"""TMAS continuous-count parsing, station matching and count metrics.

All fixtures are written inline; nothing here reads data/tmas/.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from analysis import station_counts as sc

UTM16N = "EPSG:32616"

_VOL_HEADER = (
    "Record_Type|State_Code|F_System|Station_Id|Travel_Dir|Travel_Lane|Year_Record|"
    "Month_Record|Day_Record|Day_of_Week|Restrictions|Time_Increment|"
    + "|".join(f"Hour_{h:02d}" for h in range(24))
)


def _vol_row(station, direction, lane, day, hours):
    fields = ["3", "47", "12", station, str(direction), str(lane), "2025", "3",
              str(day), "2", "", ""] + [str(v) for v in hours]
    return "|".join(fields)


def _write_vol(tmp_path, rows):
    path = tmp_path / "test.VOL"
    path.write_text("\n".join([_VOL_HEADER] + rows) + "\n")
    return path


# -- Volume parsing ---------------------------------------------------------

def test_lane_rows_are_summed_when_there_is_no_lane_zero(tmp_path):
    path = _write_vol(tmp_path, [
        _vol_row("000111", 1, 1, 17, [10] * 24),
        _vol_row("000111", 1, 2, 17, [5] * 24),
    ])
    out = sc.read_volumes(path, day=17)
    assert len(out) == 24
    assert set(out["count"]) == {15}
    assert set(out["lane_source"]) == {"sum_lanes"}
    assert set(out["n_lane_rows"]) == {2}


def test_lane_zero_row_wins_over_the_individual_lanes(tmp_path):
    path = _write_vol(tmp_path, [
        _vol_row("000111", 1, 0, 17, [12] * 24),
        _vol_row("000111", 1, 1, 17, [10] * 24),
        _vol_row("000111", 1, 2, 17, [5] * 24),
    ])
    out = sc.read_volumes(path, day=17)
    assert set(out["count"]) == {12}, "lane-0 is all lanes, not one more lane"
    assert set(out["lane_source"]) == {"lane_0"}


def test_other_days_and_stations_are_filtered_out(tmp_path):
    path = _write_vol(tmp_path, [
        _vol_row("000111", 1, 1, 17, [7] * 24),
        _vol_row("000111", 1, 1, 18, [99] * 24),
        _vol_row("000999", 1, 1, 17, [99] * 24),
    ])
    out = sc.read_volumes(path, day=17, station_ids=["000111"])
    assert set(out["count"]) == {7}


def test_rows_from_another_month_or_year_are_filtered_out(tmp_path):
    other_year = _vol_row("000111", 1, 1, 17, [99] * 24).replace("|2025|3|", "|2024|3|")
    other_month = _vol_row("000111", 1, 1, 17, [99] * 24).replace("|2025|3|", "|2025|2|")
    path = _write_vol(tmp_path, [
        _vol_row("000111", 1, 1, 17, [7] * 24), other_year, other_month,
    ])
    out = sc.read_volumes(path, day=17, year=2025, month=3)
    assert set(out["count"]) == {7}


def test_each_direction_is_kept_separate(tmp_path):
    path = _write_vol(tmp_path, [
        _vol_row("000540", 3, 1, 17, [4] * 24),
        _vol_row("000540", 7, 1, 17, [9] * 24),
    ])
    out = sc.read_volumes(path, day=17)
    assert out.groupby("travel_dir")["count"].first().to_dict() == {3: 4, 7: 9}


# -- TMG direction codes ----------------------------------------------------

def test_single_direction_codes_map_to_one_bearing():
    assert sc.direction_bearings(1) == (0.0,)
    assert sc.direction_bearings(3) == (90.0,)
    assert sc.direction_bearings(5) == (180.0,)
    assert sc.direction_bearings(7) == (270.0,)
    assert sc.direction_bearings(2) == (45.0,)


def test_combined_direction_codes_map_to_two_opposed_bearings():
    assert sc.direction_bearings(9) == (0.0, 180.0)
    assert sc.direction_bearings(0) == (90.0, 270.0)


def test_unknown_direction_code_is_rejected():
    with pytest.raises(KeyError):
        sc.direction_bearings(42)


# -- Station -> edge matching -----------------------------------------------

def _edges():
    """Northbound and southbound carriageways by a station, plus a far edge."""
    return gpd.GeoDataFrame(
        {
            "u": [1, 3, 5],
            "v": [2, 4, 6],
            "bearing": [0.0, 180.0, 90.0],
            "highway": ["motorway"] * 3,
            "geometry": [
                LineString([(0.0, -100.0), (0.0, 100.0)]),     # northbound
                LineString([(20.0, 100.0), (20.0, -100.0)]),   # southbound
                LineString([(5000.0, 0.0), (5100.0, 0.0)]),    # far away
            ],
        },
        crs=UTM16N,
    )


def _stations(rows):
    return gpd.GeoDataFrame(
        {
            "station_id": [s for s, _, _ in rows],
            "travel_dir": [d for _, d, _ in rows],
            "geometry": [Point(*p) for _, _, p in rows],
        },
        crs=UTM16N,
    )


def test_a_northbound_station_matches_only_the_northbound_carriageway():
    out = sc.match_stations_to_edges(_stations([("A", 1, (0.0, 0.0))]), _edges())
    assert list(out["u"]) == [1]
    assert list(out["highway"]) == ["motorway"]
    assert out.iloc[0]["match_kind"] == "50m"
    assert out.iloc[0]["bearing_group"] == 0


def test_a_combined_code_matches_both_carriageways_as_separate_groups():
    out = sc.match_stations_to_edges(_stations([("A", 9, (10.0, 0.0))]), _edges())
    assert sorted(out["u"]) == [1, 3]
    assert sorted(out["bearing_group"]) == [0, 1]


def test_a_station_just_out_of_reach_is_rescued_by_the_wider_radius():
    out = sc.match_stations_to_edges(_stations([("A", 1, (-70.0, 0.0))]), _edges())
    assert list(out["u"]) == [1]
    assert out.iloc[0]["match_kind"] == "100m"


def test_a_station_nowhere_near_a_loaded_edge_matches_nothing():
    assert sc.match_stations_to_edges(
        _stations([("A", 1, (-9000.0, 0.0))]), _edges()
    ).empty


# -- Synthetic counts per station-direction-hour ----------------------------

def test_loads_average_within_a_carriageway_and_sum_across_them():
    """Consecutive edges are one cross-section; opposed carriageways add up."""
    matches = pd.DataFrame({
        "station_id": ["A"] * 3,
        "travel_dir": [9] * 3,
        "bearing_group": [0, 0, 1],
        "u": [1, 3, 5],
        "v": [2, 4, 6],
        "match_kind": ["50m"] * 3,
    })
    edges_all = pd.DataFrame({
        "u": [1, 3, 5], "v": [2, 4, 6],
        "hour": [pd.Timestamp("2025-03-17 07:00")] * 3,
        "load": [10, 20, 7],
    })
    out = sc.station_hourly_loads(matches, edges_all)
    assert out.iloc[0]["synthetic"] == pytest.approx(15.0 + 7.0)


def test_an_hour_with_no_load_on_a_matched_edge_counts_as_zero():
    matches = pd.DataFrame({
        "station_id": ["A", "A"], "travel_dir": [1, 1], "bearing_group": [0, 0],
        "u": [1, 3], "v": [2, 4], "match_kind": ["50m"] * 2,
    })
    h7 = pd.Timestamp("2025-03-17 07:00")
    h8 = pd.Timestamp("2025-03-17 08:00")
    edges_all = pd.DataFrame({
        "u": [1, 3, 1], "v": [2, 4, 2], "hour": [h7, h7, h8], "load": [10, 20, 4],
    })
    out = sc.station_hourly_loads(matches, edges_all).set_index("hour")["synthetic"]
    assert out[h7] == pytest.approx(15.0)
    assert out[h8] == pytest.approx(2.0), "edge (3,4) is unloaded at 08:00, not absent"


def test_next_day_spillover_folds_into_hour_zero():
    """Traversal loads run to 00:00 the next day, the same clock hour as 00:00."""
    day = pd.Timestamp("2025-03-17")
    hours = [day + pd.Timedelta(hours=h) for h in range(25)]
    synthetic = pd.DataFrame({
        "station_id": ["A"] * 25, "travel_dir": [1] * 25,
        "hour": hours, "synthetic": [1.0] * 24 + [5.0],
    })
    out = sc.synthetic_by_clock_hour(synthetic)
    assert len(out) == 24
    assert out.set_index("hour_of_day").loc[0, "synthetic"] == pytest.approx(6.0)
    assert out.set_index("hour_of_day").loc[7, "synthetic"] == pytest.approx(1.0)


def test_25_load_buckets_yield_24_station_hours_end_to_end():
    matches = pd.DataFrame({
        "station_id": ["A"], "travel_dir": [1], "bearing_group": [0],
        "u": [1], "v": [2], "match_kind": ["50m"], "highway": ["motorway"],
    })
    day = pd.Timestamp("2025-03-17")
    edges_all = pd.DataFrame({
        "u": [1] * 25, "v": [2] * 25,
        "hour": [day + pd.Timedelta(hours=h) for h in range(25)],
        "load": [2] * 25,
    })
    out = sc.synthetic_by_clock_hour(sc.station_hourly_loads(matches, edges_all))
    assert len(out) == 24
    assert out["synthetic"].sum() == pytest.approx(50.0)


# -- Highest road class within a carriageway --------------------------------

def test_a_side_street_is_dropped_from_a_freeway_cross_section():
    matches = pd.DataFrame({
        "station_id": ["A"] * 3, "travel_dir": [1] * 3, "bearing_group": [0, 0, 0],
        "u": [1, 3, 5], "v": [2, 4, 6],
        "highway": ["motorway", "residential", "motorway"],
        "match_kind": ["50m"] * 3,
    })
    kept, dropped = sc.keep_highest_class(matches)
    assert sorted(kept["u"]) == [1, 5]
    assert list(dropped["highway"]) == ["residential"]


def test_each_carriageway_keeps_its_own_best_class():
    matches = pd.DataFrame({
        "station_id": ["A"] * 3, "travel_dir": [9] * 3, "bearing_group": [0, 1, 1],
        "u": [1, 3, 5], "v": [2, 4, 6],
        "highway": ["residential", "primary", "residential"],
        "match_kind": ["50m"] * 3,
    })
    kept, dropped = sc.keep_highest_class(matches)
    assert sorted(kept["u"]) == [1, 3], "group 0 only has residential, so it keeps it"
    assert list(dropped["u"]) == [5]


def test_an_unranked_class_loses_to_a_ranked_one():
    matches = pd.DataFrame({
        "station_id": ["A"] * 2, "travel_dir": [1] * 2, "bearing_group": [0, 0],
        "u": [1, 3], "v": [2, 4], "highway": ["living_street", "tertiary"],
        "match_kind": ["50m"] * 2,
    })
    kept, _ = sc.keep_highest_class(matches)
    assert list(kept["highway"]) == ["tertiary"]


# -- Metrics ----------------------------------------------------------------

def test_geh_is_zero_for_a_perfect_match_and_known_otherwise():
    assert sc.geh([100.0], [100.0])[0] == pytest.approx(0.0)
    assert np.isnan(sc.geh([0.0], [0.0])[0]), "no flow either way is not a match"
    assert sc.geh([200.0], [100.0])[0] == pytest.approx(np.sqrt(2 * 100 ** 2 / 300))


def test_profile_shape_correlation_ignores_scale():
    hours = [5, 6, 7]
    df = pd.DataFrame({
        "station_id": ["A"] * 3 + ["B"] * 3,
        "travel_dir": [1] * 6,
        "hour_of_day": hours * 2,
        "synthetic": [1.0, 2.0, 3.0] + [3.0, 2.0, 1.0],
        "observed": [100.0, 200.0, 300.0] + [100.0, 200.0, 300.0],
    })
    per, pooled = sc.profile_shape_correlation(df, hours)
    assert per.set_index("station_id").loc["A", "shape_r"] == pytest.approx(1.0)
    assert per.set_index("station_id").loc["B", "shape_r"] == pytest.approx(-1.0)
    assert pooled == pytest.approx(0.0, abs=1e-9)
