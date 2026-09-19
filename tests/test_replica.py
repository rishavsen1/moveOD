"""Unit tests for the Replica OD held-out benchmark module. No network or file
I/O: geojson and OD frames are built as small in-memory toy frames."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from analysis import replica


# --------------------------------------------------------------------------- purposes


def test_purpose_totals_pass_when_purposes_sum_to_total():
    od = pd.DataFrame({
        "other_activity_type_count": [1.0, 2.0],
        "work_count": [2.0, 3.0],
        "school_count": [0.0, 0.0],
        "eat_count": [0.0, 0.0],
        "shop_count": [0.0, 0.0],
        "social_count": [0.0, 0.0],
        "recreation_count": [0.0, 0.0],
        "maintenance_count": [0.0, 0.0],
        "stage_count": [0.0, 0.0],
        "lodging_count": [0.0, 0.0],
        "region_departure_count": [0.0, 0.0],
        "commercial_count": [0.0, 0.0],
        "home_count": [0.0, 0.0],
        "total_count": [3.0, 5.0],
    })
    replica.validate_purpose_totals(od)  # must not raise


def test_purpose_totals_raises_when_purposes_do_not_sum_to_total():
    od = pd.DataFrame({col: [0.0] for col in replica.REPLICA_PURPOSE_COLUMNS})
    od.loc[0, "work_count"] = 5.0
    od["total_count"] = [999.0]
    with pytest.raises(ValueError):
        replica.validate_purpose_totals(od)


def test_select_purpose_returns_requested_column():
    od = pd.DataFrame({"work_count": [1.0, 2.0], "shop_count": [3.0, 4.0]})
    pd.testing.assert_series_equal(replica.select_purpose(od, "work_count"), od["work_count"])


def test_select_purpose_rejects_unknown_column():
    od = pd.DataFrame({"work_count": [1.0]})
    with pytest.raises(ValueError):
        replica.select_purpose(od, "not_a_real_purpose")


# --------------------------------------------------------------------------- zone_to_block_group


def _toy_county_geojson() -> gpd.GeoDataFrame:
    """Two block groups in the target county (FIPS 12001) and one in a
    neighbouring county (FIPS 12003), all disjoint unit squares."""
    return gpd.GeoDataFrame(
        {
            "GEOID": ["120010001001", "120010002001", "120030001001"],
            "COUNTYFP": ["001", "001", "003"],
            "geometry": [
                box(-84.0, 34.0, -83.0, 35.0),   # target county, block group A
                box(-83.0, 34.0, -82.0, 35.0),   # target county, block group B
                box(-86.0, 34.0, -85.0, 35.0),   # neighbouring county
            ],
        },
        crs="EPSG:4326",
    )


def _toy_od(zones: list[tuple[str, float, float]]) -> pd.DataFrame:
    """A minimal OD frame carrying only the columns zone_to_block_group needs."""
    rows = []
    for name, lon, lat in zones:
        rows.append({
            "origin_name": name, "origin_centroidLon": lon, "origin_centroidLat": lat,
            "destination_name": name, "destination_centroidLon": lon, "destination_centroidLat": lat,
        })
    return pd.DataFrame(rows)


def test_zone_to_block_group_excludes_outside_and_neighbouring_county():
    od = _toy_od([
        ("in-target", -83.5, 34.5),       # inside target county, block group A
        ("in-neighbour", -85.5, 34.5),    # inside a real polygon, but county FIPS 12003 -- must be excluded
        ("outside-all", -80.0, 34.5),     # centroid outside every polygon -- must be excluded
    ])
    zone_map, diagnostics = replica.zone_to_block_group(od, _toy_county_geojson(), "12001")

    assert zone_map == {"in-target": "120010001001"}
    assert diagnostics["n_zones_total"] == 3
    assert diagnostics["n_zones_matched"] == 1
    assert diagnostics["n_block_groups_covered"] == 1
    assert diagnostics["n_block_groups_in_county"] == 2


def test_zone_to_block_group_keeps_leading_zero_geoids():
    """A GEOID that has round-tripped through int (dropping a leading zero,
    as a state FIPS of '01' would) must still be recovered as a 12-char
    zero-padded string, never silently truncated."""
    gdf = gpd.GeoDataFrame(
        {
            "GEOID": [10010201001],  # int64: leading zero of state FIPS 01 already lost
            "COUNTYFP": ["001"],
            "geometry": [box(-87.0, 32.0, -86.0, 33.0)],
        },
        crs="EPSG:4326",
    )
    od = _toy_od([("z1", -86.5, 32.5)])
    zone_map, _ = replica.zone_to_block_group(od, gdf, "01001")

    assert zone_map == {"z1": "010010201001"}
    assert all(len(v) == 12 for v in zone_map.values())


# --------------------------------------------------------------------------- intra_county_flows


def _toy_flow_od() -> pd.DataFrame:
    return pd.DataFrame({
        "origin_name": ["z1", "z1", "z2"],
        "destination_name": ["z2", "z3", "z1"],
        "work_count": [10.0, 5.0, 7.0],
    })


def test_intra_county_flows_drops_unmapped_zones_and_sums_by_block_group():
    zone_map = {"z1": "120010001001", "z2": "120010002001"}  # z3 unmapped (outside county)
    flows = replica.intra_county_flows(_toy_flow_od(), zone_map, purpose="work_count", level="block_group")

    assert set(flows.columns) == {"origin_bg", "destination_bg", "trips"}
    assert len(flows) == 2  # the z1->z3 row is dropped
    row = flows[(flows["origin_bg"] == "120010001001") & (flows["destination_bg"] == "120010002001")]
    assert row["trips"].iloc[0] == 10.0


def test_intra_county_flows_tract_level_truncates_to_eleven_chars():
    zone_map = {"z1": "120010001001", "z2": "120010002001"}  # distinct block groups, distinct tracts
    flows = replica.intra_county_flows(_toy_flow_od(), zone_map, purpose="work_count", level="tract")

    assert set(flows.columns) == {"origin_tract", "destination_tract", "trips"}
    assert (flows["origin_tract"].str.len() == 11).all()
    row = flows[(flows["origin_tract"] == "12001000100") & (flows["destination_tract"] == "12001000200")]
    assert row["trips"].iloc[0] == 10.0


# --------------------------------------------------------------------------- agreement_matrix


def _series(index_pairs: list[tuple[str, str]], values: list[float]) -> pd.Series:
    index = pd.MultiIndex.from_tuples(index_pairs, names=["origin_tract", "destination_tract"])
    return pd.Series(values, index=index, name="workers")


def test_agreement_matrix_identical_series_scores_perfectly():
    a = _series([("t1", "t2"), ("t2", "t3"), ("t3", "t1")], [10.0, 20.0, 30.0])
    result = replica.agreement_matrix({"a": a, "b": a.copy()})

    row = result.iloc[0]
    assert row["ssi"] == pytest.approx(1.0)
    assert row["spearman_r"] == pytest.approx(1.0)
    assert row["pearson_r"] == pytest.approx(1.0)
    assert row["cpc_a_covers_b"] == pytest.approx(1.0)
    assert row["cpc_b_covers_a"] == pytest.approx(1.0)


def test_agreement_matrix_disjoint_series_scores_zero_overlap():
    a = _series([("t1", "t2")], [10.0])
    b = _series([("t3", "t4")], [10.0])
    result = replica.agreement_matrix({"a": a, "b": b})

    row = result.iloc[0]
    assert row["ssi"] == pytest.approx(0.0)
    assert row["cpc_a_covers_b"] == pytest.approx(0.0)
    assert row["cpc_b_covers_a"] == pytest.approx(0.0)
    assert row["n_pairs"] == 2


def test_agreement_matrix_scaled_series_is_not_invariant_in_ssi_but_is_in_spearman():
    a = _series([("t1", "t2"), ("t2", "t3"), ("t3", "t1")], [10.0, 20.0, 40.0])
    b = a.rename("workers") * 2.0  # pure scaling by a constant

    result = replica.agreement_matrix({"a": a, "b": b})
    row = result.iloc[0]

    # SSI is magnitude-sensitive: 2*sum(min(a,b)) / sum(a+b) = 2*sum(a) / (3*sum(a)) = 2/3.
    assert row["ssi"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert row["ssi"] != pytest.approx(1.0)
    # Spearman is rank-based and invariant to any positive monotonic scaling.
    assert row["spearman_r"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- support restriction


def test_union_support_is_union_of_all_indices():
    a = _series([("t1", "t2"), ("t2", "t3")], [1.0, 2.0])
    b = _series([("t2", "t3"), ("t3", "t4")], [3.0, 4.0])
    support = replica.union_support({"a": a, "b": b})

    assert set(support) == {("t1", "t2"), ("t2", "t3"), ("t3", "t4")}


def test_restrict_to_support_zero_fills_missing_pairs():
    a = _series([("t1", "t2")], [5.0])
    support = pd.MultiIndex.from_tuples(
        [("t1", "t2"), ("t2", "t3")], names=["origin_tract", "destination_tract"],
    )
    restricted = replica.restrict_to_support({"a": a}, support)

    assert list(restricted["a"].index) == list(support)
    assert restricted["a"].loc[("t1", "t2")] == 5.0
    assert restricted["a"].loc[("t2", "t3")] == 0.0


def test_support_indices_excludes_all_zero_keys_but_keeps_any_positive():
    """A flow-pair that is zero in every source carries no commute information and must be
    excluded from the union (it would otherwise inflate rank correlation as a tied zero pair);
    a pair with a positive value in even one source must be kept."""
    index = pd.MultiIndex.from_tuples(
        [("t1", "t2"), ("t3", "t4")], names=["origin_tract", "destination_tract"],
    )
    zero_everywhere, positive_in_one = ("t1", "t2"), ("t3", "t4")
    matrices = {
        "moveod": pd.Series([0.0, 5.0], index=index),
        "ctpp": pd.Series([0.0, 0.0], index=index),
        "replica": pd.Series([0.0, 0.0], index=index),
        "lodes": pd.Series([0.0, 0.0], index=index),
    }
    supports = replica._support_indices(matrices)

    assert zero_everywhere not in supports["union"]
    assert positive_in_one in supports["union"]


def test_agreement_matrix_on_restricted_support_uses_fixed_pair_count():
    a = _series([("t1", "t2")], [5.0])
    b = _series([("t2", "t3")], [7.0])
    support = pd.MultiIndex.from_tuples(
        [("t1", "t2"), ("t2", "t3"), ("t3", "t4")], names=["origin_tract", "destination_tract"],
    )
    restricted = replica.restrict_to_support({"a": a, "b": b}, support)
    result = replica.agreement_matrix(restricted)

    assert (result["n_pairs"] == 3).all()
