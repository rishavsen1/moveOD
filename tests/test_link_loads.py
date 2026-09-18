"""Per-edge load accumulation in the router and the INRIX link-load comparison.

Everything here runs on toy graphs and toy frames; no test touches the real
county runs or the INRIX extracts on disk.
"""

import datetime

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString

from analysis import link_loads
from generate.generate_routing_df import edge_loads_to_frame, get_routed

HOUR = pd.Timestamp("2025-03-17 07:00:00")
DESIRED_DATE = datetime.date(2025, 3, 17)

# Projected CRS used by the matcher tests so distances are plain metres.
UTM16N = "EPSG:32616"


# ── Toy road graph ──────────────────────────────────────────────────────────

def _toy_graph():
    """Two origins feeding a shared link to one destination.

        1 ──┐
            ├── 3 ──> 4
        2 ──┘

    Node 5 offers 1 -> 5 -> 4, which is shorter in metres but far slower, so the
    router must never pick it. Nodes 1, 2 and 4 are the only origins, and no
    route passes through an origin, which is what makes the per-origin load
    conservation identity exact.
    """
    G = nx.MultiDiGraph(crs="epsg:4326")
    coords = {
        1: (-85.30, 35.00),
        2: (-85.30, 35.04),
        3: (-85.28, 35.02),
        4: (-85.24, 35.02),
        5: (-85.28, 35.08),
    }
    for node, (x, y) in coords.items():
        G.add_node(node, x=x, y=y)
    for u, v, travel_time, length in [
        (1, 3, 10.0, 100.0),
        (2, 3, 10.0, 200.0),
        (3, 4, 10.0, 400.0),
        (1, 5, 900.0, 10.0),
        (5, 4, 900.0, 10.0),
    ]:
        G.add_edge(u, v, travel_time=travel_time, length=length, highway="residential")
    return G, coords


def _toy_od_df(coords):
    """Three trips 1->4, two trips 2->4 and one zero-edge trip 4->4."""
    trips = [(1, 4)] * 3 + [(2, 4)] * 2 + [(4, 4)]
    rows = []
    for origin, dest in trips:
        rows.append({
            "origin_geoid": f"{origin}",
            "destination_geoid": f"{dest}",
            "departure_datetime": "2025-03-17 07:15:00",
            "origin_lon": coords[origin][0],
            "origin_lat": coords[origin][1],
            "destination_lon": coords[dest][0],
            "destination_lat": coords[dest][1],
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def toy_routed():
    G, coords = _toy_graph()
    od_df = _toy_od_df(coords)
    hourly_graphs = {HOUR: G}
    routing_df, edge_loads = get_routed(
        od_df,
        DESIRED_DATE,
        hourly_graphs,
        post_calibration=True,
        parallel=False,
        return_edge_loads=True,
    )
    return G, od_df, routing_df, edge_loads


# ── Part 1: edge loads ──────────────────────────────────────────────────────

def test_known_path_yields_known_edge_counts(toy_routed):
    _, _, _, edge_loads = toy_routed
    assert set(edge_loads) == {HOUR}
    assert edge_loads[HOUR] == {(1, 3): 3, (2, 3): 2, (3, 4): 5}


def test_load_conservation_per_origin(toy_routed):
    _, _, routing_df, edge_loads = toy_routed
    loads = edge_loads[HOUR]

    for origin in routing_df["origin_node"].unique():
        out_load = sum(c for (u, _), c in loads.items() if u == origin)
        from_origin = routing_df[routing_df["origin_node"] == origin]
        with_edges = int((from_origin["travel_distance_mi"] > 0).sum())
        assert out_load == with_edges, f"origin {origin}"


def test_loaded_length_matches_routed_distance(toy_routed):
    """Sum of load x edge length reproduces the routed distance of every trip."""
    G, _, routing_df, edge_loads = toy_routed
    lengths = {(u, v): G[u][v][0]["length"] for u, v in edge_loads[HOUR]}
    loaded_m = sum(load * lengths[e] for e, load in edge_loads[HOUR].items())
    routed_m = routing_df["travel_distance_mi"].sum() / 0.000621371
    assert loaded_m == pytest.approx(routed_m, rel=1e-9)


def test_flag_off_leaves_routing_df_unchanged():
    G, coords = _toy_graph()
    od_df = _toy_od_df(coords)
    hourly_graphs = {HOUR: G}
    plain = get_routed(od_df, DESIRED_DATE, hourly_graphs,
                       post_calibration=True, parallel=False)
    assert isinstance(plain, pd.DataFrame)

    with_loads, _ = get_routed(od_df, DESIRED_DATE, hourly_graphs,
                               post_calibration=True, parallel=False,
                               return_edge_loads=True)
    assert_frame_equal(plain, with_loads)


def test_empty_frame_returns_empty_loads():
    routing_df, edge_loads = get_routed(
        pd.DataFrame(), DESIRED_DATE, {HOUR: _toy_graph()[0]},
        post_calibration=True, parallel=False, return_edge_loads=True,
    )
    assert routing_df.empty
    assert edge_loads == {}


def _straight_line_graph():
    """1 -> 3 -> 4, with a 20-minute first leg and a 10-minute second leg."""
    G = nx.MultiDiGraph(crs="epsg:4326")
    for node, (x, y) in {1: (-85.30, 35.00), 3: (-85.28, 35.00),
                         4: (-85.26, 35.00)}.items():
        G.add_node(node, x=x, y=y)
    G.add_edge(1, 3, travel_time=1200.0, length=100.0, highway="primary")
    G.add_edge(3, 4, travel_time=600.0, length=400.0, highway="primary")
    return G


def _one_trip_at(hhmm):
    return pd.DataFrame([{
        "origin_geoid": "1", "destination_geoid": "4",
        "departure_datetime": f"2025-03-17 {hhmm}:00",
        "origin_lon": -85.30, "origin_lat": 35.00,
        "destination_lon": -85.26, "destination_lat": 35.00,
    }])


@pytest.mark.parametrize("mode,expected", [
    ("departure", {HOUR: {(1, 3): 1, (3, 4): 1}}),
    ("traversal", {HOUR: {(1, 3): 1},
                   pd.Timestamp("2025-03-17 08:00:00"): {(3, 4): 1}}),
])
def test_edge_load_hours_departure_vs_traversal(mode, expected):
    """A 07:50 departure is still on its second edge at 08:10."""
    G = _straight_line_graph()
    _, loads = get_routed(_one_trip_at("07:50"), DESIRED_DATE, {HOUR: G},
                          post_calibration=True, parallel=False,
                          return_edge_loads=True, edge_load_hours=mode)
    assert loads == expected


def test_edge_load_hours_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="edge_load_hours"):
        get_routed(_one_trip_at("07:50"), DESIRED_DATE, {HOUR: _straight_line_graph()},
                   post_calibration=True, parallel=False,
                   return_edge_loads=True, edge_load_hours="arrival")


def test_traversal_hours_conserve_total_load(toy_routed):
    """Re-keying by traversal hour moves loads between hours, never creates them."""
    G, od_df, _, dep_loads = toy_routed
    _, trav_loads = get_routed(od_df, DESIRED_DATE, {HOUR: G},
                               post_calibration=True, parallel=False,
                               return_edge_loads=True, edge_load_hours="traversal")
    total = lambda d: sum(c for per in d.values() for c in per.values())
    assert total(trav_loads) == total(dep_loads)


def test_second_graph_set_is_not_answered_from_the_first_ones_cache():
    """Regression: compiled views are cached per hour, not per graph.

    cli.py routes the same trips twice over the same hour keys -- base graphs,
    then MSSR-adjusted ones -- so a stale view would silently return the first
    set's travel times.
    """
    od_df = _one_trip_at("07:50")
    base = _straight_line_graph()
    faster = _straight_line_graph()
    for _, _, data in faster.edges(data=True):
        data["travel_time"] /= 2.0

    slow = get_routed(od_df, DESIRED_DATE, {HOUR: base},
                      post_calibration=True, parallel=False)
    quick = get_routed(od_df, DESIRED_DATE, {HOUR: faster},
                       post_calibration=True, parallel=False)
    assert slow["travel_time_min"].iat[0] == pytest.approx(30.0)
    assert quick["travel_time_min"].iat[0] == pytest.approx(15.0)


def test_edge_loads_to_frame_keeps_the_routed_parallel_edge():
    """Parallel edges resolve the way _compile_graph does: min travel_time."""
    G, _ = _toy_graph()
    # A second 1->3 edge that is shorter but slower than the original.
    G.add_edge(1, 3, travel_time=999.0, length=1.0, highway="service")

    frame = edge_loads_to_frame({HOUR: {(1, 3): 7}}, G)
    assert list(frame.columns) == [
        "u", "v", "hour", "load", "length_m", "highway", "geometry",
    ]
    row = frame.iloc[0]
    assert (row["u"], row["v"], row["load"]) == (1, 3, 7)
    assert row["length_m"] == pytest.approx(100.0)
    assert row["highway"] == "residential"
    assert isinstance(row["geometry"], LineString)


# ── Part 2: XD segment -> OSM edge matching ─────────────────────────────────

def _toy_edges():
    return gpd.GeoDataFrame(
        {
            "u": [1, 2],
            "v": [3, 4],
            "load": [10.0, 5.0],
            "bearing": [0.0, 90.0],
            "geometry": [
                LineString([(0.0, 0.0), (0.0, 100.0)]),
                LineString([(1000.0, 0.0), (1100.0, 0.0)]),
            ],
        },
        crs=UTM16N,
    )


def _toy_segments():
    return gpd.GeoDataFrame(
        {
            "xd": [101, 102],
            "bearing_deg": [2.0, 0.0],
            "geometry": [
                LineString([(5.0, 0.0), (5.0, 100.0)]),        # 5 m off edge (1, 3)
                LineString([(5000.0, 0.0), (5000.0, 100.0)]),  # nowhere near an edge
            ],
        },
        crs=UTM16N,
    )


def test_matcher_pairs_one_segment_and_drops_the_other():
    matches = link_loads.match_segments_to_edges(_toy_segments(), _toy_edges())
    assert list(matches["xd"]) == [101]
    assert (matches.iloc[0]["u"], matches.iloc[0]["v"]) == (1, 3)
    assert matches.iloc[0]["match_kind"] == "buffer"


def test_matcher_rejects_an_edge_pointing_the_other_way():
    segments = _toy_segments().iloc[:1].copy()
    segments["bearing_deg"] = [180.0]
    matches = link_loads.match_segments_to_edges(
        segments, _toy_edges(), fallback_m=0.0
    )
    assert matches.empty


def test_segment_loads_aggregates_matched_edges():
    matches = pd.DataFrame({
        "xd": [1, 1, 2],
        "u": [1, 2, 3],
        "v": [2, 3, 4],
        "load": [10.0, 20.0, 4.0],
        "match_kind": ["buffer"] * 3,
    })
    out = link_loads.segment_loads(matches).set_index("xd")
    assert out.loc[1, "match_kind"] == "buffer", "the match kind must survive"
    assert out.loc[1, "load_mean"] == pytest.approx(15.0)
    assert out.loc[1, "load_max"] == pytest.approx(20.0)
    assert out.loc[1, "n_edges"] == 2
    assert out.loc[2, "load_mean"] == pytest.approx(4.0)


def test_bearing_difference_wraps_around_north():
    diffs = link_loads.bearing_difference([350.0, 10.0, 180.0], [10.0, 350.0, 0.0])
    assert list(diffs) == pytest.approx([20.0, 20.0, 180.0])


# ── Part 2: statistics helpers ──────────────────────────────────────────────

def test_spearman_is_one_for_a_monotone_pair():
    rho, p, n = link_loads.spearman([1, 2, 3, 4, 5], [2.0, 9.0, 11.0, 30.0, 31.0])
    assert rho == pytest.approx(1.0)
    assert n == 5
    assert p < 0.05


def test_spearman_is_nan_when_there_is_nothing_to_rank():
    rho, p, n = link_loads.spearman([1.0], [2.0])
    assert np.isnan(rho)
    assert n == 1


def _two_class_frame():
    """Class A: load and congestion move together. Class B: they move apart.

    Pooled, B's high loads sit at low congestion, so the unstratified rho is
    dragged down; within classes the two effects are +1 and -1.
    """
    return pd.DataFrame({
        "frc": ["A"] * 6 + ["B"] * 6,
        "load_mean": [1.0, 2, 3, 4, 5, 6] + [100.0, 200, 300, 400, 500, 600],
        "congestion": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] + [0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
    })


def test_stratified_spearman_reports_each_class():
    out = link_loads.stratified_spearman(
        _two_class_frame(), "load_mean", "congestion", "frc"
    ).set_index("frc")
    assert out.loc["A", "rho"] == pytest.approx(1.0)
    assert out.loc["B", "rho"] == pytest.approx(-1.0)
    assert list(out["n"]) == [6, 6]


def test_within_class_rank_spearman_cancels_the_class_confound():
    rho, _, n = link_loads.within_class_rank_spearman(
        _two_class_frame(), "load_mean", "congestion", "frc"
    )
    assert n == 12
    assert rho == pytest.approx(0.0, abs=1e-12)


def test_within_class_rank_spearman_skips_classes_too_small_to_rank():
    df = pd.concat([_two_class_frame(),
                    pd.DataFrame({"frc": ["C"], "load_mean": [7.0],
                                  "congestion": [0.9]})], ignore_index=True)
    _, _, n = link_loads.within_class_rank_spearman(
        df, "load_mean", "congestion", "frc"
    )
    assert n == 12


def test_decile_means_bin_load_and_average_congestion():
    load = np.arange(100, dtype=float)
    congestion = load / 100.0
    out = link_loads.decile_means(load, congestion, n_bins=10)
    assert len(out) == 10
    assert list(out["n"]) == [10] * 10
    assert out["congestion_mean"].is_monotonic_increasing
    assert out.iloc[0]["congestion_mean"] == pytest.approx(0.045)
    assert out.iloc[-1]["congestion_mean"] == pytest.approx(0.945)
