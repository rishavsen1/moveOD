import os
import sys

# CRITICAL: Set environment variables BEFORE any other imports to suppress Streamlit in workers
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import osmnx as ox
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from bisect import bisect_right
import multiprocessing as mp
from tqdm import tqdm
import os
import math
import logging

from generate.config import *
from generate.utils import calculate_speed_shift, apply_mssr_to_existing_graphs
from generate.resources import plan_workers, usable_cpus


# ── Pre-computed lookup tables (module-level constants) ─────────────────────

_DEP_EDGES = [
    0, 300, 330, 360, 390, 420, 450, 480, 510, 540, 600, 660, 720, 960, 1440,
]

_DEP_NAMES = [
    "12am_to_4:59am",
    "5am_to_5:29am",
    "5:30am_to_5:59am",
    "6am_to_6:29am",
    "6:30am_to_6:59am",
    "7am_to_7:29am",
    "7:30am_to_7:59am",
    "8am_to_8:29am",
    "8:30am_to_8:59am",
    "9am_to_9:59am",
    "10am_to_10:59am",
    "11am_to_11:59am",
    "12pm_to_3:59pm",
    "4pm_to_11:59pm",
]

# Travel-time bin edges (minutes) and labels for O(log n) bisect lookup
_TT_EDGES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90]
_TT_LABELS = [
    "under_5_minutes", "5_to_9_minutes", "10_to_14_minutes",
    "15_to_19_minutes", "20_to_24_minutes", "25_to_29_minutes",
    "30_to_34_minutes", "35_to_39_minutes", "40_to_44_minutes",
    "45_to_59_minutes", "60_to_89_minutes", "90_minutes_and_over",
]

_M_TO_MI = 0.000621371


# ── Fast bin helpers using bisect (O(log n) instead of O(n)) ────────────────

def get_travel_time_bin(minutes):
    """Maps travel time in minutes to census travel time bin format"""
    return _TT_LABELS[bisect_right(_TT_EDGES, minutes)]


def get_census_time_bin_index(minutes_of_day):
    """Maps minutes-since-midnight to census departure time bin index (0-13)"""
    idx = bisect_right(_DEP_EDGES, minutes_of_day) - 1
    return max(0, min(idx, len(_DEP_NAMES) - 1))


# ── Multiprocessing worker initializer & batch function ─────────────────────

# Aim for this many routing chunks per worker so a slow chunk cannot stall a core.
CHUNKS_PER_WORKER = 4

# A chunk's SSSP result is (sources x nodes) for both distances and
# predecessors, so cap sources per chunk to keep that matrix bounded.
_SSSP_MATRIX_BUDGET_BYTES = 128 << 20  # 128 MB
_SSSP_BYTES_PER_CELL = 12              # float64 distance + int32 predecessor


_worker_graphs = None  # Set per-worker by _init_worker
_worker_csr = {}       # hour_key -> compiled CSR view of that hour's graph
_worker_nodes = {}     # hour_key -> inverse of that view's node_idx
_worker_edge_loads = False  # Set per-worker by _init_worker; see _route_batch
_worker_load_hours = "departure"  # "departure" or "traversal"; see _route_batch


def _compile_graph(G):
    """Compile a MultiDiGraph into CSR form for scipy's Dijkstra.

    For each node pair the *minimum* travel_time edge is kept, which is the edge
    nx.shortest_path would have followed, and that same edge's length is kept
    alongside it so distances correspond to the chosen route. (Summing every
    parallel edge, as a naive csr_matrix build does, would inflate the weights.)
    """
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    best = {}
    for u, v, data in G.edges(data=True):
        key = (node_idx[u], node_idx[v])
        tt = float(data.get("travel_time", 0.0))
        prev = best.get(key)
        if prev is None or tt < prev[0]:
            best[key] = (tt, float(data.get("length", 0.0)))

    n = len(nodes)
    rows = np.fromiter((k[0] for k in best), dtype=np.int32, count=len(best))
    cols = np.fromiter((k[1] for k in best), dtype=np.int32, count=len(best))
    times = np.fromiter((v[0] for v in best.values()), dtype=np.float64, count=len(best))
    matrix = csr_matrix((times, (rows, cols)), shape=(n, n))
    lengths = {k: v[1] for k, v in best.items()}
    return node_idx, matrix, lengths


def _graph_view(hour_key):
    view = _worker_csr.get(hour_key)
    if view is None:
        G = _worker_graphs.get(hour_key)
        if G is None:
            return None
        view = _compile_graph(G)
        _worker_csr[hour_key] = view
    return view


def _graph_nodes(hour_key):
    """Compiled index -> OSM node id, i.e. the inverse of a view's node_idx.

    Cached beside the view because it is only needed when edge loads are being
    collected, and then only once per view rather than once per chunk.
    """
    nodes = _worker_nodes.get(hour_key)
    if nodes is None:
        node_idx = _graph_view(hour_key)[0]
        nodes = [None] * len(node_idx)
        for node, i in node_idx.items():
            nodes[i] = node
        _worker_nodes[hour_key] = nodes
    return nodes


def _init_worker(hourly_graphs, collect_edge_loads=False, load_hours="departure"):
    """Called once per worker process to store the shared graphs in its global scope.

    The edge-load settings travel as globals rather than inside every chunk's
    args tuple so _route_batch reads them once instead of unpacking them per
    chunk.
    """
    global _worker_graphs, _worker_edge_loads, _worker_load_hours
    if hourly_graphs is not _worker_graphs:
        # The compiled views are keyed by hour alone, so they must not outlive
        # the graphs they came from: routing a second set of graphs over the
        # same hour keys (cli.py does exactly that, base then MSSR-adjusted)
        # would otherwise be answered from the first set's cache.
        _worker_csr.clear()
        _worker_nodes.clear()
    _worker_graphs = hourly_graphs
    _worker_edge_loads = collect_edge_loads
    _worker_load_hours = load_hours


def _walk_route(pred_row, dist_row, dest, root, lengths, by_hour,
                hour_bucket, dep_s):
    """Walk the predecessor tree from dest back to root, summing edge lengths.

    When by_hour is given each traversed edge is also counted into it: under
    hour_bucket, or -- when dep_s (the trip's departure, seconds since the
    epoch) is given instead -- under the hour the trip is actually on the edge.
    """
    bucket = None
    if by_hour is not None and hour_bucket is not None:
        bucket = by_hour.setdefault(hour_bucket, {})
    total = 0.0
    cur = dest
    while cur != root:
        prev = pred_row[cur]
        if prev < 0:
            return np.inf
        total += lengths.get((prev, cur), 0.0)
        if by_hour is not None:
            if dep_s is not None:
                # dist_row[prev] is the time from the origin to this edge's
                # tail, so this is when the trip is actually on the edge.
                bucket = by_hour.setdefault(int((dep_s + dist_row[prev]) // 3600), {})
            bucket[(prev, cur)] = bucket.get((prev, cur), 0) + 1
        cur = prev
    return total


def _merge_edge_loads(batch_results):
    """Sum the per-chunk {hour_key: {(u, v): trips}} maps into one."""
    edge_loads = {}
    for _, _, _, loads in batch_results:
        for hour_key, per_edge in (loads or {}).items():
            bucket = edge_loads.setdefault(hour_key, {})
            for edge, count in per_edge.items():
                bucket[edge] = bucket.get(edge, 0) + count
    return edge_loads


def _route_batch(args):
    """Route one chunk via one single-source Dijkstra per distinct origin.

    Dijkstra already computes the distance to every node on the way to any one
    of them, so routing per trip re-explores the graph once per trip even when
    trips share an origin. Here each distinct origin is solved once and every
    destination is an array lookup.

    The distance returned is the true optimal cost, so this also avoids
    re-summing the path with G[u][v][0], which picks edge key 0 rather than the
    minimum-weight parallel edge the router actually followed.

    The fourth return value is None unless _init_worker armed edge-load
    collection, in which case it is {hour_key: {(u, v): trips}} counted during
    the predecessor walk this function already performs. Under load_hours
    "traversal" an edge is filed under the hour the trip is actually on it
    rather than the hour it departed, so one chunk can return several hours.
    """
    (hour_key, src_nodes, trip_src_pos, trip_dst_nodes, row_idx,
     trip_dep_s) = args
    n_trips = len(row_idx)
    collect = _worker_edge_loads
    traversal = collect and _worker_load_hours == "traversal"

    travel_time_s = np.full(n_trips, np.nan, dtype=np.float64)
    distance_m = np.full(n_trips, np.inf, dtype=np.float64)

    view = _graph_view(hour_key)
    if view is None:
        return row_idx, travel_time_s, distance_m, None
    node_idx, matrix, lengths = view

    # Loads are bucketed by whole hours since the epoch, which is what a
    # traversal timestamp floors to. hour_key is floored explicitly rather than
    # assumed to be on the hour: a sub-hourly TIME_INTERVAL would otherwise put
    # a fractional value here.
    by_hour = {} if collect else None
    dep_bucket = (int(pd.Timestamp(hour_key).floor("1H").value // 3_600_000_000_000)
                  if collect else 0)

    src_cols = np.array([node_idx[s] for s in src_nodes], dtype=np.int64)
    dist, pred = dijkstra(
        matrix, directed=True, indices=src_cols, return_predecessors=True
    )
    if dist.ndim == 1:                      # scipy squeezes a single source
        dist = dist[None, :]
        pred = pred[None, :]

    for t in range(n_trips):
        si = trip_src_pos[t]
        dj = node_idx.get(trip_dst_nodes[t])
        if dj is None:
            continue
        cost = dist[si, dj]
        if not np.isfinite(cost):
            continue
        travel_time_s[t] = cost

        # cost is finite, so the walk always reaches the root and the counts it
        # writes are never left half-finished.
        distance_m[t] = _walk_route(
            pred[si], dist[si], dj, src_cols[si], lengths, by_hour,
            None if traversal else dep_bucket,
            trip_dep_s[t] if traversal else None,
        )

    if not collect:
        return row_idx, travel_time_s, distance_m, None

    nodes = _graph_nodes(hour_key)
    named = {
        pd.Timestamp(hour * 3600, unit="s"):
            {(nodes[a], nodes[b]): c for (a, b), c in per_edge.items()}
        for hour, per_edge in by_hour.items()
    }
    return row_idx, travel_time_s, distance_m, named


# ── Vectorized OD-pair builders (no iterrows) ──────────────────────────────

def _build_arrays_from_df(od_df, desired_date, post_calibration=False):
    """
    Vectorized extraction of all arrays needed for routing.
    Returns (origin_lats, origin_lons, dest_lats, dest_lons,
             departure_times, origin_geoids, dest_geoids).
    """
    if not post_calibration:
        dep_ts = pd.to_datetime(od_df["departure_time"])
        origin_lats = od_df["origin_loc_lat"].values
        origin_lons = od_df["origin_loc_lon"].values
        dest_lats = od_df["dest_loc_lat"].values
        dest_lons = od_df["dest_loc_lon"].values
        origin_geoids = od_df["h_geocode"].values
        dest_geoids = od_df["w_geocode"].values
    else:
        dep_ts = pd.to_datetime(od_df["departure_datetime"])
        origin_lats = od_df["origin_lat"].values
        origin_lons = od_df["origin_lon"].values
        dest_lats = od_df["destination_lat"].values
        dest_lons = od_df["destination_lon"].values
        origin_geoids = od_df["origin_geoid"].values
        dest_geoids = od_df["destination_geoid"].values

    # Reconstruct timestamps with the desired date but original time-of-day
    time_strs = dep_ts.dt.strftime("%H:%M:%S")
    departure_times = pd.to_datetime(str(desired_date) + " " + time_strs)

    return origin_lats, origin_lons, dest_lats, dest_lons, departure_times, origin_geoids, dest_geoids


# ── Main entry point ───────────────────────────────────────────────────────

def get_routed(od_df, desired_date, hourly_graphs_arg, post_calibration=False,
               parallel=True, return_edge_loads: bool = False,
               edge_load_hours: str = "departure"):
    """Route every OD pair; with return_edge_loads also count trips per edge.

    With return_edge_loads set the return value becomes
    (routing_df, {hour_key: {(u, v): trips}}) where (u, v) are OSM node ids of
    the edges the routes actually traversed. routing_df itself is unaffected.

    edge_load_hours picks which hour an edge is filed under: "departure" keys
    every edge of a trip by the hour the trip left, "traversal" by the hour the
    trip is actually on that edge. Only the latter is comparable with an hourly
    traffic count.
    """
    if edge_load_hours not in ("departure", "traversal"):
        raise ValueError(
            f"edge_load_hours must be 'departure' or 'traversal', got {edge_load_hours!r}"
        )
    hourly_graphs = hourly_graphs_arg
    n_pairs = len(od_df)

    if n_pairs == 0:
        return (pd.DataFrame(), {}) if return_edge_loads else pd.DataFrame()

    # 1) Vectorized data extraction (replaces iterrows + dict building)
    (origin_lats, origin_lons, dest_lats, dest_lons,
     departure_times, origin_geoids, dest_geoids) = _build_arrays_from_df(
        od_df, desired_date, post_calibration
    )

    print(f"Preparing {n_pairs} OD pairs for routing")

    # 2) Batch nearest-node lookup (already vectorized via osmnx)
    G_0 = next(iter(hourly_graphs.values()))
    orig_nodes = ox.distance.nearest_nodes(G_0, X=origin_lons, Y=origin_lats)
    dest_nodes = ox.distance.nearest_nodes(G_0, X=dest_lons, Y=dest_lats)

    # 3) Group by hourly graph, vectorised. Everything downstream travels as
    #    numpy arrays: one Python tuple per trip (carrying a pandas Timestamp)
    #    costs hundreds of bytes, which on a million-trip county dominates the
    #    parent's footprint and hence every forked child's.
    dep_ns = departure_times.values.astype("datetime64[ns]").astype(np.int64)
    # Seconds since the epoch, so a worker can add a travel time to a departure
    # and floor the result to an hour. Only traversal-hour loads need it.
    need_dep_s = return_edge_loads and edge_load_hours == "traversal"
    dep_s = dep_ns / 1e9 if need_dep_s else None
    hour_floor = pd.DatetimeIndex(departure_times).floor(TIME_INTERVAL)
    orig_nodes = np.asarray(orig_nodes)
    dest_nodes = np.asarray(dest_nodes)

    order = np.argsort(hour_floor.asi8, kind="stable")
    sorted_keys = hour_floor.asi8[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, boundaries)

    n_workers = max(1, min(usable_cpus() - 1, n_pairs))
    if parallel:
        # Sized from the parent's own RSS: each forked child starts from the
        # parent's image, so that -- not the graph alone -- is the per-worker cost.
        n_workers = plan_workers(n_workers, label="routing workers")
    # Several chunks per worker: hourly buckets are very uneven (rush hour
    # dominates), so one chunk each leaves most cores idle on the biggest bucket.
    target_chunk = max(50, math.ceil(n_pairs / (n_workers * CHUNKS_PER_WORKER)))

    # Chunk by *distinct origin* rather than by trip, so each origin's SSSP runs
    # exactly once. Sources per chunk are capped so the (sources x nodes) result
    # matrices stay bounded on large graphs.
    chunks = []
    n_sources_total = 0
    for idx in groups:
        if len(idx) == 0:
            continue
        hour_key = pd.Timestamp(hour_floor.asi8[idx[0]])
        G = hourly_graphs.get(hour_key)
        n_nodes = G.number_of_nodes() if G is not None else 1
        cap = max(1, _SSSP_MATRIX_BUDGET_BYTES // max(1, n_nodes * _SSSP_BYTES_PER_CELL))

        hour_origins = orig_nodes[idx]
        uniq, inverse = np.unique(hour_origins, return_inverse=True)
        n_sources_total += len(uniq)
        per_chunk = int(min(cap, max(1, math.ceil(len(uniq) / (n_workers * CHUNKS_PER_WORKER)))))

        for beg in range(0, len(uniq), per_chunk):
            src_sel = np.arange(beg, min(beg + per_chunk, len(uniq)))
            keep = np.isin(inverse, src_sel)
            if not keep.any():
                continue
            trip_pos = idx[keep]
            chunks.append((
                hour_key,
                uniq[src_sel],
                (inverse[keep] - beg).astype(np.int32),
                dest_nodes[trip_pos],
                trip_pos,
                dep_s[trip_pos] if need_dep_s else None,
            ))

    total_chunks = len(chunks)
    print(
        f"Grouped into {len(groups)} hourly buckets -> {total_chunks} chunks for "
        f"{n_workers} workers ({n_sources_total:,} distinct origins for {n_pairs:,} trips)"
    )

    # 4) Execute. fork is the only start method that works inside Streamlit's
    #    script runner (forkserver/spawn fail with KeyError: '__main__').
    if parallel and total_chunks > 1:
        print(f"Routing in parallel using {n_workers} processes")
        _st_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
        _prev_level = _st_logger.level
        _st_logger.setLevel(logging.ERROR)

        # With fork, children inherit hourly_graphs through the parent's address
        # space. Passing it via initargs would pickle it and rebuild a *second*
        # copy inside every child.
        if mp.get_start_method(allow_none=True) == "fork":
            _init_worker(hourly_graphs, return_edge_loads, edge_load_hours)
            pool_kwargs = {}
        else:
            pool_kwargs = {
                "initializer": _init_worker,
                "initargs": (hourly_graphs, return_edge_loads, edge_load_hours),
            }

        try:
            with mp.Pool(n_workers, **pool_kwargs) as pool:
                # imap (ordered) keeps results reproducible; row_idx makes the
                # ordering explicit regardless.
                batch_results = list(
                    tqdm(pool.imap(_route_batch, chunks),
                         total=total_chunks, desc="Routing chunks")
                )
        finally:
            _st_logger.setLevel(_prev_level)
    else:
        print("Routing sequentially")
        _init_worker(hourly_graphs, return_edge_loads, edge_load_hours)
        batch_results = [
            _route_batch(chunk) for chunk in tqdm(chunks, desc="Routing chunks")
        ]

    # 5) Scatter the per-chunk arrays back into full-length columns
    travel_time_s = np.full(n_pairs, np.nan, dtype=np.float64)
    distance_m = np.full(n_pairs, np.nan, dtype=np.float64)
    for row_idx, tt, dist, _ in batch_results:
        travel_time_s[row_idx] = tt
        distance_m[row_idx] = dist
    edge_loads = _merge_edge_loads(batch_results) if return_edge_loads else {}

    routed = np.isfinite(travel_time_s)
    print(f"Done: {int(routed.sum())}/{n_pairs} succeeded")

    travel_time_min = travel_time_s / 60.0
    dep_index = pd.DatetimeIndex(departure_times)
    routing_df = pd.DataFrame({
        "origin_geoid": origin_geoids,
        "destination_geoid": dest_geoids,
        "origin_node": orig_nodes,
        "destination_node": dest_nodes,
        "departure_time": dep_index,
        "departure_time_bin": [
            get_census_time_bin_index(m) for m in (dep_index.hour * 60 + dep_index.minute)
        ],
        "arrival_time": dep_index + pd.to_timedelta(travel_time_s, unit="s"),
        "travel_time_min": travel_time_min,
        "travel_time_bin": [get_travel_time_bin(v) for v in travel_time_min],
        "travel_distance_mi": distance_m * _M_TO_MI,
    })
    routing_df = routing_df[routed].reset_index(drop=True)

    if len(routing_df) > 0:
        print(f"Unique origin CBGs: {routing_df['origin_geoid'].nunique()}")
        print(f"Unique destination CBGs: {routing_df['destination_geoid'].nunique()}")

    if return_edge_loads:
        return routing_df, edge_loads
    return routing_df


def edge_loads_to_frame(edge_loads, G) -> gpd.GeoDataFrame:
    """Turn get_routed's edge loads into a GeoDataFrame, one row per edge-hour.

    Where parallel edges exist the *minimum travel_time* one is described, which
    is the edge _compile_graph keeps and therefore the edge the loads were
    counted on. (Minimum length would pick a different edge: the two criteria
    only agree when parallel edges share a speed.)
    """
    wanted = {edge for per_edge in edge_loads.values() for edge in per_edge}
    best = {}
    for u, v, data in G.edges(data=True):
        if (u, v) not in wanted:
            continue
        tt = float(data.get("travel_time", 0.0))
        if (u, v) not in best or tt < best[(u, v)][0]:
            best[(u, v)] = (tt, data)

    rows = []
    for hour_key, per_edge in edge_loads.items():
        for (u, v), load in per_edge.items():
            entry = best.get((u, v))
            data = entry[1] if entry else {}
            highway = data.get("highway")
            rows.append({
                "u": u,
                "v": v,
                "hour": pd.Timestamp(hour_key),
                "load": load,
                "length_m": float(data.get("length", 0.0)),
                "highway": highway[0] if isinstance(highway, list) else highway,
                "geometry": data.get("geometry") or LineString(
                    [(G.nodes[u]["x"], G.nodes[u]["y"]),
                     (G.nodes[v]["x"], G.nodes[v]["y"])]
                ),
            })

    return gpd.GeoDataFrame(
        rows,
        columns=["u", "v", "hour", "load", "length_m", "highway", "geometry"],
        geometry="geometry",
        crs=G.graph.get("crs", "epsg:4326"),
    )


def mean_speed_shift_is_uniform(hourly_graphs):
    """True when no edge keeps a measured INRIX speed.

    apply_mssr_to_existing_graphs only rescales edges without INRIX data. When
    there are none, every edge is scaled by the same factor.
    """
    distinct = {id(G): G for G in hourly_graphs.values()}
    for G in distinct.values():
        for _, _, data in G.edges(data=True):
            if data.get("inrix_speed", False):
                return False
    return True


def rescale_routing_df(routing_df, psi):
    """Apply a *uniform* mean speed shift to an already-routed frame.

    Scaling every road speed by psi divides each edge travel time by psi. A
    uniform scaling does not change which path is shortest, so the routes -- and
    therefore travel_distance_mi -- are unchanged and the travel times can be
    rescaled directly. This makes the second routing pass redundant whenever
    mean_speed_shift_is_uniform() holds.
    """
    out = routing_df.copy()
    out["travel_time_min"] = out["travel_time_min"] / psi
    out["arrival_time"] = out["departure_time"] + pd.to_timedelta(
        out["travel_time_min"] * 60.0, unit="s"
    )
    out["travel_time_bin"] = out["travel_time_min"].map(get_travel_time_bin)
    return out


def perform_mean_speed_shift(routing_df, travel_time_to_work_by_geoid, hourly_graphs):
    psi = calculate_speed_shift(routing_df, travel_time_to_work_by_geoid)
    print(f"Mean Speed Shift Ratio (psi): {psi:.4f}")

    # Create hourly graphs with speed shift
    hourly_graphs_adjusted = apply_mssr_to_existing_graphs(hourly_graphs, psi)

    return hourly_graphs_adjusted
