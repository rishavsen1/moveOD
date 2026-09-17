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
import numpy as np
from bisect import bisect_right
import multiprocessing as mp
from tqdm import tqdm
import os
import math
import logging

from generate.config import *
from generate.utils import calculate_speed_shift, apply_mssr_to_existing_graphs


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

_worker_graphs = None  # Set per-worker by _init_worker


def _init_worker(hourly_graphs):
    """Called once per worker process to store the shared graphs in its global scope."""
    global _worker_graphs
    _worker_graphs = hourly_graphs


def _route_batch(args):
    """
    Route a batch of OD pairs that share the same hourly graph.
    Receives (hour_key, task_list).
    The worker reads the graph from its process-local _worker_graphs.
    Each task is (orig_node, dest_node, dep_time, origin_geoid, dest_geoid).
    Returns a list of result dicts (or None for failures).
    """
    hour_key, tasks = args

    G = _worker_graphs.get(hour_key)
    if G is None:
        return [None] * len(tasks)

    results = []
    for orig_node, dest_node, dep_time, origin_geoid, dest_geoid in tasks:
        try:
            route = nx.shortest_path(G, source=orig_node, target=dest_node, weight="travel_time")
            # Walk edges once, accumulate both metrics
            total_tt = 0.0
            total_dist = 0.0
            for u, v in zip(route[:-1], route[1:]):
                edge = G[u][v][0]
                total_tt += edge.get("travel_time", 0)
                total_dist += edge.get("length", 0)
            total_distance_mi = total_dist * _M_TO_MI
        except nx.NetworkXNoPath:
            total_tt = np.nan
            total_distance_mi = np.inf

        travel_time_min = total_tt / 60.0
        dep_mins = dep_time.hour * 60 + dep_time.minute

        results.append({
            "origin_geoid": origin_geoid,
            "destination_geoid": dest_geoid,
            "origin_node": orig_node,
            "destination_node": dest_node,
            "departure_time": dep_time,
            "departure_time_bin": get_census_time_bin_index(dep_mins),
            "arrival_time": dep_time + pd.to_timedelta(total_tt, unit="s"),
            "travel_time_min": travel_time_min,
            "travel_time_bin": get_travel_time_bin(travel_time_min),
            "travel_distance_mi": total_distance_mi,
        })
    return results


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

def get_routed(od_df, desired_date, hourly_graphs_arg, post_calibration=False, parallel=False):
    hourly_graphs = hourly_graphs_arg
    n_pairs = len(od_df)

    if n_pairs == 0:
        return pd.DataFrame()

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

    # 3) Group tasks by hourly graph key for cache-locality
    #    Each graph is only looked-up once per batch instead of per-pair.
    from collections import defaultdict
    hour_buckets = defaultdict(list)
    for i in range(n_pairs):
        dep_time = departure_times.iloc[i] if hasattr(departure_times, 'iloc') else departure_times[i]
        hour_key = dep_time.floor(TIME_INTERVAL)
        hour_buckets[hour_key].append((
            orig_nodes[i], dest_nodes[i], dep_time,
            origin_geoids[i], dest_geoids[i],
        ))

    # 4) Sub-chunk large hourly batches so work distributes evenly across cores.
    #    E.g. if 7am has 5000 pairs and 11pm has 50, without sub-chunking one
    #    core would be stuck on the 7am batch while others sit idle.
    n_workers = min((os.cpu_count() or 4) - 1, n_pairs)
    n_workers = max(1, n_workers)
    # Target: ~equal-sized chunks across all workers
    target_chunk = max(50, math.ceil(n_pairs / n_workers))

    chunks = []
    for hour_key, tasks in hour_buckets.items():
        # Split this hour's tasks into sub-chunks
        for start in range(0, len(tasks), target_chunk):
            chunks.append((hour_key, tasks[start : start + target_chunk]))

    total_chunks = len(chunks)
    print(f"Grouped into {len(hour_buckets)} hourly buckets → {total_chunks} chunks for {n_workers} workers")

    # 5) Execute with multiprocessing.Pool (fork) for true parallelism.
    #    fork is the only start method that works inside Streamlit's script runner
    #    (forkserver/spawn fail with KeyError: '__main__').
    #    We silence the harmless ScriptRunContext warnings that forked workers emit.
    if parallel and total_chunks > 1:
        print(f"Routing in parallel using {n_workers} processes")
        _st_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
        _prev_level = _st_logger.level
        _st_logger.setLevel(logging.ERROR)
        try:
            with mp.Pool(n_workers, initializer=_init_worker, initargs=(hourly_graphs,)) as pool:
                batch_results = list(
                    tqdm(pool.imap_unordered(_route_batch, chunks),
                         total=total_chunks, desc="Routing chunks")
                )
        finally:
            _st_logger.setLevel(_prev_level)
    else:
        print("Routing sequentially")
        _init_worker(hourly_graphs)  # set up the global for single-process path
        batch_results = [
            _route_batch(chunk)
            for chunk in tqdm(chunks, desc="Routing chunks")
        ]

    # 5) Flatten & filter
    all_results = [r for batch in batch_results for r in batch if r is not None]
    print(f"Done: {len(all_results)}/{n_pairs} succeeded")

    routing_df = pd.DataFrame(all_results)

    if len(routing_df) > 0:
        print(f"Unique origin CBGs: {routing_df['origin_geoid'].nunique()}")
        print(f"Unique destination CBGs: {routing_df['destination_geoid'].nunique()}")

    return routing_df


def perform_mean_speed_shift(routing_df, travel_time_to_work_by_geoid, hourly_graphs):
    mssr = calculate_speed_shift(routing_df, travel_time_to_work_by_geoid)
    print(f"Mean Speed Shift Ratio (MSSr): {mssr:.4f}")

    # Create hourly graphs with speed shift
    hourly_graphs_adjusted = apply_mssr_to_existing_graphs(hourly_graphs, mssr)

    return hourly_graphs_adjusted
