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


_worker_graphs = None  # Set per-worker by _init_worker


def _init_worker(hourly_graphs):
    """Called once per worker process to store the shared graphs in its global scope."""
    global _worker_graphs
    _worker_graphs = hourly_graphs


def _route_batch(args):
    """Route one chunk of OD pairs that share an hourly graph.

    The chunk arrives as numpy arrays rather than a list of per-trip tuples:
    a Python tuple carrying a pandas Timestamp costs a few hundred bytes per
    trip, which on a million-trip county dominates both the parent's footprint
    (and therefore every forked child's) and the pickling cost of the results.
    Results go back as arrays for the same reason.
    """
    hour_key, orig_nodes, dest_nodes, dep_ns, row_idx = args
    n = len(orig_nodes)

    travel_time_s = np.full(n, np.nan, dtype=np.float64)
    distance_m = np.full(n, np.nan, dtype=np.float64)

    G = _worker_graphs.get(hour_key)
    if G is None:
        return row_idx, travel_time_s, distance_m

    for i in range(n):
        try:
            route = nx.shortest_path(
                G, source=orig_nodes[i], target=dest_nodes[i], weight="travel_time"
            )
            tt = 0.0
            dist = 0.0
            for u, v in zip(route[:-1], route[1:]):
                edge = G[u][v][0]
                tt += edge.get("travel_time", 0)
                dist += edge.get("length", 0)
            travel_time_s[i] = tt
            distance_m[i] = dist
        except nx.NetworkXNoPath:
            travel_time_s[i] = np.nan
            distance_m[i] = np.inf

    return row_idx, travel_time_s, distance_m


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

def get_routed(od_df, desired_date, hourly_graphs_arg, post_calibration=False, parallel=True):
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

    # 3) Group by hourly graph, vectorised. Everything downstream travels as
    #    numpy arrays: one Python tuple per trip (carrying a pandas Timestamp)
    #    costs hundreds of bytes, which on a million-trip county dominates the
    #    parent's footprint and hence every forked child's.
    dep_ns = departure_times.values.astype("datetime64[ns]").astype(np.int64)
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

    chunks = []
    for idx in groups:
        if len(idx) == 0:
            continue
        hour_key = pd.Timestamp(hour_floor.asi8[idx[0]])
        for beg in range(0, len(idx), target_chunk):
            sel = idx[beg : beg + target_chunk]
            chunks.append(
                (hour_key, orig_nodes[sel], dest_nodes[sel], dep_ns[sel], sel)
            )

    total_chunks = len(chunks)
    print(f"Grouped into {len(groups)} hourly buckets -> {total_chunks} chunks for {n_workers} workers")

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
            _init_worker(hourly_graphs)
            pool_kwargs = {}
        else:
            pool_kwargs = {"initializer": _init_worker, "initargs": (hourly_graphs,)}

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
        _init_worker(hourly_graphs)
        batch_results = [
            _route_batch(chunk) for chunk in tqdm(chunks, desc="Routing chunks")
        ]

    # 5) Scatter the per-chunk arrays back into full-length columns
    travel_time_s = np.full(n_pairs, np.nan, dtype=np.float64)
    distance_m = np.full(n_pairs, np.nan, dtype=np.float64)
    for row_idx, tt, dist in batch_results:
        travel_time_s[row_idx] = tt
        distance_m[row_idx] = dist

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

    return routing_df


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
