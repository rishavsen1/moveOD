import pandas as pd
import numpy as np
import ast
import random
import gc
import psutil
import zlib

np.random.seed(123)

from generate.config import TIME_INTERVAL

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def _mem_gb():
    """Current process RSS in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)

def _available_gb():
    """Available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def extract_latlon(x):
    if isinstance(x, str):
        coords = ast.literal_eval(x)
    else:
        coords = x
    return pd.Series({"lat": coords[0], "lon": coords[1]})


def sample_buildings(syn_df, bldg_df, key, lat_col, lon_col, out_lat, out_lon):
    draws = []
    for cbg, grp in syn_df.groupby(key):
        pool = bldg_df[bldg_df[key] == cbg][[lat_col, lon_col]]
        if pool.empty:
            # no buildings in this CBG
            tmp = pd.DataFrame({out_lat: [np.nan] * len(grp), out_lon: [np.nan] * len(grp)}, index=grp.index)
        else:
            samp = pool.sample(n=len(grp), replace=True).reset_index(drop=True)
            samp.index = grp.index
            tmp = pd.DataFrame({out_lat: samp[lat_col], out_lon: samp[lon_col]}, index=grp.index)
        draws.append(tmp)
    return pd.concat(draws)


# Calculate OD distribution
def calculate_od_distribution(od_df, lodes_dict):
    """Calculate OD distribution for each origin"""
    od_dist = {}

    # Group by origin
    for origin, group in od_df.groupby("h_geocode"):
        # Calculate total trips for this origin
        total_trips = lodes_dict[origin]
        if total_trips <= 0:
            continue

        # Calculate probability for each destination
        od_dist[origin] = {}
        for _, row in group.iterrows():
            dest = row["w_geocode"]
            count = row["total_jobs"]
            od_dist[origin][dest] = count / total_trips

    return od_dist


def get_departure_time_dist(dep_tbl):
    dep_tbl = dep_tbl.loc[:, ~dep_tbl.columns.duplicated()]
    est_cols = [c for c in dep_tbl.columns if c.endswith("_estimate") and c != "total_estimate"]

    dep_tbl = dep_tbl[["GEO_ID", "total_estimate", *est_cols]].set_index("GEO_ID")

    dep_bin_names = est_cols
    dep_to_idx = {name: i for i, name in enumerate(dep_bin_names)}

    q_dict = {}
    count = 0
    for o, row in dep_tbl.iterrows():
        total = row["total_estimate"]
        if total > 0:
            q_dict[o] = {dep_to_idx[col]: row[col] / total for col in dep_bin_names if row[col] > 0}
        else:
            q_dict[o] = {}
            count += 1

    return q_dict


def get_travel_time_dist(tt_tbl):

    tt_tbl = tt_tbl.loc[:, ~tt_tbl.columns.duplicated()]
    est_cols = [c for c in tt_tbl.columns if c.endswith("_estimate") and c != "total_estimate"]

    tt_tbl = tt_tbl[["GEO_ID", "total_estimate", *est_cols]].set_index("GEO_ID")

    tt_bin_names = est_cols
    tt_to_idx = {name: i for i, name in enumerate(tt_bin_names)}

    p_dict = {}
    for o, row in tt_tbl.iterrows():
        total = row["total_estimate"]
        p_dict[o] = {tt_to_idx[col]: row[col] / total for col in tt_bin_names if row[col] > 0}

    return p_dict, tt_bin_names


def get_initial_od_dist(od_df):
    import pandas as pd

    dep_edges = [
        0,
        5 * 60,
        5 * 60 + 30,
        6 * 60,
        6 * 60 + 30,
        7 * 60,
        7 * 60 + 30,
        8 * 60,
        8 * 60 + 30,
        9 * 60,
        10 * 60,
        11 * 60,
        12 * 60,
        16 * 60,
        24 * 60,
    ]
    dep_labels = list(range(len(dep_edges) - 1))

    t = pd.to_datetime(od_df["departure_time"])
    mins = t.dt.hour * 60 + t.dt.minute

    od_df["departure_time_bin"] = pd.cut(mins, bins=dep_edges, labels=dep_labels, right=False).astype(int)

    w_dict = (
        od_df.groupby(["h_geocode", "departure_time_bin", "w_geocode"])
        .size()
        .groupby(level=0)  # normalise per origin
        .apply(lambda v: (v / v.sum()).to_dict())
        .to_dict()
    )

    return w_dict


def add_census_time_bins(df, tt_bin_names, travel_time_col="travel_time_min"):
    """
    Add a time_bin column based on travel time values matching census categories
    without removing any rows
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Census travel time bin edges
    bin_edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90, float("inf")]

    # Create time_bin column - use integers that match indices in p_dict
    labels = list(range(len(bin_edges) - 1))

    # Fill missing travel times with a default value
    # Option 1: Use median travel time
    median_time = df[travel_time_col].median()
    df[travel_time_col + "_filled"] = df[travel_time_col].fillna(median_time)

    # Create time_bin column
    df["time_bin"] = pd.cut(df[travel_time_col + "_filled"], bins=bin_edges, labels=labels, right=False)

    # Convert to integer - no dropping of NaN values
    df["time_bin"] = df["time_bin"].astype(int)

    # Optional: Remove the temporary column
    # df = df.drop(columns=[travel_time_col + '_filled'])

    # Print summary of the bins created
    bin_counts = df["time_bin"].value_counts().sort_index()
    print("Travel time bins created:")
    for i, count in bin_counts.items():
        bin_name = tt_bin_names[i].replace("_estimate", "")
        print(f"  Bin {i} ({bin_name}): {count} trips")

    return df


def fill_travel_times(routes, cand):

    # fill travel time for the missing ones
    missing = cand["_merge"] == "left_only"
    if missing.any():
        # median by O‑D over ALL departure bins
        med_tt = routes.groupby(["origin_geoid", "destination_geoid"])["travel_time_min"].median()
        cand.loc[missing, "travel_time_min"] = cand.loc[missing].apply(
            lambda r: med_tt.get((r.origin_geoid, r.destination_geoid), np.nan), axis=1
        )
        cand.loc[missing, "travel_distance_mi"] = cand.loc[missing, "travel_time_min"] / 2

    cand = cand.drop(columns=["_merge"])

    return cand


def get_all_candidates(od_df, dep_pos, routing_df, tt_bin_names):
    """Build candidate (origin, departure_bin, destination) rows.

    For large datasets the full Cartesian product can exceed available RAM.
    We now build it in *per-origin chunks* so peak memory stays bounded.
    """
    od_counts = od_df.groupby(["h_geocode", "w_geocode"]).size().reset_index(name="count")

    total_by_origin = od_counts.groupby("h_geocode")["count"].sum()
    od_counts["share"] = od_counts["count"] / od_counts["h_geocode"].map(total_by_origin)
    od_counts.loc[od_counts["share"].isna(), "share"] = 0

    od_share_df = od_counts[["h_geocode", "w_geocode", "share"]]
    od_pos = (
        od_share_df[od_share_df.share > 0]
        .groupby("h_geocode")["w_geocode"]
        .apply(list)
        .to_dict()
    )

    routing_df[["origin_geoid", "destination_geoid"]] = (
        routing_df[["origin_geoid", "destination_geoid"]].astype(str)
    )
    routes = routing_df[
        ["origin_geoid", "departure_time_bin", "destination_geoid",
         "travel_time_min", "travel_distance_mi"]
    ]

    # --- Build candidates per-origin to cap peak memory ----------------
    origins_to_process = sorted(dep_pos.keys() & od_pos.keys())
    print(f"Building candidates for {len(origins_to_process)} origins …")

    CHUNK = 200          # origins per batch
    all_chunks = []
    for chunk_start in range(0, len(origins_to_process), CHUNK):
        batch_origins = origins_to_process[chunk_start : chunk_start + CHUNK]
        needs = []
        for o in batch_origins:
            for s in dep_pos[o]:
                for d in od_pos[o]:
                    needs.append((o, s, d))
        if not needs:
            continue
        need_df = pd.DataFrame(needs, columns=["origin_geoid", "departure_time_bin", "destination_geoid"])
        # downcast to save memory
        need_df["departure_time_bin"] = need_df["departure_time_bin"].astype(np.int16)

        cand = need_df.merge(
            routes,
            on=["origin_geoid", "departure_time_bin", "destination_geoid"],
            how="left",
            indicator=True,
        )
        cand = fill_travel_times(routes, cand)
        cand = add_census_time_bins(cand, tt_bin_names)
        all_chunks.append(cand)
        del needs, need_df, cand
        gc.collect()

    cand_with_bins = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    gc.collect()

    # Downcast numerics to cut memory roughly in half
    for col in ["departure_time_bin", "time_bin"]:
        if col in cand_with_bins.columns:
            cand_with_bins[col] = cand_with_bins[col].astype(np.int16)
    for col in ["travel_time_min", "travel_distance_mi"]:
        if col in cand_with_bins.columns:
            cand_with_bins[col] = cand_with_bins[col].astype(np.float32)

    print(f"Candidates table: {len(cand_with_bins):,} rows, "
          f"{cand_with_bins.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    return cand_with_bins


def ipf_with_time_penalty(trips_o, od_probs, p_bo, max_iter=30):
    """IPF with time penalty - always produces a solution"""
    import numpy as np
    from scipy.optimize import minimize

    # Initialize weights
    weights = np.ones(len(trips_o)) / len(trips_o)

    # Create mappings for faster lookup
    dest_indices = {}
    for d in od_probs:
        mask = trips_o.destination_geoid == d
        if mask.any():
            dest_indices[d] = np.where(mask)[0]

    time_indices = {}
    for t in trips_o.departure_time_bin.unique():
        mask = trips_o.departure_time_bin == t
        if mask.any():
            time_indices[t] = np.where(mask)[0]

    tt_indices = {}
    for b in p_bo:
        mask = trips_o.time_bin == b
        if mask.any():
            tt_indices[b] = np.where(mask)[0]

    # Run IPF iterations
    for iteration in range(max_iter):
        # Step 1: Adjust for OD distribution (STRICT)
        for d, target in od_probs.items():
            if d in dest_indices and len(dest_indices[d]) > 0:
                indices = dest_indices[d]
                current = weights[indices].sum()
                if current > 0:
                    weights[indices] *= target / current

        # Step 2: Adjust for departure time distribution (STRICT)
        for t, indices in time_indices.items():
            # Get target from time distribution in data
            target = len(indices) / len(trips_o)
            current = weights[indices].sum()
            if current > 0:
                weights[indices] *= target / current

        # Step 3: Adjust for travel time, with penalty factor
        penalty_factor = 0.5  # Smaller means softer constraint
        for b, target in p_bo.items():
            if b in tt_indices and len(tt_indices[b]) > 0:
                indices = tt_indices[b]
                current = weights[indices].sum()
                if current > 0:
                    # Soft adjustment - partial step
                    ratio = (target / current) ** penalty_factor
                    weights[indices] *= ratio

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()

    return weights


def _process_single_origin(args):
    """
    Worker function to process a single origin for ILP calibration.
    Designed to be picklable for multiprocessing.
    """
    import os
    import pulp
    import numpy as np
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    o, cand_subset, od_probs, p_bo, q_so, w_od_prop, N_o, alpha, cplex_path = args

    # Seed from the origin id, not from inherited process state. Workers are
    # forked and each may handle several origins, so without this the sampling
    # below depends on how origins happen to be distributed across the pool.
    # crc32 is used rather than hash() because str hashing is salted per process.
    np.random.seed(zlib.crc32(str(o).encode()) & 0xFFFFFFFF)

    result = {"origin": o, "df": None, "status": "skipped", "error": None}

    try:
        trips_o = cand_subset.copy()
        if len(trips_o) == 0:
            result["status"] = "no_candidates"
            return result

        # Get unique destinations and departure_time_bins
        destinations = sorted(trips_o.destination_geoid.unique())
        time_bins = sorted(trips_o.departure_time_bin.unique())

        # Build OD and OS integer marginals (rounded, sum to N_o)
        od_targets = np.array([od_probs.get(d, 0) for d in destinations])
        if od_targets.sum() > 0:
            od_targets = np.round(od_targets / od_targets.sum() * N_o).astype(int)
            od_targets[-1] += N_o - od_targets.sum()
        else:
            result["status"] = "no_od_targets"
            return result

        os_targets = np.array([q_so.get(s, 0) for s in time_bins])
        if os_targets.sum() > 0:
            os_targets = np.round(os_targets / os_targets.sum() * N_o).astype(int)
            os_targets[-1] += N_o - os_targets.sum()
        else:
            result["status"] = "no_os_targets"
            return result

        # Build ILP
        prob = pulp.LpProblem(f"ILP_Cal_{o}", pulp.LpMinimize)

        # Integer variable for each (s, d) cell: number of trips assigned
        var = {}
        for s in time_bins:
            for d in destinations:
                key = (s, d)
                if ((trips_o.departure_time_bin == s) & (trips_o.destination_geoid == d)).any():
                    var[key] = pulp.LpVariable(f"x_{s}_{d}", lowBound=0, cat="Integer")

        if not var:
            result["status"] = "no_variables"
            return result

        # Travel-time slacks (soft constraint)
        e_plus = {b: pulp.LpVariable(f"ep_{b}", lowBound=0) for b in p_bo}
        e_minus = {b: pulp.LpVariable(f"em_{b}", lowBound=0) for b in p_bo}

        # Convert w_od_prop to expected counts
        w_od = {}
        for s in time_bins:
            for d in destinations:
                w_od[(s, d)] = w_od_prop.get((s, d), 0) * N_o

        # L1 slack variables for OD deviation
        g_plus = {}
        g_minus = {}
        for key in var:
            g_plus[key] = pulp.LpVariable(f"gplus_{key[0]}_{key[1]}", lowBound=0)
            g_minus[key] = pulp.LpVariable(f"gminus_{key[0]}_{key[1]}", lowBound=0)

        # Add L1 deviation constraints
        for key in var:
            prob += var[key] - w_od[key] == g_minus[key] - g_plus[key]

        # Objective
        prob += pulp.lpSum(e_plus[b] + e_minus[b] for b in p_bo) + alpha * pulp.lpSum(
            g_plus[key] + g_minus[key] for key in var
        )

        # Constraints
        prob += pulp.lpSum(var.values()) == N_o

        for i, d in enumerate(destinations):
            prob += pulp.lpSum(var[(s, d)] for s in time_bins if (s, d) in var) == od_targets[i]

        for j, s in enumerate(time_bins):
            prob += pulp.lpSum(var[(s, d)] for d in destinations if (s, d) in var) == os_targets[j]

        # pi[(s, d)][b] = fraction of the candidates in cell (s, d) whose travel
        # time falls in bin b.  sum_b pi[(s, d)][b] == 1, so summing this
        # constraint over b reduces to sum(var) == N_o, matching the right-hand
        # sides.  Using a 0/1 indicator instead would add each cell at full
        # weight to every bin it touches, making the constraints unsatisfiable
        # by construction and forcing the excess into the slacks.
        cell_sizes = trips_o.groupby(["departure_time_bin", "destination_geoid"]).size()
        cell_bin_counts = trips_o.groupby(["departure_time_bin", "destination_geoid", "time_bin"]).size()

        pi = {}
        for (s, d, b), n in cell_bin_counts.items():
            total = cell_sizes[(s, d)]
            if total > 0:
                pi.setdefault(b, {})[(s, d)] = n / total

        for b, pbo in p_bo.items():
            terms = [coef * var[key] for key, coef in pi.get(b, {}).items() if key in var]
            if terms:
                prob += pulp.lpSum(terms) + e_minus[b] - e_plus[b] == int(round(pbo * N_o))

        # Solve
        status = None
        
        if cplex_path != None:
            try:
                solver = pulp.CPLEX_CMD(path=str(cplex_path), msg=False, timeLimit=30)
                status = prob.solve(solver)
            except Exception:                
                status = None

        if status is None:
            try:
                solver = pulp.HiGHS(msg=False, timeLimit=30)
                status = prob.solve(solver)
            except Exception:
                status = None

        if status is None:
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
            status = prob.solve(solver)

        if status == pulp.LpStatusOptimal:
            trips_o["calibrated_weight"] = 0.0
            for key, v in var.items():
                count = int(round(v.value()))
                s, d = key
                candidates = trips_o[(trips_o.departure_time_bin == s) & (trips_o.destination_geoid == d)]
                if len(candidates) == 0 or count == 0:
                    continue
                chosen_idx = np.random.choice(candidates.index, size=count, replace=True)
                for idx in chosen_idx:
                    trips_o.loc[idx, "calibrated_weight"] += 1
            result["df"] = trips_o
            result["status"] = "success"
        else:
            # Fallback to IPF
            weights = ipf_with_time_penalty(trips_o, od_probs, p_bo)
            if weights is not None:
                trips_o["calibrated_weight"] = weights * N_o
                result["df"] = trips_o
                result["status"] = "fallback"
            else:
                result["status"] = "failed"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"

    return result


def _safe_n_workers(cand_mem_bytes, n_workers_requested):
    """Throttle workers so forked children don't blow past available RAM.

    Each fork-ed worker inherits a copy-on-write view of the parent.
    In practice the child touches most pages (pandas / numpy / pulp),
    so we budget ~(cand_mem + 0.5 GB) per worker.
    """
    avail = _available_gb()
    per_worker_gb = max(cand_mem_bytes / 1e9 + 0.5, 1.0)
    safe = max(1, int(avail // per_worker_gb))
    chosen = min(n_workers_requested, safe)
    if chosen < n_workers_requested:
        print(f"[mem-guard] Throttling workers {n_workers_requested} → {chosen} "
              f"(avail {avail:.1f} GB, ~{per_worker_gb:.1f} GB/worker)")
    return chosen


def calibrate_with_strict_od_time_ilp(cand, od_df, p_dict, q_dict, w_dict, lodes_dict, alpha=1.0, n_workers=None):
    """
    Integer calibration: assigns integer counts to (departure_time_bin, destination) cells
    to strictly match OD and departure time marginals for each origin,
    with an L1 penalty for deviation from the initial ODS distribution.

    Parallelized version using multiprocessing for Streamlit compatibility.

    Large datasets are processed in *batches* so we never hold all work_items
    in memory at once, and the worker pool is memory-throttled.

    Parameters
    ----------
    n_workers : int, optional
        Number of parallel workers. Defaults to CPU count - 1, capped by
        available RAM.  Set to 1 for sequential.
    """
    import os
    import pulp
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import multiprocessing as mp

    od_distribution = calculate_od_distribution(od_df, lodes_dict)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # Get all origins to process
    origins = [
        o for o in cand.origin_geoid.unique()
        if o in od_distribution and o in p_dict and o in lodes_dict and int(lodes_dict[o]) > 0
    ]

    cplex_path = os.environ.get("CPLEX_PATH", "")

    # Memory-aware worker throttling
    cand_mem = cand.memory_usage(deep=True).sum()
    n_workers = _safe_n_workers(cand_mem, n_workers)

    print(f"Calibrating {len(origins)} origins  "
          f"(cand {cand_mem / 1e6:.0f} MB, {n_workers} workers, "
          f"RSS {_mem_gb():.1f} GB, avail {_available_gb():.1f} GB)")

    calibrated_dfs = []
    success_count = 0
    fallback_count = 0

    # --- Process in batches to bound peak memory ----------------------
    BATCH = 200  # origins per batch
    for batch_start in range(0, len(origins), BATCH):
        batch_origins = origins[batch_start : batch_start + BATCH]

        work_items = []
        for o in batch_origins:
            N_o = int(lodes_dict[o])
            cand_subset = cand[cand.origin_geoid == o].copy()
            if len(cand_subset) == 0:
                continue
            od_probs = od_distribution[o]
            p_bo = p_dict[o]
            q_so = q_dict[o]
            w_od_prop = w_dict.get(o, {}) if w_dict is not None else {}
            work_items.append((o, cand_subset, od_probs, p_bo, q_so, w_od_prop, N_o, alpha, cplex_path))

        if not work_items:
            continue

        batch_label = (f"Batch {batch_start // BATCH + 1}/"
                       f"{(len(origins) + BATCH - 1) // BATCH}")

        if n_workers > 1 and len(work_items) > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_process_single_origin, work_items, chunksize=1),
                    total=len(work_items),
                    desc=f"ILP {batch_label}",
                ))
        else:
            results = [
                _process_single_origin(args)
                for args in tqdm(work_items, desc=f"ILP {batch_label}")
            ]

        for res in results:
            if res["df"] is not None:
                calibrated_dfs.append(res["df"])
                if res["status"] == "success":
                    success_count += 1
                elif res["status"] == "fallback":
                    fallback_count += 1
            elif res["status"] == "error":
                print(f"Error with origin {res['origin']}: {res['error']}")

        # Free batch memory
        del work_items, results
        gc.collect()

    # Concatenate all results
    if calibrated_dfs:
        result = pd.concat(calibrated_dfs, ignore_index=True)
        del calibrated_dfs
        gc.collect()
        print(f"Successfully calibrated {success_count} origins with ILP")
        print(f"Used fallback method for {fallback_count} origins")
        print(f"Total trips: {len(result)}")
        return result
    else:
        print("No origins were successfully calibrated")
        return pd.DataFrame()


def post_calibrating_assignment(calibrated_trips, origin_buildings, dest_buildings, ms_buildings_df):
    # Seed both generators up front: sample_departure_time below uses the
    # `random` module, which was never seeded here, while np.random.seed further
    # down ran only after that sampling had already happened.
    random.seed(42)
    np.random.seed(42)

    df = calibrated_trips[calibrated_trips["calibrated_weight"] > 0].copy()

    # Vectorised expansion — avoids the O(N×weight) Python loop that
    # previously caused OOM on large datasets.
    weights = df["calibrated_weight"].astype(int).values
    repeat_idx = np.repeat(np.arange(len(df)), weights)
    synthetic_df = df.iloc[repeat_idx].reset_index(drop=True)

    departure_time_bins = [
        (0, 299),
        (300, 329),
        (330, 359),
        (360, 389),
        (390, 419),
        (420, 449),
        (450, 479),
        (480, 509),
        (510, 539),
        (540, 599),
        (600, 659),
        (660, 719),
        (720, 959),
        (960, 1439),
    ]

    def sample_departure_time(bin_idx):
        start, end = departure_time_bins[bin_idx]
        depart_time = random.randint(start, end) * 60
        return depart_time
        # return np.random.randint(start, end + 1)

    synthetic_df["departure_time"] = synthetic_df["departure_time_bin"].apply(sample_departure_time)
    synthetic_df["departure_datetime"] = pd.to_datetime("2025-03-10") + pd.to_timedelta(
        synthetic_df["departure_time"], unit="s"
    )

    home_locs = []
    for o, grp in synthetic_df.groupby("origin_geoid"):
        houses = origin_buildings[origin_buildings.GEOID == str(o)]
        if len(houses) == 0:
            if len(houses) == 0:
                # If no buildings, sample from ms_buildings_df and extract lat/lon from 'location'
                tmp = ms_buildings_df.sample(n=len(grp), replace=True).reset_index(drop=True)
                tmp.index = grp.index
                tmp_latlon = tmp["location"].apply(lambda loc: pd.Series({"origin_lat": loc[0], "origin_lon": loc[1]}))
                tmp = tmp_latlon  # Keep this line, remove the next one
        else:
            # sample with replacement
            samp = houses.sample(n=len(grp), replace=True).reset_index(drop=True)
            samp.index = grp.index
            tmp = samp[["lat", "lon"]].rename(columns={"lat": "origin_lat", "lon": "origin_lon"})
        home_locs.append(tmp)
    home_locs = pd.concat(home_locs)
    synthetic_df = synthetic_df.join(home_locs)

    # 4. For each destination CBG, sample a work location
    work_locs = []
    for d, grp in synthetic_df.groupby("destination_geoid"):
        offices = dest_buildings[dest_buildings.GEOID == str(d)]
        if len(offices) == 0:
            tmp = ms_buildings_df.sample(n=len(grp), replace=True).reset_index(drop=True)
            tmp.index = grp.index
            tmp_latlon = tmp["location"].apply(
                lambda loc: pd.Series({"destination_lat": loc[0], "destination_lon": loc[1]})
            )
            tmp = tmp_latlon  # Keep this, remove the next line
        else:
            samp = offices.sample(n=len(grp), replace=True).reset_index(drop=True)
            samp.index = grp.index
            tmp = samp[["lat", "lon"]].rename(columns={"lat": "destination_lat", "lon": "destination_lon"})
        work_locs.append(tmp)
    work_locs = pd.concat(work_locs)

    synthetic_df = synthetic_df.join(work_locs)

    return synthetic_df


def calibrate_with_ilp(
    od_df,
    routing_df,
    res_locations,
    combined_work_locations,
    ms_buildings_df,
    census_depart_times_df,
    travel_time_to_work_by_geoid,
):

    dep_tbl = census_depart_times_df.set_index("GEO_ID")
    dep_pos = {
        o: [i for i, col in enumerate(dep_tbl.columns[2:-4:2]) if dep_tbl.loc[o, col] > 0] for o in dep_tbl.index
    }
    empty_keys = [key for key, value in dep_pos.items() if not value]
    od_df = od_df[~od_df["h_geocode"].isin(empty_keys)]

    origin_buildings = res_locations
    dest_buildings = combined_work_locations

    home_pts = origin_buildings.apply(lambda row: extract_latlon(row["location"]), axis=1)
    origin_buildings = pd.concat([origin_buildings, home_pts], axis=1)

    work_pts = dest_buildings.apply(lambda row: extract_latlon(row["location"]), axis=1)
    dest_buildings = pd.concat([dest_buildings, work_pts], axis=1)

    odt = od_df.groupby(["h_geocode", "w_geocode"])["total_jobs"].first().reset_index()[["h_geocode", "total_jobs"]]
    lodes_dict = odt.groupby(["h_geocode"])["total_jobs"].sum().to_dict()

    q_dict = get_departure_time_dist(census_depart_times_df.copy())
    p_dict, tt_bin_names = get_travel_time_dist(travel_time_to_work_by_geoid.copy())
    w_dict = get_initial_od_dist(od_df)

    cand_with_bins = get_all_candidates(od_df, dep_pos, routing_df, tt_bin_names)

    # Run calibration with strict OD and time distribution
    calibrated_trips = calibrate_with_strict_od_time_ilp(
        cand=cand_with_bins, od_df=od_df, p_dict=p_dict, q_dict=q_dict, w_dict=w_dict, lodes_dict=lodes_dict, alpha=1
    )

    calibrated_trips = post_calibrating_assignment(calibrated_trips, origin_buildings, dest_buildings, ms_buildings_df)

    return calibrated_trips
