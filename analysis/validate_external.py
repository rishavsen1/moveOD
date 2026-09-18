"""Held-out external validation for Move-OD synthetic trips.

`analysis/figures_from_output.py` compares the synthetic trips against ACS
B08302 (time of departure) and B08303 (travel time to work). Both tables are
pipeline *inputs*, so that comparison is circular. This script compares the same
trips against ACS tables the pipeline never reads -- B08602 (time arriving at
work, workplace geography), B08603 (travel time to work, workplace geography)
and B08133 (aggregate travel time by time of departure) -- and against two
deterministic baselines, so the ILP calibration has to earn its fidelity.

Usage:
    python analysis/validate_external.py --state Tennessee --county Hamilton \
        --run-id 2025-03-17_2025-03-17
    python analysis/validate_external.py --all-counties --skip-baselines

Caveat discovered while building this (ACS5 2019-2023, checked 2026-09-18):
none of the three held-out tables is published at block-group level. The API
returns one row per block group, but every estimate on those rows is null.
B08602 and B08603 carry data only at county level; B08133 is published for a
minority of tracts (9 of 87 in Hamilton TN) and for the county. Each table is
therefore compared at the finest geography that actually carries data, and the
`geography` column of metrics.csv records which one that was. The geography
loop is generic and picks up finer geographies automatically if they appear.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.figures_from_output import (  # noqa: E402
    TT_BIN_EDGES,
    TT_BIN_LABELS,
    find_calibrated_csv,
    find_census_table,
    find_initial_frame,
    find_run_dir,
    pick_first_col,
    read_table,
)

LOGGER = logging.getLogger(__name__)

ACS_YEAR = 2021
ACS_API_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5"

# B08302/B08602 departure/arrival bins, in minutes since midnight, right-open.
DEP_BIN_EDGES = [0, 300, 330, 360, 390, 420, 450, 480, 510, 540, 600, 660, 720, 960, 1440]
DEP_BIN_LABELS = [
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
DEP_BIN_MIDPOINTS = [150.0, 315.0, 345.0, 375.0, 405.0, 435.0, 465.0, 495.0, 525.0, 570.0, 630.0, 690.0, 840.0, 1200.0]
# Midpoints of the ACS travel-time bins; the open-ended 90+ bin is charged 105 min.
TT_BIN_MIDPOINTS = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 52.5, 75.0, 105.0]
DEP_ACS_COLUMNS = [f"{label}_estimate" for label in DEP_BIN_LABELS]

# calibrated_move_od/ also holds sub-samples and helper tables; a real day is
# "<date>", "<date>_<n>" or "<County>_<date>".
CALIB_STEM_RE = re.compile(r"^(?:.*_)?(\d{4}-\d{2}-\d{2})(?:_\d+)?$")
GEO_LENGTH = {"bg": 12, "tract": 11, "county": 5}
GEO_LEVELS = ("bg", "tract", "county")
ACS_GEO_FOR = {"bg": "block group:*", "tract": "tract:*", "county": None}
# Dense IPF seed is (origins x destinations x 14); skip the baseline above this.
MAX_IPF_CELLS = 5e7
MIN_JOINT_WORKERS = 20
METHOD_COLORS = {
    "calibrated": "#E45756",
    "initial": "#4C78A8",
    "ipf": "#54A24B",
    "uniform": "#9D755D",
    "acs": "#F4A261",
}


# --------------------------------------------------------------------------- geoids


def normalize_geoid(values: pd.Series) -> pd.Series:
    """Zero-pad geoids to 12 characters (int64 round-trips drop leading zeros)."""
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(12)


def truncate_geoid(geoids: pd.Series, level: str) -> pd.Series:
    """Truncate 12-char block-group geoids to a coarser census geography."""
    if level not in GEO_LENGTH:
        raise ValueError(f"Unknown geography level: {level!r}")
    return geoids.str[: GEO_LENGTH[level]]


# --------------------------------------------------------------------------- binning


def bin_departure_minutes(minutes: pd.Series) -> pd.Series:
    """Bin minutes-since-midnight into the 14 ACS B08302/B08602 bins."""
    codes = pd.cut(pd.to_numeric(minutes, errors="coerce"), bins=DEP_BIN_EDGES, right=False, labels=False)
    return pd.Series(codes, index=minutes.index).astype("Float64").astype("Int64")


def bin_travel_minutes(minutes: pd.Series) -> pd.Series:
    """Bin travel minutes into the 12 ACS B08303/B08603 bins (right-open)."""
    return pd.cut(
        pd.to_numeric(minutes, errors="coerce"),
        bins=TT_BIN_EDGES,
        labels=TT_BIN_LABELS,
        right=False,
        include_lowest=True,
    )


def travel_bin_codes(minutes: pd.Series) -> pd.Series:
    """Integer bin index (0..11) for travel minutes, <NA> outside the bins."""
    codes = bin_travel_minutes(minutes).cat.codes
    return pd.Series(codes, index=minutes.index).astype("Int64").replace(-1, pd.NA)


# --------------------------------------------------------------------------- metrics


def _as_probabilities(counts: np.ndarray | pd.Series) -> np.ndarray | None:
    arr = np.asarray(counts, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    total = arr.sum()
    return None if total <= 0 else arr / total


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """0.5 * sum |p - q| on the normalised distributions."""
    pp, qq = _as_probabilities(p), _as_probabilities(q)
    if pp is None or qq is None:
        return float("nan")
    return float(0.5 * np.abs(pp - qq).sum())


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance in base 2 (0 identical, 1 disjoint)."""
    pp, qq = _as_probabilities(p), _as_probabilities(q)
    if pp is None or qq is None:
        return float("nan")
    return float(jensenshannon(pp, qq, base=2))


def wasserstein1(p: np.ndarray, q: np.ndarray, midpoints: np.ndarray) -> float:
    """Wasserstein-1 distance between two binned distributions, in bin units."""
    pp, qq = _as_probabilities(p), _as_probabilities(q)
    if pp is None or qq is None:
        return float("nan")
    mids = np.asarray(midpoints, dtype=float)
    return float(wasserstein_distance(mids, mids, pp, qq))


def distribution_metrics(syn: np.ndarray, acs: np.ndarray, midpoints: np.ndarray) -> dict[str, float]:
    return {
        "tvd": total_variation_distance(syn, acs),
        "js": js_distance(syn, acs),
        "w1": wasserstein1(syn, acs, midpoints),
    }


# --------------------------------------------------------------------------- trip frames


def _minutes_from_column(series: pd.Series) -> pd.Series:
    """Minutes since midnight from a timestamp, or from a numeric time-of-day column.

    Numeric `departure_time` columns are seconds-since-midnight in some runs and
    minutes-since-midnight in others, so the unit is inferred from the range.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        stamps = series
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.5:
            return numeric / (60.0 if numeric.max(skipna=True) > 1440 else 1.0)
        stamps = pd.to_datetime(series, errors="coerce")
    return stamps.dt.hour * 60.0 + stamps.dt.minute + stamps.dt.second / 60.0


def to_trip_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise any Move-OD trip table to origin/dest/departure/travel/arrival.

    Handles both the current calibrated schema (seconds-since-midnight
    `departure_time`, no arrival column) and the older schema that carries
    timestamps plus `arrival_time`. The arrival minute is always derived as
    departure + travel time so every method is treated identically; where the
    table also carries an arrival column, the disagreement is logged.
    """
    origin_col = pick_first_col(frame, ["origin_geoid", "origin_bg", "h_geocode"])
    dest_col = pick_first_col(frame, ["destination_geoid", "dest_geoid", "dest_bg", "w_geocode"])
    dep_col = pick_first_col(frame, ["departure_datetime", "departure_time", "go_time"])
    tt_col = pick_first_col(frame, ["travel_time_min_filled", "travel_time_min", "travel_time_minutes"])
    missing = [name for name, col in
               [("origin", origin_col), ("destination", dest_col), ("departure", dep_col), ("travel time", tt_col)]
               if col is None]
    if missing:
        raise KeyError(f"Trip table is missing column(s) for: {', '.join(missing)}")

    trips = pd.DataFrame(
        {
            "origin_bg": normalize_geoid(frame[origin_col]),
            "dest_bg": normalize_geoid(frame[dest_col]),
            "dep_min": _minutes_from_column(frame[dep_col]) % 1440.0,
            "tt_min": pd.to_numeric(frame[tt_col], errors="coerce"),
        }
    )
    trips["arr_min"] = (trips["dep_min"] + trips["tt_min"]) % 1440.0
    arr_col = pick_first_col(frame, ["arrival_time", "arrival_datetime"])
    if arr_col is not None:
        stored = _minutes_from_column(frame[arr_col]) % 1440.0
        drift = (stored - trips["arr_min"]).abs()
        LOGGER.info("Derived arrival differs from %s by at most %.3f min", arr_col, drift.max(skipna=True))
    return trips.dropna(subset=["dep_min", "tt_min", "arr_min"]).reset_index(drop=True)


# --------------------------------------------------------------------------- held-out ACS


def _acs_request(
    table: str, state_fips: str, county_fips: str, geo_level: str, acs_year: int = ACS_YEAR
) -> pd.DataFrame | None:
    """One ACS group() request. Never logs the API key."""
    from generate.config import CENSUS_API_KEY

    geo_for = ACS_GEO_FOR[geo_level] or f"county:{county_fips}"
    geo_in = f"state:{state_fips}" if geo_level == "county" else f"state:{state_fips} county:{county_fips}"
    params = {"get": f"group({table})", "for": geo_for, "in": geo_in}
    if CENSUS_API_KEY and not CENSUS_API_KEY.startswith(("NO_KEY", "YOUR_")):
        params["key"] = CENSUS_API_KEY

    try:
        response = requests.get(ACS_API_TEMPLATE.format(year=acs_year), params=params, timeout=120)
    except requests.RequestException as exc:
        LOGGER.warning("ACS %s %d at %s level failed: %s", table, acs_year, geo_level, exc)
        return None
    LOGGER.info("ACS request: %s", re.sub(r"([?&]key=)[^&]*", r"\1<redacted>", response.url))
    if response.status_code != 200:
        LOGGER.warning("ACS %s %d at %s level returned HTTP %s", table, acs_year, geo_level,
                       response.status_code)
        return None
    payload = response.json()
    return pd.DataFrame(payload[1:], columns=payload[0])


def _acs_geoid(raw: pd.DataFrame) -> pd.Series:
    return raw["GEO_ID"].astype(str).str.split("US").str[-1]


def load_acs_table(
    table: str, n_bins: int, state_fips: str, county_fips: str, geo_level: str, cache_dir: Path,
    acs_year: int = ACS_YEAR,
) -> pd.DataFrame | None:
    """Load one ACS table at `geo_level`, cached as parquet under the run dir.

    Returns a frame indexed by geoid with integer bin columns 0..n_bins-1 plus a
    "total" column, or None when the table carries no data at that geography
    (the API returns all-null rows for workplace-geography tables below county).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{table}_{geo_level}_{acs_year}.parquet"
    if cache_path.exists():
        raw = pd.read_parquet(cache_path)
    else:
        raw = _acs_request(table, state_fips, county_fips, geo_level, acs_year)
        if raw is None:
            return None
        raw.to_parquet(cache_path, index=False)

    columns = [f"{table}_{index:03d}E" for index in range(1, n_bins + 2)]
    if not set(columns).issubset(raw.columns):
        LOGGER.warning("ACS %s %d at %s level is missing expected variables", table, acs_year, geo_level)
        return None
    values = raw[columns].apply(pd.to_numeric, errors="coerce")
    values = values.mask(values < 0)  # -666666666 and friends are ACS jam values
    frame = values.iloc[:, 1:].copy()
    frame.columns = range(n_bins)
    frame["total"] = values.iloc[:, 0]
    frame.index = _acs_geoid(raw)
    frame = frame.dropna(how="all")
    if frame.empty or float(frame["total"].fillna(0).sum()) <= 0:
        return None
    return frame.fillna(0.0)


def resolve_acs_table(
    table: str, n_bins: int, state_fips: str, county_fips: str, cache_dir: Path, acs_year: int = ACS_YEAR
) -> tuple[pd.DataFrame, str] | None:
    """Return the table at the finest census geography that actually carries data."""
    for level in GEO_LEVELS:
        frame = load_acs_table(table, n_bins, state_fips, county_fips, level, cache_dir, acs_year)
        if frame is not None:
            LOGGER.info("ACS %s %d resolved at %s level (%d units)", table, acs_year, level, len(frame))
            return frame, level
        LOGGER.info("ACS %s %d carries no data at %s level", table, acs_year, level)
    LOGGER.warning("ACS %s %d is unavailable at every geography tried", table, acs_year)
    return None


def acs_county_total(
    table: str, n_bins: int, state_fips: str, county_fips: str, cache_dir: Path, acs_year: int = ACS_YEAR
) -> float:
    """County-level total for one ACS table, NaN when the fetch failed."""
    frame = load_acs_table(table, n_bins, state_fips, county_fips, "county", cache_dir, acs_year)
    return float("nan") if frame is None else float(frame["total"].sum())


def aggregate_acs(acs: pd.DataFrame, source_level: str, target_level: str) -> pd.DataFrame | None:
    """Roll an ACS table up from its published geography to a coarser one."""
    if GEO_LENGTH[target_level] > GEO_LENGTH[source_level]:
        return None
    units = acs.index.str[: GEO_LENGTH[target_level]]
    return acs.groupby(units).sum()


def comparable_levels(source_level: str) -> list[str]:
    return [level for level in GEO_LEVELS if GEO_LENGTH[level] <= GEO_LENGTH[source_level]]


# --------------------------------------------------------------------------- tests


def synthetic_bin_counts(trips: pd.DataFrame, geo_col: str, level: str, codes: pd.Series, n_bins: int) -> pd.DataFrame:
    """Trip counts per geography unit x bin, reindexed to the full bin set."""
    valid = codes.notna()
    units = truncate_geoid(trips.loc[valid, geo_col], level)
    counts = pd.crosstab(units, codes[valid].astype(int))
    return counts.reindex(columns=range(n_bins), fill_value=0).fillna(0)


def compare_at_level(
    syn: pd.DataFrame, acs: pd.DataFrame, midpoints: list[float]
) -> tuple[dict[str, float], pd.DataFrame]:
    """Per-unit and pooled distribution metrics for one method at one geography."""
    with_acs = acs.index[acs[list(range(len(midpoints)))].sum(axis=1) > 0]
    syn = syn.reindex(with_acs, fill_value=0).fillna(0)
    compared = with_acs[syn.sum(axis=1) > 0]
    if len(compared) == 0:
        empty = {"units_with_acs": float(len(with_acs)), "units_compared": 0.0, "coverage": 0.0}
        return empty, pd.DataFrame(columns=["tvd", "js", "w1"])

    bins = list(range(len(midpoints)))
    per_unit = pd.DataFrame(
        [distribution_metrics(syn.loc[unit, bins].to_numpy(), acs.loc[unit, bins].to_numpy(), midpoints)
         for unit in compared],
        index=compared,
    )
    weights = acs.loc[compared, bins].sum(axis=1).to_numpy()
    results: dict[str, float] = {
        "units_with_acs": float(len(with_acs)),
        "units_compared": float(len(compared)),
        "coverage": float(len(compared) / len(with_acs)),
    }
    pooled = distribution_metrics(
        syn.loc[compared, bins].sum().to_numpy(), acs.loc[compared, bins].sum().to_numpy(), midpoints
    )
    for metric in ("tvd", "js", "w1"):
        values = per_unit[metric].to_numpy(dtype=float)
        finite = np.isfinite(values)
        results[f"{metric}_pooled"] = pooled[metric]
        results[f"{metric}_wmean"] = float(np.average(values[finite], weights=weights[finite])) if finite.any() else float("nan")
        for pct in (10, 50, 90):
            results[f"{metric}_p{pct}"] = float(np.percentile(values[finite], pct)) if finite.any() else float("nan")
    return results, per_unit


def run_distribution_test(
    test_name: str,
    trips_by_method: dict[str, pd.DataFrame],
    acs: pd.DataFrame,
    acs_level: str,
    geo_col: str,
    value_col: str,
    bin_fn: Callable[[pd.Series], pd.Series],
    midpoints: list[float],
) -> tuple[list[dict], dict]:
    """Compare every method's binned distribution against one held-out ACS table."""
    rows: list[dict] = []
    pooled_counts: dict[str, pd.Series] = {}
    unit_tvd: dict[str, pd.Series] = {}
    finest = comparable_levels(acs_level)[0]
    n_bins = len(midpoints)
    for method, trips in trips_by_method.items():
        codes = bin_fn(trips[value_col])
        for level in comparable_levels(acs_level):
            syn = synthetic_bin_counts(trips, geo_col, level, codes, n_bins)
            results, per_unit = compare_at_level(syn, aggregate_acs(acs, acs_level, level), midpoints)
            n_units = int(results.get("units_compared", 0))
            rows += [
                {"method": method, "test": test_name, "geography": level, "metric": key,
                 "value": value, "n_units": n_units}
                for key, value in results.items()
            ]
            if level == finest and not per_unit.empty:
                unit_tvd[method] = per_unit["tvd"]
            if level == "county":
                pooled_counts[method] = syn.sum()
    pooled_counts["acs"] = aggregate_acs(acs, acs_level, "county")[list(range(n_bins))].sum()
    return rows, {"pooled": pooled_counts, "unit_tvd": unit_tvd}


def run_joint_test(
    trips_by_method: dict[str, pd.DataFrame],
    aggregate_time: pd.DataFrame,
    agg_level: str,
    departures: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """Mean travel minutes per origin unit x departure bin vs B08133 / B08302."""
    workers = aggregate_acs(departures, "bg", agg_level)
    if workers is None:
        return [], pd.DataFrame()
    bins = list(range(len(DEP_BIN_MIDPOINTS)))
    shared = aggregate_time.index.intersection(workers.index)
    # A zero aggregate means the ACS cell was suppressed (nulls are read as 0), not a
    # zero-minute commute, so those cells are dropped rather than compared.
    acs_mean = (aggregate_time.loc[shared, bins] / workers.loc[shared, bins]).where(
        (workers.loc[shared, bins] >= MIN_JOINT_WORKERS) & (aggregate_time.loc[shared, bins] > 0)
    )
    reference = acs_mean.stack().rename("acs_mean").reset_index()
    reference.columns = ["unit", "dep_bin", "acs_mean"]
    reference["workers"] = [workers.loc[u, b] for u, b in zip(reference["unit"], reference["dep_bin"])]
    reference = reference.dropna(subset=["acs_mean"])

    rows: list[dict] = []
    scatter: list[pd.DataFrame] = []
    for method, trips in trips_by_method.items():
        frame = pd.DataFrame(
            {"unit": truncate_geoid(trips["origin_bg"], agg_level),
             "dep_bin": bin_departure_minutes(trips["dep_min"]), "tt_min": trips["tt_min"]}
        ).dropna()
        syn = frame.groupby(["unit", "dep_bin"])["tt_min"].mean().rename("syn_mean").reset_index()
        syn["dep_bin"] = syn["dep_bin"].astype(int)
        merged = reference.merge(syn, on=["unit", "dep_bin"], how="inner")
        merged["method"] = method
        rows += _joint_metrics(method, merged, agg_level)
        scatter.append(merged)
    return rows, pd.concat(scatter, ignore_index=True) if scatter else pd.DataFrame()


def _joint_metrics(method: str, merged: pd.DataFrame, geography: str) -> list[dict]:
    """Weighted R-squared and bias (synthetic - ACS) overall and per departure bin."""
    if merged.empty:
        return [{"method": method, "test": "joint_dep_tt", "geography": geography,
                 "metric": "n_cells", "value": 0.0, "n_units": 0}]
    weights = merged["workers"].to_numpy(dtype=float)
    acs = merged["acs_mean"].to_numpy(dtype=float)
    syn = merged["syn_mean"].to_numpy(dtype=float)
    mean_acs = float(np.average(acs, weights=weights))
    ss_res = float(np.sum(weights * (syn - acs) ** 2))
    ss_tot = float(np.sum(weights * (acs - mean_acs) ** 2))
    rows = [
        {"metric": "r2_weighted", "value": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")},
        {"metric": "bias_mean", "value": float(np.average(syn - acs, weights=weights))},
        {"metric": "rmse_weighted", "value": float(np.sqrt(ss_res / weights.sum()))},
        {"metric": "n_cells", "value": float(len(merged))},
    ]
    for dep_bin, group in merged.groupby("dep_bin"):
        bias = np.average(group["syn_mean"] - group["acs_mean"], weights=group["workers"])
        rows.append({"metric": f"bias_{DEP_BIN_LABELS[int(dep_bin)]}", "value": float(bias)})
    for row in rows:
        row.update({"method": method, "test": "joint_dep_tt", "geography": geography,
                    "n_units": int(merged["unit"].nunique())})
    return rows


# --------------------------------------------------------------------------- baselines


def fit_ipf(
    seed: np.ndarray, od_target: np.ndarray, os_target: np.ndarray, max_iter: int = 200, tol: float = 1e-6
) -> tuple[np.ndarray, int, float]:
    """Iterative proportional fitting of an (origin, destination, departure bin) table.

    Fits the two marginals the ILP also enforces: `od_target[o, d]` (LODES) and
    `os_target[o, s]` (LODES origin total x B08302 departure shares). Returns the
    fitted table, the iterations used and the final max absolute marginal error.
    """
    table = np.asarray(seed, dtype=float).copy()
    od = np.asarray(od_target, dtype=float)
    departures = np.asarray(os_target, dtype=float)
    iteration, error = 0, float("inf")
    for iteration in range(1, max_iter + 1):
        current = table.sum(axis=2)
        table *= np.divide(od, current, out=np.zeros_like(od), where=current > 0)[:, :, None]
        current = table.sum(axis=1)
        table *= np.divide(departures, current, out=np.zeros_like(departures), where=current > 0)[:, None, :]
        error = float(max(np.abs(table.sum(axis=2) - od).max(), np.abs(table.sum(axis=1) - departures).max()))
        if error < tol:
            break
    return table, iteration, error


def largest_remainder_round(weights: np.ndarray, total: int) -> np.ndarray:
    """Round fractional cell weights to integers that sum exactly to `total`."""
    arr = np.asarray(weights, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    if total <= 0 or arr.sum() <= 0:
        return np.zeros(arr.shape, dtype=np.int64)
    scaled = arr * (total / arr.sum())
    counts = np.floor(scaled).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _sample_travel_times(pools: dict, origin: str, dest: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw travel minutes from the (o, d) pool, falling back to origin then global."""
    for key in ((origin, dest), origin, "*"):
        pool = pools.get(key)
        if pool is not None and len(pool):
            return rng.choice(pool, size=size, replace=True)
    return np.full(size, np.nan)


def expand_cells_to_trips(
    table: np.ndarray, origins: list[str], dests: list[str], travel_pools: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Turn a fitted (o, d, s) table into one row per trip."""
    array = np.asarray(table, dtype=float)
    counts = largest_remainder_round(array.ravel(), int(round(float(array.sum())))).reshape(array.shape)
    o_idx, d_idx, s_idx = np.nonzero(counts)
    cell_counts = counts[o_idx, d_idx, s_idx]
    if cell_counts.sum() == 0:
        return pd.DataFrame(columns=["origin_bg", "dest_bg", "dep_min", "tt_min", "arr_min"])

    rep_o = np.repeat(o_idx, cell_counts)
    rep_d = np.repeat(d_idx, cell_counts)
    rep_s = np.repeat(s_idx, cell_counts)
    dep = rng.uniform(np.asarray(DEP_BIN_EDGES[:-1], dtype=float)[rep_s],
                      np.asarray(DEP_BIN_EDGES[1:], dtype=float)[rep_s])

    travel = np.full(len(rep_o), np.nan)
    pair_codes = rep_o.astype(np.int64) * len(dests) + rep_d
    order = np.argsort(pair_codes, kind="stable")
    for chunk in np.split(order, np.flatnonzero(np.diff(pair_codes[order])) + 1):
        origin_i, dest_i = divmod(int(pair_codes[chunk[0]]), len(dests))
        travel[chunk] = _sample_travel_times(travel_pools, origins[origin_i], dests[dest_i], len(chunk), rng)

    trips = pd.DataFrame(
        {"origin_bg": np.asarray(origins)[rep_o], "dest_bg": np.asarray(dests)[rep_d],
         "dep_min": dep, "tt_min": travel}
    )
    trips["arr_min"] = (trips["dep_min"] + trips["tt_min"]) % 1440.0
    return trips


def _departure_shares(departures: pd.DataFrame) -> pd.DataFrame:
    bins = list(range(len(DEP_BIN_MIDPOINTS)))
    totals = departures[bins].sum(axis=1)
    keep = totals > 0
    return departures.loc[keep, bins].div(totals[keep], axis=0)


def _ipf_seed(initial: pd.DataFrame, o_index: dict[str, int], d_index: dict[str, int]) -> np.ndarray:
    """Seed the IPF table with the uncalibrated assignment's own (o, d, s) counts."""
    seed = np.zeros((len(o_index), len(d_index), len(DEP_BIN_MIDPOINTS)))
    cells = pd.DataFrame(
        {"o": initial["origin_bg"].map(o_index), "d": initial["dest_bg"].map(d_index),
         "s": bin_departure_minutes(initial["dep_min"])}
    ).dropna()
    np.add.at(seed, (cells["o"].astype(int).to_numpy(), cells["d"].astype(int).to_numpy(),
                     cells["s"].astype(int).to_numpy()), 1.0)
    return seed


def _travel_time_pools(initial: pd.DataFrame) -> dict:
    pools: dict = {key: group.to_numpy(dtype=float) for key, group in initial.groupby(["origin_bg", "dest_bg"])["tt_min"]}
    pools.update({key: group.to_numpy(dtype=float) for key, group in initial.groupby("origin_bg")["tt_min"]})
    pools["*"] = initial["tt_min"].to_numpy(dtype=float)
    return pools


def build_ipf_baseline(
    initial: pd.DataFrame, lodes: pd.DataFrame, departures: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame | None:
    """IPF baseline: LODES (o, d) totals x B08302 (o, s) totals, seeded with `initial`."""
    shares = _departure_shares(departures)
    pairs = pd.DataFrame(
        {"origin_bg": normalize_geoid(lodes["h_geocode"]), "dest_bg": normalize_geoid(lodes["w_geocode"]),
         "total_jobs": pd.to_numeric(lodes["total_jobs"], errors="coerce").fillna(0.0)}
    )
    origins = sorted(set(pairs["origin_bg"]) & set(shares.index))
    dropped = pairs[~pairs["origin_bg"].isin(origins)]
    if not dropped.empty:
        lost_jobs = float(dropped["total_jobs"].sum())
        log = LOGGER.warning if lost_jobs > 0 else LOGGER.info
        log("IPF: skipped %d origin block group(s) missing from B08302, carrying %.0f LODES job(s); "
            "the expanded baseline is that much smaller than the LODES total",
            dropped["origin_bg"].nunique(), lost_jobs)
    pairs = pairs[pairs["origin_bg"].isin(origins)]
    dests = sorted(set(pairs["dest_bg"]))
    if len(origins) * len(dests) * len(DEP_BIN_MIDPOINTS) > MAX_IPF_CELLS:
        LOGGER.warning("IPF baseline skipped: %d x %d x 14 cells exceed the dense-table budget",
                       len(origins), len(dests))
        return None

    o_index = {geoid: i for i, geoid in enumerate(origins)}
    d_index = {geoid: i for i, geoid in enumerate(dests)}
    od_target = np.zeros((len(origins), len(dests)))
    np.add.at(od_target, (pairs["origin_bg"].map(o_index).to_numpy(), pairs["dest_bg"].map(d_index).to_numpy()),
              pairs["total_jobs"].to_numpy(dtype=float))
    os_target = od_target.sum(axis=1)[:, None] * shares.loc[origins].to_numpy(dtype=float)

    table, iterations, error = fit_ipf(_ipf_seed(initial, o_index, d_index) + 1e-6, od_target, os_target)
    LOGGER.info("IPF: %d iteration(s), max marginal error %.3g", iterations, error)
    return expand_cells_to_trips(table, origins, dests, _travel_time_pools(initial), rng)


def build_uniform_baseline(initial: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Trivial lower bound: keep the routed travel times, redraw departures uniformly."""
    trips = initial.copy()
    trips["dep_min"] = rng.uniform(0.0, 1440.0, len(trips))
    trips["arr_min"] = (trips["dep_min"] + trips["tt_min"]) % 1440.0
    return trips


# --------------------------------------------------------------------------- figures


def _percent(series: pd.Series, n_bins: int) -> np.ndarray:
    values = np.asarray(series.reindex(range(n_bins)).fillna(0), dtype=float)
    total = values.sum()
    return values / total * 100 if total > 0 else values


def plot_binned_comparison(pooled: dict[str, pd.Series], labels: list[str], title: str, out_path: Path) -> None:
    """Grouped bars: every method plus the held-out ACS table, as % per bin."""
    series = {name: pooled[name] for name in ("calibrated", "initial", "ipf", "uniform", "acs") if name in pooled}
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    plt.figure(figsize=(11, 4.5))
    for offset, (name, counts) in enumerate(series.items()):
        plt.bar(x + (offset - (len(series) - 1) / 2) * width, _percent(counts, len(labels)), width,
                label=name.upper() if name == "acs" else name.capitalize(),
                color=METHOD_COLORS.get(name), alpha=0.85, edgecolor="black", linewidth=0.4)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Trip distribution (%)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_unit_tvd(unit_tvd: dict[str, dict[str, pd.Series]], out_path: Path) -> None:
    """Per-unit TVD distributions by method, for both distribution tests."""
    fig, axes = plt.subplots(1, len(unit_tvd), figsize=(5 * max(len(unit_tvd), 1), 4), squeeze=False)
    for axis, (test_name, by_method) in zip(axes[0], unit_tvd.items()):
        names = [name for name in ("calibrated", "initial", "ipf", "uniform") if name in by_method]
        data = [by_method[name].to_numpy(dtype=float) for name in names]
        if data:
            axis.boxplot(data, labels=names, showmeans=True)
        axis.set_ylabel("TVD per unit")
        n_units = len(data[0]) if data else 0
        axis.set_title(f"{test_name} ({n_units} unit(s))")
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Per-unit total variation distance vs held-out ACS")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_joint_scatter(scatter: pd.DataFrame, out_path: Path) -> None:
    """Synthetic vs ACS mean travel minutes per origin unit x departure bin."""
    plt.figure(figsize=(6, 6))
    for method, group in scatter.groupby("method"):
        plt.scatter(group["acs_mean"], group["syn_mean"], s=np.sqrt(group["workers"]),
                    alpha=0.6, label=method, color=METHOD_COLORS.get(method))
    finite = pd.concat([scatter["acs_mean"], scatter["syn_mean"]]).replace([np.inf, -np.inf], np.nan).dropna()
    limit = [0, float(finite.max()) * 1.05 if len(finite) else 1.0]
    plt.plot(limit, limit, color="black", linewidth=0.8, linestyle="--", label="1:1")
    plt.xlim(limit)
    plt.ylim(limit)
    plt.xlabel("ACS B08133 / B08302 mean travel time (min)")
    plt.ylabel("Synthetic mean travel time (min)")
    plt.title("Mean travel time by departure bin")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------- run driver


def code_sha() -> str:
    """HEAD's short sha, suffixed "+dirty" when the tree has uncommitted changes."""
    repo = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout

    try:
        sha = git("rev-parse", "--short", "HEAD").strip()
        return f"{sha}+dirty" if git("status", "--porcelain").strip() else sha
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def load_input_census(run_dir: Path, stem: str, columns: list[str]) -> pd.DataFrame | None:
    """Load a pipeline-input census table (B08302/B08303) into bin-indexed form."""
    path = find_census_table(run_dir / "census_data", stem)
    if path is None:
        return None
    raw = read_table(path)
    geo_col = pick_first_col(raw, ["GEO_ID", "geoid", "origin_geoid"])
    if geo_col is None or not set(columns).issubset(raw.columns):
        LOGGER.warning("Census input %s in %s has an unexpected schema", stem, run_dir)
        return None
    frame = raw[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame.columns = range(len(columns))
    frame["total"] = frame[list(range(len(columns)))].sum(axis=1)
    frame.index = normalize_geoid(raw[geo_col])
    return frame.groupby(level=0).sum()


def initial_frame_for_day(run_dir: Path, day: str) -> Path | None:
    """Prefer the day's own post-MSSR frame; fall back to the run-level lookup."""
    match = CALIB_STEM_RE.match(day)
    date = match.group(1) if match else day
    candidate = run_dir / "intermediate" / date / "post_mssr_routing_df.parquet"
    return candidate if candidate.exists() else find_initial_frame(run_dir)


def build_methods(
    run_dir: Path, calib_csv: Path, day: str, departures: pd.DataFrame | None, seed: int, skip_baselines: bool
) -> dict[str, pd.DataFrame]:
    """Trip frames for calibrated, initial and (optionally) the two baselines."""
    methods = {"calibrated": to_trip_frame(pd.read_csv(calib_csv))}
    initial_path = initial_frame_for_day(run_dir, day)
    if initial_path is None:
        LOGGER.warning("No initial (post-MSSR) frame under %s; baselines are unavailable", run_dir)
        return methods
    try:
        methods["initial"] = to_trip_frame(read_table(initial_path))
    except (KeyError, ValueError) as exc:
        # Older runs fall back to lodes_combs/, which carries OD pairs but no routing.
        LOGGER.warning("Unusable initial frame %s (%s); calibrated-only comparison", initial_path, exc)
        return methods
    if skip_baselines:
        return methods

    methods["uniform"] = build_uniform_baseline(methods["initial"], np.random.default_rng(seed))
    lodes_path = run_dir / "census_data" / "county_lodes_adjusted.csv"
    if departures is None or not lodes_path.exists():
        LOGGER.warning("IPF baseline skipped: missing %s or B08302 input", lodes_path)
        return methods
    ipf = build_ipf_baseline(methods["initial"], pd.read_csv(lodes_path), departures, np.random.default_rng(seed))
    if ipf is not None and not ipf.empty:
        methods["ipf"] = ipf
    return methods


def sanity_counts(
    run_dir: Path, trips: pd.DataFrame, departures: pd.DataFrame | None, arrival: pd.DataFrame | None,
    state_fips: str, county_fips: str, acs_year: int = ACS_YEAR
) -> dict[str, float]:
    """Trip count vs the B08302 total the pipeline scales LODES to, vs B08604/B08602.

    Also re-fetches B08302 at `acs_year` so a vintage mismatch shows up: the
    pipeline's lodes_combs_county.py reads 2022 while lodes_combs.py reads 2021.
    """
    cache_dir = run_dir / "census_data" / "held_out"
    local_total = float(departures["total"].sum()) if departures is not None else float("nan")
    acs_total = acs_county_total("B08302", 14, state_fips, county_fips, cache_dir, acs_year)
    if pd.notna(local_total) and pd.notna(acs_total) and abs(local_total - acs_total) > 0.5:
        LOGGER.warning("B08302 vintage mismatch: the run stores %.0f workers, ACS %d reports %.0f",
                       local_total, acs_year, acs_total)
    return {
        "synthetic_trips": float(len(trips)),
        "b08302_total": local_total,
        "b08302_acs_total": acs_total,
        "b08604_worker_total": acs_county_total("B08604", 0, state_fips, county_fips, cache_dir, acs_year),
        "b08602_bin_total": float(arrival[list(range(14))].sum().sum()) if arrival is not None else float("nan"),
    }


def _sanity_line(counts: dict[str, float], acs_year: int = ACS_YEAR) -> str:
    return (
        "Sanity: {synthetic_trips:,.0f} synthetic trips | B08302 (residence, pipeline input) "
        "{b08302_total:,.0f} workers, ACS {year} reports {b08302_acs_total:,.0f} | "
        "B08604 (workplace) {b08604_worker_total:,.0f} workers | "
        "B08602 county bins sum {b08602_bin_total:,.0f}".format(year=acs_year, **counts)
    )


def _held_out_tests(
    methods: dict[str, pd.DataFrame], departures: pd.DataFrame | None, state_fips: str, county_fips: str,
    cache_dir: Path, out_dir: Path, label: str, acs_year: int = ACS_YEAR
) -> tuple[list[dict], dict[str, dict[str, pd.Series]], pd.DataFrame | None]:
    """The three held-out tests plus their figures; also returns B08602 for the sanity line."""
    rows: list[dict] = []
    unit_tvd: dict[str, dict[str, pd.Series]] = {}
    specs = [
        ("arrival_dest", "B08602", "arr_min", bin_departure_minutes, DEP_BIN_MIDPOINTS, DEP_BIN_LABELS,
         "Arrival time at destination vs ACS B08602"),
        ("traveltime_dest", "B08603", "tt_min", travel_bin_codes, TT_BIN_MIDPOINTS, TT_BIN_LABELS,
         "Travel time at destination vs ACS B08603"),
    ]
    arrival = None
    for test_name, table, value_col, bin_fn, midpoints, labels, title in specs:
        resolved = resolve_acs_table(table, len(midpoints), state_fips, county_fips, cache_dir, acs_year)
        if resolved is None:
            LOGGER.warning("%s skipped: no held-out %s data", test_name, table)
            continue
        if table == "B08602":
            arrival = resolved[0]
        test_rows, extra = run_distribution_test(
            test_name, methods, resolved[0], resolved[1], "dest_bg", value_col, bin_fn, midpoints)
        rows += test_rows
        unit_tvd[test_name] = extra["unit_tvd"]
        plot_binned_comparison(extra["pooled"], labels, f"{title} - {label}", out_dir / f"{test_name}_county.png")

    aggregate_time = resolve_acs_table("B08133", 14, state_fips, county_fips, cache_dir, acs_year)
    if aggregate_time is not None and departures is not None:
        joint_rows, scatter = run_joint_test(methods, aggregate_time[0], aggregate_time[1], departures)
        rows += joint_rows
        if not scatter.empty:
            plot_joint_scatter(scatter, out_dir / "joint_dep_tt_scatter.png")
    return rows, unit_tvd, arrival


def validate_run(run_dir: Path, calib_csv: Path, state: str, county: str, seed: int, skip_baselines: bool,
                 acs_year: int = ACS_YEAR) -> pd.DataFrame:
    """Run all held-out tests for one day of one run and write that day's outputs.

    Outputs go to run_dir/validation/<day>/ so a run with several days does not
    overwrite itself; run_dir/validation/metrics.csv concatenates the day files.
    """
    day = calib_csv.stem
    out_dir = run_dir / "validation" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "census_data" / "held_out"

    departures = load_input_census(run_dir, "census_depart_times", DEP_ACS_COLUMNS)
    methods = build_methods(run_dir, calib_csv, day, departures, seed, skip_baselines)
    first_geoid = methods["calibrated"]["origin_bg"].iloc[0]
    state_fips, county_fips = first_geoid[:2], first_geoid[2:5]
    LOGGER.info("%s/%s %s (%s): methods = %s", state, county, run_id_of(run_dir), day, ", ".join(methods))

    rows, unit_tvd, arrival = _held_out_tests(methods, departures, state_fips, county_fips, cache_dir,
                                              out_dir, f"{county}, {state} ({day})", acs_year)
    if unit_tvd:
        plot_unit_tvd(unit_tvd, out_dir / "bg_tvd_distribution.png")
    counts = sanity_counts(run_dir, methods["calibrated"], departures, arrival, state_fips, county_fips, acs_year)
    rows += [{"method": "reference", "test": "sanity", "geography": "county", "metric": key,
              "value": value, "n_units": 1} for key, value in counts.items()]

    metrics = _finalise_metrics(rows, state, county, run_id_of(run_dir), day, seed)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    write_summary(out_dir / "summary.md", metrics, counts, state, county, day, acs_year)
    collect_run_metrics(run_dir)
    LOGGER.info(_sanity_line(counts, acs_year))
    return metrics


def collect_run_metrics(run_dir: Path) -> Path | None:
    """Concatenate every day's metrics.csv into run_dir/validation/metrics.csv."""
    day_files = sorted((run_dir / "validation").glob("*/metrics.csv"))
    if not day_files:
        return None
    combined = pd.concat([pd.read_csv(path) for path in day_files], ignore_index=True)
    out_path = run_dir / "validation" / "metrics.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def calibrated_csvs(run_dir: Path) -> list[Path]:
    """Date-named trip CSVs in calibrated_move_od/, one per day.

    The folder also collects sub-samples (df_sample_200.csv) and helper tables
    (geoids.csv); those are logged and skipped rather than validated as days.
    """
    paths, skipped = [], []
    for path in sorted((run_dir / "calibrated_move_od").glob("*.csv")):
        (paths if CALIB_STEM_RE.match(path.stem) else skipped).append(path)
    if skipped:
        LOGGER.info("Ignoring %d non-day file(s) in %s: %s", len(skipped), run_dir / "calibrated_move_od",
                    ", ".join(path.name for path in skipped))
    return paths


def run_id_of(run_dir: Path) -> str:
    return run_dir.name


def _finalise_metrics(rows: list[dict], state: str, county: str, run_id: str, day: str, seed: int) -> pd.DataFrame:
    metrics = pd.DataFrame(rows)
    metrics.insert(0, "day", day)
    metrics.insert(0, "run_id", run_id)
    metrics.insert(0, "county", county)
    metrics.insert(0, "state", state)
    metrics["seed"] = seed
    metrics["code_sha"] = code_sha()
    metrics["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return metrics[["state", "county", "run_id", "day", "method", "test", "geography", "metric", "value",
                    "n_units", "seed", "code_sha", "generated_at"]]


def _markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a markdown table (tabulate is not a project dependency)."""
    display = frame.reset_index()
    header = [str(column) for column in display.columns]

    def cell(value: object) -> str:
        return float_format.format(value) if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in display.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _headline_table(metrics: pd.DataFrame, test: str, wanted: list[str]) -> str:
    subset = metrics[(metrics["test"] == test) & (metrics["geography"] == "county") & metrics["metric"].isin(wanted)]
    if subset.empty:
        return f"_No held-out data for {test}._\n"
    table = subset.pivot_table(index="method", columns="metric", values="value").reindex(columns=wanted)
    return _markdown_table(table)


def write_summary(path: Path, metrics: pd.DataFrame, counts: dict[str, float], state: str, county: str,
                  day: str, acs_year: int = ACS_YEAR) -> None:
    """Short markdown report: headline numbers, coverage and the sanity line."""
    joint = metrics[(metrics["test"] == "joint_dep_tt") & metrics["metric"].isin(["r2_weighted", "bias_mean", "rmse_weighted", "n_cells"])]
    joint_table = ("_No held-out B08133 data._\n" if joint.empty else
                   _markdown_table(joint.pivot_table(index="method", columns="metric", values="value")))
    coverage = metrics[metrics["metric"].isin(["coverage", "units_compared", "units_with_acs"])]
    coverage_table = ("_n/a_\n" if coverage.empty else
                      _markdown_table(coverage.pivot_table(index=["test", "geography", "method"],
                                                           columns="metric", values="value"), "{:.3f}"))
    sections = [
        f"# Held-out validation - {county}, {state} ({day})\n",
        "Held-out tables: B08602 (arrival time, workplace geography), B08603 (travel time, "
        "workplace geography), B08133 (aggregate travel time by departure bin). None of these is "
        "read by the generation pipeline.\n",
        f"\n{_sanity_line(counts, acs_year)}\n",
        "\n## arrival_dest vs B08602 (county level)\n",
        _headline_table(metrics, "arrival_dest", ["tvd_pooled", "js_pooled", "w1_pooled", "tvd_wmean"]),
        "\n## traveltime_dest vs B08603 (county level)\n",
        _headline_table(metrics, "traveltime_dest", ["tvd_pooled", "js_pooled", "w1_pooled", "tvd_wmean"]),
        "\n## joint_dep_tt vs B08133 / B08302\n",
        joint_table,
        "\nA positive bias means the synthetic commutes run longer than the ACS aggregate "
        "minutes imply. Where that happens, part of it is built into the target rather than "
        "the routing: the pipeline's speed shift aims at a midpoint-weighted mean travel time, "
        "which for Hamilton is 23.41 min (the 90+ bin counted as 105) against an ACS "
        "aggregate-minute mean of 21.66 min (B08013/B08303 and B08133/B08302 agree) -- a 1.75 "
        "min head start on the measured bias.\n",
        "\n## Coverage\n",
        coverage_table,
    ]
    path.write_text("".join(sections))


# --------------------------------------------------------------------------- cross-county sweep


def discover_runs(output_root: Path) -> list[tuple[Path, Path, str, str]]:
    """Every (run_dir, day csv, state, county) under the output root."""
    found = []
    for calib_dir in sorted(output_root.glob("*/*/*/calibrated_move_od")):
        run_dir = calib_dir.parent
        for csv_path in calibrated_csvs(run_dir):
            found.append((run_dir, csv_path, run_dir.parent.parent.name, run_dir.parent.name))
    return found


def _county_tvd_frame(metrics: pd.DataFrame, test: str) -> pd.DataFrame:
    trips = (metrics[metrics["metric"] == "synthetic_trips"]
             .set_index(["state", "county", "run_id", "day"])["value"].rename("trips"))
    subset = metrics[(metrics["test"] == test) & (metrics["geography"] == "county")
                     & (metrics["metric"] == "tvd_pooled") & metrics["method"].isin(["calibrated", "initial"])]
    if subset.empty:
        return pd.DataFrame()
    table = subset.pivot_table(index=["state", "county", "run_id", "day"], columns="method", values="value")
    return table.join(trips).sort_values("trips", ascending=False)


def plot_cross_county_tvd(metrics: pd.DataFrame, out_path: Path) -> None:
    """County-level TVD per county, calibrated vs initial, sorted by trip count."""
    tests = ["arrival_dest", "traveltime_dest"]
    frames = {test: _county_tvd_frame(metrics, test) for test in tests}
    frames = {test: frame for test, frame in frames.items() if not frame.empty}
    if not frames:
        LOGGER.warning("No county-level TVD rows to plot across counties")
        return
    fig, axes = plt.subplots(len(frames), 1, figsize=(max(8, 0.35 * max(len(f) for f in frames.values())), 4 * len(frames)),
                             squeeze=False)
    for axis, (test, frame) in zip(axes[:, 0], frames.items()):
        x = np.arange(len(frame))
        methods = [name for name in ("initial", "calibrated") if name in frame.columns]
        width = 0.8 / len(methods)
        for offset, method in enumerate(methods):
            axis.bar(x + (offset - (len(methods) - 1) / 2) * width, frame[method].to_numpy(dtype=float), width,
                     label=method, color=METHOD_COLORS.get(method), edgecolor="black", linewidth=0.3)
        axis.set_xticks(x)
        axis.set_xticklabels([f"{county}, {state} {day}" for state, county, _, day in frame.index],
                             rotation=90, fontsize=7)
        axis.set_ylabel("County-level TVD")
        axis.set_title(f"{test} vs held-out ACS (counties sorted by trip count)")
        axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_all_counties(output_root: Path, seed: int, skip_baselines: bool, acs_year: int = ACS_YEAR) -> pd.DataFrame:
    """Validate every run under the output root, never crashing on a bad run."""
    collected: list[pd.DataFrame] = []
    for run_dir, csv_path, state, county in discover_runs(output_root):
        metrics = _validate_or_log(run_dir, csv_path, state, county, seed, skip_baselines, acs_year)
        if metrics is not None:
            collected.append(metrics)
    if not collected:
        LOGGER.warning("No runs validated under %s", output_root)
        return pd.DataFrame()

    metrics = pd.concat(collected, ignore_index=True)
    summary_dir = output_root / "validation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_dir / "cross_county_metrics.csv", index=False)
    plot_cross_county_tvd(metrics, summary_dir / "cross_county_tvd.png")
    LOGGER.info("Wrote %s", summary_dir / "cross_county_metrics.csv")
    return metrics


def _validate_or_log(run_dir: Path, csv_path: Path, state: str, county: str, seed: int,
                     skip_baselines: bool, acs_year: int) -> pd.DataFrame | None:
    """Validate one day, logging and swallowing anything that run alone breaks on."""
    label = f"{state}/{county}/{run_dir.name}/{csv_path.stem}"
    try:
        metrics = validate_run(run_dir, csv_path, state, county, seed, skip_baselines, acs_year)
    except Exception as exc:  # one bad day must not stop the others
        LOGGER.warning("SKIP %s: %s: %s", label, type(exc).__name__, exc)
        return None
    LOGGER.info("OK   %s", label)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out external validation of Move-OD synthetic trips.")
    parser.add_argument("--output-root", default="move_OD", help="Root output directory")
    parser.add_argument("--state", default="Tennessee", help="State name")
    parser.add_argument("--county", default="Hamilton", help="County name")
    parser.add_argument("--run-id", default=None, help="Run folder name (e.g., 2025-03-17_2025-03-17)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the deterministic baselines")
    parser.add_argument("--acs-year", type=int, default=ACS_YEAR, help="ACS 5-year vintage for the held-out tables")
    parser.add_argument("--skip-baselines", action="store_true", help="Only compare calibrated and initial")
    parser.add_argument("--all-counties", action="store_true", help="Validate every run under the output root")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_root = Path(args.output_root)
    if args.all_counties:
        run_all_counties(output_root, args.seed, args.skip_baselines, args.acs_year)
        return

    run_dir = find_run_dir(output_root, args.state, args.county, args.run_id)
    for csv_path in calibrated_csvs(run_dir) or [find_calibrated_csv(run_dir)]:
        metrics = _validate_or_log(run_dir, csv_path, args.state, args.county, args.seed,
                                   args.skip_baselines, args.acs_year)
        if metrics is None:
            continue
        print((run_dir / "validation" / csv_path.stem / "summary.md").read_text())
        LOGGER.info("Wrote %d metric rows to %s", len(metrics),
                    run_dir / "validation" / csv_path.stem / "metrics.csv")


if __name__ == "__main__":
    main()
