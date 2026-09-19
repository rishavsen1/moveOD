"""CTPP Part 3 held-out validation for Move-OD synthetic trips.

LODES (administrative wage records) supplies Move-OD's origin-destination
flows; ACS B08302/B08303 supply its departure-time and travel-time marginals.
Both are pipeline *inputs*. CTPP (Census Transportation Planning Products)
Part 3 is survey-based and is the only public source that publishes
home-tract-to-work-tract *flows* cross-tabulated by departure time and travel
time -- exactly the per-flow joint structure the pipeline's integer program
synthesizes and that nothing else in this repo tests. See
`analysis/validate_external.py` for the sibling held-out-ACS-table tests this
module deliberately does not duplicate.

CTPP API notes (verified against the live API, 2026-09-18):
    requests.post("https://ctppdata.transportation.org/api/data/2021",
      headers={"x-api-key": CTPP_API_KEY},
      params={"page": 1, "size": 1000, "exclude_null": False},
      json={"geo": "C1100US47065", "get": "b302104_e1,b302104_e2,...",
            "d-geo": "C3100US"})
`geo="C1100US" + state+county FIPS` returns every tract origin inside that
county; `d-geo="C3100US"` returns all destination tracts nationally. There is
no metadata or table-list endpoint. All CTPP values are perturbed and rounded
to the nearest 5 or 10; a published flow needs >= 3 unweighted survey
observations, so CTPP undercounts the true flow universe (see
`CTPP_COVERAGE_NOTE`).

Table structures below (column counts, category order) were verified
empirically against ACS for the 2021 CTPP release and are asserted at import
time so a future vintage change cannot pass silently instead of being
re-derived.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.figures_from_output import find_run_dir  # noqa: E402
from analysis.validate_external import (  # noqa: E402
    DEP_BIN_LABELS,
    TT_BIN_LABELS,
    bin_departure_minutes,
    calibrated_csvs,
    code_sha,
    find_calibrated_csv,
    load_input_census,
    normalize_geoid,
    run_id_of,
    to_trip_frame,
    total_variation_distance,
    travel_bin_codes,
)

LOGGER = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

CTPP_API_URL = "https://ctppdata.transportation.org/api/data/{year}"
CTPP_PAGE_SIZE = 1000
CTPP_YEAR = 2021
GEO_PREFIX_RE = re.compile(r"^C\d{4}US")

MIN_FLOW_WORKERS_JOINT = 50.0
TABLE_DRIFT_TVD_THRESHOLD = 0.05

# ACS B08303/travel_time_to_work.parquet column stems, in TT_BIN_LABELS order.
TT_ACS_COLUMNS = [
    f"{stem}_estimate"
    for stem in (
        "under_5_minutes", "5_to_9_minutes", "10_to_14_minutes", "15_to_19_minutes",
        "20_to_24_minutes", "25_to_29_minutes", "30_to_34_minutes", "35_to_39_minutes",
        "40_to_44_minutes", "45_to_59_minutes", "60_to_89_minutes", "90_minutes_and_over",
    )
]
DEP_ACS_COLUMNS = [f"{label}_estimate" for label in DEP_BIN_LABELS]

# CTPP orders its 14 departure categories starting at 5:00 a.m., with
# "12:00 a.m. to 4:59 a.m." last -- the opposite end from ACS B08302's own
# order. This is `DEP_BIN_LABELS` rotated left by one; verified: reordering
# this way gives a county TVD of 0.0094 against ACS B08302 for Hamilton
# County, TN, vs 0.1766 with the naive (unrotated) order.
CTPP_DEPARTURE_LABELS = DEP_BIN_LABELS[1:] + DEP_BIN_LABELS[:1]

# Column counts and category labels verified empirically against ACS for the
# 2021 CTPP release. `labels[i]` is the real-world category for column e(i+1).
TABLE_SPECS: dict[str, dict[str, object]] = {
    "b302100": {"n_cols": 1, "labels": ["total"]},
    "b302103": {
        "n_cols": 18,
        "labels": [
            "total", "drove_alone", "carpooled_2", "carpooled_3", "carpooled_4",
            "carpooled_5_6", "carpooled_7plus", "bus", "streetcar", "subway",
            "railroad", "ferryboat", "bicycle", "walked", "taxicab", "motorcycle",
            "other", "worked_at_home",
        ],
    },
    "b302104": {
        "n_cols": 17,
        "labels": ["total", "traveled", *CTPP_DEPARTURE_LABELS, "worked_at_home"],
    },
    "b302106": {
        "n_cols": 15,
        "labels": ["total", "traveled", *TT_BIN_LABELS, "worked_at_home"],
    },
}
for _table, _spec in TABLE_SPECS.items():
    assert len(_spec["labels"]) == _spec["n_cols"], (
        f"{_table}: {len(_spec['labels'])} label(s) for {_spec['n_cols']} column(s) -- "
        "the verified CTPP table shape has drifted and must be re-derived, not patched."
    )
assert len(CTPP_DEPARTURE_LABELS) == 14
assert CTPP_DEPARTURE_LABELS[0] == "5am_to_5:29am"
assert CTPP_DEPARTURE_LABELS[-1] == "12am_to_4:59am"

# e2..e7 of b302103: drove alone + carpooled 2..7+, the modes MoveOD's
# road-only assignment can actually be compared against.
CAR_MODE_COLUMNS = [f"b302103_e{i}" for i in range(2, 8)]

CTPP_COVERAGE_NOTE = (
    "CTPP flow tables only publish flows with >= 3 unweighted survey "
    "observations, so they cover a minority-but-large share of a county's "
    "ACS commuters (85.4% for Hamilton County, TN in the 2021 release)."
)


# --------------------------------------------------------------------------- geoids


def _ctpp_api_key() -> str:
    key = os.getenv("CTPP_API_KEY")
    if not key:
        raise RuntimeError("CTPP_API_KEY not set (see .env / .env.example)")
    return key


def _strip_geo_prefix(values: pd.Series) -> pd.Series:
    """Strip CTPP's summary-level prefix (C1100US, C3100US, ...) and zero-pad to 11 chars.

    Tract geoids must never be allowed to round-trip through int (a past bug
    in this project dropped leading zeros for low-numbered state FIPS codes).
    """
    stripped = values.astype(str).str.replace(GEO_PREFIX_RE, "", regex=True)
    return stripped.str.zfill(11)


# --------------------------------------------------------------------------- fetch


def _page_ctpp(geo: str, variables: list[str], table: str, year: int) -> list[dict]:
    """POST /api/data/{year} repeatedly until every flow row has been collected."""
    api_key = _ctpp_api_key()
    rows: list[dict] = []
    total: int | None = None
    page = 1
    while total is None or len(rows) < total:
        response = requests.post(
            CTPP_API_URL.format(year=year),
            headers={"x-api-key": api_key},
            params={"page": page, "size": CTPP_PAGE_SIZE, "exclude_null": False},
            json={"geo": geo, "get": ",".join(variables), "d-geo": "C3100US"},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        total = int(payload["total"])
        data = payload.get("data", [])
        rows.extend(data)
        LOGGER.info("CTPP %s %d page %d: %d/%d flow(s)", table, year, page, len(rows), total)
        if not data:
            break
        page += 1
    return rows


def fetch_flows(
    state_fips: str, county_fips: str, table: str, n_cols: int, year: int = CTPP_YEAR,
    cache_dir: Path = Path("data/ctpp"),
) -> pd.DataFrame:
    """Paged CTPP fetch for every origin tract in one county, cached as parquet.

    Returns the raw API rows (origin/destination geoid and name columns plus
    `{table}_e1..e{n_cols}` numeric estimate columns); margin (`_m`) columns
    are dropped. Uses the parquet cache at
    `{cache_dir}/{table}_{state_fips}{county_fips}_tract_{year}.parquet` when
    present, fetching from the live API otherwise.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{table}_{state_fips}{county_fips}_tract_{year}.parquet"
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        LOGGER.info("CTPP %s %d: %d cached flow(s) from %s", table, year, len(frame), cache_path)
        return frame

    variables = [f"{table}_e{i}" for i in range(1, n_cols + 1)]
    geo = f"C1100US{state_fips}{county_fips}"
    frame = pd.DataFrame(_page_ctpp(geo, variables, table, year))
    for column in variables:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[[c for c in frame.columns if not c.endswith("_m")]]
    frame.to_parquet(cache_path, index=False)
    LOGGER.info("CTPP %s %d: fetched %d flow(s) from the API, cached to %s", table, year, len(frame), cache_path)
    return frame


# --------------------------------------------------------------------------- decode


def decode(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Tidy long rows: origin_tract, destination_tract, category, workers.

    `category` carries the real-world label for each estimate column (the
    CTPP-to-ACS departure reordering is baked into `TABLE_SPECS[table]["labels"]`,
    not applied after the fact), and both tract ids are zero-padded 11-char
    strings stripped of their `C1100US`/`C3100US` summary-level prefix.
    """
    spec = TABLE_SPECS[table]
    n_cols = int(spec["n_cols"])
    labels = list(spec["labels"])
    value_cols = [f"{table}_e{i}" for i in range(1, n_cols + 1)]
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{table}: response is missing column(s) {missing}; the API shape may have changed")

    long = df.melt(id_vars=["origin_geoid", "destination_geoid"], value_vars=value_cols,
                    var_name="_col", value_name="workers")
    col_num = long["_col"].str.extract(rf"^{re.escape(table)}_e(\d+)$")[0].astype(int)
    label_by_col = {i + 1: label for i, label in enumerate(labels)}
    long["category"] = col_num.map(label_by_col)
    long["origin_tract"] = _strip_geo_prefix(long["origin_geoid"])
    long["destination_tract"] = _strip_geo_prefix(long["destination_geoid"])
    long["workers"] = pd.to_numeric(long["workers"], errors="coerce").fillna(0.0)
    return long[["origin_tract", "destination_tract", "category", "workers"]]


def _flow_totals(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    workers = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    frame = pd.DataFrame({
        "origin_tract": _strip_geo_prefix(df["origin_geoid"]),
        "destination_tract": _strip_geo_prefix(df["destination_geoid"]),
        "workers": workers,
    })
    return frame.groupby(["origin_tract", "destination_tract"], as_index=False)["workers"].sum()


def flow_totals_all_workers(df100: pd.DataFrame) -> pd.DataFrame:
    """Tidy origin_tract/destination_tract/workers from b302100 e1 (all workers)."""
    return _flow_totals(df100, ["b302100_e1"])


def flow_totals_travelers(df104: pd.DataFrame) -> pd.DataFrame:
    """Tidy flow frame from b302104 e2 (workers who travel, i.e. not WFH) -- the
    preferred commute-flow weight, since it excludes workers with no trip."""
    return _flow_totals(df104, ["b302104_e2"])


def flow_totals_car_modes(df103: pd.DataFrame) -> pd.DataFrame:
    """Tidy flow frame from b302103 e2..e7 (car modes only: drove alone + carpool)."""
    return _flow_totals(df103, CAR_MODE_COLUMNS)


# --------------------------------------------------------------------------- table-drift guard


class CTPPTableDriftError(RuntimeError):
    """Raised when a CTPP table's county aggregate no longer matches the run's own ACS input."""


def check_table_against_acs(
    ctpp_totals: pd.Series, acs_totals: pd.Series, table: str,
    threshold: float = TABLE_DRIFT_TVD_THRESHOLD, strict: bool = True,
) -> float:
    """County-level TVD between a CTPP category table and the matching ACS input.

    Raises (or, with `strict=False`, logs loudly) when the two disagree by more
    than `threshold`, so a future CTPP vintage that reorders or reshapes its
    categories cannot silently produce wrong per-flow comparisons.
    """
    tvd = total_variation_distance(
        np.asarray(ctpp_totals, dtype=float), np.asarray(acs_totals, dtype=float)
    )
    message = f"CTPP {table} county aggregate diverges from this run's ACS input by TVD={tvd:.4f} (threshold {threshold})"
    if tvd > threshold:
        if strict:
            raise CTPPTableDriftError(message)
        LOGGER.error(message)
    else:
        LOGGER.info("CTPP %s matches this run's ACS input: TVD=%.4f", table, tvd)
    return tvd


def _category_totals(decoded: pd.DataFrame, labels: list[str]) -> pd.Series:
    totals = decoded.groupby("category")["workers"].sum()
    return totals.reindex(labels, fill_value=0.0)


def departure_drift_check(run_dir: Path, decoded_104: pd.DataFrame, strict: bool = True) -> float:
    """b302104 departure categories vs this run's own census_depart_times.parquet (B08302)."""
    acs = load_input_census(run_dir, "census_depart_times", DEP_ACS_COLUMNS)
    if acs is None:
        LOGGER.warning("No census_depart_times table under %s; skipping CTPP departure drift check", run_dir)
        return float("nan")
    acs_totals = acs[list(range(len(DEP_BIN_LABELS)))].sum()
    ctpp_totals = _category_totals(decoded_104, DEP_BIN_LABELS)
    return check_table_against_acs(ctpp_totals, acs_totals, "b302104", strict=strict)


def traveltime_drift_check(run_dir: Path, decoded_106: pd.DataFrame, strict: bool = True) -> float:
    """b302106 travel-time categories vs this run's own travel_time_to_work.parquet (B08303)."""
    acs = load_input_census(run_dir, "travel_time_to_work", TT_ACS_COLUMNS)
    if acs is None:
        LOGGER.warning("No travel_time_to_work table under %s; skipping CTPP travel-time drift check", run_dir)
        return float("nan")
    acs_totals = acs[list(range(len(TT_BIN_LABELS)))].sum()
    ctpp_totals = _category_totals(decoded_106, TT_BIN_LABELS)
    return check_table_against_acs(ctpp_totals, acs_totals, "b302106", strict=strict)


# --------------------------------------------------------------------------- flow-level metrics


def filter_intra_county(flows: pd.DataFrame, county_fips5: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Keep flows with both endpoints inside `county_fips5`; report what was discarded.

    Move-OD generates intra-county trips only, so any CTPP (or LODES) flow
    that crosses the county line has to be dropped before comparison.
    """
    origin_in = flows["origin_tract"].str[:5] == county_fips5
    dest_in = flows["destination_tract"].str[:5] == county_fips5
    keep = origin_in & dest_in
    discarded = flows.loc[~keep]
    stats = {
        "n_kept": int(keep.sum()),
        "workers_kept": float(flows.loc[keep, "workers"].sum()),
        "n_discarded": int((~keep).sum()),
        "workers_discarded": float(discarded["workers"].sum()),
    }
    return flows.loc[keep].reset_index(drop=True), stats


def _align_flows(compared: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Outer-join two (origin_tract, destination_tract, workers) frames, zero-filled."""
    left = compared.set_index(["origin_tract", "destination_tract"])["workers"].rename("a")
    right = reference.set_index(["origin_tract", "destination_tract"])["workers"].rename("b")
    return pd.concat([left, right], axis=1).fillna(0.0)


def cpc_ssi(compared: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Common part of commuters (denominator = reference) and the Sorensen similarity index.

    CPC = sum(min(a, b)) / sum(b), with CTPP (`reference`) as `b` -- an
    asymmetric coverage-style score. SSI = 2 * sum(min(a, b)) / (sum(a) + sum(b)),
    the symmetric Sorensen-Dice index. Both are 1.0 for identical flow sets
    and 0.0 for disjoint ones.
    """
    merged = _align_flows(compared, reference)
    a, b = merged["a"].to_numpy(dtype=float), merged["b"].to_numpy(dtype=float)
    min_sum = float(np.minimum(a, b).sum())
    sum_a, sum_b = float(a.sum()), float(b.sum())
    cpc = min_sum / sum_b if sum_b > 0 else float("nan")
    ssi = 2 * min_sum / (sum_a + sum_b) if (sum_a + sum_b) > 0 else float("nan")
    return {"cpc": cpc, "ssi": ssi}


def flow_correlations(compared: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Pearson and Spearman correlation of flow counts over the union of flow ids, zero-filled.

    Zero-filling (rather than only comparing flows present on both sides)
    means the correlation also reflects whether each side puts a flow where
    the other has none.
    """
    merged = _align_flows(compared, reference)
    if len(merged) < 2 or merged["a"].std() == 0 or merged["b"].std() == 0:
        return {"pearson_r": float("nan"), "spearman_r": float("nan")}
    pearson_r, _ = pearsonr(merged["a"], merged["b"])
    spearman_r, _ = spearmanr(merged["a"], merged["b"])
    return {"pearson_r": float(pearson_r), "spearman_r": float(spearman_r)}


def missing_flow_shares(compared: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Flows (and their worker share) present on only one side of the comparison."""
    merged = _align_flows(compared, reference)
    ref_only = merged[(merged["b"] > 0) & (merged["a"] == 0)]
    cmp_only = merged[(merged["a"] > 0) & (merged["b"] == 0)]
    ref_total, cmp_total = float(merged["b"].sum()), float(merged["a"].sum())
    return {
        "workers_ctpp": ref_total,
        "workers_compared": cmp_total,
        "n_flows_compared": float(len(merged)),
        "n_ctpp_only_flows": float(len(ref_only)),
        "worker_share_ctpp_only": float(ref_only["b"].sum() / ref_total) if ref_total > 0 else float("nan"),
        "n_compared_only_flows": float(len(cmp_only)),
        "worker_share_compared_only": float(cmp_only["a"].sum() / cmp_total) if cmp_total > 0 else float("nan"),
    }


def flow_comparison_metrics(compared: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """All `ctpp_flows` metrics for one method against one CTPP flow-weight variant."""
    metrics: dict[str, float] = {}
    metrics.update(cpc_ssi(compared, reference))
    metrics.update(flow_correlations(compared, reference))
    metrics.update(missing_flow_shares(compared, reference))
    metrics.update(flow_coverage_decomposition(compared, reference))
    return metrics


def flow_coverage_decomposition(compared: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Split the single CPC number into coverage (does the flow exist on both
    sides at all) vs. magnitude (given it does, do the two sides agree on size).

    `structural_ceiling_cpc` = 1 - worker_share_ctpp_only is the best CPC any
    matrix could reach without inventing flows CTPP itself never publishes
    (suppressed for too few survey observations) -- the honest upper bound.
    """
    merged = _align_flows(compared, reference)
    ctpp_present = merged["b"] > 0
    compared_present = merged["a"] > 0
    overlap = merged[ctpp_present & compared_present]

    overlap_cpc = (float(np.minimum(overlap["a"], overlap["b"]).sum() / overlap["b"].sum())
                   if overlap["b"].sum() > 0 else float("nan"))

    total_a, total_b = float(merged["a"].sum()), float(merged["b"].sum())
    rescaled_a = merged["a"] * (total_b / total_a) if total_a > 0 else merged["a"]
    rescaled_cpc = (float(np.minimum(rescaled_a, merged["b"]).sum() / total_b) if total_b > 0 else float("nan"))

    worker_share_ctpp_only = missing_flow_shares(compared, reference)["worker_share_ctpp_only"]
    ceiling = 1.0 - worker_share_ctpp_only if np.isfinite(worker_share_ctpp_only) else float("nan")

    return {
        "n_ctpp_flows_total": float(ctpp_present.sum()),
        "n_ctpp_flows_covered": float((ctpp_present & compared_present).sum()),
        "cpc_overlap_only": overlap_cpc,
        "cpc_rescaled_to_ctpp_total": rescaled_cpc,
        "structural_ceiling_cpc": ceiling,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


# --------------------------------------------------------------------------- per-flow joint tests


def ctpp_bin_matrix(decoded: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Pivot decoded long rows into a (origin_tract, destination_tract) x label matrix."""
    subset = decoded[decoded["category"].isin(labels)]
    pivot = subset.pivot_table(index=["origin_tract", "destination_tract"], columns="category",
                               values="workers", aggfunc="sum", fill_value=0.0)
    return pivot.reindex(columns=labels, fill_value=0.0)


def _synthetic_bin_matrix(trips: pd.DataFrame, codes: pd.Series, labels: list[str]) -> pd.DataFrame:
    valid = codes.notna()
    units = pd.DataFrame({
        "origin_tract": trips.loc[valid, "origin_bg"].str[:11],
        "destination_tract": trips.loc[valid, "dest_bg"].str[:11],
        "bin": codes[valid].astype(int),
    })
    pivot = pd.crosstab([units["origin_tract"], units["destination_tract"]], units["bin"])
    pivot = pivot.reindex(columns=range(len(labels)), fill_value=0)
    return pivot.rename(columns=dict(enumerate(labels)))


def synthetic_departure_matrix(trips: pd.DataFrame) -> pd.DataFrame:
    """(origin_tract, destination_tract) x DEP_BIN_LABELS trip counts."""
    return _synthetic_bin_matrix(trips, bin_departure_minutes(trips["dep_min"]), DEP_BIN_LABELS)


def synthetic_traveltime_matrix(trips: pd.DataFrame) -> pd.DataFrame:
    """(origin_tract, destination_tract) x TT_BIN_LABELS trip counts."""
    return _synthetic_bin_matrix(trips, travel_bin_codes(trips["tt_min"]), TT_BIN_LABELS)


def _qualifying_flows(
    synthetic_bins: pd.DataFrame, ctpp_bins: pd.DataFrame, ctpp_flow_workers: pd.Series, min_workers: float,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    """Flows with >= `min_workers` CTPP travellers, as aligned (keys, syn_counts, ctpp_counts, weights) arrays.

    A flow missing from `synthetic_bins` or `ctpp_bins` gets an all-zero row
    rather than being dropped, so the caller can tell "no data" from "small
    but present" (the former makes a per-flow TVD undefined, not zero).
    """
    keep = ctpp_flow_workers[ctpp_flow_workers >= min_workers]
    keys = list(keep.index)
    syn = np.zeros((len(keys), synthetic_bins.shape[1]))
    ctpp = np.zeros((len(keys), ctpp_bins.shape[1]))
    for i, key in enumerate(keys):
        if key in synthetic_bins.index:
            syn[i] = synthetic_bins.loc[key].to_numpy(dtype=float)
        if key in ctpp_bins.index:
            ctpp[i] = ctpp_bins.loc[key].to_numpy(dtype=float)
    return keys, syn, ctpp, keep.to_numpy(dtype=float)


def per_flow_tvd(
    synthetic_bins: pd.DataFrame, ctpp_bins: pd.DataFrame, ctpp_flow_workers: pd.Series,
    min_workers: float = MIN_FLOW_WORKERS_JOINT,
) -> pd.DataFrame:
    """TVD per (origin_tract, destination_tract) flow with >= `min_workers` CTPP travellers.

    Flows below the threshold are excluded outright (not down-weighted) --
    CTPP's per-flow bin split is too noisy below a few dozen travellers to
    make a per-flow TVD meaningful.
    """
    keys, syn, ctpp, weights = _qualifying_flows(synthetic_bins, ctpp_bins, ctpp_flow_workers, min_workers)
    rows = [
        {"origin_tract": origin, "destination_tract": destination,
         "tvd": total_variation_distance(syn[i], ctpp[i]), "workers": float(weights[i])}
        for i, (origin, destination) in enumerate(keys)
    ]
    return pd.DataFrame(rows, columns=["origin_tract", "destination_tract", "tvd", "workers"])


def _effective_bins(counts: np.ndarray) -> np.ndarray:
    """Per-row exp(Shannon entropy): the "effective number of occupied bins".

    1.0 when a row puts everything in one bin, `n_bins` when it is uniform
    across all of them. NaN for an all-zero row (no distribution to measure).
    """
    totals = counts.sum(axis=1)
    probs = np.divide(counts, totals[:, None], out=np.zeros_like(counts, dtype=float),
                      where=totals[:, None] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0, probs * np.log(probs), 0.0)
    entropy = -terms.sum(axis=1)
    return np.where(totals > 0, np.exp(entropy), np.nan)


def joint_floor_metrics(
    synthetic_bins: pd.DataFrame, ctpp_bins: pd.DataFrame, ctpp_flow_workers: pd.Series,
    min_workers: float = MIN_FLOW_WORKERS_JOINT, n_reps: int = 200, seed: int = 0,
) -> dict[str, float]:
    """Monte Carlo sampling-noise floor for the worker-weighted mean per-flow TVD.

    A per-flow TVD of e.g. 0.69 means little on its own: each flow has few
    synthetic trips spread over many bins, so even a *perfect* model would
    show a nonzero TVD just from multinomial sampling noise. For each
    qualifying flow, this draws `n_synthetic` trips from CTPP's own
    distribution `n_reps` times (seeded) and recomputes the same
    worker-weighted TVD; `tvd_wmean_floor_rounded` repeats it after rounding
    each draw to the nearest 5, mimicking CTPP's own published rounding.
    """
    keys, syn, ctpp, weights = _qualifying_flows(synthetic_bins, ctpp_bins, ctpp_flow_workers, min_workers)
    n_synthetic = syn.sum(axis=1)
    ctpp_totals = ctpp.sum(axis=1)
    valid = (n_synthetic > 0) & (ctpp_totals > 0)
    if not valid.any():
        return {"tvd_wmean_floor": float("nan"), "tvd_wmean_floor_sd": float("nan"),
                "tvd_wmean_floor_rounded": float("nan")}

    probs = ctpp[valid] / ctpp_totals[valid][:, None]
    counts = n_synthetic[valid].astype(int)
    flow_weights = weights[valid]
    rng = np.random.default_rng(seed)
    tvd = np.empty((len(counts), n_reps))
    tvd_rounded = np.empty((len(counts), n_reps))
    for i in range(len(counts)):
        draws = rng.multinomial(counts[i], probs[i], size=n_reps)
        draw_probs = draws / counts[i]
        tvd[i] = 0.5 * np.abs(draw_probs - probs[i]).sum(axis=1)
        rounded = np.round(draws / 5.0) * 5.0
        rounded_totals = np.where(rounded.sum(axis=1) > 0, rounded.sum(axis=1), 1.0)
        tvd_rounded[i] = 0.5 * np.abs(rounded / rounded_totals[:, None] - probs[i]).sum(axis=1)

    rep_means = np.average(tvd, axis=0, weights=flow_weights)
    rep_means_rounded = np.average(tvd_rounded, axis=0, weights=flow_weights)
    return {
        "tvd_wmean_floor": float(rep_means.mean()),
        "tvd_wmean_floor_sd": float(rep_means.std()),
        "tvd_wmean_floor_rounded": float(rep_means_rounded.mean()),
    }


def joint_test_summary(per_flow: pd.DataFrame, pooled_syn: np.ndarray, pooled_ctpp: np.ndarray) -> dict[str, float]:
    """Worker-weighted mean/percentile TVD across qualifying flows, plus the pooled TVD.

    A qualifying flow with zero synthetic trips at all has an undefined
    (NaN) per-flow TVD -- `total_variation_distance` returns NaN rather than
    treating "no data" as "perfect agreement" -- so those flows are dropped
    from the mean/percentiles (as `compare_at_level` in validate_external.py
    does for the same reason) but counted separately rather than silently.
    """
    result = {"n_flows": float(len(per_flow)), "tvd_pooled": total_variation_distance(pooled_syn, pooled_ctpp)}
    if per_flow.empty:
        result.update({"tvd_wmean": float("nan"), "tvd_p10": float("nan"), "tvd_p50": float("nan"),
                       "tvd_p90": float("nan"), "n_flows_no_synthetic": 0.0})
        return result
    finite = per_flow["tvd"].notna()
    result["n_flows_no_synthetic"] = float((~finite).sum())
    scored = per_flow.loc[finite]
    if scored.empty:
        result.update({"tvd_wmean": float("nan"), "tvd_p10": float("nan"), "tvd_p50": float("nan"),
                       "tvd_p90": float("nan")})
        return result
    result["tvd_wmean"] = float(np.average(scored["tvd"], weights=scored["workers"]))
    for pct in (10, 50, 90):
        result[f"tvd_p{pct}"] = float(np.percentile(scored["tvd"], pct))
    return result


# --------------------------------------------------------------------------- input frames


def synthetic_tract_flows(trips: pd.DataFrame) -> pd.DataFrame:
    """Tract-to-tract trip counts from a Move-OD trip frame (one row = one commuter)."""
    frame = pd.DataFrame({
        "origin_tract": trips["origin_bg"].str[:11],
        "destination_tract": trips["dest_bg"].str[:11],
    })
    return frame.groupby(["origin_tract", "destination_tract"]).size().reset_index(name="workers")


def load_lodes_tract_flows(run_dir: Path) -> pd.DataFrame:
    """Tract-to-tract LODES job counts from census_data/county_lodes_adjusted.csv."""
    path = run_dir / "census_data" / "county_lodes_adjusted.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    lodes = pd.read_csv(path, dtype=str)
    frame = pd.DataFrame({
        "origin_tract": normalize_geoid(lodes["h_geocode"]).str[:11],
        "destination_tract": normalize_geoid(lodes["w_geocode"]).str[:11],
        "workers": pd.to_numeric(lodes["total_jobs"], errors="coerce").fillna(0.0),
    })
    return frame.groupby(["origin_tract", "destination_tract"], as_index=False)["workers"].sum()


# --------------------------------------------------------------------------- figures


def plot_flow_scatter(compared: pd.DataFrame, reference: pd.DataFrame, out_path: Path) -> None:
    """Synthetic vs CTPP tract-to-tract flow counts on log axes, coloured by intra-tract."""
    merged = _align_flows(compared, reference).rename(columns={"a": "synthetic", "b": "ctpp"}).reset_index()
    if merged.empty:
        LOGGER.warning("No flows to plot for %s", out_path)
        return
    intra = merged["origin_tract"] == merged["destination_tract"]
    plt.figure(figsize=(6, 6))
    for label, mask, color in (("intra-tract", intra, "#E45756"), ("inter-tract", ~intra, "#4C78A8")):
        subset = merged.loc[mask]
        if not subset.empty:
            plt.scatter(subset["ctpp"].clip(lower=0.5), subset["synthetic"].clip(lower=0.5),
                        s=14, alpha=0.5, label=label, color=color)
    plt.xscale("log")
    plt.yscale("log")
    upper = float(merged[["ctpp", "synthetic"]].to_numpy().max()) * 1.5 if len(merged) else 1.0
    limits = [0.5, max(upper, 1.0)]
    plt.plot(limits, limits, color="black", linewidth=0.8, linestyle="--", label="1:1")
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel("CTPP flow workers (b302100)")
    plt.ylabel("Synthetic flow trips")
    plt.title("Synthetic vs CTPP tract-to-tract flows")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------- markdown


def _markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a markdown table (tabulate is not a project dependency)."""
    display = frame.reset_index()
    header = [str(column) for column in display.columns]

    def cell(value: object) -> str:
        return float_format.format(value) if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in display.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _floor_headline(metrics: pd.DataFrame, test: str) -> str:
    """One line leading with the noise-floor ratio, not the raw TVD (easy to overread alone)."""
    subset = metrics[(metrics["test"] == test) & (metrics["method"] == "synthetic")].set_index("metric")["value"]
    if subset.empty or "tvd_ratio_to_floor" not in subset.index:
        return f"- {test}: no data.\n"
    return (
        f"- {test}: tvd_wmean={subset['tvd_wmean']:.4f} is **{subset['tvd_ratio_to_floor']:.1f}x** the "
        f"sampling-noise floor a perfect model would still show ({subset['tvd_wmean_floor']:.4f} "
        f"+/- {subset['tvd_wmean_floor_sd']:.4f}, {subset['tvd_wmean_floor_rounded']:.4f} with CTPP-style "
        f"rounding) -- effective bins occupied: synthetic {subset['eff_bins_synthetic_median']:.2f} vs "
        f"CTPP {subset['eff_bins_ctpp_median']:.2f} (median per flow).\n"
    )


def _flow_agreement_note(metrics: pd.DataFrame) -> str:
    """Flag when synthetic and LODES score identically on CPC/SSI -- the check is then of LODES, not synthesis."""
    flows = metrics[(metrics["test"] == "ctpp_flows") & metrics["metric"].isin(["cpc", "ssi"])
                    & metrics["method"].isin(["synthetic", "lodes"])]
    if flows.empty:
        return ""
    wide = flows.pivot_table(index=["geography", "metric"], columns="method", values="value")
    if not {"synthetic", "lodes"}.issubset(wide.columns):
        return ""
    max_gap = float((wide["synthetic"] - wide["lodes"]).abs().max())
    if max_gap < 1e-3:
        return (
            "\nSynthetic and LODES score identically (to four decimal places) on CPC and SSI above: "
            "Move-OD's tract-to-tract flow structure is inherited almost entirely from LODES, so this "
            "table is the first independent check of *that input*, not of the synthesis step -- "
            "building-level sampling and calibration barely move the flow-level answer (the small gap "
            "that does show up is in the correlation metrics, not CPC/SSI).\n"
        )
    return ""


def _pivot_or_placeholder(subset: pd.DataFrame, index: list[str], columns: list[str], placeholder: str) -> str:
    if subset.empty:
        return placeholder
    return _markdown_table(subset.pivot_table(index=index, columns="metric", values="value").reindex(columns=columns))


def _summary_tables(metrics: pd.DataFrame) -> tuple[str, str, str]:
    flow_table = _pivot_or_placeholder(
        metrics[metrics["test"] == "ctpp_flows"], ["geography", "method"],
        ["cpc", "ssi", "pearson_r", "spearman_r", "n_flows_compared", "n_ctpp_flows_total",
         "n_ctpp_flows_covered", "cpc_overlap_only", "cpc_rescaled_to_ctpp_total", "structural_ceiling_cpc",
         "worker_share_ctpp_only", "worker_share_compared_only"],
        "_No CTPP flow comparison available._\n",
    )
    joint_table = _pivot_or_placeholder(
        metrics[metrics["test"].isin(["ctpp_departure_joint", "ctpp_traveltime_joint"])], ["test", "method"],
        ["tvd_wmean", "tvd_wmean_floor", "tvd_wmean_floor_sd", "tvd_wmean_floor_rounded", "tvd_ratio_to_floor",
         "tvd_pooled", "eff_bins_synthetic_median", "eff_bins_ctpp_median", "n_flows"],
        "_No per-flow joint comparison available._\n",
    )
    reference = metrics[metrics["test"] == "reference"]
    reference_table = "_n/a_\n" if reference.empty else _markdown_table(reference.set_index("metric")[["value"]])
    return flow_table, joint_table, reference_table


def write_ctpp_summary(path: Path, metrics: pd.DataFrame, state: str, county: str, day: str) -> None:
    """Short markdown report: flow-agreement headlines, joint-test noise-floor ratios, coverage."""
    flow_table, joint_table, reference_table = _summary_tables(metrics)
    sections = [
        f"# CTPP held-out flow validation - {county}, {state} ({day})\n",
        "CTPP Part 3 publishes home-tract-to-work-tract flows cross-tabulated by "
        "departure time (b302104) and travel time (b302106) -- the only public "
        "source for the per-flow joint structure Move-OD's integer program "
        "synthesizes. Neither LODES flows nor ACS B08302/B08303 marginals (both "
        "pipeline inputs) carry this joint structure.\n",
        f"\n{CTPP_COVERAGE_NOTE}\n",
        "\n## ctpp_flows: tract-to-tract flow agreement\n",
        "`cpc_overlap_only` and `cpc_rescaled_to_ctpp_total` separate coverage (does the flow exist on "
        "both sides) from magnitude (given it does, do the sizes agree); `structural_ceiling_cpc` is "
        "the best CPC achievable without inventing flows CTPP itself suppresses.\n",
        flow_table,
        _flow_agreement_note(metrics),
        "\n## Per-flow joint structure (departure and travel time)\n",
        "Held out at the flow level: the ILP constrains origin departure marginals "
        "only, never the departure or travel-time distribution of an individual "
        "origin-destination flow. Only `synthetic` has per-trip departure/travel-time "
        "data; LODES carries neither, so it is not run for these two tests.\n",
        _floor_headline(metrics, "ctpp_departure_joint"),
        _floor_headline(metrics, "ctpp_traveltime_joint"),
        "\nCTPP travel times are self-reported (not routed), include parking and walking to/from the "
        "vehicle, and are heaped on multiples of five minutes -- a reporting confound that inflates "
        "the travel-time TVD (and its floor) but does not apply to departure timing, which CTPP asks "
        "for directly.\n",
        joint_table,
        "\n## Reference\n",
        reference_table,
    ]
    path.write_text("".join(sections))


# --------------------------------------------------------------------------- driver


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


def _ctpp_flows_rows(
    methods: dict[str, pd.DataFrame], variants: dict[str, pd.DataFrame]
) -> list[dict]:
    rows = []
    for geography, reference in variants.items():
        for method, compared in methods.items():
            metrics = flow_comparison_metrics(compared, reference)
            n_units = int(metrics.get("n_flows_compared", 0))
            rows += [{"method": method, "test": "ctpp_flows", "geography": geography, "metric": key,
                     "value": value, "n_units": n_units} for key, value in metrics.items()]
    return rows


def _effective_bins_medians(syn: np.ndarray, ctpp: np.ndarray) -> dict[str, float]:
    """Median effective-bins-occupied, restricted to flows with >0 synthetic trips
    (the same "scored" set `joint_test_summary` uses for tvd_wmean), for a fair
    synthetic-vs-CTPP comparison of how "spread out" each side's profile is."""
    scored = syn.sum(axis=1) > 0
    if not scored.any():
        return {"eff_bins_synthetic_median": float("nan"), "eff_bins_ctpp_median": float("nan")}
    return {
        "eff_bins_synthetic_median": float(np.nanmedian(_effective_bins(syn[scored]))),
        "eff_bins_ctpp_median": float(np.nanmedian(_effective_bins(ctpp[scored]))),
    }


def _joint_rows(
    test_name: str, trips: pd.DataFrame, decoded: pd.DataFrame, labels: list[str],
    synthetic_matrix_fn, ctpp_flow_workers: pd.Series, seed: int,
) -> list[dict]:
    ctpp_bins = ctpp_bin_matrix(decoded, labels)
    syn_bins = synthetic_matrix_fn(trips)
    per_flow = per_flow_tvd(syn_bins, ctpp_bins, ctpp_flow_workers, MIN_FLOW_WORKERS_JOINT)
    qualifying = ctpp_bins.reindex(per_flow.set_index(["origin_tract", "destination_tract"]).index, fill_value=0.0)
    syn_qualifying = syn_bins.reindex(qualifying.index, fill_value=0.0)
    syn_arr, ctpp_arr = syn_qualifying.to_numpy(), qualifying.to_numpy()

    summary = joint_test_summary(per_flow, syn_arr.sum(axis=0), ctpp_arr.sum(axis=0))
    floor = joint_floor_metrics(syn_bins, ctpp_bins, ctpp_flow_workers, MIN_FLOW_WORKERS_JOINT, seed=seed)
    floor["tvd_ratio_to_floor"] = _safe_ratio(summary["tvd_wmean"], floor["tvd_wmean_floor"])
    eff_bins = _effective_bins_medians(syn_arr, ctpp_arr)

    combined = {**summary, **floor, **eff_bins}
    rows = [{"method": "synthetic", "test": test_name, "geography": "tract", "metric": key,
            "value": value, "n_units": int(summary["n_flows"])} for key, value in combined.items()]
    rows.append({"method": "lodes", "test": test_name, "geography": "tract",
                "metric": "skipped_no_time_dimension", "value": 1.0, "n_units": 0})
    return rows


def _reference_rows(
    run_dir: Path, df100: pd.DataFrame, df103: pd.DataFrame, df104: pd.DataFrame,
    dep_tvd: float, tt_tvd: float, discard_all: dict[str, float],
) -> list[dict]:
    """County-level reference numbers: car-mode/WFH shares, CTPP coverage, drift, discards."""
    acs_non_wfh_total = float(load_input_census(run_dir, "census_depart_times", DEP_ACS_COLUMNS)["total"].sum())
    total_workers_all = float(df100["b302100_e1"].sum())
    car_workers_all = float(df103[CAR_MODE_COLUMNS].sum().sum())
    wfh_workers_all = float(df103["b302103_e18"].sum())
    ctpp_travelers_total = float(df104["b302104_e2"].sum())
    values = {
        "car_mode_share_of_travelers": car_workers_all / (total_workers_all - wfh_workers_all),
        "worked_at_home_share": wfh_workers_all / total_workers_all,
        "ctpp_coverage_of_acs_travelers": ctpp_travelers_total / acs_non_wfh_total,
        "ctpp_b302104_vs_acs_tvd": dep_tvd,
        "ctpp_b302106_vs_acs_tvd": tt_tvd,
        "ctpp_flows_discarded_outside_county": float(discard_all["n_discarded"]),
        "ctpp_workers_discarded_outside_county": float(discard_all["workers_discarded"]),
    }
    return [{"method": "reference", "test": "reference", "geography": "county", "metric": metric,
            "value": value, "n_units": 1} for metric, value in values.items()]


def _fetch_and_decode(state_fips: str, county_fips: str, year: int, cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Fetch (or load from cache) the four CTPP tables this module needs, and decode two of them."""
    tables = {name: fetch_flows(state_fips, county_fips, name, n_cols, year, cache_dir)
              for name, n_cols in (("b302100", 1), ("b302103", 18), ("b302104", 17), ("b302106", 15))}
    tables["decoded104"] = decode(tables["b302104"], "b302104")
    tables["decoded106"] = decode(tables["b302106"], "b302106")
    return tables


def _intra_county_variants(
    tables: dict[str, pd.DataFrame], county_fips5: str
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """The three CTPP flow-weight variants, each restricted to intra-county flows."""
    raw = {
        "tract_all_workers": flow_totals_all_workers(tables["b302100"]),
        "tract_travelers": flow_totals_travelers(tables["b302104"]),
        "tract_car_modes": flow_totals_car_modes(tables["b302103"]),
    }
    filtered, discards = {}, {}
    for name, flows in raw.items():
        filtered[name], discards[name] = filter_intra_county(flows, county_fips5)
        LOGGER.info("CTPP intra-county filter (%s): kept %d flow(s)/%d worker(s), discarded %d/%d",
                    name, discards[name]["n_kept"], discards[name]["workers_kept"],
                    discards[name]["n_discarded"], discards[name]["workers_discarded"])
    return filtered, discards


def run_ctpp_validation(
    run_dir: Path, calib_csv: Path, state: str, county: str, seed: int,
    year: int = CTPP_YEAR, strict_drift: bool = True, cache_dir: Path = Path("data/ctpp"),
) -> pd.DataFrame:
    """Run the CTPP held-out flow and per-flow-joint tests for one day of one run."""
    day = calib_csv.stem
    out_dir = run_dir / "validation" / day
    out_dir.mkdir(parents=True, exist_ok=True)

    trips = to_trip_frame(pd.read_csv(calib_csv))
    state_fips, county_fips = trips["origin_bg"].iloc[0][:2], trips["origin_bg"].iloc[0][2:5]
    county_fips5 = state_fips + county_fips
    LOGGER.info("CTPP validation for %s/%s %s (%s)", state, county, run_id_of(run_dir), day)

    tables = _fetch_and_decode(state_fips, county_fips, year, cache_dir)
    dep_tvd = departure_drift_check(run_dir, tables["decoded104"], strict=strict_drift)
    tt_tvd = traveltime_drift_check(run_dir, tables["decoded106"], strict=strict_drift)
    variants, discards = _intra_county_variants(tables, county_fips5)

    methods = {"synthetic": synthetic_tract_flows(trips), "lodes": load_lodes_tract_flows(run_dir)}
    rows = _ctpp_flows_rows(methods, variants)

    ctpp_flow_workers = variants["tract_travelers"].set_index(["origin_tract", "destination_tract"])["workers"]
    rows += _joint_rows("ctpp_departure_joint", trips, tables["decoded104"], DEP_BIN_LABELS,
                        synthetic_departure_matrix, ctpp_flow_workers, seed)
    rows += _joint_rows("ctpp_traveltime_joint", trips, tables["decoded106"], TT_BIN_LABELS,
                        synthetic_traveltime_matrix, ctpp_flow_workers, seed)
    rows += _reference_rows(run_dir, tables["b302100"], tables["b302103"], tables["b302104"],
                            dep_tvd, tt_tvd, discards["tract_all_workers"])

    metrics = _finalise_metrics(rows, state, county, run_id_of(run_dir), day, seed)
    metrics.to_csv(out_dir / "ctpp_metrics.csv", index=False)
    write_ctpp_summary(out_dir / "ctpp_summary.md", metrics, state, county, day)
    plot_flow_scatter(methods["synthetic"], variants["tract_all_workers"], out_dir / "ctpp_flow_scatter.png")
    LOGGER.info("Wrote %d CTPP metric row(s) to %s", len(metrics), out_dir / "ctpp_metrics.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CTPP Part 3 held-out flow validation for Move-OD.")
    parser.add_argument("--output-root", default="move_OD", help="Root output directory")
    parser.add_argument("--state", default="Tennessee", help="State name")
    parser.add_argument("--county", default="Hamilton", help="County name")
    parser.add_argument("--run-id", default=None, help="Run folder name (e.g., 2025-03-17_2025-03-17)")
    parser.add_argument("--year", type=int, default=CTPP_YEAR, help="CTPP data release year")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the metrics rows and the Monte Carlo noise floor")
    parser.add_argument("--no-strict-drift", dest="strict_drift", action="store_false", default=True,
                        help="Warn instead of raising when a CTPP table drifts from this run's ACS input")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = find_run_dir(Path(args.output_root), args.state, args.county, args.run_id)
    for csv_path in calibrated_csvs(run_dir) or [find_calibrated_csv(run_dir)]:
        metrics = run_ctpp_validation(run_dir, csv_path, args.state, args.county, args.seed,
                                      args.year, args.strict_drift)
        print((run_dir / "validation" / csv_path.stem / "ctpp_summary.md").read_text())
        LOGGER.info("Wrote %d metric rows for %s", len(metrics), csv_path.stem)


if __name__ == "__main__":
    main()
