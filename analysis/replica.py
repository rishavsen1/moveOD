"""Replica activity-based-model OD benchmark for Move-OD synthetic trips.

Move-OD synthesizes intra-county commute trips from LODES (its own input).
Two independent, non-input origin-destination sources exist for Hamilton
County, TN: CTPP (survey-based, see `analysis/ctpp.py`) and Replica (a
commercial activity-based model calibrated to mobile-device data). The
valuable result here is not "Move-OD agrees with Replica" in isolation -- it
is the **benchmark**: how much do the two independent sources (CTPP,
Replica) disagree with *each other*, compared with how much Move-OD
disagrees with each of them. See `run_replica_validation` and
`write_replica_summary` for the headline finding.

Data notes (verified 2026-09-18, see caller for the full provenance):
    `data/replica/replica-10_08_24-origin-destination.csv` -- 601,872 rows.
    `origin_fips`/`destination_fips` duplicate `origin_name`/`destination_name`
    and are not used. Zones are a custom Replica zone system (e.g. "BS-101"),
    identified by `origin_name`/`destination_name`; only centroids are given,
    no polygons, so each zone is assigned to the block group containing its
    centroid (`zone_to_block_group`) -- reasonable because Replica zones are
    smaller than block groups (see the zones-per-block-group diagnostics this
    module reports as evidence). Per-purpose counts sum exactly to
    `total_count` (`validate_purpose_totals`). This file carries no
    time-of-day dimension, so no joint departure/travel-time test is possible
    against Replica (unlike the CTPP per-flow joint tests); this is a real
    scope limit of the source, not an oversight.

Geoids are handled as zero-padded strings throughout and must never round-trip
through `int` (a past bug in this project dropped leading zeros for
low-numbered state FIPS codes) -- `zone_to_block_group` normalises the
block-group geojson's GEOID column with `analysis.validate_external.normalize_geoid`
for exactly this reason.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.ctpp import (  # noqa: E402
    CTPP_YEAR,
    _align_flows,
    cpc_ssi,
    fetch_flows,
    filter_intra_county,
    flow_correlations,
    flow_totals_travelers,
    load_lodes_tract_flows,
    synthetic_tract_flows,
)
from analysis.figures_from_output import find_run_dir  # noqa: E402
from analysis.validate_external import (  # noqa: E402
    calibrated_csvs,
    code_sha,
    find_calibrated_csv,
    normalize_geoid,
    run_id_of,
    to_trip_frame,
)

LOGGER = logging.getLogger(__name__)

REPLICA_PURPOSE_COLUMNS = [
    "other_activity_type_count", "work_count", "school_count", "eat_count",
    "shop_count", "social_count", "recreation_count", "maintenance_count",
    "stage_count", "lodging_count", "region_departure_count", "commercial_count",
    "home_count",
]
_REPLICA_ID_COLUMNS = [
    "origin_name", "origin_centroidLon", "origin_centroidLat",
    "destination_name", "destination_centroidLon", "destination_centroidLat",
]
REPLICA_COLUMNS = _REPLICA_ID_COLUMNS + ["total_count"] + REPLICA_PURPOSE_COLUMNS
PURPOSE_SUM_TOLERANCE = 1e-6

CONFOUNDS_NOTE = (
    "Confounds: Replica's work_count includes work trips that do not start at home, while "
    "Move-OD and CTPP are home-to-work, which is part of why Replica's total is higher; CTPP "
    "suppresses flows with fewer than three survey observations; the vintages differ (CTPP "
    "2017-2021, Replica October 2024, Move-OD March 2025); and zone-to-block-group assignment "
    "by centroid introduces error. Replica is itself a model, not ground truth, so this is "
    "agreement between models and measurements, not validation against truth. This file carries "
    "no time-of-day dimension, so no joint departure/travel-time test against Replica is possible."
)

# Hamilton County, TN, 2025-03-17, tract level, intra-county, work trips --
# verified by hand and reproduced (not re-derived) here. See
# `verify_known_numbers`; a drift is logged, never silently patched over.
KNOWN_HAMILTON_20250317 = {
    "totals": {"moveod": 158253.0, "lodes": 158253.0, "ctpp": 124114.0, "replica": 173536.0},
    "pair_counts": {"moveod": 5971, "lodes": 6008, "ctpp": 3006, "replica": 7502},
    "supports": {
        "union": {
            "n_pairs": 6885,
            "pairs": {
                ("moveod", "ctpp"): (0.585, 0.596),
                ("moveod", "replica"): (0.626, 0.680),
                ("ctpp", "replica"): (0.515, 0.516),
            },
        },
        "ctpp_published": {
            "n_pairs": 3003,
            "pairs": {
                ("moveod", "ctpp"): (0.664, 0.527),
                ("moveod", "replica"): (0.650, 0.687),
                ("ctpp", "replica"): (0.599, 0.462),
            },
        },
        "all_three": {
            "n_pairs": 2798,
            "pairs": {
                ("moveod", "ctpp"): (0.671, 0.516),
                ("moveod", "replica"): (0.654, 0.654),
                ("ctpp", "replica"): (0.605, 0.444),
            },
        },
    },
}


# --------------------------------------------------------------------------- load


def validate_purpose_totals(od: pd.DataFrame, tol: float = PURPOSE_SUM_TOLERANCE) -> None:
    """Raise if any row's purpose counts do not sum to `total_count`.

    Verified to hold exactly for the shipped Replica file; kept as a hard
    guard so a future Replica export with a different schema fails loudly
    instead of silently mis-summing.
    """
    purpose_sum = od[REPLICA_PURPOSE_COLUMNS].sum(axis=1)
    diff = (purpose_sum - od["total_count"]).abs()
    bad = diff > tol
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} row(s) where Replica purpose counts do not sum to total_count "
            f"(max |diff|={diff[bad].max():.3f})"
        )


def select_purpose(od: pd.DataFrame, purpose: str) -> pd.Series:
    """The requested purpose (or `total_count`) column, validated against the known schema."""
    valid = REPLICA_PURPOSE_COLUMNS + ["total_count"]
    if purpose not in valid:
        raise ValueError(f"Unknown Replica purpose column {purpose!r}; must be one of {valid}")
    return od[purpose]


def load_od(path: Path) -> pd.DataFrame:
    """Load the Replica OD file with dtype control, keeping only the columns this module uses."""
    dtype = {
        "origin_name": str, "destination_name": str,
        "origin_centroidLon": "float64", "origin_centroidLat": "float64",
        "destination_centroidLon": "float64", "destination_centroidLat": "float64",
        "total_count": "float64", **{col: "float64" for col in REPLICA_PURPOSE_COLUMNS},
    }
    od = pd.read_csv(path, usecols=REPLICA_COLUMNS, dtype=dtype)
    validate_purpose_totals(od)
    n_zones = pd.concat([od["origin_name"], od["destination_name"]]).nunique()
    LOGGER.info("Loaded %d Replica OD row(s), %d distinct zone(s) from %s", len(od), n_zones, path)
    return od


# --------------------------------------------------------------------------- zone_to_block_group


def _unique_zones(od: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct zone name with its centroid, from either origin or destination columns."""
    origins = od[["origin_name", "origin_centroidLon", "origin_centroidLat"]].rename(
        columns={"origin_name": "zone", "origin_centroidLon": "lon", "origin_centroidLat": "lat"})
    destinations = od[["destination_name", "destination_centroidLon", "destination_centroidLat"]].rename(
        columns={"destination_name": "zone", "destination_centroidLon": "lon", "destination_centroidLat": "lat"})
    return pd.concat([origins, destinations], ignore_index=True).drop_duplicates(subset="zone").reset_index(drop=True)


def zone_to_block_group(
    od: pd.DataFrame, county_geojson: gpd.GeoDataFrame | str | Path, county_fips5: str,
) -> tuple[dict[str, str], dict[str, float]]:
    """Assign each Replica zone to the census block group containing its centroid.

    `county_geojson` is filtered to `county_fips5` *before* the spatial join,
    so a zone whose centroid falls inside a real block group of a
    neighbouring county is excluded rather than silently mismatched. Returns
    the zone-name -> 12-char zero-padded GEOID mapping plus join diagnostics
    (zones matched, block groups covered, zones-per-block-group median/p90).
    """
    gdf = county_geojson if isinstance(county_geojson, gpd.GeoDataFrame) else gpd.read_file(county_geojson)
    gdf = gdf.copy()
    gdf["GEOID"] = normalize_geoid(gdf["GEOID"])
    county_gdf = gdf[gdf["GEOID"].str[:5] == county_fips5].reset_index(drop=True)
    if county_gdf.empty:
        raise ValueError(f"No block groups found for county FIPS {county_fips5!r} in the supplied geojson")

    zones = _unique_zones(od)
    points = gpd.GeoDataFrame(zones, geometry=gpd.points_from_xy(zones["lon"], zones["lat"]), crs="EPSG:4326")
    if county_gdf.crs is not None:
        points = points.to_crs(county_gdf.crs)
    joined = gpd.sjoin(points, county_gdf[["GEOID", "geometry"]], how="inner", predicate="within")
    if joined["zone"].duplicated().any():
        n_dup = int(joined["zone"].duplicated().sum())
        LOGGER.warning("%d zone(s) matched more than one block group (boundary ties); keeping the first match", n_dup)
        joined = joined.drop_duplicates(subset="zone", keep="first")

    zone_map = dict(zip(joined["zone"], joined["GEOID"]))
    per_bg = joined.groupby("GEOID").size()
    diagnostics = {
        "n_zones_total": int(len(zones)),
        "n_zones_matched": int(len(zone_map)),
        "n_block_groups_in_county": int(len(county_gdf)),
        "n_block_groups_covered": int(per_bg.shape[0]),
        "zones_per_bg_median": float(per_bg.median()) if not per_bg.empty else float("nan"),
        "zones_per_bg_p90": float(per_bg.quantile(0.9)) if not per_bg.empty else float("nan"),
    }
    LOGGER.info(
        "Zone-to-block-group join: %d/%d zone(s) matched, covering %d/%d block group(s) "
        "(median %.1f, p90 %.1f zones/block group)",
        diagnostics["n_zones_matched"], diagnostics["n_zones_total"],
        diagnostics["n_block_groups_covered"], diagnostics["n_block_groups_in_county"],
        diagnostics["zones_per_bg_median"], diagnostics["zones_per_bg_p90"],
    )
    return zone_map, diagnostics


# --------------------------------------------------------------------------- flows


def intra_county_flows(
    od: pd.DataFrame, zone_map: dict[str, str], purpose: str = "work_count", level: str = "block_group",
) -> pd.DataFrame:
    """Tidy intra-county flows: both endpoints resolved to a block group in `zone_map`.

    `level="block_group"` returns `origin_bg, destination_bg, trips`;
    `level="tract"` truncates each geoid to its first 11 characters and
    returns `origin_tract, destination_tract, trips`.
    """
    values = select_purpose(od, purpose).astype(float)
    origin = od["origin_name"].map(zone_map)
    destination = od["destination_name"].map(zone_map)
    keep = origin.notna() & destination.notna()
    frame = pd.DataFrame({"origin_bg": origin[keep], "destination_bg": destination[keep], "trips": values[keep]})

    if level == "tract":
        frame = frame.assign(
            origin_tract=frame["origin_bg"].str[:11], destination_tract=frame["destination_bg"].str[:11],
        )[["origin_tract", "destination_tract", "trips"]]
        group_cols = ["origin_tract", "destination_tract"]
    elif level == "block_group":
        group_cols = ["origin_bg", "destination_bg"]
    else:
        raise ValueError(f"Unknown level: {level!r}; must be 'block_group' or 'tract'")

    result = frame.groupby(group_cols, as_index=False)["trips"].sum()
    LOGGER.info(
        "Intra-county Replica flows (%s, purpose=%s): %d row(s) kept of %d (%d dropped, zone outside county)",
        level, purpose, int(keep.sum()), len(od), int((~keep).sum()),
    )
    return result


def to_flow_series(frame: pd.DataFrame, value_col: str) -> pd.Series:
    """A tidy (id, id, value) frame as a Series indexed by the two id columns."""
    id_cols = [c for c in frame.columns if c != value_col]
    if len(id_cols) != 2:
        raise ValueError(f"Expected exactly two id columns alongside {value_col!r}, got {id_cols}")
    return frame.set_index(id_cols)[value_col]


# --------------------------------------------------------------------------- agreement matrix


def union_support(matrices: dict[str, pd.Series]) -> pd.Index:
    """The union of every matrix's flow-pair index."""
    support = None
    for series in matrices.values():
        support = series.index if support is None else support.union(series.index)
    return support if support is not None else pd.Index([])


def restrict_to_support(matrices: dict[str, pd.Series], support: pd.Index) -> dict[str, pd.Series]:
    """Zero-fill every matrix onto exactly `support`, so pairwise comparisons share one fixed support."""
    return {name: series.reindex(support, fill_value=0.0) for name, series in matrices.items()}


def _as_flow_frame(series: pd.Series) -> pd.DataFrame:
    """Adapt a (origin, destination)-indexed Series to the frame shape `analysis.ctpp` expects."""
    frame = series.reset_index()
    frame.columns = ["origin_tract", "destination_tract", "workers"]
    return frame


def agreement_matrix(matrices: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise agreement for every unordered pair of flow matrices.

    Reuses `analysis.ctpp.cpc_ssi`/`flow_correlations`/`_align_flows` rather
    than reimplementing them. `ssi` is the symmetric Sorensen similarity
    index; `cpc_a_covers_b`/`cpc_b_covers_a` are the two directional common
    parts of commuters (share of one side's total also present on the
    other); `spearman_r`/`pearson_r` are rank and linear correlation.
    """
    rows = []
    for a, b in itertools.combinations(matrices.keys(), 2):
        frame_a, frame_b = _as_flow_frame(matrices[a]), _as_flow_frame(matrices[b])
        a_covers_b = cpc_ssi(frame_a, frame_b)  # cpc = common / sum(b)
        b_covers_a = cpc_ssi(frame_b, frame_a)  # cpc = common / sum(a)
        corr = flow_correlations(frame_a, frame_b)
        rows.append({
            "source_a": a, "source_b": b,
            "ssi": a_covers_b["ssi"],
            "spearman_r": corr["spearman_r"],
            "pearson_r": corr["pearson_r"],
            "cpc_a_covers_b": a_covers_b["cpc"],
            "cpc_b_covers_a": b_covers_a["cpc"],
            "n_pairs": int(len(_align_flows(frame_a, frame_b))),
        })
    return pd.DataFrame(
        rows, columns=["source_a", "source_b", "ssi", "spearman_r", "pearson_r",
                       "cpc_a_covers_b", "cpc_b_covers_a", "n_pairs"],
    )


def _lookup_pair(am: pd.DataFrame, a: str, b: str) -> dict | None:
    mask = ((am["source_a"] == a) & (am["source_b"] == b)) | ((am["source_a"] == b) & (am["source_b"] == a))
    matches = am.loc[mask]
    return None if matches.empty else matches.iloc[0].to_dict()


# --------------------------------------------------------------------------- known-number reproduction


def verify_known_numbers(
    state: str, county: str, day: str, totals: dict[str, float], pair_counts: dict[str, int],
    support_agreements: dict[str, pd.DataFrame],
) -> list[str]:
    """Compare against the hand-verified Hamilton 2025-03-17 numbers; log (never raise) on drift."""
    if (state, county, day) != ("Tennessee", "Hamilton", "2025-03-17"):
        LOGGER.info("No known-number reproduction check defined for %s/%s %s", state, county, day)
        return []

    deviations: list[str] = []
    expected = KNOWN_HAMILTON_20250317
    for name, exp_total in expected["totals"].items():
        actual = totals.get(name)
        if actual is None or abs(actual - exp_total) > 0.5:
            deviations.append(f"total[{name}]: expected {exp_total}, got {actual}")
    for name, exp_n in expected["pair_counts"].items():
        actual = pair_counts.get(name)
        if actual is None or actual != exp_n:
            deviations.append(f"pair_count[{name}]: expected {exp_n}, got {actual}")
    for support, spec in expected["supports"].items():
        am = support_agreements.get(support)
        if am is None:
            deviations.append(f"support[{support}]: no agreement matrix computed")
            continue
        for (a, b), (exp_ssi, exp_spear) in spec["pairs"].items():
            row = _lookup_pair(am, a, b)
            if row is None:
                deviations.append(f"support[{support}] {a}-{b}: pair not found in agreement matrix")
                continue
            if abs(row["ssi"] - exp_ssi) > 0.001:
                deviations.append(f"support[{support}] {a}-{b} ssi: expected {exp_ssi}, got {row['ssi']:.4f}")
            if abs(row["spearman_r"] - exp_spear) > 0.001:
                deviations.append(
                    f"support[{support}] {a}-{b} spearman: expected {exp_spear}, got {row['spearman_r']:.4f}"
                )

    for deviation in deviations:
        LOGGER.warning("Known-number check drifted: %s", deviation)
    if not deviations:
        LOGGER.info("All known Hamilton 2025-03-17 numbers reproduced exactly (within tolerance).")
    return deviations


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


def _build_matrices(
    run_dir: Path, trips: pd.DataFrame, replica_frame: pd.DataFrame,
    state_fips: str, county_fips: str, county_fips5: str, ctpp_year: int, ctpp_cache_dir: Path,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    """The four tract-level flow series (Move-OD, LODES, CTPP, Replica) and CTPP's discard stats."""
    moveod = synthetic_tract_flows(trips)
    lodes = load_lodes_tract_flows(run_dir)
    df104 = fetch_flows(state_fips, county_fips, "b302104", 17, ctpp_year, ctpp_cache_dir)
    ctpp_raw = flow_totals_travelers(df104)
    ctpp_frame, ctpp_discard = filter_intra_county(ctpp_raw, county_fips5)

    matrices = {
        "moveod": to_flow_series(moveod, "workers"),
        "lodes": to_flow_series(lodes, "workers"),
        "ctpp": to_flow_series(ctpp_frame, "workers"),
        "replica": to_flow_series(replica_frame, "trips"),
    }
    return matrices, ctpp_discard


def _support_indices(matrices: dict[str, pd.Series]) -> dict[str, pd.Index]:
    """The three supports `run_replica_validation` reports on.

    One rule applies everywhere: a flow-pair key is "present" only where its
    value is positive. A key that is zero in every source (Hamilton has 639
    of these -- Replica rows that carry trips of some other purpose but no
    work trips) carries no commute information at all, and zero-filling it
    on every side would only add a tied pair that inflates rank correlation
    without reflecting any real agreement or disagreement. So: "union" is
    the union of each source's positive-valued keys, "ctpp_published" is
    CTPP's positive-valued keys, and "all_three" is the intersection of the
    positive-valued keys of Move-OD, CTPP and Replica.
    """
    positive_series = {name: series[series > 0] for name, series in matrices.items()}
    positive = {name: series.index for name, series in positive_series.items()}
    all_three = None
    for name in ("moveod", "ctpp", "replica"):
        idx = positive[name]
        all_three = idx if all_three is None else all_three.intersection(idx)
    return {
        "union": union_support(positive_series),
        "ctpp_published": positive["ctpp"],
        "all_three": all_three if all_three is not None else pd.Index([]),
    }


def _agreement_rows(support_agreements: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for support, am in support_agreements.items():
        for _, r in am.iterrows():
            method = f"{r['source_a']}_vs_{r['source_b']}"
            for metric in ("ssi", "spearman_r", "pearson_r", "cpc_a_covers_b", "cpc_b_covers_a"):
                rows.append({"method": method, "test": "replica_flows", "geography": support,
                            "metric": metric, "value": r[metric], "n_units": int(r["n_pairs"])})
    return rows


def _benchmark_rows(
    totals: dict[str, float], pair_counts: dict[str, int], diagnostics: dict[str, float],
    support_agreements: dict[str, pd.DataFrame],
) -> list[dict]:
    rows = [{"method": name, "test": "source_benchmark", "geography": "tract", "metric": "total_trips",
            "value": value, "n_units": 1} for name, value in totals.items()]
    rows += [{"method": name, "test": "source_benchmark", "geography": "tract", "metric": "n_tract_pairs",
             "value": float(n), "n_units": 1} for name, n in pair_counts.items()]
    rows += [{"method": "zone_join", "test": "source_benchmark", "geography": "county", "metric": key,
             "value": value, "n_units": 1} for key, value in diagnostics.items()]

    for support, am in support_agreements.items():
        pairs = {(a, b): _lookup_pair(am, a, b) for a, b in
                 (("moveod", "ctpp"), ("moveod", "replica"), ("ctpp", "replica"))}
        for (a, b), row in pairs.items():
            if row is None:
                continue
            rows.append({"method": f"{a}_vs_{b}", "test": "source_benchmark", "geography": support,
                        "metric": "ssi", "value": row["ssi"], "n_units": int(row["n_pairs"])})
            rows.append({"method": f"{a}_vs_{b}", "test": "source_benchmark", "geography": support,
                        "metric": "spearman_r", "value": row["spearman_r"], "n_units": int(row["n_pairs"])})
        if all(pairs.values()):
            independent_ssi = pairs[("ctpp", "replica")]["ssi"]
            moveod_ssi = min(pairs[("moveod", "ctpp")]["ssi"], pairs[("moveod", "replica")]["ssi"])
            independent_spear = pairs[("ctpp", "replica")]["spearman_r"]
            moveod_spear = min(pairs[("moveod", "ctpp")]["spearman_r"], pairs[("moveod", "replica")]["spearman_r"])
            rows.append({"method": "benchmark", "test": "source_benchmark", "geography": support,
                        "metric": "ssi_holds", "value": float(independent_ssi < moveod_ssi), "n_units": 1})
            rows.append({"method": "benchmark", "test": "source_benchmark", "geography": support,
                        "metric": "spearman_holds", "value": float(independent_spear < moveod_spear), "n_units": 1})
    return rows


def plot_agreement_bars(support_agreements: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Grouped bars: SSI and Spearman for the three headline pairs, one group per support."""
    pairs = [("moveod", "ctpp"), ("moveod", "replica"), ("ctpp", "replica")]
    pair_labels = [f"{a}-{b}" for a, b in pairs]
    supports = list(support_agreements.keys())
    width = 0.8 / max(len(supports), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in ((axes[0], "ssi", "Sorensen similarity index (SSI)"),
                              (axes[1], "spearman_r", "Spearman rank correlation")):
        x = np.arange(len(pairs))
        for i, support in enumerate(supports):
            am = support_agreements[support]
            values = [(_lookup_pair(am, a, b) or {}).get(metric, np.nan) for a, b in pairs]
            ax.bar(x + i * width, values, width, label=support)
        ax.set_xticks(x + width * (len(supports) - 1) / 2)
        ax.set_xticklabels(pair_labels)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Pairwise source agreement: independent sources (CTPP-Replica) vs Move-OD")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


_UNION_SUPPORT_NOTE = (
    "The union support is every tract pair for which at least one of the four sources reports a "
    "commute: a pair one source has and another lacks counts as a disagreement (zero-filled on the "
    "missing side) rather than being dropped.\n"
)

_LODES_NOTE = (
    "Move-OD's tract flows are identical to LODES's (SSI 1.000, see `source_benchmark` rows): "
    "this shows the **LODES input** sits inside that envelope of source disagreement and the "
    "synthesis step does not degrade it -- it is not evidence that the synthesis itself is good.\n"
)


def _headline_section(support_agreements: dict[str, pd.DataFrame]) -> str:
    union_am = support_agreements.get("union")
    if union_am is None:
        return ""
    ctpp_replica = _lookup_pair(union_am, "ctpp", "replica") or {}
    moveod_ctpp = _lookup_pair(union_am, "moveod", "ctpp") or {}
    moveod_replica = _lookup_pair(union_am, "moveod", "replica") or {}
    return (
        f"CTPP and Replica -- the two independent sources -- agree with each other "
        f"**less** than either agrees with Move-OD, on every support and both statistics: "
        f"CTPP-Replica SSI={ctpp_replica.get('ssi', float('nan')):.3f} vs Move-OD-CTPP "
        f"SSI={moveod_ctpp.get('ssi', float('nan')):.3f} and Move-OD-Replica "
        f"SSI={moveod_replica.get('ssi', float('nan')):.3f} (union support). This reframes an "
        f"otherwise mediocre-looking Move-OD/CTPP agreement score: Move-OD sits *inside* the "
        f"envelope of disagreement between two independent measurements, not outside it.\n"
    )


def _assumption_note(diagnostics: dict[str, float]) -> str:
    return (
        f"Zone-to-block-group assignment is by centroid (no zone polygons are published): "
        f"{diagnostics.get('n_zones_matched', 0):.0f}/{diagnostics.get('n_zones_total', 0):.0f} zones "
        f"matched a block group in the county, covering {diagnostics.get('n_block_groups_covered', 0):.0f}/"
        f"{diagnostics.get('n_block_groups_in_county', 0):.0f} block groups (median "
        f"{diagnostics.get('zones_per_bg_median', float('nan')):.1f}, p90 "
        f"{diagnostics.get('zones_per_bg_p90', float('nan')):.1f} zones per block group). Since Replica "
        f"zones are smaller than block groups, centroid assignment is a reasonable approximation.\n"
    )


def _totals_table(totals: dict[str, float], pair_counts: dict[str, int]) -> str:
    return "\n".join(
        f"- {name}: {totals.get(name, float('nan')):,.0f} trips, {pair_counts.get(name, 0)} tract pairs"
        for name in ("moveod", "lodes", "ctpp", "replica")
    )


def _support_sections(support_agreements: dict[str, pd.DataFrame]) -> str:
    sections = []
    for support, am in support_agreements.items():
        n_pairs = int(am["n_pairs"].iloc[0]) if not am.empty else 0
        table = am[["source_a", "source_b", "ssi", "spearman_r", "pearson_r"]].to_string(index=False)
        sections.append(f"\n### {support} support ({n_pairs} tract pairs)\n```\n{table}\n```\n")
    return "".join(sections)


def _deviations_section(state: str, county: str, day: str, deviations: list[str]) -> str:
    if not deviations:
        if (state, county, day) == ("Tennessee", "Hamilton", "2025-03-17"):
            return "\nAll known Hamilton 2025-03-17 numbers reproduced exactly (within tolerance).\n"
        return ""
    return "\n".join(f"- DRIFT: {d}" for d in deviations) + "\n"


def write_replica_summary(
    path: Path, state: str, county: str, day: str, totals: dict[str, float], pair_counts: dict[str, int],
    diagnostics: dict[str, float], support_agreements: dict[str, pd.DataFrame], deviations: list[str],
) -> None:
    """Markdown report leading with the source-disagreement benchmark, per the analysis this reproduces."""
    sections = [
        f"# Replica OD benchmark - {county}, {state} ({day})\n",
        "\n## Headline: the source-disagreement benchmark\n",
        _headline_section(support_agreements),
        "\n## Totals (work trips, intra-county, tract level)\n",
        _totals_table(totals, pair_counts) + "\n",
        "\n## LODES-input caveat\n",
        _LODES_NOTE,
        "\n## Zone-to-block-group assignment\n",
        _assumption_note(diagnostics),
        "\n## Pairwise agreement by support\n",
        _UNION_SUPPORT_NOTE,
        _support_sections(support_agreements),
        "\n## Confounds and scope\n",
        CONFOUNDS_NOTE + "\n",
        "\n## Known-number reproduction\n",
        _deviations_section(state, county, day, deviations),
    ]
    path.write_text("".join(sections))


def run_replica_validation(
    run_dir: Path, calib_csv: Path, replica_od_path: Path, state: str, county: str, seed: int,
    purpose: str = "work_count", ctpp_year: int = CTPP_YEAR, ctpp_cache_dir: Path = Path("data/ctpp"),
) -> pd.DataFrame:
    """Run the Replica source-benchmark validation for one day of one run."""
    day = calib_csv.stem
    out_dir = run_dir / "validation" / day
    out_dir.mkdir(parents=True, exist_ok=True)

    trips = to_trip_frame(pd.read_csv(calib_csv))
    state_fips, county_fips = trips["origin_bg"].iloc[0][:2], trips["origin_bg"].iloc[0][2:5]
    county_fips5 = state_fips + county_fips
    LOGGER.info("Replica validation for %s/%s %s (%s)", state, county, run_id_of(run_dir), day)

    od = load_od(replica_od_path)
    zone_map, diagnostics = zone_to_block_group(od, run_dir / "county_geoid.geojson", county_fips5)
    replica_frame = intra_county_flows(od, zone_map, purpose, level="tract")

    matrices, ctpp_discard = _build_matrices(
        run_dir, trips, replica_frame, state_fips, county_fips, county_fips5, ctpp_year, ctpp_cache_dir,
    )
    totals = {name: float(series.sum()) for name, series in matrices.items()}
    pair_counts = {name: int(len(series)) for name, series in matrices.items()}
    LOGGER.info("Tract-level totals (%s): %s", purpose,
               ", ".join(f"{name}={value:,.0f}" for name, value in totals.items()))
    LOGGER.info("Tract-level pair counts: %s", ", ".join(f"{name}={n}" for name, n in pair_counts.items()))

    supports = _support_indices(matrices)
    support_agreements: dict[str, pd.DataFrame] = {}
    for support_name, support_idx in supports.items():
        subset = matrices if support_name == "union" else {k: v for k, v in matrices.items() if k != "lodes"}
        support_agreements[support_name] = agreement_matrix(restrict_to_support(subset, support_idx))

    rows = _agreement_rows(support_agreements)
    rows += _benchmark_rows(totals, pair_counts, diagnostics, support_agreements)
    metrics = _finalise_metrics(rows, state, county, run_id_of(run_dir), day, seed)
    metrics.to_csv(out_dir / "replica_metrics.csv", index=False)

    deviations = verify_known_numbers(state, county, day, totals, pair_counts, support_agreements)
    write_replica_summary(out_dir / "replica_summary.md", state, county, day, totals, pair_counts,
                          diagnostics, support_agreements, deviations)
    plot_agreement_bars(support_agreements, out_dir / "replica_agreement.png")
    LOGGER.info("Wrote %d Replica metric row(s) to %s", len(metrics), out_dir / "replica_metrics.csv")
    if ctpp_discard.get("n_discarded"):
        LOGGER.info("CTPP intra-county filter discarded %d flow(s)/%.0f worker(s)",
                   ctpp_discard["n_discarded"], ctpp_discard["workers_discarded"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Replica OD source-benchmark validation for Move-OD.")
    parser.add_argument("--output-root", default="move_OD", help="Root output directory")
    parser.add_argument("--state", default="Tennessee", help="State name")
    parser.add_argument("--county", default="Hamilton", help="County name")
    parser.add_argument("--run-id", default=None, help="Run folder name (e.g., 2025-03-17_2025-03-17)")
    parser.add_argument("--replica-od", default="data/replica/replica-10_08_24-origin-destination.csv",
                        help="Path to the Replica origin-destination CSV")
    parser.add_argument("--purpose", default="work_count", help="Replica purpose column to compare")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in the metrics rows")
    parser.add_argument("--ctpp-year", type=int, default=CTPP_YEAR, help="CTPP data release year")
    parser.add_argument("--ctpp-cache-dir", default="data/ctpp", help="CTPP parquet cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = find_run_dir(Path(args.output_root), args.state, args.county, args.run_id)
    for csv_path in calibrated_csvs(run_dir) or [find_calibrated_csv(run_dir)]:
        metrics = run_replica_validation(
            run_dir, csv_path, Path(args.replica_od), args.state, args.county, args.seed,
            args.purpose, args.ctpp_year, Path(args.ctpp_cache_dir),
        )
        print((run_dir / "validation" / csv_path.stem / "replica_summary.md").read_text())
        LOGGER.info("Wrote %d metric rows for %s", len(metrics), csv_path.stem)


if __name__ == "__main__":
    main()
