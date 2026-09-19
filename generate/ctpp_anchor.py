"""CTPP-anchored initial OD distribution for the ILP calibration (opt-in).

`get_initial_od_dist` builds the ILP's anchor `w[(s, d)]` from an assignment
that draws each commuter's departure block at random *within the origin*
(`LodesComb.assign_departure_times_by_cbg` groups by `h_geocode` only), so the
anchor encodes `P(S | D, O) = P(S | O)`: in expectation it is the product of
the two marginals the hard constraints already pin. The joint cell value is
therefore unconstrained by anything observed, and measured per-flow departure
profiles miss CTPP B302104 by far more than sampling noise explains.

CTPP B302104 publishes, per home-tract to work-tract pair, the workers leaving
in each of the 14 ACS departure blocks. Writing tau(.) for the 12-char block
group -> 11-char tract truncation and

    q[T, T', s] = c[T, T', s] / sum_s' c[T, T', s']

this module replaces *only* the anchor, per origin:

    w'[(s, d)] = p_od[d] * q[tau(o), tau(d), s]   where the tract pair is published
    w'[(s, d)] = w[(s, d)]                        otherwise

with `p_od[d] = sum_s w[(s, d)]` taken from today's anchor, so the O-D marginal
of the anchor is preserved exactly (constraint (2) of the ILP depends on it,
and flow-level validation has to stay comparable). Two guards apply: `q` is
restricted to the departure blocks the candidate frame actually offers for that
(o, d) -- the ILP creates no variable for a cell with no candidate row, so
anchor mass outside that set would be silently discarded -- and a published
pair whose mass inside those blocks is zero falls back to today's anchor.

Opt-in: nothing here runs unless `calibrate_with_ilp(..., ctpp_anchor=True)`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TRACT_LEN = 11
N_DEPARTURE_BINS = 14
ANCHOR_SUM_TOLERANCE = 1e-9

SharesByPair = dict[tuple[str, str], np.ndarray]


def departure_shares_by_tract_pair(decoded: pd.DataFrame, dep_labels: list[str]) -> SharesByPair:
    """q[T, T', .] as a unit-sum array per published tract pair, in `dep_labels` order.

    `decoded` is `analysis.ctpp.decode(..., "b302104")` output: long rows of
    origin_tract / destination_tract / category / workers, whose `category`
    already carries CTPP's 5 a.m.-first ordering translated to ACS labels.
    Pairs with no travellers in any block are omitted rather than kept as a
    zero row, so `apply_ctpp_anchor` treats them as unpublished.
    """
    subset = decoded[decoded["category"].isin(dep_labels)]
    pivot = subset.pivot_table(index=["origin_tract", "destination_tract"], columns="category",
                               values="workers", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(columns=dep_labels, fill_value=0.0)
    totals = pivot.to_numpy(dtype=float).sum(axis=1)
    shares: SharesByPair = {}
    for (origin_tract, destination_tract), row, total in zip(pivot.index, pivot.to_numpy(dtype=float), totals):
        if total > 0:
            shares[(str(origin_tract), str(destination_tract))] = row / total
    return shares


def candidate_bins_by_pair(cand: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    """Departure blocks the candidate frame offers for each (origin, destination).

    These are exactly the cells `_process_single_origin` creates a variable
    for, so they bound where anchor mass can usefully land.
    """
    unique = cand[["origin_geoid", "destination_geoid", "departure_time_bin"]].drop_duplicates()
    bins: dict[tuple[str, str], list[int]] = {}
    for origin, destination, dep_bin in unique.itertuples(index=False):
        bins.setdefault((str(origin), str(destination)), []).append(int(dep_bin))
    return {key: sorted(value) for key, value in bins.items()}


def select_anchor_pairs(pairs, seed: int, fraction: float = 0.5) -> set[tuple[str, str]]:
    """A seeded random `fraction` of `pairs`, for the split-half held-out evaluation.

    Sorted first so the selection depends on the pair set and the seed alone,
    never on the order the caller happened to build it in.
    """
    ordered = sorted(set(pairs))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(ordered), size=int(round(len(ordered) * fraction)), replace=False)
    return {ordered[i] for i in chosen}


def _anchored_cells(
    profile: np.ndarray, available: list[int], p_od: float, destination: str,
) -> tuple[dict[tuple[int, str], float], float] | None:
    """Cells for one (o, d), or None when the pair has no usable mass to place."""
    mass = float(profile[available].sum())
    if mass <= 0:
        return None
    return {(s, destination): p_od * float(profile[s]) / mass for s in available}, mass


def _by_destination(cells: dict) -> dict[str, dict[tuple[int, str], float]]:
    grouped: dict[str, dict[tuple[int, str], float]] = {}
    for key, share in cells.items():
        grouped.setdefault(key[1], {})[key] = share
    return grouped


def apply_ctpp_anchor(
    w_dict: dict, cand: pd.DataFrame, shares: SharesByPair, origin_workers: dict | None = None,
) -> tuple[dict, dict]:
    """Rebuild `w_dict`'s joint cells from CTPP per-flow departure profiles.

    Returns the transformed `w_dict` plus a stats dict (anchored/fallback flow
    counts and the worker-weighted share of CTPP departure mass that survived
    the restriction to available candidate blocks). Every destination's total
    anchor share is preserved to `ANCHOR_SUM_TOLERANCE`, asserted per flow.
    """
    available_bins = candidate_bins_by_pair(cand)
    stats = {"n_flows_anchored": 0, "n_flows_unpublished": 0, "n_flows_no_candidates": 0,
             "n_flows_zero_available_mass": 0, "n_cells_anchored": 0}
    mass_kept = mass_total = 0.0

    anchored_w: dict = {}
    for origin, cells in w_dict.items():
        origin_tract = str(origin)[:TRACT_LEN]
        weight = float(origin_workers.get(origin, 0)) if origin_workers is not None else 1.0
        out: dict[tuple[int, str], float] = {}
        for destination, flow_cells in _by_destination(cells).items():
            p_od = float(sum(flow_cells.values()))
            profile = shares.get((origin_tract, str(destination)[:TRACT_LEN]))
            available = available_bins.get((str(origin), str(destination)))
            if profile is None:
                stats["n_flows_unpublished"] += 1
            elif not available:
                stats["n_flows_no_candidates"] += 1
            else:
                anchored = _anchored_cells(profile, available, p_od, destination)
                if anchored is None:
                    stats["n_flows_zero_available_mass"] += 1
                else:
                    new_cells, mass = anchored
                    assert abs(sum(new_cells.values()) - p_od) <= ANCHOR_SUM_TOLERANCE, (
                        f"CTPP anchor moved the O-D marginal for {origin}->{destination}: "
                        f"{sum(new_cells.values())!r} != {p_od!r}"
                    )
                    out.update(new_cells)
                    stats["n_flows_anchored"] += 1
                    stats["n_cells_anchored"] += len(new_cells)
                    mass_kept += p_od * weight * mass
                    mass_total += p_od * weight
                    continue
            out.update(flow_cells)
        anchored_w[origin] = out

    stats["mass_retained"] = mass_kept / mass_total if mass_total > 0 else float("nan")
    LOGGER.info(
        "CTPP anchor: %d flow(s) anchored (%d cells), %d unpublished, %d without candidates, "
        "%d with zero available mass; %.1f%% of CTPP departure mass retained after restricting "
        "to available blocks",
        stats["n_flows_anchored"], stats["n_cells_anchored"], stats["n_flows_unpublished"],
        stats["n_flows_no_candidates"], stats["n_flows_zero_available_mass"],
        100.0 * stats["mass_retained"],
    )
    return anchored_w, stats


def load_ctpp_departure_shares(
    state_fips: str, county_fips: str, year: int = 2021, cache_dir: Path = Path("data/ctpp"),
) -> SharesByPair:
    """Fetch (or read from the parquet cache) B302104 and reduce it to q[T, T', .].

    `analysis.ctpp` is imported here rather than at module scope: it pulls in
    matplotlib, scipy and requests, none of which the default pipeline needs.
    """
    from analysis.ctpp import TABLE_SPECS, decode, fetch_flows
    from analysis.validate_external import DEP_BIN_LABELS

    frame = fetch_flows(state_fips, county_fips, "b302104", int(TABLE_SPECS["b302104"]["n_cols"]),
                        year, cache_dir)
    shares = departure_shares_by_tract_pair(decode(frame, "b302104"), DEP_BIN_LABELS)
    LOGGER.info("CTPP b302104 %d: %d tract pair(s) with a published departure profile", year, len(shares))
    return shares
