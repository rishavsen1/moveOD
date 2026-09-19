"""Unit tests for the opt-in CTPP-anchored initial OD distribution.

Every test runs on toy in-memory frames; nothing here touches the network or
the CTPP parquet cache (`load_ctpp_departure_shares` is the only function that
does, and it is a thin wrapper over `analysis.ctpp`).
"""

import numpy as np
import pandas as pd
import pytest

from analysis.validate_external import DEP_BIN_LABELS
from generate.calibrate_ilp import get_initial_od_dist
from generate.ctpp_anchor import (
    N_DEPARTURE_BINS,
    apply_ctpp_anchor,
    candidate_bins_by_pair,
    departure_shares_by_tract_pair,
    select_anchor_pairs,
)

ORIGIN = "470650001001"
DEST_A = "470650002001"
DEST_B = "470650003001"


def _cand(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["origin_geoid", "destination_geoid", "departure_time_bin"])


def _shares(**by_pair: list[float]) -> dict[tuple[str, str], np.ndarray]:
    """Keyed by a `<origin>|<destination>` tract-pair string for kwarg friendliness."""
    return {tuple(key.split("|")): np.asarray(value, dtype=float) for key, value in by_pair.items()}


def _profile(**weight_by_bin: float) -> list[float]:
    row = [0.0] * N_DEPARTURE_BINS
    for bin_index, weight in weight_by_bin.items():
        row[int(bin_index.lstrip("b"))] = weight
    return row


# --------------------------------------------------------------------------- decode


def test_departure_shares_normalise_per_tract_pair():
    decoded = pd.DataFrame(
        [("47065000100", "47065000200", label, workers)
         for label, workers in zip(DEP_BIN_LABELS, [0, 0, 0, 0, 0, 30, 10, 0, 0, 0, 0, 0, 0, 0])]
        + [("47065000100", "47065000200", "worked_at_home", 999.0)],
        columns=["origin_tract", "destination_tract", "category", "workers"],
    )
    shares = departure_shares_by_tract_pair(decoded, DEP_BIN_LABELS)
    profile = shares[("47065000100", "47065000200")]
    assert profile.sum() == pytest.approx(1.0)
    assert profile[5] == pytest.approx(0.75)
    assert profile[6] == pytest.approx(0.25)


def test_departure_shares_drop_pairs_with_no_travellers():
    decoded = pd.DataFrame(
        [("47065000100", "47065000200", label, 0.0) for label in DEP_BIN_LABELS],
        columns=["origin_tract", "destination_tract", "category", "workers"],
    )
    assert departure_shares_by_tract_pair(decoded, DEP_BIN_LABELS) == {}


def test_departure_shares_keep_leading_zero_tract_ids():
    """Alabama is state FIPS 01: a tract id that round-trips through int loses the zero."""
    decoded = pd.DataFrame(
        [("01001020100", "01001020200", label, workers)
         for label, workers in zip(DEP_BIN_LABELS, [1.0] + [0.0] * 13)],
        columns=["origin_tract", "destination_tract", "category", "workers"],
    )
    assert ("01001020100", "01001020200") in departure_shares_by_tract_pair(decoded, DEP_BIN_LABELS)


# --------------------------------------------------------------------------- candidate bins


def test_candidate_bins_by_pair_collects_available_blocks():
    cand = _cand([(ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 7), (ORIGIN, DEST_B, 1)])
    bins = candidate_bins_by_pair(cand)
    assert bins[(ORIGIN, DEST_A)] == [5, 7]
    assert bins[(ORIGIN, DEST_B)] == [1]


# --------------------------------------------------------------------------- the transform


def test_anchor_preserves_the_per_destination_sum_exactly():
    w_dict = {ORIGIN: {(5, DEST_A): 0.3, (7, DEST_A): 0.1, (5, DEST_B): 0.6}}
    cand = _cand([(ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 7), (ORIGIN, DEST_B, 5), (ORIGIN, DEST_B, 7)])
    shares = _shares(**{
        "47065000100|47065000200": _profile(b5=1.0, b7=3.0),
        "47065000100|47065000300": _profile(b5=1.0, b7=1.0),
    })
    anchored, _ = apply_ctpp_anchor(w_dict, cand, shares)

    for dest, expected in ((DEST_A, 0.4), (DEST_B, 0.6)):
        total = sum(value for (_, d), value in anchored[ORIGIN].items() if d == dest)
        assert total == pytest.approx(expected, abs=1e-12)
    assert anchored[ORIGIN][(5, DEST_A)] == pytest.approx(0.1)
    assert anchored[ORIGIN][(7, DEST_A)] == pytest.approx(0.3)
    assert anchored[ORIGIN][(5, DEST_B)] == pytest.approx(0.3)
    assert anchored[ORIGIN][(7, DEST_B)] == pytest.approx(0.3)


def test_anchor_renormalises_over_available_blocks_only():
    """Half of CTPP's mass sits in a block the candidate set has no variable for."""
    w_dict = {ORIGIN: {(5, DEST_A): 1.0}}
    cand = _cand([(ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 6)])
    shares = _shares(**{"47065000100|47065000200": _profile(b5=0.25, b6=0.25, b12=0.5)})
    anchored, stats = apply_ctpp_anchor(w_dict, cand, shares)

    assert set(anchored[ORIGIN]) == {(5, DEST_A), (6, DEST_A)}
    assert anchored[ORIGIN][(5, DEST_A)] == pytest.approx(0.5)
    assert anchored[ORIGIN][(6, DEST_A)] == pytest.approx(0.5)
    assert stats["mass_retained"] == pytest.approx(0.5)
    assert stats["n_cells_anchored"] == 2


def test_unpublished_tract_pair_is_left_untouched():
    w_dict = {ORIGIN: {(5, DEST_A): 0.4, (9, DEST_A): 0.6}}
    cand = _cand([(ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 9)])
    anchored, stats = apply_ctpp_anchor(w_dict, cand, {})

    assert anchored[ORIGIN] == w_dict[ORIGIN]
    assert stats["n_flows_anchored"] == 0
    assert stats["n_flows_unpublished"] == 1


def test_published_pair_with_zero_available_mass_falls_back():
    """CTPP puts every traveller in a block the ILP has no variable for."""
    w_dict = {ORIGIN: {(5, DEST_A): 0.4, (9, DEST_A): 0.6}}
    cand = _cand([(ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 9)])
    shares = _shares(**{"47065000100|47065000200": _profile(b12=1.0)})
    anchored, stats = apply_ctpp_anchor(w_dict, cand, shares)

    assert anchored[ORIGIN] == w_dict[ORIGIN]
    assert stats["n_flows_zero_available_mass"] == 1
    assert stats["n_flows_anchored"] == 0


def test_destination_absent_from_the_candidate_frame_falls_back():
    w_dict = {ORIGIN: {(5, DEST_A): 1.0}}
    shares = _shares(**{"47065000100|47065000200": _profile(b5=0.5, b6=0.5)})
    anchored, stats = apply_ctpp_anchor(w_dict, _cand([(ORIGIN, DEST_B, 5)]), shares)

    assert anchored[ORIGIN] == w_dict[ORIGIN]
    assert stats["n_flows_no_candidates"] == 1


def test_anchor_keys_stay_plain_ints_so_the_ilp_lookup_resolves():
    """The ILP looks cells up with the candidate frame's int16 bin ids."""
    w_dict = {ORIGIN: {(np.int64(5), DEST_A): 1.0}}
    cand = _cand([(ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 6)])
    cand["departure_time_bin"] = cand["departure_time_bin"].astype(np.int16)
    shares = _shares(**{"47065000100|47065000200": _profile(b5=0.5, b6=0.5)})
    anchored, _ = apply_ctpp_anchor(w_dict, cand, shares)

    assert all(type(s) is int for s, _ in anchored[ORIGIN])
    assert anchored[ORIGIN].get((np.int16(6), DEST_A)) == pytest.approx(0.5)


def test_leading_zero_block_group_ids_survive_the_tract_truncation():
    origin, dest = "010010201001", "010010202001"
    w_dict = {origin: {(5, dest): 1.0}}
    cand = _cand([(origin, dest, 5), (origin, dest, 6)])
    shares = _shares(**{"01001020100|01001020200": _profile(b5=0.25, b6=0.75)})
    anchored, stats = apply_ctpp_anchor(w_dict, cand, shares)

    assert stats["n_flows_anchored"] == 1
    assert anchored[origin][(6, dest)] == pytest.approx(0.75)


def test_anchor_with_no_published_pairs_reproduces_get_initial_od_dist():
    """Regression guard for the flag-off path: an empty share table is a no-op."""
    od_df = pd.DataFrame({
        "h_geocode": [ORIGIN] * 4,
        "w_geocode": [DEST_A, DEST_A, DEST_B, DEST_B],
        "departure_time": ["2025-03-17 07:10:00", "2025-03-17 08:40:00",
                           "2025-03-17 07:15:00", "2025-03-17 16:30:00"],
    })
    baseline = get_initial_od_dist(od_df.copy())
    cand = _cand([(ORIGIN, DEST_A, 5), (ORIGIN, DEST_A, 8), (ORIGIN, DEST_B, 5), (ORIGIN, DEST_B, 13)])
    anchored, stats = apply_ctpp_anchor(baseline, cand, {})

    assert anchored == baseline
    assert stats["n_flows_anchored"] == 0


# --------------------------------------------------------------------------- split-half


def test_select_anchor_pairs_is_a_seeded_deterministic_half():
    pairs = [(f"4706500{i:04d}", "47065000200") for i in range(100)]
    first = select_anchor_pairs(pairs, seed=7)
    assert first == select_anchor_pairs(list(reversed(pairs)), seed=7)
    assert len(first) == 50
    assert first != select_anchor_pairs(pairs, seed=8)
    assert first <= set(pairs)
