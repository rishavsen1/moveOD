"""Unit tests for the held-out external validation script."""

import numpy as np
import pandas as pd
import pytest

from analysis.validate_external import (
    CALIB_STEM_RE,
    DEP_BIN_MIDPOINTS,
    TLFD_BIN_EDGES,
    TT_BIN_LABELS,
    bin_departure_minutes,
    bin_travel_minutes,
    expand_cells_to_trips,
    fit_ipf,
    js_distance,
    largest_remainder_round,
    normalize_geoid,
    run_face_validity_test,
    to_trip_frame,
    total_variation_distance,
    truncate_geoid,
    wasserstein1,
)


# --------------------------------------------------------------------------- binning


def test_departure_bin_edges():
    minutes = pd.Series([0, 299, 300, 329, 330, 959, 960, 1439])
    bins = bin_departure_minutes(minutes)
    assert list(bins) == [0, 0, 1, 1, 2, 12, 13, 13]


def test_departure_bin_out_of_range_is_nan():
    bins = bin_departure_minutes(pd.Series([-1.0, 1440.0]))
    assert bins.isna().all()


def test_travel_time_bin_edges():
    minutes = pd.Series([0.0, 4.99, 5.0, 9.99, 59.9, 60.0, 89.9, 90.0, 500.0])
    labels = bin_travel_minutes(minutes)
    assert list(labels.astype(str)) == [
        "<5",
        "<5",
        "5-9",
        "5-9",
        "45-59",
        "60-89",
        "60-89",
        "90+",
        "90+",
    ]
    assert TT_BIN_LABELS[0] == "<5" and TT_BIN_LABELS[-1] == "90+"


# --------------------------------------------------------------------------- geoids


def test_normalize_geoid_restores_leading_zeros():
    # Iowa (19) keeps 12 digits; Alabama (01) and Colorado (08) lose the leading
    # zero when the CSV is read back as int64.
    raw = pd.Series([190039501001, 10730144001, 80310080111])
    norm = normalize_geoid(raw)
    assert list(norm) == ["190039501001", "010730144001", "080310080111"]
    assert (norm.str.len() == 12).all()


def test_normalize_geoid_handles_strings_and_floats():
    raw = pd.Series(["470650004001", 470650004001.0, " 470650004001 "])
    assert list(normalize_geoid(raw)) == ["470650004001"] * 3


def test_truncate_geoid_levels():
    geoids = pd.Series(["010730144001"])
    assert list(truncate_geoid(geoids, "bg")) == ["010730144001"]
    assert list(truncate_geoid(geoids, "tract")) == ["01073014400"]
    assert list(truncate_geoid(geoids, "county")) == ["01073"]
    with pytest.raises(ValueError):
        truncate_geoid(geoids, "state")


# --------------------------------------------------------------------------- metrics


def test_metrics_identical_distributions_are_zero():
    p = np.array([10.0, 20.0, 30.0])
    q = np.array([1.0, 2.0, 3.0])  # same shape, different scale
    mids = np.array([1.0, 2.0, 3.0])
    assert total_variation_distance(p, q) == pytest.approx(0.0)
    assert js_distance(p, q) == pytest.approx(0.0, abs=1e-12)
    assert wasserstein1(p, q, mids) == pytest.approx(0.0)


def test_metrics_disjoint_distributions():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    mids = np.array([0.0, 10.0])
    assert total_variation_distance(p, q) == pytest.approx(1.0)
    assert js_distance(p, q) == pytest.approx(1.0)  # base-2 JS distance maxes out at 1
    assert wasserstein1(p, q, mids) == pytest.approx(10.0)


def test_tvd_half_overlap():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.5, 0.5])
    assert total_variation_distance(p, q) == pytest.approx(0.5)


def test_departure_midpoints_match_bin_count():
    assert len(DEP_BIN_MIDPOINTS) == 14


# --------------------------------------------------------------------------- IPF


def test_fit_ipf_matches_both_marginals_on_toy_table():
    rng = np.random.default_rng(0)
    seed = rng.random((3, 3, 2)) + 0.01
    od = np.array([[5.0, 3.0, 2.0], [1.0, 6.0, 3.0], [4.0, 4.0, 2.0]])
    # (o, s) marginal must share the same origin totals as the (o, d) marginal.
    os_share = np.array([[0.4, 0.6], [0.25, 0.75], [0.5, 0.5]])
    os_target = od.sum(axis=1)[:, None] * os_share

    table, iterations, error = fit_ipf(seed, od, os_target, max_iter=200, tol=1e-9)

    assert error < 1e-9
    assert iterations <= 200
    np.testing.assert_allclose(table.sum(axis=2), od, atol=1e-8)
    np.testing.assert_allclose(table.sum(axis=1), os_target, atol=1e-8)


def test_fit_ipf_tolerates_zero_marginal_cells():
    seed = np.ones((2, 2, 2))
    od = np.array([[4.0, 0.0], [0.0, 6.0]])
    os_target = np.array([[1.0, 3.0], [2.0, 4.0]])
    table, _, error = fit_ipf(seed, od, os_target, max_iter=200, tol=1e-9)
    assert error < 1e-9
    assert table[0, 1, :].sum() == pytest.approx(0.0)
    assert table.sum() == pytest.approx(10.0)


# --------------------------------------------------------------------------- expansion


def test_largest_remainder_round_preserves_total():
    weights = np.array([1.4, 1.4, 1.4, 0.8])
    counts = largest_remainder_round(weights, 5)
    assert counts.sum() == 5
    assert counts.dtype == np.int64
    assert list(counts) == [2, 1, 1, 1]


def test_largest_remainder_round_zero_total():
    assert largest_remainder_round(np.array([1.0, 2.0]), 0).sum() == 0


def test_expand_cells_to_trips_preserves_totals():
    origins = ["010730144001", "010730144002"]
    dests = ["010730144001", "010730144002"]
    table = np.zeros((2, 2, 14))
    table[0, 0, 0] = 3.0
    table[0, 1, 5] = 2.0
    table[1, 1, 13] = 4.0
    pools = {
        ("010730144001", "010730144001"): np.array([3.0]),
        ("010730144001", "010730144002"): np.array([11.0]),
        ("010730144002", "010730144002"): np.array([22.0]),
    }
    trips = expand_cells_to_trips(table, origins, dests, pools, np.random.default_rng(42))

    assert len(trips) == 9
    counts = trips.groupby(["origin_bg", "dest_bg"]).size()
    assert counts[("010730144001", "010730144001")] == 3
    assert counts[("010730144001", "010730144002")] == 2
    assert counts[("010730144002", "010730144002")] == 4
    # departure minutes must land inside their bin, travel times inside the pool
    assert list(bin_departure_minutes(trips["dep_min"]).value_counts().sort_index().index) == [0, 5, 13]
    assert set(trips["tt_min"]) <= {3.0, 11.0, 22.0}


def test_expand_cells_to_trips_is_deterministic():
    origins = dests = ["190039501001", "190039501002"]
    table = np.zeros((2, 2, 14))
    table[0, 1, 7] = 25.0
    pools = {"190039501001": np.arange(1.0, 30.0)}
    a = expand_cells_to_trips(table, origins, dests, pools, np.random.default_rng(42))
    b = expand_cells_to_trips(table, origins, dests, pools, np.random.default_rng(42))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 25


# --------------------------------------------------------------------------- time columns


def _one_trip(departure: object, extra: dict | None = None) -> pd.DataFrame:
    row = {"origin_geoid": [470650004001], "destination_geoid": [470650012003],
           "departure_time": [departure], "travel_time_min": [12.0]}
    row.update(extra or {})
    return pd.DataFrame(row)


def test_departure_time_seconds_and_minutes_bin_identically():
    # 19200 s and 320 min are both 05:20; the unit is inferred from the range.
    from_seconds = to_trip_frame(_one_trip(19200))
    from_minutes = to_trip_frame(_one_trip(320))
    assert from_seconds["dep_min"].iloc[0] == pytest.approx(320.0)
    assert from_minutes["dep_min"].iloc[0] == pytest.approx(320.0)
    assert bin_departure_minutes(from_seconds["dep_min"]).iloc[0] == 1
    assert bin_departure_minutes(from_minutes["dep_min"]).iloc[0] == 1


def test_departure_datetime_wins_over_an_ambiguous_numeric_column():
    # Davidson stores minutes in departure_time; reading it as seconds would put
    # a 16:00 departure at 00:16, so the timestamp column has to take precedence.
    frame = _one_trip(960, {"departure_datetime": ["2025-03-10 16:00:00"]})
    trips = to_trip_frame(frame)
    assert trips["dep_min"].iloc[0] == pytest.approx(960.0)
    assert bin_departure_minutes(trips["dep_min"]).iloc[0] == 13


def test_arrival_is_always_derived_even_when_the_table_stores_one():
    frame = _one_trip("2025-03-10 05:20:00", {"arrival_time": ["2025-03-10 23:00:00"]})
    trips = to_trip_frame(frame)
    assert trips["arr_min"].iloc[0] == pytest.approx(332.0)


def test_calibrated_stem_filter_accepts_days_and_rejects_helpers():
    accepted = ["2025-03-10", "2025-03-10_1", "Polk_2025-03-10", "Santa_Fe_2025-03-10"]
    rejected = ["df_sample_200", "df_sample_1000", "geoids", "notes"]
    assert all(CALIB_STEM_RE.match(stem) for stem in accepted)
    assert not any(CALIB_STEM_RE.match(stem) for stem in rejected)
    assert CALIB_STEM_RE.match("Polk_2025-03-10").group(1) == "2025-03-10"


# --------------------------------------------------------------------------- face validity


def _metric(rows: list[dict], name: str) -> float:
    """Pull the single row with this metric name out of `run_face_validity_test`'s output."""
    matches = [row["value"] for row in rows if row["metric"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} row, got {len(matches)}"
    return matches[0]


def _face_validity_frame(**overrides) -> pd.DataFrame:
    base = {
        "origin_geoid": ["470650004001"],
        "destination_geoid": ["470650004001"],
        "travel_time_min_filled": [10.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_face_validity_rows_are_tagged_calibrated_county():
    rows = run_face_validity_test(_face_validity_frame())
    assert rows  # at least one metric was produced
    assert all(row["method"] == "calibrated" for row in rows)
    assert all(row["test"] == "face_validity" for row in rows)
    assert all(row["geography"] == "county" for row in rows)


def test_implied_speed_excludes_zero_travel_time_and_flags_share_thresholds():
    frame = _face_validity_frame(
        origin_geoid=[1, 1, 1, 1, 1],
        destination_geoid=[2, 2, 2, 2, 2],
        travel_time_min_filled=[0.0, 10.0, 5.0, 2.0, 30.0],
        travel_distance_mi=[5.0, 2.0, 5.0, 5.0, 1.0],
    )
    rows = run_face_validity_test(frame)

    # distance is present on all 5 rows, but the zero-travel-time row is excluded
    # from every implied-speed metric (12, 60, 150 and 2 mph -- 4 values, not 5).
    assert _metric(rows, "n_trips_with_distance") == pytest.approx(5.0)
    assert _metric(rows, "implied_speed_mph_p50") == pytest.approx(np.percentile([12.0, 60.0, 150.0, 2.0], 50))
    assert _metric(rows, "share_over_80mph") == pytest.approx(0.25)  # only the 150 mph trip
    assert _metric(rows, "share_under_3mph") == pytest.approx(0.25)  # only the 2 mph trip


def test_intra_bg_and_intra_tract_share_on_four_trips():
    geoid_a = "470650004001"  # tract 47065000400, block group 1
    geoid_b = "470650004002"  # same tract, block group 2
    geoid_c = "470650009001"  # different tract
    frame = _face_validity_frame(
        origin_geoid=[geoid_a, geoid_a, geoid_a, geoid_b],
        destination_geoid=[geoid_a, geoid_b, geoid_c, geoid_b],
        travel_time_min_filled=[10.0, 10.0, 10.0, 10.0],
    )
    rows = run_face_validity_test(frame)

    # (a,a) and (b,b) stay in one BG; (a,a), (a,b) and (b,b) stay in one tract.
    assert _metric(rows, "share_intra_bg") == pytest.approx(0.5)
    assert _metric(rows, "share_intra_tract") == pytest.approx(0.75)


def test_commuters_per_point_aggregates_duplicated_coordinates():
    n_home_a, n_home_b, n_work = 12, 3, 15
    assert n_home_a + n_home_b == n_work
    frame = _face_validity_frame(
        origin_geoid=[1] * n_work,
        destination_geoid=[2] * n_work,
        travel_time_min_filled=[10.0] * n_work,
        origin_lat=[40.0] * n_home_a + [40.1] * n_home_b,
        origin_lon=[-85.0] * n_home_a + [-85.1] * n_home_b,
        destination_lat=[41.0] * n_work,
        destination_lon=[-86.0] * n_work,
    )
    rows = run_face_validity_test(frame)

    assert _metric(rows, "home_pts_n") == pytest.approx(2.0)
    assert _metric(rows, "commuters_per_home_pt_max") == pytest.approx(12.0)
    assert _metric(rows, "share_home_pts_over_10") == pytest.approx(0.5)  # only the 12-trip point
    assert _metric(rows, "work_pts_n") == pytest.approx(1.0)
    assert _metric(rows, "commuters_per_work_pt_max") == pytest.approx(15.0)
    assert not any(row["metric"] == "share_work_pts_over_10" for row in rows)


def test_tlfd_shares_sum_to_one():
    # One trip per bin, including the open-ended 50+ bin.
    distances = [0.5, 1.5, 2.5, 4.0, 6.0, 8.0, 12.0, 17.0, 25.0, 40.0, 60.0]
    assert len(distances) == len(TLFD_BIN_EDGES) - 1
    frame = _face_validity_frame(
        origin_geoid=list(range(len(distances))),
        destination_geoid=list(range(len(distances))),
        travel_time_min_filled=[10.0] * len(distances),
        travel_distance_mi=distances,
    )
    rows = run_face_validity_test(frame)

    tlfd_rows = [row for row in rows if row["metric"].startswith("tlfd_share_")]
    assert len(tlfd_rows) == len(distances)
    assert sum(row["value"] for row in tlfd_rows) == pytest.approx(1.0)
    # each bin got exactly one of the eleven trips
    assert all(row["value"] == pytest.approx(1.0 / len(distances)) for row in tlfd_rows)
