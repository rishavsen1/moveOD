"""Unit tests for the CTPP held-out flow validation module. No network access:
`fetch_flows`'s HTTP layer is mocked in the one paging test; everything else
runs on toy in-memory frames."""

import numpy as np
import pandas as pd
import pytest

from analysis import ctpp
from analysis.validate_external import DEP_BIN_LABELS, TT_BIN_LABELS


# --------------------------------------------------------------------------- table shape


def test_table_specs_match_verified_column_counts():
    assert ctpp.TABLE_SPECS["b302100"]["n_cols"] == 1
    assert ctpp.TABLE_SPECS["b302103"]["n_cols"] == 18
    assert ctpp.TABLE_SPECS["b302104"]["n_cols"] == 17
    assert ctpp.TABLE_SPECS["b302106"]["n_cols"] == 15
    for spec in ctpp.TABLE_SPECS.values():
        assert len(spec["labels"]) == spec["n_cols"]


# --------------------------------------------------------------------------- departure reordering


def test_ctpp_departure_order_is_acs_order_rotated():
    """CTPP starts its 14 departure categories at 5:00 a.m.; ACS starts at 12:00 a.m."""
    assert ctpp.CTPP_DEPARTURE_LABELS == DEP_BIN_LABELS[1:] + DEP_BIN_LABELS[:1]
    assert ctpp.CTPP_DEPARTURE_LABELS[0] == "5am_to_5:29am"
    assert ctpp.CTPP_DEPARTURE_LABELS[-1] == "12am_to_4:59am"


def test_decode_b302104_applies_ctpp_departure_reordering():
    """e16 is CTPP's LAST departure column but ACS's FIRST bin ("12am_to_4:59am").

    A naive positional zip of e3..e16 against DEP_BIN_LABELS (i.e. e3 ->
    DEP_BIN_LABELS[0], ..., e16 -> DEP_BIN_LABELS[13]) would instead label
    this column "4pm_to_11:59pm" -- a plausible-looking but wrong answer,
    since both are "late" bins. This test would fail under that naive order.
    """
    row = {"origin_geoid": "C1100US47065000400", "destination_geoid": "C3100US47065000100",
           "b302104_e1": 100, "b302104_e2": 90, "b302104_e17": 10}
    # e3..e16, in CTPP's own column order (5am first, 12am-4:59am last).
    ctpp_order_values = [5, 10, 15, 20, 10, 8, 6, 4, 3, 2, 1, 1, 2, 3]
    assert len(ctpp_order_values) == 14
    for offset, value in enumerate(ctpp_order_values):
        row[f"b302104_e{offset + 3}"] = value

    decoded = ctpp.decode(pd.DataFrame([row]), "b302104")
    by_category = decoded.set_index("category")["workers"]

    assert by_category["5am_to_5:29am"] == 5  # e3: first CTPP column
    assert by_category["4pm_to_11:59pm"] == 2  # e15: second-to-last CTPP column
    assert by_category["12am_to_4:59am"] == 3  # e16: last CTPP column, first ACS bin
    # The naive (unrotated) mapping would have called e16's value "4pm_to_11:59pm",
    # and that bin's real value (from e15) is different, so this is discriminating.
    assert by_category["12am_to_4:59am"] != by_category["4pm_to_11:59pm"]
    assert by_category["total"] == 100
    assert by_category["traveled"] == 90
    assert by_category["worked_at_home"] == 10


# --------------------------------------------------------------------------- decode / geoids


def test_decode_zero_pads_leading_zero_state_tract_ids():
    df = pd.DataFrame([{"origin_geoid": "C1100US01001000100", "destination_geoid": "C3100US19001000200",
                        "b302100_e1": 42}])
    decoded = ctpp.decode(df, "b302100")
    assert decoded.loc[0, "origin_tract"] == "01001000100"
    assert decoded.loc[0, "destination_tract"] == "19001000200"
    assert isinstance(decoded.loc[0, "origin_tract"], str)
    assert len(decoded.loc[0, "origin_tract"]) == 11
    assert len(decoded.loc[0, "destination_tract"]) == 11


def test_decode_raises_on_missing_columns():
    df = pd.DataFrame([{"origin_geoid": "C1100US47065000400", "destination_geoid": "C3100US47065000100"}])
    with pytest.raises(KeyError):
        ctpp.decode(df, "b302100")


# --------------------------------------------------------------------------- CPC / SSI


def test_cpc_ssi_identical_flows_are_one():
    a = pd.DataFrame({"origin_tract": ["A", "A"], "destination_tract": ["B", "C"], "workers": [10.0, 20.0]})
    b = a.copy()
    result = ctpp.cpc_ssi(a, b)
    assert result["cpc"] == pytest.approx(1.0)
    assert result["ssi"] == pytest.approx(1.0)


def test_cpc_ssi_disjoint_flows_are_zero():
    a = pd.DataFrame({"origin_tract": ["A"], "destination_tract": ["B"], "workers": [10.0]})
    b = pd.DataFrame({"origin_tract": ["C"], "destination_tract": ["D"], "workers": [10.0]})
    result = ctpp.cpc_ssi(a, b)
    assert result["cpc"] == pytest.approx(0.0)
    assert result["ssi"] == pytest.approx(0.0)


def test_cpc_uses_reference_as_denominator_asymmetrically():
    """CPC = sum(min(a,b)) / sum(b): halving the reference should double CPC (until it caps)."""
    compared = pd.DataFrame({"origin_tract": ["A"], "destination_tract": ["B"], "workers": [10.0]})
    reference_full = pd.DataFrame({"origin_tract": ["A"], "destination_tract": ["B"], "workers": [20.0]})
    reference_half = pd.DataFrame({"origin_tract": ["A"], "destination_tract": ["B"], "workers": [10.0]})
    cpc_full = ctpp.cpc_ssi(compared, reference_full)["cpc"]
    cpc_half = ctpp.cpc_ssi(compared, reference_half)["cpc"]
    assert cpc_full == pytest.approx(0.5)
    assert cpc_half == pytest.approx(1.0)


# --------------------------------------------------------------------------- intra-county filter


def test_filter_intra_county_drops_cross_county_flows():
    flows = pd.DataFrame({
        "origin_tract": ["47065000100", "47065000100"],
        "destination_tract": ["47065000200", "13295020102"],
        "workers": [50.0, 30.0],
    })
    kept, stats = ctpp.filter_intra_county(flows, "47065")
    assert len(kept) == 1
    assert kept.iloc[0]["destination_tract"] == "47065000200"
    assert stats["n_kept"] == 1
    assert stats["workers_kept"] == pytest.approx(50.0)
    assert stats["n_discarded"] == 1
    assert stats["workers_discarded"] == pytest.approx(30.0)


# --------------------------------------------------------------------------- per-flow joint TVD


def test_per_flow_tvd_excludes_flows_below_worker_threshold():
    idx = pd.MultiIndex.from_tuples([("O1", "D1"), ("O2", "D2")], names=["origin_tract", "destination_tract"])
    syn = pd.DataFrame({"a": [30, 5], "b": [30, 5]}, index=idx)
    ctpp_bins = pd.DataFrame({"a": [40, 3], "b": [20, 2]}, index=idx)
    flow_workers = pd.Series([60.0, 5.0], index=idx)

    result = ctpp.per_flow_tvd(syn, ctpp_bins, flow_workers, min_workers=50.0)

    assert len(result) == 1
    assert result.iloc[0]["origin_tract"] == "O1"
    assert result.iloc[0]["destination_tract"] == "D1"
    assert result.iloc[0]["workers"] == pytest.approx(60.0)


def test_per_flow_tvd_computes_correct_distance():
    idx = pd.MultiIndex.from_tuples([("O1", "D1")], names=["origin_tract", "destination_tract"])
    syn = pd.DataFrame({"a": [10], "b": [0]}, index=idx)
    ctpp_bins = pd.DataFrame({"a": [0], "b": [10]}, index=idx)
    flow_workers = pd.Series([100.0], index=idx)

    result = ctpp.per_flow_tvd(syn, ctpp_bins, flow_workers, min_workers=50.0)

    assert result.iloc[0]["tvd"] == pytest.approx(1.0)  # fully disjoint distributions


# --------------------------------------------------------------------------- effective bins


def test_effective_bins_single_bin_is_one():
    counts = np.array([[10.0, 0.0, 0.0, 0.0]])
    assert ctpp._effective_bins(counts)[0] == pytest.approx(1.0)


def test_effective_bins_uniform_equals_bin_count():
    counts = np.array([[5.0, 5.0, 5.0, 5.0]])
    assert ctpp._effective_bins(counts)[0] == pytest.approx(4.0)


def test_effective_bins_all_zero_row_is_nan():
    counts = np.array([[0.0, 0.0]])
    assert np.isnan(ctpp._effective_bins(counts)[0])


# --------------------------------------------------------------------------- Monte Carlo noise floor


def test_joint_floor_metrics_is_reproducible_with_same_seed():
    idx = pd.MultiIndex.from_tuples([("O1", "D1"), ("O2", "D2")], names=["origin_tract", "destination_tract"])
    syn = pd.DataFrame({"a": [40.0, 20.0], "b": [20.0, 40.0]}, index=idx)
    ctpp_bins = pd.DataFrame({"a": [30.0, 30.0], "b": [30.0, 30.0]}, index=idx)
    flow_workers = pd.Series([60.0, 60.0], index=idx)

    first = ctpp.joint_floor_metrics(syn, ctpp_bins, flow_workers, min_workers=50.0, n_reps=50, seed=7)
    second = ctpp.joint_floor_metrics(syn, ctpp_bins, flow_workers, min_workers=50.0, n_reps=50, seed=7)

    assert first == second


def test_joint_floor_metrics_shrinks_toward_zero_with_large_sample_size():
    """A flow with millions of synthetic trips drawn from CTPP's own distribution
    should reproduce it almost exactly -- the floor should be small, not the
    ~0.5-0.7 range a real (small-sample) per-flow TVD showed in production."""
    idx = pd.MultiIndex.from_tuples([("O1", "D1")], names=["origin_tract", "destination_tract"])
    syn = pd.DataFrame({"a": [2_000_000.0], "b": [1_000_000.0]}, index=idx)
    ctpp_bins = pd.DataFrame({"a": [40.0], "b": [60.0]}, index=idx)
    flow_workers = pd.Series([100.0], index=idx)

    floor = ctpp.joint_floor_metrics(syn, ctpp_bins, flow_workers, min_workers=50.0, n_reps=100, seed=1)

    assert floor["tvd_wmean_floor"] < 0.01


def test_joint_floor_metrics_no_qualifying_flows_is_nan():
    idx = pd.MultiIndex.from_tuples([("O1", "D1")], names=["origin_tract", "destination_tract"])
    syn = pd.DataFrame({"a": [0.0]}, index=idx)
    ctpp_bins = pd.DataFrame({"a": [10.0]}, index=idx)
    flow_workers = pd.Series([5.0], index=idx)  # below min_workers

    floor = ctpp.joint_floor_metrics(syn, ctpp_bins, flow_workers, min_workers=50.0)

    assert np.isnan(floor["tvd_wmean_floor"])


# --------------------------------------------------------------------------- CPC coverage/magnitude decomposition


def test_flow_coverage_decomposition_overlap_and_rescaled_cpc():
    # Flow A: both sides have it, disagree on size. Flow B: CTPP-only. Flow C: compared-only.
    compared = pd.DataFrame({"origin_tract": ["A", "C"], "destination_tract": ["A", "C"], "workers": [10.0, 5.0]})
    reference = pd.DataFrame({"origin_tract": ["A", "B"], "destination_tract": ["A", "B"], "workers": [20.0, 10.0]})

    result = ctpp.flow_coverage_decomposition(compared, reference)

    assert result["n_ctpp_flows_total"] == pytest.approx(2.0)
    assert result["n_ctpp_flows_covered"] == pytest.approx(1.0)
    # Overlap-only CPC: min(10, 20) / 20 (reference total restricted to the overlapping flow A).
    assert result["cpc_overlap_only"] == pytest.approx(10.0 / 20.0)
    # Structural ceiling: 1 - (CTPP workers on flows compared lacks) / (CTPP workers) = 1 - 10/30.
    assert result["structural_ceiling_cpc"] == pytest.approx(1.0 - 10.0 / 30.0)


def test_flow_coverage_decomposition_identical_flows_has_ceiling_one():
    flows = pd.DataFrame({"origin_tract": ["A"], "destination_tract": ["A"], "workers": [10.0]})
    result = ctpp.flow_coverage_decomposition(flows, flows.copy())
    assert result["structural_ceiling_cpc"] == pytest.approx(1.0)
    assert result["cpc_overlap_only"] == pytest.approx(1.0)
    assert result["cpc_rescaled_to_ctpp_total"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- misc helpers


def test_safe_ratio_handles_zero_and_nan_denominator():
    assert ctpp._safe_ratio(1.0, 2.0) == pytest.approx(0.5)
    assert np.isnan(ctpp._safe_ratio(1.0, 0.0))
    assert np.isnan(ctpp._safe_ratio(1.0, float("nan")))
    assert np.isnan(ctpp._safe_ratio(float("nan"), 2.0))


# --------------------------------------------------------------------------- table-drift guard


def test_check_table_against_acs_raises_over_threshold():
    ctpp_totals = pd.Series([100.0, 0.0])
    acs_totals = pd.Series([0.0, 100.0])
    with pytest.raises(ctpp.CTPPTableDriftError):
        ctpp.check_table_against_acs(ctpp_totals, acs_totals, "b302104", strict=True)


def test_check_table_against_acs_warns_when_not_strict(caplog):
    ctpp_totals = pd.Series([100.0, 0.0])
    acs_totals = pd.Series([0.0, 100.0])
    tvd = ctpp.check_table_against_acs(ctpp_totals, acs_totals, "b302104", strict=False)
    assert tvd == pytest.approx(1.0)


def test_check_table_against_acs_passes_within_threshold():
    ctpp_totals = pd.Series([50.0, 50.0])
    acs_totals = pd.Series([51.0, 49.0])
    tvd = ctpp.check_table_against_acs(ctpp_totals, acs_totals, "b302104", strict=True)
    assert tvd < 0.05


# --------------------------------------------------------------------------- fetch paging (mocked)


def test_fetch_flows_pages_until_total_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(ctpp, "_ctpp_api_key", lambda: "dummy-key")
    calls = []

    def fake_post(url, headers=None, params=None, json=None, timeout=None):
        calls.append(params["page"])
        if params["page"] == 1:
            data = [{"origin_geoid": "C1100US47065000100", "destination_geoid": "C3100US47065000200",
                     "b302100_e1": "10"}]
        else:
            data = [{"origin_geoid": "C1100US47065000300", "destination_geoid": "C3100US47065000400",
                     "b302100_e1": "20"}]

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"size": 1, "page": params["page"], "total": 2, "data": data}

        return FakeResponse()

    monkeypatch.setattr(ctpp.requests, "post", fake_post)

    frame = ctpp.fetch_flows("47", "065", "b302100", 1, year=2021, cache_dir=tmp_path)

    assert calls == [1, 2]
    assert len(frame) == 2
    assert sorted(frame["b302100_e1"].tolist()) == [10.0, 20.0]
    assert pd.api.types.is_numeric_dtype(frame["b302100_e1"])

    cache_path = tmp_path / "b302100_47065_tract_2021.parquet"
    assert cache_path.exists()

    # Second call should be served from cache with no further HTTP calls.
    calls.clear()
    cached_frame = ctpp.fetch_flows("47", "065", "b302100", 1, year=2021, cache_dir=tmp_path)
    assert calls == []
    assert len(cached_frame) == 2
