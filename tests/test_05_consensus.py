"""Section 3.3, Table 4 - averaging the three tools does not recover accuracy."""
from __future__ import annotations

import pytest

TOL = 0.015


def test_complete_case_size(consensus_table, expected):
    assert int(consensus_table["n"].iloc[0]) == expected["consensus"]["n_complete_case"]
    assert consensus_table["n"].nunique() == 1, "all predictors must be scored on the same compounds"


def test_tools_agree_with_each_other(inter_tool_correlation, expected):
    want = expected["consensus"]["inter_tool_r"]
    for row in inter_tool_correlation.itertuples():
        key = f"{row.tool_a} vs {row.tool_b}"
        assert float(row.pearson_r) == pytest.approx(want[key], abs=TOL)
    assert float(inter_tool_correlation["pearson_r"].mean()) == \
        pytest.approx(expected["consensus"]["mean_inter_tool_r"], abs=TOL)


@pytest.mark.parametrize("predictor", [
    "ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0", "Consensus (mean)", "Consensus (median)"])
def test_correlation_with_experiment(predictor, consensus_table, expected):
    got = float(consensus_table.set_index("predictor").loc[predictor, "pearson_r"])
    assert got == pytest.approx(expected["consensus"]["vs_measured_r"][predictor], abs=TOL)


@pytest.mark.parametrize("predictor", [
    "ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0", "Consensus (mean)", "Consensus (median)"])
def test_published_confidence_interval(predictor, consensus_table, expected):
    """Bootstrap bounds move by up to ~0.02 between runs, hence the wider tolerance."""
    row = consensus_table.set_index("predictor").loc[predictor]
    lo, hi = expected["consensus"]["pearson_ci"][predictor]
    assert float(row["pearson_ci_lo"]) == pytest.approx(lo, abs=0.04)
    assert float(row["pearson_ci_hi"]) == pytest.approx(hi, abs=0.04)


def test_consensus_does_not_beat_the_best_single_tool(consensus_table):
    t = consensus_table.set_index("predictor")["pearson_r"]
    best_single = t[["ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0"]].max()
    assert t["Consensus (mean)"] <= best_single + 0.01


def test_agreement_and_accuracy_are_decoupled(consensus_table, inter_tool_correlation):
    """The paper's central inference, as a single inequality.

    The tools correlate with one another an order of magnitude more strongly than
    any of them correlates with the measurement, which is what makes consensus a
    measure of shared bias rather than of correctness.
    """
    mutual = float(inter_tool_correlation["pearson_r"].mean())
    versus_truth = float(consensus_table["pearson_r"].abs().max())
    assert mutual > 0.8
    assert versus_truth < 0.15
    assert mutual > 5 * versus_truth


def test_consensus_reduces_error_without_improving_ranking(consensus_table):
    """RMSE drops through variance averaging while the correlation does not move."""
    t = consensus_table.set_index("predictor")
    singles = t.loc[["ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0"]]
    assert t.loc["Consensus (mean)", "rmse"] < singles["rmse"].max()
    assert t.loc["Consensus (mean)", "r2"] < 0
