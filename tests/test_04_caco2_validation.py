"""Section 3.2, Table 3 - none of the three tools predicts the measured permeability.

This is the study's central claim, so it is checked twice: once as the exact
published numbers, and once as the qualitative statement those numbers support,
which would survive a small change in the data but not a change in conclusion.
"""
from __future__ import annotations

import pytest

from npadmet import datasets as D
from tests.conftest import compute

TOOLS = ["ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0"]
TOL = {"pearson_r": 0.01, "pearson_p": 0.02, "spearman": 0.01, "rmse": 0.01,
       "mae": 0.01, "r2": 0.05, "ccc": 0.01, "ba_bias": 0.01, "ba_loa": 0.02,
       "auroc": 0.01, "mcc": 0.01}
BOOTSTRAP_TOL = 0.04   # percentile-bootstrap bounds are themselves random


def test_experimental_dataset_size(expected):
    exp = compute(D.caco2_experimental)
    want = expected["experimental_dataset"]
    assert len(exp) == want["n_compounds"]
    assert int((exp["papp_ab_cm_s"] > 0).sum()) == want["n_with_papp"]
    assert int(exp["efflux_ratio"].notna().sum()) == want["n_with_efflux_ratio"]


@pytest.mark.parametrize("tool", TOOLS)
def test_compounds_per_tool(tool, caco2_table, expected):
    got = int(caco2_table.set_index("tool").loc[tool, "n"])
    assert got == expected["caco2_validation"][tool]["n"]


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("metric", list(TOL))
def test_published_metric(tool, metric, caco2_table, expected):
    got = float(caco2_table.set_index("tool").loc[tool, metric])
    want = expected["caco2_validation"][tool][metric]
    assert got == pytest.approx(want, abs=TOL[metric])


# --- the same claims, stated qualitatively ---------------------------------

def test_no_tool_correlates_with_experiment(caco2_table):
    assert caco2_table["pearson_r"].abs().max() < 0.15
    assert caco2_table["pearson_p"].min() > 0.05


def test_confidence_intervals_exclude_a_useful_correlation(caco2_table, expected):
    """A weak correlation cannot be excluded, but a useful one can."""
    assert caco2_table["pearson_ci_hi"].max() < expected["caco2_validation"]["pearson_ci_upper_max"]
    assert (caco2_table["pearson_ci_lo"] < 0).all()


@pytest.mark.parametrize("tool", TOOLS)
def test_published_confidence_interval(tool, caco2_table, expected):
    """Bootstrap bounds move by up to ~0.02 between runs, hence the wider tolerance."""
    row = caco2_table.set_index("tool").loc[tool]
    lo, hi = expected["caco2_validation"][tool]["pearson_ci"]
    assert float(row["pearson_ci_lo"]) == pytest.approx(lo, abs=BOOTSTRAP_TOL)
    assert float(row["pearson_ci_hi"]) == pytest.approx(hi, abs=BOOTSTRAP_TOL)


def test_r2_against_the_identity_line_is_negative(caco2_table):
    assert (caco2_table["r2"] < 0).all()


def test_binary_permeability_call_is_near_chance(caco2_table):
    assert caco2_table["auroc"].max() < 0.65
    assert caco2_table["mcc"].abs().max() < 0.2


def test_individual_predictions_are_uncertain_by_orders_of_magnitude(caco2_table):
    """Limits of agreement of +/-1.6 to +/-1.9 log units."""
    assert (caco2_table["ba_loa"] > 1.5).all()
    assert caco2_table["ba_bias"].abs().max() < 0.2


def test_predictions_are_not_simply_offset(caco2_table):
    """Bias is small while the spread is large: this is scatter, not a calibration shift."""
    assert (caco2_table["ba_loa"] > 5 * caco2_table["ba_bias"].abs()).all()
