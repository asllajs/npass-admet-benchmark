"""Section 3.5 - the failure tracks distance from the training domain.

The positive control needs ADMET-AI predictions on the Wang benchmark, which are
produced by an optional step; if they have not been generated the corresponding
tests skip and say so.
"""
from __future__ import annotations

import pytest

TOOLS = ["ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0"]


def test_natural_products_lie_outside_the_training_domain(domain_gap, expected):
    got = domain_gap.set_index("group")["median_nn_tanimoto"]
    want = expected["applicability_domain"]["median_nn_tanimoto"]
    assert float(got["natural products"]) == pytest.approx(want["natural products"], abs=0.01)
    assert float(got["drug-like held out"]) == pytest.approx(want["drug-like held out"], abs=0.01)
    assert got["natural products"] < got["drug-like held out"]


def test_fraction_below_the_applicability_domain_cutoff(domain_gap, expected):
    got = domain_gap.set_index("group")["frac_below_cutoff"]
    want = expected["applicability_domain"]["frac_below_cutoff"]
    for group, value in want.items():
        assert float(got[group]) == pytest.approx(value, abs=0.01)


def test_domain_gap_is_significant(domain_gap, expected):
    p = float(domain_gap["mannwhitney_p"].iloc[0])
    assert p < 1e-10
    assert p == pytest.approx(expected["applicability_domain"]["mannwhitney_p"], rel=0.2)


@pytest.mark.parametrize("tool", TOOLS)
def test_distance_does_not_triage_individual_compounds(tool, error_vs_distance, expected):
    """Once everything is out of domain, distance stops separating good from bad."""
    row = error_vs_distance.set_index("tool").loc[tool]
    want = expected["applicability_domain"]["spearman_abserr_vs_nn"][tool]
    assert float(row["spearman_abserr_vs_nn"]) == pytest.approx(want, abs=0.02)
    assert float(row["spearman_p"]) > 0.05


@pytest.mark.parametrize("tool", TOOLS)
def test_error_grows_with_molecular_weight(tool, mw_confound, expected):
    row = mw_confound.set_index("tool").loc[tool]
    want = expected["mw_confound"][tool]
    assert float(row["spearman"]) == pytest.approx(want["spearman"], abs=0.02)
    assert float(row["spearman_p"]) < want["p_max"]


def test_measured_permeability_also_declines_with_molecular_weight(mw_confound, expected):
    """The trend the predictions failed to capture; it is not a rescaled dosing artefact."""
    row = mw_confound[mw_confound["quantity"] == "measured log Papp vs MW"].iloc[0]
    want = expected["mw_confound"]["measured log Papp vs MW"]
    assert float(row["spearman"]) == pytest.approx(want["spearman"], abs=0.02)
    assert float(row["spearman_p"]) < want["p_max"]


def test_in_domain_positive_control(wang_control, expected):
    """The same tool, the same endpoint, inside the chemistry it was trained on."""
    got = wang_control.set_index("subset")
    want = expected["wang_positive_control"]
    for subset in ("test", "all"):
        assert int(got.loc[subset, "n"]) == want[subset]["n"]
        assert float(got.loc[subset, "pearson_r"]) == pytest.approx(want[subset]["pearson_r"], abs=0.01)
        assert float(got.loc[subset, "r2"]) == pytest.approx(want[subset]["r2"], abs=0.01)


def test_the_endpoint_is_learnable_in_domain(wang_control, caco2_table):
    """Drug-like r ~ 0.9 against natural-product r ~ 0.1 for the same endpoint."""
    in_domain = float(wang_control.set_index("subset").loc["test", "pearson_r"])
    out_of_domain = float(caco2_table.set_index("tool").loc["ADMET-AI", "pearson_r"])
    assert in_domain > 0.85
    assert out_of_domain < 0.15
