"""Section 3.4 / Table 5: the null result is not an artefact of uniform-mass dosing.

The assay dosed 100 ug/mL rather than a fixed molar concentration, so heavier
compounds were tested at a lower molar concentration. If that were producing the
null result, the correlation would come back among the lighter compounds. These
tests assert that it does not, at three resolutions.
"""
from __future__ import annotations

import pytest

LIGHT = "MW < 500 Da"
HEAVY = "MW >= 500 Da"


def test_strata_sizes(mw_stratified, expected):
    """112 of the 134 benchmark compounds are below 500 Da, 22 above."""
    exp = expected["mw_stratified"]
    assert int(mw_stratified[mw_stratified["stratum"] == LIGHT]["n"].max()) == exp["n_light"]
    assert int(mw_stratified[mw_stratified["stratum"] == HEAVY]["n"].max()) == exp["n_heavy"]


def test_no_correlation_in_the_light_stratum(mw_stratified, expected):
    """Below 500 Da the correlation is no better than on the full set."""
    exp = expected["mw_stratified"]["pearson_r_light"]
    got = mw_stratified[mw_stratified["stratum"] == LIGHT].set_index("tool")
    for tool, value in exp.items():
        assert got.loc[tool, "pearson_r"] == pytest.approx(value, abs=0.01)
    assert got["pearson_r"].abs().max() < 0.15
    assert got["pearson_p"].min() > 0.05


def test_heavy_stratum_is_not_positive_either(mw_stratified, expected):
    """Above 500 Da the correlations are negative, not merely weaker."""
    exp = expected["mw_stratified"]["pearson_r_heavy"]
    got = mw_stratified[mw_stratified["stratum"] == HEAVY].set_index("tool")
    for tool, value in exp.items():
        assert got.loc[tool, "pearson_r"] == pytest.approx(value, abs=0.01)
    assert (got["pearson_r"] < 0).all()


def test_no_quartile_shows_a_signal(mw_quartiles, expected):
    """Twelve tool-by-quartile combinations, none of them significant."""
    exp = expected["mw_stratified"]
    assert mw_quartiles["pearson_r"].abs().max() == pytest.approx(
        exp["max_abs_r_any_quartile"], abs=0.01)
    assert mw_quartiles["pearson_p"].min() == pytest.approx(
        exp["min_p_any_quartile"], abs=0.02)
    assert mw_quartiles["pearson_p"].min() > 0.05


def test_partial_correlation_removes_the_residual_signal(mw_partial, expected):
    """The one marginal rank correlation is a molecular-weight effect.

    This is the sharpest form of the claim: holding molecular weight fixed, the
    tools carry no information about permeability at all.
    """
    exp = expected["mw_stratified"]["partial_spearman"]
    got = mw_partial.set_index("tool")
    for tool, values in exp.items():
        assert got.loc[tool, "spearman_raw"] == pytest.approx(values["raw"], abs=0.01)
        assert got.loc[tool, "spearman_partial_mw"] == pytest.approx(
            values["partial"], abs=0.01)
    # the raw ADMET-AI association is nominally significant; the partial one is not
    assert got.loc["ADMET-AI", "p_raw"] < 0.05
    assert got["p_partial"].min() > 0.5
    assert got["spearman_partial_mw"].abs().max() < 0.05
