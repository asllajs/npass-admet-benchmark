"""Section 3.9 - biosynthetic class changes agreement, not accuracy."""
from __future__ import annotations

import pytest

ORDER = ["Shikimates and Phenylpropanoids", "Terpenoids", "Alkaloids",
         "Polyketides", "Fatty acids", "Carbohydrates", "Amino acids and Peptides"]


@pytest.mark.slow
@pytest.mark.parametrize("pathway", ORDER)
def test_caco2_agreement_per_pathway(pathway, class_agreement, expected):
    got = float(class_agreement.set_index("pathway").loc[pathway, "caco2"])
    want = expected["class_stratified"]["caco2_agreement_by_pathway"][pathway]
    assert got == pytest.approx(want, abs=0.02)


@pytest.mark.slow
def test_agreement_falls_from_rigid_to_polar_scaffolds(class_agreement):
    """The ordering claimed in section 3.9, not just the individual values."""
    caco2 = class_agreement.set_index("pathway")["caco2"]
    assert caco2["Shikimates and Phenylpropanoids"] > caco2["Terpenoids"] > caco2["Alkaloids"]
    assert caco2["Alkaloids"] > caco2["Polyketides"] >= caco2["Fatty acids"]
    assert caco2["Carbohydrates"] > caco2["Amino acids and Peptides"]
    assert caco2["Amino acids and Peptides"] < 0.3


@pytest.mark.parametrize("tool", ["ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0"])
def test_error_does_not_differ_across_pathways(tool, class_kruskal, expected):
    got = class_kruskal[tool]["p"]
    want = expected["class_stratified"]["kruskal_absolute_error_p"][tool]
    assert got == pytest.approx(want, abs=0.01)
    assert got > 0.05, "the paper reports no detectable difference in error across pathways"


def test_error_is_uniformly_high(class_accuracy, expected):
    """Median absolute error near half a log unit in every class."""
    lo, hi = expected["class_stratified"]["median_abs_err_range"]
    assert float(class_accuracy["median_abs_err"].min()) == pytest.approx(lo, abs=0.02)
    assert float(class_accuracy["median_abs_err"].max()) == pytest.approx(hi, abs=0.02)
    assert 0.2 < float(class_accuracy["median_abs_err"].median()) < 0.8


def test_pathways_with_enough_measured_compounds(class_accuracy, expected):
    assert sorted(class_accuracy["pathway"].unique()) == \
        expected["class_stratified"]["pathways_tested_accuracy"]
