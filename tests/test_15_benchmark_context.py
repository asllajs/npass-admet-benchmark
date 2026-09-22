"""Sections 2.1 and 2.4 and the discussion: what the benchmark is made of.

Three properties of the reference set that the manuscript states and a reader
should be able to check: that tautomer handling does not explain the result, that
the benchmark is not a uniform sample of the library (and is biased in the tools'
favour), and that its two halves behave differently.

The last of these is reported in the paper precisely because the released data
make it reproducible; the test exists so that it stays reported correctly.
"""
from __future__ import annotations

import pytest


def test_tautomerism_applies_to_most_compounds(tautomers, expected):
    """69% of the benchmark has more than one enumerated tautomer."""
    exp = expected["benchmark_context"]["tautomers"]
    assert len(tautomers) == exp["n_compounds"]
    assert int(tautomers["tautomerism_relevant"].sum()) == exp["n_with_multiple"]
    assert int(tautomers["canonical_differs_from_submitted"].sum()) == \
        exp["n_canonical_differs"]


def test_tautomerism_does_not_explain_the_error(tautomer_effect, expected):
    """Error is indistinguishable between compounds with and without tautomers.

    The tools do not disclose their own preprocessing, so this bounds how much a
    tautomer mismatch could account for: on this benchmark, nothing.
    """
    exp = expected["benchmark_context"]["tautomers"]
    assert tautomer_effect["mannwhitney_p"].min() == pytest.approx(
        exp["min_mannwhitney_p"], abs=0.05)
    assert tautomer_effect["mannwhitney_p"].min() > 0.05
    diff = (tautomer_effect["mae_tautomeric"] - tautomer_effect["mae_single_form"]).abs()
    assert diff.max() < 0.1


@pytest.mark.slow
def test_benchmark_is_closer_to_drug_like_chemistry_than_the_library(representativeness,
                                                                    expected):
    """The benchmark is lighter and less lipophilic than the library it stands for.

    This is a limitation stated in the paper, and it runs in the conservative
    direction: the tools are tested on the part of the space most like their
    training data.
    """
    exp = expected["benchmark_context"]["representativeness"]
    got = representativeness.set_index("descriptor")
    for desc, values in exp.items():
        assert got.loc[desc, "benchmark_median"] == pytest.approx(
            values["benchmark_median"], rel=0.02)
        assert got.loc[desc, "library_median"] == pytest.approx(
            values["library_median"], rel=0.02)
    assert got.loc["MolWt", "benchmark_median"] < got.loc["MolWt", "library_median"]
    assert got.loc["MolLogP", "benchmark_median"] < got.loc["MolLogP", "library_median"]
    assert got.loc["FractionCSP3", "benchmark_median"] < got.loc["FractionCSP3",
                                                                 "library_median"]
    # but it is not a narrow slice: most of the library falls inside its range
    assert got.loc["MolWt", "library_frac_within_benchmark_range"] > 0.8


def test_the_two_halves_of_the_benchmark_disagree(subset_heterogeneity, expected):
    """Compounds tabulated in one of the three source papers correlate positively, the rest negatively."""
    exp = expected["benchmark_context"]["subset_heterogeneity"]
    got = subset_heterogeneity.set_index(["subset", "tool"])
    for key, values in exp.items():
        subset, tool = [s.strip() for s in key.split("|")]
        assert int(got.loc[(subset, tool), "n"]) == values["n"]
        assert got.loc[(subset, tool), "pearson_r"] == pytest.approx(values["r"], abs=0.01)

    published = subset_heterogeneity[subset_heterogeneity["subset"] == "tabulated in a source publication"]
    unpublished = subset_heterogeneity[
        subset_heterogeneity["subset"] == "not tabulated there"]
    assert (published["pearson_r"] > 0).all()
    assert (unpublished["pearson_r"] < 0).all()
    assert subset_heterogeneity["fisher_z_p_between_halves"].min() == pytest.approx(
        expected["benchmark_context"]["min_fisher_z_p_between_halves"], abs=0.001)


def test_the_pooled_benchmark_is_still_null(subset_heterogeneity):
    """The heterogeneity does not rescue any tool on the benchmark as a whole."""
    whole = subset_heterogeneity[subset_heterogeneity["subset"] == "all"]
    assert whole["pearson_r"].max() < 0.15
    assert whole["pearson_p"].min() > 0.05


def test_measurements_follow_permeability_descriptors_where_published(descriptor_signal,
                                                                     expected):
    """The reference data behave as Caco-2 permeability should, on the published half.

    Directions are fixed in EXPECTED_SIGN before the data are looked at. The
    drug-like reference set meets all six; the published half of the benchmark
    meets all six; the unpublished half meets two, which is why the manuscript
    reports the heterogeneity rather than pooling silently.
    """
    exp = expected["benchmark_context"]["descriptor_signal_matches"]
    got = descriptor_signal[descriptor_signal["has_prior"]]
    n_prior = int(got.groupby("group").size().max())
    assert n_prior == expected["benchmark_context"]["descriptor_signal_n_with_prior"]
    for group, n_match in exp.items():
        assert int(got[got["group"] == group]["matches_expectation"].sum()) == n_match
    assert exp["drug-like (Wang 2016)"] == n_prior
    assert exp["benchmark, in published tables"] == n_prior
    assert exp["benchmark, not in published tables"] < n_prior / 2


# --- section 3.10: monolayer integrity -------------------------------------

def test_filtering_on_integrity_raises_the_correlation(teer_sensitivity, expected):
    """Correlations rise monotonically as the TEER threshold tightens.

    The manuscript reports this rather than filtering, because part of the null
    result is measurement quality and saying so is more useful than a headline
    number that hides it.
    """
    exp = expected["benchmark_context"]["monolayer_integrity"]
    at20 = teer_sensitivity[teer_sensitivity["cutoff"] == "-20"].set_index("tool")
    at10 = teer_sensitivity[teer_sensitivity["cutoff"] == "-10"].set_index("tool")
    for tool, value in exp["pearson_r_at_-20"].items():
        assert at20.loc[tool, "pearson_r"] == pytest.approx(value, abs=0.01)
    for tool, value in exp["pearson_r_at_-10"].items():
        assert at10.loc[tool, "pearson_r"] == pytest.approx(value, abs=0.01)

    unfiltered = teer_sensitivity[teer_sensitivity["cutoff"] == "none"].set_index("tool")
    for tool in unfiltered.index:
        assert at10.loc[tool, "pearson_r"] > unfiltered.loc[tool, "pearson_r"]
    # at the conventional criterion one tool becomes nominally significant
    assert at20["pearson_p"].min() < 0.05


def test_filtering_does_not_make_the_predictions_usable(teer_sensitivity, expected):
    """R2 and AUROC are unmoved by the filter, which is the point of the table."""
    exp = expected["benchmark_context"]["monolayer_integrity"]
    assert teer_sensitivity["r2"].max() == pytest.approx(exp["r2_range"][1], abs=0.05)
    assert teer_sensitivity["r2"].min() == pytest.approx(exp["r2_range"][0], abs=0.05)
    assert (teer_sensitivity["r2"] < 0).all()
    assert teer_sensitivity["auroc"].max() == pytest.approx(exp["auroc_range"][1], abs=0.02)
    assert teer_sensitivity["auroc"].max() < 0.65
    # the largest correlation anywhere still explains under a tenth of the variance
    assert teer_sensitivity["pearson_r"].max() ** 2 < 0.10


def test_the_integrity_filter_is_not_chemically_neutral(teer_composition, expected):
    """It preferentially discards the heavier, less permeable compounds."""
    exp = expected["benchmark_context"]["monolayer_integrity"]["composition_at_-20"]
    got = teer_composition.set_index("group")
    for group, values in exp.items():
        assert int(got.loc[group, "n"]) == values["n"]
        assert got.loc[group, "median_mw"] == pytest.approx(values["median_mw"], abs=2.0)
    assert got.loc["dropped", "median_mw"] > got.loc["kept", "median_mw"]
    assert got.loc["dropped", "median_measured_log"] < got.loc["kept", "median_measured_log"]


def test_filtered_measurements_follow_permeability_descriptors(teer_descriptor_signal,
                                                               expected):
    """Five of six expected directions unfiltered, six of six on every filtered subset."""
    exp = expected["benchmark_context"]["monolayer_integrity"]["descriptor_signs"]
    got = teer_descriptor_signal.set_index("cutoff")
    for cutoff, value in exp.items():
        assert int(got.loc[cutoff, "signs_matching"]) == value
    assert int(got.loc["none", "signs_matching"]) == 5
    assert all(int(got.loc[c, "signs_matching"]) == 6 for c in got.index if c != "none")
