"""Section 3.5 / Table 6: the applicability-domain flag does not identify the failures.

Most of the benchmark is in-domain by the conventional cutoff, so if distance
from the training set governed accuracy, those compounds would be predicted
better. They are not, at any cutoff, and prediction error does not rank with
distance. That is a stronger statement than the aggregate domain gap, and it is
what these tests hold the paper to.
"""
from __future__ import annotations

import pytest


def _group(df, prefix):
    return df[df["group"].str.startswith(prefix)].set_index("tool")


def test_three_quarters_of_the_benchmark_is_in_domain(domain_split, expected):
    """98 of 134 compounds sit at or above the conventional 0.30 cutoff."""
    exp = expected["domain_split"]
    assert int(_group(domain_split, "in-domain")["n"].max()) == exp["n_in_domain"]
    assert int(_group(domain_split, "out-of-domain")["n"].max()) == exp["n_out_of_domain"]
    assert exp["n_in_domain"] > 2 * exp["n_out_of_domain"]


def test_in_domain_compounds_are_not_predicted_better(domain_split, expected):
    """In-domain correlations are marginally higher but not significant."""
    exp = expected["domain_split"]
    inside = _group(domain_split, "in-domain")
    outside = _group(domain_split, "out-of-domain")
    for tool, value in exp["pearson_r_in"].items():
        assert inside.loc[tool, "pearson_r"] == pytest.approx(value, abs=0.01)
    for tool, value in exp["pearson_r_out"].items():
        assert outside.loc[tool, "pearson_r"] == pytest.approx(value, abs=0.01)
    assert inside["pearson_p"].min() > 0.05
    assert inside["pearson_r"].max() < 0.25


def test_the_two_groups_do_not_differ_significantly(domain_split, expected):
    """Fisher z on the difference between the in- and out-of-domain correlations."""
    exp = expected["domain_split"]["fisher_z_p"]
    got = _group(domain_split, "in-domain")
    for tool, value in exp.items():
        assert got.loc[tool, "fisher_z_p"] == pytest.approx(value, abs=0.02)
    assert got["fisher_z_p"].min() > 0.05


def test_conclusion_does_not_depend_on_the_cutoff(domain_thresholds, expected):
    """Repeat at 0.20 to 0.40: the in-domain correlation never becomes useful."""
    exp = expected["domain_split"]
    assert domain_thresholds["cutoff"].nunique() == exp["n_cutoffs_tested"]
    assert domain_thresholds["r_in"].max() == pytest.approx(
        exp["max_r_in_any_cutoff"], abs=0.01)
    assert domain_thresholds["r_in"].max() < 0.25
    # a couple of marginal hits out of thirty tests, none surviving multiplicity
    assert int((domain_thresholds["p_in"] < 0.05).sum()) == \
        exp["n_significant_r_in_across_cutoffs"]
    assert int((domain_thresholds["p_in"] < 0.01).sum()) == 0


def test_error_does_not_rank_with_distance(domain_quintiles, expected):
    """Binned error is flat and non-monotone, so distance cannot triage compounds."""
    exp = expected["domain_split"]
    assert domain_quintiles["spearman_err_vs_nn"].abs().max() == pytest.approx(
        exp["max_abs_spearman_err_vs_nn"], abs=0.02)
    assert domain_quintiles["spearman_p"].min() > 0.05

    for tool, values in exp["mae_by_quintile"].items():
        got = domain_quintiles[domain_quintiles["tool"] == tool].sort_values("quantile")
        assert list(got["mae"].round(3)) == pytest.approx(values, abs=0.02)
        # not monotone: the most similar quintile is not the most accurate one
        assert got["mae"].iloc[-1] > got["mae"].min()


def test_most_distant_pathways_are_not_the_worst_predicted(domain_pathways, expected):
    """The pathway furthest from the training domain is not the one with most error."""
    exp = expected["domain_split"]["pathway_median_nn"]
    got = domain_pathways.set_index("pathway")
    for pathway, value in exp.items():
        assert got.loc[pathway, "median_nn"] == pytest.approx(value, abs=0.02)
    closest = got["median_nn"].idxmax()
    worst = got[[c for c in got.columns if c.startswith("mae_")]].mean(axis=1).idxmax()
    assert closest != worst


@pytest.mark.slow
def test_benchmark_sits_closer_to_training_chemistry_than_the_library(chemspace_overlap,
                                                                     expected):
    """70% of the library is out of domain against 27% of the benchmark.

    The benchmark is therefore a conservative test: it is drawn from the part of
    natural-product space most like the tools' training data, and they fail on it
    anyway.
    """
    exp = expected["domain_split"]["chemspace_overlap"]
    got = chemspace_overlap.set_index("population")
    for population, values in exp.items():
        assert got.loc[population, "median_nn_tanimoto"] == pytest.approx(
            values["median_nn"], abs=0.01)
        assert got.loc[population, "frac_below_0.30"] == pytest.approx(
            values["frac_below_0.30"], abs=0.02)
    assert (got.loc["NPASS 3.0 library", "frac_below_0.30"]
            > 2 * got.loc["benchmark (measured)", "frac_below_0.30"])
    assert (got.loc["benchmark (measured)", "median_nn_tanimoto"]
            < got.loc["drug-like held out", "median_nn_tanimoto"])
