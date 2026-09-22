"""The claims of the abstract, one test each.

Everything above checks published numbers. These check the statements those
numbers are used to make, so that a reader can see at a glance whether the
paper's conclusions still follow from the released data.
"""
from __future__ import annotations

import pytest


def test_claim_tools_agree_on_descriptors_but_not_on_endpoints(agreement_table):
    """"agreed almost perfectly on deterministic descriptors ... only weakly on
    the ADMET endpoints themselves"."""
    tab = agreement_table.set_index("code")
    assert tab.loc[["mw", "tpsa", "hbd", "qed"], "threeway"].min() > 0.99
    admet = tab.drop(index=["mw", "tpsa", "hbd", "qed", "hba", "logp", "logs", "ppb"])
    assert admet["threeway"].median() < 0.5


test_claim_tools_agree_on_descriptors_but_not_on_endpoints = pytest.mark.slow(
    test_claim_tools_agree_on_descriptors_but_not_on_endpoints)


def test_claim_no_tool_tracks_experimental_permeability(caco2_table):
    """"none of them tracked the experimental apparent permeability"."""
    assert caco2_table["pearson_r"].max() < 0.15
    assert caco2_table["pearson_p"].min() > 0.05   # none reaches significance
    assert (caco2_table["r2"] < -1).all()
    assert caco2_table["auroc"].max() < 0.65


def test_claim_consensus_does_not_recover_accuracy(consensus_table, inter_tool_correlation):
    """"Combining the three tools into a consensus predictor did not recover accuracy
    ... because their errors are correlated rather than independent"."""
    t = consensus_table.set_index("predictor")["pearson_r"]
    assert t["Consensus (mean)"] < 0.1
    assert t["Consensus (mean)"] <= t[["ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0"]].max() + 0.01
    assert inter_tool_correlation["pearson_r"].min() > 0.8


def test_claim_pgp_predictions_are_uninformative(efflux_table):
    """"Predicted P-glycoprotein substrate probabilities were likewise uninformative"."""
    assert (efflux_table["auroc"] - 0.5).abs().max() < 0.1


def test_claim_agreement_is_class_dependent_but_error_is_not(class_agreement, class_kruskal):
    """"inter-tool agreement was itself class-dependent ... whereas the experimental
    error was uniformly high across pathways"."""
    caco2 = class_agreement.set_index("pathway")["caco2"]
    assert caco2.max() - caco2.min() > 0.4
    assert all(v["p"] > 0.05 for v in class_kruskal.values())


test_claim_agreement_is_class_dependent_but_error_is_not = pytest.mark.slow(
    test_claim_agreement_is_class_dependent_but_error_is_not)


def test_claim_benchmark_lies_outside_the_training_domain(domain_gap):
    """"median nearest-neighbor Tanimoto 0.39 versus 0.77 for held-out drug-like
    compounds"."""
    got = domain_gap.set_index("group")["median_nn_tanimoto"]
    assert got["natural products"] < 0.5 < got["drug-like held out"]
    assert float(domain_gap["mannwhitney_p"].iloc[0]) < 1e-10


def test_claim_endpoint_is_learnable_inside_the_training_domain(wang_control, caco2_table):
    """"the endpoint is learnable inside the drug-like domain and fails only on transfer".

    The in-domain anchor is the general tool reproducing the drug-like benchmark.
    The out-of-scaffold anchor is in test_14; together they are what replaced the
    same-panel cross-validated ceiling cited in the submitted version.
    """
    assert float(wang_control.set_index("subset").loc["test", "r2"]) > 0.7
    assert (caco2_table["r2"] < 0).all()


def test_claim_failure_is_not_a_dosing_artefact(mw_partial):
    """"holding molecular weight fixed removes the one marginal rank correlation"."""
    got = mw_partial.set_index("tool")
    assert got["spearman_partial_mw"].abs().max() < 0.05
    assert got["p_partial"].min() > 0.5


def test_claim_domain_flag_is_not_a_reliability_estimate(domain_split, domain_quintiles):
    """"three quarters lie above the cutoff and are predicted no better"."""
    inside = domain_split[domain_split["group"].str.startswith("in-domain")]
    outside = domain_split[domain_split["group"].str.startswith("out-of-domain")]
    assert int(inside["n"].max()) > 2 * int(outside["n"].max())
    assert inside["pearson_p"].min() > 0.05
    assert inside["fisher_z_p"].min() > 0.05
    assert domain_quintiles["spearman_p"].min() > 0.05


def test_claim_failure_is_at_the_domain_boundary_not_scaffold_novelty(generalization_table):
    """"a fixed model survives a scaffold split and collapses only on transfer"."""
    t = generalization_table
    def cell(regime):
        m = ((t["features"] == "morgan") & (t["train_set"] == "drug-like (Wang 2016)")
             & (t["regime"] == regime))
        return float(t[m].iloc[0]["r2_identity"])
    assert cell("random CV") > cell("scaffold CV") > 0
    assert cell("transfer to natural products") < -1
