"""Sections 3.6 and 3.7 / Table 7: where the generalization fails, and what a split is worth.

Two claims are checked here, both with one model held fixed across every cell.

First, that the failure sits at the domain boundary rather than at scaffold
novelty: a model trained on drug-like Caco-2 data keeps most of its accuracy
under a strict Bemis-Murcko scaffold split and loses all of it on transfer to
natural products.

Second, that the cross-validated accuracy reported for natural-product
permeability models depends on whether folds are drawn over measurements or over
compounds. The replicate-level tests skip unless the Kim et al. (2026)
supplementary document has been downloaded, since it is third-party material that
is not redistributed here.
"""
from __future__ import annotations

import pytest

DRUG_LIKE = "drug-like (Wang 2016)"
NATURAL = "natural products (this study)"


def _cell(table, features, train_set, regime):
    m = ((table["features"] == features) & (table["train_set"] == train_set)
         & (table["regime"] == regime))
    assert m.sum() == 1, f"expected one row for {features}/{train_set}/{regime}"
    return table[m].iloc[0]


def test_scaffold_counts(scaffold_counts, expected):
    """The two chemistries share few scaffolds, which is the premise of the test."""
    exp = expected["generalization"]["scaffold_counts"]
    for key, value in exp.items():
        assert scaffold_counts[key] == value
    assert scaffold_counts["shared_scaffolds"] < 0.25 * scaffold_counts["benchmark_scaffolds"]


def test_scaffold_split_costs_little_within_drug_like_space(generalization_table, expected):
    """R2 0.72 -> 0.57 going from a random split to a scaffold-disjoint one."""
    exp = expected["generalization"]["morgan"]
    rand = _cell(generalization_table, "morgan", DRUG_LIKE, "random CV")
    scaf = _cell(generalization_table, "morgan", DRUG_LIKE, "scaffold CV")
    assert rand["r2_identity"] == pytest.approx(exp["drug_like_random_cv"]["r2"], abs=0.02)
    assert scaf["r2_identity"] == pytest.approx(exp["drug_like_scaffold_cv"]["r2"], abs=0.02)
    assert scaf["r2_identity"] > 0.4          # still generalizes to unseen scaffolds
    assert rand["r2_identity"] - scaf["r2_identity"] < 0.25   # and the cost is modest


def test_transfer_to_natural_products_collapses(generalization_table, expected):
    """The same model, applied across the domain boundary, is worse than useless."""
    exp = expected["generalization"]["morgan"]
    tr = _cell(generalization_table, "morgan", DRUG_LIKE, "transfer to natural products")
    assert tr["pearson_r"] == pytest.approx(exp["drug_like_transfer"]["r"], abs=0.02)
    assert tr["r2_identity"] == pytest.approx(exp["drug_like_transfer"]["r2"], abs=0.05)
    assert tr["r2_identity"] < -1
    assert tr["pearson_r"] < 0.15


def test_auroc_degrades_on_the_same_three_steps(generalization_table, expected):
    """Reviewer 3 asked for AUROC as well as R2, and it tells the same story.

    Classification of high versus low permeability survives a scaffold split and
    falls to chance on transfer, which is the regression result in a metric that
    does not depend on calibration.
    """
    exp = expected["generalization"]["morgan"]
    rand = _cell(generalization_table, "morgan", DRUG_LIKE, "random CV")
    scaf = _cell(generalization_table, "morgan", DRUG_LIKE, "scaffold CV")
    tr = _cell(generalization_table, "morgan", DRUG_LIKE, "transfer to natural products")
    assert rand["auroc"] == pytest.approx(exp["drug_like_random_cv"]["auroc"], abs=0.02)
    assert scaf["auroc"] == pytest.approx(exp["drug_like_scaffold_cv"]["auroc"], abs=0.02)
    assert tr["auroc"] == pytest.approx(exp["drug_like_transfer"]["auroc"], abs=0.03)
    assert rand["auroc"] > scaf["auroc"] > 0.8      # scaffold novelty is survivable
    assert tr["auroc"] < 0.6                        # leaving the domain is not
    assert (scaf["auroc"] - tr["auroc"]) > 3 * (rand["auroc"] - scaf["auroc"])


def test_the_descriptor_variant_degrades_the_same_way(generalization_table, expected):
    """The result is a property of the data, not of the fingerprint."""
    exp = expected["generalization"]["descriptor"]
    rand = _cell(generalization_table, "descriptor", DRUG_LIKE, "random CV")
    scaf = _cell(generalization_table, "descriptor", DRUG_LIKE, "scaffold CV")
    tr = _cell(generalization_table, "descriptor", DRUG_LIKE, "transfer to natural products")
    assert rand["auroc"] == pytest.approx(exp["drug_like_random_cv_auroc"], abs=0.02)
    assert scaf["auroc"] == pytest.approx(exp["drug_like_scaffold_cv_auroc"], abs=0.02)
    assert tr["auroc"] == pytest.approx(exp["drug_like_transfer_auroc"], abs=0.03)
    assert rand["r2_identity"] > scaf["r2_identity"] > 0 > tr["r2_identity"]


def test_the_reverse_transfer_is_reported_too(generalization_table, expected):
    """Training on natural products and predicting drug-like compounds also fails.

    Reported because it rules out the reading that natural-product chemistry is
    simply the harder of the two: neither direction transfers usefully.
    """
    for features in ("morgan", "descriptor"):
        exp = expected["generalization"][features]["np_transfer_to_drug_like"]
        row = _cell(generalization_table, features, NATURAL, "transfer to drug-like")
        assert row["pearson_r"] == pytest.approx(exp["r"], abs=0.03)
        assert row["r2_identity"] == pytest.approx(exp["r2"], abs=0.03)
        assert row["r2_identity"] < 0.1
        assert row["auroc"] < 0.65


def test_the_degradation_is_ordered(generalization_table):
    """random split > scaffold split >> transfer. The ordering is the claim."""
    rand = _cell(generalization_table, "morgan", DRUG_LIKE, "random CV")["r2_identity"]
    scaf = _cell(generalization_table, "morgan", DRUG_LIKE, "scaffold CV")["r2_identity"]
    tr = _cell(generalization_table, "morgan", DRUG_LIKE,
               "transfer to natural products")["r2_identity"]
    assert rand > scaf > 0 > tr
    # the transfer step costs far more than the scaffold step
    assert (scaf - tr) > 4 * (rand - scaf)


def test_descriptor_features_behave_the_same_way(generalization_table, expected):
    """The conclusion is not an artefact of the fingerprint representation."""
    exp = expected["generalization"]["descriptor"]
    for regime, key in (("random CV", "drug_like_random_cv_r2"),
                        ("scaffold CV", "drug_like_scaffold_cv_r2"),
                        ("transfer to natural products", "drug_like_transfer_r2")):
        got = _cell(generalization_table, "descriptor", DRUG_LIKE, regime)["r2_identity"]
        assert got == pytest.approx(exp[key], abs=0.05)


def test_natural_products_alone_do_not_train_a_model(generalization_table, expected):
    """Reported as an exploratory control: 134 compounds of this diversity are not enough."""
    exp = expected["generalization"]["morgan"]
    rand = _cell(generalization_table, "morgan", NATURAL, "random CV")["r2_identity"]
    scaf = _cell(generalization_table, "morgan", NATURAL, "scaffold CV")["r2_identity"]
    assert rand == pytest.approx(exp["np_random_cv_r2"], abs=0.05)
    assert scaf == pytest.approx(exp["np_scaffold_cv_r2"], abs=0.05)
    assert rand < 0 and scaf < 0


def test_134_is_not_in_itself_too_few(learning_curve, expected):
    """On drug-like data the same model does fine at the benchmark's sample size.

    This separates "not enough compounds" from "not enough coverage of the space".
    """
    exp = expected["generalization"]["learning_curve_at_134"]
    row = learning_curve[learning_curve["train_size"] == 134].iloc[0]
    assert row["pearson_r_mean"] == pytest.approx(exp["pearson_r_mean"], abs=0.05)
    assert row["r2_mean"] == pytest.approx(exp["r2_mean"], abs=0.05)
    assert row["r2_mean"] > 0.25


# --- section 3.7: the replicate-level re-analysis --------------------------

def test_the_panel_is_replicate_level_and_heavily_zero(replicate_zeros, expected):
    """248 rows for 83 compounds, 37% of them recorded at exactly zero."""
    exp = expected["replicate_reanalysis"]
    assert int(replicate_zeros["n_rows"].iloc[0]) == exp["n_rows"]
    assert int(replicate_zeros["n_compounds"].iloc[0]) == exp["n_compounds"]
    assert int(replicate_zeros["n_rows_zero"].iloc[0]) == exp["n_rows_zero"]
    assert int(replicate_zeros["n_compounds_all_zero"].iloc[0]) == exp["n_compounds_all_zero"]


def test_replicate_agreement_caps_a_row_level_split(replicate_spread, expected):
    """Within-compound variance is not a function of structure, so it sets a ceiling."""
    exp = expected["replicate_reanalysis"]["log_scale"]
    row = replicate_spread.set_index("scale").loc["log10 (zeros excluded)"]
    assert int(row["n_rows"]) == exp["n_rows"]
    assert int(row["n_compounds"]) == exp["n_compounds"]
    assert row["mean_within_compound_sd"] == pytest.approx(
        exp["within_compound_sd"], abs=0.005)
    assert row["between_compound_sd"] == pytest.approx(
        exp["between_compound_sd"], abs=0.01)
    assert row["max_achievable_r2_row_level"] == pytest.approx(
        exp["max_achievable_r2_row_level"], abs=0.005)
    assert row["max_achievable_r2_row_level"] > 0.95


def test_the_raw_scale_ceiling_matches_the_reported_accuracy(replicate_spread, expected):
    """On the scale the source study reports, the row-level ceiling is 0.98.

    The accuracy reported for this panel, 0.955, sits just below that ceiling, which
    is what a row-level split makes attainable without the model learning structure.
    """
    exp = expected["replicate_reanalysis"]["raw_scale"]
    row = replicate_spread.set_index("scale").loc["raw papp (zeros retained)"]
    assert int(row["n_rows"]) == exp["n_rows"]
    assert int(row["n_compounds"]) == exp["n_compounds"]
    assert row["max_achievable_r2_row_level"] == pytest.approx(
        exp["max_achievable_r2_row_level"], abs=0.005)
    assert row["max_achievable_r2_row_level"] > 0.955


def test_compound_level_split_removes_the_reported_accuracy(replicate_protocols, expected):
    """R2 falls from 0.77 to below zero once replicates are kept together."""
    exp = expected["replicate_reanalysis"]["protocols"]
    got = replicate_protocols.set_index("split")
    for split, values in exp.items():
        assert got.loc[split, "r2_identity"] == pytest.approx(values["r2"], abs=0.05)
    assert got.loc["row-level random CV", "r2_identity"] > 0.7
    assert got.loc["compound-level random CV", "r2_identity"] < 0


def test_the_raw_scale_shows_the_same_protocol_effect(replicate_protocols, expected):
    """The reported figure reproduces under a row-level split and not a compound-level one.

    This is the comparison that bears on the published R2 = 0.955, because it is on
    the same value scale. Keeping replicates together costs the model 0.80 of R2.
    """
    exp = expected["replicate_reanalysis"]["protocols"]
    got = replicate_protocols.set_index("split")
    row = got.loc["raw scale, row-level random CV"]
    comp = got.loc["raw scale, compound-level random CV"]
    assert row["r2_identity"] == pytest.approx(
        exp["raw scale, row-level random CV"]["r2"], abs=0.02)
    assert comp["r2_identity"] == pytest.approx(
        exp["raw scale, compound-level random CV"]["r2"], abs=0.02)
    assert row["r2_identity"] > 0.9
    assert 0 < comp["r2_identity"] < 0.2
    assert row["r2_identity"] - comp["r2_identity"] > 0.7


def test_the_resolved_subset_is_reported_not_the_whole_panel(replicate_protocols, expected):
    """Structures resolve for 45 of 83 compounds, and the tables say so."""
    exp = expected["replicate_reanalysis"]
    got = replicate_protocols.set_index("split")
    assert int(got.loc["raw scale, row-level random CV", "n"]) == exp["raw_scale"]["n_rows"]
    assert int(got.loc["raw scale, row-level random CV", "n_compounds"]) == \
        exp["raw_scale"]["n_compounds"]
    assert int(got.loc["compound-level random CV", "n"]) == exp["log_scale"]["n_rows"]
    assert int(got.loc["compound-level random CV", "n_compounds"]) == \
        exp["log_scale"]["n_compounds"]
    assert exp["raw_scale"]["n_compounds"] < exp["n_compounds"]
    assert got.loc["compound-level scaffold CV", "r2_identity"] < 0
