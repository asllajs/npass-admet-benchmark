"""Shared fixtures.

Every analysis is computed once per session and reused, because several of them
read the 150 MB library table. Anything whose input has not been downloaded is
turned into an explicit skip - the run says what is missing instead of quietly
passing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from npadmet import (agreement, benchmark, caco2, chemspace, classes, consensus,
                     coverage, domain, generalization, stratify)
from npadmet.datasets import DataUnavailable

ROOT = Path(__file__).resolve().parent.parent


def compute(fn, *args, **kwargs):
    """Run an analysis, or skip the test if its input is not on disk."""
    try:
        return fn(*args, **kwargs)
    except DataUnavailable as exc:
        pytest.skip(str(exc))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: reads the whole library table")


@pytest.fixture(scope="session")
def expected() -> dict:
    return json.loads((ROOT / "expected_results.json").read_text(encoding="utf-8"))


# --- library scale ---------------------------------------------------------

@pytest.fixture(scope="session")
def library_coverage():
    return compute(coverage.library_coverage)


@pytest.fixture(scope="session")
def crosswalk_summary():
    return compute(coverage.crosswalk_summary)


@pytest.fixture(scope="session")
def agreement_table():
    return compute(agreement.agreement_table)


@pytest.fixture(scope="session")
def class_agreement():
    return compute(classes.agreement_by_pathway)


@pytest.fixture(scope="session")
def chemical_space():
    return compute(chemspace.chemical_space)


# --- experimental benchmark ------------------------------------------------

@pytest.fixture(scope="session")
def caco2_table():
    return compute(caco2.validation_table)


@pytest.fixture(scope="session")
def efflux_table():
    return compute(caco2.efflux_table)


@pytest.fixture(scope="session")
def consensus_table():
    return compute(consensus.consensus_table)


@pytest.fixture(scope="session")
def inter_tool_correlation():
    return compute(lambda: consensus.inter_tool_correlation(consensus.complete_case()))


@pytest.fixture(scope="session")
def class_accuracy():
    return compute(classes.accuracy_table)


@pytest.fixture(scope="session")
def class_kruskal():
    return compute(classes.accuracy_kruskal)


# --- applicability domain --------------------------------------------------

@pytest.fixture(scope="session")
def domain_gap():
    return compute(domain.domain_gap)


@pytest.fixture(scope="session")
def error_vs_distance():
    return compute(domain.error_vs_distance)


@pytest.fixture(scope="session")
def mw_confound():
    return compute(domain.molecular_weight_confound)


@pytest.fixture(scope="session")
def wang_control():
    return compute(domain.wang_positive_control)


# --- molecular-weight stratification (section 3.4) -------------------------

@pytest.fixture(scope="session")
def mw_stratified():
    return compute(stratify.stratified_table)


@pytest.fixture(scope="session")
def mw_quartiles():
    return compute(stratify.quartile_table)


@pytest.fixture(scope="session")
def mw_partial():
    return compute(stratify.partial_table)


# --- in-domain versus out-of-domain (section 3.5) --------------------------

@pytest.fixture(scope="session")
def domain_split():
    return compute(domain.domain_split_table)


@pytest.fixture(scope="session")
def domain_thresholds():
    return compute(domain.threshold_sensitivity)


@pytest.fixture(scope="session")
def domain_quintiles():
    return compute(domain.error_quintiles)


@pytest.fixture(scope="session")
def domain_pathways():
    return compute(domain.domain_by_pathway)


# --- generalization (sections 3.6 and 3.7) ---------------------------------

@pytest.fixture(scope="session")
def generalization_table():
    return compute(generalization.generalization_table, kinds=("morgan", "descriptor"))


@pytest.fixture(scope="session")
def scaffold_counts():
    return compute(generalization.scaffold_counts)


@pytest.fixture(scope="session")
def learning_curve():
    return compute(generalization.learning_curve)


@pytest.fixture(scope="session")
def replicate_protocols():
    return compute(generalization.replicate_protocols)


@pytest.fixture(scope="session")
def replicate_spread():
    return compute(generalization.replicate_spread)


@pytest.fixture(scope="session")
def replicate_zeros():
    return compute(generalization.replicate_zero_fraction)


# --- what the benchmark is made of (sections 2.1, 2.4, discussion) ---------

@pytest.fixture(scope="session")
def tautomers():
    return compute(benchmark.tautomer_table)


@pytest.fixture(scope="session")
def tautomer_effect(tautomers):
    return compute(benchmark.tautomer_effect, tautomers)


@pytest.fixture(scope="session")
def representativeness():
    return compute(benchmark.representativeness)


@pytest.fixture(scope="session")
def subset_heterogeneity():
    return compute(benchmark.subset_heterogeneity)


@pytest.fixture(scope="session")
def descriptor_signal():
    return compute(benchmark.descriptor_signal)


@pytest.fixture(scope="session")
def chemspace_overlap():
    return compute(domain.chemspace_overlap)


# --- monolayer integrity (section 3.10) ------------------------------------

@pytest.fixture(scope="session")
def teer_sensitivity():
    return compute(benchmark.teer_sensitivity)


@pytest.fixture(scope="session")
def teer_composition():
    return compute(benchmark.teer_composition)


@pytest.fixture(scope="session")
def teer_descriptor_signal():
    return compute(benchmark.teer_descriptor_signal)
