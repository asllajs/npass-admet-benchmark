"""Is the benchmark really a set of natural products?

The paper's subject is natural products, so it is worth confirming rather than
assuming that the validation set is one. NPASS 3.0 is itself a curated
natural-product database, which makes membership an objective check, and the
released library table is already on disk for the other analyses. The comparison is
made after counter-ions are stripped, so that a compound supplied as a salt or
hydrate is matched to its parent entry rather than counted as a miss.
"""
from __future__ import annotations

import pytest
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr

from npadmet import datasets as D
from npadmet.caco2 import CACO2_COL
from tests.conftest import compute

RDLogger.DisableLog("rdApp.*")


def desalted_key(smiles: str) -> str | None:
    """Connectivity block of the largest fragment, i.e. with counter-ions removed."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    largest = max(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True),
                  key=lambda f: f.GetNumHeavyAtoms())
    return Chem.MolToInchiKey(largest)[:14]


@pytest.fixture(scope="module")
def benchmark():
    return compute(D.caco2_experimental)


@pytest.fixture(scope="module")
def npass_keys():
    lib = compute(D.library, ("inchikey",))
    return set(lib["inchikey"].astype(str).str[:14])


def test_dataset_size(benchmark, expected):
    want = expected["benchmark_provenance"]
    assert len(benchmark) == want["n_compounds"]
    assert int((benchmark["papp_ab_cm_s"] > 0).sum()) == want["n_with_papp"]


def test_every_row_carries_a_measurement(benchmark):
    """Compounds assayed without detectable transport are outside the benchmark."""
    measured = ["papp_ab_cm_s", "papp_ba_cm_s", "efflux_ratio"]
    assert benchmark[measured].notna().any(axis=1).all()


@pytest.mark.slow
def test_every_compound_is_a_catalogued_natural_product(benchmark, npass_keys, expected):
    """Every benchmark compound appears in the natural-product catalogue."""
    keys = benchmark["smiles"].map(desalted_key)
    found = keys.isin(npass_keys)
    assert int(found.sum()) == expected["benchmark_provenance"]["n_in_npass_after_desalting"]
    assert found.all(),         f"not in NPASS 3.0: {list(benchmark.loc[~found, 'compound_name'])}"


def test_salt_forms_are_counted(benchmark, expected):
    """Recorded because these compounds were submitted to the tools as salts."""
    salts = int(benchmark["smiles"].astype(str).str.contains(r"\.").sum())
    assert salts == expected["benchmark_provenance"]["n_salt_forms"]


def test_biosynthetic_assignment(benchmark, expected):
    unclassified = int((benchmark["npc_pathway"] == "Unclassified").sum())
    assert unclassified == expected["benchmark_provenance"]["n_unclassified_pathway"]


@pytest.mark.parametrize("tool", list(CACO2_COL))
def test_conclusion_holds_without_the_salt_forms(tool, expected):
    """The verdict is not an artefact of submitting counter-ions to the predictors."""
    b = compute(D.caco2_benchmark)
    free_base = b[~b["smiles"].astype(str).str.contains(r"\.")]
    d = free_base[["measured_log", CACO2_COL[tool]]].dropna()
    r, p = pearsonr(d["measured_log"], d[CACO2_COL[tool]])
    assert abs(r) < 0.15
    assert p > 0.05
