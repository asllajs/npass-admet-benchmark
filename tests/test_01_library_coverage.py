"""Section 3.1 - how much of the library each tool covers.

If any of these fail, the compound-level join is different from the one the paper
used and nothing downstream can be compared.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_library_size(library_coverage, expected):
    assert library_coverage["n_compounds"] == expected["library_coverage"]["n_compounds"]


def test_one_structure_per_compound(library_coverage):
    """Standardized SMILES were in one-to-one correspondence with InChIKeys."""
    assert library_coverage["n_unique_inchikey"] == library_coverage["n_compounds"]


@pytest.mark.parametrize("tool", ["admet_ai", "admetlab3", "admetsar3"])
def test_per_tool_coverage(tool, library_coverage, expected):
    got = library_coverage["coverage_pct"][tool]
    want = expected["library_coverage"]["coverage_pct"][tool]
    assert got == pytest.approx(want, abs=0.05)


def test_all_three_tools(library_coverage, expected):
    want = expected["library_coverage"]
    assert library_coverage["n_all_three"] == want["n_all_three"]
    assert library_coverage["pct_all_three"] == pytest.approx(want["pct_all_three"], abs=0.05)


def test_no_compound_without_any_prediction(library_coverage):
    assert library_coverage["n_no_tool"] == 0
