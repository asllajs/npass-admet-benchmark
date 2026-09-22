"""Chemical-space structure of the library, computed on every classified compound."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_uses_the_whole_library(chemical_space, expected):
    """No subsampling: the projection covers every compound with a pathway."""
    df, _, _ = chemical_space
    assert len(df) == expected["chemical_space"]["n_compounds"]


def test_pathway_counts(chemical_space, expected):
    df, _, order = chemical_space
    assert len(order) == expected["chemical_space"]["n_pathways"]
    for pathway, n in expected["chemical_space"]["n_by_pathway"].items():
        assert int((df["pathway"] == pathway).sum()) == n


def test_two_components_carry_most_of_the_variance(chemical_space, expected):
    _, evr, _ = chemical_space
    want = expected["chemical_space"]
    assert float(evr[0]) == pytest.approx(want["explained_variance_pc1"], abs=0.01)
    assert float(evr[1]) == pytest.approx(want["explained_variance_pc2"], abs=0.01)
    assert float(evr[0] + evr[1]) == pytest.approx(want["explained_variance_pc1_pc2"], abs=0.01)


def test_saturation_gradient_across_pathways(chemical_space, expected):
    """The aromatic-to-saturated gradient the figure is there to show."""
    df, _, order = chemical_space
    want = expected["chemical_space"]["median_fsp3"]
    for pathway, value in want.items():
        got = float(np.nanmedian(df.loc[df["pathway"] == pathway, "fsp3"]))
        assert got == pytest.approx(value, abs=0.02)
    assert want["Shikimates and Phenylpropanoids"] < want["Alkaloids"] < want["Terpenoids"]
    assert want["Carbohydrates"] > want["Terpenoids"]
