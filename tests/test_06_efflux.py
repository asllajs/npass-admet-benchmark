"""Section 3.2 - predicted P-glycoprotein substrate probability against measured efflux."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("tool", ["ADMETlab 3.0", "admetSAR 3.0"])
def test_published_efflux_metrics(tool, efflux_table, expected):
    row = efflux_table.set_index("tool").loc[tool]
    want = expected["efflux_pgp"][tool]
    assert int(row["n"]) == want["n"]
    assert int(row["n_efflux_positive"]) == want["n_efflux_positive"]
    assert float(row["auroc"]) == pytest.approx(want["auroc"], abs=0.01)


def test_pgp_predictions_are_uninformative(efflux_table):
    """AUROC either side of 0.5 - no better than sorting the compounds at random."""
    assert (efflux_table["auroc"] - 0.5).abs().max() < 0.1


def test_only_two_tools_report_substrate_probability(efflux_table):
    """ADMET-AI reports P-gp inhibition, which is a different question."""
    assert set(efflux_table["tool"]) == {"ADMETlab 3.0", "admetSAR 3.0"}
