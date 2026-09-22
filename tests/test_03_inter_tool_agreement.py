"""Section 3.8 - the three tools agree on descriptors, not on ADMET endpoints."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

TOL = 0.02
DETERMINISTIC = ["mw", "tpsa", "hbd", "qed"]


def test_deterministic_descriptors_agree_almost_perfectly(agreement_table, expected):
    """The internal control on the InChIKey join: identical descriptors must match."""
    got = agreement_table.set_index("code").loc[DETERMINISTIC, "threeway"]
    assert got.min() >= expected["inter_tool_agreement"]["deterministic_descriptor_min_ccc"]


def test_complete_case_count(agreement_table, expected):
    n = agreement_table.set_index("code").loc["caco2", "n_all3"]
    assert int(n) == expected["inter_tool_agreement"]["n_complete_case_all_three"]


@pytest.mark.parametrize("code", [
    "caco2", "logs", "ppb", "vdss", "hia", "bbb", "pgp_inhibitor", "pgp_substrate",
    "cyp1a2_inh", "cyp2c19_inh", "cyp2c9_inh", "cyp2c9_sub", "cyp2d6_inh",
    "cyp2d6_sub", "cyp3a4_inh", "cyp3a4_sub", "herg", "ames", "dili",
])
def test_threeway_agreement_per_endpoint(code, agreement_table, expected):
    got = float(agreement_table.set_index("code").loc[code, "threeway"])
    want = expected["inter_tool_agreement"]["threeway"][code]
    assert got == pytest.approx(want, abs=TOL)


def test_caco2_agreement_is_moderate(agreement_table):
    """Mutually correlated Caco-2 predictions - the premise of the consensus test."""
    got = float(agreement_table.set_index("code").loc["caco2", "threeway"])
    assert 0.5 < got < 0.7


def test_cyp2c9_substrate_is_no_better_than_chance(agreement_table):
    got = float(agreement_table.set_index("code").loc["cyp2c9_sub", "threeway"])
    assert got <= 0.0


def test_most_classification_endpoints_are_only_fairly_agreed(agreement_table):
    """"Most classification endpoints reach Fleiss' kappa of just 0.2-0.5"."""
    cls = agreement_table[(agreement_table["value_type"] == "classification")
                          & agreement_table["comparable"]]
    in_band = ((cls["threeway"] >= 0.15) & (cls["threeway"] <= 0.55)).sum()
    assert in_band > len(cls) / 2


def test_descriptors_agree_better_than_endpoints(agreement_table):
    """The headline contrast of section 3.8, stated as an inequality."""
    tab = agreement_table.set_index("code")
    descriptors = tab.loc[DETERMINISTIC, "threeway"].min()
    endpoints = tab.drop(index=DETERMINISTIC + ["hba", "logp"])["threeway"].max()
    assert descriptors > endpoints
