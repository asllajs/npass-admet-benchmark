"""Table 2 - the endpoint-harmonization crosswalk.

The crosswalk is the piece of the study a reader is most likely to want to reuse
or to disagree with, so its shape and its individual rules are checked
explicitly rather than assumed.
"""
from __future__ import annotations

import pytest

from npadmet import datasets as D
from tests.conftest import compute

VALID_TRANSFORMS = {"none", "invert", "scale100", "log10_pos", "neglog10_to_hours"}

# The documentation-versus-deployment discrepancies described in section 2.3.
# These are the individual judgement calls a reader is most likely to want to
# check, so each one is asserted by name rather than being covered in aggregate.
EXPECTED_TRANSFORMS = {
    ("hia", "ADMETlab 3.0"): "invert",      # reports P(HIA < 30%), i.e. the unfavourable class
    ("f20", "ADMETlab 3.0"): "invert",      # reports P(F < 20%)
    ("f30", "ADMETlab 3.0"): "invert",
    ("f50", "ADMETlab 3.0"): "invert",
    ("pampa", "ADMETlab 3.0"): "invert",
    ("ppb", "admetSAR 3.0"): "scale100",    # a 0-1 fraction, documented as a percentage
    ("vdss", "ADMET-AI"): "log10_pos",      # raw L/kg, not log10 as the other two report
    ("t_half", "admetSAR 3.0"): "neglog10_to_hours",
}


@pytest.fixture(scope="module")
def crosswalk():
    return compute(D.crosswalk)


@pytest.fixture(scope="module")
def canonical():
    return compute(D.canonical_endpoints)


def test_crosswalk_shape(crosswalk_summary, expected):
    want = expected["crosswalk"]
    assert crosswalk_summary["n_native_mappings"] == want["n_native_mappings"]
    assert crosswalk_summary["n_canonical_endpoints"] == want["n_canonical_endpoints"]
    assert crosswalk_summary["n_comparable_multi_source"] == want["n_comparable_multi_source"]


def test_endpoints_per_admet_category(crosswalk_summary, expected):
    assert crosswalk_summary["categories"] == expected["crosswalk"]["categories"]


def test_incomparable_endpoints_are_flagged(crosswalk_summary, expected):
    """Half-life, clearance and acute toxicity use different assays or dose bases."""
    assert crosswalk_summary["not_comparable_endpoints"] == \
        expected["crosswalk"]["not_comparable_endpoints"]


def test_every_mapping_declares_a_transform(crosswalk):
    assert crosswalk["transform"].notna().all()
    unknown = set(crosswalk["transform"].unique()) - VALID_TRANSFORMS
    assert not unknown, f"undocumented transform rule(s): {sorted(unknown)}"


@pytest.mark.parametrize(("endpoint", "tool"), sorted(EXPECTED_TRANSFORMS))
def test_documented_polarity_and_unit_corrections(endpoint, tool, crosswalk):
    row = crosswalk[(crosswalk["endpoint"] == endpoint) & (crosswalk["tool"] == tool)]
    assert len(row) == 1, f"{endpoint} x {tool} should appear exactly once"
    assert row["transform"].iloc[0] == EXPECTED_TRANSFORMS[(endpoint, tool)]


def test_no_other_mapping_is_transformed(crosswalk):
    """Everything not listed above is passed through unchanged."""
    transformed = {(r.endpoint, r.tool) for r in crosswalk.itertuples()
                   if r.transform != "none"}
    assert transformed == set(EXPECTED_TRANSFORMS)


def test_every_mapping_names_a_tool_and_a_native_endpoint(crosswalk):
    assert set(crosswalk["tool"].unique()) == {"ADMET-AI", "ADMETlab 3.0", "admetSAR 3.0"}
    assert crosswalk["native_name"].notna().all()


@pytest.mark.slow
def test_crosswalk_matches_the_released_table(canonical):
    """Every canonical endpoint claimed by the crosswalk exists in the data."""
    columns = set(compute(D.library_columns))
    missing = [f"{row.code}__{tool}"
               for row in canonical.itertuples()
               for tool in ("admet_ai", "admetlab3", "admetsar3")
               if f"{row.code}__{tool}" in columns]
    assert missing, "no endpoint columns found - is the library file the right one?"
    for row in canonical.itertuples():
        present = sum(f"{row.code}__{t}" in columns
                      for t in ("admet_ai", "admetlab3", "admetsar3"))
        assert present == row.n_sources, f"{row.code}: crosswalk says {row.n_sources} tools, data has {present}"
