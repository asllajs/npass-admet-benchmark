"""Verification suite for the three-tool ADMET benchmark on the NPASS 3.0 library.

The package recomputes every quantitative claim of the paper from the public data
and exposes each analysis as a function returning a table, so that the pytest
suite in ``tests/`` can assert the published values rather than eyeball a figure.
"""
from __future__ import annotations

__version__ = "2.0.0"

from . import (  # noqa: F401
    agreement,
    benchmark,
    caco2,
    chemspace,
    classes,
    config,
    consensus,
    coverage,
    datasets,
    domain,
    generalization,
    metrics,
    plots,
    stratify,
)
