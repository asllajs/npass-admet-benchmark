"""Plain matplotlib helpers.

These figures exist to make the numbers visible, not to reproduce the typeset
figures of the manuscript. Matplotlib's default style and default font are used
on purpose: nothing here depends on a bundled typeface or a house style sheet, so
the plots look the same on any machine with a stock matplotlib installation.
Everything is written as PNG into ``output/``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcdefaults()

import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

DPI = 150
TOOL_COLORS = {"admet_ai": "tab:blue", "admetlab3": "tab:orange", "admetsar3": "tab:green"}


def figure(width: float = 7.0, height: float = 4.5, **kwargs):
    return plt.subplots(figsize=(width, height), **kwargs)


def save(fig, name: str) -> str:
    """Write ``output/<name>.png`` and return the path."""
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    path = C.OUTPUT / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def annotate(ax, text: str) -> None:
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8)
