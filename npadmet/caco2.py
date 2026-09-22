"""Validation against experimental Caco-2 permeability (manuscript section 3.2).

This is the study's central result and the part that a reader most needs to be
able to check: the three tools agree with one another but none of them tracks the
measured apparent permeability of natural products.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config as C
from . import datasets as D
from . import plots
from .metrics import auroc, binary_metrics, continuous_metrics

CACO2_COL = {t: f"caco2__{t}" for t in C.TOOLS}
PGP_COL = {"admetlab3": "pgp_substrate__admetlab3", "admetsar3": "pgp_substrate__admetsar3"}


def validation_table(n_boot: int = C.N_BOOTSTRAP) -> pd.DataFrame:
    """Per-tool agreement with the measured log Papp (manuscript Table 3)."""
    df = D.caco2_benchmark()
    rows = []
    for tool in C.TOOLS:
        measured = df["measured_log"].to_numpy()
        predicted = df[CACO2_COL[tool]].to_numpy()
        rows.append({"tool": C.TOOL_LABELS[tool],
                     **continuous_metrics(measured, predicted, n_boot=n_boot),
                     **binary_metrics(measured, predicted)})
    return pd.DataFrame(rows)


def efflux_table() -> pd.DataFrame:
    """Predicted P-glycoprotein substrate probability against the measured efflux ratio.

    Only two of the three tools report a P-gp *substrate* probability; ADMET-AI
    reports inhibition instead, which is a different question.
    """
    df = D.caco2_experimental().merge(D.caco2_predictions(), on="inchikey14", how="inner")
    rows = []
    for tool, col in PGP_COL.items():
        d = df[["efflux_ratio", col]].dropna()
        d = d[d["efflux_ratio"] > 0]
        y = (d["efflux_ratio"] >= C.EFFLUX_THRESHOLD).astype(int).to_numpy()
        score = d[col].to_numpy()
        rho, p = spearmanr(d["efflux_ratio"], d[col])
        rows.append({"tool": C.TOOL_LABELS[tool], "n": int(len(d)),
                     "n_efflux_positive": int(y.sum()),
                     "auroc": float(auroc(y, score)),
                     "spearman": float(rho), "spearman_p": float(p)})
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, efflux: pd.DataFrame) -> dict:
    t = table.set_index("tool")
    return {
        "per_tool": {tool: {k: (round(float(v), 3) if isinstance(v, float) else int(v))
                            for k, v in t.loc[tool].items()}
                     for tool in t.index},
        "pearson_r_range": [round(float(t["pearson_r"].min()), 3),
                            round(float(t["pearson_r"].max()), 3)],
        "min_pearson_p": round(float(t["pearson_p"].min()), 3),
        "r2_range": [round(float(t["r2"].min()), 2), round(float(t["r2"].max()), 2)],
        "auroc_range": [round(float(t["auroc"].min()), 2), round(float(t["auroc"].max()), 2)],
        "efflux": {r["tool"]: {"n": int(r["n"]), "n_efflux_positive": int(r["n_efflux_positive"]),
                               "auroc": round(float(r["auroc"]), 2)}
                   for _, r in efflux.iterrows()},
    }


def figure_validation() -> str:
    """Measured against predicted, plus Bland-Altman, for each tool."""
    df = D.caco2_benchmark()
    fig, axes = plots.figure(width=10.5, height=6.5, nrows=2, ncols=3)
    lim = (-8, -3)
    for j, tool in enumerate(C.TOOLS):
        d = df[["measured_log", CACO2_COL[tool]]].dropna()
        x, y = d["measured_log"].to_numpy(), d[CACO2_COL[tool]].to_numpy()
        color = plots.TOOL_COLORS[tool]

        ax = axes[0, j]
        ax.scatter(x, y, s=14, color=color, alpha=0.6)
        ax.plot(lim, lim, ls="--", color="0.4", lw=1)
        ax.axhline(C.PERM_THRESHOLD, ls=":", color="0.7", lw=0.8)
        ax.axvline(C.PERM_THRESHOLD, ls=":", color="0.7", lw=0.8)
        ax.set(xlim=lim, ylim=lim, xlabel="Measured log Papp (cm/s)",
               ylabel="Predicted log Papp (cm/s)" if j == 0 else "",
               title=C.TOOL_LABELS[tool])
        r, _ = pearsonr(x, y)
        plots.annotate(ax, f"n = {len(d)}\nr = {r:+.2f}\n"
                           f"RMSE = {np.sqrt(np.mean((y - x) ** 2)):.2f}")

        ax2 = axes[1, j]
        diff = y - x
        bias, sd = diff.mean(), diff.std()
        ax2.scatter((x + y) / 2, diff, s=14, color=color, alpha=0.6)
        ax2.axhline(bias, color="0.2", lw=1)
        ax2.axhline(bias + 1.96 * sd, ls="--", color="0.5", lw=0.8)
        ax2.axhline(bias - 1.96 * sd, ls="--", color="0.5", lw=0.8)
        ax2.set(xlabel="Mean of measured and predicted (log cm/s)",
                ylabel="Predicted - measured (log units)" if j == 0 else "")
        plots.annotate(ax2, f"bias = {bias:+.2f}\nLoA = +/-{1.96 * sd:.2f}")
    return plots.save(fig, "fig_caco2_validation")


def figure_efflux() -> str:
    df = D.caco2_experimental().merge(D.caco2_predictions(), on="inchikey14", how="inner")
    fig, axes = plots.figure(width=8.0, height=4.0, ncols=2)
    for ax, (tool, col) in zip(axes, PGP_COL.items()):
        d = df[["efflux_ratio", col]].dropna()
        d = d[d["efflux_ratio"] > 0]
        y = (d["efflux_ratio"] >= C.EFFLUX_THRESHOLD).astype(int).to_numpy()
        score = d[col].to_numpy()
        groups = [score[y == 0], score[y == 1]]
        ax.boxplot(groups, tick_labels=["ER < 2", "ER >= 2"], showfliers=False, widths=0.5)
        rng = np.random.default_rng(C.SEED)
        for i, g in enumerate(groups, 1):
            ax.scatter(np.full(len(g), i) + rng.uniform(-0.1, 0.1, len(g)), g,
                       s=14, color=plots.TOOL_COLORS[tool], alpha=0.5)
        ax.set(title=C.TOOL_LABELS[tool], xlabel="Measured efflux ratio class",
               ylabel="Predicted P-gp substrate probability")
        plots.annotate(ax, f"n = {len(d)}\nAUROC = {auroc(y, score):.2f}")
    return plots.save(fig, "fig_efflux_pgp")
