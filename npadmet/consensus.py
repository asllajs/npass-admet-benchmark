"""Does averaging the three tools recover accuracy? (manuscript section 3.3, Table 4)

Scored on the complete-case subset, so every predictor - each single tool and the
two consensus rules - is judged on exactly the same compounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from . import config as C
from . import datasets as D
from . import plots
from .caco2 import CACO2_COL
from .metrics import binary_metrics, continuous_metrics


def complete_case() -> pd.DataFrame:
    df = D.caco2_benchmark()
    cols = list(CACO2_COL.values())
    cc = df.dropna(subset=cols).copy()
    cc["consensus_mean"] = cc[cols].mean(axis=1)
    cc["consensus_median"] = cc[cols].median(axis=1)
    return cc


def inter_tool_correlation(cc: pd.DataFrame) -> pd.DataFrame:
    """How strongly the three tools agree with *each other* on these compounds."""
    rows = []
    for i, a in enumerate(C.TOOLS):
        for b in C.TOOLS[i + 1:]:
            r, p = pearsonr(cc[CACO2_COL[a]], cc[CACO2_COL[b]])
            rows.append({"tool_a": C.TOOL_LABELS[a], "tool_b": C.TOOL_LABELS[b],
                         "pearson_r": float(r), "pearson_p": float(p)})
    return pd.DataFrame(rows)


def consensus_table(n_boot: int = C.N_BOOTSTRAP) -> pd.DataFrame:
    cc = complete_case()
    predictors = [(C.TOOL_LABELS[t], CACO2_COL[t]) for t in C.TOOLS]
    predictors += [("Consensus (mean)", "consensus_mean"),
                   ("Consensus (median)", "consensus_median")]
    rows = []
    for label, col in predictors:
        measured = cc["measured_log"].to_numpy()
        predicted = cc[col].to_numpy()
        rows.append({"predictor": label,
                     **continuous_metrics(measured, predicted, n_boot=n_boot),
                     **binary_metrics(measured, predicted)})
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, inter: pd.DataFrame) -> dict:
    t = table.set_index("predictor")
    single = t.loc[[C.TOOL_LABELS[x] for x in C.TOOLS], "pearson_r"]
    return {
        "n_complete_case": int(t["n"].iloc[0]),
        "inter_tool_r": {f"{r.tool_a} vs {r.tool_b}": round(float(r.pearson_r), 3)
                         for r in inter.itertuples()},
        "mean_inter_tool_r": round(float(inter["pearson_r"].mean()), 3),
        "vs_measured_r": {k: round(float(v), 3) for k, v in t["pearson_r"].items()},
        "best_single_r": round(float(single.max()), 3),
        "consensus_mean_r": round(float(t.loc["Consensus (mean)", "pearson_r"]), 3),
        "consensus_median_r": round(float(t.loc["Consensus (median)", "pearson_r"]), 3),
        "consensus_beats_best_single": bool(
            t.loc["Consensus (mean)", "pearson_r"] > single.max()),
    }


def figure(table: pd.DataFrame) -> str:
    cc = complete_case()
    fig, (ax, ax2) = plots.figure(width=10.0, height=4.5, ncols=2)

    x = cc["measured_log"].to_numpy()
    y = cc["consensus_mean"].to_numpy()
    lim = (-8, -3)
    ax.scatter(x, y, s=16, color="black", alpha=0.55)
    ax.plot(lim, lim, ls="--", color="0.5", lw=1)
    ax.axhline(C.PERM_THRESHOLD, ls=":", color="0.7", lw=0.8)
    ax.axvline(C.PERM_THRESHOLD, ls=":", color="0.7", lw=0.8)
    ax.set(xlim=lim, ylim=lim, xlabel="Measured log Papp (cm/s)",
           ylabel="Mean-consensus predicted log Papp (cm/s)")
    plots.annotate(ax, f"n = {len(cc)}\nr = {pearsonr(x, y)[0]:+.2f}")

    labels = table["predictor"].tolist()
    values = table["pearson_r"].tolist()
    colors = [plots.TOOL_COLORS[t] for t in C.TOOLS] + ["black", "0.6"]
    ax2.barh(np.arange(len(labels)), values, color=colors, edgecolor="0.2")
    ax2.axvline(0, color="0.3", lw=0.8)
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Pearson r against measured log Papp")
    return plots.save(fig, "fig_consensus")
