"""Class-stratified analysis by NPClassifier biosynthetic pathway (section 3.9).

Two questions that behave differently and are therefore kept apart:

* accuracy - does a tool's error against experiment depend on pathway?
  (benchmark set, Kruskal-Wallis on the absolute error)
* agreement - does inter-tool agreement depend on pathway?
  (full library, three-way CCC / Fleiss' kappa within each pathway)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr

from . import config as C
from . import datasets as D
from . import plots
from .caco2 import CACO2_COL
from .metrics import ccc, fleiss_kappa

KEY_ENDPOINTS = (("caco2", "reg"), ("logs", "reg"), ("ppb", "reg"), ("vdss", "reg"),
                 ("hia", "cls"), ("bbb", "cls"), ("pgp_inhibitor", "cls"),
                 ("cyp3a4_inh", "cls"), ("cyp2d6_inh", "cls"),
                 ("herg", "cls"), ("ames", "cls"), ("dili", "cls"))


# --------------------------------------------------------------------------
# Accuracy by pathway (benchmark set)
# --------------------------------------------------------------------------

def _benchmark_with_pathway() -> tuple[pd.DataFrame, list[str]]:
    df = D.caco2_benchmark()
    counts = df.groupby("npc_pathway")["measured_log"].count()
    keep = counts[counts >= C.MIN_PATHWAY_N_ACCURACY]
    order = keep.sort_values(ascending=False).index.tolist()
    return df[df["npc_pathway"].isin(order)].copy(), order


def accuracy_table() -> pd.DataFrame:
    """Per-pathway, per-tool error against the measured permeability (Table S8)."""
    df, order = _benchmark_with_pathway()
    rows = []
    for tool in C.TOOLS:
        col = CACO2_COL[tool]
        for pathway in order:
            d = df.loc[df["npc_pathway"] == pathway, ["measured_log", col]].dropna()
            abs_err = (d[col] - d["measured_log"]).abs()
            rows.append({
                "pathway": pathway, "tool": C.TOOL_LABELS[tool], "n": int(len(d)),
                "pearson_r": float(pearsonr(d["measured_log"], d[col])[0])
                if len(d) >= C.MIN_PATHWAY_N_ACCURACY else np.nan,
                "median_abs_err": float(abs_err.median()),
            })
    return pd.DataFrame(rows)


def accuracy_kruskal() -> dict:
    """Is the absolute error distributed differently across pathways?"""
    df, order = _benchmark_with_pathway()
    out = {}
    for tool in C.TOOLS:
        col = CACO2_COL[tool]
        groups = []
        for pathway in order:
            d = df.loc[df["npc_pathway"] == pathway, ["measured_log", col]].dropna()
            g = (d[col] - d["measured_log"]).abs().to_numpy()
            if len(g):
                groups.append(g)
        h, p = kruskal(*groups)
        out[C.TOOL_LABELS[tool]] = {"H": round(float(h), 3), "p": round(float(p), 4),
                                    "n_groups": len(groups)}
    return out


# --------------------------------------------------------------------------
# Agreement by pathway (full library)
# --------------------------------------------------------------------------

def _threeway(sub: pd.DataFrame, code: str, kind: str) -> float:
    cols = [f"{code}__{t}" for t in C.TOOLS if f"{code}__{t}" in sub.columns]
    if len(cols) < 2:
        return np.nan
    d = sub[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < C.MIN_N_AGREEMENT:
        return np.nan
    if kind == "cls":
        return fleiss_kappa((d.to_numpy() > 0.5).astype(int))
    pairs = [ccc(d[a].to_numpy(), d[b].to_numpy())
             for i, a in enumerate(cols) for b in cols[i + 1:]]
    return float(np.nanmean(pairs))


def agreement_by_pathway() -> pd.DataFrame:
    """Three-way agreement per (pathway, endpoint) over the whole library."""
    cols = ["pathway"] + [f"{code}__{t}" for code, _ in KEY_ENDPOINTS for t in C.TOOLS]
    wide = D.library(columns=tuple(cols))
    wide["pathway"] = D.primary(wide["pathway"])

    counts = wide["pathway"].value_counts()
    pathways = [p for p in counts.index
                if counts[p] >= C.MIN_PATHWAY_N_AGREEMENT and p != "Unclassified"]

    mat = pd.DataFrame(index=pathways, columns=[c for c, _ in KEY_ENDPOINTS], dtype=float)
    for pathway in pathways:
        sub = wide[wide["pathway"] == pathway]
        for code, kind in KEY_ENDPOINTS:
            mat.loc[pathway, code] = _threeway(sub, code, kind)
    mat.index.name = "pathway"
    return mat.reset_index()


def summarize(accuracy: pd.DataFrame, kruskal_result: dict,
              agreement: pd.DataFrame) -> dict:
    caco2 = agreement.set_index("pathway")["caco2"]
    return {
        "caco2_agreement_by_pathway": {k: round(float(v), 2) for k, v in caco2.items()},
        "kruskal_absolute_error": kruskal_result,
        "median_abs_err_range": [round(float(accuracy["median_abs_err"].min()), 2),
                                 round(float(accuracy["median_abs_err"].max()), 2)],
        "pathways_tested_accuracy": sorted(accuracy["pathway"].unique()),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figure_accuracy() -> str:
    df, order = _benchmark_with_pathway()
    fig, ax = plots.figure(width=9.0, height=4.5)
    width = 0.26
    for k, tool in enumerate(C.TOOLS):
        col = CACO2_COL[tool]
        data = [(df.loc[df["npc_pathway"] == p, col]
                 - df.loc[df["npc_pathway"] == p, "measured_log"]).abs().dropna().to_numpy()
                for p in order]
        positions = np.arange(len(order)) + (k - 1) * width
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(plots.TOOL_COLORS[tool])
            patch.set_alpha(0.65)
        for median in bp["medians"]:
            median.set_color("black")
        bp["boxes"][0].set_label(C.TOOL_LABELS[tool])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([p.replace(" and ", " &\n") for p in order], fontsize=8)
    ax.set_ylabel("|predicted - measured| (log Papp)")
    ax.set_title("Caco-2 prediction error by biosynthetic pathway")
    ax.legend(fontsize=8)
    return plots.save(fig, "fig_class_accuracy")


def figure_agreement(mat: pd.DataFrame) -> str:
    m = mat.set_index("pathway")
    fig, ax = plots.figure(width=9.0, height=4.5)
    im = ax.imshow(m.to_numpy(dtype=float), aspect="auto", vmin=-0.1, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Three-way agreement (CCC / kappa)", shrink=0.7)
    ax.set_title("Inter-tool agreement by pathway (full library)")
    return plots.save(fig, "fig_class_agreement")
