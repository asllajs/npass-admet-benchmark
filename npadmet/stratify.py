"""Is the null result an artefact of the assay's uniform-mass dosing? (section 3.4)

The Caco-2 assay dosed every compound at 100 ug/mL rather than at a fixed molar
concentration, so heavier molecules were tested at a proportionally lower molar
concentration, and prediction error does rise with molecular weight. If that
convention were producing the null result rather than a genuine chemistry effect,
the correlation between prediction and measurement should reappear among the
lighter compounds, where the convention bites least.

Three views are computed, in increasing resolution:

``stratified_table``   a hard split at 500 Da, the comparison the reviewers asked for
``quartile_table``     the same metrics per quartile, which would expose a gradient
                       that a two-way split could hide
``partial_table``      the rank correlation between measured and predicted values
                       after the rank-linear effect of molecular weight is removed
                       from both sides

The third is the decisive one: it asks whether the tools carry any information
about permeability that molecular size does not already supply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from scipy.stats import t as student_t

from . import config as C
from . import datasets as D
from . import plots
from .caco2 import CACO2_COL
from .metrics import auroc, binary_metrics, continuous_metrics


def _strata(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        f"MW < {C.MW_SPLIT:.0f} Da": df["mw"] < C.MW_SPLIT,
        f"MW >= {C.MW_SPLIT:.0f} Da": df["mw"] >= C.MW_SPLIT,
        "all": df["mw"].notna(),
    }


def stratified_table(n_boot: int = C.N_BOOTSTRAP) -> pd.DataFrame:
    """Validation metrics computed separately below and above the split."""
    df = D.caco2_benchmark()
    rows = []
    for label, mask in _strata(df).items():
        sub = df[mask]
        for tool in C.TOOLS:
            d = sub[["measured_log", CACO2_COL[tool]]].dropna()
            if len(d) < C.MIN_N_SUBGROUP:
                rows.append({"stratum": label, "tool": C.TOOL_LABELS[tool],
                             "n": int(len(d)), "note": f"n < {C.MIN_N_SUBGROUP}"})
                continue
            m, p = d["measured_log"].to_numpy(), d[CACO2_COL[tool]].to_numpy()
            rows.append({"stratum": label, "tool": C.TOOL_LABELS[tool],
                         **continuous_metrics(m, p, n_boot=n_boot),
                         **binary_metrics(m, p)})
    return pd.DataFrame(rows)


def quartile_table() -> pd.DataFrame:
    """The same comparison in four molecular-weight bins."""
    df = D.caco2_benchmark()
    d = df[df["mw"].notna()].copy()
    d["mw_q"] = pd.qcut(d["mw"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for q, sub in d.groupby("mw_q", observed=True):
        for tool in C.TOOLS:
            s = sub[["measured_log", CACO2_COL[tool]]].dropna()
            if len(s) < C.MIN_N_SUBGROUP:
                continue
            m, p = s["measured_log"].to_numpy(), s[CACO2_COL[tool]].to_numpy()
            r, pv = pearsonr(m, p)
            rho, rho_p = spearmanr(m, p)
            y = (m > C.PERM_THRESHOLD).astype(int)
            rows.append({"mw_quartile": str(q),
                         "mw_min": float(sub["mw"].min()), "mw_max": float(sub["mw"].max()),
                         "tool": C.TOOL_LABELS[tool], "n": int(len(s)),
                         "pearson_r": float(r), "pearson_p": float(pv),
                         "spearman": float(rho), "spearman_p": float(rho_p),
                         "mae": float(np.mean(np.abs(p - m))),
                         "auroc": float(auroc(y, p)) if 0 < y.sum() < len(y) else np.nan})
    return pd.DataFrame(rows)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Spearman correlation of x and y with the rank-linear effect of z removed.

    Ranks are taken first, then z is regressed out of each of the two rank
    vectors, and the residuals are correlated. This is the standard partial
    Spearman; it is written out here rather than imported so that the definition
    used in the paper is visible.
    """
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    n = len(rx)

    def residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        slope = np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1)
        return a - slope * (b - b.mean()) - a.mean()

    r = float(np.corrcoef(residual(rx, rz), residual(ry, rz))[0, 1])
    dof = n - 3
    if dof <= 0 or abs(r) >= 1:
        return r, float("nan")
    tstat = r * np.sqrt(dof / (1 - r ** 2))
    return r, float(2 * student_t.sf(abs(tstat), dof))


def partial_table() -> pd.DataFrame:
    """Does any association survive once molecular weight is held fixed?"""
    df = D.caco2_benchmark()
    rows = []
    for tool in C.TOOLS:
        d = df[["measured_log", CACO2_COL[tool], "mw"]].dropna()
        m = d["measured_log"].to_numpy()
        p = d[CACO2_COL[tool]].to_numpy()
        w = d["mw"].to_numpy()
        rho_raw, p_raw = spearmanr(m, p)
        rho_par, p_par = partial_spearman(m, p, w)
        rows.append({"tool": C.TOOL_LABELS[tool], "n": int(len(d)),
                     "spearman_raw": float(rho_raw), "p_raw": float(p_raw),
                     "spearman_partial_mw": rho_par, "p_partial": p_par})
    return pd.DataFrame(rows)


def summarize(strat: pd.DataFrame, quart: pd.DataFrame, part: pd.DataFrame) -> dict:
    light = strat[strat["stratum"] == f"MW < {C.MW_SPLIT:.0f} Da"]
    heavy = strat[strat["stratum"] == f"MW >= {C.MW_SPLIT:.0f} Da"]
    return {
        "mw_split_da": C.MW_SPLIT,
        "n_light": int(light["n"].max()),
        "n_heavy": int(heavy["n"].max()),
        "pearson_r_light": {r.tool: round(float(r.pearson_r), 3) for r in light.itertuples()},
        "pearson_r_heavy": {r.tool: round(float(r.pearson_r), 3) for r in heavy.itertuples()},
        "max_abs_r_light": round(float(light["pearson_r"].abs().max()), 3),
        "min_p_light": round(float(light["pearson_p"].min()), 3),
        "max_abs_r_any_quartile": round(float(quart["pearson_r"].abs().max()), 3),
        "min_p_any_quartile": round(float(quart["pearson_p"].min()), 3),
        "partial_spearman": {r.tool: {"raw": round(float(r.spearman_raw), 3),
                                      "partial": round(float(r.spearman_partial_mw), 3),
                                      "p_partial": round(float(r.p_partial), 3)}
                             for r in part.itertuples()},
        "max_abs_partial_spearman": round(float(part["spearman_partial_mw"].abs().max()), 3),
    }


def figure() -> str:
    """Measured against predicted in each stratum, one column per tool."""
    df = D.caco2_benchmark()
    light = df["mw"] < C.MW_SPLIT
    fig, axes = plots.figure(width=10.5, height=6.5, nrows=2, ncols=3)
    lim = (-8, -3)
    for j, tool in enumerate(C.TOOLS):
        col = CACO2_COL[tool]
        for i, (mask, label) in enumerate([(light, f"MW < {C.MW_SPLIT:.0f} Da"),
                                           (~light, f"MW >= {C.MW_SPLIT:.0f} Da")]):
            ax = axes[i, j]
            d = df[mask][["measured_log", col]].dropna()
            ax.scatter(d["measured_log"], d[col], s=16, alpha=0.6,
                       color=plots.TOOL_COLORS[tool])
            ax.plot(lim, lim, ls="--", color="0.4", lw=1)
            ax.set(xlim=lim, ylim=lim,
                   title=C.TOOL_LABELS[tool] if i == 0 else "",
                   xlabel="Measured log Papp (cm/s)" if i == 1 else "",
                   ylabel=f"{label}\npredicted log Papp" if j == 0 else "")
            if len(d) >= C.MIN_N_SUBGROUP:
                r, pv = pearsonr(d["measured_log"], d[col])
                plots.annotate(ax, f"n = {len(d)}\nr = {r:+.2f}\np = {pv:.2f}")
    return plots.save(fig, "fig_mw_stratified")
