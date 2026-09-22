"""Chemical-space structure of the library by biosynthetic pathway (Fig. 2).

Every compound with a pathway assignment is used; nothing is subsampled. The two
descriptors shown are a principal-component projection of six deterministic
physicochemical properties and the fraction of sp3 carbons, which is the usual
summary of natural-product structural saturation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from . import config as C
from . import datasets as D
from . import plots

RDLogger.DisableLog("rdApp.*")

DESCRIPTORS = ("mw__admet_ai", "logp__admet_ai", "tpsa__admet_ai",
               "hba__admet_ai", "hbd__admet_ai", "qed__admet_ai")
N_PATHWAYS = 7
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442")


def _pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = (x - x.mean(0)) / x.std(0)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    return u[:, :2] * s[:2], (s ** 2 / (s ** 2).sum())[:2]


def fsp3(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles))
    return rdMolDescriptors.CalcFractionCSP3(mol) if mol is not None else np.nan


def chemical_space(with_fsp3: bool = True) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = D.library(columns=("np_id", "smiles", "pathway", *DESCRIPTORS)).dropna(subset=list(DESCRIPTORS))
    df["pathway"] = D.primary(df["pathway"])
    top = [p for p in df["pathway"].value_counts().index if p != "Unclassified"][:N_PATHWAYS]
    df = df[df["pathway"].isin(top)].reset_index(drop=True)

    x = df[list(DESCRIPTORS)].to_numpy(dtype=float)
    lo, hi = np.nanpercentile(x, [1, 99], axis=0)   # winsorize so a few giants do not own PC1
    scores, evr = _pca2(np.clip(x, lo, hi))
    df["pc1"], df["pc2"] = scores[:, 0], scores[:, 1]
    if with_fsp3:
        df["fsp3"] = df["smiles"].map(fsp3)
    order = df["pathway"].value_counts().index.tolist()
    return df, evr, order


def summarize(df: pd.DataFrame, evr: np.ndarray, order: list[str]) -> dict:
    return {
        "n_compounds": int(len(df)),
        "n_pathways": len(order),
        "explained_variance_pc1": round(float(evr[0]), 3),
        "explained_variance_pc2": round(float(evr[1]), 3),
        "explained_variance_pc1_pc2": round(float(evr[0] + evr[1]), 3),
        "n_by_pathway": {p: int((df["pathway"] == p).sum()) for p in order},
        "median_fsp3": {p: round(float(np.nanmedian(df.loc[df["pathway"] == p, "fsp3"])), 2)
                        for p in order} if "fsp3" in df.columns else {},
    }


def figure(df: pd.DataFrame, evr: np.ndarray, order: list[str]) -> str:
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(order)}
    fig, (ax, ax2) = plots.figure(width=11.0, height=4.6, ncols=2,
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(len(df))   # draw in random order: no pathway sits on top
    ax.scatter(df["pc1"].to_numpy()[perm], df["pc2"].to_numpy()[perm], s=1,
               c=[colors[p] for p in df["pathway"].to_numpy()[perm]],
               alpha=0.08, linewidths=0, rasterized=True)
    for p in order:
        ax.scatter([], [], s=24, color=colors[p], label=f"{p} (n = {(df['pathway'] == p).sum():,})")
    ax.set(xlim=np.percentile(df["pc1"], [0.3, 99.7]),
           ylim=np.percentile(df["pc2"], [0.3, 99.7]),
           xlabel=f"PC1 ({evr[0] * 100:.0f}% of variance)",
           ylabel=f"PC2 ({evr[1] * 100:.0f}% of variance)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Chemical space of {len(df):,} classified compounds")

    if "fsp3" in df.columns:
        data = [df.loc[df["pathway"] == p, "fsp3"].dropna().to_numpy() for p in order]
        bp = ax2.boxplot(data, vert=False, showfliers=False, widths=0.6, patch_artist=True)
        for patch, p in zip(bp["boxes"], order):
            patch.set_facecolor(colors[p])
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
        ax2.set_yticks(range(1, len(order) + 1))
        ax2.set_yticklabels([p.replace(" and ", " &\n") for p in order], fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("Fraction of sp3 carbons")
    return plots.save(fig, "fig_chemical_space")
