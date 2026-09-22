"""Inter-tool agreement across the library (manuscript section 3.8, Fig. 8).

Regression endpoints are compared with Lin's concordance correlation coefficient,
classification endpoints with Cohen's kappa pairwise and Fleiss' kappa three-way,
in both cases on complete cases only. The three-way figure for a regression
endpoint is the mean of its three pairwise CCCs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import datasets as D
from . import plots
from .metrics import ccc, cohen_kappa, fleiss_kappa

CATEGORY_ORDER = {"P": 0, "A": 1, "D": 2, "M": 3, "E": 4, "T": 5}
MIN_PAIR_N = 20
DESCRIPTOR_CONTROL = ("mw", "tpsa", "hbd", "qed")


def _endpoint_columns() -> list[str]:
    canon = D.canonical_endpoints()
    available = set(D.library_columns())
    return [f"{code}__{t}" for code in canon["code"] for t in C.TOOLS
            if f"{code}__{t}" in available]


def agreement_table() -> pd.DataFrame:
    """One row per canonical endpoint with pairwise and three-way agreement."""
    canon = D.canonical_endpoints()
    wide = D.library(columns=tuple(["np_id"] + _endpoint_columns()))

    rows = []
    for _, e in canon.iterrows():
        code, vtype = e["code"], e["value_type"]
        cols = {t: f"{code}__{t}" for t in C.TOOLS if f"{code}__{t}" in wide.columns}
        rec = {"category": e["category"], "code": code, "value_type": vtype,
               "comparable": bool(e["comparable"]), "n_sources": len(cols)}

        for (a, b), label in zip(C.TOOL_PAIRS, C.PAIR_LABELS):
            rec[label] = np.nan
            if a in cols and b in cols:
                d = wide[[cols[a], cols[b]]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(d) >= MIN_PAIR_N:
                    x, y = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
                    rec[label] = (ccc(x, y) if vtype == "regression"
                                  else cohen_kappa(x > 0.5, y > 0.5))
                    rec[f"n_{label}"] = len(d)

        if len(cols) == 3:
            d = wide[[cols[t] for t in C.TOOLS]].apply(pd.to_numeric, errors="coerce").dropna()
            rec["threeway"] = (fleiss_kappa((d.to_numpy() > 0.5).astype(int))
                               if vtype == "classification"
                               else float(np.nanmean([rec[l] for l in C.PAIR_LABELS])))
            rec["n_all3"] = len(d)
        else:
            rec["threeway"] = float(np.nanmean([rec[l] for l in C.PAIR_LABELS]))
            rec["n_all3"] = np.nan
        rows.append(rec)

    tab = pd.DataFrame(rows)
    tab["_order"] = tab["category"].map(CATEGORY_ORDER)
    return (tab.sort_values(["_order", "code"]).drop(columns="_order")
               .reset_index(drop=True))


def summarize(tab: pd.DataFrame) -> dict:
    """The claims the manuscript makes about agreement, as plain numbers."""
    by_code = tab.set_index("code")
    cls = tab[(tab["value_type"] == "classification") & tab["comparable"]]
    return {
        "threeway": {c: round(float(v), 3) for c, v in by_code["threeway"].items()},
        "descriptor_control_min_ccc": round(float(by_code.loc[list(DESCRIPTOR_CONTROL),
                                                             "threeway"].min()), 3),
        "n_classification_between_0.2_and_0.5": int(
            ((cls["threeway"] >= 0.2) & (cls["threeway"] <= 0.5)).sum()),
        "n_classification_comparable": int(len(cls)),
        "lowest_comparable": (tab.loc[tab.loc[tab["comparable"], "threeway"].idxmin(), "code"]
                              if tab["comparable"].any() else None),
    }


def figure(tab: pd.DataFrame) -> str:
    """Heatmap of endpoint x tool-pair agreement (equivalent of manuscript Fig. 8)."""
    cols = list(C.PAIR_LABELS) + ["threeway"]
    mat = tab.set_index("code")[cols]
    fig, ax = plots.figure(width=6.0, height=9.0)
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", vmin=-0.1, vmax=1.0,
                   cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["AI-L3", "AI-S3", "L3-S3", "3-way"])
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([f"[{c}] {code}" for c, code in zip(tab["category"], tab["code"])],
                       fontsize=7)
    for i in range(len(mat)):
        for j in range(len(cols)):
            v = mat.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Agreement (CCC / kappa)", shrink=0.4)
    ax.set_title("Inter-tool agreement per endpoint")
    return plots.save(fig, "fig_model_agreement")
