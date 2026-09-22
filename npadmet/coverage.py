"""Library coverage and the size of the harmonization crosswalk (manuscript section 3.1)."""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import datasets as D


def library_coverage() -> dict:
    """How many of the standardized compounds each tool returned predictions for."""
    flags = ["has_admet_ai", "has_admetlab3", "has_admetsar3"]
    df = D.library(columns=("np_id", "inchikey", *flags))
    n = len(df)
    all_three = df[flags].all(axis=1).sum()
    return {
        "n_compounds": int(n),
        "n_unique_inchikey": int(df["inchikey"].nunique()),
        "coverage_pct": {t: round(100 * float(df[f"has_{t}"].mean()), 2) for t in C.TOOLS},
        "n_all_three": int(all_three),
        "pct_all_three": round(100 * float(all_three) / n, 2),
        "n_no_tool": int((~df[flags].any(axis=1)).sum()),
    }


def crosswalk_summary() -> dict:
    """Shape of the endpoint crosswalk that puts the three tools on one scale."""
    cw = D.crosswalk()
    canon = D.canonical_endpoints()
    comparable_multi = canon[(canon["comparable"]) & (canon["n_sources"] >= 2)]
    return {
        "n_native_mappings": int(len(cw)),
        "n_canonical_endpoints": int(len(canon)),
        "n_comparable_multi_source": int(len(comparable_multi)),
        "n_not_comparable": int((~canon["comparable"]).sum()),
        "categories": {k: int(v) for k, v in canon["category"].value_counts().sort_index().items()},
        "not_comparable_endpoints": sorted(canon.loc[~canon["comparable"], "code"]),
    }


def coverage_table() -> pd.DataFrame:
    cov = library_coverage()
    rows = [{"tool": C.TOOL_LABELS[t],
             "n_with_prediction": int(round(cov["coverage_pct"][t] / 100 * cov["n_compounds"])),
             "coverage_pct": cov["coverage_pct"][t]} for t in C.TOOLS]
    rows.append({"tool": "all three", "n_with_prediction": cov["n_all_three"],
                 "coverage_pct": cov["pct_all_three"]})
    return pd.DataFrame(rows)
