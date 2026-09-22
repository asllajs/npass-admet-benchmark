"""What the benchmark is made of (sections 2.1, 2.4, and the discussion).

Three questions about the reference set rather than about the tools:

``tautomer_table``      for how many of the 134 compounds is tautomerism formally
                        available, and does prediction error differ for them?
                        The tools do not disclose their own preprocessing, so this
                        bounds how much a tautomer mismatch could explain.
``representativeness``  how the benchmark compares with the library it stands for,
                        on the descriptors that govern passive permeability. The
                        benchmark is lighter and less lipophilic than the library,
                        which biases the test in the tools' favour.
``subset_heterogeneity``the benchmark divides into the compounds tabulated in the
                        source publications and those that are not, and the two
                        halves behave differently. Reported in the manuscript
                        rather than left for a reader of the released data to find.
``descriptor_signal``   whether the measurements move with polar surface area,
                        hydrogen bonding and size in the directions Caco-2
                        permeability is known to take. Stated as expectations in
                        advance so the check cannot be read off after the fact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import ks_2samp, mannwhitneyu, norm, pearsonr, spearmanr

from . import config as C
from . import datasets as D
from .caco2 import CACO2_COL

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_FN = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "FractionCSP3": Descriptors.FractionCSP3,
    "RingCount": Descriptors.RingCount,
}

# Expected sign of the association with log Papp, from the permeability
# literature. "0" means no strong prior, and those are excluded from the tally.
EXPECTED_SIGN = {
    "TPSA": "-",
    "MolLogP": "+",
    "NumHDonors": "-",
    "NumHAcceptors": "-",
    "MolWt": "-",
    "NumRotatableBonds": "-",
    "FractionCSP3": "0",
    "RingCount": "0",
}

LIBRARY_SAMPLE = 20000   # descriptors are recomputed from SMILES, so a sample is used


def describe(smiles) -> pd.DataFrame:
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        rows.append({k: (fn(mol) if mol is not None else np.nan)
                     for k, fn in DESCRIPTOR_FN.items()})
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------
# tautomers
# --------------------------------------------------------------------------

def tautomer_table(max_tautomers: int = 200) -> pd.DataFrame:
    """Enumerate tautomers of each benchmark compound."""
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    df = D.caco2_benchmark()
    rows = []
    for r in df.itertuples():
        mol = Chem.MolFromSmiles(str(r.smiles))
        if mol is None:
            continue
        try:
            n_forms = len(enumerator.Enumerate(mol))
            canonical = Chem.MolToSmiles(enumerator.Canonicalize(mol))
        except Exception:
            n_forms, canonical = 1, Chem.MolToSmiles(mol)
        rows.append({"inchikey14": r.inchikey14, "compound_name": r.compound_name,
                     "n_tautomers": int(n_forms),
                     "tautomerism_relevant": bool(n_forms > 1),
                     "canonical_differs_from_submitted":
                         bool(canonical != Chem.MolToSmiles(mol))})
    return pd.DataFrame(rows)


def tautomer_effect(taut: pd.DataFrame) -> pd.DataFrame:
    """Does prediction error differ where tautomerism applies?"""
    df = D.caco2_benchmark().merge(
        taut[["inchikey14", "tautomerism_relevant"]], on="inchikey14", how="inner")
    rows = []
    for tool in C.TOOLS:
        col = CACO2_COL[tool]
        s = df[[col, "measured_log", "tautomerism_relevant"]].dropna()
        err = (s[col] - s["measured_log"]).abs()
        a, b = err[s["tautomerism_relevant"]], err[~s["tautomerism_relevant"]]
        p = float(mannwhitneyu(a, b).pvalue) if min(len(a), len(b)) > 3 else np.nan
        rows.append({"tool": C.TOOL_LABELS[tool],
                     "n_tautomeric": int(len(a)), "mae_tautomeric": float(a.mean()),
                     "n_single_form": int(len(b)), "mae_single_form": float(b.mean()),
                     "mannwhitney_p": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# representativeness
# --------------------------------------------------------------------------

def representativeness(n_sample: int = LIBRARY_SAMPLE) -> pd.DataFrame:
    """The benchmark against the library, on the descriptors that govern permeability."""
    bench = describe(D.caco2_benchmark()["smiles"])
    cols = D.library_columns()
    smi_col = next((c for c in ("smiles", "canonical_smiles", "std_smiles") if c in cols), None)
    if smi_col is None:
        raise D.DataUnavailable(
            "the library table has no SMILES column, so the benchmark cannot be "
            "compared with it")
    lib_smiles = D.library(columns=(smi_col,))[smi_col].dropna()
    lib_smiles = lib_smiles.sample(n=min(n_sample, len(lib_smiles)), random_state=C.SEED)
    lib = describe(lib_smiles)

    rows = []
    for desc in DESCRIPTOR_FN:
        b, l = bench[desc].dropna(), lib[desc].dropna()
        ks, p_ks = ks_2samp(b, l)
        lo, hi = b.quantile(0.05), b.quantile(0.95)
        rows.append({"descriptor": desc,
                     "benchmark_n": int(len(b)), "benchmark_median": float(b.median()),
                     "benchmark_p5": float(lo), "benchmark_p95": float(hi),
                     "library_n": int(len(l)), "library_median": float(l.median()),
                     "library_p5": float(l.quantile(0.05)),
                     "library_p95": float(l.quantile(0.95)),
                     "ks_statistic": float(ks), "ks_p": float(p_ks),
                     "library_frac_within_benchmark_range": float(
                         ((l >= lo) & (l <= hi)).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# published versus unpublished halves
# --------------------------------------------------------------------------

def _published_flag(df: pd.DataFrame) -> pd.Series:
    """Which benchmark compounds appear in the published tables of the source papers.

    The flag is derived by matching compound names against the text of those
    papers, so it is an annotation produced here rather than a measurement, and it
    ships with the repository instead of with the data archive. Name matching is
    imperfect for salt forms and alternative names; the split is good enough for a
    difference this large but should not be treated as definitive per compound.
    """
    if "published" in df.columns:
        return df["published"].astype("boolean")
    if not C.FILE_PUBLICATION_STATUS.exists():
        raise D.DataUnavailable(
            f"missing {C.FILE_PUBLICATION_STATUS.name} in data/; it is tracked in this "
            f"repository, not on Zenodo, and records which benchmark compounds appear "
            f"in the published tables")
    status = pd.read_csv(C.FILE_PUBLICATION_STATUS).set_index("inchikey14")["published"]
    return df["inchikey14"].map(status).astype("boolean")


def subset_heterogeneity() -> pd.DataFrame:
    """Agreement with the measurement, for each half of the benchmark."""
    df = D.caco2_benchmark()
    flag = _published_flag(df)
    rows = []
    for label, mask in [("tabulated in a source publication", flag == True),      # noqa: E712
                        ("not tabulated there", flag == False),  # noqa: E712
                        ("all", pd.Series(True, index=df.index))]:
        sub = df[mask.fillna(False) if hasattr(mask, "fillna") else mask]
        for tool in C.TOOLS:
            s = sub[["measured_log", CACO2_COL[tool]]].dropna()
            if len(s) < C.MIN_N_SUBGROUP:
                continue
            r, p = pearsonr(s["measured_log"], s[CACO2_COL[tool]])
            rows.append({"subset": label, "tool": C.TOOL_LABELS[tool], "n": int(len(s)),
                         "pearson_r": float(r), "pearson_p": float(p)})
    out = pd.DataFrame(rows)
    for tool in C.TOOL_LABELS.values():
        a = out[(out["subset"] == "tabulated in a source publication") & (out["tool"] == tool)]
        b = out[(out["subset"] == "not tabulated there") & (out["tool"] == tool)]
        if len(a) and len(b):
            se = np.sqrt(1 / (int(a["n"].iloc[0]) - 3) + 1 / (int(b["n"].iloc[0]) - 3))
            z = np.arctanh(float(a["pearson_r"].iloc[0])) - np.arctanh(float(b["pearson_r"].iloc[0]))
            out.loc[out["tool"] == tool, "fisher_z_p_between_halves"] = float(
                2 * norm.sf(abs(z / se)))
    return out


def descriptor_signal() -> pd.DataFrame:
    """Do the measurements move with the descriptors permeability is known to follow?"""
    bench = D.caco2_benchmark().reset_index(drop=True)
    bench = pd.concat([bench, describe(bench["smiles"])], axis=1)
    try:
        flag = _published_flag(bench)
    except D.DataUnavailable:
        flag = pd.Series(pd.NA, index=bench.index, dtype="boolean")

    wang = D.wang_caco2().rename(columns={"measured_logPapp": "measured_log"})
    wang = pd.concat([wang.reset_index(drop=True), describe(wang["smiles"])], axis=1)

    groups = {
        "benchmark (all)": bench,
        "benchmark, in published tables": bench[flag == True],      # noqa: E712
        "benchmark, not in published tables": bench[flag == False],  # noqa: E712
        "drug-like (Wang 2016)": wang,
    }
    rows = []
    for label, sub in groups.items():
        if len(sub) < 10:
            continue
        for desc, expected in EXPECTED_SIGN.items():
            d = sub[["measured_log", desc]].dropna()
            if len(d) < 10:
                continue
            r, p_r = pearsonr(d["measured_log"], d[desc])
            rho, p_rho = spearmanr(d["measured_log"], d[desc])
            observed = "+" if rho > 0 else "-"
            rows.append({"group": label, "descriptor": desc, "n": int(len(d)),
                         "expected_sign": expected, "observed_sign": observed,
                         "pearson_r": float(r), "pearson_p": float(p_r),
                         "spearman": float(rho), "spearman_p": float(p_rho),
                         "has_prior": expected != "0",
                         "matches_expectation": expected == "0" or expected == observed})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def summarize(taut: pd.DataFrame, effect: pd.DataFrame, rep: pd.DataFrame | None,
              het: pd.DataFrame | None, signal: pd.DataFrame | None) -> dict:
    out = {
        "tautomers": {
            "n_compounds": int(len(taut)),
            "n_with_multiple_tautomers": int(taut["tautomerism_relevant"].sum()),
            "median_tautomer_count": float(taut["n_tautomers"].median()),
            "n_canonical_differs": int(taut["canonical_differs_from_submitted"].sum()),
            "min_mannwhitney_p": round(float(effect["mannwhitney_p"].min()), 3),
        },
    }
    if rep is not None:
        r = rep.set_index("descriptor")
        out["representativeness"] = {
            d: {"benchmark_median": round(float(r.loc[d, "benchmark_median"]), 2),
                "library_median": round(float(r.loc[d, "library_median"]), 2),
                "ks_p": float(r.loc[d, "ks_p"]),
                "library_frac_in_range": round(
                    float(r.loc[d, "library_frac_within_benchmark_range"]), 3)}
            for d in ("MolWt", "MolLogP", "TPSA", "FractionCSP3")}
    if het is not None:
        out["subset_heterogeneity"] = {
            f"{row['subset']} | {row['tool']}": {"n": int(row["n"]),
                                                 "r": round(float(row["pearson_r"]), 3),
                                                 "p": round(float(row["pearson_p"]), 3)}
            for _, row in het.iterrows()}
        out["min_fisher_z_p_between_halves"] = round(
            float(het["fisher_z_p_between_halves"].min()), 4)
    if signal is not None:
        real = signal[signal["has_prior"]]
        out["descriptor_signal_matches"] = {
            g: int(sub["matches_expectation"].sum())
            for g, sub in real.groupby("group")}
        out["descriptor_signal_total_with_prior"] = int(
            real.groupby("group").size().max())
    return out


# --------------------------------------------------------------------------
# monolayer integrity (section 3.10)
# --------------------------------------------------------------------------

def _teer(df: pd.DataFrame) -> pd.Series:
    if C.TEER_COL not in df.columns:
        raise D.DataUnavailable(
            f"caco2_experimental.csv has no '{C.TEER_COL}' column; re-download the "
            f"data record, which carries the monolayer-integrity measurement from "
            f"version 2 onwards")
    return pd.to_numeric(df[C.TEER_COL], errors="coerce")


def teer_subset(df: pd.DataFrame, cutoff: float | None) -> pd.DataFrame:
    """Compounds whose monolayer held, at the given threshold on the TEER change."""
    if cutoff is None:
        return df
    t = _teer(df)
    return df[t.notna() & (t >= cutoff)]


def teer_sensitivity(cutoffs=C.TEER_THRESHOLDS) -> pd.DataFrame:
    """Repeat the Caco-2 validation on the compounds surviving each threshold.

    The benchmark is deliberately unfiltered, because its inclusion criteria were
    fixed before any prediction was scored. This table reports what filtering
    would have done: the correlations rise as the threshold tightens, while the
    identity-line R2 and the AUROC do not, which is the distinction section 3.10
    turns on.
    """
    from .metrics import auroc, bootstrap_ci, r2_identity

    df = D.caco2_benchmark()
    _teer(df)  # fail early with a useful message if the column is missing
    rows = []
    for cut in cutoffs:
        sub = teer_subset(df, cut)
        label = "none" if cut is None else f"{cut:.0f}"
        for tool in C.TOOLS:
            d = sub[["measured_log", CACO2_COL[tool]]].dropna()
            if len(d) < C.MIN_N_SUBGROUP:
                continue
            m, p = d["measured_log"].to_numpy(), d[CACO2_COL[tool]].to_numpy()
            r, pv = pearsonr(m, p)
            # Percentile bootstrap, matching the intervals reported for the
            # unfiltered validation, so one interval method is used throughout.
            lo, hi = bootstrap_ci(m, p, lambda a, b: pearsonr(a, b)[0])
            y = (m > C.PERM_THRESHOLD).astype(int)
            rows.append({"cutoff": label, "n_kept": int(len(sub)),
                         "tool": C.TOOL_LABELS[tool], "n": int(len(d)),
                         "pearson_r": float(r),
                         "ci_lo": float(lo),
                         "ci_hi": float(hi),
                         "pearson_p": float(pv),
                         "r2": float(r2_identity(m, p)),
                         "auroc": float(auroc(y, p)) if 0 < y.sum() < len(y) else np.nan,
                         "mae": float(np.mean(np.abs(p - m)))})
    return pd.DataFrame(rows)


def teer_composition(cutoff: float = -20.0) -> pd.DataFrame:
    """What the filter removes, since it is not neutral with respect to chemistry."""
    df = D.caco2_benchmark()
    kept = teer_subset(df, cutoff)
    dropped = df[~df["inchikey14"].isin(kept["inchikey14"])]
    rows = []
    for label, sub in (("kept", kept), ("dropped", dropped)):
        rows.append({"group": label, "cutoff": cutoff, "n": int(len(sub)),
                     "median_mw": float(sub["mw"].median()),
                     "median_measured_log": float(sub["measured_log"].median())})
    return pd.DataFrame(rows)


def teer_descriptor_signal(cutoffs=C.TEER_THRESHOLDS) -> pd.DataFrame:
    """Do the surviving measurements follow the determinants of permeability?"""
    df = D.caco2_benchmark().reset_index(drop=True)
    df = pd.concat([df, describe(df["smiles"])], axis=1)
    rows = []
    for cut in cutoffs:
        sub = teer_subset(df, cut)
        match = sig = total = 0
        for desc, expected in EXPECTED_SIGN.items():
            if expected == "0":
                continue
            total += 1
            d = sub[["measured_log", desc]].dropna()
            if len(d) < 10:
                continue
            rho, p = spearmanr(d["measured_log"], d[desc])
            ok = (expected == "+") == (rho > 0)
            match += ok
            sig += ok and p < 0.05
        rows.append({"cutoff": "none" if cut is None else f"{cut:.0f}",
                     "n": int(len(sub)), "signs_matching": match,
                     "of": total, "significant": sig})
    return pd.DataFrame(rows)


def summarize_teer(sens: pd.DataFrame, comp: pd.DataFrame, sig: pd.DataFrame) -> dict:
    at20 = sens[sens["cutoff"] == "-20"].set_index("tool")
    at10 = sens[sens["cutoff"] == "-10"].set_index("tool")
    c = comp.set_index("group")
    return {
        "n_kept": {row["cutoff"]: int(row["n_kept"])
                   for _, row in sens.drop_duplicates("cutoff").iterrows()},
        "pearson_r_at_-20": {t: round(float(at20.loc[t, "pearson_r"]), 3) for t in at20.index},
        "pearson_r_at_-10": {t: round(float(at10.loc[t, "pearson_r"]), 3) for t in at10.index},
        "min_p_at_-20": round(float(at20["pearson_p"].min()), 3),
        "min_p_at_-10": round(float(at10["pearson_p"].min()), 3),
        "r2_range": [round(float(sens["r2"].min()), 2), round(float(sens["r2"].max()), 2)],
        "auroc_range": [round(float(sens["auroc"].min()), 2),
                        round(float(sens["auroc"].max()), 2)],
        "composition_at_-20": {g: {"n": int(c.loc[g, "n"]),
                                   "median_mw": round(float(c.loc[g, "median_mw"]), 0),
                                   "median_measured_log": round(
                                       float(c.loc[g, "median_measured_log"]), 2)}
                               for g in c.index},
        "descriptor_signs": {row["cutoff"]: int(row["signs_matching"])
                             for _, row in sig.iterrows()},
    }
