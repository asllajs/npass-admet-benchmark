"""Why the tools fail: distance from the drug-like training domain (section 3.5).

The reference chemistry is the Wang et al. (2016) drug-like Caco-2 benchmark,
which is the source of the Therapeutics Data Commons ``Caco2_Wang`` task that
ADMET-AI's Caco-2 head is trained on and is representative of the assay chemistry
behind the other two tools. For every benchmark natural product we take the
nearest-neighbour ECFP4 Tanimoto similarity to the Wang *training* partition, and
contrast it with the held-out Wang test compounds.

This module also holds the two controls reported alongside the main result: the
in-domain positive control (ADMET-AI on the Wang test set) and the molecular
weight confound check for the uniform-mass dosing of the Caco-2 assay.

Section 3.5 of the revision asks a sharper question than the aggregate gap. Most
of the benchmark is in-domain by the conventional 0.30 cutoff, so if the domain
argument works as usually assumed, those compounds should be predicted better
than the rest. ``domain_split_table`` compares them, ``threshold_sensitivity``
repeats the comparison at five cutoffs so that nothing rests on a convention, and
``error_quintiles`` replaces the single error-versus-distance correlation with
binned means, which is where the absence of any monotone relationship shows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from . import config as C
from . import datasets as D
from . import plots
from .caco2 import CACO2_COL
from .metrics import r2_identity

RDLogger.DisableLog("rdApp.*")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    return _MORGAN.GetFingerprint(mol) if mol is not None else None


def nearest_neighbour_tanimoto(query_fps: list, reference_fps: list) -> np.ndarray:
    out = np.full(len(query_fps), np.nan)
    for i, q in enumerate(query_fps):
        if q is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q, reference_fps)
        if sims:
            out[i] = max(sims)
    return out


def _wang_split_fingerprints() -> tuple[list, pd.DataFrame]:
    wang = D.wang_caco2()
    train = wang[wang["split"] == "Tr"]
    test = wang[wang["split"] == "Te"].copy()
    train_fps = [f for f in (fingerprint(s) for s in train["smiles"]) if f is not None]
    return train_fps, test


def domain_distance() -> pd.DataFrame:
    """Benchmark natural products with their distance to the training domain."""
    train_fps, _ = _wang_split_fingerprints()
    df = D.caco2_benchmark()
    df = df[df["smiles"].notna()].copy()
    df["nn_tanimoto"] = nearest_neighbour_tanimoto(
        [fingerprint(s) for s in df["smiles"]], train_fps)
    return df


def domain_gap() -> pd.DataFrame:
    """Natural products against held-out drug-like compounds, same reference set."""
    train_fps, test = _wang_split_fingerprints()
    np_nn = domain_distance()["nn_tanimoto"].dropna().to_numpy()
    test_nn = nearest_neighbour_tanimoto([fingerprint(s) for s in test["smiles"]], train_fps)
    test_nn = test_nn[~np.isnan(test_nn)]
    _, p = mannwhitneyu(np_nn, test_nn, alternative="less")
    return pd.DataFrame([
        {"group": "natural products", "n": int(len(np_nn)),
         "median_nn_tanimoto": float(np.median(np_nn)),
         "frac_below_cutoff": float(np.mean(np_nn < C.AD_CUTOFF)),
         "mannwhitney_p": float(p)},
        {"group": "drug-like held out", "n": int(len(test_nn)),
         "median_nn_tanimoto": float(np.median(test_nn)),
         "frac_below_cutoff": float(np.mean(test_nn < C.AD_CUTOFF)),
         "mannwhitney_p": float(p)},
    ])


def chemspace_overlap(n_sample: int = 20000) -> pd.DataFrame:
    """Distance to the drug-like training set for the library, the benchmark and the control.

    Section 2.4 makes the point that the benchmark sits closer to the tools'
    training chemistry than the library it stands for, so the null result is
    conservative. That comparison needs the library as well as the benchmark,
    which is why it lives here rather than in ``domain_gap``.
    """
    train_fps, test = _wang_split_fingerprints()
    test_fps = [fingerprint(s) for s in test["smiles"]]

    bench = domain_distance()["nn_tanimoto"].dropna().to_numpy()
    test_nn = nearest_neighbour_tanimoto(test_fps, train_fps)
    test_nn = test_nn[~np.isnan(test_nn)]

    cols = D.library_columns()
    smi_col = next((c for c in ("smiles", "canonical_smiles", "std_smiles") if c in cols), None)
    if smi_col is None:
        raise D.DataUnavailable("the library table has no SMILES column")
    lib_smiles = D.library(columns=(smi_col,))[smi_col].dropna()
    lib_smiles = lib_smiles.sample(n=min(n_sample, len(lib_smiles)), random_state=C.SEED)
    lib_nn = nearest_neighbour_tanimoto([fingerprint(s) for s in lib_smiles], train_fps)
    lib_nn = lib_nn[~np.isnan(lib_nn)]

    rows = []
    for label, nn in (("NPASS 3.0 library", lib_nn),
                      ("benchmark (measured)", bench),
                      ("drug-like held out", test_nn)):
        rows.append({"population": label, "n": int(len(nn)),
                     "median_nn_tanimoto": float(np.median(nn)),
                     "q1": float(np.percentile(nn, 25)),
                     "q3": float(np.percentile(nn, 75)),
                     "frac_below_0.30": float(np.mean(nn < 0.30)),
                     "frac_below_0.40": float(np.mean(nn < 0.40))})
    return pd.DataFrame(rows)


def error_vs_distance() -> pd.DataFrame:
    """Within the natural products, does the distance predict which ones are wrong?"""
    df = domain_distance()
    rows = []
    for tool in C.TOOLS:
        col = CACO2_COL[tool]
        d = df[[col, "measured_log", "nn_tanimoto"]].dropna()
        err = (d[col] - d["measured_log"]).abs()
        rho, p = spearmanr(err, d["nn_tanimoto"])
        q1, q2 = np.quantile(d["nn_tanimoto"], [1 / 3, 2 / 3])
        rows.append({"tool": C.TOOL_LABELS[tool], "n": int(len(d)),
                     "spearman_abserr_vs_nn": float(rho), "spearman_p": float(p),
                     "mae_low_similarity_tertile": float(err[d["nn_tanimoto"] <= q1].mean()),
                     "mae_high_similarity_tertile": float(err[d["nn_tanimoto"] >= q2].mean())})
    return pd.DataFrame(rows)


def _in_domain(df: pd.DataFrame, cutoff: float) -> pd.Series:
    return df["nn_tanimoto"] >= cutoff


def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> float:
    """Two-sided p-value for the difference between two independent correlations."""
    from scipy.stats import norm
    if min(n1, n2) < 4 or not (abs(r1) < 1 and abs(r2) < 1):
        return np.nan
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(2 * norm.sf(abs(np.arctanh(r1) - np.arctanh(r2)) / se))


def domain_split_table(cutoff: float = C.AD_CUTOFF,
                       n_boot: int = C.N_BOOTSTRAP) -> pd.DataFrame:
    """Full validation metrics computed separately inside and outside the domain.

    If distance from the training set governed accuracy, the in-domain group
    would be predicted better. The Fisher z column tests that difference rather
    than leaving it to be eyeballed.
    """
    from .metrics import binary_metrics, continuous_metrics

    df = domain_distance()
    groups = {f"in-domain (NN >= {cutoff:.2f})": _in_domain(df, cutoff),
              f"out-of-domain (NN < {cutoff:.2f})": ~_in_domain(df, cutoff)}
    rows = []
    for label, mask in groups.items():
        sub = df[mask]
        for tool in C.TOOLS:
            d = sub[["measured_log", CACO2_COL[tool]]].dropna()
            if len(d) < C.MIN_N_SUBGROUP:
                rows.append({"group": label, "tool": C.TOOL_LABELS[tool],
                             "n": int(len(d)), "note": f"n < {C.MIN_N_SUBGROUP}"})
                continue
            m, p = d["measured_log"].to_numpy(), d[CACO2_COL[tool]].to_numpy()
            rows.append({"group": label, "tool": C.TOOL_LABELS[tool],
                         **continuous_metrics(m, p, n_boot=n_boot),
                         **binary_metrics(m, p)})
    out = pd.DataFrame(rows)

    for tool in C.TOOL_LABELS.values():
        rows_t = out[out["tool"] == tool]
        if len(rows_t) == 2 and rows_t["pearson_r"].notna().all():
            a, b = rows_t.iloc[0], rows_t.iloc[1]
            out.loc[out["tool"] == tool, "fisher_z_p"] = fisher_z_test(
                float(a["pearson_r"]), int(a["n"]), float(b["pearson_r"]), int(b["n"]))
    return out


def threshold_sensitivity(cutoffs: tuple[float, ...] = C.AD_THRESHOLDS) -> pd.DataFrame:
    """Repeat the split at several cutoffs, so no conclusion rests on 0.30 alone."""
    df = domain_distance()
    rows = []
    for cut in cutoffs:
        inside = _in_domain(df, cut)
        for tool in C.TOOLS:
            col = CACO2_COL[tool]
            rec = {"cutoff": cut, "tool": C.TOOL_LABELS[tool],
                   "n_in": int((inside & df[col].notna()).sum()),
                   "n_out": int((~inside & df[col].notna()).sum())}
            for label, mask in (("in", inside), ("out", ~inside)):
                d = df[mask][["measured_log", col]].dropna()
                if len(d) < C.MIN_N_SUBGROUP:
                    rec[f"r_{label}"] = np.nan
                    rec[f"p_{label}"] = np.nan
                    rec[f"mae_{label}"] = np.nan
                    continue
                m, p = d["measured_log"].to_numpy(), d[col].to_numpy()
                r, pv = pearsonr(m, p)
                rec[f"r_{label}"] = float(r)
                rec[f"p_{label}"] = float(pv)
                rec[f"mae_{label}"] = float(np.mean(np.abs(p - m)))
            rec["fisher_z_p"] = fisher_z_test(rec["r_in"], rec["n_in"],
                                              rec["r_out"], rec["n_out"])
            rows.append(rec)
    return pd.DataFrame(rows)


def error_quintiles(n_bins: int = 5) -> pd.DataFrame:
    """Mean absolute error per quintile of distance, with a bootstrap interval.

    A single correlation coefficient can hide a non-monotone relationship; the
    binned means are what show that the most similar compounds are not the best
    predicted.
    """
    df = domain_distance()
    rng = np.random.default_rng(C.SEED)
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    rows = []
    for tool in C.TOOLS:
        col = CACO2_COL[tool]
        d = df[["measured_log", col, "nn_tanimoto"]].dropna().copy()
        d["err"] = (d[col] - d["measured_log"]).abs()
        d["quantile"] = pd.qcut(d["nn_tanimoto"], n_bins, labels=labels)
        rho, p = spearmanr(d["err"], d["nn_tanimoto"])
        for q, sub in d.groupby("quantile", observed=True):
            boot = rng.choice(sub["err"].to_numpy(), (2000, len(sub)), replace=True).mean(axis=1)
            rows.append({"tool": C.TOOL_LABELS[tool], "quantile": str(q), "n": int(len(sub)),
                         "nn_min": float(sub["nn_tanimoto"].min()),
                         "nn_max": float(sub["nn_tanimoto"].max()),
                         "mae": float(sub["err"].mean()),
                         "mae_ci_lo": float(np.percentile(boot, 2.5)),
                         "mae_ci_hi": float(np.percentile(boot, 97.5)),
                         "spearman_err_vs_nn": float(rho), "spearman_p": float(p)})
    return pd.DataFrame(rows)


def domain_by_pathway(cutoff: float = C.AD_CUTOFF) -> pd.DataFrame:
    """How far each biosynthetic pathway sits from the domain, and how it is predicted."""
    df = domain_distance().copy()
    df["pathway"] = D.primary(df["npc_pathway"])
    rows = []
    for path, sub in df.groupby("pathway"):
        if len(sub) < 5:
            continue
        rec = {"pathway": path, "n": int(len(sub)),
               "median_nn": float(sub["nn_tanimoto"].median()),
               "frac_out_of_domain": float((sub["nn_tanimoto"] < cutoff).mean())}
        for tool in C.TOOLS:
            s = sub[["measured_log", CACO2_COL[tool]]].dropna()
            rec[f"mae_{tool}"] = (float((s[CACO2_COL[tool]] - s["measured_log"]).abs().mean())
                                  if len(s) >= 3 else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("median_nn", ascending=False).reset_index(drop=True)


def summarize_split(split: pd.DataFrame, sens: pd.DataFrame,
                    quint: pd.DataFrame, paths: pd.DataFrame) -> dict:
    inside = split[split["group"].str.startswith("in-domain")]
    outside = split[split["group"].str.startswith("out-of-domain")]
    return {
        "cutoff": C.AD_CUTOFF,
        "n_in_domain": int(inside["n"].max()),
        "n_out_of_domain": int(outside["n"].max()),
        "pearson_r_in": {r.tool: round(float(r.pearson_r), 3) for r in inside.itertuples()},
        "pearson_r_out": {r.tool: round(float(r.pearson_r), 3) for r in outside.itertuples()},
        "min_fisher_z_p": round(float(split["fisher_z_p"].min()), 3),
        "max_r_in_any_cutoff": round(float(sens["r_in"].max()), 3),
        "n_significant_r_in_across_cutoffs": int((sens["p_in"] < 0.05).sum()),
        "max_abs_spearman_err_vs_nn": round(float(quint["spearman_err_vs_nn"].abs().max()), 3),
        "min_p_err_vs_nn": round(float(quint["spearman_p"].min()), 3),
        "mae_by_quantile": {t: [round(float(v), 3) for v in
                                quint[quint["tool"] == t].sort_values("quantile")["mae"]]
                            for t in quint["tool"].unique()},
        "pathway_median_nn": {r.pathway: round(float(r.median_nn), 2)
                              for r in paths.itertuples()},
    }


def figure_split(split: pd.DataFrame, quint: pd.DataFrame) -> str:
    """Accuracy by domain membership, and error against distance in bins."""
    fig, (ax, ax2) = plots.figure(width=10.0, height=4.2, ncols=2)

    width = 0.26
    for k, tool in enumerate(C.TOOLS):
        label = C.TOOL_LABELS[tool]
        vals, errs = [], []
        for prefix in ("in-domain", "out-of-domain"):
            row = split[(split["tool"] == label) & split["group"].str.startswith(prefix)]
            if len(row) and pd.notna(row["pearson_r"].iloc[0]):
                vals.append(float(row["pearson_r"].iloc[0]))
                errs.append(1.96 / np.sqrt(max(int(row["n"].iloc[0]) - 3, 1)))
            else:
                vals.append(np.nan)
                errs.append(0.0)
        ax.bar(np.arange(2) + (k - 1) * width, vals, width, yerr=errs, capsize=3,
               color=plots.TOOL_COLORS[tool], label=label)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(range(2))
    ax.set_xticklabels([f"in-domain\n(NN >= {C.AD_CUTOFF:.2f})",
                        f"out-of-domain\n(NN < {C.AD_CUTOFF:.2f})"])
    ax.set_ylabel("Pearson r with measured log Papp")
    ax.legend(fontsize=8)

    for tool in C.TOOLS:
        q = quint[quint["tool"] == C.TOOL_LABELS[tool]].sort_values("quantile")
        x = np.arange(len(q))
        ax2.errorbar(x, q["mae"],
                     yerr=[q["mae"] - q["mae_ci_lo"], q["mae_ci_hi"] - q["mae"]],
                     marker="o", capsize=3, color=plots.TOOL_COLORS[tool],
                     label=C.TOOL_LABELS[tool])
    ax2.set_xticks(range(len(q)))
    ax2.set_xticklabels(sorted(quint["quantile"].unique()))
    ax2.set(xlabel="Quintile of nearest-neighbour similarity (low to high)",
            ylabel="Mean absolute error (log units)")
    ax2.legend(fontsize=8)
    return plots.save(fig, "fig_domain_split")


def wang_positive_control() -> pd.DataFrame:
    """ADMET-AI on the drug-like benchmark its Caco-2 head derives from."""
    pred = D.wang_admetai_predictions()
    rows = []
    for label, mask in [("test", pred["Dataset"].astype(str).str.startswith("Te")),
                        ("all", pred["Dataset"].notna())]:
        d = pred[mask][["measured_logPapp", "ai_caco2"]].dropna()
        y, yhat = d["measured_logPapp"].to_numpy(), d["ai_caco2"].to_numpy()
        r, p = pearsonr(y, yhat)
        rows.append({"subset": label, "n": int(len(d)), "pearson_r": float(r),
                     "pearson_p": float(p), "r2": float(r2_identity(y, yhat)),
                     "rmse": float(np.sqrt(np.mean((y - yhat) ** 2)))})
    return pd.DataFrame(rows)


def molecular_weight_confound() -> pd.DataFrame:
    """The assay dosed a uniform mass, so molar concentration falls with size.

    If the near-zero correlation were an artefact of that dosing rather than of
    domain mismatch, the tools' error would track molecular weight and the
    measured permeability would not.
    """
    df = D.caco2_benchmark()
    rows = []
    d0 = df[["measured_log", "mw"]].dropna()
    rho, p = spearmanr(d0["measured_log"], d0["mw"])
    rows.append({"quantity": "measured log Papp vs MW", "tool": "-", "n": int(len(d0)),
                 "spearman": float(rho), "spearman_p": float(p)})
    for tool in C.TOOLS:
        col = CACO2_COL[tool]
        d = df[[col, "measured_log", "mw"]].dropna()
        err = (d[col] - d["measured_log"]).abs()
        rho, p = spearmanr(err, d["mw"])
        rows.append({"quantity": "|error| vs MW", "tool": C.TOOL_LABELS[tool],
                     "n": int(len(d)), "spearman": float(rho), "spearman_p": float(p)})
    return pd.DataFrame(rows)


def summarize(gap: pd.DataFrame, err: pd.DataFrame, confound: pd.DataFrame,
              positive_control: pd.DataFrame | None) -> dict:
    g = gap.set_index("group")
    out = {
        "median_nn_tanimoto": {k: round(float(v), 3)
                               for k, v in g["median_nn_tanimoto"].items()},
        "frac_below_cutoff": {k: round(float(v), 3)
                              for k, v in g["frac_below_cutoff"].items()},
        "mannwhitney_p": float(g["mannwhitney_p"].iloc[0]),
        "spearman_abserr_vs_nn": {r.tool: round(float(r.spearman_abserr_vs_nn), 3)
                                  for r in err.itertuples()},
        "min_spearman_p_abserr_vs_nn": round(float(err["spearman_p"].min()), 3),
        "mw_confound": {r.tool: {"spearman": round(float(r.spearman), 3),
                                 "p": float(r.spearman_p)}
                        for r in confound.itertuples()},
    }
    if positive_control is not None:
        pc = positive_control.set_index("subset")
        out["wang_positive_control"] = {
            s: {"n": int(pc.loc[s, "n"]), "pearson_r": round(float(pc.loc[s, "pearson_r"]), 3),
                "r2": round(float(pc.loc[s, "r2"]), 3)} for s in pc.index}
    return out


def figure(gap: pd.DataFrame) -> str:
    train_fps, test = _wang_split_fingerprints()
    df = domain_distance()
    np_nn = df["nn_tanimoto"].dropna().to_numpy()
    test_nn = nearest_neighbour_tanimoto([fingerprint(s) for s in test["smiles"]], train_fps)
    test_nn = test_nn[~np.isnan(test_nn)]

    fig, (ax, ax2) = plots.figure(width=10.0, height=4.5, ncols=2)
    bins = np.linspace(0, 1, 26)
    ax.hist(test_nn, bins=bins, density=True, color="0.75", edgecolor="0.3",
            label=f"drug-like held out (n = {len(test_nn)})")
    ax.hist(np_nn, bins=bins, density=True, histtype="step", color="tab:blue", lw=1.6,
            label=f"natural products (n = {len(np_nn)})")
    ax.axvline(C.AD_CUTOFF, ls="--", color="0.35")
    ax.set(xlabel="Nearest-neighbour Tanimoto to drug-like Caco-2 training set",
           ylabel="Density")
    ax.legend(fontsize=8)

    for tool in C.TOOLS:
        col = CACO2_COL[tool]
        d = df[[col, "measured_log", "nn_tanimoto"]].dropna()
        ax2.scatter(d["nn_tanimoto"], (d[col] - d["measured_log"]).abs(),
                    s=14, alpha=0.6, color=plots.TOOL_COLORS[tool], label=C.TOOL_LABELS[tool])
    ax2.axvline(C.AD_CUTOFF, ls="--", color="0.35")
    ax2.set(xlabel="Nearest-neighbour Tanimoto to training set",
            ylabel="|predicted - measured| (log units)")
    ax2.legend(fontsize=8)
    return plots.save(fig, "fig_applicability_domain")


def figure_confound() -> str:
    df = D.caco2_benchmark()
    fig, axes = plots.figure(width=11.0, height=3.6, ncols=3)
    for ax, tool in zip(axes, C.TOOLS):
        col = CACO2_COL[tool]
        d = df[[col, "measured_log", "mw"]].dropna()
        err = (d[col] - d["measured_log"]).abs()
        ax.scatter(d["mw"], err, s=14, alpha=0.6, color=plots.TOOL_COLORS[tool])
        rho, p = spearmanr(err, d["mw"])
        ax.set(xlabel="Molecular weight (Da)", title=C.TOOL_LABELS[tool],
               ylabel="|predicted - measured| (log units)" if tool == C.TOOLS[0] else "")
        plots.annotate(ax, f"rho = {rho:+.2f}\np = {p:.1e}")
    return plots.save(fig, "fig_mw_confound")
