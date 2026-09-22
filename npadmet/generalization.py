"""Where the generalization fails, and what a split protocol is worth (sections 3.6, 3.7).

Two experiments, both built on one deliberately plain random forest that is held
fixed throughout, so that the only thing varying between cells is which compounds
the model was allowed to see.

**Section 3.6 - leaving the training chemistry.** A model trained on the drug-like
Caco-2 reference set is evaluated under a random split, a strict Bemis-Murcko
scaffold-disjoint split, and transfer to the natural-product benchmark. If the
failure on natural products were a matter of unfamiliar scaffolds, the scaffold
split would already show it. It does not: the scaffold split costs relatively
little and the transfer costs everything, which places the failure at the domain
boundary rather than at scaffold novelty. The reciprocal direction is run as well.

**Section 3.7 - tightening the split on one fixed panel.** The natural-product
permeability model cited in the submitted version as a matched-domain ceiling was
evaluated by cross-validation on a panel of 83 compounds measured in triplicate,
with each replicate a separate row. Replicates of one compound are structurally
identical and numerically near-identical, so a split drawn over rows places
near-duplicates of every test compound in the training set. ``replicate_protocols``
evaluates the same model under row-level, compound-level and compound-level
scaffold folds, and ``replicate_spread`` reports the variance decomposition that
caps what a row-level split can reach.

The random forest is a control, not a proposed predictor. Nothing here is tuned.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr

from . import config as C
from . import datasets as D
from . import plots
from .metrics import r2_identity

RDLogger.DisableLog("rdApp.*")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# A short, standard descriptor block. Kept small on purpose: the point is what a
# plain model does, not how well a tuned one could do.
DESCRIPTORS = (
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("MolMR", Descriptors.MolMR),
)

DRUG_LIKE = "drug-like (Wang 2016)"
NATURAL = "natural products (this study)"


def _model():
    """The one model used everywhere in this module."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=C.N_TREES, min_samples_leaf=C.MIN_LEAF,
                                 n_jobs=-1, random_state=C.SEED)


# --------------------------------------------------------------------------
# features and scaffolds
# --------------------------------------------------------------------------

def featurize(smiles, kind: str = "morgan") -> tuple[np.ndarray, np.ndarray]:
    """Return a feature matrix and a mask of the rows whose SMILES parsed."""
    fps, descs, ok = [], [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        ok.append(mol is not None)
        if mol is None:
            continue
        arr = np.zeros(2048, dtype=np.float64)
        for bit in _MORGAN.GetFingerprint(mol).GetOnBits():
            arr[bit] = 1.0
        fps.append(arr)
        descs.append([fn(mol) for _, fn in DESCRIPTORS])
    fp = np.array(fps) if fps else np.zeros((0, 2048))
    desc = np.array(descs, dtype=float) if descs else np.zeros((0, len(DESCRIPTORS)))
    mask = np.array(ok)
    if kind == "morgan":
        return fp, mask
    if kind == "descriptor":
        return desc, mask
    return np.hstack([fp, desc]), mask


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold as canonical SMILES; acyclic molecules get their own key."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        smi = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""
    return smi if smi else f"__acyclic__{Chem.MolToSmiles(mol)}"


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------

def random_folds(n: int, k: int = C.N_FOLDS, seed: int = C.SEED) -> list[np.ndarray]:
    return [np.sort(f) for f in np.array_split(np.random.default_rng(seed).permutation(n), k)]


def grouped_folds(groups, k: int = C.N_FOLDS, seed: int = C.SEED) -> list[np.ndarray]:
    """Folds that never split a group across the train/test boundary.

    Groups are assigned largest-first to whichever fold is currently smallest,
    the usual greedy balancing for scaffold splits. It keeps the folds close to
    equal size without ever dividing a group.

    Ties between equally sized groups are broken by a seeded draw rather than by
    dictionary insertion order, so the folds do not depend on the order the
    caller happened to pass the groups in. The draw is assigned after the
    size sort, which is what makes the result reproducible across callers.
    """
    buckets: dict = defaultdict(list)
    for i, g in enumerate(groups):
        buckets[g].append(i)
    by_size = sorted(buckets.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(len(by_size))
    order = [g for _, g in sorted(zip(tiebreak, by_size),
                                  key=lambda kv: (-len(kv[1]), kv[0]))]
    folds: list[list[int]] = [[] for _ in range(k)]
    for grp in order:
        folds[min(range(k), key=lambda j: len(folds[j]))].extend(grp)
    return [np.sort(np.array(f, dtype=int)) for f in folds]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def _score(y: np.ndarray, yhat: np.ndarray, binary: bool = True) -> dict:
    r, p = pearsonr(y, yhat)
    rho, _ = spearmanr(y, yhat)
    out = {"n": int(len(y)), "pearson_r": float(r), "pearson_p": float(p),
           "spearman": float(rho), "r2_identity": float(r2_identity(y, yhat)),
           "rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
           "mae": float(np.mean(np.abs(yhat - y)))}
    if binary:
        from .metrics import auroc
        label = (y > C.PERM_THRESHOLD).astype(int)
        out["auroc"] = (float(auroc(label, yhat))
                        if 0 < label.sum() < len(label) else float("nan"))
    return out


def cross_validate(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray],
                   binary: bool = True) -> dict:
    """Out-of-fold predictions pooled across folds, then scored once."""
    pred = np.full(len(y), np.nan)
    for fold in folds:
        train = np.setdiff1d(np.arange(len(y)), fold)
        if len(train) < 10 or len(fold) == 0:
            continue
        pred[fold] = _model().fit(X[train], y[train]).predict(X[fold])
    ok = ~np.isnan(pred)
    return _score(y[ok], pred[ok], binary=binary)


def transfer(X_tr, y_tr, X_te, y_te) -> dict:
    """Train on one chemistry in full, predict the other."""
    return _score(y_te, _model().fit(X_tr, y_tr).predict(X_te))


# --------------------------------------------------------------------------
# datasets in the shape this module wants
# --------------------------------------------------------------------------

def _wang_frame() -> pd.DataFrame:
    w = D.wang_caco2().copy()
    return w.rename(columns={"measured_logPapp": "measured_log"})[["smiles", "measured_log"]]


def _benchmark_frame() -> pd.DataFrame:
    b = D.caco2_benchmark()
    return b.loc[b["smiles"].notna(), ["smiles", "measured_log"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# section 3.6
# --------------------------------------------------------------------------

def generalization_table(kinds: tuple[str, ...] = ("morgan", "descriptor")) -> pd.DataFrame:
    """One fixed model, three regimes, two training chemistries."""
    rows = []
    for kind in kinds:
        wang, nps = _wang_frame(), _benchmark_frame()
        Xw, okw = featurize(wang["smiles"], kind)
        wang = wang[okw].reset_index(drop=True)
        Xn, okn = featurize(nps["smiles"], kind)
        nps = nps[okn].reset_index(drop=True)
        yw, yn = wang["measured_log"].to_numpy(), nps["measured_log"].to_numpy()
        scaf_w = [murcko_scaffold(s) for s in wang["smiles"]]
        scaf_n = [murcko_scaffold(s) for s in nps["smiles"]]

        for train_set, X, y, scaf, Xo, yo, other in (
                (DRUG_LIKE, Xw, yw, scaf_w, Xn, yn, "natural products"),
                (NATURAL, Xn, yn, scaf_n, Xw, yw, "drug-like")):
            rows.append({"features": kind, "train_set": train_set, "regime": "random CV",
                         **cross_validate(X, y, random_folds(len(y)))})
            rows.append({"features": kind, "train_set": train_set, "regime": "scaffold CV",
                         **cross_validate(X, y, grouped_folds(scaf))})
            rows.append({"features": kind, "train_set": train_set,
                         "regime": f"transfer to {other}", **transfer(X, y, Xo, yo)})
    return pd.DataFrame(rows)


def scaffold_counts() -> dict:
    """How the two chemistries relate at the scaffold level."""
    wang, nps = _wang_frame(), _benchmark_frame()
    sw = [murcko_scaffold(s) for s in wang["smiles"]]
    sn = [murcko_scaffold(s) for s in nps["smiles"]]
    set_w = set(sw)
    return {"drug_like_n": len(sw), "drug_like_scaffolds": len(set_w),
            "benchmark_n": len(sn), "benchmark_scaffolds": len(set(sn)),
            "shared_scaffolds": len(set_w & set(sn)),
            "benchmark_on_a_drug_like_scaffold": int(sum(s in set_w for s in sn))}


def learning_curve(sizes=(50, 100, 134, 200, 400, 800), n_repeat: int = 10) -> pd.DataFrame:
    """What the same model reaches on drug-like data at natural-product sample sizes.

    This separates "too few compounds" from "no learnable signal": if a model
    trained on 134 drug-like compounds does well, then 134 is not the problem.
    """
    wang = _wang_frame()
    X, mask = featurize(wang["smiles"], "morgan")
    y = wang[mask]["measured_log"].to_numpy()
    rng = np.random.default_rng(C.SEED)
    rows = []
    for size in sizes:
        size = min(size, len(y) - 100)
        scores = []
        for _ in range(n_repeat):
            idx = rng.permutation(len(y))
            tr, te = idx[:size], idx[size:size + 200]
            pred = _model().fit(X[tr], y[tr]).predict(X[te])
            scores.append((pearsonr(y[te], pred)[0], r2_identity(y[te], pred)))
        arr = np.array(scores)
        rows.append({"train_size": int(size), "n_repeat": n_repeat,
                     "pearson_r_mean": float(arr[:, 0].mean()),
                     "pearson_r_sd": float(arr[:, 0].std()),
                     "r2_mean": float(arr[:, 1].mean()),
                     "r2_sd": float(arr[:, 1].std())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# section 3.7
# --------------------------------------------------------------------------

def replicate_spread() -> pd.DataFrame:
    """Variance decomposition of the replicate-level panel.

    ``max_achievable_r2_row_level`` is the ceiling a structure-based model can
    reach when folds are drawn over rows: within-compound variance is not a
    function of structure, so it is the only part such a model cannot explain,
    and a row-level split lets it be memorized instead.
    """
    rows = []
    for scale in ("log10 (zeros excluded)", "raw papp (zeros retained)"):
        kim = D.kim2026_replicates()
        kim = kim[kim["smiles"].notna()].copy()
        if scale.startswith("log10"):
            kim = kim[kim["papp"] > 0].copy()
            kim["value"] = np.log10(kim["papp"])
        else:
            kim["value"] = kim["papp"]
        per = kim.groupby("key")["value"].agg(["count", "mean", "std"])
        per = per[per["count"] >= 2].dropna(subset=["std"])
        within = float((per["std"] ** 2).mean())
        between = float(per["mean"].var(ddof=1))
        rows.append({
            "scale": scale,
            "n_rows": int(len(kim)), "n_compounds": int(kim["key"].nunique()),
            "median_replicates": float(per["count"].median()),
            "mean_within_compound_sd": float(per["std"].mean()),
            "between_compound_sd": float(np.sqrt(between)),
            "max_achievable_r2_row_level": float(1 - within / (within + between)),
        })
    return pd.DataFrame(rows)


def replicate_zero_fraction() -> pd.DataFrame:
    """How much of the panel is recorded at exactly zero permeability."""
    kim = D.kim2026_replicates()
    per = kim.groupby("key")["papp"].mean()
    return pd.DataFrame([{
        "n_rows": int(len(kim)), "n_compounds": int(kim["key"].nunique()),
        "n_rows_zero": int((kim["papp"] == 0).sum()),
        "frac_rows_zero": float((kim["papp"] == 0).mean()),
        "n_compounds_all_zero": int((per == 0).sum()),
    }])


def replicate_protocols() -> pd.DataFrame:
    """The same model on the same panel, under five split protocols on two scales.

    The gap between a row-level and a compound-level split is the leakage; the gap
    between a compound-level random and a compound-level scaffold split is the
    scaffold-generalization cost. Both scales are reported because the source study
    states its accuracy on the raw permeability scale, which retains the rows
    recorded as exactly zero, while this paper works on the log scale, which cannot.
    """
    raw = D.kim2026_replicates()
    raw = raw[raw["smiles"].notna()].reset_index(drop=True)
    Xr, mask_r = featurize(raw["smiles"], "both")
    raw = raw[mask_r].reset_index(drop=True)
    yr = raw["papp"].to_numpy()

    log = D.kim2026_replicates()
    log = log[(log["papp"] > 0) & log["smiles"].notna()].reset_index(drop=True)
    log["measured_log"] = np.log10(log["papp"])
    Xl, mask_l = featurize(log["smiles"], "both")
    log = log[mask_l].reset_index(drop=True)
    yl = log["measured_log"].to_numpy()
    scaffolds = [murcko_scaffold(s) for s in log["smiles"]]

    rows = [
        {"split": "raw scale, row-level random CV",
         "n_compounds": int(raw["key"].nunique()),
         **cross_validate(Xr, yr, random_folds(len(yr)), binary=False)},
        {"split": "raw scale, compound-level random CV",
         "n_compounds": int(raw["key"].nunique()),
         **cross_validate(Xr, yr, grouped_folds(raw["key"].tolist()), binary=False)},
        {"split": "row-level random CV",
         "n_compounds": int(log["key"].nunique()),
         **cross_validate(Xl, yl, random_folds(len(yl)))},
        {"split": "compound-level random CV",
         "n_compounds": int(log["key"].nunique()),
         **cross_validate(Xl, yl, grouped_folds(log["key"].tolist()))},
        {"split": "compound-level scaffold CV",
         "n_compounds": int(log["key"].nunique()),
         **cross_validate(Xl, yl, grouped_folds(scaffolds))},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def summarize(table: pd.DataFrame, counts: dict,
              protocols: pd.DataFrame | None, spread: pd.DataFrame | None) -> dict:
    m = table[table["features"] == "morgan"].set_index(["train_set", "regime"])
    out = {
        "scaffold_counts": counts,
        "drug_like": {
            "random_cv": {"r": round(float(m.loc[(DRUG_LIKE, "random CV"), "pearson_r"]), 3),
                          "r2": round(float(m.loc[(DRUG_LIKE, "random CV"), "r2_identity"]), 3)},
            "scaffold_cv": {"r": round(float(m.loc[(DRUG_LIKE, "scaffold CV"), "pearson_r"]), 3),
                            "r2": round(float(m.loc[(DRUG_LIKE, "scaffold CV"), "r2_identity"]), 3)},
            "transfer": {"r": round(float(m.loc[(DRUG_LIKE, "transfer to natural products"),
                                                "pearson_r"]), 3),
                         "r2": round(float(m.loc[(DRUG_LIKE, "transfer to natural products"),
                                                 "r2_identity"]), 2)},
        },
        "natural_products": {
            "random_cv_r2": round(float(m.loc[(NATURAL, "random CV"), "r2_identity"]), 3),
            "scaffold_cv_r2": round(float(m.loc[(NATURAL, "scaffold CV"), "r2_identity"]), 3),
        },
    }
    if protocols is not None:
        p = protocols.set_index("split")
        out["replicate_protocols"] = {
            s: {"r": round(float(p.loc[s, "pearson_r"]), 3),
                "r2": round(float(p.loc[s, "r2_identity"]), 3)} for s in p.index}
    if spread is not None:
        by_scale = spread.set_index("scale")
        out["replicate_spread"] = {
            scale: {
                "within_sd": float(row["mean_within_compound_sd"]),
                "between_sd": float(row["between_compound_sd"]),
                "max_achievable_r2_row_level": round(float(row["max_achievable_r2_row_level"]), 3)}
            for scale, row in by_scale.iterrows()}
    return out


def figure(table: pd.DataFrame, protocols: pd.DataFrame | None) -> str:
    """(a) leaving the training chemistry; (b) tightening the split."""
    ncols = 2 if protocols is not None else 1
    fig, axes = plots.figure(width=5.2 * ncols, height=4.0, ncols=ncols)
    axes = np.atleast_1d(axes)

    m = table[table["features"] == "morgan"].set_index(["train_set", "regime"])
    regimes = ["random CV", "scaffold CV", "transfer to natural products"]
    vals = [float(m.loc[(DRUG_LIKE, r), "r2_identity"]) for r in regimes]
    ax = axes[0]
    ax.plot(range(3), vals, marker="o", color="tab:blue",
            label="trained on drug-like Caco-2 data")
    for x, v in enumerate(vals):
        ax.annotate(f"{v:+.2f}", (x, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    nv = float(m.loc[(NATURAL, "scaffold CV"), "r2_identity"])
    ax.scatter([1], [nv], marker="s", s=40, color="tab:red", zorder=5,
               label="trained on the natural products")
    ax.annotate(f"{nv:+.2f}", (1, nv), textcoords="offset points", xytext=(12, -4),
                fontsize=8, color="tab:red")
    ax.axhline(0, color="0.5", ls=":", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["random\nsplit", "Bemis-Murcko\nscaffold split",
                        "transfer to\nnatural products"])
    ax.set_ylabel("R2 (identity line)")
    ax.set_title("Leaving the training chemistry")
    ax.legend(fontsize=8, loc="lower left")

    if protocols is not None:
        ax2 = axes[1]
        p = protocols.set_index("split")
        names = ["row-level random CV", "compound-level random CV",
                 "compound-level scaffold CV"]
        vals = [float(p.loc[n, "r2_identity"]) for n in names if n in p.index]
        ax2.plot(range(len(vals)), vals, marker="o", color="tab:green")
        for x, v in enumerate(vals):
            ax2.annotate(f"{v:+.2f}", (x, v), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8)
        ax2.axhline(0, color="0.5", ls=":", lw=0.8)
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(["replicates split\nacross folds", "compound-level\nsplit",
                             "compound-level\nscaffold split"][:len(vals)])
        ax2.set_ylabel("R2 (identity line)")
        ax2.set_title("Tightening the split, same compounds")
    return plots.save(fig, "fig_scaffold_generalization")
