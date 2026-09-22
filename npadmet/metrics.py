"""Agreement and accuracy statistics.

Deliberately written out rather than imported from a statistics package, so that
each definition used in the paper is inspectable in one place. Every function
takes plain arrays and returns plain floats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from . import config as C


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (agreement about the identity line)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).mean()
    denom = x.var() + y.var() + (mx - my) ** 2
    return 2 * cov / denom if denom > 0 else np.nan


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan


def fleiss_kappa(mat: np.ndarray) -> float:
    """Fleiss' kappa for a binary (n_subjects x n_raters) matrix with no missing cells."""
    mat = np.asarray(mat, int)
    n, m = mat.shape
    if n == 0 or m < 2:
        return np.nan
    n1 = mat.sum(axis=1)
    n0 = m - n1
    p_i = (n1 * (n1 - 1) + n0 * (n0 - 1)) / (m * (m - 1))
    p1 = n1.sum() / (n * m)
    pe = p1 ** 2 + (1 - p1) ** 2
    return (p_i.mean() - pe) / (1 - pe) if (1 - pe) > 0 else np.nan


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC (equivalent to the Mann-Whitney U statistic, ties averaged)."""
    y_true = np.asarray(y_true, int)
    npos, nneg = int(y_true.sum()), int((1 - y_true).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(score)
    return (r[y_true == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def mcc(y: np.ndarray, yhat: np.ndarray) -> float:
    """Matthews correlation coefficient."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / denom if denom > 0 else np.nan


def r2_identity(measured: np.ndarray, predicted: np.ndarray) -> float:
    """R^2 referenced to the identity line, not to a refitted regression.

    This is the quantity that goes negative when a prediction is worse than
    simply returning the mean of the measurements. It also absorbs any constant
    offset between assay scales, which is why the manuscript treats the
    correlation coefficients, not this number, as the primary evidence.
    """
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    ss_res = np.sum((measured - predicted) ** 2)
    ss_tot = np.sum((measured - measured.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn, n: int = C.N_BOOTSTRAP,
                 seed: int = C.SEED) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a two-sample statistic."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        try:
            v = fn(x[s], y[s])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def continuous_metrics(measured: np.ndarray, predicted: np.ndarray,
                       n_boot: int = C.N_BOOTSTRAP) -> dict:
    """Full agreement summary of a prediction against a measurement, on complete cases."""
    d = pd.DataFrame({"m": measured, "p": predicted}).dropna()
    x, y = d["m"].to_numpy(), d["p"].to_numpy()
    r, pval = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    diff = y - x
    lo, hi = bootstrap_ci(x, y, lambda a, b: pearsonr(a, b)[0], n=n_boot)
    return {
        "n": int(len(d)),
        "pearson_r": float(r),
        "pearson_ci_lo": lo,
        "pearson_ci_hi": hi,
        "pearson_p": float(pval),
        "spearman": float(rho),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "r2": float(r2_identity(x, y)),
        "ccc": float(ccc(x, y)),
        "ba_bias": float(diff.mean()),
        "ba_loa": float(1.96 * diff.std()),
    }


def binary_metrics(measured: np.ndarray, predicted: np.ndarray,
                   threshold: float = C.PERM_THRESHOLD) -> dict:
    """Treat the same threshold as a permeability call on both sides of the comparison."""
    d = pd.DataFrame({"m": measured, "p": predicted}).dropna()
    y = (d["m"] > threshold).astype(int).to_numpy()
    score = d["p"].to_numpy()
    yhat = (score > threshold).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "auroc": float(auroc(y, score)),
        "mcc": float(mcc(y, yhat)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_acc": float(np.nanmean([sens, spec])),
        "accuracy": float((tp + tn) / len(y)),
        "n_permeable": int(y.sum()),
    }
