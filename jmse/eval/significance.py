"""Significance tests for paired differences in AUC and regression skill (rigor follow-up).

The manuscript claims the probabilistic alarm beats the point alarm and that the LSTM beats the
baselines. Reporting seed mean +/- std shows consistency but is not a hypothesis test. Here we add
two complementary tests for a paired difference (both forecasters scored on the SAME data, so the
estimates are correlated):

  delong_test                 the DeLong (1988) test for two correlated ROC AUCs, via the fast
                              midrank algorithm (Sun & Xu 2014). Standard and recognized, but its
                              variance assumes independent cases.
  auc_diff_cluster_bootstrap  a record-level (cluster) bootstrap of the AUC difference, which
                              respects the strong within-record dependence of the 1 Hz windows and
                              is therefore the primary test here.
  metric_diff_cluster_bootstrap  the same cluster bootstrap for any paired regression metric
                              difference (e.g. R^2 of model A vs model B).

A difference is significant at level alpha when its (1-alpha) bootstrap interval excludes zero.
"""
import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------------- DeLong
def _compute_midrank(x):
    """Midranks (ties averaged), as required by the fast DeLong algorithm."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """Fast DeLong structural components for k classifiers (rows), positives-first columns.

    preds_sorted: (k, N) with the first m columns the positive cases. Returns (aucs (k,), cov (k,k)).
    """
    k, N = preds_sorted.shape
    n = N - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, N))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(labels, score_a, score_b):
    """DeLong test for two correlated ROC AUCs. Returns (auc_a, auc_b, z, p_two_sided)."""
    labels = np.asarray(labels).astype(int).ravel()
    sa, sb = np.asarray(score_a, float).ravel(), np.asarray(score_b, float).ravel()
    order = np.argsort(-labels)                                   # positives (label 1) first
    m = int(labels.sum())
    if m == 0 or m == labels.size:
        return float("nan"), float("nan"), float("nan"), float("nan")
    preds = np.vstack([sa, sb])[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if abs(aucs[0] - aucs[1]) < 1e-12 else np.inf
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else 0.0
    return float(aucs[0]), float(aucs[1]), float(z), p


# ----------------------------------------------------------------------- cluster bootstrap
def _bootstrap_p(boot_diffs):
    """Two-sided bootstrap p-value: 2 * min(frac<=0, frac>=0), clipped to [0,1]."""
    b = np.asarray(boot_diffs, float)
    p = 2.0 * min(np.mean(b <= 0), np.mean(b >= 0))
    return float(min(1.0, p))


def auc_diff_cluster_bootstrap(labels, score_a, score_b, groups,
                               n_boot=2000, alpha=0.05, seed=0):
    """Record-level cluster bootstrap of AUC(a) - AUC(b). Returns auc_a/auc_b/diff/lo/hi/p."""
    from jmse.earlywarning.roc import roc_auc
    labels = np.asarray(labels).astype(int).ravel()
    sa, sb = np.asarray(score_a, float).ravel(), np.asarray(score_b, float).ravel()
    groups = np.asarray(groups).ravel()
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    auc_a, auc_b = roc_auc(labels, sa), roc_auc(labels, sb)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        lab = labels[rows]
        if lab.all() or not lab.any():
            diffs[b] = 0.0
            continue
        diffs[b] = roc_auc(lab, sa[rows]) - roc_auc(lab, sb[rows])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "diff": float(auc_a - auc_b),
            "lo": float(lo), "hi": float(hi), "p": _bootstrap_p(diffs)}


def metric_diff_cluster_bootstrap(y, yhat_a, yhat_b, groups, stat_fn,
                                  n_boot=2000, alpha=0.05, seed=0):
    """Record-level cluster bootstrap of stat_fn(a) - stat_fn(b) for two predictions of the same y."""
    y = np.asarray(y, float)
    ya, yb = np.asarray(yhat_a, float), np.asarray(yhat_b, float)
    groups = np.asarray(groups).ravel()
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    point = stat_fn(y, ya) - stat_fn(y, yb)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        diffs[b] = stat_fn(y[rows], ya[rows]) - stat_fn(y[rows], yb[rows])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "p": _bootstrap_p(diffs)}
