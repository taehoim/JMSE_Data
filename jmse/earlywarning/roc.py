"""Detection metrics for the early-warning alarm (C2): ROC/PR-AUC, F1, operating points.

`roc_auc` is implemented directly via the Mann-Whitney U statistic (with average ranks
for ties) so the headline number is auditable; PR-AUC and curve points use scikit-learn.
All functions take flat (label, score) pairs — the caller flattens (N, horizon) or slices
a single horizon. Degenerate single-class inputs return NaN rather than raising.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve


def _flat(labels, scores):
    return np.asarray(labels).astype(bool).ravel(), np.asarray(scores, float).ravel()


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve via Mann-Whitney U; NaN if labels are single-class."""
    y, s = _flat(labels, scores)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = _average_ranks(s[order])
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved to their average (for the sorted array)."""
    n = len(sorted_vals)
    ranks = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def pr_auc(labels, scores) -> float:
    """Average precision (area under precision-recall); NaN if labels are single-class."""
    y, s = _flat(labels, scores)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def confusion_at(labels, scores, thr: float):
    """(tp, fp, fn, tn) for the decision score >= thr."""
    y, s = _flat(labels, scores)
    fired = s >= thr
    tp = int((fired & y).sum())
    fp = int((fired & ~y).sum())
    fn = int((~fired & y).sum())
    tn = int((~fired & ~y).sum())
    return tp, fp, fn, tn


def f1_at(labels, scores, thr: float) -> float:
    tp, fp, fn, _ = confusion_at(labels, scores, thr)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def best_f1(labels, scores):
    """Best F1 over candidate thresholds (the unique scores) and its threshold."""
    y, s = _flat(labels, scores)
    best_f, best_t = 0.0, 0.5
    for thr in np.unique(s):
        f = f1_at(y, s, thr)
        if f > best_f:
            best_f, best_t = f, float(thr)
    return best_f, best_t


def operating_point_at_fpr(labels, scores, target_fpr: float) -> dict:
    """Highest-recall operating point whose false-alarm rate (FPR) <= target_fpr.

    Returns threshold, fpr, tpr, precision at that point. Used to compare alarms at an
    equal false-alarm budget (the C2 punchline: probabilistic beats point at matched FPR).
    """
    y, s = _flat(labels, scores)
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= target_fpr + 1e-12
    i = np.where(ok)[0][-1]                               # largest FPR within budget -> max TPR
    t = thr[i]
    tp, fp, fn, _ = confusion_at(y, s, t)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"threshold": float(t), "fpr": float(fpr[i]), "tpr": float(tpr[i]),
            "precision": float(precision)}


def roc_points(labels, scores):
    """(fpr, tpr) arrays for plotting the ROC curve."""
    y, s = _flat(labels, scores)
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def pr_points(labels, scores):
    """(recall, precision) arrays for plotting the PR curve."""
    y, s = _flat(labels, scores)
    precision, recall, _ = precision_recall_curve(y, s)
    return recall, precision
