"""Record- and condition-level bootstrap confidence intervals (P7, statistics rigor).

The benchmark and generalization tables report seed standard deviations, which capture training
noise but not *sampling* uncertainty: the ID test set is 15 records (5 tonnages x 3 sea states),
and the windows within a record are strongly dependent, so a naive per-window bootstrap would be
wildly over-confident. We instead resample whole records (the cluster/block bootstrap), which is
the honest unit of replication here, and report percentile CIs for the overall metric and for each
condition. We also separate the pooled (micro) average from the record-balanced (macro) average,
since the abundant easy records otherwise dominate a pooled mean.

Statistics operate on (N, H) target/prediction arrays and match jmse.eval.metrics exactly.
"""
import numpy as np
import pandas as pd

from jmse.eval.metrics import _r2


def overall_r2(y, yhat) -> float:
    """Pooled overall R^2 (all cells), identical to metrics.per_horizon_metrics['overall']['r2']."""
    return float(_r2(np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()))


def overall_rmse_deg(y, yhat) -> float:
    """Pooled overall RMSE in degrees."""
    diff = np.asarray(yhat, float) - np.asarray(y, float)
    return float(np.degrees(np.sqrt(np.mean(diff ** 2))))


def _percentile_ci(samples, alpha):
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def group_bootstrap_ci(y, yhat, groups, stat_fn, n_boot=2000, alpha=0.05, seed=0) -> dict:
    """Cluster bootstrap over records: resample unique groups with replacement, recompute stat.

    Returns {point, lo, hi} where point is the statistic on the full sample and [lo, hi] is the
    (1-alpha) percentile interval over n_boot resamples. Reproducible given `seed`.
    """
    y, yhat, groups = np.asarray(y, float), np.asarray(yhat, float), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    point = stat_fn(y, yhat)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in chosen])
        boots[b] = stat_fn(y[rows], yhat[rows])
    lo, hi = _percentile_ci(boots, alpha)
    return {"point": float(point), "lo": lo, "hi": hi}


def per_condition_ci(y, yhat, cond, groups, stat_fn, cond_name="cond",
                     n_boot=2000, alpha=0.05, seed=0) -> pd.DataFrame:
    """Per-condition point estimate and record-bootstrap CI (one row per unique condition value).

    For each condition value, the bootstrap resamples only the records belonging to that condition,
    so the CI reflects between-record variability within the condition.
    """
    y, yhat, cond, groups = (np.asarray(y, float), np.asarray(yhat, float),
                             np.asarray(cond), np.asarray(groups))
    rows = []
    for v in np.unique(cond):
        m = cond == v
        ci = group_bootstrap_ci(y[m], yhat[m], groups[m], stat_fn, n_boot, alpha, seed)
        rows.append({cond_name: v, "point": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                     "n_groups": int(np.unique(groups[m]).size), "n": int(m.sum())})
    return pd.DataFrame(rows)


def macro_micro_ci(y, yhat, groups, stat_fn, n_boot=2000, alpha=0.05, seed=0) -> dict:
    """Pooled (micro) vs record-balanced (macro) statistic, each with a record-bootstrap CI.

    macro = mean over records of the per-record statistic (condition-balanced); micro = statistic
    on the pooled cells. A large micro-macro gap means the easy records dominate the pooled number.
    """
    y, yhat, groups = np.asarray(y, float), np.asarray(yhat, float), np.asarray(groups)
    uniq = np.unique(groups)

    def macro_stat(yy, pp, gg):
        return float(np.mean([stat_fn(yy[gg == g], pp[gg == g]) for g in np.unique(gg)]))

    micro = group_bootstrap_ci(y, yhat, groups, stat_fn, n_boot, alpha, seed)
    # macro bootstrap (resample records, average per-record stat)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    macro_point = macro_stat(y, yhat, groups)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        boots[b] = float(np.mean([stat_fn(y[idx_by_group[g]], yhat[idx_by_group[g]])
                                  for g in chosen]))
    lo, hi = _percentile_ci(boots, alpha)
    return {"micro": micro, "macro": {"point": float(macro_point), "lo": lo, "hi": hi}}
