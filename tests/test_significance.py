"""Tests for significance testing of paired AUC / metric differences (rigor follow-up)."""
import numpy as np

from jmse.earlywarning import roc
from jmse.eval import significance as sig


def _rng(s=0):
    return np.random.default_rng(s)


def test_delong_auc_matches_mann_whitney():
    r = _rng(1)
    labels = (r.uniform(size=2000) < 0.3).astype(int)
    score = labels * 0.6 + r.uniform(size=2000)            # informative
    auc_a, auc_b, z, p = sig.delong_test(labels, score, score)
    assert abs(auc_a - roc.roc_auc(labels, score)) < 1e-6  # DeLong AUC == MW AUC
    assert abs(auc_a - auc_b) < 1e-9 and p > 0.99           # identical scores -> no difference


def test_delong_detects_real_difference():
    r = _rng(2)
    labels = (r.uniform(size=4000) < 0.3).astype(int)
    good = labels * 1.2 + r.normal(size=4000)              # strong separation
    bad = labels * 0.2 + r.normal(size=4000)               # weak separation
    auc_g, auc_b, z, p = sig.delong_test(labels, good, bad)
    assert auc_g > auc_b
    assert p < 1e-6                                          # difference highly significant


def test_cluster_bootstrap_auc_diff_excludes_zero_when_real():
    r = _rng(3)
    groups = np.repeat(np.arange(12), 400)
    labels = (r.uniform(size=groups.size) < 0.3).astype(int)
    good = labels * 1.0 + r.normal(size=groups.size)
    bad = labels * 0.2 + r.normal(size=groups.size)
    out = sig.auc_diff_cluster_bootstrap(labels, good, bad, groups, n_boot=300, seed=0)
    assert out["diff"] > 0
    assert out["lo"] > 0                                     # CI excludes zero -> significant
    assert out["p"] < 0.05


def test_cluster_bootstrap_auc_diff_includes_zero_when_none():
    r = _rng(4)
    groups = np.repeat(np.arange(12), 400)
    labels = (r.uniform(size=groups.size) < 0.3).astype(int)
    s = labels * 0.8 + r.normal(size=groups.size)
    out = sig.auc_diff_cluster_bootstrap(labels, s, s.copy(), groups, n_boot=300, seed=0)
    assert abs(out["diff"]) < 1e-9
    assert out["lo"] <= 0 <= out["hi"]                      # CI brackets zero -> not significant


def test_metric_diff_cluster_bootstrap_for_r2():
    r = _rng(5)
    groups = np.repeat(np.arange(10), 300)
    y = r.normal(size=(3000, 2))
    good = y + 0.2 * r.normal(size=y.shape)
    bad = y + 0.6 * r.normal(size=y.shape)
    from jmse.eval.bootstrap import overall_r2
    out = sig.metric_diff_cluster_bootstrap(y, good, bad, groups, overall_r2, n_boot=300, seed=0)
    assert out["diff"] > 0 and out["lo"] > 0                 # 'good' model significantly better
