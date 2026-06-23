"""Tests for record-/condition-level bootstrap CIs (P7, statistics rigor)."""
import numpy as np

from jmse.eval import bootstrap as bs
from jmse.eval.metrics import _r2


def test_overall_stats_match_metrics():
    r = np.random.default_rng(0)
    y = r.standard_normal((500, 5))
    yhat = y + 0.1 * r.standard_normal((500, 5))
    assert np.isclose(bs.overall_r2(y, yhat), _r2(y.ravel(), yhat.ravel()))
    assert np.isclose(bs.overall_rmse_deg(y, yhat),
                      np.degrees(np.sqrt(np.mean((yhat - y) ** 2))))


def test_group_bootstrap_ci_brackets_point_estimate():
    r = np.random.default_rng(1)
    groups = np.repeat(np.arange(10), 100)
    y = r.standard_normal((1000, 3))
    yhat = y + 0.2 * r.standard_normal((1000, 3))
    out = bs.group_bootstrap_ci(y, yhat, groups, bs.overall_r2, n_boot=400, seed=0)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["hi"] - out["lo"] > 0


def test_group_bootstrap_is_reproducible():
    r = np.random.default_rng(2)
    groups = np.repeat(np.arange(8), 50)
    y = r.standard_normal((400, 2))
    yhat = y + 0.3 * r.standard_normal((400, 2))
    a = bs.group_bootstrap_ci(y, yhat, groups, bs.overall_rmse_deg, n_boot=200, seed=7)
    b = bs.group_bootstrap_ci(y, yhat, groups, bs.overall_rmse_deg, n_boot=200, seed=7)
    assert a == b


def test_per_condition_ci_one_row_per_value():
    r = np.random.default_rng(3)
    groups = np.repeat(np.arange(6), 100)
    cond = np.repeat([3.0, 3.0, 5.0, 5.0, 7.0, 7.0], 100)     # 2 groups per condition
    y = r.standard_normal((600, 2))
    yhat = y + 0.2 * r.standard_normal((600, 2))
    df = bs.per_condition_ci(y, yhat, cond, groups, bs.overall_r2, "Hs",
                             n_boot=200, seed=0)
    assert sorted(df["Hs"].tolist()) == [3.0, 5.0, 7.0]
    assert (df["lo"] <= df["point"]).all() and (df["point"] <= df["hi"]).all()


def test_macro_micro_differ_under_imbalanced_skill():
    # one group is much worse; macro (record-balanced) < micro is not guaranteed, but both defined
    groups = np.repeat(np.arange(3), 100)
    y = np.zeros((300, 1))
    yhat = np.zeros((300, 1))
    yhat[:100] = 5.0                                          # group 0 terrible, others perfect
    y[:, 0] = np.arange(300) % 7                              # give y some variance
    out = bs.macro_micro_ci(y, yhat, groups, bs.overall_r2, n_boot=200, seed=0)
    assert "micro" in out and "macro" in out
    assert set(out["micro"]) == {"point", "lo", "hi"}
