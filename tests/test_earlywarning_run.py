import numpy as np
from scipy.stats import norm

from jmse.earlywarning.run import _per_horizon_auc, predictive


def test_predictive_gaussian_form():
    z = {"mean": np.array([[0.0, 1.0]]), "std": np.array([[1.0, 2.0]])}
    point, prob_fn = predictive("ensemble", z)
    assert np.allclose(point, z["mean"])               # point score = mean
    theta = 0.5
    assert np.allclose(prob_fn(theta), norm.sf((theta - z["mean"]) / z["std"]))


def test_predictive_quantile_form_uses_median_as_point():
    taus = np.array([0.05, 0.5, 0.95])
    q = np.empty((2, 1, 3))
    q[0, 0] = [0.0, 0.3, 0.9]
    q[1, 0] = [-0.2, 0.1, 0.4]
    z = {"quantiles": q, "taus": taus}
    point, prob_fn = predictive("quantile", z)
    assert np.allclose(point[:, 0], [0.3, 0.1])        # median = tau==0.5 slice
    p = prob_fn(0.3)
    assert p.shape == (2, 1) and np.all((p >= 0) & (p <= 1))


def test_per_horizon_auc_shape_and_values():
    # perfectly separable per column -> AUC 1.0 each
    label = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=bool)
    score = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
    auc = _per_horizon_auc(label, score)
    assert auc.shape == (2,)
    assert np.allclose(auc, 1.0)
