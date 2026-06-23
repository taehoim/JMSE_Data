import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from jmse.eval.metrics import per_horizon_metrics


def test_matches_sklearn_per_column():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(500, 5))
    yhat = y + rng.normal(scale=0.3, size=(500, 5))
    m = per_horizon_metrics(y, yhat)
    for k in range(5):
        assert abs(m["r2"][k] - r2_score(y[:, k], yhat[:, k])) < 1e-9
        assert abs(m["rmse"][k] - mean_squared_error(y[:, k], yhat[:, k]) ** 0.5) < 1e-9
        assert abs(m["mae"][k] - mean_absolute_error(y[:, k], yhat[:, k])) < 1e-9


def test_monotonic_degradation_is_monotonic():
    # predictions degrade with horizon -> per-horizon RMSE must strictly increase
    rng = np.random.default_rng(1)
    y = rng.normal(size=(2000, 5))
    yhat = y + rng.normal(scale=1.0, size=(2000, 5)) * np.arange(1, 6)
    m = per_horizon_metrics(y, yhat)
    assert all(np.diff(m["rmse"]) > 0)
    assert all(np.diff(m["r2"]) < 0)


def test_overall_aggregates_present():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(100, 5)); yhat = y + 0.1
    m = per_horizon_metrics(y, yhat)
    assert {"rmse", "mae", "r2"} <= set(m["overall"])
    assert len(m["rmse"]) == 5 and len(m["r2"]) == 5
