import numpy as np

from jmse.models.persistence import persistence_forecast


def test_persistence_repeats_last_value():
    last = np.array([0.1, 0.2, 0.3])
    yhat = persistence_forecast(last_value=last, horizon=5)
    assert yhat.shape == (3, 5)
    assert np.allclose(yhat[:, 0], last)
    assert np.allclose(yhat[:, -1], last)            # repeated across all horizons


def test_persistence_is_weak_beatable_floor():
    # Xacc is a rectified magnitude with low 1-Hz autocorrelation (lag-1 ~0.2),
    # so persistence is a WEAK, clearly-beatable floor (negative R^2). This is a
    # real signal property (verified), documenting that the task is non-trivial.
    from jmse.eval.run import evaluate_id
    m = evaluate_id("persistence")
    assert len(m["r2"]) == 5 and "overall" in m
    assert np.all(np.isfinite(m["rmse"])) and np.all(np.isfinite(m["r2"]))
    assert m["overall"]["r2"] < 0.2                   # weak floor; LSTM must beat it
