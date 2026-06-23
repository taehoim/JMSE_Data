"""Deep-ensemble uncertainty: combine M independently-seeded models' predictions.

Each member contributes a point-prediction array (N, horizon) in target units (radians);
the ensemble mean is the forecast and the across-member spread is the uncertainty. These
are pure aggregations — the members are produced by the benchmark sweep's per-seed runs,
so no extra training is needed beyond what the sweep already does.
"""
import numpy as np


def _stack(members):
    if not members:
        raise ValueError("ensemble requires at least one member prediction array")
    arr = np.stack([np.asarray(m, float) for m in members], axis=0)   # (M, N, horizon)
    if arr.ndim != 3:
        raise ValueError(f"each member must be 2-D (N, horizon); got stack {arr.shape}")
    return arr


def ensemble_moments(members):
    """Return (mean, std) across M member arrays, each (N, horizon).

    std is the population standard deviation (ddof=0) over members — the ensemble's
    predictive spread, zero for a single member.
    """
    arr = _stack(members)
    return arr.mean(axis=0), arr.std(axis=0)


def ensemble_samples(members) -> np.ndarray:
    """Return members as an (N, horizon, M) sample tensor for sample-based CRPS."""
    return np.moveaxis(_stack(members), 0, -1)
