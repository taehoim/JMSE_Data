"""Persistence baseline: predict the last observed target for every future step.

This is the performance floor every learned model must beat.
"""
import numpy as np


def persistence_forecast(last_value: np.ndarray, horizon: int) -> np.ndarray:
    """last_value: (N,) -> (N, horizon), each column equal to last_value."""
    last_value = np.asarray(last_value, float).reshape(-1, 1)
    return np.repeat(last_value, horizon, axis=1)
