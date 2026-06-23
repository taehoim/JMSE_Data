"""Per-horizon regression metrics (column-wise RMSE/MAE/R^2 + overall).

Fixes the non-monotonic per-step values in the original paper's Tables 5-7 by
computing each horizon's metric strictly column-wise and validating against sklearn.
All metrics are in the units of the provided arrays (caller converts to degrees).
"""
import numpy as np


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: (N, horizon). Returns rmse/mae/r2 lists (len horizon) + overall."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    assert y_true.shape == y_pred.shape and y_true.ndim == 2, (y_true.shape, y_pred.shape)
    H = y_true.shape[1]
    diff = y_pred - y_true

    rmse = [float(np.sqrt(np.mean(diff[:, k] ** 2))) for k in range(H)]
    mae = [float(np.mean(np.abs(diff[:, k]))) for k in range(H)]
    r2 = [float(_r2(y_true[:, k], y_pred[:, k])) for k in range(H)]

    overall = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "r2": float(_r2(y_true.ravel(), y_pred.ravel())),
    }
    return {"rmse": rmse, "mae": mae, "r2": r2, "overall": overall}


def to_degrees(metrics: dict) -> dict:
    """Convert rad-based rmse/mae (not r2) to degrees for reporting."""
    out = {"r2": list(metrics["r2"]), "overall": dict(metrics["overall"])}
    out["rmse"] = [float(np.degrees(v)) for v in metrics["rmse"]]
    out["mae"] = [float(np.degrees(v)) for v in metrics["mae"]]
    out["overall"]["rmse"] = float(np.degrees(metrics["overall"]["rmse"]))
    out["overall"]["mae"] = float(np.degrees(metrics["overall"]["mae"]))
    return out
