"""Evaluation harness: run a model on the in-distribution test set and record metrics.

Targets/predictions are kept in radians; reported RMSE/MAE are converted to degrees.
Persistence is implemented here directly; learned models plug in via `predict_fn`.
"""
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from jmse import config
from jmse.data.windowing import build_id_arrays
from jmse.eval.metrics import per_horizon_metrics, to_degrees
from jmse.models.persistence import persistence_forecast


def save_metrics(model_name: str, split: str, deg: dict) -> Path:
    rows = [
        {"horizon_s": k + 1, "rmse_deg": deg["rmse"][k], "mae_deg": deg["mae"][k], "r2": deg["r2"][k]}
        for k in range(len(deg["r2"]))
    ]
    rows.append({"horizon_s": "overall", "rmse_deg": deg["overall"]["rmse"],
                 "mae_deg": deg["overall"]["mae"], "r2": deg["overall"]["r2"]})
    out_dir = config.RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def evaluate_id(
    model_name: str,
    predict_fn: Optional[Callable] = None,
    arrays: Optional[dict] = None,
    split: str = "id",
    save_preds_path: Optional[Path] = None,
    write: bool = True,
) -> dict:
    """Evaluate a model on a test set. Returns degree-based metrics dict; writes CSV.

    - model_name == "persistence": repeats last observed Xacc.
    - otherwise: `predict_fn(X_test) -> (N, horizon)` predictions in radians.
    `arrays` lets the caller supply ID or OOD arrays; `split` names the output CSV
    (id | loho | lovo). This is the single eval path used by both persistence and
    learned models (train.py). When `save_preds_path` is given, the per-window
    (y_true, y_pred) arrays (radians) are saved there for paired significance tests.
    `write=False` skips the metrics CSV (used by --smoke so sanity runs never clobber
    committed reference results like results/lstm_id/id_metrics.csv).
    """
    d = arrays if arrays is not None else build_id_arrays()
    y_true = d["y_test"]                                  # radians

    if model_name == "persistence":
        y_pred = persistence_forecast(d["last_obs_test"], horizon=y_true.shape[1])
    else:
        if predict_fn is None:
            raise ValueError(f"predict_fn required for model '{model_name}'")
        y_pred = np.asarray(predict_fn(d["X_test"]), float)

    if save_preds_path is not None:
        save_preds_path = Path(save_preds_path)
        save_preds_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_preds_path, y_true=y_true, y_pred=y_pred)

    deg = to_degrees(per_horizon_metrics(y_true, y_pred))
    if write:
        save_metrics(model_name, split, deg)
    return deg
