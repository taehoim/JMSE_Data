"""GM-floor confound study (R3.4): clamped vs unclamped 10 t generalization.

The fishingVessel.m hydrostatics clamp the transverse GM at a 0.30 m floor. Only the
10 t vessel's true GM falls below that floor, so the 10 t cell carries a GM discontinuity
relative to the 20-50 t vessels. Reviewer R3 asked how much of the (large) 10 t
generalization gap is that artificial discontinuity versus genuine small-vessel dynamics.

This study answers it by building the 10 t Leave-One-Vessel-Out fold TWICE -- once from
the clamped canonical data and once from the unclamped data (where only the 10 t GM, and
hence its roll dynamics, differ) -- training the same LSTM backbone on each, and tabulating
the 10 t test metrics side by side. A smaller clamped-vs-unclamped gap means the GM
discontinuity explains little of the 10 t difficulty; a larger one means it explains a lot.

Usage:
    python -m jmse.eval.gm_floor [--config jmse/run_configs/lstm_id.yaml] [--smoke]
Outputs results/gm_floor/T_gmfloor.csv (one row per variant). The heavy multi-seed run is
Phase 3; --smoke is for development.
"""
import argparse

import numpy as np
import pandas as pd

from jmse import config
from jmse.data.curate import build_curated_dataset
from jmse.data.windowing import build_ood_arrays
from jmse.earlywarning.alarm import exceedance_labels
from jmse.eval.run import evaluate_id
from jmse.models.base import set_seed
from jmse.train import DEVICE, fit_neural, load_config

_HOLD_TON = 10                                   # the only vessel whose GM hits the floor
_HEAD_RAD = config.DANGER_THRESHOLDS_RAD[0]      # headline danger threshold (15 deg)

# variant -> curate source for build_curated_dataset
_VARIANT_SOURCE = {"clamped": "multi", "unclamped": "multi_unclamped"}


def unclamped_resim_available() -> bool:
    """True iff the re-simulated unclamped-EOM CSVs are present (R3.4 sim output)."""
    d = config.RAW_DATA_DIR_MULTI_UNCLAMPED
    return d.is_dir() and any(d.glob("6Dof_*_withXacc.csv"))


def curated_variant(variant: str) -> pd.DataFrame:
    """Curated frame for a GM-floor variant ('clamped' -> canonical, 'unclamped' -> floor off).

    The 'unclamped' variant needs the R3.4 re-simulated 10 t motion (true sub-floor GM in
    the EOM restoring term). That data is produced in the sim domain; until it lands this
    raises a clear, actionable error rather than silently using stale/identical motion.
    """
    if variant == "unclamped" and not unclamped_resim_available():
        raise FileNotFoundError(
            "Unclamped-EOM re-simulation not present yet: expected withXacc CSVs in "
            f"{config.RAW_DATA_DIR_MULTI_UNCLAMPED} (override with JMSE_UNCLAMPED_DIR). "
            "The GM-floor study needs the 10 t motion re-simulated with the true sub-floor "
            "GM; the deprecated GZ-only set shares clamped motion and would give a null tie."
        )
    return build_curated_dataset(source=_VARIANT_SOURCE[variant])


def _exceedance_recall(y_true_rad: np.ndarray, y_pred_rad: np.ndarray, theta: float) -> float:
    """Recall of danger-threshold exceedance (Xacc > theta) over all (window, horizon) cells.

    NaN when there are no positive labels (no dangerous-roll events in the test fold)."""
    y = exceedance_labels(y_true_rad, theta)
    p = exceedance_labels(y_pred_rad, theta)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    return float((p & y).sum() / pos)


def _metrics_row(variant: str, gm_10t: float, deg: dict,
                 y_true_rad: np.ndarray, y_pred_rad: np.ndarray) -> dict:
    """One side-by-side row: 10 t test RMSE/R2/exceedance for a variant."""
    o = deg["overall"]
    labels = exceedance_labels(y_true_rad, _HEAD_RAD)
    return {
        "variant": variant,
        "gm_10t_m": round(float(gm_10t), 4),
        "n_test_windows": int(len(y_true_rad)),
        "rmse_deg": float(o["rmse"]),
        "mae_deg": float(o["mae"]),
        "r2": float(o["r2"]),
        "exceed15_prevalence": float(labels.mean()),
        "exceed15_recall": _exceedance_recall(y_true_rad, y_pred_rad, _HEAD_RAD),
    }


def _run_variant(variant: str, base_cfg: dict, seed: int, smoke: bool) -> dict:
    """Build the 10 t LOVO fold for one variant, train the LSTM, return the metrics row."""
    df = curated_variant(variant)
    gm_10t = float(df[df.tonnage == _HOLD_TON]["GM"].iloc[0])
    d = build_ood_arrays(hold_ton=_HOLD_TON, df=df)

    set_seed(seed)
    cfg = {**base_cfg, "seed": seed, "name": f"gmfloor_{variant}"}
    model, yscaler = fit_neural(cfg, d, smoke=smoke)

    # Evaluate on the 10 t test fold; capture per-window preds for the exceedance metrics.
    y_pred_rad = np.asarray(model.predict(d["X_test"], device=DEVICE, target_scaler=yscaler), float)
    deg = evaluate_id(f"gmfloor_{variant}", predict_fn=lambda X: y_pred_rad,
                      arrays=d, split="lovo", write=False)
    return _metrics_row(variant, gm_10t, deg, d["y_test"], y_pred_rad)


def run_gm_floor(base_config="jmse/run_configs/lstm_id.yaml", seed=0, smoke=False, out_dir=None):
    """Tabulate clamped-vs-unclamped 10 t test metrics -> T_gmfloor.csv."""
    # Fail fast (before training the clamped variant) if the re-sim isn't ready yet.
    if not unclamped_resim_available():
        raise FileNotFoundError(
            "GM-floor study requires the unclamped-EOM re-simulation, not present yet: "
            f"expected withXacc CSVs in {config.RAW_DATA_DIR_MULTI_UNCLAMPED} "
            "(override with JMSE_UNCLAMPED_DIR). Deferred to Phase 3."
        )
    from pathlib import Path
    out_dir = Path(out_dir) if out_dir is not None else config.RESULTS_DIR / "gm_floor"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(base_config)

    rows = [_run_variant(v, base, seed, smoke) for v in ("clamped", "unclamped")]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "T_gmfloor.csv", index=False)
    return out_dir, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="jmse/run_configs/lstm_id.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir, table = run_gm_floor(args.config, seed=args.seed, smoke=args.smoke)
    print("\n=== T_gmfloor: clamped vs unclamped 10 t (LOVO) ===")
    print(table.to_string(index=False))
    print(f"wrote {out_dir / 'T_gmfloor.csv'}")


if __name__ == "__main__":
    main()
