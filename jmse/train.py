"""Config-driven training entrypoint for all learned forecasters.

Usage:
    python -m jmse.train --config jmse/run_configs/lstm_id.yaml            # full run (GPU)
    python -m jmse.train --config jmse/run_configs/lstm_id.yaml --smoke    # fast CPU smoke

Writes results/<name>/{id_metrics.csv (or <split>_metrics.csv), checkpoint.pt, scalers.npz}.
Targets are StandardScaler-scaled per horizon for training and inverse-scaled at eval.
"""
import argparse

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jmse import config
from jmse.data.windowing import (
    build_combined_holdout, build_id_arrays, build_loro_arrays, build_ood_arrays,
)
from jmse.eval.run import evaluate_id
from jmse.models.base import set_seed
from jmse.models.classical import build_classical, is_classical
from jmse.models.registry import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_arrays(cfg: dict) -> dict:
    regime = cfg.get("regime", "id")
    if regime == "id":
        return build_id_arrays()
    if regime in ("loho", "lovo"):
        return build_ood_arrays(hold_hs=cfg.get("hold_hs"), hold_ton=cfg.get("hold_ton"))
    if regime == "loro":
        return build_loro_arrays(hold_real=cfg["hold_real"])
    if regime == "combined":
        return build_combined_holdout(hold_hs=cfg.get("hold_hs"), hold_ton=cfg.get("hold_ton"))
    raise ValueError(f"Unknown regime: {regime}")


def train_classical(cfg: dict, d: dict, save_preds_path=None, write=True) -> dict:
    """Fit a closed-form classical baseline (AR / Kalman) on the target history and
    evaluate through the same `evaluate_id` path. CPU-only; no gradient loop, no seed
    dependence, so it is reported once (not over SEEDS)."""
    model = build_classical(cfg["model"], order=cfg.get("order", 8))
    model.fit(d["yhist_train"])
    horizon = d["y_train"].shape[1]
    name = cfg["name"]
    split = "id" if cfg.get("regime", "id") == "id" else cfg["regime"]
    deg = evaluate_id(
        name,
        predict_fn=lambda X: model.predict(d["yhist_test"], horizon=horizon),  # X ignored
        arrays=d,
        split=split,
        save_preds_path=save_preds_path,
        write=write,
    )
    return deg


def _smoke_subset(Xtr, ytr, Xva, yva):
    """Representative subset for fast CPU smoke checks (shared by all training paths)."""
    rng = np.random.default_rng(0)
    ti = rng.permutation(len(Xtr))[:4000]
    vi = rng.permutation(len(Xva))[:1000]
    return Xtr[ti], ytr[ti], Xva[vi], yva[vi]


def _make_loaders(Xtr, ytr_s, Xva, yva_s, cfg):
    g = torch.Generator().manual_seed(cfg.get("seed", 0))   # reproducible shuffling

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                           torch.as_tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=cfg.get("batch_size", 128), shuffle=shuffle,
                          generator=g if shuffle else None)

    return loader(Xtr, ytr_s, True), loader(Xva, yva_s, False)


def _fit_loop(model, train_loader, val_loader, loss_fn, cfg, smoke, epochs_override=None):
    """Generic Adam + early-stopping loop; `loss_fn(pred, target)` selects the objective.

    Loads and returns the model at its best validation loss. Shared by the MSE point
    forecasters and the pinball-loss quantile forecaster (P2 UQ)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    epochs = epochs_override or (2 if smoke else cfg.get("epochs", 100))
    patience = cfg.get("patience", 12)
    best_val, best_state, no_improve = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
                          for xb, yb in val_loader])
        print(f"epoch {ep:3d}/{epochs}  val_loss={vl:.6f}")
        if vl < best_val - 1e-6:
            best_val, best_state, no_improve = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {ep}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def fit_neural_on(cfg, Xtr, ytr, Xva, yva, smoke=False, epochs_override=None):
    """Train a point (MSE) forecaster on arbitrary (X, y) targets. Returns (model, yscaler).

    Target-agnostic core: the C5 representation study trains this on phi / theta / GZ as
    well as Xacc, reusing the exact same loop, scaling, and early stopping."""
    if smoke:
        Xtr, ytr, Xva, yva = _smoke_subset(Xtr, ytr, Xva, yva)
    yscaler = StandardScaler().fit(ytr)                 # scale targets per horizon (train only)
    train_loader, val_loader = _make_loaders(Xtr, yscaler.transform(ytr),
                                             Xva, yscaler.transform(yva), cfg)
    model = build_model(cfg["model"], n_features=Xtr.shape[2], horizon=ytr.shape[1],
                        hidden=cfg.get("hidden", 128), layers=cfg.get("layers", 2),
                        dropout=cfg.get("dropout", 0.3)).to(DEVICE)
    model = _fit_loop(model, train_loader, val_loader, nn.MSELoss(), cfg, smoke, epochs_override)
    return model, yscaler


def fit_neural(cfg: dict, d: dict, smoke: bool = False, epochs_override: int = None):
    """Train a point (MSE) forecaster on the scaled Xacc targets. Returns (model, yscaler).

    Reused by train() and by the P2 UQ runner (MC-Dropout / Deep Ensemble)."""
    return fit_neural_on(cfg, d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                         smoke=smoke, epochs_override=epochs_override)


def train(cfg: dict, smoke: bool = False, epochs_override: int = None, save_preds_path=None) -> dict:
    set_seed(cfg.get("seed", 0))
    d = build_arrays(cfg)

    if is_classical(cfg["model"]):
        return train_classical(cfg, d, save_preds_path=save_preds_path, write=not smoke)

    model, yscaler = fit_neural(cfg, d, smoke=smoke, epochs_override=epochs_override)

    # evaluate on test via the single shared eval path (inverse-scaled to radians)
    name = cfg["name"]
    split = "id" if cfg.get("regime", "id") == "id" else cfg["regime"]
    deg = evaluate_id(
        name,
        predict_fn=lambda X: model.predict(X, device=DEVICE, target_scaler=yscaler),
        arrays=d,
        split=split,
        save_preds_path=save_preds_path,
        write=not smoke,                                  # --smoke never clobbers committed CSVs
    )
    if not smoke:
        out_dir = config.RESULTS_DIR / name
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "checkpoints" / "model.pt")
        np.savez(out_dir / "scalers.npz",
                 feat_mean=d["scaler"].mean_, feat_scale=d["scaler"].scale_,
                 y_mean=yscaler.mean_, y_scale=yscaler.scale_)
    return deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"device={DEVICE}  config={cfg['name']}  smoke={args.smoke}")
    deg = train(cfg, smoke=args.smoke, epochs_override=args.epochs)
    for k in range(len(deg["r2"])):
        print(f"  t+{k+1}s: RMSE={deg['rmse'][k]:.2f} deg, R2={deg['r2'][k]:.4f}")
    print(f"  overall: RMSE={deg['overall']['rmse']:.2f} deg, R2={deg['overall']['r2']:.4f}")


if __name__ == "__main__":
    main()
