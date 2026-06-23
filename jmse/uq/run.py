"""UQ orchestrator (Task 18, C1): run the three UQ methods, score calibration -> T5 + F6.

For the shared LSTM backbone it produces a predictive distribution on the ID test set
for each method and scores it with jmse.uq.calibration:
  - MC-Dropout : one trained model, T dropout-active passes -> (mean, std), Gaussian CRPS.
  - Deep Ensemble : M seed models -> across-member (mean, std), Gaussian CRPS.
  - Quantile : pinball-trained quantiles (0.05/0.5/0.95) -> [q05,q95] interval, median
               point forecast, quantile-decomposition CRPS (coarse with 3 levels).

All widths/scores are reported in degrees. Heavy (trains 1 + M + 1 models) -> use
--smoke for a CPU infra check; the full run is the user's GPU job.

Usage:
    python -m jmse.uq.run --config jmse/run_configs/uq_id.yaml [--smoke]
Outputs under results/uq[_smoke]/: T5_uq.csv, reliability.csv, preds/<method>.npz, F6.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jmse import config as cfgmod
from jmse.data.windowing import build_id_arrays
from jmse.eval import stats
from jmse.models.base import set_seed
from jmse.train import DEVICE, _fit_loop, _make_loaders, _smoke_subset, fit_neural, load_config
from jmse.uq import calibration as cal
from jmse.uq.ensemble import ensemble_moments
from jmse.uq.mc_dropout import mc_dropout_predict
from jmse.uq.quantile import QuantileForecaster, pinball_loss

RELIABILITY_LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def _deg(x):
    return float(np.degrees(x))


def fit_quantile(cfg, d, taus, smoke=False):
    """Train a QuantileForecaster with the pinball loss. Returns (model, yscaler)."""
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    if smoke:
        Xtr, ytr, Xva, yva = _smoke_subset(Xtr, ytr, Xva, yva)
    yscaler = StandardScaler().fit(ytr)
    train_loader, val_loader = _make_loaders(Xtr, yscaler.transform(ytr),
                                             Xva, yscaler.transform(yva), cfg)
    model = QuantileForecaster(n_features=Xtr.shape[2], hidden=cfg.get("hidden", 128),
                               layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.3),
                               horizon=ytr.shape[1], taus=taus).to(DEVICE)
    loss_fn = lambda pred, target: pinball_loss(pred, target, taus)   # noqa: E731
    model = _fit_loop(model, train_loader, val_loader, loss_fn, cfg, smoke)
    return model, yscaler


def _aleatoric_std(mean_val, y_val):
    """Per-horizon homoscedastic aleatoric std (radians) from validation residuals."""
    return np.sqrt(np.mean((np.asarray(mean_val) - np.asarray(y_val)) ** 2, axis=0))  # (H,)


def _total_std(epistemic_std, aleatoric_h):
    """Combine model-spread (epistemic) with observation-noise (aleatoric) in quadrature."""
    return np.sqrt(epistemic_std ** 2 + np.asarray(aleatoric_h)[None, :] ** 2)


def _gaussian_metrics(y_true, mean, std, coverage):
    """PICP / MPIW(deg) / CRPS(deg) / point-RMSE(deg) + reliability for a Gaussian forecast."""
    lo, hi = cal.gaussian_interval(mean, std, coverage)
    nominal, empirical = cal.reliability_curve(y_true.ravel(), mean.ravel(), std.ravel(),
                                               RELIABILITY_LEVELS)
    return {
        "picp": cal.picp(y_true.ravel(), lo.ravel(), hi.ravel()),
        "mpiw_deg": _deg(cal.mpiw(lo, hi)),
        "crps_deg": _deg(np.mean(cal.crps_gaussian(y_true.ravel(), mean.ravel(), std.ravel()))),
        "point_rmse_deg": _deg(np.sqrt(np.mean((mean - y_true) ** 2))),
    }, (nominal, empirical)


def run_uq(cfg, smoke=False, seed=None, ensemble_seeds=None, out_dir=None, arrays=None):
    seed = cfg.get("seed", 0) if seed is None else seed
    out_dir = Path(out_dir) if out_dir else cfgmod.RESULTS_DIR / ("uq_smoke" if smoke else "uq")
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    coverage = cfg.get("coverage", 0.90)

    set_seed(seed)
    d = arrays if arrays is not None else build_id_arrays()
    y_true = d["y_test"]                                 # (N, H) radians
    y_val = d["y_val"]
    rows, reliability = [], []

    # --- MC-Dropout -------------------------------------------------------------
    # epistemic std from T dropout passes (test) + aleatoric std from validation residuals.
    set_seed(seed)
    model, ysc = fit_neural({**cfg, "model": "lstm", "seed": seed}, d, smoke=smoke)
    mean, epi = mc_dropout_predict(model, d["X_test"], T=(5 if smoke else cfg.get("mc_passes", 30)),
                                   device=DEVICE, target_scaler=ysc)
    aleat = _aleatoric_std(model.predict(d["X_val"], device=DEVICE, target_scaler=ysc), y_val)
    std = _total_std(epi, aleat)
    m, rel = _gaussian_metrics(y_true, mean, std, coverage)
    rows.append({"method": "MC-Dropout", **m})
    reliability.append(("MC-Dropout", *rel))
    np.savez(preds_dir / "mc_dropout.npz", y_true=y_true, mean=mean, std=std, epistemic=epi)

    # --- Deep Ensemble ----------------------------------------------------------
    if ensemble_seeds is None:
        ensemble_seeds = ([seed, seed + 1] if smoke
                          else cfg.get("ensemble_seeds", list(cfgmod.SEEDS)))
    seeds = ensemble_seeds
    members, members_val = [], []
    for s in seeds:
        set_seed(s)
        mem, ysc_m = fit_neural({**cfg, "model": "lstm", "seed": s}, d, smoke=smoke)
        members.append(mem.predict(d["X_test"], device=DEVICE, target_scaler=ysc_m))
        members_val.append(mem.predict(d["X_val"], device=DEVICE, target_scaler=ysc_m))
    mean, epi = ensemble_moments(members)
    aleat = _aleatoric_std(np.mean(members_val, axis=0), y_val)
    std = _total_std(epi, aleat)
    m, rel = _gaussian_metrics(y_true, mean, std, coverage)
    rows.append({"method": "Deep Ensemble", **m})
    reliability.append(("Deep Ensemble", *rel))
    np.savez(preds_dir / "ensemble.npz", y_true=y_true, mean=mean, std=std, epistemic=epi)

    # --- Quantile Regression ----------------------------------------------------
    taus = tuple(cfg.get("taus", (0.05, 0.5, 0.95)))
    set_seed(seed)
    qmodel, ysc_q = fit_quantile({**cfg, "seed": seed}, d, taus, smoke=smoke)
    q = qmodel.predict_quantiles(d["X_test"], device=DEVICE, target_scaler=ysc_q)   # (N,H,Q)
    lo, med, hi = q[:, :, 0], q[:, :, taus.index(0.5)], q[:, :, -1]
    crps = np.mean(cal.crps_from_quantiles(y_true.ravel(), q.reshape(-1, len(taus)), np.array(taus)))
    rows.append({
        "method": "Quantile", "picp": cal.picp(y_true.ravel(), lo.ravel(), hi.ravel()),
        "mpiw_deg": _deg(cal.mpiw(lo, hi)), "crps_deg": _deg(crps),
        "point_rmse_deg": _deg(np.sqrt(np.mean((med - y_true) ** 2))),
    })
    # quantile gives one central interval -> a single reliability point at its nominal coverage
    q_nominal = taus[-1] - taus[0]                       # e.g. 0.95 - 0.05 = 0.90
    reliability.append(("Quantile", np.array([q_nominal]),
                        np.array([cal.picp(y_true.ravel(), lo.ravel(), hi.ravel())])))
    np.savez(preds_dir / "quantile.npz", y_true=y_true, quantiles=q, taus=np.array(taus))

    # --- tables -----------------------------------------------------------------
    t5 = pd.DataFrame(rows)[["method", "picp", "mpiw_deg", "crps_deg", "point_rmse_deg"]]
    t5.to_csv(out_dir / "T5_uq.csv", index=False)
    rel_rows = []
    for name, nominal, empirical in reliability:
        for nom, emp in zip(np.atleast_1d(nominal), np.atleast_1d(empirical)):
            rel_rows.append({"method": name, "nominal": float(nom), "empirical": float(emp)})
    pd.DataFrame(rel_rows).to_csv(out_dir / "reliability.csv", index=False)
    return out_dir, t5


T5_METRICS = ["picp", "mpiw_deg", "crps_deg", "point_rmse_deg"]


def run_uq_multiseed(cfg, seeds=(0, 1, 2), smoke=False):
    """Run the full UQ comparison once per seed and report mean +/- std (S1).

    Each seed re-trains the MC-Dropout/Quantile models and a disjoint deep ensemble (member
    seeds offset per table seed), writing its predictive distributions to <uq>/seed<s>/preds so
    the early-warning stage can be aggregated the same way. Emits T5_uq_by_seed.csv (raw),
    T5_uq_meanstd.csv (mean +/- std for the manuscript), and T5_uq.csv (means, for plot_F6)."""
    base = cfgmod.RESULTS_DIR / ("uq_smoke" if smoke else "uq")
    n_members = 2 if smoke else len(cfg.get("ensemble_seeds", list(cfgmod.SEEDS)))
    per_seed = []
    for s in seeds:
        ens = [s * 100 + i for i in range(n_members)]    # disjoint ensemble members per table seed
        _, t5 = run_uq(cfg, smoke=smoke, seed=s, ensemble_seeds=ens, out_dir=base / f"seed{s}")
        per_seed.append(t5.assign(seed=s))
    raw = pd.concat(per_seed, ignore_index=True)
    raw.to_csv(base / "T5_uq_by_seed.csv", index=False)
    meanstd = stats.mean_std_over_seeds(raw, ["method"], T5_METRICS)
    meanstd.to_csv(base / "T5_uq_meanstd.csv", index=False)
    stats.mean_only(meanstd, T5_METRICS).to_csv(base / "T5_uq.csv", index=False)
    return base, meanstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="jmse/run_configs/uq_id.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="run the UQ table over these seeds and report mean+/-std; omit for single-seed")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"device={DEVICE}  UQ run  smoke={args.smoke}  seeds={args.seeds}")
    if args.seeds and len(args.seeds) > 1:
        out_dir, ms = run_uq_multiseed(cfg, seeds=args.seeds, smoke=args.smoke)
        print(ms.to_string(index=False))
        print(f"wrote {out_dir / 'T5_uq_meanstd.csv'}")
        return
    out_dir, t5 = run_uq(cfg, smoke=args.smoke)
    print(t5.to_string(index=False))
    print(f"wrote {out_dir / 'T5_uq.csv'}")
    if not args.no_plots:
        from jmse.plots.uq import plot_F6
        print("figure:", plot_F6(out_dir))


if __name__ == "__main__":
    main()
