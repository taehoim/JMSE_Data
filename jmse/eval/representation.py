"""Target-representation study (Task 26, C5): direct vs reconstruction vs GZ-space.

Three ways to obtain an Xacc forecast, all evaluated in the SAME Xacc space (degrees) on
the SAME test windows for a fair comparison:
  - direct        : predict Xacc.
  - reconstruction: predict (phi, theta), set Xacc = sqrt(phi^2 + theta^2).
  - gz            : predict GZ, invert Xacc = arcsin(GZ / GM).
GZ is a deterministic monotone transform of Xacc (GZ = GM*sin(Xacc), GM const per vessel),
so the gz path is expected to match direct up to that transform (M5 guardrail) — it is not
independent skill. The reconstruction path additionally carries propagated angle error; the
runner reports the analytically *predicted* reconstruction RMSE (jensen bias + delta-method
variance from validation residuals) next to the observed one, closing the C5 loop.

Usage:
    python -m jmse.eval.representation [--config jmse/run_configs/lstm_id.yaml] [--smoke]
Outputs under results/representation[_smoke]/: T8_representation.csv, predicted_vs_observed.csv (+ F10).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jmse import config
from jmse.data.representation import build_representation_arrays
from jmse.eval import error_propagation as ep
from jmse.eval import stats
from jmse.eval.metrics import per_horizon_metrics, to_degrees
from jmse.models.base import set_seed
from jmse.train import DEVICE, fit_neural_on, load_config

T8_METRICS = ["rmse_deg", "mae_deg", "r2"]


def _metrics_rows(name, y_true, y_pred):
    deg = to_degrees(per_horizon_metrics(y_true, y_pred))
    rows = [{"representation": name, "horizon_s": k + 1, "rmse_deg": deg["rmse"][k],
             "mae_deg": deg["mae"][k], "r2": deg["r2"][k]} for k in range(len(deg["r2"]))]
    rows.append({"representation": name, "horizon_s": "overall", "rmse_deg": deg["overall"]["rmse"],
                 "mae_deg": deg["overall"]["mae"], "r2": deg["overall"]["r2"]})
    return rows


def run_representation(base_config="jmse/run_configs/lstm_id.yaml", smoke=False,
                       seed=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else config.RESULTS_DIR / ("representation_smoke" if smoke else "representation")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(base_config)
    if seed is not None:                                  # multi-seed driver varies the init (S1)
        set_seed(seed)
    d = build_representation_arrays()
    y_true = d["y_test"]                                  # Xacc radians

    def fit_predict(ytr_key, yva_key):
        m, ysc = fit_neural_on(cfg, d["X_train"], d[ytr_key], d["X_val"], d[yva_key], smoke=smoke)
        pred_test = m.predict(d["X_test"], device=DEVICE, target_scaler=ysc)
        pred_val = m.predict(d["X_val"], device=DEVICE, target_scaler=ysc)
        return pred_test, pred_val

    rows = []
    # direct
    yx_test, _ = fit_predict("y_train", "y_val")
    rows += _metrics_rows("direct", y_true, yx_test)

    # reconstruction: separate phi and theta forecasters
    phi_test, phi_val = fit_predict("phi_train", "phi_val")
    th_test, th_val = fit_predict("theta_train", "theta_val")
    recon_test = ep.reconstruct(phi_test, th_test)
    rows += _metrics_rows("reconstruction", y_true, recon_test)

    # gz-space: predict GZ, invert through the known per-vessel GM
    gz_test, _ = fit_predict("gz_train", "gz_val")
    gm = d["gm_test"][:, None]
    yz_test = np.arcsin(np.clip(gz_test / gm, -1.0, 1.0))
    rows += _metrics_rows("gz", y_true, yz_test)

    t8 = pd.DataFrame(rows)
    t8.to_csv(out_dir / "T8_representation.csv", index=False)

    # validate the analytic propagation against Monte-Carlo on the real test-angle geometry
    var_phi = np.mean((phi_val - d["phi_val"]) ** 2, axis=0)     # (H,) angle-error variance per horizon
    var_theta = np.mean((th_val - d["theta_val"]) ** 2, axis=0)
    pv = _propagation_validation(d, var_phi, var_theta)
    pv.to_csv(out_dir / "propagation_validation.csv", index=False)
    np.savez(out_dir / "preds.npz", y_true=y_true, direct=yx_test, recon=recon_test, gz=yz_test,
             phi=phi_test, theta=th_test)
    return out_dir, t8, pv


def _propagation_validation(d, var_phi, var_theta, min_deg=10.0, n_mc=200):
    """Analytic vs Monte-Carlo reconstruction error on the true test angles.

    Injects *unbiased* Gaussian noise of the model's per-horizon angle-error variance onto
    the true (phi, theta) and reconstructs; this isolates the propagation effect (free of the
    model's own prediction bias). Restricted to the danger regime (true Xacc > min_deg) where
    the delta method is valid (r >> noise); near zero inclination the linearization breaks
    down and over-predicts — reported honestly in the paper."""
    rng = np.random.default_rng(0)
    phi_t, theta_t, y = d["phi_test"], d["theta_test"], d["y_test"]
    H = y.shape[1]
    rows = []
    for k in range(H):
        mask = np.degrees(y[:, k]) > min_deg
        ph, th = phi_t[mask, k], theta_t[mask, k]
        if ph.size == 0:
            continue
        sphi, sth = np.sqrt(var_phi[k]), np.sqrt(var_theta[k])
        true_r = np.hypot(ph, th)
        rec = np.hypot(ph[None, :] + sphi * rng.standard_normal((n_mc, ph.size)),
                       th[None, :] + sth * rng.standard_normal((n_mc, ph.size)))
        mc_bias = float(np.degrees((rec.mean(0) - true_r).mean()))
        mc_rmse = float(np.degrees(np.sqrt(((rec - true_r[None, :]) ** 2).mean())))
        an_bias = float(np.degrees(ep.jensen_bias(ph, th, var_phi[k], var_theta[k]).mean()))
        an_rmse = float(np.degrees(np.sqrt(
            (ep.predicted_reconstruction_rmse(ph, th, var_phi[k], var_theta[k]) ** 2).mean())))
        rows.append({"horizon_s": k + 1, "n_danger": int(ph.size),
                     "analytic_bias_deg": an_bias, "mc_bias_deg": mc_bias,
                     "analytic_rmse_deg": an_rmse, "mc_rmse_deg": mc_rmse})
    return pd.DataFrame(rows)


def run_representation_multiseed(base_config="jmse/run_configs/lstm_id.yaml",
                                 seeds=(0, 1, 2), smoke=False):
    """Run the target-representation comparison once per seed and report mean +/- std (S1).

    Each seed re-trains the direct/reconstruction/GZ forecasters; per-seed artifacts go to
    <representation>/seed<s>. Emits T8_representation_by_seed.csv (raw), T8_representation_meanstd.csv
    (mean +/- std for the manuscript), and T8_representation.csv (means, for plot_F10)."""
    base = config.RESULTS_DIR / ("representation_smoke" if smoke else "representation")
    per_seed = []
    for s in seeds:
        _, t8, _ = run_representation(base_config, smoke=smoke, seed=s, out_dir=base / f"seed{s}")
        per_seed.append(t8.assign(seed=s))
    raw = pd.concat(per_seed, ignore_index=True)
    raw.to_csv(base / "T8_representation_by_seed.csv", index=False)
    meanstd = stats.mean_std_over_seeds(raw, ["representation", "horizon_s"], T8_METRICS)
    meanstd.to_csv(base / "T8_representation_meanstd.csv", index=False)
    stats.mean_only(meanstd, T8_METRICS).to_csv(base / "T8_representation.csv", index=False)
    return base, meanstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="jmse/run_configs/lstm_id.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="run the representation table over these seeds and report mean+/-std")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    if args.seeds and len(args.seeds) > 1:
        out_dir, ms = run_representation_multiseed(args.config, seeds=args.seeds, smoke=args.smoke)
        print("=== T8 mean +/- std over seeds (overall horizon) ===")
        print(ms[ms.horizon_s == "overall"].to_string(index=False))
        print(f"wrote {out_dir / 'T8_representation_meanstd.csv'}")
        return
    out_dir, t8, pv = run_representation(args.config, smoke=args.smoke)
    print("=== T8: target representation (Xacc-space, deg) ===")
    print(t8[t8.horizon_s == "overall"].to_string(index=False))
    print("\n=== propagation validation: analytic vs Monte-Carlo (danger regime) ===")
    print(pv.to_string(index=False))
    print(f"wrote {out_dir / 'T8_representation.csv'}")
    if not args.no_plots:
        from jmse.plots.representation import plot_F10
        print("figure:", plot_F10(out_dir))


if __name__ == "__main__":
    main()
