"""UQ-rigor orchestrator (P5, C1): conditional coverage, conformal calibration, ECE/Brier.

Consumes the per-seed predictive distributions already written by `jmse.uq.run --seeds ...`
(results/uq/seed<s>/preds/{quantile,ensemble}.npz) and scores three rigor questions the
marginal PICP of Table 5 cannot answer:

  T9  conditional coverage  -- is the 90% quantile band calibrated within each horizon,
      sea state, tonnage, and (crucially) the rare above-threshold regime, or only on average?
  T10 conformal calibration -- does a distribution-free wrapper (split conformal on the
      ensemble mean; CQR on the quantile band) reach nominal coverage, and at what width?
      Calibrated on the earlier temporal half of each test group, evaluated on the later half.
  T11 exceedance-probability calibration -- ECE and Brier of P(theta_TIA(t+k) > theta_d), the
      quantity the alarm thresholds, for each danger threshold.

Everything is computed from saved arrays (no model training), then aggregated mean +/- std
over seeds. Usage:  python -m jmse.uq.rigor [--seeds 0 1 2 3 4] [--smoke]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jmse import config as cfgmod
from jmse.data.windowing import build_id_arrays
from jmse.earlywarning import alarm
from jmse.uq import calibration as cal
from jmse.uq import conditional as cond
from jmse.uq import conformal as cf
from jmse.uq import probscore as ps

ALPHA = 0.10                      # 90% nominal coverage, matching Table 5


def _keys_for(groups, group_keys, which):
    """Map per-window group ids to their tonnage (which=0) or Hs (which=1) value."""
    lut = np.array([gk[which] for gk in group_keys], float)
    return lut[np.asarray(groups, int)]


def compute_conditional(quant, taus, y_true, groups, group_keys, thetas_rad):
    """Conditional coverage of the nominal-90% quantile band [q_lo, q_hi]."""
    lo, hi = quant[:, :, 0], quant[:, :, -1]
    ton = _keys_for(groups, group_keys, 0)
    hs = _keys_for(groups, group_keys, 1)
    frames = [
        cond.coverage_by_horizon(y_true, lo, hi).assign(dim="horizon"),
        cond.coverage_by_key(y_true, lo, hi, hs, "value").assign(dim="Hs"),
        cond.coverage_by_key(y_true, lo, hi, ton, "value").assign(dim="tonnage"),
    ]
    # marginal + above/below the lowest danger threshold (the alarm-relevant tail)
    cov, w = cond.coverage_width(y_true, lo, hi)
    frames.append(pd.DataFrame([{"dim": "marginal", "value": np.nan, "coverage": cov,
                                 "mpiw_deg": np.degrees(w), "n": y_true.size}]))
    reg = cond.coverage_by_regime(y_true, lo, hi, thetas_rad[0])
    reg = reg.rename(columns={"regime": "value"}).assign(dim="regime",
                                                         mpiw_deg=np.nan)[["dim", "value", "coverage", "mpiw_deg", "n"]]
    frames.append(reg)
    out = pd.concat([f.rename(columns={"horizon_s": "value"}) if "horizon_s" in f else f
                     for f in frames], ignore_index=True)
    out["value"] = out["value"].astype(str)
    return out[["dim", "value", "coverage", "mpiw_deg", "n"]]


def compute_conformal(quant, taus, mean, std, y_true, groups):
    """Parametric vs distribution-free coverage/width on the eval half of each group."""
    cal_m = cf.first_half_mask(groups)
    eval_m = ~cal_m
    qlo, qhi = quant[:, :, 0], quant[:, :, -1]
    g_lo, g_hi = cal.gaussian_interval(mean, std, 1 - ALPHA)
    rows = []

    def _row(name, lo, hi):
        c, w = cond.coverage_width(y_true[eval_m], lo[eval_m], hi[eval_m])
        rows.append({"method": name, "coverage": c, "mpiw_deg": np.degrees(w)})

    _row("Quantile band", qlo, qhi)
    lo_cqr, hi_cqr = cf.cqr(qlo[cal_m], qhi[cal_m], y_true[cal_m], qlo, qhi, ALPHA)
    _row("Quantile + CQR", lo_cqr, hi_cqr)
    _row("Ensemble Gaussian", g_lo, g_hi)
    lo_sc, hi_sc = cf.split_conformal(mean[cal_m], y_true[cal_m], mean, ALPHA)
    _row("Ensemble + split-conformal", lo_sc, hi_sc)
    out = pd.DataFrame(rows)
    out["n_eval"] = int(eval_m.sum())
    return out


def compute_probcalib(quant, taus, mean, std, y_true, thetas_rad, thetas_deg):
    """ECE and Brier of the exceedance probability P(y > theta) for each danger threshold."""
    rows = []
    for td, th in zip(thetas_deg, thetas_rad):
        label = (y_true > th).astype(int).ravel()
        p_q = alarm.prob_exceed_quantile(quant, taus, th).ravel()
        p_g = alarm.prob_exceed_gaussian(mean, std, th).ravel()
        for name, p in (("Quantile", p_q), ("Ensemble Gaussian", p_g)):
            rows.append({"threshold_deg": td, "prob_source": name, "prevalence": float(label.mean()),
                         "brier": ps.brier_score(p, label),
                         "ece": ps.expected_calibration_error(p, label, n_bins=10)})
    return pd.DataFrame(rows)


def compute_rigor(quant, taus, mean, std, y_true, groups, group_keys, thetas_rad, thetas_deg):
    """All three rigor frames for one seed (pure; arrays in, DataFrames out)."""
    return {
        "conditional": compute_conditional(quant, taus, y_true, groups, group_keys, thetas_rad),
        "conformal": compute_conformal(quant, taus, mean, std, y_true, groups),
        "probcalib": compute_probcalib(quant, taus, mean, std, y_true, thetas_rad, thetas_deg),
    }


def _agg(frames, group_cols, metric_cols):
    """Concatenate per-seed frames and report mean/std over seeds for the metric columns."""
    raw = pd.concat(frames, ignore_index=True)
    g = raw.groupby(group_cols, sort=False)
    agg = g[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m in metric_cols for s in ("mean", "std")]
    for m in metric_cols:
        agg[f"{m}_std"] = agg[f"{m}_std"].fillna(0.0)
    extra = [c for c in raw.columns if c not in group_cols + metric_cols + ["seed"]]
    if extra:
        agg = agg.join(g[extra].first())
    return agg.reset_index()


def run_rigor(seeds=(0, 1, 2, 3, 4), smoke=False, out_dir=None):
    base = cfgmod.RESULTS_DIR / ("uq_smoke" if smoke else "uq")
    out_dir = Path(out_dir) if out_dir else base
    d = build_id_arrays()
    y_true, groups, gk = d["y_test"], d["group_test"], d["group_keys"]
    thr_rad, thr_deg = cfgmod.DANGER_THRESHOLDS_RAD, cfgmod.DANGER_THRESHOLDS_DEG

    per = {"conditional": [], "conformal": [], "probcalib": []}
    for s in seeds:
        sdir = base / f"seed{s}" / "preds"
        zq = np.load(sdir / "quantile.npz")
        ze = np.load(sdir / "ensemble.npz")
        r = compute_rigor(zq["quantiles"], zq["taus"], ze["mean"], ze["std"],
                          y_true, groups, gk, thr_rad, thr_deg)
        for k in per:
            per[k].append(r[k].assign(seed=s))

    cond_t = _agg(per["conditional"], ["dim", "value"], ["coverage", "mpiw_deg"])
    conf_t = _agg(per["conformal"], ["method"], ["coverage", "mpiw_deg"])
    prob_t = _agg(per["probcalib"], ["threshold_deg", "prob_source"], ["brier", "ece"])
    cond_t.to_csv(out_dir / "T9_conditional_coverage.csv", index=False)
    conf_t.to_csv(out_dir / "T10_conformal.csv", index=False)
    prob_t.to_csv(out_dir / "T11_probcalib.csv", index=False)
    return out_dir, cond_t, conf_t, prob_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir, cond_t, conf_t, prob_t = run_rigor(seeds=args.seeds, smoke=args.smoke)
    print("== conditional coverage =="); print(cond_t.to_string(index=False))
    print("== conformal =="); print(conf_t.to_string(index=False))
    print("== prob calibration =="); print(prob_t.to_string(index=False))
    print(f"wrote T9/T10/T11 under {out_dir}")


if __name__ == "__main__":
    main()
