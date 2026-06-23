"""F12: UQ-rigor figure — conditional coverage and conformal calibration (P5, C1).

(a) coverage of the 90% quantile band vs forecast horizon, with the nominal line;
(b) coverage by condition (sea state Hs) and by inclination regime (below/above 15 deg),
    exposing the rare large-heel tail under-coverage that the marginal PICP hides;
(c) conformal correction: parametric vs distribution-free coverage for both methods,
    showing split-conformal / CQR pull empirical coverage onto the nominal target.

Reads the T9/T10 CSVs produced by jmse.uq.rigor.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from jmse.plots.style import add_grid

NOMINAL = 0.90


def plot_F12(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    cond = pd.read_csv(out_dir / "T9_conditional_coverage.csv")
    conf = pd.read_csv(out_dir / "T10_conformal.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) coverage vs horizon
    h = cond[cond["dim"] == "horizon"].copy()
    h["value"] = h["value"].astype(float)
    h = h.sort_values("value")
    axes[0].axhline(NOMINAL, ls="--", color="grey", lw=1, label="nominal 0.90")
    axes[0].errorbar(h["value"], h["coverage_mean"], yerr=h["coverage_std"],
                     marker="o", capsize=3)
    axes[0].set_xlabel("Forecast horizon (s)")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("(a) Coverage vs horizon")
    axes[0].set_ylim(0.78, 0.98)
    add_grid(axes[0])
    axes[0].legend(fontsize=8)

    # (b) coverage by Hs and by regime
    hs = cond[cond["dim"] == "Hs"].copy()
    reg = cond[cond["dim"] == "regime"].copy()
    labels = [f"Hs={v}" for v in hs["value"]] + [f"{v} 15deg" for v in reg["value"]]
    vals = list(hs["coverage_mean"]) + list(reg["coverage_mean"])
    errs = list(hs["coverage_std"]) + list(reg["coverage_std"])
    colors = ["#4C72B0"] * len(hs) + ["#55A868", "#C44E52"][: len(reg)]
    axes[1].bar(range(len(vals)), vals, yerr=errs, capsize=3, color=colors)
    axes[1].axhline(NOMINAL, ls="--", color="grey", lw=1)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("(b) Coverage by sea state and regime")
    axes[1].set_ylim(0.6, 1.0)
    add_grid(axes[1], axis="y")

    # (c) conformal correction
    order = ["Quantile band", "Quantile + CQR", "Ensemble Gaussian", "Ensemble + split-conformal"]
    conf = conf.set_index("method").reindex(order).reset_index()
    barcolors = ["#4C72B0", "#8FB0DD", "#DD8452", "#E8B894"]
    axes[2].bar(range(len(conf)), conf["coverage_mean"], yerr=conf["coverage_std"],
                capsize=3, color=barcolors)
    axes[2].axhline(NOMINAL, ls="--", color="grey", lw=1, label="nominal 0.90")
    axes[2].set_xticks(range(len(conf)))
    axes[2].set_xticklabels(["Quantile", "Quantile\n+CQR", "Ensemble", "Ensemble\n+conformal"],
                            fontsize=8)
    axes[2].set_ylabel("Coverage (eval half)")
    axes[2].set_title("(c) Distribution-free calibration")
    axes[2].set_ylim(0.85, 0.95)
    add_grid(axes[2], axis="y")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "F12_uq_rigor.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
