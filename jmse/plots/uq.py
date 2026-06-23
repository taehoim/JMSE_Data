"""F6: uncertainty figure — reliability diagram + prediction intervals (C1).

(a) Reliability diagram: empirical vs nominal central coverage for each UQ method
    (Gaussian methods as curves, the quantile method as its single 90% point); the
    diagonal is perfect calibration.
(b) Prediction intervals: a slice of the t+1 test set showing the 90% interval and the
    point/median forecast against the truth, for the best-calibrated Gaussian method.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jmse.uq import calibration as cal
from jmse.plots.style import add_grid


def _best_gaussian(preds_dir: Path, coverage=0.90):
    """Pick the Gaussian method whose 90% PICP is closest to nominal, for panel (b)."""
    best, best_gap = None, np.inf
    for name in ("ensemble", "mc_dropout"):
        p = preds_dir / f"{name}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        lo, hi = cal.gaussian_interval(z["mean"], z["std"], coverage)
        gap = abs(cal.picp(z["y_true"].ravel(), lo.ravel(), hi.ravel()) - coverage)
        if gap < best_gap:
            best, best_gap = (name, z), gap
    return best


def plot_F6(out_dir: Path, coverage=0.90, n_show=120) -> Path:
    out_dir = Path(out_dir)
    rel = pd.read_csv(out_dir / "reliability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) reliability diagram
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
    for method, sub in rel.groupby("method"):
        sub = sub.sort_values("nominal")
        style = dict(marker="o")
        if len(sub) == 1:                               # quantile -> single point
            style = dict(marker="D", markersize=9, linestyle="none")
        axes[0].plot(sub["nominal"], sub["empirical"], label=method, **style)
    axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("(a) Reliability diagram")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal", "box")
    add_grid(axes[0])

    # (b) prediction intervals for the best-calibrated Gaussian method (t+1)
    best = _best_gaussian(out_dir / "preds", coverage)
    if best is not None:
        name, z = best
        mean, std, y = z["mean"][:, 0], z["std"][:, 0], z["y_true"][:, 0]
        lo, hi = cal.gaussian_interval(mean, std, coverage)
        idx = np.arange(min(n_show, len(y)))
        axes[1].fill_between(idx, np.degrees(lo[idx]), np.degrees(hi[idx]),
                             alpha=0.3, label=f"{int(coverage*100)}% interval")
        axes[1].plot(idx, np.degrees(mean[idx]), lw=1, label="forecast")
        axes[1].plot(idx, np.degrees(y[idx]), lw=1, color="k", label="truth")
        axes[1].set_xlabel("Test window (t+1 s)")
        axes[1].set_ylabel("Total inclination angle (deg)")
        label = {"ensemble": "deep ensembles", "mc_dropout": "MC-dropout"}.get(name, name)
        axes[1].set_title(f"(b) {int(coverage*100)}% prediction intervals ({label})")
        add_grid(axes[1])
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "F6_uncertainty.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
