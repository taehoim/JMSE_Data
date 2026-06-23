"""F10: target representation + error propagation (C5).

(a) Inclination-space skill (R^2 and RMSE) by horizon for the direct / reconstruction / gz paths.
(b) Analytically predicted vs observed reconstruction RMSE per horizon — the delta-method
    + Jensen-bias prediction (from validation angle residuals) tracking the measured error.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from jmse.plots.style import add_grid


def plot_F10(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    t8 = pd.read_csv(out_dir / "T8_representation.csv")
    pv = pd.read_csv(out_dir / "propagation_validation.csv")
    per_h = t8[t8["horizon_s"] != "overall"].copy()
    per_h["h"] = per_h["horizon_s"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) RMSE by horizon per representation
    for name, sub in per_h.groupby("representation"):
        sub = sub.sort_values("h")
        axes[0].plot(sub["h"], sub["rmse_deg"], marker="o", label=name)
    axes[0].set_xlabel("Forecast horizon (s)")
    axes[0].set_ylabel("Total inclination RMSE (deg)")
    axes[0].set_title("(a) Skill by target representation")
    add_grid(axes[0])
    axes[0].legend(fontsize=8)

    # (b) analytic propagation vs Monte-Carlo (danger regime) — validates the theory on data
    axes[1].plot(pv["horizon_s"], pv["mc_rmse_deg"], marker="o", label="MC RMSE")
    axes[1].plot(pv["horizon_s"], pv["analytic_rmse_deg"], marker="s", ls="--", label="analytic RMSE")
    axes[1].plot(pv["horizon_s"], pv["mc_bias_deg"], marker="^", color="grey", label="MC bias")
    axes[1].plot(pv["horizon_s"], pv["analytic_bias_deg"], marker="v", ls=":", color="grey",
                 label="analytic bias (Jensen)")
    axes[1].set_xlabel("Forecast horizon (s)")
    axes[1].set_ylabel("Reconstruction error (deg)")
    axes[1].set_title("(b) Propagation: analytic vs Monte-Carlo")
    add_grid(axes[1])
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "F10_representation.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
