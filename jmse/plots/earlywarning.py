"""Early-warning figures F7 (ROC/PR) and F8 (effective horizon + lead time) for C2.

Both read the orchestrator's self-contained outputs in results/earlywarning[_smoke]/:
  scores.npz (y_true, groups, point, prob_<thetadeg>) and per_horizon_auc.csv.
F7 contrasts the probabilistic and point alarms' ROC and PR curves at theta=15 deg.
F8 shows AUC decay with horizon (with the AUC=0.8 effective-horizon line) and the warning
lead-time distribution of the probabilistic alarm at a matched false-alarm rate.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jmse import config
from jmse.earlywarning import alarm, leadtime, roc
from jmse.plots.style import add_grid

_HEAD_DEG = config.DANGER_THRESHOLDS_DEG[0]              # headline threshold (15 deg)
_HEAD_RAD = config.DANGER_THRESHOLDS_RAD[0]


def _load(out_dir: Path):
    z = np.load(Path(out_dir) / "scores.npz")
    return z, alarm.exceedance_labels(z["y_true"], _HEAD_RAD)


def plot_F7(out_dir: Path, method: str = "") -> Path:
    out_dir = Path(out_dir)
    z, label = _load(out_dir)
    lab = label.ravel()
    prob = z[f"prob_{int(_HEAD_DEG)}"].ravel()
    point = z["point"].ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, score in (("probabilistic", prob), ("point", point)):
        fpr, tpr = roc.roc_points(lab, score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc.roc_auc(lab, score):.3f})")
        rec, prec = roc.pr_points(lab, score)
        axes[1].plot(rec, prec, label=f"{name} (AP={roc.pr_auc(lab, score):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
    axes[0].set_xlabel("False-alarm rate"); axes[0].set_ylabel("Detection rate")
    axes[0].set_title(rf"(a) ROC — $\theta_{{\mathrm{{TIA}}}}>{int(_HEAD_DEG)}^\circ$"); axes[0].legend(fontsize=8)
    add_grid(axes[0])
    axes[1].axhline(lab.mean(), ls="--", lw=0.8, color="grey", label="prevalence")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Precision-Recall"); axes[1].legend(fontsize=8)
    add_grid(axes[1])
    fig.tight_layout()
    path = out_dir / "F7_roc_pr.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def plot_F8(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    z, label = _load(out_dir)
    per_h = pd.read_csv(out_dir / "per_horizon_auc.csv")
    per_h = per_h[per_h["threshold_deg"] == _HEAD_DEG]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # (a) AUC vs horizon with the effective-horizon line
    for name, sub in per_h.groupby("alarm"):
        sub = sub.sort_values("horizon_s")
        axes[0].plot(sub["horizon_s"], sub["roc_auc"], marker="o",
                     label="probabilistic" if name == "prob" else "point")
    axes[0].axhline(0.8, ls="--", lw=0.9, color="red", label="AUC = 0.8")
    axes[0].set_xlabel("Forecast horizon (s)"); axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title(rf"(a) Effective horizon — $\theta_{{\mathrm{{TIA}}}}>{int(_HEAD_DEG)}^\circ$")
    axes[0].legend(fontsize=8)
    add_grid(axes[0])

    # (b) lead-time distribution of both alarms at a shared false-alarm budget
    prob = z[f"prob_{int(_HEAD_DEG)}"]
    point = z["point"]
    budget = config.EARLY_WARNING_FPR
    lab1d = label.ravel()
    leads = {}
    for name, score in (("prob", prob), ("point", point)):
        alpha = (roc.operating_point_at_fpr(lab1d, score.ravel(), budget)["threshold"]
                 if lab1d.any() and (~lab1d).any() else np.inf)
        leads[name] = leadtime.lead_times(z["y_true"], score, z["groups"], _HEAD_RAD, alpha)
    bins = np.arange(0.5, config.HORIZON + 1.5, 1.0)
    if leads["prob"]["leads"].size:
        axes[1].hist(leads["prob"]["leads"], bins=bins, alpha=0.75, color="#4C72B0", label="prob events")
    for name, ls, color in (("prob", "--", "k"), ("point", ":", "grey")):
        m = leads[name]["mean_lead_s"]
        if not np.isnan(m):
            axes[1].axvline(m, ls=ls, color=color, label=f"{name} mean={m:.2f}s")
    axes[1].set_xlabel("Warning lead time (s)"); axes[1].set_ylabel("Detected events")
    axes[1].set_title(f"(b) Lead time at FPR budget={budget:.2f}")
    add_grid(axes[1], axis="y")
    handles, _ = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "F8_leadtime.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
