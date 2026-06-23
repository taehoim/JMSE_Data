"""F9: generalization heatmap (C4) — R^2 per held-out fold x forecast horizon.

Rows are the ID baseline then each leave-one-out fold (sea states, then tonnages);
columns are the forecast horizons plus the overall column. Cells show mean R^2 over
seeds, so the vertical gap from the ID row visualizes the generalization penalty and
the left-to-right fade shows skill decay with horizon.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jmse import config
from jmse.eval.ood import generalization_folds


def _row_order():
    return ["ID"] + [f["label"] for f in generalization_folds()]


def plot_F9(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    per_fold = pd.read_csv(out_dir / "per_fold.csv")
    cols = [str(h) for h in range(1, config.HORIZON + 1)] + ["overall"]
    per_fold["horizon_s"] = per_fold["horizon_s"].astype(str)

    grid = (per_fold.pivot_table(index="model", columns="horizon_s", values="r2_mean")
            .reindex(index=[r for r in _row_order() if r in set(per_fold["model"])],
                     columns=cols))
    M = grid.to_numpy(float)

    # RdBu diverging map centered at 0 (R^2=0 => no skill): red = negative skill (failure), blue = positive.
    # Colorblind-safe (red-blue, not the banned red-green); the full negative range is shown without
    # clipping (vmin tracks the data minimum), and per-cell numbers keep it legible in grayscale.
    from matplotlib.colors import TwoSlopeNorm
    cmap = plt.get_cmap("RdBu")
    vmin = min(float(np.nanmin(M)), -0.05)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3, 0.5 * len(grid) + 2))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"t+{c}" if c != "overall" else "overall" for c in cols])
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(grid.index)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                r, g, b, _ = cmap(norm(M[i, j]))         # white text on dark cells, black on light
                txt = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color=txt)
    # separators: after ID row and after the LOHO block
    n_loho = sum(f["scope"] == "LOHO" for f in generalization_folds())
    for y in (0.5, 0.5 + n_loho):
        ax.axhline(y, color="black", lw=1.2)
    ax.set_xlabel("Forecast horizon (s)")
    ax.set_title("Generalization: R$^2$ by held-out fold and horizon")
    fig.colorbar(im, ax=ax, label="R$^2$ (mean over seeds)")
    fig.tight_layout()
    path = out_dir / "F9_generalization.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
