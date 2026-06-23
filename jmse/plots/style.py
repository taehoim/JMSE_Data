"""Shared plotting style helpers for manuscript figures."""


def add_grid(ax, axis: str = "both") -> None:
    """Add a light manuscript-friendly grid behind plotted data."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis=axis, linestyle=":", linewidth=0.6, alpha=0.65)
