"""Parity gate for the MATLAB 6-DOF regeneration wrapper (DISTRIBUTIONAL).

Why distributional, not pointwise
---------------------------------
The wrapper (``regen_6dof.m``) reproduces the published record's seed and wave
realization exactly: at t=0 the two trajectories are identical and stay highly
correlated for the first ~30 s (corr(phi) > 0.99). But small-fishing-vessel roll
is a sensitive nonlinear system, so last-bit floating-point differences between
the MATLAB build that produced the published CSVs and the build used here are
amplified exponentially -- corr(phi) decays 0.998 (10 s) -> 0.93 (60 s) ->
0.45 (300 s) -> ~0 (600 s). Pointwise reproduction across machines is therefore
impossible, and that is a property of the system, not a bug in the port.

What the study actually depends on is the *distribution* of the equivalent
heeling angle Xacc = sqrt(phi^2 + theta^2) (its moments and exceedance rates),
which the regenerated record reproduces tightly. This test gates on that. The
regenerated dataset (r0..rR) is the canonical dataset of record going forward;
the chaotic sensitivity itself motivates the multi-realization + uncertainty-
aware (probabilistic) warning design (reviewer R3).

Generate the parity file (from the repo root, MATLAB R2024b on PATH as matlab.exe):

    cd "00_Ref/논문코드_데이터셋/fishingVessel" && \\
      matlab.exe -batch "addpath(genpath('../../MSS_library')); regen_6dof_parity" \\
      2>&1 | tr -d '\\r'

Then run:

    python3 -m pytest tests/test_sim_parity.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo-root-anchored paths (this file lives in <root>/tests/).
_ROOT = Path(__file__).resolve().parents[1]
_FV = _ROOT / "00_Ref" / "논문코드_데이터셋" / "fishingVessel"

# Parity output produced by regen_6dof_parity.m (10 t, Hs 5, r0).
_PARITY = _FV / "6Dof_dataset_parity" / "6Dof_10ton_L7.9_B1.9_T0.9_Hs5.0_r0.csv"
# Committed published record (same condition, original seed).
_PUBLISHED = _FV / "6Dof_dataset" / "6Dof_10ton_L7.9_B1.9_T0.9_Hs5.0.csv"

_GEN_CMD = (
    'cd "00_Ref/논문코드_데이터셋/fishingVessel" && '
    "matlab.exe -batch \"addpath(genpath('../../MSS_library')); regen_6dof_parity\" "
    "2>&1 | tr -d '\\r'"
)

# Distributional tolerances: tight enough to catch a wrong config (which would
# shift the whole distribution), loose enough to absorb the chaotic single-
# realization sampling spread between two decorrelated trajectories.
_TOL_MEAN_DEG = 0.5      # |delta mean Xacc|
_TOL_STD_DEG = 0.5       # |delta std Xacc|
_TOL_MAX_DEG = 3.0       # |delta max Xacc|
_TOL_EXCEED_PP = 2.0     # |delta exceedance rate| in percentage points


def _xacc_stats(df: pd.DataFrame) -> dict:
    x = np.degrees(np.sqrt(df["phi"].to_numpy() ** 2 + df["theta"].to_numpy() ** 2))
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "max": float(x.max()),
        "ex15": 100.0 * float((x > 15).mean()),
        "ex20": 100.0 * float((x > 20).mean()),
        "ex25": 100.0 * float((x > 25).mean()),
    }


def test_sim_parity_distribution_matches_published() -> None:
    if not _PARITY.exists():
        pytest.skip(
            "Parity output not generated yet. Run (MATLAB R2024b on PATH):\n"
            f"    {_GEN_CMD}\n(expected file: {_PARITY})"
        )
    assert _PUBLISHED.exists(), f"published reference missing: {_PUBLISHED}"

    a = _xacc_stats(pd.read_csv(_PARITY))
    b = _xacc_stats(pd.read_csv(_PUBLISHED))

    assert abs(a["mean"] - b["mean"]) <= _TOL_MEAN_DEG, f"mean Xacc {a['mean']:.2f} vs {b['mean']:.2f}"
    assert abs(a["std"] - b["std"]) <= _TOL_STD_DEG, f"std Xacc {a['std']:.2f} vs {b['std']:.2f}"
    assert abs(a["max"] - b["max"]) <= _TOL_MAX_DEG, f"max Xacc {a['max']:.2f} vs {b['max']:.2f}"
    for k in ("ex15", "ex20", "ex25"):
        assert abs(a[k] - b[k]) <= _TOL_EXCEED_PP, f"{k} {a[k]:.2f}% vs {b[k]:.2f}%"


def test_sim_parity_short_horizon_tracking() -> None:
    """Sanity: regenerated r0 shares the published wave realization (high early
    correlation), confirming seed/phase parity before chaotic divergence."""
    if not _PARITY.exists():
        pytest.skip("Parity output not generated yet; see other test for command.")
    a = pd.read_csv(_PARITY)
    b = pd.read_csv(_PUBLISHED)
    corr30 = float(np.corrcoef(a["phi"].to_numpy()[:30], b["phi"].to_numpy()[:30])[0, 1])
    assert corr30 > 0.9, f"early corr(phi) over 30 s = {corr30:.3f} (expected >0.9: same realization)"
