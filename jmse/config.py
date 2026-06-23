"""Central configuration: paths, vessel table, hydrostatic constants, windowing."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_FV_DIR = ROOT / "00_Ref" / "논문코드_데이터셋" / "fishingVessel"
# Legacy single-realization set (single wave phase, 15 files); kept for reference.
RAW_DATA_DIR = _FV_DIR / "6Dof_dataset_withXacc"
# Canonical dataset of record: 6 independent wave-phase realizations per (tonnage, Hs)
# cell, 90 files. r0 carries the published seed (R3 expansion).
RAW_DATA_DIR_MULTI = _FV_DIR / "6Dof_dataset_multi_withXacc"
# GM-floor confound study (R3.4): the 10 t roll motion RE-SIMULATED with the true
# (sub-floor) GM in the EOM restoring term -- the scientifically valid unclamped variant.
# Overridable via JMSE_UNCLAMPED_DIR; defaults to the re-sim output dir, which the sim
# domain (00_Ref) produces. Absent until that re-sim lands -> gm_floor raises a clear error.
RAW_DATA_DIR_MULTI_UNCLAMPED = Path(
    os.environ.get("JMSE_UNCLAMPED_DIR",
                   _FV_DIR / "6Dof_dataset_multi_unclampedEOM_withXacc"))
# Deprecated GZ-only artifact: same motion as clamped, GZ re-derived with the true GM.
# Kept for provenance; NOT used by the study (its Xacc is identical to the clamped set).
RAW_DATA_DIR_MULTI_UNCLAMPED_GZONLY = _FV_DIR / "6Dof_dataset_multi_unclamped_withXacc"
PROCESSED_DIR = ROOT / "data_processed"
RESULTS_DIR = ROOT / "results"

# Vessel principal dimensions (from generate6Dof_dataset.m / fishingVessel.m)
VESSELS = {  # tonnage: (L, B, T) in metres
    10: (7.93, 1.90, 0.95),
    20: (9.99, 2.40, 1.20),
    30: (11.43, 2.74, 1.37),
    40: (12.58, 3.02, 1.51),
    50: (13.56, 3.25, 1.63),
}
HS_VALUES = [3.0, 5.0, 7.0]

# Hydrostatic constants used by the simulation (MUST match fishingVessel.m)
CB = 0.68          # block coefficient
CW = 0.80          # waterplane area coefficient
KG_RATIO = 0.60    # KG = KG_RATIO * T
GM_FLOOR = 0.30    # hard clamp applied in fishingVessel.m (kept for GZ consistency)

FEATURES = ["u", "v", "w", "p", "q", "r", "phi", "theta"]
TARGET = "Xacc"            # theta_TIA (rad)
LOOKBACK = 20              # input window length (s) @ 1 Hz
HORIZON = 5                # predict t+1..t+5 (s)

DANGER_THRESHOLDS_DEG = [15, 20, 25]
DANGER_THRESHOLDS_RAD = [float(np.radians(d)) for d in DANGER_THRESHOLDS_DEG]

# Early-warning lead-time comparison: both alarms are thresholded to this shared
# false-alarm budget (FPR), so warning lead times are compared at an equal cost.
EARLY_WARNING_FPR = 0.10

SEEDS = [0, 1, 2, 3, 4]
