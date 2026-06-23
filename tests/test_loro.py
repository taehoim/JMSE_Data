"""Leave-One-Realization-Out (LORO): isolates phase-realization generalization.

A LORO fold holds out one wave-phase realization k: ALL windows with realization==k
form the test set, and train/val see no realization-k window. Crucially every (ton, Hs)
cell appears in BOTH train and test (only the realization differs), so the fold measures
generalization to an unseen wave phase rather than to an unseen vessel or sea state.
"""
import numpy as np

from jmse import config
from jmse.data.windowing import build_loro_arrays
from jmse.eval.ood import generalization_folds


def _reals_of(d, split):
    keys = d["group_keys"]
    return {keys[g][2] for g in np.unique(d[f"group_{split}"])}


def _cells_of(d, split):
    keys = d["group_keys"]
    return {keys[g][:2] for g in np.unique(d[f"group_{split}"])}


def test_loro_holds_out_exactly_the_held_realization():
    k = 2
    d = build_loro_arrays(hold_real=k)
    assert _reals_of(d, "test") == {k}            # test is exactly realization k
    assert k not in _reals_of(d, "train")         # absent from train (no leakage)
    assert k not in _reals_of(d, "val")


def test_loro_every_cell_in_both_train_and_test():
    """Every (ton, Hs) appears in train AND test; only the realization differs."""
    d = build_loro_arrays(hold_real=0)
    all_cells = {(t, h) for t in config.VESSELS for h in config.HS_VALUES}
    assert _cells_of(d, "test") == all_cells
    assert _cells_of(d, "train") == all_cells


def test_loro_test_windows_contiguous_per_group():
    d = build_loro_arrays(hold_real=3)
    g, t = d["group_test"], d["tidx_test"]
    for gid in np.unique(g):
        assert np.all(np.diff(t[g == gid]) == 1)


def test_loro_scaler_fit_and_finite():
    d = build_loro_arrays(hold_real=1)
    for k in ("X_train", "y_train", "X_test", "y_test"):
        assert not np.isnan(d[k]).any()
    # scaled train features are ~zero-mean (scaler fit on train only)
    assert abs(d["X_train"].reshape(-1, d["X_train"].shape[-1]).mean()) < 0.1


def test_generalization_folds_include_loro():
    folds = generalization_folds()
    loro = [f for f in folds if f["scope"] == "LORO"]
    assert len(loro) == 6                          # one fold per held realization
    assert {f["hold_real"] for f in loro} == set(range(6))
    assert all(f["regime"] == "loro" for f in loro)
