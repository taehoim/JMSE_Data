import numpy as np

from jmse.data.windowing import (
    make_windows, chrono_split, fit_scaler, apply_scaler, build_id_arrays,
)


def test_window_shapes_and_no_boundary_straddle():
    values = np.arange(100).reshape(-1, 1).astype(float)
    target = np.arange(100).astype(float)
    X, y, idx = make_windows(values, target, lookback=20, horizon=5)
    # correct count is N - L - H + 1 = 100 - 20 - 5 + 1 = 76 (not 75)
    assert X.shape == (76, 20, 1)
    assert y.shape == (76, 5)
    # window 0: input ends at t=19; targets are t+1..t+5
    assert X[0, -1, 0] == 19
    assert list(y[0]) == [20, 21, 22, 23, 24]
    assert idx[0] == 19 and idx[-1] == 94


def test_chrono_split_no_overlap_and_ordered():
    tr, va, te = chrono_split(100, train=0.7, val=0.15)          # embargo=0 default
    assert (tr.start, tr.stop) == (0, 70)
    assert (va.start, va.stop) == (70, 85)
    assert (te.start, te.stop) == (85, 100)


def test_chrono_split_embargo_creates_timestep_gap():
    # embargo = L+H-1 windows ensures train/val/test windows share NO timestep.
    L, H = 20, 5
    embargo = L + H - 1
    tr, va, te = chrono_split(1000, train=0.7, val=0.15, embargo=embargo)
    # last train window ends at timestep (tr.stop-1)+L+H-1; first val window starts at va.start
    assert va.start >= (tr.stop - 1) + L + H
    assert te.start >= (va.stop - 1) + L + H
    assert va.start < va.stop and te.start < te.stop             # non-empty for real group sizes


def test_scaler_fit_on_train_only():
    Xtr = np.ones((10, 20, 3))
    Xte = np.ones((4, 20, 3)) * 5.0
    sc = fit_scaler(Xtr)
    assert sc.mean_.shape == (3,)
    assert np.allclose(sc.mean_, 1.0)              # fit on TRAIN stats only
    out = apply_scaler(Xte, sc)
    assert out.shape == Xte.shape


def test_build_id_arrays_shapes_and_scaler():
    d = build_id_arrays()
    for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        assert k in d
    assert d["X_train"].shape[1:] == (20, len(__import__("jmse.config", fromlist=["FEATURES"]).FEATURES))
    assert d["y_train"].shape[1] == 5
    assert not np.isnan(d["X_train"]).any()
    # scaled train features are ~zero-mean
    assert abs(d["X_train"].reshape(-1, d["X_train"].shape[-1]).mean()) < 0.1


def test_build_id_arrays_feature_subset_and_lookback():
    # C5 ablation knobs: feature subset sets the channel count; lookback sets window length
    d = build_id_arrays(lookback=10, features=["phi", "theta"])
    assert d["X_train"].shape[1:] == (10, 2)
    assert d["y_train"].shape[1] == 5                      # horizon unchanged
    # the Xacc target is unaffected by the feature choice
    full = build_id_arrays(lookback=10)
    assert np.allclose(d["y_test"], full["y_test"])
