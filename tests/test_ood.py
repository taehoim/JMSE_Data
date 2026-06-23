import numpy as np
import pandas as pd

from jmse import config
from jmse.data.windowing import build_ood_arrays
from jmse.eval.ood import _build_t7, generalization_folds


def _hs_of(d, split):
    keys = d["group_keys"]
    return {keys[g][1] for g in np.unique(d[f"group_{split}"])}


def _ton_of(d, split):
    keys = d["group_keys"]
    return {keys[g][0] for g in np.unique(d[f"group_{split}"])}


def test_generalization_folds_structure():
    folds = generalization_folds()
    regimes = [f["regime"] for f in folds]
    assert regimes.count("loho") == len(config.HS_VALUES)        # one per sea state
    assert regimes.count("lovo") == len(config.VESSELS)          # one per tonnage
    # every LOHO fold names a held Hs; every LOVO fold a held tonnage
    assert {f["hold_hs"] for f in folds if f["regime"] == "loho"} == set(config.HS_VALUES)
    assert {f["hold_ton"] for f in folds if f["regime"] == "lovo"} == set(config.VESSELS)


def test_loho_leaves_out_held_sea_state():
    hs = 5.0
    d = build_ood_arrays(hold_hs=hs)
    assert _hs_of(d, "test") == {hs}                             # test is exactly the held Hs
    assert hs not in _hs_of(d, "train")                          # absent from train (no leakage)
    assert hs not in _hs_of(d, "val")
    # all five tonnages appear in the held-out test (held Hs spans every vessel)
    assert _ton_of(d, "test") == set(config.VESSELS)


def test_lovo_leaves_out_held_tonnage():
    ton = 30
    d = build_ood_arrays(hold_ton=ton)
    assert _ton_of(d, "test") == {ton}
    assert ton not in _ton_of(d, "train")
    assert ton not in _ton_of(d, "val")
    assert _hs_of(d, "test") == set(config.HS_VALUES)


def test_ood_test_windows_contiguous_per_group():
    # required so OOD early-warning / time-ordered analysis stays valid
    d = build_ood_arrays(hold_ton=30)
    g, t = d["group_test"], d["tidx_test"]
    for gid in np.unique(g):
        assert np.all(np.diff(t[g == gid]) == 1)


def _overall(model, scope, r2, rmse=5.0):
    return {"model": model, "scope": scope, "seed": 0, "horizon_s": "overall",
            "rmse_deg": rmse, "mae_deg": rmse * 0.7, "r2": r2}


def test_build_t7_scope_aggregation_and_delta():
    raw = pd.DataFrame([
        _overall("ID", "ID", 0.60),
        _overall("Hs=3", "LOHO", 0.30), _overall("Hs=5", "LOHO", 0.50),   # LOHO mean 0.40
        _overall("10t", "LOVO", 0.55), _overall("50t", "LOVO", 0.45),     # LOVO mean 0.50
    ])
    t7 = _build_t7(raw)
    assert list(t7["scope"]) == ["ID", "LOHO", "LOVO"]                    # ordered ID->LOHO->LOVO
    loho = t7[t7.scope == "LOHO"].iloc[0]
    assert abs(loho["r2_mean"] - 0.40) < 1e-9
    assert abs(loho["delta_r2_vs_id"] - (0.40 - 0.60)) < 1e-9             # degradation vs ID
    assert int(loho["n_folds"]) == 2
    assert abs(t7[t7.scope == "ID"].iloc[0]["delta_r2_vs_id"]) < 1e-12    # ID vs itself = 0


def test_combined_holdout_trains_on_neither_axis_tests_only_corner():
    from jmse.data.windowing import build_combined_holdout
    from jmse.eval.ood import combined_holdout_folds
    hs, ton = 7.0, 10
    d = build_combined_holdout(hold_hs=hs, hold_ton=ton)
    # test is exactly the single joint cell (projecting away the realization axis:
    # group_keys are (ton, hs, real), and the held cell spans all 6 realizations)
    keys = d["group_keys"]
    test_cells = {keys[g][:2] for g in np.unique(d["group_test"])}
    assert test_cells == {(ton, hs)}
    # training sees NEITHER the held sea state NOR the held vessel
    assert hs not in _hs_of(d, "train") and hs not in _hs_of(d, "val")
    assert ton not in _ton_of(d, "train") and ton not in _ton_of(d, "val")
    # the 8 neither-axis cells form train/val (4 tonnages x 2 sea states)
    assert _ton_of(d, "train") == set(config.VESSELS) - {ton}
    assert _hs_of(d, "train") == set(config.HS_VALUES) - {hs}


def test_combined_holdout_folds_are_corners_plus_center():
    from jmse.eval.ood import combined_holdout_folds
    folds = combined_holdout_folds()
    kinds = [f["kind"] for f in folds]
    assert kinds.count("corner") == 4 and kinds.count("center") == 1
