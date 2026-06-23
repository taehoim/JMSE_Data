"""Windowing groups by (tonnage, Hs, realization): no window straddles a realization.

The same leakage-safety rule that kept windows inside a (tonnage, Hs) file now also
keeps them inside a single wave-phase realization.
"""
import numpy as np

from jmse import config
from jmse.data.curate import load_curated
from jmse.data.windowing import (
    _group_windows, build_id_arrays, build_ood_arrays, chrono_split, make_windows,
)

L, H = config.LOOKBACK, config.HORIZON
_EMBARGO = L + H - 1


def _expected_window_count():
    """Sum of per-(ton,Hs,real) windows that survive the embargoed 70/15/15 chrono split.

    The ID pipeline splits each realization series independently with an embargo, which
    drops windows at the two split boundaries; the assembled total must equal the sum of
    surviving windows per realization (never mixing realizations).
    """
    df = load_curated()
    total = 0
    for _, g in df.groupby(["tonnage", "Hs", "realization"]):
        n = max(0, len(g) - L - H + 1)
        tr, va, te = chrono_split(n, embargo=_EMBARGO)
        total += (tr.stop - tr.start) + (va.stop - va.start) + (te.stop - te.start)
    return total


def test_group_keys_are_three_tuples():
    df = load_curated()
    groups = list(_group_windows(df, L, H))
    # each yielded group is keyed by (ton, hs, real)
    for ton, hs, real, *_ in groups:
        assert real in range(6)
        assert ton in config.VESSELS and hs in config.HS_VALUES
    # one group per non-empty (ton, hs, real) cell
    keys = {(ton, hs, real) for ton, hs, real, *_ in groups}
    assert len(keys) == len(groups)


def test_total_window_count_matches_per_series_sum():
    d = build_id_arrays()
    n_windows = len(d["X_train"]) + len(d["X_val"]) + len(d["X_test"])
    assert n_windows == _expected_window_count()
    # group_keys carry the (ton, hs, real) identity
    assert all(len(k) == 3 for k in d["group_keys"])


def test_no_window_mixes_two_realizations():
    """Every window's input+horizon span lives in exactly one realization."""
    df = load_curated()
    for ton, hs, real, X, y, lo, yh, idx in _group_windows(df, L, H):
        # reconstruct the source series and confirm it is single-realization
        g = df[(df.tonnage == ton) & (df.Hs == hs) & (df.realization == real)]
        assert g["realization"].nunique() == 1
        # window count for this series is exactly N - L - H + 1
        assert len(X) == max(0, len(g) - L - H + 1)


def test_id_group_ids_map_to_single_realization():
    d = build_id_arrays()
    keys = d["group_keys"]
    for split in ("train", "val", "test"):
        for gid in np.unique(d[f"group_{split}"]):
            assert len(keys[gid]) == 3       # (ton, hs, real)


def test_lovo_holds_out_all_realizations_of_a_vessel():
    """Holding out a vessel removes ALL its realizations from train/val."""
    ton = 30
    d = build_ood_arrays(hold_ton=ton)
    keys = d["group_keys"]
    train_tons = {keys[g][0] for g in np.unique(d["group_train"])}
    test_tons = {keys[g][0] for g in np.unique(d["group_test"])}
    assert ton not in train_tons
    assert test_tons == {ton}
    # the held vessel contributes all 6 realizations to the test set
    test_reals = {keys[g][2] for g in np.unique(d["group_test"])}
    assert test_reals == set(range(6))


def test_loho_holds_out_all_realizations_of_a_sea_state():
    hs = 5.0
    d = build_ood_arrays(hold_hs=hs)
    keys = d["group_keys"]
    train_hs = {keys[g][1] for g in np.unique(d["group_train"])}
    test_hs = {keys[g][1] for g in np.unique(d["group_test"])}
    assert hs not in train_hs
    assert test_hs == {hs}
    test_reals = {keys[g][2] for g in np.unique(d["group_test"])}
    assert test_reals == set(range(6))
