"""Tests for k-of-n hysteresis / evidence-accumulation alarm debounce (R3.7).

All synthetic, no GPU. The k-of-n rule: at each step t the debounced alarm latches when at
least k of the last n raw decisions (steps t-n+1..t inclusive) are True. k=1,n=1 == identity.
"""
import numpy as np
import pytest

from jmse.earlywarning import hysteresis as hy
from tests.test_events_run import _synthetic_scores


# ---- the k-of-n debounce function -------------------------------------------------------

def test_k1_n1_is_identity():
    raw = np.array([0, 1, 0, 1, 1, 0, 1], bool)
    out = hy.k_of_n(raw, k=1, n=1)
    assert out.dtype == bool
    assert out.tolist() == raw.tolist()


def test_k1_any_n_latches_for_n_minus_1_steps_after_a_fire():
    # k=1 means "any of the last n" -> latches as soon as a True is in the window, and the
    # window always includes t itself, so a True at t always fires; but it also LATCHES on
    # a recent True for n-1 further steps.
    raw = np.array([0, 1, 0, 0, 0, 0], bool)
    out = hy.k_of_n(raw, k=1, n=3)
    # t=1 fires; t=2,3 still see the t=1 True within the last 3 -> stay latched; t=4 drops it.
    assert out.tolist() == [False, True, True, True, False, False]


def test_k2_n3_latches_only_after_two_of_three():
    raw = np.array([0, 1, 0, 1, 1, 0], bool)
    out = hy.k_of_n(raw, k=2, n=3)
    # windows (last 3): t0:[0]->0; t1:[0,1]->1<2->0; t2:[0,1,0]->1<2->0;
    # t3:[1,0,1]->2>=2->1; t4:[0,1,1]->2>=2->1; t5:[1,1,0]->2>=2->1
    assert out.tolist() == [False, False, False, True, True, True]


def test_isolated_single_spikes_suppressed_at_k_ge_2():
    raw = np.zeros(12, bool)
    raw[[2, 6, 10]] = True                      # three isolated 1-step spikes
    out = hy.k_of_n(raw, k=2, n=3)
    assert not out.any()                        # no two-in-three anywhere -> fully suppressed


def test_sustained_exceedance_still_raises_with_latch_delay():
    raw = np.zeros(10, bool)
    raw[3:] = True                              # sustained run starting at t=3
    out = hy.k_of_n(raw, k=2, n=3)
    # need 2 of last 3 True: t=3 has only itself, t=4 has {3,4}=2 -> first latch at t=4.
    assert not out[3]
    assert out[4:].all()
    first_latch = int(np.flatnonzero(out)[0])
    assert first_latch - 3 == 1                 # latch delay = k-1 = 1 step


def test_latch_delay_equals_k_minus_1_for_fresh_sustained_run():
    raw = np.zeros(20, bool)
    raw[5:] = True
    for k in (1, 2, 3, 4):
        out = hy.k_of_n(raw, k=k, n=5)
        first_latch = int(np.flatnonzero(out)[0])
        assert first_latch - 5 == k - 1        # sustained run latches after k-1 extra steps


def test_all_false_and_all_true():
    assert not hy.k_of_n(np.zeros(8, bool), k=2, n=3).any()
    # all-True with k==n: the leading n-1 steps have a sub-length window and cannot reach k;
    # every step from index n-1 on latches. (k=1,n=1 is the full-True identity, tested above.)
    out = hy.k_of_n(np.ones(8, bool), k=3, n=3)
    assert out.tolist() == [False, False] + [True] * 6
    assert hy.k_of_n(np.ones(8, bool), k=1, n=1).all()


def test_series_shorter_than_n():
    raw = np.array([1, 1], bool)
    # k=2,n=5: t0 has 1 True (<2)->0; t1 has 2 True (>=2)->1
    assert hy.k_of_n(raw, k=2, n=5).tolist() == [False, True]
    # empty series
    assert hy.k_of_n(np.array([], bool), k=1, n=3).tolist() == []


def test_invalid_k_n_raise():
    with pytest.raises(ValueError):
        hy.k_of_n(np.ones(3, bool), k=0, n=3)        # k < 1
    with pytest.raises(ValueError):
        hy.k_of_n(np.ones(3, bool), k=3, n=2)        # k > n


def test_accepts_probability_series_via_threshold():
    # a probability/score series can be passed with an alarm threshold alpha
    score = np.array([0.1, 0.6, 0.2, 0.7, 0.8])
    out = hy.k_of_n(score, k=2, n=3, alpha=0.5)
    # raw = [0,1,0,1,1]; 2-of-3 latches at t=3 onward
    assert out.tolist() == [False, False, False, True, True]


# ---- monotonicity of the operator-facing burden -----------------------------------------

def test_fire_count_non_increasing_in_k_at_fixed_n():
    # requiring more of the same window to be True can only turn firing steps off, never on
    rng = np.random.default_rng(0)
    raw = rng.random(2000) < 0.25
    prev = np.inf
    for k in (1, 2, 3, 4, 5):
        fires = hy.k_of_n(raw, k=k, n=5).sum()
        assert fires <= prev + 1e-9
        prev = fires


def test_fire_count_non_decreasing_in_n_at_fixed_k():
    # widening the window can only make the k-of-n condition easier to meet -> more firing steps
    rng = np.random.default_rng(1)
    raw = rng.random(2000) < 0.25
    prev = -np.inf
    for n in (2, 3, 5, 7, 10):
        fires = hy.k_of_n(raw, k=2, n=n).sum()
        assert fires >= prev - 1e-9
        prev = fires


# ---- the sweep driver -------------------------------------------------------------------

def test_sweep_schema_and_identity_baseline(tmp_path):
    base = tmp_path / "ew"
    for s in range(2):
        d = base / f"seed{s}"
        d.mkdir(parents=True)
        np.savez(d / "scores.npz", **_synthetic_scores(seed=s))
    out_dir, grid = hy.run_hysteresis(
        seeds=(0, 1), threshold_deg=15, grid=[(1, 1), (2, 3), (3, 5)],
        base=base, out_dir=tmp_path / "out")
    for col in ("k", "n", "false_alert_rate", "false_episodes_per_hour",
                "lead_time_s", "detection_rate", "n_seeds"):
        assert col in grid.columns
    assert len(grid) == 3
    assert grid["detection_rate"].between(0, 1).all()
    assert (out_dir / "T_hysteresis.csv").exists()
    # (1,1) is the raw-alarm identity baseline -> a valid detection rate is produced
    base_row = grid[(grid.k == 1) & (grid.n == 1)].iloc[0]
    assert base_row["n_seeds"] == 2


def test_sweep_detection_non_increasing_with_k(tmp_path):
    base = tmp_path / "ew"
    for s in range(2):
        d = base / f"seed{s}"
        d.mkdir(parents=True)
        np.savez(d / "scores.npz", **_synthetic_scores(seed=s))
    _, grid = hy.run_hysteresis(
        seeds=(0, 1), threshold_deg=15, grid=[(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)],
        base=base, out_dir=tmp_path / "out")
    at_n5 = grid[grid.n == 5].sort_values("k")
    # requiring more evidence to latch can only miss more events -> detection cannot rise with k
    assert (at_n5["detection_rate"].diff().dropna() <= 1e-9).all()
    # and the longest lead is at the raw (k=1) operating point; latch delay shrinks it as k grows
    leads = at_n5["lead_time_s"].dropna()
    assert leads.iloc[0] >= leads.iloc[-1] - 1e-9


def test_sweep_missing_predictions_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="scores.npz"):
        hy.run_hysteresis(seeds=(0,), threshold_deg=15, grid=[(1, 1)],
                          base=tmp_path / "absent", out_dir=tmp_path / "out")
