import numpy as np

from jmse.earlywarning import leadtime as lt


def test_effective_horizon_contiguous_run():
    assert lt.effective_horizon(np.array([0.95, 0.90, 0.85, 0.70, 0.60])) == 3
    assert lt.effective_horizon(np.array([0.70, 0.95, 0.95])) == 0      # first already below
    assert lt.effective_horizon(np.array([0.9, 0.9, 0.9])) == 3         # all pass -> H
    assert lt.effective_horizon(np.array([0.9, 0.7, 0.9])) == 1         # stops at first dip
    assert lt.effective_horizon(np.array([0.85, 0.82]), min_auc=0.80) == 2


def _single_group(y, scores):
    n = len(y)
    return dict(y_true=np.asarray(y), scores=np.asarray(scores),
                groups=np.zeros(n, int))


def test_lead_time_earliest_detection():
    # H=3, theta=0. Build a danger event at absolute future index e=3 (cells r+k=3),
    # and a second danger event at e=1 (cells r+k=1) that is never alarmed (missed).
    H = 3
    y = np.full((5, H), -0.5)
    for r, k in [(3, 0), (2, 1), (1, 2)]:    # e = 3 event (same physical timestep)
        y[r, k] = 0.5
    for r, k in [(1, 0), (0, 1)]:            # e = 1 event
        y[r, k] = 0.5
    scores = np.zeros((5, H))
    scores[3, 0] = 0.9                        # detects e=3 at lead 1
    scores[2, 1] = 0.9                        # detects e=3 at lead 2 (earliest -> wins)
    scores[1, 2] = 0.1                        # would be lead 3 but below alpha
    # e=1 cells stay at 0 -> missed
    out = lt.lead_times(**_single_group(y, scores), theta=0.0, alpha=0.5)
    assert out["n_events"] == 2
    assert out["leads"].tolist() == [2]      # e=3 detected with max lead 2; e=1 missed
    assert abs(out["detection_rate"] - 0.5) < 1e-9
    assert out["mean_lead_s"] == 2.0


def test_lead_time_counts_false_alarms_cell_level():
    H = 2
    y = np.full((3, H), -1.0)                 # no danger anywhere
    scores = np.zeros((3, H))
    scores[0, 0] = 0.9                        # spurious alarm (false positive)
    scores[2, 1] = 0.7                        # another false positive
    out = lt.lead_times(**_single_group(y, scores), theta=0.0, alpha=0.5)
    assert out["n_events"] == 0
    assert np.isnan(out["detection_rate"])
    # 2 fired cells, all 6 cells negative -> FPR = 2/6
    assert abs(out["fpr"] - 2 / 6) < 1e-9
    assert out["leads"].size == 0


def test_lead_time_respects_group_boundaries():
    # two groups; an event in group 0 must not be "detected" by a window in group 1
    H = 2
    y = np.full((4, H), -0.5)
    y[1, 0] = 0.5; y[0, 1] = 0.5             # group 0 event at e=1
    scores = np.zeros((4, H))
    scores[0, 1] = 0.9                        # group 0 window detects it at lead 2
    groups = np.array([0, 0, 1, 1])
    out = lt.lead_times(y, scores, groups, theta=0.0, alpha=0.5)
    assert out["n_events"] == 1 and out["leads"].tolist() == [2]
