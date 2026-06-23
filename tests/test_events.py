"""Tests for event-based early-warning metrics (P6, C2 rigor)."""
import numpy as np

from jmse.earlywarning import events as ev


def test_segment_events_finds_contiguous_runs():
    label = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    runs = ev.segment_events(label)
    assert runs == [(1, 2), (5, 5), (7, 9)]


def test_segment_events_empty_and_full():
    assert ev.segment_events(np.zeros(5, int)) == []
    assert ev.segment_events(np.ones(3, int)) == [(0, 2)]


def test_decision_stream_any_horizon_threshold():
    score = np.array([[0.1, 0.2], [0.4, 0.9], [0.0, 0.0]])
    fires = ev.decision_stream(score, alpha=0.5, refractory=0)
    assert fires.tolist() == [False, True, False]      # window 1 has a horizon >= 0.5


def test_decision_stream_refractory_suppresses_following_fires():
    score = np.array([[1.0], [1.0], [1.0], [0.0], [1.0]])
    fires = ev.decision_stream(score, alpha=0.5, refractory=2)
    # fire at 0 suppresses 1,2; 3 is below anyway; 4 fires again
    assert fires.tolist() == [True, False, False, False, True]


def test_event_detection_lead_time_within_window():
    # onset at time 5; horizon 3 -> a warning at windows 2,3,4 detects it.
    fires = np.zeros(10, bool)
    fires[3] = True                                    # warns at t=3 about [4,6] -> covers onset 5
    events = [(5, 6)]
    detected, leads = ev.event_detection(events, fires, horizon=3)
    assert detected.tolist() == [True]
    assert leads.tolist() == [2]                       # 5 - 3


def test_event_detection_missed_when_no_fire_in_window():
    fires = np.zeros(10, bool)
    fires[0] = True                                    # too early: covers [1,3], onset is 5
    detected, leads = ev.event_detection([(5, 6)], fires, horizon=3)
    assert detected.tolist() == [False]
    assert np.isnan(leads[0])


def test_false_alert_episodes_counts_unassociated_runs():
    # onset at 6; horizon 2. fires at {0,1} (false) and {4,5} (true, covers [5..7]).
    fires = np.zeros(10, bool)
    fires[[0, 1, 4, 5]] = True
    events = [(6, 6)]
    n_false, n_total = ev.false_alert_episodes(fires, events, horizon=2)
    assert n_total == 2 and n_false == 1


def test_event_metrics_end_to_end():
    rng = np.random.default_rng(0)
    n = 200
    actual = np.zeros(n)
    actual[50:55] = 1.0                                # one exceedance event
    actual[120:123] = 1.0                              # another
    label = (actual > 0.5).astype(int)
    score = np.zeros((n, 3))
    score[47, 1] = 1.0                                 # warns before first onset (lead 3)
    score[118, 0] = 1.0                                # warns before second onset (lead 2)
    score[10, 2] = 1.0                                 # a false alarm
    m = ev.event_metrics(label, score, alpha=0.5, horizon=3, refractory=0, fs_hz=1.0)
    assert m["n_events"] == 2
    assert m["detection_rate"] == 1.0
    assert m["missed_event_rate"] == 0.0
    assert m["false_episodes"] == 1
    assert 0.0 < m["false_episodes_per_hour"] < 100.0
    assert m["lead_p50_s"] > 0
