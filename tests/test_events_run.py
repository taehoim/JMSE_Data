"""Test the event-based runner's per-seed scoring on a synthetic scores bundle (P6)."""
import numpy as np

from jmse.earlywarning import events_run as er


def _synthetic_scores(seed=0):
    """Two groups, each a contiguous timeline with planted exceedance events in BOTH halves.

    The actual heel timeline `a` is windowed into y_true[i,k]=a[i+1+k]; the probabilistic/point
    score is an oracle-ish 0.9-on-exceedance / 0.05-elsewhere signal so that (a) alpha is
    selectable on the calibration half and (b) pre-onset windows carry a high any-horizon score
    and the events are detected. The naive channel stays flat (a weak baseline).
    """
    H = 5
    theta = np.radians(15)
    blocks, groups = [], []
    for g in range(2):
        n = 400
        a = np.full(n + H, np.radians(5.0))                  # calm baseline angle timeline
        for onset in (90, 150, 250, 330):                    # events in both halves
            a[onset:onset + 4] = np.radians(20.0)
        y = np.stack([a[1 + k: 1 + k + n] for k in range(H)], axis=1)   # (n,H)
        oracle = np.where(y > theta, 0.9, 0.05)
        naive = np.full((n, H), np.radians(5.0))
        blocks.append((y, oracle, oracle.copy(), naive))
        groups.append(np.full(n, g))
    cat = lambda j: np.concatenate([b[j] for b in blocks])   # noqa: E731
    return {"y_true": cat(0), "groups": np.concatenate(groups), "point": cat(1),
            "naive": cat(3), "prob_15": cat(2), "prob_20": cat(2), "prob_25": cat(2)}


def test_score_seed_schema_and_detection():
    z = _synthetic_scores()
    df = er.score_seed(z, theta_deg=15, theta_rad=np.radians(15),
                       fpr=0.10, refractory=5, horizon=5)
    assert set(df["alarm"]) == {"point", "prob", "naive"}
    for col in ("detection_rate", "missed_event_rate", "false_episodes_per_hour",
                "precision", "lead_p50_s", "n_events"):
        assert col in df.columns
    # the probabilistic channel was given clean pre-onset warnings -> should detect events
    prob_row = df[df["alarm"] == "prob"].iloc[0]
    assert prob_row["n_events"] >= 1
    assert prob_row["detection_rate"] > 0.0
    assert 0.0 <= prob_row["missed_event_rate"] <= 1.0
