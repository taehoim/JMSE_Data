"""Event-based early-warning metrics (P6, C2 rigor).

Table 6 scores the alarm at the per-cell (window x horizon) level, which is the right unit for
ROC/PR but not for an operator: what a watch officer experiences is *events* (episodes of large
heel) and *alerts* (episodes of the alarm sounding). This module evaluates the alarm at that
operational granularity, on a single reconstructed group timeline (1 Hz, stride-1 windows):

  exceedance event   a maximal run of consecutive timesteps with actual theta_TIA > theta_d
  decision stream    at each time t the alarm fires if the predictive score for ANY horizon
                     1..H reaches the operating threshold alpha ("will exceed within H s"),
                     optionally with a refractory period that suppresses re-fires for R s
  detection          an event is detected if a fire occurs in the H-second window before its
                     onset; the lead time is onset - earliest such fire
  false-alert episode a maximal run of fires not associated with any upcoming event onset

Reported metrics: detection (= recall) and missed-event rates, false-alert episodes per hour,
event-level precision, and lead-time quantiles. All functions take 1-D / (N,H) arrays for one
group; the runner aggregates across groups.
"""
import numpy as np


def segment_events(label_1d: np.ndarray):
    """Maximal runs of nonzero values as inclusive (start, end) index pairs."""
    lab = np.asarray(label_1d).astype(bool).astype(int)
    if lab.size == 0:
        return []
    d = np.diff(np.concatenate([[0], lab, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def decision_stream(score: np.ndarray, alpha: float, refractory: int = 0) -> np.ndarray:
    """Per-time fire decision: any-horizon score >= alpha, with an optional refractory period.

    score is (N, H). The raw decision fires when max_k score[t, k] >= alpha. If refractory > 0,
    once the alarm fires the next `refractory` steps are suppressed (modeling alert-suppression /
    avoiding a single event re-triggering a burst of alerts). Returns a boolean (N,).
    """
    raw = np.asarray(score, float).max(axis=1) >= alpha
    if refractory <= 0:
        return raw
    out = np.zeros_like(raw)
    lock = -1
    for t in np.flatnonzero(raw):
        if t > lock:
            out[t] = True
            lock = t + refractory
    return out


def event_detection(events, fires: np.ndarray, horizon: int):
    """Per-event detection flag and lead time.

    A warning issued at window i covers onsets in [i+1, i+H]; equivalently an event with onset a
    is detected by any fire in [a-H, a-1]. Lead time = a - (earliest covering fire). Returns
    (detected: bool (E,), leads: float (E,) with NaN where missed).
    """
    fires = np.asarray(fires, bool)
    detected = np.zeros(len(events), bool)
    leads = np.full(len(events), np.nan)
    for e, (start, _end) in enumerate(events):
        lo = max(0, start - horizon)
        window = np.flatnonzero(fires[lo:start]) + lo
        if window.size:
            detected[e] = True
            leads[e] = start - window[0]
    return detected, leads


def false_alert_episodes(fires: np.ndarray, events, horizon: int):
    """(n_false, n_total) alarm episodes. A fire episode is a maximal run of consecutive fires;
    it is a *true* alert if any of its fire times t has an event onset in [t+1, t+H], else *false*.
    """
    onsets = np.array([s for s, _ in events], int)
    episodes = segment_events(np.asarray(fires, bool).astype(int))
    n_total = len(episodes)
    n_false = 0
    for (a, b) in episodes:
        covered = False
        for t in range(a, b + 1):
            if np.any((onsets >= t + 1) & (onsets <= t + horizon)):
                covered = True
                break
        if not covered:
            n_false += 1
    return n_false, n_total


def event_raw_from_fires(label_1d, fires: np.ndarray, horizon: int) -> dict:
    """Raw event/alert counts + lead array for one group, given a *pre-computed* fire stream.

    Shared core of `event_raw`: takes the boolean per-time decision directly so alternative
    decision layers (e.g. a k-of-n hysteresis debounce, jmse.earlywarning.hysteresis) can be
    evaluated with the identical event-counting logic instead of duplicating it.
    """
    events = segment_events(label_1d)
    fires = np.asarray(fires, bool)
    detected, leads = event_detection(events, fires, horizon)
    n_false, n_total = false_alert_episodes(fires, events, horizon)
    return {
        "n_events": len(events), "n_detected": int(detected.sum()),
        "n_false": n_false, "n_episodes": n_total, "n_samples": len(label_1d),
        "leads": leads[~np.isnan(leads)],
    }


def event_raw(label_1d, score, alpha, horizon, refractory=0) -> dict:
    """Raw event/alert counts and the lead-time array for one group timeline (units: samples).

    Returned so the runner can pool across groups before forming rates and lead quantiles
    (averaging per-group rates would over-weight short groups). Keys: n_events, n_detected,
    n_false, n_episodes, n_samples, leads (1-D array of detected-event leads in samples).
    """
    fires = decision_stream(score, alpha, refractory)
    return event_raw_from_fires(label_1d, fires, horizon)


def pool_event_metrics(raws, fs_hz=1.0) -> dict:
    """Pool a list of `event_raw` dicts (e.g. per group) into one metric summary.

    Rates are formed from pooled counts; lead quantiles from the pooled lead array. fs_hz
    converts sample counts to hours and leads to seconds.
    """
    n_ev = sum(r["n_events"] for r in raws)
    n_det = sum(r["n_detected"] for r in raws)
    n_false = sum(r["n_false"] for r in raws)
    n_epi = sum(r["n_episodes"] for r in raws)
    n_samp = sum(r["n_samples"] for r in raws)
    leads = np.concatenate([r["leads"] for r in raws]) / fs_hz if raws else np.array([])
    dur_h = n_samp / fs_hz / 3600.0
    return {
        "n_events": n_ev, "n_detected": n_det,
        "detection_rate": float(n_det / n_ev) if n_ev else float("nan"),
        "missed_event_rate": float(1 - n_det / n_ev) if n_ev else float("nan"),
        "false_episodes": n_false, "n_episodes": n_epi,
        "false_episodes_per_hour": float(n_false / dur_h) if dur_h else float("nan"),
        "precision": float((n_epi - n_false) / n_epi) if n_epi else float("nan"),
        "lead_mean_s": float(leads.mean()) if leads.size else float("nan"),
        "lead_p10_s": float(np.quantile(leads, 0.10)) if leads.size else float("nan"),
        "lead_p50_s": float(np.quantile(leads, 0.50)) if leads.size else float("nan"),
        "lead_p90_s": float(np.quantile(leads, 0.90)) if leads.size else float("nan"),
        "duration_h": dur_h,
    }


def event_metrics(label_1d, score, alpha, horizon, refractory=0, fs_hz=1.0) -> dict:
    """Event-based detection summary for a single group timeline (convenience wrapper)."""
    return pool_event_metrics([event_raw(label_1d, score, alpha, horizon, refractory)], fs_hz)
