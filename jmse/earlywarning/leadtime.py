"""Lead-time and effective-horizon analysis for the early-warning alarm (C2).

Within one (tonnage, Hs) group the test windows are contiguous in time (stride 1), so a
future timestep e is predicted by every window/horizon pair with r + k = e: window r
forecasts it k+1 seconds ahead. A danger event (Xacc(e) > theta) is *detected* if any of
those cells fires (score >= alpha); its lead time is the earliest such firing, i.e. the
largest (k+1). Events are counted once per future timestep (deduped across the windows
that see them); false alarms are counted per decision cell (each spurious alarm is a cost),
matching the cell-level FPR used by the ROC analysis.
"""
import numpy as np

from jmse.earlywarning.alarm import EXCEEDS


def effective_horizon(auc_by_h, min_auc: float = 0.80) -> int:
    """Largest contiguous horizon (from t+1) whose AUC stays >= min_auc.

    auc_by_h[k] is the detection AUC at horizon k+1 s. Returns 0 if even t+1 is below
    the bar (skill is generally monotone-decreasing, so the contiguous run is the honest
    'how far ahead is the alarm still reliable' summary)."""
    auc = np.asarray(auc_by_h, float)
    h = 0
    for a in auc:
        if a >= min_auc:
            h += 1
        else:
            break
    return h


def lead_times(y_true, scores, groups, theta: float, alpha: float) -> dict:
    """Event-level lead times + cell-level detection/false-alarm stats at operating alpha.

    y_true, scores: (N, H); groups: (N,) int group ids; theta in target units; alpha is the
    probabilistic-score operating point. Returns leads (s, per detected event), n_events,
    detection_rate, mean/median lead, and cell-level tpr/fpr.
    """
    y_true = np.asarray(y_true, float)
    scores = np.asarray(scores, float)
    groups = np.asarray(groups, int)
    H = y_true.shape[1]
    label = EXCEEDS(y_true, theta)                       # strict '>' (M3)
    fired = scores >= alpha

    leads, n_events, detected = [], 0, 0
    for gid in np.unique(groups):
        rows = np.where(groups == gid)[0]                # contiguous, time-ordered
        Y, F = label[rows], fired[rows]
        n = len(rows)
        for e in range(n + H - 1):                       # each future timestep in the block
            k0 = min(H - 1, e)                            # a valid representative cell (r,k), r+k=e
            r0 = e - k0
            if r0 < 0 or r0 >= n or not Y[r0, k0]:
                continue
            n_events += 1
            best = 0
            for k in range(H):                           # earliest firing = largest lead
                r = e - k
                if 0 <= r < n and F[r, k]:
                    best = max(best, k + 1)
            if best > 0:
                detected += 1
                leads.append(best)

    leads = np.asarray(leads, float)
    tp = int((fired & label).sum())
    fp = int((fired & ~label).sum())
    pos = int(label.sum())
    neg = int((~label).sum())
    return {
        "leads": leads,
        "n_events": n_events,
        "detection_rate": (detected / n_events) if n_events else float("nan"),
        "mean_lead_s": float(leads.mean()) if leads.size else float("nan"),
        "median_lead_s": float(np.median(leads)) if leads.size else float("nan"),
        "tpr": (tp / pos) if pos else float("nan"),
        "fpr": (fp / neg) if neg else float("nan"),
    }
