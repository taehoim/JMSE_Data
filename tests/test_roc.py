import numpy as np
from sklearn.metrics import roc_auc_score

from jmse.earlywarning import roc


def test_roc_auc_perfect_and_reversed():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert roc.roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc.roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_roc_auc_matches_sklearn_on_random():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 5000).astype(bool)
    scores = rng.random(5000) + 0.3 * labels             # mild signal + ties
    assert abs(roc.roc_auc(labels, scores) - roc_auc_score(labels, scores)) < 1e-9


def test_roc_auc_single_class_is_nan():
    assert np.isnan(roc.roc_auc(np.zeros(10, bool), np.random.random(10)))
    assert np.isnan(roc.roc_auc(np.ones(10, bool), np.random.random(10)))


def test_pr_auc_perfect():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert abs(roc.pr_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9


def test_confusion_and_f1_at_threshold():
    labels = np.array([1, 1, 0, 0], dtype=bool)
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    tp, fp, fn, tn = roc.confusion_at(labels, scores, 0.5)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)
    assert abs(roc.f1_at(labels, scores, 0.5) - 0.5) < 1e-9   # P=0.5,R=0.5 -> F1=0.5


def test_best_f1_finds_separating_threshold():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    f1, thr = roc.best_f1(labels, scores)
    assert abs(f1 - 1.0) < 1e-9 and 0.2 < thr <= 0.8


def test_operating_point_at_fpr_matches_false_alarm_budget():
    rng = np.random.default_rng(1)
    pos = rng.normal(2.0, 1.0, 2000)
    neg = rng.normal(0.0, 1.0, 2000)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(2000), np.zeros(2000)]).astype(bool)
    op = roc.operating_point_at_fpr(labels, scores, target_fpr=0.1)
    assert op["fpr"] <= 0.1 + 1e-6                        # respects the false-alarm budget
    assert op["tpr"] > 0.6                                # good separation -> high recall
