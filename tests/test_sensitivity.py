"""Test the operating-point sensitivity grid (rigor follow-up) on synthetic scores."""
import numpy as np

from jmse.earlywarning import sensitivity as sn
from tests.test_events_run import _synthetic_scores


def test_sensitivity_grid_schema_and_monotonicity(tmp_path):
    base = tmp_path / "ew"
    for s in range(2):
        d = base / f"seed{s}"
        d.mkdir(parents=True)
        np.savez(d / "scores.npz", **_synthetic_scores(seed=s))
    out_dir, grid = sn.run_sensitivity(seeds=(0, 1), threshold_deg=15,
                                       fprs=[0.05, 0.10, 0.20], refractories=[0, 5],
                                       base=base, out_dir=tmp_path / "out")
    assert len(grid) == 3 * 2
    for col in ("fpr_budget", "refractory_s", "detection_rate",
                "false_episodes_per_hour", "precision"):
        assert col in grid.columns
    assert grid["detection_rate"].between(0, 1).all()
    assert grid["precision"].between(0, 1).all()
    # a looser false-alarm budget cannot lower the detection rate (more firing -> >= recall)
    at_r0 = grid[grid.refractory_s == 0].sort_values("fpr_budget")
    assert (at_r0["detection_rate"].diff().dropna() >= -1e-9).all()
    assert (out_dir / "T22_sensitivity.csv").exists()


def test_cost_curve_schema_and_monotone_detection(tmp_path):
    base = tmp_path / "ew"
    for s in range(2):
        d = base / f"seed{s}"
        d.mkdir(parents=True)
        np.savez(d / "scores.npz", **_synthetic_scores(seed=s))
    out_dir, cc = sn.cost_curve(seeds=(0, 1), threshold_deg=15, fprs=[0.05, 0.10, 0.20],
                                base=base, out_dir=tmp_path / "out")
    assert set(cc["alarm"]) == {"prob", "point", "naive"}
    for col in ("fpr_budget", "false_episodes_per_hour", "detection_rate", "precision"):
        assert col in cc.columns
    # detection is non-decreasing in the budget for the probabilistic alarm
    prob = cc[cc.alarm == "prob"].sort_values("fpr_budget")
    assert (prob["detection_rate"].diff().dropna() >= -1e-9).all()
    assert (out_dir / "cost_curve.csv").exists()
