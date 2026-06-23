import numpy as np

from jmse.uq.ensemble import ensemble_moments, ensemble_samples


def test_ensemble_moments_mean_and_std():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[3.0, 4.0], [5.0, 6.0]])
    c = np.array([[2.0, 3.0], [4.0, 5.0]])
    mean, std = ensemble_moments([a, b, c])
    assert np.allclose(mean, [[2.0, 3.0], [4.0, 5.0]])
    # population std (ddof=0) across the 3 members at each cell
    assert np.allclose(std, np.std(np.stack([a, b, c]), axis=0))


def test_ensemble_moments_requires_members():
    try:
        ensemble_moments([])
    except ValueError:
        return
    raise AssertionError("empty member list should raise")


def test_ensemble_samples_stacks_members_on_last_axis():
    a = np.zeros((5, 3))
    b = np.ones((5, 3))
    s = ensemble_samples([a, b])
    assert s.shape == (5, 3, 2)
    assert np.allclose(s[..., 0], 0) and np.allclose(s[..., 1], 1)


def test_ensemble_single_member_zero_variance():
    a = np.array([[1.0, 2.0]])
    mean, std = ensemble_moments([a])
    assert np.allclose(mean, a) and np.allclose(std, 0.0)
