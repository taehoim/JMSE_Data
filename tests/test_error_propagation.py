import numpy as np

from jmse.eval import error_propagation as ep


def test_reconstruct_matches_euclidean_norm():
    phi = np.array([0.3, -0.1, 0.0])
    theta = np.array([0.4, 0.2, 0.5])
    assert np.allclose(ep.reconstruct(phi, theta), np.hypot(phi, theta))


def test_gradient_is_unit_radial_vector():
    # grad of sqrt(phi^2+theta^2) is the unit radial vector -> norm exactly 1
    phi, theta = np.array([0.3, 0.7]), np.array([0.4, -0.2])
    gphi, gtheta = ep.grad(phi, theta)
    assert np.allclose(gphi**2 + gtheta**2, 1.0)
    assert np.allclose(gphi, phi / np.hypot(phi, theta))


def test_propagated_variance_vs_montecarlo():
    phi, theta = 0.30, 0.40                              # r = 0.5
    sphi, stheta = 0.02, 0.03                            # small noise -> delta method valid
    var = ep.propagated_variance(phi, theta, sphi**2, stheta**2)
    rng = np.random.default_rng(0)
    n = 2_000_000
    recon = np.hypot(phi + sphi * rng.standard_normal(n), theta + stheta * rng.standard_normal(n))
    assert abs(var - recon.var()) / recon.var() < 0.02


def test_jensen_bias_positive_and_vs_montecarlo():
    phi, theta = 0.30, 0.40
    sphi, stheta = 0.05, 0.05
    bias = ep.jensen_bias(phi, theta, sphi**2, stheta**2)
    assert bias > 0                                      # Euclidean norm is convex -> over-estimate
    rng = np.random.default_rng(1)
    n = 4_000_000
    recon = np.hypot(phi + sphi * rng.standard_normal(n), theta + stheta * rng.standard_normal(n))
    mc_bias = recon.mean() - np.hypot(phi, theta)
    assert abs(bias - mc_bias) / mc_bias < 0.06


def test_jensen_bias_zero_without_error():
    assert ep.jensen_bias(0.3, 0.4, 0.0, 0.0) == 0.0


def test_predicted_reconstruction_rmse_combines_bias_and_std():
    phi, theta = 0.3, 0.4
    vphi, vtheta = 0.02**2, 0.03**2
    rmse = ep.predicted_reconstruction_rmse(phi, theta, vphi, vtheta)
    bias = ep.jensen_bias(phi, theta, vphi, vtheta)
    var = ep.propagated_variance(phi, theta, vphi, vtheta)
    assert abs(rmse - np.sqrt(bias**2 + var)) < 1e-12
