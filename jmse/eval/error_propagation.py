"""Analytical error propagation for the reconstruction target (C5).

The total inclination angle is exactly the Euclidean norm of the two roll/pitch angles,
Xacc = g(phi, theta) = sqrt(phi^2 + theta^2) (verified to ~5e-7 rad on the dataset). A
model that predicts (phi, theta) and reconstructs g therefore inherits a *propagated*
error from the angle predictions, with two analytic pieces (delta method):

  first-order variance   Var[g] ~= (dg/dphi)^2 Var[phi] + (dg/dtheta)^2 Var[theta] + 2 dg/dphi dg/dtheta Cov
  Jensen (curvature) bias  E[g] - g ~= 1/2 ( d2g/dphi2 Var[phi] + d2g/dtheta2 Var[theta] + 2 d2g/dphidtheta Cov )

with  dg/dphi = phi/r, dg/dtheta = theta/r,  d2g/dphi2 = theta^2/r^3, d2g/dtheta2 = phi^2/r^3,
      d2g/dphidtheta = -phi*theta/r^3,  r = sqrt(phi^2+theta^2).
Because g is convex, the Jensen bias is non-negative: reconstructing from noisy angle
predictions systematically *over-estimates* the inclination. These closed forms predict
the reconstruction-path error that the direct-Xacc path avoids (the C5 punchline), and are
validated against Monte-Carlo in the tests.
"""
import numpy as np

_EPS = 1e-12


def reconstruct(phi, theta):
    """Xacc = sqrt(phi^2 + theta^2)."""
    return np.hypot(np.asarray(phi, float), np.asarray(theta, float))


def grad(phi, theta):
    """(dg/dphi, dg/dtheta) = (phi/r, theta/r); the unit radial vector."""
    phi = np.asarray(phi, float)
    theta = np.asarray(theta, float)
    r = np.maximum(np.hypot(phi, theta), _EPS)
    return phi / r, theta / r


def propagated_variance(phi, theta, var_phi, var_theta, cov=0.0):
    """First-order (delta-method) variance of the reconstructed Xacc."""
    gphi, gtheta = grad(phi, theta)
    return gphi**2 * var_phi + gtheta**2 * var_theta + 2.0 * gphi * gtheta * cov


def jensen_bias(phi, theta, var_phi, var_theta, cov=0.0):
    """Second-order curvature (Jensen) bias E[g]-g; non-negative for the convex norm."""
    phi = np.asarray(phi, float)
    theta = np.asarray(theta, float)
    r = np.maximum(np.hypot(phi, theta), _EPS)
    r3 = r**3
    hphiphi = theta**2 / r3
    hthetatheta = phi**2 / r3
    hphitheta = -phi * theta / r3
    return 0.5 * (hphiphi * var_phi + hthetatheta * var_theta + 2.0 * hphitheta * cov)


def predicted_reconstruction_rmse(phi, theta, var_phi, var_theta, cov=0.0):
    """Predicted RMSE of the reconstructed Xacc = sqrt(bias^2 + variance) (per point)."""
    bias = jensen_bias(phi, theta, var_phi, var_theta, cov)
    var = propagated_variance(phi, theta, var_phi, var_theta, cov)
    return np.sqrt(bias**2 + var)
