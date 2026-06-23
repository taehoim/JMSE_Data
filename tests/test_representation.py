import numpy as np

from jmse.data.representation import build_representation_arrays, gm_by_tonnage
from jmse.eval.error_propagation import reconstruct


def test_gm_by_tonnage_matches_data_relation():
    gm = gm_by_tonnage()
    # GZ = GM*sin(Xacc): GM is constant per tonnage and increases with size
    assert set(gm) == {10, 20, 30, 40, 50}
    assert 0.29 < gm[10] < 0.31 and 0.43 < gm[50] < 0.45
    assert gm[10] < gm[30] < gm[50]


def test_representation_arrays_aligned_with_id_targets():
    d = build_representation_arrays()
    for split in ("train", "val", "test"):
        # the auxiliary (phi,theta) reconstruct EXACTLY the primary Xacc target -> alignment proof
        recon = reconstruct(d[f"phi_{split}"], d[f"theta_{split}"])
        assert np.allclose(recon, d[f"y_{split}"], atol=1e-5)


def test_gz_inversion_recovers_xacc():
    d = build_representation_arrays()
    gm = d["gm_test"][:, None]                            # per-window GM
    xacc_from_gz = np.arcsin(np.clip(d["gz_test"] / gm, -1.0, 1.0))
    assert np.allclose(xacc_from_gz, d["y_test"], atol=1e-5)


def test_shapes_consistent():
    d = build_representation_arrays()
    n, h = d["y_test"].shape
    assert d["phi_test"].shape == (n, h)
    assert d["theta_test"].shape == (n, h)
    assert d["gz_test"].shape == (n, h)
    assert d["gm_test"].shape == (n,)
