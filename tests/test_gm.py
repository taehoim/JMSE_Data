from jmse.data.curate import compute_gm
from jmse import config


def test_gm_canonical_precise_dimension_values():
    # Canonical GM from the PRECISE vessel dims that drove the motion dynamics
    # (config.VESSELS). NOTE: the original CSV GZ column used the *rounded* filename
    # dims, so its implied GM differs (e.g. 30-ton 0.35 vs 0.372). We adopt the
    # dynamics-consistent precise value and recompute GZ in curation.
    expected = {10: 0.3000, 20: 0.3261, 30: 0.3723, 40: 0.4104, 50: 0.4398}
    for ton, (L, B, T) in config.VESSELS.items():
        gm = compute_gm(L, B, T)
        assert abs(gm - expected[ton]) < 1e-3, (ton, gm, expected[ton])


def test_precise_vs_rounded_dim_discrepancy_is_documented():
    # 30-ton is the most affected: precise dims raise GM ~6.7% over the rounded-dim
    # value embedded in the original CSV GZ. This documents the curation correction.
    precise = compute_gm(*config.VESSELS[30])              # 11.43, 2.74, 1.37
    rounded = compute_gm(11.4, 2.7, 1.4)
    assert precise > rounded
    assert abs(precise - rounded) / rounded > 0.05


def test_gm_floor_applies_to_small_vessels():
    # 10-ton un-clamped GM is below the floor -> clamped to 0.30
    assert compute_gm(*config.VESSELS[10]) == config.GM_FLOOR
    # un-clamped must be strictly below the floor (justifies the clamp)
    assert compute_gm(*config.VESSELS[10], clamp=False) < config.GM_FLOOR


def test_gm_monotonic_increasing_with_tonnage():
    gms = [compute_gm(*config.VESSELS[t]) for t in (10, 20, 30, 40, 50)]
    assert gms == sorted(gms)
