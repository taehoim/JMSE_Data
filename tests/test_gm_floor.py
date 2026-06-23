"""GM-floor confound study (R3.4): clamped vs unclamped 10 t generalization.

The 10 t LOVO fold is built and trained twice -- once from the clamped canonical data
and once from the unclamped data (the 10 t roll motion RE-SIMULATED with the true
sub-floor GM in the EOM) -- so the side-by-side 10 t test metrics quantify how much of
the 10 t generalization gap is the GM discontinuity versus genuine small-vessel dynamics.

The unclamped-EOM re-simulation is produced in the sim domain (R3.4 task); these tests
exercise the harness structure WITHOUT requiring that data, and skip the data-dependent
assertions until it is present.
"""
import numpy as np
import pytest

from jmse.eval import gm_floor

_HAVE_RESIM = gm_floor.unclamped_resim_available()
_resim = pytest.mark.skipif(not _HAVE_RESIM,
                            reason="unclamped-EOM re-simulation not present yet (R3.4)")


# --- harness structure (always runs, no re-sim data required) ---

def test_clamped_variant_curates_and_floors_10t_gm():
    """The clamped variant is the canonical set; its 10 t GM sits at the 0.30 floor."""
    clamped = gm_floor.curated_variant("clamped")
    assert abs(clamped[clamped.tonnage == 10]["GM"].iloc[0] - 0.30) < 1e-6


def test_unclamped_raises_clear_error_when_resim_absent():
    """Without the re-sim, requesting the unclamped variant fails with an actionable message,
    not a silent fallback to clamped-identical motion."""
    if _HAVE_RESIM:
        pytest.skip("re-sim present; the absent-data error path cannot be exercised")
    with pytest.raises(FileNotFoundError, match="re-sim"):
        gm_floor.curated_variant("unclamped")
    with pytest.raises(FileNotFoundError, match="re-sim"):
        gm_floor.run_gm_floor(smoke=True)


def test_unclamped_dir_is_parametrized(monkeypatch, tmp_path):
    """The unclamped source dir is overridable (JMSE_UNCLAMPED_DIR) for the re-sim output."""
    from jmse import config
    # availability tracks the configured dir: an empty override dir reads as 'not available'
    monkeypatch.setattr(config, "RAW_DATA_DIR_MULTI_UNCLAMPED", tmp_path)
    assert gm_floor.unclamped_resim_available() is False


# --- data-dependent (skipped until the R3.4 re-sim lands) ---

@_resim
def test_gm_floor_curates_distinct_10t_gm():
    """Clamped 10 t GM is the 0.30 floor; unclamped 10 t GM is the smaller true value."""
    clamped = gm_floor.curated_variant("clamped")
    unclamped = gm_floor.curated_variant("unclamped")
    gm_c = clamped[clamped.tonnage == 10]["GM"].iloc[0]
    gm_u = unclamped[unclamped.tonnage == 10]["GM"].iloc[0]
    assert abs(gm_c - 0.30) < 1e-6           # clamped at the floor
    assert gm_u < gm_c                        # unclamped is below the floor (true GM)
    for ton in (20, 30, 40, 50):              # 20 t+ unaffected by the floor
        assert abs(clamped[clamped.tonnage == ton]["GM"].iloc[0]
                   - unclamped[unclamped.tonnage == ton]["GM"].iloc[0]) < 1e-9


@_resim
def test_run_gm_floor_table_has_both_variants_and_finite_metrics(tmp_path):
    out_dir, table = gm_floor.run_gm_floor(smoke=True, out_dir=tmp_path)
    assert set(table["variant"]) == {"clamped", "unclamped"}
    for col in ("rmse_deg", "r2"):
        assert col in table.columns
        assert np.isfinite(table[col]).all()
    assert (out_dir / "T_gmfloor.csv").exists()
