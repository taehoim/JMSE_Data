from jmse import config


def test_vessel_table_and_paths():
    assert config.RAW_DATA_DIR.exists()                       # 00_Ref data present
    assert len(config.VESSELS) == 5                           # 10..50 ton
    assert config.FEATURES == ["u", "v", "w", "p", "q", "r", "phi", "theta"]
    assert config.LOOKBACK == 20 and config.HORIZON == 5
    assert config.DANGER_THRESHOLDS_DEG == [15, 20, 25]       # sorted ascending


def test_threshold_radians_consistent():
    import numpy as np
    assert np.allclose(
        config.DANGER_THRESHOLDS_RAD,
        [np.radians(d) for d in config.DANGER_THRESHOLDS_DEG],
    )
