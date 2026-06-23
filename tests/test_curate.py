import numpy as np

from jmse.data.curate import build_curated_dataset, load_curated, compute_gm
from jmse import config


def test_curated_has_metadata_and_consistent_targets():
    build_curated_dataset()                       # idempotent; writes parquet
    df = load_curated()
    assert {"tonnage", "Hs", "realization", "Xacc", "GZ", "GZ_csv",
            "phi", "theta"}.issubset(df.columns)
    assert df["tonnage"].nunique() == 5 and df["Hs"].nunique() == 3
    # Canonical set is now the 6-realization multi set (90 files, NaN tails dropped).
    assert df["realization"].nunique() == 6
    assert df.groupby(["tonnage", "Hs"])["realization"].nunique().eq(6).all()

    # Xacc must equal sqrt(phi^2 + theta^2) (validates raw data integrity)
    recon = np.sqrt(df["phi"] ** 2 + df["theta"] ** 2)
    assert np.allclose(df["Xacc"], recon, atol=1e-5)

    # Canonical GZ = precise-dim GM * sin(Xacc) (dynamics-consistent)
    sub = df[(df.tonnage == 50) & (df.Hs == 7.0)]
    gm = compute_gm(*config.VESSELS[50])
    assert np.allclose(sub["GZ"], gm * np.sin(sub["Xacc"]), atol=1e-6)


def test_exceedance_rates_match_measurement():
    df = load_curated()
    deg = np.degrees(df["Xacc"].values)
    assert abs((deg > 15).mean() * 100 - 12.4) < 1.0
    assert abs((deg > 20).mean() * 100 - 4.9) < 1.0


def test_canonical_gz_differs_from_csv_for_30ton():
    # 30-ton is where the rounded-dim CSV GZ deviates most from canonical
    df = load_curated()
    sub = df[(df.tonnage == 30) & (df.Hs == 7.0)]
    rel = (np.abs(sub["GZ"] - sub["GZ_csv"]) / sub["GZ"].abs().clip(lower=1e-6)).median()
    assert rel > 0.05
