"""Realization axis: the canonical curated set pools 6 wave-phase realizations.

r0 carries the published seed, so the r0 subset must reproduce the published
per-(tonnage, Hs) Xacc distribution recorded in the legacy T2 summary.
"""
import numpy as np
import pandas as pd

from jmse import config
from jmse.data.curate import build_curated_dataset, load_curated

_N_REAL = 6
_N_FILES = len(config.VESSELS) * len(config.HS_VALUES) * _N_REAL   # 5 x 3 x 6 = 90
_ROWS_PER_FILE = 5001
# 3 capsize-regime files (10 t, Hs 7, r1/r3/r4) diverge and carry NaN tails that are
# dropped in curation, so the clean total is below the nominal 90 x 5001.
_CLEAN_ROWS = 441562
_NAN_TAIL_ROWS = _N_FILES * _ROWS_PER_FILE - _CLEAN_ROWS          # rows lost to divergence


def test_curated_multi_has_realization_axis():
    build_curated_dataset(force=True)                  # canonical = multi clamped
    df = load_curated()
    assert "realization" in df.columns
    assert set(df["realization"].unique()) == set(range(_N_REAL))
    assert not df[config.FEATURES + [config.TARGET]].isna().any().any()  # NaN tails dropped
    assert len(df) == _CLEAN_ROWS, (
        f"expected {_CLEAN_ROWS} clean rows = 90 files x 5001 minus {_NAN_TAIL_ROWS} "
        f"divergence NaN-tail rows (3 capsize files: 10t/Hs7 r1/r3/r4); got {len(df)}. "
        "Regenerating the multi dataset changes this count -- update _CLEAN_ROWS to match "
        "data_processed/T2_realization_counts.csv."
    )
    # every (tonnage, Hs) cell still carries all 6 realizations (prefixes survive)
    per_cell = df.groupby(["tonnage", "Hs"])["realization"].nunique()
    assert (per_cell == _N_REAL).all()


def test_r0_matches_published_distribution():
    """r0 (published seed) must match the legacy T2 summary within tolerance."""
    df = load_curated()
    r0 = df[df["realization"] == 0].copy()
    r0["_deg"] = np.degrees(r0["Xacc"])

    # Stable snapshot of the published single-realization distribution (legacy schema);
    # the live T2_dataset_summary.csv is now realization-averaged and reshaped.
    published = pd.read_csv(config.PROCESSED_DIR / "T2_dataset_summary_published.csv")
    pub = published.set_index(["tonnage", "Hs"])

    for (ton, hs), g in r0.groupby(["tonnage", "Hs"]):
        d = g["_deg"]
        ref = pub.loc[(ton, hs)]
        assert abs(d.mean() - ref["Xacc_mean_deg"]) < 0.6, (ton, hs, "mean")
        assert abs(d.std() - ref["Xacc_std_deg"]) < 0.6, (ton, hs, "std")
        assert abs((d > 15).mean() * 100 - ref["exceed_15deg_pct"]) < 2.0, (ton, hs, "ex15")


def test_realization_spread_is_nontrivial():
    """Distinct realizations must actually differ (independent wave phases)."""
    df = load_curated()
    sub = df[(df.tonnage == 10) & (df.Hs == 7.0)]
    maxima = sub.groupby("realization")["Xacc"].max().apply(np.degrees)
    # the brief reports a 38.6-61.8 deg spread of Xacc_max across r0..r5
    assert maxima.max() - maxima.min() > 10.0


def test_dataset_summary_is_realization_averaged():
    """T2_dataset_summary.csv reports mean+/-std across realizations per (ton, Hs)."""
    from jmse.data.curate import dataset_summary
    summary = dataset_summary()
    # 15 cells (5 tonnage x 3 Hs), each aggregated over realizations
    assert len(summary) == len(config.VESSELS) * len(config.HS_VALUES)
    assert {"tonnage", "Hs"}.issubset(summary.columns)
    # carries a realization-spread column (std of per-realization means)
    assert any("std" in c for c in summary.columns)


def test_realization_counts_report_exposes_uneven_coverage():
    """Per-(ton, Hs, real) clean-row counts surface the 10t/Hs7 divergence transparently."""
    import pandas as pd
    from jmse.data.curate import dataset_summary, realization_counts_report

    counts = realization_counts_report()
    # one row per (ton, Hs, real); all 90 cells present
    assert len(counts) == len(config.VESSELS) * len(config.HS_VALUES) * 6
    assert {"tonnage", "Hs", "realization", "n_rows", "n_expected", "is_full"}.issubset(counts.columns)

    # every cell except the 3 divergent 10t/Hs7 realizations is full (5001 rows)
    full = counts[counts["is_full"]]
    assert len(full) == 90 - 3
    short = counts[~counts["is_full"]].set_index(["tonnage", "Hs", "realization"])["n_rows"].to_dict()
    assert short == {(10, 7.0, 1): 4182, (10, 7.0, 3): 643, (10, 7.0, 4): 1650}

    # the report is written to disk for reviewers
    dataset_summary()                                  # emits both tables
    written = pd.read_csv(config.PROCESSED_DIR / "T2_realization_counts.csv")
    assert len(written) == 90
