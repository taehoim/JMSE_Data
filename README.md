# JMSE — Uncertainty-Aware Short-Horizon Warning of Large-Inclination Exceedance in Small Fishing Vessels

Code and data for the *Journal of Marine Science and Engineering* (MDPI) article
**"Uncertainty-Aware Short-Horizon Warning of Large-Inclination Exceedance in Small Fishing Vessels: A Simulation-Based Multi-Model Benchmark."**

This repository provides the simulated dataset, the full analysis pipeline, and a
reproduction guide so that every table and figure in the article can be regenerated.

## Dataset

6-DOF seakeeping simulations of small fishing vessels, released here as the regenerated
**multi-realization** dataset:

- **90 time series** = 5 tonnages (10–50 t) × 3 sea states (Hs ∈ {3, 5, 7} m) × **6 independent wave-phase realizations** (`r0`–`r5`; `r0` carries the original published phase seed).
- Each series: 5001 s at 1 Hz (`time, u, v, w, p, q, r, phi, theta, Xacc, GZ`); three high-sea-state 10 t series are shorter where the integrator diverges in the capsize regime (documented in `data_processed/T2_realization_counts.csv`).
- An **unclamped-GM 10 t variant** (true `GM_T = 0.258` m, without the 0.30 m hydrostatic floor) used to disentangle the metacentric-floor confound from genuine small-vessel dynamics.

Locations:
- `data_processed/curated.parquet` — assembled, analysis-ready dataset (with a `realization` column) + per-condition summaries (`T2_*`).
- `00_Ref/논문코드_데이터셋/fishingVessel/6Dof_dataset_multi_withXacc/` — the 90 raw per-condition CSVs (clamped, canonical).
- `00_Ref/논문코드_데이터셋/fishingVessel/6Dof_dataset_multi_unclampedEOM_withXacc/` — the unclamped-GM variant.

## Code

`jmse/` is a self-contained Python package: data curation, leakage-safe embargoed windowing,
a multi-model forecasting benchmark (persistence · AR · Kalman · ridge · GBM · TCN · Transformer · GRU · LSTM),
three uncertainty-quantification methods with conformal recalibration, a probabilistic
large-inclination-exceedance early-warning decision layer (with a learning-free domain-rule
baseline and a k-of-n hysteresis sweep), leave-one-out generalization folds over sea state /
vessel / wave-phase realization, a clamped-vs-unclamped metacentric-height study, and a
closed-form target-representation error-propagation analysis.

## Setup

```bash
python3 -m pip install -r requirements.txt        # loose bounds
# or, for the exact versions used to produce the released results:
python3 -m pip install -r requirements-lock.txt
python3 -m pytest -q                              # sanity tests
```

## Reproduce

See **[`REPRODUCE.md`](REPRODUCE.md)** for the full end-to-end guide: MATLAB/MSS data
generation, the GPU experiments, and a copy-pasteable *table → command → CSV* and
*figure → command* map covering every table and figure in the article.

The pre-generated dataset is included, so the experiments and analysis can be run without
re-running the MATLAB stage:

```bash
bash scripts/run_phase3.sh                        # full 5-seed experiments (GPU; ~hours)
python3 -m jmse.models.feature_baselines          # ridge/GBM window baselines
make analysis                                     # post-hoc tables/figures from saved predictions (no GPU)
```

> **MATLAB/MSS note.** Regenerating the raw simulations (Stage 1 in `REPRODUCE.md`) requires the
> third-party [Marine Systems Simulator (MSS)](https://github.com/cybergalactic/MSS) toolbox,
> which is **not redistributed here**; install it and place it at `00_Ref/MSS_library/`. The
> simulation drivers (`regen_6dof.m`, `calcXacc_regen.m`, `fishingVessel_unclamped.m`) and the
> exact configuration (`docs/MATLAB_REGEN_SPEC.md`) are provided. Using the included
> pre-generated dataset, the MATLAB stage can be skipped entirely.

## Layout

```
jmse/             analysis package (data, models, uq, earlywarning, eval, plots, train.py)
tests/            pytest sanity tests
scripts/          run_phase3.sh — full experiment driver
data_processed/   curated dataset (parquet) + per-condition summaries + manifests
results/          released tables (*.csv) and figures (*.png)
00_Ref/논문코드_데이터셋/fishingVessel/   raw 6-DOF CSVs + MATLAB simulation drivers
docs/             MATLAB_REGEN_SPEC.md (simulation recipe)
```

## Citation & license

If you use this code or data, please cite the article and this archive (see
[`CITATION.cff`](CITATION.cff)). Released under the MIT License ([`LICENSE`](LICENSE)).
