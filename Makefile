# Reproducibility entry points for the JMSE marine-safety study.
#
#   make test         run the unit/integration test suite
#   make analysis     regenerate all post-hoc tables/figures from the committed predictions (no GPU)
#   make figures      regenerate the analysis figures under results/
#   make experiments  re-run the GPU experiments that produce the predictions (slow; needs a GPU)
#   make all          analysis + figures
#
# The `analysis` target reproduces every post-hoc number from the saved per-seed predictions under
# results/ (which are committed); it needs no GPU. The `experiments` target regenerates those
# predictions from scratch and requires a CUDA GPU. See REPRODUCE.md for the full guide and the
# table/figure -> command -> CSV map.

PY ?= python3
SEEDS ?= 0 1 2 3 4

.PHONY: test analysis figures experiments all

test:
	$(PY) -m pytest -q

analysis:
	$(PY) -m jmse.uq.rigor --seeds $(SEEDS)
	$(PY) -m jmse.earlywarning.events_run --seeds $(SEEDS) --refractory 0
	$(PY) -m jmse.earlywarning.sensitivity --seeds $(SEEDS) --threshold 15
	$(PY) -m jmse.eval.statrigor --models lstm gru
	$(PY) -m jmse.eval.significance_run --seeds $(SEEDS) --n-boot 600

figures:
	$(PY) -c "from jmse.plots.uq_rigor import plot_F12; plot_F12('results/uq')"
	$(PY) -c "from jmse.plots.events import plot_F13; plot_F13('results/earlywarning')"
	$(PY) -c "from jmse.plots.events import plot_cost_curve; plot_cost_curve('results/sensitivity')"
	$(PY) -c "from jmse.plots.hysteresis import plot_F_hysteresis; plot_F_hysteresis('results/sensitivity')"

# GPU experiments that produce the predictions consumed by `analysis` (slow). For the exact
# 5-seed driver used in the paper, see scripts/run_phase3.sh.
experiments:
	$(PY) -m jmse.sweep
	$(PY) -m jmse.models.feature_baselines
	$(PY) -m jmse.uq.run --config jmse/run_configs/uq_id.yaml --seeds $(SEEDS)
	$(PY) -m jmse.earlywarning.run --method quantile --seeds $(SEEDS)
	$(PY) -m jmse.earlywarning.hysteresis --seeds $(SEEDS)
	$(PY) -m jmse.eval.ood --seeds $(SEEDS)
	$(PY) -c "from jmse.eval.ood import run_combined_holdout; run_combined_holdout(seeds=[0,1,2,3,4])"
	$(PY) -m jmse.uq.ood_uq --hold-hs 7 --seeds $(SEEDS)
	$(PY) -m jmse.eval.representation --config jmse/run_configs/lstm_id.yaml --seeds $(SEEDS)
	$(PY) -m jmse.eval.ablation --config jmse/run_configs/lstm_id.yaml --seeds $(SEEDS)

all: analysis figures
