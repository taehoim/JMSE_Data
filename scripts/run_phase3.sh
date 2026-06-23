#!/usr/bin/env bash
# Phase 3 — full multi-seed re-run of all experiments on the regenerated
# multi-realization dataset (R1 revision). Continue-on-error; per-step exit logged.
# GPU steps first, then non-GPU analysis. Re-runnable: each entrypoint overwrites
# its own results/ outputs. Generalization studies use 5 seeds per the R1 plan.
#
#   bash scripts/run_phase3.sh
set +e
cd "$(dirname "$0")/.."
SEEDS="0 1 2 3 4"
echo "###### PHASE 3 FULL RUN ######"; date '+start %Y-%m-%d %H:%M:%S'
step(){ echo; echo "===== $1 ====="; date '+  %H:%M:%S'; bash -c "$2"; echo ">>> EXIT[$1]=$?"; }

# ---- GPU experiments (regenerate predictions) ----
step sweep          "python3 -m jmse.sweep --seeds $SEEDS"
step uq_run         "python3 -m jmse.uq.run --seeds $SEEDS"
step earlywarning   "python3 -m jmse.earlywarning.run --method quantile --seeds $SEEDS"
step hysteresis     "python3 -m jmse.earlywarning.hysteresis --seeds $SEEDS"
step ood            "python3 -m jmse.eval.ood --seeds $SEEDS"
step combined       "python3 -c \"from jmse.eval.ood import run_combined_holdout; run_combined_holdout(seeds=[0,1,2,3,4])\""
step ood_uq         "python3 -m jmse.uq.ood_uq --hold-hs 7 --seeds $SEEDS"
step representation "python3 -m jmse.eval.representation --seeds $SEEDS"
step ablation       "python3 -m jmse.eval.ablation --seeds $SEEDS"
step gm_floor       "python3 -c \"
from jmse.eval.gm_floor import run_gm_floor
import pandas as pd, glob
for s in range(5):
    try: run_gm_floor(seed=s, out_dir=f'results/gm_floor/seed{s}')
    except Exception as e: print('gm_floor seed', s, 'FAILED:', e)
ps=sorted(glob.glob('results/gm_floor/seed*/T_gmfloor.csv'))
if ps:
    df=pd.concat([pd.read_csv(p).assign(seed=i) for i,p in enumerate(ps)], ignore_index=True)
    df.to_csv('results/gm_floor/T_gmfloor_allseeds.csv', index=False)
    agg=df.groupby('variant').agg(gm_10t_m=('gm_10t_m','first'), rmse_mean=('rmse_deg','mean'), rmse_std=('rmse_deg','std'), r2_mean=('r2','mean'), r2_std=('r2','std'))
    agg.to_csv('results/gm_floor/T16b_gmfloor.csv'); print(agg)
else: print('NO gm_floor per-seed csvs found')
\""

# ---- non-GPU analysis (post-process the predictions) ----
step uq_rigor       "python3 -m jmse.uq.rigor --seeds $SEEDS"
step events_run     "python3 -m jmse.earlywarning.events_run --seeds $SEEDS --refractory 0"
step sensitivity    "python3 -m jmse.earlywarning.sensitivity --seeds $SEEDS --threshold 15"
step statrigor      "python3 -m jmse.eval.statrigor --models lstm gru"
step significance   "python3 -m jmse.eval.significance_run --seeds $SEEDS --n-boot 600"

echo; echo "###### PHASE 3 DONE ######"; date '+end %Y-%m-%d %H:%M:%S'
