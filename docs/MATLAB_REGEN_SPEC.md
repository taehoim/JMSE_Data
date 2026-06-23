# MATLAB/MSS regeneration spec (author-side)

Two reviewer items (2.4 multi-realization; 2.8 leave-one-realization-out + unclamped 10 t) need new
data from the MSS generator. This is the exact recipe; outputs drop into the existing pipeline.

Source files (in `00_Ref/논문코드_데이터셋/fishingVessel/`):
- `generate6Dof_dataset.m` — sweeps tonnage × Hs, writes `6Dof_<ton>..._Hs<hs>.csv`
  (cols: `time,u,v,w,p,q,r,phi,theta`). Phase realization: `rng(12345 + hs_idx*100 + t_idx)`.
- `calculateXacc_dataset.m` — adds `Xacc=sqrt(phi^2+theta^2)`, `GZ=GM*sin(Xacc)`; GM floored `if GM_T<0.3, GM_T=0.3`.

Current config (keep unless noted): Modified PM spectrum, **T0=8 s**, heading **140°**, no spreading,
100 components ≤3 rad/s, **zero speed**, h=1 s, T_final=5000 (→5001 samples). Hs∈{3,5,7}, ton∈{10,20,30,40,50}.

## Task A — multiple wave realizations per condition (2.4, 2.8-LORO)
Goal: R≥5 independent phase realizations per (tonnage, Hs) so we can do leave-one-realization-out and
record-level CIs.

In `generate6Dof_dataset.m`, wrap the (hs, ton) loop body in `for r_idx = 1:R` and change the seed +
filename to carry `r_idx`:
```matlab
R = 5;                                   % realizations per condition
% ... inside the hs/ton loops, add:  for r_idx = 1:R
rng(12345 + hs_idx*100 + t_idx*10 + r_idx);     % distinct phase set per realization
% filename: sprintf('6Dof_%dton_L%.1f_B%.1f_T%.1f_Hs%.1f_r%d.csv', tonnage,L,B,T,Hs,r_idx)
```
Then run `calculateXacc_dataset.m` over the new files (it globs the folder). Deliver the
`6Dof_*_withXacc` CSVs. **Do not change** T0/heading/speed for this task (isolates phase variability);
a separate sweep varying T0/heading/speed would address encounter-geometry generalization.

## Task B — unclamped (or consistently clamped) 10 t (2.8 GM-confound)
Goal: re-evaluate the 10 t LOVO fold without the hydrostatic discontinuity (10 t is the only hull whose
true GM=0.258 m hits the 0.30 m floor).

Make a variant set with the floor removed:
```matlab
% in calculateXacc_dataset.m, replace the clamp with a pass-through (or floor ALL vessels identically):
% if GM_T < 0.3, GM_T = 0.3; end   ->   (delete, or set GM_FLOOR=0 for all)
```
Regenerate at least the 10 t records (all 3 Hs; with the Task-A realizations if available). Keep the
clamped set too, so we can report clamped-vs-unclamped side by side.

## Output / hand-off
- Put new `*_withXacc` CSVs in a folder and tell me the path (filenames must keep
  `<ton>ton ... Hs<hs>` and, for Task A, the `_r<idx>` suffix).
- I will: extend `jmse/data/curate.py` + `windowing.py` to carry a `realization` id, add
  leave-one-realization-out folds in `eval/ood.py`, and add the clamped-vs-unclamped 10 t comparison.
- Python-side `compute_gm(..., clamp=False)` already exists in `curate.py`, so the unclamped GM path is
  ready once the unclamped motion CSVs arrive.
