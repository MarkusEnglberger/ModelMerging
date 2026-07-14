#!/bin/bash
#SBATCH --job-name=apr-mrg3-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/merge3_clip8-%j.out
#SBATCH --error=logs/merge3_clip8-%j.out

# Run 3: fix the three grid-edge problems left by job 6124434.
#  (a) TA lambda was UNBRACKETED at the low edge (0.2->.407, 0.3->.270, 0.4->-.049):
#      sweep down to 0.05. The paper's CLIP merge baseline (lam=0.3) is badly undertuned,
#      and a tuned TA already has POSITIVE worst-task, which contradicts the current draft.
#  (b) Breadcrumbs got a single lambda (0.4) => no tuning budget. Sweep it.
#  (c) APR from {ta, bc, ada} peaked at the HIGH edge lr8 (monotone increasing) => those
#      numbers are lower bounds. Extend the lr grid upward to 24.
# TIES / DARE-TIES are pinned to their known-best cells (they are already bracketed:
# APR<-TIES peaks at an interior lr6). DARE (control) is skipped -- it is ~= TA by construction.
# AdaMerging: layer-wise only (task-wise is the weaker ablation, .333 vs .604).
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip8.yaml \
  --ta_lams 0.05 0.1 0.15 0.2 0.25 \
  --ties_densities 0.1 --ties_lams 0.8 \
  --skip_families dare \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.8 \
  --bc_densities 0.2 --bc_outliers 0.01 --bc_lams 0.1 0.2 0.3 0.4 \
  --ada_variants layer --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 --ada_workers 0 \
  --refine_from ta ties dareties bc ada --apr_lrs 4 6 8 12 16 24 --steps 5 \
  --out results/compare/merge_baselines3_clip8.json "$@"
