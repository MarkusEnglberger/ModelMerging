#!/bin/bash
#SBATCH --job-name=apr-rgmctl-clip20
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/regmean_ctl_clip20-%j.out
#SBATCH --error=logs/regmean_ctl_clip20-%j.out

# Refinement CONTROLS from the RegMean init: does RegMean + ANY local descent
# match RegMean + APR (.7485/.4857 @ lr0.5, S=5), or is the gated anchored
# update doing the work? Rebuilds exactly the init the APR run used --
# regmean@n64,nd1 (matched budget, replay-selected, deterministic given the same
# seed/buffers) -- so every arm starts from the identical state.
#
# Arms:
#   nogate  @ S5           -- anchored/distance-scaled/clipped but UNGATED
#                             (isolates the gate). lrs bracket below APR's 0.5
#                             peak; nogate diverges earlier than APR everywhere.
#   GD constant @ S5       -- matched-budget plain SGD (the fair head-to-head).
#   GD cosine   @ S15/30/60 -- the LONG-HORIZON + ANNEALED form of "GD just
#                             needs more sweeps": cosine decay to 5% of peak lr
#                             with random task order, mirroring scripts/long_apr.py.
#                             5e-4 diverged at constant lr from TA inits, so it
#                             is included here precisely to test whether
#                             annealing rescues the aggressive step size.
# --apr_lrs empty: apr<-regmean cells already exist in regmean_clip20.json.
# Cost: 3*5 + 3*5 + 2*(15+30+60)=240 sweeps + ~15 full 20-task evals.
source slurm/common.sh
python scripts/regmean_baseline.py --config configs/clip20.yaml \
  --n_probe 64 --gram_ns 64 --nondiag_scales 1.0 \
  --apr_lrs \
  --nogate_lrs 0.125 0.25 0.5 \
  --gd_lrs 1e-4 5e-4 --gd_steps 5 15 30 60 \
  --gd_schedules constant cosine --gd_lr_min_frac 0.05 --gd_order random \
  --steps 5 --select_by mean_acc \
  --out results/compare/regmean_clip20_controls.json "$@"
