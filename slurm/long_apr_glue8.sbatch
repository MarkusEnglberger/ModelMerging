#!/bin/bash
#SBATCH --job-name=apr-long-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/long_glue8-%j.out
#SBATCH --error=logs/long_glue8-%j.out

# Long-horizon APR control on GLUE-8: does S=30 + random task order + cosine lr
# annealing close the gap to calibrate-then-refine (0.662), or does plain APR plateau
# near its S=5 value (~0.59)? Arms per replay budget n in {64,32}: short baseline,
# steps-only control (S=30 fixed/constant), annealed long runs at peak lr {16,32},
# and calibrate-then-refine at the same budget as the reference.
# base_lr 16 = the known stable S=5 optimum from TA (lr32 is knife-edge, but annealing decays through it); cal settings = the winning calibrate cell.
source slurm/common.sh
python scripts/long_apr.py --config configs/glue8.yaml \
  --n_probes 64 32 --long_steps 30 --long_lrs 16 32 --schedule cosine --lr_min_frac 0.05 \
  --base_lr 16 --base_steps 5 \
  --cal_lr 0.01 --cal_l2 1e-3 --cal_steps 300 --cal_refine_lr 16 \
  --out results/compare/long_apr_glue8.json "$@"
