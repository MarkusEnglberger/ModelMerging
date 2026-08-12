#!/bin/bash
#SBATCH --job-name=apr-long-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/long_clip8-%j.out
#SBATCH --error=logs/long_clip8-%j.out

# Long-horizon APR control on CLIP-8: does S=30 + random task order + cosine lr
# annealing close the gap to calibrate-then-refine (0.825), or does plain APR plateau
# near its S=5 value (~0.60)? Arms per replay budget n in {64,32}: short baseline,
# steps-only control (S=30 fixed/constant), annealed long runs at peak lr {8,16},
# and calibrate-then-refine at the same budget as the reference.
# base_lr 8 = the known S=5 optimum from TA; cal settings = the winning calibrate cell.
source slurm/common.sh
python scripts/long_apr.py --config configs/clip8.yaml \
  --n_probes 64 32 --long_steps 30 --long_lrs 8 16 --schedule cosine --lr_min_frac 0.05 \
  --base_lr 8 --base_steps 5 \
  --cal_lr 0.03 --cal_l2 1e-3 --cal_steps 300 --cal_refine_lr 8 \
  --out results/compare/long_apr_clip8.json "$@"
