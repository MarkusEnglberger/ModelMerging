#!/bin/bash
#SBATCH --job-name=apr-rm-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/regmean_clip8-%j.out
#SBATCH --error=logs/regmean_clip8-%j.out

# RegMean on CLIP-8 (the owed label-free baseline) + APR on top. Closed-form per-linear
# merge from input Gram matrices over unlabeled inputs; reported at the matched 64-input
# budget and a larger 256-input sample (RegMean is label-free, normally given more), with
# and without off-diagonal shrinkage. The APR stage still uses only the 64 labeled examples.
source slurm/common.sh
python scripts/regmean_baseline.py --config configs/clip8.yaml \
  --n_probe 64 --gram_ns 64 256 --nondiag_scales 0.9 1.0 --eps 1e-3 --gram_bs 16 \
  --apr_lrs 4 6 8 --steps 5 \
  --out results/compare/regmean_clip8.json "$@"
