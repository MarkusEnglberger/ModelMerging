#!/bin/bash
#SBATCH --job-name=apr-rgm-clip20
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/regmean_clip20-%j.out
#SBATCH --error=logs/regmean_clip20-%j.out

# RegMean on the 20-task CLIP suite + APR on top. gram_ns 64 = matched data
# budget (report that cell), 256 = the more-data variant. apr_lrs follow the
# CLIP-8 finding that RegMean is a FAR init so APR needs small lr (interior
# peak was 0.5 on clip8; bracket 0.25-2). Reporting by mean_acc.
# Requires: prefetch_clip.py --suite 20.
source slurm/common.sh
python scripts/regmean_baseline.py --config configs/clip20.yaml \
  --n_probe 64 --gram_ns 64 256 --nondiag_scales 0.9 1.0 \
  --apr_lrs 0.25 0.5 1 2 --steps 5 --select_by mean_acc \
  --out results/compare/regmean_clip20.json "$@"
