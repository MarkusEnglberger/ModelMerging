#!/bin/bash
#SBATCH --job-name=ho-smoke
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/heldout_smoke-%j.out
#SBATCH --error=logs/heldout_smoke-%j.out

# Smoke test for scripts/heldout_retention.py on the 3-task POC (train on
# eurosat+gtsrb, hold out mnist), tiny grids: proves the split/refine/record
# path end to end before the real 14/6 job runs (chained via --dependency).
source slurm/common.sh
python scripts/heldout_retention.py --config configs/clip_poc.yaml \
  --heldout mnist \
  --ta_lams 0.3 --ties_densities 0.1 --ties_lams 0.8 \
  --apr_base_lrs 8 --apr_ta_lrs 8 --steps 2 \
  --out results/compare/_heldout_smoke.json
