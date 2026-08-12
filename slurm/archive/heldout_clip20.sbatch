#!/bin/bash
#SBATCH --job-name=ho-clip20
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/heldout_clip20-%j.out
#SBATCH --error=logs/heldout_clip20-%j.out

# HELD-OUT-TASK RETENTION on the 20-task suite: train (merge/refine) on the 14
# specialized/domain-shift tasks, hold out the 6 natural-image tasks with the
# highest base zero-shot (sun397 .63, stl10 .97, cifar10 .88, pets .83,
# food101 .83, flowers102 .66; held-out base mean ~.80 -- there is something
# to lose). Measures the paper's currently-unmeasured proxy: does the
# accuracy-proximity frontier of refining FROM THE BASE MODEL (dist 0.3-0.7)
# translate into preserved zero-shot on never-seen tasks, versus the merges at
# dist 1.0 (TA-14) and ~3 (TIES-14)?
# Arms: base / TA-14 (lambda swept) / TIES-14 / APR-from-base (lr 4,8) /
# APR-from-best-TA (lr 4); S=30 cosine random-order (the converged horizon).
source slurm/common.sh
python scripts/heldout_retention.py --config configs/clip20.yaml \
  --heldout sun397 stl10 cifar10 pets food101 flowers102 \
  --n_probe 64 \
  --ta_lams 0.08 0.1 0.15 0.2 \
  --ties_densities 0.1 --ties_lams 0.6 0.8 \
  --apr_base_lrs 4 8 --apr_ta_lrs 4 \
  --steps 30 --apr_schedule cosine --apr_order random \
  --out results/compare/heldout_retention_clip20.json
