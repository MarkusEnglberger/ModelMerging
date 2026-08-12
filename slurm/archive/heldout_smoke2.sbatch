#!/bin/bash
#SBATCH --job-name=ho-smoke2
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/heldout_smoke2-%j.out
#SBATCH --error=logs/heldout_smoke2-%j.out

# Smoke test for the ADDED baseline arms in heldout_retention.py (DARE-TIES,
# Breadcrumbs, AdaMerging, RegMean, and the gd / nogate / apr<-ada refinement
# arms), on the 3-task POC with tiny grids. Proves every new code path runs
# before the ~2.5h clip20 job is committed to it.
source slurm/common.sh
python scripts/heldout_retention.py --config configs/clip_poc.yaml \
  --heldout mnist \
  --ta_lams 0.3 --ties_densities 0.1 --ties_lams 0.8 \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.4 \
  --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
  --ada --ada_steps 20 --ada_init_lam 0.3 \
  --regmean --regmean_nd 1.0 \
  --apr_base_lrs 8 --apr_ta_lrs 8 --apr_ada_lrs 8 \
  --gd_base_lrs 1e-4 --nogate_base_lrs 2 \
  --steps 2 \
  --out results/compare/_heldout_smoke2.json
