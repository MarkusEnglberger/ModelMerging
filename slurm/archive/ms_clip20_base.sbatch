#!/bin/bash
#SBATCH --job-name=ms20-base
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ms20_base-%j.out
#SBATCH --error=logs/ms20_base-%j.out

# MULTI-SEED replication, from-base cells (--ta_lams 0 => theta_0). Submit
# once per replay seed (SEED=1..4; seed 0 = existing). Pinned at the
# seed-0-selected configs:
#   n=64: apr S30 lr8 cosine (.6831) vs gd S30 1e-4 constant (.6249)
#   n=8 (the "8 labels beat the tuned merge" headline): apr S30 lr4 cosine
#        (.6316) vs gd S30 1e-5 cosine
set -u
S=${SEED:?set SEED via --export=ALL,SEED=...}
source slurm/common.sh
echo "[ms] probe_seed=${S} base n64+n8"

python scripts/merge_baselines.py --config configs/clip20.yaml \
  --probe_seed "${S}" \
  --ta_lams 0 \
  --skip_families ties dare dareties bc ada ls \
  --refine_from ta \
  --apr_lrs 8 --apr_schedules cosine --apr_order random \
  --gd_lrs 1e-4 --gd_steps 30 --gd_schedules constant \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 30 --select_by mean_acc \
  --out "results/compare/ms_clip20_base_n64_seed${S}.json"

python scripts/merge_baselines.py --config configs/clip20.yaml \
  --probe_seed "${S}" --n_probe 8 \
  --ta_lams 0 \
  --skip_families ties dare dareties bc ada ls \
  --refine_from ta \
  --apr_lrs 4 --apr_schedules cosine --apr_order random \
  --gd_lrs 1e-5 --gd_steps 30 --gd_schedules cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 30 --select_by mean_acc \
  --out "results/compare/ms_clip20_base_n8_seed${S}.json"
