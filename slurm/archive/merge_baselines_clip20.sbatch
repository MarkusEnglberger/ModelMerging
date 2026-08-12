#!/bin/bash
#SBATCH --job-name=apr-mrg-clip20
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/merge_clip20-%j.out
#SBATCH --error=logs/merge_clip20-%j.out

# Merge baselines on the full 20-task CLIP ViT-B/32 suite (TALL-masks). The point
# (per the lambda caveat): DON'T refine APR from a hand-picked global lambda --
# initialize from inits that carry their own tuned/learned scaling:
#   * TA  -- lambda SWEPT, grid extended LOW (0.05..0.3): at 20 tasks the
#            task-arithmetic optimum falls well below the 8-task 0.2-0.3.
#   * TIES -- density + lambda swept (sign-election picks the scale).
#   * AdaMerging -- LEARNS per-task/-layer lambda_i by entropy-min, so it
#            self-adapts to 20 tasks with no hand tuning (best data-free on CLIP-8).
# APR then refines on top of each via run_refine_from; apr_lrs extended LOW since
# the optimal APR step drops with task count (3->8 tasks moved it 32->8).
# Controls dare/dareties/breadcrumbs skipped for a tractable single-seed first
# pass -- add later. --mem 128G: 20 experts + image evalsets (~2.5x the 8-task
# ~33G resident); nodes have 244G. Requires: prefetch_clip.py --suite 20.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0.05 0.08 0.1 0.15 0.2 0.3 \
  --ties_densities 0.1 0.2 --ties_lams 0.8 1.0 \
  --ada_variants task layer --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 --ada_workers 0 \
  --skip_families dare dareties bc \
  --refine_from ta ties ada --apr_lrs 1 2 4 8 --steps 5 \
  --out results/compare/merge_baselines_clip20.json "$@"
