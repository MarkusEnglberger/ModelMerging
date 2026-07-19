#!/bin/bash
#SBATCH --job-name=apr-mrg-t5
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/merge_t5_glue8-%j.out
#SBATCH --error=logs/merge_t5_glue8-%j.out

# Merge baselines + composition on the SHARED-HEAD T5 GLUE-8 track (flan-t5-base,
# text-to-text): tuned-lambda task arithmetic, TIES, DARE (control), DARE-TIES,
# Model Breadcrumbs -- plus APR refinement on top of each. This is the T5 analogue
# of slurm/merge_baselines_glue8.sbatch, filling the composition table for the
# shared-head setting. Same experts / replay buffers / LM head as t5_glue8_compare.
#
# AdaMerging is INTENTIONALLY skipped: its entropy objective is written for a single
# classification distribution per example (GLUE/CLIP heads); on T5 the logits are
# token-level over the vocab, so it needs a seq2seq adaptation (deferred).
#
# APR lr grid {2,4,8,16,32}: WIDER than a single init needs, on purpose. APR's step
# and trust region scale with |v_i| = |theta_i - theta_init|, which is init-dependent
# -- the sparser TIES/DARE-TIES inits sit farther from the experts in the dropped
# coords, so they want LOWER lr while task-arithmetic peaked at lr32 in the compare
# run. The shared grid lets each init find its own interior optimum (diverged cells
# just lose); the script warns if any family pegs at a grid edge (optimum unbracketed).
# Grids otherwise mirror the GLUE-8 merge-baselines run. Benefits from the STS-B
# metrics NaN guard, so aggregates are clean (unlike the earlier compare job).
# Requires: python scripts/prefetch_t5.py   (already cached)
source slurm/common.sh
python scripts/merge_baselines.py --config configs/t5_glue8.yaml \
  --ta_lams 0.2 0.3 0.4 0.5 \
  --ties_densities 0.1 0.2 --ties_lams 0.8 1.0 \
  --dare_densities 0.3 --dare_seeds 0 1 \
  --dt_drops 0.5 --dt_trims 0.1 0.2 --dt_lams 0.8 1.0 \
  --bc_densities 0.1 0.2 --bc_outliers 0.01 0.05 --bc_lams 0.4 \
  --skip_families ada \
  --refine_from ta ties dareties bc --apr_lrs 2 4 8 16 32 --steps 5 \
  --out results/compare/merge_baselines_t5_glue8.json "$@"
