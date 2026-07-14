#!/bin/bash
#SBATCH --job-name=apr-mrg-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/merge_clip8-%j.out
#SBATCH --error=logs/merge_clip8-%j.out

# Merge baselines on the 8-task CLIP ViT-B/32 suite: task arithmetic (lambda swept),
# TIES, DARE (control), DARE-TIES, Model Breadcrumbs, AdaMerging (task+layer wise),
# plus APR refinement on top of each init. Mirrors slurm/merge_baselines_glue8.sbatch.
#
# APR lr grid {2,3,4,6,8}: extended DOWNWARD. In job 6124118 APR-from-TIES peaked at
# lr6, the low edge of {6,8}, so its optimum was not bracketed; a better/sparser init
# has smaller |v_i| and wants a smaller lr.
# --mem 96G: ~33G resident (8 vision models + expert encoders + image evalsets) plus
# the merge families' CPU task-vector copies OOM-killed job 6124028 at 48G.
# AdaMerging keeps task vectors on the GPU (~2.8G) and uses ada_workers=0 to avoid
# forking 8 tasks x 8 eval workers.
# Requires: python scripts/prefetch_clip.py --all
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip8.yaml \
  --ta_lams 0.2 0.3 0.4 0.5 \
  --ties_densities 0.1 0.2 --ties_lams 0.8 1.0 \
  --dare_densities 0.3 --dare_seeds 0 1 \
  --dt_drops 0.5 --dt_trims 0.1 0.2 --dt_lams 0.8 1.0 \
  --bc_densities 0.1 0.2 --bc_outliers 0.01 0.05 --bc_lams 0.4 \
  --ada_variants task layer --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 --ada_workers 0 \
  --refine_from ta ties dareties bc ada --apr_lrs 2 3 4 6 8 --steps 5 \
  --out results/compare/merge_baselines2_clip8.json "$@"
