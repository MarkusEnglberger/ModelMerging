#!/bin/bash
#SBATCH --job-name=apr-mrg-mb3b
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/merge_mb3b-%j.out
#SBATCH --error=logs/merge_mb3b-%j.out

# Merge baselines on the MergeBench decoder track (4 domains incl. multilingual):
# tuned-lambda task arithmetic, Localize-and-Stitch (data-free magnitude variant),
# and AdaMerging task/layer-wise with the causal adaptation (masked prompt-token
# entropy; label-free) on the matched replay budget (probe_buffer -- eval_ds is not
# collator-compatible for causal rows and would be transductive anyway).
# TIES/DARE/BC are skipped for now: sign-election families need their own careful
# lr re-centering for composition and can be a follow-up once the landscape is known.
#
# --mem 180G (the 1-GPU billing floor): the task vectors (4 x 3B fp32 ~ 48 GB) must be
# retained to build the merge grid, but with --refine_from empty the script frees the
# unused expert-anchor clones (another 48 GB), which keeps the peak under 180G.
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/merge_baselines.py --config configs/mergebench_llama3b.yaml \
  --ta_lams 0.2 0.3 0.4 \
  --skip_families ties dare dareties bc \
  --ls_fracs 0.05 0.1 --ls_lams 1.0 \
  --ada_variants task layer --ada_data probe_buffer --ada_steps 200 --ada_bs 8 \
  --refine_from \
  --out results/compare/merge_baselines_mergebench3b.json "$@"
