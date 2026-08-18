#!/bin/bash
#SBATCH --job-name=apr-mb3b
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mergebench3b-%j.out
#SBATCH --error=logs/mergebench3b-%j.out

# Decoder-track POC (H100): MergeBench Llama-3.2-3B math/coding/instruction experts,
# fair 3-family sweep. EDIT the partition line when the H100 queue name is known.
#
# Memory sizing: base + 3 experts as fp32 task vectors ~ 48 GB CPU RAM, plus one 3B
# model on GPU (~12 GB weights + grads fp32; generation eval adds KV cache at bs 8).
# Set model_dtype: bfloat16 in the config if CPU RAM is tight.
# Wall-clock is dominated by generation eval (max_eval=200/task per eval pass); the
# grids below are deliberately small for the first run -- bracket after eval0 sanity.
# Requires: python scripts/prefetch_mergebench.py   (gated base: huggingface-cli login)
source slurm/common.sh
# reduce allocator fragmentation (~9 GB was reserved-but-unallocated at the batch-16
# OOM); lets fp32 refinement backward reclaim that headroom on the 93 GB H100.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Grid rationale (v3): the v2 sweep found APR flat across lr 4-16 with tiny
# displacements (<=0.006), so the grid extends log-spaced to 64 to find where the
# refinement actually starts to move; no-gate family dropped (per protocol
# decision), ordinary GD kept as the labeled-replay control at its two best lrs.
python scripts/compare_baselines.py --config configs/mergebench_llama3b.yaml \
  --steps 5 --clip_mode vdist \
  --apr_lrs 4 16 64 \
  --nogate_lrs \
  --gd_lrs 5e-5 2e-4 \
  --skip_inverted \
  --out results/compare/mergebench3b_compare_v3.json "$@"
