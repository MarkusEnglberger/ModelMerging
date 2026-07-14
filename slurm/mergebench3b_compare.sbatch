#!/bin/bash
#SBATCH --job-name=apr-mb3b
#SBATCH --partition=CHANGE_ME_H100_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
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
python scripts/compare_baselines.py --config configs/mergebench_llama3b.yaml \
  --steps 5 --clip_mode vdist_pre --gammas 1.0 \
  --apr_lrs 4 8 16 \
  --nogate_lrs 1 2 4 \
  --gd_lrs 1e-5 5e-5 1e-4 \
  --skip_inverted \
  --out results/compare/mergebench3b_compare.json "$@"
