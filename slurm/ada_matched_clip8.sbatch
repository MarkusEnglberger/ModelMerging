#!/bin/bash
#SBATCH --job-name=apr-ada-m-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ada_matched_clip8-%j.out
#SBATCH --error=logs/ada_matched_clip8-%j.out

# Matched-budget AdaMerging on CLIP-8.
#
# The AdaMerging numbers in jobs 6124434/6126998 use the STANDARD transductive protocol:
# unlabeled inputs drawn from the very split they are scored on, effectively unbounded
# (300 steps x bs16 = 4800 draws/task). APR gets 64 LABELED examples per task from train.
# That is not apples-to-apples, and it violates the proposal's own stated protocol, which
# asks that label-free baselines get "the same number of unlabeled examples".
#
# Here AdaMerging is given exactly APR's replay-buffer inputs (n_probe per task, from the
# TRAIN split) with labels stripped. The only remaining differences are (i) labels and
# (ii) the objective -- AdaMerging still gets ~7x more gradient steps, which is generous.
# It also makes APR<-AdaMerging an INDUCTIVE result: the 0.707 headline currently inherits
# test-set access through its init.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip8.yaml \
  --skip_families ta ties dare dareties bc \
  --ada_variants task layer --ada_data probe_buffer \
  --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 --ada_workers 0 \
  --refine_from ada --apr_lrs 4 6 8 12 --steps 5 \
  --out results/compare/ada_matched_clip8.json "$@"
