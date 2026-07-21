#!/bin/bash
#SBATCH --job-name=apr-frombase
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=logs/frombase-%j.out
#SBATCH --error=logs/frombase-%j.out

# APR / ordinary GD initialized at the BASE MODEL theta_0 (no merge at all).
# --ta_lams 0 makes merge:TA@l0 == theta_0, so refine_from ta refines from base
# with ZERO code changes; refine-cell "displacement" is then exactly
# ||refined - theta_0|| (distance from base), the quantity of interest: can
# refinement reach good performance while staying CLOSE to the base model,
# vs merges that sit ~1.0+ away (TA l0.08 = 1.04)?
# Two purposes: (1) the "do you need merging at all?" ablation; (2) a
# falsifiable test of the geometry story -- from theta_0 the good region is
# FAR (>=1.0) while GD@S30 typically travels only .14-.23, so the prediction
# is APR >> GD from base at matched budget (reverse of the near-init parity).
# From-base |v_i| = full task vectors, so APR's optimal lr may sit HIGHER than
# the from-merge lr4 (cf. clip8 lr8-32) -- grid includes 8.
# Old n=64 protocol for comparability with the init x method matrix. No nogate.
# Multi-partition + --exclude tue-gpua001: first free non-MIG GPU anywhere
# (MIG slices have 6GB VRAM and the node lacks free host RAM).
source slurm/common.sh

# matched budget S=5
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0 \
  --skip_families ties dare dareties bc ada \
  --refine_from ta \
  --apr_lrs 2 4 8 \
  --gd_lrs 5e-5 1e-4 5e-4 \
  --gd_steps 5 --gd_schedules constant \
  --steps 5 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_base_s5.json

# long budget S=30, constant + cosine/random (mirrors the matrix long arms)
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0 \
  --skip_families ties dare dareties bc ada \
  --refine_from ta \
  --apr_lrs 2 4 8 \
  --apr_schedules constant cosine --apr_order random \
  --gd_lrs 5e-5 1e-4 \
  --gd_steps 30 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 30 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_base_s30.json
