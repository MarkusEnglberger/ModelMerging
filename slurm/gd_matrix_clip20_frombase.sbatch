#!/bin/bash
#SBATCH --job-name=apr-frombase
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/frombase-%A_%a.out
#SBATCH --error=logs/frombase-%A_%a.out

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
# The two budgets are independent array tasks on Snellius, so S=5 and S=30 can
# run concurrently instead of occupying one GPU serially.
set -euo pipefail
cd /home/menglberger/ModelMerging
source slurm/common.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common_args=(
  --config configs/clip20.yaml
  --ta_lams 0
  --skip_families ties dare dareties bc ada
  --refine_from ta
  --apr_lrs 2 4 8
  --select_by mean_acc
)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # matched budget S=5
    python scripts/merge_baselines.py "${common_args[@]}" \
      --gd_lrs 5e-5 1e-4 5e-4 \
      --gd_steps 5 --gd_schedules constant \
      --steps 5 \
      --out results/compare/gd_matrix_clip20_base_s5.json "$@"
    ;;
  1)
    # long budget S=30, constant + cosine/random (mirrors matrix long arms)
    python scripts/merge_baselines.py "${common_args[@]}" \
      --apr_schedules constant cosine --apr_order random \
      --gd_lrs 5e-5 1e-4 \
      --gd_steps 30 --gd_schedules constant cosine \
      --gd_lr_min_frac 0.05 --gd_order random \
      --steps 30 \
      --out results/compare/gd_matrix_clip20_base_s30.json "$@"
    ;;
esac
