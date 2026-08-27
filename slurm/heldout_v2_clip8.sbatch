#!/bin/bash
#SBATCH --job-name=heldout-v2-clip8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/heldout_v2_clip8-%j.out
#SBATCH --error=logs/heldout_v2_clip8-%j.out
#
# Held-out zero-shot retention for the protocol-v2 CLIP-8 models (Table
# tab:heldout8). Train/held-out boundary = CLIP-8 membership, fixed by the
# literature: the 12 remaining tasks of the 20-task suite are held out, of
# which seven have real headroom above chance for the base model and are the
# reportable probes (cifar10, stl10, pets, food101, flowers102, cifar100,
# fashion_mnist); the other five are recorded but reported separately.
#
# Every refined state is loaded from cv_protocol.py --save_winners output, so
# the probed model IS the one Table tab:cv-v2 reports, for all three buffer
# draws. The merge rows use the hyperparameters v2 selected on the B=32
# budget (TA l0.2, TIES d0.1/l1, DARE-TIES dd0.5/t0.1/l0.8, BC d0.2/o0.01/l0.4).
# No refinement is run inside this job (all refinement lr grids are empty).

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${BUDGET:=32}"
W="results/cv_winners/clip8_B${BUDGET}_pretrained"
OUT="results/compare/heldout_v2_clip8_B${BUDGET}.json"

STATES=()
for d in 0 1 2; do
  for arm in apr nogate gd; do
    # the monolithic 3-draw run wrote all draws into one directory, tagged by
    # draw; per-draw array runs write draw${d}/<arm>_pretrained_draw0.pt
    for f in "${W}/${arm}_pretrained_draw${d}.pt" "${W}/draw${d}/${arm}_pretrained_draw0.pt"; do
      if [ -f "$f" ]; then STATES+=("${arm}_draw${d}=${f}"); break; fi
    done
  done
done
echo "[job] probing ${#STATES[@]} v2 winner states + merges -> ${OUT}"
printf '       %s\n' "${STATES[@]}"
guard_output "${OUT}"

python scripts/heldout_retention.py --config configs/clip20.yaml \
  --heldout cifar100 stl10 flowers102 pets pcam fer2013 emnist cifar10 food101 fashion_mnist kmnist rendered_sst2 \
  --n_probe "${BUDGET}" --probe_seed 100 \
  --state_files "${STATES[@]}" \
  --ta_lams 0.2 \
  --ties_densities 0.1 --ties_lams 1 \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.8 \
  --bc_densities 0.2 --bc_outliers 0.01 --bc_lams 0.4 \
  --apr_base_lrs --apr_ta_lrs --nogate_base_lrs --gd_base_lrs \
  --out "${OUT}"

echo "[done] -> ${OUT}"
