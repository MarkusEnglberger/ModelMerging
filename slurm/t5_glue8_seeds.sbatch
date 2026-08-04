#!/bin/bash
#SBATCH --job-name=t5-seeds
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/t5_seeds-%j.out
#SBATCH --error=logs/t5_seeds-%j.out
#
# Pinned-cell seed replication on T5-GLUE n=64 -- the CLIP-20 protocol (tab:
# clip20seeds): configurations frozen at the seed-0 selections, only the replay
# buffer redraws, so the variance isolates the buffer draw.
#
# The two claims this gates, both currently single-seed:
#   S=5  : ungated (lr32, +0.663) > GD (lr0.01, +0.631) > APR (lr128, +0.580)
#   S=20 : GD (lr0.003, +0.741) > APR (lr32, +0.651, interior after the low-lr
#          extension) > ungated (jagged, best-known lr2 +0.557)
# The ungated arc swings +-0.15 between adjacent lrs at S=20, so its cells are
# noise-suspect; seed spread on the pinned cells is the cheapest way to learn
# whether ANY of these margins are real before they enter the paper.
#
# One submission = one (STEPS, SEED): the pinned cells for that horizon, one lr
# per arm. eval0 comes from the cache (probe_seed does not enter the key), so a
# job is 1 pass for TA@l0.3 + 3 cells ~= 4 passes x ~5.4 min + refinement.
#
#   sbatch --export=ALL,STEPS=5,SEED=1  slurm/t5_glue8_seeds.sbatch
#   sbatch --export=ALL,STEPS=20,SEED=2 slurm/t5_glue8_seeds.sbatch

set -euo pipefail
source slurm/common.sh

: "${STEPS:?set STEPS (5|20)}"
: "${SEED:?set SEED}"
: "${NPROBE:=}"          # empty => config n_probe (64)

# Pinned cells per horizon. At n=64 one lr per arm (the seed-0 selections).
# At n=8 the S=5 lists ALSO pin the two dramatic single-seed cells whose
# reproducibility is the question: nogate@16 (catastrophic, -1.98) and
# gd@0.01 (the n=64 optimum that collapsed to +0.284) -- replicating only the
# winners would test the claim while ignoring its most load-bearing part.
if [ "${STEPS}" = 5 ]; then
  if [ "${NPROBE}" = 8 ]; then
    APR_LRS="512"; NG_LRS="8 16"; GD_LRS="0.003 0.01"
  else
    APR_LRS="128"; NG_LRS="32"; GD_LRS="0.01"
  fi
else
  if [ "${NPROBE}" = 8 ]; then
    APR_LRS="32"; NG_LRS="2"; GD_LRS="0.003"
  else
    APR_LRS="32"; NG_LRS="2"; GD_LRS="0.003"
  fi
fi
if [ -n "${NPROBE}" ]; then
  OUT="results/compare/t5_glue8_seeds_S${STEPS}_n${NPROBE}_seed${SEED}.json"
  NPROBE_ARG=(--n_probe "${NPROBE}")
else
  OUT="results/compare/t5_glue8_seeds_S${STEPS}_seed${SEED}.json"
  NPROBE_ARG=()
fi
echo "[job] pinned cells S=${STEPS} n=${NPROBE:-64} seed=${SEED}: apr@{${APR_LRS}} nogate@{${NG_LRS}} gd@{${GD_LRS}} -> ${OUT}"

python scripts/merge_baselines.py \
  --config configs/t5_glue8.yaml \
  --ta_lams 0.3 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --steps "${STEPS}" \
  --probe_seed "${SEED}" \
  "${NPROBE_ARG[@]}" \
  --apr_lrs ${APR_LRS} \
  --nogate_lrs ${NG_LRS} \
  --control_gd_lrs ${GD_LRS} \
  --out "${OUT}"
