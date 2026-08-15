#!/bin/bash
#SBATCH --job-name=mb3b-full
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mb3b_full-%j.out
#SBATCH --error=logs/mb3b_full-%j.out

# DEFERRED -- do not submit until the (lr, S) grid is settled at the cheap
# protocol. Cancelled once at 9 min (job 25193797) for exactly that reason: it
# re-scores the S=5 winners at 300 items, but the S=20 job running alongside it
# may move which cells win, which would make ~12 h of full-precision evaluation
# measure the wrong cells. Buy precision once, at the end.
#
# Extension of the reduced sweep (results/compare/mb3b_gd_matrix_sweep.json),
# which found the anchored arms above ordinary GD at 3B but could not be quoted:
# instruction ran at 150 items (+-0.039 se) and the headline separation rides on
# that one task, and GD's best cell sat at 1e-5, the grid's lower edge.
#
# This job fixes both, at the FULL protocol (configs/mergebench_llama3b.yaml,
# instruction=300) so the numbers are comparable to every other 300-item run on
# disk (mb3b_composition.json et al.):
#
#   apr@256, nogate@128   the two anchored winners, re-scored at full precision
#   gd@1e-5, gd@1e-3      GD's top two cells -- they TIED at 0.4578 in the sweep,
#                         so both are re-scored rather than picking one on noise
#   gd@3e-6, gd@1e-6      closes GD's low edge, the criticism that would
#                         otherwise invalidate the headline
#
# GD deliberately gets 4 cells against 1 for each anchored arm: it is the arm
# under test, and every asymmetry here is set to favour it.
#
# COST (sized from measured per-task times, not from a previous job's total --
# the mistake that killed 25178513): full-protocol pass = math 777 + coding 687
# + instruction ~2900 + multilingual 58 ~= 74 min. Passes = 3 (eval0, no cache
# exists at 300) + 1 (TA@l0.5) + 6 (cells) = 10, so ~12.3 h plus ~25 min of
# refinement compute, against a 16 h limit.
#
# NOTE eval0 at 300 will be written to results/eval0_cache/, so any later
# full-protocol job on this track skips those 3 passes (~3.7 h).

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --ta_lams 0.5 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --apr_lrs 256 \
  --nogate_lrs 128 \
  --control_gd_lrs 1e-6 3e-6 1e-5 1e-3 \
  --out results/compare/mb3b_gd_matrix_full.json
