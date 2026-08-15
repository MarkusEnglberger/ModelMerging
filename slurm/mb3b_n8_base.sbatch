#!/bin/bash
#SBATCH --job-name=mb3b-n8-base
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mb3b_n8_base-%j.out
#SBATCH --error=logs/mb3b_n8_base-%j.out
#
# MergeBench-3B from-pretrained column at n=8+8: APR and ordinary GD refined
# from theta_0 (--ta_lams 0 makes merge:TA@l0 the pretrained model, mirroring
# the t2t MODE=base runs), split-selected on held-out replay loss. Supersedes
# the retired full-protocol re-score (slurm/archive/mb3b_gd_matrix_full.sbatch):
# under the n+n protocol selection never touches the evaluation split, so the
# full-precision evaluation is bought once, for the selected cell per arm,
# instead of re-scoring eval-selected winners.
#
# Grids: constant schedule only -- the 3-seed CLIP-8 study closed the annealing
# question (+0.002 +- 0.032, N=18 pairs) and constant lets the S column reuse
# one trajectory per lr (cost max(S), not sum(S)). APR lrs bracket the from-TA
# optimum (256, still best at S=20) upward, since the from-pretrained regime
# lived at higher rates on t2t, and downward to 16 because the smoke's APR
# pick sat at its low edge and the lr*S ridge maps 256@S=20 to ~50@S=100, so
# the optimum must be interior from both sides. GD keeps its S=5-sweep range
# and closes the 1e-5 low-edge criticism with 3e-6. S in {5,20,50,100}: the pretrained t2t
# S<=200 run suggested 200 is past the optimum, and constant-schedule
# trajectories resume exactly if the selected S lands on the 100 edge.
#
# COST (H100-measured): full-protocol pass = math 777 + coding 687 +
# instruction ~2900 + multilingual 58 ~= 74 min. eval0 ~3 passes (no 300-item
# cache) + TA@l0 1 + 2 selected cells = ~6 passes ~= 7.4 h. Refinement: 13
# trajectories x 100 sweeps at n=8 ~= 5-6 h. Total ~13 h against 16 h.
# Smoke-validated by slurm/mb3b_n8_smoke.sbatch (job 25708100): selection
# recorded on 7/7 cells, buffers disjoint, only selected cells evaluated.

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --n_probe 8 --n_select 8 \
  --steps 5 20 50 100 \
  --select_by mean_acc \
  --ta_lams 0 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --apr_lrs 16 32 64 128 256 512 1024 \
  --control_gd_lrs 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 \
  --out results/compare/grid_nn_mb3b_base_n8_s100.json

echo "[done] -> results/compare/grid_nn_mb3b_base_n8_s100.json"
