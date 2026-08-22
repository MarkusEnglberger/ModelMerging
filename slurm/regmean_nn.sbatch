#!/bin/bash
#SBATCH --job-name=regmean-nn
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --output=logs/regmean_nn-%j.out
#SBATCH --error=logs/regmean_nn-%j.out

# RegMean and the three refinement arms under the strict n+n selection
# protocol. This is intentionally a standalone wave: the other merge families
# already have canonical result files and must not be recomputed or overwritten.

set -euo pipefail
source slurm/common.sh

: "${BENCH:?set BENCH (glue8|clip8|clip20)}"
: "${N:=8}"
: "${STEPS_GRID:=5 20 50}"

case "$BENCH" in
  glue8)
    CFG=configs/glue8.yaml
    APR="0.03125 0.0625 0.125 0.25 0.5 1 2"
    NG="0.03125 0.0625 0.125 0.25 0.5 1"
    GD="1e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3"
    SELECT=mean_normret
    ;;
  clip8)
    CFG=configs/clip8.yaml
    APR="0.03125 0.0625 0.125 0.25 0.5 1 2"
    NG="0.03125 0.0625 0.125 0.25 0.5 1"
    GD="1e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3"
    SELECT=mean_normret
    ;;
  clip20)
    CFG=configs/clip20_l008.yaml
    APR="0.03125 0.0625 0.125 0.25 0.5 1 2"
    NG="0.03125 0.0625 0.125 0.25 0.5 1"
    GD="1e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3"
    SELECT=mean_acc
    ;;
  *) echo "unknown BENCH=$BENCH" >&2; exit 2 ;;
esac

OUT="results/compare/grid_nn_${BENCH}_regmean_n${N}.json"
echo "[job] RegMean BENCH=$BENCH n=${N}+${N} S in {$STEPS_GRID} -> $OUT"

python scripts/merge_baselines.py \
  --config "$CFG" --n_probe "$N" --n_select "$N" --steps $STEPS_GRID \
  --select_by "$SELECT" --regmean_nondiag_scales 0.9 1.0 \
  --regmean_eps 1e-3 --regmean_bs 16 \
  --apr_lrs $APR --nogate_lrs $NG --control_gd_lrs $GD \
  --skip_families ta ties dare dareties della bc ls ada \
  --refine_from regmean --out "$OUT"
