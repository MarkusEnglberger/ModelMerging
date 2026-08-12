#!/bin/bash
#SBATCH --job-name=gdmx20-aprhi
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gdmx20_aprhi-%j.out
#SBATCH --error=logs/gdmx20_aprhi-%j.out

# Bracket the TWO APR lr curves that peaked at the HIGH EDGE (lr=8) of the
# {2,4,8} grid in gd_matrix_clip20_s5: from=ta (l0.08) and from=bc
# (d0.1,o0.01,l0.2). Both rises were tiny (+.003/+.001 from lr4), so this
# likely just confirms lr8 as the peak, but the paper table shouldn't carry
# an edge asterisk. lr {12,16} completes the bracket; every other init's APR
# curve is already interior (ties/dareties/ada lr4, regmean lr0.5, calib lr4).
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0.08 \
  --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
  --skip_families ties dare dareties ada \
  --refine_from ta bc \
  --apr_lrs 12 16 \
  --steps 5 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_aprhi.json "$@"
