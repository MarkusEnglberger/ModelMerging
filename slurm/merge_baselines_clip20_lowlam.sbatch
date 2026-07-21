#!/bin/bash
#SBATCH --job-name=apr-mrg20-low
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/merge_clip20_lowlam-%j.out
#SBATCH --error=logs/merge_clip20_lowlam-%j.out

# CORRECTED lambda sweep for the 20-task CLIP suite. Two fixes over job 6188401:
#
# 1. LAMBDA GRID EXTENDED DOWN. The first run was monotone decreasing across its
#    whole grid -- TA l0.05 -> +0.108 (BEST, at the LOW EDGE), l0.08 -> -0.100,
#    l0.1 -> -0.311, l0.15 -> -1.163, l0.2 -> -2.289, l0.3 -> -4.980 -- so the
#    optimum was never bracketed. theta = theta_0 + lam * sum_i tau_i over T=20
#    vectors, so the perturbation grows with T and lambda must fall roughly like
#    1/T: the 8-task optimum ~0.2-0.3 implies ~0.05 here, and evidently lower.
#    TIES lambda likewise re-bracketed downward (its 0.8/1.0 cells all lost).
#
# 2. SELECTION/REPORTING BY ABSOLUTE MEAN ACCURACY (--select_by mean_acc).
#    NormRet = (merge-base)/(expert-base) is degenerate on this suite: stl10
#    (base .9712 -> expert .9755, gap .0043) and food101 (.8271 -> .8622) have
#    near-zero denominators and alone dragged the mean from -1.04 to -4.94.
#    Absolute accuracy is now primary; NormRet is still recorded, along with a
#    non-degenerate-task-only variant.
#
# AdaMerging is SKIPPED here: it learns its own lambda, so the grid fix does not
# apply to it, and job 6188401 already computes it (recomputable from that JSON).
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0.01 0.02 0.03 0.04 0.05 0.08 \
  --ties_densities 0.1 0.2 --ties_lams 0.2 0.4 0.6 0.8 \
  --skip_families dare dareties bc ada \
  --refine_from ta ties --apr_lrs 1 2 4 8 \
  --steps 5 --select_by mean_acc \
  --out results/compare/merge_baselines_clip20_lowlam.json "$@"
