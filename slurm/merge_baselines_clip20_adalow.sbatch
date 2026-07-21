#!/bin/bash
#SBATCH --job-name=apr-mrg20-adalow
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/merge_clip20_adalow-%j.out
#SBATCH --error=logs/merge_clip20_adalow-%j.out

# AdaMerging on the 20-task CLIP suite, restarted from a SANE initial lambda.
#
# Why: in job 6188401 AdaMerging used the default --ada_init_lam 0.3, which at
# 20 tasks is deep in the collapse region (TA@l0.3 = 0.364 mean acc, barely above
# chance on many tasks; zero-shot floor is 0.5503). ADA-layer still climbed out to
# 0.6542 mean acc -- the BEST init of that run -- and APR<-ada was the best overall
# (0.7185 @lr4), so the family is strong despite being handicapped. Starting it at
# 0.05 (where the TA lambda sweep was heading) should let the learned per-layer
# lambdas converge somewhere much better within the same 300 steps.
#
# Only the ada family runs (ta/ties already covered by 6188401 + the lowlam job);
# apr_lrs bracket the rescored ada optimum lr4. Selection/reporting by absolute
# mean accuracy -- NormRet is degenerate on this suite (stl10 gap .0043,
# food101 gap .0351). Requires: prefetch_clip.py --suite 20.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ada_variants task layer --ada_init_lam 0.05 \
  --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 --ada_workers 0 \
  --skip_families ta ties dare dareties bc \
  --refine_from ada --apr_lrs 2 4 8 --steps 5 --select_by mean_acc \
  --out results/compare/merge_baselines_clip20_adalow.json "$@"
