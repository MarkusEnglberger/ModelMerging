#!/bin/bash
#SBATCH --job-name=apr-ls-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ls_glue8-%j.out
#SBATCH --error=logs/ls_glue8-%j.out

# Localize-and-Stitch done properly on GLUE-8: sweep the L1 penalty gamma (the collapsed
# one was 1e-4) and the stitch lambda (was hardcoded 1.0; GLUE lambda is ~0.3), select on the
# replay objective, and refine APR on top of the best learned-L&S init. Fixes the
# mis-tuned learned-mask cell omitted from the matched-budget table.
source slurm/common.sh
python scripts/ls_sweep.py --config configs/glue8.yaml \
  --n_probe 64 --ls_gammas 0 1e-7 1e-6 1e-5 --ls_lams 0.1 0.2 0.3 0.5 1.0 \
  --ls_sparsities 0.01 0.05 0.1 --ls_steps 300 --ls_lr 0.1 --ls_bs 16 \
  --apr_lrs 4 8 16 --steps 5 \
  --out results/compare/ls_sweep_glue8.json "$@"
