#!/bin/bash
#SBATCH --job-name=topkg-c20
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/topkg_clip20-%j.out
#SBATCH --error=logs/topkg_clip20-%j.out

# |g|-THRESHOLD GATE SWEEP (contingent on gate_precision_clip20 showing
# precision concentrated in high-|g| buckets -- DO NOT submit before reading
# that diagnostic). topk_g: sign(g*v)<0 prior AND |g| in the per-tensor top
# `frac`; thresholds only the noisy factor, so the small-|g| x large-|v|
# noise-amplification coordinates the eps=0 gate admits are excluded.
# In-run baseline: the standard coordinate gate on the same inits/buffers.
# lr grid shifted UP vs the coordinate gate's 2-4 optimum: a sparse gate keeps
# fewer coordinates, so the per-coordinate step must be larger for the same
# total movement. From the two inits where the gate question is sharpest:
# TIES (APR~ungated tie) and ADA (best data-free init).
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ties_densities 0.1 --ties_lams 0.6 \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families ta dare dareties bc \
  --refine_from ties ada \
  --apr_gate_modes coordinate topk_g --topk_fracs 0.01 0.05 0.2 \
  --apr_lrs 2 4 8 16 \
  --steps 5 --select_by mean_acc \
  --out results/compare/gate_topkg_clip20.json "$@"
