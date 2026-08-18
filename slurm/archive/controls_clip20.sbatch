#!/bin/bash
#SBATCH --job-name=apr-ctl-clip20
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/controls_clip20-%j.out
#SBATCH --error=logs/controls_clip20-%j.out

# Refinement CONTROLS at 20 tasks: gated APR vs nogate (ungated anchor) vs
# ordinary GD vs inverted gate, all from the SAME TA merge. This is the
# gate-ablation evidence the cancelled fixed-lam job (6188808) was meant to
# provide -- rerun from the TUNED lam=0.08 config (configs/clip20_l008.yaml;
# lam 0.3 merge is catastrophic at 20 tasks, mAcc .364 < zero-shot .550).
# lr grids centred on the 20-task optima: APR peaked at lr4 from TA; nogate
# peaked lr2 on clip8 and optima fall with task count. Rescore post hoc with
# scripts/rescore.py (compare reports store per-method scores).
# Requires: prefetch_clip.py --suite 20.
source slurm/common.sh
python scripts/compare_baselines.py --config configs/clip20_l008.yaml \
  --steps 5 --clip_mode vdist \
  --apr_lrs 1 2 4 8 \
  --nogate_lrs 0.5 1 2 4 \
  --gd_lrs 5e-5 1e-4 5e-4 \
  --out results/compare/controls_clip20.json "$@"
