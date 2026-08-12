#!/bin/bash
#SBATCH --job-name=gateprec-ho
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/gateprec_ho_n%x-%j.out
#SBATCH --error=logs/gateprec_ho_n%x-%j.out

# GATE-PRECISION, HELD-OUT VERSION. Submit with
#   sbatch --export=ALL,NPROBE=64 slurm/gate_precision_heldout.sbatch   (and 8)
#
# The first version of this diagnostic (gate_precision_clip20, job 6236595)
# measured the loss change on the SAME buffer that produced the gradient, so
# meas/pred only tested linearity -- a coordinate with g*v<0 reduces that
# buffer's loss by construction, whether or not its sign is a sampling
# artifact. It therefore could not test the noise hypothesis at all.
#
# Here the gradient still comes from the n_probe train buffer, but the loss
# change is ALSO measured on a DISJOINT held-out buffer of the same size
# (verified non-overlapping). The readout is heldout/train per bucket:
#   ~1  -> the bucket's coordinates carry signal that generalizes
#   ~0  -> they only fit the replay buffer = the noise floor a threshold targets
# If the low-|g| buckets show a large gap and the top-|g| buckets do not, the
# |g|-threshold gate (topk_g, implemented) is vindicated and its sweep
# (gate_topkg_clip20.sbatch) should run.
#
# n=64 matches the main experiments; n=8 is the worst-case budget where
# sampling noise should be most visible.
set -u
N=${NPROBE:?set NPROBE (8|64) via --export=ALL,NPROBE=...}
source slurm/common.sh
echo "[gateprec] n_probe=${N} n_val=${N} (disjoint)"
python scripts/gate_precision.py --config configs/clip20_l008.yaml \
  --n_probe "${N}" --n_val "${N}" --alphas 0.05 0.2 --stats gv g gnorm \
  --out "results/compare/gate_precision_heldout_n${N}.json"
