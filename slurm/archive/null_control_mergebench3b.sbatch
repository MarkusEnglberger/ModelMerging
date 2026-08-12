#!/bin/bash
#SBATCH --job-name=apr-null-mb3b
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/null_mb3b-%j.out
#SBATCH --error=logs/null_mb3b-%j.out

# Null control for the MergeBench decoder track: measure how much the eval moves when the
# MODEL does not. Motivated by the compare run, where ordinary GD at lr=1e-5 (||disp||
# =0.0009, i.e. numerically the merge) still "lost" ~0.04 mean retention -- a model that
# did not move cannot lose retention, so that gap bounds the noise, not an effect.
# Three tests: [A] cross-run eval0 vs the previous JSON, [B] 3x eval of the identical
# merge in one process, [C] an ordinary-GD step at lr=1e-8 (||disp||~1e-7) then eval.
# Verdict line reports the noise floor; any method gap below it is unresolvable at
# max_eval=200. Run BEFORE spending more GPU on method comparisons for this track.
#
# --mem 180G = 25% of the node, which is the 1-GPU billing floor (192 SBU/hr); going
# higher makes memory (not the GPU) the binding fraction and raises the rate.
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/null_control.py --config configs/mergebench_llama3b.yaml \
  --repeats 3 --null_lr 1e-8 --steps 5 \
  --reference results/compare/mergebench3b_compare.json \
  --out results/compare/null_control_mergebench3b.json "$@"
