#!/bin/bash
#SBATCH --job-name=apr-smoke
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.out

source slurm/common.sh

# Fine-tune the two DistilRoBERTa experts from the shared base, then merge.
# (Requires internet for dataset access if not pre-cached; assets are pre-fetched.)
if [ ! -f experts/distilroberta-mrpc/config.json ]; then
  python scripts/train_expert.py --base distilroberta-base --task mrpc \
    --out experts/distilroberta-mrpc --epochs 3
fi
if [ ! -f experts/distilroberta-stsb/config.json ]; then
  python scripts/train_expert.py --base distilroberta-base --task stsb \
    --out experts/distilroberta-stsb --epochs 3
fi

python scripts/run_merge.py --config configs/smoke.yaml "$@"
