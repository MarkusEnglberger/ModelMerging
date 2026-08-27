#!/bin/bash
#SBATCH --job-name=verify-residency
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/verify_residency-%j.out
#SBATCH --error=logs/verify_residency-%j.out
#
# Gate for adopting the resident refine() path mid-campaign. Runs the unit
# suite (which includes the CUDA bit-identity case, skipped on login nodes)
# and then the real-config check on both encoder suites: same experts, same
# buffers, same seed, shuttled vs resident, compared with torch.equal.
#
# Nothing is resubmitted until this reports PASS on both suites.

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== unit suite (includes the CUDA equivalence test) ====="
PYTHONPATH=src python -m pytest tests/ -q

for c in clip8 glue8; do
  echo ""
  echo "===== real-config verification: ${c} ====="
  python scripts/verify_residency.py --config "configs/${c}.yaml" \
    --steps 3 --budget 16 --seed 100 \
    --out "results/compare/residency_${c}.json"
done

echo ""
echo "[done] residency verified on both encoder suites"
