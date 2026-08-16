#!/bin/bash
# Recovery/edge-extension wave for the n=8+8 table.
#
# The four edge jobs use strict supersets of their original grids and write to
# *_edge.json, preserving the canonical results.  The two T5 retries write the
# canonical filenames that the failed jobs never created.
#
# Before the T5 wave, repair the partial base-model cache on a networked login
# node:
#   .venv/bin/python scripts/prefetch_t5.py --base-only
#
# Usage:
#   bash slurm/submit_table2_n8_recovery.sh edges
#   bash slurm/submit_table2_n8_recovery.sh t5
#   bash slurm/submit_table2_n8_recovery.sh all

set -euo pipefail
: "${PARTITION:=mcs.gpu.q}"

wave="${1:-all}"
if [[ "$wave" != edges && "$wave" != t5 && "$wave" != all ]]; then
  echo "usage: $0 [edges|t5|all]" >&2
  exit 2
fi

submit_edge () {
  local bench="$1" mode="$2" apr="$3" ng="$4" gd="$5"
  echo "[submit] edge extension: BENCH=$bench MODE=$mode"
  sbatch --partition="$PARTITION" \
    --export="ALL,BENCH=$bench,MODE=$mode,N=8,TAG=edge,STEPS_GRID=5 20 50 100,APR_OVR=$apr,NG_OVR=$ng,GD_OVR=$gd" \
    slurm/grid_nn.sbatch
}

if [[ "$wave" == edges || "$wave" == all ]]; then
  # GLUE/base: APR and ungated selected S=50; GD selected the maximum lr.
  submit_edge glue8 base \
    "0.25 0.5 1 2 4 8 16 32" \
    "0.125 0.25 0.5 1 2 4 8" \
    "1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2"

  # GLUE/merges: TIES APR selected S=50; several GD cells selected the
  # maximum lr, and TIES ungated selected the minimum lr.
  submit_edge glue8 merges \
    "0.5 1 2 4 8 16 32" \
    "0.25 0.5 1 2 4 8" \
    "1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2"

  # CLIP/base: APR selected S=50.  Lower rates are included for the longer
  # horizon so the lr*S tradeoff remains covered.
  submit_edge clip8 base \
    "0.5 1 2 4 8 16 32" \
    "0.5 1 2 4 8 16" \
    "1e-6 1e-5 1e-4 5e-4 1e-3 5e-3"

  # CLIP/merges: most selected cells touched S=50 and/or the low-lr edge.
  submit_edge clip8 merges \
    "0.5 1 2 4 8 16 32" \
    "0.5 1 2 4 8 16" \
    "1e-5 5e-5 1e-4 5e-4 1e-3 5e-3"
fi

if [[ "$wave" == t5 || "$wave" == all ]]; then
  # Loading with local_files_only is a stronger preflight than checking for a
  # particular filename (HF checkpoints may be sharded or use safetensors).
  if ! .venv/bin/python - <<'PY'
import os
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))
from transformers import AutoTokenizer, T5ForConditionalGeneration
AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base", local_files_only=True)
print("[preflight] FLAN-T5 base cache is complete")
PY
  then
    echo "[error] FLAN-T5 base weights are not fully cached." >&2
    echo "Run: .venv/bin/python scripts/prefetch_t5.py --base-only" >&2
    exit 1
  fi

  for mode in base merges; do
    echo "[submit] failed-job retry: BENCH=t5_glue8 MODE=$mode"
    sbatch --partition="$PARTITION" \
      --export="ALL,BENCH=t5_glue8,MODE=$mode,N=8" slurm/grid_nn.sbatch
  done
fi

echo "[submit] recovery wave submitted: $wave"
