#!/usr/bin/env python
"""Prefetch flan-T5 GLUE assets into the local HF cache (run on the login node).

Downloads theta_0 (google/flan-t5-base) + the 8 FusionBench full-FT experts.
GLUE datasets are already cached by the RoBERTa track; re-requested here anyway
(no-op if present). Compute nodes then run with HF_HUB_OFFLINE=1.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))
# This script is the online half of an offline-compute workflow.  Do not let an
# inherited setting from slurm/common.sh turn a repair attempt into another
# local-cache lookup.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

from apr.t5_gen import T5_EXPERT_CKPT, GLUE_T5_TEMPLATES

BASE = "google/flan-t5-base"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--base-only", action="store_true",
                help="repair/verify only google/flan-t5-base")
args = ap.parse_args()

print(f"[prefetch] HF_HOME={os.environ['HF_HOME']}")
print(f"[prefetch] base {BASE}")
AutoTokenizer.from_pretrained(BASE)
T5ForConditionalGeneration.from_pretrained(BASE)

if args.base_only:
    print("[prefetch] base model and tokenizer are complete")
    raise SystemExit(0)

for task, ckpt in T5_EXPERT_CKPT.items():
    print(f"[prefetch] expert {task} <- {ckpt}")
    T5ForConditionalGeneration.from_pretrained(ckpt)

for task, d in GLUE_T5_TEMPLATES.items():
    print(f"[prefetch] dataset glue/{d['glue_config']}")
    load_dataset("glue", d["glue_config"])

print("[prefetch] done. Compute jobs can now run offline.")
