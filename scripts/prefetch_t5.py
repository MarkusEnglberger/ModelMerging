#!/usr/bin/env python
"""Prefetch flan-T5 GLUE assets into the local HF cache (run on the login node).

Downloads theta_0 (google/flan-t5-base) + the 8 FusionBench full-FT experts.
GLUE datasets are already cached by the RoBERTa track; re-requested here anyway
(no-op if present). Compute nodes then run with HF_HUB_OFFLINE=1.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))

from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

from apr.t5_gen import T5_EXPERT_CKPT, GLUE_T5_TEMPLATES

BASE = "google/flan-t5-base"

print(f"[prefetch] HF_HOME={os.environ['HF_HOME']}")
print(f"[prefetch] base {BASE}")
AutoTokenizer.from_pretrained(BASE)
T5ForConditionalGeneration.from_pretrained(BASE)

for task, ckpt in T5_EXPERT_CKPT.items():
    print(f"[prefetch] expert {task} <- {ckpt}")
    T5ForConditionalGeneration.from_pretrained(ckpt)

for task, d in GLUE_T5_TEMPLATES.items():
    print(f"[prefetch] dataset glue/{d['glue_config']}")
    load_dataset("glue", d["glue_config"])

print("[prefetch] done. Compute jobs can now run offline.")
