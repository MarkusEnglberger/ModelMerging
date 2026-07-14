#!/usr/bin/env python
"""Prefetch MergeBench decoder-track assets (run on the login node, needs HF login
for the gated Llama base: `huggingface-cli login` after accepting the license).

Downloads theta_0 + the MergeBench domain experts + the replay/eval datasets for
the POC triple (math/coding/instruction). Pass --base to switch model family,
e.g. --base google/gemma-2-2b (also gated).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from apr.causal_lm import MERGEBENCH_EXPERT

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="meta-llama/Llama-3.2-3B")
ap.add_argument("--domains", nargs="*", default=["math", "coding", "instruction"])
args = ap.parse_args()

print(f"[prefetch] HF_HOME={os.environ['HF_HOME']}")
print(f"[prefetch] base {args.base}")
AutoTokenizer.from_pretrained(args.base)
AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="bfloat16")

short = args.base.split("/")[-1]
for d in args.domains:
    ckpt = MERGEBENCH_EXPERT.format(base=short, domain=d)
    print(f"[prefetch] expert {d} <- {ckpt}")
    AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="bfloat16")

print("[prefetch] datasets: gsm8k, mbpp, google/IFEval, yahma/alpaca-cleaned")
load_dataset("gsm8k", "main")
load_dataset("mbpp")
load_dataset("google/IFEval")
load_dataset("yahma/alpaca-cleaned")

print("[prefetch] done. Compute jobs can now run offline.")
