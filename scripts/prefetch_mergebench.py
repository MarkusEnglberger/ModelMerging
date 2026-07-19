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
ap.add_argument("--domains", nargs="*",
                default=["math", "coding", "instruction", "multilingual"])
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

print("[prefetch] eval datasets: gsm8k, evalplus/mbppplus, google/IFEval")
load_dataset("gsm8k", "main")
load_dataset("evalplus/mbppplus")
load_dataset("google/IFEval")

print("[prefetch] replay datasets: MergeBench per-domain val sets")
for d in args.domains:
    load_dataset(f"MergeBench/{d}_val")

if "multilingual" in args.domains:
    from apr.causal_lm import MULTILINGUAL_LANGS, _MC_BENCHES
    for hf_name, bench in _MC_BENCHES:
        for lang in MULTILINGUAL_LANGS:
            print(f"[prefetch] {bench}/{lang}")
            try:
                load_dataset(hf_name, lang, split="test")
            except Exception as e:
                print(f"[prefetch]   skip {bench}/{lang}: {type(e).__name__}")

print("[prefetch] done. Compute jobs can now run offline.")
