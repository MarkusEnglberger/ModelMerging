"""Micro-benchmark: HF generate() throughput on a MergeBench expert with the
checkpoint's default use_cache (False for MergeBench/*) vs use_cache=True.

Usage: python scripts/bench_gen_cache.py [--ckpt MergeBench/Llama-3.2-3B_instruction]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def bench(model, tok, prompts, max_new, use_cache, device):
    tok.padding_side = "left"
    batch = tok(prompts, return_tensors="pt", padding=True).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    kw = {} if use_cache is None else {"use_cache": use_cache}
    out = model.generate(**batch, max_new_tokens=max_new, min_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.pad_token_id, **kw)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_new = (out.shape[1] - batch["input_ids"].shape[1]) * out.shape[0]
    return dt, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="MergeBench/Llama-3.2-3B_instruction")
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()
    device = "cuda"  # checkpoints resolve via $HF_HOME (set by slurm/common.sh)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt,
                                                 torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"ckpt={args.ckpt}  config.use_cache={model.config.use_cache}  "
          f"generation_config.use_cache={model.generation_config.use_cache}  "
          f"attn={model.config._attn_implementation}")
    print(f"gpu={torch.cuda.get_device_name(0)}")

    prompts = [f"Write a detailed, well-structured essay of at least 400 words about "
               f"topic number {i}: the history and future of renewable energy in "
               f"coastal regions. Include an introduction and conclusion."
               for i in range(args.batch)]

    # warm-up (short)
    bench(model, tok, prompts[:4], 16, True, device)

    for label, uc in [("checkpoint default", None), ("use_cache=True", True),
                      ("use_cache=False", False)]:
        dt, n_new = bench(model, tok, prompts, args.max_new, uc, device)
        print(f"{label:20s}: batch={args.batch} new_tokens={args.max_new}  "
              f"{dt:7.1f}s  {n_new / dt:8.1f} tok/s  "
              f"{dt / args.max_new * 1000:6.1f} ms/step  "
              f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
