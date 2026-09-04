"""Prototype + benchmark: vLLM as the generation backend for the causal-LM
track, with in-process weight sync from the HF model the pipeline mutates.

For each task it (1) scores the HF expert with the pipeline's own
evaluate_generation (KV cache on, batch 48), (2) builds a vLLM engine from the
base checkpoint, pushes the expert's weights into it with load_weights(), and
scores the same rows through vLLM, and (3) reports wall time, throughput,
score agreement, and a sync-correctness check (vLLM must NOT reproduce the
expert before the sync and SHOULD after).

Run inside the scratch venv_vllm (torch 2.4.0 / vllm 0.6.3); the production
venv is untouched.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def hf_load(ckpt, tok, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype)
    if m.get_input_embeddings().weight.shape[0] != len(tok):
        m.resize_token_embeddings(len(tok))  # same as pipeline.load_expert
    m.config.use_cache = True
    m.generation_config.use_cache = True
    return m.to("cuda").eval()


def vllm_model(llm):
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


def vllm_sync(llm, hf_model):
    """Push the HF model's current weights into the vLLM engine (HF names;
    vLLM's Llama loader fuses q/k/v and gate/up itself)."""
    torch.cuda.synchronize()
    t0 = time.time()
    vllm_model(llm).load_weights(
        (n, p.detach()) for n, p in hf_model.state_dict().items())
    torch.cuda.synchronize()
    return time.time() - t0


def vllm_generate(llm, prompts, max_new):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs], sum(len(o.outputs[0].token_ids)
                                                   for o in outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--expert", default="MergeBench/Llama-3.2-3B_instruction")
    ap.add_argument("--tasks", nargs="+", default=["instruction", "math"])
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--hf_batch", type=int, default=48)
    ap.add_argument("--util", type=float, default=0.45,
                    help="vLLM gpu_memory_utilization (fraction of the GPU incl. "
                         "what the HF model already holds)")
    ap.add_argument("--skip_hf", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from apr.causal_lm import get_causal_task, evaluate_generation

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    style = "chat" if getattr(tok, "chat_template", None) else "plain"
    print(f"[setup] base={args.base} expert={args.expert} prompt_style={style} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    specs, rows, prompts = {}, {}, {}
    for t in args.tasks:
        s = get_causal_task(t)
        s.tokenizer, s.prompt_style, s.gen_dtype = tok, style, None
        specs[t] = s
        rows[t] = s.build_eval(None, args.n_eval)
        prompts[t] = [s.make_prompt(r, style, tok) for r in rows[t]]
        print(f"[data] {t}: {len(rows[t])} rows, max_new_tokens={s.max_new_tokens}",
              flush=True)

    # ---- HF reference (the pipeline's own eval path, KV cache on) ------------
    hf = hf_load(args.expert, tok, torch.bfloat16)
    hf_scores, hf_times = {}, {}
    if not args.skip_hf:
        for t in args.tasks:
            torch.cuda.synchronize(); t0 = time.time()
            hf_scores[t] = evaluate_generation(hf, rows[t], specs[t], args.hf_batch,
                                               "cuda")
            torch.cuda.synchronize(); hf_times[t] = time.time() - t0
            print(f"[hf]   {t}: primary={hf_scores[t]['primary']:.4f}  "
                  f"{hf_times[t]:.0f}s  (batch {args.hf_batch})", flush=True)
    hf_mem = torch.cuda.memory_allocated() / 2**30
    torch.cuda.empty_cache()
    print(f"[mem] HF model resident: {hf_mem:.1f} GiB", flush=True)

    # ---- vLLM engine alongside the HF model ---------------------------------
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=args.base, tokenizer=args.base, dtype="bfloat16",
              gpu_memory_utilization=args.util, max_model_len=2048, seed=0,
              enable_prefix_caching=False)
    print(f"[vllm] engine up in {time.time() - t0:.0f}s; "
          f"torch allocated now {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
          f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f})", flush=True)

    # sync-correctness probe: 8 prompts before and after pushing expert weights
    probe = prompts[args.tasks[0]][:8]
    before, _ = vllm_generate(llm, probe, 64)
    dt_sync = vllm_sync(llm, hf)
    after, _ = vllm_generate(llm, probe, 64)
    tok.padding_side = "left"
    with torch.no_grad():
        b = tok(probe, return_tensors="pt", padding=True).to("cuda")
        o = hf.generate(**b, max_new_tokens=64, do_sample=False,
                        pad_token_id=tok.pad_token_id)
        ref = tok.batch_decode(o[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
    same_before = sum(x.strip() == y.strip() for x, y in zip(before, ref))
    same_after = sum(x.strip() == y.strip() for x, y in zip(after, ref))
    print(f"[sync] load_weights took {dt_sync:.2f}s; probe (8 prompts, 64 tok): "
          f"identical to HF expert BEFORE sync {same_before}/8, AFTER sync "
          f"{same_after}/8", flush=True)

    # ---- vLLM scoring on the same rows --------------------------------------
    for t in args.tasks:
        torch.cuda.synchronize(); t0 = time.time()
        texts, n_tok = vllm_generate(llm, prompts[t], specs[t].max_new_tokens)
        gen_t = time.time() - t0
        sc = specs[t].score_generations(rows[t], texts)
        tot = time.time() - t0
        line = (f"[vllm] {t}: primary={sc['primary']:.4f}  gen {gen_t:.0f}s "
                f"({n_tok / gen_t:.0f} tok/s), +scoring {tot - gen_t:.0f}s")
        if t in hf_scores:
            line += (f"  | HF primary={hf_scores[t]['primary']:.4f} in "
                     f"{hf_times[t]:.0f}s -> speedup x{hf_times[t] / tot:.1f}")
        print(line, flush=True)

    print(f"[mem] peak torch allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)


if __name__ == "__main__":
    main()
