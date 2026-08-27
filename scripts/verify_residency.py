#!/usr/bin/env python
"""Prove the resident path is bit-identical to the shuttled one, and time both.

The synthetic test in tests/test_residency_equivalence.py pins the semantics;
this runs the real thing -- actual experts, actual replay buffers, actual
gradients -- because that is what the campaign's results depend on. The whole
point of the change is that runs made before and after it can share a table, so
"identical" has to mean torch.equal on every tensor, not "close".

Builds the context twice, once with cfg.data.resident forced False and once
True, refines the same number of sweeps from the pretrained model with the same
seed, and compares. Reports per-step wall time and peak device memory for each.

  python scripts/verify_residency.py --config configs/clip8.yaml --steps 3
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig                       # noqa: E402
from apr.pipeline import MergeContext, _log                   # noqa: E402
import dataclasses                                            # noqa: E402


def run(config_path, resident, steps, lr, budget, seed):
    cfg = ExperimentConfig.from_yaml(config_path)
    cfg.data.n_probe = budget
    cfg.data.resident = resident
    ctx = MergeContext.build(cfg)
    rc = dataclasses.replace(cfg.refine, steps=steps, lr=lr,
                             lr_schedule="constant", order="random",
                             gate_mode="coordinate", clip_mode="vdist")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    state, hist = ctx.run_refine_from(ctx.base_encoder, rc, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else 0.0)
    n_steps = len(hist)
    _log(f"[{'resident' if resident else 'shuttled'}] {wall:.1f}s for {n_steps} "
         f"task-steps = {1000*wall/max(n_steps,1):.0f} ms/step, "
         f"peak {peak:.1f} GiB, keep_on_device={ctx.keep_model_on_device}")
    return state, hist, {"wall_s": wall, "task_steps": n_steps,
                         "ms_per_step": 1000 * wall / max(n_steps, 1),
                         "peak_gib": peak,
                         "keep_model_on_device": ctx.keep_model_on_device,
                         "residency": ctx.residency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8.0)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shut, h_shut, m_shut = run(args.config, False, args.steps, args.lr,
                               args.budget, args.seed)
    res, h_res, m_res = run(args.config, True, args.steps, args.lr,
                            args.budget, args.seed)

    bad = [k for k in shut if not torch.equal(shut[k].cpu(), res[k].cpu())]
    same_hist = (len(h_shut) == len(h_res) and all(
        a["task"] == b["task"] and a["sweep"] == b["sweep"]
        and a["gate_density"] == b["gate_density"]
        and a["clipped_frac_gated"] == b["clipped_frac_gated"]
        and a["update_norm"] == b["update_norm"]
        for a, b in zip(h_shut, h_res)))

    ok = not bad and same_hist
    speedup = m_shut["ms_per_step"] / max(m_res["ms_per_step"], 1e-9)
    _log("")
    _log(f"[verify] tensors identical : {'YES' if not bad else 'NO -> ' + str(bad[:5])}")
    _log(f"[verify] history identical : {'YES' if same_hist else 'NO'}")
    _log(f"[verify] speedup           : {speedup:.2f}x "
         f"({m_shut['ms_per_step']:.0f} -> {m_res['ms_per_step']:.0f} ms/step)")
    _log(f"[verify] peak memory       : {m_shut['peak_gib']:.1f} -> "
         f"{m_res['peak_gib']:.1f} GiB")
    _log(f"[verify] RESULT            : {'PASS' if ok else 'FAIL'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": args.config, "identical": ok,
                       "mismatched_tensors": bad, "history_identical": same_hist,
                       "speedup": speedup, "shuttled": m_shut,
                       "resident": m_res}, f, indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
