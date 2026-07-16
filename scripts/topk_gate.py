#!/usr/bin/env python
"""Top-k attribution gate vs. the sign gate vs. density-matched random masks.

If the attribution signal is concentrated in the heavy-|g*v| coordinates (see
gate_precision.py), a gate that keeps only the top-k fraction by |g*v| among the
loss-decreasing coordinates should outperform BOTH the eps=0 sign gate (which lets the
noise floor vote) and a random mask of the same density (which has no signal at all).
The decisive comparison is topk@f vs random@f at each density f: any gap between them
is pure attribution signal, with sparsity effects controlled.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_retention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=16)
    ap.add_argument("--topk_fracs", type=float, nargs="*", default=[0.01, 0.05, 0.1])
    ap.add_argument("--out", default="results/compare/topk_gate.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)

    arms = [("sign-gate", dict(gate_mode="coordinate"))]
    for f in args.topk_fracs:
        arms.append((f"topk@{f:g}", dict(gate_mode="topk", topk_frac=f)))
        arms.append((f"random@{f:g}", dict(gate_mode="random", random_gate_density=f)))
    acc = {name: {"means": [], "worsts": []} for name, _ in arms}

    for seed in args.seeds:
        ctx.resample_buffers(args.n_probe, probe_seed=seed)
        _log(f"\n########## seed {seed} ##########")
        for name, kw in arms:
            rc = dataclasses.replace(cfg.refine, steps=args.steps, lr=args.lr,
                                     order="fixed", lr_schedule="constant", **kw)
            st, _ = ctx.run_refine_from(ctx.merged0, rc, seed=seed)
            ag = aggregate_retention(ctx.normret(ctx.eval_encoder(st)))
            acc[name]["means"].append(ag["mean_normret"])
            acc[name]["worsts"].append(ag["worst_normret"])
            _log(f"[seed {seed}] {name:<14} mean={ag['mean_normret']:.3f} "
                 f"worst={ag['worst_normret']:.3f}")

    report = {"config": cfg.to_dict(), "grids": vars(args), "arms": {}}
    for name, d in acc.items():
        m, w = np.array(d["means"]), np.array(d["worsts"])
        report["arms"][name] = {
            "mean": float(m.mean()),
            "mean_std": float(m.std(ddof=1)) if len(m) > 1 else 0.0,
            "worst": float(w.mean()),
            "worst_std": float(w.std(ddof=1)) if len(w) > 1 else 0.0,
            "means": d["means"], "worsts": d["worsts"]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'arm':<16}{'mean +/- std':>18}{'worst +/- std':>20}")
    print("-" * 60)
    for name, d in sorted(report["arms"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{name:<16}{d['mean']:>8.3f} +/-{d['mean_std']:>6.3f}"
              f"   {d['worst']:>8.3f} +/-{d['worst_std']:>6.3f}")
    print("=" * 60)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
