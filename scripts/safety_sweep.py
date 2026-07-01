#!/usr/bin/env python
"""Test the 'anchoring makes large steps safe' hypothesis on real models.

Three families, swept over a WIDE learning-rate range, all from the same merge
point / replay buffers:

  ordinary_gd     : -g, no anchor, no clip          -> expected to diverge early
  apr_paperclip   : -g|v|, gated, clip THEN *lr      -> move ~ lr*gamma*|v|, can overshoot
  apr_sat         : -g|v|, gated, clip AFTER *lr     -> move <= gamma*|v| for ANY lr (safe)
  nogate_sat      : -g|v|, ungated, clip AFTER *lr   -> same safety, isolates the gate

We report mean/worst normalized retention and ||displacement|| for each lr, so we
can see (a) where each method collapses and (b) that the saturating trust region
never diverges no matter how large lr is.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_retention
from apr.models import pd_sub, pd_global_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default="results/compare/safety_sweep.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)
    g = args.gamma
    S = args.steps

    methods = {}
    for lr in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1]:
        methods[f"ordinary_gd@{lr:g}"] = RefineConfig(
            steps=S, lr=lr, gate_mode="none", update_mode="grad", clip_mode="none")
    for lr in [1, 2, 4, 8, 12, 16, 32]:
        methods[f"apr_paperclip@{lr:g}"] = RefineConfig(
            steps=S, lr=lr, clip_frac=g, gate_mode="coordinate",
            update_mode="gated_grad", clip_mode="vdist")
    for lr in [1, 2, 4, 8, 16, 64, 256, 1024]:
        methods[f"apr_sat@{lr:g}"] = RefineConfig(
            steps=S, lr=lr, clip_frac=g, gate_mode="coordinate",
            update_mode="gated_grad", clip_mode="vdist_pre")
        methods[f"nogate_sat@{lr:g}"] = RefineConfig(
            steps=S, lr=lr, clip_frac=g, gate_mode="none",
            update_mode="gated_grad", clip_mode="vdist_pre")

    report = {"config": cfg.to_dict(), "gamma": g, "steps": S,
              "tasks": cfg.task_names, "methods": {}}
    mref = ctx.normret(ctx.merge_scores)
    report["methods"]["merge(S=0)"] = {
        "normret": mref, "aggregate": aggregate_retention(mref), "displacement": 0.0}

    for name, rc in methods.items():
        _log(f"\n===== {name} (clip={rc.clip_mode} lr={rc.lr}) =====")
        refined, _ = ctx.run_refine(rc, seed=cfg.seed)
        nr = ctx.normret(ctx.eval_encoder(refined))
        disp = pd_global_norm(pd_sub(refined, ctx.merged0))
        report["methods"][name] = {"normret": nr,
                                   "aggregate": aggregate_retention(nr),
                                   "displacement": disp}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'method':<22} {'mean':>7} {'worst':>7} {'||disp||':>9}")
    print("-" * 70)
    def fam(n): return n.split("@")[0]
    def lrof(n):
        try: return float(n.split("@")[1])
        except Exception: return -1
    for n in sorted(report["methods"], key=lambda n: (fam(n), lrof(n))):
        d = report["methods"][n]
        ag = d["aggregate"]
        print(f"{n:<22} {ag['mean_normret']:>7.3f} {ag['worst_normret']:>7.3f} "
              f"{d['displacement']:>9.3f}")
    print("=" * 70)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
