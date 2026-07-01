#!/usr/bin/env python
"""Diagnose (a) expert geometry and (b) how much of each update is clipped.

Reports:
  - pairwise ||theta_i - theta_j|| between experts, and ||theta_i - merge|| (the
    initial |v| scale each task update is measured against), and ||tau_i||.
  - for several refinement configs: the fraction of (gated) coordinates whose step
    is pushed to the +/- gamma|v| trust-region boundary, the gate density, and the
    total displacement ||theta_refined - merge||.

This explains why a method's displacement is large or small: a low clip fraction
means updates sit in the damped -g|v| regime (tiny steps), while a high clip
fraction means the trust region is actually binding.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.pipeline import MergeContext, _log
from apr.models import pd_sub, pd_global_norm, pd_to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default="results/compare/clip_diag.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)
    tasks = cfg.task_names
    exp = {h.name: h.expert_encoder for h in ctx.handles}

    # ---- expert geometry ----
    geom = {"pairwise_expert": {}, "expert_to_merge": {}, "tau_norm": {},
            "expert_to_base": {}}
    for i in tasks:
        geom["expert_to_merge"][i] = pd_global_norm(pd_sub(exp[i], ctx.merged0))
        geom["expert_to_base"][i] = pd_global_norm(pd_sub(exp[i], ctx.base_encoder))
        geom["tau_norm"][i] = geom["expert_to_base"][i]
    for a in range(len(tasks)):
        for b in range(a + 1, len(tasks)):
            i, j = tasks[a], tasks[b]
            geom["pairwise_expert"][f"{i}-{j}"] = pd_global_norm(pd_sub(exp[i], exp[j]))

    print("\n=== expert geometry (L2 distances over encoder params) ===")
    print("pairwise expert-expert:")
    for k, val in geom["pairwise_expert"].items():
        print(f"  ||theta_{k.split('-')[0]} - theta_{k.split('-')[1]}|| = {val:.3f}")
    print("expert-to-merge (= ||v_i|| at the merge):")
    for i in tasks:
        print(f"  {i}: {geom['expert_to_merge'][i]:.3f}   (||tau_{i}||={geom['tau_norm'][i]:.3f})")

    # ---- clip fraction per config ----
    configs = {
        "apr@lr1 (paperclip)":  RefineConfig(steps=args.steps, lr=1, clip_frac=0.5,
                                             gate_mode="coordinate", update_mode="gated_grad",
                                             clip_mode="vdist"),
        "apr@lr8 (paperclip)":  RefineConfig(steps=args.steps, lr=8, clip_frac=0.5,
                                             gate_mode="coordinate", update_mode="gated_grad",
                                             clip_mode="vdist"),
        "apr_sat@lr16":         RefineConfig(steps=args.steps, lr=16, clip_frac=0.5,
                                             gate_mode="coordinate", update_mode="gated_grad",
                                             clip_mode="vdist_pre"),
        "nogate@lr4 (paperclip)": RefineConfig(steps=args.steps, lr=4, clip_frac=0.5,
                                               gate_mode="none", update_mode="gated_grad",
                                               clip_mode="vdist"),
        "ordinary_gd@1e-3":     RefineConfig(steps=args.steps, lr=1e-3, gate_mode="none",
                                             update_mode="grad", clip_mode="none"),
    }
    report = {"config": cfg.to_dict(), "geometry": geom, "methods": {}}
    print("\n=== clip fraction & displacement per config ===")
    print(f"{'config':<24} {'gate%':>6} {'clip%gated':>10} {'clip%all':>9} {'disp':>7}")
    for name, rc in configs.items():
        refined, hist = ctx.run_refine(rc, seed=cfg.seed)
        disp = pd_global_norm(pd_sub(refined, ctx.merged0))
        gd = sum(h["gate_density"] for h in hist) / len(hist)
        cg = sum(h["clip_frac_gated"] for h in hist) / len(hist)
        ca = sum(h["clip_frac_all"] for h in hist) / len(hist)
        report["methods"][name] = {"gate_density": gd, "clip_frac_gated": cg,
                                   "clip_frac_all": ca, "displacement": disp,
                                   "history": hist}
        print(f"{name:<24} {100*gd:>5.1f}% {100*cg:>9.1f}% {100*ca:>8.1f}% {disp:>7.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
