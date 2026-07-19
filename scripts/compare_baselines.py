#!/usr/bin/env python
"""Compare the proposed APR method against ordinary replay gradient descent.

All methods start from the SAME task-arithmetic merge point, use the SAME replay
buffers, and the SAME number of gradient evaluations (S sweeps x T tasks). Only
the per-step update rule differs:

  apr           : -g*|v|, AP-gated, clipped to +/-gamma|v|   (proposed; expert-anchored)
  nogate_dist   : -g*|v|, NO gate,  clipped to +/-gamma|v|   (anchored, ungated)
  inverted_gate : -g*|v|, inverted gate                       (sanity control)
  ordinary_gd   : -g (plain SGD), no gate, no clip            (NO expert anchoring)

Ordinary GD's raw-gradient steps live on a totally different scale than the
distance-scaled APR steps, so we sweep its learning rate and keep the best, to
give the baseline a fair shot (proposal: "same hyperparameter search budget").
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


def build_methods(steps, gd_lrs, apr_lrs, nogate_lrs, gammas, clip_mode="vdist_pre",
                  skip_inverted=False):
    """APR and no-gate get their OWN lr grids (their optima differ a lot: APR likes
    high lr, no-gate diverges at high lr). inverted_gate is a fixed sanity control
    (skip with skip_inverted). clip_mode default vdist_pre = clip AFTER lr."""
    methods = {}
    for g in gammas:
        for lr in apr_lrs:
            methods[f"apr@lr{lr:g},g{g:g}"] = RefineConfig(
                steps=steps, lr=lr, clip_frac=g, gate_mode="coordinate",
                update_mode="gated_grad", clip_mode=clip_mode)
        for lr in nogate_lrs:
            methods[f"nogate@lr{lr:g},g{g:g}"] = RefineConfig(
                steps=steps, lr=lr, clip_frac=g, gate_mode="none",
                update_mode="gated_grad", clip_mode=clip_mode)
    if not skip_inverted and apr_lrs:
        methods["inverted_gate"] = RefineConfig(steps=steps, lr=apr_lrs[len(apr_lrs)//2],
                                                clip_frac=gammas[0], gate_mode="inverted",
                                                update_mode="gated_grad", clip_mode=clip_mode)
    for lr in gd_lrs:
        methods[f"ordinary_gd@{lr:g}"] = RefineConfig(
            steps=steps, lr=lr, gate_mode="none", update_mode="grad", clip_mode="none")
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--gd_lrs", type=float, nargs="*",
                    default=[1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    ap.add_argument("--apr_lrs", type=float, nargs="*", default=[16, 32, 64, 128, 256])
    ap.add_argument("--nogate_lrs", type=float, nargs="*", default=[2, 4, 8, 16, 32])
    ap.add_argument("--gammas", type=float, nargs="*", default=[1.0])
    ap.add_argument("--clip_mode", default="vdist_pre")
    ap.add_argument("--skip_inverted", action="store_true",
                    help="omit the inverted-gate sanity control")
    ap.add_argument("--n_probe", type=int, default=None, help="override replay buffer size")
    ap.add_argument("--out", default="results/compare/poc3_compare.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.n_probe is not None:
        cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)
    # the task-vector copies are only needed to build the merge (done inside build);
    # refinement anchors on expert_encoder, not tau. At 3B/fp32 these are ~34 GB of
    # dead weight in this script's CPU footprint, so drop them (keeps the node-RAM
    # request at the 1-GPU billing floor). Harmless if already empty.
    ctx.task_vectors = {}

    methods = build_methods(args.steps, args.gd_lrs, args.apr_lrs, args.nogate_lrs,
                            args.gammas, clip_mode=args.clip_mode,
                            skip_inverted=args.skip_inverted)
    report = {"config": cfg.to_dict(), "steps": args.steps,
              "base": ctx.base_scores, "expert": ctx.expert_scores,
              "tasks": cfg.task_names, "methods": {}}

    # reference: the unrefined merge (S=0)
    merge_normret = ctx.normret(ctx.merge_scores)
    report["methods"]["merge(S=0)"] = {
        "scores": ctx.merge_scores, "normret": merge_normret,
        "aggregate": aggregate_retention(merge_normret), "displacement": 0.0,
    }

    for name, rc in methods.items():
        _log(f"\n===== method: {name} ({rc.update_mode}/{rc.gate_mode}/clip={rc.clip_mode} lr={rc.lr}) =====")
        refined, history = ctx.run_refine(rc, seed=cfg.seed)
        scores = ctx.eval_encoder(refined)
        nr = ctx.normret(scores)
        disp = pd_global_norm(pd_sub(refined, ctx.merged0))
        ag = aggregate_retention(nr)
        clip_g = (sum(h.get("clip_frac_gated", 0.0) for h in history) / len(history)
                  if history else 0.0)
        report["methods"][name] = {
            "scores": scores, "normret": nr,
            "aggregate": ag, "displacement": disp,
            "gate_density": (sum(h["gate_density"] for h in history) / len(history)
                             if history else None),
            "clip_frac_gated": clip_g,
        }
        # incremental per-method result so progress is visible mid-run
        _log(f"  -> {name}: mean={ag['mean_normret']:.3f} worst={ag['worst_normret']:.3f} "
             f"disp={disp:.3f} clip%gated={100*clip_g:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # ---- pretty table ----
    tasks = cfg.task_names
    print("\n" + "=" * 92)
    hdr = f"{'method':<20} " + " ".join(f"{t:>8}" for t in tasks) + \
          f" | {'mean':>6} {'worst':>6} {'hmean':>6} {'||disp||':>8}"
    print(hdr)
    print("-" * 92)
    # merge reference first, then all methods by mean normret (best at top)
    def sortkey(item):
        n, d = item
        return (0 if n == "merge(S=0)" else 1, -d["aggregate"]["mean_normret"])
    for n, d in sorted(report["methods"].items(), key=sortkey):
        ag = d["aggregate"]
        row = f"{n:<20} " + " ".join(f"{d['normret'][t]:>8.3f}" for t in tasks)
        row += f" | {ag['mean_normret']:>6.3f} {ag['worst_normret']:>6.3f} " \
               f"{ag['hmean_normret']:>6.3f} {d['displacement']:>8.4f}"
        print(row)
    print("=" * 92)
    print("(normret: 0=pretrained floor, 1=expert ceiling; higher is better)")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
