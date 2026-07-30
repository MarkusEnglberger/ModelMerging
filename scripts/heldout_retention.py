#!/usr/bin/env python
"""Held-out-task retention: does staying close to the base model preserve its
zero-shot abilities on tasks that were never merged or refined?

Motivation. Refining directly from theta_0 reaches merge-level multitask
accuracy at a small fraction of the merges' parameter displacement (base-APR
0.3-0.7 vs TA 1.04, AdaMerging 1.68, TIES 2.95 from theta_0 on the 20-task
suite). The paper currently sells proximity as a *proxy* for preserved
zero-shot behavior on unseen tasks. This script measures that directly:

  * split the suite into TRAIN tasks (merged / refined on) and HELD-OUT tasks
    (never seen by any merge or refinement step);
  * build each model point from the TRAIN tasks only -- TA merge (lambda
    swept), TIES merge, APR from the base model, APR from the TA merge;
  * evaluate ALL tasks: train-task accuracy = multitask quality, held-out
    accuracy vs the base model's own zero-shot = retention;
  * record ||state - theta_0|| so retention can be plotted against proximity.

The held-out set should be tasks the base model is actually good at
(there is nothing to retain on a ~chance zero-shot task like KMNIST), so the
default holds out the natural-image tasks with the highest base zero-shot and
trains on the specialized/domain-shift tasks.

The context is built on the FULL suite config (all experts + eval sets), and
the train-task restriction is applied to the task vectors and refinement
handles; the replay buffers of held-out tasks are simply never used.
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import OrderedDict

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import pd_sub, pd_global_norm
from apr.taskvec import task_arithmetic_merge
from apr.merge_methods import ties_combined_tau, ties_merge
from apr.refine import refine

DEFAULT_HELDOUT = ["sun397", "stl10", "cifar10", "pets", "food101", "flowers102"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--heldout", nargs="*", default=DEFAULT_HELDOUT)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--probe_seed", type=int, default=None)
    ap.add_argument("--ta_lams", type=float, nargs="*", default=[0.08, 0.1, 0.15, 0.2])
    ap.add_argument("--ties_densities", type=float, nargs="*", default=[0.1])
    ap.add_argument("--ties_lams", type=float, nargs="*", default=[0.6, 0.8])
    ap.add_argument("--apr_base_lrs", type=float, nargs="*", default=[4, 8])
    ap.add_argument("--apr_ta_lrs", type=float, nargs="*", default=[4])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--apr_schedule", default="cosine",
                    choices=["constant", "cosine", "linear"])
    ap.add_argument("--apr_order", default="random",
                    choices=["fixed", "cyclic", "random"])
    ap.add_argument("--lr_min_frac", type=float, default=0.05)
    ap.add_argument("--out", default="results/compare/heldout_retention.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    if args.probe_seed is not None:
        cfg.data.probe_seed = args.probe_seed
    ctx = MergeContext.build(cfg)

    held = [t for t in ctx.task_names if t in set(args.heldout)]
    train = [t for t in ctx.task_names if t not in set(args.heldout)]
    assert len(held) == len(args.heldout), \
        f"unknown held-out task(s): {set(args.heldout) - set(ctx.task_names)}"
    _log(f"[split] train ({len(train)}): {', '.join(train)}")
    _log(f"[split] held-out ({len(held)}): {', '.join(held)}")

    tv_train = {n: ctx.task_vectors[n] for n in train}
    handles_train = [h for h in ctx.handles if h.name in set(train)]

    report = {"config": cfg.to_dict(), "train_tasks": train, "heldout_tasks": held,
              "grids": {k: v for k, v in vars(args).items() if k != "config"},
              "base": ctx.base_scores, "expert": ctx.expert_scores, "cells": {}}

    def agg(scores, names):
        vals = [scores[n] for n in names]
        return sum(vals) / len(vals), min(vals)

    base_h_mean, _ = agg(ctx.base_scores, held)

    def record(name, state, scores=None, **extra):
        if scores is None:
            scores = ctx.eval_encoder(state)
        tr_m, tr_w = agg(scores, train)
        h_m, h_w = agg(scores, held)
        d0 = pd_global_norm(pd_sub(state, ctx.base_encoder))
        cell = {"scores": scores, "train_mean": tr_m, "train_worst": tr_w,
                "held_mean": h_m, "held_worst": h_w,
                "held_drop_vs_base": h_m - base_h_mean, "dist_theta0": d0, **extra}
        report["cells"][name] = cell
        _log(f"  -> {name}: train={tr_m:.4f}/{tr_w:.4f}  held={h_m:.4f} "
             f"(vs base {base_h_mean:.4f}, drop {h_m - base_h_mean:+.4f})  "
             f"dist0={d0:.3f}")
        return tr_m

    # base model: scores already computed at build time; distance 0 by definition.
    record("base:theta0", ctx.base_encoder, scores=ctx.base_scores)

    # --- TA merge over the TRAIN tasks, lambda swept -------------------------
    best_ta = None  # (train_mean, name, state)
    for lam in args.ta_lams:
        state = task_arithmetic_merge(ctx.base_encoder, tv_train,
                                      {n: lam for n in train})
        m = record(f"merge:TA14@l{lam:g}", state, lam=lam)
        if best_ta is None or m > best_ta[0]:
            best_ta = (m, f"merge:TA14@l{lam:g}", state)
    _log(f"[best TA] {best_ta[1]}")

    # --- TIES merge over the TRAIN tasks ------------------------------------
    for d in args.ties_densities:
        combined = ties_combined_tau(tv_train, density=d)
        for lam in args.ties_lams:
            state = ties_merge(ctx.base_encoder, tv_train, lam=lam, density=d,
                               combined=combined)
            record(f"merge:TIES14@d{d:g},l{lam:g}", state, density=d, lam=lam)

    # --- APR refinement restricted to the TRAIN handles ----------------------
    def run_apr(start, lr, tag):
        rc = dataclasses.replace(
            cfg.refine, steps=args.steps, lr=lr,
            lr_schedule=args.apr_schedule, lr_min_frac=args.lr_min_frac,
            order=(args.apr_order if args.apr_schedule != "constant"
                   else cfg.refine.order))
        _log(f"\n===== APR ({tag}) @ lr{lr:g} S={args.steps} "
             f"{args.apr_schedule}/{rc.order} on {len(handles_train)} tasks =====")
        refined, _ = refine(start, handles_train, rc, ctx.device,
                            seed=cfg.seed, move_model=True, logger=_log)
        refined_cpu = OrderedDict((k, v.cpu()) for k, v in refined.items())
        record(f"apr:{tag}@lr{lr:g}", refined_cpu, lr=lr, steps=args.steps,
               schedule=args.apr_schedule)

    for lr in args.apr_base_lrs:
        run_apr(ctx.base_encoder, lr, "from=base14")
    for lr in args.apr_ta_lrs:
        run_apr(best_ta[2], lr, "from=ta14")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\nRETENTION vs PROXIMITY (held-out mean acc; base = "
          f"{base_h_mean:.4f}):")
    for name, c in sorted(report["cells"].items(),
                          key=lambda kv: kv[1]["dist_theta0"]):
        print(f"  {name:26s} dist0={c['dist_theta0']:6.3f}  "
              f"train={c['train_mean']:.4f}  held={c['held_mean']:.4f} "
              f"({c['held_drop_vs_base']:+.4f})")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
