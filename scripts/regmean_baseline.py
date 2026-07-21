#!/usr/bin/env python
"""RegMean baseline (label-free, data-dependent), and APR refined on top of it.

RegMean is the remaining owed baseline in the label-free tier. It merges each encoder
linear layer in closed form from input Gram matrices over unlabeled inputs, and averages
the rest. We report it at two unlabeled budgets: matched (the same 64 inputs APR sees) and
a larger sample (RegMean is label-free, so it is normally given more), and we refine APR
on top of it, since RegMean is a strong data-dependent init on both suites.

Grams are gathered on the unlabeled replay inputs; enlarging the sample uses
ctx.resample_buffers, then the buffer is reset to n_probe for the APR stage so the
refinement still uses exactly 64 labeled examples per task.
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_all
from apr.models import pd_sub, pd_global_norm
from apr.replay_baselines import make_replay_objective
from apr.regmean import regmean_merge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--gram_ns", type=int, nargs="*", default=[64, 256],
                    help="unlabeled sample sizes for the Grams (64 = matched budget)")
    ap.add_argument("--nondiag_scales", type=float, nargs="*", default=[0.9, 1.0])
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--gram_bs", type=int, default=16)
    ap.add_argument("--apr_lrs", type=float, nargs="*", default=[4, 8, 16])
    ap.add_argument("--apr_schedules", nargs="*", default=["constant"],
                    choices=["constant", "cosine", "linear"],
                    help="lr schedules for the APR arms; non-constant use --apr_order")
    ap.add_argument("--apr_order", choices=["fixed", "cyclic", "random"],
                    default="random")
    ap.add_argument("--n_val", type=int, default=0,
                    help="held-out labeled examples/task for hyperparameter "
                         "selection, drawn disjointly from the replay buffer "
                         "(32/32 keeps the total labeled budget at 64); every "
                         "cell records val_acc so selection can avoid the test set")
    ap.add_argument("--nogate_lrs", type=float, nargs="*", default=[],
                    help="also run the UNGATED anchored control from the RegMean "
                         "init (gate_mode=none, still distance-scaled + clipped)")
    ap.add_argument("--gd_lrs", type=float, nargs="*", default=[],
                    help="also run ORDINARY GD from the RegMean init (plain -g, "
                         "no gate, no distance scaling, no clip)")
    ap.add_argument("--gd_steps", type=int, nargs="*", default=[],
                    help="long-horizon sweep counts for the GD control (e.g. 15 30 "
                         "60). Defaults to --steps when empty.")
    ap.add_argument("--gd_schedules", nargs="*", default=["constant"],
                    choices=["constant", "cosine", "linear"],
                    help="lr schedule(s) for the GD control; sweeping both "
                         "constant and cosine isolates what annealing buys")
    ap.add_argument("--gd_lr_min_frac", type=float, default=0.05)
    ap.add_argument("--gd_order", choices=["fixed", "cyclic", "random"],
                    default="fixed", help="task order per sweep for the long GD runs")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--select_by", choices=["mean_normret", "mean_acc"],
                    default="mean_normret",
                    help="aggregate selected on; use mean_acc when a task has a "
                         "degenerate expert-base gap (20-task suite).")
    ap.add_argument("--out", default="results/compare/regmean.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    cfg.data.n_val = args.n_val
    ctx = MergeContext.build(cfg)
    objective, _ = make_replay_objective(ctx)
    # base/expert + per-cell raw scores so scripts/rescore.py can re-aggregate.
    report = {"config": cfg.to_dict(), "tasks": ctx.task_names, "grids": vars(args),
              "base": ctx.base_scores, "expert": ctx.expert_scores, "cells": {}}

    def record(name, state, **extra):
        scores = ctx.eval_encoder(state)
        ag = aggregate_all(scores, ctx.normret(scores), ctx.base_scores,
                           ctx.expert_scores)
        cell = {"mean": ag["mean_normret"], "worst": ag["worst_normret"],
                "mean_acc": ag["mean_acc"], "worst_acc": ag["worst_acc"],
                "scores": scores,
                "replay_obj": objective(state),
                "disp": pd_global_norm(pd_sub(state, ctx.merged0)), **extra}
        vtxt = ""
        if args.n_val > 0:
            vs = ctx.val_scores(state)
            cell["val_scores"] = vs
            cell["val_acc"] = sum(vs.values()) / len(vs)
            vtxt = f" val_acc={cell['val_acc']:.4f}"
        report["cells"][name] = cell
        _log(f"  -> {name}: acc={ag['mean_acc']:.4f}/{ag['worst_acc']:.4f} "
             f"nr={ag['mean_normret']:.3f}/{ag['worst_normret']:.3f}{vtxt}")
        return ag[args.select_by], cell["replay_obj"]

    record("merge:TA", ctx.merged0)

    best = None  # (replay_obj, name, state)
    for gn in args.gram_ns:
        # draw gn unlabeled inputs per task for the Grams (matched 64 = the default buffer)
        if gn != args.n_probe:
            ctx.resample_buffers(gn, probe_seed=cfg.seed)
        else:
            ctx.resample_buffers(args.n_probe, probe_seed=cfg.seed)
        for nd in args.nondiag_scales:
            _log(f"\n[regmean] gram_n={gn} nondiag_scale={nd}")
            state, info = regmean_merge(ctx.base_encoder, ctx.per_task, ctx.task_names,
                                        ctx.device, buffer_key="probe_buffer",
                                        nondiag_scale=nd, eps=args.eps,
                                        batch_size=args.gram_bs, logger=_log)
            nm = f"regmean@n{gn},nd{nd:g}"
            _, ro = record(nm, state, gram_n=gn, **info)
            if best is None or ro < best[0]:
                best = (ro, nm, state)

    report["best_regmean"] = best[1]
    _log(f"\n[best regmean] {best[1]}")

    # restore the matched 64-example buffer for the labeled APR stage
    ctx.resample_buffers(args.n_probe, probe_seed=cfg.seed)
    for sched in args.apr_schedules:
      for lr in args.apr_lrs:
        rc = dataclasses.replace(
            cfg.refine, steps=args.steps, lr=lr,
            lr_schedule=sched, lr_min_frac=args.gd_lr_min_frac,
            order=args.apr_order if sched != "constant" else cfg.refine.order)
        _log(f"\n===== APR from RegMean @ lr{lr:g} S={args.steps} {sched} =====")
        refined, _ = ctx.run_refine_from(best[2], rc, seed=cfg.seed)
        nm = f"apr<-regmean@lr{lr:g},S{args.steps}" + \
             ("" if sched == "constant" else f",{sched}")
        record(nm, refined, lr=lr, apr_steps=args.steps, schedule=sched)

    # Controls from the SAME RegMean init (mirrors compare_baselines.build_methods):
    # nogate = anchored/distance-scaled/clipped but ungated; gd = plain SGD.
    for lr in args.nogate_lrs:
        rc = dataclasses.replace(cfg.refine, steps=args.steps, lr=lr, gate_mode="none")
        _log(f"\n===== nogate from RegMean @ lr{lr:g} =====")
        refined, _ = ctx.run_refine_from(best[2], rc, seed=cfg.seed)
        record(f"nogate<-regmean@lr{lr:g}", refined, lr=lr)
    # Ordinary GD gets the LONG-HORIZON treatment (multiple sweep budgets +
    # optional lr annealing / random task order): the fair form of the "GD just
    # needs more steps" objection. APR cells above are all at --steps.
    gd_steps = args.gd_steps or [args.steps]
    for sched in args.gd_schedules:
        for S in gd_steps:
            # annealing/random order are no-ops at S=1 and meaningless without a
            # horizon; skip the duplicate constant-vs-cosine cell at the shortest S
            if sched != "constant" and S <= 1:
                continue
            for lr in args.gd_lrs:
                rc = dataclasses.replace(cfg.refine, steps=S, lr=lr, gate_mode="none",
                                         update_mode="grad", clip_mode="none",
                                         lr_schedule=sched,
                                         lr_min_frac=args.gd_lr_min_frac,
                                         order=args.gd_order if sched != "constant"
                                         else cfg.refine.order)
                suf = "" if sched == "constant" else f",{sched}"
                tag = f"gd<-regmean@lr{lr:g},S{S}{suf}"
                _log(f"\n===== ordinary GD from RegMean @ lr{lr:g} S={S} "
                     f"schedule={sched} =====")
                refined, _ = ctx.run_refine_from(best[2], rc, seed=cfg.seed)
                record(tag, refined, lr=lr, gd_steps=S, schedule=sched)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 68)
    print(f"{'cell':<28}{'mean':>8}{'worst':>8}{'replayObj':>12}")
    print("-" * 68)
    for n, d in sorted(report["cells"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{n:<28}{d['mean']:>8.3f}{d['worst']:>8.3f}{d['replay_obj']:>12.4f}")
    print("=" * 68)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
