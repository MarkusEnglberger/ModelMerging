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
from apr.metrics import aggregate_retention
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
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default="results/compare/regmean.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)
    objective, _ = make_replay_objective(ctx)
    report = {"config": cfg.to_dict(), "tasks": ctx.task_names, "grids": vars(args),
              "cells": {}}

    def record(name, state, **extra):
        ag = aggregate_retention(ctx.normret(ctx.eval_encoder(state)))
        report["cells"][name] = {"mean": ag["mean_normret"], "worst": ag["worst_normret"],
                                 "replay_obj": objective(state),
                                 "disp": pd_global_norm(pd_sub(state, ctx.merged0)), **extra}
        _log(f"  -> {name}: mean={ag['mean_normret']:.3f} worst={ag['worst_normret']:.3f}")
        return ag["mean_normret"], report["cells"][name]["replay_obj"]

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
    for lr in args.apr_lrs:
        rc = dataclasses.replace(cfg.refine, steps=args.steps, lr=lr)
        _log(f"\n===== APR from RegMean @ lr{lr:g} =====")
        refined, _ = ctx.run_refine_from(best[2], rc, seed=cfg.seed)
        record(f"apr<-regmean@lr{lr:g}", refined, lr=lr)

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
