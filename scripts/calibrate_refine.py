#!/usr/bin/env python
"""Calibrate-then-refine: supervised per-tensor coefficient fit, then APR.

Stage 1 (calibrate): fit ~10^3 per-tensor merge coefficients on the labeled replay
buffer (apr.calibrate) -- the supervised analogue of AdaMerging, which fixes the
low-dimensional per-tensor MIXING error that APR's coordinate-wise update cannot express.
Stage 2 (refine): run APR from the calibrated merge, fixing the residual coordinate-level
interference. Both stages use only the same n=64 labeled examples per task (inductive).

Target to beat: APR from AdaMerging (0.722 on CLIP), which uses a baseline as its init.
If calibrate-then-refine matches it, the method is self-contained; if it underfits, that
localizes the tensor-mixing signal as needing more than 64 labels (also a clean result).

Baselines in the same run: the calibrated merge alone (no refine), APR from plain TA, and
a small l2/lr grid for the calibration (selected on the replay objective).
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
from apr.calibrate import calibrate


def evalcell(ctx, state, objective):
    ag = aggregate_retention(ctx.normret(ctx.eval_encoder(state)))
    return ag, objective(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--variants", nargs="*", default=["layer", "task"],
                    choices=["layer", "task"])
    ap.add_argument("--cal_lrs", type=float, nargs="*", default=[1e-2, 3e-2])
    ap.add_argument("--l2_regs", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--cal_steps", type=int, default=300)
    ap.add_argument("--cal_bs", type=int, default=16)
    ap.add_argument("--holdout_frac", type=float, default=0.25)
    ap.add_argument("--apr_lrs", type=float, nargs="*", default=[4, 8, 16])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default="results/compare/calibrate_refine.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)
    objective, ref = make_replay_objective(ctx)
    report = {"config": cfg.to_dict(), "tasks": ctx.task_names, "grids": vars(args),
              "cells": {}}

    def record(name, state, **extra):
        ag, ro = evalcell(ctx, state, objective)
        disp = pd_global_norm(pd_sub(state, ctx.merged0))
        report["cells"][name] = {"mean": ag["mean_normret"], "worst": ag["worst_normret"],
                                 "replay_obj": ro, "disp": disp, **extra}
        _log(f"  -> {name}: mean={ag['mean_normret']:.3f} worst={ag['worst_normret']:.3f} "
             f"replay_obj={ro:.4f}")
        return ag["mean_normret"], ro

    # reference
    record("merge:TA", ctx.merged0)

    # ---- Stage 1: calibrate (grid over variant x lr x l2, select on replay) ----
    best = None  # (replay_obj, name, state, info)
    for variant in args.variants:
        for lr in args.cal_lrs:
            for l2 in args.l2_regs:
                _log(f"\n[calibrate-{variant}] lr={lr:g} l2={l2:g}")
                state, info = calibrate(
                    ctx.base_encoder, ctx.task_vectors, ctx.per_task, ctx.task_names,
                    ctx.device, layerwise=(variant == "layer"), steps=args.cal_steps,
                    lr=lr, batch_size=args.cal_bs, l2_reg=l2, seed=cfg.seed,
                    holdout_frac=args.holdout_frac, logger=_log)
                nm = f"calib:{variant}@lr{lr:g},l2{l2:g}"
                _, ro = record(nm, state, calibrate=info)
                if best is None or ro < best[0]:
                    best = (ro, nm, state, info)

    report["best_calibrated"] = best[1]
    _log(f"\n[best calibrated merge] {best[1]} (replay_obj={best[0]:.4f})")

    # ---- Stage 2: APR from the best calibrated merge, and from plain TA for contrast ----
    for lr in args.apr_lrs:
        rc = dataclasses.replace(cfg.refine, steps=args.steps, lr=lr)
        _log(f"\n===== APR from calibrated @ lr{lr:g} =====")
        refined, _ = ctx.run_refine_from(best[2], rc, seed=cfg.seed)
        record(f"apr<-calib@lr{lr:g}", refined, lr=lr)
    for lr in args.apr_lrs:
        rc = dataclasses.replace(cfg.refine, steps=args.steps, lr=lr)
        _log(f"\n===== APR from plain TA @ lr{lr:g} (contrast) =====")
        refined, _ = ctx.run_refine_from(ctx.merged0, rc, seed=cfg.seed)
        record(f"apr<-ta@lr{lr:g}", refined, lr=lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'cell':<30}{'mean':>8}{'worst':>8}{'replayObj':>12}")
    print("-" * 72)
    for n, d in sorted(report["cells"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{n:<30}{d['mean']:>8.3f}{d['worst']:>8.3f}{d['replay_obj']:>12.4f}")
    print("=" * 72)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
