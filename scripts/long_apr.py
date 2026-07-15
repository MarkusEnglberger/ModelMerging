#!/usr/bin/env python
"""Long-horizon APR control: can more steps + random task order + lr annealing
substitute for the calibration stage?

Reviewer-facing question: the two-stage method's advantage might just be that plain
APR is under-optimized (S=5, fixed order, constant lr). This script runs APR from the
task-arithmetic merge with S sweeps (default 30), random task order, and a cosine
learning-rate decay, and compares against (i) the plain S=5 recipe re-run at the same
seed/buffers, (ii) a steps-only control (S sweeps, fixed order, constant lr) that
disentangles schedule from budget, and (iii) calibrate-then-refine at the same replay
budget. Repeated at each requested n_probe (default 64 and 32).

If long-APR closes the gap to calibrate-then-refine, the two-stage story weakens to a
compute statement; if it plateaus near its S=5 value (as the S=15 ablation and the
S=35 GD ceiling suggest), the gap is structural (joint objective + subspace
regularization), not a budget artifact.
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
from apr.calibrate import calibrate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probes", type=int, nargs="*", default=[64, 32])
    ap.add_argument("--long_steps", type=int, default=30)
    ap.add_argument("--long_lrs", type=float, nargs="*", default=[8, 16],
                    help="peak lrs for the annealed long runs")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--lr_min_frac", type=float, default=0.05)
    ap.add_argument("--base_lr", type=float, default=8,
                    help="best-known S=5 lr (short baseline + steps-only control)")
    ap.add_argument("--base_steps", type=int, default=5)
    # calibrate-then-refine reference at the same budget
    ap.add_argument("--cal_lr", type=float, default=0.03)
    ap.add_argument("--cal_l2", type=float, default=1e-3)
    ap.add_argument("--cal_steps", type=int, default=300)
    ap.add_argument("--cal_refine_lr", type=float, default=8)
    ap.add_argument("--skip_calibrate", action="store_true")
    ap.add_argument("--out", default="results/compare/long_apr.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probes[0]
    ctx = MergeContext.build(cfg)
    report = {"config": cfg.to_dict(), "tasks": ctx.task_names, "grids": vars(args),
              "cells": {}}

    def run(name, rc, start=None):
        refined, hist = ctx.run_refine_from(start if start is not None else ctx.merged0,
                                            rc, seed=cfg.seed)
        ag = aggregate_retention(ctx.normret(ctx.eval_encoder(refined)))
        report["cells"][name] = {
            "mean": ag["mean_normret"], "worst": ag["worst_normret"],
            "disp": pd_global_norm(pd_sub(refined, ctx.merged0)),
            "steps": rc.steps, "lr": rc.lr, "order": rc.order,
            "lr_schedule": rc.lr_schedule,
            "lr_eff_last": hist[-1]["lr_eff"] if hist else None}
        _log(f"  -> {name}: mean={ag['mean_normret']:.3f} worst={ag['worst_normret']:.3f}")
        return refined

    for n in args.n_probes:
        ctx.resample_buffers(n, probe_seed=cfg.seed)
        tag = f"n{n}"
        _log(f"\n########## replay budget n={n} ##########")

        # (i) plain S=5 baseline at this budget/seed (re-anchor)
        rc = dataclasses.replace(cfg.refine, steps=args.base_steps, lr=args.base_lr,
                                 order="fixed", lr_schedule="constant")
        run(f"{tag}:apr-short@lr{args.base_lr:g}", rc)

        # (ii) steps-only control: long horizon, fixed order, constant lr
        rc = dataclasses.replace(cfg.refine, steps=args.long_steps, lr=args.base_lr,
                                 order="fixed", lr_schedule="constant")
        run(f"{tag}:apr-longsteps@lr{args.base_lr:g}", rc)

        # (iii) the full proposal: long horizon + random order + annealed lr
        for lr in args.long_lrs:
            rc = dataclasses.replace(cfg.refine, steps=args.long_steps, lr=lr,
                                     order="random", lr_schedule=args.schedule,
                                     lr_min_frac=args.lr_min_frac)
            run(f"{tag}:apr-long-anneal@lr{lr:g}", rc)

        # (iv) calibrate-then-refine reference at the same budget
        if not args.skip_calibrate:
            _log(f"\n[calibrate] n={n} (layer-wise, lr={args.cal_lr}, l2={args.cal_l2})")
            cal_state, cal_info = calibrate(
                ctx.base_encoder, ctx.task_vectors, ctx.per_task, ctx.task_names,
                ctx.device, layerwise=True, steps=args.cal_steps, lr=args.cal_lr,
                batch_size=16, l2_reg=args.cal_l2, seed=cfg.seed,
                holdout_frac=0.25, logger=_log)
            ag = aggregate_retention(ctx.normret(ctx.eval_encoder(cal_state)))
            report["cells"][f"{tag}:calibrated"] = {
                "mean": ag["mean_normret"], "worst": ag["worst_normret"],
                "disp": pd_global_norm(pd_sub(cal_state, ctx.merged0)),
                "calibrate": cal_info}
            _log(f"  -> {tag}:calibrated: mean={ag['mean_normret']:.3f} "
                 f"worst={ag['worst_normret']:.3f}")
            rc = dataclasses.replace(cfg.refine, steps=args.base_steps,
                                     lr=args.cal_refine_lr, order="fixed",
                                     lr_schedule="constant")
            run(f"{tag}:apr<-calib@lr{args.cal_refine_lr:g}", rc, start=cal_state)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'cell':<34}{'mean':>8}{'worst':>8}{'disp':>8}")
    print("-" * 70)
    for nname, d in sorted(report["cells"].items(),
                           key=lambda kv: (kv[0].split(':')[0], -kv[1]['mean'])):
        print(f"{nname:<34}{d['mean']:>8.3f}{d['worst']:>8.3f}{d['disp']:>8.2f}")
    print("=" * 70)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
