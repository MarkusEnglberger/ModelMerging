#!/usr/bin/env python
"""Null control: how much does the eval move when the MODEL does not?

Motivation. On the MergeBench decoder track, ordinary replay GD at lr=1e-5 moved the
merge by ||disp||=0.0009 -- essentially nothing -- yet its mean normalized retention
"dropped" by ~0.04-0.10. A model that did not move cannot lose retention, so that gap is
measurement noise, not an effect. This script measures that noise floor directly instead
of inferring it, so we know which (if any) MergeBench differences are real.

Three tests, cheapest first:

  [A] cross-run reproducibility: eval0 (base/expert/merge) is recomputed here; compare it
      to the same cells from a previous run's JSON (--reference). Same weights, same
      greedy decode, different process => any difference is pure eval nondeterminism
      (fp non-associativity flipping an argmax at a near-tie, which for a 400-512 token
      generation can cascade into a different answer).

  [B] within-run repeatability: evaluate the SAME merged state --repeats times in one
      process. Same weights, same code path. Any spread here is nondeterminism too.

  [C] negligible-perturbation sensitivity: take an ordinary-GD step at --null_lr (default
      1e-8, so ||disp|| ~ 1e-7 == numerically the merge) and evaluate. This is the direct
      analogue of the gd@1e-5 cell: it isolates how much the score swings for a model that
      is, for all practical purposes, unchanged.

Read the output as: any method difference smaller than the spread reported here is not
resolvable at this eval size, regardless of its sign.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repeats", type=int, default=3,
                    help="how many times to evaluate the identical merged state")
    ap.add_argument("--null_lr", type=float, default=1e-8,
                    help="ordinary-GD lr for the negligible-perturbation test")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--reference", default="results/compare/mergebench3b_compare.json",
                    help="previous run's JSON, for the cross-run eval0 comparison")
    ap.add_argument("--out", default="results/compare/null_control.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)
    tasks = ctx.task_names
    report = {"config": cfg.to_dict(), "tasks": tasks,
              "base": ctx.base_scores, "expert": ctx.expert_scores}

    # ---- [A] cross-run: this eval0 vs a previous run's eval0 -------------------
    if args.reference and os.path.exists(args.reference):
        ref = json.load(open(args.reference))
        report["cross_run"] = {}
        _log("\n===== [A] cross-run reproducibility (same weights, different process) =====")
        for cell, now in (("base", ctx.base_scores), ("expert", ctx.expert_scores),
                          ("merge", ctx.merge_scores)):
            prev = (ref.get(cell) if cell != "merge"
                    else ref.get("methods", {}).get("merge(S=0)", {}).get("scores"))
            if not prev:
                continue
            diffs = {t: now[t] - prev[t] for t in tasks if t in prev}
            report["cross_run"][cell] = {"now": now, "prev": prev, "diff": diffs}
            for t, dv in diffs.items():
                _log(f"  {cell:7s} {t:12s} now={now[t]:.4f} prev={prev[t]:.4f} "
                     f"diff={dv:+.4f} ({dv*200:+.1f} cases/200)")

    # ---- [B] within-run: evaluate the identical merged state N times -----------
    _log(f"\n===== [B] within-run repeatability ({args.repeats}x identical merge) =====")
    runs = [ctx.merge_scores]  # eval0 already scored the merge once
    for i in range(args.repeats - 1):
        _log(f"  repeat {i+2}/{args.repeats}")
        runs.append(ctx.eval_encoder(ctx.merged0))
    report["repeat_runs"] = runs
    spread = {}
    for t in tasks:
        vals = [r[t] for r in runs]
        spread[t] = {"min": min(vals), "max": max(vals), "range": max(vals) - min(vals)}
        _log(f"  {t:12s} " + " ".join(f"{v:.4f}" for v in vals) +
             f"   range={spread[t]['range']:.4f} ({spread[t]['range']*200:.1f} cases/200)")
    report["repeat_spread"] = spread
    aggs = [aggregate_retention(ctx.normret(r)) for r in runs]
    report["repeat_aggregates"] = aggs
    means = [a["mean_normret"] for a in aggs]
    _log(f"  mean NormRet across identical evals: " + " ".join(f"{m:.3f}" for m in means) +
         f"   RANGE={max(means)-min(means):.3f}")

    # ---- [C] negligible perturbation -------------------------------------------
    _log(f"\n===== [C] negligible perturbation (ordinary GD @ lr={args.null_lr:g}) =====")
    rc = dataclasses.replace(cfg.refine, steps=args.steps, lr=args.null_lr,
                             gate_mode="none", update_mode="grad", clip_mode="none")
    perturbed, _hist = ctx.run_refine_from(ctx.merged0, rc, seed=cfg.seed)
    disp = pd_global_norm(pd_sub(perturbed, ctx.merged0))
    pscores = ctx.eval_encoder(perturbed)
    pagg = aggregate_retention(ctx.normret(pscores))
    report["perturbed"] = {"lr": args.null_lr, "displacement": disp,
                           "scores": pscores, "aggregate": pagg}
    _log(f"  ||displacement|| = {disp:.3e}  (merge is 0)")
    for t in tasks:
        d = pscores[t] - ctx.merge_scores[t]
        _log(f"  {t:12s} merge={ctx.merge_scores[t]:.4f} perturbed={pscores[t]:.4f} "
             f"diff={d:+.4f} ({d*200:+.1f} cases/200)")
    _log(f"  mean NormRet: merge={aggs[0]['mean_normret']:.3f} "
         f"perturbed={pagg['mean_normret']:.3f} "
         f"gap={pagg['mean_normret']-aggs[0]['mean_normret']:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # ---- verdict ---------------------------------------------------------------
    floors = [max(means) - min(means),
              abs(pagg["mean_normret"] - aggs[0]["mean_normret"])]
    print("\n" + "=" * 78)
    print(f"NOISE FLOOR (mean NormRet), all tasks:")
    print(f"  [B] identical-model eval spread : {floors[0]:.3f}")
    print(f"  [C] negligible-perturbation gap : {floors[1]:.3f}")
    print(f"  => any method difference below ~{max(floors):.3f} is NOT resolvable here.")
    print("=" * 78)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
