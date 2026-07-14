#!/usr/bin/env python
"""Localize-and-Stitch, done properly: sweep the L1 penalty gamma and the stitch scale.

The learned-mask L&S in the matched-budget table was mis-tuned: gamma=1e-4 overwhelmed
the tiny replay gradients and collapsed every mask sigmoid to ~0.004, so the post-hoc
top-k selected near-arbitrary coordinates (replay loss ended up ABOVE the merge). Two
fixes, both swept here:
  - gamma: much smaller, so the loss term (not the L1 term) determines the relative
    ordering that top-k reads off. gamma is now just a tie-breaker toward sparsity.
  - stitch lambda: the paper stitches at full magnitude (lam=1), but in an 8-task merge
    where even the tuned coefficient is ~0.2, full-magnitude localized vectors overshoot;
    we sweep it like every other family.

Selection is on the replay objective (mean over tasks of L_t/L_t(merge)), never the eval
split. The learned masks are trained once per gamma and reused across the sparsity and
lambda grids. We also refine APR on top of the best learned-L&S init, since L&S is the
closest published relative of the attribution gate and the composition is the key test.
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
from apr.localize_stitch import (learn_sigmoids, masks_from_sigmoids, dataless_masks,
                                 stitch)


def evalcell(ctx, state, objective):
    scores = ctx.eval_encoder(state)
    ag = aggregate_retention(ctx.normret(scores))
    return ag, objective(state), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--ls_gammas", type=float, nargs="*", default=[0, 1e-7, 1e-6, 1e-5])
    ap.add_argument("--ls_lams", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--ls_sparsities", type=float, nargs="*", default=[0.01, 0.05, 0.1])
    ap.add_argument("--ls_steps", type=int, default=300)
    ap.add_argument("--ls_lr", type=float, default=0.1)
    ap.add_argument("--ls_bs", type=int, default=16)
    ap.add_argument("--apr_lrs", type=float, nargs="*", default=[4, 8, 16])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default="results/compare/ls_sweep.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)
    objective, ref = make_replay_objective(ctx)
    report = {"config": cfg.to_dict(), "tasks": ctx.task_names, "grids": vars(args),
              "cells": {}, "replay_ref": ref}

    def record(name, state, **extra):
        ag, ro, _ = evalcell(ctx, state, objective)
        disp = pd_global_norm(pd_sub(state, ctx.merged0))
        report["cells"][name] = {"mean": ag["mean_normret"], "worst": ag["worst_normret"],
                                 "replay_obj": ro, "disp": disp, **extra}
        _log(f"  -> {name}: mean={ag['mean_normret']:.3f} worst={ag['worst_normret']:.3f} "
             f"replay_obj={ro:.4f}")
        return ag["mean_normret"], ro, state

    # ---- data-free control: top-k magnitude masks (reference) ----
    for sp in args.ls_sparsities:
        masks = dataless_masks(ctx.task_vectors, sp)
        for lam in args.ls_lams:
            st, _ = stitch(ctx.base_encoder, ctx.task_vectors, masks, lam=lam)
            record(f"dataless@sp{sp:g},l{lam:g}", st, sparsity=sp, lam=lam, dataless=True)

    # ---- learned masks: one training per gamma, reused across sparsity/lambda ----
    best = None  # (replay_obj, name, state)
    for g in args.ls_gammas:
        _log(f"\n[learned] gamma={g:g}: training masks ({args.ls_steps} steps)")
        sigs = learn_sigmoids(ctx, steps=args.ls_steps, lr=args.ls_lr, gamma=g,
                              batch_size=args.ls_bs, logger=_log)
        for sp in args.ls_sparsities:
            masks = masks_from_sigmoids(sigs, sp)
            for lam in args.ls_lams:
                st, _ = stitch(ctx.base_encoder, ctx.task_vectors, masks, lam=lam)
                m, ro, state = record(f"learned@g{g:g},sp{sp:g},l{lam:g}", st,
                                      gamma=g, sparsity=sp, lam=lam)
                if best is None or ro < best[0]:
                    best = (ro, f"learned@g{g:g},sp{sp:g},l{lam:g}", state)
        del sigs

    if best:
        report["best_learned"] = best[1]
        _log(f"\n[best learned L&S] {best[1]} (replay_obj={best[0]:.4f})")
        # ---- APR on top of the best learned-L&S init ----
        for lr in args.apr_lrs:
            rc = dataclasses.replace(cfg.refine, steps=args.steps, lr=lr)
            _log(f"\n===== APR from best-L&S @ lr{lr:g} =====")
            refined, _ = ctx.run_refine_from(best[2], rc, seed=cfg.seed)
            record(f"apr<-ls@lr{lr:g}", refined, lr=lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'cell':<32}{'mean':>8}{'worst':>8}{'replayObj':>12}")
    print("-" * 72)
    for n, d in sorted(report["cells"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{n:<32}{d['mean']:>8.3f}{d['worst']:>8.3f}{d['replay_obj']:>12.4f}")
    print("=" * 72)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
