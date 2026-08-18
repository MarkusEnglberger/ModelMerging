#!/usr/bin/env python
"""Paper-style MaTS IA3 comparison with one shared, expensive model context.

The main-table protocol selects checkpoint/refinement hyperparameters by mean
evaluation normalized retention (oracle selection, matching main.tex).  For the
uniform task-arithmetic coefficient we additionally report the strict
replay-loss selection used by the paper's matched-data table.

Feasible shared-output baselines:
  checkpoint only: averaging, tuned TA, TIES, DARE, DARE-TIES, Breadcrumbs,
                   magnitude Localize-and-Stitch
  labeled replay:  ordinary GD, ungated distance-scaled descent, APR, matched
                   random/inverted gates
  long horizon:    20-sweep random-order cosine-annealed APR

AdaMerging is intentionally absent: its per-example classification entropy is
not defined for free-form sequence generation.  Head-only calibration is also
inapplicable because MaTS has no separate task-specific head.
"""

import argparse
import dataclasses
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.localize_stitch import dataless_masks, stitch
from apr.merge_methods import (breadcrumbs_merge, dare_ta_merge,
                               dare_ties_merge, ties_combined_tau, ties_merge)
from apr.metrics import aggregate_retention
from apr.models import pd_global_norm, pd_sub
from apr.pipeline import MergeContext, _log
from apr.replay_baselines import make_replay_objective
from apr.taskvec import task_arithmetic_merge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/mats_t5_8/paper_compare.json")
    ap.add_argument("--ta_lams", nargs="*", type=float,
                    default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--apr_lrs", nargs="*", type=float,
                    default=[2, 4, 8, 16, 32])
    ap.add_argument("--nogate_lrs", nargs="*", type=float,
                    default=[0.5, 1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--gd_lrs", nargs="*", type=float,
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    ap.add_argument("--short_steps", type=int, default=5)
    ap.add_argument("--long_steps", type=int, default=20)
    ap.add_argument("--long_lrs", nargs="*", type=float, default=[4, 8, 16, 32])
    ap.add_argument("--long_min_fracs", nargs="*", type=float, default=[0.05, 0.2])
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)
    names = ctx.task_names
    report = {
        "config": cfg.to_dict(), "tasks": names, "grids": vars(args),
        "selection_protocol": {
            "main": "maximum evaluation mean normalized retention (oracle; paper main table)",
            "lambda_strict": "minimum normalized replay loss (paper matched-data table)",
        },
        "excluded": {
            "AdaMerging": "classification entropy is undefined for sequence generation",
            "head-only": "MaTS uses one shared LM head, not task-specific classifiers",
            "RegMean": "the mergeable IA3 variables are multiplicative scales, not linear weights",
        },
        "base": ctx.base_scores, "expert": ctx.expert_scores, "methods": {},
    }

    def save():
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, args.out)

    def record(name, state, scores=None, history=None, **extra):
        if scores is None:
            scores = ctx.eval_encoder(state)
        nr = ctx.normret(scores)
        ag = aggregate_retention(nr)
        cell = {
            "scores": scores, "normret": nr, "aggregate": ag,
            "displacement_from_lam03": pd_global_norm(pd_sub(state, ctx.merged0)),
            **extra,
        }
        if history is not None:
            cell.update({
                "gate_density": (sum(x["gate_density"] for x in history) /
                                 max(len(history), 1)),
                "clipped_frac_gated": (sum(x.get("clipped_frac_gated", 0.0) for x in history) /
                                    max(len(history), 1)),
                "last_lr": history[-1].get("lr_eff") if history else None,
            })
        report["methods"][name] = cell
        save()
        _log(f"  -> {name}: mean={ag['mean_normret']:.3f} "
             f"worst={ag['worst_normret']:.3f}")
        return ag["mean_normret"]

    def best(prefix):
        cells = [(v["aggregate"]["mean_normret"], k)
                 for k, v in report["methods"].items() if k.startswith(prefix)]
        return max(cells)[1]

    # The pretrained state and uniform averaging are checkpoint-only references.
    record("merge:pretrained", ctx.base_encoder, scores=ctx.base_scores)
    avg_lam = 1.0 / len(names)
    avg_state = task_arithmetic_merge(
        ctx.base_encoder, ctx.task_vectors, {n: avg_lam for n in names})
    record(f"merge:average@l{avg_lam:g}", avg_state, lam=avg_lam)

    # Uniform task-arithmetic lambda grid. Reuse the build-time lambda=.3 score.
    ta_states = {}
    for lam in args.ta_lams:
        state = task_arithmetic_merge(
            ctx.base_encoder, ctx.task_vectors, {n: lam for n in names})
        ta_states[lam] = state
        scores = ctx.merge_scores if abs(lam - 0.3) < 1e-12 else None
        record(f"merge:TA@l{lam:g}", state, scores=scores, lam=lam)
    best_ta_name = best("merge:TA@")
    best_lam = report["methods"][best_ta_name]["lam"]
    best_ta = ta_states[best_lam]
    report["selected_ta_eval"] = best_ta_name
    _log(f"\n[selected TA by eval] {best_ta_name}")

    # Also select lambda without touching evaluation, using the same 64 labels.
    replay_objective, replay_ref = make_replay_objective(ctx)
    replay_trace = {str(lam): replay_objective(state)
                    for lam, state in ta_states.items()}
    replay_lam = min(args.ta_lams, key=lambda x: replay_trace[str(x)])
    report["lambda_replay_selection"] = {
        "selected": replay_lam, "trace": replay_trace,
        "reference_losses_at_lam03": replay_ref,
    }
    save()
    _log(f"[selected TA by replay] lambda={replay_lam:g}")

    # Checkpoint-only baselines and their paper grids.
    for density in [0.1, 0.2]:
        combined = ties_combined_tau(ctx.task_vectors, density=density)
        for lam in [0.8, 1.0]:
            state = ties_merge(ctx.base_encoder, ctx.task_vectors, lam=lam,
                               density=density, combined=combined)
            record(f"merge:TIES@d{density:g},l{lam:g}", state,
                   density=density, lam=lam)
    tuned_lams = {n: best_lam for n in names}
    for density in [0.1, 0.3, 0.5]:
        for seed in [0, 1]:
            state = dare_ta_merge(ctx.base_encoder, ctx.task_vectors, tuned_lams,
                                  density=density, seed=seed)
            record(f"merge:DARE@d{density:g},s{seed}", state,
                   density=density, seed=seed, lam=best_lam)
    for trim in [0.1, 0.2]:
        for lam in [0.8, 1.0]:
            state = dare_ties_merge(ctx.base_encoder, ctx.task_vectors, lam=lam,
                                    drop_density=0.5, trim_density=trim, seed=0)
            record(f"merge:DARETIES@dd0.5,t{trim:g},l{lam:g}", state,
                   drop_density=0.5, trim_density=trim, lam=lam)
    for density in [0.1, 0.2]:
        for outlier in [0.01, 0.05]:
            state = breadcrumbs_merge(
                ctx.base_encoder, ctx.task_vectors,
                {n: 0.4 for n in names}, density=density, outlier_frac=outlier)
            record(f"merge:BC@d{density:g},o{outlier:g},l0.4", state,
                   density=density, outlier=outlier, lam=0.4)
    for frac in [0.05, 0.1]:
        masks = dataless_masks(ctx.task_vectors, frac)
        state, info = stitch(ctx.base_encoder, ctx.task_vectors, masks, lam=1.0)
        record(f"merge:LS-mag@f{frac:g},l1", state, sparsity=frac, stitch=info)
        del masks

    # Equal-sized short-horizon grids from the optimized task-arithmetic state.
    for lr in args.apr_lrs:
        rc = dataclasses.replace(
            cfg.refine, steps=args.short_steps, lr=lr,
            gate_mode="coordinate", update_mode="gated_grad",
            order="fixed", lr_schedule="constant")
        state, hist = ctx.run_refine_from(best_ta, rc, seed=cfg.seed)
        record(f"replay:APR-S{args.short_steps}@lr{lr:g}",
               state, history=hist, lr=lr)
    for lr in args.nogate_lrs:
        rc = dataclasses.replace(
            cfg.refine, steps=args.short_steps, lr=lr,
            gate_mode="none", update_mode="gated_grad",
            order="fixed", lr_schedule="constant")
        state, hist = ctx.run_refine_from(best_ta, rc, seed=cfg.seed)
        record(f"replay:ungated-S{args.short_steps}@lr{lr:g}", state,
               history=hist, lr=lr)
    for lr in args.gd_lrs:
        rc = RefineConfig(
            steps=args.short_steps, lr=lr, gate_mode="none", update_mode="grad",
            clip_mode="none", order="fixed", lr_schedule="constant")
        state, hist = ctx.run_refine_from(best_ta, rc, seed=cfg.seed)
        record(f"replay:ordinary-GD-S{args.short_steps}@lr{lr:g}", state,
               history=hist, lr=lr)

    # Matched random and inverted controls at the best short APR hyperparameters.
    best_apr_name = best(f"replay:APR-S{args.short_steps}@")
    best_apr = report["methods"][best_apr_name]
    best_apr_lr = best_apr["lr"]
    report["selected_apr_short"] = best_apr_name
    for mode in ["random", "inverted"]:
        rc = dataclasses.replace(
            cfg.refine, steps=args.short_steps, lr=best_apr_lr,
            gate_mode=mode,
            update_mode="gated_grad", order="fixed", lr_schedule="constant")
        state, hist = ctx.run_refine_from(best_ta, rc, seed=cfg.seed)
        record(f"control:{mode}-S{args.short_steps}@lr{best_apr_lr:g}",
               state, history=hist, lr=best_apr_lr)

    # Requested long variant: random order and cosine decay, with peak and floor grid.
    for floor in args.long_min_fracs:
        for lr in args.long_lrs:
            rc = dataclasses.replace(
                cfg.refine, steps=args.long_steps, lr=lr,
                gate_mode="coordinate", update_mode="gated_grad", order="random",
                lr_schedule="cosine", lr_min_frac=floor)
            state, hist = ctx.run_refine_from(best_ta, rc, seed=cfg.seed)
            record(f"replay:APR-S{args.long_steps}-anneal@lr{lr:g},floor{floor:g}",
                   state, history=hist, lr=lr,
                   floor=floor, schedule="cosine", order="random")

    # Steps-only control separates horizon from annealing/order randomization.
    rc = dataclasses.replace(
        cfg.refine, steps=args.long_steps, lr=best_apr_lr,
        gate_mode="coordinate",
        update_mode="gated_grad", order="fixed", lr_schedule="constant")
    state, hist = ctx.run_refine_from(best_ta, rc, seed=cfg.seed)
    record(f"control:APR-S{args.long_steps}-constant@lr{best_apr_lr:g}",
           state, history=hist, lr=best_apr_lr)

    report["selected"] = {
        "TA_eval": best_ta_name,
        "TA_replay": f"merge:TA@l{replay_lam:g}",
        "APR_short_eval": best_apr_name,
        "ungated_eval": best(f"replay:ungated-S{args.short_steps}@"),
        "ordinary_GD_eval": best(f"replay:ordinary-GD-S{args.short_steps}@"),
        "APR_long_eval": best(f"replay:APR-S{args.long_steps}-anneal@"),
        "TIES_eval": best("merge:TIES@"),
        "DARE_eval": best("merge:DARE@"),
        "DARETIES_eval": best("merge:DARETIES@"),
        "Breadcrumbs_eval": best("merge:BC@"),
        "LS_eval": best("merge:LS-mag@"),
    }
    save()
    _log("\n[selected cells]\n" + json.dumps(report["selected"], indent=2))
    _log(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
