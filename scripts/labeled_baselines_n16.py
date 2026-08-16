#!/usr/bin/env python
"""Standalone labeled baselines for the strict n+n experiment tier.

Fisher statistics and Localize-and-Stitch masks are fitted on ``n_probe`` labeled
examples per task.  Their hyperparameters are selected on a disjoint ``n_select``
buffer; the evaluation split is read only for the selected state.  This script
deliberately does not run APR, ungated refinement, or ordinary replay GD on top.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.data import sample_replay_buffer
from apr.localize_stitch import learn_sigmoids, masks_from_sigmoids, stitch
from apr.metrics import aggregate_all
from apr.models import pd_global_norm, pd_sub
from apr.pipeline import MergeContext, _log
from apr.replay_baselines import fisher_merge, make_replay_objective


def evaluate(ctx, state):
    scores = ctx.eval_encoder(state)
    normret = ctx.normret(scores)
    aggregate = aggregate_all(scores, normret, ctx.base_scores, ctx.expert_scores)
    return scores, normret, aggregate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=16)
    ap.add_argument("--n_select", type=int, default=16)
    ap.add_argument("--probe_seed", type=int, default=None)
    ap.add_argument("--selection_seed", type=int, default=None)
    ap.add_argument("--baselines", nargs="*", default=["fisher", "ls-learned"],
                    choices=["fisher", "ls-learned"])
    ap.add_argument("--fisher_lams", type=float, nargs="*",
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--ls_gammas", type=float, nargs="*",
                    default=[0.0, 1e-7, 1e-6, 1e-5])
    ap.add_argument("--ls_lams", type=float, nargs="*",
                    default=[0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--ls_sparsities", type=float, nargs="*",
                    default=[0.01, 0.05, 0.1])
    ap.add_argument("--ls_steps", type=int, default=300)
    ap.add_argument("--ls_lr", type=float, default=0.1)
    ap.add_argument("--ls_bs", type=int, default=16)
    ap.add_argument("--out", default="results/compare/grid_nn_labeled_n16.json")
    args = ap.parse_args()

    if args.n_probe <= 0 or args.n_select <= 0:
        ap.error("--n_probe and --n_select must be positive")

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    if args.probe_seed is not None:
        cfg.data.probe_seed = args.probe_seed
    selection_seed = (args.selection_seed if args.selection_seed is not None
                      else cfg.data.probe_seed + 1)
    if selection_seed == cfg.data.probe_seed:
        ap.error("--selection_seed must differ from the probe seed")

    ctx = MergeContext.build(cfg)
    for name in ctx.task_names:
        info = ctx.per_task[name]
        buffer, indices = sample_replay_buffer(
            info["train_ds"], info["spec"], args.n_select, selection_seed,
            cfg.data.class_balanced, exclude_indices=info["probe_indices"],
            return_indices=True)
        if set(indices) & set(info["probe_indices"]):
            raise RuntimeError(f"train/selection overlap for {name}")
        info["selection_buffer"] = buffer

    selection_obj, selection_ref = make_replay_objective(
        ctx, buffer_key="selection_buffer")
    report = {
        "config": cfg.to_dict(), "tasks": ctx.task_names, "grids": vars(args),
        "selection_protocol": {
            "rule": "held-out replay loss on a disjoint selection buffer",
            "n_probe_per_task": args.n_probe,
            "n_select_per_task": args.n_select,
            "total_labels_per_task": args.n_probe + args.n_select,
            "probe_seed": cfg.data.probe_seed,
            "selection_seed": selection_seed,
            "train_selection_overlap": 0,
            "selection_ref_losses": selection_ref,
        },
        "base": ctx.base_scores, "expert": ctx.expert_scores, "methods": {},
    }

    def record_selected(name, state, selection_value, **extra):
        scores, normret, aggregate = evaluate(ctx, state)
        report["methods"][name] = {
            "scores": scores, "normret": normret, "aggregate": aggregate,
            "selection_obj": selection_value,
            "displacement": pd_global_norm(pd_sub(state, ctx.merged0)), **extra,
        }
        _log(f"  -> {name}: selection={selection_value:.4f} "
             f"mean={aggregate['mean_normret']:.3f} "
             f"worst={aggregate['worst_normret']:.3f}")

    if "fisher" in args.baselines:
        _log("\n[fisher] fit diagonal empirical Fishers on probe buffers")
        state, info = fisher_merge(ctx, args.fisher_lams, selection_obj, logger=_log)
        selection_value = info.pop("replay_obj")
        record_selected("labeled:fisher", state, selection_value, **info)

    if "ls-learned" in args.baselines:
        best = None
        trace = {}
        for gamma in args.ls_gammas:
            _log(f"\n[learned L&S] gamma={gamma:g}")
            sigmoids = learn_sigmoids(
                ctx, steps=args.ls_steps, lr=args.ls_lr, gamma=gamma,
                batch_size=args.ls_bs, logger=_log)
            for sparsity in args.ls_sparsities:
                masks = masks_from_sigmoids(sigmoids, sparsity)
                for lam in args.ls_lams:
                    state, stitch_info = stitch(
                        ctx.base_encoder, ctx.task_vectors, masks, lam=lam)
                    value = selection_obj(state)
                    key = f"g{gamma:g},sp{sparsity:g},l{lam:g}"
                    trace[key] = value
                    if best is None or value < best[0]:
                        best = (value, state, gamma, sparsity, lam, stitch_info, key)
                del masks
            del sigmoids
        if best is not None:
            value, state, gamma, sparsity, lam, stitch_info, key = best
            record_selected(
                "labeled:ls-learned", state, value, gamma=gamma,
                sparsity=sparsity, lam=lam, stitch=stitch_info,
                selected_cell=key, selection_trace=trace)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    _log(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
