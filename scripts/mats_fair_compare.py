#!/usr/bin/env python
"""Fair MaTS IA3 comparison, with replay selection and one-shot test evaluation.

All data-dependent methods use the configured number of training examples per
task. Merge hyperparameters and post-merge refinement settings are selected
only by mean normalised replay loss; the full MaTS evaluation split is run only
for the selected baseline, APR, and ordinary-GD state in each baseline family.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.data import sample_replay_buffer
from apr.localize_stitch import learn_sigmoids, masks_from_sigmoids, threshold_masks, stitch
from apr.merge_methods import (ties_combined_tau, ties_merge, dare_ta_merge,
                               dare_ties_merge)
from apr.metrics import aggregate_retention
from apr.models import pd_axpy_, pd_clone, pd_sub
from apr.pipeline import MergeContext, _log
from apr.replay_baselines import make_replay_objective, fisher_merge
from apr.taskvec import task_arithmetic_merge


def scaled_state(base, direction, scale):
    state = pd_clone(base)
    pd_axpy_(state, scale, direction)
    return state


def json_write(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(report, handle, indent=2)
    os.replace(tmp, path)


def choose(candidates, objective, label):
    best = None
    trace = []
    for description, state in candidates:
        value = objective(state)
        trace.append({"params": description, "replay_obj": value})
        _log(f"[select:{label}] {description} replay={value:.6f}")
        if best is None or value < best[0]:
            best = (value, description, state)
    return best[2], {"params": best[1], "replay_obj": best[0], "trace": trace}


def available_family_names(n_probe):
    return ("Average", "TaskArithmetic", "TIES", "DARE-TA", "DARE-TIES",
            f"Fisher-{n_probe}", "Localize&Stitch")


def selected_baselines(ctx, objective, requested=None):
    available = available_family_names(ctx.cfg.data.n_probe)
    requested = list(available if requested is None else requested)
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown families: {sorted(unknown)}; "
                         f"available={list(available)}")
    wanted = set(requested)
    names = ctx.task_names
    baselines = OrderedDict()
    metadata = OrderedDict()

    # Parameter/model soup: arithmetic mean of the eight experts.
    if "Average" in wanted:
        state = task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors,
                                      {name: 1.0 / len(names) for name in names})
        baselines["Average"] = state
        metadata["Average"] = {
            "params": {"lambda_per_expert": 1.0 / len(names)},
            "replay_obj": objective(state)}

    lam_grid = [round(x / 10, 1) for x in range(1, 11)]
    if "TaskArithmetic" in wanted:
        candidates = [({"lambda": lam}, task_arithmetic_merge(
            ctx.base_encoder, ctx.task_vectors, {name: lam for name in names}))
            for lam in lam_grid]
        baselines["TaskArithmetic"], metadata["TaskArithmetic"] = choose(
            candidates, objective, "TaskArithmetic")

    if "TIES" in wanted:
        candidates = []
        for density in [0.1, 0.2, 0.3]:
            combined = ties_combined_tau(ctx.task_vectors, density)
            for lam in [round(x / 10, 1) for x in range(1, 16)]:
                candidates.append(({"density": density, "lambda": lam},
                                   ties_merge(ctx.base_encoder, ctx.task_vectors,
                                              lam, density, combined)))
        baselines["TIES"], metadata["TIES"] = choose(
            candidates, objective, "TIES")

    if "DARE-TA" in wanted:
        candidates = []
        for density in [0.1, 0.3, 0.5, 0.7, 0.9]:
            unit = dare_ta_merge(
                ctx.base_encoder, ctx.task_vectors,
                {name: 1.0 for name in names}, density=density, seed=0)
            direction = pd_sub(unit, ctx.base_encoder)
            for lam in lam_grid:
                candidates.append(({
                    "keep_density": density, "lambda": lam, "seed": 0},
                    scaled_state(ctx.base_encoder, direction, lam)))
        baselines["DARE-TA"], metadata["DARE-TA"] = choose(
            candidates, objective, "DARE-TA")

    if "DARE-TIES" in wanted:
        candidates = []
        for keep in [0.1, 0.3, 0.5, 0.7, 0.9]:
            unit = dare_ties_merge(
                ctx.base_encoder, ctx.task_vectors, lam=1.0,
                drop_density=keep, trim_density=0.2, seed=0)
            direction = pd_sub(unit, ctx.base_encoder)
            for lam in [round(x / 10, 1) for x in range(1, 16)]:
                candidates.append(({
                    "keep_density": keep, "trim_density": 0.2,
                    "lambda": lam, "seed": 0},
                    scaled_state(ctx.base_encoder, direction, lam)))
        baselines["DARE-TIES"], metadata["DARE-TIES"] = choose(
            candidates, objective, "DARE-TIES")

    fisher_name = f"Fisher-{ctx.cfg.data.n_probe}"
    if fisher_name in wanted:
        _log(f"[{fisher_name}] empirical diagonal Fisher at each expert")
        state, info = fisher_merge(ctx, [1.0], objective, logger=_log)
        baselines[fisher_name] = state
        metadata[fisher_name] = info

    # He et al.: 64 shots/task, top-1% magnitude initialization (+3/-3),
    # lr=1e7, L1=1e-5, batch 16, and ten epoch-level mask updates.
    if "Localize&Stitch" in wanted:
        _log("[Localize&Stitch] fitting paper-style data-dependent masks")
        sigmoids = learn_sigmoids(
            ctx, steps=10, lr=1e7, gamma=1e-5, batch_size=16,
            init_logit=-3.0, init_sparsity=0.01, init_on=3.0,
            optimizer="official", logger=_log)
        # Keep the official rounded mask as a diagnostic. On IA3 the very large
        # published learning rate may saturate most logits, so it is not guaranteed
        # to remain sparse after adapting the method to this parameterization.
        rounded = threshold_masks(sigmoids, 0.5)
        rounded_state, rounded_info = stitch(
            ctx.base_encoder, ctx.task_vectors, rounded,
            lam=1.0, average_overlaps=True)
        rounded_info.update({"mask": "round(sigmoid)", "threshold": 0.5,
                             "replay_obj": objective(rounded_state)})
        # The comparison baseline enforces the paper's stated 1% cardinality.
        top1 = masks_from_sigmoids(sigmoids, 0.01)
        top1_state, top1_info = stitch(
            ctx.base_encoder, ctx.task_vectors, top1,
            lam=1.0, average_overlaps=True)
        top1_info.update({"mask": "learned-exact-top1%",
                          "replay_obj": objective(top1_state),
                          "rounded_mask_diagnostic": rounded_info})
        baselines["Localize&Stitch"] = top1_state
        metadata["Localize&Stitch"] = top1_info

    return (OrderedDict((name, baselines[name]) for name in requested),
            OrderedDict((name, metadata[name]) for name in requested))


def refine_grid(ctx, start, objective, family, apr_lrs, apr_steps,
                gd_lrs, gd_steps):
    candidates = []
    histories = []
    if family == "APR":
        specs = [(steps, lr, "constant" if steps <= 5 else "cosine")
                 for steps in apr_steps for lr in apr_lrs]
        for steps, lr, schedule in specs:
            cfg = dataclasses.replace(ctx.cfg.refine, steps=steps, lr=lr,
                                      lr_schedule=schedule, lr_min_frac=0.05)
            state, history = ctx.run_refine_from(start, cfg, seed=ctx.cfg.seed)
            candidates.append(({"steps": steps, "lr": lr,
                                "schedule": schedule}, state))
            histories.append(history)
    else:
        if not gd_steps or not gd_lrs:
            raise ValueError("GD step and learning-rate grids must be non-empty")
        if any(steps < 0 for steps in gd_steps):
            raise ValueError(f"GD steps must be non-negative: {gd_steps}")
        unique_steps = sorted(set(gd_steps))
        max_steps = max(unique_steps)
        trajectories = {}
        for lr in gd_lrs:
            cfg = RefineConfig(steps=max_steps, lr=lr, gate_mode="none",
                               update_mode="grad", clip_mode="none",
                               order="fixed", lr_schedule="constant")
            states, history = ctx.run_refine_checkpoints_from(
                start, cfg, unique_steps, seed=ctx.cfg.seed)
            for steps in unique_steps:
                trajectories[(steps, lr)] = (
                    states[steps],
                    [row for row in history if row["sweep"] < steps])

        # Preserve the previous candidate order and therefore tie-breaking.
        for steps in gd_steps:
            for lr in gd_lrs:
                state, history = trajectories[(steps, lr)]
                candidates.append(({"steps": steps, "lr": lr,
                                    "schedule": "constant"}, state))
                histories.append(history)
    state, info = choose(candidates, objective, family)
    winning_index = min(range(len(info["trace"])),
                        key=lambda i: info["trace"][i]["replay_obj"])
    history = histories[winning_index]
    info["mean_gate_density"] = (sum(row["gate_density"] for row in history) /
                                  len(history) if history else None)
    return state, info


def evaluate_cell(ctx, state, objective):
    scores = ctx.eval_encoder(state)
    normret = ctx.normret(scores)
    return {"scores": scores, "normret": normret,
            "aggregate": aggregate_retention(normret),
            "replay_obj": objective(state)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mats_t5_8.yaml")
    parser.add_argument("--out", default="results/mats_t5_8/fair_compare.json")
    parser.add_argument("--n_probe", type=int, default=64)
    parser.add_argument("--n_select", type=int, default=None,
                        help="disjoint validation examples/task; default=n_probe")
    parser.add_argument("--selection_seed", type=int, default=1)
    parser.add_argument("--families", nargs="+", default=None,
                        help="optional subset of baseline family names")
    parser.add_argument("--grad_batch_size", type=int, default=32,
                        help="examples per refinement forward/backward batch")
    parser.add_argument("--apr_lrs", type=float, nargs="+", default=[8, 16, 32])
    parser.add_argument("--apr_steps", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--gd_lrs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--gd_steps", type=int, nargs="+", default=[1, 5, 20])
    args = parser.parse_args()

    started = time.time()
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    if args.grad_batch_size <= 0:
        parser.error("--grad_batch_size must be positive")
    cfg.data.grad_batch_size = args.grad_batch_size
    cfg.data.eval_batch_size = 32
    available = available_family_names(args.n_probe)
    if args.families is not None:
        unknown = set(args.families) - set(available)
        if unknown:
            parser.error(f"unknown families {sorted(unknown)}; "
                         f"available={list(available)}")
    ctx = MergeContext.build(cfg)
    n_select = args.n_probe if args.n_select is None else args.n_select
    selection_indices = {}
    for name in ctx.task_names:
        info = ctx.per_task[name]
        selection_buffer, indices = sample_replay_buffer(
            info["train_ds"], info["spec"], n_select, args.selection_seed,
            cfg.data.class_balanced, exclude_indices=info["probe_indices"],
            return_indices=True)
        if set(indices) & set(info["probe_indices"]):
            raise RuntimeError(f"training/selection overlap for {name}")
        info["selection_buffer"] = selection_buffer
        selection_indices[name] = indices
    objective, ref_losses = make_replay_objective(
        ctx, buffer_key="selection_buffer")
    report = {"protocol": {
        "selection": (f"mean normalized loss on {n_select} held-out, disjoint "
                      f"training examples/task; "
                      "test evaluated once for winners"),
        "n_probe_per_task": args.n_probe, "probe_seed": cfg.data.probe_seed,
        "n_selection_per_task": n_select,
        "selection_seed": args.selection_seed,
        "train_selection_overlap": 0,
        "eval_batch_size": cfg.data.eval_batch_size,
        "grad_batch_size": cfg.data.grad_batch_size,
        "apr_lrs": args.apr_lrs, "apr_steps": args.apr_steps,
        "apr_schedule": "constant for <=5 sweeps, otherwise cosine to 5%",
        "gd_lrs": args.gd_lrs, "gd_steps": args.gd_steps,
        "gd_trajectory_reuse": "one constant-LR run per learning rate",
    }, "config": cfg.to_dict(), "base": ctx.base_scores,
        "experts": ctx.expert_scores, "replay_ref_losses": ref_losses,
        "families": OrderedDict(), "elapsed_seconds": None}
    json_write(args.out, report)

    baselines, baseline_meta = selected_baselines(
        ctx, objective, requested=args.families)
    report["protocol"]["families"] = list(baselines)
    json_write(args.out, report)
    for name, start in baselines.items():
        _log(f"\n========== {name}: APR/GD from selected baseline ==========")
        apr_state, apr_info = refine_grid(ctx, start, objective, "APR",
                                          args.apr_lrs, args.apr_steps,
                                          args.gd_lrs, args.gd_steps)
        gd_state, gd_info = refine_grid(ctx, start, objective, "GD",
                                        args.apr_lrs, args.apr_steps,
                                        args.gd_lrs, args.gd_steps)
        _log(f"[{name}] full evaluation of baseline/APR/GD winners")
        report["families"][name] = {
            "baseline_selection": baseline_meta[name],
            "APR_selection": apr_info, "GD_selection": gd_info,
            "baseline": evaluate_cell(ctx, start, objective),
            "APR": evaluate_cell(ctx, apr_state, objective),
            "GD": evaluate_cell(ctx, gd_state, objective),
        }
        report["elapsed_seconds"] = time.time() - started
        json_write(args.out, report)

    _log(f"[done] {args.out} ({(time.time()-started)/3600:.2f} h)")


if __name__ == "__main__":
    main()
