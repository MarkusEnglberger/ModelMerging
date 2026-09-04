#!/usr/bin/env python
"""Held-out retention along the XGD refinement trajectory (one buffer draw).

Table tab:heldout8 probes only the endpoint of each draw's selected cell, so
it cannot say WHEN the retention cost is paid. This script re-runs the
protocol-v3 refit itself -- from the pretrained model, the draw's full budget
buffer, the selected constant learning rate, random task order, the draw's
buffer seed -- and captures the state after selected sweep counts (a
constant-schedule trajectory checkpoints exactly; see
run_refine_checkpoints_from). Each checkpoint is then evaluated on the TRAIN
tasks and zero-shot on the reportable held-out tasks, giving train accuracy,
retention, and displacement as functions of S along the very trajectory the
main table reports the endpoint of.

The refinement sees only the TRAIN tasks: the context is built on the full
suite config (so held-out evaluation heads exist), and ctx.handles is
restricted to the train tasks in config order -- the same task list, order,
and refine seed as the cv_protocol refit, so the final checkpoint reproduces
the saved winner up to GPU nondeterminism (pass --winner to measure the gap).
No selection of any kind happens here: (lr, S_max) are the draw's selected
cell, and the held-out tasks are never read by anything but the final eval.
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import pd_sub, pd_global_norm
from apr.gradients import make_grad_fn
from apr.data import sample_replay_buffer

CLIP8 = ["sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]
HELD_REPORT = ["cifar10", "stl10", "pets", "food101", "flowers102", "cifar100",
               "fashion_mnist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/clip20.yaml")
    ap.add_argument("--train", nargs="*", default=CLIP8)
    ap.add_argument("--heldout_report", nargs="*", default=HELD_REPORT,
                    help="held-out tasks with real headroom above chance; the "
                         "same seven Table tab:heldout8 reports")
    ap.add_argument("--budget", type=int, required=True,
                    help="B labeled examples per train task (the full refit "
                         "buffer of the probed cell)")
    ap.add_argument("--buffer_seed", type=int, required=True,
                    help="the draw's buffer seed (fresh draws: 103-105)")
    ap.add_argument("--lr", type=float, required=True,
                    help="the draw's selected learning rate")
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[1, 2, 5, 10, 20, 30, 40, 50],
                    help="sweep counts to capture; max() is the horizon and "
                         "should be the draw's selected S")
    ap.add_argument("--order", default="random", choices=["fixed", "cyclic",
                                                          "random"])
    ap.add_argument("--winner", default=None,
                    help="saved cv_protocol winner .pt of this draw; the "
                         "final checkpoint is compared against it (global "
                         "parameter distance) as a reproduction check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.budget
    cfg.data.probe_seed = args.buffer_seed
    ctx = MergeContext.build(cfg)

    train = [t for t in ctx.task_names if t in set(args.train)]
    held = [t for t in ctx.task_names if t in set(args.heldout_report)]
    assert len(train) == len(args.train), \
        f"unknown train task(s): {set(args.train) - set(ctx.task_names)}"
    assert len(held) == len(args.heldout_report), \
        f"unknown held-out task(s): {set(args.heldout_report) - set(ctx.task_names)}"
    _log(f"[split] train ({len(train)}): {', '.join(train)}")
    _log(f"[split] held-out report ({len(held)}): {', '.join(held)}")

    # the apr arm refines with the config's gate/update/clip; fail loudly if
    # the config no longer describes the anchored gated update
    rf = cfg.refine
    assert (rf.gate_mode, rf.update_mode, rf.clip_mode) == \
        ("coordinate", "gated_grad", "vdist"), \
        f"config refine block is not the XGD arm: {rf}"

    # the refit's construction buffer: B examples per train task at the draw's
    # seed, exactly draw_budget_buffers + set_construction_buffer restricted
    # to the train handles (cv_protocol.py)
    handles_train = [h for h in ctx.handles if h.name in set(train)]
    for h in handles_train:
        info = ctx.per_task[h.name]
        buf = sample_replay_buffer(info["train_ds"], info["spec"], args.budget,
                                   args.buffer_seed, cfg.data.class_balanced)
        h.grad_fn = make_grad_fn(info["model"], buf, info["collator"],
                                 cfg.data.grad_batch_size
                                 or cfg.data.eval_batch_size, ctx.device)

    steps = sorted(set(args.checkpoints))
    rc = dataclasses.replace(cfg.refine, steps=max(steps), lr=args.lr,
                             order=args.order, lr_schedule="constant")
    _log(f"[refit] XGD from theta_0: lr={args.lr:g}, S={max(steps)} "
         f"({args.order} order, seed {cfg.seed}), checkpoints {steps}")
    ctx.handles = handles_train  # the refinement must see ONLY the train tasks
    states, _hist = ctx.run_refine_checkpoints_from(
        ctx.base_encoder, rc, steps, seed=cfg.seed)

    winner_gap = None
    if args.winner:
        w = torch.load(args.winner, map_location="cpu")
        winner_gap = pd_global_norm(pd_sub(states[max(steps)], w))
        _log(f"[check] ||S={max(steps)} state - saved winner|| = "
             f"{winner_gap:.6f} ({args.winner})")
        del w

    def agg(scores, names):
        vals = [scores[n] for n in names]
        return sum(vals) / len(vals), min(vals)

    base_h_mean, _ = agg(ctx.base_scores, held)
    base_t_mean, base_t_worst = agg(ctx.base_scores, train)

    report = {
        "config": cfg.to_dict(), "train_tasks": train, "heldout_tasks": held,
        "budget": args.budget, "buffer_seed": args.buffer_seed, "lr": args.lr,
        "order": args.order, "checkpoints": steps, "winner": args.winner,
        "winner_gap": winner_gap,
        "base": ctx.base_scores,
        "cells": {"S0": {"scores": {n: ctx.base_scores[n] for n in train + held},
                         "train_mean": base_t_mean, "train_worst": base_t_worst,
                         "held_mean": base_h_mean, "held_worst": None,
                         "held_drop_vs_base": 0.0, "dist_theta0": 0.0}},
    }

    for S in steps:
        st = states[S]
        scores = ctx.eval_encoder(st, names=train + held)
        tr_m, tr_w = agg(scores, train)
        h_m, h_w = agg(scores, held)
        d0 = pd_global_norm(pd_sub(st, ctx.base_encoder))
        report["cells"][f"S{S}"] = {
            "scores": scores, "train_mean": tr_m, "train_worst": tr_w,
            "held_mean": h_m, "held_worst": h_w,
            "held_drop_vs_base": h_m - base_h_mean, "dist_theta0": d0}
        _log(f"  -> S={S:3d}: train={tr_m:.4f}/{tr_w:.4f}  held={h_m:.4f} "
             f"(drop {h_m - base_h_mean:+.4f})  dist0={d0:.3f}")
        del states[S], st

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
