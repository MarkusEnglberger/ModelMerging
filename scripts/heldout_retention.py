#!/usr/bin/env python
"""Held-out-task retention: does staying close to the base model preserve its
zero-shot abilities on tasks that were never merged or refined?

Motivation. Refining directly from theta_0 reaches merge-level multitask
accuracy at a small fraction of the merges' parameter displacement (base-APR
0.3-0.7 vs TA 1.04, AdaMerging 1.68, TIES 2.95 from theta_0 on the 20-task
suite). The paper currently sells proximity as a *proxy* for preserved
zero-shot behavior on unseen tasks. This script measures that directly:

  * split the suite into TRAIN tasks (merged / refined on) and HELD-OUT tasks
    (never seen by any merge or refinement step);
  * build each model point from the TRAIN tasks only -- TA merge (lambda
    swept), TIES merge, APR from the base model, APR from the TA merge;
  * evaluate ALL tasks: train-task accuracy = multitask quality, held-out
    accuracy vs the base model's own zero-shot = retention;
  * record ||state - theta_0|| so retention can be plotted against proximity.

The held-out set should be tasks the base model is actually good at
(there is nothing to retain on a ~chance zero-shot task like KMNIST), so the
default holds out the natural-image tasks with the highest base zero-shot and
trains on the specialized/domain-shift tasks.

The context is built on the FULL suite config (all experts + eval sets), and
the train-task restriction is applied to the task vectors and refinement
handles; the replay buffers of held-out tasks are simply never used.
"""

import argparse
import dataclasses
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import OrderedDict

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import pd_sub, pd_global_norm
from apr.taskvec import task_arithmetic_merge
from apr.merge_methods import (ties_combined_tau, ties_merge, dare_ties_merge,
                               breadcrumbs_merge)
from apr.adamerging import adamerging
from apr.regmean import regmean_merge
from apr.refine import refine
from apr.data import sample_replay_buffer
from apr.tatr import tatr_omega, tatr_mask, tatr_merge

DEFAULT_HELDOUT = ["sun397", "stl10", "cifar10", "pets", "food101", "flowers102"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--heldout", nargs="*", default=DEFAULT_HELDOUT)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--probe_seed", type=int, default=None)
    ap.add_argument("--ta_lams", type=float, nargs="*", default=[0.08, 0.1, 0.15, 0.2])
    ap.add_argument("--ties_densities", type=float, nargs="*", default=[0.1])
    ap.add_argument("--ties_lams", type=float, nargs="*", default=[0.6, 0.8])
    # --- additional merge baselines, all built from the TRAIN tasks only ---
    ap.add_argument("--dt_drops", type=float, nargs="*", default=[0.5])
    ap.add_argument("--dt_trims", type=float, nargs="*", default=[0.1])
    ap.add_argument("--dt_lams", type=float, nargs="*", default=[0.4])
    ap.add_argument("--bc_densities", type=float, nargs="*", default=[0.1])
    ap.add_argument("--bc_outliers", type=float, nargs="*", default=[0.01])
    ap.add_argument("--bc_lams", type=float, nargs="*", default=[0.2])
    ap.add_argument("--tatr_specs", nargs="*", default=[],
                    help="TATR merges over the TRAIN tasks, one per draw: "
                         "label=seed<S>,r<ratio>,l<lam> (e.g. "
                         "draw0=seed103,r0.99,l0.3). TATR is data-dependent "
                         "(|grad| at theta_0 on the draw's buffer), so each "
                         "spec redraws that draw's n_probe examples per train "
                         "task with its own seed and rebuilds the merge at "
                         "the protocol-selected (ratio, lam).")
    ap.add_argument("--ada", action="store_true",
                    help="AdaMerging-layer over the TRAIN tasks (entropy-min on "
                         "their unlabeled inputs only)")
    ap.add_argument("--ada_init_lam", type=float, default=0.05)
    ap.add_argument("--ada_steps", type=int, default=300)
    ap.add_argument("--ada_lr", type=float, default=1e-3)
    ap.add_argument("--ada_bs", type=int, default=16)
    ap.add_argument("--ada_workers", type=int, default=0)
    ap.add_argument("--ada_data", default="probe_buffer",
                    choices=["probe_buffer", "eval_ds"],
                    help="unlabeled source for AdaMerging. Default is the "
                         "matched budget (APR's replay inputs, labels "
                         "discarded), which is the protocol the grid runs and "
                         "the paper use. 'eval_ds' is the standard transductive "
                         "formulation: it adapts on the split it is then scored "
                         "on and is NOT budget-comparable to the other arms.")
    ap.add_argument("--regmean", action="store_true",
                    help="RegMean over the TRAIN tasks (Grams from their probe "
                         "buffers). Its raw distance is dominated by "
                         "activation-null-space components, so its retention is "
                         "an independent test of that account.")
    ap.add_argument("--regmean_nd", type=float, default=1.0)
    ap.add_argument("--regmean_eps", type=float, default=1e-3)
    # --- refinement arms ---
    ap.add_argument("--apr_base_lrs", type=float, nargs="*", default=[4, 8])
    ap.add_argument("--apr_ta_lrs", type=float, nargs="*", default=[4])
    ap.add_argument("--apr_ada_lrs", type=float, nargs="*", default=[],
                    help="APR from the AdaMerging merge (needs --ada)")
    ap.add_argument("--gd_base_lrs", type=float, nargs="*", default=[],
                    help="ordinary GD from the base model (the descent probe on "
                         "the retention axis)")
    ap.add_argument("--nogate_base_lrs", type=float, nargs="*", default=[],
                    help="ungated anchored update from the base model")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--apr_base_steps", type=int, default=None,
                    help="override --steps for the from-base APR arm (pin at a "
                         "grid-selected cell); same pattern for the others")
    ap.add_argument("--nogate_base_steps", type=int, default=None)
    ap.add_argument("--gd_base_steps", type=int, default=None)
    ap.add_argument("--apr_ta_steps", type=int, default=None)
    ap.add_argument("--apr_ada_steps", type=int, default=None)
    ap.add_argument("--apr_schedule", default="cosine",
                    choices=["constant", "cosine", "linear"])
    ap.add_argument("--apr_order", default="random",
                    choices=["fixed", "cyclic", "random"])
    ap.add_argument("--lr_min_frac", type=float, default=0.05)
    ap.add_argument("--state_files", nargs="*", default=[],
                    help="label=path.pt pairs: evaluate saved encoder states "
                         "directly (cv_protocol.py --save_winners output), so "
                         "the probed model IS the reported one and nothing is "
                         "re-derived")
    ap.add_argument("--out", default="results/compare/heldout_retention.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    if args.probe_seed is not None:
        cfg.data.probe_seed = args.probe_seed
    ctx = MergeContext.build(cfg)

    held = [t for t in ctx.task_names if t in set(args.heldout)]
    train = [t for t in ctx.task_names if t not in set(args.heldout)]
    assert len(held) == len(args.heldout), \
        f"unknown held-out task(s): {set(args.heldout) - set(ctx.task_names)}"
    _log(f"[split] train ({len(train)}): {', '.join(train)}")
    _log(f"[split] held-out ({len(held)}): {', '.join(held)}")

    tv_train = {n: ctx.task_vectors[n] for n in train}
    handles_train = [h for h in ctx.handles if h.name in set(train)]

    report = {"config": cfg.to_dict(), "train_tasks": train, "heldout_tasks": held,
              "data_sources": {"adamerging": args.ada_data,
                               "regmean": "probe_buffer",
                               "refinement": "probe_buffer"},
              "grids": {k: v for k, v in vars(args).items() if k != "config"},
              "base": ctx.base_scores, "expert": ctx.expert_scores, "cells": {}}

    def agg(scores, names):
        vals = [scores[n] for n in names]
        return sum(vals) / len(vals), min(vals)

    base_h_mean, _ = agg(ctx.base_scores, held)

    def record(name, state, scores=None, **extra):
        if scores is None:
            scores = ctx.eval_encoder(state)
        tr_m, tr_w = agg(scores, train)
        h_m, h_w = agg(scores, held)
        d0 = pd_global_norm(pd_sub(state, ctx.base_encoder))
        cell = {"scores": scores, "train_mean": tr_m, "train_worst": tr_w,
                "held_mean": h_m, "held_worst": h_w,
                "held_drop_vs_base": h_m - base_h_mean, "dist_theta0": d0, **extra}
        report["cells"][name] = cell
        _log(f"  -> {name}: train={tr_m:.4f}/{tr_w:.4f}  held={h_m:.4f} "
             f"(vs base {base_h_mean:.4f}, drop {h_m - base_h_mean:+.4f})  "
             f"dist0={d0:.3f}")
        return tr_m

    # saved winners from protocol-v2 runs: the evaluated model IS the reported
    # one (no re-derivation), and every draw's winner can be probed for error bars
    for spec in args.state_files:
        import torch
        label, path = spec.split("=", 1)
        _log(f"[state] {label} <- {path}")
        st = torch.load(path, map_location="cpu")
        record(label, st, state_file=path)
        del st

    # base model: scores already computed at build time; distance 0 by definition.
    record("base:theta0", ctx.base_encoder, scores=ctx.base_scores)

    # --- TA merge over the TRAIN tasks, lambda swept -------------------------
    best_ta = None  # (train_mean, name, state)
    for lam in args.ta_lams:
        state = task_arithmetic_merge(ctx.base_encoder, tv_train,
                                      {n: lam for n in train})
        m = record(f"merge:TA{len(train)}@l{lam:g}", state, lam=lam)
        if best_ta is None or m > best_ta[0]:
            best_ta = (m, f"merge:TA{len(train)}@l{lam:g}", state)
    if best_ta is not None:
        _log(f"[best TA] {best_ta[1]}")

    # --- TIES merge over the TRAIN tasks ------------------------------------
    for d in args.ties_densities:
        combined = ties_combined_tau(tv_train, density=d)
        for lam in args.ties_lams:
            state = ties_merge(ctx.base_encoder, tv_train, lam=lam, density=d,
                               combined=combined)
            record(f"merge:TIES{len(train)}@d{d:g},l{lam:g}", state, density=d, lam=lam)

    # --- DARE-TIES and Breadcrumbs (checkpoint-only tier) -------------------
    for dd in args.dt_drops:
        for t in args.dt_trims:
            for lam in args.dt_lams:
                state = dare_ties_merge(ctx.base_encoder, tv_train, lam=lam,
                                        drop_density=dd, trim_density=t,
                                        seed=cfg.seed)
                record(f"merge:DARETIES{len(train)}@dd{dd:g},t{t:g},l{lam:g}", state, lam=lam)
    for d in args.bc_densities:
        for o in args.bc_outliers:
            for lam in args.bc_lams:
                state = breadcrumbs_merge(ctx.base_encoder, tv_train,
                                          {n: lam for n in train},
                                          density=d, outlier_frac=o)
                record(f"merge:BC{len(train)}@d{d:g},o{o:g},l{lam:g}", state, lam=lam)

    # --- TATR over the TRAIN tasks (labeled, data-dependent, per draw) ------
    # Omega depends on the draw's labeled buffer, so unlike the checkpoint-only
    # merges each row rebuilds its own draw's buffers with that draw's seed.
    for spec in args.tatr_specs:
        m = re.fullmatch(r"([^=]+)=seed(\d+),r([\d.]+),l([\d.]+)", spec)
        assert m, f"bad --tatr_specs entry: {spec!r}"
        label, tseed, ratio, lam = (m.group(1), int(m.group(2)),
                                    float(m.group(3)), float(m.group(4)))
        bufs = {n: sample_replay_buffer(ctx.per_task[n]["train_ds"],
                                        ctx.per_task[n]["spec"], args.n_probe,
                                        tseed, cfg.data.class_balanced)
                for n in train}
        _log(f"\n[TATR] {label}: Omega from {args.n_probe} examples/task at "
             f"seed {tseed}, mask r={ratio:g}, lam={lam:g}")
        omega = tatr_omega(ctx, bufs, names=train)
        state = tatr_merge(ctx.base_encoder, tv_train,
                           tatr_mask(omega, ratio), lam, train)
        del omega
        record(f"merge:TATR{len(train)}@r{ratio:g},l{lam:g},{label}", state,
               ratio=ratio, lam=lam, buffer_seed=tseed)
        del state

    # --- AdaMerging over the TRAIN tasks (label-free, data-dependent) --------
    ada_state = None
    if args.ada:
        _log(f"\n[AdaMerging-layer] entropy minimisation over {len(train)} train "
             f"tasks ({args.ada_steps} steps, lr={args.ada_lr}, "
             f"init_lam={args.ada_init_lam})")
        ada_state, ada_info = adamerging(
            ctx.base_encoder, tv_train, ctx.per_task, train, ctx.device,
            layerwise=True, steps=args.ada_steps, lr=args.ada_lr,
            batch_size=args.ada_bs, init_lam=args.ada_init_lam, seed=cfg.seed,
            num_workers=args.ada_workers, data_key=args.ada_data, logger=_log)
        suffix = "-matched" if args.ada_data == "probe_buffer" else "-transductive"
        record(f"merge:ADA{len(train)}-layer{suffix}", ada_state,
               ada_data=args.ada_data,
               ada_lam_per_task=ada_info.get("lam_per_task"))

    # --- RegMean over the TRAIN tasks (label-free, data-dependent) ----------
    if args.regmean:
        _log(f"\n[RegMean] Grams from the {len(train)} train tasks' probe buffers")
        rgm_state, rgm_info = regmean_merge(
            ctx.base_encoder, ctx.per_task, train, ctx.device,
            buffer_key="probe_buffer", nondiag_scale=args.regmean_nd,
            eps=args.regmean_eps, batch_size=args.ada_bs, logger=_log)
        record(f"merge:RegMean{len(train)}@nd{args.regmean_nd:g}", rgm_state)

    # --- APR refinement restricted to the TRAIN handles ----------------------
    def run_refine(start, lr, tag, kind="apr", steps=None):
        """kind: apr (gated anchored) | nogate (ungated anchored) | gd (plain)."""
        if steps is None:
            steps = args.steps
        over = dict(steps=steps, lr=lr, lr_schedule=args.apr_schedule,
                    lr_min_frac=args.lr_min_frac,
                    order=(args.apr_order if args.apr_schedule != "constant"
                           else cfg.refine.order))
        if kind == "nogate":
            over.update(gate_mode="none")
        elif kind == "gd":
            over.update(gate_mode="none", update_mode="grad", clip_mode="none")
        rc = dataclasses.replace(cfg.refine, **over)
        _log(f"\n===== {kind.upper()} ({tag}) @ lr{lr:g} S={steps} "
             f"{args.apr_schedule}/{rc.order} on {len(handles_train)} tasks "
             f"({rc.gate_mode}/{rc.update_mode}/clip={rc.clip_mode}) =====")
        refined, _ = refine(start, handles_train, rc, ctx.device,
                            seed=cfg.seed, move_model=True, logger=_log)
        refined_cpu = OrderedDict((k, v.cpu()) for k, v in refined.items())
        record(f"{kind}:{tag}@lr{lr:g}", refined_cpu, lr=lr, steps=steps,
               schedule=args.apr_schedule, kind=kind)

    nt = len(train)
    for lr in args.apr_base_lrs:
        run_refine(ctx.base_encoder, lr, f"from=base{nt}", "apr",
                   steps=args.apr_base_steps)
    for lr in args.nogate_base_lrs:
        run_refine(ctx.base_encoder, lr, f"from=base{nt}", "nogate",
                   steps=args.nogate_base_steps)
    for lr in args.gd_base_lrs:
        run_refine(ctx.base_encoder, lr, f"from=base{nt}", "gd",
                   steps=args.gd_base_steps)
    for lr in args.apr_ta_lrs:
        run_refine(best_ta[2], lr, f"from=ta{nt}", "apr",
                   steps=args.apr_ta_steps)
    if args.apr_ada_lrs:
        assert ada_state is not None, "--apr_ada_lrs requires --ada"
        for lr in args.apr_ada_lrs:
            run_refine(ada_state, lr, f"from=ada{nt}", "apr",
                       steps=args.apr_ada_steps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\nRETENTION vs PROXIMITY (held-out mean acc; base = "
          f"{base_h_mean:.4f}):")
    for name, c in sorted(report["cells"].items(),
                          key=lambda kv: kv[1]["dist_theta0"]):
        print(f"  {name:26s} dist0={c['dist_theta0']:6.3f}  "
              f"train={c['train_mean']:.4f}  held={c['held_mean']:.4f} "
              f"({c['held_drop_vs_base']:+.4f})")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
