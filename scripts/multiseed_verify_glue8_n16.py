#!/usr/bin/env python
"""Pinned multi-seed verification of the headline GLUE-8 n=16 column.

The split-selection grids choose hyperparameters on replay seed 0.  This script
keeps those choices fixed and varies only the replay-buffer draw, which measures
the run-to-run uncertainty of the reported cells without re-selecting on each
draw.  Refinement/task-order RNG stays fixed at config seed 0.
"""

import argparse
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.adamerging import adamerging
from apr.config import ExperimentConfig
from apr.merge_methods import breadcrumbs_merge, dare_ties_merge, ties_merge
from apr.metrics import aggregate_all
from apr.pipeline import MergeContext, _log
from apr.taskvec import task_arithmetic_merge


# Pinned seed-0 winners from the files named below.  The from-pretrained
# ungated winner comes from the larger horizon audit, which is the value used in
# paperICLR/main.tex; all merge-initialized winners come from the fast grid.
SOURCE_FILES = {
    "pretrained": "results/compare/grid_nn_glue8_base_n16_horizon.json",
    "merges": "results/compare/grid_nn_glue8_merges_n16_fast.json",
}

CELLS = {
    "pretrained": {
        "apr": (8.0, 50),
        "nogate": (16.0, 50),
        "gd": (1e-3, 50),
    },
    "ta": {
        "apr": (4.0, 5),
        "nogate": (2.0, 5),
        "gd": (1e-3, 20),
    },
    "ties": {
        "apr": (2.0, 20),
        "nogate": (0.5, 20),
        "gd": (5e-4, 20),
    },
    "dareties": {
        "apr": (4.0, 5),
        "nogate": (1.0, 5),
        "gd": (1e-3, 5),
    },
    "bc": {
        "apr": (1.0, 20),
        "nogate": (1.0, 5),
        "gd": (5e-4, 20),
    },
    "ada": {
        "apr": (8.0, 5),
        "nogate": (2.0, 5),
        "gd": (5e-4, 50),
    },
}


def score(ctx, state):
    scores = ctx.eval_encoder(state)
    normret = ctx.normret(scores)
    return {
        "scores": scores,
        "normret": normret,
        "aggregate": aggregate_all(
            scores, normret, ctx.base_scores, ctx.expert_scores),
    }


def refine_config(base, arm, lr, steps):
    common = {"lr": lr, "steps": steps, "lr_schedule": "constant",
              "order": "fixed"}
    if arm == "apr":
        return dataclasses.replace(base, gate_mode="coordinate", **common)
    if arm == "nogate":
        return dataclasses.replace(base, gate_mode="none", **common)
    if arm == "gd":
        return dataclasses.replace(
            base, gate_mode="none", update_mode="grad", clip_mode="none",
            **common)
    raise ValueError(f"unknown arm: {arm}")


def build_init(ctx, name):
    names = ctx.task_names
    if name == "pretrained":
        return task_arithmetic_merge(
            ctx.base_encoder, ctx.task_vectors, {n: 0.0 for n in names})
    if name == "ta":
        return task_arithmetic_merge(
            ctx.base_encoder, ctx.task_vectors, {n: 0.3 for n in names})
    if name == "ties":
        return ties_merge(
            ctx.base_encoder, ctx.task_vectors, density=0.2, lam=1.0)
    if name == "dareties":
        return dare_ties_merge(
            ctx.base_encoder, ctx.task_vectors, drop_density=0.5,
            trim_density=0.1, lam=1.0, seed=0)
    if name == "bc":
        return breadcrumbs_merge(
            ctx.base_encoder, ctx.task_vectors, {n: 0.4 for n in names},
            density=0.2, outlier_frac=0.01)
    if name == "ada":
        state, _ = adamerging(
            ctx.base_encoder, ctx.task_vectors, ctx.per_task, names, ctx.device,
            layerwise=False, steps=300, lr=1e-3, batch_size=16,
            init_lam=0.3, seed=ctx.cfg.seed, num_workers=0,
            data_key="probe_buffer", logger=_log)
        return state
    raise ValueError(f"unknown initialization: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/glue8.yaml")
    parser.add_argument("--probe_seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_eval", type=int, default=None,
                        help="optional smoke-test cap; omit for verification")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = 16
    cfg.data.probe_seed = args.probe_seed
    if args.max_eval is not None:
        cfg.data.max_eval = args.max_eval
    ctx = MergeContext.build(cfg)

    report = {
        "protocol": {
            "benchmark": "GLUE-8",
            "n_probe_per_task": 16,
            "probe_seed": args.probe_seed,
            "refine_seed": cfg.seed,
            "selection": "pinned seed-0 split-selection winners",
            "varied": "replay-buffer draw only",
            "source_files": SOURCE_FILES,
        },
        "config": cfg.to_dict(),
        "tasks": ctx.task_names,
        "base": ctx.base_scores,
        "expert": ctx.expert_scores,
        "cells": {},
    }

    for init_name, arms in CELLS.items():
        _log(f"\n######## init={init_name} seed={args.probe_seed} ########")
        start = build_init(ctx, init_name)
        init_result = score(ctx, start)
        report["cells"][f"{init_name}:alone"] = init_result
        ag = init_result["aggregate"]
        _log(f"[pinned] {init_name}:alone mean={ag['mean_normret']:.4f} "
             f"worst={ag['worst_normret']:.4f}")

        for arm, (lr, steps) in arms.items():
            rc = refine_config(cfg.refine, arm, lr, steps)
            refined, _ = ctx.run_refine_from(start, rc, seed=cfg.seed)
            result = score(ctx, refined)
            result.update({"init": init_name, "arm": arm, "lr": lr,
                           "steps": steps})
            key = f"{init_name}:{arm}"
            report["cells"][key] = result
            ag = result["aggregate"]
            _log(f"[pinned] {key} lr={lr:g} S={steps} "
                 f"mean={ag['mean_normret']:.4f} "
                 f"worst={ag['worst_normret']:.4f}")
            del refined
            if ctx.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del start

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + f".tmp{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(report, handle, indent=2)
    os.replace(tmp, args.out)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
