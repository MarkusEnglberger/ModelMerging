#!/usr/bin/env python
"""Multi-seed error bars for the headline cells.

Every table in the paper is single-seed, and the measured run-to-run spread
(~0.02 mean NormRet) is comparable to several reported gaps. This script re-runs
only the best-config cell of each comparison across several replay-buffer seeds and
reports mean +/- std, so we learn which orderings survive.

Variance source: the replay buffer (which n_probe examples are sampled), re-drawn per
seed via ctx.resample_buffers. GPU non-determinism folds in for free (fresh grads each
run). Deterministic merge-only cells (task arithmetic, TIES) are computed once.

Recipes are (init, optional-refine): the init builds a merge point at the current
seed's buffers; the refine, if present, runs Algorithm 1 from it. This covers the
composition cells (APR from TIES / AdaMerging) and the ablations (ungated, ordinary GD,
random gate) under one seed loop and one set of buffers.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_retention
from apr.taskvec import task_arithmetic_merge
from apr.merge_methods import ties_merge, dare_ties_merge
from apr.adamerging import adamerging


# --- init builders: (ctx, params) -> encoder state ---------------------------

def init_ta(ctx, p):
    lam = p.get("lam", 0.3)
    return task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors,
                                 {n: lam for n in ctx.task_names})

def init_ties(ctx, p):
    return ties_merge(ctx.base_encoder, ctx.task_vectors, lam=p["lam"], density=p["density"])

def init_dareties(ctx, p):
    return dare_ties_merge(ctx.base_encoder, ctx.task_vectors, lam=p["lam"],
                           drop_density=p["drop"], trim_density=p["trim"], seed=p.get("seed", 0))

def init_ada(ctx, p):
    state, _ = adamerging(ctx.base_encoder, ctx.task_vectors, ctx.per_task,
                          ctx.task_names, ctx.device, layerwise=p.get("layerwise", True),
                          steps=p.get("steps", 300), lr=p.get("lr", 1e-3),
                          batch_size=p.get("bs", 16), init_lam=p.get("init_lam", 0.3),
                          seed=p["_seed"], num_workers=0,
                          data_key=p.get("data_key", "probe_buffer"), logger=None)
    return state

INITS = {"ta": init_ta, "ties": init_ties, "dareties": init_dareties, "ada": init_ada}
# which inits read the replay buffer (=> vary across seeds); others are deterministic
SEED_DEP_INIT = {"ada"}


def refine_cfg(base_refine, steps, spec):
    """Build a RefineConfig for a refine spec {lr, gate}."""
    gate = spec.get("gate", "coordinate")
    if gate == "grad":  # ordinary GD: raw gradient, no gate, no clip
        return RefineConfig(steps=steps, lr=spec["lr"], gate_mode="none",
                            update_mode="grad", clip_mode="none")
    return dataclasses.replace(base_refine, steps=steps, lr=spec["lr"], gate_mode=gate)


def recipes_for(modality):
    """Best-config cells per suite (from the sweeps in results/compare/*)."""
    if modality == "clip":
        return [
            {"name": "TIES",              "init": ("ties", {"lam": 0.8, "density": 0.1})},
            {"name": "AdaMerging-layer",  "init": ("ada", {"layerwise": True})},
            {"name": "GD<-TA",            "init": ("ta", {}), "refine": {"lr": 1e-4, "gate": "grad"}},
            {"name": "ungated<-TA",       "init": ("ta", {}), "refine": {"lr": 2, "gate": "none"}},
            {"name": "random-gate<-TA",   "init": ("ta", {}), "refine": {"lr": 8, "gate": "random"}},
            {"name": "APR<-TA",           "init": ("ta", {}), "refine": {"lr": 8, "gate": "coordinate"}},
            {"name": "APR<-TIES",         "init": ("ties", {"lam": 0.8, "density": 0.1}), "refine": {"lr": 6, "gate": "coordinate"}},
            {"name": "APR<-AdaMerging",   "init": ("ada", {"layerwise": True}), "refine": {"lr": 8, "gate": "coordinate"}},
        ]
    # glue
    return [
        {"name": "TIES",            "init": ("ties", {"lam": 1.0, "density": 0.1})},
        {"name": "GD<-TA",          "init": ("ta", {}), "refine": {"lr": 1e-3, "gate": "grad"}},
        {"name": "ungated<-TA",     "init": ("ta", {}), "refine": {"lr": 4, "gate": "none"}},
        {"name": "random-gate<-TA", "init": ("ta", {}), "refine": {"lr": 8, "gate": "random"}},
        {"name": "APR<-TA",         "init": ("ta", {}), "refine": {"lr": 16, "gate": "coordinate"}},
        {"name": "APR<-DARETIES",   "init": ("dareties", {"lam": 1.0, "drop": 0.5, "trim": 0.1}), "refine": {"lr": 4, "gate": "coordinate"}},
    ]


def is_seed_dependent(rec):
    init_kind = rec["init"][0]
    return (init_kind in SEED_DEP_INIT) or ("refine" in rec)


def run_recipe(ctx, rec, steps, seed):
    init_kind, init_p = rec["init"]
    init_p = dict(init_p); init_p["_seed"] = seed
    state = INITS[init_kind](ctx, init_p)
    if "refine" in rec:
        rc = refine_cfg(ctx.cfg.refine, steps, rec["refine"])
        state, _ = ctx.run_refine_from(state, rc, seed=seed)
    scores = ctx.eval_encoder(state)
    return aggregate_retention(ctx.normret(scores)), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/compare/multiseed.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)
    recs = recipes_for(cfg.modality)

    per_recipe = {r["name"]: {"means": [], "worsts": [], "seeds": []} for r in recs}
    for si, seed in enumerate(args.seeds):
        ctx.resample_buffers(args.n_probe, probe_seed=seed)
        for rec in recs:
            # deterministic merge-only cells: compute once (on the first seed)
            if not is_seed_dependent(rec) and si > 0:
                continue
            ag, _ = run_recipe(ctx, rec, args.steps, seed)
            per_recipe[rec["name"]]["means"].append(ag["mean_normret"])
            per_recipe[rec["name"]]["worsts"].append(ag["worst_normret"])
            per_recipe[rec["name"]]["seeds"].append(seed)
            _log(f"[seed {seed}] {rec['name']:<18} mean={ag['mean_normret']:.3f} "
                 f"worst={ag['worst_normret']:.3f}")

    report = {"config": cfg.to_dict(), "seeds": args.seeds, "steps": args.steps,
              "tasks": ctx.task_names, "recipes": {}}
    for name, d in per_recipe.items():
        m, w = np.array(d["means"]), np.array(d["worsts"])
        report["recipes"][name] = {
            "mean": float(m.mean()), "mean_std": float(m.std(ddof=1)) if len(m) > 1 else 0.0,
            "worst": float(w.mean()), "worst_std": float(w.std(ddof=1)) if len(w) > 1 else 0.0,
            "n_seeds": len(m), "means": d["means"], "worsts": d["worsts"]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 68)
    print(f"{'recipe':<20}{'mean +/- std':>18}{'worst +/- std':>20}{'seeds':>8}")
    print("-" * 68)
    for name, d in sorted(report["recipes"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{name:<20}{d['mean']:>7.3f} +/-{d['mean_std']:>6.3f}   "
              f"{d['worst']:>7.3f} +/-{d['worst_std']:>6.3f}{d['n_seeds']:>8}")
    print("=" * 68)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
