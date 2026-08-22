#!/usr/bin/env python
"""Split-selected DOGE/APGD baseline for the paper's n+n protocol.

DOGE/APGD is data-free: the replay buffer is not used to construct any
candidate.  As with the paper's other checkpoint-only baselines, its global
scale is selected by loss on an n-example buffer disjoint from both the
n-example replay buffer and the evaluation split.  Only that selected candidate
is evaluated.
"""

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.apgd import apgd_merge, prepare_apgd
from apr.config import ExperimentConfig
from apr.data import sample_replay_buffer
from apr.metrics import aggregate_all
from apr.models import pd_global_norm, pd_sub
from apr.pipeline import MergeContext, _log
from apr.replay_baselines import make_replay_objective


CACHE_VERSION = 3
CACHE_DIR = "results/merge_cache"


def _cache_key(cfg, eta, args):
    return {
        "version": CACHE_VERSION,
        "method": "DOGE/APGD",
        "base_model": cfg.base_model,
        "modality": cfg.modality,
        "model_dtype": cfg.model_dtype,
        "experts": [{"name": e.name, "checkpoint": e.checkpoint}
                    for e in cfg.experts],
        "eta": eta,
        "iterations": args.iterations,
        "lr": args.apgd_lr,
        "keep_density": args.keep_density,
        "subspace_divisor": args.subspace_divisor,
        "linear_filter": "2D encoder non-layernorm weights",
    }


def _cache_path(cfg, eta, args):
    key = _cache_key(cfg, eta, args)
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"apgd_{digest}.pt"), key


def _load_cache(cfg, eta, args):
    path, key = _cache_path(cfg, eta, args)
    if not os.path.exists(path):
        return None, None
    state = torch.load(path, map_location="cpu", weights_only=True)
    _log(f"[APGD cache] hit {path}")
    return state, {"cached": True, **key}


def _save_cache(cfg, eta, args, state, info):
    path, key = _cache_path(cfg, eta, args)
    os.makedirs(CACHE_DIR, exist_ok=True)
    temporary = path + f".tmp{os.getpid()}"
    torch.save(state, temporary)
    os.replace(temporary, path)
    with open(path.replace(".pt", ".json"), "w") as handle:
        json.dump({"key": key, "apgd": info}, handle, indent=2)
    _log(f"[APGD cache] saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_probe", type=int, required=True)
    parser.add_argument("--n_select", type=int, required=True)
    parser.add_argument("--probe_seed", type=int, default=None)
    parser.add_argument("--selection_seed", type=int, default=None)
    parser.add_argument("--etas", type=float, nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--apgd_lr", type=float, default=1e-4)
    parser.add_argument("--keep_density", type=float, default=0.30)
    parser.add_argument("--subspace_divisor", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n_probe <= 0 or args.n_select <= 0:
        parser.error("--n_probe and --n_select must be positive")

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    # Base/expert/configured-merge scores do not depend on the replay budget.
    # The paired n=8 -> n=16 job can therefore evaluate them once and safely
    # reuse the cache on its second invocation.
    cfg.data.eval0_cache = True
    if args.probe_seed is not None:
        cfg.data.probe_seed = args.probe_seed
    selection_seed = (args.selection_seed if args.selection_seed is not None
                      else cfg.data.probe_seed + 1)
    if selection_seed == cfg.data.probe_seed:
        parser.error("--selection_seed must differ from the probe seed")

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

    # Expert states were already used to construct task vectors and eval0.
    # DOGE needs the vectors, not a second full copy of every expert.
    for handle in ctx.handles:
        handle.expert_encoder = None
    for name in ctx.task_names:
        ctx.per_task[name]["expert_encoder"] = None

    report = {
        "config": cfg.to_dict(),
        "tasks": ctx.task_names,
        "grids": vars(args),
        "selection_protocol": {
            "rule": "held-out replay loss on a disjoint selection buffer",
            "method_data_regime": "checkpoint-only; n affects selection only",
            "n_probe_per_task": args.n_probe,
            "n_select_per_task": args.n_select,
            "total_labels_per_task": args.n_probe + args.n_select,
            "probe_seed": cfg.data.probe_seed,
            "selection_seed": selection_seed,
            "train_selection_overlap": 0,
            "selection_ref_losses": selection_ref,
        },
        "base": ctx.base_scores,
        "expert": ctx.expert_scores,
        "methods": {},
    }

    cached = {}
    missing = []
    for eta in args.etas:
        state, info = _load_cache(cfg, eta, args)
        if state is None:
            missing.append(eta)
        else:
            cached[eta] = (state, info)
    preparation = None
    if missing:
        _log(f"[APGD] preparing shared subspaces for eta candidates {missing}")
        preparation = prepare_apgd(
            ctx.task_vectors, ctx.device,
            subspace_divisor=args.subspace_divisor, logger=_log)

    best = None
    for eta in args.etas:
        if eta in cached:
            state, apgd_info = cached[eta]
        else:
            _log(f"[APGD] eta={eta:g}")
            state, apgd_info = apgd_merge(
                ctx.base_encoder, ctx.task_vectors, eta=eta, device=ctx.device,
                preparation=preparation, iterations=args.iterations,
                lr=args.apgd_lr, keep_density=args.keep_density,
                subspace_divisor=args.subspace_divisor, logger=_log)
            _save_cache(cfg, eta, args, state, apgd_info)
        value = selection_obj(state)
        name = f"merge:APGD@eta{eta:g}"
        report["methods"][name] = {
            "selection_obj": value,
            "evaluated": False,
            "eta": eta,
            "displacement": pd_global_norm(pd_sub(state, ctx.base_encoder)),
            "apgd": apgd_info,
        }
        _log(f"  -> {name}: selection={value:.6f}")
        if best is None or value < best[0]:
            best = (value, name, state)

    value, name, state = best
    scores = ctx.eval_encoder(state)
    normret = ctx.normret(scores)
    aggregate = aggregate_all(
        scores, normret, ctx.base_scores, ctx.expert_scores)
    report["methods"][name].update({
        "scores": scores,
        "normret": normret,
        "aggregate": aggregate,
        "evaluated": True,
    })
    report["selected"] = name
    _log(f"[APGD winner] {name}: selection={value:.6f}, "
         f"mean_acc={aggregate['mean_acc']:.4f}, "
         f"mean_normret={aggregate['mean_normret']:.4f}")

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    _log(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
