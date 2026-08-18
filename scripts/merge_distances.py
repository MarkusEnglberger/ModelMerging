#!/usr/bin/env python
"""Measure ||merge - theta_0|| for the split-selected merge of each family.

Why this exists. The grid runs record a ``displacement`` field, but its
reference point differs by entry type: for a refinement cell it is the distance
moved from that cell's own initialization, while for a ``merge:`` entry it is
the distance from the config-lambda merge m(lam_cfg), NOT from theta_0. Only
task arithmetic can be converted analytically -- it is linear in lambda, so
||m(lam) - theta_0|| = lam * ||sum tau|| follows from one measured norm -- which
is why the displacement figure previously carried a TA reference point and
nothing else. Every other family (TIES, DARE-TIES, Breadcrumbs, AdaMerging)
involves trimming, sign election, random dropping or learned per-tensor
coefficients, so its theta_0-distance cannot be recovered from stored norms and
has to be measured directly.

This script rebuilds each family's selected merge and reports the norm. No
evaluation is performed -- it is pure weight arithmetic -- so it is cheap
relative to the grid runs it annotates.

The selected merge per family is read from the grid run's ``best_per_family``,
so the reported distances correspond exactly to the "alone" rows of the paper's
main grid table.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import pd_sub, pd_global_norm
from apr.taskvec import task_arithmetic_merge
from apr.merge_methods import (ties_combined_tau, ties_merge, dare_ties_merge,
                               breadcrumbs_merge)
from apr.adamerging import adamerging


def build(name, ctx, cfg, tv, names):
    """Rebuild the merge identified by a ``best_per_family`` key."""
    if name.startswith("merge:TA@l"):
        lam = float(name.split("@l")[1])
        return task_arithmetic_merge(ctx.base_encoder, tv, {n: lam for n in names})
    if name.startswith("merge:TIES@"):
        d = float(re.search(r"d([0-9.]+)", name).group(1))
        lam = float(re.search(r"l([0-9.]+)$", name).group(1))
        return ties_merge(ctx.base_encoder, tv, lam=lam, density=d,
                          combined=ties_combined_tau(tv, density=d))
    if name.startswith("merge:DARETIES@"):
        dd = float(re.search(r"dd([0-9.]+)", name).group(1))
        t = float(re.search(r"t([0-9.]+)", name).group(1))
        lam = float(re.search(r"l([0-9.]+)$", name).group(1))
        return dare_ties_merge(ctx.base_encoder, tv, lam=lam, drop_density=dd,
                               trim_density=t, seed=cfg.seed)
    if name.startswith("merge:BC@"):
        d = float(re.search(r"d([0-9.]+)", name).group(1))
        o = float(re.search(r"o([0-9.]+)", name).group(1))
        lam = float(re.search(r"l([0-9.]+)$", name).group(1))
        return breadcrumbs_merge(ctx.base_encoder, tv, {n: lam for n in names},
                                 density=d, outlier_frac=o)
    if name.startswith("merge:ADA"):
        # AdaMerging learns its coefficients by entropy minimisation, so
        # rebuilding it means re-running that loop against the same unlabeled
        # inputs. Deliberately not done here: this script is meant to be pure
        # weight arithmetic. Pass --skip "" and wire it up if the reference
        # point is wanted.
        raise NotImplementedError(
            "AdaMerging must be re-fit, not rebuilt; it is skipped by default")
    raise ValueError(f"cannot rebuild merge '{name}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", required=True,
                    help="grid_nn_*_merges_*.json providing best_per_family")
    ap.add_argument("--n_probe", type=int, default=16)
    ap.add_argument("--probe_seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=["ada"],
                    help="families to skip (AdaMerging needs a training loop)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    cfg.data.probe_seed = args.probe_seed
    ctx = MergeContext.build(cfg)

    grid = json.load(open(args.grid))
    fam = grid["best_per_family"]
    names = list(ctx.task_names)
    tv = ctx.task_vectors

    out = {"config": args.config, "grid": args.grid, "distances": {}}
    for family, key in fam.items():
        if family in args.skip:
            _log(f"[skip] {family} ({key})")
            continue
        state = build(key, ctx, cfg, tv, names)
        d0 = pd_global_norm(pd_sub(state, ctx.base_encoder))
        out["distances"][family] = {"cell": key, "dist_theta0": d0}
        _log(f"  {family:10s} {key:34s} ||merge - theta_0|| = {d0:.4f}")
        del state

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    _log(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
