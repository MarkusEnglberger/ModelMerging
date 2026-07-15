#!/usr/bin/env python
"""Multi-seed replication of the long-horizon-vs-calibrate comparison.

The single-seed control found that at n=32 on GLUE-8 the annealed long refinement
(0.620) beats calibrate-then-refine (0.563) -- a flip that lies within GLUE's ~0.06
seed variability. This script re-runs the four relevant arms over several replay-buffer
seeds and reports mean +/- std, so the flip is either confirmed or dissolved.

Arms per seed (all at the same freshly drawn buffer):
  apr-short      S=5, fixed order, constant lr        (baseline)
  apr-anneal     S=long_steps, random order, cosine   (the schedule package)
  calibrated     supervised per-tensor coefficient fit
  apr<-calib     refinement from the calibrated merge (two-stage)
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_retention
from apr.calibrate import calibrate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--short_steps", type=int, default=5)
    ap.add_argument("--long_steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=16, help="short/anneal-peak/refine lr")
    ap.add_argument("--lr_min_frac", type=float, default=0.05)
    ap.add_argument("--cal_lr", type=float, default=0.01)
    ap.add_argument("--cal_l2", type=float, default=1e-3)
    ap.add_argument("--cal_steps", type=int, default=300)
    ap.add_argument("--out", default="results/compare/multiseed_longapr.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)

    acc = {a: {"means": [], "worsts": []} for a in
           ["apr-short", "apr-anneal", "calibrated", "apr<-calib"]}

    def score(state):
        return aggregate_retention(ctx.normret(ctx.eval_encoder(state)))

    for seed in args.seeds:
        ctx.resample_buffers(args.n_probe, probe_seed=seed)
        _log(f"\n########## seed {seed} (n={args.n_probe}) ##########")

        rc = dataclasses.replace(cfg.refine, steps=args.short_steps, lr=args.lr,
                                 order="fixed", lr_schedule="constant")
        st, _ = ctx.run_refine_from(ctx.merged0, rc, seed=seed)
        ags = {"apr-short": score(st)}

        rc = dataclasses.replace(cfg.refine, steps=args.long_steps, lr=args.lr,
                                 order="random", lr_schedule="cosine",
                                 lr_min_frac=args.lr_min_frac)
        st, _ = ctx.run_refine_from(ctx.merged0, rc, seed=seed)
        ags["apr-anneal"] = score(st)

        cal_state, _ = calibrate(ctx.base_encoder, ctx.task_vectors, ctx.per_task,
                                 ctx.task_names, ctx.device, layerwise=True,
                                 steps=args.cal_steps, lr=args.cal_lr, batch_size=16,
                                 l2_reg=args.cal_l2, seed=seed, holdout_frac=0.25,
                                 logger=None)
        ags["calibrated"] = score(cal_state)

        rc = dataclasses.replace(cfg.refine, steps=args.short_steps, lr=args.lr,
                                 order="fixed", lr_schedule="constant")
        st, _ = ctx.run_refine_from(cal_state, rc, seed=seed)
        ags["apr<-calib"] = score(st)

        for a, ag in ags.items():
            acc[a]["means"].append(ag["mean_normret"])
            acc[a]["worsts"].append(ag["worst_normret"])
            _log(f"[seed {seed}] {a:<12} mean={ag['mean_normret']:.3f} "
                 f"worst={ag['worst_normret']:.3f}")

    report = {"config": cfg.to_dict(), "seeds": args.seeds, "n_probe": args.n_probe,
              "arms": {}}
    for a, d in acc.items():
        m, w = np.array(d["means"]), np.array(d["worsts"])
        report["arms"][a] = {"mean": float(m.mean()), "mean_std": float(m.std(ddof=1)),
                             "worst": float(w.mean()), "worst_std": float(w.std(ddof=1)),
                             "means": d["means"], "worsts": d["worsts"]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 62)
    print(f"{'arm':<14}{'mean +/- std':>18}{'worst +/- std':>20}")
    print("-" * 62)
    for a, d in sorted(report["arms"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{a:<14}{d['mean']:>8.3f} +/-{d['mean_std']:>6.3f}"
              f"   {d['worst']:>8.3f} +/-{d['worst_std']:>6.3f}")
    print("=" * 62)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
