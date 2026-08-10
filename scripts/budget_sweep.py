#!/usr/bin/env python
"""Small-data / few-sweep test: can APR beat ordinary GD when n and S are tiny?

Premise of the method: a SMALL labeled replay buffer. At n=64, S=5 ordinary GD
already matched APR. This sweeps smaller budgets (n in {8,16,32,64}, few sweeps)
and, in each cell, gives EACH family its own learning-rate sweep (same budget) and
reports the best, averaged over several probe seeds (small buffers are noisy).

Families (all sequential, same data, same #gradient evals):
  ordinary_gd : -g, no anchor/clip
  apr         : -g|v|, AP-gated, clip to +/-gamma|v|     (proposed)
  nogate      : -g|v|, ungated, clip to +/-gamma|v|      (isolates the gate)
"""

import argparse
import json
import os
import sys
import statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_retention
from apr.models import pd_sub, pd_global_norm


def family_configs(steps, gamma):
    fams = {"ordinary_gd": [], "apr": [], "nogate": []}
    for lr in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
        fams["ordinary_gd"].append((lr, RefineConfig(
            steps=steps, lr=lr, gate_mode="none", update_mode="grad", clip_mode="none")))
    for lr in [1, 2, 4, 8, 16, 32, 64]:
        # clip AFTER lr (vdist): step bounded by gamma*|v|; clip fires when
        # lr*|g| > gamma, i.e. when the step would overshoot the gamma-trust region.
        # Grid brackets the vdist optimum (~lr16 from highlr_diag) on both sides.
        fams["apr"].append((lr, RefineConfig(
            steps=steps, lr=lr, clip_frac=gamma, gate_mode="coordinate",
            update_mode="gated_grad", clip_mode="vdist")))
        fams["nogate"].append((lr, RefineConfig(
            steps=steps, lr=lr, clip_frac=gamma, gate_mode="none",
            update_mode="gated_grad", clip_mode="vdist")))
    return fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--n_probes", type=int, nargs="*", default=[8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default="results/compare/budget_sweep.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)  # models/data loaded once
    fams = family_configs(args.steps, args.gamma)

    merge_mean = aggregate_retention(ctx.normret(ctx.merge_scores))["mean_normret"]

    # results[n][family] = list over seeds of (best_mean, best_worst, best_lr)
    results = {n: {f: [] for f in fams} for n in args.n_probes}
    for n in args.n_probes:
        for seed in args.seeds:
            ctx.resample_buffers(n, seed)
            for fam, cfgs in fams.items():
                best = None
                for lr, rc in cfgs:
                    refined, hist = ctx.run_refine(rc, seed=cfg.seed)
                    ag = aggregate_retention(ctx.normret(ctx.eval_encoder(refined)))
                    disp = pd_global_norm(pd_sub(refined, ctx.merged0))
                    clip_g = sum(h["clip_frac_gated"] for h in hist) / len(hist)
                    cand = {"mean": ag["mean_normret"], "worst": ag["worst_normret"],
                            "lr": lr, "disp": disp, "clip_frac_gated": clip_g}
                    if best is None or cand["mean"] > best["mean"]:
                        best = cand
                results[n][fam].append(best)
                _log(f"[n={n} seed={seed}] {fam}: best mean={best['mean']:.3f} "
                     f"worst={best['worst']:.3f} disp={best['disp']:.3f} "
                     f"clip%gated={100*best['clip_frac_gated']:.3f} @lr={best['lr']:g}")

    report = {"config": cfg.to_dict(), "steps": args.steps, "gamma": args.gamma,
              "seeds": args.seeds, "merge_mean": merge_mean, "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print(f"Best-of-lr per family, mean +/- std over {len(args.seeds)} seeds "
          f"(steps={args.steps}, gamma={args.gamma}); merge baseline mean={merge_mean:.3f}")
    print(f"{'n':>4} | {'ordinary_gd':>22} {'apr (gated)':>22} {'nogate':>22}")
    print("-" * 78)
    for n in args.n_probes:
        cells = []
        for fam in ["ordinary_gd", "apr", "nogate"]:
            means = [b["mean"] for b in results[n][fam]]
            disps = [b["disp"] for b in results[n][fam]]
            clips = [b["clip_frac_gated"] for b in results[n][fam]]
            mu = st.mean(means)
            sd = st.pstdev(means) if len(means) > 1 else 0.0
            cells.append(f"{mu:.3f}+-{sd:.3f} d={st.mean(disps):.2f} clip={100*st.mean(clips):.3f}%")
        print(f"{n:>4} | {cells[0]:>38} {cells[1]:>38} {cells[2]:>38}")
    print("=" * 78)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
