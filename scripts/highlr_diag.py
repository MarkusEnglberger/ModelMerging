#!/usr/bin/env python
"""Push APR-saturating to very high lr and check the lr->infinity limit.

Theory: with clip-after-lr (vdist), as lr->inf every gated coordinate
saturates to clip(lr*(-g|v|), +/-gamma|v|) -> gamma*v (on gated coords
sign(-g)=sign(v)). So APR-saturating should CONVERGE to a fixed, lr-independent
result: "move each gated coordinate a fraction gamma toward its expert" (gated
partial interpolation). We verify by (a) sweeping lr up to 1e6 and (b) running
the interpolation limit directly (update_mode=interp), which should match.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_retention
from apr.models import pd_sub, pd_global_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default="results/compare/highlr_diag.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)
    g, S = args.gamma, args.steps

    methods = {}
    for lr in [16, 64, 256, 1024, 4096, 65536, 1_000_000]:
        methods[f"apr_sat@{lr:g}"] = RefineConfig(
            steps=S, lr=lr, clip_frac=g, gate_mode="coordinate",
            update_mode="gated_grad", clip_mode="vdist")
    # predicted lr->inf limit: gated interpolation by fraction gamma toward expert.
    # update_mode=interp gives u_raw = v (on gated coords); clip to +/-gamma|v|
    # yields exactly gamma*v, applied with unit step (vdist, lr=1).
    methods["interp_limit (gamma*v)"] = RefineConfig(
        steps=S, lr=1.0, clip_frac=g, gate_mode="coordinate",
        update_mode="interp", clip_mode="vdist")

    report = {"config": cfg.to_dict(), "gamma": g, "steps": S, "methods": {}}
    mref = ctx.normret(ctx.merge_scores)
    report["methods"]["merge(S=0)"] = {"aggregate": aggregate_retention(mref),
                                       "displacement": 0.0, "clip_frac_gated": None}

    print(f"\n{'method':<24} {'mean':>7} {'worst':>7} {'clip%gated':>10} {'disp':>8}")
    for name, rc in methods.items():
        refined, hist = ctx.run_refine(rc, seed=cfg.seed)
        nr = ctx.normret(ctx.eval_encoder(refined))
        disp = pd_global_norm(pd_sub(refined, ctx.merged0))
        cg = sum(h["clip_frac_gated"] for h in hist) / len(hist)
        ag = aggregate_retention(nr)
        report["methods"][name] = {"aggregate": ag, "displacement": disp,
                                   "clip_frac_gated": cg, "normret": nr}
        print(f"{name:<24} {ag['mean_normret']:>7.3f} {ag['worst_normret']:>7.3f} "
              f"{100*cg:>9.2f}% {disp:>8.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
