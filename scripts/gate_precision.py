#!/usr/bin/env python
"""Sign-precision of the attribution gate as a function of |g*v| magnitude.

Question (raised against the random-mask parity on GLUE): the count-based gate density
is ~chance, but is there real attribution signal concentrated in the HEAVY |g*v|
coordinates that the eps=0 gate dilutes across the noise floor?

Test, at the task-arithmetic merge, per task: bucket the loss-decreasing-predicted
coordinates (g*v<0) by |g*v| percentile, take a small interpolation step toward the
expert restricted to each bucket, theta' = theta + alpha * v * mask_bucket, and compare
the MEASURED replay-loss change against the first-order PREDICTION alpha*sum(g*v).
A control bucket of g*v>0 coordinates (top magnitude) is predicted to INCREASE loss.

If measured/predicted agreement is high in the top buckets and ~chance in the bulk,
the gate's failure on GLUE is its eps=0 threshold, not the absence of signal.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import load_encoder_state
from apr.replay_baselines import buffer_loss

BUCKETS = [(0.0, 0.001, "top 0.1%"), (0.001, 0.01, "0.1-1%"), (0.01, 0.05, "1-5%"),
           (0.05, 0.2, "5-20%"), (0.2, 0.5, "20-50%"), (0.5, 1.0, "bottom 50%")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.05, 0.2])
    ap.add_argument("--out", default="results/compare/gate_precision.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)
    dev = ctx.device
    report = {"config": cfg.to_dict(), "alphas": args.alphas, "tasks": {}}

    theta = {k: v.to(dev) for k, v in ctx.merged0.items()}
    for h in ctx.handles:
        name = h.name
        info = ctx.per_task[name]
        model = info["model"]
        model.to(dev)
        load_encoder_state(model, theta)
        g = h.grad_fn()
        expert = {k: v.to(dev) for k, v in h.expert_encoder.items()}
        v = {k: expert[k] - theta[k] for k in theta}
        prod = torch.cat([(g[k] * v[k]).reshape(-1) for k in theta])
        L0 = buffer_loss(model, info["probe_buffer"], info["collator"],
                         ctx.cfg.data.eval_batch_size, dev)

        neg = -prod.clamp(max=0)          # |g*v| where product negative, else 0
        n_neg = int((prod < 0).sum())
        if n_neg == 0:
            _log(f"  [{name}] no negative-product coordinates (degenerate config); skipping")
            report["tasks"][name] = {"L0": L0, "rows": [], "degenerate": True}
            model.to("cpu")
            continue
        order_thr = {}
        # percentile thresholds among NEGATIVE-product coords by |g*v|
        negvals = neg[prod < 0]
        for lo, hi, lbl in BUCKETS:
            k_lo = max(1, int(round(lo * n_neg)))
            k_hi = max(1, int(round(hi * n_neg)))
            hi_thr = torch.kthvalue(negvals, n_neg - k_lo + 1).values if k_lo < n_neg else negvals.max()
            lo_thr = torch.kthvalue(negvals, n_neg - k_hi + 1).values
            order_thr[lbl] = (float(lo_thr), float(hi_thr))

        rows = []
        for lo, hi, lbl in BUCKETS:
            lo_thr, hi_thr = order_thr[lbl]
            masks = {k: ((g[k] * v[k]) < 0) & ((-(g[k] * v[k])) >= lo_thr)
                        & ((-(g[k] * v[k])) <= hi_thr) for k in theta}
            nsel = sum(int(m.sum()) for m in masks.values())
            pred_unit = float(sum((g[k] * v[k] * masks[k]).sum() for k in theta))
            for a in args.alphas:
                with torch.no_grad():
                    pert = {k: theta[k] + a * v[k] * masks[k] for k in theta}
                load_encoder_state(model, pert)
                L1 = buffer_loss(model, info["probe_buffer"], info["collator"],
                                 ctx.cfg.data.eval_batch_size, dev)
                rows.append({"bucket": lbl, "alpha": a, "n_coords": nsel,
                             "pred_dloss": a * pred_unit, "meas_dloss": L1 - L0,
                             "sign_correct": (L1 - L0) < 0})
        # control: TOP-magnitude POSITIVE-product coords (predicted to hurt)
        posvals = prod.clamp(min=0)
        n_pos = int((prod > 0).sum())
        k = max(1, int(round(0.01 * n_pos)))
        thr = torch.kthvalue(posvals[prod > 0], n_pos - k + 1).values
        masks = {kk: ((g[kk] * v[kk]) > 0) & ((g[kk] * v[kk]) >= thr) for kk in theta}
        pred_unit = float(sum((g[kk] * v[kk] * masks[kk]).sum() for kk in theta))
        for a in args.alphas:
            with torch.no_grad():
                pert = {kk: theta[kk] + a * v[kk] * masks[kk] for kk in theta}
            load_encoder_state(model, pert)
            L1 = buffer_loss(model, info["probe_buffer"], info["collator"],
                             ctx.cfg.data.eval_batch_size, dev)
            rows.append({"bucket": "POS top 1% (control)", "alpha": a,
                         "n_coords": sum(int(m.sum()) for m in masks.values()),
                         "pred_dloss": a * pred_unit, "meas_dloss": L1 - L0,
                         "sign_correct": (L1 - L0) > 0})
        report["tasks"][name] = {"L0": L0, "rows": rows}
        for r in rows:
            _log(f"  [{name}] {r['bucket']:<20} a={r['alpha']:<5} pred={r['pred_dloss']:+.4f} "
                 f"meas={r['meas_dloss']:+.4f} {'OK' if r['sign_correct'] else 'X'}")
        model.to("cpu")
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # summary: sign-accuracy per bucket across tasks
    print("\nsign-agreement (measured dloss has predicted sign), by bucket:")
    from collections import defaultdict
    agg = defaultdict(list)
    for t, d in report["tasks"].items():
        for r in d["rows"]:
            agg[(r["bucket"], r["alpha"])].append(r["sign_correct"])
    for (b, a), xs in sorted(agg.items()):
        print(f"  {b:<22} alpha={a:<5}: {sum(xs)}/{len(xs)} tasks")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
