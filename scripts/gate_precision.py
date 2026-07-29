#!/usr/bin/env python
"""Sign-precision of the attribution gate as a function of coordinate magnitude.

Question (raised against the random-mask parity on GLUE): the count-based gate density
is ~chance, but is there real attribution signal concentrated in the HEAVY coordinates
that the eps=0 gate dilutes across the noise floor -- and if so, heavy BY WHICH STATISTIC?

Test, at the merge, per task: among the loss-decreasing-predicted coordinates (g*v<0),
bucket by percentile of a selection statistic, take a small interpolation step toward
the expert restricted to each bucket, theta' = theta + alpha * v * mask_bucket, and
compare the MEASURED replay-loss change against the first-order PREDICTION
alpha*sum(g*v). A control bucket of g*v>0 coordinates (top magnitude) is predicted to
INCREASE loss. The intervention is identical across statistics; only the selection
changes, so the statistics are directly comparable.

Statistics (--stats):
  gv     |g*v|              -- the original thresholding quantity (topk gate)
  g      |g|                -- significance filter on the only NOISY factor (v is
                              deterministic); a small-|g| coordinate with a large |v|
                              passes |g*v| despite carrying a coin-flip sign
  gnorm  |g| / rms_tensor(g) -- |g| with per-tensor scale heterogeneity removed

If precision is high in the top buckets of |g| but not |g*v|, the gate's failure is
that it thresholds noise amplified by fictitious distance; if flat everywhere, no
threshold can help and the ungated-anchor story stands.
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
    ap.add_argument("--n_val", type=int, default=64,
                    help="size of the DISJOINT held-out buffer used to measure the "
                         "loss change (0 = train-buffer only, the old behaviour). "
                         "The gradient is always computed on the train buffer, so "
                         "the train-vs-held-out gap per bucket is the noise readout.")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.05, 0.2])
    ap.add_argument("--stats", nargs="*", default=["gv", "g", "gnorm"],
                    choices=["gv", "g", "gnorm"],
                    help="selection statistics to bucket by (see module docstring)")
    ap.add_argument("--out", default="results/compare/gate_precision.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    cfg.data.n_val = args.n_val   # disjoint split (see data.sample_replay_buffer_split)
    ctx = MergeContext.build(cfg)
    dev = ctx.device
    report = {"config": cfg.to_dict(), "alphas": args.alphas,
              "n_probe": args.n_probe, "n_val": args.n_val, "tasks": {}}

    def losses(model, info, state):
        """(train-buffer loss, held-out-buffer loss) at `state`."""
        load_encoder_state(model, state)
        lt = buffer_loss(model, info["probe_buffer"], info["collator"],
                         ctx.cfg.data.eval_batch_size, dev)
        lv = (buffer_loss(model, info["val_buffer"], info["collator"],
                          ctx.cfg.data.eval_batch_size, dev)
              if args.n_val > 0 else None)
        return lt, lv

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
        L0, L0v = losses(model, info, theta)

        n_neg = int((prod < 0).sum())
        if n_neg == 0:
            _log(f"  [{name}] no negative-product coordinates (degenerate config); skipping")
            report["tasks"][name] = {"L0": L0, "L0_val": L0v, "rows": [], "degenerate": True}
            model.to("cpu")
            continue

        # selection statistic per tensor (same flattening order as `prod`)
        def stat_tensors(stat):
            if stat == "gv":
                return {k: -(g[k] * v[k]).clamp(max=0) for k in theta}
            if stat == "g":
                return {k: g[k].abs() for k in theta}
            if stat == "gnorm":
                out = {}
                for k in theta:
                    rms = float(torch.sqrt((g[k] ** 2).mean()))
                    out[k] = g[k].abs() / (rms if rms > 0 else 1.0)
                return out
            raise ValueError(stat)

        rows = []
        negmask = {k: (g[k] * v[k]) < 0 for k in theta}
        for stat in args.stats:
            sc = stat_tensors(stat)
            scores = torch.cat([sc[k].reshape(-1) for k in theta])[prod < 0]
            for lo, hi, lbl in BUCKETS:
                k_lo = max(1, int(round(lo * n_neg)))
                k_hi = max(1, int(round(hi * n_neg)))
                hi_thr = float(torch.kthvalue(scores, n_neg - k_lo + 1).values) \
                    if k_lo < n_neg else float(scores.max())
                lo_thr = float(torch.kthvalue(scores, n_neg - k_hi + 1).values)
                masks = {k: negmask[k] & (sc[k] >= lo_thr) & (sc[k] <= hi_thr)
                         for k in theta}
                nsel = sum(int(m.sum()) for m in masks.values())
                pred_unit = float(sum((g[k] * v[k] * masks[k]).sum() for k in theta))
                for a in args.alphas:
                    with torch.no_grad():
                        pert = {k: theta[k] + a * v[k] * masks[k] for k in theta}
                    L1, L1v = losses(model, info, pert)
                    rows.append({"stat": stat, "bucket": lbl, "alpha": a,
                                 "n_coords": nsel,
                                 "pred_dloss": a * pred_unit, "meas_dloss": L1 - L0,
                                 "meas_dloss_val": (L1v - L0v) if L0v is not None else None,
                                 "sign_correct": (L1 - L0) < 0,
                                 "sign_correct_val": ((L1v - L0v) < 0)
                                                     if L0v is not None else None})
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
            L1, L1v = losses(model, info, pert)
            rows.append({"stat": "control", "bucket": "POS top 1% (control)", "alpha": a,
                         "n_coords": sum(int(m.sum()) for m in masks.values()),
                         "pred_dloss": a * pred_unit, "meas_dloss": L1 - L0,
                         "meas_dloss_val": (L1v - L0v) if L0v is not None else None,
                         "sign_correct": (L1 - L0) > 0,
                         "sign_correct_val": ((L1v - L0v) > 0) if L0v is not None else None})
        report["tasks"][name] = {"L0": L0, "L0_val": L0v, "rows": rows}
        for r in rows:
            mv = r.get("meas_dloss_val")
            _log(f"  [{name}] {r['stat']:<7} {r['bucket']:<20} a={r['alpha']:<5} "
                 f"pred={r['pred_dloss']:+.4f} meas={r['meas_dloss']:+.4f}"
                 + (f" heldout={mv:+.4f}" if mv is not None else "")
                 + f" {'OK' if r['sign_correct'] else 'X'}")
        model.to("cpu")
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # summary: sign-accuracy per bucket across tasks
    # ---- summary: the generalization gap is the readout ----------------------
    # pred/meas are both on the buffer that produced g, so meas/pred only tests
    # linearity. What tests the NOISE hypothesis is heldout/train: a bucket whose
    # coordinates carry real signal transfers to disjoint data (ratio ~ 1); a
    # bucket that is sampling noise fits the buffer and does nothing (ratio ~ 0),
    # or hurts (< 0).
    from collections import defaultdict
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0, 0])  # pred, train, held, n, ok_t, ok_v
    for t, d in report["tasks"].items():
        for r in d["rows"]:
            k = (r["stat"], r["bucket"], r["alpha"])
            e = agg[k]
            e[0] += r["pred_dloss"]; e[1] += r["meas_dloss"]
            e[2] += (r.get("meas_dloss_val") or 0.0)
            e[3] += r["n_coords"]
            e[4] += bool(r["sign_correct"]); e[5] += bool(r.get("sign_correct_val"))
    order = ["top 0.1%", "0.1-1%", "1-5%", "5-20%", "20-50%", "bottom 50%",
             "POS top 1% (control)"]
    for a in args.alphas:
        print(f"\n=== alpha={a}: summed over tasks "
              f"(gradient from the n={args.n_probe} train buffer) ===")
        print(f"{'stat':<7} {'bucket':<22} {'coords':>13} {'pred':>8} {'train':>8} "
              f"{'heldout':>8} {'meas/pred':>9} {'HELD/TRAIN':>10} {'sign ok (t/v)':>14}")
        for (s, b, aa), e in sorted(agg.items(),
                                    key=lambda kv: (kv[0][0], order.index(kv[0][1]))):
            if aa != a:
                continue
            ratio_lin = e[1] / e[0] if e[0] else float("nan")
            ratio_gen = e[2] / e[1] if e[1] else float("nan")
            print(f"{s:<7} {b:<22} {e[3]:>13,} {e[0]:>8.3f} {e[1]:>8.3f} {e[2]:>8.3f} "
                  f"{ratio_lin:>9.2f} {ratio_gen:>10.2f} {str(e[4])+'/'+str(e[5]):>14}")
    print("\nHELD/TRAIN ~1 => the bucket's coordinates carry signal that generalizes;"
          "\n           ~0 => they only fit the replay buffer (the noise floor thresholding targets).")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
