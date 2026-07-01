#!/usr/bin/env python
"""Diagnostic: is the attribution gate mask the SAME coordinates every sweep?

Re-runs the proposed sequential refinement but, for each (sweep, task), records:
  - density    : fraction of coordinates kept (g*v < 0)
  - agree_prev : fraction of coords with the SAME on/off gate value as the same
                 task's mask in the previous sweep  (1.0 => mask frozen)
  - jaccard    : |kept_now & kept_prev| / |kept_now | kept_prev|  for the kept set
  - disp       : ||theta - theta_merge||  (how far the model has moved so far)

Run at a couple of learning rates to see whether larger steps make the mask churn.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import (load_encoder_state, pd_to, pd_clone, pd_sub, pd_global_norm)


def run_lr(ctx, lr, gamma, steps, order):
    device = ctx.device
    theta = pd_to(pd_clone(ctx.merged0), device)
    names = list(theta.keys())
    prev_mask = {}  # task -> dict[name->bool tensor]
    rows = []
    for s in range(steps):
        for task in order:
            h = next(hh for hh in ctx.handles if hh.name == task)
            h.model.to(device)
            load_encoder_state(h.model, theta)
            g = h.grad_fn()
            expert = pd_to(h.expert_encoder, device)

            kept = total = same = inter = union = 0
            cur_mask = {}
            for n in names:
                v = expert[n] - theta[n]
                m = (g[n] * v < 0)
                cur_mask[n] = m
                # apply the proposed clipped update immediately
                u = torch.clamp(-g[n] * v.abs() * m.float(),
                                min=-gamma * v.abs(), max=gamma * v.abs())
                theta[n].add_(u, alpha=lr)
                kept += int(m.sum()); total += m.numel()
                if task in prev_mask:
                    pm = prev_mask[task][n]
                    same += int((m == pm).sum())
                    inter += int((m & pm).sum())
                    union += int((m | pm).sum())
            prev_mask[task] = cur_mask
            h.model.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            disp = pd_global_norm(pd_sub(theta, pd_to(ctx.merged0, device)))
            row = {"lr": lr, "sweep": s, "task": task,
                   "density": kept / total,
                   "agree_prev": (same / total) if union or s > 0 else None,
                   "jaccard": (inter / union) if union else None,
                   "disp": disp}
            rows.append(row)
            ap = row["agree_prev"]; jc = row["jaccard"]
            _log(f"lr={lr:<4g} sweep {s} {task:5s} density={row['density']:.4f} "
                 f"agree_prev={'n/a' if ap is None else f'{ap:.4f}'} "
                 f"jaccard={'n/a' if jc is None else f'{jc:.4f}'} disp={disp:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--lrs", type=float, nargs="*", default=[1.0, 8.0])
    ap.add_argument("--out", default="results/compare/gate_stability.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)
    order = cfg.task_names

    out = {"config": cfg.to_dict(), "rows": []}
    for lr in args.lrs:
        _log(f"\n===== lr={lr} =====")
        out["rows"].extend(run_lr(ctx, lr, args.gamma, args.steps, order))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
