"""Is a bad draw a SELECTION failure or a bad BUFFER?

Under protocol v2 the GLUE-8 draws split cleanly: those whose selected cell has
a large movement budget eta*S score ~0.31-0.37 NormRet, those whose cell barely
moves score ~0.02-0.06. On the low-scoring draws the held-out objective surface
is flat and mostly ABOVE the initialization, so the argmin is an isolated cell
with no supporting neighbourhood -- consistent with selection latching onto
noise rather than with refinement being unable to help.

That leaves two hypotheses, which this script separates by PINNING a cell
instead of selecting it:

  (a) SELECTION failure -- the buffer supports refinement, but 8/16 held-out
      examples cannot see it. Then a good cell taken from another draw, refit
      on this draw's budget, should score well here too.
  (b) BUFFER failure -- this draw's examples genuinely do not support
      refinement. Then the same cell scores badly here as well.

Every cell is refit on the FULL budget from the pretrained model, exactly as
cv_protocol's refit does (same buffer draw, same seed, same random task order),
so the only difference from the reported run is that (eta, S) is given rather
than chosen. The evaluation split is read once per pinned cell.

Usage:
  python scripts/pin_cell_probe.py --config configs/glue8.yaml --budget 16 \
      --cells 101:4:50 102:4:50 100:16:5 100:0.5:100 \
      --out results/compare/pin_glue8_B16.json

  --cells takes SEED:ETA:S triples (seed 100+d is protocol draw d).
"""

import argparse
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from apr.config import ExperimentConfig                                  # noqa: E402
from apr.pipeline import MergeContext, _log                              # noqa: E402
from apr.models import pd_sub, pd_global_norm                            # noqa: E402
from cv_protocol import (ARM_KW, draw_budget_buffers,                    # noqa: E402
                         set_construction_buffer, eval_and_record,
                         clip_summary)
from apr.merge_methods import (ties_combined_tau, ties_merge,            # noqa: E402
                               dare_ties_merge, breadcrumbs_merge)
from apr.taskvec import task_arithmetic_merge                            # noqa: E402


def build_init(ctx, family: str, spec: str):
    """The state a cell refines from, built exactly as cv_protocol builds it.

    ``family`` is cv_protocol's --init name; ``spec`` is the hyperparameter
    string it records after '@' for the merge selected on that draw (TA
    'l0.05'; TIES 'd0.1,l0.8'; DARE-TIES 'dd0.5,t0.1,l0.8'; Breadcrumbs
    'd0.1,o0.05,l0.4'), so a composition cell can be pinned from the same
    merge the reported run refined.
    """
    if family == "pretrained":
        return ctx.base_encoder
    kv = {}
    for tok in spec.split(","):
        k = tok.rstrip("0123456789.-")
        kv[k] = float(tok[len(k):])
    tv, base, names = ctx.task_vectors, ctx.base_encoder, ctx.task_names
    if family == "ta":
        return task_arithmetic_merge(base, tv, {n: kv["l"] for n in names})
    if family == "ties":
        return ties_merge(base, tv, lam=kv["l"], density=kv["d"],
                          combined=ties_combined_tau(tv, density=kv["d"]))
    if family == "dareties":
        return dare_ties_merge(base, tv, lam=kv["l"], drop_density=kv["dd"],
                               trim_density=kv["t"], seed=0)
    if family == "bc":
        return breadcrumbs_merge(base, tv, {n: kv["l"] for n in names},
                                 density=kv["d"], outlier_frac=kv["o"])
    raise ValueError(f"unknown init family {family!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--cells", nargs="+", required=True,
                    help="SEED:ETA:S triples, e.g. 101:4:50; with --init other "
                         "than pretrained, SEED:ETA:S:SPEC quadruples where SPEC "
                         "is the selected merge's hyperparameter string, e.g. "
                         "104:8:20:d0.1,l0.8")
    ap.add_argument("--arm", default="apr", choices=list(ARM_KW))
    ap.add_argument("--init", default="pretrained",
                    choices=["pretrained", "ta", "ties", "dareties", "bc"],
                    help="what the cell refines from (cv_protocol's --init)")
    ap.add_argument("--order", default="random")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)

    report = {"config": args.config, "budget": args.budget, "arm": args.arm,
              "init": args.init,
              "note": "cells PINNED, not selected; refit on the full budget "
                      f"from {args.init}", "cells": {}}

    for spec in args.cells:
        parts = spec.split(":")
        seed, lr, S = int(parts[0]), float(parts[1]), int(parts[2])
        init_spec = parts[3] if len(parts) > 3 else ""
        if args.init != "pretrained" and not init_spec:
            raise SystemExit(f"--init {args.init} needs SEED:ETA:S:SPEC, got {spec}")
        init_state = build_init(ctx, args.init, init_spec)
        tag = f"seed{seed},lr{lr:g},S{S}" + (f",from={args.init}@{init_spec}" if init_spec else "")
        _log(f"\n===== PINNED {tag} (draw {seed - 100}) =====")

        # identical to cv_protocol's refit: same buffer draw, whole budget
        budget_bufs = draw_budget_buffers(ctx, args.budget, seed)
        set_construction_buffer(ctx, budget_bufs)
        rc = dataclasses.replace(cfg.refine, steps=S, lr=lr, order=args.order,
                                 lr_schedule="constant", **ARM_KW[args.arm])
        refit, hist = ctx.run_refine_from(init_state, rc, seed=cfg.seed)
        scores, nr, ag = eval_and_record(ctx, refit)
        rec = {"buffer_seed": seed, "lr": lr, "S": S, "eta_times_S": lr * S,
               "init": args.init, "init_spec": init_spec,
               "displacement": pd_global_norm(pd_sub(refit, init_state)),
               "scores": scores, "normret": nr, "aggregate": ag,
               "clip": clip_summary(hist)}
        report["cells"][tag] = rec
        _log(f"[pinned] {tag}: mean_nr={ag['mean_normret']:+.4f} "
             f"mean_acc={ag['mean_acc']:.4f} worst_nr={ag['worst_normret']:+.4f} "
             f"displ={rec['displacement']:.2f}")
        del refit
        if ctx.device.startswith("cuda"):
            torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)

    _log(f"\n[done] -> {args.out}")
    _log(f"{'cell':28s}{'eta*S':>8}{'displ':>8}{'NormRet':>10}{'acc':>8}")
    for k, v in report["cells"].items():
        _log(f"{k:28s}{v['eta_times_S']:8.0f}{v['displacement']:8.2f}"
             f"{v['aggregate']['mean_normret']:+10.4f}{v['aggregate']['mean_acc']:8.4f}")


if __name__ == "__main__":
    main()
