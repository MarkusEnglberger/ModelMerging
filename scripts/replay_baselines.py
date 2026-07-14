#!/usr/bin/env python
"""Matched-data-budget table: hold the labeled data fixed, vary only the method.

Every method below sees EXACTLY the same n_probe labeled examples per task (APR's replay
buffer, drawn from the TRAIN split). This directly tests the proposal's own stated worry:
"the strongest alternative explanation for a positive result is simply that the method
performs supervised post-merge replay fine-tuning."

  Reference          merge:TA                     task-arithmetic merge point (no data)
  Labeled, no grads  labeled:lam-global           uniform lambda tuned on replay loss
                     labeled:lam-pertask          per-task lambda_i, coordinate descent
                     labeled:cocktail             LM-Cocktail loss-weighted coefficients
                     labeled:fisher               diag empirical Fisher from the buffer
  Labeled, head only labeled:head-only            frozen encoder, refit each task head
  Labeled, masks     labeled:ls@sp,g              Localize-and-Stitch (learned masks)
                     labeled:ls-dataless@sp       L&S data-free control (top-k |tau|)
  Labeled, encoder   replay:gd@lr                 ordinary replay gradient descent
                     replay:nogate@lr             ungated distance-scaled (APR ablation)
                     replay:apr@lr                the proposed method

SELECTION PROTOCOL. Hyperparameters are chosen by the REPLAY objective (mean over tasks of
L_t(theta)/L_t(theta_merge)), never by the eval split. We nevertheless record eval for every
cell and report BOTH the replay-selected and the eval-selected (oracle) winner, so the
selection gap is visible -- with n=64 that gap is not negligible, and prior runs in this
repo silently used oracle selection.
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig, RefineConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_retention
from apr.models import pd_sub, pd_global_norm
from apr.replay_baselines import (make_replay_objective, lam_search_global,
                                  lam_search_pertask, cocktail_merge, fisher_merge,
                                  head_only_scores)
from apr.localize_stitch import (learn_sigmoids, masks_from_sigmoids, dataless_masks,
                                 stitch)


def add_cell(ctx, report, name, state, objective, scores=None, disp=None, **extra):
    """Record one method cell: replay objective (selection) + eval scores (reporting)."""
    ro = objective(state) if state is not None else None
    if scores is None:
        scores = ctx.eval_encoder(state)
    if disp is None:
        disp = pd_global_norm(pd_sub(state, ctx.merged0)) if state is not None else 0.0
    nr = ctx.normret(scores)
    ag = aggregate_retention(nr)
    report["methods"][name] = {"scores": scores, "normret": nr, "aggregate": ag,
                               "displacement": disp, "replay_obj": ro, **extra}
    _log(f"  -> {name}: mean={ag['mean_normret']:.3f} worst={ag['worst_normret']:.3f} "
         f"replay_obj={'n/a' if ro is None else format(ro, '.4f')} disp={disp:.3f}")
    return ag["mean_normret"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5, help="APR/GD/nogate sweeps")
    ap.add_argument("--lam_grid", type=float, nargs="*",
                    default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--cocktail_temp", type=float, default=1.0)
    # Localize-and-Stitch
    ap.add_argument("--ls_sparsities", type=float, nargs="*", default=[0.01, 0.05])
    ap.add_argument("--ls_gamma", type=float, default=1e-4)
    ap.add_argument("--ls_steps", type=int, default=200)
    ap.add_argument("--ls_lr", type=float, default=0.1)
    ap.add_argument("--ls_bs", type=int, default=16)
    ap.add_argument("--ls_average_overlaps", action="store_true")
    # head-only
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--head_epochs", type=int, default=20)
    # encoder-refinement families
    ap.add_argument("--apr_lrs", type=float, nargs="*", default=[2, 4, 8, 16])
    ap.add_argument("--nogate_lrs", type=float, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--gd_lrs", type=float, nargs="*", default=[5e-5, 1e-4, 5e-4, 1e-3])
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["lam", "cocktail", "fisher", "head", "ls", "gd", "nogate", "apr"])
    ap.add_argument("--out", default="results/compare/replay_baselines.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    ctx = MergeContext.build(cfg)
    names = cfg.task_names
    skip = set(args.skip)

    objective, ref = make_replay_objective(ctx)
    _log(f"\n[replay] reference losses at the merge point: "
         f"{ {k: round(v, 4) for k, v in ref.items()} }")

    report = {"config": cfg.to_dict(), "tasks": names, "grids": vars(args),
              "base": ctx.base_scores, "expert": ctx.expert_scores,
              "replay_ref_losses": ref, "methods": {}}

    # ---- reference: the merge point itself (no data used) ----
    add_cell(ctx, report, "merge:TA", ctx.merged0, objective,
             scores=ctx.merge_scores, disp=0.0)

    # ---- labeled, no gradient descent on the encoder ----
    if "lam" not in skip:
        _log("\n[lam-global] uniform lambda tuned on replay loss")
        st, info = lam_search_global(ctx, args.lam_grid, objective)
        add_cell(ctx, report, "labeled:lam-global", st, objective, **info)

        _log("\n[lam-pertask] coordinate descent on per-task lambda_i")
        st, info = lam_search_pertask(ctx, args.lam_grid, objective, passes=2,
                                      logger=_log)
        add_cell(ctx, report, "labeled:lam-pertask", st, objective, **info)

    if "cocktail" not in skip:
        _log("\n[cocktail] LM-Cocktail style loss-weighted coefficients")
        st, info = cocktail_merge(ctx, args.lam_grid, objective,
                                  temperature=args.cocktail_temp, logger=_log)
        add_cell(ctx, report, "labeled:cocktail", st, objective, **info)

    if "fisher" not in skip:
        _log("\n[fisher] diagonal empirical Fisher from the replay buffer")
        st, info = fisher_merge(ctx, args.lam_grid, objective, logger=_log)
        add_cell(ctx, report, "labeled:fisher", st, objective, **info)

    # ---- labeled, head only (encoder frozen at the merge point) ----
    if "head" not in skip:
        _log("\n[head-only] frozen merged encoder, refit each task head")
        scores, hinfo = head_only_scores(ctx, ctx.merged0, lr=args.head_lr,
                                         epochs=args.head_epochs, logger=_log)
        add_cell(ctx, report, "labeled:head-only", ctx.merged0, objective,
                 scores=scores, disp=0.0, head_only=True, head_info=hinfo)

    # ---- labeled, learned sparse masks (Localize-and-Stitch) ----
    if "ls" not in skip:
        _log(f"\n[localize-and-stitch] learning masks (gamma={args.ls_gamma}, "
             f"{args.ls_steps} steps) -- one training run serves all sparsities")
        sigs = learn_sigmoids(ctx, steps=args.ls_steps, lr=args.ls_lr,
                              gamma=args.ls_gamma, batch_size=args.ls_bs, logger=_log)
        for sp in args.ls_sparsities:
            masks = masks_from_sigmoids(sigs, sp)
            st, sinfo = stitch(ctx.base_encoder, ctx.task_vectors, masks, lam=1.0,
                               average_overlaps=args.ls_average_overlaps)
            add_cell(ctx, report, f"labeled:ls@sp{sp:g}", st, objective,
                     sparsity=sp, gamma=args.ls_gamma, **sinfo)
        del sigs
        # data-free control: top-k% magnitude masks
        for sp in args.ls_sparsities:
            masks = dataless_masks(ctx.task_vectors, sp)
            st, sinfo = stitch(ctx.base_encoder, ctx.task_vectors, masks, lam=1.0,
                               average_overlaps=args.ls_average_overlaps)
            add_cell(ctx, report, f"labeled:ls-dataless@sp{sp:g}", st, objective,
                     sparsity=sp, dataless=True, **sinfo)

    # ---- labeled, encoder refinement (same buffer, same #sweeps) ----
    fam = []
    if "gd" not in skip:
        fam += [(f"replay:gd@lr{lr:g}",
                 RefineConfig(steps=args.steps, lr=lr, gate_mode="none",
                              update_mode="grad", clip_mode="none")) for lr in args.gd_lrs]
    if "nogate" not in skip:
        fam += [(f"replay:nogate@lr{lr:g}",
                 dataclasses.replace(cfg.refine, steps=args.steps, lr=lr,
                                     gate_mode="none")) for lr in args.nogate_lrs]
    if "apr" not in skip:
        fam += [(f"replay:apr@lr{lr:g}",
                 dataclasses.replace(cfg.refine, steps=args.steps, lr=lr))
                for lr in args.apr_lrs]
    for nm, rc in fam:
        _log(f"\n===== {nm} ({rc.update_mode}/{rc.gate_mode}/clip={rc.clip_mode}) =====")
        refined, hist = ctx.run_refine_from(ctx.merged0, rc, seed=cfg.seed)
        gd = (sum(h["gate_density"] for h in hist) / len(hist)) if hist else None
        add_cell(ctx, report, nm, refined, objective, lr=rc.lr, gate_density=gd)

    # ---- selection: replay-chosen vs eval-chosen (oracle) ----
    def best_of(prefix, key):
        cells = [(v, k) for k, v in report["methods"].items() if k.startswith(prefix)]
        if not cells:
            return None
        if key == "replay":
            cells = [(v, k) for v, k in cells if v.get("replay_obj") is not None]
            if not cells:
                return None
            return min(cells, key=lambda vk: vk[0]["replay_obj"])[1]
        return max(cells, key=lambda vk: vk[0]["aggregate"]["mean_normret"])[1]

    sel = {}
    for pre in ["replay:apr@", "replay:gd@", "replay:nogate@", "labeled:ls@",
                "labeled:ls-dataless@"]:
        r, e = best_of(pre, "replay"), best_of(pre, "eval")
        if r or e:
            sel[pre.rstrip("@")] = {"replay_selected": r, "eval_selected": e,
                                    "agree": r == e}
    report["selection"] = sel
    _log("\n[selection] replay-selected vs eval-selected (oracle):\n" +
         json.dumps(sel, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 112)
    print(f"{'method':<30} " + " ".join(f"{t[:7]:>7}" for t in names) +
          f" | {'mean':>6} {'worst':>6} {'replayL':>8}")
    print("-" * 112)
    for n, d in sorted(report["methods"].items(),
                       key=lambda kv: -kv[1]["aggregate"]["mean_normret"]):
        a = d["aggregate"]
        ro = d.get("replay_obj")
        print(f"{n:<30} " + " ".join(f"{d['normret'][t]:>7.3f}" for t in names) +
              f" | {a['mean_normret']:>6.3f} {a['worst_normret']:>6.3f} "
              f"{('n/a' if ro is None else format(ro, '.4f')):>8}")
    print("=" * 112)
    print(f"all methods use the SAME {cfg.data.n_probe} labeled examples/task from train")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
