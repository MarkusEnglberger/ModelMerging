#!/usr/bin/env python
"""Merge baselines vs. APR refinement on top of each merge init.

All families are built from the SAME loaded experts / replay buffers / heads and
scored with the same NormRet, so the only thing that differs is how theta^(0) is
constructed and what (if anything) refines it.

  Checkpoint-only (data-free)
    merge:TA@l<lam>            theta_0 + sum_i lam tau_i                (task arithmetic)
    merge:TIES@d,l             trim -> sign-elect -> disjoint mean      (Yadav 2023)
    merge:DARE@d,s             drop+rescale -> task arithmetic          (CONTROL, Yu 2023)
    merge:DARETIES@dd,t,l      drop+rescale -> TIES                     (Yu 2023 x Yadav 2023)
    merge:BC@d,o,l             magnitude-band mask -> task arithmetic   (Breadcrumbs, Davari 2023)
  Data-dependent, label-free
    merge:ADA-{task,layer}     unlabeled test-time entropy minimisation (AdaMerging, Yang 2024)
  Labeled replay
    apr:from=<init>@lr         Algorithm 1 started from that init

Two questions this answers:
  1. Does APR beat the best merge that uses no labels at all?
  2. Does APR *stack* on a better init, or has the better merge already captured
     what the gate would fix?

Note on the lr grid: APR's step and its trust region both scale with the
expert distance |v_i| = |theta_i - theta_init|, which differs per init, so the
optimal lr is init-dependent. Every init therefore gets the SAME lr grid and we
report best-per-init (equal tuning budget). A grid centred on the TA optimum
silently diverges for the other inits -- that bug invalidated the first run.

DARE-alone is kept only as a control: its drop+rescale is expectation-preserving,
so over ~10^8 coordinates it concentrates back onto plain task arithmetic. It
isolates *sign election* as the active ingredient of TIES.
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_all
from apr.merge_methods import (ties_combined_tau, ties_merge, dare_ta_merge,
                               dare_ties_merge, breadcrumbs_merge)
from apr.taskvec import task_arithmetic_merge
from apr.adamerging import adamerging
from apr.replay_baselines import make_replay_objective
from apr.models import pd_sub, pd_global_norm


def eval_merge(ctx, state):
    scores = ctx.eval_encoder(state)
    nr = ctx.normret(scores)
    return scores, nr, aggregate_all(scores, nr, ctx.base_scores, ctx.expert_scores)


# Which aggregate drives best-per-family selection and the grid-edge warning.
# Set from --select_by in main(); "mean_normret" preserves the pre-20-task
# behaviour, "mean_acc" is the right choice when some task has a degenerate
# expert-minus-base gap (see metrics.DEGENERATE_GAP).
SELECT_BY = "mean_normret"


def record(report, name, scores, nr, ag, disp, **extra):
    report["methods"][name] = {"scores": scores, "normret": nr, "aggregate": ag,
                               "displacement": disp, **extra}
    nd = (f" nr*={ag['mean_normret_nondeg']:.3f}"
          if "mean_normret_nondeg" in ag else "")
    _log(f"  -> {name}: acc={ag['mean_acc']:.4f}/{ag['worst_acc']:.4f} "
         f"nr={ag['mean_normret']:.3f}/{ag['worst_normret']:.3f}{nd} "
         f"disp={disp:.3f}")
    return ag[SELECT_BY]


class Best:
    """Track the best-scoring cell of one family (by the SELECT_BY aggregate)."""

    def __init__(self, label):
        self.label, self.mean, self.name, self.state = label, None, None, None

    def offer(self, mean, name, state):
        if self.mean is None or mean > self.mean:
            self.mean, self.name, self.state = mean, name, state

    def __bool__(self):
        return self.state is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, default=None)
    ap.add_argument("--probe_seed", type=int, default=None,
                    help="override the replay-buffer sampling seed (multi-seed "
                         "replication varies THIS; cfg.seed / refine order stay fixed)")
    # --- checkpoint-only grids ---
    ap.add_argument("--ta_lams", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--ties_densities", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--ties_lams", type=float, nargs="*", default=[0.8, 1.0])
    ap.add_argument("--dare_densities", type=float, nargs="*", default=[0.3])
    ap.add_argument("--dare_seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--dt_drops", type=float, nargs="*", default=[0.5])
    ap.add_argument("--dt_trims", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--dt_lams", type=float, nargs="*", default=[0.8, 1.0])
    ap.add_argument("--dt_seed", type=int, default=0)
    ap.add_argument("--bc_densities", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--bc_outliers", type=float, nargs="*", default=[0.01, 0.05])
    ap.add_argument("--bc_lams", type=float, nargs="*", default=[0.4])
    # --- Localize-and-Stitch, data-free magnitude variant ---
    ap.add_argument("--ls_fracs", type=float, nargs="*", default=[0.05, 0.1],
                    help="per-task top-|tau| mask fraction")
    ap.add_argument("--ls_lams", type=float, nargs="*", default=[1.0],
                    help="stitch scale (paper uses 1.0)")
    # --- AdaMerging (label-free, data-dependent) ---
    ap.add_argument("--ada_variants", nargs="*", default=["task", "layer"],
                    choices=["task", "layer"])
    ap.add_argument("--ada_steps", type=int, default=300)
    ap.add_argument("--ada_lr", type=float, default=1e-3)
    ap.add_argument("--ada_bs", type=int, default=16)
    ap.add_argument("--ada_init_lam", type=float, default=0.3)
    # AdaMerging holds one live DataLoader per task; with the CLIP config's
    # eval_num_workers=8 that would fork 8*8=64 workers. Keep it small.
    ap.add_argument("--ada_workers", type=int, default=0)
    # eval_ds = standard transductive AdaMerging (unlimited unlabeled test inputs).
    # probe_buffer = matched to APR's replay budget (same n_probe inputs, labels
    # stripped), which is what the proposal's protocol actually asks for.
    ap.add_argument("--ada_data", nargs="*", default=["eval_ds"],
                    choices=["eval_ds", "probe_buffer"])
    # --- APR on top ---
    ap.add_argument("--apr_schedules", nargs="*", default=["constant"],
                    choices=["constant", "cosine", "linear"],
                    help="lr schedules for the APR arms; non-constant ones use "
                         "--apr_order (mirrors the GD arms)")
    ap.add_argument("--apr_order", choices=["fixed", "cyclic", "random"],
                    default="random",
                    help="task order for non-constant-schedule APR arms")
    ap.add_argument("--apr_gate_modes", nargs="*", default=[],
                    choices=["coordinate", "none", "topk", "topk_g"],
                    help="gate variants for the APR arms (default: config's mode). "
                         "topk_g = sign(g*v)<0 AND |g| in the per-tensor top "
                         "topk_frac (significance on the noisy factor only)")
    ap.add_argument("--topk_fracs", type=float, nargs="*", default=[0.05],
                    help="kept fraction(s) for topk/topk_g gate arms")
    ap.add_argument("--gd_lrs", type=float, nargs="*", default=[],
                    help="also run ORDINARY GD (plain -g, no gate/anchor/clip) from "
                         "each --refine_from init. GD's lr optimum has only ever "
                         "been tuned at the TA init, so give it a WIDE grid.")
    ap.add_argument("--gd_steps", type=int, nargs="*", default=[],
                    help="sweep counts for GD (default: --steps). >5 tests the "
                         "'GD just needs more sweeps' objection.")
    ap.add_argument("--gd_schedules", nargs="*", default=["constant"],
                    choices=["constant", "cosine", "linear"])
    ap.add_argument("--gd_lr_min_frac", type=float, default=0.05)
    ap.add_argument("--gd_order", choices=["fixed", "cyclic", "random"],
                    default="random", help="task order for annealed GD runs")
    ap.add_argument("--nogate_lrs", type=float, nargs="*", default=[],
                    help="also run the UNGATED anchored control from each init")
    ap.add_argument("--refine_from", nargs="*",
                    default=["ta", "ties", "dareties", "bc", "ls", "ada"],
                    choices=["ta", "ties", "dare", "dareties", "bc", "ls", "ada"])
    ap.add_argument("--apr_lrs", type=float, nargs="*", default=[2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=None, help="APR sweeps (default: config)")
    ap.add_argument("--skip_families", nargs="*", default=[],
                    choices=["ta", "ties", "dare", "dareties", "bc", "ls", "ada"])
    ap.add_argument("--select_by", choices=["mean_normret", "mean_acc"],
                    default="mean_normret",
                    help="aggregate used to pick best-per-family / best APR lr. "
                         "Use mean_acc when a task has a degenerate expert-base "
                         "gap (20-task CLIP suite: stl10, food101).")
    ap.add_argument("--out", default="results/compare/merge_baselines.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.n_probe is not None:
        cfg.data.n_probe = args.n_probe
    if args.probe_seed is not None:
        cfg.data.probe_seed = args.probe_seed
    ctx = MergeContext.build(cfg)
    steps = args.steps if args.steps is not None else cfg.refine.steps
    names = cfg.task_names
    # replay objective (mean loss on the same n=64 labeled buffers used for the
    # refinement gradients) recorded per refine cell so hyperparameters can be
    # selected WITHOUT the test set (replay-honest selection, cf. replay_baselines).
    objective, _ = make_replay_objective(ctx)

    report = {"config": cfg.to_dict(), "tasks": names, "steps": steps,
              "base": ctx.base_scores, "expert": ctx.expert_scores,
              "grids": vars(args), "methods": {}}
    global SELECT_BY
    SELECT_BY = args.select_by
    _log(f"[select] best-per-family by {SELECT_BY}")
    skip = set(args.skip_families)

    # ---- task arithmetic, lambda swept (the other families tune theirs, so must this) ----
    ta = Best("TA")
    if "ta" not in skip:
        for lam in args.ta_lams:
            state = task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors,
                                          {n: lam for n in names})
            nm = f"merge:TA@l{lam:g}"
            s, nr, ag = eval_merge(ctx, state)
            ta.offer(record(report, nm, s, nr, ag,
                            pd_global_norm(pd_sub(state, ctx.merged0)), lam=lam), nm, state)

    # ---- TIES (combined tau reused across the lambda grid) ----
    ties = Best("TIES")
    if "ties" not in skip:
        for d in args.ties_densities:
            _log(f"\n[TIES] density={d:g}")
            combined = ties_combined_tau(ctx.task_vectors, density=d)
            for lam in args.ties_lams:
                state = ties_merge(ctx.base_encoder, ctx.task_vectors, lam=lam,
                                   density=d, combined=combined)
                nm = f"merge:TIES@d{d:g},l{lam:g}"
                s, nr, ag = eval_merge(ctx, state)
                ties.offer(record(report, nm, s, nr, ag,
                                  pd_global_norm(pd_sub(state, ctx.merged0)),
                                  density=d, lam=lam), nm, state)

    # ---- DARE alone (control: expectation-preserving => ~= TA) ----
    dare = Best("DARE")
    if "dare" not in skip:
        for d in args.dare_densities:
            for sd in args.dare_seeds:
                state = dare_ta_merge(ctx.base_encoder, ctx.task_vectors, ctx.lambdas,
                                      density=d, seed=sd)
                nm = f"merge:DARE@d{d:g},s{sd}"
                s, nr, ag = eval_merge(ctx, state)
                dare.offer(record(report, nm, s, nr, ag,
                                  pd_global_norm(pd_sub(state, ctx.merged0)),
                                  density=d, seed=sd, note="control"), nm, state)

    # ---- DARE-TIES ----
    dt = Best("DARETIES")
    if "dareties" not in skip:
        for dd in args.dt_drops:
            for tr in args.dt_trims:
                for lam in args.dt_lams:
                    state = dare_ties_merge(ctx.base_encoder, ctx.task_vectors, lam=lam,
                                            drop_density=dd, trim_density=tr,
                                            seed=args.dt_seed)
                    nm = f"merge:DARETIES@dd{dd:g},t{tr:g},l{lam:g}"
                    s, nr, ag = eval_merge(ctx, state)
                    dt.offer(record(report, nm, s, nr, ag,
                                    pd_global_norm(pd_sub(state, ctx.merged0)),
                                    drop=dd, trim=tr, lam=lam), nm, state)

    # ---- Model Breadcrumbs ----
    bc = Best("BC")
    if "bc" not in skip:
        for d in args.bc_densities:
            for o in args.bc_outliers:
                for lam in args.bc_lams:
                    state = breadcrumbs_merge(ctx.base_encoder, ctx.task_vectors,
                                              {n: lam for n in names},
                                              density=d, outlier_frac=o)
                    nm = f"merge:BC@d{d:g},o{o:g},l{lam:g}"
                    s, nr, ag = eval_merge(ctx, state)
                    bc.offer(record(report, nm, s, nr, ag,
                                    pd_global_norm(pd_sub(state, ctx.merged0)),
                                    density=d, outlier=o, lam=lam), nm, state)

    # ---- Localize-and-Stitch, data-free magnitude variant ----
    # (the learned-mask variant needs per-task full-d float masks -- ~12 GB each at
    # 3B -- and lost to the magnitude variant on every track; ls_sweep.py keeps it
    # for the encoder suites)
    ls = Best("LS")
    if "ls" not in skip:
        from apr.localize_stitch import dataless_masks, stitch
        for frac in args.ls_fracs:
            masks = dataless_masks(ctx.task_vectors, frac)
            for lam in args.ls_lams:
                state, info = stitch(ctx.base_encoder, ctx.task_vectors, masks, lam=lam)
                nm = f"merge:LS-mag@f{frac:g},l{lam:g}"
                s, nr, ag = eval_merge(ctx, state)
                ls.offer(record(report, nm, s, nr, ag,
                                pd_global_norm(pd_sub(state, ctx.merged0)),
                                sparsity=frac, lam=lam, stitch=info), nm, state)
            del masks

    # ---- AdaMerging (label-free, uses unlabeled eval inputs: transductive) ----
    ada = Best("ADA")
    if "ada" not in skip:
        for dkey in args.ada_data:
            for variant in args.ada_variants:
                _log(f"\n[AdaMerging-{variant}/{dkey}] entropy minimisation "
                     f"({args.ada_steps} steps, lr={args.ada_lr})")
                state, info = adamerging(
                    ctx.base_encoder, ctx.task_vectors, ctx.per_task, names, ctx.device,
                    layerwise=(variant == "layer"), steps=args.ada_steps, lr=args.ada_lr,
                    batch_size=args.ada_bs, init_lam=args.ada_init_lam, seed=cfg.seed,
                    num_workers=args.ada_workers, data_key=dkey, logger=_log)
                tag = variant if dkey == "eval_ds" else f"{variant}-matched"
                nm = f"merge:ADA-{tag}"
                s, nr, ag = eval_merge(ctx, state)
                ada.offer(record(report, nm, s, nr, ag,
                                 pd_global_norm(pd_sub(state, ctx.merged0)),
                                 adamerging=info), nm, state)

    fams = {"ta": ta, "ties": ties, "dare": dare, "dareties": dt, "bc": bc, "ls": ls, "ada": ada}
    report["best_per_family"] = {k: v.name for k, v in fams.items() if v}
    _log("\n[best-per-family] " + json.dumps(report["best_per_family"], indent=2))

    # ---- APR on top of each init (same lr grid for every init = equal budget) ----
    for key in args.refine_from:
        fam = fams.get(key)
        if not fam:
            _log(f"[apr] skip from={key} (family not run)")
            continue
        init_label = fam.name.split(":", 1)[1]
        best_mean = None
        gate_modes = args.apr_gate_modes or [cfg.refine.gate_mode]
        for gmode in gate_modes:
         fracs = args.topk_fracs if gmode.startswith("topk") else [None]
         for frac in fracs:
          for sched in args.apr_schedules:
           for lr in args.apr_lrs:
            rc = dataclasses.replace(
                cfg.refine, steps=steps, lr=lr, gate_mode=gmode,
                topk_frac=(frac if frac is not None else cfg.refine.topk_frac),
                lr_schedule=sched, lr_min_frac=args.gd_lr_min_frac,
                order=args.apr_order if sched != "constant" else cfg.refine.order)
            _log(f"\n===== APR from {init_label} @ lr{lr:g} S={steps} {sched} "
                 f"({rc.gate_mode}"
                 f"{'' if frac is None else f'@{frac:g}'}"
                 f"/clip={rc.clip_mode}/g{rc.clip_frac:g}/{rc.order}) =====")
            refined, history = ctx.run_refine_from(fam.state, rc, seed=cfg.seed)
            s, nr, ag = eval_merge(ctx, refined)
            gd = (sum(h["gate_density"] for h in history) / len(history)) if history else None
            nm = (f"apr:from={key}@lr{lr:g}"
                  + ("" if sched == "constant" else f",{sched}")
                  + ("" if gmode == cfg.refine.gate_mode and frac is None
                     else f",{gmode}" + ("" if frac is None else f"{frac:g}")))
            m = record(report, nm, s, nr, ag,
                       pd_global_norm(pd_sub(refined, fam.state)),
                       init=init_label, lr=lr, schedule=sched, gate_mode=gmode,
                       topk_frac=frac,
                       gate_density=gd, replay_obj=objective(refined))
            if best_mean is None or m > best_mean:
                best_mean = m
                report[f"best_apr_from_{key}"] = nm
        # ---- controls from the SAME init: ungated anchor, and ordinary GD ----
        # Both isolate a different part of the update: nogate removes the gate but
        # keeps the anchored/distance-scaled/clipped step; GD removes all of it.
        for lr in args.nogate_lrs:
            rc = dataclasses.replace(cfg.refine, steps=steps, lr=lr, gate_mode="none")
            _log(f"\n===== nogate from {init_label} @ lr{lr:g} =====")
            refined, _ = ctx.run_refine_from(fam.state, rc, seed=cfg.seed)
            s, nr, ag = eval_merge(ctx, refined)
            record(report, f"nogate:from={key}@lr{lr:g}", s, nr, ag,
                   pd_global_norm(pd_sub(refined, fam.state)), init=init_label, lr=lr,
                   replay_obj=objective(refined))
        for sched in args.gd_schedules:
            for S in (args.gd_steps or [steps]):
                for lr in args.gd_lrs:
                    rc = dataclasses.replace(
                        cfg.refine, steps=S, lr=lr, gate_mode="none",
                        update_mode="grad", clip_mode="none",
                        lr_schedule=sched, lr_min_frac=args.gd_lr_min_frac,
                        order=args.gd_order if sched != "constant" else cfg.refine.order)
                    tag = f"gd:from={key}@lr{lr:g},S{S},{sched}"
                    _log(f"\n===== ordinary GD from {init_label} "
                         f"@ lr{lr:g} S={S} {sched} =====")
                    refined, _ = ctx.run_refine_from(fam.state, rc, seed=cfg.seed)
                    s, nr, ag = eval_merge(ctx, refined)
                    record(report, tag, s, nr, ag,
                           pd_global_norm(pd_sub(refined, fam.state)),
                           init=init_label, lr=lr, gd_steps=S, schedule=sched,
                           replay_obj=objective(refined))

        # a peak at the grid edge means the optimum was not bracketed
        if best_mean is not None and args.apr_lrs:
            best_lr = float(report[f"best_apr_from_{key}"].split("@lr")[1].split(",")[0])
            if best_lr == min(args.apr_lrs):
                _log(f"[warn] APR from {key} peaked at the LOW EDGE (lr={best_lr:g}) "
                     f"-- true optimum may be lower; result is a lower bound.")
            elif best_lr == max(args.apr_lrs):
                _log(f"[warn] APR from {key} peaked at the HIGH EDGE (lr={best_lr:g}).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 108)
    print(f"{'method':<30} " + " ".join(f"{t[:7]:>7}" for t in names) +
          f" | {'mAcc':>6} {'wAcc':>6} {'mNR':>7} {'wNR':>7} {'||disp||':>8}")
    print("-" * 108)
    for n, d in sorted(report["methods"].items(),
                       key=lambda kv: -kv[1]["aggregate"][SELECT_BY]):
        a = d["aggregate"]
        print(f"{n:<30} " + " ".join(f"{d['normret'][t]:>7.3f}" for t in names) +
              f" | {a['mean_acc']:>6.4f} {a['worst_acc']:>6.4f}"
              f" {a['mean_normret']:>7.3f} {a['worst_normret']:>7.3f} {d['displacement']:>8.3f}")
    print("=" * 108)
    print("(mAcc/wAcc: absolute mean/worst accuracy. mNR/wNR: normalized retention, "
          "0=pretrained floor 1=expert ceiling -- UNSTABLE for tasks whose expert "
          "barely beats base; see aggregate.degenerate_tasks.)")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
