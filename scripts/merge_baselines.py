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
    merge:DELLA@d,w,l,s        magnitude-sample -> TIES                 (Deep 2024)
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
import hashlib
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.metrics import aggregate_all
from apr.merge_methods import (ties_combined_tau, ties_merge, dare_ta_merge,
                               dare_ties_merge, della_merge, breadcrumbs_merge)
from apr.taskvec import task_arithmetic_merge
from apr.adamerging import adamerging
from apr.regmean import regmean_merge
from apr.replay_baselines import make_replay_objective
from apr.data import sample_replay_buffer
from apr.models import pd_sub, pd_global_norm


# tasks whose retention enters the aggregate (and hence best-cell selection).
# None = all. Set from --retention_tasks: on the MergeBench track the multilingual
# expert does not beat base on the MC metric (denominator ~0, sign unstable), so
# its normret is undefined and would corrupt means and family-best selection.
# Raw scores for excluded tasks are still recorded per cell.
RETENTION_TASKS = None


def eval_merge(ctx, state):
    scores = ctx.eval_encoder(state)
    nr = ctx.normret(scores)
    nr_agg = ({t: nr[t] for t in RETENTION_TASKS} if RETENTION_TASKS else nr)
    return scores, nr, aggregate_all(scores, nr_agg, ctx.base_scores, ctx.expert_scores)


# Which aggregate orders evaluated winners in the final report.
SELECT_BY = "mean_normret"
# Maps a candidate state to its loss on a DISJOINT selection buffer (lower is
# better). Every family and refinement cell is chosen by this, so no
# test information enters hyperparameter selection.
SELECT_OBJ = None


class Best:
    """Track the selection-buffer winner of one family."""

    def __init__(self, label):
        self.label, self.mean, self.name, self.state = label, None, None, None

    def offer(self, name, state, selection_obj):
        score = -selection_obj
        if self.mean is None or score > self.mean:
            self.mean, self.name, self.state = score, name, state

    def __bool__(self):
        return self.state is not None




# ---------------------------------------------------------------- merge cache
# Expensive initializations are deterministic given their inputs, but were
# rebuilt by every job: AdaMerging trains 300 steps, RegMean accumulates Gram
# matrices, Localize-and-Stitch fits masks. The cheap arithmetic merges (TIES,
# DARE, Breadcrumbs, task arithmetic) are NOT cached -- rebuilding them costs
# less than reading them back.
MERGE_CACHE_DIR = "results/merge_cache"
CACHEABLE = ("ada", "regmean", "ls")


def _merge_cache_path(cfg, family, params, extra=None):
    """Everything that determines the merged weights.

    ``extra`` carries method-specific inputs that are easy to forget and unsafe
    to omit. AdaMerging is the cautionary case: it is a TEST-TIME method whose
    unlabeled inputs may come from the evaluation split (transductive) or from
    the matched replay budget, and those give different states -- so the data
    source, the probe budget and the probe seed all belong in the key. A state
    cached under one and reused under the other would silently leak the
    evaluation split into a merge the paper reports as inductive.
    """
    key = {"base_model": cfg.base_model, "modality": cfg.modality,
           "model_dtype": cfg.model_dtype,
           "experts": [{"name": e.name, "checkpoint": e.checkpoint}
                       for e in cfg.experts],
           "family": family, "params": params,
           "n_probe": cfg.data.n_probe, "probe_seed": cfg.data.probe_seed,
           "class_balanced": cfg.data.class_balanced,
           "extra": extra or {}}
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(MERGE_CACHE_DIR, f"{family}_{h}.pt"), key


def merge_cache_load(cfg, family, params, extra=None):
    path, _ = _merge_cache_path(cfg, family, params, extra)
    if family in CACHEABLE and os.path.exists(path):
        try:
            state = torch.load(path, map_location="cpu")
            _log(f"[merge-cache] hit {os.path.basename(path)} ({family})")
            return state
        except Exception as exc:                      # corrupt/partial file
            _log(f"[merge-cache] unreadable {path} ({exc}); rebuilding")
    return None


def merge_cache_save(cfg, family, params, state, extra=None):
    if family not in CACHEABLE:
        return
    path, key = _merge_cache_path(cfg, family, params, extra)
    os.makedirs(MERGE_CACHE_DIR, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"                 # atomic: concurrent jobs
    try:
        torch.save(state, tmp)                        # share this directory
        os.replace(tmp, path)
        with open(path.replace(".pt", ".json"), "w") as fh:
            json.dump(key, fh, indent=2, default=str)
        _log(f"[merge-cache] saved {os.path.basename(path)} ({family})")
    except Exception as exc:
        # A cache is an optimization, not an experiment result. Multi-GB LLM
        # states can exceed a home quota after training has already succeeded.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        _log(f"[merge-cache] save failed ({exc}); continuing without cache")


def sweep_arm(ctx, cfg, report, start_state, steps_grid, lrs, make_cfg, name_of,
              extra_of, log_label):
    """Run one refinement arm over the (lr, S) grid.

    Two efficiencies, both only sound under the stated conditions:

    * TRAJECTORY REUSE. With a constant learning rate the state after S sweeps
      of a longer run is exactly the state a separate S-sweep run produces, so
      one trajectory at max(steps_grid) with snapshots yields every horizon.
      Horizon-dependent schedules (cosine) cannot nest and fall back to
      separate runs.
    * SELECTION-ONLY SCORING. Under the n+n protocol the evaluation split may
      not inform selection, so scoring every cell on it is pure waste -- on the
      twenty-task suite it was ~99% of the runtime. Each
      cell is scored only on the disjoint selection buffer (a few hundred
      forward passes) and just the winner is evaluated. Weights are 351 MB
      apiece for ViT-B/32, so only the running best is kept resident. There is
      deliberately no full-grid evaluation fallback.

    Returns (best_name, best_state, best_sel) or (None, None, None).
    """
    max_S = max(steps_grid)
    b_name = b_state = None
    b_sel = float("inf")
    for lr in lrs:
        rc_max = make_cfg(lr, max_S)
        nest = getattr(rc_max, "lr_schedule", "constant") == "constant"
        if nest and len(steps_grid) > 1:
            _log(f"\n===== {log_label} @ lr{lr:g} S<={max_S} "
                 f"(one trajectory, snapshots at {sorted(steps_grid)}) =====")
            states, history = ctx.run_refine_checkpoints_from(
                start_state, rc_max, sorted(steps_grid), seed=cfg.seed)
            produced = [(S, states[S]) for S in sorted(steps_grid)]
        else:
            produced = []
            for S in sorted(steps_grid):
                _log(f"\n===== {log_label} @ lr{lr:g} S={S} =====")
                st, history = ctx.run_refine_from(start_state, make_cfg(lr, S),
                                                  seed=cfg.seed)
                produced.append((S, st))
        for S, st in produced:
            nm = name_of(lr, S)
            sel = SELECT_OBJ(st)
            entry = {"selection_obj": sel,
                     "displacement": pd_global_norm(pd_sub(st, start_state))}
            entry.update(extra_of(lr, S, history))
            entry["evaluated"] = False
            report["methods"][nm] = entry
            _log(f"  -> {nm}: sel={sel:.4f} (not evaluated)")
            rank = sel
            if rank < b_sel:
                b_sel, b_name, b_state = rank, nm, st
    return b_name, b_state, b_sel


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
    ap.add_argument("--della_densities", type=float, nargs="*", default=[0.7],
                    help="mean MagPrune keep probabilities")
    ap.add_argument("--della_windows", type=float, nargs="*", default=[0.14],
                    help="full row-wise keep-probability window")
    ap.add_argument("--della_lams", type=float, nargs="*", default=[1.1])
    ap.add_argument("--della_seeds", type=int, nargs="*", default=[42])
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
    # --- RegMean (label-free, data-dependent) ---
    ap.add_argument("--regmean_nondiag_scales", type=float, nargs="*",
                    default=[0.9, 1.0],
                    help="off-diagonal Gram shrinkage values")
    ap.add_argument("--regmean_eps", type=float, default=1e-3)
    ap.add_argument("--regmean_bs", type=int, default=16)
    # --- APR on top ---
    ap.add_argument("--apr_schedules", nargs="*", default=["constant"],
                    choices=["constant", "cosine", "linear"],
                    help="lr schedules for the APR arms; non-constant ones use "
                         "--apr_order (mirrors the GD arms)")
    ap.add_argument("--apr_order", choices=["fixed", "cyclic", "random"],
                    default="random",
                    help="task order for non-constant-schedule APR arms")
    ap.add_argument("--apr_constant_order",
                    choices=["fixed", "cyclic", "random"], default=None,
                    help="optional task-order override for constant-schedule "
                         "APR arms (default: the config's order)")
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
                    default=["ta", "ties", "dareties", "della", "bc", "ls", "ada"],
                    choices=["ta", "ties", "dare", "dareties", "della", "bc", "ls",
                             "ada", "regmean"])
    ap.add_argument("--apr_lrs", type=float, nargs="*", default=[2, 4, 8, 16])
    ap.add_argument("--control_gd_lrs", type=float, nargs="*", default=[],
                    help="also run ordinary replay GD from each refine_from init "
                         "at these lrs (the composition control)")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="refinement sweep counts (default: the config's single "
                         "value). More than one value makes S part of the swept "
                         "grid for ALL THREE arms -- APR, ungated and ordinary "
                         "GD -- which matters because the (lr, S) optima trade "
                         "off along a roughly constant lr*S product, and because "
                         "the arms converge at different horizons (APR "
                         "near-converges by S=5-10 while plain GD is still "
                         "improving at S=35), so pinning one S is not neutral "
                         "between them.")
    ap.add_argument("--skip_families", nargs="*", default=[],
                    choices=["ta", "ties", "dare", "dareties", "della", "bc", "ls",
                             "ada", "regmean"])
    ap.add_argument("--n_select", type=int, required=True,
                    help="n+n protocol: draw this many EXTRA labeled examples "
                         "per task, disjoint from the replay buffer, and select "
                         "every hyperparameter on them instead of on the "
                         "evaluation split. Total labeled cost per task is "
                         "n_probe + n_select.")
    ap.add_argument("--selection_seed", type=int, default=None,
                    help="sampling seed for the selection buffer; must differ "
                         "from the probe seed (default: probe_seed + 1)")
    ap.add_argument("--select_by", choices=["mean_normret", "mean_acc"],
                    default="mean_normret",
                    help="aggregate used to order evaluated winners in output")
    ap.add_argument("--retention_tasks", nargs="*", default=None,
                    help="restrict reported retention aggregates to these tasks "
                         "(excluded tasks keep raw scores in the report)")
    ap.add_argument("--out", default="results/compare/merge_baselines.json")
    args = ap.parse_args()
    if args.retention_tasks:
        global RETENTION_TASKS
        RETENTION_TASKS = list(args.retention_tasks)
        _log(f"[retention] aggregates/selection over {RETENTION_TASKS} only")

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.n_probe is not None:
        cfg.data.n_probe = args.n_probe
    if args.probe_seed is not None:
        cfg.data.probe_seed = args.probe_seed
    ctx = MergeContext.build(cfg)
    steps_grid = args.steps if args.steps else [cfg.refine.steps]
    names = cfg.task_names
    if not args.refine_from:
        # no APR composition in this run -> the expert-anchor clones (one full
        # encoder copy per task; 4 x 12 GB fp32 at 3B) are never used. Freeing
        # them keeps the CPU-RAM request at the 1-GPU billing share. eval0 has
        # already consumed them (expert scores are computed inside build()).
        for h in ctx.handles:
            h.expert_encoder = None
        for n in names:
            ctx.per_task[n]["expert_encoder"] = None

    # ---- n+n split: a disjoint, equally sized selection buffer per task ----
    global SELECT_OBJ
    selection_protocol = None
    if args.n_select is not None:
        sel_seed = (args.selection_seed if args.selection_seed is not None
                    else cfg.data.probe_seed + 1)
        if sel_seed == cfg.data.probe_seed:
            raise SystemExit(f"selection seed {sel_seed} equals the probe seed; "
                             "the two buffers must be drawn independently")
        for name in names:
            info = ctx.per_task[name]
            buf, idx = sample_replay_buffer(
                info["train_ds"], info["spec"], args.n_select, sel_seed,
                cfg.data.class_balanced, exclude_indices=info["probe_indices"],
                return_indices=True)
            if set(idx) & set(info["probe_indices"]):
                raise RuntimeError(f"train/selection overlap for {name}")
            info["selection_buffer"] = buf
        SELECT_OBJ, sel_ref = make_replay_objective(
            ctx, buffer_key="selection_buffer")
        selection_protocol = {
            "rule": "held-out replay loss on a disjoint selection buffer",
            "n_probe_per_task": cfg.data.n_probe,
            "n_select_per_task": args.n_select,
            "total_labels_per_task": cfg.data.n_probe + args.n_select,
            "probe_seed": cfg.data.probe_seed, "selection_seed": sel_seed,
            "train_selection_overlap": 0, "selection_ref_losses": sel_ref}
        _log(f"[select] n+n split: {cfg.data.n_probe} train + "
             f"{args.n_select} select per task (seeds "
             f"{cfg.data.probe_seed}/{sel_seed}); evaluation split unused "
             f"for selection")

    report = {"config": cfg.to_dict(), "tasks": names, "steps": steps_grid,
              "base": ctx.base_scores, "expert": ctx.expert_scores,
              "grids": vars(args), "selection_protocol": selection_protocol,
              "methods": {}}
    global SELECT_BY
    SELECT_BY = args.select_by
    _log(f"[report] evaluated winners ordered by {SELECT_BY}")
    skip = set(args.skip_families)

    def offer_baseline(family, name, state, **extra):
        """Select merge cells without consulting the evaluation split.

        Under the n+n protocol, evaluating every checkpoint-only candidate is
        both unnecessary (selection is by held-out replay loss) and contrary to
        the promise that the evaluation split is read only for selected cells.
        Record selection-only metadata now
        and defer the expensive benchmark evaluation until the family winner is
        known.
        """
        disp = pd_global_norm(pd_sub(state, ctx.merged0))
        sel = SELECT_OBJ(state)
        report["methods"][name] = {
            "selection_obj": sel, "displacement": disp,
            "evaluated": False, **extra}
        _log(f"  -> {name}: sel={sel:.4f} (not evaluated)")
        family.offer(name, state, selection_obj=sel)

    # ---- task arithmetic, lambda swept (the other families tune theirs, so must this) ----
    ta = Best("TA")
    if "ta" not in skip:
        for lam in args.ta_lams:
            state = task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors,
                                          {n: lam for n in names})
            nm = f"merge:TA@l{lam:g}"
            offer_baseline(ta, nm, state, lam=lam)

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
                offer_baseline(ties, nm, state, density=d, lam=lam)

    # ---- DARE alone (control: expectation-preserving => ~= TA) ----
    dare = Best("DARE")
    if "dare" not in skip:
        for d in args.dare_densities:
            for sd in args.dare_seeds:
                state = dare_ta_merge(ctx.base_encoder, ctx.task_vectors, ctx.lambdas,
                                      density=d, seed=sd)
                nm = f"merge:DARE@d{d:g},s{sd}"
                offer_baseline(dare, nm, state, density=d, seed=sd, note="control")

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
                    offer_baseline(dt, nm, state, drop=dd, trim=tr, lam=lam)

    # ---- DELLA (row-wise magnitude-ranked sampling, then sign election) ----
    della = Best("DELLA")
    if "della" not in skip:
        for d in args.della_densities:
            for w in args.della_windows:
                for lam in args.della_lams:
                    for sd in args.della_seeds:
                        state = della_merge(ctx.base_encoder, ctx.task_vectors,
                                            lam=lam, density=d, window_size=w,
                                            seed=sd)
                        nm = f"merge:DELLA@d{d:g},w{w:g},l{lam:g},s{sd}"
                        offer_baseline(della, nm, state, density=d,
                                       window_size=w, lam=lam, seed=sd)

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
                    offer_baseline(bc, nm, state, density=d, outlier=o, lam=lam)

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
                offer_baseline(ls, nm, state, sparsity=frac, lam=lam, stitch=info)
            del masks

    # ---- AdaMerging (label-free, uses unlabeled eval inputs: transductive) ----
    ada = Best("ADA")
    if "ada" not in skip:
        for dkey in args.ada_data:
            for variant in args.ada_variants:
                _log(f"\n[AdaMerging-{variant}/{dkey}] entropy minimisation "
                     f"({args.ada_steps} steps, lr={args.ada_lr})")
                ada_params = {"variant": variant, "steps": args.ada_steps,
                              "lr": args.ada_lr, "bs": args.ada_bs,
                              "init_lam": args.ada_init_lam, "seed": cfg.seed}
                # data_key decides transductive (eval split) vs matched budget,
                # which yields DIFFERENT states -- it must be part of the key.
                ada_extra = {"data_key": dkey}
                cached = merge_cache_load(cfg, "ada", ada_params, ada_extra)
                if cached is not None:
                    state, info = cached, {"cached": True, **ada_params,
                                           "data_key": dkey}
                else:
                    state, info = adamerging(
                        ctx.base_encoder, ctx.task_vectors, ctx.per_task, names, ctx.device,
                        layerwise=(variant == "layer"), steps=args.ada_steps, lr=args.ada_lr,
                        batch_size=args.ada_bs, init_lam=args.ada_init_lam, seed=cfg.seed,
                        num_workers=args.ada_workers, data_key=dkey,
                        # 3B/fp32: resident task vectors (12 GB/task) + vocab-sized
                        # entropy logits OOM the 93 GB card; stream from CPU instead
                        tv_on_gpu=(cfg.modality != "causal_lm"), logger=_log)
                    merge_cache_save(cfg, "ada", ada_params, state, ada_extra)
                tag = variant if dkey == "eval_ds" else f"{variant}-matched"
                nm = f"merge:ADA-{tag}"
                offer_baseline(ada, nm, state, adamerging=info)

    # ---- RegMean (label-free, Grams from the matched probe inputs) ----
    regmean = Best("RegMean")
    # Opt-in via --refine_from regmean so established grid recipes do not
    # silently acquire this comparatively expensive merge family.
    if "regmean" not in skip and "regmean" in args.refine_from:
        for nd in args.regmean_nondiag_scales:
            _log(f"\n[RegMean] matched probe buffers, nondiag_scale={nd:g}")
            params = {"nondiag_scale": nd, "eps": args.regmean_eps,
                      "batch_size": args.regmean_bs, "data_key": "probe_buffer"}
            cached = merge_cache_load(cfg, "regmean", params)
            if cached is not None:
                state, info = cached, {"cached": True, **params}
            else:
                state, info = regmean_merge(
                    ctx.base_encoder, ctx.per_task, names, ctx.device,
                    buffer_key="probe_buffer", nondiag_scale=nd,
                    eps=args.regmean_eps, batch_size=args.regmean_bs,
                    logger=_log)
                merge_cache_save(cfg, "regmean", params, state)
            nm = f"merge:RegMean@nd{nd:g}"
            # selection-only, like every other family: the winner is evaluated
            # on the evaluation split once it is known (see offer_baseline)
            offer_baseline(regmean, nm, state, regmean=info)

    fams = {"ta": ta, "ties": ties, "dare": dare, "dareties": dt,
            "della": della, "bc": bc, "ls": ls, "ada": ada,
            "regmean": regmean}
    report["best_per_family"] = {k: v.name for k, v in fams.items() if v}
    _log("\n[best-per-family] " + json.dumps(report["best_per_family"], indent=2))

    for key, family in fams.items():
        if not family:
            continue
        scores, nr, ag = eval_merge(ctx, family.state)
        entry = report["methods"][family.name]
        entry.update({"scores": scores, "normret": nr, "aggregate": ag,
                      "evaluated": True})
        _log(f"  [winner] {family.name}: evaluated -> "
             f"{ag['mean_normret']:.4f}/{ag['mean_acc']:.4f}")

    # every family that consumes task vectors has run by here; the refinement
    # anchors on the expert encoders (v = theta_i - theta), never on tau, so the
    # vectors are dead weight from this point. At 4 x 3B fp32 that is ~48 GB of
    # CPU RAM, which is what lets refine_from co-exist with the retained anchors
    # inside the 1-GPU-share memory request.
    ctx.task_vectors = {}

    # ---- APR on top of each init (same lr grid for every init = equal budget) ----
    for key in args.refine_from:
        fam = fams.get(key)
        if not fam:
            _log(f"[apr] skip from={key} (family not run)")
            continue
        init_label = fam.name.split(":", 1)[1]
        gate_modes = args.apr_gate_modes or [cfg.refine.gate_mode]
        for gmode in gate_modes:
         fracs = args.topk_fracs if gmode.startswith("topk") else [None]
         for frac in fracs:
          for sched in args.apr_schedules:
            order = ((args.apr_constant_order or cfg.refine.order)
                     if sched == "constant" else args.apr_order)
            def mk(lr, S, gmode=gmode, frac=frac, sched=sched):
                return dataclasses.replace(
                    cfg.refine, steps=S, lr=lr, gate_mode=gmode,
                    topk_frac=(frac if frac is not None else cfg.refine.topk_frac),
                    lr_schedule=sched, lr_min_frac=args.gd_lr_min_frac,
                    order=order)
            def nm_of(lr, S, gmode=gmode, frac=frac, sched=sched):
                return (f"apr:from={key}@lr{lr:g}"
                        + ("" if len(steps_grid) == 1 else f",S{S}")
                        + ("" if sched == "constant" else f",{sched}")
                        + (f",{order}-order" if sched == "constant" and
                           args.apr_constant_order is not None else "")
                        + ("" if gmode == cfg.refine.gate_mode and frac is None
                           else f",{gmode}" + ("" if frac is None else f"{frac:g}")))
            def ex_of(lr, S, history, gmode=gmode, frac=frac, sched=sched):
                gdens = (sum(h["gate_density"] for h in history) / len(history)
                         if history else None)
                return dict(init=init_label, lr=lr, steps=S, schedule=sched,
                            order=order, gate_mode=gmode, topk_frac=frac,
                            gate_density=gdens)
            bn, bstate, _ = sweep_arm(
                ctx, cfg, report, fam.state, steps_grid, args.apr_lrs, mk,
                nm_of, ex_of, f"APR from {init_label}")
            if bn is not None:
                report[f"best_apr_from_{key}"] = bn
                s, nr, ag = eval_merge(ctx, bstate)
                e = report["methods"][bn]
                e.update({"scores": s, "normret": nr, "aggregate": ag,
                          "evaluated": True})
                _log(f"  [winner] {bn}: evaluated -> "
                     f"{ag['mean_normret']:.4f}/{ag['mean_acc']:.4f}")
                del bstate

        # ---- controls from the SAME init: ungated anchor, and ordinary GD ----
        # Both isolate a different part of the update: nogate removes the gate but
        # keeps the anchored/distance-scaled/clipped step; GD removes all of it.
        for arm, lrs, kw in (
                ("nogate", args.nogate_lrs, dict(gate_mode="none")),
                ("gd", args.control_gd_lrs, dict(gate_mode="none",
                                                 update_mode="grad",
                                                 clip_mode="none"))):
            if not lrs:
                continue
            def mk(lr, S, kw=kw):
                return dataclasses.replace(cfg.refine, steps=S, lr=lr, **kw)
            def nm_of(lr, S, arm=arm):
                return (f"{arm}:from={key}@lr{lr:g}"
                        + ("" if len(steps_grid) == 1 else f",S{S}"))
            def ex_of(lr, S, history):
                return dict(init=init_label, lr=lr, steps=S)
            bn, bstate, _ = sweep_arm(
                ctx, cfg, report, fam.state, steps_grid, lrs, mk, nm_of, ex_of,
                f"{arm} from {init_label}")
            if bn is not None:
                report[f"best_{arm}_from_{key}"] = bn
                s, nr, ag = eval_merge(ctx, bstate)
                e = report["methods"][bn]
                e.update({"scores": s, "normret": nr, "aggregate": ag,
                          "evaluated": True})
                _log(f"  [winner] {bn}: evaluated -> "
                     f"{ag['mean_normret']:.4f}/{ag['mean_acc']:.4f}")
                del bstate
        # a peak at the grid edge means the optimum was not bracketed
        if report.get(f"best_apr_from_{key}") and args.apr_lrs:
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
    scored = {n: d for n, d in report["methods"].items() if "aggregate" in d}
    unscored = len(report["methods"]) - len(scored)
    if unscored:
        print(f"({unscored} cells scored on the selection buffer only; the "
              f"selected cell of each arm is evaluated and listed below)")
    for n, d in sorted(scored.items(),
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
