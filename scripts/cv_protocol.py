"""Protocol v2: equal total label budget, K-fold select-then-refit, multi-draw.

WHAT THIS FIXES relative to scripts/merge_baselines.py's n+n split protocol.

1. EQUAL TOTAL BUDGET. Every method gets the same B labeled examples per task.
   The n+n split charged all methods 2n labels but let checkpoint-only merges
   use only the n selection examples (the replay half sat unused for them),
   which under-resourced their selection. Here:
     * merges (checkpoint-only and unlabeled construction) score every candidate
       on all B examples;
     * labeled methods (APR, GD, ungated) split B in two: construct on one half,
       choose (eta, S) on the other, and are then REFIT ONCE on all B at the
       selected cell. That refit model is what gets evaluated. (--rotate turns
       the split into full K-fold CV at K times the cost; off by default.)
   Select-then-refit is the standard pattern (cf. sklearn GridSearchCV
   refit=True). Its known caveat applies: the hyperparameters are selected at
   construction size B(K-1)/K and applied at size B.

2. MULTI-DRAW REPORTING. The whole procedure (draw -> select -> refit ->
   evaluate) repeats over D independent buffer draws; cells are reported as
   mean +- std over draws, never as a single draw. Draw fragility is a finding
   of this project, so it belongs in the error bars rather than a caveat.

3. INTERIORITY AS A STOPPING RULE. After selection the runner checks whether
   the winning cell sits on a grid boundary in eta or S and, if so, extends the
   grid in that direction and re-selects (up to --max_extensions). A selected
   cell on the grid edge is not a verified optimum, and we have shipped one
   before (CLIP-20's lambda). S=min counts as interior when the initialization
   itself scores worse, since S=0 is then the bracketing neighbour.

4. TRACES AND WINNERS PERSISTED. Per-task held-out loss is recorded at every
   snapshot, so questions like "does task t's loss rise with more sweeps?" are
   answerable from the output file rather than needing a re-run. The refit
   winner's weights are saved when --save_winners is passed, so downstream
   probes (drift, held-out retention) need not re-derive them.

Evaluation-split discipline is unchanged and absolute: the evaluation split is
read only for cells that selection has already chosen.

Example (CLIP-8, B=32, 4 folds, 3 draws, from the pretrained model):
  python scripts/cv_protocol.py --config configs/clip8.yaml \
      --budget 32 --folds 4 --draws 3 --init pretrained \
      --apr_lrs 1 2 4 8 16 32 --nogate_lrs 0.5 1 2 4 8 16 \
      --gd_lrs 1e-5 1e-4 5e-4 1e-3 5e-3 1e-2 --steps 5 20 50 100 \
      --out results/compare/cv_clip8_B32.json
"""

import argparse
import dataclasses
import json
import os
import random
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.gradients import make_grad_fn
from apr.data import sample_replay_buffer
from apr.replay_baselines import replay_losses
from apr.metrics import aggregate_all
from apr.models import pd_sub, pd_global_norm
from apr.merge_methods import (ties_combined_tau, ties_merge, dare_ties_merge,
                               breadcrumbs_merge)
from apr.taskvec import task_arithmetic_merge

# the three labeled arms, as deltas from the config's refine settings
ARM_KW = {
    "apr":     {},
    "nogate":  {"gate_mode": "none"},
    "gd":      {"gate_mode": "none", "update_mode": "grad", "clip_mode": "none"},
}


# ---------------------------------------------------------------------------
# buffers and folds
# ---------------------------------------------------------------------------

def draw_budget_buffers(ctx, budget: int, seed: int) -> Dict[str, list]:
    """Draw B labeled examples per task. This is the ENTIRE label budget."""
    out = {}
    for name in ctx.task_names:
        info = ctx.per_task[name]
        out[name] = sample_replay_buffer(info["train_ds"], info["spec"], budget,
                                         seed, ctx.cfg.data.class_balanced)
    return out


def make_folds(examples: list, k: int, seed: int) -> List[list]:
    """Deterministic near-equal folds of one task's budget."""
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    return [[examples[i] for i in idx[f::k]] for f in range(k)]


def set_construction_buffer(ctx, buffers: Dict[str, list]):
    """Point every task's gradient closure at the given examples."""
    for h in ctx.handles:
        info = ctx.per_task[h.name]
        info["probe_buffer"] = buffers[h.name]
        h.grad_fn = make_grad_fn(info["model"], buffers[h.name], info["collator"],
                                 ctx.cfg.data.grad_batch_size
                                 or ctx.cfg.data.eval_batch_size, ctx.device)


def set_score_buffer(ctx, buffers: Dict[str, list]):
    for name in ctx.task_names:
        ctx.per_task[name]["cv_buffer"] = buffers[name]


def scored(ctx, state, ref: Dict[str, float]) -> (float, Dict[str, float]):
    """Scale-free held-out objective: mean_t L_t(state)/L_t(reference).

    Raw losses are not commensurable across tasks, so each is normalised by the
    initialization's loss on the SAME examples. Returns (aggregate, per-task).
    """
    L = replay_losses(ctx, state, buffer_key="cv_buffer")
    agg = sum(L[n] / max(ref[n], 1e-8) for n in L) / len(L)
    return agg, L


# ---------------------------------------------------------------------------
# grid handling (interiority)
# ---------------------------------------------------------------------------

def extend_lrs(lrs: List[float], where: str) -> List[float]:
    lrs = sorted(lrs)
    ratio = lrs[1] / lrs[0] if len(lrs) > 1 else 2.0
    if where == "low":
        return [lrs[0] / ratio] + lrs
    return lrs + [lrs[-1] * ratio]


def boundary_of(sel_lr, sel_S, lrs, steps, init_obj, obj_at_min_S):
    """Which axes sit on a grid edge. S=min is interior if the initialization
    (S=0) is worse, since it then brackets the minimum from below."""
    edges = []
    lrs, steps = sorted(lrs), sorted(steps)
    if len(lrs) > 1 and sel_lr == lrs[0]:
        edges.append("lr_low")
    if len(lrs) > 1 and sel_lr == lrs[-1]:
        edges.append("lr_high")
    if len(steps) > 1 and sel_S == steps[-1]:
        edges.append("S_high")
    if sel_S == steps[0] and obj_at_min_S > init_obj:
        edges.append("S_low")          # refinement is not helping at all
    return edges


# ---------------------------------------------------------------------------
# the labeled arms: K-fold CV then refit
# ---------------------------------------------------------------------------

def cv_select_and_refit(ctx, cfg, arm, init_state, budget_bufs, folds, steps,
                        lrs, args, draw_tag):
    """K-fold select over (lr, S), auto-extend on boundary, refit on all B."""
    K = args.folds
    names = ctx.task_names
    cells: Dict[tuple, list] = {}      # (lr,S) -> per-fold objectives
    traces: Dict[str, dict] = {}
    tried = set()
    lrs = sorted(lrs)

    for attempt in range(args.max_extensions + 1):
        max_S = max(steps)
        # default: ONE 2-way split -- construct on all folds but the first,
        # select on the first. --rotate turns this into full K-fold CV.
        for lr in lrs:
            for k in (range(K) if args.rotate else [0]):
                if (lr, k) in tried:
                    continue
                tried.add((lr, k))
                # construct on every fold but k; score on fold k
                train = {n: [e for f in range(K) if f != k for e in folds[n][f]]
                         for n in names}
                held = {n: folds[n][k] for n in names}
                set_construction_buffer(ctx, train)
                set_score_buffer(ctx, held)
                ref = replay_losses(ctx, init_state, buffer_key="cv_buffer")
                rc = dataclasses.replace(cfg.refine, steps=max_S, lr=lr,
                                         order=args.order, lr_schedule="constant",
                                         **ARM_KW[arm])
                _log(f"[{draw_tag}][{arm}] lr={lr:g} fold {k+1}/{K} "
                     f"(construct {len(train[names[0]])}, hold {len(held[names[0]])})")
                states, _ = ctx.run_refine_checkpoints_from(
                    init_state, rc, sorted(steps), seed=cfg.seed)
                for S in sorted(steps):
                    agg, per_task = scored(ctx, states[S], ref)
                    cells.setdefault((lr, S), []).append(agg)
                    traces[f"lr{lr:g},S{S},fold{k}"] = {
                        "objective": agg,
                        "per_task_loss": per_task,
                        "per_task_ref": ref,
                    }
                del states

        n_needed = K if args.rotate else 1
        mean_obj = {c: sum(v) / len(v) for c, v in cells.items() if len(v) == n_needed}
        (sel_lr, sel_S) = min(mean_obj, key=mean_obj.get)
        # S=0 reference on fold 0 for the interiority test
        set_score_buffer(ctx, {n: folds[n][0] for n in names})
        ref0 = replay_losses(ctx, init_state, buffer_key="cv_buffer")
        init_obj = 1.0                                     # by construction
        obj_min_S = mean_obj[(sel_lr, min(steps))]
        edges = boundary_of(sel_lr, sel_S, lrs, steps, init_obj, obj_min_S)
        if not edges or attempt == args.max_extensions:
            break
        _log(f"[{draw_tag}][{arm}] selected cell on boundary {edges}; extending")
        if "lr_low" in edges:
            lrs = extend_lrs(lrs, "low")
        if "lr_high" in edges:
            lrs = extend_lrs(lrs, "high")
        if "S_high" in edges:
            # a longer horizon changes max_S for EVERY lr: the existing
            # trajectories stop short of it, so they must all be re-run and
            # the per-cell objectives rebuilt (they are deterministic, so the
            # old cells are reproduced exactly alongside the new S).
            steps = sorted(steps) + [max(steps) * 2]
            tried.clear()
            cells.clear()

    # ---- refit on the FULL budget at the selected cell ----
    set_construction_buffer(ctx, budget_bufs)
    rc = dataclasses.replace(cfg.refine, steps=sel_S, lr=sel_lr,
                             order=args.order, lr_schedule="constant",
                             **ARM_KW[arm])
    _log(f"[{draw_tag}][{arm}] REFIT on all {args.budget} at lr={sel_lr:g}, S={sel_S}")
    refit, _ = ctx.run_refine_from(init_state, rc, seed=cfg.seed)
    return refit, {
        "selected_lr": sel_lr, "selected_S": sel_S,
        "cv_objective": mean_obj[(sel_lr, sel_S)],
        "cv_objective_per_fold": cells[(sel_lr, sel_S)],
        "lr_grid": sorted(lrs), "S_grid": sorted(steps),
        "boundary_after_extension": edges,
        "displacement": pd_global_norm(pd_sub(refit, init_state)),
        "traces": traces,
    }


# ---------------------------------------------------------------------------
# merge candidates (scored on the FULL budget)
# ---------------------------------------------------------------------------

def merge_boundaries(best, args):
    """Which selected merge hyperparameters sit on their grid's edge."""
    import re
    out = {}
    def edge(val, grid, tag):
        g = sorted(grid)
        return ([f"{tag}_low"] if len(g) > 1 and val == g[0] else []) + \
               ([f"{tag}_high"] if len(g) > 1 and val == g[-1] else [])
    for fam, (_, nm, _) in best.items():
        nums = {k: float(v) for k, v in re.findall(r"([a-z]+)([0-9.]+)", nm.split("@")[1])}
        e = []
        if fam == "TA":
            e += ["low" if nums["l"] == min(args.ta_lams) and len(args.ta_lams) > 1 else None,
                  "high" if nums["l"] == max(args.ta_lams) and len(args.ta_lams) > 1 else None]
            e = [x for x in e if x]
        elif fam == "TIES":
            e += edge(nums["d"], args.ties_densities, "density") + edge(nums["l"], args.ties_lams, "lam")
        elif fam == "DARETIES":
            e += edge(nums["dd"], args.dt_drops, "drop") + edge(nums["t"], args.dt_trims, "trim") + \
                 edge(nums["l"], args.dt_lams, "lam")
        elif fam == "BC":
            e += edge(nums["d"], args.bc_densities, "density") + edge(nums["o"], args.bc_outliers, "outlier") + \
                 edge(nums["l"], args.bc_lams, "lam")
        if e:
            out[fam] = e
    return out


def merge_candidates(ctx, args):
    """(name, state) for every checkpoint-only candidate the grids offer."""
    out = []
    for lam in args.ta_lams:
        out.append((f"TA@l{lam:g}",
                    task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors,
                                          {n: lam for n in ctx.task_names})))
    for d in args.ties_densities:
        combined = ties_combined_tau(ctx.task_vectors, density=d)
        for lam in args.ties_lams:
            out.append((f"TIES@d{d:g},l{lam:g}",
                        ties_merge(ctx.base_encoder, ctx.task_vectors, lam=lam,
                                   density=d, combined=combined)))
    for dd in args.dt_drops:
        for tr in args.dt_trims:
            for lam in args.dt_lams:
                out.append((f"DARETIES@dd{dd:g},t{tr:g},l{lam:g}",
                            dare_ties_merge(ctx.base_encoder, ctx.task_vectors,
                                            lam=lam, drop_density=dd,
                                            trim_density=tr, seed=0)))
    for d in args.bc_densities:
        for o in args.bc_outliers:
            for lam in args.bc_lams:
                out.append((f"BC@d{d:g},o{o:g},l{lam:g}",
                            breadcrumbs_merge(ctx.base_encoder, ctx.task_vectors,
                                              {n: lam for n in ctx.task_names},
                                              density=d, outlier_frac=o)))
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", type=int, required=True,
                    help="B: total labeled examples per task, for EVERY method")
    ap.add_argument("--folds", type=int, default=2,
                    help="split the budget into this many parts; by default "
                         "labeled arms construct on all but the first part and "
                         "select on the first (a single 2-way split at K=2)")
    ap.add_argument("--rotate", action="store_true",
                    help="rotate the held-out part over all K folds (full "
                         "K-fold CV, K times the cost); off by default")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--draw_seed0", type=int, default=100)
    ap.add_argument("--init", default="pretrained",
                    choices=["pretrained"] + ["ta", "ties", "dareties", "bc"],
                    help="initialization the labeled arms refine from")
    ap.add_argument("--arms", nargs="*", default=["apr", "nogate", "gd"])
    ap.add_argument("--apr_lrs", type=float, nargs="*",
                    default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--nogate_lrs", type=float, nargs="*",
                    default=[0.5, 1, 2, 4, 8, 16])
    ap.add_argument("--gd_lrs", type=float, nargs="*",
                    default=[1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    ap.add_argument("--steps", type=int, nargs="*", default=[5, 20, 50, 100])
    ap.add_argument("--order", default="random", choices=["random", "fixed"])
    ap.add_argument("--max_extensions", type=int, default=2)
    ap.add_argument("--ta_lams", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--ties_densities", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--ties_lams", type=float, nargs="*", default=[0.8, 1.0])
    ap.add_argument("--dt_drops", type=float, nargs="*", default=[0.5])
    ap.add_argument("--dt_trims", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--dt_lams", type=float, nargs="*", default=[0.8, 1.0])
    ap.add_argument("--bc_densities", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--bc_outliers", type=float, nargs="*", default=[0.01, 0.05])
    ap.add_argument("--bc_lams", type=float, nargs="*", default=[0.4])
    ap.add_argument("--skip_merges", action="store_true")
    ap.add_argument("--save_winners", default=None,
                    help="directory to persist the refit winners' weights")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.budget      # buffers are replaced per draw anyway

    # The GD and ungated arms are ABLATIONS of APR (no anchor/no gate; anchor
    # without gate). They are run from the pretrained model only, where they
    # isolate what the anchor and the gate contribute to a cold start; from
    # merge initializations only APR is run, so composition rows compare the
    # refinement against the merge it starts from, not against its ablations.
    if args.init != "pretrained":
        dropped = [a for a in args.arms if a != "apr"]
        if dropped:
            _log(f"[protocol] init={args.init}: ablation arms {dropped} are run "
                 f"from the pretrained model only; running apr")
        args.arms = [a for a in args.arms if a == "apr"]

    ctx = MergeContext.build(cfg)

    report = {
        "protocol": {
            "name": ("equal-budget k-fold select-then-refit" if args.rotate
                     else "equal-budget split select-then-refit"),
            "budget_per_task": args.budget,
            "folds": args.folds,
            "rotate": args.rotate,
            "draws": args.draws,
            "construction_per_fold": args.budget - args.budget // args.folds,
            "refit_on": args.budget,
            "selection_rule": "mean over folds of held-out replay loss, "
                              "normalised by the initialization on the same fold",
            "init": args.init,
            "order": args.order,
            "evaluation_split_used_for": "selected cells only",
        },
        "config": cfg.to_dict(), "tasks": ctx.task_names,
        "base": ctx.base_scores, "expert": ctx.expert_scores,
        "draws": {},
    }
    if args.save_winners:
        os.makedirs(args.save_winners, exist_ok=True)

    for d in range(args.draws):
        seed = args.draw_seed0 + d
        tag = f"draw{d}"
        _log(f"\n===== {tag}: drawing B={args.budget}/task (seed {seed}) =====")
        budget_bufs = draw_budget_buffers(ctx, args.budget, seed)
        folds = {n: make_folds(budget_bufs[n], args.folds, seed) for n in ctx.task_names}
        entry = {"buffer_seed": seed, "methods": {}}

        # ---- merges: score every candidate on the FULL budget ----
        init_state = ctx.base_encoder
        if not args.skip_merges:
            set_score_buffer(ctx, budget_bufs)
            ref = replay_losses(ctx, ctx.base_encoder, buffer_key="cv_buffer")
            best, boundary = {}, {}
            for attempt in range(args.max_extensions + 1):
                for nm, state in merge_candidates(ctx, args):
                    agg, _ = scored(ctx, state, ref)
                    fam = nm.split("@")[0]
                    if fam not in best or agg < best[fam][0]:
                        best[fam] = (agg, nm, state)
                    _log(f"[{tag}][merge] {nm:32s} obj={agg:.4f}")
                # interiority for the merge grids. Task arithmetic's lambda is
                # 1-D and cheap, so it is auto-extended like eta; the other
                # families' edges are recorded so the caption can say so.
                boundary = merge_boundaries(best, args)
                ta_edge = boundary.get("TA")
                if not ta_edge or attempt == args.max_extensions:
                    break
                lams = sorted(args.ta_lams)
                if "low" in ta_edge:
                    args.ta_lams = [lams[0] / 2] + lams
                if "high" in ta_edge:
                    args.ta_lams = lams + [lams[-1] * 1.5]
                _log(f"[{tag}][merge] TA lambda on grid edge {ta_edge}; "
                     f"extending to {sorted(args.ta_lams)}")
            for fam, (agg, nm, state) in best.items():
                scores, nr, ag = eval_and_record(ctx, state)
                entry["methods"][f"merge:{fam}"] = {
                    "selected": nm, "selection_obj": agg, "scores": scores,
                    "normret": nr, "aggregate": ag,
                    "grid_boundary": boundary.get(fam, []),
                    "tier": "checkpoint-only construction"}
                _log(f"[{tag}][merge] WINNER {nm}: "
                     f"mean_nr={ag['mean_normret']:.4f} mean_acc={ag['mean_acc']:.4f}")
                if args.init != "pretrained" and fam.lower().startswith(args.init[:2]):
                    init_state = state

        # ---- labeled arms: CV select -> refit -> evaluate ----
        for arm in args.arms:
            lrs = {"apr": args.apr_lrs, "nogate": args.nogate_lrs,
                   "gd": args.gd_lrs}[arm]
            refit, info = cv_select_and_refit(ctx, cfg, arm, init_state,
                                              budget_bufs, folds, args.steps,
                                              lrs, args, tag)
            scores, nr, ag = eval_and_record(ctx, refit)
            info.update({"scores": scores, "normret": nr, "aggregate": ag,
                         "tier": "labeled construction"})
            entry["methods"][f"{arm}:from={args.init}"] = info
            _log(f"[{tag}][{arm}] EVAL mean_nr={ag['mean_normret']:.4f} "
                 f"mean_acc={ag['mean_acc']:.4f} worst_nr={ag['worst_normret']:.4f}")
            if args.save_winners:
                torch.save(refit, os.path.join(
                    args.save_winners, f"{arm}_{args.init}_{tag}.pt"))
            del refit

        report["draws"][tag] = entry
        with open(args.out, "w") as f:      # checkpoint after every draw
            json.dump(report, f, indent=1)

    report["summary"] = summarize(report)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    _log(f"\n[done] -> {args.out}")
    for k, v in report["summary"].items():
        _log(f"  {k:28s} mean_acc={v['mean_acc_mean']:.4f}±{v['mean_acc_std']:.4f}  "
             f"mean_nr={v['mean_normret_mean']:.4f}±{v['mean_normret_std']:.4f}  "
             f"worst_nr={v['worst_normret_mean']:.4f}")


def eval_and_record(ctx, state):
    """The only place the evaluation split is read, and only for selected cells."""
    scores = ctx.eval_encoder(state)
    nr = ctx.normret(scores)
    ag = aggregate_all(scores, nr, ctx.base_scores, ctx.expert_scores)
    return scores, nr, ag


def summarize(report):
    """mean/std across draws for every method present in all draws."""
    import statistics
    per_method: Dict[str, list] = {}
    for tag, entry in report["draws"].items():
        for nm, m in entry["methods"].items():
            if "aggregate" in m:
                per_method.setdefault(nm, []).append(m["aggregate"])
    out = {}
    for nm, aggs in per_method.items():
        def col(key):
            return [a[key] for a in aggs if key in a]
        def ms(key):
            v = col(key)
            return (statistics.mean(v),
                    statistics.stdev(v) if len(v) > 1 else 0.0)
        ma, sa = ms("mean_acc")
        mn, sn = ms("mean_normret")
        wn, _ = ms("worst_normret")
        out[nm] = {"n_draws": len(aggs), "mean_acc_mean": ma, "mean_acc_std": sa,
                   "mean_normret_mean": mn, "mean_normret_std": sn,
                   "worst_normret_mean": wn}
    return out


if __name__ == "__main__":
    main()
