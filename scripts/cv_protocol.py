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
import re
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.gradients import make_grad_fn
from apr.data import sample_replay_buffer
from apr.replay_baselines import (replay_losses, replay_metrics,
                                  replay_losses_and_metrics,
                                  fisher_merge_matena)
from apr.localize_stitch import learn_sigmoids, masks_from_sigmoids, stitch
from apr.regmean import regmean_merge
from apr.adamerging import adamerging
from apr.apgd import apgd_merge, prepare_apgd
from apr.metrics import aggregate_all
from apr.models import pd_sub, pd_global_norm, load_encoder_state, pd_clone
from apr.data import batches_from_buffer
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


def scored(ctx, state, ref: Dict[str, float], select_on: str = "loss"
           ) -> (float, Dict[str, float]):
    """Held-out selection objective (LOWER is better). Reads no eval split.

    ``loss`` (default): mean_t L_t(state)/L_t(reference). Raw losses are not
    commensurable across tasks, so each is normalised by the initialization's
    loss on the SAME examples.

    ``metric``: minus the mean of the tasks' own reported metrics. Held-out
    cross-entropy is a biased RANKING function whenever the grid spans a wide
    confidence range: GLUE's pretrained heads sit at CE ~ ln C (maximum
    entropy), and because -log p is unbounded above, becoming confident raises
    CE on the examples it gets wrong faster than it lowers CE on those it gets
    right -- until accuracy clears ~0.8. Selection then prefers a model that
    has not moved. Measured on GLUE-8: at the cell a pinned probe shows is
    good, held-out CE RISES on 6 of 8 tasks while the metric improves on 5 of
    those 6. The metric is noisier on a small fold but unbiased. CLIP-8 does
    not show this (zero-shot CE / ln C is 0.39-0.91, already informative), so
    the correction is expected to be inert there.

    Returns ``(primary, per_task, secondary)``. Under ``metric`` the secondary
    is the loss objective, RECORDED in the traces for diagnosis but NOT used
    for selection: on a small fold the metric is quantized (accuracy on 16
    examples moves in steps of 1/16) and distinct cells can tie exactly, and
    ties fall to the least-moving cell (select_cell / least_moving_key), not
    to a second objective. (A loss tie-break was used while the rule was being
    settled on seeds 100-102; on the fresh seeds it fired in 9/54 arm cells,
    none on GLUE-8, and 2/80 merge families.) Under ``loss`` the secondary is
    None.
    """
    if select_on == "metric":
        L, M = replay_losses_and_metrics(ctx, state, buffer_key="cv_buffer")
        sec = sum(L[n] / max(ref[n], 1e-8) for n in L) / len(L)
        return -sum(M.values()) / len(M), M, sec
    L = replay_losses(ctx, state, buffer_key="cv_buffer")
    agg = sum(L[n] / max(ref[n], 1e-8) for n in L) / len(L)
    return agg, L, None


def select_cell(primary: Dict, secondary: Dict = None, ndigits: int = 9):
    """Lexicographic argmin: primary objective, then secondary as tie-break.

    The primary is rounded before comparison so that cells whose metric is
    the same rational number (k/n on an n-example fold) compare equal rather
    than being separated by floating-point summation order. Without a
    secondary, ties fall to the smallest (lr, S) -- deterministic, and the
    least-moving of the tied cells.
    """
    def key(c):
        p = round(primary[c], ndigits)
        s = secondary[c] if secondary and secondary.get(c) is not None else 0.0
        return (p, s, c)
    return min(primary, key=key)


def least_moving_key(name: str) -> tuple:
    """Tie order for merge candidates: the least-moving one wins.

    Parses the ``k<value>`` tokens after '@' (``TIES@d0.1,l0.8`` ->
    d=0.1, l=0.8) and returns (scale, remaining values in name order), so a
    tie falls to the smaller scaling coefficient first, then to the smaller
    density / drop / trim / outlier fraction. A name without a scale token
    sorts by all of its values. Mirrors select_cell's smallest-(eta, S)
    fallback for the labeled arms.
    """
    toks = re.findall(r"([a-z]+)(-?\d+(?:\.\d+)?(?:e-?\d+)?)", name.split("@", 1)[-1])
    lam = [float(v) for k, v in toks if k == "l"]
    rest = tuple(float(v) for k, v in toks if k != "l")
    return (tuple(lam[:1]) + rest) if lam else rest


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

def ls_select_and_refit(ctx, budget_bufs, folds, args, draw_tag):
    """Localize-and-Stitch with LEARNED masks, under select-then-refit.

    L&S is the one baseline whose construction consumes labels, so it cannot
    follow the merge rule of scoring every candidate on all B: the masks would
    be trained and scored on the same examples. It follows the labeled arms
    instead -- train masks on the construction fold, choose (gamma, sparsity,
    lambda) by held-out replay loss on the other fold, then retrain the masks on
    all B at the chosen gamma and apply the chosen sparsity and lambda.

    One mask training serves an entire sparsity grid (masks_from_sigmoids
    top-k's the same sigmoids), so the cost is one training pass per gamma plus
    one refit pass, not one per candidate.
    """
    names = ctx.task_names
    K = args.folds
    train = {n: [e for f in range(K) if f != 0 for e in folds[n][f]] for n in names}
    held = {n: folds[n][0] for n in names}
    set_construction_buffer(ctx, train)
    set_score_buffer(ctx, held)
    ref = replay_losses(ctx, ctx.base_encoder, buffer_key="cv_buffer")

    best = None                      # (obj, gamma, sparsity, lam, info)
    traces = {}
    for gamma in args.ls_gammas:
        _log(f"[{draw_tag}][ls] learning masks on {len(train[names[0]])} "
             f"examples, gamma={gamma:g}")
        sig = learn_sigmoids(ctx, steps=args.ls_steps, lr=args.ls_lr,
                             gamma=gamma, batch_size=args.ls_bs, logger=_log)
        for sp in args.ls_sparsities:
            masks = masks_from_sigmoids(sig, sp)
            for lam in args.ls_lams:
                state, info = stitch(ctx.base_encoder, ctx.task_vectors,
                                     masks, lam=lam)
                agg, _, sec = scored(ctx, state, ref, args.select_on)
                # ties to the least-moving stitch: smallest scale, then the
                # sparser mask, then the smaller gamma
                key = (round(agg, 9), (lam, sp, gamma))
                traces[f"g{gamma:g},sp{sp:g},l{lam:g}"] = {
                    "objective": agg, "objective_secondary": sec,
                    "covered_frac": info["covered_frac"],
                    "overlap_frac": info["overlap_frac"]}
                if best is None or key < best[0]:
                    best = (key, gamma, sp, lam, info)
                del state
        del sig

    obj, gamma, sp, lam = best[0][0], best[1], best[2], best[3]
    _log(f"[{draw_tag}][ls] REFIT masks on all {args.budget} at gamma={gamma:g}, "
         f"applying sparsity={sp:g}, lambda={lam:g} (held-out obj {obj:.4f})")
    set_construction_buffer(ctx, budget_bufs)
    sig = learn_sigmoids(ctx, steps=args.ls_steps, lr=args.ls_lr, gamma=gamma,
                         batch_size=args.ls_bs, logger=_log)
    state, info = stitch(ctx.base_encoder, ctx.task_vectors,
                         masks_from_sigmoids(sig, sp), lam=lam)
    return state, {
        "selected": f"LS@g{gamma:g},sp{sp:g},l{lam:g}",
        "selection_obj": obj,
        "ls_gamma": gamma, "ls_sparsity": sp, "ls_lam": lam,
        "ls_grid": {"gammas": args.ls_gammas, "sparsities": args.ls_sparsities,
                    "lams": args.ls_lams, "steps": args.ls_steps},
        "covered_frac": info["covered_frac"], "overlap_frac": info["overlap_frac"],
        "traces": traces,
        "tier": "labeled construction",
    }


def clip_summary(history):
    """How often the expert-distance cap actually bound, over a trajectory.

    ``clipped_frac_gated`` is the share of GATED coordinates whose pre-clip step
    exceeded the remaining expert distance, i.e. that were snapped exactly onto
    the expert. The cap binds when eta*|g_r| > 1, a condition on gradient
    magnitude alone (|v| cancels), so this is the quantity that says whether the
    selected eta makes the update perturbative or saturating. refine() computes
    it per (sweep, task); nothing used to read it back.
    """
    if not history:
        return None
    cg = [h["clipped_frac_gated"] for h in history if "clipped_frac_gated" in h]
    ca = [h["clipped_frac_all"] for h in history if "clipped_frac_all" in h]
    gd = [h["gate_density"] for h in history if "gate_density" in h]
    if not cg:
        return None
    return {"n_steps": len(cg),
            "clipped_frac_gated_mean": sum(cg) / len(cg),
            "clipped_frac_gated_max": max(cg),
            "clipped_frac_all_mean": sum(ca) / len(ca),
            "gate_density_mean": sum(gd) / len(gd) if gd else None}


def step_cap(args, arm):
    """Horizon ceiling for an arm from --max_steps (None = unbounded).
    Tokens are either a bare integer (every arm) or ``arm=value``."""
    cap = None
    for tok in args.max_steps or []:
        if "=" in tok:
            a, v = tok.split("=", 1)
            if a == arm:
                cap = int(v)
        else:
            cap = int(tok) if cap is None else min(cap, int(tok))
    return cap


def cv_select_and_refit(ctx, cfg, arm, init_state, budget_bufs, folds, steps,
                        lrs, args, draw_tag):
    """K-fold select over (lr, S), auto-extend on boundary, refit on all B."""
    K = args.folds
    names = ctx.task_names
    cells: Dict[tuple, list] = {}      # (lr,S) -> per-fold objectives
    cells_sec: Dict[tuple, list] = {}  # (lr,S) -> per-fold tie-break objectives
    traces: Dict[str, dict] = {}
    clip_cells: Dict[str, list] = {}   # "lr..,fold.." -> cap-binding summaries
    # (lr, k) -> (CPU state at sweeps_done, sweeps_done): a constant-lr
    # trajectory is CONTINUED when the horizon grid grows, never re-run
    # (Section: run_refine_checkpoints_from resume_state/resume_sweep)
    done: Dict[tuple, tuple] = {}
    lrs = sorted(lrs)
    cap = step_cap(args, arm)
    if cap is not None:
        steps = [S for S in sorted(steps) if S <= cap] or [min(steps)]
        _log(f"[{draw_tag}][{arm}] horizon capped at S<={cap}: grid {steps}")
    S_capped = False      # selected S sits at the cap and could not be extended

    for attempt in range(args.max_extensions + 1):
        max_S = max(steps)
        # default: ONE 2-way split -- construct on all folds but the first,
        # select on the first. --rotate turns this into full K-fold CV.
        for lr in lrs:
            for k in (range(K) if args.rotate else [0]):
                prev = done.get((lr, k))
                if prev is not None and prev[1] >= max_S:
                    continue
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
                if prev is None:
                    _log(f"[{draw_tag}][{arm}] lr={lr:g} fold {k+1}/{K} "
                         f"(construct {len(train[names[0]])}, "
                         f"hold {len(held[names[0]])})")
                    states, hist = ctx.run_refine_checkpoints_from(
                        init_state, rc, sorted(steps), seed=cfg.seed)
                    new_S = sorted(steps)
                else:
                    resume_state, done_upto = prev
                    _log(f"[{draw_tag}][{arm}] lr={lr:g} fold {k+1}/{K} "
                         f"continuing S={done_upto} -> {max_S}")
                    states, hist = ctx.run_refine_checkpoints_from(
                        init_state, rc, sorted(steps), seed=cfg.seed,
                        resume_state=resume_state, resume_sweep=done_upto)
                    new_S = [S for S in sorted(steps) if S > done_upto]
                # cap-binding rate over the segment just run, per (lr, fold);
                # continuation segments accumulate into the same key
                cs = clip_summary(hist)
                if cs is not None:
                    clip_cells.setdefault(f"lr{lr:g},fold{k}", []).append(cs)
                for S in new_S:
                    agg, per_task, sec = scored(ctx, states[S], ref,
                                                args.select_on)
                    cells.setdefault((lr, S), []).append(agg)
                    cells_sec.setdefault((lr, S), []).append(sec)
                    traces[f"lr{lr:g},S{S},fold{k}"] = {
                        "objective": agg,
                        "objective_secondary": sec,
                        "per_task_loss": per_task,
                        "per_task_ref": ref,
                    }
                done[(lr, k)] = (states[max_S], max_S)
                del states

        n_needed = K if args.rotate else 1
        mean_obj = {c: sum(v) / len(v) for c, v in cells.items() if len(v) == n_needed}
        # ties in the primary fall to the smallest (lr, S), the least-moving
        # cell; the secondary objective stays in the traces for diagnosis only
        (sel_lr, sel_S) = select_cell(mean_obj)
        # S=0 reference on fold 0 for the interiority test
        set_score_buffer(ctx, {n: folds[n][0] for n in names})
        init_obj, _, _ = scored(ctx, init_state, ref, args.select_on)
        # for the loss objective this is 1.0 by construction; for the metric
        # objective it is minus the initialization's own mean metric
        obj_min_S = mean_obj[(sel_lr, min(steps))]
        edges = boundary_of(sel_lr, sel_S, lrs, steps, init_obj, obj_min_S)
        S_capped = ("S_high" in edges and cap is not None
                    and max(steps) * 2 > cap)
        if S_capped:
            # the horizon ceiling is a protocol decision: S=max stays a
            # boundary pick (recorded as such) but is not extended past it
            _log(f"[{draw_tag}][{arm}] selected S={sel_S} at the cap {cap}; "
                 f"not extending S")
            edges = [e for e in edges if e != "S_high"]
        # S_low (the initialization beats every horizon) is recorded but has
        # no extension direction, so it must not spin the loop on its own
        if not [e for e in edges if e != "S_low"] or attempt == args.max_extensions:
            break
        _log(f"[{draw_tag}][{arm}] selected cell on boundary {edges}; extending")
        if "lr_low" in edges:
            lrs = extend_lrs(lrs, "low")
        if "lr_high" in edges:
            lrs = extend_lrs(lrs, "high")
        if "S_high" in edges:
            # a longer horizon extends every trajectory IN PLACE: each (lr, k)
            # continues from its retained state, paying only the extra sweeps
            steps = sorted(steps) + [max(steps) * 2]

    done.clear()          # release the retained trajectory states
    # ---- refit on the FULL budget at the selected cell ----
    set_construction_buffer(ctx, budget_bufs)
    rc = dataclasses.replace(cfg.refine, steps=sel_S, lr=sel_lr,
                             order=args.order, lr_schedule="constant",
                             **ARM_KW[arm])
    _log(f"[{draw_tag}][{arm}] REFIT on all {args.budget} at lr={sel_lr:g}, S={sel_S}")
    refit, refit_hist = ctx.run_refine_from(init_state, rc, seed=cfg.seed)
    refit_clip = clip_summary(refit_hist)
    if refit_clip is not None:
        _log(f"[{draw_tag}][{arm}] refit cap-binding: "
             f"{refit_clip['clipped_frac_gated_mean']:.4f} of gated coords "
             f"(max {refit_clip['clipped_frac_gated_max']:.4f} over steps), "
             f"gate_density={refit_clip['gate_density_mean']:.4f}")
    return refit, {
        "selected_lr": sel_lr, "selected_S": sel_S,
        "cv_objective": mean_obj[(sel_lr, sel_S)],
        "cv_objective_per_fold": cells[(sel_lr, sel_S)],
        "lr_grid": sorted(lrs), "S_grid": sorted(steps),
        "boundary_after_extension": edges + (["S_high"] if S_capped else []),
        "S_cap": cap, "S_capped": bool(S_capped),
        "displacement": pd_global_norm(pd_sub(refit, init_state)),
        "traces": traces,
        "refit_clip": refit_clip,        # cap-binding at the EVALUATED model
        "selection_clip": clip_cells,    # cap-binding per (lr, fold) in selection
    }


# ---------------------------------------------------------------------------
# merge candidates (scored on the FULL budget)
# ---------------------------------------------------------------------------

def tatr_omega(ctx, buffers):
    """TATR's conflict scores (Sun et al., ICLR 2025, released TATR_merging).

    Omega = sum_{i != j} E[|per-example grad of task i's loss at theta_0|]
            (elementwise) |tau_j|.

    Faithful to the released code: gradients are taken at the PRETRAINED
    encoder through each task's own head, one example at a time, in absolute
    value, then averaged (order-1 variant, their default). The released run
    uses 128 examples per task; here each task contributes the draw's B
    labeled examples, the same budget every method is charged for. Only
    parameter keys enter Omega, as in the released flattening; non-parameter
    entries of a task vector are never masked.
    """
    grads = {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        load_encoder_state(info["model"], ctx.base_encoder)
        model = info["model"].to(ctx.device)
        model.eval()
        acc, count = None, 0
        for batch in batches_from_buffer(buffers[n], info["collator"], 1,
                                         ctx.device):
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = model(**batch)
                out.loss.backward()
            with torch.no_grad():
                g = {k: pp.grad.detach().abs() for k, pp in
                     model.named_parameters()
                     if k in ctx.base_encoder and pp.grad is not None}
            if acc is None:
                acc = {k: v.clone() for k, v in g.items()}
            else:
                for k in acc:
                    acc[k] += g[k]
            count += 1
        model.zero_grad(set_to_none=True)
        if not getattr(ctx, "keep_model_on_device", False):
            model.to("cpu")
        grads[n] = {k: (v / max(count, 1)).cpu() for k, v in acc.items()}
        _log(f"[tatr] |grad| at theta_0 for {n}: {count} examples, "
             f"{len(grads[n])} tensors")
    omega = {k: torch.zeros_like(v) for k, v in
             next(iter(grads.values())).items()}
    for i in ctx.task_names:
        for j in ctx.task_names:
            if i == j:
                continue
            tv = ctx.task_vectors[j]
            for k in omega:
                omega[k] += grads[i][k] * tv[k].abs().cpu()
    return omega


def tatr_mask(omega, ratio: float):
    """Released thresholding: keep the bottom ``ratio`` fraction of Omega.

    threshold = the int(ratio*N)-th smallest value; mask is strictly-below,
    exactly as ``(Omega < values_desc[N - int(ratio*N)])`` in their code.
    """
    flat = torch.cat([v.reshape(-1) for v in omega.values()])
    k = int(ratio * flat.numel())
    if k < 1:
        return {key: torch.zeros_like(v, dtype=torch.bool)
                for key, v in omega.items()}
    thr = torch.kthvalue(flat, k).values
    return {key: v < thr for key, v in omega.items()}


def gradfix_signs(ctx, buffers):
    """GradFix's per-task gradient signs at theta_0 (Rinaldi et al., 2025).

    Faithful to the released ``compute_real_gradient_signs`` with its default
    ``vote="mean"``: ONE mean gradient of the task loss over the task's
    examples at the pretrained encoder, then ``sign(-grad)`` -- the descent
    direction. Each task contributes the draw's B labeled examples.
    """
    signs = {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        load_encoder_state(info["model"], ctx.base_encoder)
        model = info["model"].to(ctx.device)
        model.eval()
        model.zero_grad(set_to_none=True)
        nb = 0
        for batch in batches_from_buffer(buffers[n], info["collator"],
                                         ctx.cfg.data.eval_batch_size,
                                         ctx.device):
            with torch.enable_grad():
                out = model(**batch)
                (out.loss * batch["labels"].shape[0]).backward()
            nb += int(batch["labels"].shape[0])
        with torch.no_grad():
            signs[n] = {k: torch.sign(-pp.grad.detach()).cpu()
                        for k, pp in model.named_parameters()
                        if k in ctx.base_encoder and pp.grad is not None}
        model.zero_grad(set_to_none=True)
        if not getattr(ctx, "keep_model_on_device", False):
            model.to("cpu")
        _log(f"[gradfix] sign(-grad) at theta_0 for {n}: {nb} examples")
    return signs


def restore_residency(ctx):
    """Undo a helper's model.to("cpu") when the context keeps models resident.

    RegMean gathers its Grams through forward hooks and AdaMerging runs an
    entropy pass; both end by moving each task model back to the CPU. That is
    correct for a CPU-resident context, but when ``keep_model_on_device`` is
    set the scoring path assumes the models are still on the device
    (replay_losses_and_metrics passes ``move_model=not keep_model_on_device``),
    so the next forward would mix CUDA inputs with CPU weights.
    """
    if getattr(ctx, "keep_model_on_device", False):
        for n in ctx.task_names:
            ctx.per_task[n]["model"].to(ctx.device)


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
        elif fam == "GRADFIX":
            e += edge(nums["l"], args.ta_lams, "lam")
        elif fam == "TATR":
            e += edge(nums["r"], args.tatr_ratios, "ratio") + edge(nums["l"], args.ta_lams, "lam")
        elif fam == "DOGE":
            e += edge(nums["eta"], args.doge_etas, "eta")
        elif fam == "REGMEAN":
            e += edge(nums["nd"], args.regmean_nondiag, "nondiag")
        elif fam == "BC":
            e += edge(nums["d"], args.bc_densities, "density") + edge(nums["o"], args.bc_outliers, "outlier") + \
                 edge(nums["l"], args.bc_lams, "lam")
        if e:
            out[fam] = e
    return out


def merge_candidates(ctx, args):
    """(name, state) for every checkpoint-only candidate the grids offer.

    Fisher merging is NOT here: its candidates depend on the draw's inputs, so
    they are built in main() once the budget buffers exist (see fisher_merge_matena).
    """
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
                    choices=["pretrained", "ta", "ties", "dareties", "bc",
                             "fisher", "ls", "regmean", "doge", "ada", "tatr", "gradfix"],
                    help="initialization the labeled arms refine from; 'fisher' "
                         "requires --fisher and 'ls' requires --ls")
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
    ap.add_argument("--select_on", default="loss", choices=["loss", "metric"],
                    help="held-out selection objective. 'loss': normalised "
                         "replay cross-entropy (default). 'metric': the tasks' "
                         "own reported metrics, which held-out CE ranks "
                         "incorrectly when the initialization is at maximum "
                         "entropy (see scored()). Applies to EVERY method.")
    ap.add_argument("--max_steps", nargs="*", default=None,
                    help="horizon ceiling: bare integer (all arms) or arm=value "
                         "tokens, e.g. apr=100. Grid entries above it are dropped "
                         "and S is never extended past it; a selection at the "
                         "cap is recorded as S_capped (a boundary pick)")
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
    ap.add_argument("--ls", action="store_true",
                    help="add Localize-and-Stitch with masks LEARNED on the "
                         "buffer, under select-then-refit")
    ap.add_argument("--ls_gammas", type=float, nargs="*",
                    default=[0.0, 1e-7, 1e-6, 1e-5], help="L1 penalty grid")
    ap.add_argument("--ls_sparsities", type=float, nargs="*",
                    default=[0.01, 0.05, 0.1])
    ap.add_argument("--ls_lams", type=float, nargs="*",
                    default=[0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--ls_steps", type=int, default=300)
    ap.add_argument("--ls_lr", type=float, default=0.1)
    ap.add_argument("--ls_bs", type=int, default=16)
    ap.add_argument("--gradfix", action="store_true",
                    help="include GradFix (mask each task vector by sign "
                         "agreement with -grad of its task loss at theta_0)")
    ap.add_argument("--tatr", action="store_true",
                    help="include TATR (task arithmetic in trust region; the "
                         "budget's labeled gradients at theta_0 define Omega)")
    ap.add_argument("--tatr_ratios", type=float, nargs="*",
                    default=[0.8, 0.9, 0.95, 0.99, 0.999],
                    help="kept fraction of Omega; released default 0.99. "
                         "ratio=1 would be plain TA, so 0.999 brackets it")
    ap.add_argument("--regmean", action="store_true",
                    help="include RegMean among the candidates (unlabeled "
                         "construction: Gram matrices from the budget inputs)")
    ap.add_argument("--regmean_nondiag", type=float, nargs="*",
                    default=[0.9, 1.0],
                    help="off-diagonal Gram shrinkage, as in merge_baselines.py")
    ap.add_argument("--regmean_eps", type=float, default=1e-3)
    ap.add_argument("--regmean_bs", type=int, default=16)
    ap.add_argument("--doge", action="store_true",
                    help="include DOGE/APGD (data-free; the budget selects eta)")
    ap.add_argument("--doge_etas", type=float, nargs="*",
                    default=[0.03, 0.05, 0.07, 0.09, 0.11, 0.13],
                    help="vision defaults; RoBERTa needs a re-centred grid")
    ap.add_argument("--doge_iters", type=int, default=400)
    ap.add_argument("--doge_lr", type=float, default=1e-4)
    ap.add_argument("--doge_density", type=float, default=0.30)
    ap.add_argument("--doge_subspace", type=int, default=6)
    ap.add_argument("--adamerging", action="store_true",
                    help="include layer-wise AdaMerging fitted on the budget "
                         "inputs (matched budget, NOT the transductive default)")
    ap.add_argument("--ada_steps", type=int, default=300)
    ap.add_argument("--ada_lr", type=float, default=1e-3)
    ap.add_argument("--ada_bs", type=int, default=16)
    ap.add_argument("--ada_init_lam", type=float, default=0.3)
    ap.add_argument("--fisher", action="store_true",
                    help="include Fisher merging (expected label-free Fisher, "
                         "per-model simplex coefficients) among the candidates")
    ap.add_argument("--fisher_points", type=int, default=50,
                    help="simplex search points, as in Matena & Raffel")
    ap.add_argument("--save_winners", default=None,
                    help="directory to persist the refit winners' weights")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.init == "fisher" and not args.fisher:
        ap.error("--init fisher requires --fisher (nothing would build it)")
    if args.init == "ls" and not args.ls:
        ap.error("--init ls requires --ls (nothing would build it)")
    for _f in ("regmean", "doge", "ada", "tatr", "gradfix"):
        _flag = "adamerging" if _f == "ada" else _f
        if args.init == _f and not getattr(args, _flag):
            ap.error(f"--init {_f} requires --{_flag} (nothing would build it)")
    if args.init != "pretrained" and args.skip_merges:
        ap.error(f"--init {args.init} needs the merges built; drop --skip_merges")

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
            "selection_rule": (
                "mean over folds of held-out replay loss, normalised by the "
                "initialization on the same fold" if args.select_on == "loss"
                else "mean over folds of the held-out per-task metric"),
            "select_on": args.select_on,
            "tie_break": ("the least-moving cell: smallest (eta, S) for the "
                          "labeled arms; smallest scale, then the remaining "
                          "hyperparameters, for merges"),
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
            best, boundary, meta = {}, {}, {}
            if args.fisher:
                # Fisher merging (Matena & Raffel): the expected Fisher is
                # label-free but depends on this draw's inputs, so its candidates
                # are built here rather than in merge_candidates -- and built
                # ONCE, outside the grid-extension loop, since they are costly.
                _log(f"[{tag}][merge] expected Fishers + "
                     f"{args.fisher_points}-point simplex search")
                for nm, state, mt in fisher_merge_matena(
                        ctx, budget_bufs, None, n_points=args.fisher_points,
                        seed=seed, logger=_log):
                    agg, _, sec = scored(ctx, state, ref, args.select_on)
                    key = (round(agg, 9), least_moving_key(nm))
                    if "FISHER" not in best or key < best["FISHER"][0]:
                        best["FISHER"] = (key, nm, state)
                        meta["FISHER"] = mt
                    _log(f"[{tag}][merge] {nm:28s} obj={agg:.4f}")
            if args.regmean:
                # RegMean (Jin et al.): needs the live models to hook, and the
                # Gram matrices depend on this draw's inputs -- hence here and
                # not in merge_candidates. Unlabeled construction: it reads the
                # budget inputs, never their labels.
                for nd in args.regmean_nondiag:
                    state, mt = regmean_merge(
                        ctx.base_encoder, ctx.per_task, ctx.task_names, ctx.device,
                        buffer_key="cv_buffer", nondiag_scale=nd,
                        eps=args.regmean_eps, batch_size=args.regmean_bs,
                        logger=_log)
                    restore_residency(ctx)
                    nm = f"REGMEAN@nd{nd:g}"
                    agg, _, sec = scored(ctx, state, ref, args.select_on)
                    key = (round(agg, 9), least_moving_key(nm))
                    if "REGMEAN" not in best or key < best["REGMEAN"][0]:
                        best["REGMEAN"] = (key, nm, state)
                        meta["REGMEAN"] = mt
                    _log(f"[{tag}][merge] {nm:28s} obj={agg:.4f}")
            if args.doge:
                # DOGE/APGD (Wei et al.): data-free, so the budget enters only
                # through the choice of the global scale eta. The shared-subspace
                # SVD does not depend on eta, so it is built ONCE here rather
                # than per candidate (each candidate is 400 Adam iterations).
                # NOTE: our port does not reproduce the published numbers at the
                # authors' own eta -- see results/compare/grid_nn_*_apgd_*.json.
                prep = prepare_apgd(ctx.task_vectors, ctx.device,
                                    subspace_divisor=args.doge_subspace,
                                    logger=_log)
                for eta in args.doge_etas:
                    state, mt = apgd_merge(
                        ctx.base_encoder, ctx.task_vectors, eta, ctx.device,
                        preparation=prep, iterations=args.doge_iters,
                        lr=args.doge_lr, keep_density=args.doge_density,
                        logger=None)
                    nm = f"DOGE@eta{eta:g}"
                    agg, _, sec = scored(ctx, state, ref, args.select_on)
                    key = (round(agg, 9), least_moving_key(nm))
                    if "DOGE" not in best or key < best["DOGE"][0]:
                        best["DOGE"] = (key, nm, state)
                        meta["DOGE"] = mt
                    _log(f"[{tag}][merge] {nm:28s} obj={agg:.4f}")
                del prep
            if args.adamerging:
                # AdaMerging (Yang et al.): entropy minimisation on UNLABELED
                # inputs. data_key MUST be the draw's budget buffer -- the
                # library default "eval_ds" is the transductive formulation and
                # would fit coefficients on the evaluation split, outside the
                # budget every other method is charged for.
                state, mt = adamerging(
                    ctx.base_encoder, ctx.task_vectors, ctx.per_task,
                    ctx.task_names, ctx.device, layerwise=True,
                    steps=args.ada_steps, lr=args.ada_lr,
                    batch_size=args.ada_bs, init_lam=args.ada_init_lam,
                    seed=seed, data_key="cv_buffer", logger=_log)
                restore_residency(ctx)
                nm = "ADA@layer"
                agg, _, sec = scored(ctx, state, ref, args.select_on)
                best["ADA"] = ((round(agg, 9), ()), nm, state)
                meta["ADA"] = mt
                _log(f"[{tag}][merge] {nm:28s} obj={agg:.4f}")
            if args.gradfix:
                # GradFix (Rinaldi et al., 2025), Mask-then-Merge with the
                # released mask_mode="normal": zero each task vector where its
                # sign disagrees with sign(-grad) of that task's loss at
                # theta_0, then task-arithmetic the masked vectors. The
                # released merge fixes the scale (mean over tasks); here the
                # coefficient is selected on the budget like TA's, the same
                # treatment every merge family receives.
                gsigns = gradfix_signs(ctx, budget_bufs)
                masked = {}
                for name2 in ctx.task_names:
                    tv = ctx.task_vectors[name2]; gs = gsigns[name2]
                    masked[name2] = {
                        k2: (torch.where(torch.sign(v2) == gs[k2].to(v2.device),
                                         v2, torch.zeros_like(v2))
                             if k2 in gs else v2)
                        for k2, v2 in tv.items()}
                del gsigns
                for lam in args.ta_lams:
                    state = pd_clone(ctx.base_encoder)
                    for name2 in ctx.task_names:
                        for k2, v2 in masked[name2].items():
                            state[k2] = state[k2] + lam * v2
                    nm = f"GRADFIX@l{lam:g}"
                    agg, _, sec = scored(ctx, state, ref, args.select_on)
                    key = (round(agg, 9), least_moving_key(nm))
                    if "GRADFIX" not in best or key < best["GRADFIX"][0]:
                        best["GRADFIX"] = (key, nm, state)
                    _log(f"[{tag}][merge] {nm:28s} obj={agg:.4f}")
                    del state
                del masked
            if args.tatr:
                # TATR (Sun et al., ICLR 2025): task arithmetic restricted to
                # the low-conflict trust region. Omega needs this draw's
                # labeled gradients at theta_0, so it is built here, once;
                # masks and merges per (ratio, lambda) are then cheap.
                omega = tatr_omega(ctx, budget_bufs)
                for ratio in args.tatr_ratios:
                    mask = tatr_mask(omega, ratio)
                    for lam in args.ta_lams:
                        state = pd_clone(ctx.base_encoder)
                        for name2 in ctx.task_names:
                            tv = ctx.task_vectors[name2]
                            for k2, v2 in tv.items():
                                m2 = mask.get(k2)
                                upd = v2 * m2.to(v2.dtype) if m2 is not None else v2
                                state[k2] = state[k2] + lam * upd
                        nm = f"TATR@r{ratio:g},l{lam:g}"
                        agg, _, sec = scored(ctx, state, ref, args.select_on)
                        key = (round(agg, 9), least_moving_key(nm))
                        if "TATR" not in best or key < best["TATR"][0]:
                            best["TATR"] = (key, nm, state)
                        _log(f"[{tag}][merge] {nm:28s} obj={agg:.4f}")
                        del state
                del omega
            for attempt in range(args.max_extensions + 1):
                for nm, state in merge_candidates(ctx, args):
                    agg, _, sec = scored(ctx, state, ref, args.select_on)
                    fam = nm.split("@")[0]
                    # same rule as the labeled arms: primary objective, ties
                    # to the least-moving candidate (smallest scale, then the
                    # remaining hyperparameters in name order)
                    key = (round(agg, 9), least_moving_key(nm))
                    if fam not in best or key < best[fam][0]:
                        best[fam] = (key, nm, state)
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
            for fam, (key, nm, state) in best.items():
                scores, nr, ag = eval_and_record(ctx, state)
                entry["methods"][f"merge:{fam}"] = {
                    "selected": nm, "selection_obj": key[0],
                    "selection_tie_key": list(key[1]), "scores": scores,
                    "normret": nr, "aggregate": ag,
                    "grid_boundary": boundary.get(fam, []),
                    "tier": ("unlabeled construction" if fam == "FISHER"
                             else "checkpoint-only construction"),
                    **({"meta": meta[fam]} if fam in meta else {})}
                _log(f"[{tag}][merge] WINNER {nm}: "
                     f"mean_nr={ag['mean_normret']:.4f} mean_acc={ag['mean_acc']:.4f}")
                if args.init != "pretrained" and fam.lower() == args.init.lower():
                    init_state = state

            # Localize-and-Stitch: labeled construction, so select-then-refit
            # rather than the merges' score-on-all-B rule (see ls_select_and_refit)
            if args.ls:
                ls_state, ls_info = ls_select_and_refit(ctx, budget_bufs, folds,
                                                        args, tag)
                scores, nr, ag = eval_and_record(ctx, ls_state)
                ls_info.update({"scores": scores, "normret": nr, "aggregate": ag})
                entry["methods"]["merge:LS"] = ls_info
                _log(f"[{tag}][ls] WINNER {ls_info['selected']}: "
                     f"mean_nr={ag['mean_normret']:.4f} mean_acc={ag['mean_acc']:.4f}")
                if args.init.lower() == "ls":
                    init_state = ls_state
                else:
                    del ls_state

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
