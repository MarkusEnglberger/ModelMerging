"""Algorithm 1: sequential expert-direction-gated replay refinement.

For each sweep s and each task i (in the chosen order), at the *current* model
state theta:

    g_i   = grad_theta L_i(theta; D_probe_i)            (Eq. 14, task i's head)
    v_i   = theta_i - theta                             (Eq. 15, merge->expert)
    m_i,r = 1[ g_i,r * v_i,r < -eps_gate ]              (Eq. 16, AP gate)
    u~_i,r= -g_i,r * |v_i,r| * m_i,r                     (Eq. 17)
    u_i,r = clip(eta * u~_i,r, -|v_i,r|, |v_i,r|)      (Eq. 18, trust region)
    theta <- theta + u_i                                  (Eq. 19, applied NOW)

The defining behaviour is *sequential*: each task's clipped update is applied to
the model immediately, before the next task's gradient and gate are computed.
The aggregated-U ablation (config.aggregated=True) instead computes every u_i at
the start-of-sweep state and applies their sum once.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional
import random

import torch

from .config import RefineConfig
from .models import (ParamDict, encoder_param_names, load_encoder_state,
                     pd_clone, pd_to, pd_global_norm)


class TaskHandle:
    """Everything needed to compute task i's refinement update."""

    def __init__(self, name: str, model, expert_encoder: ParamDict,
                 grad_fn: Callable[[], Dict[str, torch.Tensor]]):
        self.name = name
        self.model = model
        self.expert_encoder = expert_encoder  # theta_i (kept on CPU)
        # grad_fn() -> {param_name: grad_tensor} for the encoder, computed at
        # whatever encoder state the model currently holds.
        self.grad_fn = grad_fn


def _order_for_sweep(names: List[str], mode: str, sweep: int,
                     rng: random.Random) -> List[str]:
    if mode == "fixed":
        return list(names)
    if mode == "cyclic":
        k = sweep % len(names)
        return names[k:] + names[:k]
    if mode == "random":
        perm = list(names)
        rng.shuffle(perm)
        return perm
    raise ValueError(f"Unknown task order '{mode}'")


def _base_update(g: torch.Tensor, v: torch.Tensor, mode: str) -> torch.Tensor:
    """u~ before gating/clipping."""
    if mode == "gated_grad":
        return -g * v.abs()
    if mode == "grad":
        return -g
    if mode == "interp":
        return v.clone()
    raise ValueError(f"Unknown update_mode '{mode}'")


def _gate(g: torch.Tensor, v: torch.Tensor, cfg: RefineConfig,
          rng_gen: Optional[torch.Generator]) -> torch.Tensor:
    """Boolean/0-1 mask m for one parameter tensor."""
    eps = cfg.gate_eps
    mode = cfg.gate_mode
    if mode == "none":
        return torch.ones_like(g)
    if mode == "coordinate":
        return (g * v < -eps).to(g.dtype)
    if mode == "inverted":
        return (g * v > eps).to(g.dtype)
    if mode == "tensor":
        dot = (g * v).sum()
        return torch.ones_like(g) if dot < -eps else torch.zeros_like(g)
    if mode == "random":
        # density matched to the coordinate gate unless overridden
        if cfg.random_gate_density is not None:
            dens = cfg.random_gate_density
        else:
            dens = float((g * v < -eps).float().mean())
        r = torch.rand(g.shape, generator=rng_gen, device=g.device)
        return (r < dens).to(g.dtype)
    raise ValueError(f"Unknown gate_mode '{mode}'")


def _sweep_lr(cfg: RefineConfig, sweep: int) -> float:
    """Effective learning rate for a sweep under the configured schedule."""
    if cfg.lr_schedule == "constant" or cfg.steps <= 1:
        return cfg.lr
    t = sweep / (cfg.steps - 1)  # 0 -> 1 over the run
    lo = cfg.lr * cfg.lr_min_frac
    if cfg.lr_schedule == "cosine":
        import math
        return lo + 0.5 * (cfg.lr - lo) * (1.0 + math.cos(math.pi * t))
    if cfg.lr_schedule == "linear":
        return cfg.lr + (lo - cfg.lr) * t
    raise ValueError(f"Unknown lr_schedule '{cfg.lr_schedule}'")


def _compute_update(g: ParamDict, v: ParamDict, cfg: RefineConfig,
                    rng_gen: Optional[torch.Generator],
                    frozen_mask: Optional[ParamDict],
                    lr_eff: Optional[float] = None) -> (ParamDict, ParamDict, dict):
    """Return (u, mask, stats) for one task at the current state. ``lr_eff``
    overrides cfg.lr (per-sweep schedule); defaults to cfg.lr."""
    if lr_eff is None:
        lr_eff = cfg.lr
    # topk gate: global per-task threshold on |g*v| among loss-decreasing coords,
    # computed here (not in _gate) because the threshold spans all tensors.
    topk_thr = None
    if cfg.gate_mode == "topk" and frozen_mask is None:
        neg = torch.cat([torch.clamp(-(g[n] * v[n]), min=0).reshape(-1) for n in g])
        total = neg.numel()
        k = max(1, int(round(cfg.topk_frac * total)))
        topk_thr = float(torch.kthvalue(neg, total - k + 1).values)
        del neg
    u = OrderedDict()
    masks = OrderedDict()
    ap_sum = 0.0
    gated = 0
    total = 0
    clipped = 0  # gated coords whose step hit the +/- |v| boundary
    for name in g:
        gi, vi = g[name], v[name]
        if frozen_mask is not None:
            m = frozen_mask[name]
        elif cfg.gate_mode == "topk":
            prod = gi * vi
            m = ((prod < 0) & (-prod >= topk_thr)).to(gi.dtype)
        elif cfg.gate_mode == "topk_g":
            # significance filter on |g| (the only noisy factor), per-tensor
            # quantile; toward-expert prior stays sign(g*v)<0.
            absg = gi.abs()
            total_t = absg.numel()
            k = max(1, int(round(cfg.topk_frac * total_t)))
            thr = torch.kthvalue(absg.reshape(-1), total_t - k + 1).values
            m = ((gi * vi < 0) & (absg >= thr)).to(gi.dtype)
        else:
            m = _gate(gi, vi, cfg, rng_gen)
        u_raw = _base_update(gi, vi, cfg.update_mode) * m
        if cfg.clip_mode == "vdist":
            # The trust region: clip AFTER scaling by lr, so the move is bounded
            # by |v| for ANY lr and cannot overshoot the expert.
            bound = vi.abs()
            pre = lr_eff * u_raw
            u_t = torch.clamp(pre, min=-bound, max=bound)
        elif cfg.clip_mode == "none":
            pre = None
            u_t = u_raw
        else:
            raise ValueError(
                f"Unknown clip_mode '{cfg.clip_mode}' (expected 'vdist' or "
                f"'none'). The legacy clip-before-lr variant, formerly "
                f"'vdist', was removed; 'vdist_pre' is now spelled 'vdist'.")
        if pre is not None:
            # count coords pushed to the trust-region boundary (only gated ones
            # can be, since masked coords have u_raw=0 <= bound)
            clipped += int(((pre.abs() > bound) & (m > 0)).sum().item())
        if cfg.rms_normalize:
            rms = torch.sqrt((u_t ** 2).mean())
            if rms > 0:
                u_t = u_t / rms
        u[name] = u_t
        masks[name] = m
        ap_sum += float((gi * vi).sum())
        # count via bool->int64: float-dtype m.sum() accumulates in fp32 and
        # loses integer precision on >2^24-element tensors (logged density 1.0001
        # for an all-ones mask on Llama's 394M-param tied embedding)
        gated += int((m > 0).sum().item())
        total += m.numel()
    stats = {"ap_sum": ap_sum, "gate_density": gated / max(total, 1),
             "clipped_frac_gated": clipped / max(gated, 1),
             "clipped_frac_all": clipped / max(total, 1)}
    return u, masks, stats


def refine(base_merged: ParamDict, handles: List[TaskHandle], cfg: RefineConfig,
           device: str, seed: int = 0,
           move_model: bool = True,
           logger=None,
           checkpoint_callback: Optional[Callable[[int, ParamDict], None]] = None,
           start_sweep: int = 0
           ) -> (ParamDict, List[dict]):
    """Run S sweeps of refinement. Returns ``(refined_encoder, history)``.

    When provided, ``checkpoint_callback`` is called after each complete sweep
    with the one-based sweep count and current parameter state.  The callback
    must clone any state it wants to retain because refinement continues in
    place.  This supports exact checkpoint reuse for constant-schedule sweeps
    without changing the existing return type or other callers.

    ``start_sweep`` CONTINUES a constant-schedule trajectory: pass the state a
    previous call returned after ``start_sweep`` sweeps as ``base_merged``, and
    this call runs sweeps ``start_sweep .. start_sweep+cfg.steps-1`` exactly as
    the tail of one longer run would have. Equivalence holds because the only
    stochastic ingredient consumed per sweep is the task-order draw, whose
    stream is advanced by ``start_sweep`` burnt shuffles below (the torch
    generator is consumed only by random-gate modes, which do not support
    continuation), and because with a constant learning rate ``_sweep_lr`` does
    not depend on the sweep index. Horizon-dependent schedules must not use it.
    """
    if start_sweep and cfg.lr_schedule != "constant":
        raise ValueError("start_sweep requires a constant lr schedule")
    if start_sweep and cfg.gate_mode.startswith("random"):
        raise ValueError("start_sweep cannot reproduce random-gate draws")
    if start_sweep and cfg.freeze_first_gates:
        raise ValueError("start_sweep cannot reproduce sweep-0 frozen gates")
    enc_names = list(base_merged.keys())
    theta = pd_to(pd_clone(base_merged), device)
    by_name = {h.name: h for h in handles}
    order_names = [h.name for h in handles]
    py_rng = random.Random(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    frozen_masks: Dict[str, ParamDict] = {}
    history: List[dict] = []

    # advance the order stream to where the previous segment left it: one
    # shuffle of an equal-length list consumes exactly the same RNG draws
    if cfg.order == "random":
        for _ in range(start_sweep):
            py_rng.shuffle(list(order_names))

    for s in range(start_sweep, start_sweep + cfg.steps):
        lr_eff = _sweep_lr(cfg, s)
        # for the trust region, lr is already folded into the clipped update
        # inside _compute_update, so it is applied with unit step here; the
        # unconstrained baseline ("none") still needs the lr applied.
        apply_alpha = 1.0 if cfg.clip_mode == "vdist" else lr_eff
        sweep_order = _order_for_sweep(order_names, cfg.order, s, py_rng)
        # snapshot for aggregated-U mode (all updates computed at theta^(s))
        snapshot = pd_clone(theta) if cfg.aggregated else None
        agg_u: Optional[ParamDict] = None

        for k, name in enumerate(sweep_order):
            h = by_name[name]
            state = snapshot if cfg.aggregated else theta
            if move_model:
                h.model.to(device)
            load_encoder_state(h.model, state)
            g = h.grad_fn()  # encoder grads at `state`
            # free the model off-GPU now: the gate/update math below needs only the
            # param-dicts (g, v, u, theta), not the model. At 3B/fp32 each of these
            # is ~12 GB, so keeping the model resident too overflows the card.
            if move_model:
                h.model.to("cpu")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            expert = pd_to(h.expert_encoder, device)
            v = OrderedDict((n, expert[n] - state[n]) for n in enc_names)
            del expert  # only needed to form v; free its GPU copy before _compute_update

            use_frozen = cfg.freeze_first_gates and s > 0
            frozen = frozen_masks.get(name) if use_frozen else None
            u, masks, stats = _compute_update(g, v, cfg, gen, frozen, lr_eff=lr_eff)
            if cfg.freeze_first_gates and s == 0:
                frozen_masks[name] = OrderedDict((n, masks[n].clone()) for n in masks)

            if cfg.aggregated:
                if agg_u is None:
                    agg_u = OrderedDict((n, u[n].clone()) for n in u)
                else:
                    for n in u:
                        agg_u[n].add_(u[n])
            else:
                for n in enc_names:
                    theta[n].add_(u[n], alpha=apply_alpha)

            raw_norm = pd_global_norm(u)
            # the actually-applied step is apply_alpha*u (apply_alpha=lr for the
            # standard path, 1.0 for vdist_pre where lr is already inside u). The
            # raw |u| is NOT comparable across families with different lr.
            stats.update({"sweep": s, "task": name, "update_norm": raw_norm,
                          "step_norm": apply_alpha * raw_norm, "lr_eff": lr_eff})
            history.append(stats)
            if logger:
                logger(f"  sweep {s} task {name}: gate_density={stats['gate_density']:.4f} "
                       f"clip_gated={stats['clipped_frac_gated']:.4f} "
                       f"clip_all={stats['clipped_frac_all']:.4f} "
                       f"ap_sum={stats['ap_sum']:.4g} step|lr*u|={stats['step_norm']:.4g} "
                       f"(raw|u|={raw_norm:.4g})")
            # release this task's large GPU param-dicts before the next task's
            # grad_fn allocates; otherwise at 3B/fp32 they accumulate across the
            # sweep and OOM the card (a single task fits; the pile-up does not).
            # u is already folded into theta/agg_u; masks is cloned into
            # frozen_masks above when needed. The model was moved to CPU above.
            del g, v, u, masks
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if cfg.aggregated and agg_u is not None:
            for n in enc_names:
                theta[n].add_(agg_u[n], alpha=apply_alpha)

        if checkpoint_callback is not None:
            checkpoint_callback(s + 1, theta)

    return theta, history
