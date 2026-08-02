"""Localize-and-Stitch (He et al. 2024, arXiv:2408.13656).

Localize: for each task i, learn a sparse binary mask m_i over its task vector tau_i by
minimising the task loss on a small LABELED sample, with an L1 penalty pushing the mask
toward zero:

    min_S  loss_i(theta_0 + sigmoid(S) * tau_i ; D_i^probe)  +  gamma * ||sigmoid(S)||_1

Stitch: theta = theta_0 + sum_i m_i * tau_i, with m_i the rounded (binary) mask.

This is the closest published method to our attribution gate: both are data-driven
coordinate selectors over the expert directions. The difference is that L&S *learns* the
mask by gradient descent over many steps, while APR recomputes a first-order sign gate at
every update. Comparing them at a matched replay budget is the sharpest test of the gate.

Gradient trick (as in adamerging.py): theta is a smooth elementwise function of S, so

    d/dS [ loss + gamma*||sigmoid(S)||_1 ]
        = ( dloss/dtheta * tau + gamma ) * sigmoid(S) * (1 - sigmoid(S))

(the L1 term needs no abs() because sigmoid(S) >= 0). We therefore take one backward pass
w.r.t. theta and chain manually -- no autograd graph over S, memory stays O(|theta|).

Sparsity control: rather than relying on gamma to land on a target sparsity, we train with
the L1 penalty and then keep the top-`sparsity` fraction of sigmoid(S) globally. This makes
sparsity an explicit, comparable hyperparameter (the paper reports ~1% for the data-driven
variant and ~5% for the dataless one).

`dataless_masks` reproduces the paper's data-free variant: top-k% by |tau| magnitude.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from .data import batches_from_buffer
from .models import ParamDict, load_encoder_state, pd_clone


def _topk_mask(scores: ParamDict, fraction: float) -> ParamDict:
    """Binary mask keeping exactly the top fraction of coordinates globally.

    Selecting by ``value >= kth_value`` can retain vastly more than requested
    when learned sigmoid scores saturate and tie.  Explicit flat indices make
    the cardinality exact (apart from the requested rounding of ``k``).
    """
    allv = torch.cat([v.reshape(-1) for v in scores.values()])
    n = allv.numel()
    k = int(round(fraction * n))
    if k >= n:
        return OrderedDict((kk, torch.ones_like(v, dtype=torch.bool)) for kk, v in scores.items())
    if k <= 0:
        return OrderedDict((kk, torch.zeros_like(v, dtype=torch.bool)) for kk, v in scores.items())
    indices = torch.topk(allv, k, sorted=False).indices
    flat_mask = torch.zeros(n, dtype=torch.bool, device=allv.device)
    flat_mask[indices] = True
    out = OrderedDict()
    start = 0
    for name, value in scores.items():
        end = start + value.numel()
        out[name] = flat_mask[start:end].reshape(value.shape)
        start = end
    return out


def dataless_masks(task_vectors: Dict[str, ParamDict],
                   sparsity: float = 0.05) -> Dict[str, ParamDict]:
    """Data-free L&S variant: keep the top-`sparsity` fraction of |tau_i| per task."""
    return {n: _topk_mask(OrderedDict((k, v.abs()) for k, v in tau.items()), sparsity)
            for n, tau in task_vectors.items()}


def learn_mask(model, buffer, collator, base_encoder: ParamDict, tau: ParamDict,
               device: str, steps: int = 200, lr: float = 0.1, gamma: float = 1e-4,
               batch_size: int = 16, init_logit: float = 0.0,
               init_sparsity: Optional[float] = None, init_on: float = 3.0,
               optimizer: str = "adam", move_model: bool = True,
               logger=None, tag: str = "") -> ParamDict:
    """Optimise sigmoid mask logits S for one task; returns sigmoid(S) (CPU floats)."""
    keys = list(base_encoder.keys())
    base_d = {k: base_encoder[k].to(device) for k in keys}
    tau_d = {k: tau[k].to(device) for k in keys}
    if init_sparsity is None:
        S = [torch.full_like(tau_d[k], float(init_logit), requires_grad=True) for k in keys]
    else:
        initial_masks = _topk_mask(
            OrderedDict((k, tau_d[k].abs()) for k in keys), init_sparsity)
        S = [torch.where(initial_masks[k],
                         torch.full_like(tau_d[k], float(init_on)),
                         torch.full_like(tau_d[k], float(init_logit))).requires_grad_()
             for k in keys]
    if optimizer == "adam":
        opt = torch.optim.Adam(S, lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(S, lr=lr)
    elif optimizer == "official":
        opt = None
    else:
        raise ValueError(f"Unknown localization optimizer '{optimizer}'")

    step, last = 0, None
    if optimizer == "official":
        # Faithful to uiuctml/Localize-and-Stitch's train_graft: one mask
        # update per epoch, with loss gradients summed over all minibatches at
        # the same interpolated model. Their manual update is plain SGD and the
        # sign regularizer pushes logits away from the rounding boundary.
        while step < steps:
            with torch.no_grad():
                sig = [torch.sigmoid(s) for s in S]
                theta = OrderedDict((k, base_d[k] + sig[j] * tau_d[k])
                                    for j, k in enumerate(keys))
            if move_model:
                model.to(device)
            model.eval()
            load_encoder_state(model, theta)
            accumulated = [torch.zeros_like(tau_d[k]) for k in keys]
            for batch in batches_from_buffer(buffer, collator, batch_size, device):
                model.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                grads = {n: p.grad for n, p in model.named_parameters()
                         if p.grad is not None}
                with torch.no_grad():
                    for j, k in enumerate(keys):
                        if k in grads:
                            accumulated[j].add_(grads[k] * tau_d[k])
                last = float(loss)
            with torch.no_grad():
                for j, s in enumerate(S):
                    derivative = sig[j] * (1.0 - sig[j])
                    regularizer = gamma * torch.where(
                        s > 0, derivative, -derivative)
                    s.sub_(lr * accumulated[j] * derivative - regularizer)
            model.zero_grad(set_to_none=True)
            step += 1

    while step < steps:
        for batch in batches_from_buffer(buffer, collator, batch_size, device):
            if step >= steps:
                break
            with torch.no_grad():
                sig = [torch.sigmoid(s) for s in S]
                theta = OrderedDict((k, base_d[k] + sig[j] * tau_d[k])
                                    for j, k in enumerate(keys))
            if move_model:
                model.to(device)
            model.eval()
            load_encoder_state(model, theta)
            model.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()

            grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            with torch.no_grad():
                for j, k in enumerate(keys):
                    g = grads.get(k)
                    dtheta = torch.zeros_like(tau_d[k]) if g is None else g
                    # d/dS [loss + gamma*||sigmoid(S)||_1]
                    S[j].grad = (dtheta * tau_d[k] + gamma) * sig[j] * (1.0 - sig[j])
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            last = float(loss)
            step += 1

    with torch.no_grad():
        out = OrderedDict((k, torch.sigmoid(S[j]).detach().cpu())
                          for j, k in enumerate(keys))
        mean_sig = float(torch.cat([v.reshape(-1) for v in out.values()]).mean())
    if logger:
        logger(f"  [localize] {tag}: loss={last:.4f} mean_sigmoid={mean_sig:.4f}")
    if move_model:
        model.to("cpu")
    if device.startswith("cuda") and move_model:
        torch.cuda.empty_cache()
    del S, base_d, tau_d
    return out


def stitch(base_encoder: ParamDict, task_vectors: Dict[str, ParamDict],
           masks: Dict[str, ParamDict], lam: float = 1.0,
           average_overlaps: bool = False) -> Tuple[ParamDict, dict]:
    """theta = theta_0 + lam * sum_i m_i * tau_i.

    With ~1% masks, overlaps are rare; `average_overlaps` divides each coordinate by the
    number of masks covering it (reported either way)."""
    keys = list(base_encoder.keys())
    merged = pd_clone(base_encoder)
    cover = OrderedDict((k, torch.zeros_like(base_encoder[k])) for k in keys)
    acc = OrderedDict((k, torch.zeros_like(base_encoder[k])) for k in keys)
    for n, tau in task_vectors.items():
        m = masks[n]
        for k in keys:
            acc[k].add_(tau[k] * m[k])
            cover[k].add_(m[k].to(cover[k].dtype))
    tot = sum(c.numel() for c in cover.values())
    covered = sum(float((c > 0).sum()) for c in cover.values())
    overlap = sum(float((c > 1).sum()) for c in cover.values())
    if average_overlaps:
        for k in keys:
            acc[k].div_(cover[k].clamp(min=1))
    for k in keys:
        merged[k].add_(acc[k], alpha=lam)
    return merged, {"covered_frac": covered / tot, "overlap_frac": overlap / tot,
                    "lam": lam, "average_overlaps": average_overlaps}


def learn_sigmoids(ctx, steps: int = 200, lr: float = 0.1, gamma: float = 1e-4,
                   batch_size: int = 16, init_logit: float = 0.0,
                   init_sparsity: Optional[float] = None, init_on: float = 3.0,
                   optimizer: str = "adam", logger=None) -> Dict[str, ParamDict]:
    """Train the mask logits for every task and return sigmoid(S) per task.

    Sparsity is applied afterwards by :func:`_topk_mask`, so ONE training run serves an
    entire sparsity grid -- do not retrain per sparsity."""
    out = {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        out[n] = learn_mask(info["model"], info["probe_buffer"], info["collator"],
                            ctx.base_encoder, ctx.task_vectors[n], ctx.device,
                            steps=steps, lr=lr, gamma=gamma, batch_size=batch_size,
                            init_logit=init_logit, init_sparsity=init_sparsity,
                            init_on=init_on, optimizer=optimizer,
                            move_model=not ctx.keep_model_on_device,
                            logger=logger, tag=n)
    return out


def masks_from_sigmoids(sigmoids: Dict[str, ParamDict],
                        sparsity: float) -> Dict[str, ParamDict]:
    return {n: _topk_mask(sig, sparsity) for n, sig in sigmoids.items()}


def threshold_masks(sigmoids: Dict[str, ParamDict],
                    threshold: float = 0.5) -> Dict[str, ParamDict]:
    """Paper-style rounding of learned sigmoid masks to binary masks."""
    return {n: OrderedDict((k, value > threshold) for k, value in sig.items())
            for n, sig in sigmoids.items()}
