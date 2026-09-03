"""TATR: Task Arithmetic in Trust Region (Sun et al., ICLR 2025).

Ported faithfully from the released TATR_merging code; previously lived in
scripts/cv_protocol.py and moved here so the held-out retention probe can
build the same merge. The ``names`` parameter restricts the construction to a
subset of tasks (the probe merges TRAIN tasks only); the protocol runner
passes the full task list.
"""

from typing import Dict, List, Optional

import torch

from .data import batches_from_buffer
from .models import ParamDict, load_encoder_state, pd_clone
from .pipeline import _log


def tatr_omega(ctx, buffers: Dict[str, list],
               names: Optional[List[str]] = None) -> ParamDict:
    """TATR's conflict scores.

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
    names = list(names) if names is not None else list(ctx.task_names)
    grads = {}
    for n in names:
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
    for i in names:
        for j in names:
            if i == j:
                continue
            tv = ctx.task_vectors[j]
            for k in omega:
                omega[k] += grads[i][k] * tv[k].abs().cpu()
    return omega


def tatr_mask(omega: ParamDict, ratio: float) -> Dict[str, torch.Tensor]:
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


def tatr_merge(base_encoder: ParamDict, task_vectors: Dict[str, ParamDict],
               mask: Dict[str, torch.Tensor], lam: float,
               names: Optional[List[str]] = None) -> ParamDict:
    """Task arithmetic restricted to the trust region: tau coordinates where
    the mask is False are dropped; keys absent from the mask pass unmasked."""
    names = list(names) if names is not None else list(task_vectors)
    state = pd_clone(base_encoder)
    for n in names:
        tv = task_vectors[n]
        for k, v in tv.items():
            m = mask.get(k)
            upd = v * m.to(v.dtype) if m is not None else v
            state[k] = state[k] + lam * upd
    return state
