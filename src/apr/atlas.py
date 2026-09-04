"""aTLAS: knowledge composition with learned anisotropic scaling.

Zhang et al., NeurIPS 2024 (github.com/fredzzhang/atlas). A coefficient is
learned per (task, parameter block) by gradient descent on a *supervised*
objective, so the merged encoder is

    theta = theta_0 + sum_i sum_b  coef[i, b] * tau_i[b].

The released implementation (``src/composition.py::WeightedImageEncoder``,
``src/learn_coef.py::train``) initialises ``coef`` at zero, optimises with
AdamW at lr 1e-3 and weight decay 0.1 under cross-entropy, and optionally adds
an Lp penalty ``gamma * ||coef||_p`` averaged over blocks.

Structurally this is AdaMerging with a different objective and initialisation:
both learn one coefficient per (task, block) and both need the gradient of a
loss with respect to those coefficients, obtained by contracting the encoder
gradient against the task vectors. We therefore reuse AdaMerging's builder and
contraction rather than reimplementing them, which keeps the two methods
numerically comparable and confines the difference to the objective.

The label budget is the caller's: ``data_key`` names the per-task buffer the
coefficients are fitted on, and under protocol v3 that is the draw's B labeled
examples, the same budget every other method is charged for.
"""

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F

from .adamerging import _build_theta, _contract_lambda_grad, _infinite_batches
from .models import ParamDict, load_encoder_state


def _lp_reg(coef: torch.Tensor, p, gamma: float = 0.5) -> torch.Tensor:
    """Released ``lp_reg``: gamma * mean over blocks of the p-norm over tasks."""
    if p is None:
        return torch.zeros((), device=coef.device)
    return gamma * torch.norm(coef, p=p, dim=0).mean()


def atlas(base_encoder: ParamDict, task_vectors: Dict[str, ParamDict],
          per_task: Dict[str, dict], task_names: List[str], device: str,
          blockwise: bool = True, steps: int = 300, lr: float = 1e-3,
          weight_decay: float = 0.1, batch_size: int = 16,
          init_coef: float = 0.0, lp: float = None, lp_gamma: float = 0.5,
          seed: int = 0, num_workers: int = 0, log_every: int = 50,
          data_key: str = "cv_buffer", tv_on_gpu: bool = True,
          logger=None) -> (ParamDict, dict):
    """Learn anisotropic task-vector coefficients on a LABELED buffer.

    Returns ``(merged_encoder_cpu, info)``.
    """
    names = list(task_names)
    keys = list(base_encoder.keys())
    T, K = len(names), len(keys)
    enc_names = set(keys)

    shape = (T, K) if blockwise else (T,)
    coef = torch.full(shape, float(init_coef), dtype=torch.float32)
    coef.requires_grad_(True)
    opt = torch.optim.AdamW([coef], lr=lr, weight_decay=weight_decay)

    tv_dev = task_vectors
    if tv_on_gpu and device.startswith("cuda"):
        try:
            tv_dev = {n: OrderedDict((k, v.to(device)) for k, v in tv.items())
                      for n, tv in task_vectors.items()}
        except RuntimeError:                       # not enough room; stream
            tv_dev = task_vectors

    # every task contributes a supervised loss; regression heads use MSE
    fit_tasks = list(names)
    if logger:
        sizes = {n: len(per_task[n][data_key]) for n in fit_tasks}
        logger(f"[atlas] labeled source='{data_key}' sizes={sizes} "
               f"blockwise={blockwise} steps={steps} lr={lr} wd={weight_decay}")
    streams = {n: _infinite_batches(per_task[n][data_key], per_task[n]["collator"],
                                    batch_size, seed + i, num_workers)
               for i, n in enumerate(fit_tasks)}

    trace = []
    for step in range(steps):
        theta = _build_theta(base_encoder, tv_dev, coef.detach(), names, keys,
                             blockwise, device)
        grad_theta = OrderedDict((k, torch.zeros_like(theta[k])) for k in keys)
        total = 0.0
        for n in fit_tasks:
            info = per_task[n]
            model = info["model"]
            model.to(device)
            model.eval()
            load_encoder_state(model, theta)
            model.zero_grad(set_to_none=True)

            batch = next(streams[n])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)                   # the head supplies the loss
            loss = out.loss
            loss.backward()
            total += float(loss)

            for pn, p in model.named_parameters():
                if pn in enc_names and p.grad is not None:
                    grad_theta[pn].add_(p.grad)
            model.zero_grad(set_to_none=True)
            model.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        g = _contract_lambda_grad(grad_theta, tv_dev, names, keys, blockwise,
                                  device)
        coef.grad = g.to(coef.device)
        if lp is not None:                          # penalty is a function of
            reg = _lp_reg(coef, lp, lp_gamma)       # coef alone, so add its
            reg.backward()                          # gradient directly
        opt.step()
        opt.zero_grad(set_to_none=True)

        trace.append({"step": step, "loss": total / max(len(fit_tasks), 1),
                      "coef_mean": float(coef.detach().mean())})
        if logger and (step % log_every == 0 or step == steps - 1):
            logger(f"  [atlas] step {step:4d}/{steps} "
                   f"loss={trace[-1]['loss']:.4f} "
                   f"coef_mean={trace[-1]['coef_mean']:.4f}")
        del theta, grad_theta

    final = _build_theta(base_encoder, tv_dev, coef.detach(), names, keys,
                         blockwise, device)
    merged_cpu = OrderedDict((k, v.detach().cpu()) for k, v in final.items())
    c = coef.detach().cpu()
    per_task_coef = ({n: float(c[i]) for i, n in enumerate(names)} if not blockwise
                     else {n: float(c[i].mean()) for i, n in enumerate(names)})
    info = {"method": "aTLAS", "blockwise": blockwise, "steps": steps, "lr": lr,
            "weight_decay": weight_decay, "init_coef": init_coef,
            "lp": lp, "lp_gamma": lp_gamma, "batch_size": batch_size,
            "data_key": data_key,
            "n_labeled_per_task": {n: len(per_task[n][data_key]) for n in fit_tasks},
            "coef_per_task": per_task_coef,
            "coef_full": c.tolist() if blockwise else None,
            "final_loss": trace[-1]["loss"] if trace else None,
            "first_loss": trace[0]["loss"] if trace else None,
            "implementation_reference": "https://github.com/fredzzhang/atlas"}
    return merged_cpu, info
