"""Supervised per-tensor coefficient calibration (the init for calibrate-then-refine).

This is the supervised twin of AdaMerging (adamerging.py). It uses the SAME
parameterization -- one coefficient lambda_i^l per (task, encoder tensor), so

    theta(lambda) = theta_0 + sum_i lambda_i^l tau_i^l

-- and the SAME manual-gradient trick (theta is linear in lambda, so
dL/dlambda_i^l = <dL/dtheta^l, tau_i^l>, no autograd graph over lambda). The only
change is the objective: labeled cross-entropy / task loss on the replay buffer,
instead of unlabeled prediction entropy.

Why the swap. AdaMerging's entropy objective has a confident-but-wrong minimum whenever
the task head is a trained classifier (it collapses on GLUE: entropy falls to 0.16 while
accuracy drops). A labeled objective cannot: lowering it requires being correct, so it
gives a usable init on both modalities. And we are already spending 64 labeled examples
per task on the refinement, so using them here too is free signal.

The risk is the mirror image: with labels there IS an answer to memorize, and ~10^3
coefficients on 64 examples/task can overfit (the per-task-lambda baseline already did).
We therefore (i) regularize the coefficients toward the initial value with an L2 penalty,
and (ii) support early stopping on a held-out slice of the buffer. Both default on.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from .models import ParamDict, load_encoder_state
from .adamerging import _build_theta, _contract_lambda_grad, _infinite_batches
from .data import batches_from_buffer


def _task_loss_and_grad(model, theta, batch, enc_names, device) -> Tuple[float, ParamDict]:
    """Loss on one labeled batch at encoder state `theta`, and dL/dtheta (encoder)."""
    model.to(device)
    model.eval()  # frozen head + merged encoder; no dropout, matches gradients.py
    load_encoder_state(model, theta)
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    grad = OrderedDict((n, p.grad.detach().clone() if p.grad is not None
                        else torch.zeros_like(p))
                       for n, p in model.named_parameters() if n in enc_names)
    model.zero_grad(set_to_none=True)
    return float(loss), grad


def calibrate(base_encoder: ParamDict, task_vectors: Dict[str, ParamDict],
              per_task: Dict[str, dict], task_names: List[str], device: str,
              layerwise: bool = True, steps: int = 300, lr: float = 1e-2,
              batch_size: int = 16, init_lam: float = 0.3, l2_reg: float = 1e-2,
              seed: int = 0, holdout_frac: float = 0.25, patience: int = 5,
              logger=None) -> Tuple[ParamDict, dict]:
    """Fit per-tensor (layerwise) or per-task coefficients by minimizing the labeled
    replay loss + l2_reg * ||lambda - init_lam||^2. Returns (merged_encoder_cpu, info).

    Early stopping: if holdout_frac>0, each task's buffer is split into a fit slice and a
    held-out slice; training stops when mean held-out loss has not improved for `patience`
    evaluations. This is the overfitting guard for the labeled objective."""
    names = list(task_names)
    keys = list(base_encoder.keys())
    T, K = len(names), len(keys)
    enc_names = set(keys)

    tv_dev = {n: OrderedDict((k, v.to(device)) for k, v in tv.items())
              for n, tv in task_vectors.items()}

    # split each task's replay buffer into fit / holdout
    fit_buf, hold_buf = {}, {}
    for n in names:
        buf = per_task[n]["probe_buffer"]
        nh = int(round(holdout_frac * len(buf))) if holdout_frac > 0 else 0
        hold_buf[n] = buf[:nh]
        fit_buf[n] = buf[nh:] if nh < len(buf) else buf  # never leave fit empty
    collators = {n: per_task[n]["collator"] for n in names}
    fit_streams = {n: _infinite_batches(fit_buf[n], collators[n], batch_size, seed + i)
                   for i, n in enumerate(names)}

    shape = (T, K) if layerwise else (T,)
    lam = torch.full(shape, float(init_lam), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([lam], lr=lr)

    def holdout_loss():
        if holdout_frac <= 0:
            return None
        theta = _build_theta(base_encoder, tv_dev, lam.detach(), names, keys, layerwise, device)
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for n in names:
                if not hold_buf[n]:
                    continue
                for batch in batches_from_buffer(hold_buf[n], collators[n], batch_size, device):
                    per_task[n]["model"].to(device).eval()
                    load_encoder_state(per_task[n]["model"], theta)
                    bn = int(batch["labels"].shape[0])
                    tot += float(per_task[n]["model"](**batch).loss) * bn
                    ntok += bn
                    per_task[n]["model"].to("cpu")
        return tot / max(ntok, 1)

    trace = []
    best_hold, best_lam, bad = float("inf"), lam.detach().clone(), 0
    for step in range(steps):
        theta = _build_theta(base_encoder, tv_dev, lam.detach(), names, keys, layerwise, device)
        grad_theta = OrderedDict((k, torch.zeros_like(theta[k])) for k in keys)
        total = 0.0
        for n in names:
            batch = next(fit_streams[n])
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, g = _task_loss_and_grad(per_task[n]["model"], theta, batch, enc_names, device)
            total += loss
            for k in keys:
                grad_theta[k].add_(g[k])
            per_task[n]["model"].to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        glam = _contract_lambda_grad(grad_theta, tv_dev, names, keys, layerwise, device)
        glam = glam.to(lam.device) + 2.0 * l2_reg * (lam.detach() - init_lam)  # + d/dlam L2
        lam.grad = glam
        opt.step()
        opt.zero_grad(set_to_none=True)

        if holdout_frac > 0 and (step % 10 == 0 or step == steps - 1):
            hl = holdout_loss()
            trace.append({"step": step, "fit_loss": total / T, "hold_loss": hl})
            if hl < best_hold - 1e-4:
                best_hold, best_lam, bad = hl, lam.detach().clone(), 0
            else:
                bad += 1
            if logger:
                logger(f"  [calibrate] step {step:4d} fit={total/T:.4f} hold={hl:.4f} "
                       f"lam_mean={float(lam.detach().mean()):.4f} best_hold={best_hold:.4f}")
            if bad >= patience:
                logger and logger(f"  [calibrate] early stop at step {step} "
                                  f"(no holdout improvement for {patience} evals)")
                break
        else:
            trace.append({"step": step, "fit_loss": total / T})

    use_lam = best_lam if holdout_frac > 0 else lam.detach()
    final = _build_theta(base_encoder, tv_dev, use_lam, names, keys, layerwise, device)
    merged_cpu = OrderedDict((k, v.detach().cpu()) for k, v in final.items())
    lam_c = use_lam.cpu()
    lam_per_task = ({n: float(lam_c[i]) for i, n in enumerate(names)} if not layerwise
                    else {n: float(lam_c[i].mean()) for i, n in enumerate(names)})
    info = {"layerwise": layerwise, "steps_run": len(trace), "lr": lr, "l2_reg": l2_reg,
            "init_lam": init_lam, "holdout_frac": holdout_frac,
            "lam_per_task": lam_per_task, "best_holdout_loss": best_hold,
            "fit_loss_first": trace[0]["fit_loss"] if trace else None,
            "fit_loss_last": trace[-1]["fit_loss"] if trace else None}
    return merged_cpu, info
