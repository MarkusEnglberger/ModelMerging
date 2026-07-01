"""Replay-buffer task-gradient computation.

g_i = grad_theta L_i(theta; D_probe_i), restricted to the mergeable encoder
parameters, evaluated with task i's (fixed) head. We compute the gradient of the
*mean* loss over the whole replay buffer via gradient accumulation, and run in
eval() mode so dropout does not add noise to the attribution gate.
"""

from collections import OrderedDict
from typing import Callable, Dict, List

import torch

from .models import encoder_param_names


def make_grad_fn(model, buffer: List[Dict], collator, batch_size: int,
                 device: str) -> Callable[[], Dict[str, torch.Tensor]]:
    """Build a closure computing encoder grads at the model's current weights."""
    enc_names = set(encoder_param_names(model))
    n_total = len(buffer)

    # pre-pad batches once (labels/tensors are cheap to keep on CPU and move)
    raw_batches = []
    for start in range(0, n_total, batch_size):
        chunk = buffer[start: start + batch_size]
        raw_batches.append((collator(chunk), len(chunk)))

    def grad_fn() -> Dict[str, torch.Tensor]:
        model.eval()
        model.zero_grad(set_to_none=True)
        for batch_cpu, bn in raw_batches:
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
            out = model(**batch)
            # scale so accumulated grad == grad of mean-over-buffer loss
            (out.loss * (bn / n_total)).backward()
        grads = OrderedDict()
        for n, p in model.named_parameters():
            if n in enc_names:
                grads[n] = (p.grad.detach().clone() if p.grad is not None
                            else torch.zeros_like(p))
        model.zero_grad(set_to_none=True)
        return grads

    return grad_fn
