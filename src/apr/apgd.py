"""DOGE / adaptive projective gradient descent (Wei et al., ICML 2025).

This is a parameter-dictionary port of the authors' released ``DOGE_TA``
implementation.  It deliberately preserves the published implementation
choices that matter for a baseline comparison:

* operate only on two-dimensional linear weights inside the encoder;
* use layer-wise, task-aware coefficients ``lambda_i^l = eta / ||tau_i^l||``;
* learn one shared modification matrix ``Delta^l`` per layer for 400 Adam
  iterations at learning rate 1e-4;
* remove the component of every Delta gradient that lies in the shared
  task-vector subspace; and
* retain the coordinates selected by the global top 30 percent of each
  original task vector, then apply that mask to its adjusted task vector before
  merging it into the pretrained model.

The released code materialises a projection matrix ``Q Q^T``.  We instead
apply it associatively as ``Q (Q^T g)``; this is algebraically identical and
avoids an unnecessary square matrix for wide transformer layers.  Likewise,
each layer receives its own Adam instance.  Since the objective is separable
by layer and inactive layers never acquire optimizer state in the released
implementation, this produces the same per-layer optimization trajectory while
keeping GPU memory bounded.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from .models import ParamDict, pd_clone


TaskVectors = Mapping[str, ParamDict]


@dataclass(frozen=True)
class APGDPreparation:
    """Shared-subspace bases that are independent of the global scale ``eta``."""

    keys: Tuple[str, ...]
    bases: Dict[str, torch.Tensor]
    task_names: Tuple[str, ...]
    subspace_divisor: int


def apgd_linear_keys(task_vectors: TaskVectors) -> Tuple[str, ...]:
    """Return the parameters targeted by the official encoder implementation.

    DOGE's released filter is ``"encoder" in name``, ``"weight" in name`` and
    not a layer norm.  Requiring a matrix additionally makes the intended
    "linear layers only" restriction explicit and lets the same filter work for
    RoBERTa, whose LayerNorm spelling differs from the released CLIP/T5 code.
    """
    if not task_vectors:
        return ()
    first = next(iter(task_vectors.values()))
    keys = []
    for name, value in first.items():
        lower = name.lower()
        if (value.ndim == 2 and "encoder" in lower and "weight" in lower
                and "layer_norm" not in lower and "layernorm" not in lower):
            keys.append(name)
    return tuple(keys)


@torch.no_grad()
def prepare_apgd(task_vectors: TaskVectors, device: str,
                 subspace_divisor: int = 6,
                 keys: Optional[Sequence[str]] = None,
                 logger=None) -> APGDPreparation:
    """Build the shared left-singular-vector basis for every selected layer."""
    if subspace_divisor <= 0:
        raise ValueError("subspace_divisor must be positive")
    names = tuple(task_vectors)
    if len(names) < 2:
        raise ValueError("APGD requires at least two task vectors")
    selected = tuple(keys) if keys is not None else apgd_linear_keys(task_vectors)
    if not selected:
        raise ValueError("APGD found no two-dimensional encoder linear weights")

    bases: Dict[str, torch.Tensor] = OrderedDict()
    for index, key in enumerate(selected, 1):
        task_bases = []
        min_rank = min(task_vectors[names[0]][key].shape)  # full_matrices=False rank
        task_rank = max(1, min_rank // len(names))
        for name in names:
            matrix = task_vectors[name][key].detach().to(device=device,
                                                          dtype=torch.float32)
            u = torch.linalg.svd(matrix, full_matrices=False).U
            task_bases.append(u[:, :task_rank])
            del matrix, u
        joined = torch.cat(task_bases, dim=1)
        shared_rank = max(1, min(min_rank // subspace_divisor, joined.shape[1]))
        shared_u = torch.linalg.svd(joined, full_matrices=False).U[:, :shared_rank]
        bases[key] = shared_u.detach().cpu()
        if logger:
            logger(f"[APGD] subspace {index}/{len(selected)} {key}: "
                   f"task-rank={task_rank}, shared-rank={shared_rank}")
        del joined, shared_u, task_bases
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return APGDPreparation(selected, bases, names, subspace_divisor)


def _taskvector_loss(layer_vectors: torch.Tensor, delta: torch.Tensor,
                     lambdas: torch.Tensor) -> torch.Tensor:
    """Equation (5), with the row-wise reduction used by the released code."""
    merged = torch.einsum("t,tij->ij", lambdas, layer_vectors)
    merged = merged + lambdas.sum() * delta
    row_inner_products = (layer_vectors * (layer_vectors - merged)).sum(dim=-1)
    return row_inner_products.square().sum()


@torch.no_grad()
def _remove_shared_component_(gradient: torch.Tensor,
                              basis: torch.Tensor) -> None:
    """In-place ``gradient -= Q Q^T gradient`` for an orthonormal ``Q``."""
    gradient.sub_(basis @ (basis.transpose(0, 1) @ gradient))


def _released_topk_threshold(values: Mapping[str, torch.Tensor],
                             keep_density: float) -> float:
    """Reproduce the released ``topk_values_mask`` threshold exactly.

    The authors use the ``(d - int(d*K))``-th smallest magnitude and retain
    values greater than or equal to it.  That inclusive comparison generally
    keeps ``int(d*K) + 1`` entries in the absence of ties; we preserve the
    implementation's boundary convention instead of silently replacing it
    with an idealized exact-k mask.
    """
    magnitudes = torch.cat([
        value.detach().reshape(-1).abs() for value in values.values()
    ])
    d = magnitudes.numel()
    if keep_density >= 1:
        return -1.0
    kth = d - int(d * keep_density)
    return float(torch.kthvalue(magnitudes, kth).values)


def apgd_merge(base_encoder: ParamDict, task_vectors: TaskVectors, eta: float,
               device: str, preparation: Optional[APGDPreparation] = None,
               iterations: int = 400, lr: float = 1e-4,
               keep_density: float = 0.30, subspace_divisor: int = 6,
               logger=None) -> Tuple[ParamDict, dict]:
    """Run DOGE/APGD and return the merged state plus reproducibility metadata."""
    if eta <= 0:
        raise ValueError("eta must be positive")
    if iterations <= 0 or lr <= 0:
        raise ValueError("iterations and lr must be positive")
    if not 0 < keep_density <= 1:
        raise ValueError("keep_density must be in (0, 1]")
    prep = preparation or prepare_apgd(
        task_vectors, device, subspace_divisor=subspace_divisor, logger=logger)
    if tuple(task_vectors) != prep.task_names:
        raise ValueError("task-vector order differs from the APGD preparation")

    names = prep.task_names
    deltas: Dict[str, torch.Tensor] = OrderedDict()
    layer_lambdas: Dict[str, Dict[str, float]] = {name: {} for name in names}

    for index, key in enumerate(prep.keys, 1):
        layer_vectors = torch.stack([
            task_vectors[name][key].detach().to(device=device, dtype=torch.float32)
            for name in names
        ])
        lambda_values = []
        for task_index, name in enumerate(names):
            norm = float(torch.linalg.vector_norm(layer_vectors[task_index]))
            value = eta / norm if norm > 0 else 0.0
            layer_lambdas[name][key] = value
            lambda_values.append(value)
        lambdas = torch.tensor(lambda_values, device=device, dtype=torch.float32)
        basis = prep.bases[key].to(device=device, dtype=torch.float32)
        delta = torch.nn.Parameter(torch.zeros_like(layer_vectors[0]))
        optimizer = torch.optim.Adam([delta], lr=lr)
        final_loss = None
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = _taskvector_loss(layer_vectors, delta, lambdas)
            loss.backward()
            _remove_shared_component_(delta.grad, basis)
            optimizer.step()
            final_loss = float(loss.detach())
        deltas[key] = delta.detach().cpu().to(base_encoder[key].dtype)
        if logger:
            logger(f"[APGD] optimize {index}/{len(prep.keys)} {key}: "
                   f"loss={final_loss:.6g}")
        del optimizer, delta, layer_vectors, lambdas, basis, loss

    # The released implementation computes each global top-magnitude mask from
    # the *original* task vector, then applies it to (task vector + delta).
    # Keeping that order is important: delta can otherwise change which
    # coordinates survive.
    merged = pd_clone(base_encoder)
    thresholds = {}
    for name in names:
        original = OrderedDict((key, task_vectors[name][key]) for key in prep.keys)
        threshold = _released_topk_threshold(original, keep_density)
        adjusted = OrderedDict(
            (key, task_vectors[name][key] + deltas[key].to(task_vectors[name][key].dtype))
            for key in prep.keys
        )
        thresholds[name] = threshold
        for key, value in adjusted.items():
            mask = value.abs() >= threshold
            merged[key].add_(value * mask, alpha=layer_lambdas[name][key])
        del adjusted

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    info = {
        "method": "DOGE/APGD",
        "eta": eta,
        "iterations": iterations,
        "lr": lr,
        "keep_density": keep_density,
        "subspace_divisor": prep.subspace_divisor,
        "linear_keys": list(prep.keys),
        "n_linear_keys": len(prep.keys),
        "thresholds": thresholds,
        "layer_lambdas": layer_lambdas,
        "implementation_reference":
            "https://github.com/WalkerWorldPeace/DOGE",
    }
    return merged, info
