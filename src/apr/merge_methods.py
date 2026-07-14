"""Checkpoint-only interference-aware merge baselines: TIES and DARE.

These are *data-free* alternatives to the plain task-arithmetic merge
(``taskvec.task_arithmetic_merge``, Eq. 4). They consume the same task vectors
``tau_i = theta_i - theta_0`` and produce a merged encoder
``theta_0 + <combined tau>`` that can be evaluated directly (an S=0, no-replay
baseline) or used as the starting point for APR refinement.

  TIES  (Yadav et al. 2023): trim each tau to its top-``density`` magnitude
        coordinates, elect a per-coordinate sign from the trimmed sum, then
        average only the sign-agreeing entries; scale the result by a single
        lambda.
  DARE  (Yu et al. 2023, "Super Mario"): randomly drop each tau coordinate with
        probability ``1-density`` and rescale the survivors by ``1/density``
        (preserving the expectation), then merge by ordinary task arithmetic
        (DARE-TA) with the per-task lambdas.

All operations are per-tensor and run on whatever device the task vectors live
on (CPU in the current pipeline), so they add no GPU memory pressure. The trim
threshold is a *global* top-k over the whole flattened task vector, matching the
TIES paper (not per-tensor top-k).
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional
import hashlib

import torch

from .models import ParamDict, pd_clone, pd_axpy_

TaskVectors = Dict[str, ParamDict]


def _global_topk_threshold(tau: ParamDict, density: float) -> float:
    """Magnitude threshold above which the top-``density`` fraction of |tau|
    coordinates survive (global over all coordinates of the task vector)."""
    allabs = torch.cat([v.detach().reshape(-1).abs() for v in tau.values()])
    n = allabs.numel()
    k = int(round(density * n))
    if k >= n:
        return -1.0  # keep everything (|.| >= -1 always true)
    if k <= 0:
        return float("inf")  # keep nothing
    # k-th largest magnitude == (n-k+1)-th smallest
    return float(torch.kthvalue(allabs, n - k + 1).values)


def trim_task_vector(tau: ParamDict, density: float) -> ParamDict:
    """Keep the top-``density`` fraction of coordinates by |value|; zero the rest."""
    thr = _global_topk_threshold(tau, density)
    return OrderedDict((k, v * (v.abs() >= thr)) for k, v in tau.items())


def ties_combined_tau(task_vectors: TaskVectors, density: float = 0.2) -> ParamDict:
    """TIES trim + sign-election + disjoint mean; returns the combined task vector
    *before* lambda scaling. Depends only on ``density``, so a lambda sweep can
    reuse the result without re-trimming.

    Memory: the per-task trim thresholds are computed first (T scalars), then each
    parameter tensor is trimmed and stacked one key at a time, so at most one
    tensor's worth of trimmed columns is materialised at once -- we never hold a
    second full copy of all T task vectors (which OOMs on the 8-task CLIP suite)."""
    names = list(task_vectors.keys())
    thresholds = {n: _global_topk_threshold(task_vectors[n], density) for n in names}
    keys = list(task_vectors[names[0]].keys())
    out = OrderedDict()
    for key in keys:
        cols = [task_vectors[n][key] * (task_vectors[n][key].abs() >= thresholds[n])
                for n in names]
        stack = torch.stack(cols, dim=0)                              # [T, *shape]
        elected = torch.sign(stack.sum(dim=0))                        # per-coord sign
        agree = (torch.sign(stack) == elected.unsqueeze(0)) & (stack != 0)
        summed = (stack * agree).sum(dim=0)
        count = agree.sum(dim=0).clamp(min=1)                         # disjoint mean
        out[key] = summed / count
    return out


def ties_merge(base_encoder: ParamDict, task_vectors: TaskVectors,
               lam: float = 1.0, density: float = 0.2,
               combined: Optional[ParamDict] = None) -> ParamDict:
    """``theta_0 + lam * TIES(tau)``. Pass a precomputed ``combined`` (from
    :func:`ties_combined_tau`) to skip the trim when sweeping lambda."""
    tau = combined if combined is not None else ties_combined_tau(task_vectors, density)
    merged = pd_clone(base_encoder)
    pd_axpy_(merged, lam, tau)
    return merged


def dare_drop(tau: ParamDict, density: float, generator: torch.Generator) -> ParamDict:
    """Keep each coordinate with probability ``density``; rescale survivors by
    ``1/density`` so the expectation matches the undropped task vector."""
    out = OrderedDict()
    for k, v in tau.items():
        mask = (torch.rand(v.shape, generator=generator) < density).to(v.dtype)
        out[k] = v * mask / density
    return out


def dare_ta_merge(base_encoder: ParamDict, task_vectors: TaskVectors,
                  lambdas: Dict[str, float], density: float = 0.5,
                  seed: int = 0) -> ParamDict:
    """DARE drop+rescale each tau, then ordinary task arithmetic (DARE-TA).

    NOTE: this is a *control*, not a competitive baseline. The drop is
    expectation-preserving, so summed over ~10^8 coordinates the merge
    concentrates back onto plain task arithmetic. Use :func:`dare_ties_merge`
    for the variant that actually resolves interference."""
    gen = torch.Generator().manual_seed(seed)
    merged = pd_clone(base_encoder)
    for n, tau in task_vectors.items():
        dropped = dare_drop(tau, density, gen)
        pd_axpy_(merged, lambdas[n], dropped)
    return merged


# ---------------------------------------------------------------------------
# Model Breadcrumbs (Davari & Belilovsky 2023)
# ---------------------------------------------------------------------------

def _band_mask(v: torch.Tensor, density: float, outlier_frac: float) -> torch.Tensor:
    """Keep a magnitude *band*: drop the top ``outlier_frac`` coords (outliers)
    and everything below the next ``density`` fraction (negligible weights).
    Thresholds are per-tensor ("layer-wise"), as in the Breadcrumbs paper."""
    a = v.abs().reshape(-1)
    n = a.numel()
    k_hi = int(round(outlier_frac * n))          # how many top coords to discard
    k_keep = int(round(density * n))
    if k_keep <= 0:
        return torch.zeros_like(v, dtype=torch.bool)
    if k_hi > 0:
        thr_hi = torch.kthvalue(a, n - k_hi + 1).values   # k_hi-th largest
        hi_ok = v.abs() < thr_hi
    else:
        hi_ok = torch.ones_like(v, dtype=torch.bool)
    k_lo = min(n, k_hi + k_keep)
    thr_lo = torch.kthvalue(a, n - k_lo + 1).values       # k_lo-th largest
    return hi_ok & (v.abs() >= thr_lo)


def breadcrumbs_merge(base_encoder: ParamDict, task_vectors: TaskVectors,
                      lambdas: Dict[str, float], density: float = 0.1,
                      outlier_frac: float = 0.01) -> ParamDict:
    """theta_0 + sum_i lam_i * (tau_i restricted to its magnitude band)."""
    merged = pd_clone(base_encoder)
    for n, tau in task_vectors.items():
        for k in base_encoder:
            m = _band_mask(tau[k], density, outlier_frac)
            merged[k].add_(tau[k] * m, alpha=lambdas[n])
    return merged


# ---------------------------------------------------------------------------
# DARE-TIES: drop+rescale, then TIES sign-election
# ---------------------------------------------------------------------------

def _tensor_seed(base_seed: int, task: str, key: str) -> int:
    """Deterministic per-(task, tensor) seed. Independent of iteration order, so
    the same drop mask is reproduced across the threshold pass and the combine
    pass without materialising the dropped task vectors (Python's builtin hash()
    is salted per process, hence sha1)."""
    h = hashlib.sha1(f"{base_seed}|{task}|{key}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def _dropped(v: torch.Tensor, density: float, base_seed: int,
             task: str, key: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(_tensor_seed(base_seed, task, key))
    mask = (torch.rand(v.shape, generator=g) < density).to(v.dtype)
    return v * mask / density


def _global_threshold(get: Callable[[str], torch.Tensor], keys: List[str],
                      density: float) -> float:
    allabs = torch.cat([get(k).reshape(-1).abs() for k in keys])
    n = allabs.numel()
    k = int(round(density * n))
    if k >= n:
        return -1.0
    if k <= 0:
        return float("inf")
    return float(torch.kthvalue(allabs, n - k + 1).values)


def _ties_disjoint_mean(names: List[str], keys: List[str],
                        get: Callable[[str, str], torch.Tensor],
                        thresholds: Dict[str, float]) -> ParamDict:
    """Trim by per-task threshold, elect a per-coordinate sign, average the
    sign-agreeing entries. ``get(task, key)`` supplies the (possibly dropped)
    task-vector tensor; it is called once per (task, key) per pass, so it must be
    deterministic."""
    out = OrderedDict()
    for key in keys:
        cols = []
        for n in names:
            v = get(n, key)
            cols.append(v * (v.abs() >= thresholds[n]))
        stack = torch.stack(cols, dim=0)
        elected = torch.sign(stack.sum(dim=0))
        agree = (torch.sign(stack) == elected.unsqueeze(0)) & (stack != 0)
        out[key] = (stack * agree).sum(dim=0) / agree.sum(dim=0).clamp(min=1)
    return out


def dare_ties_merge(base_encoder: ParamDict, task_vectors: TaskVectors,
                    lam: float = 1.0, drop_density: float = 0.5,
                    trim_density: float = 0.2, seed: int = 0) -> ParamDict:
    """DARE drop+rescale each tau, THEN TIES (trim + sign-elect + disjoint mean),
    scaled by a single lambda. This is the pairing that actually resolves
    interference: the drop sparsifies, the sign election resolves conflicts."""
    names = list(task_vectors.keys())
    keys = list(base_encoder.keys())

    def get(n, k):
        return _dropped(task_vectors[n][k], drop_density, seed, n, k)

    thresholds = {n: _global_threshold(lambda k, n=n: get(n, k), keys, trim_density)
                  for n in names}
    combined = _ties_disjoint_mean(names, keys, get, thresholds)
    merged = pd_clone(base_encoder)
    pd_axpy_(merged, lam, combined)
    return merged
