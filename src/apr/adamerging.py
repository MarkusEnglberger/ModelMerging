"""AdaMerging (Yang et al. 2024): label-free test-time learning of merge coefficients.

    theta(lambda) = theta_0 + sum_i lambda_i^l tau_i^l
    minimize_lambda  sum_i  E_{x ~ D_i^unlabeled} H(softmax(f_i(x; theta)))

where l indexes a "layer" (we use one coefficient per encoder tensor for the
layer-wise variant, one per task for the task-wise variant), f_i uses task i's
frozen head, and H is Shannon entropy. No labels are used.

Gradient trick: because theta is *linear* in lambda,

    dL/dlambda_i^l = < dL/dtheta^l , tau_i^l >

so we never build an autograd graph over lambda. We take a single gradient with
respect to theta (exactly as gradients.make_grad_fn does, but with the entropy
loss) and contract it with each task vector. Memory stays O(|theta|) rather than
O(T |theta|), which matters because T|theta| is ~4 GB for GLUE-8 RoBERTa.

Data regime. Two configurations, selected by ``data_key``:

  data_key="eval_ds"      the standard (transductive) AdaMerging protocol: unlabeled
                          inputs drawn from the very split it is later scored on, in
                          effectively unlimited quantity. Faithful to the paper, but
                          NOT comparable to APR's 64 labeled train examples.
  data_key="probe_buffer" matched-budget: exactly the inputs in APR's replay buffer
                          (drawn from train), with the labels discarded. Isolates the
                          objective (entropy vs. gated replay) from the data advantage.

The proposal's own protocol asks that label-free baselines get "the same number of
unlabeled examples", i.e. the matched variant. Report both.

Regression tasks (STS-B, num_labels==1) have no predictive distribution, so they
contribute no entropy term. Their coefficients are still learned, via the other
tasks' gradients (theta depends on every tau).
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .models import ParamDict, load_encoder_state


def _entropy_loss(logits: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Mean Shannon entropy of the predictive distribution over a batch.

    ``mask`` (optional, bool, same shape as logits minus the vocab dim) restricts
    the mean to selected positions -- used by the causal-LM adaptation to score
    only prompt positions."""
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    if mask is not None:
        ent = ent[mask]
        if ent.numel() == 0:
            return (logits.sum() * 0.0)  # keeps the graph; contributes nothing
    return ent.mean()


def _infinite_batches(dataset, collator, batch_size: int, seed: int,
                      num_workers: int = 0):
    """Cycle shuffled unlabeled batches forever."""
    g = torch.Generator()
    g.manual_seed(seed)
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collator, num_workers=num_workers,
                            generator=g, drop_last=False)
        for batch in loader:
            yield batch


def _build_theta(base_encoder: ParamDict, task_vectors: Dict[str, ParamDict],
                 lam: torch.Tensor, names: List[str], keys: List[str],
                 layerwise: bool, device: str) -> ParamDict:
    """theta = theta_0 + sum_i lambda_i^l tau_i^l  (no autograd through lambda).

    ``lam`` is a CPU tensor, so the per-coefficient float() costs no GPU sync.
    We clone before add_ because ``.to(device)`` is a no-op when the tensor is
    already there -- otherwise we would mutate base_encoder in place."""
    theta = OrderedDict()
    with torch.no_grad():
        for j, k in enumerate(keys):
            acc = base_encoder[k].detach().clone().to(device)
            for i, n in enumerate(names):
                coef = lam[i, j] if layerwise else lam[i]
                acc.add_(task_vectors[n][k].to(device), alpha=float(coef))
            theta[k] = acc
    return theta


def _contract_lambda_grad(grad_theta: ParamDict, task_vectors: Dict[str, ParamDict],
                          names: List[str], keys: List[str], layerwise: bool,
                          device: str) -> torch.Tensor:
    """dL/dlambda_i^l = <dL/dtheta^l, tau_i^l>.

    Accumulated on-device and synced once: a float() per (task, tensor) would
    cost T*K GPU syncs per optimisation step."""
    shape = (len(names), len(keys)) if layerwise else (len(names),)
    g = torch.zeros(shape, device=device, dtype=torch.float32)
    with torch.no_grad():
        for i, n in enumerate(names):
            for j, k in enumerate(keys):
                d = (grad_theta[k] * task_vectors[n][k].to(device)).sum()
                if layerwise:
                    g[i, j] = d
                else:
                    g[i] += d
    return g


def adamerging(base_encoder: ParamDict, task_vectors: Dict[str, ParamDict],
               per_task: Dict[str, dict], task_names: List[str], device: str,
               layerwise: bool = True, steps: int = 300, lr: float = 1e-3,
               batch_size: int = 16, init_lam: float = 0.3, seed: int = 0,
               num_workers: int = 0, log_every: int = 50,
               data_key: str = "eval_ds", logger=None) -> (ParamDict, dict):
    """Learn merge coefficients by unlabeled entropy minimisation.

    Returns (merged_encoder_cpu, info) where info records the learned lambdas
    and the entropy trace."""
    names = list(task_names)
    keys = list(base_encoder.keys())
    T, K = len(names), len(keys)
    enc_names = set(keys)

    shape = (T, K) if layerwise else (T,)
    lam = torch.full(shape, float(init_lam), dtype=torch.float32)
    lam.requires_grad_(True)
    opt = torch.optim.Adam([lam], lr=lr)

    # task vectors on device: T*|theta| (~2.8 GB CLIP / ~4 GB GLUE-8) so the
    # per-step build+contract are not PCIe-bound. Fall back to CPU streaming.
    tv_dev = task_vectors
    try:
        tv_dev = {n: OrderedDict((k, v.to(device)) for k, v in tv.items())
                  for n, tv in task_vectors.items()}
    except RuntimeError as e:  # pragma: no cover - OOM path
        if logger:
            logger(f"[adamerging] task vectors stay on CPU ({e.__class__.__name__})")
        tv_dev = task_vectors

    # unlabeled batch streams; regression tasks contribute no entropy term
    ent_tasks = [n for n in names if not per_task[n]["spec"].is_regression]
    if logger and len(ent_tasks) < T:
        skipped = [n for n in names if n not in ent_tasks]
        logger(f"[adamerging] no entropy term for regression tasks {skipped} "
               f"(their lambdas still learn via the other tasks)")
    if logger:
        sizes = {n: len(per_task[n][data_key]) for n in ent_tasks}
        logger(f"[adamerging] unlabeled source='{data_key}' sizes={sizes} "
               f"({'TRANSDUCTIVE: adapts on the eval split' if data_key == 'eval_ds' else 'matched to APR replay budget'})")
    streams = {n: _infinite_batches(per_task[n][data_key], per_task[n]["collator"],
                                    batch_size, seed + i, num_workers)
               for i, n in enumerate(ent_tasks)}

    trace = []
    for step in range(steps):
        theta = _build_theta(base_encoder, tv_dev, lam.detach(), names, keys,
                             layerwise, device)

        grad_theta = OrderedDict((k, torch.zeros_like(theta[k])) for k in keys)
        total_ent = 0.0
        for n in ent_tasks:
            model = per_task[n]["model"]
            model.to(device)
            model.eval()
            load_encoder_state(model, theta)
            model.zero_grad(set_to_none=True)

            batch = next(streams[n])
            labels = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            mask = None
            if out.logits.dim() == 3 and labels is not None:
                # causal-LM adaptation: SFT batches carry prompt+response tokens
                # with the prompt loss-masked (labels==-100). To stay honestly
                # label-free, score predictive entropy ONLY at positions whose
                # next token is still prompt (response tokens are the labels and
                # must not condition the objective). Pad positions (attention 0)
                # are excluded too.
                lab = labels.to(device)
                mask = lab[:, 1:] == -100
                att = batch.get("attention_mask")
                if att is not None:
                    mask &= att[:, 1:].bool()
                ent = _entropy_loss(out.logits[:, :-1, :], mask=mask)
            else:
                ent = _entropy_loss(out.logits)
            ent.backward()
            total_ent += float(ent)

            for pn, p in model.named_parameters():
                if pn in enc_names and p.grad is not None:
                    grad_theta[pn].add_(p.grad)
            model.zero_grad(set_to_none=True)
            model.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        g = _contract_lambda_grad(grad_theta, tv_dev, names, keys, layerwise, device)
        lam.grad = g.to(lam.device)
        opt.step()
        opt.zero_grad(set_to_none=True)

        trace.append({"step": step, "entropy": total_ent / max(len(ent_tasks), 1),
                      "lam_mean": float(lam.detach().mean())})
        if logger and (step % log_every == 0 or step == steps - 1):
            logger(f"  [adamerging] step {step:4d}/{steps} "
                   f"entropy={trace[-1]['entropy']:.4f} "
                   f"lam_mean={trace[-1]['lam_mean']:.4f}")
        del theta, grad_theta

    final = _build_theta(base_encoder, tv_dev, lam.detach(), names, keys,
                         layerwise, device)
    merged_cpu = OrderedDict((k, v.detach().cpu()) for k, v in final.items())
    lam_c = lam.detach().cpu()
    lam_out = ({n: float(lam_c[i]) for i, n in enumerate(names)} if not layerwise
               else {n: float(lam_c[i].mean()) for i, n in enumerate(names)})
    info = {"layerwise": layerwise, "steps": steps, "lr": lr,
            "data_key": data_key, "transductive": data_key == "eval_ds",
            "n_unlabeled_per_task": {n: len(per_task[n][data_key]) for n in ent_tasks},
            "batch_size": batch_size, "init_lam": init_lam,
            "lam_per_task": lam_out,  # layerwise: mean over tensors
            "lam_full": lam_c.tolist() if layerwise else None,
            "entropy_tasks": ent_tasks,
            "final_entropy": trace[-1]["entropy"] if trace else None,
            "entropy_first": trace[0]["entropy"] if trace else None}
    return merged_cpu, info
