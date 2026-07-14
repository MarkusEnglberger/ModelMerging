"""RegMean (Jin et al. 2023): closed-form data-dependent merge of linear layers.

For each linear layer y = W x, RegMean chooses the merged weight that best reproduces
every expert's output on that expert's own inputs:

    W* = argmin_W sum_i || W X_i^T - W_i X_i^T ||_F^2
       = ( sum_i W_i G_i ) ( sum_i G_i )^{-1},     G_i = X_i^T X_i,

where X_i are the inputs seen by that layer on expert i's (unlabeled) data. This is the
label-free, data-dependent baseline promised in the proposal, and the standard companion
to AdaMerging on both the RoBERTa GLUE and CLIP ViT suites.

Implementation notes:
  - Grams are gathered with forward hooks on every ``nn.Linear`` under the encoder prefix,
    over the unlabeled replay inputs (labels are dropped). Non-linear parameters
    (embeddings, convolutions, biases, LayerNorm) are not covered by the regression and
    are uniformly averaged across experts, as in the paper.
  - Following the paper we shrink the off-diagonal of each Gram toward 0 (``nondiag_scale``)
    for conditioning, and add ``eps`` to the diagonal before inverting.
  - With the matched 64-example budget the Gram is estimated from far fewer inputs than the
    original RegMean (which uses training data); ``n_gram`` in the driver can enlarge the
    unlabeled sample, since RegMean uses no labels.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch

from .data import batches_from_buffer
from .models import ParamDict, load_encoder_state, pd_clone


def encoder_linear_weight_names(model) -> List[str]:
    """Names of ``<module>.weight`` for every nn.Linear under the encoder prefix."""
    prefix = model.base_model_prefix + "."
    out = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and (mod_name + ".").startswith(prefix):
            out.append(mod_name + ".weight")
    return out


@torch.no_grad()
def gather_grams(model, buffer, collator, linear_names: List[str], device: str,
                 batch_size: int) -> Dict[str, torch.Tensor]:
    """G[name] = sum over the buffer of X^T X, X = inputs to that linear layer."""
    modules = dict(model.named_modules())
    name_by_id = {id(modules[n[:-len(".weight")]]): n for n in linear_names}
    grams: Dict[str, torch.Tensor] = {}

    def hook(mod, inp, out):
        x = inp[0].detach()
        x = x.reshape(-1, x.shape[-1]).float()      # [N, in]
        g = (x.t() @ x).cpu()
        nm = name_by_id[id(mod)]
        grams[nm] = g if nm not in grams else grams[nm] + g

    handles = [modules[n[:-len(".weight")]].register_forward_hook(hook)
               for n in linear_names]
    model.to(device)
    model.eval()
    try:
        for batch in batches_from_buffer(buffer, collator, batch_size, device):
            batch.pop("labels", None)
            model(**batch)
    finally:
        for h in handles:
            h.remove()
    model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return grams


def regmean_merge(base_encoder: ParamDict, per_task: Dict[str, dict],
                  task_names: List[str], device: str, buffer_key: str = "probe_buffer",
                  nondiag_scale: float = 0.9, eps: float = 1e-3, batch_size: int = 16,
                  logger=None) -> Tuple[ParamDict, dict]:
    """Merge encoder linear weights by RegMean; uniformly average the rest."""
    names = list(task_names)
    m0 = per_task[names[0]]["model"]
    linear_names = [n for n in encoder_linear_weight_names(m0) if n in base_encoder]
    if logger:
        logger(f"[regmean] {len(linear_names)} encoder linear layers; "
               f"nondiag_scale={nondiag_scale} eps={eps}")

    sum_G: Dict[str, torch.Tensor] = {}
    sum_WG: Dict[str, torch.Tensor] = {}
    for n in names:
        model = per_task[n]["model"]
        load_encoder_state(model, per_task[n]["expert_encoder"])  # Grams at the expert
        grams = gather_grams(model, per_task[n][buffer_key], per_task[n]["collator"],
                             linear_names, device, batch_size)
        for wn in linear_names:
            G = grams[wn]
            if nondiag_scale != 1.0:  # shrink off-diagonal toward 0
                diag = torch.diag(torch.diag(G))
                G = nondiag_scale * G + (1.0 - nondiag_scale) * diag
            W = per_task[n]["expert_encoder"][wn].float()          # [out, in]
            sum_G[wn] = G if wn not in sum_G else sum_G[wn] + G
            WG = W @ G
            sum_WG[wn] = WG if wn not in sum_WG else sum_WG[wn] + WG
        if logger:
            logger(f"  [regmean] gathered grams for {n}")

    merged = pd_clone(base_encoder)
    n_in_total = 0
    for wn in linear_names:
        G = sum_G[wn]
        n_in = G.shape[0]
        G_reg = G + eps * torch.eye(n_in, dtype=G.dtype)
        W_m = sum_WG[wn] @ torch.linalg.inv(G_reg)                 # [out, in]
        merged[wn] = W_m.to(merged[wn].dtype)
        n_in_total += n_in
    # non-linear params (embeddings, convs, biases, LayerNorm): uniform expert average
    lin = set(linear_names)
    for k in base_encoder:
        if k not in lin:
            acc = sum(per_task[n]["expert_encoder"][k].float() for n in names) / len(names)
            merged[k] = acc.to(merged[k].dtype)
    return merged, {"n_linear": len(linear_names), "nondiag_scale": nondiag_scale,
                    "eps": eps, "buffer_key": buffer_key}
