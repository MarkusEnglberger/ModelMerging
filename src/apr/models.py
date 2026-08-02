"""Model loading and parameter-vector utilities.

The "mergeable encoder" is everything under the model's `base_model_prefix`
(e.g. all `roberta.*` parameters, including embeddings). Task-specific heads
(everything else, e.g. `classifier.*`) are *not* merged; they are kept per task
and used both for refinement gradients and for evaluation.

Parameters are represented as an ordered ``dict[name -> Tensor]`` (a "param
dict"). All vector operations are elementwise per tensor, which is exactly a
flattened-vector operation but avoids materialising one giant vector.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ParamDict = Dict[str, torch.Tensor]


def load_tokenizer(checkpoint: str, cache_dir: Optional[str] = None):
    return AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)


def load_classifier(checkpoint: str, num_labels: int, cache_dir: Optional[str] = None):
    """Load a sequence-classification model (encoder + head)."""
    problem_type = "regression" if num_labels == 1 else "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, problem_type=problem_type,
        cache_dir=cache_dir,
    )
    return model


def encoder_param_names(model) -> List[str]:
    """Names of mergeable encoder parameters (under base_model_prefix)."""
    if hasattr(model, "mergeable_param_names"):
        return list(model.mergeable_param_names())
    prefix = model.base_model_prefix + "."
    return [n for n, _ in model.named_parameters() if n.startswith(prefix)]


def is_head_param(model, name: str) -> bool:
    if hasattr(model, "mergeable_param_names"):
        return name not in set(model.mergeable_param_names())
    return not name.startswith(model.base_model_prefix + ".")


def get_encoder_state(model, device: Optional[str] = None) -> ParamDict:
    """Detached clone of the mergeable encoder parameters."""
    names = set(encoder_param_names(model))
    out = OrderedDict()
    for n, p in model.named_parameters():
        if n in names:
            t = p.detach().clone()
            out[n] = t.to(device) if device else t
    return out


def get_head_state(model, device: Optional[str] = None) -> ParamDict:
    """Detached clone of the task-specific head parameters."""
    out = OrderedDict()
    for n, p in model.named_parameters():
        if is_head_param(model, n):
            t = p.detach().clone()
            out[n] = t.to(device) if device else t
    return out


@torch.no_grad()
def load_encoder_state(model, state: ParamDict):
    """Copy encoder params from `state` into `model` in place."""
    params = dict(model.named_parameters())
    for n, t in state.items():
        params[n].copy_(t.to(params[n].device))


@torch.no_grad()
def load_head_state(model, state: ParamDict):
    params = dict(model.named_parameters())
    for n, t in state.items():
        params[n].copy_(t.to(params[n].device))


# ---------------------------------------------------------------------------
# Param-dict vector algebra (all elementwise; keys must match)
# ---------------------------------------------------------------------------

def pd_to(a: ParamDict, device: str) -> ParamDict:
    return OrderedDict((k, v.to(device)) for k, v in a.items())


def pd_clone(a: ParamDict) -> ParamDict:
    return OrderedDict((k, v.clone()) for k, v in a.items())


def pd_sub(a: ParamDict, b: ParamDict) -> ParamDict:
    return OrderedDict((k, a[k] - b[k].to(a[k].device)) for k in a)


def pd_add(a: ParamDict, b: ParamDict) -> ParamDict:
    return OrderedDict((k, a[k] + b[k].to(a[k].device)) for k in a)


def pd_axpy_(a: ParamDict, alpha: float, b: ParamDict) -> ParamDict:
    """In-place a <- a + alpha * b."""
    for k in a:
        a[k].add_(b[k].to(a[k].device), alpha=alpha)
    return a


def pd_scale(a: ParamDict, alpha: float) -> ParamDict:
    return OrderedDict((k, a[k] * alpha) for k in a)


def pd_global_norm(a: ParamDict) -> float:
    return float(torch.sqrt(sum((v.double() ** 2).sum() for v in a.values())))
