"""Evaluation of a (encoder, head) configuration on a task's eval split."""

from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from .tasks import TaskSpec
from .metrics import score_predictions


@torch.no_grad()
def evaluate_task(model, eval_ds, spec: TaskSpec, collator, batch_size: int,
                  device: str, num_workers: int = 0) -> Dict[str, float]:
    """Run the model over the eval split and return the GLUE-convention score.

    Specs with ``eval_kind == "generation"`` (the causal-LM track) are scored by
    free generation + a task metric function instead of logits."""
    if getattr(spec, "eval_kind", "classification") == "generation":
        from .causal_lm import evaluate_generation
        return evaluate_generation(model, eval_ds, spec, batch_size, device)
    if getattr(spec, "eval_kind", "classification") == "mc":
        from .causal_lm import evaluate_mc
        return evaluate_mc(model, eval_ds, spec, batch_size, device)
    if getattr(spec, "eval_kind", "classification") == "mats_p3":
        from .mats_t5 import evaluate_mats
        return evaluate_mats(model, eval_ds, spec, collator, batch_size, device,
                             num_workers=num_workers)
    model.eval()
    model.to(device)
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator,
                        num_workers=num_workers,
                        pin_memory=device.startswith("cuda"))
    all_preds: List = []
    all_labels: List = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        if spec.is_regression:
            preds = logits.squeeze(-1).float().cpu().numpy()
        else:
            preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return score_predictions(spec, preds, labels)
