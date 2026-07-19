"""GLUE-convention scoring and normalized-retention aggregation.

Per-task primary metric:
  CoLA -> Matthews corr; SST-2/QNLI/RTE/MNLI -> accuracy; STS-B -> mean of
  Pearson & Spearman; MRPC/QQP -> mean of accuracy & F1.

Normalized retention (Eq. 21):
  NormRet_t = (Score_t(merge) - Score_t(theta0)) /
              (Score_t(expert) - Score_t(theta0) + eps)
"""

from typing import Dict, List
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

from .tasks import TaskSpec

EPS_METRIC = 1e-8


def score_predictions(spec: TaskSpec, preds, labels) -> Dict[str, float]:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    m = spec.metric
    if m == "matthews":
        return {"matthews": float(matthews_corrcoef(labels, preds)),
                "primary": float(matthews_corrcoef(labels, preds))}
    if m == "accuracy":
        acc = float(accuracy_score(labels, preds))
        return {"accuracy": acc, "primary": acc}
    if m == "acc_f1":
        acc = float(accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds))
        return {"accuracy": acc, "f1": f1, "primary": (acc + f1) / 2.0}
    if m == "pearson_spearman":
        p = float(pearsonr(preds, labels)[0])
        s = float(spearmanr(preds, labels)[0])
        return {"pearson": p, "spearman": s, "primary": (p + s) / 2.0}
    raise ValueError(f"Unknown metric '{m}'")


def normalized_retention(merge: float, base: float, expert: float,
                         eps: float = EPS_METRIC) -> float:
    # correlation metrics are NaN when predictions are degenerate (e.g. zero-shot
    # flan-T5 on STS-B emits non-numeric strings); read "undefined" as "no signal"
    # for the base/merge scores so one NaN cannot poison the aggregates. A NaN
    # expert score is a broken setup and is left to propagate visibly.
    if np.isnan(base):
        base = 0.0
    if np.isnan(merge):
        merge = 0.0
    return (merge - base) / (expert - base + eps)


def aggregate_retention(per_task: Dict[str, float]) -> Dict[str, float]:
    """mean / worst / harmonic-mean normalized retention across tasks."""
    vals = np.array(list(per_task.values()), dtype=float)
    out = {
        "mean_normret": float(vals.mean()),
        "worst_normret": float(vals.min()),
    }
    # harmonic mean is only meaningful for positive retentions; clamp at eps
    clipped = np.clip(vals, 1e-6, None)
    out["hmean_normret"] = float(len(clipped) / np.sum(1.0 / clipped))
    return out
