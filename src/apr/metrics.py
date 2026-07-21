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


# A task whose expert barely beats the zero-shot/base model has a near-zero
# NormRet denominator, so its normalized retention explodes. On the 20-task CLIP
# suite stl10 (base .9712 -> expert .9755, gap .0043) and food101 (.8271 ->
# .8622) alone swung the mean from -1.04 to -4.94. Absolute accuracy is the
# primary metric there; NormRet is still reported, plus a variant restricted to
# tasks with a meaningful gap.
DEGENERATE_GAP = 0.05


def aggregate_absolute(scores: Dict[str, float]) -> Dict[str, float]:
    """mean / worst ABSOLUTE per-task score (accuracy for the vision suite)."""
    vals = np.array(list(scores.values()), dtype=float)
    return {"mean_acc": float(vals.mean()), "worst_acc": float(vals.min())}


def degenerate_tasks(base: Dict[str, float], expert: Dict[str, float],
                     gap_min: float = DEGENERATE_GAP) -> List[str]:
    """Tasks whose expert-minus-base gap is too small for NormRet to be stable."""
    return sorted(t for t in base if (expert[t] - base[t]) < gap_min)


def aggregate_all(scores: Dict[str, float], per_task_nr: Dict[str, float],
                  base: Dict[str, float], expert: Dict[str, float],
                  gap_min: float = DEGENERATE_GAP) -> Dict[str, float]:
    """Absolute-accuracy aggregates (primary) + NormRet (secondary), plus a
    NormRet restricted to non-degenerate-gap tasks."""
    out = aggregate_absolute(scores)
    out.update(aggregate_retention(per_task_nr))
    deg = degenerate_tasks(base, expert, gap_min)
    keep = {t: v for t, v in per_task_nr.items() if t not in deg}
    if deg and keep:
        sub = aggregate_retention(keep)
        out["mean_normret_nondeg"] = sub["mean_normret"]
        out["worst_normret_nondeg"] = sub["worst_normret"]
    out["degenerate_tasks"] = deg
    return out
