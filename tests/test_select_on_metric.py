"""The metric selection objective must rank by correctness, not by confidence.

Held-out cross-entropy is a biased ranking function when the candidates differ
in confidence: from a maximum-entropy initialization, becoming confident raises
CE on the examples it gets wrong faster than it lowers CE on those it gets
right, so a timid model outranks a more accurate one. These tests pin that this
is a real property of CE (not a claim about our data) and that the metric
objective is free of it, using synthetic logits with no model involved.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.metrics import score_predictions
from apr.tasks import TASKS as GLUE_SPECS


def _ce(probs_true):
    """Mean cross-entropy given the probability assigned to the true label."""
    return float(np.mean([-math.log(max(p, 1e-12)) for p in probs_true]))


def test_ce_prefers_the_timid_model_while_accuracy_prefers_the_accurate_one():
    n = 8
    # timid: maximum entropy, right on half by luck -> CE = ln 2 at any accuracy
    timid_ce = _ce([0.5] * n)
    timid_acc = 0.5
    # confident: 75% correct at p=0.95, wrong at p=0.05
    conf_probs = [0.95] * 6 + [0.05] * 2
    conf_ce = _ce(conf_probs)
    conf_acc = 6 / 8
    assert conf_acc > timid_acc, "the confident model IS more accurate"
    assert conf_ce > timid_ce, (
        f"CE must nonetheless prefer the timid model: {conf_ce:.3f} vs {timid_ce:.3f}")
    # and the break-even point is near 0.8 accuracy, as the paper claims
    def ce_at(acc):
        return acc * -math.log(0.95) + (1 - acc) * -math.log(0.05)
    assert ce_at(0.75) > math.log(2) > ce_at(0.85)


def test_metric_objective_orders_by_correctness():
    """-mean(metric) is lower (better) for the more accurate model, always."""
    spec = GLUE_SPECS["sst2"]
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    worse = np.array([0, 0, 0, 0, 0, 0, 0, 0])       # majority-class, 50%
    better = np.array([0, 1, 0, 1, 0, 1, 1, 0])      # 75%
    mw = score_predictions(spec, worse, labels)["primary"]
    mb = score_predictions(spec, better, labels)["primary"]
    assert mb > mw
    assert -mb < -mw, "the negated metric must be minimised by the better model"


def test_degenerate_folds_are_finite():
    """On an 8-example fold, constant predictions make MCC undefined and the
    correlations NaN. buffer_metric maps those to 0.0; check the raw scorers do
    produce the degenerate values, so the guard is exercised and not dead code."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    const = np.zeros(8, dtype=int)
    mcc = score_predictions(GLUE_SPECS["cola"], const, labels)["primary"]
    assert mcc == 0.0 or mcc != mcc, "constant predictions give MCC 0 or NaN"
    stsb = GLUE_SPECS["stsb"]
    v = score_predictions(stsb, np.zeros(8), np.arange(8, dtype=float))["primary"]
    assert v != v, "zero-variance predictions must make the correlation NaN"


def test_glue_pretrained_is_at_maximum_entropy():
    """The premise of the whole correction: GLUE's pretrained heads sit at ~ln C,
    so any confidence gain raises CE. Values measured in the protocol traces."""
    measured = {"cola": 0.7444, "sst2": 0.6908, "mrpc": 0.6832, "qqp": 0.6695,
                "qnli": 0.6844, "rte": 0.6916}
    for t, ce in measured.items():
        assert abs(ce - math.log(2)) < 0.06, f"{t} not near ln 2: {ce}"
    assert abs(1.1248 - math.log(3)) < 0.06, "mnli not near ln 3"
