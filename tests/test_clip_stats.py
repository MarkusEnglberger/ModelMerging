"""The expert-distance cap's binding rate must be counted, and counted right.

refine() has always computed clipped_frac_gated/clipped_frac_all but nothing
read them back, so they were never verified. These tests pin the semantics:

  * the cap binds exactly when eta*|g_r| > |v_r| ... no: > 1 as a FRACTION,
    i.e. when eta*|g_r| > 1, because the step is eta*g*|v| and the bound is
    |v| -- the expert distance cancels. The rate is therefore a statement
    about gradient magnitude and the learning rate alone.
  * a tiny eta binds on nothing; a huge eta binds on every gated coordinate.
  * the ungated arm still clips (it keeps the anchor); the GD arm has
    clip_mode="none" and must report a zero rate.
"""

import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import RefineConfig
from apr.refine import refine

from test_refine import make_handle


def _setup(d=256, seed=11):
    g = torch.Generator().manual_seed(seed)
    handles = []
    for name in ("a", "b"):
        expert = torch.randn(d, generator=g)
        target = torch.randn(d, generator=g)
        h, _ = make_handle(name, d, expert, target)
        handles.append(h)
    base = OrderedDict({"enc.w": torch.randn(d, generator=g)})
    return base, handles


def _cfg(lr, **kw):
    return RefineConfig(steps=2, lr=lr, order="fixed", gate_mode="coordinate",
                        lr_schedule="constant", clip_mode="vdist", **kw)


def test_clip_rate_is_reported_and_bounded():
    base, handles = _setup()
    _, hist = refine(base, handles, _cfg(1.0), "cpu", seed=0)
    assert hist, "no history recorded"
    for h in hist:
        assert "clipped_frac_gated" in h and "clipped_frac_all" in h
        assert 0.0 <= h["clipped_frac_gated"] <= 1.0
        assert 0.0 <= h["clipped_frac_all"] <= h["gate_density"] + 1e-9, \
            "clipped coords must be a subset of gated coords"


def test_clip_rate_is_monotone_in_lr():
    """Larger eta saturates more coordinates: the rate must not decrease."""
    base, handles = _setup()
    rates = []
    for lr in (1e-6, 1e-2, 1.0, 1e3):
        _, hist = refine(base, handles, _cfg(lr), "cpu", seed=0)
        rates.append(hist[0]["clipped_frac_gated"])
    assert rates[0] == 0.0, f"tiny lr must clip nothing, got {rates[0]}"
    assert rates[-1] == 1.0, f"huge lr must clip every gated coord, got {rates[-1]}"
    assert rates == sorted(rates), f"clip rate not monotone in lr: {rates}"


def test_gd_arm_reports_no_clipping():
    """clip_mode='none' (the GD ablation) has no cap to bind."""
    base, handles = _setup()
    cfg = RefineConfig(steps=2, lr=1.0, order="fixed", gate_mode="none",
                       update_mode="grad", clip_mode="none",
                       lr_schedule="constant")
    _, hist = refine(base, handles, cfg, "cpu", seed=0)
    assert all(h["clipped_frac_gated"] == 0.0 for h in hist)


def test_ungated_arm_still_clips():
    """The ungated ablation drops the gate but keeps the anchor and its cap.

    Not exactly 1.0 even at a huge eta: a coordinate with |g_r| < 1/eta stays
    below the cap however large the expert distance is, since |v| cancels out
    of the binding condition. That is the point of the statistic.
    """
    base, handles = _setup()
    cfg = RefineConfig(steps=2, lr=1e3, order="fixed", gate_mode="none",
                       clip_mode="vdist", lr_schedule="constant")
    _, hist = refine(base, handles, cfg, "cpu", seed=0)
    assert hist[0]["gate_density"] == 1.0
    assert hist[0]["clipped_frac_gated"] > 0.99
