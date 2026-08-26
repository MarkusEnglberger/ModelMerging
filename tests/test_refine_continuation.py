"""Trajectory continuation must be EXACTLY the tail of one longer run.

refine(start_sweep=N) continues a constant-schedule trajectory: the state a
previous call returned after N sweeps, refined for M further sweeps, must be
bit-identical to sweeps N..N+M-1 of a single (N+M)-sweep run -- including with
a per-sweep re-randomized task order, whose RNG stream is advanced by burning
N shuffles. cv_protocol relies on this to grow the horizon grid without
re-running trajectories.
"""

import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import RefineConfig
from apr.refine import refine

from test_refine import make_handle  # tiny quadratic task fixtures


def _setup(d=64, seed=7):
    g = torch.Generator().manual_seed(seed)
    handles = []
    for name in ("a", "b", "c"):
        expert = torch.randn(d, generator=g)
        target = torch.randn(d, generator=g)
        h, _ = make_handle(name, d, expert, target)
        handles.append(h)
    base = OrderedDict({"enc.w": torch.randn(d, generator=g)})
    return base, handles


def _cfg(steps):
    return RefineConfig(steps=steps, lr=0.7, order="random",
                        gate_mode="coordinate", lr_schedule="constant")


def test_continuation_equals_single_long_run():
    base, handles = _setup()
    full, hist_full = refine(base, handles, _cfg(9), "cpu", seed=3)

    head, hist_head = refine(base, handles, _cfg(4), "cpu", seed=3)
    tail, hist_tail = refine(head, handles, _cfg(5), "cpu", seed=3,
                             start_sweep=4)

    assert torch.equal(full["enc.w"], tail["enc.w"]), \
        "continued trajectory diverged from the single long run"
    # the task-visit sequence must also line up exactly
    order_full = [(h["sweep"], h["task"]) for h in hist_full]
    order_seg = [(h["sweep"], h["task"]) for h in hist_head + hist_tail]
    assert order_full == order_seg, "task order stream not reproduced"


def test_continuation_guards():
    base, handles = _setup()
    bad = RefineConfig(steps=2, lr=0.7, order="random", gate_mode="coordinate",
                       lr_schedule="cosine")
    try:
        refine(base, handles, bad, "cpu", seed=0, start_sweep=1)
    except ValueError:
        pass
    else:
        raise AssertionError("cosine schedule must reject start_sweep")
