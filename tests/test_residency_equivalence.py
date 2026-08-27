"""Keeping models on the device must not change a single bit of the result.

refine() shuttles each task model to the GPU and back per task-step, and copies
the expert encoder over alongside; profiling put roughly two thirds of a step in
those transfers. The resident path removes them. It is only adoptable
mid-campaign if it is EXACTLY equivalent -- otherwise runs made before and after
the change could not share a table.

Two things are checked:

  * the refined weights and the whole per-step history (gate density, clip
    counts, step norms) are identical whether the model moves or not;
  * the de-synced diagnostic counters -- accumulated on device and read back
    once per step rather than per tensor -- still equal the per-tensor counts.
    gate density and clip counts are exact integers, so these must match
    exactly; ap_sum is a float64 reduction and is only checked to tolerance.

The equivalence is device-independent, so it is exercised on CPU (where
move_model is a no-op for placement but still gates the empty_cache and the
expert copy) and on CUDA when one is available.
"""

import os
import sys
from collections import OrderedDict

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import RefineConfig
from apr.refine import refine, TaskHandle, _compute_update

from test_refine import Quad


def _make_handle(name, d, expert_vec, target_vec, device):
    """Device-aware version of test_refine.make_handle.

    The shared fixture closes over a CPU target, which breaks under
    move_model=True on CUDA: refine() moves the model to the device before
    calling grad_fn, so the parameters are on CUDA while the target is not.
    Here the model, expert and target all live on `device` from the start,
    which is also what the resident path assumes of a real context.
    """
    model = Quad(d).to(device)
    target = target_vec.to(device)
    expert = OrderedDict({"enc.w": expert_vec.to(device)})

    def grad_fn():
        w = dict(model.named_parameters())["enc.w"].detach()
        return OrderedDict({"enc.w": (w - target.to(w.device)).clone()})

    return TaskHandle(name, model, expert, grad_fn)


def _setup(d=512, n_tasks=3, seed=5, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    handles = []
    for i in range(n_tasks):
        expert = torch.randn(d, generator=g)
        target = torch.randn(d, generator=g)
        handles.append(_make_handle(f"t{i}", d, expert, target, device))
    base = OrderedDict({"enc.w": torch.randn(d, generator=g).to(device)})
    return base, handles


def _cfg(steps=4, lr=0.8):
    return RefineConfig(steps=steps, lr=lr, order="random",
                        gate_mode="coordinate", lr_schedule="constant",
                        clip_mode="vdist")


def _run(move_model, device="cpu"):
    base, handles = _setup(device=device)
    return refine(base, handles, _cfg(), device, seed=11, move_model=move_model)


def test_resident_and_shuttled_are_bit_identical_cpu():
    shuttled, h_shut = _run(move_model=True)
    resident, h_res = _run(move_model=False)
    for k in shuttled:
        assert torch.equal(shuttled[k], resident[k]), \
            f"resident path changed the weights at {k}"
    assert len(h_shut) == len(h_res)
    for a, b in zip(h_shut, h_res):
        assert a["task"] == b["task"] and a["sweep"] == b["sweep"]
        assert a["gate_density"] == b["gate_density"]
        assert a["clipped_frac_gated"] == b["clipped_frac_gated"]
        assert a["update_norm"] == b["update_norm"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_resident_and_shuttled_are_bit_identical_cuda():
    shuttled, _ = _run(move_model=True, device="cuda")
    resident, _ = _run(move_model=False, device="cuda")
    for k in shuttled:
        assert torch.equal(shuttled[k].cpu(), resident[k].cpu()), \
            f"resident path changed the weights at {k} on CUDA"


def test_desynced_counters_match_per_tensor_counts():
    """The on-device accumulators must agree with the counts they replaced."""
    torch.manual_seed(3)
    g = OrderedDict((f"t{i}", torch.randn(64)) for i in range(6))
    v = OrderedDict((f"t{i}", torch.randn(64)) for i in range(6))
    cfg = _cfg(lr=2.0)
    _u, masks, stats = _compute_update(g, v, cfg, None, None, lr_eff=cfg.lr)

    gated = clipped = total = 0
    ap = 0.0
    for name in g:
        gi, vi = g[name], v[name]
        m = masks[name]
        pre = cfg.lr * (-gi * vi.abs()) * m
        clipped += int(((pre.abs() > vi.abs()) & (m > 0)).sum())
        gated += int((m > 0).sum())
        total += m.numel()
        ap += float((gi * vi).sum())

    # exact: these are integer counts, and the update depends on the same masks
    assert stats["gate_density"] == gated / total
    assert stats["clipped_frac_gated"] == clipped / max(gated, 1)
    assert stats["clipped_frac_all"] == clipped / total
    # approximate, and deliberately so: the accumulator now sums the per-tensor
    # reductions in float64 instead of adding float32 results in Python, which
    # is strictly more accurate but differs in the last digits. ap_sum is
    # logged only -- nothing reads it back -- so float32 epsilon is the right
    # tolerance here, not exact equality.
    assert stats["ap_sum"] == pytest.approx(ap, rel=1e-6)
