"""Synthetic unit tests for the attribution gate and Algorithm 1.

Uses a tiny quadratic "task": L_i(theta) = 1/2 ||theta - target_i||^2, whose
gradient is exactly (theta - target_i). This lets us check the gate logic, the
trust-region clip, and the sequential update without any heavy model.

Run: python tests/test_refine.py   (or: pytest tests/test_refine.py)
"""

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import RefineConfig
from apr.refine import refine, TaskHandle, _gate, _compute_update


class Quad(nn.Module):
    """Holds an encoder parameter vector under prefix 'enc'."""
    base_model_prefix = "enc"

    def __init__(self, d):
        super().__init__()
        self.enc = nn.Module()
        self.enc.w = nn.Parameter(torch.zeros(d))


def make_handle(name, d, expert_vec, target_vec):
    model = Quad(d)
    expert = OrderedDict({"enc.w": expert_vec.clone()})

    def grad_fn():
        w = dict(model.named_parameters())["enc.w"].detach()
        # grad of 1/2||w - target||^2 is (w - target)
        return OrderedDict({"enc.w": (w - target_vec).clone()})

    return TaskHandle(name, model, expert, grad_fn), model


def loss_to(target, w):
    return 0.5 * float(((w - target) ** 2).sum())


def test_gate_matches_sign_condition():
    g = torch.tensor([1.0, -1.0, 2.0, -3.0])
    v = torch.tensor([-1.0, -1.0, 1.0, 1.0])  # g*v = [-1, 1, 2, -3]
    cfg = RefineConfig(gate_mode="coordinate", gate_eps=0.0)
    m = _gate(g, v, cfg, None)
    assert torch.equal(m, torch.tensor([1.0, 0.0, 0.0, 1.0])), m
    # inverted gate is the complement (excluding the ==0 boundary)
    cfg_inv = RefineConfig(gate_mode="inverted", gate_eps=0.0)
    mi = _gate(g, v, cfg_inv, None)
    assert torch.equal(mi, torch.tensor([0.0, 1.0, 1.0, 0.0])), mi
    print("ok: gate matches g*v<0 sign condition")


def test_clip_respects_trust_region():
    g = OrderedDict({"enc.w": torch.tensor([-10.0, -0.1])})
    v = OrderedDict({"enc.w": torch.tensor([2.0, 2.0])})  # g*v<0 both -> gated on
    cfg = RefineConfig(clip_frac=0.5, gate_mode="coordinate", update_mode="gated_grad")
    u, m, stats = _compute_update(g, v, cfg, None, None)
    bound = 0.5 * v["enc.w"].abs()
    assert torch.all(u["enc.w"].abs() <= bound + 1e-6), u["enc.w"]
    assert abs(stats["gate_density"] - 1.0) < 1e-9
    print("ok: clip respects +/- gamma|v| trust region")


def test_refine_descends_when_expert_is_optimum():
    torch.manual_seed(0)
    d = 8
    expert = torch.randn(d) * 3
    # target == expert: every coordinate has g*v<0, gate fully open
    h, model = make_handle("t", d, expert_vec=expert, target_vec=expert)
    start = torch.randn(d)
    base = OrderedDict({"enc.w": start.clone()})
    cfg = RefineConfig(steps=10, lr=1.0, clip_frac=0.5, gate_mode="coordinate")
    refined, hist = refine(base, [h], cfg, device="cpu", move_model=False)
    w = refined["enc.w"]
    # loss strictly decreased and moved toward the expert
    assert loss_to(expert, w) < loss_to(expert, start)
    assert (w - expert).norm() < (start - expert).norm()
    assert abs(hist[0]["gate_density"] - 1.0) < 1e-9
    print(f"ok: refine descends, dist {(start-expert).norm():.3f} -> {(w-expert).norm():.3f}")


def test_inverted_gate_blocks_progress():
    torch.manual_seed(1)
    d = 8
    expert = torch.randn(d) * 3
    h, model = make_handle("t", d, expert_vec=expert, target_vec=expert)
    start = torch.randn(d)
    base = OrderedDict({"enc.w": start.clone()})
    cfg = RefineConfig(steps=5, lr=1.0, gate_mode="inverted")
    refined, hist = refine(base, [h], cfg, device="cpu", move_model=False)
    # when target==expert, g*v<0 everywhere so inverted gate selects nothing
    assert torch.allclose(refined["enc.w"], start), "inverted gate should not move"
    assert hist[0]["gate_density"] == 0.0
    print("ok: inverted gate blocks all updates here")


def test_vdist_pre_is_safe_at_huge_lr():
    """Saturating trust region (clip after lr): for gamma<=1 the move never
    overshoots the expert, so even an enormous lr stays bounded (no divergence)."""
    torch.manual_seed(3)
    d = 16
    expert = torch.randn(d) * 5
    h, model = make_handle("t", d, expert_vec=expert, target_vec=expert)
    start = torch.randn(d)
    base = OrderedDict({"enc.w": start.clone()})
    cfg = RefineConfig(steps=5, lr=1e6, clip_frac=1.0, gate_mode="coordinate",
                       clip_mode="vdist_pre")
    refined, hist = refine(base, [h], cfg, device="cpu", move_model=False)
    w = refined["enc.w"]
    assert torch.isfinite(w).all(), "vdist_pre diverged at huge lr"
    # bounded by the expert: never lands farther from expert than it started
    assert (w - expert).norm() <= (start - expert).norm() + 1e-4
    # and with gamma=1 + huge lr it essentially reaches the expert
    assert (w - expert).norm() < 1e-3 * (start - expert).norm()
    print(f"ok: vdist_pre safe at lr=1e6, dist {(start-expert).norm():.2f} -> {(w-expert).norm():.2e}")


def test_vdist_pre_contrast_paper_clip_diverges():
    """The paper clip (clip then *lr) DOES diverge at huge lr -> motivates vdist_pre."""
    torch.manual_seed(3)
    d = 16
    expert = torch.randn(d) * 5
    h, _ = make_handle("t", d, expert_vec=expert, target_vec=expert)
    start = torch.randn(d)
    base = OrderedDict({"enc.w": start.clone()})
    cfg = RefineConfig(steps=5, lr=1e6, clip_frac=1.0, gate_mode="coordinate",
                       clip_mode="vdist")
    refined, _ = refine(base, [h], cfg, device="cpu", move_model=False)
    assert (refined["enc.w"] - expert).norm() > (start - expert).norm()
    print("ok: paper clip (vdist) overshoots/diverges at huge lr (as expected)")


def test_aggregated_vs_sequential_differ():
    torch.manual_seed(2)
    d = 6
    e1, e2 = torch.randn(d) * 2, torch.randn(d) * 2
    start = torch.randn(d)

    def run(aggregated):
        h1, _ = make_handle("a", d, e1, e1)
        h2, _ = make_handle("b", d, e2, e2)
        base = OrderedDict({"enc.w": start.clone()})
        cfg = RefineConfig(steps=3, lr=1.0, clip_frac=0.5, aggregated=aggregated)
        out, _ = refine(base, [h1, h2], cfg, device="cpu", move_model=False)
        return out["enc.w"]

    seq, agg = run(False), run(True)
    assert not torch.allclose(seq, agg, atol=1e-4), "sequential and aggregated should differ"
    print("ok: sequential != aggregated-U")


def test_constant_trajectory_checkpoints_match_independent_runs():
    """A constant-LR long run must contain the exact shorter-horizon states."""
    torch.manual_seed(4)
    d = 7
    expert = torch.randn(d)
    target = torch.randn(d)
    start = torch.randn(d)
    base = OrderedDict({"enc.w": start.clone()})

    captured = {}

    def capture(step, state):
        if step in {2, 5}:
            captured[step] = state["enc.w"].clone()

    h_long, _ = make_handle("t", d, expert, target)
    long_cfg = RefineConfig(steps=5, lr=0.2, gate_mode="none",
                            update_mode="grad", clip_mode="none",
                            lr_schedule="constant")
    long_state, _ = refine(base, [h_long], long_cfg, device="cpu",
                           move_model=False, checkpoint_callback=capture)

    for steps in (2, 5):
        h_short, _ = make_handle("t", d, expert, target)
        short_cfg = RefineConfig(steps=steps, lr=0.2, gate_mode="none",
                                 update_mode="grad", clip_mode="none",
                                 lr_schedule="constant")
        short_state, _ = refine(base, [h_short], short_cfg, device="cpu",
                                move_model=False)
        assert torch.equal(captured[steps], short_state["enc.w"])
    assert torch.equal(captured[5], long_state["enc.w"])
    print("ok: constant-LR checkpoints exactly match independent horizons")


if __name__ == "__main__":
    test_gate_matches_sign_condition()
    test_clip_respects_trust_region()
    test_refine_descends_when_expert_is_optimum()
    test_inverted_gate_blocks_progress()
    test_vdist_pre_is_safe_at_huge_lr()
    test_vdist_pre_contrast_paper_clip_diverges()
    test_aggregated_vs_sequential_differ()
    test_constant_trajectory_checkpoints_match_independent_runs()
    print("\nAll synthetic refinement tests passed.")
