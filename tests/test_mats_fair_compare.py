"""Focused tests for the optimized fair-comparison driver."""

from collections import OrderedDict
from types import SimpleNamespace

import torch

from scripts.mats_fair_compare import refine_grid, selected_baselines


def test_family_filter_skips_unrequested_baseline_searches():
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(data=SimpleNamespace(n_probe=32)),
        task_names=["a", "b"],
        base_encoder=OrderedDict({"w": torch.tensor([0.0])}),
        task_vectors={
            "a": OrderedDict({"w": torch.tensor([1.0])}),
            "b": OrderedDict({"w": torch.tensor([3.0])}),
        },
    )
    objective_calls = []

    def objective(state):
        objective_calls.append(state)
        return float(state["w"].square().sum())

    baselines, metadata = selected_baselines(
        ctx, objective, requested=["Average"])

    assert list(baselines) == ["Average"]
    assert list(metadata) == ["Average"]
    assert len(objective_calls) == 1


def test_gd_grid_runs_one_trajectory_per_learning_rate():
    class FakeContext:
        def __init__(self):
            self.cfg = SimpleNamespace(seed=0)
            self.calls = []

        def run_refine_checkpoints_from(self, start, cfg, steps, seed=0):
            self.calls.append((cfg.lr, cfg.steps, tuple(steps), seed))
            states = OrderedDict(
                (step, OrderedDict({"w": torch.tensor([cfg.lr + step])}))
                for step in steps)
            history = [
                {"sweep": sweep, "gate_density": 1.0}
                for sweep in range(cfg.steps)
            ]
            return states, history

    ctx = FakeContext()
    start = OrderedDict({"w": torch.tensor([0.0])})
    _state, info = refine_grid(
        ctx, start, lambda state: float(state["w"]), "GD",
        apr_lrs=[], apr_steps=[], gd_lrs=[1.0, 2.0], gd_steps=[20, 40, 80])

    assert ctx.calls == [
        (1.0, 80, (20, 40, 80), 0),
        (2.0, 80, (20, 40, 80), 0),
    ]
    assert [row["params"] for row in info["trace"]] == [
        {"steps": step, "lr": lr, "schedule": "constant"}
        for step in (20, 40, 80)
        for lr in (1.0, 2.0)
    ]
