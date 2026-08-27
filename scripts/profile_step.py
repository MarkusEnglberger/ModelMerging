"""Where does a task-step of refine() spend its time, and how fast is eval?

The CLIP-8 v2 run spends 353 ms per task-step (GLUE-8: 445 ms) although the
gradient itself -- forward+backward over 16 examples through a ~100M-parameter
encoder -- should take a few tens of milliseconds on an A100. This script
measures each component of one step exactly as refine() performs it:

  model.to(device) -> load_encoder_state -> grad_fn -> model.to("cpu") +
  empty_cache -> expert encoder to device -> v = expert - theta ->
  _compute_update -> apply -> pd_global_norm

then times refine() end to end for a few sweeps in two variants:

  current   models and experts live on the CPU and are moved per step
            (move_model=True, the path every v2 run has used);
  resident  every task model and expert encoder is placed on the device once
            and refine() runs with move_model=False.

and finally times one evaluation pass over the largest eval splits with the
DataLoader worker count the runs use (0) and with 8 workers.

No evaluation split is READ for any decision: the eval timing discards scores.

Usage:
  python scripts/profile_step.py --config configs/clip8.yaml --out results/profile/clip8.json
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import OrderedDict
from statistics import median

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig                                     # noqa: E402
from apr.pipeline import _make_backend, _log                                # noqa: E402
from apr.gradients import make_grad_fn                                      # noqa: E402
from apr.data import sample_replay_buffer                                   # noqa: E402
from apr.models import get_encoder_state, load_encoder_state               # noqa: E402
from apr.refine import (TaskHandle, refine, _compute_update, pd_to,        # noqa: E402
                        pd_clone, pd_global_norm)
from apr.eval import evaluate_task                                          # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class T:
    """Accumulate wall times of named blocks, synchronizing the device."""
    def __init__(self):
        self.t = {}
    def __call__(self, name):
        outer = self
        class _B:
            def __enter__(self):
                sync(); self.t0 = time.perf_counter()
            def __exit__(self, *a):
                sync(); outer.t.setdefault(name, []).append(time.perf_counter() - self.t0)
        return _B()
    def medians(self):
        return {k: median(v) for k, v in self.t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=4.0)
    ap.add_argument("--eval_tasks", type=int, default=1,
                    help="time eval on this many of the largest eval splits")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    _log(f"[load] {cfg.base_model} (modality={cfg.modality}) on {device} "
         f"{torch.cuda.get_device_name(0) if device.startswith('cuda') else ''}")
    base_encoder, load_expert = _make_backend(cfg, device)

    handles, per_task = [], {}
    bs = cfg.data.grad_batch_size or cfg.data.eval_batch_size
    for e in cfg.experts:
        spec, model, train_ds, eval_ds, collator = load_expert(e)
        expert_encoder = get_encoder_state(model)
        # the construction half of the budget, as in cv_protocol's selection
        buffer = sample_replay_buffer(train_ds, spec, args.budget // 2, args.seed,
                                      cfg.data.class_balanced)
        grad_fn = make_grad_fn(model, buffer, collator, bs, device)
        handles.append(TaskHandle(e.name, model, expert_encoder, grad_fn))
        per_task[e.name] = dict(spec=spec, model=model, eval_ds=eval_ds,
                                collator=collator)
    shared = len({id(h.model) for h in handles}) == 1
    n_par = sum(t.numel() for t in base_encoder.values())
    n_ten = len(base_encoder)
    model_bytes = sum(p.numel() * p.element_size() for p in handles[0].model.parameters())
    _log(f"[model] encoder {n_par/1e6:.1f}M params in {n_ten} tensors; "
         f"task model {model_bytes/2**20:.0f} MiB; shared model object: {shared}")

    rc = dataclasses.replace(cfg.refine, steps=args.sweeps, lr=args.lr,
                             order="random", lr_schedule="constant")
    report = {"config": args.config, "device": device, "modality": cfg.modality,
              "n_tasks": len(handles), "encoder_params": n_par,
              "encoder_tensors": n_ten, "task_model_MiB": model_bytes / 2**20,
              "shared_model": shared, "buffer_per_task": args.budget // 2}

    # ------------------------------------------------------------------
    # A. one task-step, component by component, exactly as refine() does it
    # ------------------------------------------------------------------
    tm = T()
    theta = pd_to(pd_clone(base_encoder), device)
    gen = torch.Generator(device=device); gen.manual_seed(0)
    enc_names = list(theta.keys())
    for rep in range(3):
        for h in handles[:3]:
            with tm("1 model.to(device)"):
                h.model.to(device)
            with tm("2 load_encoder_state"):
                load_encoder_state(h.model, theta)
            with tm("3 grad_fn (fwd+bwd, buffer)"):
                g = h.grad_fn()
            with tm("4 model.to(cpu)+empty_cache"):
                h.model.to("cpu"); torch.cuda.empty_cache()
            with tm("5 expert -> device"):
                expert = pd_to(h.expert_encoder, device)
            with tm("6 v = expert - theta"):
                v = OrderedDict((n, expert[n] - theta[n]) for n in enc_names)
                del expert
            with tm("7 _compute_update"):
                u, masks, stats = _compute_update(g, v, rc, gen, None, lr_eff=rc.lr)
            with tm("8 apply update"):
                for n in enc_names:
                    theta[n].add_(u[n], alpha=1.0)
            with tm("9 pd_global_norm(u)"):
                pd_global_norm(u)
            with tm("10 del + empty_cache"):
                del g, v, u, masks; torch.cuda.empty_cache()
    comp = tm.medians()
    total = sum(comp.values())
    report["components_ms"] = {k: v * 1e3 for k, v in comp.items()}
    _log("\n[A] one task-step, median over 9 (ms):")
    for k, v in comp.items():
        _log(f"   {k:32s} {v*1e3:8.1f}   ({100*v/total:4.1f}%)")
    _log(f"   {'TOTAL':32s} {total*1e3:8.1f}")
    del theta

    # ------------------------------------------------------------------
    # B. refine() end to end: current (shuttled) vs resident placement
    # ------------------------------------------------------------------
    steps = args.sweeps * len(handles)
    base_dev = pd_to(pd_clone(base_encoder), device)

    sync(); t0 = time.perf_counter()
    out_cur, _ = refine(base_dev, handles, rc, device, seed=0, move_model=True)
    sync(); t_cur = time.perf_counter() - t0
    _log(f"\n[B] refine() current  (move_model=True):  {t_cur:.1f}s for {steps} "
         f"task-steps = {1e3*t_cur/steps:.0f} ms/step")

    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    for h in handles:
        h.model.to(device)
        h.expert_encoder = pd_to(h.expert_encoder, device)
    sync(); t0 = time.perf_counter()
    out_res, _ = refine(base_dev, handles, rc, device, seed=0, move_model=False)
    sync(); t_res = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device.startswith("cuda") else 0
    same = all(torch.equal(out_cur[n], out_res[n]) for n in out_cur)
    _log(f"[B] refine() resident (move_model=False): {t_res:.1f}s = "
         f"{1e3*t_res/steps:.0f} ms/step  (x{t_cur/max(t_res,1e-9):.1f} faster); "
         f"peak device memory {peak:.1f} GiB; outputs bit-identical: {same}")
    report["refine_current_ms_per_step"] = 1e3 * t_cur / steps
    report["refine_resident_ms_per_step"] = 1e3 * t_res / steps
    report["resident_peak_GiB"] = peak
    report["resident_bit_identical"] = same
    del out_cur, out_res, base_dev

    # ------------------------------------------------------------------
    # C. evaluation throughput vs DataLoader workers
    # ------------------------------------------------------------------
    for h in handles:
        h.model.to("cpu")
    torch.cuda.empty_cache()
    biggest = sorted(per_task, key=lambda n: -len(per_task[n]["eval_ds"]))[:args.eval_tasks]
    report["eval"] = {}
    _log("\n[C] evaluation pass (items/s); evaluate_task builds autograd graphs "
         "(no torch.no_grad), so the last row wraps it in no_grad:")
    settings = [(nw, False) for nw in sorted({0, cfg.data.eval_num_workers, 16})]
    settings.append((cfg.data.eval_num_workers, True))
    for name in biggest:
        info = per_task[name]
        row = {"items": len(info["eval_ds"])}
        for nw, nograd in settings:
            key = f"workers={nw}" + ("+no_grad" if nograd else "")
            try:
                sync(); t0 = time.perf_counter()
                ctx = torch.no_grad() if nograd else torch.enable_grad()
                with ctx:
                    evaluate_task(info["model"], info["eval_ds"], info["spec"],
                                  info["collator"], cfg.data.eval_batch_size, device,
                                  num_workers=nw)
                sync(); dt = time.perf_counter() - t0
                row[key] = dt
                _log(f"   {name:10s} {len(info['eval_ds']):6d} items  {key:20s}: "
                     f"{dt:6.1f}s = {len(info['eval_ds'])/dt:7.0f} items/s")
            except Exception as ex:                       # noqa: BLE001
                row[key] = f"error: {type(ex).__name__}: {ex}"
                _log(f"   {name:10s} {key}: {row[key]}")
            info["model"].to("cpu"); torch.cuda.empty_cache()
        report["eval"][name] = row

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    _log(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
