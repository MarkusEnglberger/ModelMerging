"""End-to-end merge + refine + evaluate pipeline.

Glues the modules together for one ExperimentConfig:
  1. load base (theta_0) and each expert (encoder theta_i + fixed head),
  2. build task vectors and the task-arithmetic merge theta^(0) (Eq. 4),
  3. evaluate theta_0, each expert, and theta^(0) per task,
  4. run Algorithm 1 refinement,
  5. evaluate the refined encoder and compute normalized retention (Eq. 21).

The expensive setup (1-3) is factored into `MergeContext.build` so several
refinement rules can be compared from the *same* merge point and replay buffers
(see scripts/compare_baselines.py).
"""

from collections import OrderedDict
from typing import Dict, List
import random

import numpy as np
import torch

from .config import ExperimentConfig, RefineConfig
from .tasks import get_task
from .models import (load_tokenizer, load_classifier, get_encoder_state,
                     load_encoder_state, ParamDict, pd_sub, pd_global_norm)
from .data import (load_task_dataset, sample_replay_buffer, make_collator)
from .gradients import make_grad_fn
from .taskvec import task_vector, task_arithmetic_merge
from .refine import refine, TaskHandle
from .eval import evaluate_task
from .metrics import normalized_retention, aggregate_retention


def _make_backend(cfg: ExperimentConfig, device: str):
    """Return (base_encoder, load_expert) for the configured modality.

    ``load_expert(expert_cfg) -> (spec, model, train_ds, eval_ds, collator)``.
    Everything after this (task vectors, replay buffers, gradients, merge, eval)
    is modality-agnostic and shared.
    """
    cache = cfg.data.cache_dir
    if cfg.modality == "glue":
        tokenizer = load_tokenizer(cfg.base_model, cache_dir=cache)
        base_model = load_classifier(cfg.base_model, num_labels=2, cache_dir=cache)
        base_encoder = get_encoder_state(base_model)
        del base_model

        def load_expert(e):
            spec = get_task(e.name)
            _log(f"[load] expert {e.name} <- {e.checkpoint} (num_labels={spec.num_labels})")
            model = load_classifier(e.checkpoint, spec.num_labels, cache_dir=cache)
            train_ds, eval_ds = load_task_dataset(spec, tokenizer, cfg.data.max_length, cache)
            if cfg.data.max_eval is not None:
                eval_ds = eval_ds.select(range(min(cfg.data.max_eval, len(eval_ds))))
            collator = make_collator(tokenizer, spec.is_regression)
            return spec, model, train_ds, eval_ds, collator

        return base_encoder, load_expert

    if cfg.modality == "clip":
        from .vision import (VisionAssets, get_vision_task, VISION_EXPERT_CKPT,
                             load_vision_dataset, make_image_collator)
        assets = VisionAssets(cfg.base_model, cache_dir=cache)
        base_encoder = assets.base_vision_state
        collator = make_image_collator(assets.processor)

        def load_expert(e):
            spec = get_vision_task(e.name)
            ckpt = e.checkpoint or VISION_EXPERT_CKPT[spec.name]
            _log(f"[load] vision expert {spec.name} <- {ckpt}")
            train_ds, eval_ds, classnames = load_vision_dataset(spec, cache_dir=cache)
            if cfg.data.max_eval is not None:
                eval_ds = eval_ds.select(range(min(cfg.data.max_eval, len(eval_ds))))
            model = assets.build_classifier(spec, classnames, ckpt)
            return spec, model, train_ds, eval_ds, collator

        return base_encoder, load_expert

    raise ValueError(f"Unknown modality '{cfg.modality}' (expected 'glue' or 'clip')")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log(msg: str):
    print(msg, flush=True)


class MergeContext:
    """Reusable setup: experts loaded, merge built, base/expert/merge scored."""

    def __init__(self, cfg, device, handles, per_task, base_encoder, merged0,
                 base_scores, expert_scores, merge_scores):
        self.cfg = cfg
        self.device = device
        self.handles = handles
        self.per_task = per_task
        self.base_encoder = base_encoder
        self.merged0 = merged0
        self.base_scores = base_scores
        self.expert_scores = expert_scores
        self.merge_scores = merge_scores

    @property
    def task_names(self) -> List[str]:
        return self.cfg.task_names

    def eval_encoder(self, encoder_state: ParamDict, names=None) -> Dict[str, float]:
        out = {}
        for name in (names or self.task_names):
            info = self.per_task[name]
            load_encoder_state(info["model"], encoder_state)
            out[name] = evaluate_task(info["model"], info["eval_ds"], info["spec"],
                                      info["collator"], self.cfg.data.eval_batch_size,
                                      self.device,
                                      num_workers=self.cfg.data.eval_num_workers)["primary"]
            info["model"].to("cpu")
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return out

    def normret(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {n: normalized_retention(scores[n], self.base_scores[n],
                                        self.expert_scores[n])
                for n in self.task_names}

    def resample_buffers(self, n_probe: int, probe_seed: int):
        """Re-draw each task's replay buffer (size n_probe, seed probe_seed) and
        rebuild its gradient closure, reusing the already-loaded models/datasets.
        Lets a budget sweep vary n without reloading experts."""
        for h in self.handles:
            info = self.per_task[h.name]
            buf = sample_replay_buffer(info["train_ds"], info["spec"], n_probe,
                                       probe_seed, self.cfg.data.class_balanced)
            h.grad_fn = make_grad_fn(info["model"], buf, info["collator"],
                                     self.cfg.data.eval_batch_size, self.device)

    def run_refine(self, refine_cfg: RefineConfig, seed: int = 0):
        """Refine from the shared merge point; return (refined_cpu, history)."""
        refined, history = refine(self.merged0, self.handles, refine_cfg,
                                  self.device, seed=seed, move_model=True, logger=_log)
        refined_cpu = OrderedDict((k, v.cpu()) for k, v in refined.items())
        return refined_cpu, history

    @staticmethod
    def build(cfg: ExperimentConfig) -> "MergeContext":
        set_seed(cfg.seed)
        device = cfg.device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            _log("[warn] CUDA not available; running on CPU.")

        _log(f"[load] base model {cfg.base_model} (modality={cfg.modality})")
        base_encoder, load_expert = _make_backend(cfg, device)

        handles, per_task = [], {}
        task_vectors, lambdas = {}, {}
        for e in cfg.experts:
            spec, model, train_ds, eval_ds, collator = load_expert(e)
            expert_encoder = get_encoder_state(model)
            buffer = sample_replay_buffer(train_ds, spec, cfg.data.n_probe,
                                          cfg.data.probe_seed, cfg.data.class_balanced)
            grad_fn = make_grad_fn(model, buffer, collator, cfg.data.eval_batch_size, device)
            handles.append(TaskHandle(e.name, model, expert_encoder, grad_fn))
            per_task[e.name] = dict(spec=spec, model=model, eval_ds=eval_ds,
                                    collator=collator, expert_encoder=expert_encoder,
                                    train_ds=train_ds)
            task_vectors[e.name] = task_vector(expert_encoder, base_encoder)
            lambdas[e.name] = e.lam

        merged0 = task_arithmetic_merge(base_encoder, task_vectors, lambdas)

        ctx = MergeContext(cfg, device, handles, per_task, base_encoder, merged0,
                           {}, {}, {})
        # score base / expert / merge (one pass each)
        ctx.base_scores = ctx.eval_encoder(base_encoder)
        ctx.expert_scores = {n: ctx.eval_encoder(per_task[n]["expert_encoder"], names=[n])[n]
                             for n in cfg.task_names}
        ctx.merge_scores = ctx.eval_encoder(merged0)
        for n in cfg.task_names:
            _log(f"[eval0] {n}: base={ctx.base_scores[n]:.4f} "
                 f"expert={ctx.expert_scores[n]:.4f} merge={ctx.merge_scores[n]:.4f}")
        return ctx


def run_experiment(cfg: ExperimentConfig) -> Dict:
    ctx = MergeContext.build(cfg)
    _log(f"[refine] S={cfg.refine.steps} lr={cfg.refine.lr} gamma={cfg.refine.clip_frac} "
         f"gate={cfg.refine.gate_mode} update={cfg.refine.update_mode} agg={cfg.refine.aggregated}")
    refined_cpu, history = ctx.run_refine(cfg.refine, seed=cfg.seed)
    refined_scores = ctx.eval_encoder(refined_cpu)

    merge_normret = ctx.normret(ctx.merge_scores)
    refined_normret = ctx.normret(refined_scores)
    results = {"config": cfg.to_dict(), "tasks": {}}
    for name in cfg.task_names:
        b, x = ctx.base_scores[name], ctx.expert_scores[name]
        m, r = ctx.merge_scores[name], refined_scores[name]
        results["tasks"][name] = {
            "base": b, "expert": x, "merge": m, "refined": r,
            "merge_normret": merge_normret[name],
            "refined_normret": refined_normret[name],
            "merge_drop": x - m, "refined_drop": x - r,
        }
        _log(f"[final] {name}: merge={m:.4f} refined={r:.4f} "
             f"normret {merge_normret[name]:.3f}->{refined_normret[name]:.3f}")

    results["aggregate"] = {
        "merge": aggregate_retention(merge_normret),
        "refined": aggregate_retention(refined_normret),
    }
    results["history"] = history
    return results
