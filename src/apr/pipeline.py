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
import hashlib
import json
import os
import random
import time

import numpy as np
import torch

from .config import ExperimentConfig, RefineConfig
from .tasks import get_task
from .models import (load_tokenizer, load_classifier, get_encoder_state,
                     load_encoder_state, ParamDict, pd_sub, pd_global_norm)
from .data import (load_task_dataset, sample_replay_buffer,
                   sample_replay_buffer_split, batches_from_buffer, make_collator)
from .gradients import make_grad_fn
from .taskvec import task_vector, task_arithmetic_merge
from .refine import refine, TaskHandle
from .eval import evaluate_task
from .metrics import normalized_retention, aggregate_retention


EVAL0_CACHE_DIR = "results/eval0_cache"
# bump when eval SEMANTICS change (prompt templates, scoring rules, eval-set
# construction) in a way the config key below cannot see. Forgetting to bump
# reuses stale scores silently -- when in doubt, bump.
EVAL0_CACHE_VERSION = 1


def _eval0_cache_key(cfg: ExperimentConfig) -> dict:
    """Everything that determines base/expert/merge scores. Scores are
    deterministic given these (verified across runs), but only per batch size
    (kernel selection differs), hence eval_batch_size is part of the key."""
    return {
        "v": EVAL0_CACHE_VERSION,
        "base_model": cfg.base_model, "modality": cfg.modality,
        "model_dtype": cfg.model_dtype, "eval_dtype": cfg.eval_dtype,
        "experts": [{"name": e.name, "checkpoint": e.checkpoint, "lam": e.lam}
                    for e in cfg.experts],
        "max_eval": cfg.data.max_eval,
        "max_eval_by_task": cfg.data.max_eval_by_task,
        "eval_batch_size": cfg.data.eval_batch_size,
        "max_length": cfg.data.max_length,
    }


def _eval0_cache_path(cfg: ExperimentConfig) -> str:
    key = json.dumps(_eval0_cache_key(cfg), sort_keys=True)
    return os.path.join(EVAL0_CACHE_DIR,
                        hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")


def _eval0_cache_load(cfg: ExperimentConfig):
    path = _eval0_cache_path(cfg)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        blob = json.load(f)
    if blob.get("key") != _eval0_cache_key(cfg):  # hash collision guard
        return None
    _log(f"[eval0] loaded from cache {path} (delete the file to force re-scoring)")
    return blob


def _eval0_cache_save(cfg: ExperimentConfig, base, expert, merge):
    os.makedirs(EVAL0_CACHE_DIR, exist_ok=True)
    path = _eval0_cache_path(cfg)
    with open(path, "w") as f:
        json.dump({"key": _eval0_cache_key(cfg), "base_scores": base,
                   "expert_scores": expert, "merge_scores": merge}, f, indent=2)
    _log(f"[eval0] cached -> {path}")


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

    if cfg.modality == "t5":
        from .t5_gen import (build_t5_assets, get_t5_task, load_t5_task_dataset,
                             make_t5_collator, T5GenClassifier, T5_EXPERT_CKPT)
        from transformers import T5ForConditionalGeneration
        dtype = getattr(torch, cfg.model_dtype)
        tokenizer, base = build_t5_assets(cfg.base_model, cache_dir=cache,
                                          torch_dtype=dtype)
        base_encoder = get_encoder_state(T5GenClassifier(base, tokenizer,
                                                         get_t5_task("cola")))
        del base

        def load_expert(e):
            spec = get_t5_task(e.name)
            ckpt = e.checkpoint or T5_EXPERT_CKPT[spec.name]
            _log(f"[load] t5 expert {spec.name} <- {ckpt}")
            t5 = T5ForConditionalGeneration.from_pretrained(
                ckpt, cache_dir=cache, torch_dtype=dtype)
            model = T5GenClassifier(t5, tokenizer, spec)
            train_ds, eval_ds = load_t5_task_dataset(spec, tokenizer,
                                                     cfg.data.max_length, cache)
            if cfg.data.max_eval is not None:
                eval_ds = eval_ds.select(range(min(cfg.data.max_eval, len(eval_ds))))
            collator = make_t5_collator(tokenizer, spec.is_regression)
            return spec, model, train_ds, eval_ds, collator

        return base_encoder, load_expert

    if cfg.modality == "t5_mats":
        from .mats_t5 import (build_mats_assets, get_mats_task,
                              load_mats_task_dataset, make_mats_collator,
                              MatsT5Model, load_mats_expert, mats_checkpoint)
        dtype = getattr(torch, cfg.model_dtype)
        tokenizer, base = build_mats_assets(cfg.base_model, cache_dir=cache,
                                            torch_dtype=dtype)
        base_encoder = get_encoder_state(
            MatsT5Model(base, tokenizer, get_mats_task("cosmos_qa")))
        del base
        collator = make_mats_collator(tokenizer)

        def load_expert(e):
            spec = get_mats_task(e.name)
            ckpt = e.checkpoint or mats_checkpoint(spec.name)
            _log(f"[load] MaTS T5 expert {spec.name} <- {ckpt}")
            t5 = load_mats_expert(cfg.base_model, ckpt, cache_dir=cache,
                                  torch_dtype=dtype)
            model = MatsT5Model(t5, tokenizer, spec)
            train_ds, eval_ds = load_mats_task_dataset(
                spec, tokenizer, cfg.data.max_length, cache)
            if cfg.data.max_eval is not None:
                eval_ds = eval_ds.select(range(min(cfg.data.max_eval, len(eval_ds))))
            return spec, model, train_ds, eval_ds, collator

        return base_encoder, load_expert

    if cfg.modality == "t5_mats_ia3":
        from .mats_t5 import (build_mats_ia3_assets, get_mats_task,
                              load_mats_task_dataset, make_mats_collator,
                              MatsIA3Model, apply_mats_ia3_expert,
                              MATS_IA3_REPOS)
        dtype = getattr(torch, cfg.model_dtype)
        tokenizer, base = build_mats_ia3_assets(
            cfg.base_model, cache_dir=cache, torch_dtype=dtype)
        shared_model = MatsIA3Model(base, tokenizer, get_mats_task("cosmos_qa"))
        base_encoder = get_encoder_state(shared_model)
        collator = make_mats_collator(tokenizer)

        def load_expert(e):
            spec = get_mats_task(e.name)
            source = e.checkpoint or spec.name
            shown = e.checkpoint or MATS_IA3_REPOS[spec.name]
            _log(f"[load] MaTS IA3 expert {spec.name} <- {shown}")
            # All tasks share the frozen T5 and LM head. Only the tiny IA3 state
            # is replaced before it is snapshotted by MergeContext.build.
            model = apply_mats_ia3_expert(shared_model, source, cache_dir=cache)
            train_ds, eval_ds = load_mats_task_dataset(
                spec, tokenizer, cfg.data.max_length, cache)
            if cfg.data.max_eval is not None:
                eval_ds = eval_ds.select(range(min(cfg.data.max_eval, len(eval_ds))))
            return spec, model, train_ds, eval_ds, collator

        return base_encoder, load_expert

    if cfg.modality == "causal_lm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .causal_lm import (get_causal_task, make_causal_collator,
                                MERGEBENCH_EXPERT)
        dtype = getattr(torch, cfg.model_dtype)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, cache_dir=cache)
        # base decoder-only checkpoints (e.g. Llama-3.2-3B) ship no pad token, which
        # breaks batched padding in both the SFT collator and generation eval. Reuse
        # eos as pad (attention_mask masks it; no vocab resize, so task vectors are
        # unaffected). Set on the tokenizer object so tok(..., padding=True) sees it.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_short = cfg.base_model.split("/")[-1]
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, cache_dir=cache, torch_dtype=dtype)
        base_encoder = get_encoder_state(base_model)
        prompt_style = ("chat" if getattr(tokenizer, "chat_template", None)
                        else "plain")
        del base_model
        collator = make_causal_collator(tokenizer)

        gen_dtype = getattr(torch, cfg.eval_dtype) if cfg.eval_dtype else None

        def load_expert(e):
            spec = get_causal_task(e.name)
            spec.tokenizer = tokenizer
            spec.prompt_style = prompt_style
            spec.gen_dtype = gen_dtype  # bf16 generation eval; weights stay fp32
            ckpt = e.checkpoint or MERGEBENCH_EXPERT.format(base=base_short,
                                                            domain=spec.name)
            _log(f"[load] causal expert {spec.name} <- {ckpt} "
                 f"(prompt_style={prompt_style}, dtype={cfg.model_dtype})")
            model = AutoModelForCausalLM.from_pretrained(
                ckpt, cache_dir=cache, torch_dtype=dtype)
            # some MergeBench experts pad the embedding matrix beyond the base vocab
            # (e.g. Llama-3.2-3B_math: 128320 vs 128256); the shared base tokenizer
            # never produces ids >= len(tokenizer), so truncating back is lossless
            # and keeps the task vector well-defined against base_encoder.
            n_embed = model.get_input_embeddings().weight.shape[0]
            if n_embed != len(tokenizer):
                _log(f"[load] resize {spec.name} embeddings {n_embed} -> "
                     f"{len(tokenizer)} (expert checkpoint padding)")
                model.resize_token_embeddings(len(tokenizer))
            train_raw, make_fn = spec.build_replay(tokenizer, cfg.data.max_length,
                                                   cache, prompt_style)
            max_eval = (cfg.data.max_eval_by_task or {}).get(spec.name,
                                                             cfg.data.max_eval)
            eval_rows = spec.build_eval(cache, max_eval)
            # pre-materialise the SFT view so sample_replay_buffer indexes it
            train_ds = _CausalReplayView(train_raw, make_fn)
            return spec, model, train_ds, eval_rows, collator

        return base_encoder, load_expert

    raise ValueError(f"Unknown modality '{cfg.modality}' "
                     f"(expected 'glue', 'clip', 't5', 't5_mats', "
                     f"'t5_mats_ia3' or 'causal_lm')")


class _CausalReplayView:
    """List-like lazy view: train_ds[i] -> tokenized SFT example. Lets the
    shared sample_replay_buffer index causal-LM replay data without
    materialising the whole SFT dataset."""

    def __init__(self, raw_ds, make_fn):
        self.raw = raw_ds
        self.make = make_fn

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        return self.make(self.raw[i])


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
                 base_scores, expert_scores, merge_scores,
                 task_vectors=None, lambdas=None):
        self.cfg = cfg
        self.device = device
        self.handles = handles
        self.per_task = per_task
        self.base_encoder = base_encoder
        self.merged0 = merged0
        self.base_scores = base_scores
        self.expert_scores = expert_scores
        self.merge_scores = merge_scores
        # task vectors tau_i and their lambda_i, kept so alternative merge
        # points (TIES/DARE, see apr.merge_methods) can be built from the same
        # loaded experts without re-loading (used by scripts/merge_baselines.py).
        self.task_vectors = task_vectors or {}
        self.lambdas = lambdas or {}
        self.shared_model = len({id(h.model) for h in handles}) == 1
        self.keep_model_on_device = self.shared_model and cfg.modality == "t5_mats_ia3"
        if self.keep_model_on_device:
            handles[0].model.to(device)

    @property
    def task_names(self) -> List[str]:
        return self.cfg.task_names

    def eval_encoder(self, encoder_state: ParamDict, names=None) -> Dict[str, float]:
        out = {}
        for name in (names or self.task_names):
            info = self.per_task[name]
            load_encoder_state(info["model"], encoder_state)
            t0 = time.time()
            out[name] = evaluate_task(info["model"], info["eval_ds"], info["spec"],
                                      info["collator"], self.cfg.data.eval_batch_size,
                                      self.device,
                                      num_workers=self.cfg.data.eval_num_workers)["primary"]
            # per-eval-pass heartbeat: generation eval can run many minutes silently
            _log(f"[eval] {name}: {len(info['eval_ds'])} items -> {out[name]:.4f} "
                 f"({time.time()-t0:.0f}s)")
            if not self.keep_model_on_device:
                info["model"].to("cpu")
            if self.device.startswith("cuda") and not self.keep_model_on_device:
                torch.cuda.empty_cache()
        return out

    def normret(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {n: normalized_retention(scores[n], self.base_scores[n],
                                        self.expert_scores[n])
                for n in self.task_names}

    def resample_buffers(self, n_probe: int, probe_seed: int):
        """Re-draw each task's replay buffer (size n_probe, seed probe_seed), rebuild
        its gradient closure, AND update ``per_task[name]["probe_buffer"]`` so that
        every buffer-consuming method (APR grads, but also Fisher, head-only, matched
        AdaMerging, and the replay objective) sees the reseeded buffer. Lets a budget
        or multi-seed sweep vary n / the seed without reloading experts.
        When data.n_val > 0 the train/val split is preserved (val stays disjoint)."""
        for h in self.handles:
            info = self.per_task[h.name]
            if self.cfg.data.n_val > 0:
                buf, val, indices, val_indices = sample_replay_buffer_split(
                    info["train_ds"], info["spec"], n_probe, self.cfg.data.n_val,
                    probe_seed, self.cfg.data.class_balanced, return_indices=True)
                info["val_buffer"] = val
                info["val_indices"] = val_indices
            else:
                buf, indices = sample_replay_buffer(
                    info["train_ds"], info["spec"], n_probe, probe_seed,
                    self.cfg.data.class_balanced, return_indices=True)
            info["probe_buffer"] = buf
            info["probe_indices"] = indices
            h.grad_fn = make_grad_fn(info["model"], buf, info["collator"],
                                     self.cfg.data.grad_batch_size
                                     or self.cfg.data.eval_batch_size, self.device)

    @torch.no_grad()
    def val_scores(self, encoder_state: ParamDict, names=None) -> Dict[str, float]:
        """Per-task score on the held-out val buffer (data.n_val > 0): accuracy
        for classification, -MSE for regression. Selection WITHOUT the test set."""
        out = {}
        for name in (names or self.task_names):
            info = self.per_task[name]
            vb = info.get("val_buffer")
            if not vb:
                raise ValueError("val_scores needs data.n_val > 0")
            load_encoder_state(info["model"], encoder_state)
            info["model"].to(self.device).eval()
            correct, total, sq = 0, 0, 0.0
            for batch in batches_from_buffer(vb, info["collator"],
                                             self.cfg.data.eval_batch_size, self.device):
                labels = batch["labels"]
                logits = info["model"](**batch).logits
                if info["spec"].is_regression:
                    sq += float(((logits.squeeze(-1) - labels) ** 2).sum())
                else:
                    correct += int((logits.argmax(-1) == labels).sum())
                total += labels.numel()
            out[name] = (-sq / max(total, 1)) if info["spec"].is_regression \
                else correct / max(total, 1)
            info["model"].to("cpu")
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return out

    def run_refine_from(self, start_state: ParamDict, refine_cfg: RefineConfig,
                        seed: int = 0):
        """Refine from an arbitrary merge point; return (refined_cpu, history).

        The expert anchoring (v = theta_i - theta) is read from the task handles
        and is independent of the start point, so APR can be run on top of any
        merge (task arithmetic, TIES, DARE, ...)."""
        refined, history = refine(start_state, self.handles, refine_cfg,
                                  self.device, seed=seed,
                                  move_model=not self.keep_model_on_device, logger=_log)
        refined_cpu = OrderedDict((k, v.cpu()) for k, v in refined.items())
        return refined_cpu, history

    def run_refine_checkpoints_from(self, start_state: ParamDict,
                                    refine_cfg: RefineConfig,
                                    checkpoint_steps, seed: int = 0):
        """Refine once and retain CPU states at selected sweep counts.

        This is intended for constant-schedule horizon grids: the state after,
        for example, 20 sweeps of an 80-sweep constant-LR trajectory is exactly
        the state produced by a separate 20-sweep run.  Horizon-dependent
        schedules such as cosine decay must continue to use separate runs.
        """
        wanted = set(checkpoint_steps)
        invalid = sorted(step for step in wanted
                         if step < 0 or step > refine_cfg.steps)
        if invalid:
            raise ValueError(f"checkpoint steps outside [0, {refine_cfg.steps}]: "
                             f"{invalid}")

        checkpoints = OrderedDict()

        def save_checkpoint(step, state):
            if step in wanted:
                checkpoints[step] = OrderedDict(
                    (name, value.detach().cpu().clone())
                    for name, value in state.items())

        if 0 in wanted:
            save_checkpoint(0, start_state)
        _refined, history = refine(
            start_state, self.handles, refine_cfg, self.device, seed=seed,
            move_model=not self.keep_model_on_device, logger=_log,
            checkpoint_callback=save_checkpoint)
        missing = wanted - set(checkpoints)
        if missing:
            raise RuntimeError(f"failed to capture refinement steps: "
                               f"{sorted(missing)}")
        return checkpoints, history

    def run_refine(self, refine_cfg: RefineConfig, seed: int = 0):
        """Refine from the shared task-arithmetic merge point."""
        return self.run_refine_from(self.merged0, refine_cfg, seed)

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
            if cfg.data.n_val > 0:
                buffer, val_buffer, probe_indices, val_indices = sample_replay_buffer_split(
                    train_ds, spec, cfg.data.n_probe, cfg.data.n_val,
                    cfg.data.probe_seed, cfg.data.class_balanced, return_indices=True)
            else:
                buffer, probe_indices = sample_replay_buffer(
                    train_ds, spec, cfg.data.n_probe, cfg.data.probe_seed,
                    cfg.data.class_balanced, return_indices=True)
                val_buffer = None
                val_indices = None
            grad_fn = make_grad_fn(model, buffer, collator,
                                   cfg.data.grad_batch_size or cfg.data.eval_batch_size,
                                   device)
            handles.append(TaskHandle(e.name, model, expert_encoder, grad_fn))
            # probe_buffer is kept so a label-free baseline can be given exactly the
            # same inputs as APR's replay buffer (labels stripped) -- see the
            # matched-budget AdaMerging variant in scripts/merge_baselines.py.
            per_task[e.name] = dict(spec=spec, model=model, eval_ds=eval_ds,
                                    collator=collator, expert_encoder=expert_encoder,
                                    train_ds=train_ds, probe_buffer=buffer,
                                    val_buffer=val_buffer, val_indices=val_indices,
                                    probe_indices=probe_indices)
            task_vectors[e.name] = task_vector(expert_encoder, base_encoder)
            lambdas[e.name] = e.lam

        merged0 = task_arithmetic_merge(base_encoder, task_vectors, lambdas)

        ctx = MergeContext(cfg, device, handles, per_task, base_encoder, merged0,
                           {}, {}, {}, task_vectors=task_vectors, lambdas=lambdas)
        # score base / expert / merge (one pass each)
        cached = _eval0_cache_load(cfg) if cfg.data.eval0_cache else None
        if cached is not None:
            ctx.base_scores = cached["base_scores"]
            ctx.expert_scores = cached["expert_scores"]
            ctx.merge_scores = cached["merge_scores"]
            suffix = " (cached)"
        else:
            ctx.base_scores = ctx.eval_encoder(base_encoder)
            ctx.expert_scores = {n: ctx.eval_encoder(per_task[n]["expert_encoder"],
                                                     names=[n])[n]
                                 for n in cfg.task_names}
            ctx.merge_scores = ctx.eval_encoder(merged0)
            if cfg.data.eval0_cache:
                _eval0_cache_save(cfg, ctx.base_scores, ctx.expert_scores,
                                  ctx.merge_scores)
            suffix = ""
        for n in cfg.task_names:
            _log(f"[eval0] {n}: base={ctx.base_scores[n]:.4f} "
                 f"expert={ctx.expert_scores[n]:.4f} "
                 f"merge={ctx.merge_scores[n]:.4f}{suffix}")
        return ctx


def run_experiment(cfg: ExperimentConfig) -> Dict:
    ctx = MergeContext.build(cfg)
    _log(f"[refine] S={cfg.refine.steps} lr={cfg.refine.lr} "
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
