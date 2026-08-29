"""Labeled-replay baselines: the same n_probe labeled examples per task as APR.

Every method here consumes EXACTLY APR's replay buffer -- ``per_task[t]["probe_buffer"]``,
sampled from the TRAIN split -- and nothing else. This is the comparison the proposal
actually needs: hold the data budget fixed and vary only the method, so that "the method
merely does supervised post-merge replay fine-tuning" becomes a testable claim rather
than a concession.

  lam_search_global / lam_search_pertask
      Tune the task-arithmetic coefficients on replay loss. The cheapest alternative use
      of the same labels; pre-empts "could you not just have tuned lambda?".
  cocktail_merge
      LM-Cocktail (Xiao et al. 2023) style loss-weighted coefficients. ADAPTATION: the
      original weights candidate models by their loss on ONE target task's few-shot data,
      producing a different model per target. We need a single shared encoder, so we
      weight expert i by its mean (normalised) replay loss across ALL tasks.
  fisher_merge
      Fisher-weighted merging (Matena & Raffel 2022) with a diagonal EMPIRICAL Fisher
      estimated from the replay buffer (per-example squared gradients of the task loss at
      the expert). Note theta = sum_i F_i theta_i / sum_i F_i is identical to
      theta_0 + sum_i F_i tau_i / sum_i F_i, and the latter degrades gracefully to theta_0
      where every F_i is zero -- which happens for ~31% of RoBERTa coordinates (word
      embeddings of tokens absent from a 64-example buffer).
  head_only_scores
      Freeze the merged encoder; refit each task's head on its own buffer. Separates
      "APR repairs the encoder" from "the gains were head/encoder realignment".

Selection protocol: hyperparameters are chosen by replay-buffer loss, NEVER by the eval
split. The eval split is touched once, for reporting.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .data import batches_from_buffer
from .eval import evaluate_task
from .metrics import score_predictions
from .models import (ParamDict, load_encoder_state, get_head_state, load_head_state,
                     is_head_param, pd_clone, pd_axpy_)
from .taskvec import task_arithmetic_merge


# ---------------------------------------------------------------------------
# Replay-buffer objective (selection signal; never touches the eval split)
# ---------------------------------------------------------------------------

@torch.no_grad()
def buffer_loss(model, buffer, collator, batch_size: int, device: str,
                move_model: bool = True) -> float:
    """Mean task loss over the replay buffer at the model's current weights."""
    model.eval()
    if move_model:
        model.to(device)
    tot, n = 0.0, 0
    for batch in batches_from_buffer(buffer, collator, batch_size, device):
        bn = int(batch["labels"].shape[0])
        tot += float(model(**batch).loss) * bn
        n += bn
    if move_model:
        model.to("cpu")
    if device.startswith("cuda") and move_model:
        torch.cuda.empty_cache()
    return tot / max(n, 1)


def replay_losses(ctx, state: ParamDict,
                  buffer_key: str = "probe_buffer") -> Dict[str, float]:
    out = {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        load_encoder_state(info["model"], state)
        out[n] = buffer_loss(info["model"], info[buffer_key], info["collator"],
                             ctx.cfg.data.eval_batch_size, ctx.device,
                             move_model=not ctx.keep_model_on_device)
    return out


@torch.no_grad()
def buffer_metric(model, buffer, spec, collator, batch_size: int, device: str,
                  move_model: bool = True) -> float:
    """The task's own GLUE-convention metric on a replay buffer.

    The classification counterpart of :func:`buffer_loss`, scoring the same
    forward pass by the metric the paper reports instead of by cross-entropy.
    Cross-entropy is a poor RANKING function across models of differing
    confidence: from a maximum-entropy initialization (GLUE's pretrained heads
    sit at CE ~ ln C) any gain in confidence RAISES held-out CE until accuracy
    clears roughly 0.8, so a timid model outranks an accurate one. The metric
    has no such bias, at the cost of sampling noise on a small fold.

    Degenerate cases are real on an 8-example fold and are mapped to 0.0, the
    score of an uninformative predictor: MCC is undefined when predictions are
    constant, and Pearson/Spearman are NaN when either side has zero variance.
    """
    model.eval()
    if move_model:
        model.to(device)
    preds, labels = [], []
    for batch in batches_from_buffer(buffer, collator, batch_size, device):
        y = batch.pop("labels")
        logits = model(**batch).logits
        p = (logits.squeeze(-1) if spec.is_regression
             else logits.argmax(dim=-1))
        preds.append(p.float().cpu().numpy())
        labels.append(y.float().cpu().numpy())
    if move_model:
        model.to("cpu")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    import numpy as np
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if not spec.is_regression:
        preds = preds.astype(int)
        labels = labels.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        try:
            v = float(score_predictions(spec, preds, labels)["primary"])
        except Exception:
            return 0.0
    return 0.0 if (v != v) else v          # NaN -> 0


@torch.no_grad()
def buffer_loss_and_metric(model, buffer, spec, collator, batch_size: int,
                           device: str, move_model: bool = True
                           ) -> Tuple[float, float]:
    """Loss AND metric on a replay buffer from ONE forward pass.

    The metric is the selection criterion and the loss its tie-break, so both
    are needed for every scored cell; computing them separately would double
    the held-out passes for no reason.
    """
    import numpy as np
    model.eval()
    if move_model:
        model.to(device)
    tot, n = 0.0, 0
    preds, labels = [], []
    for batch in batches_from_buffer(buffer, collator, batch_size, device):
        out = model(**batch)
        bn = int(batch["labels"].shape[0])
        tot += float(out.loss) * bn
        n += bn
        p = (out.logits.squeeze(-1) if spec.is_regression
             else out.logits.argmax(dim=-1))
        preds.append(p.float().cpu().numpy())
        labels.append(batch["labels"].float().cpu().numpy())
    if move_model:
        model.to("cpu")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if not spec.is_regression:
        preds = preds.astype(int)
        labels = labels.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        try:
            v = float(score_predictions(spec, preds, labels)["primary"])
        except Exception:
            v = 0.0
    if v != v:
        v = 0.0
    return tot / max(n, 1), v


def replay_losses_and_metrics(ctx, state: ParamDict,
                              buffer_key: str = "probe_buffer"
                              ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(per-task loss, per-task metric) on a replay buffer, one pass per task."""
    L, M = {}, {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        load_encoder_state(info["model"], state)
        L[n], M[n] = buffer_loss_and_metric(
            info["model"], info[buffer_key], info["spec"], info["collator"],
            ctx.cfg.data.eval_batch_size, ctx.device,
            move_model=not ctx.keep_model_on_device)
    return L, M


def replay_metrics(ctx, state: ParamDict,
                   buffer_key: str = "probe_buffer") -> Dict[str, float]:
    """Per-task metric on a replay buffer, the selection counterpart of
    :func:`replay_losses`. Reads no evaluation split."""
    out = {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        load_encoder_state(info["model"], state)
        out[n] = buffer_metric(info["model"], info[buffer_key], info["spec"],
                               info["collator"], ctx.cfg.data.eval_batch_size,
                               ctx.device,
                               move_model=not ctx.keep_model_on_device)
    return out


def make_replay_objective(ctx, buffer_key: str = "probe_buffer"
                          ) -> Tuple[Callable[[ParamDict], float], Dict[str, float]]:
    """Scale-free replay objective: mean over tasks of L_t(theta) / L_t(theta_merge).

    Raw losses are not commensurable across tasks (cross-entropy vs. MSE for STS-B), so
    each is normalised by its value at the fixed task-arithmetic merge point. Lower is
    better; the merge point scores exactly 1.0."""
    ref = replay_losses(ctx, ctx.merged0, buffer_key=buffer_key)

    def objective(state: ParamDict) -> float:
        L = replay_losses(ctx, state, buffer_key=buffer_key)
        return sum(L[n] / max(ref[n], 1e-8) for n in L) / len(L)

    return objective, ref


# ---------------------------------------------------------------------------
# 1. Replay-tuned task-arithmetic coefficients
# ---------------------------------------------------------------------------

def lam_search_global(ctx, lams: List[float], objective) -> Tuple[ParamDict, dict]:
    """Uniform lambda chosen by replay loss."""
    best = None
    trace = {}
    for lam in lams:
        state = task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors,
                                      {n: lam for n in ctx.task_names})
        v = objective(state)
        trace[lam] = v
        if best is None or v < best[0]:
            best = (v, lam, state)
    return best[2], {"lam": best[1], "replay_obj": best[0], "trace": trace}


def lam_search_pertask(ctx, lams: List[float], objective, passes: int = 2,
                       init_lam: Optional[float] = None,
                       logger=None) -> Tuple[ParamDict, dict]:
    """Per-task lambda_i by coordinate descent on the replay objective."""
    names = ctx.task_names
    cur = {n: (init_lam if init_lam is not None else ctx.lambdas[n]) for n in names}

    def build(lmap):
        return task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors, lmap)

    best_v = objective(build(cur))
    for p in range(passes):
        for n in names:
            keep = cur[n]
            for lam in lams:
                if lam == keep:
                    continue
                cur[n] = lam
                v = objective(build(cur))
                if v < best_v:
                    best_v, keep = v, lam
            cur[n] = keep
        if logger:
            logger(f"  [lam_pertask] pass {p}: obj={best_v:.4f} "
                   f"lams={ {k: round(v, 3) for k, v in cur.items()} }")
    return build(cur), {"lams": cur, "replay_obj": best_v}


# ---------------------------------------------------------------------------
# 2. LM-Cocktail style loss-weighted merging (single-model adaptation)
# ---------------------------------------------------------------------------

def cocktail_merge(ctx, lams: List[float], objective, temperature: float = 1.0,
                   logger=None) -> Tuple[ParamDict, dict]:
    """w_i = softmax(-L_i / T), L_i = mean normalised replay loss of expert i across
    all tasks (each scored with its own head). theta = theta_0 + lam * sum_i w_i tau_i,
    with lam selected on replay."""
    names = ctx.task_names
    # normalise each task's loss by the merge point's, as in make_replay_objective
    ref_losses = replay_losses(ctx, ctx.merged0)

    expert_loss = {}
    for i in names:
        theta_i = ctx.per_task[i]["expert_encoder"]
        L = replay_losses(ctx, theta_i)
        expert_loss[i] = sum(L[t] / max(ref_losses[t], 1e-8) for t in names) / len(names)
    if logger:
        logger(f"  [cocktail] expert mean-normalised replay losses: "
               f"{ {k: round(v, 3) for k, v in expert_loss.items()} }")

    ls = torch.tensor([expert_loss[n] for n in names])
    w = torch.softmax(-ls / temperature, dim=0)
    wmap = {n: float(w[i]) for i, n in enumerate(names)}

    best = None
    for lam in lams:
        state = task_arithmetic_merge(ctx.base_encoder, ctx.task_vectors,
                                      {n: lam * wmap[n] for n in names})
        v = objective(state)
        if best is None or v < best[0]:
            best = (v, lam, state)
    return best[2], {"weights": wmap, "temperature": temperature,
                     "lam": best[1], "replay_obj": best[0],
                     "expert_loss": expert_loss}


# ---------------------------------------------------------------------------
# 3. Fisher merging (diagonal empirical Fisher from the replay buffer)
# ---------------------------------------------------------------------------

def _diag_fisher(model, buffer, collator, device: str, enc_names: List[str],
                 move_model: bool = True) -> ParamDict:
    """F[r] = (1/n) sum_x (d loss(x) / d theta_r)^2, per-example (batch size 1)."""
    model.eval()
    if move_model:
        model.to(device)
    F = OrderedDict((n, torch.zeros_like(p, device="cpu"))
                    for n, p in model.named_parameters() if n in set(enc_names))
    count = 0
    for batch in batches_from_buffer(buffer, collator, 1, device):
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        for n, p in model.named_parameters():
            if n in F and p.grad is not None:
                F[n].add_((p.grad.detach() ** 2).cpu())
        count += 1
    model.zero_grad(set_to_none=True)
    if move_model:
        model.to("cpu")
    if device.startswith("cuda") and move_model:
        torch.cuda.empty_cache()
    for n in F:
        F[n].div_(max(count, 1))
    return F


def _expected_fisher(model, buffer, collator, device: str, enc_names: List[str],
                     is_regression: bool = False, exact_max_classes: int = 64,
                     n_samples: int = 16, seed: int = 0,
                     move_model: bool = True) -> ParamDict:
    """Diagonal EXPECTED Fisher of \\citet{matena2022fisher}, eq. (3):

        F[r] = (1/N) sum_i E_{y ~ p_theta(y|x_i)} ( d log p_theta(y|x_i) / d theta_r )^2

    The expectation is over labels drawn from the MODEL's own predictive
    distribution, so this consumes the buffer's inputs and never its labels --
    unlike ``_diag_fisher`` above, which squares the gradient of the loss at the
    ground-truth label (the "empirical Fisher", a different estimator).

    Following the paper, the expectation is computed exactly when the number of
    classes is small and estimated by sampling from p_theta otherwise. For a
    regression head the unit-variance Gaussian likelihood gives F = (d mu)^2.
    """
    model.eval()
    if move_model:
        model.to(device)
    keep = set(enc_names)
    named = [(n, p) for n, p in model.named_parameters()
             if n in keep and p.requires_grad]
    plist = [p for _, p in named]
    F = OrderedDict((n, torch.zeros_like(p, device="cpu")) for n, p in named)
    gen = torch.Generator(); gen.manual_seed(seed)
    count = 0

    for batch in batches_from_buffer(buffer, collator, 1, device):
        inputs = {k: v for k, v in batch.items() if k != "labels"}   # LABEL-FREE
        logits = model(**inputs).logits.squeeze(0)
        if is_regression:
            targets, weights = [logits.squeeze()], [1.0]
        else:
            logp = torch.log_softmax(logits, dim=-1)
            probs = logp.exp().detach()
            n_cls = logp.shape[-1]
            if n_cls <= exact_max_classes:
                targets = [logp[c] for c in range(n_cls)]
                weights = [float(probs[c]) for c in range(n_cls)]
            else:
                draw = torch.multinomial(probs.float().cpu(), n_samples,
                                         replacement=True, generator=gen)
                targets = [logp[int(c)] for c in draw]
                weights = [1.0 / n_samples] * len(draw)
        for j, (t, w) in enumerate(zip(targets, weights)):
            if w == 0.0:
                continue
            grads = torch.autograd.grad(t, plist, retain_graph=(j < len(targets) - 1),
                                        allow_unused=True)
            for (n, _), g in zip(named, grads):
                if g is not None:
                    F[n].add_((g.detach() ** 2).cpu(), alpha=w)
        count += 1

    model.zero_grad(set_to_none=True)
    if move_model:
        model.to("cpu")
    if device.startswith("cuda") and move_model:
        torch.cuda.empty_cache()
    for n in F:
        F[n].div_(max(count, 1))
    return F


def fisher_lambda_candidates(task_names: List[str], n_points: int = 50,
                             seed: int = 0):
    """The simplex search of \\citet{matena2022fisher}: per-model coefficients
    with lambda_i >= 0 and sum_i lambda_i = 1. Point 0 is the uniform weighting;
    the rest are Dirichlet(1) draws, i.e. uniform over the simplex."""
    T = len(task_names)
    out = [("uniform", {n: 1.0 / T for n in task_names})]
    rng = torch.Generator(); rng.manual_seed(seed)
    for i in range(max(0, n_points - 1)):
        w = torch.distributions.Dirichlet(torch.ones(T)).sample()
        out.append((f"dir{i}", {n: float(w[j]) for j, n in enumerate(task_names)}))
    return out


def fisher_merge_matena(ctx, buffers, objective, n_points: int = 50,
                        eps: float = 1e-12, seed: int = 0, logger=None):
    """Fisher merging as specified by \\citet{matena2022fisher}:

        theta* = sum_i lambda_i F_i theta_i / sum_i lambda_i F_i

    with the diagonal EXPECTED Fisher of each expert (label-free; computed on
    that expert's own inputs) and per-model coefficients on the simplex. Where
    every F_i vanishes the ratio is undefined and we default to theta_0, the
    analogue of the paper's "privileged target model".

    Note the structural difference from ``fisher_merge``: there a single global
    lambda scales one Fisher-weighted average, which can push the result outside
    the experts' hull; here the lambda_i sit inside both sums, so the result is
    a convex combination of the experts coordinate by coordinate.

    Returns (best_state, info, all_candidates) where all_candidates is a list of
    (name, state, meta) for the caller to score under its own protocol.
    """
    keys = list(ctx.base_encoder.keys())
    fishers, tau_fishers = {}, {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        load_encoder_state(info["model"], info["expert_encoder"])   # Fisher AT the expert
        F = _expected_fisher(info["model"], buffers[n], info["collator"],
                             ctx.device, keys,
                             is_regression=info["spec"].is_regression,
                             seed=seed, move_model=not ctx.keep_model_on_device)
        tau = ctx.task_vectors[n]
        fishers[n] = F
        tau_fishers[n] = OrderedDict((k, F[k] * tau[k]) for k in keys)
        if logger:
            nz = sum(float((F[k] > 0).float().sum()) for k in keys)
            tot = sum(F[k].numel() for k in keys)
            logger(f"  [fisher-expected] {n}: nonzero coords {nz/tot:.3f}")

    cands = []
    for label, lam in fisher_lambda_candidates(ctx.task_names, n_points, seed):
        num = OrderedDict((k, torch.zeros_like(ctx.base_encoder[k])) for k in keys)
        den = OrderedDict((k, torch.zeros_like(ctx.base_encoder[k])) for k in keys)
        for n in ctx.task_names:
            w = lam[n]
            if w == 0.0:
                continue
            for k in keys:
                num[k].add_(tau_fishers[n][k], alpha=w)
                den[k].add_(fishers[n][k], alpha=w)
        state = pd_clone(ctx.base_encoder)
        for k in keys:
            state[k].add_(num[k] / (den[k] + eps))
        cands.append((f"FISHER@{label}", state, {"lambdas": lam}))
        del num, den
    return cands


def fisher_merge(ctx, lams: List[float], objective, eps: float = 1e-12,
                 logger=None) -> Tuple[ParamDict, dict]:
    """theta = theta_0 + lam * sum_i F_i tau_i / (sum_i F_i + eps).

    Equivalent to the classic sum_i F_i theta_i / sum_i F_i (since sum_i F_i theta_0 /
    sum_i F_i = theta_0), but well-defined where every F_i vanishes -- there it returns
    theta_0 rather than 0. lam=1 is pure Fisher merging; lam is swept on replay."""
    keys = list(ctx.base_encoder.keys())
    num = OrderedDict((k, torch.zeros_like(ctx.base_encoder[k])) for k in keys)
    den = OrderedDict((k, torch.zeros_like(ctx.base_encoder[k])) for k in keys)

    for n in ctx.task_names:
        info = ctx.per_task[n]
        load_encoder_state(info["model"], info["expert_encoder"])  # Fisher AT the expert
        F = _diag_fisher(info["model"], info["probe_buffer"], info["collator"],
                         ctx.device, keys, move_model=not ctx.keep_model_on_device)
        tau = ctx.task_vectors[n]
        for k in keys:
            num[k].add_(F[k] * tau[k])
            den[k].add_(F[k])
        if logger:
            nz = sum(float((F[k] > 0).float().mean()) * F[k].numel() for k in keys)
            tot = sum(F[k].numel() for k in keys)
            logger(f"  [fisher] {n}: nonzero-Fisher coords {nz/tot:.3f}")
        del F

    ratio = OrderedDict((k, num[k] / (den[k] + eps)) for k in keys)
    zero_frac = (sum(float((den[k] == 0).sum()) for k in keys) /
                 sum(den[k].numel() for k in keys))

    best = None
    for lam in lams:
        state = pd_clone(ctx.base_encoder)
        pd_axpy_(state, lam, ratio)
        v = objective(state)
        if best is None or v < best[0]:
            best = (v, lam, state)
    return best[2], {"lam": best[1], "replay_obj": best[0],
                     "zero_fisher_frac": zero_frac}


# ---------------------------------------------------------------------------
# 4. Head-only recalibration on the merged encoder
# ---------------------------------------------------------------------------

def head_only_scores(ctx, state: ParamDict, lr: float = 1e-3, epochs: int = 20,
                     batch_size: int = 16, logger=None) -> Tuple[Dict[str, float], dict]:
    """Freeze the merged encoder; refit each task's head on its own replay buffer.

    The head is trained in place and then RESTORED, so later evaluations are unaffected.
    Returns eval-split scores (the encoder is never updated, so this isolates how much of
    APR's gain is really encoder repair)."""
    scores, info_out = {}, {}
    for n in ctx.task_names:
        info = ctx.per_task[n]
        model = info["model"]
        orig_head = get_head_state(model)
        orig_rg = {pn: p.requires_grad for pn, p in model.named_parameters()}
        load_encoder_state(model, state)
        model.to(ctx.device)
        model.eval()  # frozen encoder; eval mode keeps dropout out of the head fit

        head_params = []
        for pn, p in model.named_parameters():
            head = is_head_param(model, pn)
            p.requires_grad_(head)
            if head:
                head_params.append(p)
        if not head_params:
            logger and logger(f"  [head_only] {n}: no trainable head params, skipping")
            for pn, p in model.named_parameters():
                p.requires_grad_(orig_rg[pn])
            scores[n] = ctx.eval_encoder(state, names=[n])[n]
            continue

        opt = torch.optim.Adam(head_params, lr=lr)
        last = None
        for _ in range(epochs):
            for batch in batches_from_buffer(info["probe_buffer"], info["collator"],
                                             batch_size, ctx.device):
                opt.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                opt.step()
                last = float(loss)
        model.zero_grad(set_to_none=True)

        scores[n] = evaluate_task(model, info["eval_ds"], info["spec"], info["collator"],
                                  ctx.cfg.data.eval_batch_size, ctx.device,
                                  num_workers=ctx.cfg.data.eval_num_workers)["primary"]
        info_out[n] = {"final_head_loss": last, "n_head_params": len(head_params)}
        if logger:
            logger(f"  [head_only] {n}: head loss {last:.4f} score {scores[n]:.4f}")

        # restore original head + the exact original requires_grad flags, so nothing
        # downstream (APR refinement, later evals) is contaminated
        load_head_state(model, orig_head)
        for pn, p in model.named_parameters():
            p.requires_grad_(orig_rg[pn])
        model.to("cpu")
        if ctx.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return scores, info_out
