"""Decoder-only LLM track (modality: "causal_lm") -- MergeBench experts.

Experts are the published MergeBench full-SFT domain models (HF org
``MergeBench``, e.g. MergeBench/Llama-3.2-3B_math) sharing theta_0 with the
corresponding base model. The mergeable parameters are everything under the
model's ``base_model_prefix`` ("model" for Llama/Gemma-2); ``lm_head`` is tied
to the input embeddings on the 2-3B models, so it is merged implicitly.

POC domains (judge-free automatic metrics only):
  math        replay = GSM8K train (SFT format), eval = GSM8K test exact-match
  coding      replay = MBPP train,               eval = MBPP test pass@1
              (sandboxed subprocess execution with a timeout)
  instruction replay = an SFT instruction set (alpaca-cleaned),
              eval = IFEval-lite: the subset of google/IFEval prompts whose
              constraints our checkers implement, strict prompt-level accuracy

Prompting: ``prompt_style`` = "chat" applies the tokenizer's chat template
(matching how instruct-based experts were SFT'd); "plain" uses a bare
instruction+response format for base-model experts. VALIDATE with [eval0]:
if an expert does not clearly beat the base model on its own domain, the
prompt format does not match its SFT recipe -- fix that before trusting runs.

Eval is generation-based and therefore slow: subsample eval sets during sweeps
(``data.max_eval``) and report full-set numbers only for final configurations.
"""

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Task specs
# ---------------------------------------------------------------------------

@dataclass
class CausalTaskSpec:
    """Duck-typed against tasks.TaskSpec for the shared pipeline (name, metric,
    is_regression, num_labels) plus generation-eval fields."""
    name: str
    metric: str                      # label for reporting; scoring is metric_fn
    build_replay: Callable           # (tokenizer, max_length, cache_dir) -> list[dict]
    build_eval: Callable             # (cache_dir, max_eval) -> list[dict]
    make_prompt: Callable            # (row, style, tokenizer) -> str
    score_generations: Callable      # (rows, texts) -> dict with "primary"
    max_new_tokens: int = 256
    eval_kind: str = "generation"
    is_regression: bool = False
    num_labels: int = 2  # unused; kept for duck-typing
    prompt_style: str = "plain"      # overridden from config at build time
    tokenizer: Optional[object] = None  # injected at build time


def _wrap_prompt(instruction: str, style: str, tokenizer) -> str:
    if style == "chat" and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True)
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _sft_example(tokenizer, prompt: str, response: str, max_length: int) -> Dict:
    """Tokenized SFT pair with the prompt masked out of the loss."""
    p = tokenizer(prompt, add_special_tokens=True, truncation=True,
                  max_length=max_length)["input_ids"]
    r = tokenizer(response + tokenizer.eos_token, add_special_tokens=False,
                  truncation=True, max_length=max_length)["input_ids"]
    ids = (p + r)[:max_length]
    labels = ([-100] * len(p) + r)[:max_length]
    return {"input_ids": ids, "labels": labels}


# ---------------------------------------------------------------------------
# math: GSM8K
# ---------------------------------------------------------------------------

GSM8K_INSTR = ("Solve the following math problem step by step. "
               "End your answer with '#### <final numeric answer>'.\n\n{q}")
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Number after '####' if present, else the LAST number in the text."""
    m = re.search(r"####\s*([^\n]+)", text)
    cand = m.group(1) if m else text
    nums = _NUM_RE.findall(cand)
    if not nums and not m:
        return None
    if not nums:
        nums = _NUM_RE.findall(text)
        if not nums:
            return None
    return nums[-1].replace(",", "").replace("$", "").rstrip(".")


def _mergebench_val_replay(hf_name: str, prompt_field: str, response_field: str):
    """Replay drawn from MergeBench's released per-domain validation sets
    (1k samples from each expert's actual SFT training distribution). This is the
    same pool MergeBench gives its data-dependent baselines (Fisher/RegMean/L&S,
    1,000 examples each), so an n=64 replay buffer is protocol-consistent -- and it
    fixes the earlier objective mismatch where the instruction gradient was computed
    on alpaca prose while the metric scored IFEval constraint compliance."""
    def build(tokenizer, max_length, cache_dir, style):
        ds = load_dataset(hf_name, split="train", cache_dir=cache_dir)
        def make(row):
            return _sft_example(tokenizer,
                                _wrap_prompt(row[prompt_field], style, tokenizer),
                                row[response_field], max_length)
        return ds, make
    return build


def _gsm8k_eval(cache_dir, max_eval):
    ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    if max_eval:
        ds = ds.select(range(min(max_eval, len(ds))))
    return list(ds)


def _gsm8k_score(rows, texts) -> Dict[str, float]:
    hits = 0
    for row, text in zip(rows, texts):
        gold = extract_gsm8k_answer(row["answer"])
        pred = extract_gsm8k_answer(text)
        try:
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-4
        except (TypeError, ValueError):
            ok = pred == gold
        hits += int(ok)
    acc = hits / max(len(rows), 1)
    return {"exact_match": acc, "primary": acc}


# ---------------------------------------------------------------------------
# coding: MBPP (sandboxed pass@1)
# ---------------------------------------------------------------------------

MBPP_INSTR = ("Write a Python function for the following task. Return only code.\n\n"
              "{text}\n\nYour solution must pass this test:\n{test}\n")
_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _FENCE_RE.search(text)
    code = m.group(1) if m else text
    # drop anything after a new prose block if the model kept talking
    return code.strip()


def run_python_tests(code: str, tests: List[str], timeout: float = 10.0) -> bool:
    """Execute candidate code + asserts in a subprocess. NOTE: this executes
    model-generated code with user permissions -- standard for MBPP/HumanEval
    scoring, but do not point it at untrusted checkpoints."""
    prog = code + "\n\n" + "\n".join(tests) + "\nprint('__PASS__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        out = subprocess.run([sys.executable, "-I", path], capture_output=True,
                             text=True, timeout=timeout)
        return "__PASS__" in out.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        os.unlink(path)


def _mbpp_eval(cache_dir, max_eval):
    # MBPP+ (EvalPlus): same problems as MBPP but ~35x more tests per problem, and
    # the split MergeBench's coding numbers use (via bigcode-eval). The stronger
    # suites separate base from expert far better than the 3 original asserts
    # (plain-MBPP eval0 gave expert-base = 0.05, too small to carry retention).
    ds = load_dataset("evalplus/mbppplus", split="test", cache_dir=cache_dir)
    if max_eval:
        ds = ds.select(range(min(max_eval, len(ds))))
    return list(ds)


def _mbpp_score(rows, texts) -> Dict[str, float]:
    # row["test"] is a self-contained EvalPlus script (defines assertion() + all
    # input/expected pairs and calls assertion(func(*inp), exp, 0)); append it to
    # the candidate code and run in the sandbox. Longer timeout: plus-suites run
    # hundreds of assertions.
    #
    # Scored in a thread pool: each item is an independent subprocess, so the
    # work is I/O-wait on subprocess.run and threads suffice (no GIL contention,
    # no fork of the CUDA-initialised parent). Serial scoring of 300 items took
    # 88-142 s with the GPU at 0% utilisation, which is the whole coding eval
    # once generation is fast. Order-independent: results are gathered by index,
    # and each subprocess is isolated (python -I on its own temp file), so the
    # score is identical to the serial version.
    # sched_getaffinity, not cpu_count: under Slurm the latter reports the whole
    # node (94 cores) rather than the job's --cpus-per-task allocation.
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:                      # not Linux
        n_cpu = os.cpu_count() or 4
    workers = max(1, min(n_cpu, max(len(rows), 1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        passed = list(pool.map(
            lambda rt: run_python_tests(extract_code(rt[1]), [rt[0]["test"]],
                                        timeout=30.0),
            zip(rows, texts)))
    acc = sum(int(p) for p in passed) / max(len(rows), 1)
    return {"pass@1": acc, "primary": acc}


# ---------------------------------------------------------------------------
# instruction: IFEval (official checkers, all 541 prompts)
# ---------------------------------------------------------------------------

def _ifeval_eval(cache_dir, max_eval):
    # official checker suite (apr.ifeval) covers all 25 instruction types, so no
    # coverage filter: the previous 12 hand-written checkers restricted scoring to
    # a biased 223/541 subset (only the constraint families that were easy to
    # regex), which both skewed the construct and doubled the noise.
    rows = list(load_dataset("google/IFEval", split="train", cache_dir=cache_dir))
    if max_eval:
        rows = rows[:max_eval]
    return rows


def _ifeval_score(rows, texts) -> Dict[str, float]:
    """Prompt-level STRICT accuracy with the official IFEval checkers."""
    from .ifeval import strict_follows_all
    hits = sum(int(strict_follows_all(row["prompt"], row["instruction_id_list"],
                                      row.get("kwargs"), text))
               for row, text in zip(rows, texts))
    acc = hits / max(len(rows), 1)
    return {"prompt_level_strict": acc, "primary": acc}


# ---------------------------------------------------------------------------
# multilingual: M-MMLU + M-ARC (Okapi), multiple-choice log-likelihood.
# Scored like lm-eval's MC tasks: one forward pass per prompt, compare the
# log-probs of the option letters at the final position. No generation at all,
# so this eval has NONE of the decode-chaos sensitivity of the other domains
# (a near-tie argmax flip cannot cascade), and it is ~100x cheaper per example.
# ---------------------------------------------------------------------------

MULTILINGUAL_LANGS = ["de", "es", "fr", "ru", "zh", "hi"]
_MC_BENCHES = [("alexandrainst/m_mmlu", "m_mmlu"), ("alexandrainst/m_arc", "m_arc")]


def _multilingual_eval(cache_dir, max_eval):
    """Round-robin over (bench, language) streams so the subset is balanced;
    each stream is shuffled with a fixed seed first (M-MMLU's test split is
    subject-sorted, so taking a prefix unshuffled would bias toward one subject)."""
    import random as _random
    streams = []
    for hf_name, bench in _MC_BENCHES:
        for lang in MULTILINGUAL_LANGS:
            try:
                ds = load_dataset(hf_name, lang, split="test", cache_dir=cache_dir)
            except Exception as e:  # missing language config: skip, keep balance
                print(f"[multilingual] skip {bench}/{lang}: {type(e).__name__}", flush=True)
                continue
            rows = []
            for r in ds:
                options = []
                for letter in ("a", "b", "c", "d", "e"):
                    v = r.get(f"option_{letter}")
                    if v is None or str(v).strip() in ("", "None"):
                        continue
                    options.append((letter.upper(), str(v)))
                if len(options) < 2 or r["answer"] not in [o[0] for o in options]:
                    continue
                rows.append({"question": r["instruction"], "options": options,
                             "answer": r["answer"], "bench": bench, "lang": lang})
            _random.Random(0).shuffle(rows)
            streams.append(rows)
    out, i = [], 0
    limit = max_eval or sum(len(s) for s in streams)
    while len(out) < limit and any(streams):
        s = streams[i % len(streams)]
        if s:
            out.append(s.pop(0))
        i += 1
        if i > 10_000_000:
            break
    return out


def _mc_prompt(row) -> str:
    # standard MMLU-style plain format (matches lm-eval; no SFT wrapper, so the
    # comparison base-vs-expert is the community-standard one)
    lines = [row["question"]]
    lines += [f"{letter}. {text}" for letter, text in row["options"]]
    lines.append("Answer:")
    return "\n".join(lines)


@torch.no_grad()
def evaluate_mc(model, eval_rows: List[Dict], spec: CausalTaskSpec,
                batch_size: int, device: str) -> Dict[str, float]:
    """Multiple-choice accuracy by option-letter log-likelihood at the last
    position. Honors spec.gen_dtype (bf16 eval) with the same cast-and-restore
    discipline as evaluate_generation."""
    tok = spec.tokenizer
    model.eval()
    model.to(device)
    gen_dtype = getattr(spec, "gen_dtype", None)
    orig_dtype = next(model.parameters()).dtype
    cast = gen_dtype is not None and gen_dtype != orig_dtype
    buf_backup = ({n: b.detach().clone() for n, b in model.named_buffers()
                   if b.is_floating_point()} if cast else {})
    if cast:
        model.to(gen_dtype)
    letter_ids = {}
    for letter in ("A", "B", "C", "D", "E"):
        ids = tok(" " + letter, add_special_tokens=False)["input_ids"]
        letter_ids[letter] = ids[-1]
    hits = 0
    try:
        for s in range(0, len(eval_rows), batch_size):
            rows = eval_rows[s: s + batch_size]
            prompts = [_mc_prompt(r) for r in rows]
            tok.padding_side = "left"  # last position aligned across the batch
            batch = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024, add_special_tokens=True).to(device)
            logits = model(**batch).logits[:, -1, :].float()
            for i, r in enumerate(rows):
                letters = [o[0] for o in r["options"]]
                scores = torch.stack([logits[i, letter_ids[L]] for L in letters])
                hits += int(letters[int(scores.argmax())] == r["answer"])
    finally:
        if cast:
            model.to(orig_dtype)
            for n, b in model.named_buffers():
                if n in buf_backup:
                    b.copy_(buf_backup[n])
    acc = hits / max(len(eval_rows), 1)
    return {"mc_acc": acc, "primary": acc}


def _mc_score_stub(rows, texts):  # pragma: no cover - mc path never generates
    raise RuntimeError("multilingual task is scored by evaluate_mc, not generation")


# ---------------------------------------------------------------------------
# Registry + prompt builders per task
# ---------------------------------------------------------------------------

def _mk_prompt_math(row, style, tok):
    return _wrap_prompt(GSM8K_INSTR.format(q=row["question"]), style, tok)

def _mk_prompt_code(row, style, tok):
    # mbppplus rows: task text in "prompt", original asserts in "test_list"
    return _wrap_prompt(MBPP_INSTR.format(text=row["prompt"],
                                          test=row["test_list"][0]), style, tok)

def _mk_prompt_instr(row, style, tok):
    return _wrap_prompt(row["prompt"], style, tok)


CAUSAL_TASKS: Dict[str, dict] = {
    "math": dict(metric="exact_match", eval=_gsm8k_eval,
                 replay=_mergebench_val_replay("MergeBench/math_val",
                                               "query", "response"),
                 prompt=_mk_prompt_math, score=_gsm8k_score, max_new_tokens=400),
    "coding": dict(metric="pass@1", eval=_mbpp_eval,
                   replay=_mergebench_val_replay("MergeBench/coding_val",
                                                 "instruction", "response"),
                   prompt=_mk_prompt_code, score=_mbpp_score, max_new_tokens=400),
    "instruction": dict(metric="ifeval_strict", eval=_ifeval_eval,
                        replay=_mergebench_val_replay("MergeBench/instruction_val",
                                                      "instruction", "output"),
                        prompt=_mk_prompt_instr,
                        # 1024 (was 512): 34/541 IFEval prompts carry "at least N
                        # words" constraints (median 300, p90 800 words); at 512
                        # tokens 15 of those were mathematically unsatisfiable,
                        # auto-failing any model that complies. 1024 leaves ~6.
                        score=_ifeval_score, max_new_tokens=1024),
    "multilingual": dict(metric="mc_acc", eval=_multilingual_eval,
                         replay=_mergebench_val_replay("MergeBench/multilingual_val",
                                                       "inputs", "targets"),
                         prompt=_mk_prompt_instr,  # unused: mc path never generates
                         score=_mc_score_stub, max_new_tokens=8, eval_kind="mc"),
}

# MergeBench checkpoint naming: MergeBench/<base>_<domain>
MERGEBENCH_EXPERT = "MergeBench/{base}_{domain}"


def get_causal_task(name: str) -> CausalTaskSpec:
    if name not in CAUSAL_TASKS:
        raise KeyError(f"Unknown causal task '{name}'. Known: {sorted(CAUSAL_TASKS)}")
    d = CAUSAL_TASKS[name]
    return CausalTaskSpec(name=name, metric=d["metric"], build_replay=d["replay"],
                          build_eval=d["eval"], make_prompt=d["prompt"],
                          score_generations=d["score"],
                          max_new_tokens=d["max_new_tokens"],
                          eval_kind=d.get("eval_kind", "generation"))


# ---------------------------------------------------------------------------
# Replay sampling + collator (right-padded SFT batches)
# ---------------------------------------------------------------------------

def sample_causal_replay(train_ds, make_fn, n: int, seed: int) -> List[Dict]:
    import random
    rng = random.Random(seed)
    idx = rng.sample(range(len(train_ds)), min(n, len(train_ds)))
    return [make_fn(train_ds[i]) for i in idx]


def make_causal_collator(tokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(features: List[Dict]):
        L = max(len(f["input_ids"]) for f in features)
        ids = torch.full((len(features), L), pad_id, dtype=torch.long)
        lab = torch.full((len(features), L), -100, dtype=torch.long)
        att = torch.zeros((len(features), L), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}
    return collate


# ---------------------------------------------------------------------------
# Generation-based evaluation (dispatched from eval.evaluate_task)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_generation(model, eval_rows: List[Dict], spec: CausalTaskSpec,
                        batch_size: int, device: str) -> Dict[str, float]:
    tok = spec.tokenizer
    model.eval()
    model.to(device)
    # Optional: run generation in a lower precision (e.g. bfloat16) to speed up the
    # autoregressive decode, then restore the original storage dtype. This is safe
    # for the weight-space method because the merge / task vectors / gate / refine
    # gradients are all computed on the fp32 master ParamDict (see MergeContext),
    # which is never touched here; the live model is scratch that has its fp32
    # weights re-copied in before every eval and gradient step. We also snapshot and
    # restore floating buffers (rotary inv_freq etc.) so nothing leaks across evals.
    gen_dtype = getattr(spec, "gen_dtype", None)
    orig_dtype = next(model.parameters()).dtype
    cast = gen_dtype is not None and gen_dtype != orig_dtype
    buf_backup = ({n: b.detach().clone() for n, b in model.named_buffers()
                   if b.is_floating_point()} if cast else {})
    if cast:
        model.to(gen_dtype)
    texts = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        for s in range(0, len(eval_rows), batch_size):
            rows = eval_rows[s: s + batch_size]
            prompts = [spec.make_prompt(r, spec.prompt_style, tok) for r in rows]
            tok.padding_side = "left"  # decoder-only generation needs left padding
            batch = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024, add_special_tokens=True).to(device)
            out = model.generate(**batch, max_new_tokens=spec.max_new_tokens,
                                 do_sample=False, pad_token_id=pad_id)
            gen = out[:, batch["input_ids"].shape[1]:]
            texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
    finally:
        if cast:
            model.to(orig_dtype)  # params reloaded from fp32 master before next use
            for n, b in model.named_buffers():
                if n in buf_backup:
                    b.copy_(buf_backup[n])
    return spec.score_generations(eval_rows, texts)
