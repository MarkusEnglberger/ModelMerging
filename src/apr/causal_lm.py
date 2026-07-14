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


def _gsm8k_replay(tokenizer, max_length, cache_dir, style):
    ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    def make(row):
        return _sft_example(tokenizer,
                            _wrap_prompt(GSM8K_INSTR.format(q=row["question"]),
                                         style, tokenizer),
                            row["answer"], max_length)
    return ds, make


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


def _mbpp_replay(tokenizer, max_length, cache_dir, style):
    ds = load_dataset("mbpp", split="train", cache_dir=cache_dir)
    def make(row):
        prompt = _wrap_prompt(MBPP_INSTR.format(text=row["text"],
                                                test=row["test_list"][0]),
                              style, tokenizer)
        return _sft_example(tokenizer, prompt, row["code"], max_length)
    return ds, make


def _mbpp_eval(cache_dir, max_eval):
    ds = load_dataset("mbpp", split="test", cache_dir=cache_dir)
    if max_eval:
        ds = ds.select(range(min(max_eval, len(ds))))
    return list(ds)


def _mbpp_score(rows, texts) -> Dict[str, float]:
    hits = sum(int(run_python_tests(extract_code(t), r["test_list"]))
               for r, t in zip(rows, texts))
    acc = hits / max(len(rows), 1)
    return {"pass@1": acc, "primary": acc}


# ---------------------------------------------------------------------------
# instruction: IFEval-lite (verifiable-constraint subset) + alpaca replay
# ---------------------------------------------------------------------------

def _count_words(t): return len(re.findall(r"\b\w+\b", t))
def _count_sentences(t): return max(1, len(re.findall(r"[.!?]+(?:\s|$)", t)))

def _rel(n, relation, thr):
    return n < thr if relation == "less than" else n >= thr

# checker: (response, kwargs) -> bool. Subset of google/IFEval instruction ids.
IFEVAL_CHECKERS: Dict[str, Callable] = {
    "keywords:existence": lambda t, k: all(
        re.search(rf"\b{re.escape(w)}\b", t, re.I) for w in k["keywords"]),
    "keywords:forbidden_words": lambda t, k: not any(
        re.search(rf"\b{re.escape(w)}\b", t, re.I) for w in k["forbidden_words"]),
    "keywords:frequency": lambda t, k: _rel(
        len(re.findall(rf"\b{re.escape(k['keyword'])}\b", t, re.I)),
        k["relation"], k["frequency"]),
    "length_constraints:number_words": lambda t, k: _rel(
        _count_words(t), k["relation"], k["num_words"]),
    "length_constraints:number_sentences": lambda t, k: _rel(
        _count_sentences(t), k["relation"], k["num_sentences"]),
    "change_case:english_lowercase": lambda t, k: t == t.lower(),
    "change_case:english_capital": lambda t, k: t == t.upper(),
    "startend:quotation": lambda t, k: t.strip().startswith('"') and t.strip().endswith('"'),
    "startend:end_checker": lambda t, k: t.strip().endswith(k["end_phrase"].strip()),
    "detectable_content:number_placeholders": lambda t, k: len(
        re.findall(r"\[.*?\]", t)) >= k["num_placeholders"],
    "detectable_format:number_bullet_lists": lambda t, k: len(
        re.findall(r"^\s*[*-]\s", t, re.M)) == k["num_bullets"],
    "detectable_format:title": lambda t, k: bool(re.search(r"<<[^\n<>]+>>", t)),
}


def _ifeval_eval(cache_dir, max_eval):
    ds = load_dataset("google/IFEval", split="train", cache_dir=cache_dir)
    rows = [r for r in ds
            if all(i in IFEVAL_CHECKERS for i in r["instruction_id_list"])]
    covered = len(rows) / max(len(ds), 1)
    print(f"[ifeval-lite] {len(rows)}/{len(ds)} prompts covered "
          f"({covered:.0%}) by {len(IFEVAL_CHECKERS)} checkers", flush=True)
    if max_eval:
        rows = rows[:max_eval]
    return rows


def _ifeval_score(rows, texts) -> Dict[str, float]:
    """Strict prompt-level accuracy: every constraint of the prompt satisfied."""
    hits = 0
    for row, text in zip(rows, texts):
        kwargs_list = row.get("kwargs") or [{}] * len(row["instruction_id_list"])
        ok = True
        for inst_id, kw in zip(row["instruction_id_list"], kwargs_list):
            kw = {k: v for k, v in (kw or {}).items() if v is not None}
            try:
                if not IFEVAL_CHECKERS[inst_id](text, kw):
                    ok = False
                    break
            except (KeyError, TypeError):
                ok = False
                break
        hits += int(ok)
    acc = hits / max(len(rows), 1)
    return {"prompt_level_strict": acc, "primary": acc}


def _alpaca_replay(tokenizer, max_length, cache_dir, style):
    ds = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=cache_dir)
    def make(row):
        instr = row["instruction"] + (
            f"\n\n{row['input']}" if row.get("input") else "")
        return _sft_example(tokenizer, _wrap_prompt(instr, style, tokenizer),
                            row["output"], max_length)
    return ds, make


# ---------------------------------------------------------------------------
# Registry + prompt builders per task
# ---------------------------------------------------------------------------

def _mk_prompt_math(row, style, tok):
    return _wrap_prompt(GSM8K_INSTR.format(q=row["question"]), style, tok)

def _mk_prompt_code(row, style, tok):
    return _wrap_prompt(MBPP_INSTR.format(text=row["text"],
                                          test=row["test_list"][0]), style, tok)

def _mk_prompt_instr(row, style, tok):
    return _wrap_prompt(row["prompt"], style, tok)


CAUSAL_TASKS: Dict[str, dict] = {
    "math": dict(metric="exact_match", replay=_gsm8k_replay, eval=_gsm8k_eval,
                 prompt=_mk_prompt_math, score=_gsm8k_score, max_new_tokens=400),
    "coding": dict(metric="pass@1", replay=_mbpp_replay, eval=_mbpp_eval,
                   prompt=_mk_prompt_code, score=_mbpp_score, max_new_tokens=400),
    "instruction": dict(metric="ifeval_strict", replay=_alpaca_replay,
                        eval=_ifeval_eval, prompt=_mk_prompt_instr,
                        score=_ifeval_score, max_new_tokens=512),
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
                          max_new_tokens=d["max_new_tokens"])


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
    texts = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
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
    return spec.score_generations(eval_rows, texts)
