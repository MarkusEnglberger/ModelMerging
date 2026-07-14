"""T5 text-to-text GLUE track (modality: "t5").

Shared-output setting: the merged model is the ENTIRE seq2seq model (encoder,
decoder, shared embeddings, lm_head) and every task decodes through the same LM
head -- there are no task-specific heads, which removes the oracle-task-id
caveat of the multi-head RoBERTa setting. Experts are the FusionBench full
fine-tunes ``tanganke/flan-t5-base_glue-<task>`` with theta_0 = google/flan-t5-base.

Prompt templates are FusionBench's, VERBATIM (including the 'Answere' typo in
CoLA) -- the experts were trained on these exact strings, so any edit here
breaks them. Source:
fusion_bench/tasks/flan_t5_text_generation/glue_prompt_templates.py

Evaluation = classification-as-generation by candidate scoring: for each class
verbalizer we teacher-force its token sequence and take the summed log-prob;
the argmax is the prediction. This is deterministic, needs |C| forward passes
(2-3 for classification, 26 for STS-B's 0.2-quantized scale), and returns the
same (preds, labels) interface as the other tracks, so metrics.py, eval.py,
gradients.py, refine.py and the whole pipeline are reused unchanged.

The wrapper's forward contract (mirrors ClipImageClassifier):
  - ``target_ids`` present            -> ``.loss`` = seq2seq CE on the gold target
                                         (used by replay gradients / buffer loss)
  - ``labels`` absent (popped by eval) -> ``.logits`` = candidate scores (B, C),
                                         or the predicted float as (B, 1) for STS-B
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ---------------------------------------------------------------------------
# Task registry: FusionBench templates + verbalizers (verbatim).
# ---------------------------------------------------------------------------

GLUE_T5_TEMPLATES: Dict[str, dict] = {
    "cola": dict(
        glue_config="cola",
        template="Indicate if the following sentence is grammatically correct or not: "
                 "\"{sentence}\". Answere 'acceptable' or 'unacceptable'.",  # sic
        fields=("sentence",),
        verbalizers=["unacceptable", "acceptable"],  # label 0, 1
        metric="matthews",
    ),
    "sst2": dict(
        glue_config="sst2",
        template="Given the sentence '{sentence}', determine the sentiment. "
                 "Is it positive or negative?",
        fields=("sentence",),
        verbalizers=["negative", "positive"],
        metric="accuracy",
    ),
    "mrpc": dict(
        glue_config="mrpc",
        template="Are the following sentences '{sentence1}' and '{sentence2}' "
                 "conveying the same meaning?",
        fields=("sentence1", "sentence2"),
        verbalizers=["no", "yes"],
        metric="acc_f1",
    ),
    "stsb": dict(
        glue_config="stsb",
        template="Consider the sentences '{sentence1}' and '{sentence2}'. "
                 "Rate similarity on a scale from 1 to 5.",
        fields=("sentence1", "sentence2"),
        verbalizers=None,  # regression; candidates built below
        metric="pearson_spearman",
    ),
    "qqp": dict(
        glue_config="qqp",
        template="Do the questions '{question1}' and '{question2}' have the same intent?",
        fields=("question1", "question2"),
        verbalizers=["no", "yes"],
        metric="acc_f1",
    ),
    "mnli": dict(
        glue_config="mnli",
        template="Does the premise: '{premise}' logically imply, contradict, "
                 "or is neutral to the hypothesis: '{hypothesis}'?",
        fields=("premise", "hypothesis"),
        verbalizers=["entailment", "neutral", "contradiction"],
        metric="accuracy",
        eval_split="validation_matched",
    ),
    "qnli": dict(
        glue_config="qnli",
        template="Given the context: '{sentence}', does the question '{question}' "
                 "have an answer based on the information provided?",
        fields=("sentence", "question"),
        verbalizers=["yes", "no"],
        metric="accuracy",
    ),
    "rte": dict(
        glue_config="rte",
        template="Does the text: '{sentence1}' entail that '{sentence2}' is true?",
        fields=("sentence1", "sentence2"),
        verbalizers=["yes", "no"],
        metric="accuracy",
    ),
}

# STS-B: targets are "{:.1f}" strings; we score a 0.2-quantized candidate grid.
# Quantization adds a little metric noise but keeps eval to 26 scored candidates.
STSB_CANDIDATES: List[str] = [f"{v / 5:.1f}" for v in range(0, 26)]  # 0.0 .. 5.0

T5_EXPERT_CKPT = {t: f"tanganke/flan-t5-base_glue-{t}" for t in GLUE_T5_TEMPLATES}


@dataclass
class T5TaskSpec:
    """Duck-typed like tasks.TaskSpec: shared code touches name / metric /
    is_regression / num_labels only."""
    name: str
    glue_config: str
    template: str
    fields: Tuple[str, ...]
    verbalizers: Optional[List[str]]
    metric: str
    eval_split: str = "validation"
    label_key: str = "label"

    @property
    def is_regression(self) -> bool:
        return self.verbalizers is None

    @property
    def num_labels(self) -> int:
        return 1 if self.is_regression else len(self.verbalizers)


def get_t5_task(name: str) -> T5TaskSpec:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in GLUE_T5_TEMPLATES:
        raise KeyError(f"Unknown t5 task '{name}'. Known: {sorted(GLUE_T5_TEMPLATES)}")
    d = dict(GLUE_T5_TEMPLATES[key])
    return T5TaskSpec(name=key, glue_config=d.pop("glue_config"),
                      eval_split=d.pop("eval_split", "validation"), **d)


def target_text(spec: T5TaskSpec, label) -> str:
    if spec.is_regression:
        return f"{float(label):.1f}"
    return spec.verbalizers[int(label)]


# ---------------------------------------------------------------------------
# Model wrapper: whole seq2seq model is mergeable; no task-specific head.
# ---------------------------------------------------------------------------

class T5GenClassifier(nn.Module):
    """flan-T5 + fixed candidate answers for one GLUE task.

    ``base_model_prefix = "t5"`` puts EVERY parameter (shared embeddings,
    encoder, decoder, lm_head) under the mergeable prefix: in the shared-output
    setting the LM head is common to all tasks and is merged like everything
    else. `is_head_param` is False for all parameters, so nothing is frozen.
    """

    base_model_prefix = "t5"

    def __init__(self, t5: T5ForConditionalGeneration, tokenizer, spec: T5TaskSpec):
        super().__init__()
        self.t5 = t5
        self.spec = spec
        cands = STSB_CANDIDATES if spec.is_regression else spec.verbalizers
        self.candidate_values = ([float(c) for c in cands] if spec.is_regression
                                 else list(range(len(cands))))
        # pre-tokenized candidate targets, padded to a common length with -100
        enc = tokenizer(list(cands), padding=True, return_tensors="pt")
        ids = enc.input_ids  # (C, L) includes </s>
        ids[enc.attention_mask == 0] = -100
        self.register_buffer("cand_ids", ids, persistent=False)

    def _candidate_scores(self, input_ids, attention_mask) -> torch.Tensor:
        """Summed log-prob of each candidate target given the input: (B, C)."""
        B = input_ids.shape[0]
        C, L = self.cand_ids.shape
        enc_out = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state
        scores = []
        for c in range(C):
            tgt = self.cand_ids[c].unsqueeze(0).expand(B, L).contiguous()
            out = self.t5(encoder_outputs=(hidden,), attention_mask=attention_mask,
                          labels=tgt)
            logp = torch.log_softmax(out.logits, dim=-1)  # (B, L, V)
            mask = (tgt != -100)
            safe = tgt.masked_fill(~mask, 0)
            tok = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # (B, L)
            scores.append((tok * mask).sum(dim=-1))  # sum log-prob
        return torch.stack(scores, dim=1)  # (B, C)

    def forward(self, input_ids=None, attention_mask=None, target_ids=None,
                labels=None, **_):
        loss = None
        if target_ids is not None:
            loss = self.t5(input_ids=input_ids, attention_mask=attention_mask,
                           labels=target_ids).loss
        logits = None
        if labels is None:  # eval path (evaluate_task pops the metric labels)
            with torch.no_grad():
                sc = self._candidate_scores(input_ids, attention_mask)
            if self.spec.is_regression:
                vals = torch.tensor(self.candidate_values, device=sc.device)
                logits = vals[sc.argmax(dim=1)].unsqueeze(-1)  # (B, 1) floats
            else:
                logits = sc  # (B, C); argmax = class index
        return _Out(logits=logits, loss=loss)


@dataclass
class _Out:
    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Data: template-rendered inputs; target text + scalar label per example.
# ---------------------------------------------------------------------------

def load_t5_task_dataset(spec: T5TaskSpec, tokenizer, max_length: int,
                         cache_dir: Optional[str] = None):
    """Tokenized (train, eval) datasets with input_ids / target_ids / labels."""
    ds = load_dataset("glue", spec.glue_config, cache_dir=cache_dir)

    def render(batch):
        texts = [spec.template.format(**{f: v for f, v in zip(spec.fields, vals)})
                 for vals in zip(*[batch[f] for f in spec.fields])]
        enc = tokenizer(texts, truncation=True, max_length=max_length)
        tgt = tokenizer([target_text(spec, y) for y in batch[spec.label_key]],
                        truncation=True, max_length=16)
        enc["target_ids"] = tgt["input_ids"]
        return enc

    keep = {spec.label_key}
    cols = [c for c in ds["train"].column_names if c not in keep]
    ds = ds.map(render, batched=True, remove_columns=cols)
    ds = ds.rename_column(spec.label_key, "labels")
    return ds["train"], ds[spec.eval_split]


def make_t5_collator(tokenizer, is_regression: bool):
    def collate(features: List[Dict]):
        labels = [f["labels"] for f in features]
        inputs = [{"input_ids": f["input_ids"],
                   "attention_mask": f.get("attention_mask",
                                           [1] * len(f["input_ids"]))}
                  for f in features]
        batch = tokenizer.pad(inputs, return_tensors="pt")
        tmax = max(len(f["target_ids"]) for f in features)
        tgt = torch.full((len(features), tmax), -100, dtype=torch.long)
        for i, f in enumerate(features):
            ids = torch.tensor(f["target_ids"], dtype=torch.long)
            tgt[i, : len(ids)] = ids
        batch["target_ids"] = tgt
        batch["labels"] = torch.tensor(
            labels, dtype=torch.float if is_regression else torch.long)
        return batch
    return collate


def build_t5_assets(base_model: str, cache_dir: Optional[str] = None,
                    torch_dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    kw = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    base = T5ForConditionalGeneration.from_pretrained(base_model,
                                                      cache_dir=cache_dir, **kw)
    return tokenizer, base
