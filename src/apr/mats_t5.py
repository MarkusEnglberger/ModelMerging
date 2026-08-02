"""MaTS/T0 eight-task benchmark (Tam et al., TMLR 2023).

This is the shared-output IA3 setting from the authors' ``p3_eight_qa`` mixture.
All tasks use the same frozen T5 language-model head; there is no task-specific
classifier. Data rendering follows the released implementation: original-task
PromptSource templates, a cyclic template at evaluation time, and the authors'
train-tail holdouts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
import re
import string
from collections import Counter
import os


@dataclass(frozen=True)
class MatsTaskSpec:
    name: str
    dataset_stash: Tuple[str, ...]
    template_stash: Tuple[str, ...]
    label_key: Optional[str]
    label_map: Optional[Dict[str, int]] = None
    holdout: int = 1000
    ignored_templates: Tuple[str, ...] = ()
    metric: str = "accuracy"
    eval_kind: str = "mats_p3"

    @property
    def is_regression(self) -> bool:
        # This keeps the shared replay sampler from requiring a special type.
        # The supplied config disables class balancing (ROPES has no class label).
        return self.metric == "squad"

    @property
    def num_labels(self) -> int:
        return 1 if self.metric == "squad" else 2

    def label(self, row) -> int:
        if self.label_key is None:
            return 0
        value = row[self.label_key]
        if self.name == "social_iqa":
            return int(value) - 1
        if self.label_map is not None:
            return self.label_map[str(value)]
        return int(value)


_LETTERS_2 = {"A": 0, "B": 1}
_LETTERS_8 = {c: i for i, c in enumerate("ABCDEFGH")}
MATS_TASKS: Dict[str, MatsTaskSpec] = {
    "cosmos_qa": MatsTaskSpec("cosmos_qa", ("cosmos_qa",), ("cosmos_qa",), "label"),
    "social_iqa": MatsTaskSpec(
        "social_iqa", ("social_i_qa",), ("social_i_qa",), "label",
        ignored_templates=("Check if a random answer is valid or not",)),
    "paws": MatsTaskSpec(
        "paws", ("paws", "labeled_final"), ("paws", "labeled_final"), "label"),
    "quail": MatsTaskSpec("quail", ("quail",), ("quail",), "correct_answer_id"),
    "wiki_qa": MatsTaskSpec("wiki_qa", ("wiki_qa",), ("wiki_qa",), "label"),
    "quartz": MatsTaskSpec(
        "quartz", ("quartz",), ("quartz",), "answerKey", _LETTERS_2, holdout=200),
    "qasc": MatsTaskSpec(
        "qasc", ("qasc",), ("qasc",), "answerKey", _LETTERS_8, holdout=500),
    "ropes": MatsTaskSpec(
        "ropes", ("ropes",), ("ropes",), None, metric="squad"),
}

MATS_IA3_REPOS = {task: f"r-three/eight-qa-{task}-ia3" for task in MATS_TASKS}
MATS_IA3_FILENAMES = {
    "cosmos_qa": "checkpoint_1299.pt",
    "paws": "checkpoint_2699.pt",
    "qasc": "checkpoint_499.pt",
    "quail": "checkpoint_1399.pt",
    "quartz": "checkpoint_1799.pt",
    "ropes": "checkpoint_799.pt",
    "social_iqa": "checkpoint_1099.pt",
    "wiki_qa": "checkpoint_99.pt",
}

# Exact PromptSource configs retained by the MaTS reader after filtering for
# ``original_task`` and Accuracy/SQuAD metrics. These are materialized as safe
# Parquet by bigscience/P3, so runtime does not execute legacy dataset scripts.
MATS_P3_CONFIGS = {
    "cosmos_qa": [
        "cosmos_qa_description_context_question_answer_text",
        "cosmos_qa_description_context_question_text",
        "cosmos_qa_description_context_question_answer_id",
        "cosmos_qa_context_description_question_answer_text",
        "cosmos_qa_no_prompt_id", "cosmos_qa_no_prompt_text",
        "cosmos_qa_context_description_question_answer_id",
        "cosmos_qa_context_question_description_answer_id",
        "cosmos_qa_context_description_question_text",
        "cosmos_qa_context_question_description_answer_text",
    ],
    "social_iqa": [
        "social_i_qa_I_was_wondering",
        "social_i_qa_Show_choices_and_generate_answer",
        "social_i_qa_Generate_answer",
        "social_i_qa_Show_choices_and_generate_index",
    ],
    "paws": [
        "paws_labeled_final_task_description_no_label",
        "paws_labeled_final_Meaning",
        "paws_labeled_final_context_question_no_label",
        "paws_labeled_final_Rewrite_no_label",
        "paws_labeled_final_context_question",
        "paws_labeled_final_Concatenation",
        "paws_labeled_final_Concatenation_no_label",
        "paws_labeled_final_Meaning_no_label",
        "paws_labeled_final_PAWS_ANLI_GPT3",
        "paws_labeled_final_Rewrite",
        "paws_labeled_final_PAWS_ANLI_GPT3_no_label",
    ],
    "quail": [
        "quail_context_question_answer_description_id",
        "quail_context_question_answer_description_text",
        "quail_description_context_question_answer_id",
        "quail_context_question_description_answer_text",
        "quail_context_question_description_answer_id",
        "quail_no_prompt_id", "quail_context_description_question_answer_id",
        "quail_no_prompt_text", "quail_context_description_question_answer_text",
        "quail_description_context_question_answer_text",
    ],
    "wiki_qa": [
        "wiki_qa_Is_This_True_", "wiki_qa_automatic_system",
        "wiki_qa_found_on_google", "wiki_qa_exercise",
        "wiki_qa_Decide_good_answer",
    ],
    "quartz": [
        "quartz_use_info_from_question_paragraph",
        "quartz_paragraph_question_plain_concat",
        "quartz_use_info_from_paragraph_question",
        "quartz_answer_question_based_on", "quartz_answer_question_below",
        "quartz_read_passage_below_choose", "quartz_having_read_above_passage",
        "quartz_given_the_fact_answer_the_q",
    ],
    "qasc": [
        "qasc_qa_with_separated_facts_1", "qasc_qa_with_separated_facts_3",
        "qasc_qa_with_separated_facts_4", "qasc_qa_with_separated_facts_5",
        "qasc_qa_with_separated_facts_2",
    ],
    "ropes": [
        "ropes_prompt_beginning", "ropes_prompt_bottom_hint_beginning",
        "ropes_given_background_situation", "ropes_plain_bottom_hint",
        "ropes_plain_background_situation", "ropes_background_new_situation_answer",
        "ropes_background_situation_middle", "ropes_new_situation_background_answer",
        "ropes_prompt_mix", "ropes_read_background_situation",
    ],
}

# Paths are exactly those selected by the released MaTS checkpoint registry.
# The bucket README asks users to put ``exp_out`` under a local ``mms`` folder.
_CKPT_REL = {
    "cosmos_qa": "exp_out/p3/cosmos_qa/google-t5-large-lm-adapt/2023-04-30-10-16-25/checkpoints/checkpoint_399.pt",
    "paws": "exp_out/p3/paws/google-t5-large-lm-adapt/2023-04-30-00-43-21/checkpoints/checkpoint_1299.pt",
    "qasc": "exp_out/p3/qasc/google-t5-large-lm-adapt/2023-04-29-21-15-10/checkpoints/checkpoint_199.pt",
    "quail": "exp_out/p3/quail/google-t5-large-lm-adapt/2023-04-29-21-11-24/checkpoints/checkpoint_999.pt",
    "quartz": "exp_out/p3/quartz/google-t5-large-lm-adapt/2023-04-29-21-13-32/checkpoints/checkpoint_399.pt",
    "ropes": "exp_out/p3/ropes/google-t5-large-lm-adapt/2023-04-29-21-11-51/checkpoints/checkpoint_499.pt",
    "social_iqa": "exp_out/p3/social_iqa/google-t5-large-lm-adapt/2023-04-29-21-13-12/checkpoints/checkpoint_799.pt",
    "wiki_qa": "exp_out/p3/wiki_qa/google-t5-large-lm-adapt/2023-04-29-21-15-46/checkpoints/checkpoint_499.pt",
}


def get_mats_task(name: str) -> MatsTaskSpec:
    key = name.lower().replace("-", "_")
    if key not in MATS_TASKS:
        raise KeyError(f"Unknown MaTS task '{name}'. Known: {sorted(MATS_TASKS)}")
    return MATS_TASKS[key]


def mats_checkpoint(task: str, root: str = "mms") -> str:
    return str(Path(root) / _CKPT_REL[get_mats_task(task).name])


def _templates(spec: MatsTaskSpec):
    try:
        from promptsource.templates import DatasetTemplates
    except ImportError as exc:
        raise ImportError(
            "The MaTS benchmark needs promptsource==0.2.3 to reproduce its P3 "
            "templates. Install the optional dependency from requirements.txt."
        ) from exc
    allowed = "Squad" if spec.metric == "squad" else "Accuracy"
    out = []
    for template in DatasetTemplates(*spec.template_stash).templates.values():
        meta = template.metadata
        if (meta.original_task and all(m == allowed for m in meta.metrics)
                and template.name not in spec.ignored_templates):
            out.append(template)
    if not out:
        raise RuntimeError(f"No compatible PromptSource templates for {spec.name}")
    return out


class _RenderedView:
    """Lazy PromptSource view, avoiding a large materialised P3 cross-product."""

    def __init__(self, raw, templates, tokenizer, spec, max_length, evaluation):
        self.raw = raw
        self.templates = templates
        self.tokenizer = tokenizer
        self.spec = spec
        self.max_length = max_length
        self.evaluation = evaluation

    def __len__(self):
        # MaTS trains on the example x template cross-product, but evaluates with
        # template ``example_index % num_templates``. Represent both lazily.
        return len(self.raw) if self.evaluation else len(self.raw) * len(self.templates)

    def __getitem__(self, index):
        if isinstance(index, str):
            if index != "labels":
                raise KeyError(index)
            return [self[i]["labels"] for i in range(len(self))]
        index = int(index)
        if self.evaluation:
            raw_index, template_index = index, index % len(self.templates)
        else:
            raw_index, template_index = divmod(index, len(self.templates))
        row = self.raw[raw_index]
        template = self.templates[template_index]
        input_text, target_text = template.apply(row)
        enc = self.tokenizer(input_text, truncation=True, max_length=self.max_length)
        target = self.tokenizer(target_text, truncation=True, max_length=64)
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"])),
            "target_ids": target["input_ids"],
            "labels": self.spec.label(row),
        }
        if not self.evaluation:
            return item
        choices = template.get_answer_choices_list(row)
        if choices is not None:
            item["choice_ids"] = [
                self.tokenizer(c, truncation=True, max_length=64)["input_ids"]
                for c in choices
            ]
        else:
            answers = row.get("answers", {"text": [target_text]})
            item["references"] = list(answers.get("text", [target_text]))
            item["example_id"] = str(row.get("id", raw_index))
        return item

    def select(self, indices):
        return _RenderedView(self.raw.select(indices), self.templates, self.tokenizer,
                             self.spec, self.max_length, self.evaluation)


def load_mats_task_dataset(spec: MatsTaskSpec, tokenizer, max_length: int,
                           cache_dir: Optional[str] = None):
    data_root = Path(os.environ.get("MATS_DATA_DIR", ".mats_data")) / spec.name
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Safe Parquet snapshot not found at {data_root}. Run "
            "`python scripts/prefetch_mats_t5.py` on a networked node first."
        )
    template_sets = [load_from_disk(str(data_root / f"template_{i}"))
                     for i in range(len(MATS_P3_CONFIGS[spec.name]))]
    return (_P3View(template_sets, tokenizer, spec, max_length, False),
            _P3View(template_sets, tokenizer, spec, max_length, True))


class _P3View:
    """Lazy MaTS view over the official pre-rendered BigScience P3 Parquet."""

    def __init__(self, template_sets, tokenizer, spec, max_length, evaluation,
                 indices=None):
        self.sets = template_sets
        self.tokenizer = tokenizer
        self.spec = spec
        self.max_length = max_length
        self.evaluation = evaluation
        train_len = len(template_sets[0]["train"])
        self.train_len = train_len - spec.holdout
        if self.train_len <= 0:
            raise RuntimeError(f"Invalid {spec.name} holdout for P3 train split")
        if evaluation:
            # The released QASC reader evaluates on its 500-example train-tail
            # holdout because its public test split is unlabeled.
            eval_len = spec.holdout if spec.name == "qasc" else len(
                template_sets[0]["validation"])
            self.indices = list(range(eval_len)) if indices is None else list(indices)
        else:
            self.indices = None

    def __len__(self):
        return len(self.indices) if self.evaluation else self.train_len * len(self.sets)

    def _row(self, raw_index, template_index):
        if self.evaluation and self.spec.name == "qasc":
            return self.sets[template_index]["train"][self.train_len + raw_index]
        split = "validation" if self.evaluation else "train"
        return self.sets[template_index][split][raw_index]

    def __getitem__(self, index):
        if isinstance(index, str):
            if index != "labels":
                raise KeyError(index)
            return [self[i]["labels"] for i in range(len(self))]
        index = int(index)
        if self.evaluation:
            raw_index = self.indices[index]
            template_index = raw_index % len(self.sets)
        else:
            raw_index, template_index = divmod(index, len(self.sets))
        row = self._row(raw_index, template_index)
        input_text, target_text = (row["inputs_pretokenized"],
                                   row["targets_pretokenized"])
        enc = self.tokenizer(input_text, truncation=True, max_length=self.max_length)
        target = self.tokenizer(target_text, truncation=True, max_length=64)
        choices = row.get("answer_choices")
        label = 0
        if choices:
            # P3 preserves the whitespace emitted around a Jinja template's
            # target (often one or two leading newlines), while answer_choices
            # stores the unpadded strings. PromptSource treats these as the same
            # answer, so recover the label after stripping template whitespace.
            try:
                label = [str(choice).strip() for choice in choices].index(
                    target_text.strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"P3 target is not an answer choice for {self.spec.name}: "
                    f"{target_text!r}") from exc
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"])),
            "target_ids": target["input_ids"], "labels": label,
        }
        if self.evaluation:
            if choices:
                item["choice_ids"] = [
                    self.tokenizer(c, truncation=True, max_length=64)["input_ids"]
                    for c in choices]
            else:
                item["references"] = [target_text]
                item["example_id"] = str(raw_index)
        return item

    def select(self, indices):
        if not self.evaluation:
            raise ValueError("select is only supported for the MaTS evaluation view")
        selected = [self.indices[int(i)] for i in indices]
        return _P3View(self.sets, self.tokenizer, self.spec, self.max_length,
                       True, selected)


def make_mats_collator(tokenizer):
    def collate(features: List[Dict]):
        inputs = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                  for f in features]
        batch = tokenizer.pad(inputs, return_tensors="pt")
        tmax = max(len(f["target_ids"]) for f in features)
        targets = torch.full((len(features), tmax), -100, dtype=torch.long)
        for i, feature in enumerate(features):
            ids = torch.as_tensor(feature["target_ids"], dtype=torch.long)
            targets[i, :len(ids)] = ids
        batch["target_ids"] = targets
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        if "choice_ids" in features[0]:
            n_choices = max(len(f["choice_ids"]) for f in features)
            max_choice_len = max(len(c) for f in features for c in f["choice_ids"])
            choices = torch.full((len(features), n_choices, max_choice_len),
                                 -100, dtype=torch.long)
            choice_mask = torch.zeros((len(features), n_choices), dtype=torch.bool)
            for i, feature in enumerate(features):
                for j, choice in enumerate(feature["choice_ids"]):
                    choices[i, j, :len(choice)] = torch.as_tensor(choice)
                    choice_mask[i, j] = True
            batch["choice_ids"] = choices
            batch["choice_mask"] = choice_mask
        if "references" in features[0]:
            batch["references"] = [f["references"] for f in features]
            batch["example_ids"] = [f["example_id"] for f in features]
        return batch
    return collate


class MatsT5Model(nn.Module):
    """Whole T5 model: every parameter, including the LM head, is mergeable."""

    base_model_prefix = "t5"

    def __init__(self, t5, tokenizer, spec):
        super().__init__()
        self.t5 = t5
        self.tokenizer = tokenizer
        self.spec = spec

    def forward(self, input_ids=None, attention_mask=None, target_ids=None,
                labels=None, **_):
        loss = None
        if target_ids is not None:
            loss = self.t5(input_ids=input_ids, attention_mask=attention_mask,
                           labels=target_ids).loss
        return _Output(None, loss)

    def score_choices(self, input_ids, attention_mask, choice_ids, choice_mask):
        """MaTS candidate score: summed (not length-normalized) token log-prob."""
        batch_size, n_choices, length = choice_ids.shape
        enc = self.t5.get_encoder()(input_ids=input_ids, attention_mask=attention_mask,
                                    return_dict=True).last_hidden_state
        hidden = enc[:, None].expand(-1, n_choices, -1, -1).reshape(
            batch_size * n_choices, enc.shape[1], enc.shape[2])
        mask = attention_mask[:, None].expand(-1, n_choices, -1).reshape(
            batch_size * n_choices, attention_mask.shape[1])
        targets = choice_ids.reshape(batch_size * n_choices, length)
        out = self.t5(encoder_outputs=(hidden,), attention_mask=mask, labels=targets)
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        valid_tokens = targets.ne(-100)
        safe_targets = targets.masked_fill(~valid_tokens, 0)
        token_scores = logp.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        scores = (token_scores * valid_tokens).sum(-1).reshape(batch_size, n_choices)
        return scores.masked_fill(~choice_mask, torch.finfo(scores.dtype).min)


class IA3Linear(nn.Module):
    """T5 linear layer with the output-channel scaling used by MaTS IA3."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.ia3_scale = nn.Parameter(torch.ones(
            base.out_features, 1, dtype=base.weight.dtype,
            device=base.weight.device))

    def forward(self, inputs):
        output = self.base(inputs)
        # Checkpoints store (out_features, 1); broadcast over batch/sequence.
        return output * self.ia3_scale.squeeze(-1)


def inject_mats_ia3(t5):
    """Inject the exact 192 k/v/wi_1 adapters used in the released setup."""
    for parameter in t5.parameters():
        parameter.requires_grad_(False)
    for module_name, module in list(t5.named_modules()):
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in {"SelfAttention", "EncDecAttention", "DenseReluDense"}:
            continue
        for child_name, child in list(module.named_children()):
            if child_name in {"k", "v", "wi_1"} and isinstance(child, nn.Linear):
                setattr(module, child_name, IA3Linear(child))
    scales = [n for n, _ in t5.named_parameters() if n.endswith(".ia3_scale")]
    expected = 3 * t5.config.num_layers + 5 * t5.config.num_decoder_layers
    if len(scales) != expected:
        raise RuntimeError(f"Expected {expected} T5 IA3 tensors, found {len(scales)}")
    return t5


class MatsIA3Model(MatsT5Model):
    """MaTS model whose merge space is only the IA3 scaling parameters."""

    def mergeable_param_names(self):
        return [name for name, _ in self.named_parameters()
                if name.endswith(".ia3_scale")]


@dataclass
class _Output:
    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]


def build_mats_assets(base_model: str, cache_dir=None, torch_dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    model = T5ForConditionalGeneration.from_pretrained(base_model, cache_dir=cache_dir,
                                                        **kwargs)
    return tokenizer, model


def build_mats_ia3_assets(base_model: str, cache_dir=None, torch_dtype=None):
    tokenizer, model = build_mats_assets(base_model, cache_dir, torch_dtype)
    return tokenizer, inject_mats_ia3(model)


def _load_ia3_state(checkpoint: str, cache_dir=None):
    path = Path(checkpoint)
    if not path.is_file():
        task = get_mats_task(checkpoint).name
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(
            repo_id=MATS_IA3_REPOS[task], filename=MATS_IA3_FILENAMES[task],
            cache_dir=cache_dir))
    return torch.load(path, map_location="cpu", weights_only=True)


def apply_mats_ia3_expert(model, task_or_checkpoint: str, cache_dir=None):
    """Load an official IA3 checkpoint into an already-instantiated model.

    Keeping the frozen T5 backbone shared is important for the eight-task MaTS
    benchmark: the experts differ only in their 192 IA3 tensors.  This helper
    lets the pipeline snapshot each expert without constructing eight identical
    copies of T5-large.
    """
    source = _load_ia3_state(task_or_checkpoint, cache_dir)
    target = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for name, value in source.items():
            if not name.startswith("transformer.base_model.model."):
                raise RuntimeError(f"Unexpected IA3 checkpoint key: {name}")
            name = name[len("transformer.base_model.model."):]
            name = name.replace(".ia3_l.default", ".ia3_scale")
            # Accept either the bare injected T5 or the MatsIA3Model wrapper.
            candidates = (name, f"t5.{name}")
            matched = next((candidate for candidate in candidates if candidate in target), None)
            if matched is None:
                raise RuntimeError(f"IA3 checkpoint key does not match model: {name}")
            target[matched].copy_(value.to(dtype=target[matched].dtype))
            loaded.add(matched)
    expected = {n for n in target if n.endswith(".ia3_scale")}
    if loaded != expected:
        raise RuntimeError(f"IA3 checkpoint coverage mismatch: missing={sorted(expected-loaded)[:4]}")
    return model


def load_mats_ia3_expert(base_model: str, task_or_checkpoint: str, cache_dir=None,
                          torch_dtype=None):
    """Load one official ``r-three/eight-qa-*-ia3`` expert."""
    _, model = build_mats_ia3_assets(base_model, cache_dir, torch_dtype)
    return apply_mats_ia3_expert(model, task_or_checkpoint, cache_dir)


def load_mats_expert(base_model: str, checkpoint: str, cache_dir=None,
                     torch_dtype=None):
    """Load an official ``checkpoint_*.pt`` into current Transformers T5."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"MaTS expert checkpoint not found: {path}. Download the official "
            "bucket so its exp_out directory is under ./mms, or set checkpoint "
            "explicitly in the YAML."
        )
    kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    model = T5ForConditionalGeneration.from_pretrained(base_model, cache_dir=cache_dir,
                                                        **kwargs)
    checkpoint_state = torch.load(path, map_location="cpu", weights_only=True)
    state = {}
    for name, value in checkpoint_state.items():
        if name.startswith("transformer."):
            name = name[len("transformer."):]
        state[name] = value
    if "shared.weight" in state:
        state.setdefault("encoder.embed_tokens.weight", state["shared.weight"])
        state.setdefault("decoder.embed_tokens.weight", state["shared.weight"])
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in MaTS checkpoint: {incompatible.unexpected_keys[:8]}")
    # Checkpoints intentionally contain only trainable parameters; full-model
    # checkpoints should nevertheless cover every non-aliased T5 parameter.
    allowed_missing = {"encoder.embed_tokens.weight", "decoder.embed_tokens.weight",
                       "lm_head.weight"}
    missing = [k for k in incompatible.missing_keys if k not in allowed_missing]
    if missing:
        raise RuntimeError(f"Missing keys in full-model MaTS checkpoint: {missing[:8]}")
    return model


def _normalize_answer(text: str) -> List[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).split()


def _squad_scores(prediction: str, references: Sequence[str]):
    pred = _normalize_answer(prediction)
    exact = float(any(pred == _normalize_answer(ref) for ref in references))
    f1s = []
    for ref_text in references:
        ref = _normalize_answer(ref_text)
        common = sum((Counter(pred) & Counter(ref)).values())
        if not pred or not ref:
            f1s.append(float(pred == ref))
        elif common == 0:
            f1s.append(0.0)
        else:
            precision, recall = common / len(pred), common / len(ref)
            f1s.append(2 * precision * recall / (precision + recall))
    return exact, max(f1s, default=0.0)


@torch.no_grad()
def evaluate_mats(model, eval_ds, spec, collator, batch_size, device,
                  num_workers=0):
    """Official-style P3 evaluation: candidate accuracy or SQuAD EM/F1."""
    model.eval()
    model.to(device)
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator,
                        num_workers=num_workers,
                        pin_memory=str(device).startswith("cuda"))
    if spec.metric == "accuracy":
        correct = total = 0
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch.pop("target_ids")
            scores = model.score_choices(**{k: v.to(device) for k, v in batch.items()})
            correct += int(scores.argmax(-1).eq(labels).sum())
            total += len(labels)
        value = correct / max(total, 1)
        return {"accuracy": value, "primary": value}

    exacts, f1s = [], []
    for batch in loader:
        references = batch.pop("references")
        batch.pop("example_ids")
        batch.pop("labels")
        batch.pop("target_ids")
        generated = model.t5.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=64,
        )
        predictions = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for prediction, refs in zip(predictions, references):
            exact, f1 = _squad_scores(prediction, refs)
            exacts.append(exact)
            f1s.append(f1)
    exact = sum(exacts) / max(len(exacts), 1)
    f1 = sum(f1s) / max(len(f1s), 1)
    # Released scorer's ``average`` averages all returned SQuAD metrics.
    return {"exact_match": exact, "f1": f1, "primary": (exact + f1) / 2}
