"""GLUE task registry.

Each task carries enough metadata to (a) load and tokenize its data, (b) build
the right classification/regression head, and (c) score predictions with the
GLUE-convention metric. We follow the standard GLUE-8 setting used in model
merging studies: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE (WNLI excluded).
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TaskSpec:
    name: str  # our canonical lowercase id
    glue_config: str  # HuggingFace `glue` config name
    text_keys: Tuple[str, ...]  # 1 or 2 input columns
    num_labels: int  # 1 => regression (STS-B)
    metric: str  # primary metric id for selection/retention
    # column holding the gold label in the HF dataset:
    label_key: str = "label"
    # eval split name (MNLI uses a non-standard validation split name):
    eval_split: str = "validation"

    @property
    def is_regression(self) -> bool:
        return self.num_labels == 1


# Canonical GLUE-8 specs.
TASKS = {
    "cola": TaskSpec("cola", "cola", ("sentence",), 2, "matthews"),
    "sst2": TaskSpec("sst2", "sst2", ("sentence",), 2, "accuracy"),
    "mrpc": TaskSpec("mrpc", "mrpc", ("sentence1", "sentence2"), 2, "acc_f1"),
    "stsb": TaskSpec("stsb", "stsb", ("sentence1", "sentence2"), 1, "pearson_spearman"),
    "qqp": TaskSpec("qqp", "qqp", ("question1", "question2"), 2, "acc_f1"),
    "mnli": TaskSpec("mnli", "mnli", ("premise", "hypothesis"), 3, "accuracy",
                     eval_split="validation_matched"),
    "qnli": TaskSpec("qnli", "qnli", ("question", "sentence"), 2, "accuracy"),
    "rte": TaskSpec("rte", "rte", ("sentence1", "sentence2"), 2, "accuracy"),
}

# Interpretable task clusters used for the H4 (related-task) analysis.
CLUSTERS = {
    "paraphrase_similarity": ["mrpc", "qqp", "stsb"],
    "entailment": ["mnli", "qnli", "rte"],
    "single_sentence": ["cola", "sst2"],
}


def get_task(name: str) -> TaskSpec:
    key = name.lower().replace("-", "").replace("_", "")
    # normalise common variants (sts-b, sst-2, ...)
    alias = {"stsb": "stsb", "sst2": "sst2", "mnlimatched": "mnli"}
    key = alias.get(key, key)
    if key not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Known: {sorted(TASKS)}")
    return TASKS[key]
