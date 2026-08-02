#!/usr/bin/env python
"""Prefetch the reproducible MaTS/T0 IA3 benchmark assets.

Uses BigScience's generated P3 Parquet files rather than executing the legacy
dataset repositories' loading scripts.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))

from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
import requests
from transformers import AutoTokenizer, T5ForConditionalGeneration

from apr.mats_t5 import (MATS_TASKS, MATS_IA3_FILENAMES, MATS_IA3_REPOS,
                         MATS_P3_CONFIGS)

BASE = "google/t5-large-lm-adapt"

print(f"[prefetch] HF_HOME={os.environ['HF_HOME']}")
print(f"[prefetch] base {BASE}")
AutoTokenizer.from_pretrained(BASE)
T5ForConditionalGeneration.from_pretrained(BASE)

data_root = os.path.abspath(os.environ.get("MATS_DATA_DIR", ".mats_data"))
os.makedirs(data_root, exist_ok=True)
for task, configs in MATS_P3_CONFIGS.items():
    print(f"[prefetch] P3 task {task}: {len(configs)} templates")
    task_root = os.path.join(data_root, task)
    os.makedirs(task_root, exist_ok=True)
    for index, config in enumerate(configs):
        output = os.path.join(task_root, f"template_{index}")
        if os.path.isdir(output):
            print(f"  [{index + 1}/{len(configs)}] cached {config}")
            continue
        response = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": "bigscience/P3", "config": config}, timeout=60)
        response.raise_for_status()
        files = response.json()["parquet_files"]
        needed = {"train"} if task == "qasc" else {"train", "validation"}
        data_files = {
            split: [entry["url"] for entry in files if entry["split"] == split]
            for split in needed
        }
        if any(not paths for paths in data_files.values()):
            raise RuntimeError(f"Missing P3 Parquet split for {config}: {data_files}")
        print(f"  [{index + 1}/{len(configs)}] download {config}")
        DatasetDict(load_dataset("parquet", data_files=data_files)).save_to_disk(output)

for task in MATS_TASKS:
    path = hf_hub_download(repo_id=MATS_IA3_REPOS[task],
                           filename=MATS_IA3_FILENAMES[task])
    print(f"[prefetch] IA3 expert {task}: {path}")

print("[prefetch] done. All assets can now be used offline.")
