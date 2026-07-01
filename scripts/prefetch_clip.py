#!/usr/bin/env python
"""Pre-fetch CLIP/ViT assets into .hf_cache on a networked node.

The compute nodes run offline (HF_HUB_OFFLINE=1), so the base backbone, the
per-task fine-tuned vision experts, and the image datasets must be downloaded
here first. Run this on the login node before submitting slurm/clip_poc.sbatch.

Usage:
    python scripts/prefetch_clip.py --tasks eurosat gtsrb mnist
    python scripts/prefetch_clip.py --all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

from datasets import load_dataset
from transformers import CLIPModel, CLIPVisionModel, CLIPImageProcessor, CLIPTokenizer

from apr.vision import VISION_TASKS, VISION_EXPERT_CKPT, get_vision_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/clip-vit-base-patch32")
    ap.add_argument("--tasks", nargs="*", default=["eurosat", "gtsrb", "mnist"])
    ap.add_argument("--all", action="store_true", help="fetch the full 8-task suite")
    args = ap.parse_args()

    tasks = sorted(VISION_TASKS) if args.all else args.tasks
    cache = os.environ["HF_HOME"]
    print(f"[prefetch] HF_HOME={cache}")

    print(f"[prefetch] base backbone {args.base}")
    CLIPModel.from_pretrained(args.base)
    CLIPImageProcessor.from_pretrained(args.base)
    CLIPTokenizer.from_pretrained(args.base)

    for t in tasks:
        spec = get_vision_task(t)
        ckpt = VISION_EXPERT_CKPT[spec.name]
        print(f"[prefetch] {spec.name}: expert {ckpt}")
        CLIPVisionModel.from_pretrained(ckpt)
        print(f"[prefetch] {spec.name}: dataset {spec.dataset}"
              f"{'' if spec.dataset_config is None else ' ('+spec.dataset_config+')'}")
        kw = {} if spec.dataset_config is None else {"name": spec.dataset_config}
        load_dataset(spec.dataset, **kw)

    print("[prefetch] done.")


if __name__ == "__main__":
    main()
