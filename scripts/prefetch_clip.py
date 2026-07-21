#!/usr/bin/env python
"""Pre-fetch CLIP/ViT assets into .hf_cache on a networked node.

The compute nodes run offline (HF_HUB_OFFLINE=1), so the base backbone, the
per-task fine-tuned vision experts, and the image datasets must be downloaded
here first. Run this on the login node before submitting slurm/clip_poc.sbatch.

Usage:
    python scripts/prefetch_clip.py --tasks eurosat gtsrb mnist
    python scripts/prefetch_clip.py --suite 8      # original task-arithmetic set
    python scripts/prefetch_clip.py --suite 20     # full TALL-masks suite
    python scripts/prefetch_clip.py --all
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

from datasets import load_dataset
from transformers import CLIPModel, CLIPVisionModel, CLIPImageProcessor, CLIPTokenizer

from apr.vision import (VISION_TASKS, VISION_EXPERT_CKPT, VISION_SUITES,
                        get_vision_task, load_vision_dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/clip-vit-base-patch32")
    ap.add_argument("--tasks", nargs="*", default=["eurosat", "gtsrb", "mnist"])
    ap.add_argument("--suite", choices=sorted(VISION_SUITES),
                    help="fetch a named suite: 8, 14, or 20 tasks")
    ap.add_argument("--all", action="store_true", help="fetch every registered task")
    ap.add_argument("--verify", action="store_true",
                    help="also load each task via load_vision_dataset and print "
                         "num classes + sizes (catches split/label-key breakage)")
    args = ap.parse_args()

    if args.all:
        tasks = sorted(VISION_TASKS)
    elif args.suite:
        tasks = list(VISION_SUITES[args.suite])
    else:
        tasks = args.tasks
    cache = os.environ["HF_HOME"]
    print(f"[prefetch] HF_HOME={cache}")
    print(f"[prefetch] {len(tasks)} task(s): {', '.join(tasks)}")

    print(f"[prefetch] base backbone {args.base}")
    CLIPModel.from_pretrained(args.base)
    CLIPImageProcessor.from_pretrained(args.base)
    CLIPTokenizer.from_pretrained(args.base)

    failures = []
    for t in tasks:
        spec = get_vision_task(t)
        try:
            ckpt = VISION_EXPERT_CKPT[spec.name]
            print(f"[prefetch] {spec.name}: expert {ckpt}")
            CLIPVisionModel.from_pretrained(ckpt)
            print(f"[prefetch] {spec.name}: dataset {spec.dataset}"
                  f"{'' if spec.dataset_config is None else ' ('+spec.dataset_config+')'}")
            kw = {} if spec.dataset_config is None else {"name": spec.dataset_config}
            load_dataset(spec.dataset, **kw)
            if args.verify:
                tr, ev, names = load_vision_dataset(spec)
                print(f"[prefetch] {spec.name}: OK  classes={len(names)} "
                      f"train={len(tr)} eval={len(ev)}  e.g. {names[:3]}")
        except Exception as e:  # noqa: BLE001 -- report and continue
            failures.append((spec.name, repr(e)))
            print(f"[prefetch] {spec.name}: FAILED -- {e}")
            traceback.print_exc()

    if failures:
        print(f"[prefetch] done with {len(failures)} failure(s):")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print("[prefetch] done.")


if __name__ == "__main__":
    main()
