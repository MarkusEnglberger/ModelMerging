#!/usr/bin/env python
"""Run a merge + refine + evaluate experiment from a YAML config.

Usage:
    python scripts/run_merge.py --config configs/poc3.yaml [--override key=val ...]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import run_experiment


def apply_overrides(cfg: ExperimentConfig, overrides):
    """Allow `refine.steps=2`, `data.n_probe=16`, `tag=foo` style overrides."""
    for ov in overrides:
        key, _, val = ov.partition("=")
        # cast
        for caster in (int, float):
            try:
                val = caster(val)
                break
            except ValueError:
                continue
        if isinstance(val, str) and val.lower() in ("true", "false"):
            val = val.lower() == "true"
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="output json path")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    os.makedirs(cfg.output_dir, exist_ok=True)
    results = run_experiment(cfg)

    out = args.out or os.path.join(cfg.output_dir, f"{cfg.tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] results -> {out}")

    agg = results["aggregate"]
    print(f"[summary] merge  : mean={agg['merge']['mean_normret']:.3f} "
          f"worst={agg['merge']['worst_normret']:.3f}")
    print(f"[summary] refined: mean={agg['refined']['mean_normret']:.3f} "
          f"worst={agg['refined']['worst_normret']:.3f}")


if __name__ == "__main__":
    main()
