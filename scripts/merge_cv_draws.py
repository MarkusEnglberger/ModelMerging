"""Merge single-draw cv_protocol output files into one multi-draw report.

Array jobs run one draw per task (slurm/cv_protocol_glue8.sbatch), each writing
a file whose draws dict holds a single entry tagged "draw0". This re-tags each
file's entry by its position on the command line, concatenates them, and
recomputes the cross-draw summary with cv_protocol's own summarize().

Usage:
  python scripts/merge_cv_draws.py OUT.json IN_draw0.json IN_draw1.json ...
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from cv_protocol import summarize  # noqa: E402


def main():
    out_path, *in_paths = sys.argv[1:]
    assert in_paths, "need at least one input file"
    merged = None
    for i, path in enumerate(in_paths):
        d = json.load(open(path))
        if merged is None:
            merged = {k: v for k, v in d.items() if k != "draws"}
            merged["draws"] = {}
            merged["protocol"]["draws"] = len(in_paths)
            merged["merged_from"] = in_paths
        assert d["protocol"]["budget_per_task"] == merged["protocol"]["budget_per_task"], path
        for tag, entry in d["draws"].items():
            merged["draws"][f"draw{i}"] = entry
    merged["summary"] = summarize(merged)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=1)
    print(f"[merged] {len(in_paths)} draws -> {out_path}")
    for nm, v in merged["summary"].items():
        print(f"  {nm:24s} acc={v['mean_acc_mean']:.4f}±{v['mean_acc_std']:.4f} "
              f"nr={v['mean_normret_mean']:.4f}±{v['mean_normret_std']:.4f} "
              f"(n={v['n_draws']})")


if __name__ == "__main__":
    main()
