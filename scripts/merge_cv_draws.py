"""Merge single-draw cv_protocol output files into one multi-draw report.

Array jobs run one draw per task (slurm/cv_protocol_glue8.sbatch), each writing
a file whose draws dict holds a single entry tagged "draw0". This re-tags each
file's entry by its draw index, concatenates them, and recomputes the
cross-draw summary with cv_protocol's own summarize().

Inputs are either bare paths (assigned draw indices 0, 1, 2, ... in order) or
``IDX:path`` tokens. Several files may name the same index: their method
dicts are stitched, later files overriding earlier ones method by method
(e.g. an APR-only rerun under a horizon cap replacing the APR entry of a full
run while keeping its merges, nogate and gd). Every draw's ``buffer_seed`` must
agree across the files stitched into it.

Usage:
  python scripts/merge_cv_draws.py OUT.json IN_draw0.json IN_draw1.json ...
  python scripts/merge_cv_draws.py OUT.json 0:full_d0.json 0:apr_d0.json 1:d1.json
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from cv_protocol import summarize  # noqa: E402


def main():
    out_path, *tokens = sys.argv[1:]
    assert tokens, "need at least one input file"
    merged = None
    stitched = {}                      # draw tag -> list of source files
    for i, tok in enumerate(tokens):
        if ":" in tok and tok.split(":", 1)[0].isdigit():
            idx, path = tok.split(":", 1)
            idx = int(idx)
        else:
            idx, path = i, tok
        d = json.load(open(path))
        if merged is None:
            merged = {k: v for k, v in d.items() if k != "draws"}
            merged["draws"] = {}
        assert d["protocol"]["budget_per_task"] == merged["protocol"]["budget_per_task"], path
        tag = f"draw{idx}"
        for _, entry in d["draws"].items():
            if tag in merged["draws"]:
                have = merged["draws"][tag]
                assert have["buffer_seed"] == entry["buffer_seed"], \
                    f"{path}: buffer seed {entry['buffer_seed']} != {have['buffer_seed']} for {tag}"
                overridden = sorted(set(have["methods"]) & set(entry["methods"]))
                have["methods"].update(entry["methods"])
                if overridden:
                    print(f"[stitch] {tag}: {path} overrides {overridden}")
            else:
                merged["draws"][tag] = entry
            stitched.setdefault(tag, []).append(path)
    merged["protocol"]["draws"] = len(merged["draws"])
    merged["merged_from"] = stitched
    merged["summary"] = summarize(merged)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=1)
    print(f"[merged] {len(merged['draws'])} draws -> {out_path}")
    for nm, v in merged["summary"].items():
        print(f"  {nm:24s} acc={v['mean_acc_mean']:.4f}±{v['mean_acc_std']:.4f} "
              f"nr={v['mean_normret_mean']:.4f}±{v['mean_normret_std']:.4f} "
              f"(n={v['n_draws']})")


if __name__ == "__main__":
    main()
