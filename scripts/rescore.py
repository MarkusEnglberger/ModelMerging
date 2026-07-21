#!/usr/bin/env python
"""Re-report an existing results JSON by ABSOLUTE mean accuracy.

The merge_baselines / compare_baselines reports store the raw per-task scores
for every method, so runs made before the metric change (which selected and
printed mean NormRet) can be re-scored post hoc -- no GPU re-run needed.

Why: NormRet = (merge-base)/(expert-base) blows up when an expert barely beats
the zero-shot base. On the 20-task CLIP suite stl10 (gap .0043) and food101
(gap .0351) alone moved the reported mean from -1.04 to -4.94.

Usage:
    python scripts/rescore.py results/compare/merge_baselines_clip20.json
    python scripts/rescore.py <json> --top 20 --gap_min 0.05
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.metrics import aggregate_all, degenerate_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--gap_min", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=None, help="show only the top N")
    ap.add_argument("--sort_by", default="mean_acc",
                    choices=["mean_acc", "worst_acc", "mean_normret",
                             "mean_normret_nondeg"])
    args = ap.parse_args()

    with open(args.report) as f:
        rep = json.load(f)
    base, expert = rep["base"], rep["expert"]
    methods = rep.get("methods") or rep.get("cells") or {}

    deg = degenerate_tasks(base, expert, args.gap_min)
    print(f"# {args.report}")
    print(f"# tasks={len(base)}  degenerate (expert-base < {args.gap_min}): "
          f"{deg or 'none'}")
    for t in deg:
        print(f"#   {t}: base={base[t]:.4f} expert={expert[t]:.4f} "
              f"gap={expert[t]-base[t]:.4f}")
    print(f"# expert ceiling: mean_acc="
          f"{sum(expert.values())/len(expert):.4f}   "
          f"zero-shot floor: mean_acc={sum(base.values())/len(base):.4f}")

    rows = []
    for name, d in methods.items():
        scores = d.get("scores")
        if not scores:
            continue  # cells-style report without raw scores
        nr = d.get("normret") or {
            t: (scores[t] - base[t]) / (expert[t] - base[t]) for t in scores}
        ag = aggregate_all(scores, nr, base, expert, args.gap_min)
        rows.append((name, ag))
    if not rows:
        print("(no methods with raw per-task scores in this report)")
        return

    rows.sort(key=lambda r: -r[1].get(args.sort_by, float("-inf")))
    if args.top:
        rows = rows[:args.top]

    print()
    print(f"{'method':<34} {'mAcc':>7} {'wAcc':>7} {'mNR':>9} {'mNR*':>8}")
    print("-" * 70)
    for name, ag in rows:
        nd = ag.get("mean_normret_nondeg")
        print(f"{name:<34} {ag['mean_acc']:>7.4f} {ag['worst_acc']:>7.4f} "
              f"{ag['mean_normret']:>9.3f} "
              f"{(f'{nd:.3f}' if nd is not None else '-'):>8}")
    print("-" * 70)
    print("mAcc/wAcc = absolute mean/worst accuracy (primary).")
    print("mNR = normalized retention over ALL tasks (unstable here).")
    print("mNR* = normalized retention over non-degenerate tasks only.")


if __name__ == "__main__":
    main()
