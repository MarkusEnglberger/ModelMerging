"""Consolidate the MaTS bracketing / multi-seed runs into paper-ready numbers.

Reads the ``bracket_s<seed><group>.json`` reports written by
``scripts/mats_fair_compare.py`` and answers the two questions the paper's gap
list raises for the shared-output track:

1. *Are the refinement grids bracketed?*  For every selected cell we report
   whether the winning learning rate or horizon sits on the boundary of the grid
   it was chosen from.  A boundary selection means the optimum is outside the
   grid and the number is not yet a tuned optimum.
2. *Do the single-seed conclusions survive a redraw?*  For each family we report
   mean +/- std across replay-buffer seeds, plus the paired APR-minus-GD margin,
   which is the quantity the claim actually rests on (the per-seed pairing
   removes the buffer-draw variance common to both methods).

Usage:
    python scripts/mats_consolidate.py                 # all bracket_s*.json
    python scripts/mats_consolidate.py --glob 'results/mats_t5_8/bracket_s*.json'
    python scripts/mats_consolidate.py --latex         # also emit a LaTeX table
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def mean_std(xs):
    """Sample mean and standard deviation (ddof=1; None when undefined)."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    if n == 1:
        return mu, float("nan")
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, var ** 0.5


def edge_flags(params, lrs, steps):
    """Which grid boundaries the selected cell sits on."""
    flags = []
    lr, st = params.get("lr"), params.get("steps")
    if lrs and lr is not None:
        if lr <= min(lrs):
            flags.append("lr@min")
        if lr >= max(lrs):
            flags.append("lr@max")
    if steps and st is not None:
        if st <= min(steps):
            flags.append("S@min")
        if st >= max(steps):
            flags.append("S@max")
    return flags


def load(paths):
    """Group (family, seed) -> report fragment across the split job files."""
    runs = {}
    for path in sorted(paths):
        with open(path) as fh:
            report = json.load(fh)
        proto = report["protocol"]
        seed = proto.get("probe_seed")
        for family, fam in report.get("families", {}).items():
            if "APR" not in fam:          # job died before this family finished
                continue
            runs[(family, seed)] = (fam, proto, os.path.basename(path))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="results/mats_t5_8/bracket_s*.json")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    paths = glob.glob(args.glob)
    if not paths:
        raise SystemExit(f"no reports matched {args.glob!r}")
    runs = load(paths)
    if not runs:
        raise SystemExit(f"{len(paths)} report(s) matched but none has a "
                         f"finished family")

    families = sorted({f for f, _ in runs})
    seeds = sorted({s for _, s in runs})
    print(f"# {len(paths)} report(s), {len(families)} famil(ies), "
          f"seeds {seeds}\n")

    # ---- 1. bracketing audit -------------------------------------------
    print("## Grid bracketing (boundary selections are NOT tuned optima)")
    # The schedule is a function of the horizon ("constant" for S<=5, otherwise
    # cosine), so a short winner confounds horizon with schedule; print it.
    print(f"{'family':18s} {'seed':>4s} {'method':>6s} {'lr':>7s} {'S':>4s} "
          f"{'sched':>9s}  edge")
    n_edge = 0
    for family in families:
        for seed in seeds:
            if (family, seed) not in runs:
                continue
            fam, proto, _ = runs[(family, seed)]
            for method in ("APR", "GD"):
                sel = fam.get(f"{method}_selection")
                if not sel:
                    continue
                key = method.lower()
                flags = edge_flags(sel["params"], proto.get(f"{key}_lrs"),
                                   proto.get(f"{key}_steps"))
                n_edge += bool(flags)
                print(f"{family:18s} {seed:>4} {method:>6s} "
                      f"{sel['params'].get('lr'):>7} {sel['params'].get('steps'):>4} "
                      f"{sel['params'].get('schedule', '?'):>9s}  "
                      f"{','.join(flags) if flags else '-- interior --'}")
    print(f"\n{n_edge} boundary selection(s); the grid is bracketed only where "
          f"every cell reads 'interior'.\n")

    # ---- 2. multi-seed aggregation -------------------------------------
    print("## Across replay-buffer seeds (mean +/- std)")
    header = (f"{'family':18s} {'n':>2s} {'baseline':>16s} {'APR':>16s} "
              f"{'GD':>16s} {'APR-GD (paired)':>17s}")
    print(header)
    rows = []
    for family in families:
        cells = defaultdict(list)
        paired = []
        for seed in seeds:
            if (family, seed) not in runs:
                continue
            fam, _, _ = runs[(family, seed)]
            for method in ("baseline", "APR", "GD"):
                agg = fam.get(method, {}).get("aggregate")
                if agg:
                    cells[method].append(agg["mean_normret"])
            if fam.get("APR", {}).get("aggregate") and \
               fam.get("GD", {}).get("aggregate"):
                paired.append(fam["APR"]["aggregate"]["mean_normret"] -
                              fam["GD"]["aggregate"]["mean_normret"])
        if not cells["APR"]:
            continue
        stats = {m: mean_std(cells[m]) for m in ("baseline", "APR", "GD")}
        pm, ps = mean_std(paired)
        fmt = lambda t: f"{t[0]:.4f}+/-{t[1]:.4f}" if t[1] == t[1] else f"{t[0]:.4f}   (1 seed)"
        print(f"{family:18s} {len(cells['APR']):>2} "
              f"{fmt(stats['baseline']):>16s} {fmt(stats['APR']):>16s} "
              f"{fmt(stats['GD']):>16s} "
              f"{(f'{pm:+.4f}+/-{ps:.4f}' if ps == ps else f'{pm:+.4f}'):>17s}")
        rows.append((family, len(cells["APR"]), stats, (pm, ps)))

    # ---- 3. safety check ------------------------------------------------
    print("\n## Safety: did any refined run land below its initialization?")
    violations = []
    for (family, seed), (fam, _, src) in sorted(runs.items()):
        base = fam.get("baseline", {}).get("aggregate", {}).get("mean_normret")
        for method in ("APR", "GD"):
            got = fam.get(method, {}).get("aggregate", {}).get("mean_normret")
            if base is not None and got is not None and got < base:
                violations.append((family, seed, method, got - base, src))
    if violations:
        for family, seed, method, delta, src in violations:
            print(f"  BELOW INIT  {family} seed={seed} {method} "
                  f"{delta:+.4f}  ({src})")
    else:
        print("  none: every refined cell finished at or above its merge.")

    if args.latex and rows:
        print("\n% --- LaTeX ---")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Initialization & Merge & APR & GD & APR $-$ GD \\\\")
        print("\\midrule")
        for family, n, stats, (pm, ps) in rows:
            cell = lambda t: (f"${t[0]:.3f}\\pm{t[1]:.3f}$" if t[1] == t[1]
                              else f"${t[0]:.3f}$")
            pair = (f"${pm:+.3f}\\pm{ps:.3f}$" if ps == ps else f"${pm:+.3f}$")
            name = family.replace("&", "\\&")
            print(f"{name} & {cell(stats['baseline'])} & {cell(stats['APR'])} "
                  f"& {cell(stats['GD'])} & {pair} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
