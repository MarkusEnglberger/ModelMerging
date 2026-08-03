"""Find measured results that exist only in logs/ and never reached results/.

Motivation: ``merge_baselines.py`` and friends write their JSON report only at
the very end, so a job that was cancelled, superseded, or whose output path was
overwritten leaves its summary table in logs/ alone. Two such mb3b runs (jobs
24650025 and 24739458, the only decoder-track runs that ever measured the
ungated and ordinary-GD arms) were missed for exactly this reason.

This scans every log for a printed summary table, extracts the method rows, and
reports which logs have no corresponding results file -- i.e. measurements that
are invisible to any analysis that reads results/ only.

    python scripts/harvest_logs.py                 # report
    python scripts/harvest_logs.py --json out.json # machine-readable
"""

import argparse
import json
import os
import re
from collections import OrderedDict

LOG_DIR = "logs"
RESULT_DIRS = ["results"]

# A summary row: "<method>  <task scores...> | <mean> <worst> ..." or the
# fair_compare/paper_compare "[eval] name -> 0.1234" style.
ROW = re.compile(r"^(?P<name>[A-Za-z][\w:@.,=&\-+()]*?)\s+"
                 r"(?P<nums>(?:-?\d+\.\d+\s+)+)\|\s+"
                 r"(?P<agg>-?\d+\.\d+)")
HEADER = re.compile(r"^method\s")
OUTPATH = re.compile(r"->\s+(results/[\w./\-]+\.json)|"
                     r"--out\s+(results/[\w./\-]+\.json)")


def parse_log(path):
    """Return (rows, declared_out) for one log file."""
    rows = OrderedDict()
    declared = set()
    in_table = False
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = OUTPATH.search(line)
                if m:
                    declared.add(m.group(1) or m.group(2))
                if HEADER.match(line):
                    in_table = True
                    continue
                if in_table:
                    m = ROW.match(line.rstrip())
                    if m:
                        rows[m.group("name")] = float(m.group("agg"))
                    elif line.strip() and not line.startswith("-"):
                        # table ended; keep scanning for a later one
                        in_table = False
    except OSError:
        pass
    return rows, declared


def existing_results():
    out = set()
    for root in RESULT_DIRS:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".json"):
                    out.add(os.path.join(dirpath, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=LOG_DIR)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    have = existing_results()
    stranded, linked = [], []
    for f in sorted(os.listdir(args.log_dir)):
        if not f.endswith((".out", ".log")):
            continue
        path = os.path.join(args.log_dir, f)
        rows, declared = parse_log(path)
        if not rows:
            continue
        missing = {d for d in declared if d not in have}
        entry = {"log": path, "n_methods": len(rows), "declared_out": sorted(declared),
                 "missing_out": sorted(missing),
                 "best": max(rows.items(), key=lambda kv: kv[1]) if rows else None,
                 "methods": rows}
        (stranded if (missing or not declared) else linked).append(entry)

    print(f"# scanned {args.log_dir}: {len(stranded) + len(linked)} log(s) with a summary table\n")
    print(f"## STRANDED -- measurements with no results/ file ({len(stranded)})")
    for e in stranded:
        arms = [m for m in e["methods"] if re.match(r"(nogate|ungated|ordinary_gd|gd[:@])", m)]
        note = f"  [has {len(arms)} ungated/GD arm(s)]" if arms else ""
        print(f"  {e['log']:44s} {e['n_methods']:>3d} methods"
              f"  best={e['best'][0]}={e['best'][1]:.4f}{note}")
        for d in e["missing_out"]:
            print(f"      declared but absent: {d}")
    print(f"\n## linked to a results file ({len(linked)})")
    for e in linked:
        print(f"  {e['log']:44s} {e['n_methods']:>3d} methods -> {', '.join(e['declared_out'])}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"stranded": stranded, "linked": linked}, fh, indent=2)
        print(f"\n[written] {args.json}")


if __name__ == "__main__":
    main()
