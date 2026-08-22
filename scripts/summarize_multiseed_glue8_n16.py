#!/usr/bin/env python
"""Aggregate pinned GLUE-8 n=16 verification reports over replay seeds."""

import argparse
import glob
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob", default="results/compare/ms_glue8_n16_seed*.json")
    parser.add_argument(
        "--out", default="results/compare/multiseed_glue8_n16.json")
    parser.add_argument("--expect", type=int, default=5)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if len(paths) != args.expect:
        raise SystemExit(
            f"expected {args.expect} reports matching {args.glob!r}, got "
            f"{len(paths)}: {paths}")
    reports = [json.load(open(path)) for path in paths]
    seeds = [r["protocol"]["probe_seed"] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate replay seeds: {seeds}")

    keys = list(reports[0]["cells"])
    if any(set(r["cells"]) != set(keys) for r in reports[1:]):
        raise SystemExit("reports do not contain the same pinned cells")

    summary = {
        "protocol": reports[0]["protocol"],
        "seeds": seeds,
        "source_reports": paths,
        "cells": {},
    }
    summary["protocol"]["probe_seed"] = seeds
    for key in keys:
        means = [r["cells"][key]["aggregate"]["mean_normret"]
                 for r in reports]
        worsts = [r["cells"][key]["aggregate"]["worst_normret"]
                  for r in reports]
        summary["cells"][key] = {
            "mean": float(np.mean(means)),
            "mean_std": float(np.std(means, ddof=1)),
            "worst": float(np.mean(worsts)),
            "worst_std": float(np.std(worsts, ddof=1)),
            "means": means,
            "worsts": worsts,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"{'cell':<22} {'mean +/- std':>18} {'worst +/- std':>20}")
    print("-" * 62)
    for key, values in summary["cells"].items():
        print(f"{key:<22} {values['mean']:>7.3f} +/-{values['mean_std']:>6.3f}"
              f"   {values['worst']:>7.3f} +/-{values['worst_std']:>6.3f}")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
