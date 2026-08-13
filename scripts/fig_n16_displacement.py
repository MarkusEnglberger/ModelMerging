#!/usr/bin/env python
"""Figure: score vs parameter displacement for the n=16+16 deployment tier.

One panel per benchmark. Each point is a SPLIT-SELECTED winner of one
refinement arm from one initialization row of Table 2 (the n+n protocol);
x is the parameter distance the refinement moved from ITS OWN initialization
(log scale), y is the reported score. From-pretrained points (ringed) measure
distance from theta_0 itself -- the literal drift-from-pretrained quantity.

Encoding follows fig_heldout_frontier.py's print-safe convention: identity by
fixed CVD-safe color (Okabe-Ito, validated) AND marker shape, so the figure
survives grayscale; selective direct labels; recessive grid.

Displacement caveat stated in the caption: composition-row x values are
distances from each merge initialization, not from theta_0, and parameter
distance is a within-family diagnostic only (the paper measures capability
drift functionally; cf. the held-out study).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "paperICLR/figs/n16_displacement.pdf")

BENCH = [("glue8", "GLUE-8 (RoBERTa)", "mean_normret", "normalized retention"),
         ("clip8", "CLIP-8 (ViT-B/32)", "mean_normret", "normalized retention"),
         ("t5_glue8", "GLUE-8 t2t (flan-T5)", "mean_normret", "normalized retention"),
         ("clip20", "CLIP-20 (ViT-B/32)", "mean_acc", "top-1 accuracy")]
ROWS = ["ta", "ties", "dareties", "bc", "ada"]

# fixed arm assignment (never cycled); validated: #0072B2,#E69F00,#009E73 pass
# the six checks on the light surface (orange contrast WARN covered by shape +
# direct labels).
ARM = {"apr":    dict(color="#0072B2", marker="o", label="APR"),
       "gd":     dict(color="#E69F00", marker="^", label="GD"),
       "nogate": dict(color="#009E73", marker="D", label="ungated")}


def load(bench, mode):
    for suffix in ("_fast", ""):
        p = os.path.join(ROOT, f"results/compare/grid_nn_{bench}_{mode}_n16{suffix}.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def winner(methods, arm, init):
    cand = [(k, v) for k, v in methods.items()
            if k.startswith(f"{arm}:from={init}@")
            and v.get("selection_obj") is not None]
    if not cand:
        return None
    k, v = min(cand, key=lambda kv: kv[1]["selection_obj"])
    return k, v


CFG_LAM = {"glue8": 0.3, "clip8": 0.3, "t5_glue8": 0.3, "clip20": 0.08}


def ta_reference(bench, metric):
    """(dist from theta0, score) of the best TA merge cell. TA is linear in
    lambda -- m(lam) = theta0 + lam*sum(tau) -- so its theta0-distance follows
    exactly from the measured ||theta0 - m(lam_cfg)|| in the base-mode file."""
    base, mer = load(bench, "base"), load(bench, "merges")
    if not (base and mer):
        return None
    D = base["methods"]["merge:TA@l0"]["displacement"]  # ||theta0 - m(lam_cfg)||
    v = mer["methods"][mer["best_per_family"]["ta"]]
    return v["lam"] * D / CFG_LAM[bench], v["aggregate"][metric]


def collect(bench, metric):
    """[(arm, init, displacement, score, from_pretrained)]"""
    pts = []
    base = load(bench, "base")
    if base:
        for arm in ARM:
            w = winner(base["methods"], arm, "ta")
            if w and w[1].get("aggregate"):
                pts.append((arm, "pretrained", w[1]["displacement"],
                            w[1]["aggregate"][metric], True))
    mer = load(bench, "merges")
    if mer:
        ok_ada = "probe_buffer" in (mer["grids"].get("ada_data") or [])
        for init in ROWS:
            if init == "ada" and not ok_ada:
                continue
            for arm in ARM:
                w = winner(mer["methods"], arm, init)
                if w and w[1].get("aggregate"):
                    pts.append((arm, init, w[1]["displacement"],
                                w[1]["aggregate"][metric], False))
    return pts


def main():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))

    print(f"{'bench':9s}{'arm':8s}{'init':11s}{'disp':>9s}{'score':>8s}")
    for ax, (bench, title, metric, ylab) in zip(axes.flat, BENCH):
        pts = collect(bench, metric)
        for arm, init, d, s, from_pre in sorted(pts, key=lambda p: not p[4]):
            a = ARM[arm]
            ax.scatter(d, s, c=a["color"], marker=a["marker"],
                       s=64 if from_pre else 26,
                       edgecolors="black" if from_pre else "white",
                       linewidths=1.1 if from_pre else 0.6,
                       zorder=5 if from_pre else 3)
            print(f"{bench:9s}{arm:8s}{init:11s}{d:>9.3f}{s:>8.3f}")
        # the tuned TA merge as a reference: gray square (de-emphasized family,
        # directly labeled -- same convention as the held-out frontier figure)
        ref = ta_reference(bench, metric)
        if ref:
            ax.scatter(*ref, c="#8a8a8a", marker="s", s=40, zorder=4,
                       edgecolors="white", linewidths=0.6)
            ax.annotate("TA merge", ref, textcoords="offset points",
                        xytext=(-4, 6), ha="right", fontsize=7, color="#555555")
        # the do-nothing reference: pretrained alone
        floor = {"clip20": 0.550}.get(bench, 0.0)
        ax.axhline(floor, color="#8a8a8a", lw=0.8, ls=":", zorder=1)
        ax.text(0.985, floor, r"$\theta_0$", ha="right",
                va="bottom", fontsize=7, color="#6a6a6a",
                transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", lw=0.3, alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("parameter displacement from initialization (log)")

    handles = [Line2D([], [], color=a["color"], marker=a["marker"], ls="",
                      markersize=6, label=a["label"]) for a in ARM.values()]
    handles.append(Line2D([], [], color="#555555", marker="o", ls="",
                          markersize=8, markerfacecolor="none",
                          markeredgewidth=1.1, label="from pretrained (ringed)"))
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
