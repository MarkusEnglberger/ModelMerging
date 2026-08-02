#!/usr/bin/env python
"""Figure: training-accuracy vs held-out-retention frontier (fig:clip20heldout).

Reads results/compare/heldout_retention_clip20_full.json and renders the
accuracy--retention scatter with the Pareto frontier as a step line.

Encoding (print-safe): identity is carried by BOTH a fixed CVD-safe palette
(Okabe-Ito) and marker shape, so the figure survives grayscale printing;
selective direct labels on the frontier and on the cautionary points only.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "results/compare/heldout_retention_clip20_full.json")
OUT = os.path.join(HERE, "..", "Model Merging Causal/figs/heldout_frontier.pdf")

# fixed family assignment (never cycled): ours=blue/circle, controls=orange/tri,
# merges=gray/square, base=black/star
FAM = {
    "apr":    dict(color="#0072B2", marker="o", label="APR (ours)", z=5, s=46),
    "ctl":    dict(color="#E69F00", marker="^", label="GD / ungated from base", z=4, s=42),
    "merge":  dict(color="#8a8a8a", marker="s", label="Merges (train-14)", z=3, s=38),
    "base":   dict(color="#000000", marker="*", label=r"Base model $\theta_0$", z=6, s=140),
}

LABELS = {  # selective direct labels: frontier + cautionary points
    "base:theta0": (r"$\theta_0$", (5, 5)),
    "gd:from=base14@lr5e-05": ("GD 5e-5", (5, 4)),
    "apr:from=base14@lr4": ("APR base lr4", (-8, 7)),
    "apr:from=base14@lr8": ("APR base lr8", (2, -13)),
    "apr:from=ada14@lr4": (r"APR$\leftarrow$AdaMerging", (-30, 9)),
    "merge:BC14@d0.1,o0.01,l0.2": ("Breadcrumbs", (6, -11)),
    "merge:TA14@l0.2": (r"TA $\lambda$=0.2", (6, 3)),
    "merge:RegMean14@nd1": ("RegMean (dist 386)", (-40, -13)),
    "merge:TIES14@d0.1,l0.8": (r"TIES $\lambda$=0.8", (6, -3)),
    "nogate:from=base14@lr2": ("ungated lr2", (4, -11)),
    "nogate:from=base14@lr8": ("ungated lr8", (5, -3)),
}


def family(name):
    if name.startswith("base"):
        return "base"
    if name.startswith(("gd:", "nogate:")):
        return "ctl"
    if name.startswith("apr:"):
        return "apr"
    return "merge"


SRC2 = os.path.join(HERE, "..", "results/compare/heldout_nogate_sweep.json")


def main():
    d = json.load(open(SRC))
    pts = {k: (v["train_mean"], v["held_mean"]) for k, v in d["cells"].items()}
    if os.path.exists(SRC2):
        for k, v in json.load(open(SRC2))["cells"].items():
            if k.startswith("nogate") and k not in pts:
                pts[k] = (v["train_mean"], v["held_mean"])

    # Pareto frontier (maximize both axes)
    front = []
    best_h = -1
    for k, (t, h) in sorted(pts.items(), key=lambda kv: -kv[1][0]):
        if h > best_h:
            front.append((t, h))
            best_h = h
    front = sorted(front)  # ascending train

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    # frontier step line, recessive, behind the points
    fx = [t for t, _ in front]
    fy = [h for _, h in front]
    ax.step(fx, fy, where="post", color="#0072B2", lw=1.2, alpha=0.35, zorder=1)

    seen = set()
    for k, (t, h) in pts.items():
        f = FAM[family(k)]
        lbl = f["label"] if f["label"] not in seen else None
        seen.add(f["label"])
        ax.scatter(t, h, c=f["color"], marker=f["marker"], s=f["s"],
                   zorder=f["z"], label=lbl,
                   edgecolors="white", linewidths=0.6)
        if k in LABELS:
            txt, (dx, dy) = LABELS[k]
            ax.annotate(txt, (t, h), textcoords="offset points", xytext=(dx, dy),
                        fontsize=7.5, color="#333333")

    ax.axhline(pts["base:theta0"][1], color="#bbbbbb", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel("Training-task accuracy (14-task mean)")
    ax.set_ylabel("Held-out zero-shot accuracy (6-task mean)")
    ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[done] -> {OUT}  ({len(pts)} points, {len(front)} on frontier)")


if __name__ == "__main__":
    main()
