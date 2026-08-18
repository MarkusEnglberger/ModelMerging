#!/usr/bin/env python
"""Figure: score vs distance from theta_0, in a single absolute frame.

Every point shares ONE origin: x is the distance from the pretrained model
theta_0 (log scale), y is the absolute score. That is the only frame in which
the scatter is readable. An earlier version also plotted refinement cells
launched from each merge, with x measured from THEIR OWN initialization -- but
those points mix origins on both axes (a cell inherits its initialization's
score, and its x is measured from a different point), so two points at the same
(x, y) could represent a +0.5 gain and a +0.03 gain. No trend could be read off
that cloud. Only the from-pretrained arms share an origin, so only they are
plotted; the composition results live in the grid table, where they are legible.

Merge baselines appear as gray reference squares at their own theta_0-distance,
showing where the merge cloud sits relative to the refinement. Their distances
come from measure scripts/merge_distances.py, because the grid runs record merge
displacement relative to the config-lambda merge rather than to theta_0, and
only task arithmetic (linear in lambda) can be converted analytically.

Cross-family caveat for the caption: distances are comparable in magnitude but
not as a drift proxy ACROSS merge families (RegMean sits hundreds of units out
while retaining well), which is why capability drift is measured functionally
elsewhere. The claim this figure supports is the order-of-magnitude one.

Encoding follows fig_heldout_frontier.py's print-safe convention: identity by
fixed CVD-safe color (Okabe-Ito, validated) AND marker shape, so the figure
survives grayscale; selective direct labels; recessive grid.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "paperICLR/figs/n16_displacement.pdf")

BENCH = [("glue8", "GLUE-8 (RoBERTa)", "mean_normret", "normalized retention"),
         ("clip8", "CLIP-8 (ViT-B/32)", "mean_normret", "normalized retention"),
         ("clip20", "CLIP-20 (ViT-B/32)", "mean_acc", "top-1 accuracy")]
ROWS = ["ta", "ties", "dareties", "bc", "ada"]

# fixed arm assignment (never cycled); validated: #0072B2,#E69F00,#009E73 pass
# the six checks on the light surface (orange contrast WARN covered by shape +
# direct labels).
ARM = {"apr":    dict(color="#0072B2", marker="o", label="APR"),
       "gd":     dict(color="#E69F00", marker="^", label="GD"),
       "nogate": dict(color="#009E73", marker="D", label="ungated")}


# Explicit file pins. The default discovery order picks up the older S<=50
# grids, whose winners no longer match the paper's table (which now reports the
# extended S<=200 search). Pin the files the table was built from so the figure
# and the table cannot drift apart.
PIN = {
    ("glue8", "base"): "grid_nn_glue8_base_n16_horizon.json",
    ("clip8", "base"): "grid_nn_clip8_base_n16_s200_tatiesdaretiesbcada.json",
    ("clip8", "merges"): "grid_nn_clip8_merges_n16_s200_tatiesdareties.json",
}


def load(bench, mode):
    pin = PIN.get((bench, mode))
    if pin:
        p = os.path.join(ROOT, "results/compare", pin)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
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


CFG_LAM = {"glue8": 0.3, "clip8": 0.3, "clip20": 0.08}
FAM_LABEL = {"ta": "TA", "ties": "TIES", "dareties": "DARE-TIES",
             "bc": "Breadcrumbs", "ada": "AdaMerging"}


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
    """[(arm, displacement_from_theta0, score)] -- from-pretrained arms only.

    In the base-mode run the refinement starts at theta_0, so the recorded
    ``displacement`` IS the distance from theta_0 (verified against the
    independent measurement in pretrain_drift_glue8.json: apr 2.4679,
    nogate 5.3202, gd 0.4522 agree exactly)."""
    pts = []
    base = load(bench, "base")
    if not base:
        return pts
    for arm in ARM:
        w = winner(base["methods"], arm, "ta")
        if w and w[1].get("aggregate"):
            pts.append((arm, w[1]["displacement"], w[1]["aggregate"][metric]))
    return pts


def merge_refs(bench, metric):
    """[(family_label, dist_theta0, score)] for each family's selected merge.

    TA is derived analytically (linear in lambda); the rest are read from the
    measured file if present, and silently omitted otherwise."""
    refs = []
    ta = ta_reference(bench, metric)
    if ta:
        refs.append(("TA", ta[0], ta[1]))
    mer = load(bench, "merges")
    mpath = os.path.join(ROOT, f"results/compare/merge_distances_{bench}.json")
    if mer and os.path.exists(mpath):
        meas = json.load(open(mpath))["distances"]
        for fam, rec in meas.items():
            if fam == "ta":
                continue
            cell = mer["methods"].get(rec["cell"])
            if cell and cell.get("aggregate"):
                refs.append((FAM_LABEL.get(fam, fam), rec["dist_theta0"],
                             cell["aggregate"][metric]))
    return refs


def main():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1))

    print(f"{'bench':9s}{'what':13s}{'dist0':>10s}{'score':>8s}")
    for ax, (bench, title, metric, ylab) in zip(axes.flat, BENCH):
        # merge cloud first, recessive, behind the refinement points
        for lbl, d, sc in merge_refs(bench, metric):
            ax.scatter(d, sc, c="#8a8a8a", marker="s", s=34, zorder=3,
                       edgecolors="white", linewidths=0.6)
            ax.annotate(lbl, (d, sc), textcoords="offset points",
                        xytext=(-4, 5), ha="right", fontsize=6.2,
                        color="#555555")
            print(f"{bench:9s}{('merge ' + lbl):13s}{d:>10.3f}{sc:>8.3f}")
        for arm, d, sc in collect(bench, metric):
            a = ARM[arm]
            ax.scatter(d, sc, c=a["color"], marker=a["marker"], s=64,
                       edgecolors="black", linewidths=1.1, zorder=5)
            print(f"{bench:9s}{a['label']:13s}{d:>10.3f}{sc:>8.3f}")
        # the do-nothing reference: pretrained alone
        floor = {"clip20": 0.550}.get(bench, 0.0)
        ax.axhline(floor, color="#8a8a8a", lw=0.8, ls=":", zorder=1)
        ax.text(0.985, floor, r"$\theta_0$", ha="right",
                va="bottom", fontsize=7, color="#6a6a6a",
                transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        # narrow log ranges (CLIP-8 spans barely a decade) otherwise print
        # overlapping minor-tick labels on top of each other
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", lw=0.3, alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat:
        ax.set_xlabel(r"distance from $\theta_0$ (log)")

    handles = [Line2D([], [], color=a["color"], marker=a["marker"], ls="",
                      markersize=6, label=a["label"]) for a in ARM.values()]
    handles.append(Line2D([], [], color="#8a8a8a", marker="s", ls="",
                          markersize=6, label="merge baselines"))
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
