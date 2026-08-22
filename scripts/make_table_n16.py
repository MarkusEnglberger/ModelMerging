"""Generate a split-selected Table 2 panel for a requested n+n budget.

Selection mirrors merge_baselines.py exactly: within a group the winner MINIMISES
selection_obj (held-out replay loss on the disjoint selection buffer). The
evaluation split is read only at that winner, so every number printed here is
one a practitioner could have obtained without ever touching test data.
"""
import argparse
import json
import os

BENCH = ["glue8", "clip8", "t5_glue8", "clip20"]          # column order
HEAD = {"glue8": "GLUE-8", "clip8": "CLIP-8",
        "t5_glue8": "GLUE-8 t2t", "clip20": "CLIP-20"}
MEAN = {"clip20": "mean_acc"}          # default mean_normret
WORST = {"clip20": "worst_acc"}        # default worst_normret
ROWS = [("pretrained", "Pretrained"), ("ta", "Task arithmetic"),
        ("bc", "Breadcrumbs"), ("dareties", "DARE-TIES"),
        ("ties", "TIES"), ("apgd", "DOGE/APGD"),
        ("ada", "AdaMerging"), ("regmean", "RegMean"),
        ("fisher", "Fisher merging"), ("ls_learned", "Learned L\\&S")]
ARMS = [("alone", "alone"), ("gd", "$+$GD"),
        ("nogate", "$+$ungated"), ("apr", "$+$APR")]


def load(bench, mode, n):
    # A recovery wave may extend a boundary-hit grid without overwriting the
    # canonical run.  Prefer that strict superset once it exists, otherwise
    # retain the established n=16 fast-file and canonical fallbacks.
    for suffix in ("_edge", "_fast", ""):
        p = f"results/compare/grid_nn_{bench}_{mode}_n{n}{suffix}.json"
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p, suffix == "_fast"
    return None, None, None


def cell(v, mkey, wkey):
    ag = v.get("aggregate")
    if not ag:
        return None
    return {"mean": ag.get(mkey), "worst": ag.get(wkey),
            "lr": v.get("lr"), "S": v.get("steps")}


def pick(methods, pred, mkey, wkey):
    cand = [(k, v) for k, v in methods.items()
            if pred(k) and v.get("selection_obj") is not None]
    if not cand:
        return None
    k, v = min(cand, key=lambda kv: kv[1]["selection_obj"])
    c = cell(v, mkey, wkey)
    if c:
        c["cell"] = k
        c["n_cells"] = len(cand)
    return c


def collect(bench, n):
    mkey, wkey = MEAN.get(bench, "mean_normret"), WORST.get(bench, "worst_normret")
    out, meta = {}, {}
    base, bp, bfast = load(bench, "base", n)
    if base:
        m = base["methods"]
        r = {}
        z = m.get("merge:TA@l0")
        if z:
            r["alone"] = cell(z, mkey, wkey)
        for a in ("apr", "nogate", "gd"):
            r[a] = pick(m, lambda k, a=a: k.startswith(f"{a}:from=ta@"), mkey, wkey)
        out["pretrained"] = r
        meta["base"] = {"file": bp, "S": base.get("steps")}
    mer, mp, mfast = load(bench, "merges", n)
    if mer:
        m = mer["methods"]
        # The paper reports the MATCHED-BUDGET AdaMerging (n unlabeled inputs per
        # task, the same budget APR gets labeled). A run left on the standard
        # transductive default adapted to the evaluation split itself, so its
        # AdaMerging cells answer a different question and are withheld.
        fams = ["ta", "ties", "dareties", "bc", "ada"]
        if "probe_buffer" not in (mer["grids"].get("ada_data") or []):
            fams.remove("ada")
            meta["ada_withheld"] = mer["grids"].get("ada_data")
        for key in fams:
            name = (mer.get("best_per_family") or {}).get(key)
            r = {"alone": cell(m[name], mkey, wkey) if name in m else None}
            for a in ("apr", "nogate", "gd"):
                r[a] = pick(m, lambda k, a=a, key=key:
                            k.startswith(f"{a}:from={key}@"), mkey, wkey)
            if any(r.values()):
                out[key] = r
        meta["merges"] = {"file": mp, "S": mer.get("steps")}
    labeled_path = f"results/compare/grid_nn_{bench}_labeled_n{n}.json"
    if os.path.exists(labeled_path):
        with open(labeled_path) as f:
            labeled = json.load(f)
        methods = labeled.get("methods", {})
        mapping = {"fisher": "labeled:fisher",
                   "ls_learned": "labeled:ls-learned"}
        for row, method in mapping.items():
            if method in methods:
                out[row] = {"alone": cell(methods[method], mkey, wkey)}
        meta["labeled"] = {"file": labeled_path, "S": None}
    apgd_path = f"results/compare/grid_nn_{bench}_apgd_n{n}.json"
    if os.path.exists(apgd_path):
        with open(apgd_path) as f:
            apgd = json.load(f)
        selected = apgd.get("selected")
        if selected in apgd.get("methods", {}):
            out["apgd"] = {
                "alone": cell(apgd["methods"][selected], mkey, wkey)}
            meta["apgd"] = {"file": apgd_path, "S": None}
    regmean_path = f"results/compare/grid_nn_{bench}_regmean_n{n}.json"
    if os.path.exists(regmean_path):
        with open(regmean_path) as f:
            regmean = json.load(f)
        m = regmean.get("methods", {})
        name = (regmean.get("best_per_family") or {}).get("regmean")
        r = {"alone": cell(m[name], mkey, wkey) if name in m else None}
        for a in ("apr", "nogate", "gd"):
            r[a] = pick(m, lambda k, a=a:
                        k.startswith(f"{a}:from=regmean@"), mkey, wkey)
        if any(r.values()):
            out["regmean"] = r
            meta["regmean"] = {"file": regmean_path,
                               "S": regmean.get("steps")}
    return out, meta


def _w(w):
    """Worst-task, in the paper's convention: sign in braces, leading 0 dropped."""
    sign = "{-}" if w < 0 else "{+}"
    a = abs(w)
    body = f"{a:.2f}"[1:] if a < 1 else f"{a:.2f}"
    return sign + body


def fmt(c, best, smax=None):
    if c is None or c.get("mean") is None:
        return "---"
    m, w = c["mean"], c["worst"]
    ms = f"{m:.3f}"
    if best:
        ms = r"\mathbf{" + ms + "}"
    # a cell whose selected horizon is the longest offered has not been shown to
    # have an interior optimum: its value is a lower bound, not a maximum.
    if smax is not None and c.get("S") is not None and c["S"] >= smax:
        ms += r"^{\dagger}"
    return f"${ms}\\ ({_w(w)})$"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16,
                    help="replay and selection examples per task (default: 16)")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be positive")

    data = {b: collect(b, args.n) for b in BENCH}
    # bold the best mean per benchmark column
    best = {}
    for b in BENCH:
        vals = [(rk, ak, c["mean"]) for rk, r in data[b][0].items()
                for ak, c in r.items() if c and c.get("mean") is not None]
        if vals:
            best[b] = max(vals, key=lambda t: t[2])[:2]

    lines = []
    for i, (rk, rlabel) in enumerate(ROWS):
        if i:
            lines.append(r"\midrule")
        row_arms = (ARMS if rk not in {"apgd", "fisher", "ls_learned"}
                    else [("alone", "alone")])
        for j, (ak, alabel) in enumerate(row_arms):
            left = rlabel if j == 0 else ""
            cells = []
            for b in BENCH:
                c = (data[b][0].get(rk) or {}).get(ak)
                # the pretrained model IS the retention zero point, by definition
                if rk == "pretrained" and ak == "alone" and b != "clip20" and c:
                    cells.append("$0$")
                    continue
                src = ("base" if rk == "pretrained" else
                       "regmean" if rk == "regmean" else "merges")
                sm = (data[b][1].get(src) or {}).get("S")
                cells.append(fmt(c, best.get(b) == (rk, ak),
                                 max(sm) if sm and ak != "alone" else None))
            lines.append(f"{left:<16}& {alabel:<11}& " + " & ".join(cells) + r" \\")
    print("\n".join(lines))

    print("\n%%% provenance")
    for b in BENCH:
        for k, v in data[b][1].items():
            if isinstance(v, dict) and "file" in v:
                print(f"%   {b:9s} {k:8s} {v['file']}  S={v['S']}")
            else:
                print(f"%   {b:9s} {k:8s} {v}")
    print("\n%%% selected cells + EDGE AUDIT (a cell at a grid edge is a lower bound)")
    GRIDKEY = {"apr": "apr_lrs", "nogate": "nogate_lrs", "gd": "control_gd_lrs"}
    for b in BENCH:
        for src in ("base", "merges", "regmean"):
            if src not in data[b][1]:
                continue
            with open(data[b][1][src]["file"]) as f:
                g = json.load(f)
            grids, S = g["grids"], g["steps"]
            for rk, r in data[b][0].items():
                expected_src = ("base" if rk == "pretrained" else
                                "regmean" if rk == "regmean" else "merges")
                if src != expected_src:
                    continue
                for ak, c in r.items():
                    if not c or not c.get("cell") or ak == "alone":
                        continue
                    lrs = grids.get(GRIDKEY[ak]) or []
                    flags = []
                    if lrs and c["lr"] is not None:
                        if c["lr"] <= min(lrs):
                            flags.append("lr@MIN")
                        if c["lr"] >= max(lrs):
                            flags.append("lr@MAX")
                    if S and c["S"] is not None and c["S"] >= max(S):
                        flags.append("S@MAX")
                    print(f"%   {b:9s} {rk:11s} {ak:7s} lr={c['lr']:<8g} S={c['S']:<4}"
                          f" of {c.get('n_cells','?'):>2} cells  "
                          + (" ".join(flags) if flags else "interior"))


main()
