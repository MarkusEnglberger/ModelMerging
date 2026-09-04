"""Emit the tabular bodies of the two main results tables (tab:b32 / tab:b16).

Reads the protocol-v3 result files and prints one LaTeX body per budget with
per-column bolding computed, so the hand-maintained tables cannot drift from
the data. Row layout: pretrained; the four checkpoint-only merges each paired
with its +XGD row; the from-pretrained arms; then the budget-spending
constructions (aTLAS, TATR, RegMean, Fisher, L&S paired with +XGD where run,
GradFix and DOGE unpaired).
"""
import json, glob, statistics as st, sys

SUITES = ("clip8", "glue8", "clip20")
def draws(pat):
    fs = [f for f in sorted(glob.glob(pat)) if "superseded" not in f]
    return [list(json.load(open(f))["draws"].values())[0]["methods"] for f in fs]

def agg(D, key, field):
    v = [d[key]["aggregate"][field] for d in D]
    return (st.mean(v), st.stdev(v))

def cells(B, key_by_suite, suffix_by_suite):
    out = []
    for su in SUITES:
        D = draws(f"results/compare/cv_{su}_B{B}_{suffix_by_suite}_draw?.json")
        k = key_by_suite
        if k == "APRINIT":
            k = [x for x in D[0] if x.startswith("apr:from=")][0] if D else None
        if not D or k not in D[0]:
            out += [None, None]; continue
        out += [agg(D, k, "mean_acc"), agg(D, k, "worst_acc")]
    return out

def rows_for(B):
    r = []
    r.append(("Pretrained $\\thetab$", [(0.478,None),(0.315,None),(0.354,None),(-0.047,None),(0.550,None),(0.100,None)]))
    r.append(("MIDRULE", None))
    for fam, init in (("TA","ta"),("BC","bc"),("DARETIES","dareties"),("TIES","ties")):
        nm = {"TA":"Task arithmetic","BC":"Breadcrumbs","DARETIES":"DARE-TIES","TIES":"TIES"}[fam]
        r.append((nm, cells(B, f"merge:{fam}", "pretrained_v3fresh")))
        r.append(("\\quad $+$XGD", cells(B, "APRINIT", f"{init}_v3fresh")))
    r.append(("MIDRULE", None))
    for arm in ("gd","nogate","apr"):
        nm = {"gd":"$\\thetab+$GD","nogate":"$\\thetab+$ungated","apr":"$\\thetab+$XGD"}[arm]
        r.append((nm, cells(B, f"{arm}:from=pretrained", "pretrained_v3fresh")))
    r.append(("MIDRULE", None))
    pairs = [("aTLAS","merge:ATLAS","pretrained_v3atlas","atlas"),
             ("TATR","merge:TATR","pretrained_v3tatr","tatr"),
             ("RegMean","merge:REGMEAN","pretrained_v3regmean","regmean"),
             ("Fisher (replay)","merge:FISHER","pretrained_v3fisherls","fisher"),
             ("Localize-and-Stitch","merge:LS","pretrained_v3fisherls","ls")]
    for nm, key, suf, init in pairs:
        r.append((nm, cells(B, key, suf)))
        r.append(("\\quad $+$XGD", cells(B, "APRINIT", f"{init}_v3fresh")))
    r.append(("GradFix", cells(B, "merge:GRADFIX", "pretrained_v3gradfix")))
    r.append(("DOGE/APGD", cells(B, "merge:DOGE", "pretrained_v3doge")))
    return r

def emit(B):
    rows = rows_for(B)
    # column maxima over the mean of each (mean, worst) column
    best = [max((row[1][c][0] for _, row in enumerate(rows)
                 if row[1] and row[1][c]) , default=None) for c in range(6)
            ] if False else None
    vals = [row for row in rows if row[1]]
    best = []
    for c in range(6):
        best.append(max(v[1][c][0] for v in vals if v[1][c] is not None))
    body = []
    for name, cs in rows:
        if cs is None:
            body.append("\\midrule"); continue
        parts = []
        for c, cell in enumerate(cs):
            if cell is None:
                parts.append("--"); continue
            m, sd = cell
            t = f"{m:.3f}"
            if abs(m - best[c]) < 5e-4: t = f"\\mathbf{{{t}}}"
            parts.append(f"${t}$" if sd is None else f"${t}_{{\\pm{sd:.3f}}}$")
        body.append(f"{name:24s} & " + " & ".join(parts) + " \\\\")
    return "\n".join(body)

if __name__ == "__main__":
    for B in (32, 16):
        print(f"%%%% BODY B={B} %%%%")
        print(emit(B))
