#!/usr/bin/env python
"""Phase 34 — the resolution scaling law as a curve, not three points.

WHAT THIS REPLACES. p26_matrix.py:242-284 fits log r = a + b log n through
exactly three points -- city (n=1), gu (n=25), dong (n=424) -- and then
extrapolates that single exponent to 1,831 polygons, and p27_survey.py:345 pushes
the same exponent out to six to ten MILLION locations. Three points spanning 17x
cannot identify a power law over six orders of magnitude, and the advisor is
right that it gets blocked. The fit is exactly identified up to its residual,
which p26 says in a comment and which this script turns into a measurement.

THE FIX USES DATA ALREADY IN HAND. The 424 dong can be merged into k groups for
every k from 424 down to 1, many times over, which turns three points into the
whole curve with a confidence band. Doing it TWO ways answers a second question
for free:

    adjacency arm  merge only with a spatial neighbour  -> groups look like real
                                                          administrative areas
    random arm     merge with anyone at all             -> groups are spatially
                                                          meaningless

Both arms are the SAME stochastic agglomerative process -- "the currently
smallest group merges with a uniformly chosen (adjacent | any) group" -- with one
constraint toggled. That is what makes the gap between them interpretable: it is
how much of the scaling comes from spatial proximity rather than from aggregation
alone. Merging smallest-first is what keeps group sizes comparable; a pure
random-pair coalescent instead produces one giant group and a cloud of
singletons, which is a different regime and not what a partition into k areas
means.

THE THREE ANCHORS. A sweep that does not reproduce what is already known is
wrong, so the script asserts:

    k = 424   both arms must equal p26's dong rung
    k = 1     both arms must equal p26's city rung
    k = 25    the ADJACENCY arm's median must land near the real gu rung

The third is not bookkeeping. A real 자치구 is a contiguous clump of ~17 dong, so
if simulated contiguous clumps of 17 dong reproduce the measured gu value, the
merge algorithm is generating administratively realistic geography -- and that
belongs in the paper, not just in the test suite.

WHY EVERY STATISTIC IS RECOMPUTED AT EVERY k. p32 found that the mutual-
information ladder implies a different exponent from the assortativity ladder.
If the exponent depends on which statistic is scaled, then "the" scaling exponent
does not exist, and that is a second and independent reason the three-point
extrapolation cannot be defended -- one only the sweep can show.

GEOMETRY WITHOUT GEOPANDAS. Adjacency comes from the 행정동 boundary GeoJSON by
shared boundary vertices, using the ring machinery already in dl_covariates.py.
scipy, networkx and geopandas are all deliberately absent from requirements.txt;
a 424-node planar graph does not justify adding one.

    python eda/p34_ksweep.py [--months 202001,202012] [--reps 200]
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calendar_kr as K
from common import AGES
from p26_matrix import cells
from p32_pmix import excess_stats
from paths import DATA_ROOT, FIG, RESULTS, ROOT, require

SEOUL_LO, SEOUL_HI = 1101000, 1125999
BOUNDARY = DATA_ROOT / "raw" / "covariates" / "hjd_20200101.geojson"
# The product's code space is frozen at 2020 (p22), so one vintage is enough.
# 통계청 splits 오류2동 differently from the product: the product has never had
# 항동, and calls 오류2동 1117068 where 통계청 says 1117073. p21 folds 항동's
# population into 오류2동; adjacency is folded the same way, or the join silently
# drops a dong. See p20c_covariates.py:16-41.
CODE_REMAP = {1117073: 1117068}
CODE_FOLD = {1117074: 1117068}
VERTEX_ROUND = 6                      # ~0.1 m at Seoul's latitude
SHARED_VERTICES = 2                   # one shared point is a corner, not an edge
SEED = 20260819
ARM_SEED = {"adjacency": 0, "random": 1}

out = {}

COMMITTED_MONTHS = [202001, 202012, 202312, 202402, 202512, 202606]
CANONICAL_REPS = 200


def output_tag(panel, reps, yms):
    """Only the canonical configuration may write the canonical filename.

    A --panel run writes its OWN files: results_p34.json is what p31's gate and
    p36's independent recompute read, and both index it by ym across all six
    months of the WE sweep, so a W-only run landing on that name would replace
    what they check with another panel's numbers and still look like a
    successful run.

    Tagging the panel alone is not enough, and p38 proved it on 2026-08-20: a
    27-minute 79-month run was destroyed by a `--reps 20 --months 202001` smoke
    test. Both were panel WE, so the panel tag was empty for both, both wrote
    results_p38.json, and the cheap one landed last. Nothing failed at the time
    -- the loss surfaced only because the gate was checking numbers the file no
    longer contained. Two runs of the same panel at different SCALES must not
    share a path either, so a run at any other scale is named for what it is.
    """
    tag = "" if panel == "WE" else f"_{panel}"
    if reps != CANONICAL_REPS or yms != COMMITTED_MONTHS:
        tag += f"_r{reps}_m{len(yms)}"
    return tag



# ------------------------------------------------------------------- adjacency
def seoul_rings(path):
    """dong code -> list of exterior rings, with the two code traps applied."""
    g = json.load(open(path, encoding="utf-8"))
    rings = defaultdict(list)
    for f in g["features"]:
        try:
            code = int(f["properties"]["adm_cd"])
        except (KeyError, TypeError, ValueError):
            continue
        code = CODE_REMAP.get(code, code)
        code = CODE_FOLD.get(code, code)
        if not (SEOUL_LO <= code <= SEOUL_HI):
            continue
        geom = f["geometry"]
        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for p in polys:
            rings[code].append(p[0])          # exterior ring; holes touch nobody
    return rings


def build_adjacency(rings, codes):
    """Two dong are neighbours when their boundaries share at least two vertices.

    A single shared vertex is a corner touch, which is not a shared border, and
    admitting it would connect diagonal neighbours across a crossroads.
    """
    at = defaultdict(set)
    for code, rs in rings.items():
        for r in rs:
            for x, y in r:
                at[(round(x, VERTEX_ROUND), round(y, VERTEX_ROUND))].add(code)
    shared = defaultdict(int)
    for owners in at.values():
        if len(owners) < 2:
            continue
        o = sorted(owners)
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                shared[(o[i], o[j])] += 1
    idx = {c: i for i, c in enumerate(codes)}
    adj = [set() for _ in codes]
    for (a, b), n in shared.items():
        if n >= SHARED_VERTICES and a in idx and b in idx:
            adj[idx[a]].add(idx[b])
            adj[idx[b]].add(idx[a])
    return adj


def connected(adj, nodes=None):
    nodes = set(range(len(adj))) if nodes is None else set(nodes)
    if not nodes:
        return True
    start = next(iter(nodes))
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in nodes and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen == nodes


# ------------------------------------------------------------------- the sweep
def base_cube(ym, panel, imp=1.5, dows=None):
    """X[dong, cell, age] where cell is (arrival hour, day of week).

    Built to p26_matrix.matrices' definition exactly -- each (dow, hour, place)
    cell divided by that weekday's occurrences so every cell is an average day --
    because the k=424 anchor compares against p26's stored dong rung.
    """
    dows = list(range(1, 8)) if dows is None else list(dows)
    nd = {d: K.cell_exposure(ym, d)["n_days"] for d in dows}
    d = cells(ym)
    d = d if panel == "WE" else d[d.dest_attr == panel]
    d = d[d.dest_attr != "H"]
    d = d[d.dow_n.isin(dows)]
    d = d[(d.d_dong >= SEOUL_LO) & (d.d_dong <= SEOUL_HI)]
    d = d.assign(n=(d.v_obs + imp * d.n_masked) / d.dow_n.map(nd))
    codes = np.array(sorted(d.d_dong.unique()))
    ci = {c: i for i, c in enumerate(codes)}
    hi = {h: i for i, h in enumerate(sorted(d.arr_hour.unique()))}
    di = {w: i for i, w in enumerate(dows)}
    ai = {a: i for i, a in enumerate(AGES)}
    X = np.zeros((len(codes), len(hi) * len(di), len(AGES)))
    np.add.at(X, (d.d_dong.map(ci).to_numpy(),
                  d.arr_hour.map(hi).to_numpy() * len(di)
                  + d.dow_n.map(di).to_numpy(),
                  d.age.map(ai).to_numpy()), d.n.to_numpy())
    return codes, X, len(dows)


def contrib(Y):
    """sum_c outer(n_c, n_c) / n_c.  for one group's cells."""
    tot = Y.sum(1)
    m = tot > 0
    if not m.any():
        return np.zeros((Y.shape[1], Y.shape[1]))
    Z = Y[m]
    return (Z / tot[m][:, None]).T @ Z


def sweep(X, adj, ndows, pop, rng, use_adjacency, record):
    """Agglomerate 424 -> 1, recording the statistics at every k.

    A is maintained incrementally: a merge only changes the two groups involved,
    so each step costs one 168x16 pass rather than a full rebuild. That is what
    makes 200 replicates of a 424-step sweep cheap enough to be routine.
    """
    n0 = X.shape[0]
    Y = X.copy()
    alive = np.ones(n0, bool)
    size = np.ones(n0, int)
    nbr = [set(s) for s in adj] if use_adjacency else None
    A = np.zeros((X.shape[2], X.shape[2]))
    for i in range(n0):
        A += contrib(Y[i])
    rows = []
    record(rows, A / ndows, n0, pop)
    for k in range(n0 - 1, 0, -1):
        live = np.flatnonzero(alive)
        u = live[np.argmin(size[live])]
        if use_adjacency:
            cand = [v for v in nbr[u] if alive[v]]
            if not cand:                       # only if the graph disconnects
                cand = [v for v in live if v != u]
        else:
            cand = live[live != u]
        v = cand[rng.integers(len(cand))] if len(cand) else None
        if v is None:
            break
        assert v != u and alive[v], "a group cannot absorb itself"
        A -= contrib(Y[u])
        A -= contrib(Y[v])
        Y[u] += Y[v]
        A += contrib(Y[u])
        alive[v] = False
        size[u] += size[v]
        if use_adjacency:
            nbr[u] |= nbr[v]
            nbr[u].discard(u)
            nbr[u].discard(v)
            for w in nbr[v]:
                # w == u must be skipped, or u lands in its own neighbour set
                # and can be drawn as its own merge partner -- which doubles
                # Y[u] and kills the group. The k=1 anchor is what caught it.
                if alive[w] and w != u:
                    nbr[w].discard(v)
                    nbr[w].add(u)
        record(rows, A / ndows, k, pop)
    return rows


def stat_row(A, n_loc, pop):
    """Everything measured at one resolution."""
    C = A / pop[:, None]
    w = np.linalg.eigvals(C)
    ex = excess_stats(A)
    return dict(n_loc=int(n_loc), assortativity=ex["assortativity"],
                nmi=ex["nmi"], mi_bits=ex["mi_bits"], half_l1=ex["half_l1"],
                cramers_v=ex["cramers_v"],
                lead_eigenvalue=float(max(w.real)),
                persons_per_loc=float(pop.sum() / n_loc))


def local_slope(n, r, window=0.45):
    """d log r / d log n estimated in a sliding window, rather than one exponent.

    If the curve bends, a single exponent is not identified and no extrapolation
    beyond the observed range is licensed -- which is the whole point.
    """
    ln, lr = np.log(np.asarray(n, float)), np.log(np.asarray(r, float))
    o = np.argsort(ln)
    ln, lr = ln[o], lr[o]
    res = []
    for i, x in enumerate(ln):
        m = np.abs(ln - x) <= window
        if m.sum() >= 3:
            b = np.polyfit(ln[m], lr[m], 1)[0]
            res.append((float(np.exp(x)), float(b), int(m.sum())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="202001,202012")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--panel", default="WE")
    args = ap.parse_args()
    yms = [int(x) for x in args.months.split(",")]
    tag = output_tag(args.panel, args.reps, yms)
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    ladder = {(x["ym"], x["panel"], x["level"]): x for x in p26["ladder"]}

    # ------------------------------------------------------- 34.1 the geography
    print("=== 34.1 dong adjacency from the 2020 boundary file ===")
    require(BOUNDARY, "행정동 boundaries (run eda/dl_covariates.py)")
    codes0, X0, ndows = base_cube(yms[0], args.panel)
    codes = list(codes0)
    print(f"  {len(codes)} Seoul dong appear as destinations in {yms[0]}")
    rings = seoul_rings(BOUNDARY)
    print(f"  {len(rings)} dong in the boundary file after the code remap "
          f"({CODE_REMAP} and folding {list(CODE_FOLD)})")
    missing = [c for c in codes if c not in rings]
    assert not missing, f"no geometry for {missing}"
    adj = build_adjacency(rings, codes)
    deg = np.array([len(a) for a in adj])
    gu = np.array([c // 1000 for c in codes])
    gu_ok = all(connected(adj, np.flatnonzero(gu == g)) for g in np.unique(gu))
    checks = dict(n_nodes=len(codes), connected=bool(connected(adj)),
                  min_degree=int(deg.min()), mean_degree=float(deg.mean()),
                  max_degree=int(deg.max()), every_gu_connected=bool(gu_ok))
    for k, v in checks.items():
        print(f"    {k:<20}{v}")
    assert checks["n_nodes"] == 424, "the product has 424 Seoul dong"
    assert checks["connected"], "Seoul is not an archipelago"
    assert checks["min_degree"] >= 1, "an isolated dong means the join failed"
    assert 3.5 <= checks["mean_degree"] <= 7.0, "implausible for a planar map"
    assert checks["every_gu_connected"], "a 자치구 must be contiguous"
    out["adjacency"] = dict(checks, edges=int(deg.sum() // 2),
                            degree_hist=np.bincount(deg).tolist())
    print("  all five self-gates pass")

    # -------------------------------------------------------- 34.2 the sweeps
    sweeps, anchors, slopes = {}, {}, {}
    for ym in yms:
        pop = pops[ym]
        codes_m, X, ndows = (codes0, X0, ndows) if ym == yms[0] \
            else base_cube(ym, args.panel)
        assert list(codes_m) == codes, "dong set moved between months"
        print(f"\n=== 34.2 {ym} {args.panel}: sweeping 424 -> 1, "
              f"{args.reps} replicates per arm ===")
        for arm, use_adj in (("adjacency", True), ("random", False)):
            # NOT hash(arm): Python randomises string hashing per process, so
            # that seed changes between runs and the sweep stops being
            # reproducible. Caught because the two 2020 months moved in the
            # third decimal when the run was extended to six months.
            rng = np.random.default_rng([SEED, ym, ARM_SEED[arm]])
            acc = defaultdict(list)
            for rep in range(args.reps):
                rows = sweep(X, adj, ndows, pop, rng, use_adj,
                             lambda rs, A, k, p: rs.append(stat_row(A, k, p)))
                for r in rows:
                    acc[r["n_loc"]].append(r)
                if (rep + 1) % 50 == 0:
                    print(f"    {arm}: {rep + 1}/{args.reps}", flush=True)
            band = []
            for n_loc in sorted(acc):
                g = acc[n_loc]
                e = {}
                for s in ("assortativity", "nmi", "mi_bits", "half_l1",
                          "cramers_v", "lead_eigenvalue"):
                    v = np.array([x[s] for x in g], float)
                    e[s] = dict(median=float(np.median(v)),
                                lo=float(np.percentile(v, 2.5)),
                                hi=float(np.percentile(v, 97.5)))
                band.append(dict(n_loc=n_loc, n_rep=len(g),
                                 persons_per_loc=g[0]["persons_per_loc"], **e))
            sweeps[f"{ym}|{args.panel}|{arm}"] = band
            by_n = {b["n_loc"]: b for b in band}
            print(f"    {arm}: n=424 assort {by_n[424]['assortativity']['median']:.5f}"
                  f" | n=25 {by_n[25]['assortativity']['median']:.5f} "
                  f"[{by_n[25]['assortativity']['lo']:.5f}, "
                  f"{by_n[25]['assortativity']['hi']:.5f}]"
                  f" | n=1 {by_n[1]['assortativity']['median']:.5f}")

        # ------------------------------------------------- 34.3 the three anchors
        print(f"\n=== 34.3 {ym}: does the sweep reproduce what p26 measured? ===")
        a_ok = {}
        for arm in ("adjacency", "random"):
            by_n = {b["n_loc"]: b for b in sweeps[f"{ym}|{args.panel}|{arm}"]}
            for lev, n_loc in (("dong", 424), ("city", 1)):
                got = by_n[n_loc]["assortativity"]["median"]
                want = ladder[(ym, args.panel, lev)]["assortativity"]
                ok = abs(got - want) < 1e-9
                a_ok[f"{arm}|{lev}"] = dict(got=got, want=want, ok=bool(ok))
                print(f"    {arm:<10}{lev:<6}n={n_loc:<4}{got:.6f} vs p26 "
                      f"{want:.6f}  {'exact' if ok else 'MISMATCH'}")
                assert ok, f"{arm} {lev} does not reproduce p26"
        by_n = {b["n_loc"]: b for b in sweeps[f"{ym}|{args.panel}|adjacency"]}
        want = ladder[(ym, args.panel, "gu")]["assortativity"]
        g25 = by_n[25]["assortativity"]
        inside = g25["lo"] <= want <= g25["hi"]
        a_ok["adjacency|gu25"] = dict(got=g25["median"], want=want,
                                      lo=g25["lo"], hi=g25["hi"],
                                      inside_band=bool(inside),
                                      ratio=float(g25["median"] / want))
        print(f"    adjacency k=25 median {g25['median']:.5f} "
              f"[{g25['lo']:.5f}, {g25['hi']:.5f}] vs the REAL 자치구 rung "
              f"{want:.5f} -- {'inside the band' if inside else 'outside'}"
              f" (ratio {g25['median'] / want:.2f})")
        rnd25 = {b["n_loc"]: b for b in
                 sweeps[f"{ym}|{args.panel}|random"]}[25]["assortativity"]
        print(f"    for contrast the random arm at k=25 gives "
              f"{rnd25['median']:.5f} ({rnd25['median'] / want:.2f}x the real gu)")
        anchors[str(ym)] = a_ok

        # ------------------------------------------ 34.4 is one exponent enough?
        print(f"\n=== 34.4 {ym}: the exponent is not a constant ===")
        sl = {}
        for arm in ("adjacency", "random"):
            band = sweeps[f"{ym}|{args.panel}|{arm}"]
            n = [b["n_loc"] for b in band]
            for st in ("assortativity", "nmi"):
                r = [b[st]["median"] for b in band]
                good = [(a, b) for a, b in zip(n, r) if a > 0 and b > 0]
                ls = local_slope([a for a, _ in good], [b for _, b in good])
                sl[f"{arm}|{st}"] = [dict(n=a, slope=b, support=c) for a, b, c in ls]
                lo = [b for a, b, _ in ls if a <= 30]
                hi = [b for a, b, _ in ls if a >= 200]
                # the same three points p26 fits, taken off this same curve, so
                # the comparison is exponent-to-exponent and not method-to-method
                pts = [[b for b in band if b["n_loc"] == m][0][st]["median"]
                       for m in (1, 25, 424)]
                three = float(np.polyfit(np.log([1.0, 25.0, 424.0]),
                                         np.log(np.maximum(pts, 1e-12)), 1)[0])
                print(f"    {arm:<10}{st:<15}three-point exponent {three:.3f} | "
                      f"local slope near n=25 {np.median(lo):.3f}, near n=424 "
                      f"{np.median(hi):.3f}")
                sl[f"{arm}|{st}|three_point"] = float(three)
        slopes[str(ym)] = sl

    out["sweep"] = sweeps
    out["anchors"] = anchors
    out["local_slope"] = slopes

    # ------------------------------------------- 34.5 what the extrapolation cost
    print("\n=== 34.5 what this does to the published extrapolation ===")
    pubs = {(x["ym"], x["panel"]): x for x in p26["resolution_prediction"]}
    poly = {}
    for ym in yms:
        band = sweeps[f"{ym}|{args.panel}|adjacency"]
        ls = out["local_slope"][str(ym)]["adjacency|assortativity"]
        near = [x["slope"] for x in ls if x["n"] >= 200]
        lo_s, hi_s = float(np.min(near)), float(np.max(near))
        r424 = [b for b in band if b["n_loc"] == 424][0]["assortativity"]["median"]
        pred = sorted(r424 * (1831.0 / 424.0) ** s for s in (lo_s, hi_s))
        pub = pubs.get((ym, args.panel))
        poly[str(ym)] = dict(
            slope_lo=lo_s, slope_hi=hi_s, r_dong=r424,
            r_polygon_lo=pred[0], r_polygon_hi=pred[1],
            published_point=(pub["r_pred_polygon"] if pub else None),
            published_slope=(pub["slope"] if pub else None))
        print(f"  {ym}: local slope over n in [200, 424] runs {lo_s:.3f} to "
              f"{hi_s:.3f}" + (f", against the three-point fit's {pub['slope']:.3f}"
                               if pub else ""))
        print(f"  {'':<8}1,831-polygon assortativity becomes [{pred[0]:.5f}, "
              f"{pred[1]:.5f}]" + (f", not the point {pub['r_pred_polygon']:.5f}"
                                   if pub else ""))
    out["polygon_prediction_interval"] = poly
    print("\n  and the 6-10 million location figure in p27 rests on extrapolating")
    print("  four orders of magnitude past the largest n this curve reaches;")
    print("  it is retracted rather than re-estimated.")
    out["extrapolation_support"] = dict(n_min=1, n_max=424,
                                        log10_span=float(math.log10(424)),
                                        published_n_loc_target=6.3e6,
                                        log10_span_required=float(math.log10(6.3e6)))

    # ------------------------------------------------------------------ figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ym = yms[-1]
    pub = pubs.get((ym, args.panel))
    for arm, col in (("adjacency", "#1f77b4"), ("random", "#d62728")):
        band = sweeps[f"{ym}|{args.panel}|{arm}"]
        n = np.array([b["n_loc"] for b in band], float)
        m = np.array([b["assortativity"]["median"] for b in band])
        lo = np.array([b["assortativity"]["lo"] for b in band])
        hi = np.array([b["assortativity"]["hi"] for b in band])
        k = m > 0
        axes[0].plot(n[k], m[k], color=col, lw=1.4, label=f"{arm} merge")
        axes[0].fill_between(n[k], lo[k], hi[k], color=col, alpha=.22, lw=0)
    for lev, n_loc in (("city", 1), ("gu", 25), ("dong", 424)):
        axes[0].plot([n_loc], [ladder[(ym, args.panel, lev)]["assortativity"]],
                     "k*", ms=13, zorder=5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of locations")
    axes[0].set_ylabel("assortativity")
    axes[0].set_title(f"{ym} {args.panel}: the whole curve, with the three\n"
                      "published points (stars) on top", fontsize=10)
    axes[0].axvspan(424, 1e7, color="0.85", zorder=0)
    axes[0].text(700, axes[0].get_ylim()[1] * .5,
                 "unsupported by these data", fontsize=8, rotation=90)
    axes[0].set_xlim(0.8, 1e4)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3, which="both")
    for arm, col in (("adjacency", "#1f77b4"), ("random", "#d62728")):
        ls = out["local_slope"][str(ym)][f"{arm}|assortativity"]
        axes[1].plot([x["n"] for x in ls], [x["slope"] for x in ls],
                     color=col, lw=1.4, label=f"{arm} merge")
    if pub:
        axes[1].axhline(pub["slope"], color="k", ls="--", lw=1,
                        label=f"three-point fit ({pub['slope']:.3f})")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of locations")
    axes[1].set_ylabel("local d log r / d log n")
    axes[1].set_title("The exponent is not a constant", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / f"p34_ksweep{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(out["adjacency"]["degree_hist"])),
           out["adjacency"]["degree_hist"], color="#4c72b0")
    ax.set_xlabel("number of neighbouring dong")
    ax.set_ylabel("dong")
    ax.set_title(f"Adjacency graph: {out['adjacency']['edges']} edges, "
                 f"mean degree {out['adjacency']['mean_degree']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"p34_adjacency{tag}.png", dpi=150)
    plt.close(fig)

    with open(f"{RESULTS}/results_p34{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {RESULTS}/results_p34{tag}.json, "
          f"{FIG}/p34_ksweep{tag}.png, {FIG}/p34_adjacency{tag}.png")


if __name__ == "__main__":
    main()
