#!/usr/bin/env python
"""Phase 38 — the resolution sweep on all 79 months.

WHY NOW. p34_ksweep.py answered the advisor's point 4 -- the scaling exponent is
not a constant, so p27's extrapolation to six million locations is retracted --
on six months. p37_timeseries.py then re-ran the matrices and the ladder on all
79 months and found the published six are NOT representative: their upper edge
is 21% below the 79-month maximum and 202012 is the lowest month of all 79. The
sweep was left at six because it is a Monte Carlo (200 replicates x 424 merge
levels x 2 arms per month) rather than one pass per month, and the letter of
2026-08-19 carries an open question in its place: does the low-biased six-month
window change what the sweep concluded? It costs about half an hour to answer
instead of arguing, so this script answers it.

SAME ESTIMATOR, LITERALLY. Nothing here is re-implemented. The adjacency graph,
the agglomerative sweep, the per-k statistics and the local-slope estimator are
imported from p34_ksweep and called; the denominator comes from
p37_timeseries.population. A second implementation would be a different object
measured on more months, which answers a different question. p36_recompute.py is
where independence lives.

WHAT IS NOT REWRITTEN. results_p34.json is left alone. p31's citation gate reads
it, p36 recomputes its arithmetic, and determinism_check.sh runs it twice;
rewriting it with 79 months would invalidate that chain to add rows. Instead the
six published months are re-swept here and asserted BIT FOR BIT against it --
the rng is seeded per (month, arm), so extending the month list cannot move a
month that was already run, and if it does, that is a finding and not a rounding
difference. The anchor runs on the first published month reached, so a break
surfaces in the first minute rather than after the whole hour.

THREE ANCHOR LAYERS, all of them cheap:

    p34   the six published months must come back with zero deviation
    p37   k=424 and k=1 must equal the measured dong and city rungs, every month
    p26   the six published rungs p37 reproduces are the published ones

STORAGE. The full 424-level band with confidence intervals for 79 months x 2
arms is ~70 MB of JSON, most of it never read, so the band is stored on a log
ladder (REPORT_K). But a summary nobody can recompute is a number to be taken on
trust, and the two claims that need the whole curve -- the local slope near the
dong scale, and where the adjacency/random gap peaks -- would be exactly that.
So the median curve itself is kept at every one of the 424 levels for the two
statistics those claims are made on, medians only, which costs about 2 MB and
makes every derived number in this file recomputable from it. The full band with
its 95% intervals, for the six published months, is already in results_p34.json.

    python eda/p38_ksweep79.py                      # every month on disk
    python eda/p38_ksweep79.py --months 202001,202012 --reps 50
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import months_on_disk
from p34_ksweep import (ARM_SEED, BOUNDARY, SEED, base_cube, build_adjacency,
                        connected, local_slope, seoul_rings, stat_row, sweep)
from p37_timeseries import population
from paths import DERIVED, FIG, RESULTS, ROOT, require

ARMS = ("adjacency", "random")
# p34's order, so the two files can be compared field by field
STATS = ("assortativity", "nmi", "mi_bits", "half_l1", "cramers_v",
         "lead_eigenvalue")
SLOPE_STATS = ("assortativity", "nmi")
PUBLISHED_MONTHS = [202001, 202012, 202312, 202402, 202512, 202606]
# A log ladder plus the three rungs the paper names (424 dong, 25 gu, 1 city)
# and 17, the mean number of dong in a real 자치구.
REPORT_K = (1, 2, 3, 5, 8, 12, 17, 25, 35, 50, 75, 100, 150, 200, 300, 424)
# The two windows the memo quotes its exponents from: "near the gu scale" and
# "near the dong scale". Fixed here so every month is summarised the same way.
LOW_N, HIGH_N = 30, 200
out = {}


def cube_for(ym, panel, tries=6, wait=20):
    """base_cube, allowing for the fact that the data live on a USB drive.

    The sweep reads the drive once per month for half an hour, so one dropout
    costs the whole run: on 2026-08-20 the drive vanished at month 15 of 79 and
    took 27 minutes of Monte Carlo with it. A drive that is really gone is still
    fatal -- `require` has already checked it once and the loop gives up and
    raises -- but it is no longer fatal on the first attempt.
    """
    for attempt in range(1, tries + 1):
        try:
            return base_cube(ym, panel)
        except Exception as exc:                     # duckdb IOException, OSError
            if attempt == tries or not any(
                    w in str(exc) for w in ("No files found", "No such file",
                                            "I/O error", "IO Error",
                                            "Input/output error")):
                raise
            print(f"    {ym}: {type(exc).__name__} reading the data root "
                  f"(attempt {attempt}/{tries}); is the drive still mounted? "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)


def band_from(acc):
    """p34_ksweep.main's aggregation, factored out so both files agree by
    construction rather than by inspection."""
    band = []
    for n_loc in sorted(acc):
        g = acc[n_loc]
        e = {}
        for s in STATS:
            v = np.array([x[s] for x in g], float)
            e[s] = dict(median=float(np.median(v)),
                        lo=float(np.percentile(v, 2.5)),
                        hi=float(np.percentile(v, 97.5)))
        band.append(dict(n_loc=n_loc, n_rep=len(g),
                         persons_per_loc=g[0]["persons_per_loc"], **e))
    return band


def sweep_month(ym, X, adj, ndows, pop, reps):
    """Both arms of one month. The rng is seeded from (SEED, ym, arm) and from
    nothing else, which is what makes a month's result independent of which
    other months the run happens to include."""
    bands = {}
    for arm in ARMS:
        rng = np.random.default_rng([SEED, ym, ARM_SEED[arm]])
        acc = defaultdict(list)
        for _ in range(reps):
            rows = sweep(X, adj, ndows, pop, rng, arm == "adjacency",
                         lambda rs, A, k, p: rs.append(stat_row(A, k, p)))
            for r in rows:
                acc[r["n_loc"]].append(r)
        bands[arm] = band_from(acc)
    return bands


def deviation_from_p34(ym, bands, p34, panel="WE"):
    """Every number p34 published for this month, compared to what was just run.

    Not a tolerance check: the same code, the same data and the same seed must
    give the same float. A non-zero deviation means the sweep is not the object
    the paper describes, whatever the size of the number.
    """
    dev, n = 0.0, 0
    for arm in ARMS:
        want = {r["n_loc"]: r for r in p34["sweep"][f"{ym}|{panel}|{arm}"]}
        for row in bands[arm]:
            w = want[row["n_loc"]]
            for s in STATS:
                for q in ("median", "lo", "hi"):
                    dev = max(dev, abs(row[s][q] - w[s][q]))
                    n += 1
            dev = max(dev, abs(row["persons_per_loc"] - w["persons_per_loc"]))
            n += 1
    return dev, n


def slope_summary(band, stat):
    """The local slope curve, reduced to what the paper quotes."""
    n = [b["n_loc"] for b in band]
    r = [b[stat]["median"] for b in band]
    good = [(a, b) for a, b in zip(n, r) if a > 0 and b > 0]
    ls = local_slope([a for a, _ in good], [b for _, b in good])
    # local_slope reports n as exp(log n), so its n are 199.99999999999997-style
    # floats: `a >= 200` actually selects n >= 201, and a REPORT_K lookup keyed
    # on the raw float finds 3 of the 16 rungs. p34's published slopes are
    # computed with exactly this rule, so the rule stays and the integer window
    # it really selects is recorded next to the numbers -- otherwise nothing
    # here can be recomputed from the stored curve, which is the whole point of
    # storing it. Rounding is for the KEYS only; no window moves.
    at = {int(round(x[0])): x[1] for x in ls}
    low = [b for a, b, _ in ls if a <= LOW_N]
    high = [b for a, b, _ in ls if a >= HIGH_N]
    low_n = [int(round(a)) for a, _, _ in ls if a <= LOW_N]
    high_n = [int(round(a)) for a, _, _ in ls if a >= HIGH_N]
    pts = [[b for b in band if b["n_loc"] == m][0][stat]["median"]
           for m in (1, 25, 424)]
    three = float(np.polyfit(np.log([1.0, 25.0, 424.0]),
                             np.log(np.maximum(pts, 1e-12)), 1)[0])
    return ls, dict(
        three_point=three,
        low_median=float(np.median(low)), high_median=float(np.median(high)),
        near_lo=float(np.min(high)), near_hi=float(np.max(high)),
        low_window=[min(low_n), max(low_n)], n_low=len(low_n),
        high_window=[min(high_n), max(high_n)], n_high=len(high_n),
        curve_min=float(np.min([b for _, b, _ in ls])),
        curve_max=float(np.max([b for _, b, _ in ls])),
        slope_at={str(k): float(at[k]) for k in REPORT_K if k in at})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--panel", default="WE")
    args = ap.parse_args()
    tag = "" if args.panel == "WE" else f"_{args.panel}"
    yms = ([int(x) for x in args.months.split(",")] if args.months
           else months_on_disk())
    require(DERIVED / "regpop.parquet", "denominator")
    require(BOUNDARY, "행정동 boundaries (run eda/dl_covariates.py)")
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    # The p34 anchor has to compare like with like: a --panel W run reproduces
    # results_p34_W.json, not the WE file. deviation_from_p34 already looks up
    # f"{ym}|{panel}|{arm}", so reading results_p34.json regardless of panel
    # raises KeyError on a "|W|" key the WE file has never held. p34's panel
    # files were committed at 2641ded, so the anchor is available per panel.
    _p34_tag = "" if args.panel == "WE" else f"_{args.panel}"
    p34 = json.load(open(f"{ROOT}/eda/results_p34{_p34_tag}.json"))
    p37 = json.load(open(f"{ROOT}/eda/results_p37.json"))
    reg = pd.read_parquet(DERIVED / "regpop.parquet")
    fo = pd.read_parquet(DERIVED / "foreign.parquet")

    # The measured ladder for every month, from p37. p26 only has the six, and
    # the k=424 / k=1 anchors need one rung per month or 73 of the 79 months
    # would be swept with nothing to check them against.
    lad = {(r["ym"], r["level"]): r["assortativity"] for r in p37["rows"]
           if r["panel"] == args.panel and r["level"] in ("dong", "gu", "city")}
    pub_lad = {(x["ym"], x["level"]): x["assortativity"] for x in p26["ladder"]
               if x["panel"] == args.panel}
    print(f"=== 38.0 {len(yms)} months: {yms[0]}..{yms[-1]}, "
          f"{args.reps} replicates per arm, panel {args.panel} ===")
    tie = [(ym, lev) for (ym, lev) in pub_lad if (ym, lev) in lad]
    worst = max(abs(lad[k] - pub_lad[k]) for k in tie)
    assert worst == 0.0, "p37's ladder has drifted from p26's published rungs"
    print(f"  p37's ladder reproduces p26's {len(tie)} published rungs exactly "
          f"(max |diff| {worst:.1e}), so the anchors below are tied to the "
          f"published numbers")

    # --------------------------------------------------------- 38.1 the graph
    codes0, X0, ndows0 = cube_for(yms[0], args.panel)
    codes = list(codes0)
    rings = seoul_rings(BOUNDARY)
    adj = build_adjacency(rings, codes)
    deg = np.array([len(a) for a in adj])
    gu = np.array([c // 1000 for c in codes])
    graph = dict(n_nodes=len(codes), connected=bool(connected(adj)),
                 min_degree=int(deg.min()), mean_degree=float(deg.mean()),
                 max_degree=int(deg.max()),
                 every_gu_connected=bool(all(connected(adj, np.flatnonzero(gu == g))
                                             for g in np.unique(gu))),
                 edges=int(deg.sum() // 2),
                 degree_hist=np.bincount(deg).tolist())
    assert graph["n_nodes"] == 424 and graph["connected"]
    assert graph["min_degree"] >= 1 and graph["every_gu_connected"]
    # The geometry is one file and one vintage, so it must be the same graph p34
    # swept. If it is not, the two sets of curves are not comparable and the
    # bit-for-bit anchor below would fail for a reason nobody would find.
    same_graph = all(graph[k] == p34["adjacency"][k] for k in
                     ("n_nodes", "edges", "min_degree", "max_degree",
                      "degree_hist"))
    assert same_graph, "the adjacency graph differs from the one p34 swept"
    print(f"  adjacency graph identical to p34's: {graph['n_nodes']} nodes, "
          f"{graph['edges']} edges, mean degree {graph['mean_degree']:.2f}")
    out["adjacency"] = dict(graph, identical_to_p34=bool(same_graph))

    # --------------------------------------------------------- 38.2 the sweeps
    bands_k, anchors, slopes, poly, gap, curve = {}, {}, {}, {}, {}, {}
    p34_dev, p34_n, p34_months = 0.0, 0, []
    pubs = {(x["ym"], x["panel"]): x for x in p26["resolution_prediction"]}
    for i, ym in enumerate(yms, 1):
        pop = population(reg, fo, ym)
        if ym == yms[0]:
            X, ndows = X0, ndows0
        else:
            codes_m, X, ndows = cube_for(ym, args.panel)
            assert list(codes_m) == codes, f"{ym}: the dong set moved"
        bands = sweep_month(ym, X, adj, ndows, pop, args.reps)
        by = {arm: {b["n_loc"]: b for b in bands[arm]} for arm in ARMS}

        # --- anchor 1: the published months, bit for bit
        if ym in PUBLISHED_MONTHS and args.reps == 200:
            d, n = deviation_from_p34(ym, bands, p34, args.panel)
            assert d == 0.0, (
                f"{ym}: this sweep does not reproduce results_p34.json "
                f"(max deviation {d:.3e} over {n} numbers)")
            p34_dev, p34_n = max(p34_dev, d), p34_n + n
            p34_months.append(ym)

        # --- anchor 2: both ends are deterministic and must equal the measured
        # rungs -- with 424 groups nothing is merged, with 1 group there is only
        # one partition, so neither end is Monte Carlo at all
        a = {}
        for arm in ARMS:
            for lev, k in (("dong", 424), ("city", 1)):
                got, want = by[arm][k]["assortativity"]["median"], lad[(ym, lev)]
                assert abs(got - want) < 1e-9, f"{ym} {arm} {lev}: {got} vs {want}"
                a[f"{arm}|{lev}"] = dict(got=got, want=want)
        g25 = by["adjacency"][25]["assortativity"]
        r25 = by["random"][25]["assortativity"]
        want_gu = lad[(ym, "gu")]
        a["adjacency|gu25"] = dict(got=g25["median"], want=want_gu,
                                   lo=g25["lo"], hi=g25["hi"],
                                   inside_band=bool(g25["lo"] <= want_gu <= g25["hi"]),
                                   ratio=float(g25["median"] / want_gu))
        a["random|gu25"] = dict(got=r25["median"], want=want_gu,
                                ratio=float(r25["median"] / want_gu))
        anchors[str(ym)] = a

        # --- the exponent, on the full curve
        sl = {}
        for arm in ARMS:
            for st in SLOPE_STATS:
                ls, summ = slope_summary(bands[arm], st)
                sl[f"{arm}|{st}"] = summ
                if arm == "adjacency" and st == "assortativity":
                    near = [x[1] for x in ls if x[0] >= HIGH_N]
                    lo_s, hi_s = float(np.min(near)), float(np.max(near))
        slopes[str(ym)] = sl

        # --- what it does to the retracted extrapolation
        r424 = by["adjacency"][424]["assortativity"]["median"]
        pred = sorted(r424 * (1831.0 / 424.0) ** s for s in (lo_s, hi_s))
        pb = pubs.get((ym, args.panel))
        poly[str(ym)] = dict(slope_lo=lo_s, slope_hi=hi_s, r_dong=r424,
                             r_polygon_lo=pred[0], r_polygon_hi=pred[1],
                             published_point=(pb["r_pred_polygon"] if pb else None),
                             published_slope=(pb["slope"] if pb else None))

        # --- the arm gap. Both ends are 1 by construction, so the peak is
        # looked for strictly between them.
        rat = {k: by["adjacency"][k]["assortativity"]["median"]
                  / by["random"][k]["assortativity"]["median"]
               for k in by["adjacency"]
               if 1 < k < 424 and by["random"][k]["assortativity"]["median"] > 0}
        pk = max(rat, key=rat.get)
        gap[str(ym)] = dict(peak_k=int(pk), peak_ratio=float(rat[pk]),
                            ratio_at={str(k): float(rat[k]) for k in REPORT_K
                                      if k in rat})
        for arm in ARMS:
            key = f"{ym}|{args.panel}|{arm}"
            bands_k[key] = [b for b in bands[arm] if b["n_loc"] in REPORT_K]
            # Positional: element i is k = i + 1, asserted rather than assumed
            # because an off-by-one here would silently shift every slope
            # recomputed from this file by one merge level.
            assert [b["n_loc"] for b in bands[arm]] == list(range(1, 425))
            curve[key] = {st: [b[st]["median"] for b in bands[arm]]
                          for st in SLOPE_STATS}
        print(f"  [{i:>2}/{len(yms)}] {ym}  dong {r424:.5f}  "
              f"adj/real-gu {a['adjacency|gu25']['ratio']:.3f}  "
              f"rand/real-gu {a['random|gu25']['ratio']:.3f}  "
              f"peak k={pk} ({rat[pk]:.3f})  slope[{HIGH_N},424] "
              f"{lo_s:.3f}-{hi_s:.3f}"
              + ("  [p34 anchor exact]" if ym in p34_months else ""), flush=True)

    out.update(months=yms, reps=args.reps, panel=args.panel,
               report_k=list(REPORT_K), band=bands_k, curve=curve,
               anchors=anchors, local_slope=slopes,
               polygon_prediction_interval=poly, arm_gap=gap)
    out["p34_anchor"] = dict(months=p34_months, n_numbers=p34_n,
                             max_deviation=p34_dev)
    if p34_months:
        print(f"\n=== 38.3 anchor: {len(p34_months)} published months, "
              f"{p34_n:,} numbers, max deviation {p34_dev:.1e} ===")
        print("  extending the month list did not move a month that was "
              "already run, which is what the per-month seed is for")

    # ------------------------------------------- 38.4 the exponent, 79 months
    ymk = [str(y) for y in yms]
    print(f"\n=== 38.4 is the exponent a constant in ANY month? ===")
    exp_sum = {}
    for st in SLOPE_STATS:
        lows = [slopes[y][f"{arm}|{st}"]["low_median"] for y in ymk for arm in ARMS]
        highs = [slopes[y][f"{arm}|{st}"]["high_median"] for y in ymk for arm in ARMS]
        named = lows + highs
        three = [slopes[y][f"{arm}|{st}"]["three_point"] for y in ymk for arm in ARMS]
        bends = sum(1 for y in ymk for arm in ARMS
                    if slopes[y][f"{arm}|{st}"]["high_median"]
                    > slopes[y][f"{arm}|{st}"]["low_median"])
        exp_sum[st] = dict(named_lo=float(min(named)), named_hi=float(max(named)),
                           low_median=float(np.median(lows)),
                           high_median=float(np.median(highs)),
                           three_lo=float(min(three)), three_hi=float(max(three)),
                           n_bending_up=int(bends), n_curves=len(ymk) * len(ARMS))
        print(f"  {st:<15} local slope runs {min(named):.3f} to {max(named):.3f} "
              f"across {len(ymk)} months x 2 arms; the three-point fit gives "
              f"{min(three):.3f}-{max(three):.3f}")
        print(f"  {'':<15} the curve bends upward (n={HIGH_N}+ steeper than "
              f"n<={LOW_N}) in {bends} of {len(ymk) * len(ARMS)} curves")
    # the claim p31 checks six times, now checkable 79 times
    beats = sum(1 for y in ymk if poly[y]["published_slope"] is None
                or poly[y]["slope_lo"] > poly[y]["published_slope"])
    n_pub_sl = sum(1 for y in ymk if poly[y]["published_slope"] is not None)
    exp_sum["local_exceeds_three_point"] = dict(
        n_months=len(ymk),
        n_with_published_slope=n_pub_sl,
        n_local_above=int(sum(1 for y in ymk
                              if poly[y]["published_slope"] is not None
                              and poly[y]["slope_lo"] > poly[y]["published_slope"])))
    out["exponent"] = exp_sum

    # ------------------------------------------ 38.5 the arm gap, 79 months
    print("\n=== 38.5 how much of the scaling is spatial proximity? ===")
    pk = np.array([gap[y]["peak_k"] for y in ymk])
    pr = np.array([gap[y]["peak_ratio"] for y in ymk])
    r25 = np.array([gap[y]["ratio_at"]["25"] for y in ymk])
    adj_ratio = np.array([anchors[y]["adjacency|gu25"]["ratio"] for y in ymk])
    rnd_ratio = np.array([anchors[y]["random|gu25"]["ratio"] for y in ymk])
    inside = int(sum(anchors[y]["adjacency|gu25"]["inside_band"] for y in ymk))
    print(f"  the real 자치구 rung is inside the adjacency arm's 95% band in "
          f"{inside} of {len(ymk)} months (ratio {adj_ratio.min():.3f}-"
          f"{adj_ratio.max():.3f})")
    print(f"  the random arm falls short of the real gu rung by "
          f"{(1 - rnd_ratio.max()) * 100:.1f}% to {(1 - rnd_ratio.min()) * 100:.1f}%")
    print(f"  adjacency/random peaks at k={int(np.median(pk))} (median), "
          f"range {pk.min()}-{pk.max()}, peak ratio {pr.min():.3f}-{pr.max():.3f}")
    out["arm_gap_summary"] = dict(
        n_months=len(ymk), gu_inside_band=inside,
        adj_over_real_gu=[float(adj_ratio.min()), float(adj_ratio.max())],
        adj_over_real_gu_median=float(np.median(adj_ratio)),
        random_over_real_gu=[float(rnd_ratio.min()), float(rnd_ratio.max())],
        random_shortfall_pct=[float((1 - rnd_ratio.max()) * 100),
                              float((1 - rnd_ratio.min()) * 100)],
        peak_k=[int(pk.min()), int(pk.max())], peak_k_median=float(np.median(pk)),
        peak_k_in_17_35=int(((pk >= 17) & (pk <= 35)).sum()),
        peak_ratio=[float(pr.min()), float(pr.max())],
        peak_ratio_median=float(np.median(pr)),
        ratio_at_25=[float(r25.min()), float(r25.max())],
        ratio_at_25_median=float(np.median(r25)))

    # --------------------------- 38.6 was the six-month window a bad window?
    # p37 found it was, for the level of the dong-level assortativity. The
    # question this script exists to answer is whether it was also a bad window
    # for the things the SWEEP concluded, which are ratios and slopes rather
    # than levels.
    print("\n=== 38.6 the published six months against the full span ===")
    pub_i = [i for i, y in enumerate(yms) if y in PUBLISHED_MONTHS]
    win = {}
    quantities = dict(
        r_dong=np.array([poly[y]["r_dong"] for y in ymk]),
        slope_lo=np.array([poly[y]["slope_lo"] for y in ymk]),
        slope_hi=np.array([poly[y]["slope_hi"] for y in ymk]),
        adj_over_real_gu=adj_ratio, random_over_real_gu=rnd_ratio,
        arm_ratio_at_25=r25, peak_ratio=pr)
    for name, v in quantities.items():
        six = v[pub_i]
        # where the six sit in the 79: 0 = all of them below the full range's
        # midpoint of the ranks, 1 = at the top
        rank = [int((v <= x).sum()) for x in six]
        cov = float((six.max() - six.min()) / (v.max() - v.min())) \
            if v.max() > v.min() else float("nan")
        win[name] = dict(all_min=float(v.min()), all_max=float(v.max()),
                         all_median=float(np.median(v)),
                         six_min=float(six.min()), six_max=float(six.max()),
                         six_ranks=rank, span_covered=cov)
        print(f"  {name:<20} six {six.min():.4f}-{six.max():.4f}   "
              f"79 {v.min():.4f}-{v.max():.4f}   "
              f"the six cover {cov:.0%} of the full spread")
    out["published_window"] = win

    # --------------------------------------------------- 38.7 the retraction
    # None of this moves with the month: it is the span of n the curve covers
    # against the span the published extrapolation needed.
    out["extrapolation_support"] = dict(n_min=1, n_max=424,
                                        log10_span=float(math.log10(424)),
                                        published_n_loc_target=6.3e6,
                                        log10_span_required=float(math.log10(6.3e6)))
    print(f"\n=== 38.7 the retraction is unchanged: the curve supports "
          f"log10 span {math.log10(424):.2f}, the published extrapolation "
          f"needed {math.log10(6.3e6):.2f} ===")

    # -------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for ym in yms:
        b = bands_k[f"{ym}|{args.panel}|adjacency"]
        n = np.array([x["n_loc"] for x in b], float)
        r = np.array([x["assortativity"]["median"] for x in b])
        k = r > 0
        pub = ym in PUBLISHED_MONTHS
        ax.plot(n[k], r[k], "-", lw=1.6 if pub else 0.7,
                color="#A8434E" if pub else "0.72", zorder=3 if pub else 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvspan(424, 1e7, color="0.85", zorder=0)
    ax.text(700, ax.get_ylim()[1] * .4, "unsupported by these data",
            fontsize=7, rotation=90)
    ax.set_xlim(0.8, 1e4)
    ax.set_xlabel("number of locations")
    ax.set_ylabel("assortativity")
    ax.set_title(f"{len(yms)} monthly curves, adjacency arm\n"
                 "(red = the six published months)", fontsize=9)
    ax.grid(alpha=.3, which="both")

    ax = axes[1]
    lo = np.array([poly[y]["slope_lo"] for y in ymk])
    hi = np.array([poly[y]["slope_hi"] for y in ymk])
    tp = np.array([slopes[y]["adjacency|assortativity"]["three_point"] for y in ymk])
    x = np.arange(len(yms))
    ax.fill_between(x, lo, hi, color="#0E7C86", alpha=.25, lw=0,
                    label=f"local slope, n in [{HIGH_N}, 424]")
    ax.plot(x, tp, "-", color="#BC8034", lw=1.3, label="three-point fit")
    ax.plot(pub_i, tp[pub_i], "*", color="#A8434E", ms=11, zorder=5,
            label="published months")
    ax.set_xticks(range(0, len(yms), 6))
    ax.set_xticklabels([str(y) for y in yms][::6], rotation=90, fontsize=7)
    ax.set_ylabel("d log r / d log n")
    ax.set_title("the three-point exponent is below the local\n"
                 "slope near the dong scale, in every month", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    ax = axes[2]
    ax.plot(x, r25, "-", color="#0E7C86", lw=1.3, label="adjacency / random at k=25")
    ax.plot(x, adj_ratio, "-", color="#4C6E8A", lw=1.3,
            label="adjacency / the real gu rung")
    ax.plot(x, rnd_ratio, "-", color="#BC8034", lw=1.3,
            label="random / the real gu rung")
    ax.plot(pub_i, r25[pub_i], "*", color="#A8434E", ms=11, zorder=5)
    ax.axhline(1.0, color="0.4", lw=.8, ls="--")
    ax.set_xticks(range(0, len(yms), 6))
    ax.set_xticklabels([str(y) for y in yms][::6], rotation=90, fontsize=7)
    ax.set_ylabel("ratio")
    ax.set_title("space, not aggregation: the gap at the gu scale\n"
                 "and the two arms against the measured rung", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.suptitle(f"Phase 38 — the k-sweep on {len(yms)} months, "
                 f"{yms[0]}–{yms[-1]}, {args.reps} replicates per arm", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"p38_ksweep79{tag}.png", dpi=150)
    plt.close(fig)

    with open(f"{RESULTS}/results_p38{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {RESULTS}/results_p38{tag}.json, "
          f"{FIG}/p38_ksweep79{tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
