#!/usr/bin/env python
"""Phase 41 — the school-semester cycle, promoted to a result of its own.

WHY THIS EXISTS. The paper's headline is a null: dong-level co-arrival resolves
about 0.4% of the age entropy, the best rank-1 approximation of the passive
matrix is proportionate mixing itself, and IPF onto the survey margins closes
4% and 1% of the gap. The one reading a reviewer can give that costs us the
whole paper is "your instrument is broken, so of course it reads zero". The
semester cycle is the only evidence against that reading, and it has the shape
evidence has to have: **no school label of any kind enters the matrix** -- the
estimator sees destination dong, arrival hour, day of week, destination
attribute and age band, and nothing else -- yet the concentration statistic
rises in the Korean academic term and falls in the vacations, and the one year
it does not is 2020, the year schools did not open in March. p37 found this and
recorded it as section 4 of a memo with the cycle drawn as shading on the
ladder figure. The advisor's 8-21 letter, point 4, says that is the wrong
weight: it is not a face-validity footnote, it is the premise of the null, and
it needs its own Results subsection, its own figure, and statistics done
properly rather than a p-value quoted from a table.

WHAT THIS SCRIPT DOES NOT DO. It does not touch the parquet, does not rebuild a
single matrix, and does not re-implement the estimator. Everything here is a
statement about numbers already in results_p37.json, which is itself anchored
bit for bit to the six published months in results_p26.json. The only quantity
computed here that p37 did not already store is the per-band decomposition of
the assortativity numerator, and that is a decomposition of the published
statistic rather than a new one: Newman assortativity on the co-arrival matrix
is

    r = (tr e - sum_a r_a^2) / (1 - sum_a r_a^2),   e = A / A.sum(),  r_a = e_a.

so the band-wise terms c_a = (e_aa - r_a^2) / (1 - sum_b r_b^2) sum to r
exactly. Section 41.0 asserts that identity against p37's stored value on all
79 months and against p26's published value on the six, so the decomposition is
tied to the number in the paper rather than merely consistent with it.

results_p37.json is NOT rewritten. It is gated by p31_report_audit.py and
independently recomputed by p36_recompute.py, and adding rows to it would
invalidate that chain to save one file. This is what p37 did to p26 and what
p38 did to p34; p41 does the same and writes results_p41.json.

WHAT THE STATISTICS ARE FOR, AND WHAT THEY ARE NOT FOR. The primary reported
test is the sign test on the seven per-year term-minus-vacation differences,
and it is reported because the advisor asked for a number that separates "weak"
from "weak but real", not because it carries the argument. It cannot: with
seven years the smallest attainable one-sided p is 1/128 = 0.0078, so the test
has no resolution to spend. What carries the argument is structural and no
p-value expresses it -- the single exception lands exactly on the year schools
did not open, and (41.6) the 2020 anomaly is confined to the age bands that go
to school while the same year's adult seasonal contrast survives intact.

    python eda/p41_semester.py
    python eda/p41_semester.py --mc 500000     # more permutation draws
"""
import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import AGE_LABEL, AGES
from paths import FIG, ROOT

# The Korean academic year: two semesters, March-July and September-February,
# with the long vacation in July-August and the short one in January-February.
# Months 3-6 and 9-11 are unambiguously in term; 1, 2, 7 and 8 are unambiguously
# in vacation; December straddles the end of the second semester and is left out
# of BOTH sets rather than assigned to one. This is p37's convention, kept
# unchanged so the two files describe the same partition.
TERM = (3, 4, 5, 6, 9, 10, 11)
VAC = (1, 2, 7, 8)
CONTROL_YEAR = 2020

# Bands. 0-9 / 10-14 / 15-19 are compulsory schooling in Korea (elementary
# through high school); 20-24 is university, which also went fully remote in
# 2020. 25+ is the placebo channel: those bands do not follow an academic
# calendar, so whatever seasonality they carry is not schooling.
SCHOOL_IDX = [AGES.index(a) for a in (0, 10, 15)]
STUDENT_IDX = [AGES.index(a) for a in (0, 10, 15, 20)]
ADULT_IDX = [AGES.index(a) for a in AGES if a >= 25]
GROUPS = {"school_0_19": SCHOOL_IDX, "student_0_24": STUDENT_IDX,
          "adult_25plus": ADULT_IDX}

# The ten (panel, level, statistic) readings the same 79 months support. The
# headline is the first; the other nine exist so that "6 of 7 positive" can be
# checked for whether it is a property of the cycle or of one statistic.
VARIANTS = [("WE", "dong", "assortativity"),
            ("WE", "dong", "nmi"),
            ("WE", "dong", "half_l1"),
            ("WE", "dong", "cramers_v"),
            ("WE", "dong", "diag_excess_sum"),
            ("WE", "dong_holidayfree", "assortativity"),
            ("W", "dong", "assortativity"),
            ("E", "dong", "assortativity"),
            ("WE", "gu", "assortativity"),
            ("WE", "city", "assortativity")]

SEED = 20260821
out = {}


# ------------------------------------------------------------------ statistics
def contrast(vals_by_month):
    """median(term months) - median(vacation months), within one year.

    Medians rather than means because a single month can be moved a long way by
    a holiday cluster (설 연휴 lands in January or February and 추석 in September
    or October), and the point of the statistic is the level of the term, not
    its mean over a set that a moveable feast can reshape. p37 reported both;
    both are computed here and both are stored.
    """
    t = [v for m, v in vals_by_month.items() if m in TERM]
    v = [v for m, v in vals_by_month.items() if m in VAC]
    if not t or not v:
        return np.nan
    return float(np.median(t) - np.median(v))


def perm_null(vals_by_month):
    """Every within-year relabelling of the term/vacation split, exhaustively.

    The labelled months are held fixed and only WHICH of them is called "term"
    is permuted, keeping the two group sizes. December is not in either set and
    stays out, so the permutation is over C(11, 7) = 330 assignments in a full
    year and C(7, 4) = 35 in 2026. Exhaustive, so no seed and no Monte Carlo
    error; the seeded draw in 41.4b is only for the across-year statistic, whose
    null is the 7-fold convolution of these and has 330^6 x 35 support points.

    Permuting within the year is what respects the structure a pooled test
    destroys: the level moves by half its own range between 2020 and 2024, so a
    pooled shuffle would compare a 2024 term month against a 2020 vacation month
    and read the trend as a cycle.
    """
    months = [m for m in vals_by_month if m in TERM or m in VAC]
    x = np.array([vals_by_month[m] for m in months], float)
    nt = sum(1 for m in months if m in TERM)
    dist = []
    for c in itertools.combinations(range(len(months)), nt):
        cs = set(c)
        rest = [i for i in range(len(months)) if i not in cs]
        dist.append(np.median(x[list(c)]) - np.median(x[rest]))
    return np.array(dist, float)


def sign_test(diffs):
    """Exact binomial sign test, ties dropped. n = 7, so the floor is 1/128."""
    d = [x for x in diffs if x != 0]
    n, k = len(d), sum(1 for x in d if x > 0)
    one = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return dict(n=n, n_positive=k, n_dropped_ties=len(diffs) - n,
                p_one_sided=float(one), p_two_sided=float(min(1.0, 2 * one)),
                numerator=sum(comb(n, i) for i in range(k, n + 1)),
                denominator=2 ** n)


def poisson_binomial_tail(ps, k):
    """P(sum of independent Bernoulli(ps) >= k), exactly, by convolution.

    Needed because the permutation null of the per-year SIGN is not a fair coin:
    with 7 term months against 4 vacation months the median-difference statistic
    is not symmetric about zero, and the per-year P(perm > 0) computed from the
    330 relabellings runs from 0.36 to 0.62. The binomial sign test assumes 0.5;
    this is the same test without that assumption.
    """
    poly = np.array([1.0])
    for p in ps:
        poly = np.convolve(poly, [1 - p, p])
    return float(poly[k:].sum())


def by_year(series):
    """{year: {calendar month: value}} from {ym: value}."""
    d = defaultdict(dict)
    for ym, v in series.items():
        d[ym // 100][ym % 100] = v
    return dict(sorted(d.items()))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=200_000)
    args = ap.parse_args()

    p37 = json.load(open(f"{ROOT}/eda/results_p37.json"))
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    rows = p37["rows"]
    series = {(r["panel"], r["level"], s): {}
              for r in rows for s in ("assortativity", "nmi", "half_l1",
                                      "cramers_v", "diag_excess_sum")}
    for r in rows:
        for s in ("assortativity", "nmi", "half_l1", "cramers_v",
                  "diag_excess_sum"):
            series[(r["panel"], r["level"], s)][r["ym"]] = r[s]
    head = series[("WE", "dong", "assortativity")]
    yms = sorted(head)
    years = sorted({y // 100 for y in yms})
    print(f"=== 41.0 {len(yms)} months, {yms[0]}..{yms[-1]}, {len(years)} years ===")

    # --------------------------------------------------------- 41.0 the anchor
    # The band decomposition has to sum back to the published assortativity, on
    # every month against p37 and on the six published months against p26. If it
    # does not, this section is describing a different object than the paper's.
    bands, anch37, anch26 = {}, [], []
    for ym in yms:
        A = np.array(p37["matrices_we_dong"][str(ym)]["A"], float)
        e = A / A.sum()
        r = e.sum(1)
        denom = 1.0 - float((r * r).sum())
        c = (np.diag(e) - r * r) / denom
        bands[ym] = c
        got, want = float(c.sum()), head[ym]
        anch37.append(dict(ym=ym, rel=abs(got - want) / abs(want)))
        assert anch37[-1]["rel"] < 1e-12, f"{ym}: {got} vs p37 {want}"
        key = f"{ym}|WE|dong"
        if key in p26["matrices"]:
            w26 = p26["matrices"][key]["assortativity"]
            anch26.append(dict(ym=ym, rel=abs(got - w26) / abs(w26)))
            assert anch26[-1]["rel"] < 1e-12, f"{ym}: {got} vs p26 {w26}"
    print(f"  band decomposition sums to the stored assortativity in "
          f"{len(anch37)}/{len(yms)} months, max relative deviation "
          f"{max(a['rel'] for a in anch37):.2e}")
    print(f"  and to the PUBLISHED value in {len(anch26)} of the six p26 months, "
          f"max {max(a['rel'] for a in anch26):.2e}")
    out["anchor"] = dict(vs_p37=anch37, vs_p26=anch26,
                         max_rel_p37=max(a["rel"] for a in anch37),
                         max_rel_p26=max(a["rel"] for a in anch26))

    # ------------------------------------------------- 41.1 the calendar profile
    print("\n=== 41.1 the calendar-month profile (a guide, not the evidence) ===")
    cal = {}
    for m in range(1, 13):
        v = [head[ym] for ym in yms if ym % 100 == m]
        cal[m] = dict(n=len(v), median=float(np.median(v)),
                      lo=float(min(v)), hi=float(max(v)))
    order = sorted(cal, key=lambda m: -cal[m]["median"])
    print("  ranked by median: " + " ".join(
        f"{m}({cal[m]['median']:.4f})" for m in order))
    print(f"  March is rank {order.index(3) + 1}/12, September rank "
          f"{order.index(9) + 1}/12, February rank {order.index(2) + 1}/12 "
          f"(i.e. {12 - order.index(2)} from the bottom)")
    # Pooling across years is the weak part of this table, so the same question
    # is asked inside each year, where no pooling can manufacture a peak.
    march_rank, feb_rank = {}, {}
    for y, mv in by_year(head).items():
        o = sorted(mv, key=lambda m: -mv[m])
        march_rank[y] = dict(rank=o.index(3) + 1, n=len(o))
        feb_rank[y] = dict(rank_from_bottom=len(o) - o.index(2), n=len(o))
    n_march_top = sum(1 for y in years if march_rank[y]["rank"] == 1)
    print(f"  March is the top month of its OWN year in {n_march_top}/{len(years)} "
          f"years: " + " ".join(f"{y}:{march_rank[y]['rank']}/{march_rank[y]['n']}"
                                for y in years))
    out["calendar"] = dict(by_month=cal, rank_order=order,
                           march_rank_within_year=march_rank,
                           february_rank_within_year=feb_rank,
                           n_years_march_top=n_march_top)

    # ---------------------------------------------- 41.2 the per-year contrasts
    print("\n=== 41.2 term minus vacation, one number per year ===")
    yr = by_year(head)
    per_year = {}
    for y in years:
        mv = yr[y]
        t = [v for m, v in mv.items() if m in TERM]
        v = [x for m, x in mv.items() if m in VAC]
        per_year[y] = dict(
            n_months=len(mv), n_term=len(t), n_vac=len(v),
            median_term=float(np.median(t)), median_vac=float(np.median(v)),
            contrast_median=contrast(mv),
            contrast_mean=float(np.mean(t) - np.mean(v)),
            march=mv.get(3), february=mv.get(2),
            march_minus_february=(float(mv[3] - mv[2])
                                  if 3 in mv and 2 in mv else None))
        print(f"  {y}  term {per_year[y]['median_term']:.5f}  "
              f"vac {per_year[y]['median_vac']:.5f}  "
              f"contrast {per_year[y]['contrast_median']:+.5f}  "
              f"Mar-Feb {per_year[y]['march_minus_february']:+.5f}")
    diffs = [per_year[y]["contrast_median"] for y in years]
    out["per_year"] = per_year

    # ------------------------------------------------------- 41.3 the sign test
    print("\n=== 41.3 the sign test (primary, and it carries no load) ===")
    st = sign_test(diffs)
    st_mar = sign_test([per_year[y]["march_minus_february"] for y in years])
    print(f"  contrast positive in {st['n_positive']}/{st['n']} years; exact "
          f"one-sided p = {st['numerator']}/{st['denominator']} = "
          f"{st['p_one_sided']:.6f}, two-sided {st['p_two_sided']:.6f}")
    print(f"  March minus February positive in {st_mar['n_positive']}/"
          f"{st_mar['n']}; one-sided p = {st_mar['p_one_sided']:.6f}")
    print(f"  floor: with n = {st['n']} the smallest attainable one-sided p is "
          f"{1 / 2 ** st['n']:.6f}, so this test has no resolution to spend")
    out["sign_test"] = dict(contrast=st, march_minus_february=st_mar,
                            smallest_attainable_one_sided=1 / 2 ** st["n"])

    # ------------------------------------------------ 41.4 permutation, exactly
    print("\n=== 41.4a within-year permutation, exhaustive ===")
    nulls = {y: perm_null(yr[y]) for y in years}
    p_pos, p_year = [], {}
    for y in years:
        d = nulls[y]
        obs = diffs[years.index(y)]
        p_pos.append(float((d > 0).mean()))
        p_year[y] = dict(n_perm=int(d.size),
                         p_ge_observed=float((d >= obs - 1e-15).mean()),
                         rank=int((d < obs).sum() + 1),
                         null_median=float(np.median(d)),
                         null_lo=float(np.quantile(d, 0.025)),
                         null_hi=float(np.quantile(d, 0.975)),
                         p_perm_positive=p_pos[-1])
        print(f"  {y}: {d.size:>3} relabellings, observed ranks "
              f"{p_year[y]['rank']}/{d.size}, exact p = "
              f"{p_year[y]['p_ge_observed']:.5f}   P(perm > 0) = {p_pos[-1]:.4f}")
    k = st["n_positive"]
    p_count = poisson_binomial_tail(p_pos, k)
    print(f"  exact permutation P(at least {k} of {len(years)} years positive) "
          f"= {p_count:.6f}   [the binomial sign test's {st['p_one_sided']:.6f} "
          f"assumes P = 0.5; the relabellings say {min(p_pos):.2f}-{max(p_pos):.2f}]")

    print("\n=== 41.4b within-year permutation, the across-year magnitude ===")
    rng = np.random.default_rng(SEED)
    draws = np.zeros(args.mc)
    for y in years:
        d = nulls[y]
        draws += d[rng.integers(0, d.size, args.mc)]
    draws /= len(years)
    obs_mean = float(np.mean(diffs))
    p_mc = float((int((draws >= obs_mean - 1e-18).sum()) + 1) / (args.mc + 1))
    print(f"  statistic = mean over years of the contrast; observed "
          f"{obs_mean:+.6f}, null mean {draws.mean():+.2e}, sd {draws.std():.2e}")
    print(f"  seed {SEED}, {args.mc:,} draws, p = {p_mc:.6f} "
          f"(the floor is 1/{args.mc + 1})")
    print("  ! this p is not calibrated evidence: within-year relabelling treats "
          "adjacent months as exchangeable, and they are not")
    out["permutation"] = dict(
        per_year=p_year, exhaustive=True,
        p_count_at_least=p_count, count_observed=k,
        p_perm_positive=p_pos,
        mc=dict(seed=SEED, n_draws=int(args.mc), statistic="mean_over_years",
                observed=obs_mean, p=p_mc, null_mean=float(draws.mean()),
                null_sd=float(draws.std()), floor=1 / (args.mc + 1)))

    # ------------------------------------------- 41.5 is 6/7 a property of one
    #                                             statistic, or of the cycle
    print("\n=== 41.5 the same contrast under ten readings of the same months ===")
    rob = {}
    for panel, level, stat in VARIANTS:
        s = series[(panel, level, stat)]
        v = [contrast(mv) for _, mv in by_year(s).items()]
        i0 = years.index(CONTROL_YEAR)
        others = [x for j, x in enumerate(v) if j != i0]
        key = f"{panel}|{level}|{stat}"
        rob[key] = dict(
            per_year={str(y): v[j] for j, y in enumerate(years)},
            n_positive=sum(1 for x in v if x > 0), n=len(v),
            control=v[i0], others_min=min(others), others_max=max(others),
            control_rank=int(sorted(v).index(v[i0]) + 1),
            control_over_others_min=float(v[i0] / min(others)))
        print(f"  {key:<38} +{rob[key]['n_positive']}/7  2020 {v[i0]:+.5f}  "
              f"others {min(others):+.5f}..{max(others):+.5f}  "
              f"2020 rank {rob[key]['control_rank']}/7  "
              f"ratio {rob[key]['control_over_others_min']:+.3f}")
    n_neg = sum(1 for k_ in rob if rob[k_]["control"] < 0)
    n_rank1 = sum(1 for k_ in rob if rob[k_]["control_rank"] == 1)
    print(f"  -> 2020 is NEGATIVE in {n_neg}/{len(rob)} readings but is the "
          f"SMALLEST of the seven years in {n_rank1}/{len(rob)}")
    out["robustness"] = dict(variants=rob, n_control_negative=n_neg,
                             n_control_rank1=n_rank1)

    # ---------------------------------------------------- 41.6 the 2020 confound
    # 2020 was the year schools stayed shut AND the pandemic year, and p37 said
    # plainly that it does not try to separate them. It cannot be separated with
    # these data, but it CAN be narrowed, and the direction matters: a uniform
    # level shift is weak evidence, a deficit confined to the bands that go to
    # school is strong. The decomposition answers it without adding an
    # assumption, because the contrast is a WITHIN-year difference and a level
    # shift cancels out of it.
    print("\n=== 41.6 the 2020 confound: uniform shift or school-specific? ===")
    band_year = {y: {} for y in years}
    for y in years:
        for m, ym in ((ym % 100, ym) for ym in yms if ym // 100 == y):
            band_year[y][m] = bands[ym]
    per_band = {}
    for i, a in enumerate(AGES):
        s = {ym: float(bands[ym][i]) for ym in yms}
        v = {y: contrast(mv) for y, mv in by_year(s).items()}
        per_band[AGE_LABEL[a]] = v
    grp = {}
    for name, idx in GROUPS.items():
        s = {ym: float(bands[ym][idx].sum()) for ym in yms}
        yv = by_year(s)
        v = [contrast(yv[y]) for y in years]
        gst = sign_test(v)
        ps = [float((perm_null(yv[y]) > 0).mean()) for y in years]
        others = [x for j, x in enumerate(v) if years[j] != CONTROL_YEAR]
        grp[name] = dict(
            per_year={str(y): v[j] for j, y in enumerate(years)},
            sign_test=gst,
            p_count_at_least=poisson_binomial_tail(ps, gst["n_positive"]),
            control=v[years.index(CONTROL_YEAR)],
            others_median=float(np.median(others)),
            retention=float(v[years.index(CONTROL_YEAR)] / np.median(others)))
        print(f"  {name:<14} {gst['n_positive']}/7 positive  "
              f"sign p {gst['p_one_sided']:.5f}  perm p {grp[name]['p_count_at_least']:.5f}  "
              f"2020 {grp[name]['control'] * 1e3:+7.3f}e-3  "
              f"other years' median {grp[name]['others_median'] * 1e3:+7.3f}e-3  "
              f"2020 retains {grp[name]['retention']:+.1%}")
    print("  -> the term signal lives in the bands that go to school, and it is "
          "those bands, not all of them, that 2020 switches off")
    # The level channel, reported because it does NOT support the same reading.
    lev = {}
    c20 = np.median(np.array([bands[ym] for ym in yms
                              if ym // 100 == CONTROL_YEAR]), axis=0)
    coth = np.median(np.array([bands[ym] for ym in yms
                               if ym // 100 != CONTROL_YEAR]), axis=0)
    for i, a in enumerate(AGES):
        lev[AGE_LABEL[a]] = dict(control=float(c20[i]), others=float(coth[i]),
                                 ratio=float(c20[i] / coth[i]))
    worst = min(lev, key=lambda b: lev[b]["ratio"])
    print(f"  ! the LEVEL channel does not separate them: 2020's deepest deficit "
          f"is {worst} at {lev[worst]['ratio']:.2f}x, deeper than any school "
          f"band, so only the CONTRAST channel is clean")
    out["bands"] = dict(per_band_contrast=per_band, groups=grp,
                        level_2020_vs_rest=lev, deepest_level_deficit=worst)

    # ------------------------------------------------------------ 41.7 placebos
    print("\n=== 41.7 two placebos ===")
    shift = {}
    for kshift in range(12):
        T = {(m - 1 + kshift) % 12 + 1 for m in TERM}
        V = {(m - 1 + kshift) % 12 + 1 for m in VAC}
        v = []
        for y in years:
            mv = yr[y]
            t = [x for m, x in mv.items() if m in T]
            vv = [x for m, x in mv.items() if m in V]
            v.append(float(np.median(t) - np.median(vv)) if t and vv else np.nan)
        shift[kshift] = dict(
            per_year=[None if np.isnan(x) else x for x in v],
            near_degenerate_with=(kshift + 6) % 12,
            mean_excluding_control=float(np.nanmean(
                [x for j, x in enumerate(v) if years[j] != CONTROL_YEAR])))
    best = max(shift, key=lambda s: shift[s]["mean_excluding_control"])
    print("  cyclic shift of the academic labelling, mean over 2021-2026 (x1e3): "
          + " ".join(f"{s}:{shift[s]['mean_excluding_control'] * 1e3:+.2f}"
                     for s in range(12)))
    print(f"  argmax at shift {best}. Shifts k and k+6 are near-DEGENERATE -- the "
          f"vacation set {{1,2,7,8}} is invariant under six months, which is what "
          f"having two symmetric semesters means, and the term sets differ only "
          f"by exchanging one month -- so there are six distinct alignments, not "
          f"twelve, and the true one is the largest of the six")
    # Is the whole thing March and February?
    T2, V2 = tuple(m for m in TERM if m != 3), tuple(m for m in VAC if m != 2)
    nomar = {}
    for y in years:
        mv = yr[y]
        t = [x for m, x in mv.items() if m in T2]
        v = [x for m, x in mv.items() if m in V2]
        nomar[y] = float(np.median(t) - np.median(v))
    st2 = sign_test([nomar[y] for y in years])
    print(f"  with March and February deleted from the labelling the contrast is "
          f"{st2['n_positive']}/7 positive, 2020 {nomar[CONTROL_YEAR]:+.5f} vs "
          f"others {min(nomar[y] for y in years if y != CONTROL_YEAR):+.5f}.."
          f"{max(nomar[y] for y in years if y != CONTROL_YEAR):+.5f}")
    out["placebo"] = dict(cyclic_shift=shift, argmax_shift=best,
                          n_distinct_alignments=6,
                          shift_k_near_degenerate_with_k_plus_6=True,
                          without_march_february=dict(
                              per_year={str(y): nomar[y] for y in years},
                              sign_test=st2))

    # ---------------------------------------------------------------- the figure
    figure(years, yr, per_year, p_year, per_band, c20, coth)

    out["config"] = dict(term_months=list(TERM), vacation_months=list(VAC),
                         december_excluded=True, control_year=CONTROL_YEAR,
                         seed=SEED, n_mc=int(args.mc), n_months=len(yms),
                         source="eda/results_p37.json (not modified)")
    with open(f"{ROOT}/eda/results_p41.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p41.json")
    return 0


def figure(years, yr, per_year, p_year, per_band, c20, coth):
    """Three panels, one job each: the cycle, the test, the confound.

    Panel A has to make three things visible without the reader hunting: March
    is the peak, 2020 is flat, and the other six years are all there. p37 drew
    the cycle as shading under a 79-point sawtooth, which shows the trend and
    hides the cycle; folding the series onto the calendar month puts the seven
    years on top of each other and the 2020 line simply fails to rise.
    """
    C_TEAL, C_SLATE, C_OCHRE, C_RED = "#0E7C86", "#4C6E8A", "#BC8034", "#A8434E"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    for m in TERM:
        ax.axvspan(m - 0.5, m + 0.5, color=C_TEAL, alpha=0.07, lw=0)
    shades = plt.cm.GnBu(np.linspace(0.45, 0.95, len(years) - 1))
    allv = [v for y in years for v in yr[y].values()]
    # Headroom below for the legend and above for the March annotation, so
    # neither sits on top of a line; 202012 is the lowest month of all 79 and
    # has to stay inside the axes rather than being clipped at the frame.
    span = max(allv) - min(allv)
    ax.set_ylim(min(allv) - 0.42 * span, max(allv) + 0.16 * span)
    j = 0
    for y in years:
        mv = yr[y]
        x = sorted(mv)
        v = [mv[m] for m in x]
        if y == CONTROL_YEAR:
            ax.plot(x, v, "-o", c=C_RED, lw=2.4, ms=4, zorder=6,
                    label=f"{y} (schools shut)")
        else:
            ax.plot(x, v, "-o", c=shades[j], lw=1.1, ms=2.6, alpha=0.95, label=str(y))
            j += 1
    med = [np.median([yr[y][m] for y in years if m in yr[y]]) for m in range(1, 13)]
    ax.plot(range(1, 13), med, "--", c="0.25", lw=1.4, label="median of years")
    ax.axvline(3, color=C_OCHRE, lw=1.0, ls=":")
    ax.annotate("March: the Korean\nacademic year begins",
                xy=(3.15, ax.get_ylim()[1] - 0.02 * span), ha="left", va="top",
                fontsize=6.5, color=C_OCHRE)
    ax.annotate("2020", xy=(6, yr[CONTROL_YEAR][6]),
                xytext=(6.1, yr[CONTROL_YEAR][6] - 0.10 * span), fontsize=7,
                color=C_RED, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
                       fontsize=8)
    ax.set_xlabel("calendar month (shaded = school term)", fontsize=8)
    ax.set_ylabel("assortativity, dong level")
    ax.legend(fontsize=6, ncol=4, loc="lower center", framealpha=0.92,
              handlelength=1.6, columnspacing=1.0)
    ax.set_title("no school label enters the matrix, yet\nevery year but 2020 "
                 "rises in March", fontsize=9)

    ax = axes[1]
    # The observed value sits OUTSIDE the relabelling band in six of the seven
    # years, so it cannot be drawn as an error bar around itself. The band is
    # the null and the marker is the observation, which is also the honest way
    # round: the grey box is what the term/vacation split would be worth if the
    # labels were arbitrary.
    for i, y in enumerate(years):
        ax.add_patch(plt.Rectangle((i - 0.31, p_year[y]["null_lo"]), 0.62,
                                   p_year[y]["null_hi"] - p_year[y]["null_lo"],
                                   facecolor="0.86", edgecolor="0.6", lw=0.6,
                                   zorder=2))
    ax.plot([], [], marker="s", ls="none", c="0.86", mec="0.6", ms=8,
            label="95% of the within-year relabellings")
    v = [per_year[y]["contrast_median"] for y in years]
    for i, y in enumerate(years):
        c = C_RED if y == CONTROL_YEAR else C_TEAL
        ax.plot([i, i], [0, v[i]], "-", c=c, lw=1.6, zorder=3)
        ax.plot([i], [v[i]], "o", c=c, ms=7, zorder=4)
    ax.plot([], [], "o", c=C_TEAL, ms=7, label="observed, 2021-2026")
    ax.plot([], [], "o", c=C_RED, ms=7, label="observed, 2020")
    ax.axhline(0, color="0.2", lw=0.9, zorder=1)
    ax.set_xlim(-0.6, len(years) - 0.4)
    blo = min(p_year[y]["null_lo"] for y in years)
    bhi = max(max(v), max(p_year[y]["null_hi"] for y in years))
    ax.set_ylim(blo - 0.04 * (bhi - blo), bhi + 0.34 * (bhi - blo))
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=7)
    ax.set_ylabel("median(term) - median(vacation)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("6 of 7 years positive; the exception is the year\n"
                 "schools did not open (sign test one-sided p = 0.0625)",
                 fontsize=9)

    ax = axes[2]
    labs = [AGE_LABEL[a] for a in AGES]
    x = np.arange(len(labs))
    c2020 = np.array([per_band[b][CONTROL_YEAR] for b in labs]) * 1e3
    cother = np.array([np.median([per_band[b][y] for y in years
                                  if y != CONTROL_YEAR]) for b in labs]) * 1e3
    for i in SCHOOL_IDX + [AGES.index(20)]:
        ax.axvspan(i - 0.5, i + 0.5, color=C_OCHRE, alpha=0.10, lw=0)
    ax.bar(x - 0.2, cother, width=0.4, color=C_TEAL, label="2021-2026, median")
    ax.bar(x + 0.2, c2020, width=0.4, color=C_RED, label="2020")
    ax.axhline(0, color="0.2", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=90, fontsize=6.5)
    ax.set_ylabel("band's contribution to the contrast (x$10^{-3}$)")
    ax.legend(fontsize=6.5)
    ax.set_title("the cycle lives in the school and university bands,\n"
                 "and 2020 switches off exactly those (shaded)", fontsize=9)

    fig.suptitle("Phase 41 — the school-semester cycle: the instrument responds "
                 "to a calendar it was never told about", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p41_semester.png", dpi=150)
    plt.close(fig)
    print(f"\n  -> fig/p41_semester.png")


if __name__ == "__main__":
    sys.exit(main())
