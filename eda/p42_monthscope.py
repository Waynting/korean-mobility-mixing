#!/usr/bin/env python
"""Phase 42 — the 79-month passive series on the SAME day set as the floor.

WHY THIS EXISTS. memo/phase40b-month-scope.md recorded, marked "preliminary,
not a result", that about ten of the 79 months carry a passive `mi_bits` at or
above p40's declared national permutation floor, and that all ten are Korean
school-term months. It was preliminary for one reason, stated in its own
section 4: the series it counted came from the `WE|dong` rows of
results_p37.json, which are built on ALL SEVEN weekdays, while the floor in
results_p40.json is attached to a passive value built on the HOLIDAY-FREE
weekdays only. The two survey months proved the mismatch on their face --
0.019909 vs the published 0.019617 for 202312, 0.016277 vs 0.017712 for 202402.
Counting a series against a floor computed on a different day set is not a
result, and the count is what the paper's scope sentence turns on.

WHAT THIS SCRIPT SETTLES. The same estimator, on the holiday-free day set, on
all 79 months, counted against all four floors p40 measured (2 scopes x 2 survey
months). Nothing here is new machinery.

THE DAY-SET ANCHOR IS THE WHOLE POINT, so it runs first and it is bit-exact.
`matrices(df, ym, "WE", "dong", dows=holiday_free_dows(ym))` on 202312 and
202402, reduced by `symmetrise` on the 0-79 block and scored by `excess_stats`,
must return the very floats results_p40.json carries as `passive_mi_bits`
(0.019616749248339782 and 0.01771228807031904, themselves from results_p32).
If they do not, this script dies before counting anything, because a count
against a mismatched floor is exactly the thing phase40b already did once.

A SECOND ANCHOR, FOR FREE. p37 already stores the holiday-free variant of every
month, as `level == "dong_holidayfree"` -- the loop is right there at
p37_timeseries.py:120-133. The preliminary memo read `level == "dong"` instead,
which is the all-days variant, and that single filter is the entire origin of
the mismatch. So the corrected series can be checked against p37's stored one
bit for bit on all 79 months, and it is. That makes this file a recomputation
rather than a new measurement, which is the stronger of the two claims.

A THIRD CHECK, WHICH IS THE MECHANISM ITSELF. In a month whose calendar carries
no 공휴일 at all, `holiday_free_dows` returns all seven weekdays, so the
holiday-free and the all-days matrix are the SAME matrix and the two series must
agree to the last bit. They do, in every such month. That is a check on the day
set rather than on the arithmetic, and it is the one the preliminary version
could not have passed.

IMPORTS, NEVER RE-IMPLEMENTS. `cells`, `holiday_free_dows`, `matrices`, `stats`
from p26_matrix; `excess_stats`, `symmetrise` from p32_pmix; `population` from
p37_timeseries; `TERM`, `VAC` from p41_semester. The term/vacation partition is
p41's, unchanged, December in neither set -- redefining it here would let the
split be chosen after seeing which months clear.

NOTHING FROZEN IS REWRITTEN. results_p26/p32/p37/p40/p51 are read and left
alone; p31 gates them and p36 recomputes them. The only output is
results_p42.json.

THE COUNTS ARE p40's, THE SENTENCE IS p51's. Every count in 42.3b-42.5 is taken
against results_p40.json's four permutation medians, which is what p40 declared
and what the 08-22 letter published; p54 anchors on the 13 that follows and p63
asserts this file's `counts` floor is still p40's. p51_natsym.py has since
corrected the national median -- p40 built the national arm on Seoul's
population vector -- so 42.6's `sentence_allowed`, whose only job is to say what
the manuscript may write, is written against p51's floor and the count that
follows from it. Both floors and that count are read at run time; nothing in
that field is a typed number. This makes p51 a run-order dependency: p42 has to
run after p51.

FULLY DETERMINISTIC. There is no rng in this file. The permutation nulls in 42.4
are exact -- hypergeometric tails and their convolution, computed in closed form
-- so there is no seed to report and nothing for determinism_check.sh to catch
that a second run of p37 would not. It is listed there anyway, since it is
cheap (about a minute) and the point of that script is that "I ran it twice" is
a command rather than a claim.

    python eda/p42_monthscope.py
    python eda/p42_monthscope.py --months 202312,202402   # the anchor alone
"""
import argparse
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calendar_kr as K
from common import AGES, months_on_disk
from p26_matrix import cells, holiday_free_dows, matrices, stats
from p32_pmix import excess_stats, symmetrise
from p37_timeseries import population
from p41_semester import TERM, VAC          # (3,4,5,6,9,10,11) / (1,2,7,8)
from paths import DERIVED, FIG, ROOT, require

SURVEY_MONTHS = [202312, 202402]
# The four floors p40 measured: permutation median of the survey's own MI, one
# per (survey month, scope). p40's DECLARED primary is 202312|national; the
# other three are reported because p40 reports all four regardless of which way
# any of them lands, and a count is not allowed to pick its own threshold.
FLOOR_CELLS = ["202312|national", "202402|national",
               "202312|seoul", "202402|seoul"]
PRIMARY_CELL = "202312|national"
out = {}


def label(ym):
    """p41's partition, applied to a ym. December is in NEITHER set."""
    m = ym % 100
    return "term" if m in TERM else ("vacation" if m in VAC else "december")


def hyper_tail_ge(N, K_, n, k):
    """P(X >= k) for X ~ Hypergeometric(N population, K_ successes, n drawn)."""
    tot = comb(N, n)
    if tot == 0:
        return float("nan")
    return float(sum(comb(K_, i) * comb(N - K_, n - i)
                     for i in range(k, min(K_, n) + 1)
                     if 0 <= n - i <= N - K_) / tot)


def hyper_pmf(N, K_, n):
    """The full pmf of Hypergeometric(N, K_, n) as an array over 0..n."""
    tot = comb(N, n)
    return np.array([comb(K_, i) * comb(N - K_, n - i) / tot
                     if 0 <= n - i <= N - K_ and i <= K_ else 0.0
                     for i in range(n + 1)], float)


def within_year_null(per_year, target):
    """Exact null for "how many of the clearing months carry label `target`",
    when the labels are permuted WITHIN each year only.

    This is the test that matters and the pooled one is the test that flatters.
    The LEVEL trends: the year medians of this very series run 0.01372 in 2020
    to 0.02329 in 2026, a factor of 1.70 (42.3 prints them). A pooled shuffle is
    therefore free to pair a 2023 term month against a 2020 vacation month and
    read the trend as a cycle. p41's permutation is within-year for exactly this
    reason and this is the same null applied to a count instead of a median
    contrast.

    Within one year the labels are exchangeable across that year's months, so
    the count of `target` labels landing on that year's c clearing months is
    exactly Hypergeometric(n_months, n_target, c). The years are independent
    under the null, so the total is the convolution -- no Monte Carlo, no seed,
    no simulation error.

    per_year: {year: (n_months, n_target, n_clearing)}
    """
    poly = np.array([1.0])
    for _, (n_m, n_t, c) in sorted(per_year.items()):
        poly = np.convolve(poly, hyper_pmf(n_m, n_t, c))
    obs = int(target)
    return poly, float(poly[obs:].sum()), float(poly[:obs + 1].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="")
    args = ap.parse_args()
    yms = ([int(x) for x in args.months.split(",")] if args.months
           else months_on_disk())
    require(DERIVED / "regpop.parquet", "denominator")
    reg = pd.read_parquet(DERIVED / "regpop.parquet")
    fo = pd.read_parquet(DERIVED / "foreign.parquet")
    sq = [i for i, a in enumerate(AGES) if a < 80]     # no 80+ survey egos
    p37 = json.load(open(f"{ROOT}/eda/results_p37.json"))
    p40 = json.load(open(f"{ROOT}/eda/results_p40.json"))
    # p51 IS A BACKWARD DEPENDENCY AND IT IS DELIBERATE. Every count below is
    # taken against results_p40.json's four permutation medians, because those
    # are what p40 published and what the 08-22 letter quotes -- p54 anchors on
    # this file's 13 and p63 asserts that this file's `counts` floor is NOT the
    # corrected one, so that block does not move. But p51_natsym.py corrected
    # the national median (p40 built the national arm on Seoul's population
    # vector), and 42.6's `sentence_allowed` is the one field here whose whole
    # job is to say what the manuscript may write. A superseded floor typed into
    # that field as a string is the defect this load exists to remove: the
    # sentence now READS both floors and re-counts against them. The cost is
    # that p42 must be run after p51 (eda/README.md orders p42 at line 163 and
    # p51 at line 249), and it is loaded here, before the 80-second rebuild, so
    # a missing file fails in the first second rather than the last.
    p51 = json.load(open(f"{ROOT}/eda/results_p51.json"))
    print(f"=== 42.0 {len(yms)} months: {yms[0]}..{yms[-1]} ===")
    print(f"  p40 declared primary cell: {p40['declaration']['primary_cell']}")

    # ------------------------------------------------------- 42.1 the series
    # Both day sets, every month. The all-days variant is recomputed too, not
    # because the paper uses it but because the SIZE of the day-set effect is
    # the thing phase40b guessed at, and a guess is what this file replaces.
    rows = []
    for i, ym in enumerate(yms, 1):
        df = cells(ym)
        pop = population(reg, fo, ym)
        hf = holiday_free_dows(ym)
        nhol = sum(K.cell_exposure(ym, d)["n_holiday"] for d in range(1, 8))
        rec = dict(ym=ym, month=ym % 100, year=ym // 100,
                   semester=label(ym), holiday_free_dows=hf,
                   n_dows_kept=len(hf), n_holidays=int(nhol))
        for tag, dows in (("hf", hf), ("all", None)):
            A = matrices(df, ym, "WE", "dong", dows=dows)
            st = stats(A, pop)
            T, asym = symmetrise(A[np.ix_(sq, sq)] / pop[sq][:, None], pop[sq])
            ex = excess_stats(T)
            rec[f"mi_bits_{tag}"] = ex["mi_bits"]
            rec[f"nmi_{tag}"] = ex["nmi"]
            rec[f"half_l1_{tag}"] = ex["half_l1"]
            rec[f"assortativity_{tag}"] = st["assortativity"]
        rec["mi_delta"] = rec["mi_bits_hf"] - rec["mi_bits_all"]
        rows.append(rec)
        print(f"  [{i:>2}/{len(yms)}] {ym}  {rec['semester']:<9} "
              f"dows {len(hf)}/7  mi_bits holiday-free {rec['mi_bits_hf']:.6f}  "
              f"all-days {rec['mi_bits_all']:.6f}  "
              f"delta {rec['mi_delta']:+.6f}", flush=True)
    R = pd.DataFrame(rows).sort_values("ym").reset_index(drop=True)
    out["rows"] = R.to_dict("records")
    out["months"] = list(map(int, R.ym))

    # ------------------------------------------------------- 42.2 the anchors
    print("\n=== 42.2 the day-set anchor: this is the check, not a formality ===")
    anchor = dict(day_set=[], vs_p37_holidayfree=[], vs_p37_alldays=[],
                  zero_holiday_identity=[])
    ok = True
    for ym in SURVEY_MONTHS:
        r = R[R.ym == ym]
        if r.empty:
            continue
        got = float(r.mi_bits_hf.iloc[0])
        want = p40["cells"][f"{ym}|national"]["passive_mi_bits"]
        same = (got == want)
        anchor["day_set"].append(dict(ym=int(ym), got=got, stored_p40=want,
                                      abs_diff=abs(got - want),
                                      bit_exact=bool(same)))
        ok &= same
        print(f"  {ym}  holiday-free dows {list(r.holiday_free_dows.iloc[0])}  "
              f"mi_bits {got!r}")
        print(f"          p40 passive_mi_bits {want!r}   "
              f"{'BIT-EXACT' if same else 'MISMATCH'}")
        # and the all-days number, which is what the preliminary memo counted
        pre = float(r.mi_bits_all.iloc[0])
        print(f"          (all-days variant {pre:.6f} -- the preliminary memo's "
              f"number, off by {pre - want:+.6f})")
    if not ok:
        out["anchor"] = anchor
        out["fail"] = ("the holiday-free recomputation does not reproduce p40's "
                       "published passive values; the day set is still not "
                       "matched and no count may be taken from this run")
        with open(f"{ROOT}/eda/results_p42.json", "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        print("\nDAY-SET ANCHOR FAILED. Stopping before the count, and writing "
              "the failure to results_p42.json.")
        return 1
    print(f"  -> both survey months reproduce p40/p32 BIT FOR BIT. The series "
          f"below is on the floor's own day set.")

    # p37 stores the same holiday-free variant for all 79 months. It must agree.
    stored_hf = {r["ym"]: r for r in p37["rows"]
                 if r["level"] == "dong_holidayfree"}
    stored_all = {r["ym"]: r for r in p37["rows"]
                  if r["level"] == "dong" and r["panel"] == "WE"}
    for _, r in R.iterrows():
        ym = int(r.ym)
        for tag, store, key in (("hf", stored_hf, "vs_p37_holidayfree"),
                                ("all", stored_all, "vs_p37_alldays")):
            s = store.get(ym)
            if s is None:
                continue
            d = dict(ym=ym,
                     mi_abs_diff=abs(float(r[f"mi_bits_{tag}"]) - s["mi_bits"]),
                     assort_abs_diff=abs(float(r[f"assortativity_{tag}"])
                                         - s["assortativity"]))
            anchor[key].append(d)
            assert d["mi_abs_diff"] == 0.0 and d["assort_abs_diff"] == 0.0, \
                f"{ym}/{tag}: does not reproduce results_p37.json"
    for key, name in (("vs_p37_holidayfree", "dong_holidayfree"),
                      ("vs_p37_alldays", "dong (all days)")):
        a = anchor[key]
        print(f"  p37 {name:<20} {len(a)}/{len(R)} months reproduced, "
              f"max |diff| {max((x['mi_abs_diff'] for x in a), default=0):.1e}")

    # The mechanism check: no holiday in the month -> the two day sets ARE the
    # same day set, so the two series must be the same number.
    for _, r in R.iterrows():
        if r.n_dows_kept == 7:
            same = float(r.mi_bits_hf) == float(r.mi_bits_all)
            anchor["zero_holiday_identity"].append(
                dict(ym=int(r.ym), identical=bool(same),
                     abs_diff=abs(float(r.mi_bits_hf) - float(r.mi_bits_all))))
            assert same, f"{int(r.ym)}: 7 dows kept but the two series differ"
    z = anchor["zero_holiday_identity"]
    print(f"  in the {len(z)} months with no 공휴일 at all the two day sets are "
          f"the same day set, and the two series agree exactly: "
          f"{sum(x['identical'] for x in z)}/{len(z)}")
    out["anchor"] = anchor

    # -------------------------------------------- 42.3 the series, and the count
    print("\n=== 42.3 the 79-month series on the holiday-free day set ===")
    hf = R.mi_bits_hf.to_numpy(float)
    al = R.mi_bits_all.to_numpy(float)
    summ = {}
    for tag, v in (("holiday_free", hf), ("all_days", al)):
        summ[tag] = dict(
            n=int(v.size), median=float(np.median(v)),
            min=float(v.min()), max=float(v.max()),
            argmin=int(R.ym[int(np.argmin(v))]), argmax=int(R.ym[int(np.argmax(v))]),
            q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)))
        print(f"  {tag:<13} median {summ[tag]['median']:.5f}  range "
              f"{summ[tag]['min']:.5f} ({summ[tag]['argmin']}) - "
              f"{summ[tag]['max']:.5f} ({summ[tag]['argmax']})")
    d = R.mi_delta.to_numpy(float)
    moved = R[R.n_dows_kept < 7]
    print(f"  the day set moves mi_bits by {d.min():+.5f}..{d.max():+.5f} "
          f"(median {np.median(d):+.6f}); it is nonzero only in the "
          f"{len(moved)} months that contain a holiday, and it is NOT one-signed"
          f" -- {int((moved.mi_delta > 0).sum())} up, "
          f"{int((moved.mi_delta < 0).sum())} down")
    summ["day_set_effect"] = dict(
        n_months_with_holiday=int(len(moved)),
        delta_min=float(d.min()), delta_max=float(d.max()),
        delta_median=float(np.median(d)),
        delta_abs_median_where_nonzero=float(np.median(np.abs(moved.mi_delta)))
        if len(moved) else 0.0,
        n_up=int((moved.mi_delta > 0).sum()),
        n_down=int((moved.mi_delta < 0).sum()))
    ymed = R.groupby("year").mi_bits_hf.median()
    summ["year_median"] = {int(y): float(v) for y, v in ymed.items()}
    summ["year_median_ratio_max_over_min"] = float(ymed.max() / ymed.min())
    print("  year medians: " + "  ".join(f"{int(y)} {v:.5f}"
                                         for y, v in ymed.items())
          + f"   (max/min {ymed.max() / ymed.min():.2f})")
    out["summary"] = summ

    print("\n=== 42.3b the count, against every floor p40 measured ===")
    print(f"  {'floor cell':<18}{'floor':>10}{'holiday-free':>14}"
          f"{'all-days (prelim)':>19}")
    counts = {}
    for cell in FLOOR_CELLS:
        f = p40["cells"][cell]["mi_perm_median"]
        c_hf = R[R.mi_bits_hf >= f]
        c_al = R[R.mi_bits_all >= f]
        counts[cell] = dict(
            floor=f, is_primary=(cell == PRIMARY_CELL),
            n_at_or_above=int(len(c_hf)), n_months=int(len(R)),
            months=[int(x) for x in c_hf.sort_values("mi_bits_hf",
                                                     ascending=False).ym],
            n_at_or_above_all_days=int(len(c_al)),
            months_all_days=[int(x) for x in c_al.ym])
        print(f"  {cell:<18}{f:>10.5f}{len(c_hf):>10}/{len(R)}"
              f"{len(c_al):>15}/{len(R)}"
              f"{'   <- p40 PRIMARY' if cell == PRIMARY_CELL else ''}")
    out["counts"] = counts

    # F is p40's declared primary floor, and the assert is what ties it to a
    # DECLARED cell rather than to a scalar somebody chose: p40 stores the same
    # float twice and the two copies must not drift. It does not pin this file
    # to a literal -- there is no number typed here -- but it does keep every
    # count below on p40's arm, which p51 has since corrected. That is
    # deliberate and it is what p54 re-counts against; the corrected floor
    # enters only in 42.6, where the sentence is written.
    F = p40["verdict_primary"]["floor"]
    assert F == p40["cells"][PRIMARY_CELL]["mi_perm_median"], (
        f"p40's verdict_primary.floor {F!r} is not its own "
        f"cells[{PRIMARY_CELL!r}].mi_perm_median "
        f"{p40['cells'][PRIMARY_CELL]['mi_perm_median']!r}")
    clear = R[R.mi_bits_hf >= F].sort_values("mi_bits_hf", ascending=False)
    print(f"\n  the {len(clear)} months at or above the declared primary floor "
          f"{F:.5f}:")
    print(f"    {'ym':>8}{'mi_bits':>10}{'all-days':>10}  {'cal':>3}  semester")
    for _, r in clear.iterrows():
        print(f"    {int(r.ym):>8}{r.mi_bits_hf:>10.5f}{r.mi_bits_all:>10.5f}"
              f"  {int(r.month):>3}  {r.semester}")

    # The floor is itself a Monte Carlo median, and p40 stores its standard
    # error. A count that flips inside 2 s.e. of the threshold is not a count.
    se = p40["cells"][PRIMARY_CELL]["mi_perm_median_se"]
    sens = {}
    for mult in (-2, -1, 0, 1, 2):
        sens[f"{mult:+d}se"] = int((hf >= F + mult * se).sum())
    print(f"\n  floor Monte Carlo s.e. {se:.6f}; the count at the floor "
          f"+/- 1 and 2 s.e.: " + "  ".join(f"{k} {v}" for k, v in sens.items()))
    lo, hi = (p40["cells"][PRIMARY_CELL]["mi_perm_lo"],
              p40["cells"][PRIMARY_CELL]["mi_perm_hi"])
    print(f"  and against the permutation distribution's own 2.5/97.5 "
          f"({lo:.5f}, {hi:.5f}): {int((hf >= lo).sum())} and "
          f"{int((hf >= hi).sum())} of {len(R)}")
    # The floor does not sit in empty space -- it sits inside a dense band of
    # term months, which is why -2 s.e. buys three more. Naming them keeps the
    # count from being read as a sharp boundary that it is not.
    near = R[(R.mi_bits_hf < F) & (R.mi_bits_hf >= F - 3 * se)]
    print(f"  the {len(near)} months within 3 s.e. BELOW the floor: "
          + ", ".join(f"{int(r.ym)} {r.mi_bits_hf:.5f} [{r.semester}]"
                      for _, r in near.sort_values("mi_bits_hf",
                                                   ascending=False).iterrows()))
    print(f"  -> the floor is not a gap in the distribution, so {len(clear)} is "
          f"a count at a declared threshold and not a natural break in the data")
    out["floor_sensitivity"] = dict(
        floor=F, se=se, counts_at_offsets=sens,
        perm_lo=lo, perm_hi=hi,
        n_at_or_above_perm_lo=int((hf >= lo).sum()),
        n_at_or_above_perm_hi=int((hf >= hi).sum()),
        near_misses_within_3se=[dict(ym=int(r.ym), mi_bits=float(r.mi_bits_hf),
                                     semester=r.semester)
                                for _, r in near.iterrows()])

    # -------------------------------- 42.3c what the alignment actually changed
    # The preliminary count and this one differ by three months. Which three,
    # and in which direction, is the part that decides whether "about ten"
    # was a harmless approximation or a lucky one.
    set_hf = set(counts[PRIMARY_CELL]["months"])
    set_al = set(counts[PRIMARY_CELL]["months_all_days"])
    added, dropped = sorted(set_hf - set_al), sorted(set_al - set_hf)
    print("\n=== 42.3c what matching the day set changed ===")
    print(f"  added by the alignment ({len(added)}): "
          + ", ".join(f"{y} [{label(y)}]" for y in added))
    print(f"  removed by the alignment ({len(dropped)}): "
          + (", ".join(f"{y} [{label(y)}]" for y in dropped) or "none"))
    print("  The correction is one-directional here: the aligned count is "
          "LARGER, so the preliminary memo understated the problem rather than "
          "inventing it.")
    # A thin day set is a real caveat and it cuts both ways, so it is measured
    # rather than mentioned. Dropping a weekday drops a seventh of the week's
    # structure; a month down to one or three weekdays is a noisier object, and
    # 202402 -- a SURVEY month, and therefore part of the published convention
    # -- is one of them.
    thin = R[R.n_dows_kept <= 3]
    thin_clear = [int(y) for y in thin.ym if float(
        R[R.ym == y].mi_bits_hf.iloc[0]) >= F]
    print(f"\n  day-set thickness: "
          + ", ".join(f"{k} dow{'s' if k > 1 else ''}: {v}"
                      for k, v in sorted(R.n_dows_kept.value_counts().items())))
    print(f"  {len(thin)} months keep 3 weekdays or fewer "
          + ", ".join(f"{int(r.ym)}({r.n_dows_kept})" for _, r in thin.iterrows()))
    verb = "is" if len(thin_clear) == 1 else "are"
    print(f"  of those, {len(thin_clear)} {verb} in the clearing set: "
          f"{thin_clear}  -- and 202402, a SURVEY month, keeps only "
          f"{int(R[R.ym == 202402].n_dows_kept.iloc[0])}, so a thin day set is "
          f"the published convention rather than an artefact of this file")
    out["alignment_effect"] = dict(
        added=[dict(ym=int(y), semester=label(y)) for y in added],
        removed=[dict(ym=int(y), semester=label(y)) for y in dropped],
        n_added=len(added), n_removed=len(dropped),
        dows_kept_histogram={int(k): int(v)
                             for k, v in sorted(R.n_dows_kept.value_counts().items())},
        thin_months=[dict(ym=int(r.ym), n_dows_kept=int(r.n_dows_kept),
                          mi_bits_hf=float(r.mi_bits_hf),
                          mi_bits_all=float(r.mi_bits_all),
                          semester=r.semester) for _, r in thin.iterrows()],
        thin_months_clearing=thin_clear)

    # ------------------------------------------ 42.4 term or vacation, and luck
    print("\n=== 42.4 the term/vacation split of the months that clear ===")
    base = {k: int((R.semester == k).sum())
            for k in ("term", "vacation", "december")}
    print(f"  base rates over the {len(R)} months: term {base['term']}, "
          f"vacation {base['vacation']}, December {base['december']} "
          f"(p41's partition: term = {TERM}, vacation = {VAC}, "
          f"December in neither)")
    obs = {k: int((clear.semester == k).sum())
           for k in ("term", "vacation", "december")}
    print(f"  of the {len(clear)} clearing: term {obs['term']}, "
          f"vacation {obs['vacation']}, December {obs['december']}")

    k = len(clear)
    N = len(R)
    # (a) the pooled null: the clearing set is a uniform random k-subset of the
    #     79 months. This is the null the preliminary memo implicitly had in
    #     mind, and it is the WRONG one -- see within_year_null's docstring.
    p_pool_term = hyper_tail_ge(N, base["term"], k, obs["term"])
    p_pool_novac = comb(N - base["vacation"], k) / comb(N, k)
    print(f"\n  (a) pooled null (uniform random {k}-subset of all {N} months):")
    print(f"      P(term >= {obs['term']})            = {p_pool_term:.3e}")
    print(f"      P(no vacation month at all)  = {p_pool_novac:.3e}")

    # (b) the within-year null: the level trends across the seven years, so the
    #     labels are permuted within each year and never across.
    per_year_term, per_year_vac, per_year_clear = {}, {}, {}
    for y, g in R.groupby("year"):
        c = int((g.mi_bits_hf >= F).sum())
        per_year_clear[int(y)] = dict(
            n_months=int(len(g)), n_term=int((g.semester == "term").sum()),
            n_vacation=int((g.semester == "vacation").sum()),
            n_december=int((g.semester == "december").sum()), n_clearing=c,
            n_clearing_term=int(((g.mi_bits_hf >= F) & (g.semester == "term")).sum()))
        per_year_term[int(y)] = (len(g), int((g.semester == "term").sum()), c)
        per_year_vac[int(y)] = (len(g), int((g.semester == "vacation").sum()), c)
    _, p_wy_term, _ = within_year_null(per_year_term, obs["term"])
    poly_v, _, p_wy_novac = within_year_null(per_year_vac, obs["vacation"])
    print(f"\n  (b) within-year null (labels permuted inside each year only, "
          f"exact, no seed):")
    for y, v in sorted(per_year_clear.items()):
        if v["n_clearing"]:
            print(f"      {y}: {v['n_clearing']} of {v['n_months']} months "
                  f"clear, {v['n_term']} of them term-labelled -> "
                  f"P(all term this year) = "
                  f"{comb(v['n_term'], v['n_clearing']) / comb(v['n_months'], v['n_clearing']):.4f}")
    print(f"      P(term >= {obs['term']})            = {p_wy_term:.3e}")
    print(f"      P(vacation <= {obs['vacation']})         = {p_wy_novac:.3e}")
    ymed = R.groupby("year").mi_bits_hf.median()
    trend = float(ymed.max() / ymed.min())
    print(f"      This is the number to quote. The year medians run "
          f"{ymed.min():.5f} ({int(ymed.idxmin())}) to {ymed.max():.5f} "
          f"({int(ymed.idxmax())}), a factor of {trend:.2f}, and the pooled "
          f"null is free to credit that trend to the semester.")
    print("  ! neither p is calibrated evidence on its own: adjacent months are "
          "serially correlated, so even the within-year null over-counts "
          "independent trials. And this is not an independent test of the "
          "semester cycle -- it is p41's cycle seen from the floor side, as "
          "phase40b already said.")
    out["semester"] = dict(
        term_months=TERM, vacation_months=VAC, december="in neither set",
        base_rates=base, observed=obs, n_clearing=k, n_months=N,
        pooled=dict(p_term_ge=p_pool_term, p_no_vacation=p_pool_novac),
        within_year=dict(p_term_ge=p_wy_term, p_vacation_le=p_wy_novac,
                         per_year=per_year_clear, exact=True, seed=None),
        clearing=[dict(ym=int(r.ym), mi_bits=float(r.mi_bits_hf),
                       month=int(r.month), semester=r.semester)
                  for _, r in clear.iterrows()])

    # ---------------------------------------- 42.5 are the two survey months odd
    print("\n=== 42.5 where the two survey months sit in the 79 ===")
    ranks = {}
    order = R.sort_values("mi_bits_hf").reset_index(drop=True)
    for ym in SURVEY_MONTHS:
        if ym not in set(R.ym):
            continue
        i = int(order.index[order.ym == ym][0])
        v = float(order.mi_bits_hf.iloc[i])
        pct = (i + 1) / len(order)
        ranks[str(ym)] = dict(rank_ascending=i + 1, n=len(order),
                              rank_descending=len(order) - i,
                              mi_bits=v, percentile=pct,
                              semester=label(ym),
                              ratio_to_median=v / summ["holiday_free"]["median"])
        print(f"  {ym}  mi_bits {v:.5f}  rank {i + 1}/{len(order)} ascending "
              f"({pct:.0%} of months are at or below it)  "
              f"{v / summ['holiday_free']['median']:.2f}x the median  "
              f"[{label(ym)}]")
    print("  Both sit near the middle: the two months the survey exists in are "
          "not a flattering draw, and they are not an unflattering one either.")

    # ...but "near the middle of all 79" is the wrong reference class for the
    # sentence the paper actually wants to write, and this is the finding in
    # this file that costs the most. p41's partition puts 202312 in DECEMBER --
    # in neither set -- and 202402 in VACATION. NEITHER SURVEY MONTH IS A TERM
    # MONTH. So the one comparison the paper is allowed to make is made in the
    # two months where the instrument is structurally at its quietest, and the
    # months that clear the floor are drawn from a class the survey never
    # sampled. That is not a defect of the survey (Chae et al. chose December
    # and February) and it is not fixable with these data; it is a scope
    # statement the paper has to make in its own voice rather than let a
    # reviewer make for it.
    by_lab = {}
    for lab_ in ("term", "vacation", "december"):
        v = np.sort(R.loc[R.semester == lab_, "mi_bits_hf"].to_numpy(float))
        by_lab[lab_] = dict(n=int(v.size), median=float(np.median(v)),
                            min=float(v.min()), max=float(v.max()),
                            n_clearing=int((v >= F).sum()))
        print(f"  {lab_:<9} n {v.size:>2}  median {np.median(v):.5f}  "
              f"range {v.min():.5f}-{v.max():.5f}  clearing {int((v >= F).sum())}")
    term_v = np.sort(R.loc[R.semester == "term", "mi_bits_hf"].to_numpy(float))
    for ym in SURVEY_MONTHS:
        if str(ym) not in ranks:
            continue
        v = ranks[str(ym)]["mi_bits"]
        ranks[str(ym)]["n_term_months_above"] = int((term_v > v).sum())
        ranks[str(ym)]["n_term_months"] = int(term_v.size)
        ranks[str(ym)]["ratio_to_term_median"] = float(v / by_lab["term"]["median"])
        print(f"  {ym} [{label(ym)}] is below {int((term_v > v).sum())} of the "
              f"{term_v.size} term months, and sits at "
              f"{v / by_lab['term']['median']:.2f}x the term-month median")
    print("  -> NEITHER survey month is a term month. The floor comparison the "
          "paper leads with is made in December and in a vacation month, and "
          "every month that clears the floor is a term month. The survey never "
          "sampled the class of months where the passive reading is highest.")
    out["survey_month_ranks"] = ranks
    out["by_semester"] = by_lab

    # ------------------------------------------------------ 42.6 the verdict
    # THE SENTENCE IS NOT WRITTEN AGAINST THIS FILE'S OWN FLOOR, and that is the
    # point of this section. 42.3b counted against p40's four medians; p51
    # corrected the national one, so a sentence quoting p40's floor is quoting a
    # number that has been withdrawn. Every figure in the sentence below is READ
    # -- the two passive readings and the two national floors out of
    # results_p51.json, the count re-derived from this file's own series at that
    # floor -- so a re-run corrects it and nothing in it can go stale as a typed
    # string. The p40-floor count stays beside it under its own names because it
    # is what the 08-22 letter published and what p54 anchors on.
    n_pri = counts[PRIMARY_CELL]["n_at_or_above"]
    F51 = p51["cells"][PRIMARY_CELL]["mi_perm_median"]
    assert F51 != F, (
        f"results_p51.json's {PRIMARY_CELL} median is p40's own {F!r}; the "
        f"corrected national floor and the superseded one must differ, and if "
        f"they no longer do then p51 is not the correction this section reads")
    floors51 = {ym: p51["cells"][f"{ym}|national"]["mi_perm_median"]
                for ym in SURVEY_MONTHS}
    passive51 = {ym: p51["cells"][f"{ym}|national"]["passive_mi_bits"]
                 for ym in SURVEY_MONTHS}
    # p51's passive readings must BE the ones rebuilt here, or the sentence
    # would pair p51's floors with somebody else's numerator.
    for ym in SURVEY_MONTHS:
        if ym in set(R.ym):
            assert float(R[R.ym == ym].mi_bits_hf.iloc[0]) == passive51[ym], (
                f"{ym}: p51 stores passive_mi_bits {passive51[ym]!r}, this run "
                f"rebuilt {float(R[R.ym == ym].mi_bits_hf.iloc[0])!r}")
    clear51 = R[R.mi_bits_hf >= F51].sort_values("mi_bits_hf", ascending=False)
    n_cor = int(len(clear51))
    n_cor_term = int((clear51.semester == "term").sum())
    print("\n=== 42.6 what the paper may write ===")
    print(f"  p40's floor {F!r} -> {n_pri} of {len(R)} months "
          f"(this file's own count, unchanged, and p54's anchor)")
    print(f"  p51's floor {F51!r} -> {n_cor} of {len(R)} months, "
          f"{n_cor_term} of them term months (the sentence's count)")
    verdict = dict(
        primary_cell=PRIMARY_CELL, floor=F, n_months=int(len(R)),
        n_at_or_above=n_pri,
        general_claim_supported=bool(n_pri == 0),
        survey_month_claim_supported=all(
            float(R[R.ym == ym].mi_bits_hf.iloc[0]) < F for ym in SURVEY_MONTHS
            if ym in set(R.ym)),
        n_clearing_term=int((clear.semester == "term").sum()),
        survey_months_are_term=[label(ym) for ym in SURVEY_MONTHS],
        floor_corrected=F51,
        floor_corrected_source=(f"results_p51.json cells['{PRIMARY_CELL}']"
                                ".mi_perm_median"),
        floors_corrected_national={str(ym): floors51[ym]
                                   for ym in SURVEY_MONTHS},
        passive_corrected_national={str(ym): passive51[ym]
                                    for ym in SURVEY_MONTHS},
        n_at_or_above_corrected=n_cor,
        n_clearing_term_corrected=n_cor_term,
        months_corrected=[int(x) for x in clear51.ym],
        survey_month_claim_supported_corrected=all(
            float(R[R.ym == ym].mi_bits_hf.iloc[0]) < floors51[ym]
            for ym in SURVEY_MONTHS if ym in set(R.ym)),
        which_floor_the_sentence_uses=(
            "`floor`, `n_at_or_above` and `n_clearing_term` are this file's own "
            "count against results_p40.json's national permutation median. They "
            "are kept under those names because that is the count the 08-22 "
            "letter published, p54_semrecount.py anchors on it, and p63_fig2.py "
            "asserts that this file's `counts` block still holds it. "
            "`sentence_allowed` is written against `floor_corrected` instead -- "
            "p51_natsym.py's corrected national median, because p40 built the "
            "national arm on Seoul's population vector -- and against the count "
            "that follows from it. p54 re-counts at that floor and p58 replaces "
            "the null on the recounted set."),
        sentence_allowed=(
            "In 2023-12 and 2024-02, the two months the survey covers, the "
            "passive matrix's entire departure from proportionate mixing "
            f"({passive51[202312]:.4f} and {passive51[202402]:.4f} bits) is "
            "smaller than the permutation floor of the national survey it is "
            f"measured against ({floors51[202312]:.5f} and "
            f"{floors51[202402]:.5f} bits). This is a statement about those two "
            "months and does not generalise: measured the same way on the same "
            f"holiday-free day set, {n_cor} of the {len(R)} months on record "
            f"reach the 2023-12 floor, and all {n_cor_term} of them are Korean "
            f"school-term months -- a class neither survey month belongs to."),
        sentence_forbidden=(
            "The passive matrix carries less age information than the survey's "
            "sampling noise."))
    print(f"  ALLOWED:   {verdict['sentence_allowed']}")
    print(f"  FORBIDDEN: {verdict['sentence_forbidden']}")
    out["verdict"] = verdict

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8),
                             gridspec_kw=dict(width_ratios=[2.35, 1]))
    x = np.arange(len(R))
    lab = [str(y) for y in R.ym]
    ax = axes[0]
    for i, r in R.iterrows():
        if r.semester == "term":
            ax.axvspan(i - .5, i + .5, color="#0E7C86", alpha=.075, lw=0)
    ax.plot(x, al, "-", c="#B9C4C9", lw=1.0, zorder=1,
            label="all seven weekdays (the preliminary series)")
    ax.plot(x, hf, "-", c="#0E7C86", lw=1.4, zorder=2,
            label="holiday-free weekdays (the floor's day set)")
    m = R.mi_bits_hf >= F
    ax.plot(x[m.to_numpy()], hf[m.to_numpy()], "o", c="#A8434E", ms=5.5,
            zorder=4, label=f"at or above the primary floor ({int(m.sum())})")
    pub = [i for i, y in enumerate(R.ym) if y in SURVEY_MONTHS]
    ax.plot(pub, hf[pub], "*", c="#1E1E1E", ms=13, zorder=5,
            label="the two survey months")
    styles = {"202312|national": ("-", "#A8434E", 1.7),
              "202402|national": ("--", "#A8434E", 1.2),
              "202312|seoul": (":", "#4C6E8A", 1.2),
              "202402|seoul": ("-.", "#4C6E8A", 1.2)}
    for cell in FLOOR_CELLS:
        f = counts[cell]["floor"]
        ls, c, lw = styles[cell]
        if f > hf.max() * 1.25:
            continue                      # the Seoul floors are off the top
        ax.axhline(f, ls=ls, c=c, lw=lw, zorder=3)
        ax.annotate(f"{cell} floor {f:.4f}"
                    + ("  [p40 primary]" if cell == PRIMARY_CELL else ""),
                    xy=(len(R) - .5, f), xytext=(-4, 2), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.8, color=c)
    off = [c for c in FLOOR_CELLS if counts[c]["floor"] > hf.max() * 1.25]
    if off:
        # bottom, not top: the top-left corner is the legend's
        ax.annotate("the two Seoul floors are off the top of this axis ("
                    + ", ".join(f"{c.split('|')[0]} {counts[c]['floor']:.4f}"
                                for c in off) + "): 0 of 79 months reach them",
                    xy=(.5, .015), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=6.8, color="#4C6E8A")
    ax.set_xticks(range(0, len(R), 6))
    ax.set_xticklabels(lab[::6], rotation=90, fontsize=7)
    ax.set_ylabel("passive excess, mi_bits (WE panel, dong level)")
    ax.legend(fontsize=6.8, loc="upper left", framealpha=.9)
    ax.set_title(f"shaded = Korean school term (p41's partition).  "
                 f"{int(m.sum())} of {len(R)} months reach the declared floor, "
                 f"and all {int(m.sum())} are term months", fontsize=9)
    ax.set_xlim(-1, len(R))

    ax = axes[1]
    srt = R.sort_values("mi_bits_hf").reset_index(drop=True)
    col = {"term": "#0E7C86", "vacation": "#BC8034", "december": "#8A8A8A"}
    ax.barh(range(len(srt)), srt.mi_bits_hf,
            color=[col[s] for s in srt.semester], height=.86, lw=0)
    ax.axvline(F, c="#A8434E", lw=1.6)
    ax.annotate(f"primary floor\n{F:.4f}", xy=(F, len(srt) * .40),
                xytext=(4, 0), textcoords="offset points", fontsize=7,
                color="#A8434E", va="center")
    f2 = counts["202402|national"]["floor"]
    ax.axvline(f2, c="#A8434E", ls="--", lw=1.1)
    ax.annotate(f"202402 floor {f2:.4f}\n{counts['202402|national']['n_at_or_above']}"
                f" of {len(srt)} reach it", xy=(f2, len(srt) * .12),
                xytext=(4, 0), textcoords="offset points", fontsize=6.8,
                color="#A8434E", va="center")
    for ym in SURVEY_MONTHS:
        j = int(srt.index[srt.ym == ym][0])
        ax.plot(srt.mi_bits_hf.iloc[j], j, "*", c="#1E1E1E", ms=10, zorder=5)
    ax.set_yticks([])
    ax.set_ylim(-1, len(srt))
    ax.set_xlim(0, hf.max() * 1.32)
    ax.set_xlabel("mi_bits, months sorted")
    ax.set_title("every month above the floor is teal\n"
                 "teal term / orange vacation / grey Dec\n"
                 "stars = the two survey months", fontsize=8.5)
    fig.suptitle(f"Phase 42 — the {len(R)}-month passive series on the floor's "
                 f"own day set; p40's floors drawn in", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p42_monthscope.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p42_monthscope.png")

    with open(f"{ROOT}/eda/results_p42.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p42.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
