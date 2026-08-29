#!/usr/bin/env python
"""Phase 40 — the survey scope switch: 465 Seoul respondents -> 1,987 national,
and what it does to the headline sentence.

WHY NOW. The 8-21 letter, point 3. The sentence Phase 32 leads with --

    the passive matrix's entire departure from proportionate mixing (0.0196
    bits) is smaller than the sampling floor of the survey it is measured
    against (0.0713 bits), by 3.6x

-- rests on the SEOUL SUBSAMPLE of Chae et al. (Sci Data 13, 603, 2026): 465 of
the 1,987 respondents. The advisor's arithmetic is that the permutation floor
shuffles EGOS (p32_pmix.py:421), so it falls roughly as 1/n, and 465 -> 1,987 is
4.27x, which would put the 202312 floor near 0.0167 -- BELOW the passive excess,
i.e. the sentence reverses. That is an order-of-magnitude estimate. This script
measures it.

p27_survey.py has built both scopes since it was written (line ~268,
`for scope, so in (("seoul", True), ("national", False))`), and
`results_p27.json` already carries `202312|WE|national`. The 465 was a
deliberate primary choice, not a data limitation. What hardcodes Seoul is p32:
`sk = f"survey|{ym}|WE|seoul"` in 32.4, 32.4b and 32.5, and
`ego_cubes(ego, d, True, ...)` in 32.4b.

THE USER'S DECISION, APPLIED SYMMETRICALLY. An earlier proposal was asymmetric
-- national for the floor, Seoul for the point comparison. That is the shape
that flatters the paper (a bigger floor's worth of precision, a smaller
comparator's worth of concentration) and it is rejected here. National becomes
primary for BOTH the permutation floor and the point comparison. Seoul is kept
alongside as the robustness/anchor arm, never dropped.

======================================================================
DECLARED BEFORE THE RUN -- the primary cell, so the choice is not made
after seeing which cell is kinder:

    PRIMARY = permutation MEDIAN, 202312, NATIONAL scope,
              WE panel, holiday-free weekdays, 0-79 block.

    The headline sentence SURVIVES iff
        excess(passive|202312|WE|dong|holidayfree).mi_bits  <  that floor.

    p32 32.3 already names post-hoc statistic selection as a failure mode of
    this project. Choosing the cell after the run is the same failure mode
    wearing a different hat. All four cells (2 scopes x 2 months) are reported
    at both the median and the p95 regardless of which way any of them lands.
======================================================================

THE TENSION THIS DOES NOT DISSOLVE. memo/phase35-seir.md already records that a
national survey is not a legitimate control group for a Seoul-only passive
matrix, and that an earlier version of that section was burned by comparing a
national bootstrap against a Seoul point estimate. Switching symmetrically does
not answer the objection, it relocates it: the passive side is still Seoul, so
every respondent is still being matched to a Seoul object. 40.5 therefore
MEASURES the Seoul/national divergence instead of arguing about it, with the
only null that is actually available -- a random 465-respondent subsample of the
same 1,987, which is exactly "Seoul residence carries no information about
contact structure".

SEEDS, both reported.
  * ANCHOR_SEED = 20260819 -- p32's own seed. 40.1 replays p32's rng
    CONSUMPTION ORDER exactly (four null_floor calls in 32.4, then the 32.4b
    Seoul loop), so the anchor is bit-exact rather than tolerance-based. If it
    is not bit-exact the script dies before a single national number is read.
  * SEED = 20260821 -- everything new. Both scopes are drawn from this one
    stream so the Seoul and national arms of 40.2 are directly comparable, and
    the gap between 40.1's Seoul numbers and 40.2's Seoul numbers is a free
    Monte-Carlo-noise estimate on the floor itself.

IMPORTS, NEVER RE-IMPLEMENTS. Every estimator comes from p27_survey
(`load_survey`, `ego_cubes`, `contact_matrix`, `holiday_free_dows`), p32_pmix
(`excess_stats`, `rank_stats`, `null_floor`, `symmetrise`, `ipf`, `pm_null`) or
p35_seir (`marginal_ranking`, `final_size`, `optimal_allocation`, `kendall_tau`,
`scale_to_r0`, `ngm`, `top_k`). Nothing is written back to results_p27.json,
results_p32.json, results_p33.json or results_p35.json -- p31_report_audit.py
gates them and p36_recompute.py independently recomputes them. This script's
only output is results_p40.json.

    python eda/p40_natfloor.py [--boot 500] [--perm 500] [--boot35 200]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import AGES, AGE_LABEL
from paths import FIG, RESULTS, ROOT

import p32_pmix as P32
import p35_seir as P35
from p32_pmix import excess_stats, ipf, null_floor, pm_null, rank_stats
from p35_seir import (kendall_tau, marginal_ranking, ngm, optimal_allocation,
                      final_size, scale_to_r0, top_k)

SURVEY_MONTHS = [202312, 202402]
SCOPES = [("national", False), ("seoul", True)]      # national first: primary
ANCHOR_SEED = P32.SEED                                # 20260819
SEED = 20260821
PRIMARY = dict(statistic="mi_perm_median", ym=202312, scope="national",
               panel="WE", days="holiday-free", block="0-79")

out = {}
FAIL = []


def hdr(s):
    print(f"\n=== {s} ===")


# ----------------------------------------------------------------- 40.0 inputs
def load_all():
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))
    p32 = json.load(open(f"{ROOT}/eda/results_p32.json"))
    p33 = json.load(open(f"{ROOT}/eda/results_p33.json"))
    p35 = json.load(open(f"{ROOT}/eda/results_p35.json"))
    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))
    return p26, p27, p32, p33, p35, p9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=P32.BOOT)
    ap.add_argument("--perm", type=int, default=P32.PERM)
    ap.add_argument("--boot35", type=int, default=200)
    ap.add_argument("--sub", type=int, default=500)
    ap.add_argument("--r0", type=float, default=2.5)
    ap.add_argument("--budget", type=float, default=0.10)
    args = ap.parse_args()

    p26, p27, p32, p33, p35, p9 = load_all()
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]
    lbl = [AGE_LABEL[AGES[i]] for i in sq]
    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}

    out["declaration"] = dict(
        primary_cell=PRIMARY, anchor_seed=ANCHOR_SEED, seed=SEED,
        n_boot=args.boot, n_perm=args.perm, n_boot_pillar4=args.boot35,
        n_subsample=args.sub, r0=args.r0, budget_frac=args.budget,
        symmetry="national is primary for BOTH the permutation floor and the "
                 "point comparison; Seoul is reported alongside",
        survives_rule="headline survives iff passive mi_bits < the primary "
                      "cell's permutation median")
    print(__doc__.split("======================================================================")[1].strip())

    # The passive side never moves. It is Seoul either way; that is the tension.
    passive = {}
    for ym in SURVEY_MONTHS:
        A = np.array(p26["matrices"][f"{ym}|WE|dong|holidayfree"]["A"])[np.ix_(sq, sq)]
        Tp, _ = P32.symmetrise(A / pops[ym][sq][:, None], pops[ym][sq])
        n_dev = float(sum(pops[ym][i] / cov.get(AGE_LABEL[AGES[i]], np.nan)
                          for i in sq))
        passive[ym] = dict(T=Tp, n_dev=n_dev, ex=excess_stats(Tp),
                           sp=rank_stats(Tp))

    # ------------------------------------------------------------ 40.1 anchor
    hdr("40.1 ANCHOR: bit-exact replay of p32 32.4 + 32.4b (Seoul arm)")
    print("  p32 draws 32.4's four multinomial floors and 32.4b's Seoul loop "
          "from ONE rng\n  stream, so reproducing 32.4b requires replaying "
          "32.4's consumption first.")
    rng_a = np.random.default_rng(ANCHOR_SEED)
    anchor = {"seed": ANCHOR_SEED, "checks": []}

    def acheck(name, got, want, tol=0.0):
        d = abs(float(got) - float(want))
        ok = d <= tol
        anchor["checks"].append(dict(name=name, got=float(got),
                                     stored=float(want), abs_diff=d,
                                     tol=tol, ok=bool(ok)))
        if not ok:
            FAIL.append(f"anchor {name}: {got!r} vs stored {want!r}")
        return ok

    for ym in SURVEY_MONTHS:
        fp = null_floor(passive[ym]["T"], passive[ym]["n_dev"], rng=rng_a)
        Cs = np.array(p27["survey"][f"{ym}|WE|seoul"]["C"])[np.ix_(sq, sq)]
        Ts, _ = P32.symmetrise(Cs, pops[ym][sq])
        n_con = p32["null_floor"][str(ym)]["survey"]["n_effective"]
        fs = null_floor(Ts, n_con, rng=rng_a)
        acheck(f"{ym} 32.4 passive floor median",
               fp["mi_bits"]["median"],
               p32["null_floor"][str(ym)]["passive"]["floor"]["mi_bits"]["median"])
        acheck(f"{ym} 32.4 survey floor median",
               fs["mi_bits"]["median"],
               p32["null_floor"][str(ym)]["survey"]["floor"]["mi_bits"]["median"])

    from p27_survey import (contact_matrix, ego_cubes, holiday_free_dows,
                            load_survey)
    ego, d = load_survey()
    d = d[d.panel.isin(["W", "E"])]        # NOT household -- see p32 32.4b
    cubes = {}
    for ym in SURVEY_MONTHS:
        for scope, so in SCOPES:
            cubes[(ym, scope)] = ego_cubes(ego, d, so, ym=ym,
                                           dows=holiday_free_dows(ym))

    for ym in SURVEY_MONTHS:
        cube, eb, nd = cubes[(ym, "seoul")]
        pop = pops[ym][sq]
        draws, perms, assort = [], [], []
        for _ in range(P32.BOOT):
            idx = rng_a.integers(0, len(eb), len(eb))
            C, _ = contact_matrix(cube[idx], eb[idx], nd)
            T, _ = P32.symmetrise(C[np.ix_(sq, sq)], pop)
            st = excess_stats(T)
            draws.append(st["mi_bits"])
            assort.append(st["assortativity"])
        for _ in range(P32.PERM):
            C, _ = contact_matrix(cube, rng_a.permutation(eb), nd)
            T, _ = P32.symmetrise(C[np.ix_(sq, sq)], pop)
            perms.append(excess_stats(T)["mi_bits"])
        st = p32["survey_floor_measured"][str(ym)]
        got = dict(mi_perm_median=float(np.median(perms)),
                   mi_perm_p95=float(np.percentile(perms, 95)),
                   mi_boot_median=float(np.median(draws)),
                   mi_boot_lo=float(np.percentile(draws, 2.5)),
                   mi_boot_hi=float(np.percentile(draws, 97.5)),
                   assort_boot_lo=float(np.percentile(assort, 2.5)),
                   assort_boot_hi=float(np.percentile(assort, 97.5)))
        for k, v in got.items():
            acheck(f"{ym} 32.4b {k}", v, st[k])
        # and p27's own published assortativity CI, the anchor inside the anchor
        pub = p27["survey_assortativity_bootstrap"][str(ym)]
        acheck(f"{ym} p27 published assort lo (2e-2)", got["assort_boot_lo"],
               pub["lo"], 2e-2)
        acheck(f"{ym} p27 published assort hi (2e-2)", got["assort_boot_hi"],
               pub["hi"], 2e-2)

    n_ok = sum(c["ok"] for c in anchor["checks"])
    worst = max(c["abs_diff"] for c in anchor["checks"])
    anchor["n"] = len(anchor["checks"])
    anchor["n_ok"] = n_ok
    anchor["max_abs_diff"] = worst
    anchor["bit_exact"] = bool(all(c["abs_diff"] == 0.0 for c in anchor["checks"]
                                   if c["tol"] == 0.0))
    for c in anchor["checks"]:
        print(f"  {'ok ' if c['ok'] else 'FAIL'} {c['name']:<44} "
              f"{c['got']:.12g}  diff {c['abs_diff']:.1e}")
    print(f"  --> {n_ok}/{len(anchor['checks'])} pass, worst |diff| {worst:.2e}, "
          f"bit-exact on the zero-tolerance checks: {anchor['bit_exact']}")
    out["anchor_p32"] = anchor
    assert n_ok == len(anchor["checks"]) and anchor["bit_exact"], (
        "the p32 anchor does not reproduce; STOP -- fix this before reading a "
        f"single national number. {FAIL}")

    # p35's deterministic anchors, for Part B. No rng in either.
    hdr("40.1b ANCHOR: p35's marginal ranking, deterministic")
    a35 = []
    mats, scaled = {}, {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        for tag, key in (("passive_dong", f"{ym}|WE|dong|holidayfree"),
                         ("passive_gu", f"{ym}|WE|gu")):
            A = np.array(p26["matrices"][key]["A"])[np.ix_(sq, sq)]
            C, _ = P35.symmetrise(A / pops[ym][sq][:, None], pop)
            mats[(ym, tag)] = C
        Cs = np.array(p27["survey"][f"{ym}|WE|seoul"]["C"])[np.ix_(sq, sq)]
        mats[(ym, "survey_seoul")] = P35.symmetrise(Cs, pop)[0]
        Cn = np.array(p27["survey"][f"{ym}|WE|national"]["C"])[np.ix_(sq, sq)]
        mats[(ym, "survey_national")] = P35.symmetrise(Cn, pop)[0]
        for tag in ("passive_dong", "passive_gu", "survey_seoul",
                    "survey_national"):
            C = mats[(ym, tag)]
            scaled[(ym, tag)] = (C, scale_to_r0(C, args.r0))
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        s_seoul, _ = marginal_ranking(scaled[(ym, "survey_seoul")][0],
                                      scaled[(ym, "survey_seoul")][1], pop)
        for tag in ("passive_dong", "passive_gu"):
            sp_, _ = marginal_ranking(scaled[(ym, tag)][0],
                                      scaled[(ym, tag)][1], pop)
            got = kendall_tau(s_seoul, sp_)
            want = [r["tau"] for r in p35["comparison_marginal"][str(ym)]
                    if r["pair"] == f"survey vs {tag}"][0]
            ok = abs(got - want) < 1e-9
            a35.append(dict(name=f"{ym} tau survey_seoul vs {tag}", got=got,
                            stored=want, ok=bool(ok)))
            print(f"  {'ok ' if ok else 'FAIL'} {ym} tau(survey_seoul, {tag}) "
                  f"= {got:+.9f}  vs p35 {want:+.9f}")
            if not ok:
                FAIL.append(f"p35 anchor {ym} {tag}")
    out["anchor_p35"] = a35
    assert all(a["ok"] for a in a35), "p35 marginal-ranking anchor failed"

    # ------------------------------------------------- 40.2 the four cells
    hdr(f"40.2 THE FOUR CELLS: permutation floor at both scopes, both months "
        f"({args.boot} bootstraps + {args.perm} permutations, seed {SEED})")
    print("  The permutation shuffles EGO bands and keeps each ego's own alter")
    print("  tally, so it is the proportionate-mixing null at this n_ego.")
    rng = np.random.default_rng(SEED)
    cells = {}
    for ym in SURVEY_MONTHS:
        for scope, _ in SCOPES:
            cube, eb, nd = cubes[(ym, scope)]
            pop = pops[ym][sq]
            draws, perms, assort, aperm = [], [], [], []
            for _ in range(args.boot):
                idx = rng.integers(0, len(eb), len(eb))
                C, _ = contact_matrix(cube[idx], eb[idx], nd)
                T, _ = P32.symmetrise(C[np.ix_(sq, sq)], pop)
                st = excess_stats(T)
                draws.append(st["mi_bits"])
                assort.append(st["assortativity"])
            for _ in range(args.perm):
                C, _ = contact_matrix(cube, rng.permutation(eb), nd)
                T, _ = P32.symmetrise(C[np.ix_(sq, sq)], pop)
                st = excess_stats(T)
                perms.append(st["mi_bits"])
                aperm.append(st["assortativity"])
            C0, _ = contact_matrix(cube, eb, nd)
            T0, _ = P32.symmetrise(C0[np.ix_(sq, sq)], pop)
            n_con = float(np.bincount(eb, minlength=len(AGES))[sq]
                          @ C0[np.ix_(sq, sq)].sum(1) * nd)
            ex = excess_stats(T0, n_obs=n_con)
            mfloor = null_floor(T0, n_con, rng=rng)     # 32.4's multinomial one
            pas = passive[ym]["ex"]["mi_bits"]
            med = float(np.median(perms))
            p95 = float(np.percentile(perms, 95))
            # The primary verdict is a comparison of two numbers, so it needs
            # its own error bar. Two of them:
            #   * where the passive excess sits INSIDE the permutation null,
            #     which is the verdict without any summary statistic in it;
            #   * the Monte Carlo standard error of the median itself, so
            #     "survives by 18%" can be read against "the median is worth
            #     +/- x%".
            pv = np.array(perms, float)
            frac_below = float((pv < pas).mean())
            meds = [float(np.median(rng.choice(pv, len(pv), replace=True)))
                    for _ in range(1000)]
            med_se = float(np.std(meds, ddof=1))
            cells[f"{ym}|{scope}"] = dict(
                ym=ym, scope=scope, n_ego=int(len(eb)), n_contacts=n_con,
                n_ego_per_band=np.bincount(eb, minlength=len(AGES))[sq].tolist(),
                survey_mi_bits=ex["mi_bits"], passive_mi_bits=pas,
                mi_perm_median=med, mi_perm_p95=p95,
                mi_perm_median_se=med_se,
                mi_perm_median_margin_in_se=float((med - pas) / med_se)
                if med_se else np.nan,
                passive_frac_below_perm_null=frac_below,
                mi_perm_lo=float(np.percentile(perms, 2.5)),
                mi_perm_hi=float(np.percentile(perms, 97.5)),
                mi_boot_median=float(np.median(draws)),
                mi_boot_lo=float(np.percentile(draws, 2.5)),
                mi_boot_hi=float(np.percentile(draws, 97.5)),
                assort_boot_lo=float(np.percentile(assort, 2.5)),
                assort_boot_hi=float(np.percentile(assort, 97.5)),
                assort_perm_median=float(np.median(aperm)),
                assort_observed=ex["assortativity"],
                mm_bias_bits=ex["mm_bias_bits"],
                multinomial_floor_median=mfloor["mi_bits"]["median"],
                multinomial_floor_p95=mfloor["mi_bits"]["p95"],
                floor_over_passive_median=float(med / pas),
                floor_over_passive_p95=float(p95 / pas),
                survives_median=bool(pas < med), survives_p95=bool(pas < p95))
    out["cells"] = cells

    print(f"\n  {'cell':<20}{'n_ego':>7}{'n_cont':>9}{'passive':>10}"
          f"{'perm med':>10}{'perm p95':>10}{'med x':>8}{'p95 x':>8}"
          f"  survives (med / p95)")
    for ym in SURVEY_MONTHS:
        for scope, _ in SCOPES:
            c = cells[f"{ym}|{scope}"]
            star = " <-- PRIMARY" if (c["ym"] == PRIMARY["ym"]
                                      and c["scope"] == PRIMARY["scope"]) else ""
            print(f"  {ym} {scope:<13}{c['n_ego']:>7}{c['n_contacts']:>9,.0f}"
                  f"{c['passive_mi_bits']:>10.5f}{c['mi_perm_median']:>10.5f}"
                  f"{c['mi_perm_p95']:>10.5f}"
                  f"{c['floor_over_passive_median']:>8.2f}"
                  f"{c['floor_over_passive_p95']:>8.2f}"
                  f"   {'YES' if c['survives_median'] else 'NO ':<4}/ "
                  f"{'YES' if c['survives_p95'] else 'NO'}{star}")
    print(f"\n  the same four cells without a summary statistic in them -- where")
    print(f"  the passive excess sits inside the survey's own permutation null:")
    for ym in SURVEY_MONTHS:
        for scope, _ in SCOPES:
            c = cells[f"{ym}|{scope}"]
            print(f"  {ym} {scope:<10} {c['passive_frac_below_perm_null']:>6.1%} of "
                  f"permutation draws fall BELOW the passive excess; median "
                  f"{c['mi_perm_median']:.5f} +/- {c['mi_perm_median_se']:.5f} "
                  f"(MC se), margin {c['mi_perm_median_margin_in_se']:+.1f} se")

    # how well did the 1/n prediction do?
    scal = []
    for ym in SURVEY_MONTHS:
        s_, n_ = cells[f"{ym}|seoul"], cells[f"{ym}|national"]
        r_ego = n_["n_ego"] / s_["n_ego"]
        r_con = n_["n_contacts"] / s_["n_contacts"]
        obs = s_["mi_perm_median"] / n_["mi_perm_median"]
        scal.append(dict(ym=ym, ratio_n_ego=r_ego, ratio_n_contacts=r_con,
                         observed_floor_ratio=obs,
                         predicted_from_n_ego=float(s_["mi_perm_median"] / r_ego),
                         actual_national_floor=n_["mi_perm_median"]))
        print(f"\n  {ym}: n_ego x{r_ego:.3f}, n_contacts x{r_con:.3f}; the "
              f"floor actually fell by x{obs:.3f}")
        print(f"        1/n_ego predicted {s_['mi_perm_median'] / r_ego:.5f}, "
              f"measured {n_['mi_perm_median']:.5f} "
              f"({100 * (n_['mi_perm_median'] / (s_['mi_perm_median'] / r_ego) - 1):+.1f}%)")
    out["floor_scaling"] = scal

    # Monte Carlo noise on the floor itself: 40.1's Seoul arm vs 40.2's
    mc = []
    for ym in SURVEY_MONTHS:
        a = [c for c in anchor["checks"]
             if c["name"] == f"{ym} 32.4b mi_perm_median"][0]["got"]
        b = cells[f"{ym}|seoul"]["mi_perm_median"]
        mc.append(dict(ym=ym, seed_20260819=a, seed_20260821=b,
                       abs_diff=abs(a - b), rel_diff=abs(a - b) / a))
        print(f"  {ym} Seoul floor, two independent seeds: {a:.5f} vs {b:.5f} "
              f"({100 * abs(a - b) / a:.2f}% apart) -- Monte Carlo noise on the "
              f"floor")
    out["floor_monte_carlo_noise"] = mc

    # ------------------------- 40.2b why the floor did not fall as 1/n
    # The 1/n rule under-predicts the national floor. Two candidate reasons, and
    # they have opposite consequences for the paper: either the permutation
    # floor simply does not scale as 1/n (a property of the estimator, and the
    # switch is fine), or the national POPULATION is more heterogeneous between
    # respondents and that inflates the null (a property of the comparator, and
    # the switch has imported it). They are separable: hold n at 465 and change
    # only the population.
    #
    # Own rng stream (SEED + 2), so adding this section leaves every number in
    # 40.2 and everything after it bit-identical.
    hdr("40.2b is the floor change sample size, or is it population?")
    rng2 = np.random.default_rng(SEED + 2)
    decomp = {}
    N_SUB_FLOOR, PERM_SUB = 20, 200
    for ym in SURVEY_MONTHS:
        cn, ebn, ndn = cubes[(ym, "national")]
        n465 = cells[f"{ym}|seoul"]["n_ego"]
        pop = pops[ym][sq]
        meds = []
        for _ in range(N_SUB_FLOOR):
            sub = rng2.choice(len(ebn), n465, replace=False)
            cs_, eb_ = cn[sub], ebn[sub]
            pp = []
            for _ in range(PERM_SUB):
                C, _ = contact_matrix(cs_, rng2.permutation(eb_), ndn)
                T, _ = P32.symmetrise(C[np.ix_(sq, sq)], pop)
                pp.append(excess_stats(T)["mi_bits"])
            meds.append(float(np.median(pp)))
        meds = np.array(meds)
        f_seoul = cells[f"{ym}|seoul"]["mi_perm_median"]
        f_nat = cells[f"{ym}|national"]["mi_perm_median"]
        decomp[str(ym)] = dict(
            n=int(n465), n_subsamples=N_SUB_FLOOR, n_perm=PERM_SUB,
            floor_seoul_465=f_seoul, floor_national_465_median=float(np.median(meds)),
            floor_national_465_lo=float(np.percentile(meds, 2.5)),
            floor_national_465_hi=float(np.percentile(meds, 97.5)),
            floor_national_1987=f_nat,
            population_effect=float(np.median(meds) / f_seoul),
            size_effect=float(f_nat / np.median(meds)),
            seoul_inside_national_465_band=bool(
                np.percentile(meds, 2.5) <= f_seoul <= np.percentile(meds, 97.5)))
        e = decomp[str(ym)]
        print(f"  {ym}: floor at n=465 Seoul {f_seoul:.5f} | at n=465 drawn from "
              f"the national pool {e['floor_national_465_median']:.5f} "
              f"[{e['floor_national_465_lo']:.5f}, {e['floor_national_465_hi']:.5f}]"
              f" | at n=1,987 national {f_nat:.5f}")
        print(f"  {'':<6} population effect (465 Seoul -> 465 national) "
              f"x{e['population_effect']:.3f}; size effect (465 -> 1,987 within "
              f"the national pool) x{e['size_effect']:.3f}; Seoul's own floor is "
              f"{'INSIDE' if e['seoul_inside_national_465_band'] else 'OUTSIDE'}"
              f" the national-465 band")
    out["floor_decomposition"] = decomp

    prim = cells[f"{PRIMARY['ym']}|{PRIMARY['scope']}"]
    print(f"\n  PRIMARY CELL ({PRIMARY['ym']}, {PRIMARY['scope']}, permutation "
          f"median): floor {prim['mi_perm_median']:.5f} bits vs passive excess "
          f"{prim['passive_mi_bits']:.5f} bits")
    print(f"  --> the headline sentence "
          f"{'SURVIVES' if prim['survives_median'] else 'DOES NOT SURVIVE'} "
          f"at the declared primary cell.")
    out["verdict_primary"] = dict(
        cell=f"{PRIMARY['ym']}|{PRIMARY['scope']}",
        floor=prim["mi_perm_median"], passive=prim["passive_mi_bits"],
        ratio=prim["floor_over_passive_median"],
        survives=prim["survives_median"])

    # ---------------------------------------- 40.3 the point comparison
    hdr("40.3 the point comparison at national scope (32.3, re-scoped)")
    comp = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        rows = {}
        for scope, _ in SCOPES:
            C = np.array(p27["survey"][f"{ym}|WE|{scope}"]["C"])[np.ix_(sq, sq)]
            T, asym = P32.symmetrise(C, pop)
            n_con = cells[f"{ym}|{scope}"]["n_contacts"]
            rows[scope] = dict(excess_stats(T, n_obs=n_con), asymmetry=asym,
                               n_contacts=n_con, **rank_stats(T))
            # anchor against p32's stored value for the same matrix
            stored = p32["excess"].get(f"survey|{ym}|WE|{scope}")
            if stored:
                dmax = max(abs(rows[scope][k] - stored[k]) for k in
                           ("assortativity", "half_l1", "cramers_v", "mi_bits",
                            "nmi"))
                rows[scope]["p32_max_abs_diff"] = float(dmax)
                assert dmax < 1e-12, (f"40.3 does not reproduce p32's stored "
                                      f"survey|{ym}|WE|{scope}: {dmax:.2e}")
        pas = passive[ym]["ex"]
        comp[str(ym)] = dict(passive=pas, seoul=rows["seoul"],
                             national=rows["national"],
                             passive_spectrum=passive[ym]["sp"])
        print(f"\n  {ym}  {'statistic':<16}{'passive':>10}{'seoul':>10}"
              f"{'x(seoul)':>10}{'national':>10}{'x(national)':>13}"
              f"{'shift':>9}")
        for st, nm in (("half_l1", "half-L1"), ("cramers_v", "Cramer's V"),
                       ("assortativity", "assortativity"),
                       ("mi_bits", "MI (bits)"), ("nmi", "normalised MI")):
            rs = rows["seoul"][st] / pas[st]
            rn = rows["national"][st] / pas[st]
            print(f"  {'':<6}{nm:<16}{pas[st]:>10.4f}{rows['seoul'][st]:>10.4f}"
                  f"{rs:>9.1f}x{rows['national'][st]:>10.4f}{rn:>12.1f}x"
                  f"{100 * (rn / rs - 1):>+8.1f}%")
        for scope in ("seoul", "national"):
            rr = [rows[scope][s] / pas[s] for s in
                  ("half_l1", "cramers_v", "assortativity", "nmi")]
            comp[str(ym)][f"ratio_spread_{scope}"] = dict(lo=float(min(rr)),
                                                          hi=float(max(rr)))
            print(f"  {'':<6}{'ratio spans (' + scope + ')':<16}"
                  f"{min(rr):>29.1f}x {max(rr):>21.1f}x")
        print(f"  {'':<6}{'sigma1^2/sum':<16}"
              f"{passive[ym]['sp']['sigma1_share']:>10.4f}"
              f"{rows['seoul']['sigma1_share']:>10.4f}{'':>10}"
              f"{rows['national']['sigma1_share']:>10.4f}")
    out["point_comparison"] = comp

    # the theta family, re-scoped. p33 divides p32's SEOUL survey stat by the
    # theta-family passive stat; the national version divides the same passive
    # stats by the national survey's, so it is a pure re-scoping of p33.
    hdr("40.3b the coverage theta family (p33 34.4), re-scoped to national")
    th = {}
    for ym in SURVEY_MONTHS:
        fam = p33["R2_theta_family"].get(str(ym))
        if not fam:
            continue
        rows = []
        for r_ in fam:
            e = {}
            for st in ("assortativity", "half_l1", "cramers_v", "nmi"):
                e[st] = dict(
                    seoul=(comp[str(ym)]["seoul"][st] / r_[st]) if r_[st] else np.nan,
                    national=(comp[str(ym)]["national"][st] / r_[st]) if r_[st] else np.nan)
            rows.append(dict(theta=r_["theta"], **e))
        th[str(ym)] = rows
        z = [r_ for r_ in rows if r_["theta"] == 0.0][0]
        print(f"  {ym} theta=0: assortativity {z['assortativity']['seoul']:.1f}x "
              f"(seoul) -> {z['assortativity']['national']:.1f}x (national); "
              f"NMI {z['nmi']['seoul']:.1f}x -> {z['nmi']['national']:.1f}x")
        lo_n = min(min(r_[s]["national"] for s in
                       ("assortativity", "half_l1", "cramers_v", "nmi"))
                   for r_ in rows)
        lo_s = min(min(r_[s]["seoul"] for s in
                       ("assortativity", "half_l1", "cramers_v", "nmi"))
                   for r_ in rows)
        print(f"  {'':<6}smallest gap anywhere in the family: {lo_s:.1f}x "
              f"(seoul) -> {lo_n:.1f}x (national); both "
              f"{'> 1' if min(lo_s, lo_n) > 1 else 'BREAK'}")
        th[f"{ym}|family_min"] = dict(seoul=lo_s, national=lo_n)
    out["theta_family_rescoped"] = th

    # ------------------------------------------------------- 40.4 IPF (32.5)
    hdr("40.4 IPF margin calibration (32.5) at both scopes")
    ipf_out = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        Tp = passive[ym]["T"]
        row = {}
        for scope, _ in SCOPES:
            C = np.array(p27["survey"][f"{ym}|WE|{scope}"]["C"])[np.ix_(sq, sq)]
            Ts, _ = P32.symmetrise(C, pop)
            tgt = Ts.sum(1) / Ts.sum() * Tp.sum()
            X, _ = ipf(Tp, tgt, tgt)
            mult = np.divide(X.sum(1), Tp.sum(1), out=np.ones_like(tgt),
                             where=Tp.sum(1) > 0)
            eb_, ea_ = excess_stats(Tp), excess_stats(X)
            nmi_s = comp[str(ym)][scope]["nmi"]
            row[scope] = dict(
                bands=lbl, row_multiplier=mult.tolist(),
                nmi_before=eb_["nmi"], nmi_after=ea_["nmi"], nmi_survey=nmi_s,
                gap_before=float(nmi_s / eb_["nmi"]),
                gap_after=float(nmi_s / ea_["nmi"]),
                absorbed_frac=float(1 - (nmi_s / ea_["nmi"]) / (nmi_s / eb_["nmi"])),
                converged=bool(nmi_s / ea_["nmi"] < 2.0))
            print(f"  {ym} {scope:<9}: NMI gap {row[scope]['gap_before']:.1f}x "
                  f"before -> {row[scope]['gap_after']:.1f}x after "
                  f"({row[scope]['absorbed_frac']:.1%} absorbed); "
                  f"row multipliers {mult.min():.2f}-{mult.max():.2f}; "
                  f"{'CONVERGES' if row[scope]['converged'] else 'does NOT converge'}")
        # anchor: the Seoul arm must reproduce p32's stored 32.5
        stored = p32["ipf_calibration"]["months"].get(str(ym))
        if stored:
            dd = abs(row["seoul"]["gap_after"] - stored["gap_after"])
            assert dd < 1e-9, f"40.4 does not reproduce p32 32.5 for {ym}: {dd:.2e}"
            print(f"  {'':<6}   anchor: seoul gap_after reproduces p32 "
                  f"({stored['gap_after']:.6f}), |diff| {dd:.1e}")
        ipf_out[str(ym)] = row
    out["ipf_calibration"] = ipf_out

    # -------------------------------- 40.5 how far apart are the two surveys?
    hdr("40.5 THE TENSION, MEASURED: Seoul survey vs national survey")
    print("  The passive matrix is Seoul-only. Switching the survey side to")
    print("  national does not remove that; it relocates it. So: how different")
    print("  are the two survey matrices, and is the difference more than a")
    print("  random 465-respondent subsample of the same 1,987 would produce?")
    print(f"  Null: {args.sub} draws of 465 respondents WITHOUT replacement from")
    print("  the national pool -- i.e. 'Seoul residence carries no information'.")

    STATS = ("assortativity", "half_l1", "cramers_v", "nmi", "mi_bits")
    diverge = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        cs, ebs, nds = cubes[(ym, "seoul")]
        cn, ebn, ndn = cubes[(ym, "national")]
        assert nds == ndn
        Cs_, _ = contact_matrix(cs, ebs, nds)
        Cn_, _ = contact_matrix(cn, ebn, ndn)
        Ts, _ = P32.symmetrise(Cs_[np.ix_(sq, sq)], pop)
        Tn, _ = P32.symmetrise(Cn_[np.ix_(sq, sq)], pop)
        es, _, _ = pm_null(Ts)
        en, _, _ = pm_null(Tn)
        st_s, st_n = excess_stats(Ts), excess_stats(Tn)
        vs = ngm(Cs_[np.ix_(sq, sq)])[2]
        vn = ngm(Cn_[np.ix_(sq, sq)])[2]
        cos = float(vs @ vn / (np.linalg.norm(vs) * np.linalg.norm(vn)))
        # the fourth pillar's own currency, between the two survey matrices
        Ms, Mn = P35.symmetrise(Cs_[np.ix_(sq, sq)], pop)[0], \
            P35.symmetrise(Cn_[np.ix_(sq, sq)], pop)[0]
        bs, _ = marginal_ranking(Ms, scale_to_r0(Ms, args.r0), pop)
        bn, _ = marginal_ranking(Mn, scale_to_r0(Mn, args.r0), pop)
        obs = dict(
            joint_half_l1=float(np.abs(es - en).sum() / 2),
            eigvec_cosine=cos,
            eigvec_kendall=kendall_tau(vs, vn),
            eigvec_pearson=float(np.corrcoef(vs, vn)[0, 1]),
            marginal_ranking_kendall=kendall_tau(bs, bn),
            rank1_same=bool(int(np.argmax(bs)) == int(np.argmax(bn))),
            top3_overlap=len(set(top_k(bs)) & set(top_k(bn))),
            # every d_ is THIS 465-sample minus the national value, so the
            # observed Seoul deviation and the subsample null have the same sign
            # convention and can be read on one axis.
            **{f"d_{s}": float(st_s[s] - st_n[s]) for s in STATS})
        for s in STATS:
            obs[f"seoul_{s}"] = float(st_s[s])
            obs[f"national_{s}"] = float(st_n[s])

        # the exchangeability null
        n_sub = len(ebs)
        null = {k: [] for k in ("joint_half_l1", "eigvec_cosine",
                                "eigvec_kendall", "marginal_ranking_kendall")}
        null.update({f"d_{s}": [] for s in STATS})
        for _ in range(args.sub):
            idx = rng.choice(len(ebn), n_sub, replace=False)
            Cb, _ = contact_matrix(cn[idx], ebn[idx], ndn)
            Tb, _ = P32.symmetrise(Cb[np.ix_(sq, sq)], pop)
            eb_, _, _ = pm_null(Tb)
            stb = excess_stats(Tb)
            vb = ngm(Cb[np.ix_(sq, sq)])[2]
            Mb = P35.symmetrise(Cb[np.ix_(sq, sq)], pop)[0]
            qb = scale_to_r0(Mb, args.r0)
            bb, _ = marginal_ranking(Mb, qb, pop) if np.isfinite(qb) else (bn, None)
            null["joint_half_l1"].append(float(np.abs(eb_ - en).sum() / 2))
            null["eigvec_cosine"].append(
                float(vb @ vn / (np.linalg.norm(vb) * np.linalg.norm(vn))))
            null["eigvec_kendall"].append(kendall_tau(vb, vn))
            null["marginal_ranking_kendall"].append(kendall_tau(bb, bn))
            for s in STATS:
                null[f"d_{s}"].append(float(stb[s] - st_n[s]))

        summ = {}
        for k, v in null.items():
            v = np.array(v, float)
            o = obs[k]
            # two-sided: how extreme is Seoul among 465-person subsamples?
            p_lo = float((v <= o).mean())
            p_hi = float((v >= o).mean())
            summ[k] = dict(
                observed=float(o), null_median=float(np.median(v)),
                null_lo=float(np.percentile(v, 2.5)),
                null_hi=float(np.percentile(v, 97.5)),
                p_two_sided=float(min(1.0, 2 * min(p_lo, p_hi))),
                z=float((o - v.mean()) / v.std(ddof=1)) if v.std(ddof=1) else np.nan,
                inside_null=bool(np.percentile(v, 2.5) <= o <= np.percentile(v, 97.5)))
        diverge[str(ym)] = dict(observed=obs, null=summ, n_subsample=args.sub,
                                subsample_size=int(n_sub))
        print(f"\n  {ym}  {'quantity':<28}{'Seoul':>11}{'null median':>13}"
              f"{'null 95%':>22}{'p':>7}  verdict")
        for k in ("joint_half_l1", "eigvec_cosine", "eigvec_kendall",
                  "marginal_ranking_kendall", "d_assortativity", "d_nmi",
                  "d_half_l1", "d_cramers_v"):
            s_ = summ[k]
            print(f"  {'':<6}{k:<28}{s_['observed']:>11.4f}"
                  f"{s_['null_median']:>13.4f}"
                  f"   [{s_['null_lo']:>+8.4f}, {s_['null_hi']:>+8.4f}]"
                  f"{s_['p_two_sided']:>7.3f}  "
                  f"{'inside' if s_['inside_null'] else 'OUTSIDE'}")
        n_out = sum(1 for k in summ if not summ[k]["inside_null"])
        diverge[str(ym)]["n_outside_null"] = n_out
        diverge[str(ym)]["n_tested"] = len(summ)
        print(f"  {'':<6}--> {n_out} of {len(summ)} quantities put Seoul "
              f"OUTSIDE the 95% band of a random 465-person national subsample")
    out["seoul_vs_national"] = diverge

    # WHICH WAY IT CUTS. "The two survey populations differ" is only decisive
    # against the whole thing being measured. So: the Seoul->national shift in
    # each statistic, divided by the ENTIRE passive excess in that statistic.
    # Above 1 means switching the comparator population moves the comparator by
    # more than the signal the paper is trying to report.
    print("\n  which way it cuts -- the Seoul->national population shift as a")
    print("  multiple of the ENTIRE passive excess in the same statistic:")
    cut = {}
    for ym in SURVEY_MONTHS:
        row = {}
        for s in STATS:
            shift = abs(diverge[str(ym)]["observed"][f"d_{s}"])
            pas = passive[ym]["ex"][s]
            row[s] = dict(shift=float(shift), passive=float(pas),
                          multiple=float(shift / pas) if pas else np.nan,
                          # and against the sampling noise the switch buys down
                          multiple_of_null_halfwidth=float(
                              shift / max(1e-30, 0.5 * (
                                  diverge[str(ym)]["null"][f"d_{s}"]["null_hi"]
                                  - diverge[str(ym)]["null"][f"d_{s}"]["null_lo"]))))
        cut[str(ym)] = row
        print(f"  {ym}  {'statistic':<16}{'|shift|':>10}{'passive':>10}"
              f"{'x passive':>11}{'x null halfwidth':>18}")
        for s in STATS:
            r_ = row[s]
            print(f"  {'':<6}{s:<16}{r_['shift']:>10.4f}{r_['passive']:>10.4f}"
                  f"{r_['multiple']:>10.2f}x{r_['multiple_of_null_halfwidth']:>17.2f}x")
    out["divergence_vs_passive_signal"] = cut

    # ------------------------------- 40.6 Part B: the fourth pillar, confirmed
    hdr("40.6 FOURTH PILLAR at n=1,987 -- the symmetric version")
    print("  p35 35.4 already ran a national arm, but its between-matrix tau was")
    print("  the SEOUL survey's, carried across so the intervals could be read on")
    print("  one axis; the memo labels it a power statement, not a result. The")
    print("  symmetric switch requires the between-matrix tau to come from the")
    print("  NATIONAL survey's own ranking. That is what is computed here.")
    pillar = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        # the passive plan, recomputed here (SLSQP is deterministic, no rng), so
        # this section does not depend on results_p35.json's stored allocation.
        Cpd = mats[(ym, "passive_dong")]
        qpd = scale_to_r0(Cpd, args.r0)
        u_p, _, _ = optimal_allocation(Cpd, qpd, pop, budget_frac=args.budget)
        for scope, _ in SCOPES:
            cube, eb, nd = cubes[(ym, scope)]
            C0, _ = contact_matrix(cube, eb, nd)
            Cm, _ = P35.symmetrise(C0[np.ix_(sq, sq)], pop)
            q0 = scale_to_r0(Cm, args.r0)
            base, _ = marginal_ranking(Cm, q0, pop)
            between = []
            for tag in ("passive_dong", "passive_gu"):
                sp_, _ = marginal_ranking(mats[(ym, tag)],
                                          scale_to_r0(mats[(ym, tag)], args.r0),
                                          pop)
                between.append(dict(pair=f"survey_{scope} vs {tag}",
                                    tau=kendall_tau(base, sp_),
                                    top3_a=[lbl[i] for i in top_k(base)],
                                    top3_b=[lbl[i] for i in top_k(sp_)],
                                    top3_overlap=len(set(top_k(base))
                                                     & set(top_k(sp_))),
                                    rank1_same=bool(top_k(base)[0]
                                                    == top_k(sp_)[0])))
            taus, top1, tops3, regs, nulls = [], [], [], [], []
            for _ in range(args.boot35):
                idx = rng.integers(0, len(eb), len(eb))
                Cb, _ = contact_matrix(cube[idx], eb[idx], nd)
                Cb, _ = P35.symmetrise(Cb[np.ix_(sq, sq)], pop)
                qb = scale_to_r0(Cb, args.r0)
                if not np.isfinite(qb):
                    continue
                s_, _ = marginal_ranking(Cb, qb, pop)
                taus.append(kendall_tau(base, s_))
                top1.append(int(np.argmax(s_)))
                tops3.append(top_k(s_))
                # the regret and the null it has to clear, exactly as 35.5
                ub, vb, _ = optimal_allocation(Cb, qb, pop,
                                               budget_frac=args.budget)
                z0 = float((final_size(Cb, qb) * pop).sum() / pop.sum())
                zp = float((final_size(Cb, qb, u_p) * pop).sum() / pop.sum())
                regs.append((zp - vb) / (z0 - vb) if z0 > vb else np.nan)
                jdx = rng.integers(0, len(eb), len(eb))
                Cj, _ = contact_matrix(cube[jdx], eb[jdx], nd)
                Cj, _ = P35.symmetrise(Cj[np.ix_(sq, sq)], pop)
                qj = scale_to_r0(Cj, args.r0)
                uj, _, _ = optimal_allocation(Cj, qj, pop,
                                              budget_frac=args.budget)
                zj = float((final_size(Cb, qb, uj) * pop).sum() / pop.sum())
                nulls.append((zj - vb) / (z0 - vb) if z0 > vb else np.nan)
            taus = np.array(taus)
            regs = np.array([r for r in regs if np.isfinite(r)])
            nulls = np.array([r for r in nulls if np.isfinite(r)])
            lo = float(np.percentile(taus, 2.5))
            bt = [r["tau"] for r in between]
            n_ego_b = np.bincount(eb, minlength=len(AGES))[sq]
            ent = dict(
                ym=ym, scope=scope, n_respondents=int(len(eb)),
                n_ego_min=int(n_ego_b.min()), n_ego_max=int(n_ego_b.max()),
                between=between, tau_between=bt,
                tau_within_median=float(np.median(taus)),
                tau_within=[lo, float(np.percentile(taus, 97.5))],
                top1_stability=float(np.mean([i == int(np.argmax(base))
                                              for i in top1])),
                top3_mean_overlap=float(np.mean([len(set(t) & set(top_k(base)))
                                                 for t in tops3])),
                buy_first=lbl[int(np.argmax(base))],
                verdict=("separable" if bt and max(bt) < lo else "NOT separable"),
                regret=dict(
                    r0=args.r0, median=float(np.median(regs)),
                    ci=[float(np.percentile(regs, 2.5)),
                        float(np.percentile(regs, 97.5))],
                    null_median=float(np.median(nulls)),
                    null_ci=[float(np.percentile(nulls, 2.5)),
                             float(np.percentile(nulls, 97.5))],
                    exceeds_null=bool(np.percentile(regs, 2.5)
                                      > np.percentile(nulls, 97.5))))
            pillar[f"{ym}|{scope}"] = ent
            print(f"\n  {ym} [{scope}, {len(eb)} respondents, "
                  f"{n_ego_b.min()}-{n_ego_b.max()} egos per band]  buy first: "
                  f"{ent['buy_first']}")
            print(f"    within-survey tau  median {ent['tau_within_median']:+.3f}"
                  f"  95% [{lo:+.3f}, {ent['tau_within'][1]:+.3f}]")
            print(f"    between-matrix tau (this scope's OWN ranking): "
                  f"{', '.join(f'{t:+.3f}' for t in bt)}")
            print(f"    top-1 stability {ent['top1_stability']:.0%}, mean top-3 "
                  f"overlap {ent['top3_mean_overlap']:.2f}/3")
            print(f"    VERDICT: between-matrix difference is "
                  f"{'LARGER than' if ent['verdict'] == 'separable' else 'INSIDE'}"
                  f" this arm's own sampling noise -> {ent['verdict']}")
            r_ = ent["regret"]
            print(f"    regret of the passive plan at R0={args.r0}: median "
                  f"{r_['median']:.1%}, 95% [{r_['ci'][0]:.1%}, {r_['ci'][1]:.1%}]")
            print(f"    null (a SECOND draw of the same survey): median "
                  f"{r_['null_median']:.1%}, 95% [{r_['null_ci'][0]:.1%}, "
                  f"{r_['null_ci'][1]:.1%}]")
            print(f"    --> the cost of using passive data "
                  f"{'EXCEEDS' if r_['exceeds_null'] else 'does NOT exceed'} the "
                  f"cost of re-running the survey on {len(eb)} new people")
    out["pillar4"] = pillar
    n_sep = sum(1 for k, v in pillar.items()
                if v["scope"] == "national" and v["verdict"] == "separable")
    n_exc = sum(1 for v in pillar.values() if v["regret"]["exceeds_null"])
    print(f"\n  national arm: {n_sep}/{len(SURVEY_MONTHS)} months separable on "
          f"the symmetric between-matrix tau")
    print(f"  regret exceeding the resample-the-survey null: {n_exc}/"
          f"{len(pillar)} cells")
    out["pillar4_summary"] = dict(
        national_months_separable=n_sep,
        cells_regret_exceeds_null=n_exc, cells=len(pillar))

    # ------------------------------------------------- 40.7 downstream inventory
    inv = [
        dict(where="eda/p31_report_audit.py:325-326",
             what="pinned tuple (mi, n_dev, n_contacts, survey MI, MM bias, "
                  "perm median, perm p95) for both months",
             seoul="(0.0196, 2369954, 8281, 0.6907, 0.0162, 0.0713, 0.0921) / "
                   "(0.0177, 2368375, 3579, 0.6992, 0.0351, 0.0868, 0.1055)",
             national=[dict(
                 ym=ym,
                 n_contacts=cells[f"{ym}|national"]["n_contacts"],
                 survey_mi=cells[f"{ym}|national"]["survey_mi_bits"],
                 mm_bias=cells[f"{ym}|national"]["mm_bias_bits"],
                 perm_median=cells[f"{ym}|national"]["mi_perm_median"],
                 perm_p95=cells[f"{ym}|national"]["mi_perm_p95"])
                 for ym in SURVEY_MONTHS]),
        dict(where="eda/p31_report_audit.py:338-340",
             what='the ratios ("202312", 3.6), ("202402", 4.9)',
             seoul=[3.6, 4.9],
             national=[round(cells[f"{ym}|national"]["floor_over_passive_median"], 3)
                       for ym in SURVEY_MONTHS]),
        dict(where="eda/p31_report_audit.py:289-301",
             what="the statistic-dependence table: passive/survey/ratio for "
                  "half_l1, cramers_v, assortativity, nmi",
             seoul="survey column 0.3285/0.3634/0.2227/0.1798 (202312)",
             national=[dict(ym=ym, **{s: round(comp[str(ym)]["national"][s], 4)
                                      for s in ("half_l1", "cramers_v",
                                                "assortativity", "nmi")})
                       for ym in SURVEY_MONTHS]),
        dict(where="eda/p31_report_audit.py:307-308, 320-321",
             what="rank table for survey|YYYYMM|WE|seoul (sigma1 share, "
                  "sigma2/sigma1, best rank-1 resid, pm resid)",
             seoul=[0.4679, 0.601, 0.7294, 0.7355],
             national=[dict(ym=ym, sigma1_share=round(comp[str(ym)]["national"]["sigma1_share"], 4),
                            sigma2_over_sigma1=round(comp[str(ym)]["national"]["sigma2_over_sigma1"], 4),
                            best_rank1_resid=round(comp[str(ym)]["national"]["best_rank1_resid"], 4),
                            pm_rank1_resid=round(comp[str(ym)]["national"]["pm_rank1_resid"], 4))
                       for ym in SURVEY_MONTHS]),
        dict(where="eda/p31_report_audit.py:461-463",
             what="p33 NMI gap at theta=0 (34.7) and theta=+1 (16.5); the whole "
                  "R2_gaps_vs_survey family is built from the SEOUL survey stat",
             seoul=34.7,
             national=round([r for r in th["202312"]
                             if r["theta"] == 0.0][0]["nmi"]["national"], 3)),
        dict(where="eda/p31_report_audit.py:476-482, 517-519",
             what="p35 between-matrix taus 0.429 / 0.390 / 0.600 -- these are "
                  "Seoul-survey-vs-passive; the symmetric switch replaces them",
             seoul=[0.429, 0.390, 0.600],
             national=[round(t, 4) for ym in SURVEY_MONTHS
                       for t in pillar[f"{ym}|national"]["tau_between"]]),
        dict(where="eda/p31_report_audit.py:493-499",
             what="p35 within-tau lower bounds 0.619 / 0.410 and 'p35 seoul "
                  "egos 465'",
             seoul=[0.619, 0.410, 465],
             national=[round(pillar[f"{ym}|national"]["tau_within"][0], 4)
                       for ym in SURVEY_MONTHS] + [1987]),
        dict(where="eda/p31_report_audit.py:187-194",
             what="p27 survey assortativity 0.2227/0.1867 and the published "
                  "bootstrap CI 0.194-0.248 / 0.163-0.211 -- all Seoul",
             seoul=[0.2227, 0.1867],
             national=[round(comp[str(ym)]["national"]["assortativity"], 4)
                       for ym in SURVEY_MONTHS]),
        dict(where="paper/related_work_positioning.md:117",
             what='"0.0196 bits 對調查置換下限 0.0713 bits，小 3.6 倍"',
             seoul="0.0713, 3.6x",
             national=f"{cells['202312|national']['mi_perm_median']:.4f}, "
                      f"{cells['202312|national']['floor_over_passive_median']:.2f}x"),
        dict(where="eda/memo/phase32-pmix.md:41-45, 60, 79-84",
             what="the statistic table, the rank table, the floor table and the "
                  "headline sentence, all Seoul",
             seoul="see file", national="see 40.2/40.3 above"),
        dict(where="eda/memo/phase35-seir.md:7, 64-95",
             what="'Chae Seoul 465', the separability table and the regret table",
             seoul="see file", national="see 40.6 above"),
        dict(where="eda/memo/phase7-survey.md:50-51, 106",
             what="the four indicators per month and the explicit statement "
                  "that the Seoul subsample n=465 is the primary analysis",
             seoul="see file", national="see 40.3 above"),
        dict(where="eda/p27_survey.py:52",
             what="docstring: 'the Seoul subsample is the primary'",
             seoul="prose", national="prose"),
        dict(where="eda/p32_pmix.py:32, 421-430, 471-472",
             what='the docstring "465 Seoul respondents", '
                  'ego_cubes(ego, d, True, ...) in 32.4b, and '
                  'sk = f"survey|{ym}|WE|seoul" in 32.4 / 32.4b / 32.5',
             seoul="hardcoded True / 'seoul'", national="would become a flag"),
        dict(where="eda/p35_seir.py:35-42, 597-612, 793-800",
             what="'465 respondents' framing, and the national arm being "
                  "labelled a power reference rather than a result",
             seoul="see file", national="see 40.6 above"),
        dict(where="eda/README.md:226",
             what="the p35 one-liner ending '465 人證不出「比重做調查更糟」'",
             seoul="465", national="1,987 -- and the conclusion is unchanged"),
        dict(where="eda/memo/numbers-20260820.md",
             what="the number ledger carries every one of the above",
             seoul="see file", national="regenerate after any switch"),
        dict(where="Email_Discussion/advisor_report_20260819.md:19, 57",
             what="the reported headline sentence, already sent",
             seoul="0.0713, 3.6x", national="needs a correction in the next letter"),
    ]
    out["downstream_inventory"] = inv
    hdr("40.7 downstream inventory (report only -- nothing here is edited)")
    for i in inv:
        print(f"  {i['where']}\n      {i['what']}")

    # ------------------------------------------------------------------ figure
    make_figure(cells, comp, diverge, pillar, passive)

    out["fail"] = FAIL
    with open(f"{RESULTS}/results_p40.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nwrote {RESULTS}/results_p40.json and {FIG}/p40_natfloor.png")

    hdr("THE THREE ANSWERS")
    p1 = cells[f"{PRIMARY['ym']}|{PRIMARY['scope']}"]
    print(f"  1. headline at national scope: "
          f"{'SURVIVES' if p1['survives_median'] else 'DOES NOT SURVIVE'} "
          f"at the primary cell "
          f"(passive {p1['passive_mi_bits']:.5f} vs floor "
          f"{p1['mi_perm_median']:.5f}); "
          + "; ".join(
              f"{ym}/{sc} {'survives' if cells[f'{ym}|{sc}']['survives_median'] else 'flips'}"
              for ym in SURVEY_MONTHS for sc, _ in SCOPES))
    print(f"  2. Seoul vs national survey matrices: "
          + "; ".join(
              f"{ym} marginal-ranking tau "
              f"{diverge[str(ym)]['observed']['marginal_ranking_kendall']:+.3f}, "
              f"{diverge[str(ym)]['n_outside_null']}/"
              f"{diverge[str(ym)]['n_tested']} quantities outside the "
              f"exchangeability null" for ym in SURVEY_MONTHS))
    print(f"  3. fourth pillar at n=1,987: "
          + "; ".join(f"{ym} {pillar[f'{ym}|national']['verdict']}, regret "
                      f"{'exceeds' if pillar[f'{ym}|national']['regret']['exceeds_null'] else 'does NOT exceed'}"
                      f" the null" for ym in SURVEY_MONTHS))


def make_figure(cells, comp, diverge, pillar, passive):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # (a) the four cells
    ax = axes[0]
    x = np.arange(4)
    keys = [f"{ym}|{sc}" for ym in SURVEY_MONTHS for sc in ("seoul", "national")]
    med = [cells[k]["mi_perm_median"] for k in keys]
    p95 = [cells[k]["mi_perm_p95"] for k in keys]
    lo = [cells[k]["mi_perm_lo"] for k in keys]
    hi = [cells[k]["mi_perm_hi"] for k in keys]
    cols = ["#4C6E8A", "#A8434E", "#4C6E8A", "#A8434E"]
    for i in x:
        ax.plot([i, i], [lo[i], hi[i]], color=cols[i], lw=2.2, alpha=.55,
                solid_capstyle="butt")
        ax.plot(i, med[i], "o", color=cols[i], ms=8, zorder=3)
        ax.plot(i, p95[i], "_", color=cols[i], ms=16, mew=2.2, zorder=3)
        c = cells[keys[i]]
        ax.annotate(f"{c['floor_over_passive_median']:.2f}x\n"
                    f"{c['passive_frac_below_perm_null']:.0%} below",
                    (i, med[i]), textcoords="offset points", xytext=(11, -4),
                    fontsize=7.5, color=cols[i], va="top")
    for ym, ls in zip(SURVEY_MONTHS, ["-", "--"]):
        pas = passive[ym]["ex"]["mi_bits"]
        ax.axhline(pas, color="#222", ls=ls, lw=1.3,
                   label=f"passive excess {ym} = {pas:.4f}")
    ax.set_xticks(x)
    ax.set_xlim(-0.55, 3.85)
    ax.set_xticklabels([f"{k.split('|')[0]}\n{k.split('|')[1]}\n"
                        f"n={cells[k]['n_ego']}" for k in keys], fontsize=8)
    ax.set_ylabel("MI (bits)")
    ax.set_yscale("log")
    ax.set_title("(a) survey permutation floor: median, p95 (dash),\n"
                 "2.5-97.5% (bar), against the whole passive excess",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper center")
    ax.grid(alpha=.3, axis="y")

    # (b) Seoul vs national point statistics, with the exchangeability null
    ax = axes[1]
    sts = ("half_l1", "cramers_v", "assortativity", "nmi")
    w = 0.35
    for j, ym in enumerate(SURVEY_MONTHS):
        xs = np.arange(len(sts)) + (j - .5) * w
        s_ = [comp[str(ym)]["seoul"][s] for s in sts]
        n_ = [comp[str(ym)]["national"][s] for s in sts]
        ax.plot(xs, s_, "o", color=["#0E7C86", "#BC8034"][j], ms=7,
                label=f"Seoul {ym}")
        ax.plot(xs, n_, "s", color=["#0E7C86", "#BC8034"][j], ms=7, mfc="none",
                mew=2, label=f"national {ym}")
        for k, s in enumerate(sts):
            ax.plot([xs[k], xs[k]], [s_[k], n_[k]],
                    color=["#0E7C86", "#BC8034"][j], lw=1.1, alpha=.6)
    ax.set_xticks(np.arange(len(sts)))
    ax.set_xticklabels(["half-L1", "Cramer's V", "assortativity", "nMI"],
                       fontsize=8.5)
    ax.set_ylabel("value")
    ax.set_title("(b) the two survey matrices are not the same object:\n"
                 "national is MORE concentrated on every statistic",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=.3, axis="y")

    # (c) the fourth pillar
    ax = axes[2]
    ks = [f"{ym}|{sc}" for ym in SURVEY_MONTHS for sc in ("seoul", "national")]
    y = np.arange(len(ks))[::-1]
    for i, k in enumerate(ks):
        e = pillar[k]
        c = "#4C6E8A" if e["scope"] == "seoul" else "#A8434E"
        ax.plot(e["tau_within"], [y[i], y[i]], color=c, lw=3.5, alpha=.5,
                solid_capstyle="butt")
        ax.plot(e["tau_within_median"], y[i], "o", color=c, ms=7)
        for t in e["tau_between"]:
            ax.plot(t, y[i], "x", color="#222", ms=9, mew=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k.split('|')[0]} {k.split('|')[1]}\n"
                        f"({pillar[k]['verdict']})" for k in ks], fontsize=8)
    ax.set_xlabel("Kendall tau of the marginal ranking")
    ax.set_title("(c) fourth pillar: within-arm bootstrap interval (bar)\n"
                 "against its OWN between-matrix tau (x)", fontsize=9.5)
    ax.grid(alpha=.3, axis="x")

    fig.suptitle("Phase 40 — switching the survey scope from 465 Seoul to "
                 "1,987 national respondents, symmetrically", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "p40_natfloor.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
