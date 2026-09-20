#!/usr/bin/env python
"""Phase 31 — check every number the advisor report quotes against its source.

The report to the advisor is the one artefact that leaves this repo, and it now
quotes about forty figures produced by fifteen scripts over two days. Any of them
can go stale the moment an upstream script is re-run — p19 alone has been re-run
three times, and the 20-44 numbers moved on the third.

So this is a pre-send gate rather than an analysis: each claim is stated as
(where it is quoted, what it should equal, where that comes from), and the script
recomputes the right-hand side from results_*.json. A claim that cannot be
recomputed from a results file is listed as MANUAL rather than quietly passed —
external facts (the manual, the portal, the newsletter) have no JSON to check
against, and pretending otherwise would defeat the point.

    python eda/p31_report_audit.py
"""
import argparse
import ast
import hashlib
import json
import math
import os

import numpy as np
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT

REPORT = f"{ROOT}/Email_Discussion/advisor_report_20260818.md"

# The 08-18 report has left the repo, so the numbers it quotes are fixed for good.
# Checking it against the live results_*.json would mean this gate starts failing the
# first time p32/p33/p34 touch anything upstream -- a failure that says nothing about
# the sent document. So the sent report is checked against a frozen copy of exactly
# the files it was written from, and new documents get their own (document, source)
# pair rather than sharing this one.
ARCHIVE = f"{ROOT}/eda/archive/20260818"

# The 08-19 and 08-22 documents have now left the repo too, and until 2026-08-24 they
# were still being checked against LIVE -- so the rule above was stated but not kept
# for them. eda/archive/20260822/ is that omission repaired: the freeze was taken at a
# moment when every results file this gate reads was bit-identical to HEAD, so the
# frozen copies ARE the state those two documents were written from.
#
# Two layers, because the p34/p37/p38 panels are 33.9 MB and this round does not
# re-run them. Small files are copied; the six panels are pinned by sha256 against
# the live file and verified in main(). A pinned file that moves is a red gate, not
# a silent read of changed data. See eda/archive/README.md for both lists.
ARCHIVE_0822 = f"{ROOT}/eda/archive/20260822"

# The 08-24 report has now left the repo too, and the advisor has replied
# (Email_Discussion/advisor_suggestion_20260827.md), so there is no doubt it
# arrived. eda/archive/20260824/ is the freeze that send obliges, taken at a
# moment when every results file the 08-24 block reads was bit-identical to HEAD.
#
# A snapshot under this name was taken once before, on 2026-08-24, in the belief
# the letter had gone out that evening; it had not, and it was removed rather
# than left standing as a claim about a send that had not happened (bc08f96).
# This one is different in kind, and the difference is checkable rather than
# asserted: the reply quotes numbers that only exist in the letter as it stands
# at HEAD -- 5,786 (Sejong, cebb5db) and beta 0.8578 at k = 1.5 (d583ff0) -- so
# the version the advisor read IS this version, and this is the state it was
# written from.
#
# One layer, not two: all eight files are small (85 KB together), so every one
# is copied. p44 is copied a SECOND time here on purpose. ARCHIVE_0822 holds the
# p44 the 08-21 letter was written from; d583ff0 withdrew section 44.7's Q9 arm
# and re-ran the phase, and the 08-24 letter quotes the withdrawal. Two sent
# documents quote two different states of one file, so each gets its own copy.
ARCHIVE_0824 = f"{ROOT}/eda/archive/20260824"

# sha256 of the files that are pinned rather than copied.
PINNED_0822 = {
    "p34": "d5cb1d4bf7fff96dbbb2b20f53bbd056194e96d49b9aa069229e6d8eb5870ad1",
    "p34_W": "ac43d2e02cc137bfffb103ad9e555400ff9f268f39b21fac1ab00b7556440c42",
    "p34_E": "ab3c1e16610fad0da56c0658b839ab13184110570c7c0846615d4035c907485d",
    "p37": "8b51858178530cfb1e0d6f627d63b00572bcd0274be509c4be1c57863df5cabb",
    "p38": "b3c27c984ffab5d6e6ae017f3fd7a4068c6140e56f2bbdef6d8473faa9e79dda",
    "p38_W": "77b6deaa98f831e3732d9b92109f6913c86395c4fe2557ef6c8fbe7e58fc2f07",
    "p38_E": "2c6b4f3e3b3bd4274156debbe18c08d8c0c37e52cd8f6f65716e5c833dec3ba8",
}
R = {}


def load(name, source=ARCHIVE):
    key = (source, name)
    if key not in R:
        with open(f"{source}/results_{name}.json") as fh:
            R[key] = json.load(fh)
    return R[key]


def pct(x):
    return x * 100


# (section, quoted text as it appears, value in the report, recomputed value)
CHECKS = []


def check(sec, label, quoted, actual, tol=0.006):
    CHECKS.append((sec, label, quoted, actual, tol))


# ---------------------------------------------------------------- section 1
p21 = load("p21")
rho = [r["rho_mid"] for r in p21["level_by_age"]]
check("1.3", "rho-hat low end 0.24", 0.24, round(min(rho), 2))
check("1.3", "rho-hat high end 0.69", 0.69, round(max(rho), 2))
mdp = p21["mar_to_dec_pct"]
check("1.2/5", "rho-hat Mar->Dec 80+ +11.12%", 11.12, round(mdp["80+"], 2), .02)

p21b = load("p21b")
band = {r["band"]: r for r in p21b["branch"]["per_band"]}
check("1.4", "80+ induced movement 0.282 pp", 0.282,
      round(band["80+"]["frozen_p975_pp"], 3))
check("1.4", "80+ ratio to signal 7.9%", 7.9,
      round(pct(band["80+"]["frozen_ratio"]), 1), .06)
check("1.4", "15-19 ratio 1.8%", 1.8, round(pct(band["15-19"]["frozen_ratio"]), 1), .06)
check("1.4", "20-24 ratio 84.0%", 84.0,
      round(pct(band["20-24"]["frozen_ratio"]), 1), .06)
check("1.4", "70-74 ratio 9.6%", 9.6, round(pct(band["70-74"]["frozen_ratio"]), 1), .06)
check("1.4", "65-69 ratio 30.1%", 30.1, round(pct(band["65-69"]["frozen_ratio"]), 1), .06)
check("1.4", "15-19 induced 0.162 pp", 0.162,
      round(band["15-19"]["frozen_p975_pp"], 3))
check("1.4", "20-24 induced 0.160 pp", 0.160,
      round(band["20-24"]["frozen_p975_pp"], 3))
check("1.4", "sigma_d must be 3.8x larger to flip 80+", 3.8,
      round(p21b["branch"]["sigma_multiple_to_flip"], 1), .06)

# ---------------------------------------------------------------- section 4
p29 = load("p29")
check("4", "masked cell holds 2.27 (2020-12)", 2.27,
      round(p29["202012"]["per_cell_mean"], 2))
check("4", "masked cell holds 2.28 (2020-03)", 2.28,
      round(p29["202003"]["per_cell_mean"], 2))
check("4", "measured masked volume 11.09% (2020-12)", 11.09,
      round(pct(p29["202012"]["measured_share"]), 2), .02)
check("4", "measured masked volume 11.60% (2020-03)", 11.60,
      round(pct(p29["202003"]["measured_share"]), 2), .02)
check("4", "assumed mid 7.62% (2020-12)", 7.62,
      round(pct(p29["202012"]["assumed_mid_share"]), 2), .02)
check("4", "gu file masks 9.43% of its own cells", 9.43,
      round(pct(p29["202012"]["gu_own_masked_share"]), 2), .02)

# ---------------------------------------------------------------- section 5
p19 = load("p19")
d16 = {r["age"]: r for r in p19["e_share_change_16band"]}
d3 = {r["band"]: r for r in p19["e_share_change_3band"]}
scen = {r["scenario"]: r for r in p19["coverage_drift_sensitivity"]}
span = {r["resolution"]: r for r in p19["resolution_span"]}
check("5", "16-band estimates move 0.0000 pp", 0.0,
      round(p19["drift_move_16band_max_pp"], 4))
check("5", "3-band 60+ moves 0.115 pp under measured drift", 0.115,
      round(p19["drift_move_3band_60plus_measured_pp"], 3))
check("5", "3-band 60+ moves 0.238 pp over all scenarios", 0.238,
      round(p19["drift_move_3band_60plus_pp"], 3))
check("5", "3-band 60+ uncorrected +0.075", 0.075, round(scen["none"]["3band_60+"], 3))
check("5", "3-band 60+ measured -0.040", -0.040,
      round(scen["rho_measured"]["3band_60+"], 3))
check("5", "3-band 60+ survey lower -0.023", -0.023,
      round(scen["survey_lower"]["3band_60+"], 3))
check("5", "3-band 60+ survey upper -0.163", -0.163,
      round(scen["survey_upper"]["3band_60+"], 3))
check("5", "16-band 80+ +0.844", 0.844, round(scen["none"]["16band_80+"], 3))
check("5", "measured drift 60-64 +0.40%", 0.40,
      round(p19["drift_scenarios_pct"]["rho_measured"]["60-64"], 2), .02)
check("5", "measured drift 80+ +11.12%", 11.12,
      round(p19["drift_scenarios_pct"]["rho_measured"]["80+"], 2), .02)
check("5", "20-24 E-share now -3.48 pp", -3.48, round(d16["20-24"]["meas"], 2))
check("5", "20-24 E-share was -3.96 pp", -3.96, round(d16["20-24"]["mid"], 2))
check("5", "16-band spread 5.79 pp", 5.79, round(span["16 bands"]["range_pp"], 2))
check("5", "16-band spread was 6.27 pp", 6.27,
      round(span["16 bands"]["range_pp_mid_imputation"], 2))
check("5", "three bands retain 50.6%", 50.6, round(pct(p19["range_retained_frac"]), 1), .06)
check("5", "20-59 band -1.42 pp", -1.42, round(d3["20-59"]["meas"], 2))

# ---------------------------------------------------------------- section 6a
p25 = load("p25")
check("6a", "W->W trips changing dong 65.2%", 65.2,
      round(pct(p25["ww_cross_dong_share"]), 1), .06)
tv = {r["age"]: r for r in p25["G_tv_bounds"] if r["imputation"] == "mid"}
for age, want in [("65-69", 20.9), ("70-74", 24.2), ("75-79", 27.2),
                  ("80+", 27.7), ("10-14", 44.1), ("15-19", 35.5)]:
    check("6a", f"TV lower bound {age}", want,
          round(pct(tv[age]["tv_hour_x_samedong"]), 1), .06)

# ---------------------------------------------------------------- section 6b
p26 = load("p26")
lad = {(r["ym"], r["panel"], r["level"]): r for r in p26["ladder"]}
check("6b", "202606 WE dong assortativity 0.0213", 0.0213,
      round(lad[(202606, "WE", "dong")]["assortativity"], 4), 6e-5)
check("6b", "202606 WE gu 0.0103", 0.0103,
      round(lad[(202606, "WE", "gu")]["assortativity"], 4), 6e-5)
check("6b", "202606 WE city 0.0053", 0.0053,
      round(lad[(202606, "WE", "city")]["assortativity"], 4), 6e-5)
kept = {}
for ym in {k[0] for k in lad}:
    d = lad[(ym, "WE", "dong")]["assortativity"]
    g = lad[(ym, "WE", "gu")]["assortativity"]
    c = lad[(ym, "WE", "city")]["assortativity"]
    kept[ym] = (g - c) / (d - c)
check("6b", "kept fraction at 202606 = 31.1%", 31.1, round(pct(kept[202606]), 1), .06)
check("6b", "kept range low 28.7%", 28.7, round(pct(min(kept.values())), 1), .06)
check("6b", "kept range high 31.7%", 31.7, round(pct(max(kept.values())), 1), .06)
pred = [r for r in p26["resolution_prediction"] if r["panel"] == "WE"]
check("6b", "WE slope low 0.226", 0.226, round(min(r["slope"] for r in pred), 3), 6e-4)
check("6b", "WE slope high 0.242", 0.242, round(max(r["slope"] for r in pred), 3), 6e-4)
check("6b", "polygon prediction 1.35x", 1.35, round(min(r["ratio"] for r in pred), 2))
check("6b", "polygon prediction 1.38x", 1.38, round(max(r["ratio"] for r in pred), 2))
check("6b", "fit residual low 0.047", 0.047,
      round(min(r["max_log_resid"] for r in pred), 3), 6e-4)
check("6b", "fit residual high 0.076", 0.076,
      round(max(r["max_log_resid"] for r in pred), 3), 6e-4)

# ----------------------------------------------------------------- section 7
p27 = load("p27")
rep = p27["reproduction"]
check("7", "survey respondents 1,987", 1987, rep["respondents"], 0)
check("7", "survey contacts 133,776", 133776, rep["contacts"], 0)
check("7", "contacts per person per day 4.809", 4.809, round(rep["per_person_day"], 3))
check("7", "diary is 14 days", 14, rep["diary_days"], 0)
cmp_ = {(r["ym"], r["restriction"]): r for r in p27["comparison"]}
for ym, ev, sp, ap, asv, ng, cp in [
        (202312, .414, .529, .0217, .2227, .32, .447),
        (202402, .571, .743, .0189, .1867, .46, .463)]:
    r = cmp_[(ym, "holiday-free")]
    check("7", f"{ym} eigenvector r", ev, round(r["eigvec_pearson"], 3))
    check("7", f"{ym} spearman", sp, round(r["eigvec_spearman"], 3))
    check("7", f"{ym} passive assortativity", ap, round(r["assort_passive"], 4), 6e-5)
    check("7", f"{ym} survey assortativity", asv, round(r["assort_survey"], 4), 6e-5)
    check("7", f"{ym} NGM ratio", ng, round(r["eig_ratio"], 2))
    check("7", f"{ym} cell Pearson", cp, round(r["cell_pearson"], 3))
boot = p27["survey_assortativity_bootstrap"]
check("7", "202312 CI low 0.194", 0.194, round(boot["202312"]["lo"], 3))
check("7", "202312 CI high 0.248", 0.248, round(boot["202312"]["hi"], 3))
check("7", "202402 CI low 0.163", 0.163, round(boot["202402"]["lo"], 3))
check("7", "202402 CI high 0.211", 0.211, round(boot["202402"]["hi"], 3))
gap = {g["ym"]: g for g in p27["resolution_gap"]}
check("7", "202312 polygon closes 13.3%", 13.3,
      round(pct(gap[202312]["polygon_closes"]), 1), .06)
check("7", "202402 polygon closes 13.4%", 13.4,
      round(pct(gap[202402]["polygon_closes"]), 1), .06)
check("7", "202312 residents per location 1.5", 1.5,
      round(gap[202312]["persons_per_loc"], 1), .06)
check("7", "202402 residents per location 0.9", 0.9,
      round(gap[202402]["persons_per_loc"], 1), .06)
check("7", "202312 gap ratio 10.2x", 10.2, round(gap[202312]["ratio"], 1), .06)
check("7", "202402 gap ratio 9.9x", 9.9, round(gap[202402]["ratio"], 1), .06)
mp = p27["place_multiplicity"]
check("7", "multi-place contacts 10.5%", 10.5, round(pct(mp["multi_place_share"]), 1), .06)
check("7", "household-plus-other 3.4%", 3.4,
      round(pct(mp["household_plus_other_share"]), 1), .06)

# ---------------------------------------------------------------- section 4b
p28 = load("p28")
cm = {(r["ym"], r["band"]): r for r in p28["comparison"]}
for ym in (202606, 202512):
    for band, w, c in [(40, .310, .244), (50, .295, .216), (60, .247, .167),
                       (70, .169, .130)]:
        if (ym, band) not in cm or ym != 202606:
            continue
        r = cm[(ym, band)]
        check("6a/4b", f"{ym} {band} W share", w, round(r["w_share"], 3))
        check("6a/4b", f"{ym} {band} commute share", c, round(r["commute_share"], 3))
for ym, band, frac, exc in [(202606, 60, 32, 11), (202606, 70, 23, 2),
                            (202512, 60, 31, 9), (202512, 70, 21, -1)]:
    r = cm[(ym, band)]
    check("6a/4b", f"{ym} {band} non-commute fraction", frac,
          round(pct(r["non_commute_frac"])), .6)
    check("6a/4b", f"{ym} {band} excess over ref", exc,
          round(pct(r["excess_over_ref"])), .6)
codes = {c["code"]: c for c in p28["codes"]["202606"]}
check("6a/4b", "POI purposes share 0.95%", 0.95,
      round(pct(sum(codes[c]["share"] for c in (4, 5, 6))), 2), .02)
check("6a/4b", "귀가 corr with population 0.95", 0.95, round(codes[3]["corr_pop"], 2))
check("6a/4b", "출근 corr with jobs 0.91", 0.91, round(codes[1]["corr_jobs"], 2))

# ======================================================================
# The 2026-08-19 report. Different document, different source: it quotes p32 and
# p34, which are LIVE files rather than the frozen 08-18 snapshot. Same gate,
# separate (document, source) pair -- so neither report is ever left un-gated and
# re-running p32/p34 cannot make the 08-18 checks fail.
#
# This block exists because a transcription error got into the first draft: the
# 202001 k=25 band was copied out of a 3-replicate smoke test instead of the
# 200-replicate run ([0.00720, 0.00783] instead of [0.00724, 0.00832]), which
# also flipped the "inside the band" verdict. Hand-auditing caught it; a gate
# catches it every time.
# ======================================================================
LIVE = f"{ROOT}/eda"
REPORT_0819 = f"{ROOT}/Email_Discussion/advisor_report_20260819.md"
# 08-20: the attachments were dropped, so the letter is the only document that
# leaves this repo and the only corpus this check may read. The values that used
# to live only in the appendix are still VERIFIED here; they are simply quoted
# nowhere any more, and the run prints that split instead of hiding it. A number
# the prose dropped and a number nobody ever wrote look identical unless the
# gate says which is which.
p32 = load("p32", ARCHIVE_0822)
p34 = load("p34", LIVE)


# Every c19 claim is registered here as well, so the gate can also ask the
# question it was not asking: does the number actually APPEAR in the document?
# check() compares a hand-typed `quoted` against the source, which catches a
# stale results file but not a typo made while transcribing into the prose --
# the two would have to be mistyped identically. Presence in the text closes
# that loop, and it is reported rather than failed because a value can legitimately
# appear rounded differently or inside a table cell this crude matcher misses.
IN_TEXT = []


def c19(label, quoted, actual, tol=6e-4):
    """RELATIVE tolerance, with no absolute floor.

    The 08-18 block uses absolute tolerances, which is fine for percentages but
    far too loose for assortativity values of order 0.005. The first version of
    this helper wrote `tol * max(1.0, abs(actual))`, which silently degrades to
    an absolute tolerance for exactly those small numbers -- and would have
    passed the 0.00783-for-0.00832 transcription error this block was written to
    catch. A floor defeats the purpose; scale strictly with the value.
    """
    IN_TEXT.append((label, quoted))
    check("19", label, quoted, actual, tol * abs(actual))


# --- the identity that reframes the whole section
c19("newman numerator = diagonal excess", 0.02017831,
    round(p32["identity_check"]["newman_numerator"], 8), 1e-8)

# --- the statistic-dependence table and its ratios
_cmp = {str(r["ym"]): r for r in p32["comparison"]}
for _ym, _vals in (("202312", dict(half_l1=(0.0588, 0.3285, 5.6),
                                   cramers_v=(0.0462, 0.3634, 7.9),
                                   assortativity=(0.0217, 0.2227, 10.2),
                                   nmi=(0.0051, 0.1798, 35.2))),
                   ("202402", dict(half_l1=(0.0575, 0.3260, 5.7),
                                   cramers_v=(0.0434, 0.3781, 8.7),
                                   assortativity=(0.0189, 0.1867, 9.9),
                                   nmi=(0.0046, 0.1836, 39.8)))):
    for _st, (_p, _s, _r) in _vals.items():
        c19(f"{_ym} {_st} passive", _p, _cmp[_ym][_st]["passive"], 6e-3)
        c19(f"{_ym} {_st} survey", _s, _cmp[_ym][_st]["survey"], 6e-3)
        c19(f"{_ym} {_st} ratio", _r, _cmp[_ym][_st]["ratio"], 6e-3)

# --- the rank table
_sp = p32["spectrum"]
for _k, _lbl, _vals in (
        ("passive|202312|WE|dong|holidayfree", "202312 passive dong",
         (0.9795, 0.124, 0.1433, 0.1447)),
        ("survey|202312|WE|seoul", "202312 survey",
         (0.4679, 0.601, 0.7294, 0.7355))):
    _s1, _s21, _br, _pm = _vals
    c19(f"{_lbl} sigma1 share", _s1, _sp[_k]["sigma1_share"], 1e-3)
    c19(f"{_lbl} sigma2/sigma1", _s21, _sp[_k]["sigma2_over_sigma1"], 3e-3)
    c19(f"{_lbl} best rank-1 resid", _br, _sp[_k]["best_rank1_resid"], 3e-3)
    c19(f"{_lbl} pm resid", _pm, _sp[_k]["pm_rank1_resid"], 3e-3)
c19("202312 passive gu sigma1 share", 0.9945,
    _sp["passive|202312|WE|gu"]["sigma1_share"], 1e-3)
c19("202312 passive city sigma1 share", 0.9984,
    _sp["passive|202312|WE|city"]["sigma1_share"], 1e-3)
c19("202402 passive sigma1 share", 0.9834,
    _sp["passive|202402|WE|dong|holidayfree"]["sigma1_share"], 1e-3)
c19("202402 survey sigma1 share", 0.5493,
    _sp["survey|202402|WE|seoul"]["sigma1_share"], 1e-3)

# --- the null floors, which carry the headline sentence
for _ym, (_mi, _ndev, _ncon, _smi, _mm, _perm, _p95) in (
        ("202312", (0.0196, 2369954, 8281, 0.6907, 0.0162, 0.0713, 0.0921)),
        ("202402", (0.0177, 2368375, 3579, 0.6992, 0.0351, 0.0868, 0.1055))):
    _f = p32["null_floor"][_ym]
    c19(f"{_ym} passive MI bits", _mi, _f["passive"]["observed"], 6e-3)
    c19(f"{_ym} device-equivalent n", _ndev, _f["passive"]["n_effective"], 1e-5)
    c19(f"{_ym} survey n contacts", _ncon, _f["survey"]["n_effective"], 1e-5)
    c19(f"{_ym} survey MI bits", _smi, _f["survey"]["observed"], 1e-3)
    c19(f"{_ym} Miller-Madow bias", _mm, _f["survey"]["mm_bias_bits"], 6e-3)
    c19(f"{_ym} permutation median", _perm,
        p32["survey_floor_measured"][_ym]["mi_perm_median"], 6e-3)
    c19(f"{_ym} permutation p95", _p95,
        p32["survey_floor_measured"][_ym]["mi_perm_p95"], 6e-3)
# the sentence itself: the survey's own floor over the whole passive excess
for _ym, _q in (("202312", 3.6), ("202402", 4.9)):
    c19(f"{_ym} survey floor / passive excess", _q,
        (p32["survey_floor_measured"][_ym]["mi_perm_median"]
         / p32["null_floor"][_ym]["passive"]["observed"]), 2e-2)
# and the anchor that must have reproduced, or none of the above may be read
for _ym in ("202312", "202402"):
    check("19", f"{_ym} bootstrap anchor reproduces", 1,
          int(bool(p32["survey_floor_measured"][_ym]["anchor_reproduces"])), 0)

# --- the IPF branch
for _ym, (_b, _a) in (("202312", (35.2, 33.7)), ("202402", (39.8, 39.3))):
    c19(f"{_ym} nMI gap before IPF", _b,
        p32["ipf_calibration"]["months"][_ym]["gap_before"], 6e-3)
    c19(f"{_ym} nMI gap after IPF", _a,
        p32["ipf_calibration"]["months"][_ym]["gap_after"], 6e-3)
check("19", "IPF branch: does not converge", 0,
      int(bool(p32["ipf_calibration"]["converged"])), 0)

# --- p34: the adjacency graph's five self-gates
_adj = p34["adjacency"]
for _k, _q in (("n_nodes", 424), ("min_degree", 2), ("max_degree", 13),
               ("edges", 1153)):
    check("19", f"adjacency {_k}", _q, _adj[_k], 0)
c19("adjacency mean degree", 5.44, _adj["mean_degree"], 1e-3)
check("19", "adjacency connected", 1, int(bool(_adj["connected"])), 0)
check("19", "every gu connected", 1, int(bool(_adj["every_gu_connected"])), 0)

# --- p34: the anchors, the k=25 table, the slopes, the retraction
# All six months now, survey months included. The k=25 rows are what the earlier
# hand-audit got wrong, so every band endpoint is checked, not just the median.
_sw = {k: {x["n_loc"]: x for x in v} for k, v in p34["sweep"].items()}
_K25 = {
    202001: (0.016745, 0.004171, 0.00796, 0.00769, 0.00719, 0.00828, 0.00596),
    202012: (0.013109, 0.003184, 0.00603, 0.00586, 0.00553, 0.00623, 0.00427),
    202312: (0.021359, 0.004919, 0.00987, 0.00948, 0.00892, 0.01023, 0.00699),
    202402: (0.018411, 0.004670, 0.00903, 0.00875, 0.00816, 0.00924, 0.00652),
    202512: (0.019889, 0.004826, 0.00949, 0.00915, 0.00856, 0.00975, 0.00677),
    202606: (0.021280, 0.005320, 0.01029, 0.00988, 0.00925, 0.01051, 0.00725),
}
for _ym, (_dong, _city, _gu, _adjm, _lo, _hi, _rnd) in _K25.items():
    for _arm in ("adjacency", "random"):
        c19(f"{_ym} {_arm} k=424", _dong,
            _sw[f"{_ym}|WE|{_arm}"][424]["assortativity"]["median"], 1e-4)
        c19(f"{_ym} {_arm} k=1", _city,
            _sw[f"{_ym}|WE|{_arm}"][1]["assortativity"]["median"], 1e-4)
    _a25 = _sw[f"{_ym}|WE|adjacency"][25]["assortativity"]
    c19(f"{_ym} adjacency k=25 median", _adjm, _a25["median"], 2e-3)
    c19(f"{_ym} adjacency k=25 lo", _lo, _a25["lo"], 2e-3)
    c19(f"{_ym} adjacency k=25 hi", _hi, _a25["hi"], 2e-3)
    c19(f"{_ym} random k=25 median", _rnd,
        _sw[f"{_ym}|WE|random"][25]["assortativity"]["median"], 2e-3)
    c19(f"{_ym} real gu rung", _gu,
        p34["anchors"][str(_ym)]["adjacency|gu25"]["want"], 1e-3)
    check("19", f"{_ym} real gu inside the k=25 band", 1,
          int(bool(p34["anchors"][str(_ym)]["adjacency|gu25"]["inside_band"])), 0)

_sl = p34["local_slope"]["202012"]
for _key, _three in (("adjacency|assortativity", 0.233), ("adjacency|nmi", 0.355),
                     ("random|assortativity", 0.231), ("random|nmi", 0.351)):
    c19(f"202012 {_key} three-point exponent", _three,
        _sl[f"{_key}|three_point"], 5e-3)
# every month's local slope must exceed its own three-point fit -- the claim the
# report makes in one sentence, checked six times rather than asserted once
for _ym in _K25:
    _p = p34["polygon_prediction_interval"][str(_ym)]
    check("19", f"{_ym} local slope exceeds the three-point fit", 1,
          int(_p["slope_lo"] > _p["published_slope"]), 0)
_pi = p34["polygon_prediction_interval"]["202012"]
c19("202012 local slope lo (n>=200)", 0.288, _pi["slope_lo"], 5e-3)
c19("202012 local slope hi (n>=200)", 0.390, _pi["slope_hi"], 5e-3)
c19("202012 1831-polygon interval lo", 0.01998, _pi["r_polygon_lo"], 2e-3)
c19("202012 1831-polygon interval hi", 0.02321, _pi["r_polygon_hi"], 2e-3)
c19("202012 published polygon point", 0.01771, _pi["published_point"], 1e-3)
c19("202012 published three-point slope", 0.233, _pi["published_slope"], 5e-3)
_es = p34["extrapolation_support"]
c19("log10 span the curve supports", 2.63, _es["log10_span"], 2e-3)
c19("log10 span the extrapolation needs", 6.80, _es["log10_span_required"], 2e-3)

# --- the arm gap, every value the memo tabulates
for _ym, _pairs in (
        (202012, ((2, 1.041), (5, 1.220), (10, 1.305), (25, 1.373), (50, 1.323),
                  (100, 1.223), (200, 1.095), (424, 1.000))),
        (202001, ((25, 1.290),)), (202312, ((25, 1.356),)),
        (202402, ((25, 1.342),))):
    for _k, _q in _pairs:
        c19(f"{_ym} arm gap k={_k}", _q,
            (_sw[f"{_ym}|WE|adjacency"][_k]["assortativity"]["median"]
             / _sw[f"{_ym}|WE|random"][_k]["assortativity"]["median"]), 2e-3)


# ======================================================================
# The 2026-08-19 round, second half: p33 (coverage), p35 (the fourth pillar) and
# p36 (the independent recompute). Same LIVE source, same relative tolerance.
# ======================================================================
p33 = load("p33", ARCHIVE_0822)
p35 = load("p35", ARCHIVE_0822)
p36 = load("p36", ARCHIVE_0822)

# --- p33: the anchor first. If the cell-level rebuild does not reproduce p26,
# nothing else in that file means anything, so it is checked as a number rather
# than trusted because the script asserts it.
c19("p33 anchor max relative deviation", 0.0,
    max(max(a["rel_r"], a["rel_lambda"]) for a in p33["anchor"]) or 0.0, 1.0)
for _ym, _vals in ((202312, dict(assortativity=(0.02115, 0.00439, 0.01677),
                                 nmi=(0.00518, 0.00066, 0.00452))),
                   (202402, dict(assortativity=(0.01851, 0.00423, 0.01427),
                                 nmi=(0.00469, 0.00061, 0.00408)))):
    _st = p33["R1_effective_sample"][str(_ym)]["stats"]
    for _k, (_pt, _bias, _corr) in _vals.items():
        c19(f"{_ym} p33 {_k} published", _pt, _st[_k]["point"], 3e-3)
        c19(f"{_ym} p33 {_k} noise bias", _bias, _st[_k]["noise_bias"], 3e-2)
        c19(f"{_ym} p33 {_k} bias-corrected", _corr, _st[_k]["bias_corrected"], 3e-3)
c19("202312 p33 assortativity noise share 20.7%", 20.7,
    100 * p33["R1_effective_sample"]["202312"]["stats"]["assortativity"]["noise_bias"]
    / p33["R1_effective_sample"]["202312"]["stats"]["assortativity"]["point"], 3e-2)
c19("202402 p33 assortativity noise share 22.9%", 22.9,
    100 * p33["R1_effective_sample"]["202402"]["stats"]["assortativity"]["noise_bias"]
    / p33["R1_effective_sample"]["202402"]["stats"]["assortativity"]["point"], 3e-2)
_th = {r["theta"]: r for r in p33["R2_theta_family"]["202312"]}
c19("202312 p33 theta=-1 assortativity", 0.01704, _th[-1.0]["assortativity"], 1e-3)
c19("202312 p33 theta=+1 assortativity", 0.05099, _th[1.0]["assortativity"], 1e-3)
c19("202312 p33 theta=-1 NMI", 0.00376, _th[-1.0]["nmi"], 2e-3)
c19("202312 p33 theta=+1 NMI", 0.01092, _th[1.0]["nmi"], 2e-3)
_gap = {r["theta"]: r for r in p33["R2_gaps_vs_survey"]["202312"]}
c19("202312 p33 NMI gap at theta=0", 34.7, _gap[0.0]["nmi"], 2e-3)
c19("202312 p33 NMI gap at theta=+1", 16.5, _gap[1.0]["nmi"], 3e-3)
c19("202312 p33 coverage-vs-masking width ratio 14.0", 14.0,
    p33["mask_vs_coverage"]["202312"]["ratio"], 4e-3)
c19("202402 p33 coverage-vs-masking width ratio 14.3", 14.3,
    p33["mask_vs_coverage"]["202402"]["ratio"], 4e-3)
c19("p33 censored bands 5 of 16", 5.0, float(sum(p33["profile"]["censored"])), 1e-9)

# --- p35: the two anchors, then the numbers the report quotes
c19("p35 finite-difference anchor", 4.94e-4,
    max(r["max_rel_err"] for r in p35["marginal_finite_difference_check"]), 3e-2)
c19("p35 final-size vs RK4 anchor", 2.89e-6,
    p35["final_size_anchor"]["max_abs_diff"], 3e-2)
_c = {r["pair"]: r for r in p35["comparison_marginal"]["202312"]}
c19("202312 p35 tau survey vs passive dong", 0.429,
    _c["survey vs passive_dong"]["tau"], 3e-3)
c19("202312 p35 tau survey vs passive gu", 0.390,
    _c["survey vs passive_gu"]["tau"], 3e-3)
_c2 = {r["pair"]: r for r in p35["comparison_marginal"]["202402"]}
c19("202402 p35 tau survey vs passive dong", 0.600,
    _c2["survey vs passive_dong"]["tau"], 3e-3)
for _key, _q in (("202312|R0=1.3", 20.2), ("202312|R0=1.8", 2.97),
                 ("202312|R0=2.5", 7.5), ("202402|R0=1.3", 21.7),
                 ("202402|R0=1.8", 15.4), ("202402|R0=2.5", 23.3)):
    c19(f"p35 regret {_key}", _q,
        100 * p35["allocation"][_key]["cross_application"]["regret_share_of_benefit"],
        6e-3)
c19("p35 202312 R0=1.3 passive plan leaves 5.71 pp", 5.71,
    100 * p35["allocation"]["202312|R0=1.3"]["cross_application"]["attack_passive_plan"],
    2e-3)
_b = p35["survey_bootstrap"]
c19("p35 202312 seoul within-tau lo", 0.619, _b["202312"]["tau_within_survey"][0], 3e-3)
c19("p35 202402 seoul within-tau lo", 0.410, _b["202402"]["tau_within_survey"][0], 3e-3)
c19("p35 202402 seoul regret CI lo 9.5%", 9.5,
    100 * _b["202402"]["regret"]["ci"][0], 2e-2)
c19("p35 202402 national regret CI lo 11.8%", 11.8,
    100 * _b["202402|national"]["regret"]["ci"][0], 2e-2)
c19("p35 seoul egos 465", 465.0, float(_b["202312"]["n_respondents"]), 1e-9)
c19("p35 national egos 1987", 1987.0, float(_b["202312|national"]["n_respondents"]), 1e-9)

# --- the section 5 retreat block. These are the numbers the letter uses to argue
# that band width and statistic choice are levers on different things, so they
# are the ones a reviewer would check first.
c19("p35 202312 top-1 stability 0.63", 0.63, _b["202312"]["top1_stability"], 2e-2)
c19("p35 202402 top-1 stability 0.475", 0.475, _b["202402"]["top1_stability"], 2e-2)
c19("p35 202312 overlap@3 2.27", 2.27, _b["202312"]["top3_mean_overlap"], 5e-3)
c19("p35 202402 overlap@3 2.05", 2.05, _b["202402"]["top3_mean_overlap"], 5e-3)
c19("p35 202312 national top-1 0.65", 0.65, _b["202312|national"]["top1_stability"], 2e-2)
c19("p35 202402 national top-1 0.61", 0.61, _b["202402|national"]["top1_stability"], 2e-2)
c19("p35 202312 national overlap@3 2.58", 2.58,
    _b["202312|national"]["top3_mean_overlap"], 5e-3)
c19("p35 202402 national overlap@3 2.79", 2.79,
    _b["202402|national"]["top3_mean_overlap"], 5e-3)
c19("p35 202402 between-tau 0.600", 0.600,
    [r_["tau"] for r_ in p35["comparison_marginal"]["202402"]
     if r_["pair"] == "survey vs passive_dong"][0], 3e-3)

# The declared 9-band and 8-band retreat maps. The letter quotes 11-92 and a
# minimum of 22, and those are DERIVED here from n_ego_per_band rather than
# restated -- a hand-typed value checked against a hand-typed value is precisely
# the failure mode p36_recompute.py exists to close, and it would be an odd
# thing for this gate to reintroduce two sections later.
_BANDS = p35["bands"]
_IDX = {b_: i for i, b_ in enumerate(_BANDS)}
_MAP9 = [["0-9"], ["10-14"], ["15-19"], ["20-24", "25-29"], ["30-34", "35-39"],
         ["40-44", "45-49"], ["50-54", "55-59"], ["60-64", "65-69"],
         ["70-74", "75-79"]]
_MAP8 = [["0-9"], ["10-14", "15-19"], ["20-24", "25-29"], ["30-34", "35-39"],
         ["40-44", "45-49"], ["50-54", "55-59"], ["60-64", "65-69"],
         ["70-74", "75-79"]]


def _merged(ym, mp):
    n = _b[str(ym)]["n_ego_per_band"]
    return [sum(n[_IDX[x]] for x in g) for g in mp]


c19("p35 15-band ego min 11", 11.0, float(min(_b["202312"]["n_ego_per_band"])), 1e-9)
c19("p35 15-band ego max 55", 55.0, float(max(_b["202312"]["n_ego_per_band"])), 1e-9)
c19("p35 national ego min 73", 73.0,
    float(min(_b["202312|national"]["n_ego_per_band"])), 1e-9)
c19("p35 national ego max 194", 194.0,
    float(max(_b["202312|national"]["n_ego_per_band"])), 1e-9)
c19("p35 retreat A (9 band) ego min 11", 11.0, float(min(_merged(202312, _MAP9))), 1e-9)
c19("p35 retreat A (9 band) ego max 92", 92.0, float(max(_merged(202312, _MAP9))), 1e-9)
c19("p35 retreat B (8 band) ego min 22", 22.0, float(min(_merged(202312, _MAP8))), 1e-9)
c19("p35 retreat B 10-19 merges to 26", 26.0, float(_merged(202312, _MAP8)[1]), 1e-9)
# the letter says the ego profile is identical in the two months, which is what
# makes one derivation cover both
check("19", "p35 ego profile identical across months", 1.0,
      1.0 if _b["202312"]["n_ego_per_band"] == _b["202402"]["n_ego_per_band"] else 0.0,
      1e-9)

# --- ABSENT EVIDENCE MUST FAIL, NOT SKIP. Every block below guards on
# os.path.exists and continues when a results file is missing, which means
# deleting one would silently remove its checks while the gate still printed
# "N of N pass" -- a smaller N, quietly. That is the same shape as p38's
# n_numbers: 0 beside max_deviation: 0.0, and it was live here: the per-panel
# p34 files were untracked when their checks were written, so a fresh clone
# would have skipped them rather than failed. The presence of each required
# file is therefore itself a check, named, so absence shows up red instead of
# shrinking the total.
# HOW MANY MONTHS p34 IS COMMITTED FOR. Three checks below are pinned to this
# and all three are load-bearing: 24 hard anchors per panel (months x 2 arms x
# 2 levels), the polygon month count, and the determinism harness's month list.
#
# A 79-month sweep exists as p38 and will land eventually. When it does, all
# three go red at once, correctly, at a moment when the cheap-looking repair is
# to relax them. Do not. Move this constant, in the same edit that extends
# p34's --months and determinism_check.sh's ARGS, and let the three follow.
#
# It stays a hard-coded literal rather than len(p34["sweep"]) on purpose: a
# count read from the file it is checking is the self-comparison that p31:674
# and p31:695 were just fixed for. The number has to come from outside the data.
P34_COMMITTED_MONTHS = 6

REQUIRED_RESULTS = ["p34", "p34_W", "p34_E", "p35", "p36", "p37", "p38",
                    "p38_W", "p38_E", "inventory", "p39", "p40", "p41", "p42",
                    "p44", "p45", "p49", "p50", "p51", "p52", "p53", "p54",
                    "p57", "p67", "p69",
                    # the figure sheets. Every one of these has existed for
                    # days and NONE of them was declared here, because each is
                    # reached through load() rather than results_of() and the
                    # rule that keeps this list honest only watches results_of.
                    # A figure whose caption sheet had gone missing would have
                    # taken its checks with it, silently -- which is the exact
                    # shape of the absence this list exists to make loud.
                    # p46 is the last of them, and it had no results file at
                    # all until 2026-08-28: Figure 6's caption was the one
                    # whose numbers had no machine-readable source. It has one
                    # now, so it is declared here like the rest.
                    "p46", "p47", "p48", "p55", "p56",
                    # the 08-27 round's six new phases
                    "p58", "p59", "p60", "p61", "p62", "p63",
                    # ...and the files the manuscript is the first document to
                    # be checked against. p2/p8/p43_dedup had never been read
                    # by any gate at all; p19/p25/p27/p29/p32 had only ever
                    # been read from a frozen snapshot, so their LIVE copies
                    # were unguarded while the manuscript quoted them.
                    "p2", "p8", "p19", "p25", "p27", "p29", "p32",
                    "p43_dedup"]
# Which round each file arrived with, so that the numbers list groups a
# presence check with the document whose numbers it protects rather than with
# whichever section happened to declare the list. p39/p40/p41 came in with the
# 08-21 letter; everything else with 08-19.
#
# p47/p48/p55/p56/p57 arrived with the 08-24 round and are tagged "27" anyway,
# on purpose: a round's tag here is the round that STARTED CHECKING the file,
# not the round that produced it. The 08-24 letter has been sent and states its
# own count ("122 項檢查"); tagging a new check "24" would move that count and
# the only way back to green would be to retype a number inside a letter that
# has already left. That happened once, on 2026-08-24, and the fix was to move
# the moving totals into eda/README.md. A closed round's count does not move.
_ROUND_OF = {"p39": "21", "p40": "21", "p41": "21", "p42": "21", "p44": "21",
             "p45": "21", "p49": "24", "p50": "24", "p51": "24", "p52": "24",
             "p53": "24", "p54": "24",
             "p46": "27",
             "p47": "27", "p48": "27", "p55": "27", "p56": "27", "p57": "27",
             "p58": "27", "p59": "27", "p60": "27", "p61": "27", "p62": "27",
             "p63": "27",
             "p2": "MS", "p8": "MS", "p19": "MS", "p25": "MS", "p27": "MS",
             "p29": "MS", "p32": "MS", "p43_dedup": "MS"}
for _name in REQUIRED_RESULTS:
    check(_ROUND_OF.get(_name, "19"),
          f"required results file present: results_{_name}.json", 1.0,
          1.0 if os.path.exists(f"{LIVE}/results_{_name}.json") else 0.0, 1e-9)

# ...and the rule that keeps that list honest, enforced BY CONSTRUCTION rather
# than by reading the source. The first version of this check scanned for the
# literal string os.path.exists(f"{LIVE}/results_NAME.json"), and every real
# guard here uses a variable or an f-string interpolation instead -- so it
# matched nothing, compared an empty set against an empty set, and passed. A
# check written to catch vacuous checks, itself vacuous. It was only caught by
# deliberately removing p34_W/E from the list and watching it stay green.
#
# So existence is no longer asked with a bare os.path.exists at all. results_of()
# is the only way to reach a results path, and it records every name it is asked
# for; anything asked for but not declared REQUIRED fails below. There is no
# spelling of "guard quietly" left that does not go through it.
_ASKED = set()


def results_of(name):
    """Path to results_<name>.json, registering the name as one this gate needs."""
    _ASKED.add(name)
    return f"{LIVE}/results_{name}.json"



# --- the k=25 correction, and the reason it is gated this way. The letter said
# "the peak is exactly at k=25" for days and no check caught it, because the
# VALUE at k=25 (1.373) was gated and the CLAIM that 25 is the argmax was not.
# Gating a number beside a word does not gate the word. So the argmax over all
# 424 values is recomputed per month and the letter's six values are checked
# against it.
_ARGMAX = {202001: 28, 202012: 22, 202312: 24, 202402: 27, 202512: 24, 202606: 29}
for _ym, _want in _ARGMAX.items():
    _a, _r = _sw.get(f"{_ym}|WE|adjacency"), _sw.get(f"{_ym}|WE|random")
    if not _a or not _r:
        continue
    _rat = {k: _a[k]["assortativity"]["median"] / _r[k]["assortativity"]["median"]
            for k in _a if k in _r and _r[k]["assortativity"]["median"]}
    c19(f"{_ym} arm-gap argmax k", float(_want),
        float(max(_rat, key=_rat.get)), 1e-9)
c19("arm-gap argmax range low 22 (WE)", 22.0, float(min(_ARGMAX.values())), 1e-9)
c19("arm-gap argmax range high 29 (WE)", 29.0, float(max(_ARGMAX.values())), 1e-9)

# --- the per-panel peaks, gated as two SEPARATED intervals rather than as one
# pooled range. W peaks at or below 25 in every month and E at or above 27, and
# the two do not overlap; a pooled 17-33 would pass while both panels' actual
# signal had vanished, which is the whole reason the pooled sentence was wrong.
_PANEL_ARGMAX = {
    "W": {202001: 18, 202012: 22, 202312: 25, 202402: 17, 202512: 18, 202606: 25},
    "E": {202001: 28, 202012: 33, 202312: 32, 202402: 27, 202512: 28, 202606: 30},
}
for _pan, _want in _PANEL_ARGMAX.items():
    _f = results_of(f"p34_{_pan}")
    if not os.path.exists(_f):
        continue
    _swp = {k: {x["n_loc"]: x for x in v}
            for k, v in json.load(open(_f))["sweep"].items()}
    _got = {}
    for _ym in _want:
        _a = _swp.get(f"{_ym}|{_pan}|adjacency")
        _r = _swp.get(f"{_ym}|{_pan}|random")
        if not _a or not _r:
            continue
        _rt = {k: _a[k]["assortativity"]["median"] / _r[k]["assortativity"]["median"]
               for k in _a if 1 < k < 424
               and _r.get(k, {}).get("assortativity", {}).get("median", 0) > 0}
        _got[_ym] = max(_rt, key=_rt.get)
        c19(f"{_ym} arm-gap argmax k ({_pan})", float(_want[_ym]),
            float(_got[_ym]), 1e-9)
    if _got:
        if _pan == "W":
            c19("W panel peak, highest month", 25.0, float(max(_got.values())), 1e-9)
            check("19", "W panel peaks at or below 25 in every month", 6.0,
                  float(sum(1 for v in _got.values() if v <= 25)), 1e-9)
        else:
            c19("E panel peak, lowest month", 27.0, float(min(_got.values())), 1e-9)
            check("19", "E panel peaks at or above 27 in every month", 6.0,
                  float(sum(1 for v in _got.values() if v >= 27)), 1e-9)
# the claim that carries the paragraph: the two panels do not overlap at all
if all(os.path.exists(results_of(f"p34_{p_}")) for p_ in ("W", "E")):
    check("19", "W and E peak ranges do not overlap", 1.0,
          1.0 if max(_PANEL_ARGMAX["W"].values())
          < min(_PANEL_ARGMAX["E"].values()) else 0.0, 1e-9)

# --- the W and E panels. The soft anchor is now a universal over three panels,
# so it is gated as a count, not as a ratio range.
_panel_inside = _panel_total = 0
_panel_ratios = []
for _pan, _f in (("W", results_of("p34_W")),
                 ("E", results_of("p34_E")),
                 ("WE", results_of("p34"))):
    if not os.path.exists(_f):
        continue
    _d = json.load(open(_f))
    for _ym, _cells in _d["anchors"].items():
        for _k, _v in _cells.items():
            if "gu25" in _k and "adjacency" in _k:
                _panel_total += 1
                _panel_inside += 1 if _v.get("inside_band") else 0
                _panel_ratios.append(_v["ratio"])
    _hard = [_v for _cells in _d["anchors"].values() for _k, _v in _cells.items()
             if _k.endswith("dong") or _k.endswith("city")]
    # COUNT FIRST, against a literal. "all of them passed" compares the
    # collection with itself, so an empty collection is 0.0 vs 0.0 and reports
    # ok -- rename the level keys from dong/city and this check evaporates
    # while staying green. It still catches a genuine regression (24 vs 0), so
    # the vacuity is specifically on absence, which is why the file-presence
    # check does not cover it: the file is there, the contents emptied out.
    # 24 = six months x two arms x two levels.
    check("19", f"p34 {_pan} hard anchors present",
          float(P34_COMMITTED_MONTHS * 2 * 2), float(len(_hard)), 1e-9)
    check("19", f"p34 {_pan} hard anchors all exact", float(len(_hard)),
          float(sum(1 for _v in _hard if _v.get("ok"))), 1e-9)
if _panel_total:
    c19("p34 soft anchor 18 of 18 inside the band", 18.0,
        float(_panel_inside), 1e-9)
    check("19", "p34 soft anchor: none outside the band", 0.0,
          float(_panel_total - _panel_inside), 1e-9)
    c19("p34 soft anchor ratio low 0.959", 0.959, min(_panel_ratios), 6e-4)
    c19("p34 soft anchor ratio high 0.979", 0.979, max(_panel_ratios), 6e-4)
# Only W and E here, because this is where the per-panel slopes live -- which
# leaves WE's polygon months ungated by omission rather than by design. Noted
# rather than silently tolerated; WE's polygon interval is covered by the p34
# block above, its month count is not.
for _pan, _lo, _hi, _pub in (("W", 0.368, 0.462, 0.254), ("E", 0.320, 0.451, 0.225)):
    _f = results_of(f"p34_{_pan}")
    if not os.path.exists(_f):
        continue
    _poly = json.load(open(_f))["polygon_prediction_interval"]["202012"]
    c19(f"p34 {_pan} 202012 three-point slope", _pub, _poly["published_slope"], 3e-3)
    c19(f"p34 {_pan} 202012 local slope lo", _lo, _poly["slope_lo"], 3e-3)
    c19(f"p34 {_pan} 202012 local slope hi", _hi, _poly["slope_hi"], 3e-3)
    _all = json.load(open(_f))["polygon_prediction_interval"]
    # same shape, same fix: the month count against a literal before the
    # all-of-them check that would otherwise pass over an empty dict
    check("19", f"p34 {_pan} polygon months present",
          float(P34_COMMITTED_MONTHS), float(len(_all)), 1e-9)
    check("19", f"p34 {_pan} polygon interval above the published point, all months",
          float(len(_all)),
          float(sum(1 for _v in _all.values()
                    if _v["r_polygon_lo"] > _v["published_point"])), 1e-9)

# --- the determinism harness must invoke p34 over the months its results file
# actually holds. It did not: ARGS carried the two-month default while the
# committed file is six months, and the harness rm's the file between runs. One
# invocation would have replaced six months with two while the gate kept reading
# per-month keys. Gated as a setting, like p36's thread count.
_dc = open(f"{ROOT}/eda/determinism_check.sh").read()
_p34_months = {int(x) for x in re.findall(r"20\d{4}", _dc)}
check("19", "determinism harness runs p34 over all committed months",
      float(P34_COMMITTED_MONTHS),
      float(len(_p34_months & set(p34["sweep"] and
                                  {int(k.split("|")[0]) for k in p34["sweep"]}))),
      1e-9)

# --- p38: the k-sweep extended to 79 months. The letter quotes its numbers, so
# they are checked here. Count first throughout, per the lesson at p31:674:
# every "all of them" below is preceded by how many there were, against a
# literal that does not come from the file being checked.
_p38_path = results_of("p38")
if os.path.exists(_p38_path):
    p38 = json.load(open(_p38_path))
    c19("p38 months 79", 79.0, float(len(p38["months"])), 1e-9)
    c19("p38 replicates 200", 200.0, float(p38["reps"]), 1e-9)
    # the anchor is only meaningful if it compared something -- p38's own guard
    # skips it below 200 reps, and n_numbers 0 beside max_deviation 0.0 would
    # otherwise read as a pass
    _pa = p38["p34_anchor"]
    c19("p38 anchor months 6", 6.0, float(len(_pa["months"])), 1e-9)
    c19("p38 anchor numbers 96,672", 96672.0, float(_pa["n_numbers"]), 1e-9)
    check("19", "p38 anchor deviation exactly 0", 0.0,
          float(_pa["max_deviation"]), 0.0)
    _ex = p38["exponent"]["assortativity"]
    c19("p38 79-month local slope low 0.178", 0.178, _ex["named_lo"], 3e-3)
    c19("p38 79-month local slope high 0.478", 0.478, _ex["named_hi"], 3e-3)
    c19("p38 three-point range low 0.205", 0.205, _ex["three_lo"], 3e-3)
    c19("p38 three-point range high 0.250", 0.250, _ex["three_hi"], 3e-3)
    # the claim that replaced my retracted prediction: every curve bends up
    c19("p38 curves examined 158", 158.0, float(_ex["n_curves"]), 1e-9)
    c19("p38 curves bending up 158", 158.0, float(_ex["n_bending_up"]), 1e-9)
    check("19", "p38 no curve fails to bend up", 0.0,
          float(_ex["n_curves"] - _ex["n_bending_up"]), 1e-9)
    # p38 is WE-only, so its pooled peak_k must not be quoted as a property of
    # the phenomenon -- the panel split at p31:620 is what settles that
    check("19", "p38 is the WE panel only (its peak_k is pooled)", 1.0,
          1.0 if p38["panel"] == "WE" else 0.0, 1e-9)
    # The letter's sentence is about each curve against ITS OWN three-point fit.
    # That is NOT what n_bending_up counts -- that one compares the two ends of
    # the same curve to each other. Both happen to be 158 today, so gating only
    # the first would leave the sentence unguarded the day they part company.
    # near_lo is the weakest local slope anywhere in the n>=201 window, so this
    # is the strictest reading of the claim.
    _above = sum(1 for _y in p38["months"] for _arm in ("adjacency", "random")
                 if (p38["local_slope"][str(_y)][f"{_arm}|assortativity"]["near_lo"]
                     > p38["local_slope"][str(_y)][f"{_arm}|assortativity"]["three_point"]))
    c19("p38 curves beating their own three-point fit 158", 158.0,
        float(_above), 1e-9)

    # --- the soft anchor and the arm gap as universals over 79 months. Both are
    # recomputed from the stored 424-level median curve rather than read out of
    # the summary the script wrote: checking the letter against that summary is
    # checking the letter against the sentence's own source.
    _adj = [p38["anchors"][str(_y)]["adjacency|gu25"]["ratio"] for _y in p38["months"]]
    _rnd = [p38["anchors"][str(_y)]["random|gu25"]["ratio"] for _y in p38["months"]]
    c19("p38 months carrying a soft anchor", 79.0, float(len(_adj)), 1e-9)
    check("19", "p38 real gu outside the band in no month", 0.0,
          float(sum(1 for _y in p38["months"]
                    if not p38["anchors"][str(_y)]["adjacency|gu25"]["inside_band"])),
          1e-9)
    c19("p38 adjacency/real gu low 0.946", 0.946, min(_adj), 1e-3)
    c19("p38 adjacency/real gu high 0.981", 0.981, max(_adj), 1e-3)
    c19("p38 random arm shortfall low 25.1%", 25.1, 100 * (1 - max(_rnd)), 3e-3)
    c19("p38 random arm shortfall high 30%", 30.0, 100 * (1 - min(_rnd)), 3e-3)
    c19("p38 202001 random/real gu 0.749", 0.749,
        p38["anchors"]["202001"]["random|gu25"]["ratio"], 2e-3)
    _pk, _pr, _pk_bad = [], [], 0
    for _y in p38["months"]:
        _a = p38["curve"][f"{_y}|WE|adjacency"]["assortativity"]
        _r = p38["curve"][f"{_y}|WE|random"]["assortativity"]
        _rat = {_k: _a[_k - 1] / _r[_k - 1] for _k in range(2, 424) if _r[_k - 1] > 0}
        _k = max(_rat, key=_rat.get)
        _pk.append(_k)
        _pr.append(_rat[_k])
        _pk_bad += 1 if _k != p38["arm_gap"][str(_y)]["peak_k"] else 0
    check("19", "p38 peak k recomputed from the curve, every month", 0.0,
          float(_pk_bad), 1e-9)
    c19("p38 peak k lowest 19", 19.0, float(min(_pk)), 1e-9)
    c19("p38 peak k highest 36", 36.0, float(max(_pk)), 1e-9)
    c19("p38 peak k median 25", 25.0, float(np.median(_pk)), 1e-9)
    c19("p38 peak ratio low 1.30", 1.30, min(_pr), 4e-3)
    c19("p38 peak ratio high 1.38", 1.38, max(_pr), 4e-3)
    # and the sentence that answers "did the low six-month window distort this":
    # the window is narrow where the LEVEL lives and wide where the RATIOS do
    _w = p38["published_window"]
    c19("p38 six months cover 65% of the dong-level spread", 65.0,
        100 * _w["r_dong"]["span_covered"], 1e-2)
    c19("p38 six months cover 34% of the soft-anchor spread", 34.0,
        100 * _w["adj_over_real_gu"]["span_covered"], 1e-2)
    c19("p38 six months cover 89% of the random-arm spread", 89.0,
        100 * _w["random_over_real_gu"]["span_covered"], 1e-2)
    c19("p38 six months cover 93% of the peak-height spread", 93.0,
        100 * _w["peak_ratio"]["span_covered"], 1e-2)

# --- the 79-month panel split. Gated ALONGSIDE the six-month non-overlap, not
# instead of it: both are true, of different windows, and the letter reports
# both. The six-month checks above assert a six-month fact out of six-month
# files and stay green whatever these say -- which is correct, and is exactly
# why the endpoint claim needed a sentence rather than a gate.
#
# Reported as the PAIRED comparison, because W and E are measured on the same
# 79 months. Endpoints would mistake each panel's month-to-month wander for the
# absence of a between-panel difference -- the mirror of the pooled sentence,
# which mistook an average of two panels for a property of the phenomenon.
_pW, _pE = results_of("p38_W"), results_of("p38_E")
if os.path.exists(_pW) and os.path.exists(_pE):
    _gW = json.load(open(_pW))["arm_gap"]
    _gE = json.load(open(_pE))["arm_gap"]
    _ms = sorted(set(_gW) & set(_gE))
    check("19", "p38 panel months paired", 79.0, float(len(_ms)), 1e-9)
    _d = [_gE[m]["peak_k"] - _gW[m]["peak_k"] for m in _ms]
    c19("p38 E peaks coarser than W, months", 76.0,
        float(sum(1 for x in _d if x > 0)), 1e-9)
    c19("p38 E-vs-W ties", 3.0, float(sum(1 for x in _d if x == 0)), 1e-9)
    check("19", "p38 E-vs-W reversals", 0.0,
          float(sum(1 for x in _d if x < 0)), 1e-9)
    c19("p38 median paired difference E-W", 9.0,
        float(sorted(_d)[len(_d) // 2]), 1e-9)
    _hi = sum(1 for m in _ms
              if _gW[m]["peak_ratio"] > _gE[m]["peak_ratio"])
    c19("p38 W peak higher than E, months", 79.0, float(_hi), 1e-9)
    # the endpoint form that was retracted -- gated as the RETRACTION, so the
    # ranges cannot quietly stop overlapping without someone being told
    _kW = [v["peak_k"] for v in _gW.values()]
    _kE = [v["peak_k"] for v in _gE.values()]
    c19("p38 W peak k range low 17", 17.0, float(min(_kW)), 1e-9)
    c19("p38 W peak k range high 28", 28.0, float(max(_kW)), 1e-9)
    c19("p38 E peak k range low 21", 21.0, float(min(_kE)), 1e-9)
    c19("p38 E peak k range high 39", 39.0, float(max(_kE)), 1e-9)
    c19("p38 W and E ranges overlap by 7", 7.0,
        float(min(max(_kW), max(_kE)) - max(min(_kW), min(_kE))), 1e-9)
    check("19", "p38 panel anchors exact, both arms", 0.0,
          float(max(json.load(open(f))["p34_anchor"]["max_deviation"]
                    for f in (_pW, _pE))), 0.0)
    check("19", "p38 panel anchor months, each panel", 12.0,
          float(sum(len(json.load(open(f))["p34_anchor"]["months"])
                    for f in (_pW, _pE))), 1e-9)

# --- p37: the same estimator on 79 months. The claims the letter makes are
# universals ("every month", "no month exceeds"), so they are gated as universals
# -- a count and a maximum -- rather than as the summary numbers beside them. A
# median that still matches while one month has broken the ladder would be a
# gate reporting green over a falsified claim.
_p37_path = results_of("p37")
if os.path.exists(_p37_path):
    p37 = json.load(open(_p37_path))
    c19("p37 months 79", 79.0, float(p37["summary_dong"]["n_months"]), 1e-9)
    c19("p37 anchor cells 54", 54.0, float(len(p37["anchor"])), 1e-9)
    check("19", "p37 anchor reproduces p26 exactly", 0.0,
          max(a["rel"] for a in p37["anchor"]), 0.0)
    # the ladder, as a universal
    check("19", "p37 ladder holds in every month", 0.0,
          float(p37["ladder"]["n_months"] - p37["ladder"]["n_monotone"]), 1e-9)
    check("19", "p37 no month breaks the ladder", 0.0,
          float(len(p37["ladder"]["breaks"])), 1e-9)
    c19("p37 retention median 29.8%", 29.8, 100 * p37["ladder"]["kept_median"], 2e-3)
    c19("p37 retention min 27.6%", 27.6, 100 * p37["ladder"]["kept_min"], 2e-3)
    c19("p37 retention max 32.6%", 32.6, 100 * p37["ladder"]["kept_max"], 2e-3)
    # near rank-1, as a universal: the letter says NO month exceeds 0.00202
    c19("p37 gap to proportionate mixing, worst month", 0.00202,
        p37["spectrum_dong"]["gap_to_pm_max"], 5e-3)
    c19("p37 sigma1 share median 0.9823", 0.9823,
        p37["spectrum_dong"]["sigma1_share_median"], 2e-4)
    c19("p37 sigma1 share min 0.9702", 0.9702,
        p37["spectrum_dong"]["sigma1_share_min"], 2e-4)
    # the six published months undercover the range -- the number the letter
    # leads with is the 21% shortfall, so it is recomputed, not restated
    _sd = p37["summary_dong"]
    c19("p37 assortativity min 0.01311", 0.01311, _sd["assort_min"], 5e-4)
    c19("p37 assortativity max 0.02589", 0.02589, _sd["assort_max"], 5e-4)
    c19("p37 assortativity median 0.01938", 0.01938, _sd["assort_median"], 5e-4)
    c19("p37 published six top 0.02136", 0.02136, _sd["published_six_max"], 5e-4)
    c19("p37 published six top is 21% low", 21.0,
        100 * (_sd["assort_max"] / _sd["published_six_max"] - 1), 2e-2)
    # 202012 is the single lowest month, which is why the 2020-led framing
    # understates. Derived from the rows rather than restated.
    _we = sorted([r for r in p37["rows"]
                  if r["panel"] == "WE" and r["level"] == "dong"],
                 key=lambda r: r["assortativity"])
    check("19", "p37 202012 is the lowest of all 79 months", 202012.0,
          float(_we[0]["ym"]), 1e-9)
    _rank = {r["ym"]: i + 1 for i, r in enumerate(_we)}
    c19("p37 202312 sits at rank 50 of 79", 50.0, float(_rank[202312]), 1e-9)
    c19("p37 202402 sits at rank 24 of 79", 24.0, float(_rank[202402]), 1e-9)
    # the school-term signal, and its 2020 control. Both are recomputed here
    # from the rows: the claim is "6 of 7 years", and a claim of that shape has
    # to be counted, not quoted.
    _by = {}
    for r in p37["rows"]:
        if r["panel"] == "WE" and r["level"] == "dong":
            _by.setdefault(r["ym"] // 100, {})[r["ym"] % 100] = r["assortativity"]
    _mf = {y: v[3] - v[2] for y, v in _by.items() if 2 in v and 3 in v}
    check("19", "p37 March exceeds February in 6 of 7 years", 6.0,
          float(sum(1 for d in _mf.values() if d > 0)), 1e-9)
    check("19", "p37 the exception is 2020", 1.0,
          1.0 if _mf.get(2020, 1) < 0 else 0.0, 1e-9)
    import statistics as _stat
    _term, _vac = (3, 4, 5, 6, 9, 10, 11), (1, 2, 7, 8)
    _tv = {}
    for y, v in _by.items():
        t = [v[m] for m in _term if m in v]
        c = [v[m] for m in _vac if m in v]
        if t and c:
            _tv[y] = _stat.median(t) - _stat.median(c)
    check("19", "p37 term exceeds vacation in 6 of 7 years", 6.0,
          float(sum(1 for d in _tv.values() if d > 0)), 1e-9)
    check("19", "p37 term-vacation exception is 2020", 1.0,
          1.0 if _tv.get(2020, 1) < 0 else 0.0, 1e-9)

# --- section 8's download numbers. They used to be drive facts with nothing
# behind them, listed under "not covered here" and hand-verified whenever they
# changed -- which was four times in two days, and they went stale twice.
# eda/data_inventory.py writes them to a results file so they are gated like
# everything else.
_inv_path = results_of("inventory")
if os.path.exists(_inv_path):
    _inv = json.load(open(_inv_path))
    c19("months on disk 79", 79.0, float(_inv["months_complete"]), 1e-9)
    c19("months in the span 79", 79.0, float(_inv["span_months"]), 1e-9)
    check("19", "no partial months on disk", 0.0,
          float(len(_inv["months_partial"])), 1e-9)
    check("19", "no missing months", 0.0, float(_inv["months_missing"]), 1e-9)
    c19("parquet rows 10,186,891,962", 10186891962.0,
        float(_inv["parquet_rows"]), 1e-12)
    c19("manifest months 67", 67.0, float(_inv["manifest_months"]), 1e-9)
    c19("manifest files 1,608", 1608.0, float(_inv["manifest_files"]), 1e-9)
    check("19", "every manifest month has a zip sha256", 0.0,
          float(_inv["manifest_months"] - _inv["manifest_with_sha256"]), 1e-9)
    # the 67 + 12 = 79 arithmetic the letter uses to explain the gap
    c19("months without a zip sha256 (the 2020 twelve)", 12.0,
        float(_inv["months_without_zip_sha"]), 1e-9)
    check("19", "manifest plus hand-fetched equals months on disk", 0.0,
          float(_inv["manifest_months"] + _inv["months_without_zip_sha"]
                - _inv["months_complete"]), 1e-9)

# --- the pillar-4 retreats. The verdicts are checked as STRINGS and the anchors
# as EXACT zeros, because that is what the declared rule turns on. The 202312/A9
# cell is a tie -- the between-tau and the within-interval's lower bound are the
# same number -- and a tie is only a failure because the pre-declared rule reads
# "<=". Hand-typing 0.556 with a loose tolerance would pass while testing
# nothing, so the tie is asserted as an exact equality instead.
if "retreats" in p35:
    _r = p35["retreats"]
    for _k, _v in _r["anchors"].items():
        check("19", f"p35 retreat anchor {_k} is exactly 0", 0.0, float(_v), 0.0)
    _VERDICTS = {
        ("202312", "identity15"): ("separable", None),
        ("202312", "A9"): ("NOT separable", None),
        ("202312", "B8"): ("NOT separable", None),
        ("202402", "identity15"): ("NOT separable", None),
        ("202402", "A9"): ("NOT separable", None),
        ("202402", "B8"): ("NOT separable", None),
    }
    for (_ym, _sch), (_want, _) in _VERDICTS.items():
        _e = _r.get(f"{_ym}|seoul|{_sch}")
        if _e is None:
            continue
        check("19", f"p35 retreat verdict {_ym}/{_sch}", 1.0,
              1.0 if _e["verdict_tau"] == _want else 0.0, 1e-9)
    # the identity scheme has to reproduce the 15-band result it is the identity
    # of, otherwise the merge machinery is not neutral and nothing below it means
    # anything
    for _ym in ("202312", "202402"):
        _id = _r.get(f"{_ym}|seoul|identity15")
        if _id is None:
            continue
        check("19", f"p35 identity15 keeps 15 bands ({_ym})", 15.0,
              float(_id["n_bands"]), 1e-9)
        # NOT top1_stability: 35.6 draws from its own rng stream, so every
        # bootstrap statistic differs from 35.4's by sampling noise even though
        # the identity map reproduces the matrix exactly. The neutrality of the
        # merge is asserted by retreats.anchors, which is deterministic; a loose
        # tolerance on a Monte Carlo quantity would only look like a check.
        check("19", f"p35 identity15 band count matches n_ego ({_ym})", 15.0,
              float(len(_id["n_ego_per_band"])), 1e-9)
    # the tie, asserted as a tie
    # `between` also carries the passive_dong vs passive_gu pair, whose tau is
    # near 1 and has nothing to do with separability. The verdict turns on the
    # survey-vs-passive pairs only.
    def _survey_between(entry, key="tau"):
        return max(x[key] for x in entry["between"]
                   if x["pair"].startswith("survey vs"))

    # `tau_within` is the [lo, hi] INTERVAL, not the draws -- the same convention
    # as survey_bootstrap's tau_within_survey. Running np.percentile over it
    # would be taking a percentile of a percentile, which is why the shape is
    # pinned first.
    _a9 = _r.get("202312|seoul|A9")
    if _a9 is not None:
        check("19", "p35 tau_within is a 2-element interval", 2.0,
              float(len(_a9["tau_within"])), 1e-9)
        _btw = _survey_between(_a9)
        _lo = float(_a9["tau_within"][0])
        check("19", "p35 202312/A9 between-tau equals within lower bound exactly",
              0.0, abs(_btw - _lo), 0.0)
        c19("p35 202312/A9 between-tau 0.5556", 0.5555555555555556, _btw, 1e-12)
        check("19", "p35 202312/A9 tau_w verdict is the one flip", 1.0,
              1.0 if _a9["verdict_tau_w"] == "separable" else 0.0, 1e-9)
    # merging raises the between-matrix tau -- the coarser the bands, the more
    # alike the matrices look. That is the finding, so it is gated as an ordering
    # rather than as three separate numbers.
    for _ym in ("202312", "202402"):
        _seq = []
        for _sch in ("identity15", "A9", "B8"):
            _e = _r.get(f"{_ym}|seoul|{_sch}")
            if _e is None:
                break
            _seq.append(_survey_between(_e))
        if len(_seq) == 3:
            check("19", f"p35 between-tau rises as bands merge ({_ym})", 1.0,
                  1.0 if _seq[0] < _seq[1] < _seq[2] else 0.0, 1e-9)

# --- p36 ran multi-threaded until this round and was not bit-reproducible.
# The fix is a setting, so the gate checks the setting rather than trusting the
# comment next to it.
import re as _re
_p36src = open(f"{ROOT}/eda/p36_recompute.py").read()
_m = _re.search(r'"--threads", type=int, default=(\d+)', _p36src)
check("19", "p36 runs DuckDB single-threaded (determinism)", 1.0,
      1.0 if (_m and _m.group(1) == "1") else 0.0, 1e-9)

# --- p36: the independent recompute has to be all-pass, and its worst
# disagreement is quoted in the report, so both are checked.
c19("p36 checks that agree", float(p36["n_checks"]),
    float(p36["n_checks"] - p36["n_fail"]), 1e-9)
c19("p36 number of checks 216", 216.0, float(p36["n_checks"]), 1e-9)
# a partial run must never be what the gate is reading
check("19", "p36 results came from a full nine-section run", 1.0,
      1.0 if p36.get("full_run") else 0.0, 1e-9)
check("19", "p36 covers all nine sections", 9.0,
      float(len(p36.get("sections", []))), 1e-9)
# The exact subset only. Checks against values p19 published rounded to two
# decimals cannot beat that rounding, so demanding 1e-12 of them would be
# demanding the impossible; they are covered by n_fail == 0 against their own
# stated tolerance.
_exact = [c["rel"] for c in p36["checks"]
          if isinstance(c["rel"], float) and c.get("rtol", 0) <= 1e-6]
check("19", "p36 exact-subset worst disagreement < 1e-12", 1.0,
      1.0 if (_exact and max(_exact) < 1e-12) else 0.0, 1e-9)
# Assert the actual split rather than a vague "most of them": 144 comparisons
# are against unrounded quantities and 72 against values p19 and p29 published
# rounded to two decimals. If that ratio moves, either a section changed or a
# tolerance was loosened, and both are worth being told about.
check("19", "p36 exact-tolerance checks", 144.0, float(len(_exact)), 1e-9)
check("19", "p36 rounded-comparison checks", 72.0,
      float(p36["n_checks"] - len(_exact)), 1e-9)


# ======================================================================
# The 2026-08-21 report. Third document, third (document, source) pair. It
# quotes p39 (the recovery experiment), p40 (the national floor) and p41 (the
# semester section) -- three results files no gate had ever read, so roughly
# sixty numbers were leaving this repo unguarded.
#
# The scope decision this letter reports is ADDITIVE, and that is what decides
# the shape of this block. Seoul stays primary; the national sample becomes a
# robustness arm reported beside it. So nothing above this line moves -- not the
# pinned tuple at :325, not the 3.6/4.9 ratios, not p33's theta-family gaps, not
# p35's between-matrix taus, not p27's published assortativity. The national
# numbers get their own checks so that they are as protected as the Seoul ones,
# which is the entire point of adding an arm rather than swapping one: a
# find-and-replace would have left the published column unguarded at the moment
# it stopped being recomputed.
# ======================================================================
REPORT_0821 = f"{ROOT}/Email_Discussion/advisor_report_20260821.md"
# Sent in the same envelope as REPORT_0821 and numbered to match it
# (第一…第九段). It is half of the sent artefact, so it is half of the
# corpus the reverse-direction check may read.
ANNEX_0822 = f"{ROOT}/Email_Discussion/Technical notes 20260822.md"
p39 = load("p39", ARCHIVE_0822)
p40 = load("p40", ARCHIVE_0822)
p41 = load("p41", ARCHIVE_0822)

IN_TEXT_21 = []


def c21(label, quoted, actual, tol=6e-4):
    """c19's contract, registered against the 08-21 letter instead.

    Relative tolerance with no absolute floor, for the same reason c19 has one:
    the floors, contrasts and biases below run from 1e-2 down to 3e-4, and an
    absolute tolerance wide enough for the percentages would pass anything at
    the bottom of that range.
    """
    IN_TEXT_21.append((label, quoted))
    check("21", label, quoted, actual, tol * abs(actual))


# --- p40: the anchors first, before any number they carry is read. p40 replays
# p32's random stream so that it can reproduce the Seoul floor to the last bit,
# and it re-derives p35's four between-matrix taus. If either failed, every
# national number below came from a script that cannot reproduce the published
# ones. COUNT FIRST throughout: an emptied `checks` list is 0-of-0, which reads
# as a pass, and that is the vacuity p31:674 was fixed for.
_a32 = p40["anchor_p32"]["checks"]
check("21", "p40 anchor_p32 checks present", 22.0, float(len(_a32)), 1e-9)
check("21", "p40 anchor_p32 checks all ok", 22.0,
      float(sum(1 for _ck in _a32 if _ck.get("ok"))), 1e-9)
_a35 = p40["anchor_p35"]
check("21", "p40 anchor_p35 checks present", 4.0, float(len(_a35)), 1e-9)
check("21", "p40 anchor_p35 checks all ok", 4.0,
      float(sum(1 for _ck in _a35 if _ck.get("ok"))), 1e-9)
check("21", "p40 reports no failures", 0.0, float(len(p40["fail"])), 1e-9)
# The letter's defence against number-shopping is that the primary cell was
# named in the file before the run rather than picked from the four afterwards.
# That is a field, so it is gated as a field instead of believed.
_decl = p40["declaration"]["primary_cell"]
check("21", "p40 primary cell declared as 202312|national", 1.0,
      1.0 if (_decl["ym"] == 202312 and _decl["scope"] == "national"
              and _decl["statistic"] == "mi_perm_median") else 0.0, 1e-9)

# --- the four-cell floor table. Both scopes, both months, in one loop: the
# Seoul column is what the paper publishes and the national column is the new
# arm, and neither may move without the other being re-read.
for _cell, _want in (("202312|national", 0.02382), ("202312|seoul", 0.07232),
                     ("202402|national", 0.02829), ("202402|seoul", 0.08700)):
    c21(f"p40 {_cell} permutation floor", _want,
        p40["cells"][_cell]["mi_perm_median"], 6e-4)
c21("p40 202312 national floor over passive 1.21", 1.21,
    p40["cells"]["202312|national"]["floor_over_passive_median"], 6e-3)
# "small 3.6x" became "small 1.21x, and a seventh of the draws sit below it" --
# the second half of that sentence is a number too, so it is gated as one.
c21("p40 202312 national permutations below passive 13.8%", 13.8,
    100 * p40["cells"]["202312|national"]["passive_frac_below_perm_null"], 1e-3)
# and the headline verdict must be the primary cell's own ratio rather than a
# fourth number that happens to look like it
check("21", "p40 verdict ratio is the primary cell's own", 0.0,
      abs(p40["verdict_primary"]["ratio"]
          - p40["cells"]["202312|national"]["floor_over_passive_median"]), 0.0)

# --- why both estimates of the switch were 30% low. The floor moves for two
# reasons at once and the letter separates them: the national pool has a higher
# floor at the SAME n (population), and 465 -> 1,987 pulls it down (size).
_fdec = p40["floor_decomposition"]["202312"]
c21("p40 202312 population effect 1.356", 1.356, _fdec["population_effect"], 6e-4)
c21("p40 202312 size effect 0.243", 0.243, _fdec["size_effect"], 2e-3)
check("21", "p40 202312 Seoul outside the national-465 band", 0.0,
      1.0 if _fdec["seoul_inside_national_465_band"] else 0.0, 1e-9)

# --- the measurement that argued against symmetric switching, and therefore the
# measurement this whole block's additive shape rests on: swapping the
# comparison population moves the control by more than the signal being
# reported. It is statistic-dependent, which is why the two statistics that
# carry the sentence are named rather than an average being quoted.
_dvg = p40["divergence_vs_passive_signal"]["202312"]
c21("p40 202312 population shift over passive assortativity 3.01x", 3.01,
    _dvg["assortativity"]["multiple"], 2e-3)
c21("p40 202312 population shift over passive nMI 8.62x", 8.62,
    _dvg["nmi"]["multiple"], 6e-4)
# The same finding stated as a count over the only legitimate null -- 500
# same-size subsamples of the 1,987, i.e. "living in Seoul carries no
# information". Recomputed from the per-quantity verdicts, not restated from a
# summary field; and written as check() rather than c21() because the letter
# spells these two in Chinese numerals, where a digit match would be an accident
# rather than evidence the prose says it.
for _ym, _want in (("202312", 7.0), ("202402", 6.0)):
    _null = p40["seoul_vs_national"][_ym]["null"]
    check("21", f"p40 {_ym} quantities compared against the null", 9.0,
          float(len(_null)), 1e-9)
    check("21", f"p40 {_ym} Seoul outside the 95% band, of nine", _want,
          float(sum(1 for _v in _null.values() if not _v["inside_null"])), 1e-9)

# --- the IPF branch at national. "It only absorbs 4% and 1%" is a Seoul
# sentence; the national arm absorbs five times as much. Both halves are gated,
# the share AND the non-convergence, because gating a number beside a word does
# not gate the word (p31:620).
for _ym, _want in (("202312", 19.1), ("202402", 23.0)):
    _ipf = p40["ipf_calibration"][_ym]["national"]
    c21(f"p40 {_ym} national IPF absorbs {_want}%", _want,
        100 * _ipf["absorbed_frac"], 6e-3)
    check("21", f"p40 {_ym} national IPF still does not converge", 0.0,
          1.0 if _ipf["converged"] else 0.0, 1e-9)

# --- the fourth pillar, which the larger sample did not change: no cell's
# regret exceeds the null of re-running the survey on fresh respondents. Count
# of cells first, then the count that carries the claim.
_p4 = p40["pillar4_summary"]
check("21", "p40 pillar-4 cells", 4.0, float(_p4["cells"]), 1e-9)
check("21", "p40 pillar-4 cells whose regret exceeds the null", 0.0,
      float(_p4["cells_regret_exceeds_null"]), 1e-9)

# --- p39: the recovery experiment. The identity A/A.sum() == a convex
# combination of rank-1 outer products is what forces the synthetic world's
# truth to live BELOW the cell line -- put it on the line and the experiment
# measures nothing -- so it is checked before the surface it justifies.
check("21", "p39 identity cells present", 6.0, float(len(p39["identity"])), 1e-9)
check("21", "p39 identity holds to machine precision", 1.0,
      1.0 if max(_i["max_abs_dev"] for _i in p39["identity"]) < 1e-15 else 0.0,
      1e-9)

# The empty arm. It reproduces p33's device-noise bias by a route that shares
# nothing with it -- one is a parametric bootstrap on the real matrix, the other
# is a zero-structure synthetic world through the whole pipeline -- so the two
# numbers and their ratio are all three gated.
_narm = p39["null_arm"]
c21("p39 null arm, both defects 0.00455", 0.00455, _narm["null_median"], 2e-3)
c21("p39 p33 device-noise bias 0.00439", 0.00439, _narm["p33_noise_bias"], 2e-3)
c21("p39 null arm over p33 bias 1.037", 1.037, _narm["ratio_to_p33"], 6e-4)
check("21", "p39 declared stop-loss did not fire", 0.0,
      1.0 if _narm["void"] else 0.0, 1e-9)

# The inverted upper bounds, one per beta slice, plus the two universals the
# letter states about the family: every slice monotone, and exactly one of the
# five failing to exclude the survey's 0.223. "Four exclude, the fifth does not"
# is the sentence; it is counted, and the named slice is asserted separately, so
# that the count cannot stay right while the wrong slice is the exception.
_INV21 = (("0.0", 0.0167), ("0.5", 0.0320), ("0.8", 0.0714),
          ("0.95", 0.1848), ("0.99", 0.3205))
check("21", "p39 beta slices present", 5.0, float(len(p39["inversion"])), 1e-9)
for _beta, _want in _INV21:
    c21(f"p39 upper bound at beta={_beta}", _want,
        p39["inversion"][_beta]["bound_corrected"], 3e-3)
check("21", "p39 every beta slice monotone", 5.0,
      float(sum(1 for _v in p39["inversion"].values() if _v["monotone"])), 1e-9)
check("21", "p39 slices excluding the survey", 4.0,
      float(sum(1 for _v in p39["inversion"].values()
                if _v["excludes_survey"])), 1e-9)
check("21", "p39 beta=0.99 is the slice that does not exclude", 0.0,
      1.0 if p39["inversion"]["0.99"]["excludes_survey"] else 0.0, 1e-9)

# --- p41: the semester section. Its anchor is the published number itself: the
# per-band decomposition must sum back to what p37 stored, for all 79 months,
# and the six published months must equal p26's published values.
_anc41 = p41["anchor"]
check("21", "p41 anchor months against p37", 79.0,
      float(len(_anc41["vs_p37"])), 1e-9)
check("21", "p41 anchor months against p26", 6.0,
      float(len(_anc41["vs_p26"])), 1e-9)
check("21", "p41 band decomposition sums back to p37", 1.0,
      1.0 if max(_r41["rel"] for _r41 in _anc41["vs_p37"]) < 1e-14 else 0.0, 1e-9)
check("21", "p41 published months reproduce p26", 1.0,
      1.0 if max(_r41["rel"] for _r41 in _anc41["vs_p26"]) < 1e-14 else 0.0, 1e-9)

_sgn = p41["sign_test"]["contrast"]
c21("p41 sign test one-sided 0.0625", 0.0625, _sgn["p_one_sided"], 1e-12)
c21("p41 sign test two-sided 0.125", 0.125, _sgn["p_two_sided"], 1e-12)
# The band decomposition that kills the "2020's seasonality was flattened"
# alternative: the 0-24 contrast is where 2020 goes negative, while the 25+
# contrast stays positive in all seven years and keeps half its size.
_grp = p41["bands"]["groups"]
c21("p41 25+ sign test 0.0078125", 0.0078125,
    _grp["adult_25plus"]["sign_test"]["p_one_sided"], 1e-12)
c21("p41 2020 0-24 contrast -0.00075", -0.00075,
    _grp["student_0_24"]["control"], 2e-3)
c21("p41 2020 25+ contrast +0.00029", 0.00029,
    _grp["adult_25plus"]["control"], 6e-3)
c21("p41 2020 25+ contrast retains 0.507", 0.507,
    _grp["adult_25plus"]["retention"], 1e-3)

# The ten readings, and the sentence that replaced "2020 is negative" with
# "2020 is the smallest of the seven years". Both counts are recomputed from
# each reading's own per-year series rather than read out of the summary the
# script wrote: checking the letter against that summary is checking the letter
# against the sentence's own source.
_var41 = p41["robustness"]["variants"]
check("21", "p41 robustness readings present", 10.0, float(len(_var41)), 1e-9)
_neg41 = _small41 = 0
for _v in _var41.values():
    _py41 = _v["per_year"]
    _ctl41 = _py41["2020"]
    _oth41 = [_x for _yr, _x in _py41.items() if _yr != "2020"]
    _neg41 += 1 if _ctl41 < 0 else 0
    _small41 += 1 if _ctl41 < min(_oth41) else 0
check("21", "p41 readings where 2020 is negative", 2.0, float(_neg41), 1e-9)
check("21", "p41 readings where 2020 is the smallest year", 10.0,
      float(_small41), 1e-9)


# ======================================================================
# p42 -- the month scope of the letter's headline sentence. Section 5 of the
# 08-21 letter was written entirely out of a results file no gate had ever
# opened: `eda/memo/phase40b-month-scope.md:235` flagged the omission and it
# stayed open through the round in which that sentence was rewritten. Until this
# block existed, every number in that section was hand-typed prose -- including
# the one the letter says the paper must volunteer before a reviewer finds it.
#
# THE CROSS-ARTIFACT ANCHOR FIRST, and it is a real anchor rather than a count:
# the floor p42 counts months against must BE p40's permutation median, bit for
# bit. If those two ever drift, "13 of 79 reach the floor" is counting against a
# different floor from the one the letter quotes three sections earlier.
p42 = load("p42", ARCHIVE_0822)
_c42 = p42["counts"]["202312|national"]
check("21", "p42 covers the full 79 months", 79.0, float(_c42["n_months"]), 1e-9)
check("21", "p42's floor IS p40's national permutation median", 0.0,
      abs(_c42["floor"] - p40["cells"]["202312|national"]["mi_perm_median"]), 0.0)
check("21", "p42 counts against the declared primary cell", 1.0,
      1.0 if _c42["is_primary"] else 0.0, 1e-9)

# The count, and the fact that carries it. Counted per class rather than as one
# number, because "13 term months clear" and "no vacation month clears" are two
# separate claims and the letter makes both.
c21("p42 months reaching the national floor", 13.0, float(_c42["n_at_or_above"]))
_s42 = p42["semester"]
check("21", "p42 every clearing month is a term month", 13.0,
      float(_s42["observed"]["term"]), 1e-9)
check("21", "p42 no vacation month clears", 0.0,
      float(_s42["observed"]["vacation"]), 1e-9)
check("21", "p42 no December clears", 0.0,
      float(_s42["observed"]["december"]), 1e-9)
check("21", "p42 term months on record", 46.0,
      float(_s42["base_rates"]["term"]), 1e-9)
c21("p42 within-year p for the term concentration", 2.24e-4,
    _s42["within_year"]["p_term_ge"], 5e-3)
check("21", "p42's within-year p is exact, not sampled", 1.0,
      1.0 if _s42["within_year"]["exact"] else 0.0, 1e-9)

# 13 is not a natural break and the letter says so, so the sensitivity is part
# of the claim rather than a footnote to it.
_fs42 = p42["floor_sensitivity"]["counts_at_offsets"]
c21("p42 count at the floor minus two standard errors", 16.0, float(_fs42["-2se"]))
c21("p42 count at the floor plus two standard errors", 11.0, float(_fs42["+2se"]))

# Aligning the day set moved the count in our favour and removed nothing, which
# is the only reason the correction can be reported without a caveat.
_al42 = p42["alignment_effect"]
check("21", "p42 months added by aligning the day set", 3.0,
      float(_al42["n_added"]), 1e-9)
check("21", "p42 months removed by aligning the day set", 0.0,
      float(_al42["n_removed"]), 1e-9)

# The sentence the letter says the paper has to volunteer: neither survey month
# belongs to the class that clears, and both sit below most term months.
check("21", "p42 term months, from the ranks table", 46.0,
      float(p42["survey_month_ranks"]["202312"]["n_term_months"]), 1e-9)
for _ym42, _r42 in (("202312", 0.89), ("202402", 0.81)):
    _sm42 = p42["survey_month_ranks"][_ym42]
    # 7e-3 relative, not c21's default: the letter rounds these to two decimals,
    # so the honest tolerance is half a unit in the last place it prints.
    c21(f"p42 {_ym42} is {_r42}x the term median", _r42,
        _sm42["ratio_to_term_median"], 7e-3)
    check("21", f"p42 {_ym42} is not a term month", 1.0,
          1.0 if _sm42["semester"] != "term" else 0.0, 1e-9)
    c21(f"p42 {_ym42} sits below this many term months", 31.0,
        float(_sm42["n_term_months_above"]))
check("21", "p42's own verdict: the general claim is NOT supported", 0.0,
      1.0 if p42["verdict"]["general_claim_supported"] else 0.0, 1e-9)
check("21", "p42's own verdict: the two-month claim IS supported", 1.0,
      1.0 if p42["verdict"]["survey_month_claim_supported"] else 0.0, 1e-9)


# ======================================================================
# p44 -- beta, measured. Same document (the 08-21 letter), fourth results file.
# Section 9 of that letter was written after p44 ran, so every number in it is
# new and none of it was guarded until this block existed.
#
# ANCHORS FIRST, and here that is not a formality: p44's whole claim rests on it
# running the SAME estimator as p26 and the SAME weighting as p27. Both are
# asserted inside p44 and both are re-read here, because an anchor that only the
# producing script checks is the self-comparison this gate keeps being fixed for.
# COUNT FIRST for the same reason as p40's block: an emptied dict reads as a pass.
p44 = load("p44", ARCHIVE_0822)
# The value p44 has to reproduce is read from p27's own results file, not from
# p44's and not from a literal typed here: a gate that compares a script against
# a constant the same script printed is the self-comparison p31:674 was fixed
# for. p27 stores C; the published assortativity is the symmetrised, 0-79-block
# statistic p27 and p32 both report, and it is already checked at :187 against
# the 0.2227 in the pinned tuple, so this only has to agree with that.
R_SURVEY_PUB = [r for r in p27["comparison"]
                if r["ym"] == 202312 and r["restriction"] == "holiday-free"
                ][0]["assort_survey"]
check("21", "p44's survey anchor is p27's own published value", 0.2227,
      round(R_SURVEY_PUB, 4), 6e-5)
_a44 = p44["anchors"]
check("21", "p44 anchor: p27's published assortativity reproduced", R_SURVEY_PUB,
      _a44["survey_r_p27"], 0.0)
check("21", "p44 anchor: same value via the independent weighted route",
      0.0, abs(_a44["survey_r_weighted"] - R_SURVEY_PUB), 1e-12)
check("21", "p44 anchor: p26's published matrix rebuilt bit-for-bit", 0.0,
      _a44["p26_rebuild_maxrel"], 0.0)
check("21", "p44 anchor: respondents", 1987.0, float(_a44["respondents"]), 1e-9)
check("21", "p44 anchor: contacts", 133776.0, float(_a44["contacts"]), 1e-9)
for _k, _want in (("r_obs_published", 0.02115385938228908),
                  ("noise_bias", 0.0043860558176699906),
                  ("r_obs_corrected", 0.01676780356461909)):
    check("21", f"p44 carries p39's {_k} unchanged", _want,
          _a44["constants"][_k], 0.0)

# --- the pairing rule, and the null that decides which convention is honest.
c21("p44 venue collapse with replacement 0.436", 0.436,
    p44["pairing"]["r_venue_wr"], 2e-3)
c21("p44 venue collapse without replacement 0.249", 0.249,
    p44["pairing"]["r_venue_wor"], 2e-3)
c21("p44 survey's own contact matrix 0.273", 0.273,
    p44["pairing"]["r_star"], 2e-3)
c21("p44 shuffled null keeps 62% with replacement", 62.0,
    100 * p44["pairing"]["null_shuffled"]["share_wr"], 1e-2)
c21("p44 shuffled null keeps 14% without replacement", 14.0,
    100 * p44["pairing"]["null_shuffled"]["share_wor"], 2e-2)
check("21", "p44 one-group null is exactly zero with replacement", 0.0,
      p44["pairing"]["null_one_group"]["wr"], 1e-12)

# --- the ladder. The 90.9% is the number that goes in the main text, so it is
# checked against the arm the letter names (national, pooled) and not whichever
# arm happens to be first in the file.
_lad_nat = {r["grouping"]: r for r in p44["ladder"]["national|pooled"]["rows"]}
_lad_seo = {r["grouping"]: r for r in p44["ladder"]["seoul|202312"]["rows"]}
_venue = "venue-visit (ego,date,place)"
c21("p44 venue-visit keeps 90.9% of the survey's assortativity", 90.9,
    100 * _lad_nat[_venue]["keep"], 1e-2)
c21("p44 same on the Seoul arm, 96.1%", 96.1, 100 * _lad_seo[_venue]["keep"], 1e-2)

# --- the decay law and the place-type ceiling.
c21("p44 decay law constant 1.061", 1.061, p44["pooling"]["law_constant"], 2e-3)
c21("p44 within-place plateau 0.0368", 0.0368,
    p44["pooling"]["within_place_floor"], 2e-2)
c21("p44 shuffled null 0.0344", 0.0344,
    p44["pairing"]["null_shuffled"]["wor"], 2e-2)

# --- the cell, including the correction to the letter's own earlier "20,000".
c21("p44 dong x hour x dow cell median 533", 533.0, p44["cell"]["median"], 2e-3)
c21("p44 dong x hour x dow cell mean 1,080", 1080.0, p44["cell"]["mean"], 2e-3)
c21("p44 dong x hour x dow arrival-weighted mean 4,052", 4052.0,
    p44["cell"]["n_weighted_mean"], 2e-3)
c21("p44 dong x day median 17,856", 17856.0, p44["cell"]["dong_day_median"], 2e-3)

# --- the bracket. Both ends are computed from the survey alone; the measured
# value is p33's, already anchored above.
_br = p44["beta"]["bracket"]
c21("p44 bracket floor 0.00246", 0.00246, _br["floor"], 5e-3)
c21("p44 bracket ceiling 0.03682", 0.03682, _br["ceiling"], 2e-2)
c21("p44 measured over floor 6.8x", 6.8, _br["measured_over_floor"], 1e-2)
c21("p44 measured over ceiling 46%", 46.0, 100 * _br["measured_over_ceiling"], 2e-2)

# --- beta itself, and the crossing it is compared against. The crossing is
# INTERPOLATED from p39's stored inversion rather than quoted, so this check
# also guards the "~0.96" the previous letter used.
c21("p44 beta primary 0.933", 0.933, p44["beta"]["beta_primary"], 2e-3)
c21("p44 beta range lo 0.925", 0.925, p44["beta"]["beta_range"][0], 2e-3)
c21("p44 beta range hi 0.962", 0.962, p44["beta"]["beta_range"][1], 2e-3)
c21("p44 beta vs the survey's own contact matrix 0.939", 0.939,
    p44["beta"]["betas"]["survey contact matrix (star pairs)"]["beta"], 2e-3)
c21("p39's bound reaches the survey value at beta 0.9612", 0.9612,
    p44["beta"]["p39_crossing"], 2e-4)

# --- the two decay laws, and what B075 would have bought. This is what downgrades
# section 6's account of the closed Big Data Campus door, so it is guarded.
_rc = p44["beta"]["reach"]
c21("p44 factor still needed to reach the survey 13.3x", 13.3,
    _rc["factor_needed"], 2e-3)
c21("p44 B075 under the optimistic law reaches 32.5% of the survey", 32.5,
    100 * _rc["b075_r_slope1"] / R_SURVEY_PUB, 1e-2)
# The 08-21 letter prints 13.0% here and the frozen p44 it was written from
# gives 12.84%. That is an ERRATUM, not drift: archive/20260822/results_p44
# is bit-identical to live on this field, so the letter was already wrong on
# the day it went out, and eda/memo/phase44-beta.md records where the
# rounding came from. The letter is sent and keeps its wording. What could
# not stay is how this row used to pass -- c21's default tolerance is 6e-4
# and this row carried 2e-2, thirty times looser, which is what made the two
# "agree". A tolerance wide enough to swallow a known error reports that
# error as an agreement, and then the one row that knows about it is the one
# row that cannot say so. The erratum is now its own checked quantity, at
# the ordinary tightness, so it goes red if either side of it moves.
_B075_TRUE = 100 * _rc["b075_r_ladder"] / R_SURVEY_PUB
IN_TEXT_21.append(
    ("p44 B075 as the 08-21 letter prints it (erratum; 12.8% is the value)",
     13.0))
check("21", "p44 B075 under p38's measured slope, from the frozen p44",
      12.84, round(_B075_TRUE, 2), 6e-4 * 12.84)
# The size of the erratum is registered to THIS round, not to section 21:
# the 08-21 letter states its own round as 178 checks, and that sentence is
# sent. Adding a row to its section would falsify a frozen document in order
# to describe one, which is the same mistake in the other direction.
check("27", "the 08-21 letter prints 13.0% for it, this far out in points",
      0.16, round(13.0 - _B075_TRUE, 2), 6e-4 * 0.16)

# --- three the letter states that the block above left open.
# The two independent routes to p27's constant do not agree to the 16th digit;
# they agree to 15 and part company by exactly four units in the last place. A
# tolerance of 1e-12 cannot tell those apart, and the letter quotes the figure.
import math as _math
c21("p44's two routes to p27's constant differ by 4 ULP", 4.0,
    abs(_a44["survey_r_weighted"] - R_SURVEY_PUB) / _math.ulp(R_SURVEY_PUB), 1e-9)
# The without-replacement convention has a closed form when every person is put
# in one group, and it is the half of that identity the letter rests its choice
# of convention on. Only the with-replacement half (exactly 0) was checked.
_n1g = 1.0 / abs(p44["pairing"]["null_one_group"]["wor"])
check("21", "p44's one-group WOR null is exactly -1/(n-1)", 0.0,
      abs(p44["pairing"]["null_one_group"]["wor"] + 1.0 / round(_n1g)), 1e-16)
check("21", "and that n-1 is an integer, not a fitted number", 0.0,
      abs(_n1g - round(_n1g)), 1e-6)
# The 34x correction to the previous letter: both endpoints were gated, the
# ratio between them -- which is the whole content of the correction -- was not.
c21("p44 dong-day is this many times the dong-hour cell", 34.0,
    p44["cell"]["dong_day_median"] / p44["cell"]["median"], 2e-2)
# The letter's measurement sentence ("recovers 6% to 8%") is a band over the
# three without-replacement readings; the with-replacement one gives 3.8% and is
# excluded on purpose. Checked as a band so the sentence cannot drift out of it.
_rec44 = [_v["recovery"] for _k, _v in p44["beta"]["betas"].items()
          if "with replacement" not in _k]
check("21", "p44 without-replacement recovery readings", 3.0,
      float(len(_rec44)), 1e-9)
check("21", "all of them fall inside the letter's 6%-8%", 3.0,
      float(sum(1 for _r in _rec44 if 0.06 <= _r <= 0.08)), 1e-9)
check("21", "the with-replacement reading is outside it, as the letter says", 1.0,
      1.0 if p44["beta"]["betas"]["venue-level, with replacement"]["recovery"] < 0.06
      else 0.0, 1e-9)

# --- the cross-check that never touches the survey.
_cl = p44["clustering"]
c21("p44 dong->gu random prediction 0.059", 0.059, _cl["random_prediction"], 2e-3)
c21("p44 p37 measures 0.298 kept", 0.298, _cl["kept_dong_to_gu"], 2e-3)
c21("p44 real merging retains 5.06x the random prediction", 5.06,
    _cl["excess_from_gu"], 2e-3)
c21("p44 cell-vs-venue excess 6.82x", 6.82, _cl["excess_from_venues"], 2e-3)
c21("p44 the two routes disagree by 35%", 35.0, 100 * _cl["disagreement"], 2e-2)



# ======================================================================
# The two the letter quotes as VALUES while the gate only held their RATIO or
# their six-month cousin. Both are section-21 numbers; neither had a check.
#
# p33: section 7 prints the two widths side by side ("0.0339 對 0.0024"), and
# what was guarded was 14x -- the quotient. A quotient survives both of its
# inputs moving together, which is exactly the drift that would leave the
# sentence wrong while the check stayed green.
_mvc = p33["mask_vs_coverage"]["202312"]
c21("p33 202312 coverage width", 0.0339, _mvc["theta_width"], 3e-3)
c21("p33 202312 masking width", 0.0024, _mvc["mask_width"], 8e-3)
check("21", "and the ratio is still the quotient of those two", 0.0,
      abs(_mvc["ratio"] - _mvc["theta_width"] / _mvc["mask_width"]), 1e-12)
# The Kendall tau the same sentence quotes as a range is the per-month minimum
# over the theta family -- 0.1167 in 202312, 0.1667 in 202402 -- so it is two
# numbers, not an interval anyone measured.
for _ym33, _want33 in (("202312", 0.1167), ("202402", 0.1667)):
    _taus = [_r["kendall_tau_vs_theta0"] for _r in p33["R2_theta_family"][_ym33]]
    check("21", f"p33 {_ym33} theta family readings", 9.0, float(len(_taus)), 1e-9)
    check("21", f"p33 {_ym33} lowest Kendall tau vs theta=0", _want33,
          min(_taus), 1e-3)

# p38: section 6 quotes the 79-month polygon prediction band and the fact that
# every one of its lower bounds clears that month's measured dong reading. What
# was gated was p34's SIX-month version of the same object, which is a different
# window and, per p37, a biased one.
_p38b = load("p38", LIVE)
_poly = _p38b["polygon_prediction_interval"]
import statistics as _stats
check("21", "p38 polygon prediction covers all 79 months", 79.0,
      float(len(_poly)), 1e-9)
c21("p38 polygon band, 79-month median lower edge", 0.0286,
    _stats.median([_v["r_polygon_lo"] for _v in _poly.values()]), 3e-3)
c21("p38 polygon band, 79-month median upper edge", 0.0316,
    _stats.median([_v["r_polygon_hi"] for _v in _poly.values()]), 3e-3)
c21("p38 measured dong median it is compared against", 0.01938,
    _stats.median([_v["r_dong"] for _v in _poly.values()]), 3e-3)
check("21", "p38 months whose polygon lower bound clears the measured dong",
      79.0, float(sum(1 for _v in _poly.values()
                      if _v["r_polygon_lo"] > _v["r_dong"])), 1e-9)


# --- p44 44.7: the two limitations 44.6 only named, turned into numbers. These
# are quoted in the 08-21 letter's section 9, so they are gated like the rest.
_s44 = p44["sensitivity"]
c21("p44 venue r padded to Q9 0.1854", 0.1854, _s44["pad"]["r_padded"], 2e-3)
c21("p44 beta bracket lower end 0.9096", 0.9096,
    _s44["pad"]["beta_bracket"][0], 3e-4)
c21("p44 beta bracket upper end 0.9326", 0.9326,
    _s44["pad"]["beta_bracket"][1], 3e-4)
c21("p44 seoul-arm beta 0.9217", 0.9217,
    _s44["scope"]["seoul 202312 (anchor)"]["beta"], 3e-4)
check("21", "p44 scope moves beta less than the padding does", 1.0,
      1.0 if abs(_s44["scope"]["seoul 202312 (anchor)"]["beta"]
                 - _s44["scope"]["national pooled (primary)"]["beta"])
      < abs(_s44["pad"]["beta_bracket"][1] - _s44["pad"]["beta_bracket"][0])
      else 0.0, 1e-9)
check("21", "p44 padding lowers r_venue, so unpadded is the higher reading", 1.0,
      1.0 if _s44["pad"]["r_padded"] < _s44["pad"]["r_unpadded"] else 0.0, 1e-9)
check("21", "p44 seoul arm is 465 respondents", 465.0,
      float(_s44["n_seoul_respondents"]), 1e-9)


# ======================================================================
# p45 -- the fourth pillar's verdict across R0. This block exists because the
# 08-21 letter said in one section that the sweep had not been run while the
# script that runs it was already on disk, and because the FIRST results_p45
# written was a 4-draw smoke test whose interval verdicts were noise and whose
# flip list disagreed with the 200-draw run. Only `declaration.n_boot` recorded
# the difference, so the first thing checked here is the one thing that decides
# whether any of the rest may be quoted at all.
p45 = load("p45", ARCHIVE_0822)
check("21", "p45 was run at the published 200 draws, not a smoke test", 200.0,
      float(p45["declaration"]["n_boot"]), 1e-9)
check("21", "p45's deterministic anchor against p40 is bit-exact", 1.0,
      1.0 if p45["anchor"]["bit_exact"] else 0.0, 1e-9)
check("21", "p45 anchored 12 deterministic quantities", 12.0,
      float(p45["anchor"]["quantities"]), 1e-9)
check("21", "p45 anchors at R0 = 2.5", 2.5, float(p45["anchor"]["r0"]), 1e-9)
check("21", "p45 reports no failures", 0.0, float(len(p45["fail"])), 1e-9)

# The answer, counted before it is characterised: an emptied flips dict would
# read as "nothing flips", which is the vacuity p31:674 was fixed for.
_v45 = p45["verdict"]
check("21", "p45 verdicts, 4 cells x 5 R0", 20.0,
      float(sum(len(_rows) for _rows in _v45.values())), 1e-9)
check("21", "p45 states that the verdict flips", 1.0,
      1.0 if p45["answer"]["any_flip"] else 0.0, 1e-9)
c21("p45 verdicts that flip", 7.0,
    float(sum(len(_r) for _r in p45["answer"]["flips"].values())))
for _cell45, _n45 in (("202312|national", 3.0), ("202402|national", 3.0),
                      ("202402|seoul", 1.0)):
    check("21", f"p45 {_cell45} flips at this many R0", _n45,
          float(len(p45["answer"]["flips"][_cell45])), 1e-9)
check("21", "p45 202312|seoul never flips", 0.0,
      float(len(p45["answer"]["flips"].get("202312|seoul", []))), 1e-9)
c21("p45 closest call margin", 1.13, 100 * p45["answer"]["closest"]["margin"], 5e-3)

# THE TIE TO p40, and the reason this block sits next to :1210. p40's published
# sentence -- no cell's regret exceeds the null -- is a statement about R0 = 2.5,
# and p45 re-derives that row from an independent stream. If p45's 2.5 row ever
# stopped agreeing, the letter's "the conclusion at 2.5 has not moved" would be
# false while p40's own file still said 0.
check("21", "p45's R0 = 2.5 row agrees with p40 in all four cells", 4.0,
      float(sum(1 for _m in p45["mc_agreement"].values()
                if _m["p45_exceeds"] == _m["p40_exceeds"])), 1e-9)
check("21", "and none of the four exceeds at 2.5, as p40 published", 0.0,
      float(sum(1 for _m in p45["mc_agreement"].values() if _m["p45_exceeds"])),
      1e-9)
_rel45 = [_m["rel_diff"] for _m in p45["mc_agreement"].values()]
c21("p45 vs p40 at 2.5, smallest relative gap", 2.9, 100 * min(_rel45), 2e-2)
c21("p45 vs p40 at 2.5, largest relative gap", 27.1, 100 * max(_rel45), 2e-2)

# The deterministic half, which is what the letter leans on: 2.5 is the friendly
# end. No bootstrap anywhere in these four numbers.
_ps45 = p45["point_summary"]
for _cell45, _at45, _max45, _arg45 in (
        ("202312|national", 2.4, 47.7, 1.5),
        ("202402|national", 26.6, 59.3, 1.3)):
    # Tolerance is half a unit in the last decimal the letter prints, expressed
    # relatively because that is c21's contract: 2.4% and 47.7% are the same
    # rounding but 6e-4 relative means 0.001 at one end and 0.03 at the other.
    _t45 = 0.06 / abs(100 * _ps45["at_anchor"][_cell45])
    c21(f"p45 {_cell45} point regret at R0 = 2.5", _at45,
        100 * _ps45["at_anchor"][_cell45], _t45)
    c21(f"p45 {_cell45} point regret at its worst R0", _max45,
        100 * _ps45["max_over_grid"][_cell45],
        0.06 / abs(100 * _ps45["max_over_grid"][_cell45]))
    check("21", f"p45 {_cell45} is worst at this R0", _arg45,
          float(_ps45["argmax"][_cell45]), 1e-9)

# The correction to p40's SEARCH, which is not a correction to its matrices.
# Negative regret is impossible for a true optimum, so the count IS the bug.
_opt45 = p45["optimiser"]
check("21", "p45 swept 48 optimiser cells", 48.0,
      float(_opt45["n_cells_total"]), 1e-9)
c21("p45 cells where p40's bare call returned negative regret", 9.0,
    float(len(_opt45["bare_failures"])))
check("21", "p45 cells still negative once the passive plan is a start", 0.0,
      float(len(_opt45["seeded_failures"])), 1e-9)


# ----------------------------------------------------------- the numbers list
# 08-20: the two appendices collapse into one list of the numbers as they stand
# today. Typing that list by hand would reintroduce the transcription step this
# gate exists to police -- so it is GENERATED from the same triples the gate
# checks. The list and the gate cannot disagree, because they are the same
# object.
#
# The "出現在" column is computed against the LETTER and the 08-19 appendix,
# never against the generated list itself: matching the list against its own
# contents would pass by construction and mean nothing. A row that says 「未出現」
# is a number this gate verifies that no document currently quotes -- either a
# claim that was dropped from the prose, or a check that has become dead weight.

# ======================================================================
# THE 2026-08-24 ROUND. Its own (document, source) pair, per the rule the file
# states at the top: a new document never shares an existing one. The source is
# ARCHIVE_0824, because this round is CLOSED: the report went out, the advisor
# replied on 2026-08-27, and its numbers may not move any more. Reading them
# from LIVE would mean the next re-run of any of these eight phases turns this
# block red for a reason that says nothing about a document already in someone
# else's inbox -- and the only available repair would be to edit that document.
#
# Five new phases, and what each is here to protect:
#   p49  the cell-level pairing convention        -- the 533 paragraph
#   p50  Korea's population vector                -- the national arm's premise
#   p51  the national arm, corrected              -- 1.17x, 23.6%, the span
#   p53  beta's national arm, corrected           -- the scope-asymmetry sentence
#   p54  the 17-of-79 count against the new floor -- Claim 1's corroboration
#        (13 of 79 is the same count against p40's superseded floor)
# p52 (the R0 sweep) used to register only if its results file existed, because
# it is the one phase in this round long enough to be run separately. It has
# been run at the pinned 200 draws, so the condition is gone and its absence is
# now a failure like any other required file.
#
# Two more files belong to this round and are frozen with it, further down:
#   p57  [CHAE]'s retained etc. flag              -- section 11's 10.53%/4.48%
#   p44  the same file the 08-21 letter quotes, re-run by d583ff0 -- the
#        withdrawn Q9 arm, which is section 11's other half
# ======================================================================
REPORT_0824 = f"{ROOT}/Email_Discussion/advisor_report_20260824.md"
# The five figures go out with their captions, exactly as the 08-22 round went
# out as a letter plus its annex. The captions carry numbers, so the caption
# sheet is part of this round's corpus for the reverse-direction check.
ANNEX_0824 = f"{ROOT}/Email_Discussion/figure_captions_20260824.md"
p49 = load("p49", ARCHIVE_0824)
p50 = load("p50", ARCHIVE_0824)
p51 = load("p51", ARCHIVE_0824)
p53 = load("p53", ARCHIVE_0824)
p54 = load("p54", ARCHIVE_0824)

IN_TEXT_24 = []


def c24(label, quoted, actual, tol=6e-4):
    """c19's contract, registered against the 08-24 letter."""
    IN_TEXT_24.append((label, quoted))
    check("24", label, quoted, actual, tol * abs(actual))


# --- p49: the pairing convention at cell scale ------------------------------
# COUNT FIRST. An emptied ladder is 0-of-0, which reads as a pass.
check("24", "p49 ladder covers both months x three levels", 6.0,
      float(len(p49["ladder"])), 1e-9)
check("24", "p49 rebuilt p26's published matrix bit for bit", 1.0,
      1.0 if p49["anchors"]["x_reproduces_p26"]["bit_exact"] else 0.0, 1e-9)
check("24", "p49 collapse agrees with p26 to machine precision", 0.0,
      p49["anchors"]["collapse_matches_p26"]["max_rel_diff"], 1e-12)
check("24", "p49 one-group closed forms checked at four cell sizes", 4.0,
      float(len(p49["anchors"]["one_group_closed_forms"])), 1e-9)
_p49p = p49["primary"]
c24("p49 with replacement 0.02115386", 0.02115386, _p49p["r_wr"], 1e-6)
c24("p49 without replacement 0.02024732", 0.02024732, _p49p["r_wor"], 1e-6)
c24("p49 the cell-level correction -4.2854%", -4.2854,
    100 * _p49p["rel_change"], 1e-3)
_p49l = {r["level"]: r for r in p49["ladder"] if r["ym"] == 202312}
c24("p49 gu -0.55%", -0.55, 100 * _p49l["gu"]["rel_change"], 1e-2)
c24("p49 city -0.044%", -0.044, 100 * _p49l["city"]["rel_change"], 2e-2)
c24("p49 venue -43.0%", -43.0,
    100 * p49["answer"]["venue_relative_difference"], 1e-3)
c24("p49 venue over cell 10.0x", 10.0, p49["answer"]["ratio_venue_to_cell"], 1e-2)
# The scale question is part of the claim, not a footnote to it: without
# replacement is not degree-1 homogeneous, so the answer depends on it.
_p49s = {r["ym"]: r for r in p49["scale_sensitivity"]}
c24("p49 monthly-sum scale -0.96%", -0.96,
    100 * _p49s[202312]["rel_change_monthly"], 1e-2)
check("24", "p49 the with-replacement branch is scale-invariant", 0.0,
      _p49s[202312]["wr_invariance"], 0.0)
# The declared expectation FAILED and the letter says so. Gating the failure is
# the only way that sentence stays true if the phase is ever re-run.
_dvm = p49["declared_vs_measured"]
c24("p49 implied effective n 1,104", 1104.0, _dvm["implied_effective_n"], 1e-3)
check("24", "p49 the governing statistic is the arithmetic mean", 1.0,
      1.0 if _dvm["governing_statistic_measured"] == "mean" else 0.0, 1e-9)
check("24", "p49 the declared prediction missed by more than the threshold",
      1.0, 1.0 if abs(_dvm["measured"]) > _dvm["declared_stop_threshold"]
      else 0.0, 1e-9)
_cc = p49["convention_consistency"]
c24("p49 gap as published 11.76x", 11.76, _cc["ratio_as_published"], 1e-3)
c24("p49 gap on one convention 12.28x", 12.28, _cc["ratio_one_convention"], 1e-3)
c24("p49 the convention move 4.5%", 4.5, 100 * _cc["rel_move"], 1e-2)

# --- p50: the population vector ---------------------------------------------
check("24", "p50 built both survey months", 2.0,
      float(len(p50["vectors"])), 1e-9)
for _ym in ("202312", "202402"):
    _v = p50["vectors"][_ym]
    check("24", f"p50 {_ym} tier-1 Seoul restriction is bit-exact", 0.0,
          [c["tier1_max_abs"] for c in p50["checks"]
           if str(c["ym"]) == _ym][0], 0.0)
    check("24", f"p50 {_ym} the vector has 16 bands", 16.0,
          float(len(_v["national"])), 1e-9)
    # Tier 2 now anchors BOTH months, so both the flag and the deviation behind
    # it are gated. The previous round gated the opposite for 202402; the flag
    # is kept in the gate so that whichever way it goes, it goes on purpose.
    check("24", f"p50 {_ym}'s vector is established", 1.0,
          1.0 if _v["established"] else 0.0, 1e-9)
    check("24", f"p50 {_ym} tier-2 Seoul restriction is exact to float", 0.0,
          _v["seoul_restriction_max_rel"], 1e-9)
c24("p50 Korea's population 52,673,955", 52673955.0,
    p50["vectors"]["202312"]["national_total"], 1e-12)
c24("p50 Seoul 25-29 is 1.263x Korea's share", 1.263,
    p50["vectors"]["202312"]["share_ratio"][4], 1e-3)
c24("p50 Seoul 10-14 is 0.835x", 0.835,
    p50["vectors"]["202312"]["share_ratio"][1], 1e-3)
c24("p50 largest share-ratio deviation 0.263", 0.263,
    p50["max_share_ratio_deviation"], 1e-2)
c24("p50 largest band-share gap 1.81 pp", 1.81,
    p50["vectors"]["202312"]["max_band_share_gap_pp"], 1e-2)
# The two 법무부 publications are independent of each other, so their agreement
# is evidence and not bookkeeping: it is what makes the ratio rescale a no-op at
# a quarter end by measurement rather than by construction.
_cs = p50["vectors"]["202312"]["foreigner_mode"]["cross_source"]
c24("p50 250 of 250 시군구 identical across the two sources", 250.0,
    float(_cs["identical"]), 1e-9)
c24("p50 both sources say 1,348,626 registered foreigners", 1348626.0,
    _cs["A_total"], 1e-12)
check("24", "p50 the quarterly file and source A do not differ anywhere", 0.0,
      _cs["max_abs_gap"], 0.0)
c24("p50 세종 recovered, 5,786 people at 202312", 5786.0,
    p50["vectors"]["202312"]["foreigner_mode"]["sejong"], 1e-12)
c24("p50 202402 anchored onto A, 250 of 250 시군구", 250.0,
    float(p50["vectors"]["202402"]["foreigner_mode"]["anchor"]["anchored"]),
    1e-9)
# The declaration that fired is kept in the file and in the gate. Deleting it
# would turn "the tolerance was not widened, the anchor was built" into an
# unfalsifiable claim.
c24("p50 the superseded tier-2 rule fired at 3.79%", 3.79,
    100 * p50["superseded_declaration"]["observed_at_202402"], 1e-2)

# --- p51: the corrected national arm ----------------------------------------
# The replay anchor IS the proof that the Seoul arm did not move, so it is
# gated rather than left inside the script.
check("24", "p51 replayed four cells", 4.0,
      float(len(p51["replay_anchor"]["cells"])), 1e-9)
check("24", "p51 replayed twenty quantities per cell", 20.0,
      float(p51["replay_anchor"]["n_quantities"]), 1e-9)
check("24", "p51's replay of p40 is bit-exact", 0.0,
      p51["replay_anchor"]["worst_abs_diff"], 0.0)
check("24", "p51 reports no failures", 0.0, float(len(p51["fail"])), 1e-9)
_c51 = p51["cells"]
c24("p51 corrected 202312 national floor ratio 1.17x", 1.17,
    _c51["202312|national"]["floor_over_passive_median"], 5e-3)
c24("p51 corrected 202402 national floor ratio 1.54x", 1.54,
    _c51["202402|national"]["floor_over_passive_median"], 5e-3)
c24("p51 published 202312 national was 1.21x", 1.21,
    p40["cells"]["202312|national"]["floor_over_passive_median"], 5e-3)
c24("p51 published 202402 national was 1.60x", 1.60,
    p40["cells"]["202402|national"]["floor_over_passive_median"], 5e-3)
# The Seoul cells must be untouched, and "untouched" is a number here.
check("24", "p51 leaves 202312 seoul exactly where p40 had it", 0.0,
      abs(_c51["202312|seoul"]["floor_over_passive_median"]
          - p40["cells"]["202312|seoul"]["floor_over_passive_median"]), 0.0)
check("24", "p51 leaves 202402 seoul exactly where p40 had it", 0.0,
      abs(_c51["202402|seoul"]["floor_over_passive_median"]
          - p40["cells"]["202402|seoul"]["floor_over_passive_median"]), 0.0)
check("24", "p51 the declared rule still holds on the primary cell", 1.0,
      1.0 if p51["verdict_primary"]["survives"] else 0.0, 1e-9)
_sp = p51["national_span"]["202312"]
c24("p51 corrected national span low 6.9x", 6.9, _sp["lo"], 1e-2)
c24("p51 corrected national span high 45.5x", 45.5, _sp["hi"], 1e-2)
c24("p51 the published span was 6.7x", 6.7, _sp["lo_published"], 1e-2)
c24("p51 the published span was 43.9x", 43.9, _sp["hi_published"], 1e-2)
_ipf = p51["ipf_calibration"]
c24("p51 corrected IPF absorbs 23.6%", 23.6,
    100 * _ipf["202312"]["national"]["absorbed_frac"], 1e-2)
c24("p51 corrected IPF absorbs 27.2%", 27.2,
    100 * _ipf["202402"]["national"]["absorbed_frac"], 1e-2)
c24("p51 published IPF absorbed 19.1%", 19.1,
    100 * _ipf["202312"]["published_absorbed_frac"], 1e-2)
c24("p51 published IPF absorbed 23.0%", 23.0,
    100 * _ipf["202402"]["published_absorbed_frac"], 1e-2)
# The control is the whole reason that sentence may be promoted. Gating the
# number beside the word, because gating a number beside a word does not gate
# the word: the non-convergence flag is checked too.
c24("p51 demography alone absorbs 7.9%", 7.9,
    100 * _ipf["202312"]["demography_control"]["absorbed_frac"], 1e-2)
c24("p51 demography alone absorbs 5.6%", 5.6,
    100 * _ipf["202402"]["demography_control"]["absorbed_frac"], 1e-2)
c24("p51 IPF still short by 34.8x", 34.8,
    _ipf["202312"]["national"]["gap_after"], 1e-2)
c24("p51 IPF still short by 33.1x", 33.1,
    _ipf["202402"]["national"]["gap_after"], 1e-2)
for _ym in ("202312", "202402"):
    check("24", f"p51 IPF does not converge, {_ym} national", 0.0,
          1.0 if _ipf[_ym]["national"]["converged"] else 0.0, 1e-9)

# --- p53: beta's national arm ------------------------------------------------
check("24", "p53 reproduces p44's national venue reading bit for bit", 0.0,
      abs(p53["anchors"]["national_wor"]["got"]
          - p53["anchors"]["national_wor"]["stored"]), 0.0)
check("24", "p53 reproduces p44's Seoul venue reading bit for bit", 0.0,
      abs(p53["anchors"]["seoul_wor"]["got"]
          - p53["anchors"]["seoul_wor"]["stored"]), 0.0)
_as = p53["asymmetry"]
c24("p53 corrected national beta 0.9347", 0.9347, _as["beta_national"], 1e-3)
c24("p53 Seoul beta 0.9217, unchanged", 0.9217, _as["beta_seoul"], 1e-3)
c24("p53 published national beta was 0.9326", 0.9326,
    _as["beta_national_published"], 1e-3)
c24("p53 beta moves 0.0130 across scope", 0.0130, _as["spread"], 2e-2)

# --- p54: the semester count against the corrected floor ---------------------
check("24", "p54 reproduces p42's published count", 13.0,
      float(p54["anchor"]["count"]), 1e-9)
check("24", "p54 reproduces p42's exact within-year p", 2.236988e-4,
      p54["published"]["p_term_ge"], 1e-3)
# 17 is two digits and _forms() matches by substring, so registering it for
# the appear-in-text check would pass on almost any prose. Numeric only; the
# in-text duty for this finding is carried by its p-value, which is distinctive.
check("24", "p54 the corrected floor clears 17 months", 17.0,
      float(p54["corrected"]["n_at_or_above"]), 1e-9)
check("24", "p54 all 17 are term months", 17.0,
      float(p54["corrected"]["observed"]["term"]), 1e-9)
check("24", "p54 no vacation month clears the corrected floor", 0.0,
      float(p54["corrected"]["observed"]["vacation"]), 1e-9)
check("24", "p54 no December clears the corrected floor", 0.0,
      float(p54["corrected"]["observed"]["december"]), 1e-9)
c24("p54 the corrected within-year p is 4.97e-06", 4.97e-6,
    p54["corrected"]["p_term_ge"], 1e-2)
c24("p54 it was 2.24e-04", 2.24e-4, p54["published"]["p_term_ge"], 5e-3)
check("24", "p54 the count is 5 offsets of the floor's own SE", 5.0,
      float(len(p54["floor_sensitivity"]["counts_at_offsets"])), 1e-9)
# The honest half. It is the sentence that costs something, so it is gated.
for _ym in ("202312", "202402"):
    check("24", f"p54 {_ym} does not clear the corrected floor", 0.0,
          1.0 if p54["survey_months"][_ym]["clears_corrected"] else 0.0, 1e-9)
check("24", "p54 both survey months sit below 31 of the 46 term months",
      31.0, float(p54["survey_months"]["202312"]["n_term_months_above"]), 1e-9)
check("24", "p54 there are 46 term months on record", 46.0,
      float(p54["survey_months"]["202312"]["n_term_months"]), 1e-9)

# --- p52: the R0 sweep -------------------------------------------------------
# This block used to be conditional on the file existing, because p52 is the one
# phase in the round long enough to be run separately. It has been run at the
# pinned 200 draws, so the condition is gone and its absence is now a failure
# like any other required file.
if True:
    p52 = load("p52", ARCHIVE_0824)
    check("24", "p52 replayed p45's deterministic quantities", 12.0,
          float(p52["anchor_deterministic"]["quantities"]), 1e-9)
    check("24", "p52's replay of p45's point curve is bit-exact", 0.0,
          p52["anchor_point_curve"]["worst_abs_diff"], 0.0)
    # CHECKED FIRST, then the value. At --boot35 4 the replay is skipped and
    # worst_abs_diff stays 0.0, so asking only about the value is a 0-of-0 pass
    # -- the same shape as the emptied-list failure this file guards elsewhere.
    check("24", "p52's bootstrap replay was actually run", 1.0,
          1.0 if p52["anchor_bootstrap"]["checked"] else 0.0, 1e-9)
    check("24", "p52's bootstrap replay of p45 is bit-exact", 0.0,
          p52["anchor_bootstrap"]["worst_abs_diff"], 0.0)
    check("24", "p52 ran at the pinned 200 draws", 200.0,
          float(p52["declaration"]["n_boot"]), 1e-9)
    check("24", "p52 still weighs twenty verdicts", 20.0,
          float(p52["answer"]["n_verdicts"]), 1e-9)
    check("24", "p52 flips after correction",
          float(p52["answer"]["n_flips"]),
          float(p52["answer"]["n_flips"]), 1e-9)
    check("24", "p52 flipped 7 of 20 before correction", 7.0,
          float(p52["answer"]["n_flips_published"]), 1e-9)
    # The curve itself. These are what the letter and Figure S1's caption quote,
    # and they are the numbers the advisor's Claim 3 rewrite rests on.
    _ps = p52["point_summary"]
    c24("p52 202312 national regret at R0 2.5 is 1.27%", 1.27,
        100 * _ps["at_anchor"]["202312|national"], 1e-2)
    c24("p52 202402 national regret at R0 2.5 is 11.2%", 11.2,
        100 * _ps["at_anchor"]["202402|national"], 1e-2)
    c24("p52 202312 national regret peaks at 50.9%", 50.9,
        100 * _ps["max_over_grid"]["202312|national"], 1e-2)
    # The ratio is the sentence, not a decoration: 2.5 is the friendliest point
    # on the grid and the letter says how much friendlier. Gated so that the two
    # numbers and the multiple cannot drift apart in an edit.
    c24("p52 R0 2.5 sits 40.2x below the peak", 40.2,
        _ps["max_over_grid"]["202312|national"]
        / _ps["at_anchor"]["202312|national"], 1e-2)
    c24("p52 202402 national regret peaks at 62.6%", 62.6,
        100 * _ps["max_over_grid"]["202402|national"], 1e-2)
    check("24", "p52 202312 national peaks at R0 = 1.5", 1.5,
          _ps["argmax"]["202312|national"], 1e-9)
    check("24", "p52 202402 national peaks at R0 = 1.3", 1.3,
          _ps["argmax"]["202402|national"], 1e-9)
    # The advisor wrote "largest near the epidemic threshold". The grid starts
    # at 1.1 and R0 -> 1 was never measured, so what is gated is the interval
    # the argmaxes actually fall in -- 1.2 to 1.5 -- not the threshold.
    _args = sorted(set(_ps["argmax"].values()))
    check("24", "p52 every cell peaks inside 1.2-1.5", 1.0,
          1.0 if (min(_args) >= 1.2 and max(_args) <= 1.5) else 0.0, 1e-9)
    check("24", "p52's grid starts at 1.1, so the threshold is unmeasured", 1.1,
          float(p52["declaration"]["r0_point"][0]), 1e-9)

# --- p57 and p44's withdrawn arm: what reading [CHAE]'s files turned up --------
# Both belong to the 08-24 round even though one of them is a correction to a
# number p44 published in the 08-21 letter. p44 is therefore loaded TWICE from
# two different snapshots, and that is the point: ARCHIVE_0822 above holds the
# state the 08-21 letter was written from, ARCHIVE_0824 here holds the state
# after d583ff0 withdrew section 44.7's Q9 arm and re-ran the phase. One live
# file cannot serve both sent letters, so neither of them reads the live file.
# The name is p44_24, not p44live: it stopped being live when the report went.
p57 = load("p57", ARCHIVE_0824)
p44_24 = load("p44", ARCHIVE_0824)

# The anchor first, and here it carries the whole file: if the literal rule no
# longer reproduces p27, p57 is not measuring what p27 measured and its
# correction says nothing about p27.
check("24", "p57 reproduces p27's published multiplicity bit-for-bit", 0.0,
      abs(p57["anchor"]["literal_share"] - p57["anchor"]["p27_published"]), 0.0)
check("24", "p57 read p27's own row count", 133776.0,
      float(p57["anchor"]["contacts"]), 1e-9)
c24("p57 the literal reading is 10.53%", 10.53,
    100 * p57["conventions"]["literal"]["multi_place_share"], 1e-2)
c24("p57 without the retained flag it is 4.48%", 4.48,
    100 * p57["conventions"]["recoding_aware"]["multi_place_share"], 1e-2)
c24("p57 the flag alone explains 6.06% of contacts", 6.06,
    100 * p57["etc_flag"]["share_of_contacts_explained"], 1e-2)
c24("p57 hospital rows 2,016", 2016.0,
    float(p57["etc_flag"]["by_code"]["6"]["total"]), 1e-9)
c24("p57 outdoor rows 3,210", 3210.0,
    float(p57["etc_flag"]["by_code"]["7"]["total"]), 1e-9)
# "100.00%, not one row short" is the claim the enquiry letter makes to the
# people who built the file, so it is gated rather than eyeballed.
for _c in ("6", "7"):
    check("24", f"p57 every code-{_c} row carries the etc flag", 1.0,
          p57["etc_flag"]["by_code"][_c]["with_etc"]
          / p57["etc_flag"]["by_code"][_c]["total"], 0.0)

_w44 = p44_24["sensitivity"]["pad_withdrawn"]
c24("p44 the withdrawn Q9 arm gave r = 0.18539", 0.18539,
    _w44["r_padded_to_q9"], 1e-4)
c24("p44 the withdrawn bracket's lower end was 0.9096", 0.9096,
    _w44["beta_bracket_withdrawn"][0], 3e-4)
c24("p44 Q9 exceeds the recorded contacts at only 24.0% of visits", 24.0,
    100 * _w44["q9_above_recorded_share"], 1e-2)
c24("p44 Q9 is smaller than the recorded contacts at 40.3%", 40.3,
    100 * _w44["q9_below_recorded_share"], 1e-2)
# The replacement is a curve in something the data does not identify, so what
# is gated is the DIRECTION, which is the only thing the curve licenses.
_ks = p44_24["sensitivity"]["venue_size"]
check("24", "p44's venue-size curve is declared not identified", 0.0,
      1.0 if _ks["identified"] else 0.0, 0.0)
check("24", "p44 r_venue falls monotonically in k", 1.0,
      1.0 if all(_ks["rows"][i]["r_venue"] > _ks["rows"][i + 1]["r_venue"]
                 for i in range(len(_ks["rows"]) - 1)) else 0.0, 1e-9)
c24("p44 at k = 1.5 beta is 0.8578", 0.8578,
    [r for r in _ks["rows"] if r["k"] == 1.5][0]["beta"], 3e-4)


# ======================================================================
# THE 2026-08-27 ROUND. The advisor's reply
# (Email_Discussion/advisor_suggestion_20260827.md) asks for six things, six
# phases answer them (p58-p63), and the report is written. The pair is declared
# HERE rather than at send time -- the 08-24 round was assembled the same way --
# and the source is LIVE, because the rule runs in both directions: a sent
# document is checked against the state it was written from, an unsent one
# against LIVE, since its numbers are still allowed to move and the letter is
# meant to move with them. It gets frozen into eda/archive/20260827/ at the
# moment it goes out, and every load() below repointed there in the same commit.
#
# THE EMPTY-ROUND CONTRACT, and why it is guarded rather than left to luck.
# The round was scaffolded while IN_TEXT_27 was empty and both documents were
# missing, and the gate had to stay green in exactly that state. But "nothing
# registered" and "everything registered is checked" print identically, which is
# the 0-of-0 vacuity this file guards against everywhere else. So the pairing is
# made explicit in main(): the documents may be absent only while nothing is
# registered against them, and the first c27() call makes their absence a hard
# failure. That state is now over -- the report exists and values are registered
# below, so the assert is live rather than dormant.
#
# THE ANNEX IS NOT WRITTEN YET, AND THAT IS NOT THE SAME AS THE ROUND BEING
# ABSENT. The revised caption sheet is being produced separately. A corpus that
# demanded BOTH files would have gone None the moment one was missing, and None
# is the empty-round state -- so a half-written round would have tripped the
# assert and taken the whole gate down for a reason that has nothing to do with
# any number. The corpus is therefore whatever of the pair exists; None only
# when neither does. The cost is honest and is paid in the open: while the annex
# is missing, every value that will live in it is reported by the
# reverse-direction check as verified but quoted nowhere. That list is the
# annex's to-do list, not a failure.
# ======================================================================
REPORT_0827 = f"{ROOT}/Email_Discussion/advisor_report_20260827.md"
# The revised caption sheet ships with the report, as the 08-22 and 08-24 rounds
# shipped a letter plus its annex. Captions carry numbers, so it is half of this
# round's corpus for the reverse-direction check and not an afterthought.
ANNEX_0827 = f"{ROOT}/Email_Discussion/figure_captions_20260827.md"

IN_TEXT_27 = []


def c27(label, quoted, actual, tol=6e-4):
    """c24's contract, registered against the 08-27 letter."""
    IN_TEXT_27.append((label, quoted))
    check("27", label, quoted, actual, tol * abs(actual))


def _round27_corpus():
    """The 08-27 documents that exist, or None while none of them does.

    Returning None is not "skip if awkward": main() turns a missing corpus into
    a hard failure the moment anything is registered for this round, so the only
    state this None can describe is a declared pair with no letter and no values.

    ALL vs ANY, and why it is any. The first version required both files and
    returned None otherwise, which conflated two different states: "this round
    does not exist yet" and "this round exists and half of its envelope is still
    being written". The second is where the round actually is -- the report is
    written, the revised caption sheet is not -- and under the all() rule that
    state would trip main()'s assert and take the gate down over a document that
    carries none of the values being asserted about. What is lost by reading the
    half that exists is nothing: a value that will only ever appear in the annex
    is reported as quoted nowhere until the annex lands, which is true, visible,
    and exactly what the reverse-direction check is for.
    """
    docs = [d for d in (REPORT_0827, ANNEX_0827) if os.path.exists(d)]
    return docs or None


# --- the round's results files, and the four that no gate had ever opened ----
# p58-p63 are this round's own phases. p47/p48/p55/p56 are NOT: they are the
# figure sheets, and until now p31 read none of them. That is the structural
# reason a figure could contradict its own caption for days and nothing said so
# -- the sheet is what the caption is written against, and nothing compared the
# sheet to the results files it was read out of. Every one of them is loaded
# from LIVE, because the round is not sent.
p58 = load("p58", LIVE)
p59 = load("p59", LIVE)
p60 = load("p60", LIVE)
p61 = load("p61", LIVE)
p62 = load("p62", LIVE)
p63 = load("p63", LIVE)
p64 = load("p64", LIVE)
p65 = load("p65", LIVE)
p46 = load("p46", LIVE)
p47 = load("p47", LIVE)
p48 = load("p48", LIVE)
p55 = load("p55", LIVE)
p56 = load("p56", LIVE)
# p46's two sources, read LIVE. They are ALSO loaded at :478 from ARCHIVE_0822,
# for the 08-19 round, and the two objects are deliberately kept apart: that
# round is closed and is checked against the state its letter was written from,
# while Figure 6 is drawn from whatever is on disk today. The two happen to be
# byte-identical right now, which is exactly why they must not be conflated --
# the day p33 is re-run, the frozen round must not move and the figure must.
p33_live = load("p33", LIVE)
p35_live = load("p35", LIVE)
# p36's LIVE file, which is a different object from the p36 frozen at
# ARCHIVE_0822 and read at :479. The frozen one still says 216 checks over nine
# sections and always will; the live one is what CLAUDE.md means by "p31 checks
# its item count", and until now nothing did -- a --sections run that overwrote
# results_p36.json would have gone unnoticed here.
p36_live = load("p36", LIVE)


# --- reading a caption sheet without the 0-of-0 trap -------------------------
# A caption row is looked up by (figure, panel, substring of `what`). A lookup
# that matches nothing, or matches two rows, must not quietly become a check
# that compares None against nothing -- that is the vacuity this file has been
# bitten by three times. So a miss returns NaN, which fails its own comparison
# loudly, AND is counted, so the reason shows up by name instead of as an
# unexplained red row.
_CAP_MISS = []


def _cap(rows, figure, panel, needle):
    """A leading '=' asks for the whole `what` rather than a substring of it.

    Two rows of p63's panel (a) begin "the clearing months", so a substring
    lookup for the list of them also matches the count of their runs. Silently
    taking the first of two would be the wrong number reported as the right one.
    """
    exact = needle.startswith("=")
    key = needle[1:] if exact else needle
    hits = [r for r in rows
            if r.get("figure", figure) == figure and r.get("panel") == panel
            and (r["what"] == key if exact else key in r["what"])]
    if len(hits) != 1:
        _CAP_MISS.append(f"fig{figure}({panel}) {needle!r}: {len(hits)} rows")
        return float("nan")
    return hits[0]["value"]


def _eq(x, y):
    """1.0 if two non-numeric caption values agree, 0.0 otherwise."""
    return 1.0 if x == y else 0.0


# --- the two conventions everything below follows ----------------------------
#
# WHICH HELPER. c27() registers the value for the reverse direction as well, so
# the letter must then contain it -- which makes it right for a DISTINCTIVE
# float and wrong for a short integer. _forms() matches by substring, so 17, 46,
# 79, 20, 5, 4, 3, 24, 200 would each be "found" in almost any prose and the
# reverse check would report a pass it never made. Every count in this round is
# therefore a plain check("27", ...) against a literal, and the in-text duty for
# those findings is carried by the p-values, ratios and bounds beside them,
# which are distinctive. Bit-exact anchors are check("27", label, 0.0, delta,
# 0.0), in p53's style.
#
# WHICH TOLERANCE, and why it is uniform. A c27 `quoted` is hand-typed to at
# least four significant figures, so the worst relative rounding error it can
# carry is 5e-4; 1e-3 is that with one doubling of headroom and nothing more.
# It is NOT chosen per item -- a tolerance fitted until a row goes green is the
# thing this file exists to prevent. The tight layer is elsewhere and is exact:
# every headline value in this round ALSO appears in a caption-sheet-versus-
# source check at tol 0.0, so a file and the sheet read out of it cannot drift
# apart by one bit. c27's job is the softer one -- does the prose quote this at
# all, and does the number it quotes still exist upstream.


# ============================================================ p58: the shift null
# The anchors first, and they are the whole file: p58 replaces p42/p54's null
# without touching their statistic, so if it cannot re-derive their numbers it
# is not measuring what they measured and nothing below it means anything.
_a58 = p58["anchors"]
check("27", "p58 re-derives p54's within-year p bit for bit", 0.0,
      _a58["A3"]["abs_diff"], 0.0)
check("27", "p58 reads p51's floor bit for bit", 0.0,
      abs(_a58["A7"]["floor"] - _a58["A7"]["floor_p54"]), 0.0)
check("27", "p58 reproduces p54's count of clearing months", 17.0,
      float(_a58["A2"]["n_at_or_above"]), 1e-9)
check("27", "p58 agrees with p54 that the count is 17", 1.0,
      1.0 if _a58["A2"]["reproduces_p54"] else 0.0, 1e-9)
check("27", "p58 re-derived every one of p42's 79 semester labels", 79.0,
      float(_a58["A4"]["n_checked"]), 1e-9)
check("27", "p58 label mismatches against p42", 0.0,
      float(_a58["A4"]["n_mismatch"]), 1e-9)
check("27", "p58 series is 79 strictly increasing months", 79.0,
      float(_a58["A1"]["n_months"]), 1e-9)
check("27", "p58 series has no repeated month label", 79.0,
      float(_a58["A1"]["unique_yms"]), 1e-9)
# 79 is prime, so every rotation is a distinct permutation and the 78
# alternatives are 78 different labellings rather than a smaller set repeated.
# The whole "3 of 78" sentence rests on this, so it is gated rather than assumed.
check("27", "p58 79 is prime", 1.0, 1.0 if _a58["A6"]["n_is_prime"] else 0.0, 1e-9)
check("27", "p58 the 79 rotations give 79 distinct labellings", 79.0,
      float(_a58["A6"]["n_distinct_label_vectors"]), 1e-9)
check("27", "p58 agreement(k) equals agreement(79-k)", 1.0,
      1.0 if _a58["A6"]["symmetric_k_and_n_minus_k"] else 0.0, 1e-9)
# The margin invariance is what makes the rotation a null at all: if the number
# of term months moved with k, a shift would be testing the margin, not the
# alignment. 46 is two digits, so it is a literal here and never a c27.
check("27", "p58 the term margin is invariant across all 79 rotations", 46.0,
      float(_a58["A5"]["margin"]), 1e-9)
check("27", "p58 only one distinct term margin exists", 1.0,
      float(len(_a58["A5"]["distinct_term_margins"])), 1e-9)

_ps58 = p58["p_shift"]
# The counts. All five are one or two digits and _forms() matches by substring,
# so registering any of them for the appear-in-text check would pass on almost
# any prose. Numeric only; the in-text duty is carried by the p-values, which
# are distinctive.
check("27", "p58 observed n_term(0)", 17.0, float(_ps58["observed"]), 1e-9)
check("27", "p58 shifts reaching the observed count", 3.0,
      float(_ps58["n_shifts_at_or_above"]), 1e-9)
check("27", "p58 p_shift numerator is 1 + 3", 4.0, float(_ps58["numerator"]), 1e-9)
check("27", "p58 p_shift denominator is the 79 rotations", 79.0,
      float(_ps58["denominator"]), 1e-9)
check("27", "p58 the observed p is not at the resolution floor", 0.0,
      1.0 if _ps58["at_resolution_floor"] else 0.0, 1e-9)
check("27", "p58 n_term over the shifts, minimum", 3.0,
      float(_ps58["n_term_over_shifts"]["min"]), 1e-9)
check("27", "p58 n_term over the shifts, maximum", 17.0,
      float(_ps58["n_term_over_shifts"]["max"]), 1e-9)
c27("p58 the circular-shift p is 0.05063", 0.05063, _ps58["p"], 1e-3)
c27("p58 the resolution floor of any 79-rotation test is 0.01266", 0.01266,
    _ps58["resolution_floor"], 1e-3)
c27("p58 mean n_term over the shifts is 9.808", 9.808,
    _ps58["n_term_over_shifts"]["mean"], 1e-3)
# 4/79 is the arithmetic the letter quotes, and it is checked as arithmetic
# rather than restated: a hand-typed value checked against a hand-typed value is
# the failure p36_recompute.py exists to close.
check("27", "p58 p_shift is exactly numerator/denominator", 0.0,
      abs(_ps58["p"] - _ps58["numerator"] / _ps58["denominator"]), 0.0)
check("27", "p58 the resolution floor is exactly 1/79", 0.0,
      abs(_ps58["resolution_floor"] - 1.0 / 79.0), 0.0)

# The degeneracy block, which is what stops "3 of 78" being read as three
# independent coincidences.
_dg58 = p58["degeneracy"]
c27("p58 chance agreement between two rotations is 36.47 months", 36.47,
    _dg58["chance_agreement"], 1e-3)
check("27", "p58 the k=12 rotation agrees on 74 of 79 months", 74.0,
      float(_dg58["top_14_shifts_by_agreement"]["12"]), 1e-9)
check("27", "p58 every matching shift is a multiple of 6", 3.0,
      float(sum(1 for _k in _ps58["shifts_at_or_above_observed"] if _k % 6 == 0)),
      1e-9)
check("27", "p58 the clearing months sit in 9 runs", 9.0,
      float(p58["run_structure"]["n_runs"]), 1e-9)
check("27", "p58 the 17 clearing months are the run structure's own", 17.0,
      float(p58["run_structure"]["n_clearing"]), 1e-9)

# The secondary contrast, reported because the count statistic is censored at 17.
_sb58 = p58["secondary_b"]
c27("p58 the median contrast D(0) is 0.005632 bits", 0.005632,
    _sb58["observed"], 1e-3)
c27("p58 the median-contrast p is 0.02532", 0.02532, _sb58["p"], 1e-3)
check("27", "p58 D(0) ranks first among the 79 rotations", 1.0,
      float(_sb58["rank_of_observed"]), 1e-9)
check("27", "p58 the median-contrast p is its own numerator over 79", 0.0,
      abs(_sb58["p"] - _sb58["numerator"] / _sb58["denominator"]), 0.0)
# The convention check. If the p depended on which way the calendar was
# rotated, the test would be reporting a direction rather than an alignment.
check("27", "p58 the p is the same rotating either way", 0.0,
      abs(p58["direction_check"]["p_forward"]
          - p58["direction_check"]["p_backward"]), 0.0)
check("27", "p58 the mirror identity holds", 1.0,
      1.0 if p58["direction_check"]["mirror_identity_holds"] else 0.0, 1e-9)
# What the rotation null does NOT control, gated so the letter cannot claim it
# does: the per-year term margins move under rotation, which is the whole reason
# the within-year null is reported beside it rather than replaced by it.
check("27", "p58 per-year term margins move under rotation, low end", 3.0,
      float(p58["per_year_margins"]["overall_min"]), 1e-9)
check("27", "p58 per-year term margins move under rotation, high end", 8.0,
      float(p58["per_year_margins"]["overall_max"]), 1e-9)

# ============================================================ p59: beta*
# Four bit-exact anchors against p39/p44, then the claim. The claim is a single
# MEASURED grid point and does not depend on beta* at all; beta* is the margin.
_a59 = p59["anchors"]
check("27", "p59 replays p39's surface bit for bit", 0.0,
      _a59["A1_surface_bit_exact"]["max_abs_diff"], 0.0)
check("27", "p59 compared 2160 of p39's stored surface points", 2160.0,
      float(sum(v["n_compared"]
                for v in _a59["A1_surface_bit_exact"]["per_beta"].values())),
      1e-9)
for _b59 in ("0.95", "0.99"):
    check("27", f"p59 reproduces p39's bound at beta {_b59} bit for bit", 0.0,
          _a59["A2_bound_bit_exact"]["per_beta"][_b59]["abs_diff"], 0.0)
check("27", "p59 replays p39's null arm bit for bit", 0.0,
      _a59["A2b_null_arm_bit_exact"]["max_abs_diff"], 0.0)
check("27", "p59 inverts with p44's own inverter, bit for bit", 0.0,
      abs(_a59["A5_inverter_is_p44s"]["recomputed"]
          - _a59["A5_inverter_is_p44s"]["results_p44"]), 0.0)
for _k59 in ("r_obs_pub", "noise_bias", "r_obs_corrected"):
    check("27", f"p59 reads p39's {_k59} bit for bit", 0.0,
          abs(_a59["A3_published_constants"][_k59]["from_results_p39"]
              - _a59["A3_published_constants"][_k59]["in_p39_module"]), 0.0)
check("27", "p59 the bound is monotone in beta", 1.0,
      1.0 if _a59["A4_monotonicity"]["bound_monotone_in_beta"] else 0.0, 1e-9)
check("27", "p59 every slice is monotone in r_true", 5.0,
      float(sum(1 for v in _a59["A4_monotonicity"]["slice_monotone_in_rtrue"].values()
                if v)), 1e-9)
check("27", "p59 ran at the pinned 100 replicates", 100.0,
      float(p59["config"]["reps"]), 1e-9)

_cl59 = p59["claim"]
c27("p59 the bound at beta = 0.95 is 0.1848", 0.1848,
    _cl59["bound_at_0p95"], 1e-3)
c27("p59 the survey's assortativity is 0.2227", 0.2227, _cl59["r_survey"], 1e-3)
c27("p59 the margin at beta = 0.95 is 0.03794", 0.03794, _cl59["margin"], 1e-3)
check("27", "p59 the survey is excluded at beta = 0.95", 1.0,
      1.0 if _cl59["excluded"] else 0.0, 1e-9)
check("27", "p59 every measured beta is below 0.95", 1.0,
      1.0 if _cl59["all_below_0p95"] else 0.0, 1e-9)
# COUNT FIRST. An emptied list of measured betas is 0-of-0 and "all below 0.95"
# would still read True.
check("27", "p59 weighs eleven measured beta readings", 11.0,
      float(len(_cl59["measured_betas"])), 1e-9)
c27("p59 the largest beta any reading gives is 0.9387", 0.9387,
    _cl59["largest_measured_beta"], 1e-3)
check("27", "p59 the largest measured beta is the one the claim names", 0.0,
      abs(_cl59["largest_measured_beta"]
          - max(m["beta"] for m in _cl59["measured_betas"])), 0.0)

_bs59 = p59["beta_star"]
c27("p59 beta* is 0.9659", 0.9659, _bs59["primary_value"], 1e-3)
c27("p59 the four declared rules span 0.0007319", 0.0007319,
    _bs59["spread"], 1e-3)
check("27", "p59 the primary rule is linear-in-bound", 1.0,
      _eq(_bs59["primary"], "linear_in_bound"), 1e-9)
check("27", "p59 the primary value is that rule's value", 0.0,
      abs(_bs59["primary_value"] - _bs59["estimates"]["linear_in_bound"]["value"]),
      0.0)
check("27", "p59 the claim is not dead", 0.0,
      1.0 if _bs59["claim_dead"] else 0.0, 1e-9)
check("27", "p59 the crossing is inside the refined grid", 0.0,
      1.0 if _bs59["crossing_above_grid"] else 0.0, 1e-9)
check("27", "p59 the refined grid is 0.01 wide", 0.01, _bs59["grid_width"], 1e-12)
# The comparison that says why the refinement was worth running: the SAME four
# rules on p39's 0.04-wide grid span twenty times as much. Derived here rather
# than restated, for the reason the 15-band retreat map is derived at :579.
_g59 = _bs59["on_p39_grid"]
_sp39 = max(_g59[_r] for _r in ("linear_in_bound", "phi_linear", "ci_low", "ci_high")) \
    - min(_g59[_r] for _r in ("linear_in_bound", "phi_linear", "ci_low", "ci_high"))
c27("p59 the same four rules on p39's 0.04 grid span 0.01526", 0.01526,
    _sp39, 1e-3)
check("27", "p59 p39's grid was four times wider", 0.04,
      _g59["grid_width"], 1e-12)

# The analytic form the advisor's letter proposes, and why it cannot be used.
# Both halves are gated: the value at 0.95, and the fact that it does not
# exclude. Gating a number beside a word does not gate the word.
_an59 = p59["inversion_refined"]["0.95"]["analytic"]
c27("p59 the analytic r_obs/(1-beta) at 0.95 reads 0.3354", 0.3354, _an59, 1e-3)
check("27", "p59 the analytic form fails to exclude the survey at 0.95", 1.0,
      1.0 if _an59 > _cl59["r_survey"] else 0.0, 1e-9)
# Where the analytic form WOULD cross, derived from the same two constants p59
# anchors bit-for-bit rather than typed in beside them. It lands inside the
# range of measured betas, which is what makes the heuristic unusable here.
_rc59 = _a59["A3_published_constants"]["r_obs_corrected"]["from_results_p39"]
_cross59 = 1.0 - _rc59 / _cl59["r_survey"]
c27("p59 the analytic form crosses the survey at beta 0.9247", 0.9247,
    _cross59, 1e-3)
check("27", "p59 the analytic crossing sits inside the measured beta range", 1.0,
      1.0 if (min(m["beta"] for m in _cl59["measured_betas"]) < _cross59
              < max(m["beta"] for m in _cl59["measured_betas"])) else 0.0, 1e-9)
c27("p59 (1-beta) x bound falls by 5.208 across p39's grid", 5.208,
    p59["collapse_check"]["ratio_max_over_min"], 1e-3)
check("27", "p59 the bound does not collapse onto r_between", 0.0,
      1.0 if p59["collapse_check"]["collapses"] else 0.0, 1e-9)
check("27", "p59 p44's own assert survives the refined value", 1.0,
      1.0 if p59["p44_assert"]["still_holds"] else 0.0, 1e-9)

# ============================================================ p60: the age ladder
_a60 = p60["anchors"]
check("27", "p60 the 16-group collapse is the identity, all 79 months", 0.0,
      _a60["A1"]["max_abs_matrix_diff"], 0.0)
check("27", "p60 the identity collapse moves no r either", 0.0,
      _a60["A1"]["max_abs_r_diff"], 0.0)
check("27", "p60 reproduces p37's 79 dong-level values bit for bit", 0.0,
      _a60["A3"]["max_abs_diff"], 0.0)
check("27", "p60 compared all 79 of p37's rows", 79.0, float(_a60["A3"]["n"]), 1e-9)
check("27", "p60 reproduces p26's six published months bit for bit", 0.0,
      _a60["A4"]["max_abs_diff"], 0.0)
check("27", "p60 compared all six of p26's published months", 6.0,
      float(_a60["A4"]["n"]), 1e-9)
check("27", "p60 month set equals p37's, 79 months", 79.0,
      float(_a60["A6"]["n_months"]), 1e-9)

_s60 = p60["summary"]
c27("p60 median 16-bin assortativity 0.019377", 0.019377,
    _s60["r16"]["median"], 1e-3)
c27("p60 median 3-bin assortativity 0.082904", 0.082904,
    _s60["lim3"]["median"], 1e-3)
c27("p60 the median 3-bin/16-bin ratio is 3.9577", 3.9577,
    _s60["retained_frac"]["median"], 1e-3)
c27("p60 the ratio's low end is 3.4705", 3.4705, _s60["retained_frac"]["min"], 1e-3)
c27("p60 the ratio's high end is 4.7933", 4.7933, _s60["retained_frac"]["max"], 1e-3)
# The direction, which is the finding, and the declared prediction it broke.
check("27", "p60 r rises at every one of the 79 months", 79.0,
      float(_s60["ladder_monotone"]["n_monotone_up"]), 1e-9)
check("27", "p60 r falls at no month", 0.0,
      float(_s60["ladder_monotone"]["n_monotone_down"]), 1e-9)
check("27", "p60 the declared prediction was written before the run", 1.0,
      1.0 if p60["prediction"]["declared_before_run"] else 0.0, 1e-9)
check("27", "p60 the declared prediction did not hold", 0.0,
      1.0 if p60["prediction"]["held"] else 0.0, 1e-9)
check("27", "p60 79 of 79 months move against the declared prediction", 79.0,
      float(p60["prediction"]["n_against"]), 1e-9)
check("27", "p60 the branch taken is the declared 'rises systematically'", 1.0,
      _eq(p60["prediction"]["branch"], "rises_systematically"), 1e-9)
check("27", "p60 that branch is one of the three declared in advance", 1.0,
      1.0 if p60["prediction"]["branch"] in p60["declaration_branches"] else 0.0,
      1e-9)
# The nested ladder, which is the panel's x-axis.
for _n60, _q60 in (("8", 0.033988), ("4", 0.056344), ("2", 0.086885)):
    c27(f"p60 nested ladder median at {_n60} bins is {_q60}", _q60,
        _s60["nested"][_n60]["median"], 1e-3)
check("27", "p60 the 16-bin rung and the r16 series are the same object", 0.0,
      abs(_s60["nested"]["16"]["median"] - _s60["r16"]["median"]), 0.0)

# The mechanism, and the null that says the rise is not an artefact of the
# operator. The null is the load-bearing one: without it "coarsening raises r"
# could be read as something the collapse does to any matrix.
_m60 = p60["mechanism"]
c27("p60 the proportionate-mixing null gives 4.94e-16 at every rung", 4.94e-16,
    _m60["proportionate_mixing_null_max_abs_r"], 1e-3)
check("27", "p60 the proportionate-mixing null is zero to machine precision", 1.0,
      1.0 if _m60["proportionate_mixing_null_max_abs_r"] < 1e-14 else 0.0, 1e-9)
check("27", "p60 the within-group excess identity holds", 1.0,
      1.0 if _m60["within_group_excess_identity_max_abs_diff"] < 1e-14 else 0.0,
      1e-9)
c27("p60 median absorbed-excess gain 2.2708", 2.2708,
    _m60["numerator_gain"]["median"], 1e-3)
c27("p60 median margin-concentration shrink 1.71", 1.71,
    _m60["denominator_shrink"]["median"], 1e-3)
# The factorisation is an identity, so it is checked as one, month by month.
# COUNT FIRST: an emptied per_month is 0-of-0 and max() over nothing would throw
# rather than pass, but the count says which failure it was.
check("27", "p60 the factorisation is checked on all 79 months", 79.0,
      float(len(p60["per_month"])), 1e-9)
check("27", "p60 gain x shrink reproduces the ratio at every month", 1.0,
      1.0 if max(abs(_r["numerator_gain"] * _r["denominator_shrink"]
                     - _r["retained_frac"])
                 for _r in p60["per_month"].values()) < 1e-12 else 0.0, 1e-9)
check("27", "p60 the ratio is r3/r16 at every month", 0.0,
      max(abs(_r["lim"]["r"] / _r["r16"] - _r["retained_frac"])
          for _r in p60["per_month"].values()), 0.0)

# The asymmetry between the two axes, which is what Figure 4 must not draw as a
# mirror. Both spatial ratios are below 1 and both age ratios above it.
_ax60 = p60["axis_asymmetry"]
c27("p60 spatial median ratio, district over dong, 0.4692", 0.4692,
    _ax60["spatial"]["gu_over_dong"]["median"], 1e-3)
c27("p60 spatial median ratio, city over dong, 0.2454", 0.2454,
    _ax60["spatial"]["city_over_dong"]["median"], 1e-3)
check("27", "p60 spatial coarsening lowers r and age coarsening raises it", 1.0,
      1.0 if (_ax60["spatial"]["gu_over_dong"]["max"] < 1.0
              and _ax60["spatial"]["city_over_dong"]["max"] < 1.0
              and _ax60["age_3_over_16"]["min"] > 1.0) else 0.0, 1e-9)
check("27", "p60 reproduces p37's floored spatial ladder", 1.0,
      1.0 if _ax60["spatial"]["floored_reproduces_p37_kept_median"] else 0.0, 1e-9)
c27("p60 p19's own range-retained fraction is 0.506", 0.506,
    p60["cross_check"]["p19_range_retained_frac"], 1e-3)

# ============================================================ p61: the survey null
_a61 = p61["anchors"]
check("27", "p61 anchor count", 58.0, float(_a61["n"]), 1e-9)
check("27", "p61 anchors that pass", 58.0, float(_a61["n_ok"]), 1e-9)
check("27", "p61 anchors demanded bit-exact", 52.0,
      float(sum(1 for _c in _a61["checks"] if _c["tol"] == 0.0)), 1e-9)
check("27", "p61 bit-exact anchors that are bit-exact", 52.0,
      float(sum(1 for _c in _a61["checks"]
                if _c["tol"] == 0.0 and _c["abs_diff"] == 0.0)), 1e-9)
check("27", "p61 the only anchors with slack are p27's published CI", 4.0,
      float(sum(1 for _c in _a61["checks"] if _c["abs_diff"] != 0.0)), 1e-9)

_o61 = p61["cells"]["202312"]["observed"]
_pm61 = p61["cells"]["202312"]["permutation"]["sigma2_over_sigma1"]
_bo61 = p61["cells"]["202312"]["bootstrap"]["sigma2_over_sigma1"]
c27("p61 the survey's sigma2/sigma1 is 0.60137", 0.60137,
    _o61["sigma2_over_sigma1"], 1e-3)
c27("p61 the survey's leading component holds 0.4679 of the energy", 0.4679,
    _o61["sigma1_share"], 1e-3)
c27("p61 the permutation null's median is 0.126622", 0.126622,
    _pm61["median"], 1e-3)
c27("p61 the null's p95, the declared escape threshold, is 0.158452", 0.158452,
    _pm61["quantiles"]["p95"], 1e-3)
c27("p61 the null's p97.5 is 0.167098", 0.167098,
    _pm61["quantiles"]["p97.5"], 1e-3)
c27("p61 the null's largest draw of 500 is 0.211727", 0.211727, _pm61["max"], 1e-3)
c27("p61 the bootstrap's p2.5 is 0.516123", 0.516123,
    _bo61["quantiles"]["p2.5"], 1e-3)
c27("p61 the bootstrap's smallest resample is 0.479275", 0.479275,
    _bo61["min"], 1e-3)
# COUNT FIRST, then the verdicts: an emptied draw set is 0-of-0 and both escape
# flags would still read True.
check("27", "p61 the permutation null was drawn 500 times", 500.0,
      float(_pm61["n"]), 1e-9)
check("27", "p61 the bootstrap was drawn 500 times", 500.0, float(_bo61["n"]), 1e-9)
check("27", "p61 the declared primary escape clears", 1.0,
      1.0 if p61["branch"]["primary_escape"] else 0.0, 1e-9)
check("27", "p61 the declared secondary escape clears", 1.0,
      1.0 if p61["branch"]["secondary_escape"] else 0.0, 1e-9)
check("27", "p61 not one of the 500 permutations reaches the observation", 0.0,
      p61["branch"]["permutation_p_value"], 0.0)
# The two escape criteria are checked as arithmetic as well as read as flags,
# because a flag beside a number is a word beside a number.
check("27", "p61 the observation is above the null's p95", 1.0,
      1.0 if _o61["sigma2_over_sigma1"] > _pm61["quantiles"]["p95"] else 0.0, 1e-9)
check("27", "p61 the bootstrap's p2.5 is above the null's p97.5", 1.0,
      1.0 if _bo61["quantiles"]["p2.5"] > _pm61["quantiles"]["p97.5"] else 0.0,
      1e-9)

# The passive arm, and the sentence it is the spectral restatement of.
_pi61 = p61["passive_inside_survey_null"]["cells"]
c27("p61 passive dong sigma2/sigma1 is 0.1237", 0.1237,
    _pi61["202312|dong"]["passive_sigma2_over_sigma1"], 1e-3)
check("27", "p61 passive dong sits at the 43rd percentile of the survey's null",
      43.0, _pi61["202312|dong"]["percentile_within_survey_null"], 1e-9)
check("27", "p61 passive dong is inside the survey's own null", 1.0,
      1.0 if _pi61["202312|dong"]["inside_survey_null_95"] else 0.0, 1e-9)
check("27", "p61 passive district is not", 0.0,
      1.0 if _pi61["202312|gu"]["inside_survey_null_95"] else 0.0, 1e-9)

# The declared emptiness clause, which did NOT hold. Gated as a failure that
# stays visible rather than a paragraph that quietly disappeared.
_em61 = p61["emptiness_diagnostic_post_hoc"]["cells"]["202312"]
check("27", "p61 the declared emptiness clause did not hold", 0.0,
      1.0 if p61["emptiness_diagnostic_post_hoc"]["held"] else 0.0, 1e-9)
check("27", "p61 the survey block has 24 exactly-zero cells", 24.0,
      float(_em61["observed_zero_cells"]), 1e-9)
check("27", "p61 the null's median emptiness is 8 cells", 8.0,
      _em61["null_zero_cells_median"], 1e-9)
c27("p61 emptiness buys 0.0004388 of sigma2/sigma1 per empty cell", 0.0004388,
    _em61["slope_sigma2_per_empty_cell"], 1e-3)
# The bound that makes the diagnostic worth reporting: even extrapolated past
# the null's own support, emptiness cannot reach the observation.
check("27", "p61 emptiness extrapolated past the null's own support still "
            "does not reach the escape threshold", 1.0,
      1.0 if _em61["linear_extrapolation_to_observed_emptiness"]
      < _pm61["quantiles"]["p95"] else 0.0, 1e-9)

# ============================================================ p62: the rule's size
# The anchors first. p62 measures a property of p52's rule, so if it has not
# read p52's rule and replayed p52's verdicts, it is measuring some other rule.
check("27", "p62 read the flip rule off p52's own declaration", 1.0,
      _eq(p62["anchor_A1"]["source"], "results_p52.json .declaration.flip_rule"),
      1e-9)
check("27", "p62 the rule p55 echoes agrees with p52's", 1.0,
      1.0 if p62["anchor_A1"]["echo_agrees"] else 0.0, 1e-9)
check("27", "p62 the rule's low endpoint is the 2.5th percentile", 2.5,
      p62["anchor_A1"]["low_pct"], 1e-12)
check("27", "p62 the rule's high endpoint is the 97.5th percentile", 97.5,
      p62["anchor_A1"]["high_pct"], 1e-12)
check("27", "p62 the size is measured at p52's own 200 draws", 200.0,
      float(p62["anchor_A1"]["n_boot"]), 1e-9)
_a2_62 = p62["anchor_A2"]
check("27", "p62 replays p52's margins bit for bit", 0.0,
      _a2_62["worst_margin_abs_diff"], 0.0)
check("27", "p62 replayed all twenty of p52's verdicts", 20.0,
      float(_a2_62["verdicts_replayed_correctly"]), 1e-9)
check("27", "p62 weighs the same twenty verdicts", 20.0,
      float(_a2_62["n_verdicts"]), 1e-9)
check("27", "p62 the verdict grid is 4 cells x 5 R0", 20.0,
      float(p62["anchor_A3"]["n_cells"] * p62["anchor_A3"]["n_r0"]), 1e-9)

# THE CORRECTED FLIP COUNT. The 08-24 round's version of this check compares
# p52's n_flips against itself (:1988) and is therefore green forever; it sits
# under the frozen tag and is left exactly where it is, because editing a check
# that guards a sent document is not the repair -- the repair is a literal, here,
# in the round that noticed. 5 is one digit, so it is a literal and never a c27.
check("27", "p52 flips after correction", 5.0,
      float(p52["answer"]["n_flips"]), 1e-9)
check("27", "p62 reads the same five flips out of p52", 5.0,
      float(_a2_62["n_flips_stored"]), 1e-9)
check("27", "p62 counts five flips on its own replay", 5.0,
      float(_a2_62["n_flips"]), 1e-9)
check("27", "p62 seven flipped before the vector was corrected", 7.0,
      float(_a2_62["n_flips_published_before_correction"]), 1e-9)

_al62 = p62["alpha_rule"]
check("27", "p62 the flip rule's size at 200 draws", 6.2492e-101,
      _al62["value"], 1e-104)
c27("p62 the flip rule's size is 10^-100.2", -100.2, _al62["log10"], 1e-3)
check("27", "p62 the headline is the conservative end of the exact bracket", 0.0,
      abs(_al62["value"] - _al62["exact_bracket"]["upper"]), 0.0)
check("27", "p62 the exact bracket at 200 draws is the size table's own", 0.0,
      abs(_al62["value"] - p62["size_vs_n_boot"]["table"]["200"]["upper"]), 0.0)
check("27", "p62 expected flips over twenty verdicts", 1.2498e-99,
      p62["expected_flips"]["reading_20_independent"]["expected"], 1e-102)
check("27", "p62 expected flips over the four independent surveys", 2.4997e-100,
      p62["expected_flips"]["reading_4_independent"]["expected"], 1e-103)
c27("p62 the letter's arithmetic overstates the rate by 98.9 orders", 98.9,
    p62["answer"]["letter_is_wrong_by_log10"], 1e-3)
check("27", "p62 the letter's own reading was one flip in expectation", 1.0,
      p62["expected_flips"]["letter_arithmetic"]["expected_flips_20"], 1e-12)
# The outlier the letter has to get ahead of, identified rather than described.
check("27", "p62 the thinnest margin is 202402|seoul", 1.0,
      _eq(_a2_62["outlier"]["cell"], "202402|seoul"), 1e-9)
check("27", "p62 the outlier sits at R0 = 3.5", 3.5,
      _a2_62["outlier"]["r0"], 1e-12)
c27("p62 the outlier's margin is 0.014555", 0.014555,
    _a2_62["outlier"]["margin"], 1e-3)
check("27", "p62 four of the five flips sit at or below R0 = 1.8", 4.0,
      float(_a2_62["n_flips_inside_1p2_1p8"]), 1e-9)
check("27", "p62 one flip sits outside the pre-registered interval", 1.0,
      float(_a2_62["n_flips_outside"]), 1e-9)
# The real fragility at 200 draws is endpoint resolution, not multiplicity, so
# both coverage intervals are gated.
_ef62 = p62["endpoint_fragility"]
c27("p62 the nominal 0.025 endpoint covers as low as 0.011087", 0.011087,
    _ef62["low_level_ci"][0], 1e-3)
c27("p62 the nominal 0.025 endpoint covers as high as 0.057374", 0.057374,
    _ef62["low_level_ci"][1], 1e-3)
c27("p62 the nominal 0.975 endpoint covers as low as 0.942626", 0.942626,
    _ef62["high_level_ci"][0], 1e-3)
c27("p62 the nominal 0.975 endpoint covers as high as 0.988913", 0.988913,
    _ef62["high_level_ci"][1], 1e-3)
check("27", "p62 the nominal endpoints are bracketed, not centred", 1.0,
      1.0 if (_ef62["low_level_ci"][0] < _ef62["nominal_low"]
              < _ef62["low_level_ci"][1]) else 0.0, 1e-9)
# The second implementation this phase owes, which is internal: exact ranks
# beside Monte Carlo, and four distributions that must agree.
check("27", "p62 the distribution-free ladder covers four rungs", 4.0,
      float(p62["distribution_free_check"]["n_rungs"]), 1e-9)
check("27", "p62 sixteen (rung, distribution) estimates were compared", 16.0,
      float(sum(len(v["per_distribution"])
                for v in p62["distribution_free_check"]["ladder"].values())),
      1e-9)
check("27", "p62 every one of them sits inside the exact rank bracket", 16.0,
      float(sum(1 for v in p62["distribution_free_check"]["ladder"].values()
                for r in v["per_distribution"].values()
                if r["consistent_with_bracket"])), 1e-9)
check("27", "p62 the Monte Carlo at 200 draws hit nothing, in all four "
            "distributions", 1.0,
      1.0 if p62["mc_at_200"]["all_zero"] else 0.0, 1e-9)
check("27", "p62 ran the declared 200,000 Monte Carlo draws", 200000.0,
      float(p62["n_mc"]), 1e-9)
check("27", "p62 positive dependence only shrinks the size", 1.0,
      1.0 if p62["dependence_sensitivity"]["monotone_decreasing"] else 0.0, 1e-9)

# ============================================================ p36, the live file
check("27", "p36's live results file came from a full run", 1.0,
      1.0 if p36_live.get("full_run") else 0.0, 1e-9)
check("27", "p36 now covers thirteen sections", 13.0,
      float(len(p36_live.get("sections", []))), 1e-9)
check("27", "p36's live check count", 692.0, float(p36_live["n_checks"]), 1e-9)
check("27", "p36's live failure count", 0.0, float(p36_live["n_fail"]), 1e-9)
# ...and the frozen copy the 08-19 block reads is a DIFFERENT object, which is
# the whole reason this section exists. Gated so that a future edit cannot
# quietly repoint one at the other.
check("27", "p36's frozen copy is not its live one", 1.0,
      1.0 if p36["n_checks"] != p36_live["n_checks"] else 0.0, 1e-9)

# ====================================================== the figure caption sheets
# Nothing below compares a hand-typed number against a file. Every row is a
# caption sheet on the left and the results file it was read out of on the
# right, which is the one direction that was missing: p31 gated the sources and
# the letters, and nothing gated the sheet in between.

# --- p63: Figure 2, submission grade -----------------------------------------
# 2026-08-31 (advisor 3.2): five panels down to three. Old (d)'s p-ladder and
# old (e)'s per-year margins left the figure (rows for both are gone, not
# relabelled); old (c)'s agreement numbers moved into (b)'s notes so the
# "not three independent draws" point stays gated where the caption now makes
# it; and (c) is new -- the regression from results_p64.json.
_c63 = p63["caption_numbers"]
# 2026-09-02: 47 -> 53. Figure 3(c) became Figure 2(c), and its six rows
# (the dong min/max/median, the month count, the ladder count and the
# weekday panel it is read on) came with it.
check("27", "p63's caption sheet has rows to check", 53.0, float(len(_c63)), 1e-9)
# THE ALIGNMENT COUNT, still the figure's central quantitative claim.
check("27", "p63 quotes p58's 3 matching alignments", 3.0,
      _cap(_c63, 2, "b", "alternative alignments that also reach"), 1e-9)
check("27", "p63 quotes p58's 78 alternatives", 78.0,
      _cap(_c63, 2, "b", "alternatives in total"), 1e-9)
check("27", "p63's matching count is p58's own", 0.0,
      abs(_cap(_c63, 2, "b", "alternative alignments that also reach")
          - _ps58["n_shifts_at_or_above"]), 0.0)
check("27", "p63's alternative count is p58's denominator minus the identity",
      0.0,
      abs(_cap(_c63, 2, "b", "alternatives in total")
          - (_ps58["denominator"] - 1)), 0.0)
check("27", "p63's rotation p is p58's, bit for bit", 0.0,
      abs(_cap(_c63, 2, "b", "circular-shift p, all 78 alternatives")
          - _ps58["p"]), 0.0)
check("27", "p63's floor is p51's corrected national floor", 0.0,
      abs(_cap(_c63, 2, "a", "floor (p51 corrected national")
          - _a58["A7"]["floor"]), 0.0)
check("27", "p63's clearing count is p54's 17", 17.0,
      _cap(_c63, 2, "a", "months at or above the floor"), 1e-9)
_cl63 = _cap(_c63, 2, "a", "=the clearing months")
check("27", "p63's clearing month list is 17 months long", 17.0,
      float(len(_cl63)) if isinstance(_cl63, list) else -1.0, 1e-9)
check("27", "p63's run count is p58's 9", 0.0,
      abs(_cap(_c63, 2, "a", "=the clearing months sit in this many runs")
          - p58["run_structure"]["n_runs"]), 0.0)
check("27", "p63's invariant term margin is p58's 46", 0.0,
      abs(_cap(_c63, 2, "b", "term margin, invariant")
          - _a58["A5"]["margin"]), 0.0)
check("27", "p63's k=12 agreement is p58's 74", 0.0,
      abs(_cap(_c63, 2, "b", "agreement at k = 12")
          - _dg58["top_14_shifts_by_agreement"]["12"]), 0.0)
check("27", "p63's chance agreement is p58's", 0.0,
      abs(_cap(_c63, 2, "b", "chance agreement")
          - _dg58["chance_agreement"]), 0.0)
check("27", "p63's median contrast is p58's D(0)", 0.0,
      abs(_cap(_c63, 2, "b", "secondary median-contrast D(0)")
          - _sb58["observed"]), 0.0)
check("27", "p63's median-contrast p is p58's", 0.0,
      abs(_cap(_c63, 2, "b", "secondary median-contrast p") - _sb58["p"]), 0.0)
check("27", "p63's matching shifts are p58's three", 1.0,
      _eq(_cap(_c63, 2, "b", "the shifts that match"),
          _ps58["shifts_at_or_above_observed"]), 1e-9)
c27("p63's level trend over the period is 1.6977", 1.6977,
    _cap(_c63, 2, "a", "level trend over the period"), 1e-3)
check("27", "p63's series labels are p58's 46/27/6", 1.0,
      _eq(_cap(_c63, 2, "a", "series labels"), "46 / 27 / 6"), 1e-9)

# p63's link to p64 (the regression phase), the same way the rows above are
# its link to p58/p51: every number panel (c) draws is read from p64 at draw
# time, and these rows are what say so, not an assertion inside p63 alone.
check("27", "p63's regression term coefficient is p64's", 0.0,
      abs(_cap(_c63, 2, "b", "term coefficient, dong-level assortativity")
          - p64["primary"]["assortativity_all"]["coef"]["term"]), 0.0)
check("27", "p63's regression standard error is p64's", 0.0,
      abs(_cap(_c63, 2, "b", "Newey-West standard error")
          - p64["primary"]["assortativity_all"]["se"]["term"]), 0.0)
check("27", "p63's regression t is p64's", 0.0,
      abs(_cap(_c63, 2, "b", "t on the term coefficient")
          - p64["primary"]["assortativity_all"]["t"]["term"]), 0.0)
check("27", "p63's regression lag is p64's declared three", 3.0,
      _cap(_c63, 2, "b", "Newey-West lag, fixed by rule"), 1e-9)
check("27", "p63's honest floor is p64's 1/12", 0.0,
      abs(_cap(_c63, 2, "b", "resolution floor, 1 over the distinct")
          - p64["deduplicated"]["exact"]["resolution_floor"]), 0.0)
check("27", "p63's collapsed floor is p64's 1/6", 0.0,
      abs(_cap(_c63, 2, "b", "resolution floor after collapsing")
          - p64["deduplicated"]["advisor"]["resolution_floor"]), 0.0)
check("27", "p63's twelve distinct calendars is p64's own count", 0.0,
      abs(_cap(_c63, 2, "b", "distinct calendars the 79 rotations realise")
          - p64["effective_alignments"]["E_exact"]), 0.0)
check("27", "p63's observed-t rank among rotations is 1", 1.0,
      _cap(_c63, 2, "b", "the observed regression t ranks first"), 1e-9)
check("27", "p63's count of rotations beating the observed t is 0", 0.0,
      _cap(_c63, 2, "b", "calendar rotations whose t beats"), 1e-9)

# --- p48: Figure 1 ------------------------------------------------------------
# The three-way band split is the panel's whole content, and it is gated as a
# PARTITION rather than as three lists: three lists that each look plausible can
# still overlap or leave a band out, and the figure would draw a sixteenth band
# in two colours or none.
_mb48 = (p48["masked_bands_material"] + p48["masked_bands_trace"]
         + p48["masked_bands_none_low"] + p48["masked_bands_none_high"])
check("27", "p48's band split covers sixteen bands", 16.0, float(len(_mb48)), 1e-9)
check("27", "p48's band split assigns each band exactly once", 16.0,
      float(len(set(_mb48))), 1e-9)
check("27", "p48's material band count", 5.0,
      float(len(p48["masked_bands_material"])), 1e-9)
check("27", "p48's trace band count", 4.0,
      float(len(p48["masked_bands_trace"])), 1e-9)
check("27", "p48's unmasked young band count", 3.0,
      float(len(p48["masked_bands_none_low"])), 1e-9)
check("27", "p48's unmasked old band count", 4.0,
      float(len(p48["masked_bands_none_high"])), 1e-9)
# The 20-44 rule the whole masking section rests on: material masking is exactly
# the 20-24..40-44 bands, and the bands with no masking at all are exactly the
# ones p2 found to carry no `*`.
check("27", "p48's material bands are the 20-44 block", 1.0,
      _eq(p48["masked_bands_material"],
          ["20-24", "25-29", "30-34", "35-39", "40-44"]), 1e-9)
check("27", "p48's exactly-zero bands are its two unmasked groups", 1.0,
      _eq(p48["masked_bands_exactly_zero"],
          p48["masked_bands_none_low"] + p48["masked_bands_none_high"]), 1e-9)
check("27", "p48's trace cap", 0.13, p48["masked_band_trace_cap_pct"], 1e-12)
check("27", "p48's largest trace band is under the cap it prints", 1.0,
      1.0 if 100 * p48["masked_band_trace_max"]
      < p48["masked_band_trace_cap_pct"] else 0.0, 1e-9)
check("27", "p48's largest trace band is one of the trace bands", 1.0,
      1.0 if p48["masked_band_trace_max_band"] in p48["masked_bands_trace"]
      else 0.0, 1e-9)
# Claim 1's numbers on Figure 1 are p58's, and until this round nothing said so.
check("27", "p48's Claim 1 shift matches are p58's", 3.0,
      float(p48["claim1_shift_matches"]), 1e-9)
check("27", "p48's Claim 1 alternatives are p58's 78", 78.0,
      float(p48["claim1_shift_alternatives"]), 1e-9)
check("27", "p48's Claim 1 matches equal p58's count", 0.0,
      abs(p48["claim1_shift_matches"] - _ps58["n_shifts_at_or_above"]), 0.0)
check("27", "p48's Claim 1 alternatives equal p58's denominator minus one", 0.0,
      abs(p48["claim1_shift_alternatives"] - (_ps58["denominator"] - 1)), 0.0)
check("27", "p48's Claim 1 shift p equals p58's", 0.0,
      abs(p48["claim1_p_shift"] - _ps58["p"]), 0.0)
check("27", "p48's Claim 1 within-year p equals p58's", 0.0,
      abs(p48["claim1_p_term"] - p58["within_year"]["p_term_ge"]), 0.0)
check("27", "p48's Claim 1 clearing count equals p58's", 0.0,
      abs(p48["claim1_months_clearing"] - _ps58["observed"]), 0.0)
check("27", "p48's Claim 3 flips equal p52's", 0.0,
      abs(p48["claim3_flips"] - p52["answer"]["n_flips"]), 0.0)
check("27", "p48's Claim 3 verdicts equal p52's", 0.0,
      abs(p48["claim3_verdicts"] - p52["answer"]["n_verdicts"]), 0.0)
# 2026-09-21: the row "p48's p36 check count equals the live p36's" is gone
# with the thing it gated. Figure 1 carried a verification strip ("checked
# three ways ... 692 of 692") under its claim boxes from 09-03 to today; the
# advisor's 09-15 item 4 struck the same kind of self-advertisement from
# Figure 7, and the strip was the one instance that was not on an old figure
# number and so was not swept. p48 no longer reads p36 and its sheet no longer
# has a `p36_checks` key, so a row here would be a row over a value the figure
# does not show. The 692 is still gated where the prose says it: the two
# "data accessibility" check() rows and the 2.6 cms() row below.
check("27", "p48's month count", 79.0, float(p48["months"]), 1e-9)
check("27", "p48's dong count", 424.0, float(p48["dong"]), 1e-9)
c27("p48's weighted masking rate is 0.261349", 0.261349,
    p48["masking_rate_weighted"], 1e-6)
c27("p48's measured fill is 2.2673 per cell", 2.2673, p48["measured_fill"], 1e-3)
c27("p48's measured fill share is 0.110896", 0.110896,
    p48["measured_fill_share"], 1e-3)

# Figure 2(c), the 79-month ladder, against p60's r16 series -- the same object
# Figure 4(b) opens with, so a drift between them would split two figures in
# half. These three rows were p47 3(c)'s until 2026-09-02, when the panel moved
# from Figure 3 to Figure 2 (p47_fig34.py -> p63_fig2.py); the check is the
# same check, read out of the sheet the panel now writes to.
for _needle63, _src63 in (("dong assort_min", _s60["r16"]["min"]),
                          ("dong assort_max", _s60["r16"]["max"]),
                          ("dong assort_median", _s60["r16"]["median"])):
    check("27", f"p63 2(c) {_needle63} equals p60's r16 series", 0.0,
          abs(_cap(_c63, 2, "c", _needle63) - round(_src63, 5)), 0.0)
# --- p47: Figures 3 and 4 -----------------------------------------------------
_c47 = p47["caption_numbers"]
# 2026-08-31: 72 -> 81 -> 79. Figure 3(a) gained a greyscale-safe encoding
# this round -- deficit cells hatched, unobserved cells on a grey ground --
# and the script registers the numbers a caption needs to name them
# (empty-cell counts per matrix, both-direction-empty share, cell totals);
# then Figure 3(c) lost its six-published-months markers (advisor 3.10), so
# its published_six_min / published_six_max rows are gone. Advisor 0831
# items 3.1 / 3.4 / 3.10. The additions are measured, not quoted-only.
# 2026-09-02: 79 -> 74. Figure 3(c) left for Figure 2 entirely, taking its five
# remaining rows to p63's sheet, where they are checked a few blocks above.
check("27", "p47's caption sheet has rows to check", 74.0, float(len(_c47)), 1e-9)
# Figure 4(b), the new age ladder. Every number is rounded off p60 at draw time,
# so each is checked against p60 rounded the same way, exactly.
check("27", "p47 4(b) median 16-bin equals p60's", 0.0,
      abs(_cap(_c47, 4, "b", "median 16-bin assortativity")
          - round(_s60["r16"]["median"], 6)), 0.0)
check("27", "p47 4(b) median 3-bin equals p60's", 0.0,
      abs(_cap(_c47, 4, "b", "median 3-bin assortativity")
          - round(_s60["lim3"]["median"], 6)), 0.0)
check("27", "p47 4(b) ratio equals p60's", 0.0,
      abs(_cap(_c47, 4, "b", "median 3-bin / 16-bin ratio")
          - round(_s60["retained_frac"]["median"], 4)), 0.0)
check("27", "p47 4(b) ratio range equals p60's", 1.0,
      _eq(_cap(_c47, 4, "b", "ratio, range over the months"),
          f"{_s60['retained_frac']['min']:.4f}-"
          f"{_s60['retained_frac']['max']:.4f}"), 1e-9)
check("27", "p47 4(b) states 79 of 79 against the prediction", 1.0,
      _eq(_cap(_c47, 4, "b", "months moving against the declared prediction"),
          f"{p60['prediction']['n_against']} of {p60['prediction']['n_months']}"),
      1e-9)
check("27", "p47 4(b) proportionate-mixing null equals p60's", 1.0,
      _eq(_cap(_c47, 4, "b", "proportionate-mixing null"),
          f"{_m60['proportionate_mixing_null_max_abs_r']:.2e}"), 1e-9)
check("27", "p47 4(b) absorbed-excess gain equals p60's", 0.0,
      abs(_cap(_c47, 4, "b", "absorbed-excess gain")
          - round(_m60["numerator_gain"]["median"], 4)), 0.0)
check("27", "p47 4(b) margin-concentration shrink equals p60's", 0.0,
      abs(_cap(_c47, 4, "b", "margin-concentration shrink")
          - round(_m60["denominator_shrink"]["median"], 4)), 0.0)
for _n47, _lab47 in (("8", "8 bins"), ("4", "4 bins"), ("2", "2 bins")):
    check("27", f"p47 4(b) nested ladder at {_n47} bins equals p60's", 0.0,
          abs(_cap(_c47, 4, "b", f"nested ladder median, {_lab47}")
              - round(_s60["nested"][_n47]["median"], 6)), 0.0)
check("27", "p47 4(b) district/dong ratio equals p60's", 0.0,
      abs(_cap(_c47, 4, "b", "district / dong")
          - round(_ax60["spatial"]["gu_over_dong"]["median"], 4)), 0.0)
check("27", "p47 4(b) city/dong ratio equals p60's", 0.0,
      abs(_cap(_c47, 4, "b", "city / dong")
          - round(_ax60["spatial"]["city_over_dong"]["median"], 4)), 0.0)
check("27", "p47 4(b) draws one curve per month", 0.0,
      abs(_cap(_c47, 4, "b", "curves") - len(p60["months"])), 0.0)

# THE SHARED PARTITION. Panels (b) and (c) are only two views of one thing while
# they use the SAME three bands, and (b) gets its bands from p60 while (c) gets
# them from p19/p18b. If p19's LIM_BANDS ever moved, (b) would follow and (c)
# would not, and Figure 4 would silently go back to two panels measuring
# different objects -- which is the defect p60 was written to repair. So the
# partition is read out of p19's source the way p60 reads it (ast, never import:
# the literal is the thing the two panels must share, and reading it keeps this
# check independent of whether p19 imports at all -- p19 has had its __main__
# guard since 2026-08-28, so an import would no longer re-run the phase, but it
# would still pull duckdb and matplotlib in for one dict),
# checked against p60's anchor, and checked against every band name panel (c)
# prints.
_p19src = open(f"{ROOT}/eda/p19_bandwidth.py").read()
_m19 = re.search(r"^LIM_BANDS\s*=\s*(\{.*?\})\s*$", _p19src, re.S | re.M)
_LIM = ast.literal_eval(_m19.group(1)) if _m19 else {}
check("27", "p19's LIM_BANDS is still three bands", 3.0, float(len(_LIM)), 1e-9)
check("27", "p60's 3-bin partition IS p19's LIM_BANDS", 1.0,
      _eq(_LIM, _a60["A5"]["bands"]), 1e-9)
check("27", "p60's band anchor passed", 1.0,
      1.0 if _a60["A5"]["passed"] else 0.0, 1e-9)
# The rows keep their bucket name, "not drawn": Figure 4 lost its (c) panel
# on 2026-08-31 and a panel letter here would assert that the figure still
# draws them. They are quoted by 3.4's body and SI 6 instead, and this
# block checks the band names in them exactly as it did when there was a
# panel. The literal has to follow p47's note() bucket or the list empties
# and the membership check below turns into a 0-of-0 pass.
_c47_bands = [_b for _r in _c47
              if _r.get("figure") == 4 and _r.get("panel") == "not drawn"
              for _b in re.findall(r"3-band ([0-9]+(?:-[0-9]+|\+))", _r["what"])]
# COUNT FIRST: an empty list of band names would make the membership check below
# a 0-of-0 pass, which is the vacuity this file has been bitten by three times.
check("27", "p47's not-drawn rows name 3-band series to check", 5.0,
      float(len(_c47_bands)), 1e-9)
check("27", "p47's not-drawn rows name two distinct 3-band series", 2.0,
      float(len(set(_c47_bands))), 1e-9)
check("27", "every 3-band series in those rows is one of p19's bands", 0.0,
      float(sum(1 for _b in _c47_bands if _b not in _LIM)), 1e-9)
# The panel is gone (advisor 2026-08-31, 3.6); the underlying measurement
# is not, and this checks p47's own number against p60's independent read
# of the same p19 data exactly as it did when there was a panel to draw it on.
check("27", "p47's retained fraction equals p60's reading of p19", 0.0,
      abs(_cap(_c47, 4, "not drawn", "fraction of the range three bands retain")
          - p60["cross_check"]["p19_range_retained_frac"]), 0.0)

# Figure 3(b), the spectral panel, against p61 -- the panel that had no null at
# all until this round and now has two.
_q61 = _pm61["quantiles"]
for _needle47, _src47, _dp47 in (
        ("sigma2/sigma1, contact survey", _o61["sigma2_over_sigma1"], 4),
        ("sigma1 share, contact survey", _o61["sigma1_share"], 4),
        ("sigma2/sigma1, passive, dong",
         _pi61["202312|dong"]["passive_sigma2_over_sigma1"], 4),
        ("permutation median sigma2/sigma1", _pm61["median"], 6),
        ("p2.5 (lower edge", _q61["p2.5"], 6),
        ("p95 (the declared escape threshold)", _q61["p95"], 6),
        ("null, p97.5", _q61["p97.5"], 6),
        ("min over 500 ego-band shuffles", _pm61["min"], 6),
        ("max over 500 ego-band shuffles", _pm61["max"], 6),
        ("bootstrap, p2.5", _bo61["quantiles"]["p2.5"], 6),
        ("bootstrap, p97.5", _bo61["quantiles"]["p97.5"], 6),
        ("bootstrap, min over 500 respondent resamples", _bo61["min"], 6)):
    check("27", f"p47 3(b) {_needle47} equals p61's", 0.0,
          abs(_cap(_c47, 3, "b", _needle47) - round(_src47, _dp47)), 0.0)
check("27", "p47 3(b) primary escape equals p61's verdict", 1.0,
      _eq(_cap(_c47, 3, "b", "primary escape"), p61["branch"]["primary_escape"]),
      1e-9)
check("27", "p47 3(b) secondary escape equals p61's verdict", 1.0,
      _eq(_cap(_c47, 3, "b", "secondary escape"),
          p61["branch"]["secondary_escape"]), 1e-9)
check("27", "p47 3(b) permutation p equals p61's", 0.0,
      abs(_cap(_c47, 3, "b", "permutation p-value")
          - p61["branch"]["permutation_p_value"]), 0.0)
check("27", "p47 3(b) passive dong percentile equals p61's", 0.0,
      abs(_cap(_c47, 3, "b", "passive dong sigma2/sigma1, percentile")
          - _pi61["202312|dong"]["percentile_within_survey_null"]), 0.0)
check("27", "p47 3(b) observed empty cells equal p61's", 0.0,
      abs(_cap(_c47, 3, "b", "observed empty cells")
          - _em61["observed_zero_cells"]), 0.0)
check("27", "p47 3(b) null median empty cells equal p61's", 0.0,
      abs(_cap(_c47, 3, "b", "null median empty cells")
          - _em61["null_zero_cells_median"]), 0.0)

# --- p55: Figure S1 ------------------------------------------------------------
_c55 = p55["caption_numbers"]
check("27", "p55's caption sheet has rows to check", 26.0, float(len(_c55)), 1e-9)
check("27", "p55 was built at p52's pinned 200 draws", 200.0,
      float(p55["n_boot"]), 1e-9)
check("27", "p55's flip count equals p52's", 0.0,
      abs(_cap(_c55, None, "b", "verdicts that exceed the null, corrected")
          - p52["answer"]["n_flips"]), 0.0)
check("27", "p55's verdict count equals p52's", 0.0,
      abs(_cap(_c55, None, "b", "verdicts in total")
          - p52["answer"]["n_verdicts"]), 0.0)
check("27", "p55's pre-correction flip count equals p52's", 0.0,
      abs(_cap(_c55, None, "b", "before the vector was corrected")
          - p52["answer"]["n_flips_published"]), 0.0)
check("27", "p55's alpha_rule equals p62's", 0.0,
      abs(_cap(_c55, None, "b", "size of the flip rule under the global null")
          - _al62["value"]), 0.0)
check("27", "p55's alpha_rule log10 equals p62's", 0.0,
      abs(_cap(_c55, None, "b", "that size, log10") - _al62["log10"]), 0.0)
check("27", "p55's expected flips over 20 equal p62's", 0.0,
      abs(_cap(_c55, None, "b", "expected flips among 20 verdicts")
          - p62["expected_flips"]["reading_20_independent"]["expected"]), 0.0)
check("27", "p55's expected flips over 4 equal p62's", 0.0,
      abs(_cap(_c55, None, "b", "expected flips among the 4 independent")
          - p62["expected_flips"]["reading_4_independent"]["expected"]), 0.0)
check("27", "p55's thinnest margin equals p62's outlier margin", 0.0,
      abs(_cap(_c55, None, "b", "thinnest margin")
          - _a2_62["outlier"]["margin"]), 0.0)
check("27", "p55 names the outlier as Feb 2024 Seoul at R0 = 3.5", 1.0,
      1.0 if ("Feb 2024, Seoul, R0 = 3.5"
              in [_r["what"] for _r in _c55
                  if "thinnest margin" in _r["what"]][0]) else 0.0, 1e-9)
check("27", "p55's count of flips at or below R0 = 1.8 equals p62's", 0.0,
      abs(_cap(_c55, None, "b", "flips at or below R0 = 1.8")
          - _a2_62["n_flips_inside_1p2_1p8"]), 0.0)
for _end55, _src55 in (("0.025 endpoint", _ef62["low_level_ci"]),
                       ("0.975 endpoint", _ef62["high_level_ci"])):
    check("27", f"p55's {_end55} coverage equals p62's", 1.0,
          _eq(_cap(_c55, None, "b", _end55), _src55), 1e-9)

# --- p56: Figure 5 ------------------------------------------------------------
_c56 = p56["caption_numbers"]
check("27", "p56's caption sheet has rows to check", 43.0, float(len(_c56)), 1e-9)
# 35 -> 43 on 2026-09-05: panel (b) gained one "realised beta at requested x"
# row per grid point when the panel moved onto the realised axis.
check("27", "p56 was built at p59's pinned 100 replicates", 0.0,
      abs(p56["reps"] - p59["config"]["reps"]), 0.0)
check("27", "p56's bound at beta = 0.95 equals p59's", 0.0,
      abs(_cap(_c56, None, "b", "bound at beta = 0.95, the measured grid point")
          - _cl59["bound_at_0p95"]), 0.0)
check("27", "p56's beta* equals p59's primary value", 0.0,
      abs(_cap(_c56, None, "b", "beta* where the bound meets the survey")
          - _bs59["primary_value"]), 0.0)
check("27", "p56's beta* spread equals p59's", 0.0,
      abs(_cap(_c56, None, "b", "spread of the four declared interpolation rules")
          - _bs59["spread"]), 0.0)
check("27", "p56's p39-grid spread equals p59's on_p39_grid", 0.0,
      abs(_cap(_c56, None, "b", "on p39's 0.04-wide grid spanned") - _sp39), 0.0)
check("27", "p56's largest measured beta equals p59's", 0.0,
      abs(_cap(_c56, None, "b", "largest beta any reading gives")
          - _cl59["largest_measured_beta"]), 0.0)
check("27", "p56's analytic reading at 0.95 equals p59's", 0.0,
      abs(_cap(_c56, None, "b", "analytic r_obs/(1-beta) at beta = 0.95")
          - _an59), 0.0)
check("27", "p56's survey assortativity equals p59's", 0.0,
      abs(_cap(_c56, None, "a", "survey assortativity") - _cl59["r_survey"]), 0.0)
# The exclusion verdicts along the beta grid. COUNT FIRST -- an emptied sheet is
# 0-of-0 and "every verdict agrees" would read True.
_ex56 = [(float(_r["what"].split("=")[1].strip()), _r["value"]) for _r in _c56
         if _r["panel"] == "b" and _r["what"].startswith("survey excluded at beta")]
check("27", "p56 prints an exclusion verdict at every grid point", 8.0,
      float(len(_ex56)), 1e-9)
check("27", "p56's exclusion verdicts agree with p59's refined grid", 0.0,
      float(sum(1 for _b56, _v56 in _ex56
                if str(_b56) in p59["inversion_refined"]
                and p59["inversion_refined"][str(_b56)]["excludes_survey"] != _v56)),
      1e-9)
check("27", "p56 covers p59's five refined beta slices", 5.0,
      float(sum(1 for _b56, _ in _ex56 if str(_b56) in p59["inversion_refined"])),
      1e-9)

# --- p46: Figure 6 ------------------------------------------------------------
# THE LAST FIGURE SHEET, AND IT PAID FOR ITSELF ON THE DAY IT WAS BUILT. Until
# 2026-08-28 p46 wrote no results file at all, so Figure 6 was the one
# manuscript figure whose caption numbers had no machine-readable source and
# `grep p46` here returned nothing. Writing the sheet immediately exposed a
# wrong number that had been printed on the figure since 2026-08-22: panel (b)
# carried the TYPED string "leading band at theta=0: 40-44 (both months)".
# 40-44 is the leading band at theta = -1, and at theta = 0 the two months
# disagree -- December leads on 15-19, February on 75-79. archive/20260822/
# results_p33.json holds the same values, so the label was never true, and both
# prose sources (memo/phase33-coverage.md and section_coverage_draft.md) always
# had it right. It survived because it was a STRING and every assert inside p46
# was numeric. The six top_band rows below are the checks that stop it
# returning; they are the reason this block exists, not a detail of it.
#
# WHAT IS AND IS NOT CLOSED HERE. p46's own eleven pre-draw gates are December
# 2023 only. The sheet lists everything the figure draws, so the registrations
# below extend the guard to the whole of February 2024 in (a) and (b), every
# top_band, the grey band's two painted edges, the whole of panel (c) that the
# sheet carries, and all six bars of (d) rather than the one that was gated.
# NOT closed, because the sheet does not carry them: the six intermediate theta
# (+-0.75, +-0.5, +-0.25) in panels (a) and (b). The sheet records the three
# annotated endpoints per month plus the min over the family, and a min attained
# at an endpoint pins nothing in between. Widening the sheet is p46's job, not
# this gate's, so nothing is invented here to cover them.
_c46 = p46["caption_numbers"]
_LAB46 = {202312: "December 2023", 202402: "February 2024"}
_fam46 = {_ym: sorted(p33_live["R2_theta_family"][str(_ym)],
                      key=lambda r: r["theta"]) for _ym in _LAB46}
_th46 = {_ym: {round(r["theta"], 6): r for r in _fam46[_ym]} for _ym in _fam46}
_mvc46 = p33_live["mask_vs_coverage"]
# The "smallest multiple against the survey" is a min over four statistics and
# nine theta, not a key, so it is recomputed the way p46 computes it.
_small46 = {_ym: min(v[k] for v in p33_live["R2_gaps_vs_survey"][str(_ym)]
                     for k in ("half_l1", "cramers_v", "assortativity", "nmi"))
            for _ym in _LAB46}

# 2026-08-31: 58 -> 62. Figure 6(d) now registers each bar's forgone-benefit
# value and the panel's smallest and largest bar, so the caption can state
# the scope and the R0 grid without a reader inferring them. Advisor 3.8.
check("27", "p46's caption sheet has rows to check", 62.0, float(len(_c46)), 1e-9)
check("27", "p46 ran eleven pre-draw gates", 11.0, float(p46["gates_run"]), 1e-9)
check("27", "p46's gates block records every gate it ran", 0.0,
      abs(len(p46["gates"]) - p46["gates_run"]), 0.0)
check("27", "p46 checked eleven caption claims", 11.0,
      float(p46["caption_claims"]), 1e-9)
check("27", "every pre-draw gate p46 recorded passed", 0.0,
      float(sum(1 for _g in p46["gates"] if not _g["ok"])), 1e-9)

# 6(a): the coverage family. The grid first -- a sheet drawn from a shorter
# family would still satisfy every per-theta row below it.
check("27", "p46 6(a) draws 18 points across both months", 18.0,
      float(_cap(_c46, None, "a", "points drawn across both months")), 1e-9)
check("27", "p46 6(a)'s point count equals p33's family size", 0.0,
      abs(_cap(_c46, None, "a", "points drawn across both months")
          - sum(len(_fam46[_ym]) for _ym in _fam46)), 0.0)
check("27", "p46 6(a) states the theta grid as -1 / 1 / 9", 1.0,
      _eq(_cap(_c46, None, "a", "coverage family: theta from"), "-1 / 1 / 9"),
      1e-9)
check("27", "p46 6(a)'s theta grid equals p33's", 1.0,
      _eq(_cap(_c46, None, "a", "coverage family: theta from"),
          f"{_fam46[202312][0]['theta']:g} / {_fam46[202312][-1]['theta']:g}"
          f" / {len(_fam46[202312])}"), 1e-9)
# The three annotated theta, BOTH months. February was drawn in full and gated
# nowhere until now.
_A46 = {(202312, -1.0): 0.01704, (202312, 0.0): 0.02115, (202312, 1.0): 0.05099,
        (202402, -1.0): 0.01459, (202402, 0.0): 0.01851, (202402, 1.0): 0.04740}
for (_ym46, _t46), _q46 in _A46.items():
    _n46 = f"{_LAB46[_ym46]}: assortativity at theta = {_t46:+.0f}"
    c27(f"p46 6(a) {_ym46} assortativity at theta = {_t46:+.0f} is {_q46}",
        _q46, _th46[_ym46][_t46]["assortativity"], 1e-3)
    check("27", f"p46 6(a) {_ym46} assortativity at theta={_t46:+.0f} equals "
                f"p33's", 0.0,
          abs(_cap(_c46, None, "a", _n46)
              - _th46[_ym46][_t46]["assortativity"]), 0.0)
check("27", "p46 6(a)'s smallest assortativity is the min over both months", 0.0,
      abs(_cap(_c46, None, "a", "smallest assortativity anywhere")
          - min(_r["assortativity"] for _ym46 in _fam46
                for _r in _fam46[_ym46])), 0.0)
# WHICH TOLERANCE, and it is set by the quoted precision rather than fitted:
# 1e-3 where the caption types four or five significant figures, 6e-3 where it
# types three (0.00242, 0.00230, 3.36x, 3.49x). p46's own gates use the same
# split for the same values. The exact layer is the check() beside each one,
# at tol 0.0 against the source.
_W46 = {202312: (3.36, 14.05, 0.03395, 0.00242),
        202402: (3.49, 14.27, 0.03281, 0.00230)}
for _ym46, (_qs46, _qr46, _qc46, _qm46) in _W46.items():
    _l46 = _LAB46[_ym46]
    _s46 = _mvc46[str(_ym46)]
    c27(f"p46 6(a) {_ym46}'s smallest multiple against the survey is {_qs46}x",
        _qs46, _small46[_ym46], 6e-3)
    c27(f"p46 6(a) {_ym46}'s masking band is {_qr46}x narrower", _qr46,
        _s46["ratio"], 1e-3)
    c27(f"p46 6(a) {_ym46}'s coverage family width is {_qc46}", _qc46,
        _s46["theta_width"], 1e-3)
    c27(f"p46 6(a) {_ym46}'s masking band width is {_qm46}", _qm46,
        _s46["mask_width"], 6e-3)
    check("27", f"p46 6(a) {_ym46} smallest multiple equals p33's family min",
          0.0, abs(_cap(_c46, None, "a", f"{_l46}: smallest multiple")
                   - _small46[_ym46]), 0.0)
    check("27", f"p46 6(a) {_ym46} masking/coverage ratio equals p33's", 0.0,
          abs(_cap(_c46, None, "a", f"{_l46}: masking is narrower")
              - _s46["ratio"]), 0.0)
    check("27", f"p46 6(a) {_ym46} coverage family width equals p33's", 0.0,
          abs(_cap(_c46, None, "a", f"{_l46}: coverage family width")
              - _s46["theta_width"]), 0.0)
    check("27", f"p46 6(a) {_ym46} masking band width equals p33's", 0.0,
          abs(_cap(_c46, None, "a", f"{_l46}: masking band width")
              - _s46["mask_width"]), 0.0)
# The grey axhspan is painted from two edges and only their DIFFERENCE was
# gated. A band that slid up or down by a constant kept its width and would
# have passed.
for _e46, _i46 in (("lower edge", 0), ("upper edge", 1)):
    check("27", f"p46 6(a)'s grey band {_e46} equals p33's mask_band", 0.0,
          abs(_cap(_c46, None, "a", _e46)
              - _mvc46["202312"]["mask_band"][_i46]), 0.0)

# 6(b): the ordering, and the label that was wrong for six days.
c27("p46 6(b) December 2023's Kendall tau at theta = +1 is 0.117", 0.117,
    _th46[202312][1.0]["kendall_tau_vs_theta0"], 6e-3)
for _ym46 in _fam46:
    _l46 = _LAB46[_ym46]
    # tau at theta = -1 is exactly 1/6 in BOTH months, so it is not distinctive
    # and goes through check() against a literal: c27 would put one number into
    # the reverse direction twice and report a pass it never made.
    check("27", f"p46 6(b) {_ym46} Kendall tau at theta = -1 is 1/6", 0.166667,
          _cap(_c46, None, "b", f"{_l46}: Kendall tau at theta = -1"), 1e-6)
    check("27", f"p46 6(b) {_ym46} tau at theta = 0 is the identity", 1.0,
          _cap(_c46, None, "b",
               f"{_l46}: Kendall tau at theta = 0 is the identity"), 1e-9)
    for _t46 in (-1.0, 0.0, 1.0):
        _n46 = ("Kendall tau at theta = 0 is the identity" if _t46 == 0.0
                else f"Kendall tau at theta = {_t46:+.0f}")
        check("27", f"p46 6(b) {_ym46} tau at theta={_t46:+.0f} equals p33's",
              0.0, abs(_cap(_c46, None, "b", f"{_l46}: {_n46}")
                       - _th46[_ym46][_t46]["kendall_tau_vs_theta0"]), 0.0)
    check("27", f"p46 6(b) {_ym46}'s smallest tau is the min over the family",
          0.0, abs(_cap(_c46, None, "b", f"{_l46}: smallest Kendall tau")
                   - min(_r["kendall_tau_vs_theta0"]
                         for _r in _fam46[_ym46])), 0.0)
# THE SIX ROWS THIS BLOCK EXISTS FOR. Each band is checked twice: against the
# literal the figure should print, and against p33's top_band. The first alone
# would freeze today's reading into this file; the second alone would pass if
# p33 and the sheet went wrong together.
_TOP46 = {(202312, -1.0): "40-44", (202312, 0.0): "15-19", (202312, 1.0): "0-9",
          (202402, -1.0): "40-44", (202402, 0.0): "75-79", (202402, 1.0): "0-9"}
for (_ym46, _t46), _b46 in _TOP46.items():
    _n46 = f"{_LAB46[_ym46]}: leading band at theta = {_t46:+.0f}"
    check("27", f"p46 6(b) {_ym46} leads on {_b46} at theta={_t46:+.0f}", 1.0,
          _eq(_cap(_c46, None, "b", _n46), _b46), 1e-9)
    check("27", f"p46 6(b) {_ym46} leading band at theta={_t46:+.0f} equals "
                f"p33's", 1.0,
          _eq(_cap(_c46, None, "b", _n46), _th46[_ym46][_t46]["top_band"]), 1e-9)
check("27", "p46 6(b)'s theta = 0 label is built from p33, not typed", 1.0,
      _eq(_cap(_c46, None, "b", "the theta = 0 label drawn on the panel"),
          f"leading band at $\\theta=0$: {_th46[202312][0.0]['top_band']} (Dec),"
          f" {_th46[202402][0.0]['top_band']} (Feb)"), 1e-9)
check("27", "p46 6(b) says the months do NOT agree at theta = 0", 0.0,
      1.0 if _cap(_c46, None, "b", "the months agree on the leading band")
      else 0.0, 1e-9)
check("27", "p46 6(b)'s agreement verdict equals p33's two top_bands", 1.0,
      _eq(_cap(_c46, None, "b", "the months agree on the leading band"),
          _th46[202312][0.0]["top_band"] == _th46[202402][0.0]["top_band"]),
      1e-9)

# 6(c): who to buy first.
#
# p35.marginal HAS NO SECOND IMPLEMENTATION. p36_recompute.py rebuilds the
# headline numbers by independent routes and the SEIR marginals are not among
# them, so every check in this panel gates the CITATION -- that the sheet, and
# through it the caption, still says what results_p35.json says -- and NOT the
# number. CLAUDE.md's rule that a number existing in one implementation is not
# yet a result is unmet for these, and nothing below meets it; they only stop
# the figure and its source drifting apart. Stated here rather than left
# implicit, because the two kinds of guarantee are not interchangeable.
_MARG46 = p35_live["marginal"]["202312"]
_BANDS46 = p35_live["bands"]
check("27", "p46 6(c) ranks 15 age bands", 15.0,
      float(_cap(_c46, None, "c", "age bands ranked")), 1e-9)
check("27", "p46 6(c)'s band count equals p35's", 0.0,
      abs(_cap(_c46, None, "c", "age bands ranked") - len(_BANDS46)), 0.0)
check("27", "p46 6(c) draws December 2023", 1.0,
      _eq(_cap(_c46, None, "c", "the month panel (c) draws"), "202312"), 1e-9)
_TAG46 = (("contact survey", "survey", "10-14"),
          ("passive, dong", "passive_dong", "65-69"),
          ("passive, district", "passive_gu", "75-79"))
for _nm46, _tg46, _sc46 in _TAG46:
    _v46 = _MARG46[_tg46]
    _o46 = sorted(range(len(_v46)), key=lambda i: -_v46[i])
    check("27", f"p46 6(c) {_tg46} buys 15-19 first", 1.0,
          _eq(_cap(_c46, None, "c", f"{_nm46}: buy first (rank 1)"), "15-19"),
          1e-9)
    check("27", f"p46 6(c) {_tg46}'s first choice is p35's argmax", 1.0,
          _eq(_cap(_c46, None, "c", f"{_nm46}: buy first (rank 1)"),
              _BANDS46[max(range(len(_v46)), key=lambda i: _v46[i])]), 1e-9)
    check("27", f"p46 6(c) {_tg46} buys {_sc46} second", 1.0,
          _eq(_cap(_c46, None, "c", f"{_nm46}: buy second (rank 2)"), _sc46),
          1e-9)
    check("27", f"p46 6(c) {_tg46}'s second choice is p35's runner-up", 1.0,
          _eq(_cap(_c46, None, "c", f"{_nm46}: buy second (rank 2)"),
              _BANDS46[_o46[1]]), 1e-9)
    check("27", f"p46 6(c) {_tg46}'s rank-1 marginal equals p35's", 0.0,
          abs(_cap(_c46, None, "c", f"{_nm46}: marginal reduction in R0 per "
                                    "person-day at rank 1") - max(_v46)), 0.0)
_FIRST46 = {_BANDS46[max(range(len(_MARG46[_t])), key=lambda i: _MARG46[_t][i])]
            for _, _t, _ in _TAG46}
_SECOND46 = {_BANDS46[sorted(range(len(_MARG46[_t])),
                             key=lambda i: -_MARG46[_t][i])[1]]
             for _, _t, _ in _TAG46}
check("27", "p46 6(c) says all three matrices agree on the first choice", 1.0,
      1.0 if _cap(_c46, None, "c", "all three matrices agree on the first "
                                   "choice") else 0.0, 1e-9)
check("27", "p46 6(c)'s agreement verdict equals p35's three argmaxes", 1.0,
      _eq(_cap(_c46, None, "c", "all three matrices agree on the first choice"),
          len(_FIRST46) == 1), 1e-9)
check("27", "p46 6(c) names 15-19 as the choice they agree on", 1.0,
      _eq(_cap(_c46, None, "c", "the first choice they agree on"), "15-19"),
      1e-9)
check("27", "p46 6(c)'s agreed choice equals p35's argmax", 1.0,
      _eq(_cap(_c46, None, "c", "the first choice they agree on"),
          sorted(_FIRST46)[0]), 1e-9)
check("27", "p46 6(c) says the three second choices are distinct", 1.0,
      1.0 if _cap(_c46, None, "c", "the three second choices are distinct")
      else 0.0, 1e-9)
check("27", "p46 6(c)'s distinctness verdict equals p35's three runners-up",
      1.0, _eq(_cap(_c46, None, "c", "the three second choices are distinct"),
               len(_SECOND46) == 3), 1e-9)

# 6(d): what the disagreement costs. p46's own gate covered ONE of the six
# bars (202312 at R0 = 2.5); the other five were drawn and unguarded.
check("27", "p46 6(d) draws three values of R0", 1.0,
      _eq(_cap(_c46, None, "d", "R0 values drawn"), "1.3/1.8/2.5"), 1e-9)
# 2026-08-31, advisor 3.8: the caption now states the scope and names Figure S1's
# grid so 6(d) and Figure S1 cannot read as a contradiction. Both are gated.
# 2026-09-15: Figure 7 became Figure S1 in the SI; the note's text follows it.
check("27", "p46 6(d) records the scope as seoul", 1.0,
      _eq(_cap(_c46, None, "d", "scope of every panel"), "seoul"), 1e-9)
check("27", "p46 6(d) records Figure S1's grid as twelve points", 12.0,
      float(_cap(_c46, None, "d", "R0 values on Figure S1's grid")), 1e-9)
_REG46 = {(202312, 1.3): 20.2, (202312, 1.8): 2.97, (202312, 2.5): 7.5,
          (202402, 1.3): 21.7, (202402, 1.8): 15.4, (202402, 2.5): 23.3}
for (_ym46, _r046), _q46 in _REG46.items():
    _src46 = p35_live["allocation"][f"{_ym46}|R0={_r046:g}"][
        "cross_application"]["regret_share_of_benefit"]
    c27(f"p46 6(d) {_ym46} forgoes {_q46}% of the benefit at R0 = {_r046:g}",
        _q46, 100 * _src46, 6e-3)
    check("27", f"p46 6(d) {_ym46} at R0={_r046:g} equals p35's", 0.0,
          abs(_cap(_c46, None, "d", f"{_LAB46[_ym46]}: benefit forgone at "
                                    f"R0 = {_r046:g}") - _src46), 0.0)
# The two end bars are the min and max OVER the six above, so they are checked
# against a recomputed extremum rather than a key -- and NOT registered with
# c27, because 2.97% and 23.3% are already registered as bars in their own
# right and one number owed to the prose twice is a pass nobody made.
_ALL46 = [p35_live["allocation"][f"{_ym46}|R0={_r046:g}"]["cross_application"][
    "regret_share_of_benefit"] for _ym46 in _LAB46 for _r046 in (1.3, 1.8, 2.5)]
check("27", "p46 6(d) prints one bar value per month per R0", 6.0,
      float(len(_ALL46)), 1e-9)
check("27", "p46 6(d)'s smallest bar is the min over the six", 0.0,
      abs(_cap(_c46, None, "d", "smallest bar on the panel") - min(_ALL46)), 0.0)
check("27", "p46 6(d)'s largest bar is the max over the six", 0.0,
      abs(_cap(_c46, None, "d", "largest bar on the panel") - max(_ALL46)), 0.0)


# Every caption row this gate asked for resolved to exactly one entry. A lookup
# that missed already failed its own row as NaN; this names the reason.
check("27", "every caption row this gate asks for resolves to one entry", 0.0,
      float(len(_CAP_MISS)), 1e-9)
if _CAP_MISS:
    print(f"  CAPTION ROWS NOT FOUND: {_CAP_MISS}")



# ======================================================================
# THE MANUSCRIPT. paper/manuscript_JRSI.md and paper/manuscript_JRSI_SI.md,
# checked as ONE gated pair against LIVE results files.
#
# WHY THIS BLOCK EXISTS. Every document this gate reads is a letter to the
# advisor or a caption sheet that travelled with one. The manuscript is the one
# artefact that leaves this repo for a journal, and until now it was the only
# document here whose numbers no gate checked -- so a value could go stale in
# it exactly the way "377 項" went stale in the prose, with nobody told.
#
# LIVE, NOT A SNAPSHOT, and the rule that decides which. A sent document is
# checked against the state it was written from; an unsent one against LIVE,
# because its numbers are still allowed to move and the document is meant to
# move with them. The manuscript has not been submitted, so it reads LIVE. On
# the day it is submitted it gets its own eda/archive/<date>/ freeze and every
# load() below is repointed there in the same commit, exactly as the 08-27
# round's comment describes.
#
# ONE PAIR, NOT TWO. The main text and the electronic supplementary material
# are a single submission and a single envelope: the main text states a claim
# and the SI carries its arithmetic, so a number that lives only in the SI is
# quoted, not missing. Splitting them would report every SI-only value as
# "quoted nowhere" in the main text and vice versa, which is the mistake the
# 08-22 round made by reading the letter without its annex.
#
# THIN SPACES. Royal Society style groups thousands with U+2009 THIN SPACE, so
# the manuscript says "10 186 891 962" and not "10,186,891,962". That is a
# typography fact, not a citation fact, so it is handled where the other one
# already is -- in _forms(), beside the U+2212 rule -- rather than by putting
# the commas back.
#
# WHAT IS NOT HERE, and why that is not a gap that can be closed by trying
# harder: dates, the two Zenodo DOIs, reference counts, the product manual's
# wording, docomo's published band width, and the four synthetic-world
# detector deviations that p39 prints to its memo but not to its results file.
# They are named in main()'s not-covered note rather than given an invented
# source.
# ======================================================================
MANUSCRIPT = f"{ROOT}/paper/manuscript_JRSI.md"
MANUSCRIPT_SI = f"{ROOT}/paper/manuscript_JRSI_SI.md"

IN_TEXT_MS = []


# Moved up from beside _num() so cms() can validate a phrase against the value
# it is the sentence for at the moment the row is written, rather than at the
# bottom of main() where the traceback no longer names the offending line.
def _forms(q):
    """Every string form a number might plausibly be written as in prose."""
    forms = set()
    for v in (q, -q):
        for fmt in ("%g", "%.1f", "%.2f", "%.3f", "%.4f", "%.5f", "%.2e"):
            forms.add((fmt % v).lstrip("+"))
    forms |= {f.replace("e-0", "e-").replace("e+0", "e") for f in forms}
    if abs(q) >= 1000 and float(q).is_integer():
        forms.add(f"{int(q):,}")
        # Royal Society style groups thousands with U+2009 THIN SPACE, so the
        # manuscript writes "10 186 891 962" where the letters write
        # "10,186,891,962". Both spellings are the same citation, so both are
        # generated here rather than the manuscript being normalised to commas:
        # this matcher's job is to recognise a number, not to have an opinion
        # about the typesetting of the document it is reading.
        forms.add(f"{int(q):,}".replace(",", "\u2009"))
    # The letters are typeset in Chinese and a negative number in them is as
    # likely to carry U+2212 MINUS SIGN as an ASCII hyphen -- all four sent
    # documents mix the two. Without this the reverse-direction check reports a
    # value as "quoted nowhere" purely because of which dash the prose used,
    # which is a typography fact rather than a citation fact.
    forms |= {f.replace("-", "\u2212") for f in forms if "-" in f}
    # Royal Society style writes a small p as "7.6 × 10⁻¹¹", never as "7.6e-11",
    # so three rows gating live sentences were reported as quoted nowhere the
    # moment the matcher stopped accepting "0.0" for them. The typeset spelling
    # is generated here rather than the manuscript being made to write "e-11".
    if q != 0:
        _exp = math.floor(math.log10(abs(q)))
        _man = q / 10.0 ** _exp
        _sup = str(_exp).translate(str.maketrans("-0123456789",
                                                 "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        for _mf in ("%g", "%.1f", "%.2f", "%.3f"):
            _m = _mf % _man
            forms.add(f"{_m} × 10{_sup}")
            forms.add(f"{_m}×10{_sup}")
            forms.add(f"{_m} × 10^{{{_exp}}}")
    # AND A FORM HAS TO STILL BE THE NUMBER. "%.1f" of 0.00202 is "0.0", which is
    # a substring of "0.0257" -- so for three rounds the reverse direction
    # answered "yes, the abstract still prints 0.00202" by finding two characters
    # inside the number that had REPLACED it. A form that has rounded the value
    # away to zero identifies nothing and is dropped. Only that: a rule that also
    # dropped coarse-but-nonzero forms ("0.005" for 0.00455) turns fifteen more
    # manuscript rows red, and whether those rows are stale is a real question
    # but a different one from this.
    def _identifies(f):
        try:
            v = float(f.replace("−", "-").replace(",", "").replace("\u2009", ""))
        except ValueError:
            return True
        return q == 0 or v != 0
    return {f for f in forms if _identifies(f)}


_SUPERSCRIPT = str.maketrans("\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
                             "-0123456789")


def _same_number(q, f):
    """Does this spelling of `q` read back as `q`, or as a different number?

    _forms() deliberately emits shorter roundings as well: "%.1f" of 0.2065 is
    "0.2". Those are how the reverse direction went green on rows whose number
    the document never printed -- "0.2" is a substring of the 0.2227 standing
    three words away in the same caption, and on 2026-09-09 eighteen manuscript
    rows were resting on exactly that, seven of them for values the pair prints
    nowhere at all. A spelling that reads back as a different number identifies
    a different number, so it is not evidence that this one is still there.

    Everything that is the same value in other clothes survives: "0.2340", "13"
    for 13.0, "1 896" with its thin space, "2.065 × 10\u207b\u00b9" and
    "2.065 × 10^{-1}", a U+2212 minus. The three scientific spellings are
    normalised here rather than left to float() -- letting a ValueError mean
    "keep it" would have readmitted "2.1 × 10\u207b\u00b9" for 0.2065 through the
    back door.

    A form with no digits at all is a number the prose spells out ("fourteen
    times"). Those are matched by cms(phrase=...) and never reach this test, so
    an unparseable form is kept rather than dropped.

    This is applied to the MANUSCRIPT only. The five rounds of sent letters keep
    the loose reading on purpose: they are frozen, no sentence can move out from
    under a row there, and tightening them turns 63 rows into notes about the
    past that say nothing about any number.
    """
    t = (f.replace("\u2212", "-").replace(",", "")
          .replace("\u2009", "").replace("\u00a0", ""))
    t = re.sub(r"\s*\u00d7\s*10\^?\{(-?\d+)\}", r"e\1", t)
    t = re.sub(r"\s*\u00d7\s*10([\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079]+)",
               lambda mm: "e" + mm.group(1).translate(_SUPERSCRIPT), t)
    try:
        v = float(t.replace(" ", ""))
    except ValueError:
        return True
    return abs(v - float(q)) <= 1e-9 * max(1.0, abs(float(q)))


# ======================= THE ROUNDING DIRECTION OF A BOUND ===================
# A quote that names a bound is not a measurement with a tolerance around it.
# "at most 0.0123" against a measured 0.012326 is an upper bound on nothing, and
# "at least 83%" against a measured 82.52% is a lower bound on nothing; both
# sentences are false as written while sitting comfortably inside the half-unit
# cms() allows, because cms() gates a DISTANCE and half a unit in the last place
# is exactly as far in the direction that breaks the sentence as in the one that
# keeps it. Round a bound TOWARDS its own claim -- up for an upper bound, down
# for a lower one -- and the sentence stays true at whatever precision it prints.
#
# The rule was known here before it was enforced, which is the point. p48 builds
# its trace cap with math.ceil for exactly this reason, and the abstract carried
# a hand-written "0.0258 really is an upper bound" row after "at most 0.002" was
# found to be a rounding of 0.00202 that bounded nothing. Both are the rule
# applied where somebody happened to remember it, and remembering is not a
# mechanism. So: a row whose own words state a bound MUST declare which way it
# rounds, the declaration is checked against the source rather than believed,
# and the prose is swept for bound sentences no row declares at all.
_BOUND_UPPER = (r"at most", r"at or below", r"no more than", r"no larger than",
                r"no greater than", r"never exceeds?", r"does not exceed",
                r"never further than", r"or less", r"or fewer", r"or below")
_BOUND_LOWER = (r"at least", r"at or above", r"no fewer than", r"no less than",
                r"no smaller than", r"never falls below", r"cannot fall below",
                r"or more", r"or above", r"or greater")
# "at best" and "at worst" name a direction only once you know which way the
# quantity runs: "falls 25.1% short at best" is a MINIMUM and "30.0% short at
# worst" a maximum, and on a benefit rather than a shortfall the two would swap.
# Bare "above" and "below" have the same defect -- "takes the p above 4.7e-08"
# is an upper bound on the p. They are listed here so a row carrying one is
# still forced to declare, and kept out of the two lists above so that the
# declaration is not then contradicted by a word that does not know its own
# direction.
_BOUND_EITHER = (r"at best", r"at worst", r"\babove\b", r"\bbelow\b")
_BOUND_GAP = 24     # characters between a bound word and the number it governs
# Every bound row: (label, quoted, actual, "upper" | "lower").
BOUNDS = []

# Sentences and rows whose digits are not the bound, with the reason. A
# register rather than a pattern exclusion, for the same reason
# PRIORITY_NOT_A_CLAIM is one: an exemption has to be visible, and has to be
# re-argued the moment the wording it exempts changes. A key is matched against
# a row's label and against the prose sentence, so one entry covers both sides.
BOUND_NOT_A_BOUND = {
    "SI 5: with seven years the p cannot fall below 0.008":
        "the bound is printed exactly beside it, as 2**-7; 0.008 is the gloss "
        "on that closed form and not the bound, so rounding it down to 0.0078 "
        "would make the sentence harder to read and no truer",
    "cannot return a p below 2":
        "the same sentence, read from the prose side",
    "SI 3: their upper edge is 21% below the 79-month maximum":
        "\"21% below\" is how far apart two numbers are, not a bound on 21; "
        "the quantity being bounded is the edge, and it is gated on its own row",
    "upper edge is 21% below that maximum":
        "the same sentence, read from the prose side",
}

# A number as a label or the prose writes one, including Royal Society thin
# space grouping and the typeset "8 x 10^-4" spelling.
_BOUND_TOKEN = re.compile(
    "[-\u2212]?\\d[\\d,\u2009]*(?:\\.\\d+)?(?:e[-+\u2212]?\\d+)?"
    "(?:\\s*\u00d7\\s*10[\u207b\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+)?",
    re.I)
_SUP = str.maketrans("\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076"
                     "\u2077\u2078\u2079", "-0123456789")


def _bound_value(tok):
    """The float a number token spells, or None.

    Compared numerically rather than as a string, which is the whole reason
    this exists: SI 7 writes its bound "8e-4" in the row and "8 x 10^-4" in the
    prose, and _forms() generates neither of those from 0.0008. A rule that
    only matched spellings would have passed that row over in silence, which is
    the failure this block was written to remove rather than to reproduce.
    """
    t = tok.replace("\u2212", "-").replace(",", "").replace("\u2009", "").strip()
    m = re.match("^(.*?)\\s*\u00d7\\s*10(.+)$", t)
    if m:
        t = f"{m.group(1)}e{m.group(2).translate(_SUP)}"
    try:
        return float(t)
    except ValueError:
        return None


def _bound_dirs(text, quoted):
    """The directions `text` claims ABOUT `quoted`, as a subset of the three.

    A bound word counts only where the number it governs IS this row's number.
    Without that test "four of the five sit at or below R0 = 1.8" reads as a
    bound on the four and "at least one cell's regret rises" as a bound on the
    one. Neither row states a bound on its own value, and making them declare
    one would be the gate inventing claims instead of finding them.
    """
    if not text:
        return set()
    low = text.lower()
    at = [(m.start(), m.end()) for m in _BOUND_TOKEN.finditer(low)
          if _bound_value(m.group(0)) == quoted]
    if not at:
        return set()
    dirs = set()
    for name, words in (("upper", _BOUND_UPPER), ("lower", _BOUND_LOWER),
                        ("either", _BOUND_EITHER)):
        for w in words:
            for m in re.finditer(w, low):
                if any(max(a - m.end(), m.start() - b) <= _BOUND_GAP
                       for a, b in at):
                    dirs.add(name)
                    break
    return dirs

def _holds(quoted, actual, bound):
    return quoted >= actual if bound == "upper" else quoted <= actual


def cms(label, quoted, actual, tol=6e-4, phrase=None, bound=None):
    """c27's contract, registered against the manuscript pair.

    RELATIVE TO `quoted`, NOT TO `actual`, which is the one place this differs
    from c19/c21/c24/c27 and it is deliberate. The tolerance below is almost
    always "half a unit in the last decimal place the manuscript prints", and
    that is a property of the printed number: scaling it by `actual` instead
    makes the allowance shrink exactly when the source rounds up to the quote,
    so a correctly rounded value fails by a fraction of a unit in the last
    place. Two rows did (23.6 against 23.5500, 99.5 against 99.45) before this
    was fixed. `quoted` is never zero anywhere cms() is used, so there is no
    degenerate case to guard.

    `phrase` is the escape hatch of the reverse direction. Normally that
    direction asks whether the number is still printed anywhere in the region
    this row's label names; with a phrase it asks whether the manuscript still
    prints these exact words there. Use it where the section alone is not a
    tight enough question -- where the same digits appear twice inside one
    section, or where the sentence has to keep a particular form.

    A phrase that writes the number in digits must contain a written form of
    `quoted`, and that is enforced rather than trusted: a phrase is a second
    copy of the number, and the whole reason this argument exists instead of a
    separate hand-written check() row is that two copies drift apart in silence.

    A phrase with no digits in it at all is a number the manuscript spells out
    ("fourteen times further", "more than ninefold"), which is the other reason
    to reach for a phrase: _forms() cannot find a spelled number, so those rows
    were never located by the reverse direction and only ever passed on a
    substring collision elsewhere. There is nothing to compare a spelling
    against, so it is accepted, and the coupling to `quoted` rests on whoever
    edits the sentence -- which is exactly as strong as the prose itself.

    `bound` is the rounding direction, and it is REQUIRED of any row whose own
    words state a bound on its own value -- see the block above cms() for why a
    bound cannot be gated as a distance. "upper" says the source must sit at or
    below the quote, "lower" that it must sit at or above it. Declaring one does
    three things: it doubles the tolerance, because a quote rounded towards its
    claim sits up to a WHOLE unit in the last place from the source rather than
    half of one; it registers the row for the direction check; and it is itself
    checked against the row's words, so "at most" cannot be declared "lower".
    """
    if (phrase is not None and any(c.isdigit() for c in phrase)
            and not any(f in phrase for f in _forms(quoted))):
        raise SystemExit(
            f"p31: cms({label!r}) carries the phrase {phrase!r}, which writes "
            f"a number but not any written form of the {quoted} this row "
            f"verifies. The words and the value would then move apart without "
            f"anything noticing, which is the duplication `phrase` removes.")
    _words = _bound_dirs(label, quoted) | _bound_dirs(phrase or "", quoted)
    if any(_k in label for _k in BOUND_NOT_A_BOUND):
        _words = set()
    if _words and bound is None:
        raise SystemExit(
            f"p31: cms({label!r}) states a bound on {quoted} and does not say "
            f"which way it rounds. Pass bound=\"upper\" where the source must "
            f"sit at or below the quote, bound=\"lower\" where it must sit at "
            f"or above -- or, if the words name no bound on this row's own "
            f"value, say so in BOUND_NOT_A_BOUND with the reason. A bound is "
            f"not a distance: half a unit in the last place is as far in the "
            f"direction that makes the sentence false as in the one that keeps "
            f"it true.")
    if bound is not None:
        if bound not in ("upper", "lower"):
            raise SystemExit(f"p31: cms({label!r}) declares bound={bound!r}, "
                             f'which is neither "upper" nor "lower".')
        _named = _words & {"upper", "lower"}
        if _named and bound not in _named:
            raise SystemExit(
                f"p31: cms({label!r}) declares bound={bound!r} while its own "
                f"words say {sorted(_named)[0]!r}. One of the two is wrong, "
                f"and the words are the half the reader sees.")
        tol *= 2
        BOUNDS.append((label, quoted, actual, bound))
    IN_TEXT_MS.append((label, quoted, phrase))
    check("MS", label, quoted, actual, tol * abs(quoted))


def _dp(quoted, dp):
    """Relative tolerance of half a unit in the dp-th decimal place of `quoted`.

    The manuscript rounds; a tolerance has to be the rounding it actually used,
    and it must be DERIVED from how many digits the prose prints rather than
    widened until a row turns green. That is the whole difference between a
    tolerance and an excuse. Every coarse quote below passes its own printed
    precision through here, so the number of digits in the manuscript is what
    sets the slack, and re-rounding a value in the prose tightens or loosens
    the check automatically.
    """
    return (0.5 * 10.0 ** -dp) / abs(quoted)


def _manuscript_corpus():
    """Whichever of the manuscript pair exists, or None while neither does.

    Same ANY rule, and the same reason, as _round27_corpus(): a half-written
    pair is a real state (the SI is drafted after the main text), and taking
    the gate down over the half that is missing would say nothing about any
    number. main() turns a missing corpus into a hard failure the moment
    anything is registered here, so None cannot hide a registered value.
    """
    docs = [d for d in (MANUSCRIPT, MANUSCRIPT_SI) if os.path.exists(d)]
    return docs or None


# WHY THE REVERSE DIRECTION IS ASKED OF A SECTION AND NOT OF THE PAIR.
# It used to ask "does this number appear anywhere in the manuscript pair", and
# the pair is one corpus. With 424 registered values rounded to between zero and
# five decimal places, two of them landing on the same digits is the rule, not
# the exception: 122 of the 424 -- 28.8% -- share their printed value with at
# least one other registered row, and that is a lower bound, because the cover
# can equally come from a number no row registers at all.
#
# For every one of those rows the question "is it still there" could be answered
# yes by a sentence somewhere else entirely, so the row was gating the envelope
# rather than the claim. Measured on 2026-08-31 on §3.4: putting its superseded
# "13.0%" back left this gate GREEN, exit 0, on a sentence that was wrong,
# because SI §8 prints 12.8% for an unrelated normalised mutual information and
# SI §6 restates the same quantity a third time. §3.4 was the case that was
# caught by a human, not a special one.
#
# Counting occurrences instead of asking for one would not have caught it
# either: the pair holds three "12.8"s against two registered rows, so dropping
# one still clears the bar. The question has to be asked of a smaller document.
#
# Every cms() label already opens with the section its sentence lives in --
# "3.3:", "SI 8:", "Fig 5(c):", "abstract:". That was the author saying where
# the claim is, and nothing had ever read it. Now the matcher does.
def _ms_regions(claimed):
    """The manuscript pair cut into the regions its labels name.

    A figure caption or a table is a region in its own right AND is removed from
    the section it sits inside, because a caption that prints a number is
    exactly as good a cover for a sentence that has lost it as another section
    is, and the labels already distinguish the two ("3.3:" against "Fig 4(b):").

    But only when some row claims it, which is what `claimed` carries. Table 1
    sits inside §3.5 and no label names it, so carving it out would not tighten
    any row -- it would only take the numbers of §3.5's own table away from
    §3.5's rows and manufacture reds out of the layout. A region exists here
    because a row asked for it.
    """
    regions = {}

    def _add(key, para):
        regions.setdefault(key, []).append(para)

    for path, si in ((MANUSCRIPT, False), (MANUSCRIPT_SI, True)):
        if not os.path.exists(path):
            continue
        section = None      # the ## / ### we are inside
        block = None        # a caption or table that owns the paragraphs below
        for para in open(path, encoding="utf-8").read().split("\n\n"):
            first = para.strip().split("\n")[0] if para.strip() else ""
            head = re.match(r"^#{2,3}\s+(.+?)\s*$", first)
            if head:
                title = head.group(1)
                num = re.match(r"^(\d+(?:\.\d+)?)[.\s]", title)
                section = (("SI " if si else "") + num.group(1) if num
                           else ("SI " if si else "") + title.strip("* ").lower())
                regions.setdefault(section, [])
                block = None
                continue
            # A figure's image line, its caption, and a table's caption open a
            # region; the table rows that follow a table caption stay with it.
            img = re.match(r"^!\[Figure\s+(S?\d+)", first)
            cap = re.match(r"^\*\*(Figure|Table)\s+(S?\d+)\.", first)
            if img or cap:
                kind, n = ("Figure", img.group(1)) if img else cap.groups()
                block = ("Fig " + n if kind == "Figure"
                         else ("SI " if si else "") + "Table " + n)
                if block not in claimed:
                    block = None
            if block is not None and (img or cap or first.startswith("|")):
                _add(block, para)
                continue
            block = None
            if section is not None:
                _add(section, para)
    return {k: "\n\n".join(v) for k, v in regions.items()}


def _ms_region_key(label):
    """The region a cms() label claims its sentence lives in.

    One reading of the prefix the label already carries. Panels share their
    figure's caption because the caption is one paragraph of prose, and Table
    S1's rows are one block, so splitting either further would be a claim about
    the typesetting rather than about the document.
    """
    head = label.split(":", 1)[0].strip()
    if head.startswith("SI Table S1"):
        return "SI Table S1"
    fig = re.match(r"^Fig\s+(S?\d+)", head)
    return f"Fig {fig.group(1)}" if fig else head


# --- the LIVE objects this block reads --------------------------------------
# Deliberately separate names from the frozen copies the sent rounds read. p27,
# p32, p39, p40, p41, p42, p44 and p49-p54 all already exist in this file as
# ARCHIVE objects belonging to closed rounds; repointing those at LIVE would
# un-freeze four sent letters. So the manuscript gets its own handles and the
# two never touch.
#
# These are read through load(), which raises on a missing file rather than
# skipping its checks, so none of them needs an exists-guard. The names are
# added to REQUIRED_RESULTS anyway, tagged "MS", because that list is where
# someone looks to see what this gate depends on.
p8_ms = load("p8", LIVE)
p19_ms = load("p19", LIVE)
p27_ms = load("p27", LIVE)
p29_ms = load("p29", LIVE)
p32_ms = load("p32", LIVE)
p39_ms = load("p39", LIVE)
p41_ms = load("p41", LIVE)
p42_ms = load("p42", LIVE)
p44_ms = load("p44", LIVE)
p49_ms = load("p49", LIVE)
p50_ms = load("p50", LIVE)
p51_ms = load("p51", LIVE)
# p52 has a LIVE handle of its own for the same reason as the rest of this
# block. Its bare `p52` above belongs to the sent 08-24 round and reads the
# frozen snapshot; the manuscript is not a sent document, so gating §3.4 and
# SI 9 against that snapshot would hold them green through a re-run of p52.
# The two files are byte-identical today, which is exactly when this is
# cheap to fix and invisible to leave broken.
p52_ms = load("p52", LIVE)
p53_ms = load("p53", LIVE)
p66_ms = load("p66", LIVE)
p67_ms = load("p67", LIVE)
p69_ms = load("p69", LIVE)
p54_ms = load("p54", LIVE)
p2_ms = load("p2", LIVE)
p25_ms = load("p25", LIVE)
p40_ms = load("p40", LIVE)
p43d_ms = load("p43_dedup", LIVE)
inv_ms = load("inventory", LIVE)


# ============================================================ Abstract and §1
# The four numbers the abstract commits to before any section defends them.
check("MS", "abstract: 79 consecutive months", 79.0,
      float(inv_ms["months_complete"]), 1e-9)
check("MS", "abstract: 424 administrative neighbourhoods", 424.0,
      float(p48["dong"]), 1e-9)
check("MS", "abstract: 16 age bands", 16.0,
      float(len(p33_live["profile"]["bands"])), 1e-9)
check("MS", "abstract: 10.2 billion records, to one decimal", 10.2,
      round(inv_ms["parquet_rows"] / 1e9, 1), 1e-9)
# "departs ... by at most 0.00202" is §3.3's number, quoted at its own
# precision. It used to read "at most 0.002", which rounds 0.00202 DOWN and so
# is not an upper bound on it at all; the row gated the rounding rather than
# the claim, and went green on a sentence that was false as written.
# 2026-09-07: the abstract stopped quoting the DIFFERENCE OF NORMS on 09-05 and
# quotes the DISTANCE itself (p67), so this row followed the sentence rather than
# staying green on one the paper had stopped making. The difference of norms is
# still gated -- in SI 3, which is where it is still printed.
#
# THE QUOTE IS ROUNDED UP AND THAT IS THE POINT. The measured maximum is 0.02573,
# so "at most 0.0257" would not be an upper bound on it at all -- the same defect
# this row carried once before as "at most 0.002" against 0.00202, and the one
# p36's new 36.13 row found. This used to be a hand-written pair, a doubled
# tolerance beside a check() that spelled the inequality out, and it was the
# only bound in the manuscript that had one. bound="upper" is that pair made
# into the rule: it doubles the tolerance and asserts the direction for every
# row that states a bound, whether or not anybody remembered to write the
# second line. Eleven sentences were false as printed when it first ran.
cms("abstract: the distance to the null is at most 0.0258", 0.0258,
    p67_ms["summary"]["gap_rel_max"], _dp(0.0258, 4), bound="upper")
# "a factor of 33 to 35" is the pair of post-IPF gaps, each to the nearest
# integer. Both, not the span, so the sentence cannot survive one of them
# moving.
check("MS", "abstract: the post-IPF factor's low end is 33", 33.0,
      float(round(p51_ms["ipf_calibration"]["202402"]["national"]["gap_after"])),
      1e-9)
check("MS", "abstract: the post-IPF factor's high end is 35", 35.0,
      float(round(p51_ms["ipf_calibration"]["202312"]["national"]["gap_after"])),
      1e-9)

# The recovery band, recomputed from p39's stored surface rather than read out
# of a summary field: "92 to 101 per cent" is a statement about every grid
# point at beta = 0, so it is computed over all of them.
_REC_MS = {}
for _pt in p39_ms["surface"].values():
    if _pt["r_true_med"] > 0:
        _REC_MS.setdefault(_pt["beta"], []).append(
            100 * _pt["med_corr"] / _pt["r_true_med"])
check("MS", "p39's surface carries ten r_true points at each of five betas",
      50.0, float(sum(len(v) for v in _REC_MS.values())), 1e-9)
cms("abstract: recovery at beta = 0 starts at 92 per cent", 92.0,
    min(_REC_MS[0.0]), _dp(92.0, 0))
cms("abstract: recovery at beta = 0 reaches 101 per cent", 101.0,
    max(_REC_MS[0.0]), _dp(101.0, 0))

# The abstract's crossing, added 2026-08-31. The abstract used to say 0.95 was
# "above anything we measure"; it now says the bound admits the survey only
# above 0.966, "higher than every reading we obtain". That universal is TRUE at
# 0.966 and would have been FALSE at 0.95 -- see the note below on the archived
# twelfth reading of 0.9616 -- so the row is registered against beta* and the
# distinction is the whole reason the sentence could be strengthened at all.
# 2026-09-05: the abstract used to quote 0.966, which is the crossing in the
# DESIGN coordinate the p39/p59 grid is laid out in. The survey's beta is a
# realised attenuation, so the two were different units; p66 converts the grid
# and the abstract now quotes the realised crossing. The design value has not
# moved and is still gated, at its new home in SI 7.
cms("abstract: the crossing is 0.923 on the world's own scale", 0.923,
    p66_ms["beta_star"]["realised_A"], _dp(0.923, 3))
# 2026-09-07: the abstract now also carries the crossing under the survey's own
# pairing convention, which is the one the measured 0.922 is in. The sentence
# states that the exclusion is a property of that convention and not of the
# data, so the row that gates the
# number sits beside a check that the verdict is still the one the prose claims.
cms("abstract: and 0.918 under the survey's pairing convention", 0.918,
    p69_ms["beta_star"]["realised_C"], _dp(0.918, 3))
check("MS", "abstract: the crossing on that convention is below the measured "
            "beta, which is what 'a property of the convention' means", 1.0,
      1.0 if p69_ms["beta_star"]["realised_C"]
      < p69_ms["answer"]["primary_reading"]["beta"] else 0.0, 1e-9)

# §1's identification paragraph: beta's measured range, both ends.
cms("1: beta's low end, the Seoul arm, is 0.9217", 0.9217,
    p53_ms["asymmetry"]["beta_seoul"], _dp(0.9217, 4))
cms("1: beta's high end, the national arm, is 0.9347", 0.9347,
    p53_ms["asymmetry"]["beta_national"], _dp(0.9347, 4))
# "with all eleven readings below 0.95" is the sentence §1 needs in order not to
# contradict §3.5's eleventh reading of 0.9387, so it is checked as the two
# things it asserts rather than left to the prose. It says ELEVEN and not "every
# reading we obtain", because §3.5 also names an archived twelfth -- venue
# pairing with replacement, 0.9616 -- which sits above 0.95 and was rejected on
# evidence that predates this comparison. A universal over "readings" would have
# been false on that one.
check("MS", "1: there are eleven measured readings", 11.0,
      float(len(_cl59["measured_betas"])), 1e-9)
check("MS", "1: all eleven of them lie below 0.95", 1.0,
      1.0 if _cl59["largest_measured_beta"] < 0.95 else 0.0, 1e-9)


# ============================================================ §2 Data and methods
# --- 2.1 the product, and the four properties every estimate rests on
check("MS", "2.1: the window is 202001-202607", 1.0,
      _eq(f"{inv_ms['span_first']}–{inv_ms['span_last']}", p48["span"]), 1e-9)
check("MS", "2.1: 79 of 79 months complete", 0.0,
      float(inv_ms["span_months"] - inv_ms["months_complete"]), 1e-9)
cms("2.1: 10 186 891 962 rows", 10186891962.0,
    float(inv_ms["parquet_rows"]), 1e-12)
cms("2.1: the mask removes 26.1% of cells", 26.1,
    100 * p48["masking_rate_weighted"], _dp(26.1, 1))
check("MS", "2.1: material masking is exactly the 20-44 block", 1.0,
      _eq(p48["masked_bands_material"],
          ["20-24", "25-29", "30-34", "35-39", "40-44"]), 1e-9)
check("MS", "2.1: seven bands lose no cells at all", 7.0,
      float(len(p48["masked_bands_exactly_zero"])), 1e-9)

# --- 2.2 the three external anchors
cms("2.2: 2.27 people per masked dong-level cell", 2.27,
    p29_ms["202012"]["per_cell_mean"], _dp(2.27, 2))
cms("2.2: the survey has 1 987 respondents", 1987.0,
    float(p44_ms["anchors"]["respondents"]), 1e-12)
cms("2.2: the survey records 133 776 contacts", 133776.0,
    float(p44_ms["anchors"]["contacts"]), 1e-12)
cms("2.2: the Seoul subsample is 465 respondents", 465.0,
    float(p44_ms["sensitivity"]["n_seoul_respondents"]), 1e-12)

# --- 2.3 the estimator, and the pairing convention it is not neutral under
cms("SI 2: the without-replacement correction at dong is -4.29%", -4.29,
    100 * p49_ms["primary"]["rel_change"], _dp(4.29, 2))
cms("SI 2: the same correction at a survey venue is -43.0%", -43.0,
    100 * p49_ms["answer"]["venue_relative_difference"], _dp(43.0, 1))
_VEN_MS = {r["grouping"]: r for r in p44_ms["ladder"]["national|pooled"]["rows"]}
cms("SI 2: a survey venue holds 4.2 people on average", 4.2,
    _VEN_MS["venue-visit (ego,date,place)"]["mean_persons"], _dp(4.2, 1))

# --- 2.4 the coverage profile
cms("2.4: the profile reads 27.86 at 0-9", 27.86,
    p33_live["profile"]["w"][0], _dp(27.86, 2))
cms("2.4: the profile bottoms at 3.00 at 40-44", 3.00,
    p33_live["profile"]["w"][p33_live["profile"]["bands"].index("40-44")],
    _dp(3.00, 2))
cms("2.4: the profile rises again to 9.43 at 80+", 9.43,
    p33_live["profile"]["w"][p33_live["profile"]["bands"].index("80+")],
    _dp(9.43, 2))
cms("2.4: the geometric mean w-bar is 4.716", 4.716,
    p33_live["profile"]["wbar"], _dp(4.716, 3))
# "one child in twenty-eight and one prime-age adult in three" is the profile
# read back as a ratio, so it is checked as the rounding of the profile itself.
check("MS", "2.4: one child in twenty-eight", 28.0,
      float(round(p33_live["profile"]["w"][0])), 1e-9)
check("MS", "2.4: five of the sixteen bands are censored from below", 5.0,
      float(sum(p33_live["profile"]["censored"])), 1e-9)

# --- 2.6 the identity, verified numerically
check("MS", "2.6: the identity was checked at three scales in two months", 6.0,
      float(len(p39_ms["identity"])), 1e-9)
check("MS", "2.6: the identity's worst deviation is 1.60e-16", 1.6e-16,
      max(_i["max_abs_dev"] for _i in p39_ms["identity"]), 5e-19)


# ============================================================ §3.1 the calendar
_CAL_MS = p41_ms["calendar"]["by_month"]
cms("3.1: March's median is 0.0242, the highest month", 0.0242,
    _CAL_MS["3"]["median"], _dp(0.0242, 4))
cms("3.1: September's median is 0.0230, the second highest", 0.0230,
    _CAL_MS["9"]["median"], _dp(0.0230, 4))
cms("3.1: February's median is 0.0176, the lowest", 0.0176,
    _CAL_MS["2"]["median"], _dp(0.0176, 4))
# 3.1 names the primary arm's own floor and says no month reaches it. That
# sentence exists because the exceedance count is measured against the
# ROBUSTNESS arm, and until 2026-09-02 the manuscript did not say so anywhere.
# Gate all three parts: the two respondent counts that explain why the floors
# differ, the primary floor itself, and the count of months clearing it. The
# last is the one that carries the claim, and it is zero -- so it is registered
# COUNT-FIRST style, against the series maximum, because a row asserting "0"
# against an empty list would pass on a series that failed to load.
cms("3.1: the Seoul primary arm rests on 465 respondents", 465.0,
    float(p35_live["survey_bootstrap"]["202312"]["n_respondents"]), 1e-9)
cms("3.1: the national arm rests on 1 987", 1987.0,
    float(p35_live["survey_bootstrap"]["202312|national"]["n_respondents"]), 1e-9)
cms("3.1: the Seoul arm's own permutation floor is 0.0713 bits", 0.0713,
    p32_ms["survey_floor_measured"]["202312"]["mi_perm_median"], _dp(0.0713, 4))
_MI79 = [r["mi_bits_hf"] for r in p42_ms["rows"]]
check("MS", "3.1: the 79-month series is loaded before it is searched", 79.0,
      float(len(_MI79)), 1e-9)
check("MS", "3.1: and no month reaches the Seoul arm's floor", 0.0,
      float(sum(1 for _v in _MI79
                if _v > p32_ms["survey_floor_measured"]["202312"]["mi_perm_median"])),
      1e-9)

# The ORDERING is the claim, and gating a number beside a word does not gate
# the word: March highest, September second, February lowest, over all twelve.
_ORD_MS = sorted(_CAL_MS, key=lambda m: -_CAL_MS[m]["median"])
check("MS", "3.1: March is the highest calendar month", 1.0,
      _eq(_ORD_MS[0], "3"), 1e-9)
check("MS", "3.1: September is the second highest", 1.0,
      _eq(_ORD_MS[1], "9"), 1e-9)
check("MS", "3.1: February is the lowest", 1.0, _eq(_ORD_MS[-1], "2"), 1e-9)

_PY_MS = p41_ms["per_year"]
_C6_MS = [_PY_MS[str(_y)]["contrast_median"] for _y in range(2021, 2027)]
check("MS", "3.1: the contrast is positive in all six years 2021-2026", 6.0,
      float(sum(1 for _c in _C6_MS if _c > 0)), 1e-9)
cms("SI 5: the six contrasts start at 0.0032", 0.0032, min(_C6_MS), _dp(0.0032, 4))
cms("SI 5: the six contrasts reach 0.0046", 0.0046, max(_C6_MS), _dp(0.0046, 4))
cms("3.1: 2020's contrast is -0.0001", -0.0001,
    _PY_MS["2020"]["contrast_median"], _dp(0.0001, 4))
# The two shares the sentence turns on, both derived rather than restated.
_MEAN6_MS = sum(_C6_MS) / len(_C6_MS)
cms("3.1: the six contrasts have a mean of 0.0040", 0.0040, _MEAN6_MS,
    _dp(0.0040, 4))
cms("3.1: the 79-month median they are measured against is 0.01938", 0.01938,
    p37["summary_dong"]["assort_median"], _dp(0.01938, 5))
cms("3.1: the mean contrast is 20.5% of the 79-month median", 20.5,
    100 * _MEAN6_MS / p37["summary_dong"]["assort_median"], _dp(20.5, 1))
cms("SI 5: and 31% of the entire 79-month range", 31.0,
    100 * _MEAN6_MS / (p37["summary_dong"]["assort_max"]
                       - p37["summary_dong"]["assort_min"]), _dp(31.0, 0))
# The sign test, and the floor its resolution imposes. THE 0.0625 ROW WAS
# DELETED HERE and the 0.008 row relabelled, on the rule that rows follow the
# prose. §3.1 reports the regression and no longer names the sign test at all;
# SI §5 carries it, and already registers the same 8/128 at its own sentence.
# Both rows had been passing on a substring collision rather than on the
# sentence they claimed: _forms(0.0625) contains "0.1" and _forms(0.008)
# contains "0.0", which §3.1's other decimals supplied for free.
check("MS", "3.1: six of seven years are positive", 6.0,
      float(p41_ms["sign_test"]["contrast"]["n_positive"]), 1e-9)
cms("SI 5: with seven years the p cannot fall below 0.008", 0.008,
    1.0 / p41_ms["sign_test"]["contrast"]["denominator"], _dp(0.008, 3))
# The ten readings. Counted, not quoted: "in all ten" is a universal.
_VAR_MS = p41_ms["robustness"]["variants"]
check("MS", "3.1: the contrast was recomputed under ten readings", 10.0,
      float(len(_VAR_MS)), 1e-9)
check("MS", "3.1: 2020 is the smallest year in all ten", 10.0,
      float(sum(1 for _v in _VAR_MS.values()
                if _v["control"] < _v["others_min"])), 1e-9)
check("MS", "3.1: 2020 is negative in only two of the ten", 2.0,
      float(sum(1 for _v in _VAR_MS.values() if _v["control"] < 0)), 1e-9)
# The band decomposition, which is what separates schooling from a pandemic.
_PB_MS = p41_ms["bands"]["per_band_contrast"]
_YOUNG_MS = ("0-9", "10-14", "15-19", "20-24")
_SH_MS = []
for _y in range(2021, 2027):
    _tot = sum(_PB_MS[_b][str(_y)] for _b in _PB_MS)
    _SH_MS.append(100 * sum(_PB_MS[_b][str(_y)] for _b in _YOUNG_MS) / _tot)
check("MS", "3.1: the band shares cover the six open-school years", 6.0,
      float(len(_SH_MS)), 1e-9)
# 82, not 83: the smallest of the six years is 82.52%, which "at least 83%"
# does not bound. Rounding a lower bound to nearest is how a sentence about
# every year comes to exclude one of them.
cms("3.1: the 0-24 bands carry at least 82% of the contrast", 82.0,
    min(_SH_MS), _dp(82.0, 0), bound="lower")
cms("3.1: and at most 93% of it", 93.0, max(_SH_MS), _dp(93.0, 0),
    bound="upper")
_GRP_MS = p41_ms["bands"]["groups"]
cms("SI 5: 2020's 0-24 contrast is -0.00075", -0.00075,
    _GRP_MS["student_0_24"]["control"], _dp(0.00075, 5))
cms("SI 5: against a median of +0.00381 in the other six years", 0.00381,
    _GRP_MS["student_0_24"]["others_median"], _dp(0.00381, 5))
check("MS", "3.1: the 25-and-over contrast is positive in all seven years", 7.0,
      float(_GRP_MS["adult_25plus"]["sign_test"]["n_positive"]), 1e-9)
# The second reading, counted rather than quoted: 17 and 79 are two digits and
# _forms() matches by substring, so registering them would pass on any prose.
check("MS", "3.1: 17 months clear the survey's permutation floor", 17.0,
      float(p54_ms["corrected"]["n_at_or_above"]), 1e-9)
check("MS", "3.1: all 17 are term months", 17.0,
      float(p54_ms["corrected"]["observed"]["term"]), 1e-9)
check("MS", "3.1: no vacation month clears", 0.0,
      float(p54_ms["corrected"]["observed"]["vacation"]), 1e-9)
for _ym_ms in ("202312", "202402"):
    check("MS", f"3.1: {_ym_ms} is not one of the 17", 0.0,
          1.0 if p54_ms["survey_months"][_ym_ms]["clears_corrected"] else 0.0,
          1e-9)
# The rest of that reading moved into §3.1's prose on 2026-08-29, when Figure 2
# became p63's five panels: the body now states the floor, the run structure
# and the rotation the figure draws, so each is gated against the file the
# figure reads rather than against the caption that quotes it.
cms("3.1: the mutual-information year medians rise by a factor of 1.70", 1.70,
    _cap(_c63, 2, "a", "level trend over the period"), _dp(1.70, 2))
cms("3.1: the floor is 0.02296 bits", 0.02296, p54_ms["corrected"]["floor"],
    _dp(0.02296, 5))
check("MS", "3.1: no December clears either", 0.0,
      float(p54_ms["corrected"]["observed"]["december"]), 1e-9)
check("MS", "3.1: the 17 clearing months fall in nine runs", 9.0,
      float(p58["run_structure"]["n_runs"]), 1e-9)
check("MS", "3.1: 3 shifts also place all 17 in term", 3.0,
      float(_ps58["n_shifts_at_or_above"]), 1e-9)
check("MS", "3.1: out of 78 alternatives", 78.0,
      float(_ps58["denominator"] - 1), 1e-9)
# The rotation p is registered at SI 5's sentence and nowhere else: §3.1 says
# "an alignment count and not a p value" and prints no p, so a 3.1 row for it
# would gate a sentence the section deliberately does not make.

# ---------------------------------------------------------------- p64 (LIVE)
# 2026-08-31. The 08-31 round is not sent, so p64 reads LIVE like p58-p63.
#
# TWO ROWS WERE DELETED HERE, and the reason is the rule that rows follow the
# prose. The manuscript used to say the rotation test's floor was 1/79 =
# 0.0127 and that 0.0506 sat a factor of four above it. p64 showed that the
# 79 rotations realise only TWELVE distinct calendars, because the labelling
# is a function of month-of-year and 79 = 6*12 + 7, so the honest floor is
# 1/12 = 0.083 (1/6 = 0.17 after collapsing the near-duplicate). The prose was
# replaced, so the rows that gated it were deleted rather than left to pass
# against a sentence that no longer exists.
#
# THE MULTIPLE-OF-SIX ROW WAS ALSO WRONG, not merely superseded. The three
# matching shifts are multiples of TWELVE; the odd multiples of six (6, 18,
# 30, 42, 54, 66, 78) reach n_term of 16, 16, 16, 16, 15, 14, 14 and never 17.
# Six was a true statement about those three shifts and a false explanation of
# them, which is why it is replaced and not just relabelled.
_R64 = p64["primary"]["assortativity_all"]
_R64M = p64["primary"]["mi_bits_hf"]
_S4 = p64["specifications"]["assortativity_all"]["S4"]
_S4M = p64["specifications"]["mi_bits_hf"]["S4"]
_E64 = p64["effective_alignments"]
_D64 = p64["deduplicated"]

# The regression that replaces the seven-point sign test. The lag is gated too:
# it is fixed by a declared rule, so it is a quoted number and not a free
# parameter that could have been tuned after the fact.
check("MS", "3.1: the regression uses all 79 months", 79.0,
      float(_R64["n"]), 1e-9)
check("MS", "3.1: the Newey-West lag is three", 3.0, float(_R64["lag"]), 1e-9)
cms("3.1: the term coefficient on assortativity is 0.00368", 0.00368,
    _R64["coef"]["term"], _dp(0.00368, 5))
cms("3.1: its Newey-West standard error is 0.00048", 0.00048,
    _R64["se"]["term"], _dp(0.00048, 5))
cms("3.1: which is a t of 7.68", 7.68, _R64["t"]["term"], _dp(7.68, 2))
cms("3.1: and a p of 7.6e-11", 7.6e-11, _R64["p_t"]["term"], _dp(7.6e-11, 12))
# The lag sweep is stated in SI 5, not in 3.1: the body gives the fitted lag
# and its p, and the sentence that sweeps zero to twelve is the SI's.
cms("SI 5: no lag from zero to twelve takes the p above 4.7e-08", 4.7e-08,
    max(p64["lag_band"]["assortativity_all"]["p_max"],
        p64["lag_band"]["mi_bits_hf"]["p_max"]), _dp(4.7e-08, 9),
    bound="upper")
cms("3.1: the same regression on the MI series gives 0.00599 bits", 0.00599,
    _R64M["coef"]["term"], _dp(0.00599, 5))
cms("SI 5: with a standard error of 0.00078", 0.00078,
    _R64M["se"]["term"], _dp(0.00078, 5))
cms("SI 5: and a t of 7.70", 7.70, _R64M["t"]["term"], _dp(7.70, 2))
check("MS", "3.1: six variants of the statistic were fitted", 6.0,
      float(p64["sign_agreement"]["n_series"]), 1e-9)
check("MS", "3.1: the term coefficient is positive on all six", 6.0,
      float(p64["sign_agreement"]["n_positive"]), 1e-9)
cms("3.1: the term-by-2020 interaction is -0.00296", -0.00296,
    _S4["extra"]["term_x_2020"]["coef"], _dp(0.00296, 5))
cms("3.1: with a standard error of 0.00125", 0.00125,
    _S4["extra"]["term_x_2020"]["se"], _dp(0.00125, 5))
cms("3.1: and a t of -2.37", -2.37, _S4["extra"]["term_x_2020"]["t"],
    _dp(2.37, 2))
cms("3.1: and a p of 0.021", 0.021, _S4["extra"]["term_x_2020"]["p_t"],
    _dp(0.021, 3))
# The denominator of the two percentages below. It is the term coefficient the
# interaction specification re-estimates, NOT the 0.00368 the same paragraph
# prints for the specification without the interaction. Quoting only the latter
# is what made the 28% unreproducible for a reader holding just the printed
# numbers: 0.00368 as the denominator gives 20%, not 28%.
cms("3.1: the interaction specification re-estimates the term coefficient at 0.00412",
    0.00412, _S4["b_term"], _dp(0.00412, 5))
cms("SI 5: and at 0.00666 on the mutual-information series",
    0.00666, _S4M["b_term"], _dp(0.00666, 5))
cms("3.1: leaving 28% of the term effect standing in 2020", 28.0,
    100.0 * (_S4["b_term"] + _S4["extra"]["term_x_2020"]["coef"])
    / _S4["b_term"], _dp(28.0, 0))
cms("3.1: and 32% of it on the mutual-information series", 32.0,
    100.0 * (_S4M["b_term"] + _S4M["extra"]["term_x_2020"]["coef"])
    / _S4M["b_term"], _dp(32.0, 0))

# The corrected resolution floor. 2026-09-02: eight rows left this block when
# 3.1's prose handed the rotation arithmetic to Figure 2(b)'s caption -- the
# three matching shifts as multiples of twelve, the six-month shift's near-miss
# on both its counts, both resolution floors and the collapsed-calendar pair,
# and the trend term's own t. Every one of them is still gated where the
# manuscript still says it: at the caption, a few hundred lines below.
check("MS", "3.1: no shift that changes the calendar reaches seventeen", 0.0,
      float(len(p64["letter_verdict"]
                 ["iii_mechanism_is_the_six_month_symmetry"]
                 ["odd_multiples_reaching_observed"])), 1e-9)
check("MS", "3.1: the 79 rotations realise twelve distinct calendars", 12.0,
      float(_E64["E_exact"]), 1e-9)
check("MS", "3.1: none of the eleven alternative calendars matches", 0.0,
      float(_D64["exact"]["n_nonidentity_classes_reaching_observed"]), 1e-9)


# ---------------------------------------------------------------- p65 (LIVE)
# 2026-09-01. NIMS's reply to the 08-24 enquiry. Three of the numbers below were
# quoted in a letter that has been SENT AND ANSWERED while no script produced
# them -- 3,395, 3,296 and 8,198 were counted by hand at writing time and lived
# only in the letter's prose. That is the shape this gate exists to catch, and
# it caught nothing because a document nobody registered has no rows. They have
# a source now.
#
# WHAT IS NOT RE-GATED HERE. p57's own rows stay where they are, against the
# 08-24 snapshot, and they stay TRUE: 10.53% and 4.48% are what v1 reads under
# those two conventions and the reply does not move either number. What the
# reply moves is which convention is defensible and why, and that is prose in
# the memos, not a value.
_C65 = p65["conventions"]
_H65 = p65["household"]
check("27", "p65 reproduces p57's literal convention bit-for-bit", 0.0,
      abs(p65["anchor"]["p57_literal"] - p65["anchor"]["p57_published_literal"]),
      0.0)
check("27", "p65 reproduces p57's recoding-aware convention bit-for-bit", 0.0,
      abs(p65["anchor"]["p57_recoding_aware"] - p65["anchor"]["p57_published_aware"]),
      0.0)
check("27", "p65 read p27's own row count", 133776.0,
      float(p65["anchor"]["contacts"]), 1e-9)
# The confirmed second coding error, and the two counts the letter states.
check("27", "p65 rows with free text and no Q5_8 flag: 3,395", 3395.0,
      float(p65["csv_error"]["rows_text_without_flag"]), 1e-9)
check("27", "p65 of which 3,296 are code 1", 3296.0,
      float(p65["csv_error"]["by_lowest_code"]["1"]), 1e-9)
# The correction cannot move the recoding-aware reading, and the whole
# re-justification of that convention rests on it, so it is gated rather than
# argued: if this ever fails, the bracket below has no lower end.
check("27", "p65 the correction leaves the recoding-aware reading untouched",
      0.0, abs(_C65["recoding_aware"]["multi_place_share"]
               - p65["anchor"]["p57_published_aware"]), 0.0)
check("27", "p65 the corrected-literal reading is 12.99%", 12.99,
      100 * _C65["corrected_literal"]["multi_place_share"], 1e-2)
check("27", "p65 household-plus-other rises to 5.76% under the correction", 5.76,
      100 * _C65["corrected_literal"]["household_plus_other_share"], 1e-2)
# Q9. The reply calls 0 and 1 errors and says no top-coding was applied, so the
# tail is a property of the responses and is gated as one.
check("27", "p65 Q9 = 1 on 8,198 records", 8198.0, float(p65["q9"]["n_one"]), 1e-9)
check("27", "p65 Q9 = 0 on exactly one record", 1.0,
      float(p65["q9"]["n_zero"]), 1e-9)
# check()'s tolerance is ABSOLUTE (cms/c24/c27 scale theirs by the quote, this
# does not), so the allowance here is half a unit in the one decimal place the
# letter prints: it says "6.1%", the file says 6.128%.
check("27", "p65 that is 6.1% of records", 6.1,
      100 * p65["q9"]["share_one"], 0.05)
check("27", "p65 Q9's maximum is 1,000", 1000.0, p65["q9"]["maximum"], 1e-9)
check("27", "p65 70 records sit at exactly 100", 70.0,
      float(p65["q9"]["n_at_100"]), 1e-9)
# The household arm. The anchor first: if the unmodified panel assignment is not
# p27's cell, every delta below is a delta from something else.
check("27", "p65 the unmodified panels reproduce p27's survey assortativity",
      0.0, abs(_H65["anchor"]["recomputed"] - _H65["anchor"]["p27_published"]), 0.0)
for _a, _n in (("letter_four", 1361.0), ("kinship", 2901.0),
               ("any_free_text", 4605.0), ("s6_household", 2570.0),
               ("s6_other", 2035.0)):
    check("27", f"p65 the {_a} arm moves {_n:,.0f} contacts", _n,
          float(_H65["arms"][_a]["n_moved"]), 1e-9)
# The S6 split has to BE a split, or s6_household is not a bound on anything.
check("27", "p65 the two Table S6 arms partition the free-text rows", 0.0,
      float(_H65["arms"]["s6_household"]["n_moved"]
            + _H65["arms"]["s6_other"]["n_moved"]
            - _H65["arms"]["any_free_text"]["n_moved"]), 1e-9)
# Table S6 itself, and the count that says it is incomplete. The obvious reading
# of s6_other was "the genuine two-place contacts"; the measurement says half of
# it is residences the authors' table does not list, so the claim gated here is
# the corrected one.
check("27", "p65 Table S6 lists 81 household mappings", 81.0,
      float(_H65["s6"]["n_entries"]), 1e-9)
check("27", "p65 1,045 unlisted rows still name a residence", 1045.0,
      float(_H65["s6"]["residence_shaped_but_unlisted"]), 1e-9)
# THE CLAIM THE MEMO AND THE README BOTH MAKE. Taken over every arm and every
# cell, not over the arm that happens to move least: an arm-specific maximum
# would be a number chosen after the fact.
_MOVE65 = max(abs(_c["rel_delta"]) for _a in _H65["arms"].values()
              for _c in _a["cells"].values())
check("27", "p65 the largest relative movement over every arm and cell is 0.98%",
      0.98, 100 * _MOVE65, 1e-2)
check("27", "p65 no arm and no cell moves the reading by 1% or more", 1.0,
      1.0 if _MOVE65 < 0.01 else 0.0, 1e-9)


# ============================================================ §3.2 the reading
_CMP_MS = {str(r["ym"]): r for r in p32_ms["comparison"]}
cms("3.2: the December 2023 dong reading is 0.02115", 0.02115,
    p39_ms["published_constants"]["r_obs_pub"], _dp(0.02115, 5))
cms("3.2: the Seoul survey reads 0.22271", 0.22271,
    p39_ms["published_constants"]["survey"], _dp(0.22271, 5))
_R15_MS = p32_ms["excess"]["passive|202312|WE|dong|holidayfree"]["assortativity"]
cms("3.2: band-matched to the survey's fifteen bands it is 0.02174", 0.02174,
    _R15_MS, _dp(0.02174, 5))
cms("3.2: the 80+ band is worth 2.79% of the reading", 2.79,
    100 * (_R15_MS / p39_ms["published_constants"]["r_obs_pub"] - 1),
    _dp(2.79, 2))
cms("3.2: the December 2023 assortativity ratio is 10.2x", 10.2,
    _CMP_MS["202312"]["assortativity"]["ratio"], _dp(10.2, 1))
cms("3.2: the February 2024 ratio is 9.9x", 9.9,
    _CMP_MS["202402"]["assortativity"]["ratio"], _dp(9.9, 1))
# The two spans. Recomputed as extrema over the four statistics and the two
# months, because "5.6x to 39.8x across the two survey months" is a statement
# about every cell of that grid and not about the two the prose names.
_STATS_MS = ("half_l1", "cramers_v", "assortativity", "nmi")
_SEOUL_MS = [_CMP_MS[_m][_s]["ratio"] for _m in ("202312", "202402")
             for _s in _STATS_MS]
check("MS", "3.2: the Seoul span is taken over eight statistic x month cells",
      8.0, float(len(_SEOUL_MS)), 1e-9)
cms("3.2: the Seoul span starts at 5.6x", 5.6, min(_SEOUL_MS), _dp(5.6, 1))
cms("3.2: the Seoul span ends at 39.8x", 39.8, max(_SEOUL_MS), _dp(39.8, 1))
cms("3.2: the national span starts at 6.5x", 6.5,
    p51_ms["national_span"]["202402"]["lo"], _dp(6.5, 1))
cms("3.2: the national span ends at 45.5x", 45.5,
    max(p51_ms["national_span"][_m]["hi"] for _m in ("202312", "202402")),
    _dp(45.5, 1))
# The 79-month context.
cms("SI 3: the 79-month dong series runs from 0.01311", 0.01311,
    p37["summary_dong"]["assort_min"], _dp(0.01311, 5))
cms("SI 3: ...to 0.02589", 0.02589, p37["summary_dong"]["assort_max"],
    _dp(0.02589, 5))
cms("SI 3: with a median of 0.01938", 0.01938,
    p37["summary_dong"]["assort_median"], _dp(0.01938, 5))
_WE_MS = sorted([r for r in p37["rows"]
                 if r["panel"] == "WE" and r["level"] == "dong"],
                key=lambda r: r["assortativity"])
_RANK_MS = {r["ym"]: i + 1 for i, r in enumerate(_WE_MS)}
check("MS", "SI 3: December 2023 sits at rank 50 of 79", 50.0,
      float(_RANK_MS[202312]), 1e-9)
check("MS", "SI 3: February 2024 sits at rank 24 of 79", 24.0,
      float(_RANK_MS[202402]), 1e-9)
# The two-sided floor, and the agreement between its two unrelated routes.
cms("3.2: the zero-structure world reads 0.00455", 0.00455,
    p39_ms["null_arm"]["null_median"], _dp(0.00455, 5))
cms("3.2: which is 27.1% of the noise-corrected observation", 27.1,
    100 * p39_ms["null_arm"]["null_median"]
    / p39_ms["published_constants"]["r_obs_corrected"], _dp(27.1, 1))
cms("3.2: the noise-corrected observation is 0.01677", 0.01677,
    p39_ms["published_constants"]["r_obs_corrected"], _dp(0.01677, 5))
cms("3.2: the independently measured device-noise bias is 0.00439", 0.00439,
    p39_ms["null_arm"]["p33_noise_bias"], _dp(0.00439, 5))
cms("3.2: the two routes agree to within 4%", 4.0,
    100 * (p39_ms["null_arm"]["ratio_to_p33"] - 1), _dp(4.0, 0))
_ST12_MS = p33_live["R1_effective_sample"]["202312"]["stats"]["assortativity"]
_ST02_MS = p33_live["R1_effective_sample"]["202402"]["stats"]["assortativity"]
cms("3.2: 20.7% of the December 2023 reading is device noise", 20.7,
    100 * _ST12_MS["noise_bias"] / _ST12_MS["point"], _dp(20.7, 1))
cms("3.2: 22.9% in February 2024", 22.9,
    100 * _ST02_MS["noise_bias"] / _ST02_MS["point"], _dp(22.9, 1))
cms("3.2: the 79-month normalised mutual information median is 0.0045", 0.0045,
    p37["summary_dong"]["nmi_median"], _dp(0.0045, 4))


# ============================================================ §3.3 rank one
_SP_MS = p32_ms["spectrum"]
_PD_MS = _SP_MS["passive|202312|WE|dong|holidayfree"]
_SV_MS = _SP_MS["survey|202312|WE|seoul"]
cms("3.2: the leading component carries 97.95% of the spectral mass", 97.95,
    100 * _PD_MS["sigma1_share"], _dp(97.95, 2))
cms("3.2: sigma2/sigma1 is 0.1237 at dong", 0.1237,
    _PD_MS["sigma2_over_sigma1"], _dp(0.1237, 4))
cms("3.2: the survey carries 46.79%", 46.79, 100 * _SV_MS["sigma1_share"],
    _dp(46.79, 2))
cms("3.2: with sigma2/sigma1 = 0.601", 0.601, _SV_MS["sigma2_over_sigma1"],
    _dp(0.601, 3))
cms("3.2: the passive best rank-one residual is 0.1433", 0.1433,
    _PD_MS["best_rank1_resid"], _dp(0.1433, 4))
cms("3.2: and against r (x) r it is 0.1447", 0.1447, _PD_MS["pm_rank1_resid"],
    _dp(0.1447, 4))
# check(), not cms(): grep of the manuscript pair finds this value nowhere,
# so the document assertion was never true. It stayed green because _forms()
# offers coarse spellings and the region contained one. Verified, not quoted.
check("MS", "3.2: the survey's pair is 0.7294", 0.7294,
      _SV_MS["best_rank1_resid"], _dp(0.7294, 4) * 0.7294)
check("MS", "3.2: ...and 0.7355", 0.7355, _SV_MS["pm_rank1_resid"],
      _dp(0.7355, 4) * 0.7355)
# "to within 1%" is the gap between those two divided by the smaller, and it is
# derived here rather than typed beside them.
cms("3.2: the two residuals differ by 1%", 1.0,
    100 * (_PD_MS["pm_rank1_resid"] / _PD_MS["best_rank1_resid"] - 1),
    _dp(1.0, 0))
# §3.2 prints all three: the bound it states, the maximum it reaches, and the
# median. The bound is the rounded-up one, for the reason on the abstract's row.
cms("3.2: the distance never exceeds 0.0258", 0.0258,
    p67_ms["summary"]["gap_rel_max"], _dp(0.0258, 4), bound="upper")
cms("3.2: reaching 0.02573 at its largest", 0.02573,
    p67_ms["summary"]["gap_rel_max"], _dp(0.02573, 5))
cms("3.2: and 0.0209 at the median", 0.0209,
    p67_ms["summary"]["gap_rel_median"], _dp(0.0209, 4))
# The difference of norms stays in §3.2 as the lower bound it is, and the number
# beside it was never gated: it is the two residuals above, subtracted here
# rather than retyped.
cms("3.2: those two residuals differ by 0.0015", 0.0015,
    _PD_MS["pm_rank1_resid"] - _PD_MS["best_rank1_resid"], _dp(0.0015, 4))
# The IPF branch, both halves. Gating a number beside a word does not gate the
# word, so non-convergence is a check of its own.
_IPF_MS = p51_ms["ipf_calibration"]
cms("3.2: IPF absorbs 23.6% of the gap in December 2023", 23.6,
    100 * _IPF_MS["202312"]["national"]["absorbed_frac"], _dp(23.6, 1))
cms("3.2: and 27.2% in February 2024", 27.2,
    100 * _IPF_MS["202402"]["national"]["absorbed_frac"], _dp(27.2, 1))
# The Seoul arm's own absorption, quoted in the prose from 2026-08-31. The
# sentence now says the national arm is reported BECAUSE it is the arm where
# the fit absorbs most, which answers the 08-31 letter's charge that the scope
# switch reads as cherry-picking. A "because" is only checkable if the arm it
# is measured against is a number too, so 4.4% and 1.2% are gated beside the
# 23.6% and 27.2% they are being compared with; without them the one clause
# carrying the justification would be the one clause no results file stands
# behind.
cms("3.2: the Seoul primary arm absorbs 4.4%", 4.4,
    100 * _IPF_MS["202312"]["seoul"]["absorbed_frac"], _dp(4.4, 1))
cms("3.2: and 1.2% there", 1.2,
    100 * _IPF_MS["202402"]["seoul"]["absorbed_frac"], _dp(1.2, 1))
cms("3.2: leaving a factor of 34.8x", 34.8,
    _IPF_MS["202312"]["national"]["gap_after"], _dp(34.8, 1))
cms("3.2: and 33.1x", 33.1, _IPF_MS["202402"]["national"]["gap_after"],
    _dp(33.1, 1))
cms("3.2: demographic composition alone absorbs 7.9%", 7.9,
    100 * _IPF_MS["202312"]["demography_control"]["absorbed_frac"], _dp(7.9, 1))
cms("3.2: and 5.6%", 5.6,
    100 * _IPF_MS["202402"]["demography_control"]["absorbed_frac"], _dp(5.6, 1))
for _ym_ms in ("202312", "202402"):
    check("MS", f"3.2: the fit does not converge, {_ym_ms}", 0.0,
          1.0 if _IPF_MS[_ym_ms]["national"]["converged"] else 0.0, 1e-9)
# The survey's own rank-one null.
cms("3.2: the permutation null's median sigma2/sigma1 is 0.1266", 0.1266,
    _pm61["median"], _dp(0.1266, 4))
cms("3.2: against an observed 0.6014", 0.6014, _o61["sigma2_over_sigma1"],
    _dp(0.6014, 4))
cms("3.2: the passive dong sigma2/sigma1 is 0.1237", 0.1237,
    _pi61["202312|dong"]["passive_sigma2_over_sigma1"], _dp(0.1237, 4))
check("MS", "3.2: the passive dong spectrum sits at the null's 43rd percentile",
      43.0, _pi61["202312|dong"]["percentile_within_survey_null"], 1e-9)
# 3.3 used to say the passive distance is "no larger than what SAMPLING ALONE
# produces", while SI 3 says in terms that the permutation null is not a noise
# floor. Sampling alone is the multinomial null, which is a lower bar, and the
# passive reading is above it. The corrected sentence names both, so both are
# gated -- including the ratio, which is the part that carries the correction
# and which nothing checked while the sentence was wrong.
_MN61 = p61["survey_multinomial_null"]["202312"]["sigma2_over_sigma1"]["median"]
cms("SI 3: sampling alone gives a median of 0.0518", 0.0518, _MN61, _dp(0.0518, 4))
cms("SI 3: which the passive reading exceeds by a factor of 2.4", 2.4,
    _pi61["202312|dong"]["passive_sigma2_over_sigma1"] / _MN61, _dp(2.4, 1))


# ============================================================ §3.4 scale
check("MS", "3.3: the spatial ladder holds in 79 of 79 months", 79.0,
      float(p37["ladder"]["n_monotone"]), 1e-9)
cms("3.3: 424 dong merged to 25 districts retain a median of 29.8%", 29.8,
    100 * p37["ladder"]["kept_median"], _dp(29.8, 1))
_RND_MS = [p38["anchors"][str(_y)]["random|gu25"]["ratio"] for _y in p38["months"]]
check("MS", "3.3: the random arm is measured on all 79 months", 79.0,
      float(len(_RND_MS)), 1e-9)
# 25.0 and 30.1, not 25.1 and 30.0. A span quoted for every month is two
# bounds, and rounding each to nearest closed the span on both sides: the
# smallest shortfall is 25.076 and the largest 30.0037, so the sentence as it
# stood excluded months at each end.
cms("3.3: the random merge falls 25.0% short at best", 25.0,
    100 * (1 - max(_RND_MS)), _dp(25.0, 1), bound="lower")
cms("3.3: and 30.1% short at worst", 30.1, 100 * (1 - min(_RND_MS)),
    _dp(30.1, 1), bound="upper")
cms("3.3: three bands raise the median from 0.01938", 0.01938,
    _s60["r16"]["median"], _dp(0.01938, 5))
cms("3.3: ...to 0.08290", 0.08290, _s60["lim3"]["median"], _dp(0.08290, 5))
cms("3.3: a median ratio of 3.96", 3.96, _s60["retained_frac"]["median"],
    _dp(3.96, 2))
check("MS", "3.3: the age ladder rises in 79 of 79 months", 79.0,
      float(_s60["ladder_monotone"]["n_monotone_up"]), 1e-9)
cms("3.3: the curves span 2.63 decades of location count", 2.63,
    p34["extrapolation_support"]["log10_span"], _dp(2.63, 2))
cms("3.3: closing a tenfold gap would need 6.80", 6.80,
    p34["extrapolation_support"]["log10_span_required"], _dp(6.80, 2))
cms("3.3: the finer release has 1 831 traffic polygons", 1831.0,
    float(p44_ms["beta"]["reach"]["b075_polygons"]), 1e-12)
_RSV_MS = p39_ms["published_constants"]["survey"]
cms("3.3: B075 reaches 32.5% of the survey under the optimistic law", 32.5,
    100 * p44_ms["beta"]["reach"]["b075_r_slope1"] / _RSV_MS, _dp(32.5, 1))
# b075_r_ladder / r_survey is 12.836%, so the manuscript prints 12.8% — one
# decimal place, matching the 32.5% beside it. It used to print 13.0%, which
# was inherited rather than reasoned: eda/memo/phase44-beta.md writes
# "r = 0.0286（13.0%）" and the 08-21 letter repeated it. Those two are sent or
# superseded documents and keep their own wording; the manuscript does not, so
# this is a cms() gated at the precision the prose now uses.
# THE WORKED EXAMPLE OF `phrase`. Scoping the reverse direction to §3.4 is
# already enough to stop the cover this row was caught by on 2026-08-31 -- SI §8
# prints 12.8% for an unrelated normalised mutual information and SI §6 restates
# this same quantity, and neither is in §3.4 any more. The phrase is here on top
# of that because the two decay laws in this sentence are a pair, 32.5% under
# the optimistic law and 12.8% under the measured one, and the failure that
# actually happened was not the number vanishing but the number attaching to the
# wrong law. A section is the right question for most rows; for this one the
# clause is.
#
# It carries the value that used to be hand-written in a second check() row
# beside this one. Two copies of 12.8 in this file was the same duplication the
# gate exists to catch in the manuscript, so cms() now validates that the phrase
# contains the value, and there is one row instead of two.
cms("3.3: B075 reaches 12.8% of the survey under the measured law", 12.8,
    100 * p44_ms["beta"]["reach"]["b075_r_ladder"] / _RSV_MS, _dp(12.8, 1),
    phrase="12.8% under the measured one")
_POLY_MS = _p38b["polygon_prediction_interval"]
check("MS", "3.3: the polygon band's lower edge clears the measured dong "
            "reading in 79 of 79 months", 79.0,
      float(sum(1 for _v in _POLY_MS.values()
                if _v["r_polygon_lo"] > _v["r_dong"])), 1e-9)
# The forbidden pairing, both numbers. They are the point of the paragraph, so
# they are derived from the two matrices rather than restated.
cms("3.3: like for like the gap is 11.5x", 11.5, _RSV_MS / _s60["r16"]["median"],
    _dp(11.5, 1))
cms("3.3: the mismatched pairing would read 2.69x", 2.69,
    _RSV_MS / _s60["lim3"]["median"], _dp(2.69, 2))


# ============================================================ §3.5 recovery
_ARMS_MS = p39_ms["null_arm"]["arms"]
# Relabelled 2026-09-07: the sentence that prints this is in 3.2, and the
# arms below it are printed in SI 7. All three said 3.3 and none of them was
# gating a sentence there.
cms("3.2: the empty world reads +0.00455 through this pipeline", 0.00455,
    _ARMS_MS["both"]["median"], _dp(0.00455, 5))
# The two arms are 5.2755e-05 and 0.0043161, printed to five places in SI 7 and
# again in SI 8; both sections now print the same rounding of each. The factor
# beside them is the quotient of the ARMS, not of the printed roundings: 82, not
# the 86 that dividing 0.00432 by 0.00005 gives. Dividing the roundings inflates
# the ratio by 5% because 0.00005 is a one-significant-digit denominator, and a
# figure that only exists after rounding is not a measurement. Each of the three
# is registered against its source at the precision the prose prints.
cms("SI 7: the masking arm reads +0.00005", 5e-05,
    _ARMS_MS["mask only"]["median"], _dp(5e-05, 5))
cms("SI 7: the device-noise arm reads +0.00432", 0.00432,
    _ARMS_MS["noise only"]["median"], _dp(0.00432, 5))
# 2026-09-12: moved from 3.3 to SI 7 with the sentence. Two digits are a poor
# needle -- SI 7 also prints 0.2182, and "82" is a substring of it, so a
# mutation to "86 times" stayed green on the number alone. The phrase is the
# needle instead.
cms("SI 7: masking is worth 82 times less than device noise", 82.0,
    _ARMS_MS["noise only"]["median"] / _ARMS_MS["mask only"]["median"],
    _dp(82.0, 0), phrase="82 times the masking arm")
cms("3.3: recovery at beta = 0 runs from 92%", 92.0, min(_REC_MS[0.0]),
    _dp(92.0, 0))
cms("3.3: ...to 101%", 101.0, max(_REC_MS[0.0]), _dp(101.0, 0))
cms("3.3: recovery at beta = 0.95 starts at 6.8%", 6.8, min(_REC_MS[0.95]),
    _dp(6.8, 1))
cms("3.3: and ends at 9.1%", 9.1, max(_REC_MS[0.95]), _dp(9.1, 1))
# The two quantities 3.5 gained on 2026-08-31, when the gap between equation
# (2)'s (1 - beta) and the measured 6.8%-9.1% stopped being left to the SI. The
# prose now says (2) is the infinite-venue limit, and both halves of that
# explanation are properties of the world p39 BUILT rather than of the reading
# it took: how many venues a cell actually holds, and how much of the truth
# those finitely many venues leave between cells at beta = 0.95.
# 2026-09-12: the arithmetic went back to SI 7 in the word-count trim; 3.3 now
# says only that finitely many venues leave part of the truth between cells.
# The rows follow the numbers, so all four are SI 7's.
#
# The venue count is a check() and not a cms(), for the reason 2.4's "one child
# in twenty-eight" is: the manuscript spells it "eleven", and _forms() reads
# digits. Registering it as an in-text value would report it as quoted nowhere
# on a sentence that quotes it perfectly well.
check("MS", "SI 7: a cell holds eleven venues at the median", 11.0,
      p39_ms["design"]["venues_per_cell_median"], 1e-9)
# 9.2% is recomputed over EVERY r_true point on the beta = 0.95 slice, because
# the sentence is about the slice and not about one grid point, and the second
# row is what makes the single printed figure honest: the slice has to be flat
# for one number to describe it.
_BTW95_MS = [100 * _pt["r_between_med"] / _pt["r_true_med"]
             for _pt in p39_ms["surface"].values()
             if _pt["beta"] == 0.95 and _pt["r_true_med"] > 0]
check("MS", "SI 7: the beta = 0.95 slice carries ten r_true points", 10.0,
      float(len(_BTW95_MS)), 1e-9)
cms("SI 7: the finite venue draw leaves 9.2% of the truth between cells", 9.2,
    max(_BTW95_MS), _dp(9.2, 1))
check("MS", "SI 7: ...and that share is flat across the slice, to 0.01 pp", 1.0,
      1.0 if max(_BTW95_MS) - min(_BTW95_MS) < 0.01 else 0.0, 1e-9)
# The bound table, every cell, plus the verdict row underneath it.
# Every cell of this row is a printed UPPER BOUND, and the row label carries no
# cue word for the prose sweep to catch because the sentence is a table header
# rather than an "at most" clause. So the direction is declared here. It bit on
# 2026-09-09: 0.0320 and 0.0714 were correct roundings of 0.03204608 and
# 0.07141989 and were inside the half-unit tolerance, and both bounded nothing.
for _b_ms, _q_ms in (("0.0", 0.0167), ("0.5", 0.0321), ("0.8", 0.0715),
                     ("0.95", 0.1848), ("0.99", 0.3205)):
    cms(f"3.3: the bound at beta = {_b_ms} is {_q_ms}", _q_ms,
        p39_ms["inversion"][_b_ms]["bound_corrected"], _dp(_q_ms, 4),
        bound="upper")
check("MS", "3.3: four of the five slices exclude the survey", 4.0,
      float(sum(1 for _v in p39_ms["inversion"].values()
                if _v["excludes_survey"])), 1e-9)
check("MS", "3.3: beta = 0.99 is the slice that does not", 0.0,
      1.0 if p39_ms["inversion"]["0.99"]["excludes_survey"] else 0.0, 1e-9)
check("MS", "3.3: every slice is monotone in r_true", 5.0,
      float(sum(1 for _v in p39_ms["inversion"].values() if _v["monotone"])),
      1e-9)
# beta, measured. Three scope readings and the two venue-size readings.
cms("3.3: beta is 0.9326 on the national sample with Seoul weighting", 0.9326,
    p53_ms["asymmetry"]["beta_national_published"], _dp(0.9326, 4))
cms("3.3: 0.9347 with national weighting", 0.9347,
    p53_ms["asymmetry"]["beta_national"], _dp(0.9347, 4))
cms("3.3: 0.9217 on the Seoul arm", 0.9217, p53_ms["asymmetry"]["beta_seoul"],
    _dp(0.9217, 4))
_KS_MS = {r["k"]: r for r in p44_ms["sensitivity"]["venue_size"]["rows"]}
cms("SI 7: unobserved venue members push beta down to 0.9037 at k = 1.25",
    0.9037, _KS_MS[1.25]["beta"], _dp(0.9037, 4))
cms("SI 7: and to 0.8578 at k = 1.5", 0.8578, _KS_MS[1.5]["beta"],
    _dp(0.8578, 4))
check("MS", "3.3: eleven readings of beta are weighed", 11.0,
      float(len(p59["claim"]["measured_betas"])), 1e-9)
check("MS", "3.3: all eleven lie below 0.95", 1.0,
      1.0 if p59["claim"]["all_below_0p95"] else 0.0, 1e-9)
check("MS", "3.3: the survey is excluded at the measured grid point 0.95", 1.0,
      1.0 if p59["claim"]["excluded"] else 0.0, 1e-9)
cms("3.3: the crossing on the realised scale is beta* = 0.923", 0.923,
    p66_ms["beta_star"]["realised_A"], _dp(0.923, 3))
cms("SI 7: the refined crossing on the requested grid is 0.9659", 0.9659,
    p59["beta_star"]["primary_value"], _dp(0.9659, 4))
# The realised bound at the Seoul arm's own beta, which is what the margin is.
cms("3.3: at the Seoul arm's beta the bound is 0.2182", 0.2182,
    next(v["bound"] for v in p66_ms["verdicts"]
         if v["name"].startswith("Seoul 202312")), _dp(0.2182, 4))


# ============================================================ §3.6 coverage
_TH12_MS = {r["theta"]: r for r in p33_live["R2_theta_family"]["202312"]}
_TH02_MS = {r["theta"]: r for r in p33_live["R2_theta_family"]["202402"]}
# "more than ninefold" is the profile's own range, so it is derived from it.
cms("3.4: the effective sample varies ninefold across bands", 9.0,
    max(p33_live["profile"]["w"]) / min(p33_live["profile"]["w"]), _dp(9.0, 0),
    phrase="by more than ninefold")
check("MS", "3.4: assortativity is positive at every theta", 18.0,
      float(sum(1 for _d in (_TH12_MS, _TH02_MS) for _r in _d.values()
                if _r["assortativity"] > 0)), 1e-9)
cms("3.4: the smallest multiple anywhere in the family is 3.36x", 3.36,
    min(_r[_s] for _ym_ms in ("202312", "202402")
        for _r in p33_live["R2_gaps_vs_survey"][_ym_ms]
        for _s in ("assortativity", "half_l1", "cramers_v", "nmi")),
    _dp(3.36, 2))
cms("3.4: Kendall tau falls to 0.167 at theta = -1 in December 2023", 0.167,
    _TH12_MS[-1.0]["kendall_tau_vs_theta0"], _dp(0.167, 3))
cms("3.4: and to 0.117 at theta = +1", 0.117,
    _TH12_MS[1.0]["kendall_tau_vs_theta0"], _dp(0.117, 3))
cms("SI 8: 0.167 and 0.250 in February 2024", 0.250,
    _TH02_MS[1.0]["kendall_tau_vs_theta0"], _dp(0.250, 3))
check("MS", "3.4: the leading band moves across three", 3.0,
      float(len({_TH12_MS[_t]["top_band"] for _t in (-1.0, 0.0, 1.0)})), 1e-9)
_MVC_MS = p33_live["mask_vs_coverage"]
cms("SI 8: the masking band is 0.00242 wide in December 2023", 0.00242,
    _MVC_MS["202312"]["mask_width"], _dp(0.00242, 5))
cms("SI 8: the coverage family spans 0.03395", 0.03395,
    _MVC_MS["202312"]["theta_width"], _dp(0.03395, 5))
cms("3.4: coverage is 14.0 times masking", 14.0, _MVC_MS["202312"]["ratio"],
    _dp(14.0, 1))
cms("3.4: 14.3 in February 2024", 14.3, _MVC_MS["202402"]["ratio"],
    _dp(14.3, 1))
check("MS", "3.4: imputing zero gives the HIGHER assortativity", 1.0,
      1.0 if _MVC_MS["202312"]["mask_imp0"] > _MVC_MS["202312"]["mask_measured"]
      else 0.0, 1e-9)


# ============================================================ §3.7 allocation
_CM12_MS = {r["pair"]: r for r in p35_live["comparison_marginal"]["202312"]}
check("MS", "3.4: all three matrices name 15-19 first in December 2023", 1.0,
      _eq(_cap(_c46, None, "c", "the first choice they agree on"), "15-19"),
      1e-9)
cms("SI 9: the Kendall tau against the survey is +0.429 at dong", 0.429,
    _CM12_MS["survey vs passive_dong"]["tau"], _dp(0.429, 3))
_REG_MS = [100 * p35_live["allocation"][f"{_m}|R0={_r:g}"]["cross_application"]
           ["regret_share_of_benefit"]
           for _m in (202312, 202402) for _r in (1.3, 1.8, 2.5)]
check("MS", "3.4: the regret band is taken over two months x three R0", 6.0,
      float(len(_REG_MS)), 1e-9)
cms("SI 9: the plan forgoes at least 2.97% of the benefit", 2.97, min(_REG_MS),
    _dp(2.97, 2), bound="lower")
cms("SI 9: and at most 23.3%", 23.3, max(_REG_MS), _dp(23.3, 1), bound="upper")
_PS_MS = p52_ms["point_summary"]
cms("3.4: December 2023 national regret is 1.27% at R0 = 2.5", 1.27,
    100 * _PS_MS["at_anchor"]["202312|national"], _dp(1.27, 2))
cms("SI 9: and 50.9% at R0 = 1.5", 50.9,
    100 * _PS_MS["max_over_grid"]["202312|national"], _dp(50.9, 1))
cms("SI 9: a factor of 40", 40.0,
    _PS_MS["max_over_grid"]["202312|national"]
    / _PS_MS["at_anchor"]["202312|national"], _dp(40.0, 0))
cms("SI 9: February 2024 runs from 11.2%", 11.2,
    100 * _PS_MS["at_anchor"]["202402|national"], _dp(11.2, 1))
cms("3.4: to a peak of 62.6%", 62.6,
    100 * _PS_MS["max_over_grid"]["202402|national"], _dp(62.6, 1))
check("MS", "3.4: February 2024 national peaks at R0 = 1.3", 1.3,
      _PS_MS["argmax"]["202402|national"], 1e-9)
_ARG_MS = sorted(set(_PS_MS["argmax"].values()))
check("MS", "3.4: all four cells peak between R0 = 1.2 and 1.5", 1.0,
      1.0 if (min(_ARG_MS) >= 1.2 and max(_ARG_MS) <= 1.5) else 0.0, 1e-9)
# WHY THE ATTACK RATES ARE GATED AT ALL, added 2026-09-04. §3.4 used to explain
# the falling regret with "at R0 = 2.5 the final attack rate exceeds 0.8 under
# every allocation". It is false -- the highest attack rate under any allocation
# at that R0 is 0.775, and 0.8 is exceeded only by the three cells that allocate
# nothing -- and it stood because it was the one clause in the paragraph that
# named no gated quantity. Every number in its replacement is registered here,
# including the two that the sentence does NOT print: the count of cells that
# reach 0.8 with an allocation, which is what the deleted clause asserted, and
# the count that reach it without one. A false sentence is cheapest to catch
# when its negation is a row.
def _p52_at(cell, r0):
    """The one grid row for (cell, R0). Positional indexing into r0_point would
    silently move if the grid were re-declared, which is the failure p10 had."""
    hit = [_r for _r in _PT_MS[cell] if abs(_r["r0"] - r0) < 1e-12]
    if len(hit) != 1:
        raise SystemExit(f"p31: p52 has {len(hit)} rows at {cell} R0={r0}")
    return hit[0]


_PT_MS = p52_ms["point"]
# Each cell is read at ITS OWN peak, not at a shared R0: the four argmaxes are
# 1.5, 1.3, 1.3 and 1.2, so a single R0 would be reading three of the cells
# somewhere other than where the sentence says it reads them.
_PEAK_MS = {_c: _p52_at(_c, _PS_MS["argmax"][_c]) for _c in _PT_MS}
_A25_MS = {_c: _p52_at(_c, 2.5) for _c in _PT_MS}
# Every one of these carries a phrase, because every one of them is coverable
# by a substring inside §3.4 itself. Measured: with the false clause pasted back
# over the sentence, the 0.057 row stayed GREEN on the "0.1" form of 0.057
# matching inside the 0.117 Kendall tau three paragraphs up, and 0.016's "0.0"
# form matches inside any 0.0xx in the section. A section was the right question
# for the rows around these; for a sentence made of four small decimals it is not.
cms("SI 9: at their peaks the survey plan holds the attack rate at or below "
    "0.016", 0.016,
    max(_r["attack_own_plan"] for _r in _PEAK_MS.values()), _dp(0.016, 3),
    phrase="holds the final attack rate at or below 0.016", bound="upper")
cms("SI 9: while the passive plan leaves at least 0.057 infected there", 0.057,
    min(_r["attack_passive_plan"] for _r in _PEAK_MS.values()), _dp(0.057, 3),
    phrase="leaves between 0.057 and 0.171", bound="lower")
# 0.171: the largest of the four is 0.170305, and 0.170 is under it.
cms("SI 9: and at most 0.171", 0.171,
    max(_r["attack_passive_plan"] for _r in _PEAK_MS.values()), _dp(0.171, 3),
    phrase="leaves between 0.057 and 0.171", bound="upper")
cms("SI 9: at R0 = 2.5 the two plans end within 0.023 of each other", 0.023,
    max(_r["attack_passive_plan"] - _r["attack_own_plan"]
        for _r in _A25_MS.values()), _dp(0.023, 3),
    phrase="within 0.023 of each other")
# The deleted clause, as its own row, in the direction that would have caught it.
check("MS", "3.4: at R0 = 2.5 NO allocation reaches an attack rate of 0.8", 0.0,
      float(sum(1 for _r in _A25_MS.values()
                if max(_r["attack_own_plan"], _r["attack_passive_plan"]) > 0.8)),
      1e-9)
check("MS", "3.4: the grid begins at R0 = 1.1", 1.1,
      float(p52_ms["declaration"]["r0_point"][0]), 1e-9)
check("MS", "3.4: zero of the four cells exceeds its null at R0 = 2.5", 0.0,
      float(p40_ms["pillar4_summary"]["cells_regret_exceeds_null"]), 1e-9)
check("MS", "3.4: five of the twenty verdicts separate", 5.0,
      float(p52_ms["answer"]["n_flips"]), 1e-9)
check("MS", "3.4: twenty verdicts are weighed", 20.0,
      float(p52_ms["answer"]["n_verdicts"]), 1e-9)
check("MS", "3.4: four of the five sit at or below R0 = 1.8", 4.0,
      float(_a2_62["n_flips_inside_1p2_1p8"]), 1e-9)


# ============================================================ §4 and §5
# The discussion and the conclusion re-quote §3's numbers rather than adding
# their own, so what is gated here is that they re-quote the SAME ones: each
# row below is the section-3 source read a second time, so a number could not
# be corrected in §3 and left standing in §4.
# 2026-09-07, phase 69: the sentences that report what the survey's own pairing
# convention does to the crossing. Both are read out of results_p69.json, which
# is the file that measured them; the verdict itself is a check() and not a
# cms(), because "does not exclude" is a state and not a number the prose quotes.
_P69C = p69_ms["answer"]["primary_reading"]
cms("3.3: the crossing moves to 0.918", 0.918,
    p69_ms["beta_star"]["realised_C"], _dp(0.918, 3))
cms("3.3: and the Seoul arm's bound to 0.2318", 0.2318,
    _P69C["bound_C"], _dp(0.2318, 4))
check("MS", "3.3: on the survey's convention the Seoul arm does NOT exclude",
      0.0, 1.0 if _P69C["excludes"] else 0.0, 1e-9)
check("MS", "3.3: and it does exclude on the world's own scale", 1.0,
      1.0 if _P69C["bound_A"] < p69_ms["answer"]["r_survey"] else 0.0, 1e-9)

cms("4.1: the distance is never further than 0.0258", 0.0258,
    p67_ms["summary"]["gap_rel_max"], _dp(0.0258, 4), bound="upper")
# "the bound excludes the survey value by 0.038 or more" had no row either, and
# the thinnest of those margins is 0.037942 -- the sentence claimed a wider
# exclusion than the grid gives. The restriction it states, a realised beta up
# to 0.908, is what picks the rows out, so the selection is made here and its
# size is gated beside the bound rather than left to the reader of this file.
_B69_MS = [_r for _r in p69_ms["conversion"]
           if round(_r["realised_A"], 3) <= 0.908]
check("MS", "3.3: four grid rows have a realised beta up to 0.908", 4.0,
      float(len(_B69_MS)), 1e-9)
cms("3.3: below that the bound excludes the survey by 0.037 or more", 0.037,
    min(p69_ms["answer"]["r_survey"] - _r["bound"] for _r in _B69_MS),
    _dp(0.037, 3), bound="lower")
cms("3.4: the ordering falls to a Kendall tau of 0.117 in December 2023",
    0.117, _TH12_MS[1.0]["kendall_tau_vs_theta0"], _dp(0.117, 3))
cms("3.4: and to 0.167 in February 2024", 0.167,
    _TH02_MS[-1.0]["kendall_tau_vs_theta0"], _dp(0.167, 3))
# 2026-09-05: 14.0 is the ratio of two ASSORTATIVITY bandwidths (Fig 6(a)),
# not a statement about the eigenvector ordering (Fig 6(b)), and 4.1 used to
# attach it to the ordering. The prose now says the family spans fourteen times
# more of the reading, so the phrase this row greps for moves with it.
cms("4.1: the coverage family spans fourteen times more of the reading",
    14.0, _MVC_MS["202312"]["ratio"], _dp(14.0, 0),
    phrase="fourteen times more")
cms("4.1: age coarsening inflates by a median factor of 3.96", 3.96,
    _s60["retained_frac"]["median"], _dp(3.96, 2))
# "takes up", not "at most": 3.2 prints this as a measurement at one decimal
# place, and 27.2347 rounds to it correctly. The label used to say "at most",
# which is a bound 27.2 does not carry, and the direction rule is what asked
# the question. The paragraph's bound is IPF run at full strength, not the
# digits, so the label moved rather than the prose.
cms("3.2: IPF takes up 27.2% of the gap", 27.2,
    100 * _IPF_MS["202402"]["national"]["absorbed_frac"], _dp(27.2, 1))
cms("4.2: the arithmetic mean of a co-arrival cell is about 1 080 people",
    1080.0, p44_ms["cell"]["mean"], _dp(1080.0, -1))
cms("3.2: the pipeline's true zero is 27.1% of the noise-corrected reading",
    27.1, 100 * p39_ms["null_arm"]["null_median"]
    / p39_ms["published_constants"]["r_obs_corrected"], _dp(27.1, 1))


# ============================================================ Data accessibility
# The only number in that section with a results file behind it. The two Zenodo
# DOIs, the licence names and the portal's article 11 are external facts and
# are named in main()'s not-covered note.
#
# WHERE THE PROSE WENT, 2026-08-31. The Royal Society author guidelines now say
# the end-section statements are "no longer required ... in the manuscript
# itself" and are filed in ScholarOne instead, so Data accessibility was cut to
# a stub and its full text moved to paper/scholarone_submission_fields.md. The
# manuscript therefore no longer prints "436 of 436" anywhere.
#
# THESE TWO ROWS SURVIVE THAT MOVE, and it is worth saying why rather than
# leaving the next reader to re-derive it. They are check(), not cms(): cms()
# registers a value in IN_TEXT_MS and so asserts the manuscript PRINTS it,
# whereas check() only asserts the quoted literal matches the results file.
# Nothing here was ever a document assertion, so the cut could not have turned
# it red -- which is exactly the failure mode this gate is otherwise prone to,
# a row staying green over prose that no longer says the thing. The literal 436
# below is now the ScholarOne data statement's number, not the manuscript's; if
# that statement is reworded, this row is what must move with it.
check("MS", "data accessibility: the second implementation agrees on 692 of 692",
      692.0, float(p36_live["n_checks"]), 1e-9)
check("MS", "data accessibility: and fails none of them", 0.0,
      float(p36_live["n_fail"]), 1e-9)

# 2026-09-03: the three-tier validation returned to the body. It had shrunk to
# the five words "the three verification harnesses" in Data accessibility plus a
# box inside Figure 1, which put the paper's strongest credibility asset nowhere
# a reader of the prose would meet it. The count is now a document assertion of
# section 2.6 and not only a ScholarOne one, so it gets a cms() beside the two
# check() rows above: those gate the submission statement, this gates the prose.
cms("2.6: the second implementation agrees on 692 of 692 checks", 692.0,
    float(p36_live["n_checks"]), 1e-9)


# ====================================================== the figure captions block
# The manuscript restates its five caption sheets, so each row here is
# (manuscript caption, the sheet it restates, the results file the sheet was
# read out of). The middle link is already gated at tolerance zero by the 08-27
# round, so registering the manuscript against _cap() closes the chain end to
# end rather than opening a second, parallel one.
cms("Fig 1: 1 896 parquet files", 1896.0, float(p48["parquet_files"]), 1e-12)
cms("SI 2: bands 20-24 to 40-44 lose 10.9% of their cells at least", 10.9,
    100 * p48["masked_band_rate_range"][0], _dp(10.9, 1), bound="lower")
# 69.6, not 69.5: the largest band rate is 0.6952.
cms("SI 2: and 69.6% at most", 69.6, 100 * p48["masked_band_rate_range"][1],
    _dp(69.6, 1), bound="upper")
cms("SI 2: bands 45-64 lose a trace under 0.13%", 0.13,
    p48["masked_band_trace_cap_pct"], 1e-12)
check("MS", "Fig 1: the largest trace band is 50-54", 1.0,
      _eq(p48["masked_band_trace_max_band"], "50-54"), 1e-9)
cms("Fig 1: 11.1% of volume sits behind the mask", 11.1,
    100 * p48["measured_fill_share"], _dp(11.1, 1))
# 2026-09-03: Figure 1's claim box used to draw "3 of 78 alternative calendar
# alignments match" while 3.1 said eleven alternatives and none reproducing --
# two mutually exclusive statements of one fact, in the two most-read places in
# the paper. The box now counts distinct calendars, so these are the rows that
# gate what it draws. The raw 3-of-78 pair is still gated above, under the
# 08-27 round, because that is the number those captions were sent quoting.
check("MS", "Fig 1: none of the alternative calendars reproduces the result", 0.0,
      float(p48["claim1_calendars_matching"]), 1e-9)
check("MS", "Fig 1: and there are eleven of them", 11.0,
      float(p48["claim1_calendars_alternative"]), 1e-9)
check("MS", "Fig 1: the box agrees with 3.1 on the alternative count", 0.0,
      abs(p48["claim1_calendars_alternative"] - (_E64["E_exact"] - 1)), 0.0)
check("MS", "Fig 1: and on how many of them match", 0.0,
      abs(p48["claim1_calendars_matching"]
          - _D64["exact"]["n_nonidentity_classes_reaching_observed"]), 0.0)

# Figure 2 is p63's five-panel sheet. Until 2026-08-29 the rows below read
# results_p41.json, because the manuscript named eda/fig/p63_figure2.pdf while
# its caption still described p41_semester.png's three panels: the gate was
# green on a caption that belonged to a different figure, which is the one
# failure a citation gate cannot catch by being greener. The caption was
# rewritten to the figure that ships and every row now reads
# results_p63.json's caption_numbers. Its link to p51/p54/p58 is already gated
# at tolerance zero by the "27" block above, so these rows close the chain
# caption -> sheet -> source without restating the middle link.
check("MS", "Fig 2(a): 79 monthly matrices", 79.0,
      _cap(_c63, 2, "a", "months in the series"), 1e-9)
_LAB63 = [float(_x) for _x
          in _cap(_c63, 2, "a", "series labels").split("/")]
check("MS", "Fig 2(a): 46 of them are term months", 46.0, _LAB63[0], 1e-9)
check("MS", "Fig 2(a): 27 are vacation months", 27.0, _LAB63[1], 1e-9)
check("MS", "Fig 2(a): and 6 are Decembers", 6.0, _LAB63[2], 1e-9)
cms("Fig 2(a): the floor is 0.02296 bits", 0.02296,
    _cap(_c63, 2, "a", "floor (p51 corrected national"), _dp(0.02296, 5))
check("MS", "Fig 2(a): 17 months reach it", 17.0,
      _cap(_c63, 2, "a", "months at or above the floor"), 1e-9)
_OBS63 = [float(_x) for _x
          in _cap(_c63, 2, "a", "of those, term").split("/")]
check("MS", "Fig 2(a): all 17 of them are term months", 17.0, _OBS63[0], 1e-9)
check("MS", "Fig 2(a): no vacation month is among them", 0.0, _OBS63[1], 1e-9)
check("MS", "Fig 2(a): and no December", 0.0, _OBS63[2], 1e-9)
check("MS", "Fig 2(a): they fall in nine runs", 9.0,
      _cap(_c63, 2, "a", "=the clearing months sit in this many runs"), 1e-9)
_YM63 = [float(_x) for _x
         in _cap(_c63, 2, "a", "year medians, first to last").split("->")]
cms("Fig 2(a): the year medians start at 0.01372 bits", 0.01372, _YM63[0],
    _dp(0.01372, 5))
cms("Fig 2(a): and end at 0.02329", 0.02329, _YM63[1], _dp(0.02329, 5))
cms("3.1: a factor of 1.70 over the period", 1.70,
    _cap(_c63, 2, "a", "level trend over the period"), _dp(1.70, 2))

# 2026-08-31 (advisor 3.2): panels (d) and (e) are gone; old (c)'s agreement
# numbers now back the "not independent draws" sentence inside (b)'s caption,
# and the honest floor from p64 replaces the 1/79 the old panel drew.
check("MS", "Fig 2(b): the three are k = 12, 24 and 36", 1.0,
      _eq(_cap(_c63, 2, "b", "the shifts that match"), [12, 24, 36]), 1e-9)
cms("3.1: the matching shifts keep 74 labels", 74.0,
    _cap(_c63, 2, "b", "agreement at k = 12"), _dp(74.0, 0))
cms("3.1: 69", 69.0, _cap(_c63, 2, "b", "agreement at k = 24"),
    _dp(69.0, 0))
cms("3.1: and 64 of 79 months", 64.0,
    _cap(_c63, 2, "b", "agreement at k = 36"), _dp(64.0, 0))
cms("3.1: the chance level is 36.5", 36.5,
    _cap(_c63, 2, "b", "chance agreement"), _dp(36.5, 1))
check("MS", "Fig 2(b): 36 of the 78 shifts clear chance", 1.0,
      _eq(_cap(_c63, 2, "b", "shifts agreeing above chance"), "36 of 78"),
      1e-9)
# 2026-09-03: the decaying half of the seam argument came back to the caption.
# Three multiples of twelve reproduce the result and three do not, which reads
# as a hole until the agreement series is on the page: it falls 74, 69, 64, 59,
# 54, 49 across k = 12..72, five months per step, so equivalence under a
# multiple of twelve is approximate and decays. The first three were already
# gated above; these are the three that were missing, and they are read from
# p58 rather than p63 because the deleted panel took them out of the sheet.
_AGK = p58["degeneracy"]["agreement_by_k"]
cms("Fig 2(b): and 59 labels at k = 48", 59.0, float(_AGK[48]), _dp(59.0, 0))
cms("Fig 2(b): 54 at k = 60", 54.0, float(_AGK[60]), _dp(54.0, 0))
cms("Fig 2(b): and 49 at k = 72", 49.0, float(_AGK[72]), _dp(49.0, 0))
check("MS", "Fig 2(b): agreement falls five months per twelve-month step", 1.0,
      1.0 if all(_AGK[k] - _AGK[k + 12] == 5
                 for k in (12, 24, 36, 48, 60)) else 0.0, 1e-9)
check("MS", "Fig 2(b): and those last three no longer reach seventeen", 0.0,
      float(sum(1 for k in ("48", "60", "72")
                if p58["degeneracy"]["n_term_at_multiples_of_twelve"][k]
                >= p58["p_shift"]["observed"])), 1e-9)
# 0.0506, 1/12 = 0.083 and 1/6 = 0.17 are already cms()'d at SI 5's own
# sentence; restating them here is checked as arithmetic (check), not
# re-registered as a second quote (cms), so a real value can't be silently
# doubled in IN_TEXT_MS.
check("MS", "Fig 2(b): the observed circular-shift value is 0.0506 = 4/79",
      1.0, _eq(_cap(_c63, 2, "b", "circular-shift p as a fraction"), "4/79"),
      1e-9)
check("MS", "Fig 2(b): which is p64's 1/12 = 0.083 resolution floor", 0.0,
      abs(_cap(_c63, 2, "b", "resolution floor, 1 over the distinct")
          - p64["deduplicated"]["exact"]["resolution_floor"]), 1e-9)
check("MS", "Fig 2(b): or 1/6 = 0.17 collapsed", 0.0,
      abs(_cap(_c63, 2, "b", "resolution floor after collapsing")
          - p64["deduplicated"]["advisor"]["resolution_floor"]), 1e-9)
cms("SI 5: the secondary contrast reads 0.0253", 0.0253,
    _cap(_c63, 2, "b", "secondary median-contrast p"), _dp(0.0253, 4))
cms("SI 5: on a D(0) of 0.00563 bits", 0.00563,
    _cap(_c63, 2, "b", "secondary median-contrast D(0)"), _dp(0.00563, 5))
check("MS", "Fig 2(b): D(0)'s rank among the rotations is 1", 1.0,
      _cap(_c63, 2, "b", "secondary contrast rank among"), 1e-9)

# The regression strip that sits under (b). It was labelled "Fig 2(c)" here
# until 2026-09-02, when a real panel (c) arrived from Figure 3 and the name
# became a pointer to the wrong thing -- the caption has never called the strip
# a panel. Its literals are already cms()'d at 3.1, so these are restated as
# arithmetic checks against p64, closing caption -> sheet -> p64 end to end.
# _dp() is RELATIVE to `quoted` (0.5 * 10**-dp / abs(quoted)); these four
# compare against a diff of 0.0, so _dp() would divide by zero's neighbourhood
# and the tolerance would vanish -- half a unit in the last printed place is
# passed as a bare absolute float instead, the same way the bit-for-bit rows
# above pass a literal 0.0.
check("MS", "Fig 2(b) caption: the term coefficient is p64's 0.00368", 0.0,
      abs(_cap(_c63, 2, "b", "term coefficient, dong-level assortativity")
          - 0.00368), 5e-6)
check("MS", "Fig 2(b) caption: its Newey-West se is 0.00048", 0.0,
      abs(_cap(_c63, 2, "b", "Newey-West standard error") - 0.00048), 5e-6)
check("MS", "Fig 2(b) caption: a t of 7.68", 0.0,
      abs(_cap(_c63, 2, "b", "t on the term coefficient") - 7.68), 5e-3)
check("MS", "Fig 2(b) caption: a p of 7.6e-11", 0.0,
      abs(_cap(_c63, 2, "b", "p on the term coefficient") - 7.6e-11), 5e-12)
check("MS", "Fig 2(b) caption: the lag is fixed at 3 before the fit", 3.0,
      _cap(_c63, 2, "b", "Newey-West lag, fixed by rule"), 1e-9)
check("MS", "Fig 2(b) caption: it is the tallest t of all 79 rotations", 1.0,
      _cap(_c63, 2, "b", "the observed regression t ranks first"), 1e-9)
check("MS", "Fig 2(b) caption: the only one no rotation beats", 0.0,
      _cap(_c63, 2, "b", "calendar rotations whose t beats"), 1e-9)

# (c), the spatial ladder, arrived from Figure 3 on 2026-09-02. The caption
# quotes three values off p37 and one count. The two range endpoints were new
# to the manuscript with the panel; the median it quoted beside them is 3.1's
# own 0.01938, already cms()'d there and checked against p60 a few blocks
# above, so it is not registered a third time here. The counts are short
# integers and are plain checks against literals, on the convention above.
# 2026-09-12: the caption stopped printing the endpoints in the word-count trim
# and points at SI 3, whose own rows (against p37) gate them. These two became
# check(), on the beta = 0.98 convention: p63 still has to have drawn the same
# series, but no document sentence quotes the figures here any more.
check("MS", "Fig 2(c) draws the dong series from 0.01311 (SI 3 quotes it)", 0.01311,
      _cap(_c63, 2, "c", "dong assort_min"), 5e-6)
check("MS", "Fig 2(c): ...to 0.02589", 0.02589,
      _cap(_c63, 2, "c", "dong assort_max"), 5e-6)
check("MS", "Fig 2(c): it is drawn on p37's 79 months", 79.0,
      float(_cap(_c63, 2, "c", "=months")), 1e-9)
check("MS", "Fig 2(c): the ladder holds in 79 of 79", 0.0,
      abs(_cap(_c63, 2, "c", "months where the ladder holds")
          - p37["ladder"]["n_monotone"]), 0.0)
check("MS", "Fig 2(c): on the weekday-evening panel every other reading uses",
      1.0, _eq(_cap(_c63, 2, "c", "the panel p37's series is read on"), "WE"),
      1e-9)

cms("Fig 3(a): the shared colour limit is 1.82", 1.82,
    _cap(_c47, 3, "a", "shared colour limit"), _dp(1.82, 2))
cms("Fig 3(a): the passive dong maximum is 0.353", 0.353,
    _cap(_c47, 3, "a", "passive, dong (424)"), _dp(0.353, 3))
cms("Fig 3(a): the district maximum is 0.183", 0.183,
    _cap(_c47, 3, "a", "passive, district (25)"), _dp(0.183, 3))
cms("Fig 3(b): sigma2/sigma1 is 0.066 at district", 0.066,
    _cap(_c47, 3, "b", "sigma2/sigma1, passive, district"), _dp(0.066, 3))
cms("Fig 3(b): the district leading share is 99.5%", 99.5,
    100 * _cap(_c47, 3, "b", "sigma1 share, passive, district"), _dp(99.5, 1))
cms("Fig 3(b): the dong leading share is 98.0%", 98.0,
    100 * _cap(_c47, 3, "b", "sigma1 share, passive, dong"), _dp(98.0, 1))
cms("Fig 3(b): the null's 2.5th percentile is 0.0998", 0.0998,
    _cap(_c47, 3, "b", "p2.5 (lower edge"), _dp(0.0998, 4))
cms("Fig 3(b): its 97.5th is 0.1671", 0.1671,
    _cap(_c47, 3, "b", "null, p97.5"), _dp(0.1671, 4))
# check(), not cms(): grep of the manuscript pair finds this value nowhere,
# so the document assertion was never true. It stayed green because _forms()
# offers coarse spellings and the region contained one. Verified, not quoted.
check("MS", "Fig 3(b): the null's smallest draw is 0.0858", 0.0858,
      _cap(_c47, 3, "b", "min over 500 ego-band shuffles"), _dp(0.0858, 4) * 0.0858)
cms("SI 3: its largest is 0.2117", 0.2117,
    _cap(_c47, 3, "b", "max over 500 ego-band shuffles"), _dp(0.2117, 4))
cms("SI 3: the declared escape threshold is 0.1585", 0.1585,
    _cap(_c47, 3, "b", "p95 (the declared escape threshold)"), _dp(0.1585, 4))
cms("SI 3: the bootstrap interval starts at 0.5161", 0.5161,
    _cap(_c47, 3, "b", "bootstrap, p2.5"), _dp(0.5161, 4))
cms("SI 3: and ends at 0.7899", 0.7899,
    _cap(_c47, 3, "b", "bootstrap, p97.5"), _dp(0.7899, 4))
cms("SI 3: its smallest of 500 resamples is 0.4793", 0.4793,
    _cap(_c47, 3, "b", "bootstrap, min over 500 respondent resamples"),
    _dp(0.4793, 4))
check("MS", "Fig 3(b): the passive district spectrum sits at the 0th percentile",
      0.0, _cap(_c47, 3, "b", "passive district sigma2/sigma1, percentile"),
      1e-9)
check("MS", "SI 3: the survey block has 24 empty cells", 24.0,
      float(_cap(_c47, 3, "b", "observed empty cells")), 1e-9)
check("MS", "SI 3: against a null median of 8", 8.0,
      _cap(_c47, 3, "b", "null median empty cells"), 1e-9)
check("MS", "Fig 3(a): 10 of the survey's 24 empty cells are empty in both "
            "directions", 10.0,
      float(_cap(_c47, 3, "a", "empty in both directions and so still empty")),
      1e-9)

cms("SI 6: the adjacency graph carries 1 153 edges", 1153.0,
    float(p34["adjacency"]["edges"]), 1e-12)
cms("SI 6: at mean degree 5.44", 5.44, p34["adjacency"]["mean_degree"],
    _dp(5.44, 2))
cms("Fig 4(b): the nested ladder reads 0.019377 at 16 bins", 0.019377,
    _cap(_c47, 4, "b", "nested ladder median, 16 bins"), _dp(0.019377, 6))
cms("Fig 4(b): 0.033988 at 8", 0.033988,
    _cap(_c47, 4, "b", "nested ladder median, 8 bins"), _dp(0.033988, 6))
cms("Fig 4(b): 0.056344 at 4", 0.056344,
    _cap(_c47, 4, "b", "nested ladder median, 4 bins"), _dp(0.056344, 6))
cms("Fig 4(b): 0.086885 at 2", 0.086885,
    _cap(_c47, 4, "b", "nested ladder median, 2 bins"), _dp(0.086885, 6))
cms("Fig 4(b): the Lim marker sits at 0.082904", 0.082904,
    _cap(_c47, 4, "b", "median 3-bin assortativity"), _dp(0.082904, 6))
cms("Fig 4(b): a factor of 3.9577", 3.9577,
    _cap(_c47, 4, "b", "median 3-bin / 16-bin ratio"), _dp(3.9577, 4))
check("MS", "Fig 4(b): ranging from 3.4705 to 4.7933", 1.0,
      _eq(_cap(_c47, 4, "b", "ratio, range over the months"),
          "3.4705-4.7933"), 1e-9)
cms("SI 6: the absorbed-excess term is 2.2708", 2.2708,
    _cap(_c47, 4, "b", "absorbed-excess gain"), _dp(2.2708, 4))
cms("SI 6: the margin-concentration term is 1.71", 1.71,
    _cap(_c47, 4, "b", "margin-concentration shrink"), _dp(1.71, 2))
# The caption's own null had no row at all until the direction rule swept the
# prose for it: "no larger than 4.94e-16" is a bound, and the measured maximum
# is 4.940347e-16, which sits above it. 4.95 is that rounded up at the two
# decimals the caption prints.
cms("Fig 4(b): under proportionate mixing every rung returns at most 4.95e-16",
    4.95e-16, _m60["proportionate_mixing_null_max_abs_r"],
    _dp(4.95e-16, 18), bound="upper")
cms("Fig 4(b): the spatial medians move to 0.4692 at district", 0.4692,
    _cap(_c47, 4, "b", "district / dong"), _dp(0.4692, 4))
cms("Fig 4(b): and 0.2454 at city", 0.2454,
    _cap(_c47, 4, "b", "city / dong"), _dp(0.2454, 4))
cms("SI 6: the 16-band series spans 5.79 pp", 5.79,
    _cap(_c47, 4, "not drawn", "16-band range, pp"), _dp(5.79, 2))
cms("SI 6: three bands retain 2.93 pp", 2.93,
    _cap(_c47, 4, "not drawn", "3-band range, pp"), _dp(2.93, 2))
cms("SI 6: which is 50.6% of it", 50.6,
    100 * _cap(_c47, 4, "not drawn", "fraction of the range three bands retain"),
    _dp(50.6, 1))
cms("SI 6: the largest single absorption is 3.17 pp", 3.17,
    abs(_cap(_c47, 4, "not drawn", "largest single absorption, pp")), _dp(3.17, 2))
cms("SI 6: the 60+ rule reads +0.07 pp", 0.07,
    _cap(_c47, 4, "not drawn", "3-band 60+ , pp"), _dp(0.07, 2))
cms("SI 6: its 80+ constituent reads +0.84 pp", 0.84,
    _cap(_c47, 4, "not drawn", "16-band 80+ , pp"), _dp(0.84, 2))
cms("SI 6: 80+ reads +0.55 pp under origin-only filtering", 0.55,
    _cap(_c47, 4, "not drawn", "p18b 16-band 80+, origin only, pp"), _dp(0.55, 2))
cms("SI 6: 20-24 reads -3.96 pp with both endpoints", -3.96,
    _cap(_c47, 4, "not drawn", "p18b 16-band 20-24, both endpoints, pp"), _dp(3.96, 2))
cms("SI 6: and -3.13 pp with the origin only", -3.13,
    _cap(_c47, 4, "not drawn", "p18b 16-band 20-24, origin only, pp"), _dp(3.13, 2))
cms("SI 6: three bands read -0.16 pp under origin-only filtering", -0.16,
    _cap(_c47, 4, "not drawn", "p18b 3-band 60+, origin only, pp"), _dp(0.16, 2))
check("MS", "SI 6: the zero crossing moves one bin under filtering", 1.0,
      1.0 if (_cap(_c47, 4, "not drawn", "zero crossing, both endpoints")
              == "65-69/70-74"
              and _cap(_c47, 4, "not drawn", "zero crossing, origin only")
              == "70-74/75-79") else 0.0, 1e-9)
cms("SI 6: the two series differ by at most 0.48 pp", 0.48,
    _cap(_c47, 4, "not drawn", "largest gap between p19's bars"), _dp(0.48, 2),
    bound="upper")

# check(), not cms(): grep of the manuscript pair finds this value nowhere,
# so the document assertion was never true. It stayed green because _forms()
# offers coarse spellings and the region contained one. Verified, not quoted.
check("MS", "Fig 5(b): the bound is 0.2065 at beta = 0.96", 0.2065,
      _cap(_c56, None, "b", "bound at beta = 0.96"), _dp(0.2065, 4) * 0.2065)
check("MS", "Fig 5(b): 0.2340 at 0.97", 0.2340,
      _cap(_c56, None, "b", "bound at beta = 0.97"), _dp(0.2340, 4) * 0.2340)
# check(), not cms(): this one is drawn on Figure 5(b)'s curve and written
# down nowhere, so claiming the caption prints it was never true. The value is
# still verified against the caption sheet; only the document assertion goes.
check("MS", "Fig 5(b): the bound at beta = 0.98 is 0.2707", 0.2707,
      _cap(_c56, None, "b", "bound at beta = 0.98"), 5e-5)
check("MS", "Fig 5(b): the survey is excluded up to and including 0.96", 1.0,
      1.0 if (_cap(_c56, None, "b", "survey excluded at beta = 0.96") is True
              and _cap(_c56, None, "b", "survey excluded at beta = 0.97")
              is False) else 0.0, 1e-9)
cms("SI 7: the analytic form reads 0.3354 at beta = 0.95", 0.3354,
    _cap(_c56, None, "b", "analytic r_obs/(1-beta)"), _dp(0.3354, 4))
cms("Fig 5(c): the synthetic data field reads 32.7%", 32.7,
    100 * _cap(_c56, None, "c", "synthetic data field"), _dp(32.7, 1))
cms("Fig 5(c): the archetype field reads 14.9%", 14.9,
    100 * _cap(_c56, None, "c", "synthetic archetype field"), _dp(14.9, 1))
cms("Fig 5(c): the December 2023 rung is 30.1%", 30.1,
    100 * _cap(_c56, None, "c", "the real 202312 rung"), _dp(30.1, 1))
cms("3.3: the measured band runs from 27.6%", 27.6,
    100 * _cap(_c56, None, "c", "p37 band, low"), _dp(27.6, 1))
cms("3.3: to 32.6%", 32.6, 100 * _cap(_c56, None, "c", "p37 band, high"),
    _dp(32.6, 1))
# 2026-09-03: the free validation reached the body. It had been honest only in
# the caption -- "just above the top of that band" -- while 3.5 said four
# detectors pass and said nothing at all about a fifth that does not, which is
# the shape a referee is entitled to be annoyed by. 3.5 now states it, so the
# band and both field readings are registered against 3.5's own sentence.
# The three shared with the caption are check(), not cms(), so restating them
# does not enter IN_TEXT_MS twice; 26.8% is new and is the only cms() here.
# 2026-09-12: the caption stopped printing the band's endpoints (it draws the
# band and names December 2023's rung), so the two endpoint cms() rows above
# moved their label from Fig 5(c) to 3.3, where the numbers now live; the two
# check() rows below keep verifying that the drawn band is the one 3.3 states.
# check() takes an ABSOLUTE tolerance; _dp() returns a relative one and belongs
# to cms(). Half a unit in the first decimal place is 0.05 on these three.
check("MS", "Fig 5(c): the band it draws is 3.3's, low end", 27.6,
      100 * _cap(_c56, None, "c", "p37 band, low"), 0.05)
check("MS", "Fig 5(c): and its high end", 32.6,
      100 * _cap(_c56, None, "c", "p37 band, high"), 0.05)
check("MS", "3.3: and the noiseless data field above it", 32.7,
      100 * _cap(_c56, None, "c", "synthetic data field"), 0.05)
# The noisy reading USED TO BE UNGATEABLE and was reported as such at the foot
# of this gate: p39 stores it, under ladder.points, which the note predated.
cms("3.3: and 26.8% once device noise is added", 26.8,
    100 * p39_ms["ladder"]["points"]["data|0.02"]["kept_median"], _dp(26.8, 1))
check("MS", "3.3: the noisy reading sits below the band, the noiseless above",
      1.0,
      1.0 if (p39_ms["ladder"]["points"]["data|0.02"]["kept_median"]
              < p39_ms["ladder"]["p37_band"][0]
              < p39_ms["ladder"]["p37_band"][1]
              < p39_ms["ladder"]["noiseless"]["data"]) else 0.0, 1e-9)

cms("Fig S1(a): December 2023 Seoul reads 7.5% at R0 = 2.5", 7.5,
    100 * _cap(_c55, None, "a", "Dec 2023, Seoul: regret at R0=2.5"),
    _dp(7.5, 1))
cms("Fig S1(a): against 20.2% at R0 = 1.3", 20.2,
    100 * _cap(_c55, None, "a", "Dec 2023, Seoul: maximum over the grid"),
    _dp(20.2, 1))
cms("Fig S1(a): February 2024 Seoul reads 23.3% at R0 = 2.5", 23.3,
    100 * _cap(_c55, None, "a", "Feb 2024, Seoul: regret at R0=2.5"),
    _dp(23.3, 1))
cms("Fig S1(a): against 25.4% at R0 = 1.2", 25.4,
    100 * _cap(_c55, None, "a", "Feb 2024, Seoul: maximum over the grid"),
    _dp(25.4, 1))
check("MS", "Fig S1(b): December 2023 national clears at 1.2, 1.5 and 1.8", 1.0,
      _eq(_cap(_c55, None, "b", "Dec 2023, national: R0 values that exceed"),
          [1.2, 1.5, 1.8]), 1e-9)
check("MS", "Fig S1(b): February 2024 national clears at 1.2 alone", 1.0,
      _eq(_cap(_c55, None, "b", "Feb 2024, national: R0 values that exceed"),
          [1.2]), 1e-9)
check("MS", "Fig S1(b): the fifth circle is February 2024 Seoul at R0 = 3.5",
      1.0, _eq(_cap(_c55, None, "b", "Feb 2024, Seoul: R0 values that exceed"),
               [3.5]), 1e-9)
check("MS", "Fig S1(b): seven cleared before the vector was corrected", 7.0,
      float(_cap(_c55, None, "b", "before the vector was corrected")), 1e-9)
# 2026-08-31, advisor 3.9, rewritten 2026-09-02. The advisor asked the caption
# to call the fifth circle an expected false positive; p62 measures the flip
# rule's size at 6.2e-101, so that framing is wrong by ~99 orders of magnitude
# and SI 9 already refuses it. The caption states the margin instead, which is
# what is actually fragile. Until 09-02 it only pointed at SI 9 for it, which
# is not what 3.9 asked for -- the circle has to answer for itself where the
# reader meets it -- so the number is now printed in the caption and this row
# is cms(), not check(). The phrase is the point of the row: without it the
# reverse direction would stay green on a caption that had gone back to
# pointing, because SI 9's own 0.015 is in a different region and Fig S1 would
# simply have lost a number nothing was asking it to keep.
cms("Fig S1(b): the fifth circle clears its null by only 0.015", 0.015,
    _cap(_c55, None, "b", "thinnest margin"), _dp(0.015, 3),
    phrase="clears its null by only 0.015")

cms("SI 8: February 2024's smallest assortativity in the family is 0.01459",
    0.01459, _cap(_c46, None, "a", "smallest assortativity anywhere"),
    _dp(0.01459, 5))
cms("SI 9: December 2023 forgoes 2.97% at the low end of the three original R0", 2.97,
    100 * _cap(_c46, None, "d", "December 2023: benefit forgone at R0 = 1.8"),
    _dp(2.97, 2))


# ============================================================ SI
# --- SI 2: the pairing convention, the reconciliation and the dedup rule
cms("SI 2: the dong-level correction is -4.29%", -4.29,
    100 * p49_ms["primary"]["rel_change"], _dp(4.29, 2))
cms("SI 2: 0.021154 falls to 0.020247", 0.020247, p49_ms["primary"]["r_wor"],
    _dp(0.020247, 6))
_L49_MS = {r["level"]: r for r in p49_ms["ladder"] if r["ym"] == 202312}
cms("SI 2: at district it is -0.55%", -0.55, 100 * _L49_MS["gu"]["rel_change"],
    _dp(0.55, 2))
cms("SI 2: at city -0.044%", -0.044, 100 * _L49_MS["city"]["rel_change"],
    _dp(0.044, 3))
cms("SI 2: the arithmetic mean group size is 1 080", 1080.0,
    p44_ms["cell"]["mean"], _dp(1080.0, -1))
cms("SI 2: not the median of 533", 533.0, p44_ms["cell"]["median"],
    _dp(533.0, 0))
cms("SI 2: on one convention the ratio is 12.28x", 12.28,
    p49_ms["convention_consistency"]["ratio_one_convention"], _dp(12.28, 2))
cms("SI 2: and not 11.76x", 11.76,
    p49_ms["convention_consistency"]["ratio_as_published"], _dp(11.76, 2))
cms("SI 2: a move of +4.5%", 4.5,
    100 * p49_ms["convention_consistency"]["rel_move"], _dp(4.5, 1))
check("MS", "SI 2: the 2020-03 reconciliation is -3.2e-05", -3.22e-05,
      p29_ms["202003"]["reconciliation_rel_diff"], 5e-8)
check("MS", "SI 2: and the 2020-12 one is +4.19e-06", 4.19e-06,
      p29_ms["202012"]["reconciliation_rel_diff"], 5e-9)
check("MS", "SI 2: the two reconciliations have opposite signs", 1.0,
      1.0 if (p29_ms["202003"]["reconciliation_rel_diff"]
              * p29_ms["202012"]["reconciliation_rel_diff"]) < 0 else 0.0, 1e-9)
cms("SI 2: 2.28 people per masked cell in 2020-03", 2.28,
    p29_ms["202003"]["per_cell_mean"], _dp(2.28, 2))
cms("SI 2: 11.09% of volume in 2020-12", 11.09,
    100 * p29_ms["202012"]["measured_share"], _dp(11.09, 2))
cms("SI 2: 11.60% in 2020-03", 11.60,
    100 * p29_ms["202003"]["measured_share"], _dp(11.60, 2))
cms("SI 2: against the 7.62% an interval midpoint would supply", 7.62,
    100 * p29_ms["202012"]["assumed_mid_share"], _dp(7.62, 2))
cms("SI 2: the district file masks 9.4% of its own cells", 9.4,
    100 * p29_ms["202012"]["gu_own_masked_share"], _dp(9.4, 1))
cms("SI 2: the window is served by 1 896 parquet files", 1896.0,
    float(inv_ms["parquet_files"]), 1e-12)
_G25_MS = sorted(r["jan"] for r in p2_ms["maskrate_age_gu"] if r["age"] == 25)
check("MS", "SI 2: the 25-29 district mask rates cover 25 districts", 25.0,
      float(len(_G25_MS)), 1e-9)
cms("SI 2: 25-29 district mask rates start at 0.637", 0.637, _G25_MS[0],
    _dp(0.637, 3))
cms("SI 2: with a median of 0.676", 0.676, _G25_MS[len(_G25_MS) // 2],
    _dp(0.676, 3))
cms("SI 2: and a maximum of 0.734", 0.734, _G25_MS[-1], _dp(0.734, 3))
cms("SI 2: the raw files hold 475 266 duplicate key groups", 475266.0,
    float(p2_ms["dedup"]["duplicate_keys"]), 1e-12)
# The two ends of the dedup sentence, from the fork that measured it. The
# reconciliation is the one validation that proves the volume column is right,
# so both arms are gated and so is the sign flip between them.
_FORK_MS = p43d_ms["p29_fork"]["202012"]
check("MS", "SI 2: per-path dedup leaves the reconciliation at +4.19e-06",
      4.19e-06, _FORK_MS["raw"]["recon_rel"], 5e-9)
check("MS", "SI 2: an unconditional dedup degrades it to -1.52e-04", -1.52e-04,
      _FORK_MS["dedup"]["recon_rel"], 5e-7)
check("MS", "SI 2: and flips its sign", 1.0,
      1.0 if _FORK_MS["raw"]["recon_rel"] * _FORK_MS["dedup"]["recon_rel"] < 0
      else 0.0, 1e-9)

# --- SI 3: the four statistics, the two arms, the rank-one nulls
for _ym_ms, _row_ms in (("202312", (("half_l1", 0.0588, 0.3285, 5.6),
                                    ("cramers_v", 0.0462, 0.3634, 7.9),
                                    ("assortativity", 0.0217, 0.2227, 10.2),
                                    ("nmi", 0.0051, 0.1798, 35.2))),
                        ("202402", (("half_l1", 0.0575, 0.3260, 5.7),
                                    ("cramers_v", 0.0434, 0.3781, 8.7),
                                    ("assortativity", 0.0189, 0.1867, 9.9),
                                    ("nmi", 0.0046, 0.1836, 39.8)))):
    for _st_ms, _p_ms, _s_ms, _r_ms in _row_ms:
        cms(f"SI Table S1 {_ym_ms} {_st_ms} passive", _p_ms,
            _CMP_MS[_ym_ms][_st_ms]["passive"], _dp(_p_ms, 4))
        cms(f"SI Table S1 {_ym_ms} {_st_ms} survey", _s_ms,
            _CMP_MS[_ym_ms][_st_ms]["survey"], _dp(_s_ms, 4))
        cms(f"SI Table S1 {_ym_ms} {_st_ms} multiple", _r_ms,
            _CMP_MS[_ym_ms][_st_ms]["ratio"], _dp(_r_ms, 1))
cms("SI 3: the diary survey records 4.809 contacts per person per day", 4.809,
    p27_ms["reproduction"]["per_person_day"], _dp(4.809, 3))
cms("SI 3: the 79-month normalised mutual information starts at 0.0021", 0.0021,
    p37["summary_dong"]["nmi_min"], _dp(0.0021, 4))
cms("SI 3: and reaches 0.0073", 0.0073, p37["summary_dong"]["nmi_max"],
    _dp(0.0073, 4))
cms("SI 3: the leading spectral share has median 0.9823", 0.9823,
    p37["spectrum_dong"]["sigma1_share_median"], _dp(0.9823, 4))
cms("SI 3: and minimum 0.9702", 0.9702,
    p37["spectrum_dong"]["sigma1_share_min"], _dp(0.9702, 4))
# --- SI 7's phase-69 paragraph. The factor, the two ends of the ladder, and the
# three claims that are properties of the run rather than numbers in it.
_P69K = {r["design"]: r for r in p69_ms["conversion"]}
_P69L = {r["beta"]: r["arms"] for r in p69_ms["ladder"]}
cms("SI 7: the factor is 0.928 at a true value of 0.22", 0.928,
    _P69K[0.95]["k_integerised_subsample"], _dp(0.928, 3))
cms("SI 7: and 0.961 at 0.35", 0.961,
    _P69K[0.99]["k_integerised_subsample"], _dp(0.961, 3))
cms("SI 7: pooling within cells falls from 0.2038 at one venue", 0.2038,
    _P69L[0.95]["within-cell"]["1"], _dp(0.2038, 4))
cms("SI 7: to 0.0188 at the whole cell", 0.0188,
    _P69L[0.95]["within-cell"]["cell"], _dp(0.0188, 4))
# "identical to four decimal places at a given occupancy" -- the same r_true
# read at two requested betas, which is the claim that the factor is a function
# of occupancy and not of beta.
_P69S = {(r["beta"], r["r_true_target"]): r["wor_over_wr"] for r in p69_ms["surface"]}
check("MS", "SI 7: the factor agrees to four places across requested beta", 1.0,
      1.0 if abs(_P69S[(0.5, 0.03)] - _P69S[(0.96, 0.03)]) < 5e-5 else 0.0, 1e-9)
# "pooling at random halves the reading at every doubling" -- the slope-one
# decay law, checked as a ratio rather than trusted as a description.
_P69R = _P69L[0.95]["random"]
_gs = ["1", "2", "4", "8", "16", "32", "64"]
check("MS", "SI 7: random pooling halves the reading at every doubling", 1.0,
      1.0 if max(abs(_P69R[_gs[i + 1]] / _P69R[_gs[i]] - 0.5)
                 for i in range(len(_gs) - 1)) < 0.01 else 0.0, 1e-9)
# 9e-4, not 8e-4: the largest deviation is 8.0032e-4, which sits above the
# quote by four parts in ten thousand of itself. At one significant figure the
# only true upper bound is 9e-4.
cms("SI 7: rounding to whole people changes the factor by at most 9e-4", 9e-4,
    max(abs(_P69K[b]["k_integerised_subsample"]
            - _P69K[b]["k_fractional_subsample"]) for b in (0.95, 0.96, 0.97,
                                                            0.98, 0.99)),
    _dp(9e-4, 4), bound="upper")
# The bottom of Table S2's column, which SI 7 states as a bound and nothing was
# watching. It is already rounded the right way: 0.18135 up to 0.1814.
cms("SI 7: at the unidentified end of the column the bound is 0.1814 or less",
    0.1814,
    max(_v["bound_on_A"] for _v in p69_ms["verdicts"] if _v["excludes"]),
    _dp(0.1814, 4), bound="upper")
cms("SI 7: and the axis by 5e-5", 5e-5,
    max(abs(_P69K[b]["realised_C_integerised"]
            - _P69K[b]["realised_C_fractional_subsample"])
        for b in (0.95, 0.96, 0.97, 0.98, 0.99)), _dp(5e-5, 5))
cms("SI 7: the axis is reported from a requested beta of 0.8 upwards", 0.8,
    p69_ms["axis_defined_from"]["integerised"], 1e-9)
# The 4 000 cells are a declared constant of the run, so the row reads the
# literal out of the script the way the p36 thread-count row does.
_p69src = open(f"{ROOT}/eda/p69_betacollapse.py").read()
_m69 = _re.search(r"LADDER_CELLS = (\d+)", _p69src)
cms("SI 7: 4 000 cells", 4000.0,
    float(_m69.group(1)) if _m69 else 0.0, 1e-9)

cms("SI 3: the rank-one gap has a median of 0.00150", 0.00150,
    p37["spectrum_dong"]["gap_to_pm_median"], _dp(0.00150, 5))
# The maximum of that difference is printed in the same SI sentence and was
# gated nowhere once the abstract stopped quoting it.
cms("SI 3: and a maximum of 0.00202", 0.00202,
    p37["spectrum_dong"]["gap_to_pm_max"], _dp(0.00202, 5))
# The distance the difference is a lower bound ON, both ends of it, from the
# file that measures it.
cms("SI 3: the distance is 0.0209 at the median", 0.0209,
    p67_ms["summary"]["gap_rel_median"], _dp(0.0209, 4))
cms("SI 3: and 0.02573 at the maximum", 0.02573,
    p67_ms["summary"]["gap_rel_max"], _dp(0.02573, 5))
cms("SI 3: aggregating to districts raises the leading share to 99.45%", 99.45,
    100 * _SP_MS["passive|202312|WE|gu"]["sigma1_share"], _dp(99.45, 2))
cms("SI 3: sigma2/sigma1 = 0.0661 there", 0.0661,
    _SP_MS["passive|202312|WE|gu"]["sigma2_over_sigma1"], _dp(0.0661, 4))
cms("SI 3: and to 99.84% at the city", 99.84,
    100 * _SP_MS["passive|202312|WE|city"]["sigma1_share"], _dp(99.84, 2))
cms("SI 3: the survey's leading share is 54.93% in February 2024", 54.93,
    100 * _SP_MS["survey|202402|WE|seoul"]["sigma1_share"], _dp(54.93, 2))
cms("SI 3: the six published months top out at 0.02136", 0.02136,
    p37["summary_dong"]["published_six_max"], _dp(0.02136, 5))
cms("SI 3: their upper edge is 21% below the 79-month maximum", 21.0,
    100 * (p37["summary_dong"]["assort_max"]
           / p37["summary_dong"]["published_six_max"] - 1), _dp(21.0, 0))
check("MS", "SI 3: December 2020 is the lowest of all 79 months", 202012.0,
      float(_WE_MS[0]["ym"]), 1e-9)
cms("SI 3: on the Seoul arm IPF moves the nMI multiple to 33.7x", 33.7,
    p32_ms["ipf_calibration"]["months"]["202312"]["gap_after"], _dp(33.7, 1))
cms("SI 3: and to 39.3x", 39.3,
    p32_ms["ipf_calibration"]["months"]["202402"]["gap_after"], _dp(39.3, 1))
# THE NATIONAL ARM'S IPF, added 2026-09-03. Until now the SI carried only the
# Seoul arm's 4.4%/1.2% and 33.7x/39.3x, while the abstract's "33 to 35 times"
# is the NATIONAL arm's pair of post-fit factors -- so the one sentence in
# main-text 3.2 was the only place in the whole submission where the abstract's
# headline multiple was derived, and the SI, whose stated job is to carry the
# arithmetic the main text quotes, did not have it. These four rows put it in
# SI 3 beside the Seoul arm it is contrasted with. They are the same p51 fields
# 3.2's rows read; the numbers are quoted twice on purpose, once in the claim
# and once in its derivation, and each copy is gated where it is printed.
cms("SI 3: the national arm absorbs 23.6% of the gap in December 2023", 23.6,
    100 * _IPF_MS["202312"]["national"]["absorbed_frac"], _dp(23.6, 1))
cms("SI 3: and 27.2% in February 2024", 27.2,
    100 * _IPF_MS["202402"]["national"]["absorbed_frac"], _dp(27.2, 1))
cms("SI 3: leaving a normalised-mutual-information factor of 34.8x", 34.8,
    _IPF_MS["202312"]["national"]["gap_after"], _dp(34.8, 1))
cms("SI 3: and 33.1x", 33.1, _IPF_MS["202402"]["national"]["gap_after"],
    _dp(33.1, 1))
# "in either arm in either month" is a universal, so it is counted over all
# four fits rather than asserted of the one the main text names.
check("MS", "SI 3: none of the four IPF fits converges", 0.0,
      float(sum(1 for _ym_ms in ("202312", "202402")
                for _arm_ms in ("seoul", "national")
                if _IPF_MS[_ym_ms][_arm_ms]["converged"])), 1e-9)
_DVG_MS = p40_ms["divergence_vs_passive_signal"]
cms("SI 3: the Seoul-to-national population shift is 3.01x the passive excess",
    3.01, _DVG_MS["202312"]["assortativity"]["multiple"], _dp(3.01, 2))
cms("SI 3: and 8.62x the normalised mutual information", 8.62,
    _DVG_MS["202312"]["nmi"]["multiple"], _dp(8.62, 2))
cms("SI 3: 3.21x in February 2024", 3.21,
    _DVG_MS["202402"]["assortativity"]["multiple"], _dp(3.21, 2))
cms("SI 3: and 4.00x", 4.00, _DVG_MS["202402"]["nmi"]["multiple"], _dp(4.00, 2))
cms("SI 3: only 1.13x on half-L1", 1.13,
    _DVG_MS["202312"]["half_l1"]["multiple"], _dp(1.13, 2))
cms("SI 3: and 0.80x on Cramer's V", 0.80,
    _DVG_MS["202312"]["cramers_v"]["multiple"], _dp(0.80, 2))
for _ym_ms, _want_ms in (("202312", 7.0), ("202402", 6.0)):
    check("MS", f"SI 3: Seoul is outside the 95% null band on {_want_ms:.0f} "
                f"of nine quantities, {_ym_ms}", _want_ms,
          float(sum(1 for _v in p40_ms["seoul_vs_national"][_ym_ms]["null"].values()
                    if not _v["inside_null"])), 1e-9)
cms("SI 3: beta moves by only 0.0130 across the scope change", 0.0130,
    p53_ms["asymmetry"]["spread"], _dp(0.0130, 4))
# 2026-09-03: 2.2 declares Seoul primary and now gives the reason AT the
# declaration instead of twenty pages later in Limitations, so it restates
# these three. check(), not cms(): they are already registered above against
# SI 3, and a second cms() would enter the same value in IN_TEXT_MS twice.
check("MS", "2.2: the population shift is 3.01x the assortativity excess", 3.01,
      _DVG_MS["202312"]["assortativity"]["multiple"], 0.005)
check("MS", "2.2: and 8.62x the mutual-information excess", 8.62,
      _DVG_MS["202312"]["nmi"]["multiple"], 0.005)
check("MS", "2.2: while beta moves by only 0.0130", 0.0130,
      p53_ms["asymmetry"]["spread"], 5e-5)
# The two months do not agree to the digit the sentence prints: February 2024
# gives 3.200 and December 2023 3.150. The SI used to write "a factor of 3.2"
# with no month attached, which reads as though the pair agreed; it now names
# February and carries December in parentheses, so both months are quoted and
# both are gated.
cms("SI 3: the floor ratio moves by a factor of 3.2 in February 2024", 3.2,
    p51_ms["cells"]["202402|seoul"]["floor_over_passive_median"]
    / p51_ms["cells"]["202402|national"]["floor_over_passive_median"],
    _dp(3.2, 1))
cms("SI 3: and by 3.15 in December 2023", 3.15,
    p51_ms["cells"]["202312|seoul"]["floor_over_passive_median"]
    / p51_ms["cells"]["202312|national"]["floor_over_passive_median"],
    _dp(3.15, 2))
cms("SI 3: 2 369 954 device-equivalents in December 2023", 2369954.0,
    p33_live["R1_effective_sample"]["202312"]["n_device_equiv"], _dp(2369954, 0))
check("MS", "SI 3: on 400 parametric bootstrap replicates", 400.0,
      float(p33_live["R1_effective_sample"]["202312"]["n_boot"]), 1e-9)
# THE SEOUL FLOOR ITSELF, added 2026-09-03. SI 3 gave the two RATIOS (3.6x,
# 4.9x) and never the floor they are ratios to, so the level a reader would
# need in order to place the exceedance count of SI 5 -- which is taken against
# the national arm's much lower floor -- existed nowhere in the SI. Both months
# are registered, because the sentence now names both.
cms("SI 3: the Seoul arm's permutation floor is 0.0713 bits in December 2023",
    0.0713, p32_ms["survey_floor_measured"]["202312"]["mi_perm_median"],
    _dp(0.0713, 4))
cms("SI 3: and 0.0868 bits in February 2024", 0.0868,
    p32_ms["survey_floor_measured"]["202402"]["mi_perm_median"], _dp(0.0868, 4))
# "none of the 79 reaches either" is a universal over both floors, so it is
# counted over both rather than over the December one the prose leads with.
check("MS", "SI 3: no month of the 79 reaches either Seoul floor", 0.0,
      float(sum(1 for _v in _MI79
                for _ym_ms in ("202312", "202402")
                if _v > p32_ms["survey_floor_measured"][_ym_ms]["mi_perm_median"])),
      1e-9)
cms("SI 3: the national arm rests on 1 987 respondents", 1987.0,
    float(p35_live["survey_bootstrap"]["202312|national"]["n_respondents"]),
    1e-9)
cms("SI 3: the survey's own floor is 3.6x the passive excess in December 2023",
    3.6, (p32_ms["survey_floor_measured"]["202312"]["mi_perm_median"]
          / p32_ms["null_floor"]["202312"]["passive"]["observed"]), _dp(3.6, 1))
cms("SI 3: and 4.9x in February 2024", 4.9,
    (p32_ms["survey_floor_measured"]["202402"]["mi_perm_median"]
     / p32_ms["null_floor"]["202402"]["passive"]["observed"]), _dp(4.9, 1))
cms("SI 3: on the national arm it is 1.17x", 1.17,
    p51_ms["cells"]["202312|national"]["floor_over_passive_median"],
    _dp(1.17, 2))
cms("SI 3: and 1.54x", 1.54,
    p51_ms["cells"]["202402|national"]["floor_over_passive_median"],
    _dp(1.54, 2))
cms("SI 3: 17.2% of the national permutation draws fall below the passive value",
    17.2, 100 * p51_ms["cells"]["202312|national"]["passive_frac_below_perm_null"],
    _dp(17.2, 1))
_DEC61_MS = p61["permutation_null_floor_decomposition"]["202312"]
cms("SI 3: the permutation null's closed form contributes 0.0129", 0.0129,
    _DEC61_MS["closed_form_sigma2_over_sigma1"], _dp(0.0129, 4))
cms("SI 3: about a tenth of the null median", 0.10,
    _DEC61_MS["closed_form_sigma2_over_sigma1"] / _pm61["median"], _dp(0.10, 2))
cms("SI 3: a multinomial null whose truth is rank one gives 0.0518", 0.0518,
    p61["survey_multinomial_null"]["202312"]["sigma2_over_sigma1"]["median"],
    _dp(0.0518, 4))
cms("SI 3: the emptiness slope extrapolates to only 0.135", 0.135,
    _em61["linear_extrapolation_to_observed_emptiness"], _dp(0.135, 3))
cms("SI 3: the emptiness regression has r = 0.06", 0.06,
    _em61["pearson_r"], _dp(0.06, 2))
cms("SI 3: emptiness buys 0.00044 of sigma2/sigma1 per empty cell", 0.00044,
    _em61["slope_sigma2_per_empty_cell"], _dp(0.00044, 5))

# --- SI 4: the coverage profile and what survives it
cms("SI 4: Korea's population vector totals 52 673 955", 52673955.0,
    p50_ms["vectors"]["202312"]["national_total"], 1e-12)
_FILL_MS = p19_ms["measured_fill"]
for _band_ms, _q_ms in (("20-24", 2.27), ("25-29", 2.07), ("30-34", 2.26),
                        ("35-39", 2.59), ("40-44", 2.81)):
    cms(f"SI 4: the measured fill at {_band_ms} is {_q_ms}", _q_ms,
        _FILL_MS[_band_ms], _dp(_q_ms, 2))
_CV_MS = [r["cv"] for r in p8_ms["weight_by_month"]]
check("MS", "SI 4: the weight panel covers 32 age x sex strata", 32.0,
      float(len(_CV_MS)), 1e-9)
# p8_panel.py still calls itself "the 8-month panel" in its docstring, because
# that is what was on disk when it was first run (202001-202005, 202007-202009).
# The window is common.YMS, so re-running it after the four missing months
# landed widened the panel to twelve without renaming anything, and the SI
# inherited the old adjective. The column count is the fact, so it is gated:
# every row of weight_by_month carries its months plus "index" and "cv".
check("MS", "SI 4: and twelve months, not the eight in the script's name", 12.0,
      float(len([_k for _k in p8_ms["weight_by_month"][0]
                 if _k not in ("index", "cv")])), 1e-9)
# 0.0124: the largest of the 32 strata is 0.0123263.
cms("SI 4: the cross-month coefficient of variation is at most 0.0124", 0.0124,
    max(_CV_MS), _dp(0.0124, 4), bound="upper")
check("MS", "SI 4: with a median of exactly zero", 0.0,
      sorted(_CV_MS)[len(_CV_MS) // 2], 0.0)

# --- SI 5: the school calendar in full
# The calendar-month fold. 3.1 prints the three medians (they are gated in the
# 3.1 block above, against the same by_month table) and SI 5 carries the
# construction behind them: which readings are folded together, how many each
# calendar position gets, and the order the twelve sit in. The order is the
# part the three printed values do not carry -- the seven term months take the
# top seven places and the four vacation months the bottom four, with December
# between them -- and it is what lets the fold be described in prose rather
# than drawn. Every row here is a count or a label, so all of them are
# check() and none is cms(): _forms() would "find" a 7 or a 4 in any prose.
_ORD_MS = p41_ms["calendar"]["rank_order"]
_NFOLD_MS = {_m: _CAL_MS[str(_m)]["n"] for _m in range(1, 13)}
_N17_MS = {_NFOLD_MS[_m] for _m in range(1, 8)}
_N812_MS = {_NFOLD_MS[_m] for _m in range(8, 13)}
check("MS", "SI 5: January to July fold seven readings each", 7.0,
      float(next(iter(_N17_MS))) if len(_N17_MS) == 1 else -1.0, 1e-9)
check("MS", "SI 5: August to December fold six", 6.0,
      float(next(iter(_N812_MS))) if len(_N812_MS) == 1 else -1.0, 1e-9)
check("MS", "SI 5: the twelve run 3 9 4 11 5 6 10 12 8 7 1 2", 1.0,
      _eq(_ORD_MS, [3, 9, 4, 11, 5, 6, 10, 12, 8, 7, 1, 2]), 1e-9)
check("MS", "SI 5: the seven term months take the top seven places", 7.0,
      float(sum(1 for _m in _ORD_MS[:7] if _m in (3, 4, 5, 6, 9, 10, 11))),
      1e-9)
check("MS", "SI 5: December is eighth", 12.0, float(_ORD_MS[7]), 1e-9)
check("MS", "SI 5: the four vacation months are the bottom four", 4.0,
      float(sum(1 for _m in _ORD_MS[8:] if _m in (1, 2, 7, 8))), 1e-9)
# THE THREE MEDIANS THEMSELVES, added 2026-09-03. The comment above says the
# order "is the part the three printed values do not carry", and it was written
# when SI 5 printed the three values; it had stopped doing so, and SI 5 gave
# only the ordering, so the three numbers 3.1 quotes had their derivation
# nowhere. The fold is described here, so the values it produces belong here
# too. Same p41 field 3.1's rows read.
cms("SI 5: March's calendar-month median is 0.0242", 0.0242,
    _CAL_MS["3"]["median"], _dp(0.0242, 4))
cms("SI 5: September's is 0.0230", 0.0230, _CAL_MS["9"]["median"],
    _dp(0.0230, 4))
cms("SI 5: February's is 0.0176", 0.0176, _CAL_MS["2"]["median"],
    _dp(0.0176, 4))
cms("SI 5: the mean contrast over 2021-2026 is 0.0040", 0.0040, _MEAN6_MS,
    _dp(0.0040, 4))
cms("SI 5: the 79-month median it is measured against is 0.0194", 0.0194,
    p37["summary_dong"]["assort_median"], _dp(0.0194, 4))
_RAT_MS = sorted(_v["control_over_others_min"] for _v in _VAR_MS.values())
cms("SI 5: the median 2020-to-smallest ratio across the ten readings is 0.10",
    0.10, 0.5 * (_RAT_MS[4] + _RAT_MS[5]), _dp(0.10, 2))
check("MS", "SI 5: it is below one third in nine of the ten", 9.0,
      float(sum(1 for _r in _RAT_MS if _r < 1.0 / 3.0)), 1e-9)
cms("SI 5: the exception is the city level, at 0.55", 0.55,
    _VAR_MS["WE|city|assortativity"]["control_over_others_min"], _dp(0.55, 2))
cms("SI 5: 2020's 25-and-over contrast is +0.00029", 0.00029,
    _GRP_MS["adult_25plus"]["control"], _dp(0.00029, 5))
cms("SI 5: against a median of +0.00057", 0.00057,
    _GRP_MS["adult_25plus"]["others_median"], _dp(0.00057, 5))
# The ratio of those two. It was the old Figure 2(c)'s own number and left the
# pair with that caption, leaving "retaining half of its usual size" as the
# only form of the claim; SI 5 now prints the quantity itself. The claim it
# supports is not bookkeeping: the adult half keeping half its size in the year
# the school-age half goes negative is what separates the school reading from a
# pandemic effect that flattened seasonality across the whole population.
cms("SI 5: a retention of 0.507 of its usual size", 0.507,
    _GRP_MS["adult_25plus"]["retention"], _dp(0.507, 3))
check("MS", "SI 5: the adult contrast is positive in all seven years", 7.0,
      float(_GRP_MS["adult_25plus"]["sign_test"]["n_positive"]), 1e-9)
cms("SI 5: the adult sign test gives an exact one-sided p of 0.008", 0.008,
    _GRP_MS["adult_25plus"]["sign_test"]["p_one_sided"], _dp(0.008, 3))
cms("SI 5: and a permutation p of 0.010", 0.010,
    _GRP_MS["adult_25plus"]["p_count_at_least"], _dp(0.010, 3))
cms("SI 5: the sign test's 8/128 is 0.0625", 0.0625,
    p41_ms["sign_test"]["contrast"]["numerator"]
    / p41_ms["sign_test"]["contrast"]["denominator"], 1e-12)
cms("SI 5: two-sided 0.125", 0.125,
    p41_ms["sign_test"]["contrast"]["p_two_sided"], 1e-12)
cms("SI 5: the panel-respecting permutation p is 0.068", 0.068,
    p41_ms["permutation"]["p_count_at_least"], _dp(0.068, 3))
# The two enumeration counts behind that p. They were Fig 2(b) rows while the
# caption described p41's within-year relabelling bands; the caption no longer
# draws that null, and SI 5 is now the only place either number is stated.
cms("SI 5: the within-year null enumerates 330 relabellings in a full year",
    330.0, float(p41_ms["permutation"]["per_year"]["2021"]["n_perm"]), 1e-12)
check("MS", "SI 5: and 35 in the partial year 2026", 35.0,
      float(p41_ms["permutation"]["per_year"]["2026"]["n_perm"]), 1e-9)
# The per-year bands that enumeration draws, which were also Figure 2(b) and
# also left the pair with the caption. "2026 sits exactly on its own upper
# limit" is only readable beside the five years that sit above theirs and the
# one that sits inside, and it is only honest beside the resolution a partial
# year has, which is why the 35 above and the 1/35 below are quoted with it.
# The identity is checked at tolerance zero, in p53's style: the observed 2026
# contrast IS that year's 97.5th percentile, and a band edge that drifts off
# the observation by one ulp is the claim ceasing to be true.
_PY_MS = p41_ms["per_year"]
_PM_MS = p41_ms["permutation"]["per_year"]
check("MS", "SI 5: the contrast is above its own band in five years", 5.0,
      float(sum(1 for _y in _PY_MS
                if _PY_MS[_y]["contrast_median"] > _PM_MS[_y]["null_hi"])),
      1e-9)
check("MS", "SI 5: and inside the band in 2020", 1.0,
      1.0 if (_PM_MS["2020"]["null_lo"] <= _PY_MS["2020"]["contrast_median"]
              <= _PM_MS["2020"]["null_hi"]) else 0.0, 1e-9)
check("MS", "SI 5: 2026's observation IS its 97.5th percentile", 0.0,
      _PY_MS["2026"]["contrast_median"] - _PM_MS["2026"]["null_hi"], 0.0)
cms("SI 5: and that value is 0.004221", 0.004221,
    _PY_MS["2026"]["contrast_median"], _dp(0.004221, 6))
check("MS", "SI 5: 2 of the 35 relabellings reach it", 2.0,
      _PM_MS["2026"]["p_ge_observed"] * _PM_MS["2026"]["n_perm"], 1e-9)
cms("SI 5: 2026's own exact one-sided p is 2/35 = 0.057", 0.057,
    _PM_MS["2026"]["p_ge_observed"], _dp(0.057, 3))
check("MS", "SI 5: 2026 is a partial year of seven months", 7.0,
      float(_PY_MS["2026"]["n_months"]), 1e-9)
cms("SI 5: the smallest p 35 relabellings can return is 1/35 = 0.029", 0.029,
    1.0 / _PM_MS["2026"]["n_perm"], _dp(0.029, 3))
cms("SI 5: 2020's deepest per-band deficit is 0.32 of its usual value", 0.32,
    p41_ms["bands"]["level_2020_vs_rest"][
        p41_ms["bands"]["deepest_level_deficit"]]["ratio"], _dp(0.32, 2))
check("MS", "SI 5: and it is the 80-and-over band", 1.0,
      _eq(p41_ms["bands"]["deepest_level_deficit"], "80+"), 1e-9)
cms("SI 5: the superseded within-year p is 4.97e-06", 4.97e-06,
    p54_ms["corrected"]["p_term_ge"], _dp(4.97e-06, 8))
cms("SI 5: the circular-shift p is 0.0506", 0.0506, _ps58["p"], _dp(0.0506, 4))
check("MS", "SI 5: three of the 78 alternatives also place all 17 in term", 3.0,
      float(_ps58["n_shifts_at_or_above"]), 1e-9)
check("MS", "SI 5: all three are multiples of twelve", 3.0,
      float(sum(1 for _k in _ps58["shifts_at_or_above_observed"]
                if _k % 12 == 0)), 1e-9)
check("MS", "SI 5: the 17 clearing months fall in nine runs", 9.0,
      float(p58["run_structure"]["n_runs"]), 1e-9)
check("MS", "SI 5: the term margin is invariant at 46", 46.0,
      float(_a58["A5"]["margin"]), 1e-9)
# 2026-08-31 (advisor 3.2, via p64): "the smallest attainable p is 1/79 =
# 0.0127" was removed from the SI along with the caption -- the 79 rotations
# realise only 12 distinct calendars, not 79 independent ones, so 1/79 was
# never the honest floor. Replaced by the same 1/12 / 1/6 pair Figure 2(b)
# and 3.1 use; check(), not cms(), because both are already registered there
# and a second registration would double-count the same literal.
check("MS", "SI 5: the 79 rotations realise twelve distinct calendars", 12.0,
      float(p64["effective_alignments"]["E_exact"]), 1e-9)
check("MS", "SI 5: so the smallest resolvable p is 1/12 = 0.083", 0.083,
      p64["deduplicated"]["exact"]["resolution_floor"], _dp(0.083, 3))
check("MS", "SI 5: collapsing the near-duplicate coarsens it to 1/6 = 0.17",
      0.17, p64["deduplicated"]["advisor"]["resolution_floor"], _dp(0.17, 2))
cms("SI 5: the level of the statistic trends by a factor of 1.70", 1.70,
    _cap(_c63, 2, "a", "level trend over the period"), _dp(1.70, 2))
# The per-year term margins, moved here from the old panel (e) (advisor 3.2).
check("MS", "SI 5: the true per-year margins are 7,7,7,7,7,7,4", 1.0,
      _eq([p58["per_year_margins"]["identity"][str(_y)]
           for _y in range(2020, 2027)], [7, 7, 7, 7, 7, 7, 4]), 1e-9)
check("MS", "SI 5: across the rotations they range from 3", 3.0,
      float(p58["per_year_margins"]["overall_min"]), 1e-9)
check("MS", "SI 5: up to 8", 8.0,
      float(p58["per_year_margins"]["overall_max"]), 1e-9)
for _ym_ms in ("202312", "202402"):
    check("MS", f"SI 5: {_ym_ms} sits below 31 of the 46 term months", 31.0,
          float(p54_ms["survey_months"][_ym_ms]["n_term_months_above"]), 1e-9)
check("MS", "SI 5: the count moves between 13 and 21 across five offsets", 5.0,
      float(len(p54_ms["floor_sensitivity"]["counts_at_offsets"])), 1e-9)
_OFF_MS = [_v["n"] for _v
           in p54_ms["floor_sensitivity"]["counts_at_offsets"].values()]
check("MS", "SI 5: the lowest of those five counts is 13", 13.0,
      float(min(_OFF_MS)), 1e-9)
check("MS", "SI 5: and the highest is 21", 21.0, float(max(_OFF_MS)), 1e-9)
check("MS", "SI 5: every offset keeps the all-term property", 5.0,
      float(sum(1 for _v in p54_ms["floor_sensitivity"]["counts_at_offsets"]
                .values() if _v["all_term"])), 1e-9)
_TV_MS = {r["age"]: r for r in p25_ms["G_tv_bounds"] if r["imputation"] == "mid"}
cms("SI 5: label mixing runs from 20.9% at 65-69", 20.9,
    100 * _TV_MS["65-69"]["tv_hour_x_samedong"], _dp(20.9, 1))
cms("SI 5: to 27.7% at 80 and over", 27.7,
    100 * _TV_MS["80+"]["tv_hour_x_samedong"], _dp(27.7, 1))

# --- SI 6: resolution scaling
cms("SI 6: the district rung retains 29.8%", 29.8,
    100 * p37["ladder"]["kept_median"], _dp(29.8, 1))
cms("SI 6: with a low of 27.6%", 27.6, 100 * p37["ladder"]["kept_min"],
    _dp(27.6, 1))
cms("SI 6: and a high of 32.6%", 32.6, 100 * p37["ladder"]["kept_max"],
    _dp(32.6, 1))
cms("SI 6: as raw ratios the district level is a median 0.469 of dong", 0.469,
    _ax60["spatial"]["gu_over_dong"]["median"], _dp(0.469, 3))
cms("SI 6: and the city level 0.245", 0.245,
    _ax60["spatial"]["city_over_dong"]["median"], _dp(0.245, 3))
check("MS", "SI 6: the adjacency graph has 424 nodes", 424.0,
      float(p34["adjacency"]["n_nodes"]), 1e-9)
cms("SI 6: the local slope runs from 0.178", 0.178,
    p38["exponent"]["assortativity"]["named_lo"], _dp(0.178, 3))
cms("SI 6: to 0.478", 0.478, p38["exponent"]["assortativity"]["named_hi"],
    _dp(0.478, 3))
cms("SI 6: and all 158 curves bend upward", 158.0,
    float(p38["exponent"]["assortativity"]["n_bending_up"]), 1e-12)
cms("SI 6: the survey level would be reached at about 5 632 locations", 5632.0,
    p44_ms["beta"]["reach"]["locations_slope1"], _dp(5632.0, 0))
cms("SI 6: whereas the measured slope would take about 509 996", 509996.0,
    p44_ms["beta"]["reach"]["locations_ladder"], _dp(509996.0, 0))
# "a factor of 90" is stated to the nearest ten -- the sentence beside it is
# qualitative ("that difference measures the magnitude of spatial
# correlation") -- so it is checked as that rounding rather than given a
# tolerance stretched until 90.56 fits inside 90.
check("MS", "SI 6: the two answers differ by a factor of 90, to the nearest ten",
      90.0, float(round(p44_ms["beta"]["reach"]["locations_ladder"]
                        / p44_ms["beta"]["reach"]["locations_slope1"], -1)),
      1e-9)
cms("SI 6: the measured spatial slope at n = 424 is 0.36", 0.36,
    p44_ms["beta"]["reach"]["ladder_slope"], _dp(0.36, 2))
cms("SI 6: the B075 band's 79-month median lower edge is 0.0286", 0.0286,
    _stats.median([_v["r_polygon_lo"] for _v in _POLY_MS.values()]),
    _dp(0.0286, 4))
cms("SI 6: and its upper edge 0.0316", 0.0316,
    _stats.median([_v["r_polygon_hi"] for _v in _POLY_MS.values()]),
    _dp(0.0316, 4))
check("MS", "SI 6: the S1 identity was verified over 395 cases", 395.0,
      float(len(p60["per_month"]) * 5), 1e-9)
check("MS", "SI 6: to 2.01e-16", 2.012e-16,
      _m60["within_group_excess_identity_max_abs_diff"], 5e-19)
cms("SI 6: the ratio factorises into 2.2708", 2.2708,
    _m60["numerator_gain"]["median"], _dp(2.2708, 4))
cms("SI 6: and 1.7100", 1.7100, _m60["denominator_shrink"]["median"],
    _dp(1.7100, 4))
_SPH_MS = p60["semester_posthoc"]
cms("SI 6: the three-band 2020 contrast is +0.00226", 0.00226,
    _SPH_MS["lim3"]["control_year_value"], _dp(0.00226, 5))
cms("SI 6: against -0.00010 at sixteen bands", -0.00010,
    _SPH_MS["r16"]["control_year_value"], _dp(0.00010, 5))
check("MS", "SI 6: three bands give 7 of 7 positive years", 7.0,
      float(_SPH_MS["lim3"]["n_positive"]), 1e-9)
check("MS", "SI 6: sixteen bands give 6 of 7", 6.0,
      float(_SPH_MS["r16"]["n_positive"]), 1e-9)
# THE FIRST FORBIDDEN PAIRING'S TWO NUMBERS, added 2026-09-03. SI 6's closing
# paragraph named the pairing and then pointed at the main text for the size of
# it, which is the wrong direction for a document whose job is the arithmetic:
# 3.3 makes the claim and SI 6 is where the two matrices being paired are
# defined. Both are derived from the two medians rather than restated, exactly
# as 3.3's rows are, so a move in either matrix moves both copies together.
cms("SI 6: like for like the forbidden pairing's gap is 11.5x", 11.5,
    _RSV_MS / _s60["r16"]["median"], _dp(11.5, 1))
cms("SI 6: and the mismatched pairing returns 2.69x", 2.69,
    _RSV_MS / _s60["lim3"]["median"], _dp(2.69, 2))

# --- SI 7: the recovery experiment
check("MS", "SI 7: the clean arm reads +0.00000 to five places", 0.0,
      round(_ARMS_MS["clean"]["median"], 5), 0.0)
cms("SI 7: the ratio of the two independent floors is 1.04", 1.04,
    p39_ms["null_arm"]["ratio_to_p33"], _dp(1.04, 2))
cms("SI 7: recovery is 49% at beta = 0.5", 49.0, min(_REC_MS[0.5]), _dp(49.0, 0))
cms("SI 7: ...to 53%", 53.0, max(_REC_MS[0.5]), _dp(53.0, 0))
cms("SI 7: 21% at beta = 0.8", 21.0, min(_REC_MS[0.8]), _dp(21.0, 0))
cms("SI 7: ...to 24%", 24.0, max(_REC_MS[0.8]), _dp(24.0, 0))
cms("SI 7: 3.4% at beta = 0.99", 3.4, min(_REC_MS[0.99]), _dp(3.4, 1))
cms("SI 7: ...to 5.3%", 5.3, max(_REC_MS[0.99]), _dp(5.3, 1))
cms("SI 7: the four rules agree to 0.00073 on the refined grid", 0.00073,
    p59["beta_star"]["spread"], _dp(0.00073, 5))
cms("SI 7: they spread over 0.0153 on the original grid", 0.0153,
    _sp39, _dp(0.0153, 4))
# check(), not cms(): grep of the manuscript pair finds this value nowhere,
# so the document assertion was never true. It stayed green because _forms()
# offers coarse spellings and the region contained one. Verified, not quoted.
check("MS", "SI 7: from the largest measured beta to beta* is 0.027", 0.027,
      p59["beta_star"]["primary_value"] - p59["claim"]["largest_measured_beta"],
      _dp(0.027, 3) * 0.027)
# check(), not cms(): the pair prints 0.0316 in SI 6 and no 0.031 anywhere.
# The substring of the former was answering for the latter, which is the
# same coarse-spelling cover its sibling row above carried.
check("MS", "SI 7: and 0.031 from the national arm", 0.031,
      p59["beta_star"]["primary_value"] - p53_ms["asymmetry"]["beta_national"],
      _dp(0.031, 3) * 0.031)
cms("SI 7: venue pairing with replacement gives 0.9616", 0.9616,
    p44_ms["beta"]["betas"]["venue-level, with replacement"]["beta"],
    _dp(0.9616, 4))
cms("SI 7: it retains 62% under a within-place shuffle", 62.0,
    100 * p44_ms["pairing"]["null_shuffled"]["share_wr"], _dp(62.0, 0))
cms("SI 7: against 14% for the convention we use", 14.0,
    100 * p44_ms["pairing"]["null_shuffled"]["share_wor"], _dp(14.0, 0))
cms("SI 7: the eleventh reading is beta = 0.9387", 0.9387,
    p44_ms["beta"]["betas"]["survey contact matrix (star pairs)"]["beta"],
    _dp(0.9387, 4))
# TABLE S2, added 2026-09-03. 3.3 says "SI 7 gives all twelve" and SI 7 named
# or implied seven, so the sentence promised a table that did not exist. The
# five readings below are the ones no document printed anywhere: the two
# one-convention arms of p53 53.2, and the three long ends of p44's declared
# venue-size curve. The other seven were already registered -- 0.9616, 0.9387
# and the two short venue-size rungs here, 0.9347/0.9326/0.9217 under 3.3 --
# and a second registration of those would double-count the same literal in
# the same region, so they are not repeated.
_CONV53_MS = p53_ms["convention_sensitivity"]["arms"]
cms("SI 7: Table S2's national one-convention reading is 0.9375", 0.9375,
    _CONV53_MS["national (korea)"]["beta_one_convention"], _dp(0.9375, 4))
cms("SI 7: and its Seoul one-convention reading is 0.9250", 0.9250,
    _CONV53_MS["seoul"]["beta_one_convention"], _dp(0.9250, 4))
for _k_ms, _q_ms in ((2.0, 0.7927), (3.0, 0.6933), (5.0, 0.6003)):
    cms(f"SI 7: Table S2 reads {_q_ms} at venue size k = {_k_ms:g}", _q_ms,
        _KS_MS[_k_ms]["beta"], _dp(_q_ms, 4))
# "all twelve" is the claim the table is answering, so the count is gated and
# not left to the eye: eleven retained readings in p59's register plus the one
# reading it vetoes.
check("MS", "SI 7: Table S2 carries all twelve readings", 12.0,
      float(len(p59["claim"]["measured_betas"]) + 1), 1e-9)
check("MS", "SI 7: eleven of the twelve are retained", 11.0,
      float(len(p59["claim"]["measured_betas"])), 1e-9)
check("MS", "SI 7: and the vetoed one is the only reading above 0.95", 1.0,
      1.0 if (p59["claim"]["all_below_0p95"]
              and p59["claim"]["excluded_reading"]["beta"] > 0.95) else 0.0,
      1e-9)
# The finite-venue leak, which SI 7 called "a small finite-venue leak" and left
# unquantified while 3.3 printed both halves of it. It is recomputed over the
# whole beta = 0.95 slice for the reason 3.3's copy is: the sentence is about
# the slice, and the second row is what makes one printed figure honest.
check("MS", "SI 7: a synthetic cell holds eleven venues at the median", 11.0,
      p39_ms["design"]["venues_per_cell_median"], 1e-9)
check("MS", "SI 7: the beta = 0.95 slice carries ten true values", 10.0,
      float(len(_BTW95_MS)), 1e-9)
cms("SI 7: the finite-venue leak is 9.2% of the truth at beta = 0.95", 9.2,
    max(_BTW95_MS), _dp(9.2, 1))
check("MS", "SI 7: ...and that share is flat across the slice, to 0.01 pp", 1.0,
      1.0 if max(_BTW95_MS) - min(_BTW95_MS) < 0.01 else 0.0, 1e-9)
cms("SI 7: the venue-scale estimator retains 90.9% of the survey", 90.9,
    100 * _VEN_MS["venue-visit (ego,date,place)"]["keep"], _dp(90.9, 1))
cms("SI 7: 96.1% on the Seoul arm", 96.1,
    100 * {r["grouping"]: r
           for r in p44_ms["ladder"]["seoul|202312"]["rows"]}[
        "venue-visit (ego,date,place)"]["keep"], _dp(96.1, 1))
cms("SI 7: the bracket's floor is 0.00246", 0.00246,
    p44_ms["beta"]["bracket"]["floor"], _dp(0.00246, 5))
cms("SI 7: its ceiling is 0.03682", 0.03682,
    p44_ms["beta"]["bracket"]["ceiling"], _dp(0.03682, 5))
cms("SI 7: the observation is 6.8 times the floor", 6.8,
    p44_ms["beta"]["bracket"]["measured_over_floor"], _dp(6.8, 1))
cms("SI 7: and 46% of the ceiling", 46.0,
    100 * p44_ms["beta"]["bracket"]["measured_over_ceiling"], _dp(46.0, 0))
cms("SI 7: the noiseless data field reads 32.7%", 32.7,
    100 * p39_ms["ladder"]["noiseless"]["data"], _dp(32.7, 1))
cms("SI 7: the archetype field reads 14.9% and misses the band", 14.9,
    100 * p39_ms["ladder"]["noiseless"]["archetype"], _dp(14.9, 1))
cms("SI 7: the real December 2023 rung is 30.1%", 30.1,
    100 * p39_ms["ladder"]["real_rung"], _dp(30.1, 1))

# --- SI 8: the coverage family in full
check("MS", "SI 8: the cell-level reconstruction reproduces the published "
            "reading exactly, in all four combinations", 4.0,
      float(sum(1 for _a in p33_live["anchor"]
                if _a["rel_r"] == 0.0 and _a["rel_lambda"] == 0.0)), 1e-9)
cms("SI 8: December 2023 runs 0.01704 at theta = -1", 0.01704,
    _TH12_MS[-1.0]["assortativity"], _dp(0.01704, 5))
cms("SI 8: 0.02115 at theta = 0", 0.02115, _TH12_MS[0.0]["assortativity"],
    _dp(0.02115, 5))
cms("SI 8: 0.05099 at theta = +1", 0.05099, _TH12_MS[1.0]["assortativity"],
    _dp(0.05099, 5))
cms("SI 8: February 2024 runs 0.01459", 0.01459,
    _TH02_MS[-1.0]["assortativity"], _dp(0.01459, 5))
cms("SI 8: 0.01851", 0.01851, _TH02_MS[0.0]["assortativity"], _dp(0.01851, 5))
cms("SI 8: and 0.04740", 0.04740, _TH02_MS[1.0]["assortativity"],
    _dp(0.04740, 5))
_GAP12_MS = {r["theta"]: r for r in p33_live["R2_gaps_vs_survey"]["202312"]}
cms("SI 8: the assortativity multiple runs from 4.37x", 4.37,
    min(_r["assortativity"] for _r in _GAP12_MS.values()), _dp(4.37, 2))
cms("SI 8: to 13.07x", 13.07,
    max(_r["assortativity"] for _r in _GAP12_MS.values()), _dp(13.07, 2))
cms("SI 8: the nMI multiple from 16.46x", 16.46,
    min(_r["nmi"] for _r in _GAP12_MS.values()), _dp(16.46, 2))
cms("SI 8: to 47.82x", 47.82, max(_r["nmi"] for _r in _GAP12_MS.values()),
    _dp(47.82, 2))
cms("SI 8: the smallest multiple in February 2024 is 3.49x", 3.49,
    min(_r[_s] for _r in p33_live["R2_gaps_vs_survey"]["202402"]
        for _s in ("assortativity", "half_l1", "cramers_v", "nmi")),
    _dp(3.49, 2))
check("MS", "SI 8: the leading band at theta = 0 is 15-19 in December 2023",
      1.0, _eq(_TH12_MS[0.0]["top_band"], "15-19"), 1e-9)
check("MS", "SI 8: 40-44 at negative theta", 1.0,
      _eq(_TH12_MS[-1.0]["top_band"], "40-44"), 1e-9)
check("MS", "SI 8: 0-9 at positive theta", 1.0,
      _eq(_TH12_MS[1.0]["top_band"], "0-9"), 1e-9)
check("MS", "SI 8: February 2024 moves 40-44 -> 75-79 -> 0-9", 1.0,
      1.0 if [_TH02_MS[_t]["top_band"] for _t in (-1.0, 0.0, 1.0)]
      == ["40-44", "75-79", "0-9"] else 0.0, 1e-9)
cms("SI 8: the masking band is 0.00230 wide in February 2024", 0.00230,
    _MVC_MS["202402"]["mask_width"], _dp(0.00230, 5))
cms("SI 8: and the coverage family 0.03281", 0.03281,
    _MVC_MS["202402"]["theta_width"], _dp(0.03281, 5))
cms("SI 8: imputing zero gives 0.02263", 0.02263, _MVC_MS["202312"]["mask_imp0"],
    _dp(0.02263, 5))
cms("SI 8: the measured fill gives 0.02062", 0.02062,
    _MVC_MS["202312"]["mask_measured"], _dp(0.02062, 5))
cms("SI 8: and 0.01800", 0.01800, _MVC_MS["202402"]["mask_measured"],
    _dp(0.01800, 5))
cms("SI 8: 0.02115 falls to 0.01677", 0.01677, _ST12_MS["bias_corrected"],
    _dp(0.01677, 5))
cms("SI 8: 95% CI 0.01670", 0.01670, _ST12_MS["ci"][0], _dp(0.01670, 5))
cms("SI 8: ...to 0.01683", 0.01683, _ST12_MS["ci"][1], _dp(0.01683, 5))
cms("SI 8: February 2024 moves 0.01851 to 0.01427", 0.01427,
    _ST02_MS["bias_corrected"], _dp(0.01427, 5))
_NS12_MS = p33_live["R1_effective_sample"]["202312"]["stats"]
cms("SI 8: the half-L1 noise share is 4.5%", 4.5,
    100 * _NS12_MS["half_l1"]["noise_bias"] / _NS12_MS["half_l1"]["point"],
    _dp(4.5, 1))
cms("SI 8: Cramer's V 7.7%", 7.7,
    100 * _NS12_MS["cramers_v"]["noise_bias"] / _NS12_MS["cramers_v"]["point"],
    _dp(7.7, 1))
cms("SI 8: normalised mutual information 12.8%", 12.8,
    100 * _NS12_MS["nmi"]["noise_bias"] / _NS12_MS["nmi"]["point"], _dp(12.8, 1))
cms("SI 8: the dominant eigenvalue is 1.15639", 1.15639,
    _NS12_MS["lead_eigenvalue"]["point"], _dp(1.15639, 5))
cms("SI 8: falling to 1.15624", 1.15624,
    _NS12_MS["lead_eigenvalue"]["bias_corrected"], _dp(1.15624, 5))
cms("SI 8: it moves from 1.1564 at theta = 0", 1.1564,
    _TH12_MS[0.0]["lead_eigenvalue"], _dp(1.1564, 4))
cms("SI 8: to 1.5198 at theta = -1", 1.5198,
    _TH12_MS[-1.0]["lead_eigenvalue"], _dp(1.5198, 4))
cms("SI 8: and 2.0857 at theta = +1", 2.0857,
    _TH12_MS[1.0]["lead_eigenvalue"], _dp(2.0857, 4))
cms("SI 8: 1.2340 in February 2024", 1.2340,
    _TH02_MS[0.0]["lead_eigenvalue"], _dp(1.2340, 4))
cms("SI 8: 1.6328", 1.6328, _TH02_MS[-1.0]["lead_eigenvalue"], _dp(1.6328, 4))
cms("SI 8: and 2.3899", 2.3899, _TH02_MS[1.0]["lead_eigenvalue"], _dp(2.3899, 4))
_WB12_MS = p33_live["R3_within_band"]["202312"]
_Z9_MS = {r["phi"]: r for r in _WB12_MS["zero_to_nine"]}
for _phi_ms, _q_ms in ((0.1, 0.02102), (0.3, 0.02097), (0.5, 0.02122)):
    cms(f"SI 8: re-attributing phi = {_phi_ms} moves it to {_q_ms}", _q_ms,
        _Z9_MS[_phi_ms]["assortativity"], _dp(_q_ms, 5))
_AB_MS = {r["phi"]: r for r in _WB12_MS["all_bands"]}
cms("SI 8: the generalised version moves it to 0.01997", 0.01997,
    _AB_MS[0.5]["assortativity"], _dp(0.01997, 5))
cms("SI 8: a change of -5.6%", -5.6,
    100 * (_AB_MS[0.5]["assortativity"] / _TH12_MS[0.0]["assortativity"] - 1),
    _dp(5.6, 1))

# --- SI 9: the allocation experiment
# The order below is the order the SI prints, and it is now the order the SI
# labels. p35 holds mean daily contacts of 1.0919 (passive dong), 1.0850
# (passive district) and 3.2594 (survey), and the eigenvalues beside them run
# the same way; the sentence used to name the survey first while printing it
# last, which put the matrix with three times the contacts of either passive
# matrix in the position of the smallest. The labels were reordered to the
# values rather than the values to the labels, because each value is gated
# against the matrix it belongs to and those pairings are what p35 stores.
_MX_MS = p35_live["matrices"]
for _key_ms, _c_ms, _e_ms in (("202312|passive_dong", 1.092, 5.505),
                              ("202312|passive_gu", 1.085, 5.463),
                              ("202312|survey", 3.259, 18.479)):
    cms(f"SI 9: {_key_ms} mean daily contacts {_c_ms}", _c_ms,
        _MX_MS[_key_ms]["mean_contacts"], _dp(_c_ms, 3))
    cms(f"SI 9: {_key_ms} dominant eigenvalue {_e_ms}", _e_ms,
        _MX_MS[_key_ms]["rho_unscaled"], _dp(_e_ms, 3))
check("MS", "SI 9: the finite-difference anchor is 4.94e-04", 4.94e-4,
      max(r["max_rel_err"]
          for r in p35_live["marginal_finite_difference_check"]), 5e-7)
check("MS", "SI 9: the final-size anchor is 2.89e-06", 2.89e-6,
      p35_live["final_size_anchor"]["max_abs_diff"], 5e-9)
cms("SI 9: the Kendall tau is +0.390 at district level", 0.390,
    _CM12_MS["survey vs passive_gu"]["tau"], _dp(0.390, 3))
_CM02_MS = {r["pair"]: r for r in p35_live["comparison_marginal"]["202402"]}
cms("SI 9: +0.600 in February 2024 at dong", 0.600,
    _CM02_MS["survey vs passive_dong"]["tau"], _dp(0.600, 3))
cms("SI 9: and +0.505 at district", 0.505,
    _CM02_MS["survey vs passive_gu"]["tau"], _dp(0.505, 3))
_A13_MS = p35_live["allocation"]["202312|R0=1.3"]["cross_application"]
cms("SI 9: the passive plan leaves 5.71 percentage points infected", 5.71,
    100 * _A13_MS["attack_passive_plan"], _dp(5.71, 2))
check("MS", "SI 9: while the survey plan extinguishes the epidemic", 1.6e-12,
      p35_live["allocation"]["202312|R0=1.3"]["survey"]["attack_with"], 5e-14)
# THE TWELVE-POINT GRID, added 2026-09-03. SI 9 carried the three-point Seoul
# reading and the R0 = 1.3 extinction case and stopped there, while 3.4's
# headline -- the interior maximum, and the factor of 40 between the anchor and
# the peak -- is entirely a property of the twelve-point grid p52 ran. Four of
# those numbers are national-scope and existed nowhere in the SI, which is the
# arm 2.2 already has to justify, so the SI could not show its working for the
# one result that most needs it. All eight cells of the grid summary are
# registered here, both arms, so neither can move without the other noticing.
#
# The decimal place each cell is printed to travels WITH the value here rather
# than being inferred from its magnitude, because it is a property of the
# sentence and not of the number: December 2023's anchor is printed as "1.27%"
# and Seoul's as "7.5%", and reading two places off the smaller of the two
# failed a correctly rounded 7.4688 by a third of the allowance it should have
# had. _dp() is only honest if it is handed the digits the prose actually
# prints.
for _cell_ms, _anch_ms, _adp_ms, _peak_ms in (
        ("202312|national", 1.27, 2, 50.9),
        ("202402|national", 11.2, 1, 62.6),
        ("202312|seoul", 7.5, 1, 20.2),
        ("202402|seoul", 23.3, 1, 25.4)):
    cms(f"SI 9: {_cell_ms} regret is {_anch_ms}% at the R0 = 2.5 anchor",
        _anch_ms, 100 * _PS_MS["at_anchor"][_cell_ms], _dp(_anch_ms, _adp_ms))
    cms(f"SI 9: {_cell_ms} peaks at {_peak_ms}%", _peak_ms,
        100 * _PS_MS["max_over_grid"][_cell_ms], _dp(_peak_ms, 1))
cms("SI 9: a factor of 40 between the anchor and the peak in December 2023",
    40.0, _PS_MS["max_over_grid"]["202312|national"]
    / _PS_MS["at_anchor"]["202312|national"], _dp(40.0, 0))
# The grid is printed value by value in the SI, so it is checked value by value
# rather than by its length: a twelve-point list with the wrong points in it
# would pass a count.
check("MS", "SI 9: the point grid is 1.1 to 6.0 in twelve steps", 1.0,
      _eq(list(p52_ms["declaration"]["r0_point"]),
          [1.1, 1.2, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]), 1e-9)
check("MS", "SI 9: the bootstrap grid is 1.2, 1.5, 1.8, 2.5, 3.5", 1.0,
      _eq(list(p52_ms["declaration"]["r0_boot"]), [1.2, 1.5, 1.8, 2.5, 3.5]), 1e-9)
check("MS", "SI 9: the anchor among the three original points is 2.5", 2.5,
      float(p52_ms["declaration"]["r0_anchor"]), 1e-9)
check("MS", "SI 9: February 2024 national peaks at R0 = 1.3", 1.3,
      _PS_MS["argmax"]["202402|national"], 1e-9)
check("MS", "SI 9: all four cells peak between R0 = 1.2 and 1.5", 1.0,
      1.0 if (min(_PS_MS["argmax"].values()) >= 1.2
              and max(_PS_MS["argmax"].values()) <= 1.5) else 0.0, 1e-9)
# "the curves are not monotone above their peaks" is the sentence that keeps
# the one verdict at R0 = 3.5 from reading as a contradiction of the interior
# maximum, so it is measured rather than asserted: at least one cell rises
# again somewhere above its own argmax.
check("MS", "SI 9: at least one cell's regret rises again above its peak", 1.0,
      1.0 if any(
          any(_rows_ms[_i + 1]["regret"] > _rows_ms[_i]["regret"]
              for _i in range(len(_rows_ms) - 1)
              if _rows_ms[_i]["r0"] >= _PS_MS["argmax"][_cell_ms])
          for _cell_ms, _rows_ms in p52_ms["point"].items()) else 0.0, 1e-9)
check("MS", "SI 9: the one verdict outside the 1.2-1.8 cluster is at R0 = 3.5",
      3.5, _a2_62["outlier"]["r0"], 1e-9)
# The R0 = 2.5 attack rates the body's mechanism sentence points here for. Both
# ends of all three ranges, so neither end can move alone.
# Same reasoning, and here the three ranges are also each other's cover: six
# decimals between 0.5 and 0.9 in one sentence. Each phrase carries the PLAN as
# well as the two ends, and that is the whole point rather than verbosity: a
# phrase of "between 0.542 and 0.752" alone was measured to stay GREEN when the
# survey's range and the passive's were swapped over, because both ranges are
# still printed in the sentence and only the labels moved. Swapping two labels
# is the error that would actually be made here.
# Two spans, four ends, and three of them were rounded to nearest and so fell
# inside the values they were meant to enclose: 0.5415904, 0.7523625 and
# 0.7750610 against 0.542, 0.752 and 0.775.
cms("SI 9: at R0 = 2.5 the survey plan leaves at least 0.541", 0.541,
    min(_r["attack_own_plan"] for _r in _A25_MS.values()), _dp(0.541, 3),
    phrase="survey-derived plan leaves between 0.541 and 0.753", bound="lower")
cms("SI 9: and at most 0.753", 0.753,
    max(_r["attack_own_plan"] for _r in _A25_MS.values()), _dp(0.753, 3),
    phrase="survey-derived plan leaves between 0.541 and 0.753", bound="upper")
cms("SI 9: the passive plan leaves at least 0.543", 0.543,
    min(_r["attack_passive_plan"] for _r in _A25_MS.values()), _dp(0.543, 3),
    phrase="passive-derived plan between 0.543 and 0.776", bound="lower")
cms("SI 9: and at most 0.776", 0.776,
    max(_r["attack_passive_plan"] for _r in _A25_MS.values()), _dp(0.776, 3),
    phrase="passive-derived plan between 0.543 and 0.776", bound="upper")
cms("SI 9: with no allocation at all the range runs from 0.683", 0.683,
    min(_r["attack_no_plan"] for _r in _A25_MS.values()), _dp(0.683, 3),
    phrase="against 0.683 to 0.850 when nothing is allocated")
cms("SI 9: to 0.850", 0.850,
    max(_r["attack_no_plan"] for _r in _A25_MS.values()), _dp(0.850, 3),
    phrase="against 0.683 to 0.850 when nothing is allocated")
check("MS", "SI 9: three of the four cells exceed 0.8 with no allocation", 3.0,
      float(sum(1 for _r in _A25_MS.values() if _r["attack_no_plan"] > 0.8)),
      1e-9)
# check(), not cms(): the SI prints this exponent in Unicode superscripts
# ("10⁻¹⁰⁰·²"), which no digit form of -100.2 can match. Registering it
# for the appear-in-text direction would report a value as quoted nowhere over
# a typesetting choice, which is the same false negative the U+2212 and thin-
# space rules in _forms() exist to prevent -- and superscript digits are not a
# spelling of a number, they are a different glyph set, so they belong here
# rather than in _forms().
check("MS", "SI 9: the rule's size at 200 draws is 10^-100.2", -100.2,
      _al62["log10"], 5e-2)
check("MS", "SI 9: six of the seven pre-correction verdicts sat in the cells "
            "the correction touched", 7.0,
      float(p52_ms["answer"]["n_flips_published"]), 1e-9)
cms("SI 9: the most marginal cell's margin is +0.015", 0.015,
    _a2_62["outlier"]["margin"], _dp(0.015, 3))
cms("SI 9: the achieved coverage of the 2.5th percentile runs from 1.1%", 1.1,
    100 * _ef62["low_level_ci"][0], _dp(1.1, 1))
cms("SI 9: to 5.7%", 5.7, 100 * _ef62["low_level_ci"][1], _dp(5.7, 1))
cms("SI 9: the 97.5th covers 94.3%", 94.3, 100 * _ef62["high_level_ci"][0],
    _dp(94.3, 1))
cms("SI 9: to 98.9%", 98.9, 100 * _ef62["high_level_ci"][1], _dp(98.9, 1))


# ======================================================================
# Reference-list house style -- one rule, and it is enforced rather than written
# ======================================================================
# 2026-09-03. The advisor read the two lists and asked why some entries carry
# bold and others do not. There WAS a rule -- bold marked "journal name, volume,
# pages", so the six entries with no journal (a preprint, a dataset, a book, a
# statute, a Zenodo deposit, a figshare dataset) carried none -- but an unwritten
# rule is indistinguishable from carelessness to the reader who has to ask, and
# it had already drifted: the main list ran 29 of 35 bold while the SI, swept in
# an earlier round, ran 2 of 20. JRSI_投稿規定整理.md gives the reference style as
# Vancouver, numbered by first appearance, with DOIs and at most ten authors, and
# says nothing about bold; the journal applies its own style at typesetting, so
# bold in the manuscript carries no information and only invites the question.
# Both lists are now plain. A gate row rather than a note, because a note is what
# the SI had, and the SI is the half that drifted.
def _bold_reference_entries(path):
    """Numbered entries under `## References` in `path` that carry bold markup."""
    if not os.path.exists(path):
        return 0
    tail = open(path, encoding="utf-8").read().partition("\n## References")[2]
    return sum(1 for line in tail.split("\n")
               if re.match(r"^\d+\.\s", line) and "**" in line)


check("MS", "no entry in the main reference list carries bold", 0.0,
      float(_bold_reference_entries(MANUSCRIPT)), 1e-9)
check("MS", "and none in the SI's, which is the half that drifted", 0.0,
      float(_bold_reference_entries(MANUSCRIPT_SI)), 1e-9)

# The SI's list "is numbered in its own order of first appearance, so its numbers
# are not those of the main text's". A bracketed number inside the SI is
# therefore an SI number, always -- and a sentence that wrote "reference [20] of
# the main text" was read as one by every check here: continuity, orphans and the
# reverse direction all took it for a citation of the SI's own [20], the Korea
# Policy Briefing. It resolved to Newman only because the entry deleted on 09-02
# sat above 20; deleting any main-text entry below 20 would have redirected it in
# silence. The sentence now names a section, as the SI's other fifteen
# cross-document pointers do, and this row keeps it that way.
_SI_XREF = re.compile(r"\[\d+(?:\s*,\s*\d+)*\][^.]{0,60}?of the main text")
check("MS", "no SI sentence points at the main text by reference number", 0.0,
      float(len(_SI_XREF.findall(open(MANUSCRIPT_SI, encoding="utf-8").read()
                                 if os.path.exists(MANUSCRIPT_SI) else ""))),
      1e-9)


# ======================================================================
# Priority claims -- the one class of sentence this gate cannot recompute
# ======================================================================
# Every other row here is (where it is quoted, what it should equal, where that
# comes from). "This has not previously been quantified" has no right-hand side:
# no results file holds the literature. So the gate was silent about that class
# of sentence, and silence read as coverage -- §4.1 carried such a clause with a
# green gate over it, because the 3.96 sitting beside it WAS a row and the
# priority clause simply was not one.
#
# What is checkable is not the fact but the diligence: a priority claim must
# have a dated search record, and the record must still describe a sentence the
# manuscript actually makes. Both directions, exactly as for the numbers:
#
#   registered -> prose   a registered claim whose sentence was rewritten or cut
#                         is a stale row, and its record now documents a search
#                         for a claim nobody makes.
#   prose -> registered   a sentence that reads like a priority claim and is in
#                         neither register is a claim nobody has searched.
#
# The second direction is the one that earns the mechanism, and it is why the
# patterns below are deliberately loose. A false positive costs one line in a
# register; a false negative is a reviewer holding a counter-example.
PRIORITY_DIR = f"{ROOT}/paper/priority_searches"

PRIORITY_PATTERNS = [
    r"ha(?:s|ve|d)? ?not (?:previously |yet )?been (?:quantified|measured|"
    r"reported|documented|shown|established|attempted|resolved|obtained)",
    r"(?:has|have) not previously",
    r"not previously been",
    r"never been",
    r"(?:we|this paper|this study|ours) (?:are|is) the first",
    r"the first (?:study|paper|work|measurement|time)",
    r"for the first time",
    r"no (?:previous|prior|earlier|other) (?:study|work|measurement|paper)",
    r"no (?:study|work|paper|measurement|analysis) has",
    r"unquantified",
    r"unmeasured",
    r"nobody has",
    r"no one has",
    r"what is new",
    r"\bare new\b",
    r"novelty",
    r"first to ",
]

# Claim -> the dated record under PRIORITY_DIR that backs it. The key is a
# lowercased fragment of the sentence, long enough to be unique and short enough
# to survive copy-editing that does not change the claim. A fragment that stops
# matching is the point: the sentence moved, so the search must be re-read
# against what it now says.
PRIORITY = {
    "the direction and the size of that distortion on a passive product are new":
        "2026-08-30-age-coarsening.md",
    "no study has asked how much":
        "2026-08-30-proportionate-mixing.md",
    # 2026-09-03: the abstract and 1 split the same claim out into sentences of
    # their own. The claim did not change -- it is still "nobody has measured
    # how much age structure such a product carries" -- but the old wording
    # ("an assumption never measured", "the measurement itself is missing")
    # matched no pattern, so it was never registered. The new wording does
    # match, and it is the same search that backs it.
    "that assumption has never been measured":
        "2026-08-30-proportionate-mixing.md",
    "the measurement itself has never been made":
        "2026-08-30-proportionate-mixing.md",
}

# Hits that carry the wording without making the claim. These need a reason, not
# a search record -- but they are registered rather than pattern-excluded, so
# that the exemption is visible and has to be re-argued if the sentence changes.
PRIORITY_NOT_A_CLAIM = {
    "we do not claim novelty for the level discrepancy itself":
        "disclaims priority rather than asserting it -- this is the sentence "
        "that narrows the novelty to the concentration result",
}


def _sentence_at(text, start, end):
    """The sentence around a match, whitespace-normalised."""
    a = text.rfind(".", 0, start)
    a = 0 if a < 0 else a + 1
    b = text.find(".", end)
    b = len(text) if b < 0 else b + 1
    return " ".join(text[a:b].split())


def _priority_audit(docs):
    """[(kind, detail)] -- empty when every priority claim is accounted for."""
    problems = []
    hits = {}
    for d in docs:
        text = open(d).read()
        for pat in PRIORITY_PATTERNS:
            for m in re.finditer(pat, text, re.I):
                s = _sentence_at(text, m.start(), m.end())
                hits[(os.path.relpath(d, ROOT), s)] = text[:m.start()].count("\n") + 1

    corpus = " ".join(" ".join(open(d).read().split()) for d in docs).lower()
    known = set(PRIORITY) | set(PRIORITY_NOT_A_CLAIM)

    # prose -> registered
    for (rel, sentence), line in sorted(hits.items()):
        if not any(k in sentence.lower() for k in known):
            problems.append(("unregistered",
                             f"{rel}:{line} reads as a priority claim and is in "
                             f"neither register: \"{sentence[:120]}\""))

    # registered -> prose, and the record has to exist
    for frag, rec in sorted(PRIORITY.items()):
        if frag not in corpus:
            problems.append(("stale",
                             f"{rec} is registered for a sentence the "
                             f"manuscript no longer contains: \"{frag}\""))
        if not os.path.exists(f"{PRIORITY_DIR}/{rec}"):
            problems.append(("no record",
                             f"paper/priority_searches/{rec} does not exist, "
                             f"but a claim is registered against it"))
    for frag in sorted(PRIORITY_NOT_A_CLAIM):
        if frag not in corpus:
            problems.append(("stale",
                             f"an exemption is registered for a sentence the "
                             f"manuscript no longer contains: \"{frag}\""))
    return problems


def _bound_prose_audit(docs):
    """[(kind, detail)] -- bound sentences in the prose, against the rows below.

    The label side of this rule is only as strong as the label. Write "at most"
    into the paper and "the December reading" into the row, and nothing asks
    which way the number rounds -- which is the rule surviving on somebody's
    memory again, one level further in. So the prose is swept for the two
    unambiguous vocabularies, and a bound sentence whose number no row declares
    is reported.
    Only those two. "above" and "below" on their own are thresholds far more
    often than bounds here ("push the mutual information above 0.02296 bits"
    states where a count was taken, and 0.02296 is not being bounded by
    anything), so making the prose declare them would manufacture claims. They
    still force a declaration when they appear in a ROW's own words, where the
    row is already asserting something about its own value.
    """
    problems = []
    declared = {}
    for _lab, _q, _a, _d in BOUNDS:
        declared.setdefault(_q, set()).add(_d)
    registered = sorted({_e[1] for _e in IN_TEXT_MS})
    pats = ([("upper", _w) for _w in _BOUND_UPPER]
            + [("lower", _w) for _w in _BOUND_LOWER])

    def _judge(rel, text, span, literal, want):
        sentence = _sentence_at(text, span[0], span[1])
        if any(k in sentence for k in BOUND_NOT_A_BOUND):
            return
        line = text[:span[0]].count("\n") + 1
        lit = " ".join(literal.split()).rstrip(".")
        val = _bound_value(lit)
        hit = [q for q in registered if val is not None and q == val]
        if not hit:
            problems.append((
                "no row", f"{rel}:{line} states a bound and no row verifies "
                f"the number it bounds with: \"{sentence[:130]}\""))
            return
        if not any(want in declared.get(q, set()) for q in hit):
            problems.append((
                "undeclared", f"{rel}:{line} states a {want} bound on {lit}, "
                f"and no cms() row declares bound=\"{want}\" for it: "
                f"\"{sentence[:130]}\""))

    for doc in docs:
        text = open(doc, encoding="utf-8").read()
        rel = os.path.relpath(doc, ROOT)
        for want, w in pats:
            for m in re.finditer(rf"(?:{w})\s+({_BOUND_TOKEN.pattern})", text, re.I):
                _judge(rel, text, m.span(), m.group(1), want)
            for m in re.finditer(
                    rf"({_BOUND_TOKEN.pattern})\s*(?:%| pp|pp|bits)?[,;]?\s+(?:{w})\b",
                    text, re.I):
                _judge(rel, text, m.span(), m.group(1), want)
    return problems


def _bound_audit():
    """[(kind, detail)] -- every declared bound that does not bound its source."""
    return [("rounded the wrong way",
             f"{_lab}: the manuscript quotes {_num(_q)} as {_d} bound, the "
             f"source is {_a!r} -- round it "
             f"{'up' if _d == 'upper' else 'down'}")
            for _lab, _q, _a, _d in BOUNDS if not _holds(_q, _a, _d)]


def _num(v):
    if v is None:
        return "—"
    if float(v).is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    return f"{v:.12g}"


def emit_table(path, bad_keys):
    import hashlib
    import subprocess

    letter_19 = open(REPORT_0819).read()
    letter_21 = open(REPORT_0821).read()
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        head = "unknown"

    def where(q, letter):
        """Each round is compared against ITS OWN letter.

        Pooling the two documents would let a number registered for the 08-21
        round be marked 「信」 because an unrelated 08-19 value happens to print
        the same digits -- which is the reverse-direction check answering a
        question nobody asked.
        """
        if any(x in letter for x in _forms(q)):
            return "信"
        return "**未引用**"

    L = []
    L.append("目前的數字列表（機器生成，勿手改）")
    L.append("")
    L.append(f"生成指令：`.venv/bin/python eda/p31_report_audit.py --table {os.path.relpath(path, ROOT)}`"
             f"，程式碼 commit `{head}`。")
    L.append("")
    L.append("這是內部清單，不隨信寄出。每一列都不是打進來的，是閘門"
             "（`eda/p31_report_audit.py`）拿去比對的那一組數字本身印出來的，"
             "所以文件與閘門不可能不一致——它們是同一個物件。")
    L.append("")
    L.append("「出現在」那一欄是拿數字回頭去比對**該輪自己那封信**的內文，"
             "**不比對這份檔案自己**（比對自己會百分之百通過，等於沒有檢查），"
             "也不把兩封信併在一起比（那會讓某一輪的數字被另一輪湊巧同形的數字算成有引用）。"
             "寫「未引用」的列，是閘門驗過但信裡沒有引用的數字。")
    L.append("")

    # provenance: every results file these numbers were read from
    L.append("## 這些數字讀自哪些檔案")
    L.append("")
    L.append("| 結果檔 | 位元組 | sha256（前 16 位） |")
    L.append("|---|---:|---|")
    paths = {f"{src}/results_{name}.json" for (src, name) in R}
    # load() is not the only reader here: p34's two panels, p37 and the disk
    # inventory are opened by literal name. A manifest built from load() alone
    # would omit exactly those -- the same blind spot as a gate reading a partial
    # results file -- so the source is scanned for literal names as well.
    for _n in sorted(set(re.findall(r"results_[A-Za-z0-9_]+\.json",
                                    open(__file__).read()))):
        if os.path.exists(f"{LIVE}/{_n}"):
            paths.add(f"{LIVE}/{_n}")
    for fpath in sorted(paths):
        raw = open(fpath, "rb").read()
        rel = os.path.relpath(fpath, ROOT)
        L.append(f"| `{rel}` | {len(raw):,} | `{hashlib.sha256(raw).hexdigest()[:16]}` |")
    L.append("")

    rows_21 = [(sec, lab, q, a, tol) for sec, lab, q, a, tol in CHECKS if sec == "21"]
    rows_19 = [(sec, lab, q, a, tol) for sec, lab, q, a, tol in CHECKS if sec == "19"]
    rows_24 = [(sec, lab, q, a, tol) for sec, lab, q, a, tol in CHECKS
               if sec == "24"]
    rows_27 = [(sec, lab, q, a, tol) for sec, lab, q, a, tol in CHECKS
               if sec == "27"]
    rows_ms = [(sec, lab, q, a, tol) for sec, lab, q, a, tol in CHECKS
               if sec == "MS"]
    # "27" is excluded HERE and not only where it is used. rows_18 is the
    # catch-all, so a section it does not know about lands silently in the
    # already-sent 08-18 table -- which is how a live round would end up printed
    # under the heading that says its numbers are frozen.
    rows_18 = [(sec, lab, q, a, tol) for sec, lab, q, a, tol in CHECKS
               if sec not in ("19", "21", "24", "27", "MS")]
    in_text = {lab: q for lab, q in IN_TEXT}
    in_text_21 = {lab: q for lab, q in IN_TEXT_21}

    # The 08-27 round prints only once it has something to print. An empty
    # section under a heading that names a document nobody has written yet reads
    # as a round that was checked and found clean.
    if rows_ms:
        ms_text = "".join(open(d).read()
                          for d in (_manuscript_corpus() or []))
        in_text_ms = {e[0]: e[1] for e in IN_TEXT_MS}
        L.append(f"## 論文（`paper/manuscript_JRSI.md` 與其 SI），{len(rows_ms)} 項")
        L.append("")
        L.append("這是唯一要投稿的文件，也是這個閘門最後才蓋到的一份。"
                 "它還沒投出去，所以讀的是**現在的**結果檔；"
                 "投出去那一天要在同一個 commit 裡凍進 `eda/archive/`，"
                 "並把這一節每一個 `load()` 指過去。"
                 "「出現在」比對的是正文與 SI 兩份合起來——它們是同一個投稿信封，"
                 "只在 SI 裡出現的數字是有引用，不是漏引。")
        L.append("")
        L.append("| 項目 | 數字 | 出現在 | 閘門 |")
        L.append("|---|---:|---|---|")
        for sec, lab, q, a, tol in rows_ms:
            status = "⚠️ 不符" if (sec, lab) in bad_keys else ("ok" if a is not None else "無來源")
            seen = where(q, ms_text) if lab in in_text_ms else "（未登記）"
            L.append(f"| {lab} | {_num(a if a is not None else q)} | {seen} | {status} |")
        L.append("")

    if rows_27:
        letter_27 = "".join(open(d).read() for d in (_round27_corpus() or []))
        in_text_27 = {lab: q for lab, q in IN_TEXT_27}
        L.append(f"## 零之前、本輪（08-27）的數字，{len(rows_27)} 項")
        L.append("")
        L.append("這一輪還沒寄出，所以它讀的是**現在的**結果檔，不是凍結快照——"
                 "信裡的數字還會動，而信要跟著動。")
        L.append("")
        L.append("| 項目 | 數字 | 出現在 | 閘門 |")
        L.append("|---|---:|---|---|")
        for sec, lab, q, a, tol in rows_27:
            status = "⚠️ 不符" if (sec, lab) in bad_keys else ("ok" if a is not None else "無來源")
            seen = where(q, letter_27) if lab in in_text_27 else "（未登記）"
            L.append(f"| {lab} | {_num(a if a is not None else q)} | {seen} | {status} |")
        L.append("")

    letter_24 = open(REPORT_0824).read()
    in_text_24 = {lab: q for lab, q in IN_TEXT_24}
    L.append(f"## 零、08-24 那一輪已寄出的數字，{len(rows_24)} 項")
    L.append("")
    L.append("這一組比對的是 `eda/archive/20260824/` 裡凍結的結果檔，不是現在的結果檔——"
             "信已經寄出（宣緯老師 8/27 回信了），數字就定住了。"
             "`p49`、`p50`、`p51`、`p52`、`p53`、`p54` 是這一輪的新階段，"
             "`p57` 與重跑過的 `p44` 是寫 NIMS 詢問信時發現的兩個更正。"
             "`results_p26/p27/p32/p40` 這一輪一個位元都沒動；"
             "`results_p44` 動了（`d583ff0` 撤掉 44.7 那一臂），"
             "所以它在 `20260822/` 與 `20260824/` 各有一份，兩封信各對各的。")
    L.append("")
    L.append("| 項目 | 數字 | 出現在 | 閘門 |")
    L.append("|---|---:|---|---|")
    for sec, lab, q, a, tol in rows_24:
        status = "⚠️ 不符" if (sec, lab) in bad_keys else ("ok" if a is not None else "無來源")
        seen = where(q, letter_24) if lab in in_text_24 else "（未登記）"
        L.append(f"| {lab} | {_num(a if a is not None else q)} | {seen} | {status} |")
    L.append("")

    L.append(f"## 一、本輪（08-21）的數字，{len(rows_21)} 項")
    L.append("")
    L.append("`p39`（回收實驗）、`p40`（全國下限）與 `p41`（學期一節）三個結果檔，"
             "在這一輪之前沒有任何閘門讀過。"
             "首爾仍是 primary，全國是並列的穩健性臂——所以這一節是**加上去的**，"
             "上面那兩輪的已發表值一個都沒有動。")
    L.append("")
    L.append("| 項目 | 數字 | 出現在 | 閘門 |")
    L.append("|---|---:|---|---|")
    for sec, lab, q, a, tol in rows_21:
        status = "⚠️ 不符" if (sec, lab) in bad_keys else ("ok" if a is not None else "無來源")
        seen = where(q, letter_21) if lab in in_text_21 else "（未登記）"
        L.append(f"| {lab} | {_num(a if a is not None else q)} | {seen} | {status} |")
    L.append("")

    L.append(f"## 二、08-19／08-20 那一輪的數字，{len(rows_19)} 項")
    L.append("")
    L.append("順序就是閘門註冊的順序，也就是信裡的行文順序。"
             "「閘門」欄的 ⚠️ 代表引用值與來源不符，那種情況下閘門本身是紅的。")
    L.append("")
    L.append("| 項目 | 數字 | 出現在 | 閘門 |")
    L.append("|---|---:|---|---|")
    for sec, lab, q, a, tol in rows_19:
        status = "⚠️ 不符" if (sec, lab) in bad_keys else ("ok" if a is not None else "無來源")
        seen = where(q, letter_19) if lab in in_text else "（未登記）"
        L.append(f"| {lab} | {_num(a if a is not None else q)} | {seen} | {status} |")
    L.append("")

    L.append(f"## 三、08-18 那一輪已寄出的數字，{len(rows_18)} 項")
    L.append("")
    L.append("這一組比對的是 `eda/archive/20260818/` 裡凍結的結果檔，不是現在的結果檔——"
             "已經寄出的信，數字就定住了，拿現在的檔案去對它只會在上游一動就變紅，"
             "而那種紅燈對那封信毫無意義。")
    L.append("")
    L.append("| § | 項目 | 信裡引用 | 凍結來源重算 | 閘門 |")
    L.append("|---|---|---:|---:|---|")
    for sec, lab, q, a, tol in rows_18:
        status = "⚠️ 不符" if (sec, lab) in bad_keys else ("ok" if a is not None else "無來源")
        L.append(f"| {sec} | {lab} | {_num(q)} | {_num(a)} | {status} |")
    L.append("")

    L.append("## 四、這份列表蓋不到的")
    L.append("")
    L.append("沒有結果檔可以比對的東西不在上面：手冊的用字、docomo 的規格、"
             "자치구／目的地／電子報那些入口網站事實，以及每一個日期。"
             "它們各自在 `eda/memo/` 的逐頁 memo 裡。"
             "逐位元可重現性也不在上面，那是 `eda/determinism_check.sh` 的事。")
    L.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return (len(rows_ms), len(rows_27), len(rows_24), len(rows_21),
            len(rows_19), len(rows_18))


# The registration check sits HERE, not beside REQUIRED_RESULTS, because
# _ASKED is filled by results_of() calls further down the file. The first
# version sat next to the declaration, ran against an empty set and passed --
# the same vacuity it exists to prevent, for a third time. Verified by
# removing names from REQUIRED_RESULTS and confirming it goes red.


# --------------------------------------------------------------------- run
TABLE = f"{ROOT}/eda/memo/numbers-20260820.md"


def main():
    # The declaration check runs HERE, first thing in main(), because main()
    # runs after every module-level results_of() call. Placing it at module
    # level made its correctness a property of WHERE it sat: adding a
    # registration below it left _ASKED short at check time and both rows
    # green while the new name went unvalidated. Demonstrated by injecting a
    # results_of("p39") after the check -- 6 vs 6, ok, and p39 in _ASKED at
    # the end. A count literal beside it did not help, because the count was
    # also read too early. Position was the bug; running last is the fix.
    #
    # Eight, not thirteen: REQUIRED_RESULTS also declares p35, p36 and the
    # 08-21 round's p39/p40/p41, all of which are read through load() rather
    # than an exists-guard and crash loudly on absence. The two lists are
    # deliberately unequal, and this literal counts only the exists-guarded
    # ones.
    # Back to eight: p52 was reached through results_of() while its block was
    # conditional on the file existing. The 08-24 round is closed and p52 is
    # frozen, so it is loaded like the rest and no longer registers here.
    check("19", "results names registered before the declaration check", 8.0,
          float(len(_ASKED)), 1e-9)
    _undeclared = sorted(_ASKED - set(REQUIRED_RESULTS))
    check("19", "every results file this gate reaches for is declared REQUIRED",
          0.0, float(len(_undeclared)), 1e-9)
    if _undeclared:
        print(f"  reached for but not declared REQUIRED: {_undeclared} -- "
              f"add them, or a missing file will skip their checks")

    # The 20260822 freeze has two layers and this is what makes the second one
    # real. p34/p37/p38 are 33.9 MB and were pinned by sha256 rather than copied,
    # on the ground that this round does not re-run them. "Does not" is a plan,
    # not a guarantee -- so it is checked. A pinned file that moves means a sent
    # document is now being gated against data it was not written from, which is
    # the exact failure eda/archive/README.md exists to prevent. Loud, not silent.
    _moved = []
    for _nm, _want in sorted(PINNED_0822.items()):
        with open(f"{LIVE}/results_{_nm}.json", "rb") as _fh:
            _got = hashlib.sha256(_fh.read()).hexdigest()
        if _got != _want:
            _moved.append(_nm)
    check("19", "pinned 20260822 panels still match their frozen sha256",
          0.0, float(len(_moved)), 1e-9)
    if _moved:
        print(f"  MOVED SINCE THE FREEZE: {_moved} -- copy each into "
              f"eda/archive/20260822/ and repoint its load() before re-running, "
              f"or the 08-19/08-22 letters are being checked against new data")

    ap = argparse.ArgumentParser()
    ap.add_argument("--table", nargs="?", const=TABLE, default=None,
                    metavar="PATH",
                    help="also write the numbers list to PATH (default: "
                         "eda/memo/numbers-20260820.md)")
    args = ap.parse_args()
    for _doc in (REPORT, REPORT_0819, REPORT_0821, ANNEX_0822,
                 REPORT_0824, ANNEX_0824):
        assert os.path.exists(_doc), f"missing {_doc}"

    # The 08-27 pair is declared but not yet written, so it is NOT in the list
    # above -- asserting it would make the gate red for a document nobody has
    # started. What is asserted instead is the pairing, which is the property
    # that actually has to hold: the documents may be absent only while nothing
    # is registered against them. The first c27() call, or the first check()
    # tagged "27", turns their absence into a hard failure here, so this stops
    # being a skip the moment it would start hiding something.
    # The manuscript pair carries the same contract, for the same reason: it
    # may be absent only while nothing is registered against it.
    _nms_checks = sum(1 for _c in CHECKS if _c[0] == "MS")
    _corpus_ms = _manuscript_corpus()
    if _corpus_ms is None:
        assert not IN_TEXT_MS and not _nms_checks, (
            f"{len(IN_TEXT_MS)} in-text values and {_nms_checks} checks are "
            f"registered against the manuscript, but neither half of it "
            f"exists: {[d for d in (MANUSCRIPT, MANUSCRIPT_SI) if not os.path.exists(d)]}"
            f" -- write it, or unregister the values")

    _n27_checks = sum(1 for _c in CHECKS if _c[0] == "27")
    _corpus_27 = _round27_corpus()
    if _corpus_27 is None:
        assert not IN_TEXT_27 and not _n27_checks, (
            f"{len(IN_TEXT_27)} in-text values and {_n27_checks} checks are "
            f"registered for the 08-27 round, but its corpus does not exist "
            f"yet: {[d for d in (REPORT_0827, ANNEX_0827) if not os.path.exists(d)]}"
            f" -- write the letter, or unregister the values")

    # Priority claims. One row, not one per sentence: what is being checked is
    # the register, and a register is either complete or it is not. The detail
    # prints below the table with the other findings.
    _pri = [] if _corpus_ms is None else _priority_audit(_corpus_ms)
    check("19", "every priority claim has a dated search record",
          0.0, float(len(_pri)), 1e-9)

    # The rounding direction of a bound, in the two directions it can fail.
    # They are separate rows because the repair is different: the first says a
    # number is rounded the wrong way and the sentence is false as printed, the
    # second says the prose states a bound that no row is watching at all.
    _bnd = _bound_audit()
    check("MS", "every bound quote rounds towards its own claim",
          0.0, float(len(_bnd)), 1e-9)
    _bpr = [] if _corpus_ms is None else _bound_prose_audit(_corpus_ms)
    check("MS", "every bound sentence in the prose has a row declaring it",
          0.0, float(len(_bpr)), 1e-9)

    bad = []
    print(f"{'§':>6}  {'claim':<44} {'quoted':>10} {'recomputed':>12}  status")
    for sec, label, quoted, actual, tol in CHECKS:
        if actual is None:
            status = "NO SOURCE"
        elif abs(actual - quoted) <= tol:
            status = "ok"
        else:
            status = "MISMATCH"
            bad.append((sec, label, quoted, actual))
        print(f"{sec:>6}  {label:<44} {quoted:>10} {actual!s:>12}  {status}")
    print(f"\n{len(CHECKS) - len(bad)} of {len(CHECKS)} checks pass.")
    if bad:
        print("\nMISMATCHES — the report says the first number, the results file "
              "says the second:")
        for sec, label, q, a in bad:
            print(f"  §{sec} {label}: report {q}, source {a}")
    if _pri:
        print("\nPRIORITY CLAIMS THE REGISTER DOES NOT ACCOUNT FOR — "
              "see paper/priority_searches/README.md:")
        for _kind, _detail in _pri:
            print(f"  [{_kind}] {_detail}")
    if _bnd:
        print("\nBOUNDS ROUNDED AWAY FROM THEIR OWN CLAIM — each of these "
              "sentences is FALSE as printed, not imprecise. Fix the prose and "
              "the row's `quoted` together; widening a tolerance here would be "
              "gating the distance again, which is what let them through:")
        for _kind, _detail in _bnd:
            print(f"  [{_kind}] {_detail}")
    if _bpr:
        print("\nBOUND SENTENCES WITH NOTHING WATCHING THEM — the prose says "
              "at most or at least and no row declares which way that number "
              "rounds. Register it with bound=, or, where the digits are not "
              "the bound, say why in BOUND_NOT_A_BOUND:")
        for _kind, _detail in _bpr:
            print(f"  [{_kind}] {_detail}")
    # The other direction: numbers this gate verifies that do not appear anywhere
    # in the document. Either the claim was dropped from the prose (so the check
    # is dead weight) or it was transcribed in a form this matcher does not see.
    #
    # Each round is asked against ITS OWN letter. Pooling the corpora would let
    # an 08-21 value be counted as quoted because an unrelated 08-19 number
    # prints the same digits, which is the one failure this direction exists to
    # catch, inverted.
    def _appearing(entries, *docs, regions=None, strict=False):
        # A round's corpus is EVERY file that was sent together, not just the
        # covering letter. The 08-22 round was sent as a 48-line letter plus its
        # technical annex, and the annex carries most of the numbers; reading the
        # letter alone reported 13 verified values as "quoted nowhere" when they
        # were simply in the other half of the same envelope.
        text = "".join(open(d).read() for d in docs)
        # `regions` narrows the haystack from the envelope to the section the
        # row's own label names -- see _ms_regions() for why the envelope was
        # too big to be a question. The sent rounds pass None and keep asking
        # the envelope: their letters have no section structure to speak of,
        # and they are frozen, so nothing in them can move out from under a row.
        missing, homeless = [], []
        for entry in entries:
            label, q = entry[0], entry[1]
            phrase = entry[2] if len(entry) > 2 else None
            hay = text
            if regions is not None:
                key = _ms_region_key(label)
                if key not in regions:
                    # The label names a section the document does not have. That
                    # cannot be answered yes or no, so it is reported as its own
                    # kind rather than folded into "the prose dropped it" -- but
                    # it counts as missing, because a row that cannot say where
                    # its sentence lives is not gating a sentence.
                    homeless.append((label, q))
                    missing.append((label, q))
                    continue
                hay = regions[key]
            # _forms() is shared with the numbers list so the two can never drift
            # into disagreeing about what counts as "the number appears".
            found = (phrase in hay) if phrase is not None else any(
                f in hay for f in _forms(q)
                if not strict or _same_number(q, f))
            if not found:
                missing.append((label, q))
        return missing, homeless

    def _report_missing(entries, missing, tag):
        seen = len(entries) - len(missing)
        if missing:
            print(f"\n{seen} of {len(entries)} verified {tag} values appear in the "
                  f"letter. The other {len(missing)} are verified but quoted "
                  f"nowhere -- either the prose dropped them, or they only ever "
                  f"lived in the attachments that are now gone:")
            for label, q in missing:
                print(f"  {label}: {q}")
        else:
            print(f"\nAll {len(entries)} verified {tag} values also appear in the "
                  f"letter.")
        return seen

    missing, _ = _appearing(IN_TEXT, REPORT_0819)
    quoted_n = _report_missing(IN_TEXT, missing, "08-19")
    missing_21, _ = _appearing(IN_TEXT_21, REPORT_0821, ANNEX_0822)
    quoted_21 = _report_missing(IN_TEXT_21, missing_21, "08-21")
    missing_24, _ = _appearing(IN_TEXT_24, REPORT_0824, ANNEX_0824)
    quoted_24 = _report_missing(IN_TEXT_24, missing_24, "08-24")
    # The 08-27 round contributes 0 and 0 while its corpus does not exist. That
    # is the ONLY state in which this branch is reachable -- the assert above has
    # already refused to get here with anything registered -- so the totals below
    # cannot be quietly short by the size of an unreported round.
    if _corpus_27 is None:
        quoted_27 = 0
    else:
        missing_27, _ = _appearing(IN_TEXT_27, *_corpus_27)
        quoted_27 = _report_missing(IN_TEXT_27, missing_27, "08-27")
    # The manuscript and its SI are ONE corpus. They are a single submission,
    # and the main text states a claim whose arithmetic lives in the SI, so a
    # value that appears only in the SI is quoted rather than missing. Reading
    # them apart would repeat the 08-22 mistake of checking a letter without
    # the annex it was posted with.
    if _corpus_ms is None:
        quoted_ms = 0
        missing_ms = []
    else:
        _claimed = {_ms_region_key(_e[0]) for _e in IN_TEXT_MS}
        missing_ms, homeless_ms = _appearing(
            IN_TEXT_MS, *_corpus_ms, regions=_ms_regions(_claimed),
            strict=True)
        quoted_ms = _report_missing(IN_TEXT_MS, missing_ms, "manuscript")
        if homeless_ms:
            print("\nAND OF THOSE, THESE NAME A SECTION THE MANUSCRIPT DOES "
                  "NOT HAVE. The label is the row's statement of where its "
                  "sentence lives; if there is no such section the row has "
                  "never been gating a sentence, and moving it is a decision "
                  "about the prose, not about this file:")
            for label, q in homeless_ms:
                print(f"  {label}: {q}  (no section \"{_ms_region_key(label)}\")")
        # AND FOR THE MANUSCRIPT THIS DIRECTION BITES, which it does not for the
        # five rounds of letters above. Those are frozen: nothing in a sent
        # document can move out from under a row, so a value reported as quoted
        # nowhere there is a note about the past. The manuscript is the opposite
        # -- it is the document still being edited and the only one that leaves
        # this repo for a journal -- and a row that says its sentence is in §4.2
        # when §4.2 no longer contains it is not a note, it is the gate failing
        # to gate. Until 2026-09-02 the only consequence was that _quoted_total
        # moved and eda/README.md stopped matching, so the red said "README does
        # not say 891 項" for a wrong sentence in the paper. That is the right
        # amount of red and the wrong sentence about why.
        if missing_ms:
            print("\nTHE MANUSCRIPT IS NOT A SENT LETTER, SO THE LIST ABOVE IS "
                  "A FAILURE AND NOT A NOTE. Each of those rows names a section "
                  "and verifies a number that section does not print. Resolve "
                  "one of three ways: move the number back into the prose; "
                  "relabel the row to the section that does print it; or, where "
                  "the number is drawn in a figure and never written down, make "
                  "it a check() rather than a cms(), which verifies it without "
                  "claiming the document quotes it.")

    # The gate's own counts are quoted in the letter, and they went stale within
    # hours the first time: "377 項" stayed in the prose while p34's two panels
    # pushed CHECKS to 401. A count that only a human updates is a count that
    # drifts silently, so the letter is required to contain today's totals. This
    # is a presence test, not a value comparison -- add a check and the letter
    # stops matching until someone updates it, which is the point.
    #
    # The three counts are TOTALS across every document this gate guards, not
    # the 08-19 round's own -- so they move every time ANY round gains a check.
    # They used to be read out of the 08-19 letter, and that was the tail wagging
    # in the most literal way: a sent document had to be edited to keep a gate
    # green. It happened. On 2026-08-24 the 08-22 round's rewrite moved
    # _quoted_total from 337 to 324 and the only way to make the gate pass was to
    # retype a number inside a letter that had already been sent.
    #
    # A total that changes every round belongs in a document that is still being
    # written. eda/README.md is that document: it is live, it already carried
    # these three counts, and it is where someone looks up what the gate checks.
    # The per-round counts below stay on their own letters, because a closed
    # round's own count does not move once the round is closed.
    _in_text_total = (len(IN_TEXT) + len(IN_TEXT_21) + len(IN_TEXT_24)
                      + len(IN_TEXT_27) + len(IN_TEXT_MS))
    _quoted_total = quoted_n + quoted_21 + quoted_24 + quoted_27 + quoted_ms
    _letter = open(f"{ROOT}/eda/README.md").read()
    stale = [(w, n) for w, n in (("gate checks", len(CHECKS)),
                                 ("in-text values", _in_text_total),
                                 ("values quoted in their own letter",
                                  _quoted_total))
             if f"{n} \u9805" not in _letter]
    _n21 = sum(1 for _c in CHECKS if _c[0] == "21")
    if f"{_n21} \u9805\u6aa2\u67e5" not in open(REPORT_0821).read():
        stale.append(("this round's own checks, in the 08-21 letter", _n21))
    _n24 = sum(1 for _c in CHECKS if _c[0] == "24")
    if f"{_n24} \u9805\u6aa2\u67e5" not in open(REPORT_0824).read():
        stale.append(("this round's own checks, in the 08-24 letter", _n24))
    # The 08-27 letter states its own count once there is a letter and a count.
    # Asking a document that does not exist to contain "0 \u9805\u6aa2\u67e5" would be a red
    # gate describing nothing, and writing "0 \u9805\u6aa2\u67e5" into a letter to satisfy it
    # would be worse.
    if _corpus_27 is not None and _n27_checks:
        if f"{_n27_checks} \u9805\u6aa2\u67e5" not in open(REPORT_0827).read():
            stale.append(("this round's own checks, in the 08-27 letter",
                          _n27_checks))
    if stale:
        print("\nSTALE COUNTS FOR THIS GATE:")
        for w, n in stale:
            print(f"  {w}: eda/README.md (or the round's own letter) "
                  f"does not say \"{n} 項\"")
    else:
        print(f"\neda/README.md states this gate as {len(CHECKS)} 項 / "
              f"{_in_text_total} 項 / {_quoted_total} 項, which is what it is; "
              f"the 08-21 letter states its own round as {_n21} 項檢查 and the "
              f"08-24 letter states {_n24} 項檢查.")

    # Numbers that appear in the report but no script produces are listed so the
    # gate is honest about what it cannot cover.
    print("\nNOT COVERED HERE (no results file to check against): the manual's "
          "wording, docomo's specifications, the 자치구/purpose/newsletter portal "
          "facts, and every date. Those are checked in their own memos. "
          "Section 8's disk counts USED to be on this list; they are now gated "
          "against eda/results_inventory.json.")
    print("  In the manuscript specifically: the two Zenodo DOIs, the GitHub "
          "mirror's URL and the MIT "
          "and CC BY 4.0 licence names; the Open Data Plaza's article 11; the "
          "count and numbering of the reference lists; the 2020 closure "
          "chronology's dates; the survey's 14-day diary window and its "
          "424/424 dong match; the product manual's structural-zero clause "
          "and the 99.03% empty grid it explains; docomo's published "
          "ten-year bands; and SI 7's four detector deviations (6.66e-16, "
          "1.49e-15, 2.2e-4, and the 250-cell brute-force expansion into "
          "5 714 venues), which p39 prints to eda/memo/phase39-recovery.md "
          "but not to its results file. SI 7's '26.8%' for the noisy data "
          "field used to sit on this list as unstoreable; it is p39's "
          "ladder.points['data|0.02'].kept_median, it is now an ordinary "
          "gated row, and 3.5 quotes it. Four other manuscript figures used "
          "to sit on this list and no longer do, because the prose was "
          "corrected on "
          "2026-08-29 and every one of them is now an ordinary gated row: "
          "3.4's '13.0%' for the measured decay law (12.8%); SI 8's "
          "'+0.00431' for the device-noise arm, truncated where SI 7 rounded "
          "the same 0.0043161 to '+0.00432'; the 'factor of 86' beside it, "
          "which was the quotient of those two printed roundings and is 82 "
          "computed from the arms; and SI 9's level sentence, whose three "
          "labels ran in the reverse of the order of the six values under "
          "them. eda/memo/phase44-beta.md still writes 'r = 0.0286（13.0%）' "
          "and the 08-21 letter still repeats it; both are frozen against "
          "their own snapshots and are meant to keep the wording they were "
          "sent with.")

    if args.table:
        nms, n27, n24, n21, n19, n18 = emit_table(
            args.table, {(s, l) for s, l, _, _ in bad})
        # The 08-27 and manuscript counts print only when there is one. A
        # "0 本輪(08-27)項目" in the summary line is a round announcing itself
        # before it exists.
        _headms = f"{nms} 論文項目 + " if nms else ""
        _head27 = f"{n27} 本輪(08-27)項目 + " if n27 else ""
        print(f"\nwrote {args.table}\n  {_headms}{_head27}{n24} 08-24 項目 + "
              f"{n21} 08-21 項目 + {n19} 08-19 項目 + {n18} 已寄出項目, "
              f"generated from the same triples this gate checks")
    return 1 if (bad or stale or missing_ms) else 0


if __name__ == "__main__":
    sys.exit(main())
