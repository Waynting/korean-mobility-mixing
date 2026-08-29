#!/usr/bin/env python
"""Phase 53 — beta's national arm, weighted to Korea instead of to Seoul.

WHY THIS EXISTS. The advisor's 2026-08-24 letter makes an asymmetry load-
bearing: "beta is insensitive to scope (0.922 vs 0.933), the headline sentence
is not (3.69 vs 1.21)", and offers it as the reason the paper may choose one
survey arm and still trust beta. The reasoning is right. The number is not yet.

`p44_beta.py:168-172` post-stratifies the venue table to `POP`, and
`p44_beta.py:206-211` sets `POP` to `results_p26.json["population"]["202312"]`
-- Seoul's. It does that for BOTH scopes. So 0.9326 is not a national reading;
it is national respondents reweighted onto Seoul's age structure. The scope
comparison currently varies who answered but not the population they are
weighted to, which is half of a scope comparison.

This is a SEPARATE defect from the one p50/p51 fix. `symmetrise(C, pop)` is a
reciprocity correction; this is p27's post-stratification convention. They share
a cause (one Seoul vector used everywhere) and nothing else, and the fix for one
is not the fix for the other.

WHY THIS IS A NEW PHASE AND NOT AN EDIT TO p44. `p44_beta.py:241-244` asserts
that the Seoul-weighted survey matrix reproduces p32's published 0.22270993,
`abs(a3 - R_SURVEY) < 1e-12`. That anchor is an IDENTITY precisely because the
weighting is `pop_a x (contacts per ego of band a)` with p27's pop. Changing
`POP` in place breaks a published anchor to make a new number -- exactly
backwards. So p44 is imported unchanged and its `POP` is set from outside, once
per arm, which is the same `import p32_pmix as P32` pattern p40 already uses.

    python eda/p53_natbeta.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import AGES
from paths import ROOT

import p44_beta as P44
from p27_survey import load_survey

YM = 202312
out = {}


def say(s=""):
    print(s, flush=True)


def main():
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p44 = json.load(open(f"{ROOT}/eda/results_p44.json"))
    p49 = json.load(open(f"{ROOT}/eda/results_p49.json"))
    p50 = json.load(open(f"{ROOT}/eda/results_p50.json"))

    POP_SEOUL = np.array(p26["population"][str(YM)], float)
    POP_KOREA = np.array(p50["vectors"][str(YM)]["national"], float)
    R_OBS_CORRECTED = P44.R_OBS_CORRECTED          # 0.01676780356461909, p33's

    ego, d = load_survey()

    # p44's main() derives `place` from the Q5_* columns before it calls build(),
    # and build() keys venue-visits on it. Doing the same here rather than importing
    # a preprocessed frame keeps this phase reading the survey exactly as p44 does;
    # the assert is p44's own (p44_beta.py:224).
    _q5 = [c for c in d.columns if c.startswith("Q5_") and c[3:].isdigit()]


    def _place_of(row):
        for c in _q5:
            v = str(row[c]).strip()
            if v.isdigit():
                return int(v)
        return 0


    d = d.assign(place=d[_q5].apply(_place_of, axis=1))
    assert (d.place > 0).all(), "a contact with no place code"
    assert (len(ego), len(d)) == (1987, 133776), (len(ego), len(d))


    def r_venue(scope, ym, pop):
        """p44's 44.1 reading of the truth, at one scope, under one weighting."""
        P44.POP = np.asarray(pop, float)
        P = P44.build(ego, d, scope, ym)
        return float(P44.r_of(P44.collapse(P["U"], P["wg"]))), P


    # ==================================================================== 53.0
    say("=== 53.0 anchors: the published readings, reproduced through p44 itself ===")
    r_nat_seoulw, P_nat = r_venue("national", None, POP_SEOUL)
    r_seo_seoulw, _ = r_venue("seoul", YM, POP_SEOUL)

    want_nat = p44["pairing"]["r_venue_wor"]
    want_seo = p44["sensitivity"]["scope"][f"seoul {YM} (anchor)"]["r_venue"]
    d_nat, d_seo = abs(r_nat_seoulw - want_nat), abs(r_seo_seoulw - want_seo)
    say(f"  national pooled, Seoul weighting : {r_nat_seoulw:.17f}")
    say(f"                    results_p44    : {want_nat:.17f}   |diff| {d_nat:.1e}")
    say(f"  seoul {YM},     Seoul weighting : {r_seo_seoulw:.17f}")
    say(f"                    results_p44    : {want_seo:.17f}   |diff| {d_seo:.1e}")
    assert d_nat == 0.0 and d_seo == 0.0, (
        "p44's published venue readings do not reproduce through p44's own "
        "functions; the corrected arm below would differ by the pipeline too")
    out["anchors"] = dict(national_wor=dict(got=r_nat_seoulw, stored=want_nat,
                                            bit_exact=True),
                          seoul_wor=dict(got=r_seo_seoulw, stored=want_seo,
                                         bit_exact=True))

    # The A3 identity p44 asserts must still be an identity -- this phase must not
    # have disturbed it, and the only way to know is to check it rather than to
    # reason that imports are side-effect free.
    P44.POP = POP_SEOUL
    say(f"  p44's A3 identity is untouched: POP restored to Seoul's vector, "
        f"total {POP_SEOUL.sum():,.0f}")

    # ==================================================================== 53.1
    say("\n=== 53.1 the national arm, weighted to Korea ===")
    r_nat_koreaw, _ = r_venue("national", None, POP_KOREA)
    say(f"  national pooled, Korea weighting : {r_nat_koreaw:.17f}")
    say(f"  moved {100 * (r_nat_koreaw / r_nat_seoulw - 1):+.2f}% from the "
        f"Seoul-weighted reading")

    rows = {}
    for nm, rv, note in (
            ("national pooled, Korea weighting (corrected)", r_nat_koreaw,
             "the reported national arm"),
            ("national pooled, Seoul weighting (published)", r_nat_seoulw,
             "what results_p44.json holds"),
            (f"seoul {YM}, Seoul weighting (unchanged)", r_seo_seoulw,
             "correct as it stands; a Seoul survey on Seoul's population")):
        rows[nm] = dict(r_venue=rv, beta=1 - R_OBS_CORRECTED / rv,
                        recovery=R_OBS_CORRECTED / rv, note=note)
        say(f"  {nm:<46} r_true {rv:.5f}  beta {rows[nm]['beta']:.4f}")
    out["scope"] = rows

    b_nat = rows["national pooled, Korea weighting (corrected)"]["beta"]
    b_seo = rows[f"seoul {YM}, Seoul weighting (unchanged)"]["beta"]
    b_pub = rows["national pooled, Seoul weighting (published)"]["beta"]
    spread_now = abs(b_nat - b_seo)
    spread_pub = abs(b_pub - b_seo)
    say(f"\n  the asymmetry the advisor wants to make load-bearing:")
    say(f"    beta   across scope : {b_seo:.4f} (seoul) vs {b_nat:.4f} (korea)   "
        f"spread {spread_now:.4f}")
    say(f"    published spread    : {b_seo:.4f} vs {b_pub:.4f}   "
        f"spread {spread_pub:.4f}")
    out["asymmetry"] = dict(beta_seoul=b_seo, beta_national=b_nat,
                            beta_national_published=b_pub,
                            spread=float(spread_now),
                            spread_published=float(spread_pub))

    # ==================================================================== 53.2
    # p49 found that the two sides of this comparison use different pairing
    # conventions: r_true is without replacement (p44), while the passive reading
    # R_OBS_CORRECTED descends from p26's with-replacement matrix. beta is a ratio
    # of the two, so the mismatch lands directly on it. Reported as a sensitivity,
    # NOT as the primary: the primary definition is p44's and is not redefined here.
    say("\n=== 53.2 sensitivity: beta on one pairing convention (p49) ===")
    rel = p49["primary"]["rel_change"]
    r_obs_wor = R_OBS_CORRECTED * (1 + rel)
    say(f"  p49: the passive reading moves {100 * rel:+.4f}% under without-"
        f"replacement, so {R_OBS_CORRECTED:.6f} -> {r_obs_wor:.6f}")
    cons = {}
    for nm, rv in (("national (korea)", r_nat_koreaw), ("seoul", r_seo_seoulw)):
        cons[nm] = dict(beta_mixed=1 - R_OBS_CORRECTED / rv,
                        beta_one_convention=1 - r_obs_wor / rv)
        say(f"  {nm:<18} beta {cons[nm]['beta_mixed']:.4f} (as published, mixed "
            f"conventions) -> {cons[nm]['beta_one_convention']:.4f} (one convention)")
    say("  beta rises, because a smaller passive reading means less of the truth "
        "was recovered.")
    out["convention_sensitivity"] = dict(
        passive_mixed=R_OBS_CORRECTED, passive_one_convention=float(r_obs_wor),
        rel_change=rel, arms=cons,
        note="reported, not adopted: p44's definition of beta is the primary")

    out["declaration"] = dict(
        ym=YM, primary="national pooled, Korea weighting",
        passive_reading=R_OBS_CORRECTED,
        passive_reading_source="p33, noise-bias corrected; Seoul, and it stays "
                               "Seoul -- only the SURVEY side was mis-weighted",
        touches_nothing=["results_p26.json", "results_p44.json"],
        method="p44_beta imported unchanged; its POP global is set from outside, "
               "so p44's A3 identity against p32's 0.22270993 is not disturbed")

    with open(f"{ROOT}/eda/results_p53.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p53.json")


if __name__ == "__main__":
    main()
