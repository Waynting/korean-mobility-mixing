#!/usr/bin/env python
"""Phase 61 — does the survey matrix's distance from rank one survive its own null?

WHY THIS EXISTS. The advisor's 2026-08-27 letter, point 3: Figure 3(b) is the
only panel in the figure with no null, and it is the one that most needs one.
The caption sells it as "(b) restates (a) without reference to any null". The
survey matrix has sigma2/sigma1 = 0.601 with only 46.8% of the energy in the
leading component -- but it is built from 465 respondents and 24 of its 225
cells are exactly zero, and a small sample by itself pushes a matrix away from
rank one. The direction of that bias CUTS AGAINST US: noise makes the survey
look LESS rank-1, which exaggerates exactly the contrast the figure reports. So
the question is not rhetorical: how much of 0.601 is structure and how much is
sampling noise at n = 465 with this pattern of emptiness?

The instrument already exists. p32 32.4b measured the survey's own floor for
mutual information by resampling respondents and by permuting ego bands, and
p32.rank_stats already returns sigma2/sigma1 on any matrix. This phase runs
p32's loops again and reads the spectral statistic out of the same draws.

WHAT THE PERMUTATION NULL IS, AND WHAT IT IS NOT. Shuffling ego bands while
keeping each ego's own alter tally intact preserves the alter margin, every
ego's contact count, the per-band respondent counts and therefore the sparsity
that emptiness comes from -- only the pairing dies. In the CONTACT matrix C
that null is exactly rank one in expectation: every row becomes the same mean
alter vector. It is NOT exactly rank one after symmetrisation, because
T = (pop v' + v pop')/2 has rank two unless the alter margin v is proportional
to the population -- true proportionate mixing would force v proportional to
pop by reciprocity, and the observed survey margin is not. 61.4 computes that
deterministic component in closed form and reports it, because it inflates the
null and therefore makes escape HARDER, which is the conservative direction but
must not be passed off as pure sampling noise. A multinomial null drawn from
r r' at the same contact count is reported beside it: that one IS exactly rank
one in truth but does not carry the respondent clustering or the emptiness. The
two bracket the question from opposite sides.

======================================================================
DECLARATION -- written before any analysis code was run, and not modified
after seeing output. Quoted verbatim into results_p61.json["declaration"].

cell          survey|202312|WE|seoul, bands < 80 (sq), symmetrised against population.
              THE SAME CELL p32 PUBLISHED. Declared, not chosen after seeing four cells.
              202402 is computed and reported too, exactly as p32 reports both.
statistic     sigma2_over_sigma1 from p32_pmix.rank_stats, imported.
null          p32 32.4b's ego-band permutation (alter margins and every ego's contact count
              survive, only the pairing dies). PERM = 500, SEED = 20260827.
uncertainty   p32 32.4b's respondent bootstrap, BOOT = 500, same stream discipline.
ESCAPE CRITERION, DECLARED BEFORE THE DRAW:
  primary     observed sigma2_over_sigma1 = 0.6013701616523675 exceeds the 95th percentile of
              the permutation distribution.  (p32's own mi_perm_p95 convention.)
  secondary   the 2.5th percentile of the bootstrap distribution exceeds the 97.5th percentile
              of the permutation distribution.  (p40/p52's flip rule, echoed p55_fig7.py:19.)
              REPORTED WHETHER OR NOT IT CLEARS.
also reported sigma1_share (observed 0.4679298573664325) against the same two nulls, and the
              count of exactly-zero cells in each permuted draw, so a reader can see the null
              carries the same emptiness the observation does.
======================================================================
ADDITIONS TO THE DECLARATION -- also written before the run. These add
reported quantities; they do not touch the cell, the statistic or the escape
criterion above.

  direction     sigma2_over_sigma1 is larger when a matrix is FURTHER from rank
                one, sigma1_share is smaller. The two rules are therefore
                mirrored for sigma1_share: primary = observed below the null's
                5th percentile; secondary = the bootstrap's 97.5th percentile
                below the null's 2.5th percentile. Stated here so the mirroring
                is not a choice made at reading time.
  passive arm   TAKEN, and declared SECONDARY. The advisor asked only for the
                survey side, but 3(b)'s other two curves (0.124 at dong, 0.066
                at district) are equally presented as noiseless. The passive
                16x16 is resampled multinomially at its own device-equivalent n
                (results_p33.json R1_effective_sample n_device_equiv, cross-
                checked against results_p32.json null_floor n_effective), twice:
                from r r' for a null and from the observed joint for a sampling
                spread. A multinomial is NOT the passive matrix's true noise
                model -- devices appear in many cells, coverage weights are
                applied, masked cells are imputed -- so it overstates the
                precision and the resulting floor is a LOWER bound on the true
                passive noise floor. Lower bound is the conservative direction
                for the claim "the passive noise floor is far below its signal".
  third null    a multinomial draw from the survey's own r r' at its contact
                count (MULTI = 500). Exactly rank one in truth, but without
                respondent clustering or emptiness. Reported beside the
                permutation null, never in place of it; the primary escape
                criterion above is decided by the PERMUTATION null alone.
  streams       SEED = 20260827 for the survey bootstrap and permutation;
                SEED + 1 for the survey multinomial null; SEED + 2 for the
                passive arm. Separate streams so that adding an arm cannot move
                an arm already reported -- the lesson p51 drew from p40.
======================================================================

ANCHORS. The run aborts on any failure.

  A1  p32 32.4b's built-in anchor, carried over verbatim: the respondent
      bootstrap must reproduce results_p27.json's published assortativity CI.
      It once caught the H panel leaking into the WE cube.
  A2  rank_stats on the un-resampled cube reproduces results_p32.json
      spectrum["survey|202312|WE|seoul"] BIT FOR BIT on all four scalars and
      all 15 singular values.
  A3  n_ego == 465 and the 80+ ego band is empty -- the reason sq is bands < 80.
  A4  the observed 15x15 sq block has exactly 24 cells equal to zero.
  A5  replaying p32's rng CONSUMPTION ORDER (32.4's four null_floor calls, then
      32.4b's loops) at p32's own seed reproduces mi_perm_median =
      0.07125667197789379 bit for bit. This is the anchor p40 established
      (eda/README.md:275-277); if it fires, this script draws in a different
      sequence from p32 and nothing below it can be compared to p32.

Reads results_p26/p27/p32/p33/p9.json and the Chae microdata; writes
results_p61.json and nothing else.

    python eda/p61_survspec.py [--boot 500] [--perm 500] [--multi 500]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import AGES, AGE_LABEL
from paths import RESULTS, ROOT

import p32_pmix as P32
from p32_pmix import excess_stats, null_floor, pm_null, rank_stats, symmetrise

SURVEY_MONTHS = [202312, 202402]
ANCHOR_SEED = P32.SEED                       # 20260819 -- p32's own
SEED = 20260827                              # this phase's declared stream
PRIMARY = dict(cell="survey|202312|WE|seoul", statistic="sigma2_over_sigma1",
               block="0-79 (sq)", panel="WE", days="holiday-free",
               null="ego-band permutation", rule="observed > permutation p95")

# The four published quantities A2 must reproduce bit for bit.
RANK_KEYS = ("sigma1_share", "sigma2_over_sigma1", "best_rank1_resid",
             "pm_rank1_resid")
P32_PERM_MEDIAN_202312 = 0.07125667197789379       # A5, results_p32.json
OBS_S2S1 = 0.6013701616523675                      # declared in the criterion
OBS_S1SHARE = 0.4679298573664325
N_ZERO_PUBLISHED = 24                              # A4
N_EGO_PUBLISHED = 465                              # A3

out = {}
FAIL = []


def hdr(s):
    print(f"\n=== {s} ===", flush=True)


def say(s=""):
    print(s, flush=True)


def check(name, got, want, tol=0.0):
    """Zero-tolerance by default: an anchor that needs a tolerance says so."""
    d = abs(float(got) - float(want))
    ok = d <= tol
    rec = dict(name=name, got=float(got), stored=float(want), abs_diff=d,
               tol=tol, ok=bool(ok))
    if not ok:
        FAIL.append(f"{name}: {got!r} vs stored {want!r} (diff {d:.3e})")
    return rec


def quantiles(v):
    q = [0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5]
    p = np.percentile(np.asarray(v, float), q)
    return {f"p{x}": float(y) for x, y in zip(q, p)}


def summarise(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                median=float(np.median(v)), min=float(v.min()),
                max=float(v.max()), quantiles=quantiles(v))


def verdicts(obs, perm, boot, larger_is_structure=True):
    """Both declared rules, for one statistic, in one place.

    larger_is_structure=True  -> sigma2/sigma1: structure pushes the value UP.
    larger_is_structure=False -> sigma1 share:  structure pushes the value DOWN.
    The mirroring is declared, not decided here.
    """
    perm = np.asarray(perm, float)
    boot = np.asarray(boot, float)
    if larger_is_structure:
        thr = float(np.percentile(perm, 95))
        primary = bool(obs > thr)
        p_value = float((perm >= obs).mean())
        b_edge, n_edge = (float(np.percentile(boot, 2.5)),
                          float(np.percentile(perm, 97.5)))
        secondary = bool(b_edge > n_edge)
    else:
        thr = float(np.percentile(perm, 5))
        primary = bool(obs < thr)
        p_value = float((perm <= obs).mean())
        b_edge, n_edge = (float(np.percentile(boot, 97.5)),
                          float(np.percentile(perm, 2.5)))
        secondary = bool(b_edge < n_edge)
    return dict(observed=float(obs), larger_is_structure=bool(larger_is_structure),
                null_threshold=thr, primary_escape=primary,
                permutation_p_value=p_value,
                bootstrap_edge=b_edge, null_edge=n_edge,
                secondary_escape=secondary,
                null_median=float(np.median(perm)),
                bootstrap_median=float(np.median(boot)))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=P32.BOOT)
    ap.add_argument("--perm", type=int, default=P32.PERM)
    ap.add_argument("--multi", type=int, default=500)
    ap.add_argument("--passive", type=int, default=500)
    args = ap.parse_args()

    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))
    p32 = json.load(open(f"{ROOT}/eda/results_p32.json"))
    p33 = json.load(open(f"{ROOT}/eda/results_p33.json"))
    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]

    out["declaration"] = dict(
        text=__doc__.split("=" * 70)[1].strip(),
        additions=__doc__.split("=" * 70)[2].strip(),
        anchors=__doc__.split("ANCHORS. The run aborts on any failure.")[1]
        .split("Reads results_p26")[0].strip(),
        primary=PRIMARY, seed=SEED, anchor_seed=ANCHOR_SEED,
        n_boot=args.boot, n_perm=args.perm, n_multi=args.multi,
        n_passive=args.passive)
    say(__doc__.split("=" * 70)[1].strip())

    anchors = []

    # ------------------------------------------------ 61.1 the static anchors
    hdr("61.1 A2/A3/A4: the observed cell, before anything is resampled")
    obs = {}
    for ym in SURVEY_MONTHS:
        s = p27["survey"][f"{ym}|WE|seoul"]
        C = np.array(s["C"])[np.ix_(sq, sq)]
        n_ego = np.array(s["n_ego"], float)
        T, asym = symmetrise(C, pops[ym][sq])
        rk = rank_stats(T)
        n_obs = float(n_ego[sq] @ C.sum(1) * s["n_days"])
        obs[ym] = dict(T=T, C=C, rank=rk, n_ego=int(n_ego.sum()),
                       n_contacts=n_obs,
                       zero_cells_C=int((C == 0).sum()),
                       zero_cells_T=int((T == 0).sum()),
                       excess=excess_stats(T, n_obs=n_obs), asymmetry=asym)
        pub = p32["spectrum"][f"survey|{ym}|WE|seoul"]
        for k in RANK_KEYS:
            anchors.append(check(f"A2 {ym} {k}", rk[k], pub[k]))
        sv_got = np.array(rk["singular_values"], float)
        sv_pub = np.array(pub["singular_values"], float)
        anchors.append(check(f"A2 {ym} singular_values max|diff|",
                             float(np.abs(sv_got - sv_pub).max()), 0.0))
        anchors.append(check(f"A3 {ym} n_ego", n_ego.sum(), N_EGO_PUBLISHED))
        anchors.append(check(f"A3 {ym} n_ego 80+ band", n_ego[-1], 0.0))
        say(f"  {ym}: sigma2/sigma1 {rk['sigma2_over_sigma1']:.10f}, "
            f"sigma1 share {rk['sigma1_share']:.10f}, "
            f"n_ego {int(n_ego.sum())}, zeros in C {int((C == 0).sum())}/"
            f"{C.size}, zeros in T {int((T == 0).sum())}")
    anchors.append(check("A4 202312 zero cells in sq block",
                         obs[202312]["zero_cells_C"], N_ZERO_PUBLISHED))
    anchors.append(check("A2 202312 sigma2/sigma1 == declared constant",
                         obs[202312]["rank"]["sigma2_over_sigma1"], OBS_S2S1))
    anchors.append(check("A2 202312 sigma1 share == declared constant",
                         obs[202312]["rank"]["sigma1_share"], OBS_S1SHARE))

    # the passive side of panel 3(b), same treatment
    passive = {}
    for ym in SURVEY_MONTHS:
        for lev, key in (("dong", f"{ym}|WE|dong|holidayfree"), ("gu", f"{ym}|WE|gu")):
            if key not in p26["matrices"]:
                continue
            A = np.array(p26["matrices"][key]["A"])[np.ix_(sq, sq)]
            T, _ = symmetrise(A / pops[ym][sq][:, None], pops[ym][sq])
            rk = rank_stats(T)
            pub = p32["spectrum"][f"passive|{key}"]
            for k in RANK_KEYS:
                anchors.append(check(f"A2 passive {key} {k}", rk[k], pub[k]))
            passive[(ym, lev)] = dict(T=T, rank=rk, excess=excess_stats(T))

    # ------------------------------------------- 61.2 A5 + A1: the stream anchor
    hdr("61.2 A5 ANCHOR: bit-exact replay of p32's rng consumption order")
    say("  p32 draws 32.4's four multinomial floors and 32.4b's two survey loops")
    say("  from ONE stream, so reproducing 32.4b requires replaying 32.4 first.")
    from p27_survey import (contact_matrix, ego_cubes, holiday_free_dows,
                            load_survey)
    ego, d = load_survey()
    d = d[d.panel.isin(["W", "E"])]        # NOT household -- see p32 32.4b
    cubes = {ym: ego_cubes(ego, d, True, ym=ym, dows=holiday_free_dows(ym))
             for ym in SURVEY_MONTHS}

    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}
    n_dev = {ym: float(sum(pops[ym][i] / cov.get(AGE_LABEL[AGES[i]], np.nan)
                           for i in sq)) for ym in SURVEY_MONTHS}

    rng_a = np.random.default_rng(ANCHOR_SEED)
    for ym in SURVEY_MONTHS:                       # p32 32.4, in p32's order
        fp = null_floor(passive[(ym, "dong")]["T"], n_dev[ym], rng=rng_a)
        fs = null_floor(obs[ym]["T"], obs[ym]["n_contacts"], rng=rng_a)
        anchors.append(check(
            f"A5 {ym} 32.4 passive floor median", fp["mi_bits"]["median"],
            p32["null_floor"][str(ym)]["passive"]["floor"]["mi_bits"]["median"]))
        anchors.append(check(
            f"A5 {ym} 32.4 survey floor median", fs["mi_bits"]["median"],
            p32["null_floor"][str(ym)]["survey"]["floor"]["mi_bits"]["median"]))

    replay = {}
    for ym in SURVEY_MONTHS:                       # p32 32.4b, in p32's order
        cube, eb, nd = cubes[ym]
        pop = pops[ym][sq]
        mi_b, mi_p, assort, s2_b, s2_p = [], [], [], [], []
        for _ in range(P32.BOOT):
            idx = rng_a.integers(0, len(eb), len(eb))
            C, _ = contact_matrix(cube[idx], eb[idx], nd)
            T, _ = symmetrise(C[np.ix_(sq, sq)], pop)
            st = excess_stats(T)
            mi_b.append(st["mi_bits"])
            assort.append(st["assortativity"])
            s2_b.append(rank_stats(T)["sigma2_over_sigma1"])
        for _ in range(P32.PERM):
            C, _ = contact_matrix(cube, rng_a.permutation(eb), nd)
            T, _ = symmetrise(C[np.ix_(sq, sq)], pop)
            mi_p.append(excess_stats(T)["mi_bits"])
            s2_p.append(rank_stats(T)["sigma2_over_sigma1"])
        st = p32["survey_floor_measured"][str(ym)]
        got = dict(mi_perm_median=float(np.median(mi_p)),
                   mi_perm_p95=float(np.percentile(mi_p, 95)),
                   mi_boot_median=float(np.median(mi_b)),
                   mi_boot_lo=float(np.percentile(mi_b, 2.5)),
                   mi_boot_hi=float(np.percentile(mi_b, 97.5)),
                   assort_boot_lo=float(np.percentile(assort, 2.5)),
                   assort_boot_hi=float(np.percentile(assort, 97.5)))
        for k, v in got.items():
            anchors.append(check(f"A5 {ym} 32.4b {k}", v, st[k]))
        pub = p27["survey_assortativity_bootstrap"][str(ym)]
        anchors.append(check(f"A1 {ym} p27 published assort lo (tol 2e-2)",
                             got["assort_boot_lo"], pub["lo"], 2e-2))
        anchors.append(check(f"A1 {ym} p27 published assort hi (tol 2e-2)",
                             got["assort_boot_hi"], pub["hi"], 2e-2))
        # Free of charge: the same draws, read for the spectral statistic. A
        # second seed for the headline, at no extra randomness.
        replay[str(ym)] = dict(
            sigma2_over_sigma1_perm=summarise(s2_p),
            sigma2_over_sigma1_boot=summarise(s2_b),
            verdicts=verdicts(obs[ym]["rank"]["sigma2_over_sigma1"], s2_p, s2_b))
        say(f"  {ym}: mi_perm_median {got['mi_perm_median']!r} vs p32 "
            f"{st['mi_perm_median']!r}")
        say(f"  {ym}: ANCHOR assortativity CI [{got['assort_boot_lo']:+.4f}, "
            f"{got['assort_boot_hi']:+.4f}] vs p27's published "
            f"[{pub['lo']:+.4f}, {pub['hi']:+.4f}]")
    got_perm_median = [c["got"] for c in anchors
                       if c["name"] == "A5 202312 32.4b mi_perm_median"][0]
    anchors.append(check("A5 202312 mi_perm_median vs the quoted constant",
                         got_perm_median, P32_PERM_MEDIAN_202312))

    n_ok = sum(a["ok"] for a in anchors)
    worst = max(a["abs_diff"] for a in anchors)
    out["anchors"] = dict(n=len(anchors), n_ok=n_ok, max_abs_diff=worst,
                          checks=anchors)
    say(f"\n  {n_ok}/{len(anchors)} anchors pass, worst |diff| {worst:.3e}")
    assert not FAIL, "ANCHOR FAILURE:\n  " + "\n  ".join(FAIL)

    # --------------------------------- 61.3 the declared draw, this phase's seed
    hdr("61.3 the declared draw: sigma2/sigma1 under permutation and bootstrap")
    rng = np.random.default_rng(SEED)
    cells = {}
    for ym in SURVEY_MONTHS:
        cube, eb, nd = cubes[ym]
        pop = pops[ym][sq]
        boot = {"sigma2_over_sigma1": [], "sigma1_share": [],
                "assortativity": [], "mi_bits": [], "zero_cells_C": []}
        perm = {k: [] for k in boot}
        for _ in range(args.boot):
            idx = rng.integers(0, len(eb), len(eb))
            C, _ = contact_matrix(cube[idx], eb[idx], nd)
            Cq = C[np.ix_(sq, sq)]
            T, _ = symmetrise(Cq, pop)
            rk, ex = rank_stats(T), excess_stats(T)
            boot["sigma2_over_sigma1"].append(rk["sigma2_over_sigma1"])
            boot["sigma1_share"].append(rk["sigma1_share"])
            boot["assortativity"].append(ex["assortativity"])
            boot["mi_bits"].append(ex["mi_bits"])
            boot["zero_cells_C"].append(int((Cq == 0).sum()))
        for _ in range(args.perm):
            C, _ = contact_matrix(cube, rng.permutation(eb), nd)
            Cq = C[np.ix_(sq, sq)]
            T, _ = symmetrise(Cq, pop)
            rk, ex = rank_stats(T), excess_stats(T)
            perm["sigma2_over_sigma1"].append(rk["sigma2_over_sigma1"])
            perm["sigma1_share"].append(rk["sigma1_share"])
            perm["assortativity"].append(ex["assortativity"])
            perm["mi_bits"].append(ex["mi_bits"])
            perm["zero_cells_C"].append(int((Cq == 0).sum()))

        v2 = verdicts(obs[ym]["rank"]["sigma2_over_sigma1"],
                      perm["sigma2_over_sigma1"], boot["sigma2_over_sigma1"],
                      larger_is_structure=True)
        v1 = verdicts(obs[ym]["rank"]["sigma1_share"],
                      perm["sigma1_share"], boot["sigma1_share"],
                      larger_is_structure=False)
        cells[str(ym)] = dict(
            n_ego=obs[ym]["n_ego"], n_contacts=obs[ym]["n_contacts"],
            n_days=int(nd),
            observed=dict(
                sigma2_over_sigma1=obs[ym]["rank"]["sigma2_over_sigma1"],
                sigma1_share=obs[ym]["rank"]["sigma1_share"],
                best_rank1_resid=obs[ym]["rank"]["best_rank1_resid"],
                pm_rank1_resid=obs[ym]["rank"]["pm_rank1_resid"],
                assortativity=obs[ym]["excess"]["assortativity"],
                mi_bits=obs[ym]["excess"]["mi_bits"],
                zero_cells_C=obs[ym]["zero_cells_C"],
                zero_cells_T=obs[ym]["zero_cells_T"]),
            permutation={k: summarise(v) for k, v in perm.items()},
            bootstrap={k: summarise(v) for k, v in boot.items()},
            verdict_sigma2_over_sigma1=v2, verdict_sigma1_share=v1,
            replay_at_p32_seed=replay[str(ym)],
            # Kept in full, not only summarised: the figure has to draw this
            # distribution, and a figure script that re-ran the draw would be
            # drawing a different one. p47 reads these.
            draws=dict(
                permutation_sigma2_over_sigma1=[
                    float(x) for x in perm["sigma2_over_sigma1"]],
                bootstrap_sigma2_over_sigma1=[
                    float(x) for x in boot["sigma2_over_sigma1"]],
                permutation_zero_cells_C=[
                    int(x) for x in perm["zero_cells_C"]],
                permutation_sigma1_share=[
                    float(x) for x in perm["sigma1_share"]],
                bootstrap_sigma1_share=[
                    float(x) for x in boot["sigma1_share"]]))

        say(f"\n  {ym}   n_ego {obs[ym]['n_ego']}, "
            f"{obs[ym]['zero_cells_C']} of 225 cells empty")
        say(f"    sigma2/sigma1  observed {v2['observed']:.4f}")
        say(f"      permutation  median {v2['null_median']:.4f}  "
            f"p95 {v2['null_threshold']:.4f}  "
            f"p97.5 {v2['null_edge']:.4f}")
        say(f"      bootstrap    median {v2['bootstrap_median']:.4f}  "
            f"p2.5 {v2['bootstrap_edge']:.4f}")
        say(f"      PRIMARY   observed > null p95   -> "
            f"{'ESCAPES' if v2['primary_escape'] else 'DOES NOT ESCAPE'}"
            f"   (perm p = {v2['permutation_p_value']:.4f})")
        say(f"      SECONDARY boot p2.5 > null p97.5 -> "
            f"{'CLEARS' if v2['secondary_escape'] else 'DOES NOT CLEAR'}")
        say(f"    sigma1 share   observed {v1['observed']:.4f}  "
            f"null median {v1['null_median']:.4f}  p5 {v1['null_threshold']:.4f}")
        say(f"      PRIMARY   observed < null p5    -> "
            f"{'ESCAPES' if v1['primary_escape'] else 'DOES NOT ESCAPE'}"
            f"   (perm p = {v1['permutation_p_value']:.4f})")
        say(f"      SECONDARY boot p97.5 < null p2.5 -> "
            f"{'CLEARS' if v1['secondary_escape'] else 'DOES NOT CLEAR'}")
        z = cells[str(ym)]["permutation"]["zero_cells_C"]
        say(f"    empty cells per permuted draw: median {z['median']:.0f}, "
            f"range {z['min']:.0f}-{z['max']:.0f}, observed "
            f"{obs[ym]['zero_cells_C']}")

    # ------------------- 61.4 what the permutation null cannot make rank one
    hdr("61.4 the deterministic rank-2 component the permutation null carries")
    say("  Under the permutation every row of C becomes the same mean alter")
    say("  vector v, so T = (pop v' + v pop')/2, which is rank TWO unless v is")
    say("  proportional to pop. Reciprocity under true proportionate mixing")
    say("  would force that proportionality; the observed alter margin does not")
    say("  satisfy it. So part of the null's sigma2/sigma1 is not sampling")
    say("  noise at all -- it inflates the null and makes escape HARDER.")
    decomp = {}
    for ym in SURVEY_MONTHS:
        cube, eb, nd = cubes[ym]
        pop = pops[ym][sq]
        C0, n_ego0 = contact_matrix(cube, eb, nd)
        v = cube.sum(0)[sq] / (len(eb) * nd)          # mean alter vector per day
        Cbar = np.repeat(v[None, :], len(sq), axis=0)  # every row identical
        Tbar, _ = symmetrise(Cbar, pop)
        rk = rank_stats(Tbar)
        pu, pv = pop / np.linalg.norm(pop), v / np.linalg.norm(v)
        cos = float(pu @ pv)
        closed = float((1 - cos) / (1 + cos))
        decomp[str(ym)] = dict(
            expected_sigma2_over_sigma1=rk["sigma2_over_sigma1"],
            expected_sigma1_share=rk["sigma1_share"],
            cos_pop_alter=cos, closed_form_sigma2_over_sigma1=closed,
            null_median=cells[str(ym)]["permutation"]
            ["sigma2_over_sigma1"]["median"])
        say(f"  {ym}: cos(pop, alter margin) {cos:.4f} -> deterministic "
            f"sigma2/sigma1 {rk['sigma2_over_sigma1']:.4f} "
            f"(closed form {closed:.4f}); the permutation null's median is "
            f"{decomp[str(ym)]['null_median']:.4f}")
    out["permutation_null_floor_decomposition"] = decomp

    # ----------------------- 61.5 the third null: exactly rank one in truth
    hdr("61.5 a multinomial null drawn from r r' at the same contact count")
    say("  Exactly rank one in truth, but with neither respondent clustering")
    say("  nor the emptiness. It brackets the permutation null from the other")
    say("  side and is reported, never substituted for it.")
    rng_m = np.random.default_rng(SEED + 1)
    multi = {}
    for ym in SURVEY_MONTHS:
        e, r, E = pm_null(obs[ym]["T"])
        p = (E / E.sum()).ravel()
        n = int(min(obs[ym]["n_contacts"], 5e6))
        s2, s1, zc = [], [], []
        for _ in range(args.multi):
            dr = rng_m.multinomial(n, p).reshape(E.shape).astype(float)
            dr = (dr + dr.T) / 2
            if dr.sum() == 0:
                continue
            rk = rank_stats(dr)
            s2.append(rk["sigma2_over_sigma1"])
            s1.append(rk["sigma1_share"])
            zc.append(int((dr == 0).sum()))
        multi[str(ym)] = dict(
            n_draw=n, sigma2_over_sigma1=summarise(s2),
            sigma1_share=summarise(s1), zero_cells_T=summarise(zc),
            observed_sigma2_over_sigma1=obs[ym]["rank"]["sigma2_over_sigma1"],
            escapes_p95=bool(obs[ym]["rank"]["sigma2_over_sigma1"]
                             > float(np.percentile(s2, 95))))
        say(f"  {ym}: multinomial null sigma2/sigma1 median "
            f"{multi[str(ym)]['sigma2_over_sigma1']['median']:.4f}, p95 "
            f"{multi[str(ym)]['sigma2_over_sigma1']['quantiles']['p95']:.4f}; "
            f"observed {obs[ym]['rank']['sigma2_over_sigma1']:.4f} -> "
            f"{'escapes' if multi[str(ym)]['escapes_p95'] else 'does not escape'}")
    out["survey_multinomial_null"] = multi

    # ----------------------------- 61.6 the passive arm, declared SECONDARY
    hdr("61.6 the passive side of 3(b), which also had no null (SECONDARY)")
    say("  A multinomial is not the passive matrix's noise model -- devices")
    say("  appear in many cells, coverage weights are applied and masked cells")
    say("  are imputed -- so this OVERSTATES the precision and the floor below")
    say("  is a LOWER bound on the true passive noise floor.")
    rng_p = np.random.default_rng(SEED + 2)
    pas = {}
    for ym in SURVEY_MONTHS:
        n_p33 = p33["R1_effective_sample"][str(ym)]["n_device_equiv"]
        n_p32 = p32["null_floor"][str(ym)]["passive"]["n_effective"]
        anchors.append(check(f"passive n_dev p33 vs p32 {ym}", n_p33, n_p32,
                             1e-6))
        for lev in ("dong", "gu"):
            if (ym, lev) not in passive:
                continue
            T = passive[(ym, lev)]["T"]
            e, r, E = pm_null(T)
            n = int(min(n_dev[ym], 5e6))
            null_s2, boot_s2, null_s1, boot_s1 = [], [], [], []
            pn = (E / E.sum()).ravel()
            pe = (e / e.sum()).ravel()
            for _ in range(args.passive):
                dr = rng_p.multinomial(n, pn).reshape(E.shape).astype(float)
                dr = (dr + dr.T) / 2
                rk = rank_stats(dr)
                null_s2.append(rk["sigma2_over_sigma1"])
                null_s1.append(rk["sigma1_share"])
            for _ in range(args.passive):
                dr = rng_p.multinomial(n, pe).reshape(E.shape).astype(float)
                dr = (dr + dr.T) / 2
                rk = rank_stats(dr)
                boot_s2.append(rk["sigma2_over_sigma1"])
                boot_s1.append(rk["sigma1_share"])
            o2 = passive[(ym, lev)]["rank"]["sigma2_over_sigma1"]
            o1 = passive[(ym, lev)]["rank"]["sigma1_share"]
            pas[f"{ym}|{lev}"] = dict(
                n_device_equiv=n_dev[ym], n_draw=n,
                observed_sigma2_over_sigma1=o2, observed_sigma1_share=o1,
                null_sigma2_over_sigma1=summarise(null_s2),
                null_sigma1_share=summarise(null_s1),
                bootstrap_sigma2_over_sigma1=summarise(boot_s2),
                bootstrap_sigma1_share=summarise(boot_s1),
                verdict_sigma2_over_sigma1=verdicts(o2, null_s2, boot_s2),
                verdict_sigma1_share=verdicts(o1, null_s1, boot_s1,
                                              larger_is_structure=False),
                noise_floor_over_signal=float(
                    np.median(null_s2) / o2) if o2 else np.nan)
            v = pas[f"{ym}|{lev}"]["verdict_sigma2_over_sigma1"]
            say(f"  {ym} {lev:<5} observed {o2:.4f}   null median "
                f"{v['null_median']:.4f}  p95 {v['null_threshold']:.4f}   "
                f"boot p2.5 {v['bootstrap_edge']:.4f}   -> primary "
                f"{'escapes' if v['primary_escape'] else 'does not escape'}, "
                f"secondary "
                f"{'clears' if v['secondary_escape'] else 'does not clear'}")
    # The same comparison p32 32.4 already publishes for mutual information --
    # "the passive matrix's entire excess is smaller than the survey's own
    # floor" -- restated in the spectral statistic. It is a derived reading of
    # two distributions already computed above, not a new draw, and it decides
    # nothing: the escape criterion is settled in 61.3.
    cross = {}
    for ym in SURVEY_MONTHS:
        dr = np.array(cells[str(ym)]["draws"]["permutation_sigma2_over_sigma1"],
                      float)
        for lev in ("dong", "gu"):
            if (ym, lev) not in passive:
                continue
            o = passive[(ym, lev)]["rank"]["sigma2_over_sigma1"]
            cross[f"{ym}|{lev}"] = dict(
                passive_sigma2_over_sigma1=o,
                survey_perm_null_median=float(np.median(dr)),
                percentile_within_survey_null=float((dr < o).mean() * 100),
                inside_survey_null_95=bool(
                    float(np.percentile(dr, 2.5)) <= o
                    <= float(np.percentile(dr, 97.5))))
            say(f"  {ym} {lev:<5} passive sigma2/sigma1 {o:.4f} sits at the "
                f"{cross[f'{ym}|{lev}']['percentile_within_survey_null']:.0f}th "
                f"percentile of the SURVEY's permutation null "
                f"(median {np.median(dr):.4f})")
    out["passive_inside_survey_null"] = dict(
        note="the spectral restatement of p32 32.4's signature sentence; "
             "derived from distributions already drawn, decides nothing",
        cells=cross)

    out["passive_arm"] = dict(
        status="SECONDARY, declared; multinomial is a lower bound on the true "
               "passive noise floor",
        p33_noise_bias_assortativity=p33["R1_effective_sample"]["202312"]
        ["stats"]["assortativity"]["noise_bias"],
        cells=pas)

    # ------------------------- 61.7 a DECLARED EXPECTATION THAT DID NOT HOLD
    hdr("61.7 the emptiness clause of the declaration is wrong, and that is a "
        "finding")
    say("  The declaration says the zero-cell count is reported 'so a reader can")
    say("  see the null carries the same emptiness the observation does'. It does")
    say("  NOT. Permuting the pairing SPREADS contacts across cells and fills")
    say("  them in, so the observed emptiness is itself structure, not sparsity.")
    say("  Everything below this line is POST HOC -- added after seeing that,")
    say("  changing no cell, no statistic and no escape criterion, and reported")
    say("  as post hoc. It answers what the clause was there to answer: would a")
    say("  null AS EMPTY as the observation reach 0.601?")
    empt = {}
    for ym in SURVEY_MONTHS:
        z = np.array(cells[str(ym)]["draws"]["permutation_zero_cells_C"], float)
        s = np.array(cells[str(ym)]["draws"]["permutation_sigma2_over_sigma1"],
                     float)
        slope, intercept = np.polyfit(z, s, 1)
        top = z >= np.percentile(z, 90)
        empt[str(ym)] = dict(
            observed_zero_cells=obs[ym]["zero_cells_C"],
            null_zero_cells_median=float(np.median(z)),
            null_zero_cells_max=float(z.max()),
            slope_sigma2_per_empty_cell=float(slope),
            intercept=float(intercept),
            pearson_r=float(np.corrcoef(z, s)[0, 1]),
            emptiest_decile_zero_cells_min=float(z[top].min()),
            emptiest_decile_sigma2_median=float(np.median(s[top])),
            emptiest_decile_sigma2_max=float(s[top].max()),
            linear_extrapolation_to_observed_emptiness=float(
                intercept + slope * obs[ym]["zero_cells_C"]),
            observed_sigma2_over_sigma1=obs[ym]["rank"]["sigma2_over_sigma1"],
            null_sigma2_max=float(s.max()),
            note="linear extrapolation beyond the null's support; reported to "
                 "bound the emptiness effect, not as an estimate")
        e = empt[str(ym)]
        say(f"  {ym}: observed {e['observed_zero_cells']} empty cells, null "
            f"median {e['null_zero_cells_median']:.0f} (max "
            f"{e['null_zero_cells_max']:.0f})")
        say(f"        within the null, sigma2/sigma1 rises "
            f"{e['slope_sigma2_per_empty_cell']:+.5f} per empty cell "
            f"(r = {e['pearson_r']:+.3f}); extrapolated linearly to "
            f"{e['observed_zero_cells']} empty cells it reaches "
            f"{e['linear_extrapolation_to_observed_emptiness']:.4f}, against "
            f"an observed {e['observed_sigma2_over_sigma1']:.4f}")
        say(f"        the single emptiest permuted draw reaches "
            f"{e['null_sigma2_max']:.4f}")
    out["emptiness_diagnostic_post_hoc"] = dict(
        status="POST HOC: added after the declared emptiness clause proved "
               "false. No cell, statistic or criterion changed.",
        declared_clause="the count of exactly-zero cells in each permuted "
                        "draw, so a reader can see the null carries the same "
                        "emptiness the observation does",
        held=False, cells=empt)

    out["cells"] = cells
    out["anchors"]["n"] = len(anchors)
    out["anchors"]["n_ok"] = sum(a["ok"] for a in anchors)
    out["anchors"]["max_abs_diff"] = max(a["abs_diff"] for a in anchors)
    out["anchors"]["checks"] = anchors
    assert not FAIL, "ANCHOR FAILURE:\n  " + "\n  ".join(FAIL)

    # --------------------------------------------------------- 61.8 the branch
    hdr("61.8 which pre-declared branch this lands in")
    pv = cells["202312"]["verdict_sigma2_over_sigma1"]
    if pv["primary_escape"] and pv["secondary_escape"]:
        branch = ("escapes both: the caption sentence gets stronger -- (b) needs "
                  "no null to be read, and it survives one anyway")
    elif pv["primary_escape"]:
        branch = ("escapes the p95 but not the CI rule: quote the primary, say "
                  "plainly that the stricter rule does not clear")
    else:
        branch = ("does NOT escape: the caption sentence at "
                  "the Figure 3(b) caption is wrong as written and "
                  "must go; the two spectra are not comparable raw, and 3(b) "
                  "is redrawn as each matrix against its own null")
    out["branch"] = dict(
        primary_escape=pv["primary_escape"],
        secondary_escape=pv["secondary_escape"],
        permutation_p95=pv["null_threshold"],
        permutation_p_value=pv["permutation_p_value"],
        landed=branch)
    say(f"  {branch}")

    with open(f"{RESULTS}/results_p61.json", "w") as fh:
        json.dump(out, fh, indent=1)
    say(f"\nwrote {RESULTS}/results_p61.json")


if __name__ == "__main__":
    main()
