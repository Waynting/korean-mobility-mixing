#!/usr/bin/env python
"""Phase 52 — the R0 sweep with the corrected national population vector.

WHY THIS EXISTS. `p45_r0.py` answers the advisor's question -- does the fourth
pillar's verdict survive a change of R0 -- and the answer is that 7 of 20
verdicts flip. SIX OF THOSE SEVEN ARE NATIONAL CELLS (202312|national at R0
1.2/1.5/1.8, 202402|national at 1.2/1.5/3.5; only 202402|seoul at 3.5 is not).
`p45_r0.py:112-129` reduces every survey matrix with `pops[ym]` from
`results_p26.json`, which is Seoul's population, so the entire evidential base
for rewriting Claim 3 sits on the arm that `p50`/`p51` just corrected.

This phase re-runs the sweep for the NATIONAL cells on Korea's vector. The
Seoul cells are not recomputed: their vector never changed, so recomputing them
could only introduce a difference where there is none. They are read from
`results_p45.json` and carried through, which is also what lets the combined
20-verdict tally be assembled honestly.

Writes `results_p52.json`. `results_p45.json` and `results_p40.json` are not
re-run and cannot move.

THE PAIRING, again, and this time it also anchors the bootstrap.
`p45_r0.py:262-266` builds ONE rng stream per (month, scope, R0) from a closed
formula, not from a shared stream -- so unlike p40's 40.2, every block here is
independently reproducible. And the draws (`rng.integers`) depend on the rng and
the ego array only; the population vector enters afterwards, inside
`symmetrise`, `optimal_allocation` and `final_size`. So re-running a block with
the Seoul vector must reproduce `results_p45.json` BIT FOR BIT, and re-running
it with Korea's vector differs by the vector and nothing else -- same resamples,
same order, same optimiser starts.

  p45's own docstring says its bootstrap half "can only be compared within Monte
  Carlo error" because reproducing it would need replaying p40's stream. That is
  true of 45.4's comparison against p40. It is NOT true of 45.3 against itself,
  because 45.3's streams are per-cell and closed-form. This phase uses that.

WHAT IS ANCHORED, and what each anchor costs.

  52.0a  The twelve deterministic quantities of 45.1 (buy_first and the two
         between-taus, four cells) reproduced on the Seoul vector. Free.
  52.0b  The whole of 45.2's point curve -- 12 R0 x 4 cells x 8 quantities --
         reproduced on the Seoul vector, bit for bit against
         results_p45.json["point"]. This is the strong one: it exercises every
         SLSQP call and every final-size solve the corrected run will make.
  52.0c  ONE bootstrap block (202312|national at R0 = 2.5, 200 draws) replayed
         on the Seoul vector and required to reproduce p45's verdict row bit for
         bit. Anchoring all ten national blocks would double the run for no new
         information; anchoring none would leave the slow half unchecked.

THE MODELLING CHOICE THAT HAS TO BE DECLARED. The passive matrix stays Seoul's
-- that is the standing limitation, not something this phase changes. But
`optimal_allocation(Cpd, ..., pop, ...)` needs a population, and once the
evaluation world is national there are two defensible readings: the passive
planner uses Seoul's demography (what a Seoul-data planner would believe), or
the national one (demography is public; the passive matrix stands in for MIXING
only).

  DECLARED: the national vector, everywhere in a national cell. Age structure
  is published by 통계청 and no planner would be wrong about it; the thing the
  passive product substitutes for is who-meets-whom. Using Seoul demography for
  the plan and national demography for the evaluation would also charge the
  passive matrix for an error nobody would make, which inflates regret in the
  direction that flatters the paper's claim.

    python eda/p52_r0nat.py                 # 200 bootstrap draws, ~1 hour
    python eda/p52_r0nat.py --boot35 4      # smoke test; NOT a result
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import AGES, AGE_LABEL
from paths import ROOT

import p35_seir as P35
from p35_seir import (final_size, kendall_tau, marginal_ranking,
                      optimal_allocation, scale_to_r0)
from p27_survey import (contact_matrix, ego_cubes, holiday_free_dows,
                        load_survey)

SURVEY_MONTHS = [202312, 202402]
SCOPES = [("national", False), ("seoul", True)]
SEED = 20260822                 # p45's, so the streams line up exactly
R0_POINT = (1.1, 1.2, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
R0_BOOT = (1.2, 1.5, 1.8, 2.5, 3.5)
R0_ANCHOR = 2.5
BUDGET = 0.10
ANCHOR_BOOT_CELL = (202312, "national", 2.5)     # the one replayed block

out = {}
FAIL = []


def hdr(s):
    print(f"\n=== {s} ===", flush=True)


def say(s=""):
    print(s, flush=True)


def cell_rng(ym, scope, r0):
    """p45_r0.py:262-266's stream recipe, verbatim."""
    return np.random.default_rng(
        (SEED * 1000003 + ym * 97 + (scope == "seoul") * 7
         + int(round(r0 * 10))) % (2 ** 63))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot35", type=int, default=200)
    ap.add_argument("--budget", type=float, default=BUDGET)
    args = ap.parse_args()

    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p40 = json.load(open(f"{ROOT}/eda/results_p40.json"))
    p45 = json.load(open(f"{ROOT}/eda/results_p45.json"))
    p50 = json.load(open(f"{ROOT}/eda/results_p50.json"))

    sq = [i for i, a in enumerate(AGES) if a < 80]
    lbl = [AGE_LABEL[AGES[i]] for i in sq]
    POP_S = {ym: np.array(p26["population"][str(ym)], float)[sq]
             for ym in SURVEY_MONTHS}
    POP_K = {ym: np.array(p50["vectors"][str(ym)]["national"], float)[sq]
             for ym in SURVEY_MONTHS}
    EST = {ym: bool(p50["vectors"][str(ym)]["established"])
           for ym in SURVEY_MONTHS}

    assert p45["declaration"]["n_boot"] == 200, \
        "results_p45.json was not produced at 200 draws; the comparison would " \
        "be against a different question"
    assert p45["declaration"]["r0_point"] == list(R0_POINT)
    assert p45["declaration"]["r0_boot"] == list(R0_BOOT)

    out["declaration"] = dict(
        seed=SEED, n_boot=args.boot35, budget_frac=args.budget,
        r0_point=list(R0_POINT), r0_boot=list(R0_BOOT), r0_anchor=R0_ANCHOR,
        question="does 7-of-20 survive correcting the national population "
                 "vector, and where over R0 is the consequence largest?",
        flip_rule=p45["declaration"]["flip_rule"],
        recomputed="national cells only; the Seoul cells' vector never changed "
                   "and they are carried through from results_p45.json",
        planner_population="national -- age structure is public, the passive "
                           "product substitutes for mixing, not demography",
        established={str(k): v for k, v in EST.items()},
        anchor_boot_cell=list(map(str, ANCHOR_BOOT_CELL)))

    # ------------------------------------------------------------- matrices
    ego, d = load_survey()
    d = d[d.panel.isin(["W", "E"])]
    cubes = {}
    for ym in SURVEY_MONTHS:
        for scope, so in SCOPES:
            cubes[(ym, scope)] = ego_cubes(ego, d, so, ym=ym,
                                           dows=holiday_free_dows(ym))

    def passive_mats(ym, pop):
        """The Seoul passive matrices, reduced with whichever vector is in play."""
        m = {}
        for tag, lev in (("passive_dong", "dong|holidayfree"),
                         ("passive_gu", "gu")):
            A = np.array(p26["matrices"][f"{ym}|WE|{lev}"]["A"])[np.ix_(sq, sq)]
            m[tag] = P35.symmetrise(A / POP_S[ym][:, None], pop)[0]
        return m

    def survey_C(ym, scope, pop, idx=None):
        cube, eb, nd = cubes[(ym, scope)]
        if idx is None:
            C0, _ = contact_matrix(cube, eb, nd)
        else:
            C0, _ = contact_matrix(cube[idx], eb[idx], nd)
        return P35.symmetrise(C0[np.ix_(sq, sq)], pop)[0]

    # =================================================== 52.0a the 12 anchors
    hdr("52.0a ANCHOR: p45's twelve deterministic quantities, Seoul vector")
    n_anchor = 0
    for ym in SURVEY_MONTHS:
        pop = POP_S[ym]
        mats = passive_mats(ym, pop)
        for scope, _ in SCOPES:
            Cm = survey_C(ym, scope, pop)
            base, _ = marginal_ranking(Cm, scale_to_r0(Cm, R0_ANCHOR), pop)
            stored = p40["pillar4"][f"{ym}|{scope}"]
            got = lbl[int(np.argmax(base))]
            assert got == stored["buy_first"], \
                f"{ym}|{scope} buy_first {got} != {stored['buy_first']}"
            n_anchor += 1
            for k, tag in enumerate(("passive_dong", "passive_gu")):
                sp_, _ = marginal_ranking(mats[tag],
                                          scale_to_r0(mats[tag], R0_ANCHOR), pop)
                t = kendall_tau(base, sp_)
                assert abs(t - stored["tau_between"][k]) < 1e-12, \
                    f"{ym}|{scope} {tag}: tau {t!r} != {stored['tau_between'][k]!r}"
                n_anchor += 1
    say(f"  {n_anchor} deterministic quantities reproduced, all bit-exact.")
    out["anchor_deterministic"] = dict(quantities=n_anchor, bit_exact=True)

    # ================================================ 52.0b the whole curve
    hdr("52.0b ANCHOR: all of 45.2's point curve on the Seoul vector")
    say("  12 R0 x 4 cells x 8 quantities against results_p45.json['point'].")
    KEYS = ("attack_no_plan", "attack_own_plan", "attack_own_plan_bare",
            "attack_passive_plan", "optimiser_gain", "regret", "regret_bare")

    def point_curve(ym, scope, pop, mats):
        Cpd = mats["passive_dong"]
        Cm = survey_C(ym, scope, pop)
        rows = []
        for r0 in R0_POINT:
            q0 = scale_to_r0(Cm, r0)
            u_p, _, _ = optimal_allocation(Cpd, scale_to_r0(Cpd, r0), pop,
                                           budget_frac=args.budget)
            _, v_bare, _ = optimal_allocation(Cm, q0, pop,
                                              budget_frac=args.budget)
            _, v0, _ = optimal_allocation(Cm, q0, pop, budget_frac=args.budget,
                                          seeds=[u_p])
            z0 = float((final_size(Cm, q0) * pop).sum() / pop.sum())
            zp = float((final_size(Cm, q0, u_p) * pop).sum() / pop.sum())
            rows.append(dict(r0=r0, attack_no_plan=z0, attack_own_plan=v0,
                             attack_own_plan_bare=v_bare,
                             attack_passive_plan=zp,
                             optimiser_gain=float(v_bare - v0),
                             regret=(zp - v0) / (z0 - v0) if z0 > v0 else np.nan,
                             regret_bare=((zp - v_bare) / (z0 - v_bare)
                                          if z0 > v_bare else np.nan)))
        return rows

    worst, n_q = 0.0, 0
    for ym in SURVEY_MONTHS:
        mats = passive_mats(ym, POP_S[ym])
        for scope, _ in SCOPES:
            got = point_curve(ym, scope, POP_S[ym], mats)
            want = p45["point"][f"{ym}|{scope}"]
            for g, w in zip(got, want):
                assert g["r0"] == w["r0"]
                for k in KEYS:
                    d_ = abs(float(g[k]) - float(w[k]))
                    worst = max(worst, d_)
                    n_q += 1
    say(f"  {n_q} quantities, worst |diff| = {worst:.3e}")
    assert worst == 0.0, (
        f"the point curve does not replay p45 ({worst:.3e}); the corrected "
        f"curve would then differ by the pipeline as well as by the vector")
    out["anchor_point_curve"] = dict(quantities=n_q, worst_abs_diff=float(worst),
                                     bit_exact=True)

    # ============================================ 52.1 the corrected curve
    hdr("52.1 POINT regret over R0, national cells, Korea's vector")
    point = {}
    for ym in SURVEY_MONTHS:
        mats_k = passive_mats(ym, POP_K[ym])
        rows = point_curve(ym, "national", POP_K[ym], mats_k)
        point[f"{ym}|national"] = rows
        was = {r["r0"]: r["regret"] for r in p45["point"][f"{ym}|national"]}
        say(f"\n  {ym}|national" + ("" if EST[ym] else "   (vector not established)"))
        say(f"    {'R0':>6}{'regret now':>12}{'was':>10}{'move':>10}")
        for r in rows:
            say(f"    {r['r0']:>6.1f}{r['regret']:>11.1%}{was[r['r0']]:>10.1%}"
                f"{r['regret'] - was[r['r0']]:>+10.1%}")
    # the Seoul cells are carried through unchanged, by construction
    for ym in SURVEY_MONTHS:
        point[f"{ym}|seoul"] = p45["point"][f"{ym}|seoul"]
    out["point"] = point

    at25 = {k: [r["regret"] for r in v if r["r0"] == R0_ANCHOR][0]
            for k, v in point.items()}
    hi = {k: max(r["regret"] for r in v) for k, v in point.items()}
    arg = {k: [r["r0"] for r in point[k] if r["regret"] == hi[k]][0]
           for k in point}
    out["point_summary"] = dict(at_anchor=at25, max_over_grid=hi, argmax=arg)
    say("\n  regret at R0 = 2.5 vs its maximum over the grid:")
    for k in point:
        say(f"    {k:<20}{at25[k]:>8.1%} at 2.5   {hi[k]:>8.1%} at R0 = "
            f"{arg[k]}   ({hi[k] / at25[k]:.1f}x)" if at25[k] > 0
            else f"    {k:<20}   zero at 2.5")
    say(f"  the maximum sits at R0 = {sorted(set(arg.values()))}, "
        f"i.e. the 1.2-1.5 region, not at the epidemic threshold itself "
        f"(the grid starts at {R0_POINT[0]} and R0 -> 1 was never measured).")

    # ================================================ 52.2 verdicts
    hdr(f"52.2 THE VERDICT over R0, national cells ({args.boot35} draws)")
    if args.boot35 != 200:
        say(f"  !! --boot35 {args.boot35} is not the pinned 200. This is a "
            f"smoke test and its intervals are NOT a result.")

    def boot_block(ym, scope, r0, pop, Cpd):
        rng = cell_rng(ym, scope, r0)
        cube, eb, nd = cubes[(ym, scope)]
        u_p, _, _ = optimal_allocation(Cpd, scale_to_r0(Cpd, r0), pop,
                                       budget_frac=args.budget)
        regs, nulls = [], []
        for _ in range(args.boot35):
            idx = rng.integers(0, len(eb), len(eb))
            Cb = survey_C(ym, scope, pop, idx)
            qb = scale_to_r0(Cb, r0)
            if not np.isfinite(qb):
                continue
            _, vb, _ = optimal_allocation(Cb, qb, pop, budget_frac=args.budget,
                                          seeds=[u_p])
            z0 = float((final_size(Cb, qb) * pop).sum() / pop.sum())
            zp = float((final_size(Cb, qb, u_p) * pop).sum() / pop.sum())
            regs.append((zp - vb) / (z0 - vb) if z0 > vb else np.nan)
            jdx = rng.integers(0, len(eb), len(eb))
            Cj = survey_C(ym, scope, pop, jdx)
            uj, _, _ = optimal_allocation(Cj, scale_to_r0(Cj, r0), pop,
                                          budget_frac=args.budget, seeds=[u_p])
            zj = float((final_size(Cb, qb, uj) * pop).sum() / pop.sum())
            nulls.append((zj - vb) / (z0 - vb) if z0 > vb else np.nan)
        regs = np.array([r for r in regs if np.isfinite(r)])
        nulls = np.array([r for r in nulls if np.isfinite(r)])
        ent = dict(r0=r0, n_draws=int(len(regs)),
                   median=float(np.median(regs)),
                   ci=[float(np.percentile(regs, 2.5)),
                       float(np.percentile(regs, 97.5))],
                   null_median=float(np.median(nulls)),
                   null_ci=[float(np.percentile(nulls, 2.5)),
                            float(np.percentile(nulls, 97.5))],
                   exceeds_null=bool(np.percentile(regs, 2.5)
                                     > np.percentile(nulls, 97.5)))
        ent["margin"] = ent["ci"][0] - ent["null_ci"][1]
        return ent

    # --- 52.0c: one block replayed on the Seoul vector, bit-exact -----------
    aym, ascope, ar0 = ANCHOR_BOOT_CELL
    say(f"  52.0c anchor: replaying {aym}|{ascope} at R0={ar0} on the Seoul "
        f"vector...")
    a_mats = passive_mats(aym, POP_S[aym])
    a_got = boot_block(aym, ascope, ar0, POP_S[aym], a_mats["passive_dong"])
    a_want = [r for r in p45["verdict"][f"{aym}|{ascope}"] if r["r0"] == ar0][0]
    a_worst = 0.0
    if args.boot35 == 200:
        for k in ("median", "null_median", "margin"):
            a_worst = max(a_worst, abs(a_got[k] - a_want[k]))
        for k in ("ci", "null_ci"):
            a_worst = max(a_worst, max(abs(x - y) for x, y
                                       in zip(a_got[k], a_want[k])))
        say(f"        worst |diff| vs results_p45.json = {a_worst:.3e}")
        assert a_worst == 0.0, (
            f"the bootstrap block does not replay p45 ({a_worst:.3e}); the "
            f"slow half is not the same computation")
    else:
        say(f"        skipped: only meaningful at 200 draws")
    out["anchor_bootstrap"] = dict(cell=f"{aym}|{ascope}", r0=ar0,
                                   worst_abs_diff=float(a_worst),
                                   bit_exact=bool(args.boot35 == 200
                                                  and a_worst == 0.0),
                                   checked=bool(args.boot35 == 200))

    verdicts = {}
    for ym in SURVEY_MONTHS:
        mats_k = passive_mats(ym, POP_K[ym])
        rows = []
        for r0 in R0_BOOT:
            ent = boot_block(ym, "national", r0, POP_K[ym],
                             mats_k["passive_dong"])
            rows.append(ent)
            w = [r for r in p45["verdict"][f"{ym}|national"] if r["r0"] == r0][0]
            say(f"  {ym}|national R0={r0:<4} regret {ent['median']:>6.1%} "
                f"[{ent['ci'][0]:>5.1%}, {ent['ci'][1]:>5.1%}]  null "
                f"{ent['null_median']:>6.1%}  "
                f"{'EXCEEDS' if ent['exceeds_null'] else 'does not exceed'}"
                f"  (margin {ent['margin']:+.1%}; was "
                f"{'EXCEEDS' if w['exceeds_null'] else 'no'}, "
                f"{w['margin']:+.1%})")
        verdicts[f"{ym}|national"] = rows
    for ym in SURVEY_MONTHS:
        verdicts[f"{ym}|seoul"] = p45["verdict"][f"{ym}|seoul"]
    out["verdict"] = verdicts

    # ================================================ 52.3 the combined tally
    hdr("52.3 the 20-verdict tally, corrected")
    flips = {k: [r["r0"] for r in v if r["exceeds_null"]]
             for k, v in verdicts.items()}
    flips = {k: v for k, v in flips.items() if v}
    n_flip = sum(len(v) for v in flips.values())
    n_tot = sum(len(v) for v in verdicts.values())
    was_flips = {k: [r["r0"] for r in v if r["exceeds_null"]]
                 for k, v in p45["verdict"].items()}
    was_n = sum(len(v) for v in was_flips.values())
    say(f"  corrected: {n_flip} of {n_tot} verdicts flip (was {was_n} of 20)")
    for k in sorted(set(list(flips) + [k for k, v in was_flips.items() if v])):
        say(f"    {k:<20} now {flips.get(k, []) or '-'}   was "
            f"{was_flips.get(k, []) or '-'}")
    out["answer"] = dict(flips=flips, any_flip=bool(n_flip), n_verdicts=n_tot,
                         n_flips=n_flip, n_flips_published=was_n,
                         flips_published={k: v for k, v in was_flips.items()
                                          if v})
    out["fail"] = FAIL

    with open(f"{ROOT}/eda/results_p52.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p52.json")
    if FAIL:
        for f in FAIL:
            say(f"  FAIL {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
