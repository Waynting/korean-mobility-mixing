#!/usr/bin/env python
"""Phase 45 — the fourth pillar's verdict across R0, on the national arm.

WHY THIS EXISTS, in the letter's own words: "這個判定只在 R0 = 2.5 下成立，
R0 掃描這輪沒在全國臂重跑，換 R0 有可能翻." `p40` settled the fourth pillar at
n = 1,987 with a single R0, and the verdict it reported -- 0 of 4 cells has a
regret exceeding the "resample the survey and redo it" null -- is the one that
decides whether pillar 4 is a claim or a consequence.

AND 2.5 IS THE FRIENDLIEST PLACE TO ASK. `p35_seir.py:522-527` says it outright:
at R0 = 2.5 these matrices put the attack rate above 0.8, and in that regime
almost any allocation looks alike, so a comparison made only there UNDERSTATES
the difference between plans. The single R0 that was run is therefore the one
most likely to return "no difference" -- which is the answer we reported. That
is not a neutral choice and it has to be checked before an advisor is asked to
rule on it, not after.

WHAT IS DETERMINISTIC AND WHAT IS NOT, because that decides what can be anchored.
`optimal_allocation` is SLSQP from fixed starts and `marginal_ranking` is a
closed-form derivative, so at a given R0 the base ranking, the between-matrix
taus, the passive plan and the POINT regret carry no randomness at all. Only the
bootstrap arms do. So:

  - the deterministic quantities are anchored BIT-EXACTLY against p40's stored
    pillar4 at R0 = 2.5, and the run aborts if any of them moves;
  - the bootstrap arms use their own stream, so they cannot be bit-compared;
    they are checked against p40's stored medians within Monte Carlo error, and
    the check is reported as what it is rather than dressed up as an anchor.

Reproducing p40's stream instead would mean replaying every draw p40 makes
before 40.6 -- the same trap p40 itself had to climb out of for p32 -- and it
would buy nothing here, because the question is whether the VERDICT moves with
R0, not whether one bootstrap median reproduces.

THE GRID. Point quantities on a fine grid (cheap, 12 values); the bootstrap on
five, because each draw costs two SLSQP solves and the verdict is what needs the
interval. 1.2 is below the regime where everyone gets infected anyway, 3.5 is
above measles-free respiratory plausibility, and 2.5 is kept so the anchor has
somewhere to land.

    python eda/p45_r0.py [--boot35 200]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import p35_seir as P35
from common import AGE_LABEL, AGES
from p27_survey import contact_matrix, ego_cubes, holiday_free_dows, load_survey
from p35_seir import (final_size, kendall_tau, marginal_ranking,
                      optimal_allocation, scale_to_r0, top_k)
from paths import FIG, ROOT

SURVEY_MONTHS = [202312, 202402]
SCOPES = [("national", False), ("seoul", True)]
SEED = 20260822
R0_POINT = (1.1, 1.2, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
R0_BOOT = (1.2, 1.5, 1.8, 2.5, 3.5)
R0_ANCHOR = 2.5
BUDGET = 0.10
NA = len(AGES)

out = {}
FAIL = []


def hdr(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def say(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot35", type=int, default=200)
    ap.add_argument("--budget", type=float, default=BUDGET)
    args = ap.parse_args()

    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p40 = json.load(open(f"{ROOT}/eda/results_p40.json"))
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]
    lbl = [AGE_LABEL[AGES[i]] for i in sq]

    out["declaration"] = dict(
        seed=SEED, n_boot=args.boot35, budget_frac=args.budget,
        r0_point=list(R0_POINT), r0_boot=list(R0_BOOT), r0_anchor=R0_ANCHOR,
        question="does the fourth pillar's verdict -- regret does not exceed "
                 "the resample-the-survey null -- survive a change of R0?",
        flip_rule="a cell flips iff the 2.5th percentile of regret rises above "
                  "the 97.5th percentile of the null, which is p40's own rule")

    # ---------------------------------------------------------------- matrices
    ego, d = load_survey()
    d = d[d.panel.isin(["W", "E"])]
    cubes = {}
    for ym in SURVEY_MONTHS:
        for scope, so in SCOPES:
            cubes[(ym, scope)] = ego_cubes(ego, d, so, ym=ym,
                                           dows=holiday_free_dows(ym))
    mats = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        for tag, lev in (("passive_dong", "dong|holidayfree"), ("passive_gu", "gu")):
            A = np.array(p26["matrices"][f"{ym}|WE|{lev}"]["A"])[np.ix_(sq, sq)]
            C, _ = P35.symmetrise(A / pops[ym][sq][:, None], pop)
            mats[(ym, tag)] = C

    # ------------------------------------------------- 45.1 the bit-exact anchor
    hdr("45.1 ANCHOR: p40's deterministic pillar-4 quantities at R0 = 2.5")
    say("  Deterministic means deterministic: SLSQP from fixed starts, a")
    say("  closed-form marginal. If any of these has moved, nothing below is")
    say("  about R0 -- it is about a changed pipeline, and the run stops.")
    n_anchor = 0
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        for scope, _ in SCOPES:
            cube, eb, nd = cubes[(ym, scope)]
            C0, _ = contact_matrix(cube, eb, nd)
            Cm, _ = P35.symmetrise(C0[np.ix_(sq, sq)], pop)
            base, _ = marginal_ranking(Cm, scale_to_r0(Cm, R0_ANCHOR), pop)
            stored = p40["pillar4"][f"{ym}|{scope}"]
            got_first = lbl[int(np.argmax(base))]
            assert got_first == stored["buy_first"], \
                f"{ym}|{scope} buy_first {got_first} != {stored['buy_first']}"
            n_anchor += 1
            for k, tag in enumerate(("passive_dong", "passive_gu")):
                sp_, _ = marginal_ranking(mats[(ym, tag)],
                                          scale_to_r0(mats[(ym, tag)], R0_ANCHOR),
                                          pop)
                t = kendall_tau(base, sp_)
                assert abs(t - stored["tau_between"][k]) < 1e-12, \
                    f"{ym}|{scope} vs {tag}: tau {t!r} != {stored['tau_between'][k]!r}"
                n_anchor += 1
            say(f"  {ym}|{scope:<8} buy first {got_first:<7} between-tau "
                + ", ".join(f"{t:+.4f}" for t in stored["tau_between"])
                + "   reproduced")
    say(f"  {n_anchor} deterministic quantities, all bit-exact against p40.")
    out["anchor"] = dict(quantities=n_anchor, r0=R0_ANCHOR, bit_exact=True)

    # ------------------------------------------- 45.2 the point curve over R0
    hdr("45.2 POINT regret over R0 (no bootstrap, so no Monte Carlo error)")
    say("  regret = (z_passive_plan - z_own_optimum) / (z_no_plan - z_own_optimum),")
    say("  i.e. the share of the achievable benefit thrown away by allocating")
    say("  from the passive matrix instead of this survey arm's own.")
    point = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        Cpd = mats[(ym, "passive_dong")]
        for scope, _ in SCOPES:
            cube, eb, nd = cubes[(ym, scope)]
            C0, _ = contact_matrix(cube, eb, nd)
            Cm, _ = P35.symmetrise(C0[np.ix_(sq, sq)], pop)
            rows = []
            for r0 in R0_POINT:
                q0 = scale_to_r0(Cm, r0)
                u_p, _, _ = optimal_allocation(Cpd, scale_to_r0(Cpd, r0), pop,
                                               budget_frac=args.budget)
                # p40's call, unchanged, so the two are comparable...
                _, v_bare, _ = optimal_allocation(Cm, q0, pop,
                                                  budget_frac=args.budget)
                # ...and the same call handed the passive plan as an extra start.
                # A real optimum cannot be beaten by a candidate it was shown, so
                # this cannot be worse, and where it IS better the bare call had
                # not found the optimum. That is the whole content of the
                # negative regrets the first run produced: p35's docstring already
                # says negative regret is impossible for a true optimum and is a
                # property of the search. `seeds` exists for exactly this.
                _, v0, _ = optimal_allocation(Cm, q0, pop,
                                              budget_frac=args.budget,
                                              seeds=[u_p])
                z0 = float((final_size(Cm, q0) * pop).sum() / pop.sum())
                zp = float((final_size(Cm, q0, u_p) * pop).sum() / pop.sum())
                rows.append(dict(r0=r0, attack_no_plan=z0,
                                 attack_own_plan=v0, attack_own_plan_bare=v_bare,
                                 attack_passive_plan=zp,
                                 optimiser_gain=float(v_bare - v0),
                                 regret=(zp - v0) / (z0 - v0) if z0 > v0 else np.nan,
                                 regret_bare=((zp - v_bare) / (z0 - v_bare)
                                              if z0 > v_bare else np.nan)))
            point[f"{ym}|{scope}"] = rows
            say(f"\n  {ym}|{scope}")
            say(f"    {'R0':>6}{'attack no plan':>16}{'own plan':>11}"
                f"{'passive plan':>14}{'regret':>9}{'bare':>9}{'opt gain':>10}")
            for r in rows:
                say(f"    {r['r0']:>6.1f}{r['attack_no_plan']:>16.4f}"
                    f"{r['attack_own_plan']:>11.4f}{r['attack_passive_plan']:>14.4f}"
                    f"{r['regret']:>9.1%}{r['regret_bare']:>9.1%}"
                    f"{r['optimiser_gain']:>10.2e}")
    out["point"] = point

    # The stop-loss, stated as a rule rather than as a glance at the table.
    NEG = -0.005
    bad = [(k, r["r0"], r["regret"]) for k, v in point.items() for r in v
           if r["regret"] < NEG]
    bad_bare = [(k, r["r0"], r["regret_bare"]) for k, v in point.items()
                for r in v if r["regret_bare"] < NEG]
    say(f"\n  negative regret is impossible for a true optimum, so it is the")
    say(f"  detector for the search failing. Bare call (p40's): "
        f"{len(bad_bare)} of {sum(len(v) for v in point.values())} cells below "
        f"{NEG:.1%}.")
    if bad_bare:
        say("    " + ", ".join(f"{k}@{r0}={g:.1%}" for k, r0, g in bad_bare))
    say(f"  Seeded call: {len(bad)} below {NEG:.1%}"
        + ("." if not bad else ": " + ", ".join(f"{k}@{r0}={g:.1%}"
                                                for k, r0, g in bad)))
    gain = max((r["optimiser_gain"] for v in point.values() for r in v),
               default=0.0)
    say(f"  largest attack-rate improvement the extra start bought: {gain:.2e}")
    out["optimiser"] = dict(negative_threshold=NEG, n_cells_total=
                            sum(len(v) for v in point.values()),
                            bare_failures=[[k, r0, g] for k, r0, g in bad_bare],
                            seeded_failures=[[k, r0, g] for k, r0, g in bad],
                            max_gain=float(gain))
    if bad:
        FAIL.append(f"{len(bad)} point cells still have negative regret after "
                    "seeding; the R0 range they sit in cannot carry a verdict")

    # The letter's claim that 2.5 is the friendly end, made into a number.
    hi = {k: max(r["regret"] for r in v) for k, v in point.items()}
    at25 = {k: [r["regret"] for r in v if r["r0"] == R0_ANCHOR][0]
            for k, v in point.items()}
    say("\n  point regret at R0 = 2.5 against its maximum over the grid:")
    for k in point:
        arg = [r["r0"] for r in point[k] if r["regret"] == hi[k]][0]
        say(f"    {k:<18} {at25[k]:>7.1%} at 2.5   vs {hi[k]:>7.1%} at R0 = {arg}"
            f"   ({hi[k] / at25[k]:.2f}x)" if at25[k] else f"    {k}: zero at 2.5")
    out["point_summary"] = dict(
        at_anchor=at25, max_over_grid=hi,
        argmax={k: [r["r0"] for r in point[k] if r["regret"] == hi[k]][0]
                for k in point})

    # --------------------------------------- 45.3 the verdict, with the interval
    hdr("45.3 THE VERDICT over R0 -- p40's own rule, five R0 values")
    say("  Each cell is bootstrapped over RESPONDENTS, and the null is a second")
    say("  independent draw of the same survey allocating against the first --")
    say("  the cost of simply re-running the survey. p40's rule: the passive")
    say("  plan's cost exceeds it iff regret p2.5 > null p97.5.")
    verdicts = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        Cpd = mats[(ym, "passive_dong")]
        for scope, _ in SCOPES:
            cube, eb, nd = cubes[(ym, scope)]
            rows = []
            for r0 in R0_BOOT:
                # One stream per (month, scope, R0): the cells are independent
                # questions and a shared stream would make the answer to one
                # depend on how many draws another happened to reject.
                rng = np.random.default_rng(
                    (SEED * 1000003 + ym * 97 + (scope == "seoul") * 7
                     + int(round(r0 * 10))) % (2 ** 63))
                u_p, _, _ = optimal_allocation(Cpd, scale_to_r0(Cpd, r0), pop,
                                               budget_frac=args.budget)
                regs, nulls = [], []
                for _ in range(args.boot35):
                    idx = rng.integers(0, len(eb), len(eb))
                    Cb, _ = contact_matrix(cube[idx], eb[idx], nd)
                    Cb, _ = P35.symmetrise(Cb[np.ix_(sq, sq)], pop)
                    qb = scale_to_r0(Cb, r0)
                    if not np.isfinite(qb):
                        continue
                    _, vb, _ = optimal_allocation(Cb, qb, pop,
                                                  budget_frac=args.budget,
                                                  seeds=[u_p])
                    z0 = float((final_size(Cb, qb) * pop).sum() / pop.sum())
                    zp = float((final_size(Cb, qb, u_p) * pop).sum() / pop.sum())
                    regs.append((zp - vb) / (z0 - vb) if z0 > vb else np.nan)
                    jdx = rng.integers(0, len(eb), len(eb))
                    Cj, _ = contact_matrix(cube[jdx], eb[jdx], nd)
                    Cj, _ = P35.symmetrise(Cj[np.ix_(sq, sq)], pop)
                    uj, _, _ = optimal_allocation(Cj, scale_to_r0(Cj, r0), pop,
                                                  budget_frac=args.budget,
                                                  seeds=[u_p])
                    zj = float((final_size(Cb, qb, uj) * pop).sum() / pop.sum())
                    nulls.append((zj - vb) / (z0 - vb) if z0 > vb else np.nan)
                regs = np.array([r for r in regs if np.isfinite(r)])
                nulls = np.array([r for r in nulls if np.isfinite(r)])
                ent = dict(
                    r0=r0, n_draws=int(len(regs)),
                    median=float(np.median(regs)),
                    ci=[float(np.percentile(regs, 2.5)),
                        float(np.percentile(regs, 97.5))],
                    null_median=float(np.median(nulls)),
                    null_ci=[float(np.percentile(nulls, 2.5)),
                             float(np.percentile(nulls, 97.5))],
                    exceeds_null=bool(np.percentile(regs, 2.5)
                                      > np.percentile(nulls, 97.5)))
                ent["margin"] = ent["ci"][0] - ent["null_ci"][1]
                rows.append(ent)
                say(f"  {ym}|{scope:<8} R0={r0:<4} regret {ent['median']:>6.1%} "
                    f"[{ent['ci'][0]:>5.1%}, {ent['ci'][1]:>5.1%}]   null "
                    f"{ent['null_median']:>6.1%} [{ent['null_ci'][0]:>5.1%}, "
                    f"{ent['null_ci'][1]:>5.1%}]   "
                    f"{'EXCEEDS' if ent['exceeds_null'] else 'does not exceed'}"
                    f"  (margin {ent['margin']:+.1%})")
            verdicts[f"{ym}|{scope}"] = rows
    out["verdict"] = verdicts

    # ------------------------------------------- 45.4 the Monte Carlo agreement
    hdr("45.4 the R0 = 2.5 row against p40's, within Monte Carlo error")
    say("  Not an anchor -- a different stream cannot reproduce a bootstrap")
    say("  median bit for bit, and this run also hands the optimiser an extra")
    say("  start that p40 did not (45.2). Both differences push the SAME way:")
    say("  a better-found own-optimum can only raise regret. Reported so the")
    say("  disagreement is visible rather than assumed away.")
    agree = {}
    for k, rows in verdicts.items():
        r25 = [r for r in rows if r["r0"] == R0_ANCHOR][0]
        st = p40["pillar4"][k]["regret"]
        agree[k] = dict(
            p45_median=r25["median"], p40_median=st["median"],
            rel_diff=abs(r25["median"] - st["median"]) / abs(st["median"]),
            p45_exceeds=r25["exceeds_null"], p40_exceeds=st["exceeds_null"])
        same = "same" if agree[k]["p45_exceeds"] == agree[k]["p40_exceeds"] \
            else "DIFFERENT"
        say(f"  {k:<18} p45 {r25['median']:>6.1%}  p40 {st['median']:>6.1%}   "
            f"rel {agree[k]['rel_diff']:>6.1%}   verdict {same}")
        if agree[k]["p45_exceeds"] != agree[k]["p40_exceeds"]:
            FAIL.append(f"{k}: the R0=2.5 verdict disagrees with p40")
    out["mc_agreement"] = agree

    # ----------------------------------------------------------- 45.5 the answer
    hdr("45.5 SO DOES IT FLIP?")
    n_cells = len(verdicts)
    flips = {k: [r["r0"] for r in rows if r["exceeds_null"]]
             for k, rows in verdicts.items()}
    any_flip = {k: v for k, v in flips.items() if v}
    worst = min(((r["margin"], k, r["r0"]) for k, rows in verdicts.items()
                 for r in rows), key=lambda t: abs(t[0]))
    say(f"  cells: {n_cells}, R0 values per cell: {len(R0_BOOT)}, "
        f"so {n_cells * len(R0_BOOT)} verdicts.")
    if any_flip:
        say("  IT FLIPS. Cells and the R0 values where the passive plan's cost "
            "exceeds the null:")
        for k, v in any_flip.items():
            say(f"    {k}: R0 = {', '.join(str(x) for x in v)}")
        say("  So pillar 4's verdict is R0-dependent, and the letter cannot")
        say("  report 'no cell exceeds the null' without naming R0 = 2.5.")
    else:
        say("  IT DOES NOT FLIP. No cell at any R0 on this grid has a regret")
        say("  interval clearing the resample-the-survey null, so the verdict")
        say("  p40 reported at R0 = 2.5 is not an artefact of that choice.")
    say(f"  closest call: {worst[1]} at R0 = {worst[2]}, margin {worst[0]:+.2%} "
        "(negative = the intervals still overlap)")
    out["answer"] = dict(flips=any_flip, any_flip=bool(any_flip),
                         n_verdicts=n_cells * len(R0_BOOT),
                         closest=dict(cell=worst[1], r0=worst[2],
                                      margin=worst[0]))

    # -------------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for k, rows in point.items():
        ax[0].plot([r["r0"] for r in rows], [r["regret"] for r in rows],
                   "o-", ms=3.5, label=k)
    ax[0].axvline(R0_ANCHOR, color="0.6", ls=":", lw=1.2)
    ax[0].annotate("the only R0 p40 ran", (R0_ANCHOR, ax[0].get_ylim()[1]),
                   fontsize=7.5, ha="center", va="top", color="0.35")
    ax[0].set_xlabel("$R_0$")
    ax[0].set_ylabel("point regret of the passive plan")
    ax[0].set_title("45.2  regret over $R_0$ (deterministic)")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=.3)

    off = {k: i for i, k in enumerate(verdicts)}
    for k, rows in verdicts.items():
        x = np.array([r["r0"] for r in rows]) + (off[k] - 1.5) * 0.06
        med = [r["median"] for r in rows]
        lo = [r["median"] - r["ci"][0] for r in rows]
        hi = [r["ci"][1] - r["median"] for r in rows]
        ax[1].errorbar(x, med, yerr=[lo, hi], fmt="o", ms=3.5, capsize=2,
                       lw=1.0, label=k)
        ax[1].plot(x, [r["null_ci"][1] for r in rows], "_", ms=9, color="crimson")
    ax[1].set_xlabel("$R_0$")
    ax[1].set_ylabel("regret (95% interval)")
    ax[1].set_title("45.3  regret vs the resample-the-survey null (red bars)")
    ax[1].legend(fontsize=7.5)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p45_r0.png", dpi=150)
    say(f"\n  figure -> {FIG}/p45_r0.png")

    out["fail"] = FAIL
    with open(f"{ROOT}/eda/results_p45.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"  results -> {ROOT}/eda/results_p45.json")
    if FAIL:
        say("\nFAILURES:")
        for f in FAIL:
            say(f"  {f}")


if __name__ == "__main__":
    main()
