#!/usr/bin/env python
"""Phase 49 — the cell-level estimator under both pairing conventions.

WHY THIS EXISTS. `p44_beta.py` found that the pairing convention matters: at
VENUE scale, where a group holds a mean of 4.22 people, drawing pairs with
replacement puts a sixth of the mass on the diagonal and reads r = 0.43615,
while drawing without replacement reads r = 0.24872. That is a 43% relative
difference and it decided which number the survey arm reports.

The passive estimator (`p26_matrix.py:141`) uses the WITH-replacement form at
every level, and has never been run the other way. The size argument says the
correction is negligible there -- a cell holds hundreds of people, not four --
but "negligible" has been an argument rather than a number, and a reader asked
to accept an estimator should not have to derive its bias themselves.

So this phase runs both conventions at cell scale and reports one number. It
adds no machinery: it imports `collapse` from `p44_beta` unchanged, so the two
scales are the SAME operator at different group sizes rather than two
implementations that happen to agree.

WHAT IS DECLARED BEFORE THE RUN, and why each choice is fixed in advance.

  1. THE CELL. 202312 | WE | dong | holiday-free, because that is the published
     primary cell -- the one `p32`/`p33`/`p44` quote as 0.02115385938228908.
     Answering at any other cell would be answering about a number the paper
     does not report.

  2. THE PAIRING SCALE, which is the one real trap here. `p26_matrix.py:129`
     divides each cell's headcount by that weekday's number of occurrences in
     the month BEFORE pairing, so X is an average day's arrivals -- a float
     expectation, not an integer headcount. That matters because the two
     conventions are not the same kind of function:

         with replacement     sum_g w_g n_a n_a' / n_.            degree 1 in n
         without replacement  sum_g w_g (n_a n_a' - d_aa' n_a)/(n_.-1)   NOT

     `p44_beta.py:113-115` justifies rescaling groups by saying the operator is
     homogeneous of degree one in n. That is true of the WITH-replacement branch
     only. For the without-replacement branch the scale at which the correction
     is applied changes the answer, so it has to be chosen and defended rather
     than inherited.

     DECLARED PRIMARY: the average-day scale. The estimator's whole construction
     says the unit of co-presence is one (dong, hour, weekday) cell on one day --
     that is what `pop_day` is for, and `common.py` records that `이동인구(합)`
     is a monthly SUM which nothing may be compared across without dividing by
     the occurrence count first. Two people who were in the same dong in the same
     hour on different days did not meet.

     DECLARED BOUND: the monthly-sum scale, where n is four to five times larger
     and the correction is correspondingly smaller. Reported so the reader can
     see the answer is not an artefact of the scale choice: the primary is the
     conservative end.

     NOT DONE: integerising X. Rounding would change the with-replacement value
     and break anchor 49.0a, which is the one thing tying this to p26.

  3. THE GOVERNING SIZE STATISTIC, declared in advance so the answer is not
     framed afterwards by whichever denominator flatters it. The correction
     enters ARRIVAL-WEIGHTED -- a cell contributes in proportion to the people
     in it -- so the n that governs is `results_p44.json cell.n_weighted_mean`
     = 4052.1, not the cell median of 533.0. All four size statistics are
     reported. The advisor's "median 533, so negligible" reaches the right
     conclusion by roughly a factor of 7.6 too weak an argument.

  4. THE LADDER, not a single number. dong / gu / city, because how the
     correction decays with group size IS the mechanism sentence: contact is
     structured by venues, venues are smaller than any administrative unit, and
     the correction is large exactly where groups are small.

  5. STOP-LOSS. If the with-replacement rebuild does not reproduce p26's
     published matrix bit for bit, this script aborts and reports nothing.

EXPECTED MAGNITUDE, so a wrong answer is recognisable rather than merely
surprising. The correction is order 1/(n-1) in the governing n. At venue scale
n = 4.22 gives 1/(n-1) = 0.31 and the observed relative change is 0.43. At
n = 4052 the same form predicts about 0.025%. Anything above roughly 1% here
means the scale convention was applied wrongly, not that the estimator is
biased.

    python eda/p49_dongwor.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import AGES
from p26_matrix import cells, holiday_free_dows, matrices
from p27_survey import assortativity
from p44_beta import collapse
from paths import ROOT
import calendar_kr as K

YM = 202312                     # the declared primary month
YM2 = 202402                    # the other survey month, reported beside it
PANEL = "WE"
IMP_PUB = 1.5                   # p26's published imputation for a masked row
NA = len(AGES)
SQ = [i for i, a in enumerate(AGES) if a < 80]     # p44's 0-79 block

# Read back from the results files, not retyped, and asserted in 49.0.
R_VENUE_WR = 0.43614678635940224
R_VENUE_WOR = 0.24871774193970334
CELL_N_WEIGHTED = 4052.1185601400757
CELL_MEDIAN = 533.0475004911423

out = {}


def say(s=""):
    print(s, flush=True)


def r_of(A):
    """Newman assortativity on the 0-79 block, p44's published convention."""
    return assortativity(A[np.ix_(SQ, SQ)])


def wide_counts(df, ym, panel, level, dows):
    """The (group, age) count table `matrices()` pairs, stopped one step early.

    This is `p26_matrix.matrices` up to but not including the pairing line, so
    that the same X can be handed to both conventions. Keeping it here rather
    than editing p26 is what leaves results_p26.json untouched.
    """
    nd = {d: K.cell_exposure(ym, d)["n_days"] for d in dows}
    d = df if panel == "WE" else df[df.dest_attr == panel]
    d = d[d.dest_attr != "H"]
    d = d[d.dow_n.isin(dows)]
    d = d.assign(n=(d.v_obs + IMP_PUB * d.n_masked) / d.dow_n.map(nd))
    loc = {"dong": d.d_dong, "gu": d.d_dong // 1000,
           "city": pd.Series(0, index=d.index)}[level]
    g = (d.assign(loc=loc).groupby(["loc", "arr_hour", "dow_n", "age"],
                                   as_index=False)["n"].sum())
    wide = g.pivot_table(index=["loc", "arr_hour", "dow_n"], columns="age",
                         values="n", fill_value=0.0).reindex(columns=AGES,
                                                             fill_value=0.0)
    X = wide.to_numpy(float)
    # Occurrences per (loc, hour, dow) row, so the monthly-sum scale can be built
    # from the same table rather than from a second pass over the data.
    occ = np.array([nd[t[2]] for t in wide.index], float)
    return X[X.sum(1) > 0], occ[X.sum(1) > 0]


def brute_force(X, w):
    """Both conventions written out as pair enumeration, for 49.0c.

    O(bands^2) per group and deliberately literal: it counts ordered pairs of
    people drawn from a group, once allowing the same person twice and once not.
    It shares no line with `collapse`, so agreement is evidence rather than
    tautology.
    """
    nb = X.shape[1]
    Awr = np.zeros((nb, nb))
    Awor = np.zeros((nb, nb))
    for g in range(X.shape[0]):
        n = X[g]
        tot = n.sum()
        if tot <= 0:
            continue
        for i in range(nb):
            for j in range(nb):
                Awr[i, j] += w[g] * n[i] * n[j] / tot
        if tot > 1.0 + 1e-9:
            for i in range(nb):
                for j in range(nb):
                    pairs = n[i] * n[j] - (n[i] if i == j else 0.0)
                    Awor[i, j] += w[g] * pairs / (tot - 1.0)
    return Awr, Awor


def main():
    # ====================================================================== 49.0
    say("=== 49.0 anchors ===")
    anchors = {}

    df = cells(YM)
    hf = holiday_free_dows(YM)
    say(f"  {YM}: {len(df):,} cells, holiday-free weekdays {hf}")

    X, occ = wide_counts(df, YM, PANEL, "dong", hf)
    w_day = np.full(len(X), 1.0 / len(hf))

    # --- 49.0a X IS p26's X: p26's own expression, bit for bit ------------------
    # Two anchors, not one, and the split is the point. This first one rebuilds the
    # pairing line exactly as p26_matrix.py:141 writes it. It must be bit-exact,
    # because any difference here means `wide_counts` did not reproduce p26's cell
    # table and the whole phase is about a different estimator.
    A_pub = np.array(json.load(open(f"{ROOT}/eda/results_p26.json"))
                     ["matrices"][f"{YM}|{PANEL}|dong|holidayfree"]["A"], float)
    tot0 = X.sum(1)
    A_p26 = (X / tot0[:, None]).T @ X / len(hf)
    d_p26 = np.abs(A_p26 - A_pub).max()
    say(f"  49.0a  p26's own expression on this X vs published A: "
        f"max abs diff {d_p26:.2e}")
    assert d_p26 == 0.0, (
        f"wide_counts did not reproduce p26's cell table ({d_p26:.3e}); "
        f"everything below would be a statement about a different estimator")
    anchors["x_reproduces_p26"] = dict(max_abs_diff=float(d_p26), bit_exact=True)

    # --- 49.0a2 collapse is that same operator, to machine precision ------------
    # `collapse` computes the same sum in a different order -- it scales rows first
    # and contracts once, where p26 divides then contracts -- and float addition is
    # not associative, so the two agree to about 1e-14 rather than bit for bit.
    # That is the same effect p36 pins threads=1 to avoid, and it is reported rather
    # than hidden: bit-exactness is a claim about operation order, and this pair
    # does not have the same order. What matters is that the OPERATOR is p44's, so
    # the venue scale and the cell scale are one estimator at two group sizes.
    A_wr = collapse(X, w_day, replacement=True)
    rel = np.abs(A_wr - A_pub).max() / np.abs(A_pub).max()
    say(f"  49.0a2 collapse(replacement=True) vs published A: "
        f"max rel diff {rel:.2e} (operation order only, not a different estimator)")
    assert rel < 1e-12, (
        f"collapse's with-replacement branch is not p26's estimator ({rel:.3e})")
    anchors["collapse_matches_p26"] = dict(max_rel_diff=float(rel),
                                           bit_exact=bool(rel == 0.0),
                                           why_not_bit_exact="summation order; "
                                           "float addition is not associative")

    # --- 49.0b the one-group closed forms --------------------------------------
    # In a single group of n people all in one band, with replacement every pair is
    # a same-band pair, so r = 0; without replacement exactly n of the n^2 ordered
    # pairs are a person with themselves and are removed, so the null is -1/(n-1).
    # These are the operator's signature and they are checked at CELL scale here,
    # where n is large, precisely because p44 only ever checked them at venue scale.
    # The composition is the real age margin of the cell table, not a toy: the
    # closed form holds for ANY composition spread over more than one band, so
    # using the data's own margin tests it where it will actually be applied. A
    # single-band group is the one case it does NOT cover -- Newman's denominator
    # is 1 - sum(a^2), which is 0 when all the mass is in one band -- and that
    # degenerate case is why this check is written with a composition rather than
    # with a lump.
    p_mix = X.sum(0) / X.sum()
    assert (p_mix > 0).sum() > 1, "need a composition, not a single band"
    closed = []
    for n in (533.0, 1080.0, 4052.0, 17856.0):
        one = (p_mix * n)[None, :]
        got_wr = assortativity(collapse(one, np.ones(1), replacement=True))
        got_wor = assortativity(collapse(one, np.ones(1)))
        want_wor = -1.0 / (n - 1.0)
        closed.append(dict(n=n, wr=float(got_wr), wor=float(got_wor),
                           wor_closed_form=float(want_wor),
                           wor_abs_err=float(abs(got_wor - want_wor))))
        say(f"  49.0b  n={n:>8,.0f}  WR r={got_wr:+.3e} (closed form 0)   "
            f"WOR r={got_wor:+.6e} vs -1/(n-1)={want_wor:+.6e}")
    assert all(abs(c["wr"]) < 1e-12 for c in closed), "WR one-group null is not 0"
    assert all(c["wor_abs_err"] < 1e-9 * abs(c["wor_closed_form"])
               for c in closed), "WOR one-group null is not -1/(n-1)"
    anchors["one_group_closed_forms"] = closed

    # --- 49.0c collapse against an independent pair enumeration ----------------
    rng = np.random.default_rng(20260824)
    Xs = rng.integers(0, 40, size=(25, NA)).astype(float)
    ws = rng.random(25)
    Bwr, Bwor = brute_force(Xs, ws)
    Cwr = collapse(Xs, ws, replacement=True)
    Cwor = collapse(Xs, ws)
    e_wr = np.abs(Bwr - Cwr).max() / np.abs(Bwr).max()
    e_wor = np.abs(Bwor - Cwor).max() / np.abs(Bwor).max()
    say(f"  49.0c  collapse vs literal pair enumeration: "
        f"WR {e_wr:.2e}, WOR {e_wor:.2e}")
    assert e_wr < 1e-12 and e_wor < 1e-12, "collapse disagrees with pair enumeration"
    anchors["pair_enumeration"] = dict(wr_rel=float(e_wr), wor_rel=float(e_wor))

    # --- 49.0d the venue-scale numbers this phase is the counterpart of ---------
    p44 = json.load(open(f"{ROOT}/eda/results_p44.json"))
    for k, want in (("r_venue_wr", R_VENUE_WR), ("r_venue_wor", R_VENUE_WOR)):
        got = p44["pairing"][k]
        assert got == want, f"{k}: results_p44 has {got}, this script assumed {want}"
    say(f"  49.0d  venue scale, read back from results_p44: "
        f"WR {R_VENUE_WR:.5f} -> WOR {R_VENUE_WOR:.5f} "
        f"({100 * (R_VENUE_WOR / R_VENUE_WR - 1):+.1f}%)")
    anchors["venue_scale_readback"] = dict(wr=R_VENUE_WR, wor=R_VENUE_WOR)

    out["anchors"] = anchors

    # ====================================================================== 49.1
    say("\n=== 49.1 both conventions at cell scale, average-day (declared primary) ===")
    say(f"  {'month':>7} {'level':>5} | {'n_groups':>9} {'n_med':>9} {'n_wmean':>10} "
        f"| {'r WR':>10} {'r WOR':>10} {'delta':>11} {'rel':>9}")

    ladder = []
    for ym in (YM, YM2):
        d_ = cells(ym)
        hf_ = holiday_free_dows(ym)
        for level in ("dong", "gu", "city"):
            Xl, occl = wide_counts(d_, ym, PANEL, level, hf_)
            wl = np.full(len(Xl), 1.0 / len(hf_))
            tot = Xl.sum(1)
            a_wr = collapse(Xl, wl, replacement=True)
            a_wor = collapse(Xl, wl)
            r_wr, r_wor = assortativity(a_wr), assortativity(a_wor)
            rq_wr, rq_wor = r_of(a_wr), r_of(a_wor)
            n_w = float((tot * tot).sum() / tot.sum())
            row = dict(ym=ym, level=level, n_groups=int(len(Xl)),
                       n_median=float(np.median(tot)), n_mean=float(tot.mean()),
                       n_weighted_mean=n_w,
                       r_wr=float(r_wr), r_wor=float(r_wor),
                       delta=float(r_wor - r_wr),
                       rel_change=float(r_wor / r_wr - 1.0),
                       r_wr_sub80=float(rq_wr), r_wor_sub80=float(rq_wor),
                       rel_change_sub80=float(rq_wor / rq_wr - 1.0))
            ladder.append(row)
            say(f"  {ym:>7} {level:>5} | {len(Xl):>9,} {np.median(tot):>9,.0f} "
                f"{n_w:>10,.0f} | {r_wr:>10.6f} {r_wor:>10.6f} "
                f"{r_wor - r_wr:>+11.2e} {100 * (r_wor / r_wr - 1):>+8.4f}%")
    out["ladder"] = ladder

    prim = [r for r in ladder if r["ym"] == YM and r["level"] == "dong"][0]
    out["primary"] = prim

    # ====================================================================== 49.2
    say("\n=== 49.2 does the answer depend on the scale the correction is applied at? ===")
    say("  primary is the average-day scale; the monthly-sum scale is the bound.")
    scale = []
    for ym in (YM, YM2):
        d_ = cells(ym)
        hf_ = holiday_free_dows(ym)
        Xl, occl = wide_counts(d_, ym, PANEL, "dong", hf_)
        # Monthly-sum scale: undo the occurrence-count division, pair, then put the
        # scale back. Only the WOR branch moves -- WR is homogeneous, so its
        # invariance here is itself a check on the arithmetic.
        Xm = Xl * occl[:, None]
        wm = np.full(len(Xm), 1.0 / len(hf_)) / occl
        a_wr_m = collapse(Xm, wm, replacement=True)
        a_wor_m = collapse(Xm, wm)
        r_wr_m, r_wor_m = assortativity(a_wr_m), assortativity(a_wor_m)
        base = [r for r in ladder if r["ym"] == ym and r["level"] == "dong"][0]
        wr_moved = abs(r_wr_m - base["r_wr"])
        scale.append(dict(ym=ym, r_wr_monthly=float(r_wr_m),
                          r_wor_monthly=float(r_wor_m),
                          rel_change_monthly=float(r_wor_m / r_wr_m - 1.0),
                          rel_change_daily=base["rel_change"],
                          wr_invariance=float(wr_moved)))
        say(f"  {ym}: average-day {100 * base['rel_change']:+.4f}%   "
            f"monthly-sum {100 * (r_wor_m / r_wr_m - 1):+.4f}%   "
            f"(WR moved {wr_moved:.1e}, must be ~0)")
        assert wr_moved < 1e-12, \
            "the with-replacement branch moved under rescaling; it is degree-1 " \
            "homogeneous and must not"
    out["scale_sensitivity"] = scale

    # ====================================================================== 49.3
    say("\n=== 49.3 how big is a cell, by the four statistics that have been quoted ===")
    p44cell = p44["cell"]
    sizes = dict(cell_median=p44cell["median"], cell_mean=p44cell["mean"],
                 cell_n_weighted_mean=p44cell["n_weighted_mean"],
                 dong_day_median=p44cell["dong_day_median"],
                 venue_mean=p44cell.get("observed_venue_mean"))
    for k, v in sizes.items():
        if v is not None:
            say(f"  {k:>22}: {v:>12,.2f}   1/(n-1) = {1.0 / (v - 1):.3e}"
                if v > 1 else f"  {k:>22}: {v:>12,.2f}")
    assert p44cell["n_weighted_mean"] == CELL_N_WEIGHTED
    assert p44cell["median"] == CELL_MEDIAN
    out["sizes"] = sizes

    # ====================================================================== 49.4
    # THE DECLARED EXPECTATION FAILED, AND THAT IS THE FINDING.
    #
    # The declaration predicted a correction of order 1/(n-1) at the arrival-
    # weighted mean, about 0.025%, and said anything above 1% would mean the scale
    # convention had been applied wrongly. The measured value is -4.29%. The
    # convention is not wrong -- 49.2 shows the answer tracks 1/n exactly as it
    # should when the scale is changed, and 49.0a-c all hold -- so the prediction
    # was. It was wrong twice over, and both errors are worth writing down because
    # the second one is the advisor's too.
    #
    #   (i) MISSING THE 1/r AMPLIFICATION. The correction is order 1/n relative to
    #       the matrix's TOTAL mass, but r is a diagonal EXCESS of only 0.0212. A
    #       perturbation of the total is a perturbation of 47x its size relative to
    #       the excess. Every "the cells are big so it cannot matter" argument in
    #       this project has quietly compared the correction to the wrong baseline.
    #
    #   (ii) THE WRONG SIZE STATISTIC. Summing the removed diagonal gives
    #        sum_g w_g n_g/(n_g - 1), and n_g/(n_g - 1) -> 1, so the correction is
    #        approximately (number of groups) x w. Divided by the total mass that is
    #        1/mean(n) -- the ARITHMETIC MEAN, which weights every group equally
    #        because every group contributes one removed self-pair per person. Not
    #        the arrival-weighted mean (which is what this file declared, and is too
    #        big by 3.8x), and not the median (which the advisor's letter used, and
    #        is too small by 2.1x in the other direction).
    #
    # So the declared governing statistic is superseded by measurement. The
    # declaration block is left exactly as it was written -- choosing the statistic
    # after seeing which one fits is the number-shopping the block exists to
    # prevent -- and the correction is recorded here instead.
    say("\n=== 49.4 the declared expectation, against what was measured ===")
    r_wr_p = prim["r_wr"]
    implied_n = 1.0 / (-prim["rel_change"] * r_wr_p) + 1.0
    cands = dict(median=p44cell["median"], mean=p44cell["mean"],
                 arrival_weighted_mean=p44cell["n_weighted_mean"])
    say(f"  declared governing statistic : arrival-weighted mean "
        f"({CELL_N_WEIGHTED:,.1f})")
    say(f"  declared predicted magnitude : "
        f"{-100 * (1 / (CELL_N_WEIGHTED - 1)):+.4f}%  (no 1/r amplification)")
    say(f"  measured                     : {100 * prim['rel_change']:+.4f}%")
    say(f"  implied effective n          : {implied_n:,.1f}")
    for nm, v in cands.items():
        pred = -(1.0 / (v - 1.0)) / r_wr_p
        say(f"    vs {nm:>22} = {v:>10,.1f} -> predicts "
            f"{100 * pred:+.4f}%   (ratio {pred / prim['rel_change']:.2f})")
    out["declared_vs_measured"] = dict(
        declared_governing="n_weighted_mean",
        declared_prediction=float(-(1.0 / (CELL_N_WEIGHTED - 1.0))),
        declared_stop_threshold=0.01,
        measured=prim["rel_change"],
        implied_effective_n=float(implied_n),
        governing_statistic_measured="mean",
        candidates={k: dict(n=float(v),
                            predicted=float(-(1.0 / (v - 1.0)) / r_wr_p))
                    for k, v in cands.items()},
        verdict="the declared statistic was wrong; the arithmetic mean governs, "
                "because each group loses one self-pair per person. The 1/r "
                "amplification was omitted entirely.",
    )

    # ====================================================================== 49.5
    # A CONSEQUENCE NOBODY ASKED FOR, which is why it is reported.
    # The published comparison puts a WITH-replacement passive reading beside a
    # WITHOUT-replacement survey reading: 0.0212 (p26, WR) against 0.2487 (p44,
    # WOR). p44 changed the survey side's convention and did not change the passive
    # side's, because the passive side was assumed unaffected. It is affected by
    # 4.29%. This does not overturn anything -- the gap is 10x, not 1.04x -- but the
    # ratio that gets quoted should be built from one convention, not two.
    say("\n=== 49.5 the published comparison currently mixes the two conventions ===")
    mixed = R_VENUE_WOR / prim["r_wr"]
    consistent = R_VENUE_WOR / prim["r_wor"]
    say(f"  as published  survey WOR {R_VENUE_WOR:.5f} / passive WR  "
        f"{prim['r_wr']:.6f} = {mixed:.3f}x")
    say(f"  one convention survey WOR {R_VENUE_WOR:.5f} / passive WOR "
        f"{prim['r_wor']:.6f} = {consistent:.3f}x")
    say(f"  the gap moves by {100 * (consistent / mixed - 1):+.2f}%, "
        f"which is small, and in the direction that makes the gap LARGER.")
    out["convention_consistency"] = dict(
        survey_wor=R_VENUE_WOR,
        passive_wr=prim["r_wr"], passive_wor=prim["r_wor"],
        ratio_as_published=float(mixed), ratio_one_convention=float(consistent),
        rel_move=float(consistent / mixed - 1.0),
    )

    # The one-line answer, assembled rather than retyped.
    out["answer"] = dict(
        cell=f"{YM}|{PANEL}|dong|holidayfree",
        r_with_replacement=prim["r_wr"],
        r_without_replacement=prim["r_wor"],
        absolute_difference=prim["delta"],
        relative_difference=prim["rel_change"],
        governing_n=CELL_N_WEIGHTED,
        predicted_order=1.0 / (CELL_N_WEIGHTED - 1.0),
        venue_relative_difference=R_VENUE_WOR / R_VENUE_WR - 1.0,
        ratio_venue_to_cell=abs((R_VENUE_WOR / R_VENUE_WR - 1.0)
                                / prim["rel_change"]) if prim["rel_change"] else None,
    )

    say("\n=== the number ===")
    say(f"  at {YM} WE dong holiday-free, the published cell:")
    say(f"    with replacement     r = {prim['r_wr']:.8f}   (this is p26's published "
        f"{prim['r_wr']:.8f})")
    say(f"    without replacement  r = {prim['r_wor']:.8f}")
    say(f"    difference           {prim['delta']:+.2e}  "
        f"({100 * prim['rel_change']:+.4f}%)")
    say(f"  the same correction at venue scale is "
        f"{100 * (R_VENUE_WOR / R_VENUE_WR - 1):+.1f}%, "
        f"{out['answer']['ratio_venue_to_cell']:,.0f}x larger.")
    say("  that ratio is the mechanism: the correction is large exactly where the")
    say("  group is small, and a venue is smaller than any administrative unit.")

    out["declaration"] = dict(
        primary_cell=f"{YM}|{PANEL}|dong|holidayfree",
        pairing_scale_primary="average-day",
        pairing_scale_bound="monthly-sum",
        governing_size_statistic="n_weighted_mean",
        integerised=False,
        levels=["dong", "gu", "city"],
        stop_loss="abort unless collapse(replacement=True) reproduces "
                  "results_p26.json's published A bit for bit",
    )

    with open(f"{ROOT}/eda/results_p49.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p49.json")


if __name__ == "__main__":
    main()
