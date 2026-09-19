#!/usr/bin/env python
"""Phase 64 -- the semester claim as a regression, and what the rotation test can resolve.

WHY THIS EXISTS. The advisor's 2026-08-31 letter makes two methodological
points about section 3.1, and both of them are about resolution rather than
about the direction of the finding.

  2.1  "You have a 79-month series and you are testing it with a sign test on
       seven points." The sign test cannot go below 2^-7 = 0.0078 no matter
       what the data say, and the within-year hypergeometric treats contiguous
       semesters as exchangeable. Run MI_t ~ term_t + year FE + linear trend
       with Newey-West standard errors instead: the term coefficient is one
       number, the year effects absorb the non-stationarity the section spends
       a paragraph on, and HAC absorbs the block autocorrelation.

  2.2  Figure 2 already shows that a shift by a multiple of six lands the school
       calendar nearly back on itself -- {3,4,5,6,9,10,11} shifted by six is
       {9,10,11,12,3,4,5}, six of seven months overlapping -- and the three
       matching shifts 12, 24 and 36 are all multiples of six. So of the 79
       alignments only about six are genuinely distinct, the resolution floor
       is 1/6 = 0.17, and the observed 0.0506 sits below the test's own
       resolution. The main text instead quotes 1/79 = 0.0127.

WHAT THIS FILE DOES. Two things, in one phase, because both replace arithmetic
inside the same section and both read the same 79-month series.

  Part A (64.1-64.6)  fits the regression the letter asks for, with Newey-West
  standard errors at a lag declared before the run, on two pre-declared primary
  series -- the mutual-information series the rotation test attaches to, and
  the assortativity series the sign test was actually computed on. Five further
  specifications and four further series are declared in advance and reported
  whichever way they land.

  Part B (64.7-64.10) does not accept the advisor's six. It derives the number
  of genuinely distinct alignments from the calendar itself, checks each step
  of his argument computationally, and reports the honest resolution floor.

WHAT IS NOT REBUILT. Nothing. The 79 readings and their labels come from
results_p42.json, the floor from results_p51.json, the anchors from
results_p41/p54/p58/p63.json. No parquet, no matrices, no drive. The only
output is results_p64.json. results_p41/p42/p51/p54/p58/p63.json are read-only
here and not one byte of any of them is written.

FULLY DETERMINISTIC. There is no rng anywhere in this file: the 79 rotations
are an exhaustive enumeration, the regressions are closed-form least squares,
and the nulls that are re-derived are exact hypergeometric convolutions
imported from p42. There is no seed to report.

    .venv/bin/python eda/p64_semreg.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats as sps

from p41_semester import TERM, VAC              # (3,4,5,6,9,10,11) / (1,2,7,8)
from p42_monthscope import label, within_year_null
from paths import ROOT

# ==========================================================================
# THE DECLARATION.  Written before any line of analysis code below it and not
# edited afterwards.  CLAUDE.md: "Declare the primary cell before the run, in
# the script's `declaration` block.  Choosing after the fact is number-
# shopping."  Two things in particular are declared here because they decide
# an answer before it is seen: the HAC lag length, and the rule by which two
# circular shifts count as the same alignment.
# ==========================================================================
DECLARATION = """\
=== PART A -- the regression (advisor 2.1) ===

primary series (two, both declared, both reported, no selection between them)
  P1  mi_bits_hf   -- results_p42.json rows[*].mi_bits_hf, the holiday-free
      dong-level mutual information. This is the series Figure 2(a) draws, the
      series the 17-of-79 floor count is taken on, and the series the rotation
      test rotates. It is what the letter calls MI_t.
  P2  assortativity_all -- results_p42.json rows[*].assortativity_all. This is
      the series p41's per-year contrasts, and therefore the sign test the
      letter asks to replace, were computed on. A regression that replaces a
      sign test has to be run on the sign test's own statistic, or it changes
      the question while claiming to answer it.
  Neither is subordinate to the other. Both are reported in full.

secondary series (reported, never decisive)
  mi_bits_all, assortativity_hf, nmi_hf, half_l1_hf.

primary specification  S1
  y_t = a + b*term_t + d*dec_t + sum_y gamma_y*1{year(t)=y} + delta*tc_t + e_t
  term_t = 1 for months 3,4,5,6,9,10,11; dec_t = 1 for December; the reference
  category is a vacation month (1,2,7,8) in 2020. December gets its own
  indicator because p41's partition puts December in NEITHER set; pooling it
  into the baseline would make b a term-versus-(vacation+December) contrast,
  which is not the contrast the section states. tc_t is the centred position
  index (i - 39), i = 0..78; centring moves the intercept and cannot move b.
  Estimation: ordinary least squares. k = 10 parameters, T = 79, df = 69.

secondary specifications, all declared now and all reported
  S2  December rows dropped entirely (T = 73).
  S3  December pooled into the baseline, no dec_t (b is term vs everything).
  S4  S1 plus term_t * 1{year = 2020}. The mechanism the section argues for
      predicts this interaction is NEGATIVE and close to -b, because Seoul's
      schools did not open in March 2020.
  S5  S1 without the linear trend (year FE only).
  S6  S1 without the year fixed effects (trend only).

standard errors  Newey-West HAC, Bartlett kernel, lag
  L = floor(4 * (T/100)^(2/9)) = floor(4 * 0.79^(2/9)) = floor(3.7958) = 3
  This is the Newey-West (1994) / Stock-Watson rule of thumb evaluated at
  T = 79. THE LAG IS FIXED HERE, BEFORE THE RUN. The whole L = 0..12 curve is
  reported alongside it. L = 12 is the top of the reported band because a
  monthly series carrying a school calendar can leave a twelve-month component
  in the residual, and 12 is the most conservative lag in that neighbourhood;
  it is reported as sensitivity, not as the primary.
  No small-sample correction is applied to the meat matrix in the primary
  number; the T/(T-k) variant is reported beside it.

inference  two-sided t with df = T - k. The normal-approximation p is reported
  beside it and is never the quoted one.

predicted sign  POSITIVE, on both primary series. Term months are the months in
  which the school-age contribution to the matrix is present at all.
predicted magnitude  for P2, of the order of the 0.0040 mean within-year
  contrast p41 measured. No magnitude is predicted for P1: nothing published
  fixes the scale of a term effect in bits.

WHAT WOULD FALSIFY THE SECTION 3.1 CLAIM, stated before the run
  b <= 0 on either primary series; or |t| < 2 at the declared lag L = 3; or a
  sign change anywhere in the L = 0..12 band. Any one of those and the
  regression does not corroborate the sign test, it overturns it, and that is
  what gets written.

WHAT THE REGRESSION CANNOT DO  it does not separate "school term" from anything
  else that is seasonal on the same 3-6 / 9-11 pattern, and it does not by
  itself identify 2020 as the exceptional year. S4 does that, as an
  interaction, not as a p-value.

R-ROT  the same regression refitted under all 79 circular rotations of the
  calendar label vector, with the year effects and the trend held fixed
  (they are properties of time, not of the labels). Reported: the rank of the
  observed t(term) among the 79. This is the rotation test with a statistic
  that is not censored at 17, and its resolution is subject to exactly the
  Part B analysis below.

=== PART B -- the rotation test's effective degrees of freedom (advisor 2.2) ===

THE CRITERION, declared before anything is counted
  Two circular shifts are THE SAME ALIGNMENT when they impose the same
  calendar, that is, when they assign the same label to every month-of-year
  slot 1..12. The school labelling is a function of month-of-year alone, so a
  shift by k months imposes the calendar offset (k mod 12) on every month that
  does not wrap round the end of the window. There are therefore at most 12
  distinct alignments, and the 79 rotations realise each of them six or seven
  times, differing only in the wrap seam (79 = 6*12 + 7).
    E_exact  = number of distinct calendar offsets. Derived, not assumed.
    E(tau)   = number of classes after merging offsets whose 12-slot labellings
               agree on at least tau of the 12 slots. REPORTED FOR EVERY
               tau = 0..12, so no single threshold carries the argument.
    E_advisor= E(10). Ten of twelve is the advisor's own near-symmetry turned
               into a rule, and it is set at exactly the value his six-month
               shift achieves. That is stated rather than hidden: the threshold
               is chosen to admit his collapse, which is why the whole E(tau)
               family is reported instead of one number.

RESOLUTION FLOOR  1/E. A randomisation test whose alternative set contains E
  distinct members cannot return a p below 1/E.

DEDUPLICATED p  p_E = (1 + #{non-identity classes whose member reaches
  n_term = 17}) / E. The class representative is declared as the smallest
  k >= 1 in the class, and the sensitivity to that choice is reported by also
  taking the maximum over every member of every class -- the most
  anti-conservative rule available -- so the representative cannot be doing
  the work.

WHAT IS CHECKED ABOUT THE LETTER, item by item, reported whichever way it lands
  (i)   |TERM intersect (TERM + 6 mod 12)| = 6 of 7 ?
  (ii)  are the three matching shifts 12, 24, 36 multiples of 6 ?
  (iii) are they multiples of 6 BECAUSE of the six-month near-symmetry? Tested
        by asking whether the ODD multiples of six -- 6, 18, 30, 42, 54, 66,
        78 -- also reach 17. If they do not, the mechanism is the twelve-month
        exact period and not the six-month near-symmetry, and the letter's
        attribution is wrong even where its conclusion survives.
  (iv)  is 6 the right effective count ?

PREDICTION, before computing  E_exact = 12 by exact calendar arithmetic and
  E(10) = 6 after the collapse, so the honest floor is between 1/12 = 0.083 and
  1/6 = 0.167, and BOTH are above the observed 0.0506. On that prediction the
  advisor's conclusion holds and his number is the conservative end of a
  two-valued answer rather than the answer.

WHAT IS NOT CLAIMED  the deduplicated p is a RESOLUTION statement, not a new
  exact randomisation p-value. 79 is prime, so the cyclic group of order 79 has
  no proper non-trivial subgroup and a set of twelve representatives is not a
  subgroup; the exact-test guarantee attaches to all 79 rotations or to
  nothing. What 1/E measures is how many different calendars the test is able
  to put up as alternatives, which is the quantity the section quotes when it
  quotes a floor.
"""

BRANCH_TABLE = [
    dict(condition="b > 0 and |t| >= 2 at L = 3 on both primary series",
         framing="The regression replaces the sign test in the prose. The "
                 "seven-point sign test and its 0.0078 ceiling move to SI as "
                 "the ordinal cross-check; the term coefficient, its HAC "
                 "standard error and its p become the sentence."),
    dict(condition="b > 0 on both but |t| < 2 at L = 3",
         framing="Reported as a positive but imprecisely estimated effect. "
                 "The sign test stays, and the regression is reported as the "
                 "reason the claim is stated ordinally rather than as a "
                 "magnitude."),
    dict(condition="b <= 0 on either primary series",
         framing="The regression contradicts the sign test. Section 3.1's "
                 "second paragraph is rewritten around the contradiction "
                 "before anything else in this round is touched."),
]

BRANCH_TABLE_B = [
    dict(condition="E_exact == 12 and E(10) == 6",
         framing="The advisor's conclusion holds and his number is the "
                 "conservative half of a two-valued answer. The main text "
                 "replaces 1/79 = 0.0127 with the pair, and stops calling the "
                 "three matching shifts alternative alignments."),
    dict(condition="E_exact == 12 and E(10) != 6",
         framing="The collapse the letter proposes is not what the agreement "
                 "table supports. Report 1/12 and say why 1/6 does not follow."),
    dict(condition="E_exact != 12",
         framing="The calendar arithmetic in this file is wrong somewhere. "
                 "Nothing is reported until that is found."),
]

out = {"declaration": {"text": DECLARATION, "branch_table_A": BRANCH_TABLE,
                       "branch_table_B": BRANCH_TABLE_B}}

# The declared lag, computed from the declared formula rather than typed, so
# the number in the results file cannot drift from the rule that chose it.
T_SERIES = 79
LAG_PRIMARY = int(np.floor(4.0 * (T_SERIES / 100.0) ** (2.0 / 9.0)))
LAG_BAND = list(range(0, 13))

PRIMARY_SERIES = ("mi_bits_hf", "assortativity_all")
SECONDARY_SERIES = ("mi_bits_all", "assortativity_hf", "nmi_hf", "half_l1_hf")


def say(s=""):
    print(s, flush=True)


# ======================================================================= OLS


def ols(X, y):
    """Least squares by SVD. Returns (beta, residuals, XtX_inv)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    return beta, u, XtX_inv


def hac_cov(X, u, L, correction=False):
    """Newey-West HAC covariance, Bartlett kernel, lag L.

    S = G_0 + sum_{j=1..L} (1 - j/(L+1)) (G_j + G_j'), G_j = sum_t u_t u_{t-j}
    x_t x_{t-j}'. L = 0 is White's HC0 sandwich. `correction` applies the
    T/(T-k) small-sample factor to the meat; the primary number does not use it.
    """
    T, k = X.shape
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        G = Xu[j:].T @ Xu[:-j]
        S = S + w * (G + G.T)
    if correction:
        S = S * (T / (T - k))
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def fit(X, y, names, L=LAG_PRIMARY, correction=False):
    """One regression, reported at one lag. Everything the memo quotes is here."""
    beta, u, _ = ols(X, y)
    T, k = X.shape
    V = hac_cov(X, u, L, correction=correction)
    se = np.sqrt(np.diag(V))
    tstat = beta / se
    df = T - k
    p_t = 2.0 * sps.t.sf(np.abs(tstat), df)
    p_z = 2.0 * sps.norm.sf(np.abs(tstat))
    ybar = float(np.mean(y))
    ss_res = float(u @ u)
    ss_tot = float(((y - ybar) ** 2).sum())
    return dict(
        n=int(T), k=int(k), df=int(df), lag=int(L), correction=bool(correction),
        names=list(names),
        coef={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se)},
        t={nm: float(v) for nm, v in zip(names, tstat)},
        p_t={nm: float(v) for nm, v in zip(names, p_t)},
        p_normal={nm: float(v) for nm, v in zip(names, p_z)},
        r2=float(1.0 - ss_res / ss_tot), rss=ss_res,
        sigma=float(np.sqrt(ss_res / df)))


def design(rows_order, YEAR, MONTH, labels, spec="S1"):
    """Build (X, names) for one specification. `labels` maps ym -> label.

    Nothing here indexes a month positionally: rows_order is a list of ym
    labels and every column is built by looking each ym up in a dict.
    """
    yrs = sorted({YEAR[ym] for ym in rows_order})
    ref_year = yrs[0]
    n = len(rows_order)
    cols, names = [], []
    cols.append(np.ones(n)); names.append("const")
    cols.append(np.array([1.0 if labels[ym] == "term" else 0.0
                          for ym in rows_order])); names.append("term")
    if spec not in ("S2", "S3"):
        cols.append(np.array([1.0 if labels[ym] == "december" else 0.0
                              for ym in rows_order])); names.append("december")
    if spec == "S4":
        cols.append(np.array([1.0 if (labels[ym] == "term" and YEAR[ym] == 2020)
                              else 0.0 for ym in rows_order]))
        names.append("term_x_2020")
    if spec != "S6":
        for y in yrs[1:]:
            cols.append(np.array([1.0 if YEAR[ym] == y else 0.0
                                  for ym in rows_order]))
            names.append(f"year_{y}")
    if spec != "S5":
        idx = np.arange(n, dtype=float)
        cols.append(idx - idx.mean()); names.append("trend")
    _ = ref_year, MONTH
    return np.column_stack(cols), names


def main():
    # ==================================================================== inputs
    p41 = json.load(open(f"{ROOT}/eda/results_p41.json"))
    p42 = json.load(open(f"{ROOT}/eda/results_p42.json"))
    p51 = json.load(open(f"{ROOT}/eda/results_p51.json"))
    p54 = json.load(open(f"{ROOT}/eda/results_p54.json"))
    p58 = json.load(open(f"{ROOT}/eda/results_p58.json"))
    p63 = json.load(open(f"{ROOT}/eda/results_p63.json"))

    rows = p42["rows"]
    N = len(rows)
    YMS = sorted(r["ym"] for r in rows)
    POS = {ym: i for i, ym in enumerate(YMS)}
    YEAR = {r["ym"]: r["year"] for r in rows}
    MONTH = {r["ym"]: r["month"] for r in rows}
    TRUE = {r["ym"]: r["semester"] for r in rows}
    SER = {name: {r["ym"]: r[name] for r in rows}
           for name in PRIMARY_SERIES + SECONDARY_SERIES}

    anchors = {}
    say("=== 64.0 anchors: the run aborts on any failure ===")

    # ------------------------------------------------------------------- A1
    # The series itself. Same three checks p58 opens with, for the same reason:
    # everything downstream is keyed by ym label and never by position.
    strictly_increasing = all(YMS[i] < YMS[i + 1] for i in range(N - 1))
    assert N == 79, f"expected 79 months, got {N}"
    assert strictly_increasing, "the month series is not strictly increasing"
    assert len(set(YMS)) == N, "duplicate ym in results_p42.json rows"
    assert TERM == (3, 4, 5, 6, 9, 10, 11), f"TERM moved: {TERM}"
    assert VAC == (1, 2, 7, 8), f"VAC moved: {VAC}"
    mismatch = [ym for ym in YMS if label(ym) != TRUE[ym]]
    assert not mismatch, f"label() disagrees with rows[].semester on {mismatch}"
    counts_true = {lab: sum(1 for ym in YMS if TRUE[ym] == lab)
                   for lab in ("term", "vacation", "december")}
    say(f"  A1  n_months {N} {YMS[0]}..{YMS[-1]}, strictly increasing "
        f"{strictly_increasing}; label() reproduces rows[].semester {N}/{N}; "
        f"{counts_true}")
    anchors["A1"] = dict(n_months=N, first_ym=YMS[0], last_ym=YMS[-1],
                         strictly_increasing=strictly_increasing,
                         n_label_mismatch=len(mismatch), term=TERM,
                         vacation=VAC, label_counts=counts_true)

    # ------------------------------------------------------------------- A2
    # The regression is asked to replace p41's sign test, so it has to be run on
    # p41's own series. This proves that it is: the seven per-year contrasts are
    # rebuilt here from results_p42.json's assortativity_all and must equal the
    # ones results_p41.json stores, BIT FOR BIT. The sign test itself is then
    # re-derived from an independent binomial tail rather than read.
    years = sorted({YEAR[ym] for ym in YMS})
    contrasts, worst = {}, 0.0
    for y in years:
        t_v = [SER["assortativity_all"][ym] for ym in YMS
               if YEAR[ym] == y and MONTH[ym] in TERM]
        v_v = [SER["assortativity_all"][ym] for ym in YMS
               if YEAR[ym] == y and MONTH[ym] in VAC]
        c = float(np.median(t_v) - np.median(v_v))
        ref = p41["per_year"][str(y)]["contrast_median"]
        worst = max(worst, abs(c - ref))
        contrasts[y] = c
    assert worst == 0.0, f"per-year contrasts differ from p41 by {worst:.3e}"
    n_pos = sum(1 for c in contrasts.values() if c > 0)
    n_yr = len(contrasts)
    p_sign = float(sum(sps.binom.pmf(i, n_yr, 0.5)
                       for i in range(n_pos, n_yr + 1)))
    p_sign_floor = 0.5 ** n_yr
    assert n_pos == p41["sign_test"]["contrast"]["n_positive"], "n_positive"
    assert abs(p_sign - p41["sign_test"]["contrast"]["p_one_sided"]) < 1e-15, \
        "the sign test does not reproduce"
    assert abs(p_sign_floor
               - p41["sign_test"]["smallest_attainable_one_sided"]) < 1e-18, \
        "the sign test's floor does not reproduce"
    say(f"  A2  seven per-year contrasts rebuilt from results_p42.json: max "
        f"|diff| vs p41 {worst:.1e} (bit-exact)")
    say(f"      sign test {n_pos}/{n_yr} positive, one-sided p {p_sign!r} "
        f"(p41 {p41['sign_test']['contrast']['p_one_sided']!r})")
    say(f"      smallest p a {n_yr}-point sign test can return "
        f"{p_sign_floor!r} = 2^-{n_yr}")
    anchors["A2"] = dict(max_abs_diff_vs_p41=worst, bit_exact=bool(worst == 0.0),
                         n_years=n_yr, n_positive=n_pos, p_one_sided=p_sign,
                         p_one_sided_p41=p41["sign_test"]["contrast"][
                             "p_one_sided"],
                         smallest_attainable=p_sign_floor,
                         smallest_attainable_p41=p41["sign_test"][
                             "smallest_attainable_one_sided"],
                         contrasts={str(y): contrasts[y] for y in years})

    # ------------------------------------------------------------------- A3
    # The floor count and both exact within-year p values, corrected and
    # published, re-derived through p42's own null. The floors are compared bit
    # for bit against the files that hold them, so this is a recount and not a
    # re-measurement.
    FLOOR = p51["cells"]["202312|national"]["mi_perm_median"]
    FLOOR_PUB = p54["published"]["floor"]
    assert FLOOR == p54["corrected"]["floor"], "floor is not the one p54 counted"
    a3 = {}
    for tag, fl in (("corrected", FLOOR), ("published", FLOOR_PUB)):
        clear = [ym for ym in YMS if SER["mi_bits_hf"][ym] >= fl]
        obs = {lab: sum(1 for ym in clear if TRUE[ym] == lab)
               for lab in ("term", "vacation", "december")}
        per_year = {y: (sum(1 for ym in YMS if YEAR[ym] == y),
                        sum(1 for ym in YMS if YEAR[ym] == y
                            and TRUE[ym] == "term"),
                        sum(1 for ym in YMS if YEAR[ym] == y
                            and SER["mi_bits_hf"][ym] >= fl))
                    for y in years}
        _, p_w, _ = within_year_null(per_year, obs["term"])
        want = p54[tag]
        assert len(clear) == want["n_at_or_above"], f"{tag} count differs"
        assert clear == sorted(want["months"]), f"{tag} month set differs"
        assert obs == want["observed"], f"{tag} label split differs"
        assert abs(p_w - want["p_term_ge"]) < 1e-18, f"{tag} within-year p"
        a3[tag] = dict(floor=fl, n_at_or_above=len(clear), observed=obs,
                       months=clear, p_term_ge=p_w,
                       p_term_ge_p54=want["p_term_ge"])
        say(f"  A3  {tag:>9} floor {fl!r}: {len(clear)} months clear, "
            f"{obs['term']} term, within-year p {p_w:.15e}")
    CLEARING = a3["corrected"]["months"]
    N_OBS = a3["corrected"]["observed"]["term"]
    anchors["A3"] = a3

    # ------------------------------------------------------------------- A4
    # p58's rotation, re-implemented here rather than read, then compared to the
    # array p58 stored. If this file's rotation differed from p58's by even one
    # k, Part B would be analysing the resolution of a different test.
    def rotate(k):
        return {YMS[i]: TRUE[YMS[(i - k) % N]] for i in range(N)}

    LAB = [rotate(k) for k in range(N)]
    n_term = [sum(1 for ym in CLEARING if LAB[k][ym] == "term") for k in range(N)]
    ge = [k for k in range(1, N) if n_term[k] >= N_OBS]
    p_shift = (1 + len(ge)) / N
    agreement = [sum(1 for ym in YMS if LAB[k][ym] == TRUE[ym]) for k in range(N)]
    margins = {sum(1 for ym in YMS if LAB[k][ym] == "term") for k in range(N)}
    assert n_term == p58["p_shift"]["n_term_by_k"], "n_term_by_k differs from p58"
    assert ge == p58["p_shift"]["shifts_at_or_above_observed"], "shifts differ"
    assert p_shift == p58["p_shift"]["p"], "p_shift differs from p58"
    assert agreement == p58["degeneracy"]["agreement_by_k"], "agreement differs"
    assert margins == {counts_true["term"]}, f"term margin not invariant: {margins}"
    say(f"  A4  rotation re-implemented: n_term_by_k, matching shifts {ge} and "
        f"p = {p_shift!r} all identical to results_p58.json")
    say(f"      term margin invariant at {counts_true['term']}; "
        f"agreement(12) = {agreement[12]}")
    anchors["A4"] = dict(matching_shifts=ge, p_shift=p_shift,
                         p_shift_p58=p58["p_shift"]["p"],
                         n_term_by_k_identical=True,
                         agreement_identical=True,
                         term_margin=counts_true["term"],
                         agreement_12=agreement[12])

    # ------------------------------------------------------------------- A5
    # The level trend the section spends a paragraph on, and the thing the year
    # fixed effects are supposed to absorb. Recomputed from the same rows, checked
    # against the value p63 draws Figure 2(a) from.
    med_by_year = {y: float(np.median([SER["mi_bits_hf"][ym] for ym in YMS
                                       if YEAR[ym] == y])) for y in years}
    trend_ratio = max(med_by_year.values()) / min(med_by_year.values())
    ref_trend = [c["value"] for c in p63["caption_numbers"]
                 if c["panel"] == "a" and "level trend" in c["what"]][0]
    assert abs(trend_ratio - ref_trend) < 1e-12, "the level trend does not match p63"
    say(f"  A5  MI year medians {min(med_by_year.values()):.5f} -> "
        f"{max(med_by_year.values()):.5f}, ratio {trend_ratio:.4f} "
        f"(p63 {ref_trend:.4f}, manuscript quotes 1.70)")
    anchors["A5"] = dict(year_medians={str(y): med_by_year[y] for y in years},
                         level_trend=trend_ratio, level_trend_p63=ref_trend)

    # ------------------------------------------------------------------- A6
    # The estimator this file adds. Two internal second implementations, because
    # nothing upstream has ever run a regression and there is no published number
    # to anchor one against.
    #   (a) Frisch-Waugh-Lovell: partial the term column and the outcome on
    #       every other regressor and regress the residuals. The slope must be
    #       the multivariate coefficient.
    #   (b) HAC at L = 0 must equal the HC0 sandwich built by a different route.
    y0 = np.array([SER["mi_bits_hf"][ym] for ym in YMS])
    X0, nm0 = design(YMS, YEAR, MONTH, TRUE, "S1")
    b0, u0, _ = ols(X0, y0)
    j_term = nm0.index("term")
    Z = np.delete(X0, j_term, axis=1)
    r_y = y0 - Z @ np.linalg.lstsq(Z, y0, rcond=None)[0]
    r_x = X0[:, j_term] - Z @ np.linalg.lstsq(Z, X0[:, j_term], rcond=None)[0]
    b_fwl = float(r_x @ r_y / (r_x @ r_x))
    fwl_diff = abs(b_fwl - float(b0[j_term]))
    XtX_inv = np.linalg.inv(X0.T @ X0)
    meat = (X0 * u0[:, None]).T @ (X0 * u0[:, None])
    hc0_ref = XtX_inv @ meat @ XtX_inv
    hc0_diff = float(np.max(np.abs(hac_cov(X0, u0, 0) - hc0_ref)))
    assert fwl_diff < 1e-15, f"FWL disagrees with the multivariate fit: {fwl_diff}"
    assert hc0_diff < 1e-30, f"HAC(0) is not HC0: {hc0_diff}"
    say(f"  A6  FWL reproduces b(term) to {fwl_diff:.1e}; HAC(L=0) equals the "
        f"HC0 sandwich to {hc0_diff:.1e}")
    anchors["A6"] = dict(fwl_abs_diff=fwl_diff, hac0_vs_hc0_max_abs=hc0_diff)

    # ------------------------------------------------------------------- A7
    # statsmodels, if it is installed. It is NOT a dependency of this repo --
    # requirements.txt lists it commented out -- so its absence is recorded and
    # is not a failure. When present it is a third implementation of the same
    # sandwich, written by somebody else.
    sm_check = dict(available=False)
    try:
        import statsmodels.api as sm  # noqa: F401
        m = sm.OLS(y0, X0).fit(cov_type="HAC",
                               cov_kwds=dict(maxlags=LAG_PRIMARY,
                                             use_correction=False))
        mine = hac_cov(X0, u0, LAG_PRIMARY)
        rel = float(np.max(np.abs(np.sqrt(np.diag(mine)) - m.bse)
                           / np.abs(m.bse)))
        assert rel < 1e-10, f"statsmodels disagrees with hac_cov by {rel:.2e}"
        sm_check = dict(available=True, version=sm.__dict__.get("__version__"),
                        max_rel_diff_se=rel,
                        note="statsmodels HAC with use_correction=False")
        say(f"  A7  statsmodels HAC agrees with hac_cov to {rel:.1e} "
            f"(max relative, standard errors)")
    except ImportError:
        say("  A7  statsmodels not installed -- third implementation skipped, "
            "not a failure (it is not a dependency of this repo)")
    anchors["A7"] = sm_check
    out["anchors"] = anchors

    # ==================================================================== 64.1
    say(f"\n=== 64.1 the declared HAC lag ===")
    say(f"  L = floor(4 * (T/100)^(2/9)) at T = {T_SERIES} -> "
        f"{4.0 * (T_SERIES / 100.0) ** (2.0 / 9.0):.4f} -> L = {LAG_PRIMARY}")
    out["lag"] = dict(rule="floor(4 * (T/100)^(2/9))", T=T_SERIES,
                      raw=float(4.0 * (T_SERIES / 100.0) ** (2.0 / 9.0)),
                      L=LAG_PRIMARY, band=LAG_BAND,
                      declared_before_the_run=True)

    # ==================================================================== 64.2
    say("\n=== 64.2 PRIMARY: S1 at the declared lag, both primary series ===")
    primary = {}
    for name in PRIMARY_SERIES:
        y = np.array([SER[name][ym] for ym in YMS])
        X, nm = design(YMS, YEAR, MONTH, TRUE, "S1")
        r = fit(X, y, nm)
        rc = fit(X, y, nm, correction=True)
        r["with_small_sample_correction"] = dict(
            se=rc["se"]["term"], t=rc["t"]["term"], p_t=rc["p_t"]["term"])
        primary[name] = r
        say(f"  {name}")
        say(f"    b(term) = {r['coef']['term']:+.7f}   HAC se "
            f"{r['se']['term']:.7f}   t {r['t']['term']:+.4f}   "
            f"p {r['p_t']['term']:.6g}   (df {r['df']})")
        say(f"    b(december) = {r['coef']['december']:+.7f} "
            f"(se {r['se']['december']:.7f}, t {r['t']['december']:+.3f})   "
            f"trend {r['coef']['trend']:+.3e}   R2 {r['r2']:.4f}")
    out["primary"] = primary

    # the two comparisons the memo needs to interpret the size of b
    interp = {}
    for name in PRIMARY_SERIES:
        y = np.array([SER[name][ym] for ym in YMS])
        interp[name] = dict(
            series_median=float(np.median(y)), series_mean=float(np.mean(y)),
            series_range=float(y.max() - y.min()),
            b_over_median=float(primary[name]["coef"]["term"] / np.median(y)),
            b_over_range=float(primary[name]["coef"]["term"]
                               / (y.max() - y.min())))
    interp["assortativity_all"]["p41_mean_within_year_contrast"] = float(
        np.mean([contrasts[y] for y in years if y != 2020]))
    out["interpretation"] = interp
    say(f"  b(term) on assortativity_all vs p41's mean within-year contrast "
        f"{interp['assortativity_all']['p41_mean_within_year_contrast']:.7f}")

    # ==================================================================== 64.3
    say("\n=== 64.3 the whole declared lag band, L = 0..12 ===")
    band = {}
    for name in PRIMARY_SERIES:
        y = np.array([SER[name][ym] for ym in YMS])
        X, nm = design(YMS, YEAR, MONTH, TRUE, "S1")
        rowsL = []
        for L in LAG_BAND:
            r = fit(X, y, nm, L=L)
            rowsL.append(dict(L=L, se=r["se"]["term"], t=r["t"]["term"],
                              p_t=r["p_t"]["term"]))
        band[name] = dict(
            coef=primary[name]["coef"]["term"], rows=rowsL,
            se_min=min(d["se"] for d in rowsL),
            se_max=max(d["se"] for d in rowsL),
            p_min=min(d["p_t"] for d in rowsL),
            p_max=max(d["p_t"] for d in rowsL),
            all_p_below_05=bool(all(d["p_t"] < 0.05 for d in rowsL)),
            all_abs_t_at_least_2=bool(all(abs(d["t"]) >= 2.0 for d in rowsL)))
        say(f"  {name}: b {band[name]['coef']:+.7f} (fixed), se "
            f"{band[name]['se_min']:.7f}..{band[name]['se_max']:.7f}, "
            f"p {band[name]['p_min']:.3g}..{band[name]['p_max']:.3g}, "
            f"|t| >= 2 everywhere: {band[name]['all_abs_t_at_least_2']}")
        say("    " + "  ".join(f"L{d['L']}:t={d['t']:.2f}" for d in rowsL))
    out["lag_band"] = band

    # ==================================================================== 64.4
    say("\n=== 64.4 the declared specification family ===")
    specs = {}
    for name in PRIMARY_SERIES:
        specs[name] = {}
        for spec in ("S1", "S2", "S3", "S4", "S5", "S6"):
            keep = [ym for ym in YMS
                    if not (spec == "S2" and TRUE[ym] == "december")]
            y = np.array([SER[name][ym] for ym in keep])
            X, nm = design(keep, YEAR, MONTH, TRUE, spec)
            r = fit(X, y, nm)
            specs[name][spec] = dict(
                n=r["n"], k=r["k"], df=r["df"],
                b_term=r["coef"]["term"], se_term=r["se"]["term"],
                t_term=r["t"]["term"], p_term=r["p_t"]["term"], r2=r["r2"],
                extra={k2: dict(coef=r["coef"][k2], se=r["se"][k2],
                                t=r["t"][k2], p_t=r["p_t"][k2])
                       for k2 in ("term_x_2020", "december") if k2 in r["coef"]})
            say(f"  {name:>18} {spec}: n {r['n']:>2}  b {r['coef']['term']:+.7f}"
                f"  se {r['se']['term']:.7f}  t {r['t']['term']:+.3f}"
                f"  p {r['p_t']['term']:.4g}")
        s4 = specs[name]["S4"]["extra"].get("term_x_2020")
        if s4:
            say(f"  {name:>18} S4: term x 2020 = {s4['coef']:+.7f} "
                f"(se {s4['se']:.7f}, t {s4['t']:+.3f}); "
                f"b + interaction = {specs[name]['S4']['b_term'] + s4['coef']:+.7f}")
    out["specifications"] = specs

    # ==================================================================== 64.5
    say("\n=== 64.5 the four secondary series, S1 at the declared lag ===")
    secondary = {}
    for name in SECONDARY_SERIES:
        y = np.array([SER[name][ym] for ym in YMS])
        X, nm = design(YMS, YEAR, MONTH, TRUE, "S1")
        r = fit(X, y, nm)
        secondary[name] = dict(b_term=r["coef"]["term"], se_term=r["se"]["term"],
                               t_term=r["t"]["term"], p_term=r["p_t"]["term"],
                               r2=r["r2"])
        say(f"  {name:>18}: b {r['coef']['term']:+.7f}  se {r['se']['term']:.7f}"
            f"  t {r['t']['term']:+.3f}  p {r['p_t']['term']:.4g}")
    out["secondary"] = secondary
    all_series = dict(primary, **{k: None for k in ()})
    n_pos_all = sum(1 for nmS in PRIMARY_SERIES
                    if primary[nmS]["coef"]["term"] > 0) + \
        sum(1 for nmS in SECONDARY_SERIES if secondary[nmS]["b_term"] > 0)
    out["sign_agreement"] = dict(
        n_series=len(PRIMARY_SERIES) + len(SECONDARY_SERIES),
        n_positive=n_pos_all)
    _ = all_series

    # ==================================================================== 64.6
    say("\n=== 64.6 R-ROT: the same regression under all 79 rotations ===")
    rot = {}
    for name in PRIMARY_SERIES:
        y = np.array([SER[name][ym] for ym in YMS])
        bs, ts = [], []
        for k in range(N):
            X, nm = design(YMS, YEAR, MONTH, LAB[k], "S1")
            r = fit(X, y, nm)
            bs.append(r["coef"]["term"]); ts.append(r["t"]["term"])
        rank_b = 1 + sum(1 for k in range(N) if bs[k] > bs[0])
        rank_t = 1 + sum(1 for k in range(N) if ts[k] > ts[0])
        ge_t = [k for k in range(1, N) if ts[k] >= ts[0]]
        rot[name] = dict(
            b_by_k=bs, t_by_k=ts, rank_of_b=rank_b, rank_of_t=rank_t,
            p_rotation_t=(1 + len(ge_t)) / N, shifts_ge_t=ge_t,
            b_max_over_shifts=float(max(bs[1:])),
            t_max_over_shifts=float(max(ts[1:])),
            resolution_floor_naive=1.0 / N)
        say(f"  {name}: rank of observed t among the {N} rotations "
            f"{rank_t} (b: {rank_b}); rotation p on t = "
            f"{rot[name]['p_rotation_t']:.7f}; best alternative t "
            f"{rot[name]['t_max_over_shifts']:+.3f} vs observed {ts[0]:+.3f}")
    out["rotation_regression"] = rot

    # ==================================================================== 64.7
    say("\n=== 64.7 PART B: the alignment algebra ===")
    # The calendar is a function of month-of-year. Build the 12-slot labelling
    # once, from p41's partition, and derive every offset from it. Nothing here
    # touches the data.
    def slot_label(m):
        return ("term" if m in TERM else
                "vacation" if m in VAC else "december")

    SLOT = {m: slot_label(m) for m in range(1, 13)}
    slot_counts = {lab: sum(1 for m in SLOT if SLOT[m] == lab)
                   for lab in ("term", "vacation", "december")}
    # offset c: month m receives the label of month m - c
    OFF = {c: {m: SLOT[((m - c - 1) % 12) + 1] for m in range(1, 13)}
           for c in range(12)}
    n_offsets = len({tuple(OFF[c][m] for m in range(1, 13)) for c in range(12)})
    assert n_offsets == 12, f"the 12 calendar offsets are not distinct: {n_offsets}"
    # agreement between offset c and offset c' depends only on (c - c') mod 12
    slot_agree = [sum(1 for m in range(1, 13) if OFF[d][m] == OFF[0][m])
                  for d in range(12)]
    pairwise_ok = all(
        sum(1 for m in range(1, 13) if OFF[c][m] == OFF[cp][m])
        == slot_agree[(c - cp) % 12] for c in range(12) for cp in range(12))
    assert pairwise_ok, "offset agreement is not a function of the difference"
    slot_chance = sum(v * v for v in slot_counts.values()) / 12.0
    say(f"  the 12 calendar offsets are distinct: {n_offsets}/12")
    say(f"  slot agreement with the true calendar, offsets 0..11: {slot_agree}")
    say(f"  chance agreement over 12 slots: {slot_chance:.4f}")
    out["alignment_algebra"] = dict(
        slot_labels={str(m): SLOT[m] for m in range(1, 13)},
        slot_counts=slot_counts, n_distinct_offsets=n_offsets,
        slot_agreement_by_offset=slot_agree,
        agreement_is_function_of_difference=pairwise_ok,
        slot_chance_agreement=slot_chance,
        seam="79 = 6*12 + 7, so a shift by k imposes offset (k mod 12) on the "
             "79-k months that do not wrap and offset ((k-7) mod 12) on the k "
             "that do; k and k+12 therefore test the same calendar and differ "
             "only in the seam")

    # the seam decomposition, verified against the realised agreement
    seam_pred = []
    for k in range(N):
        bulk_c, seam_c = k % 12, (k - 7) % 12
        n_seam = k
        pred = 0
        for i in range(N):
            m = MONTH[YMS[i]]
            c = seam_c if i < n_seam else bulk_c
            pred += 1 if OFF[c][m] == SLOT[m] else 0
        seam_pred.append(pred)
    assert seam_pred == agreement, "the seam decomposition does not reproduce " \
                                   "the realised agreement"
    say(f"  the seam decomposition reproduces all {N} realised agreements: True")
    out["alignment_algebra"]["seam_decomposition_reproduces_agreement"] = True

    # ==================================================================== 64.8
    say("\n=== 64.8 E(tau): how many alignments survive each merge threshold ===")

    def classes_at(tau):
        """Merge offsets whose 12-slot labellings agree on >= tau slots.
        Union-find over the 12 offsets; returns the sorted class list."""
        parent = list(range(12))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for c in range(12):
            for cp in range(c + 1, 12):
                if slot_agree[(c - cp) % 12] >= tau:
                    ra, rb = find(c), find(cp)
                    if ra != rb:
                        parent[ra] = rb
        groups = {}
        for c in range(12):
            groups.setdefault(find(c), []).append(c)
        return sorted([sorted(v) for v in groups.values()])

    E_by_tau = {}
    for tau in range(0, 13):
        cl = classes_at(tau)
        E_by_tau[tau] = dict(E=len(cl), classes=cl)
        say(f"  tau = {tau:>2}: E = {len(cl):>2}   {cl if len(cl) <= 12 else ''}")
    E_exact = E_by_tau[12]["E"]
    E_advisor = E_by_tau[10]["E"]
    distinct_E = sorted({E_by_tau[t]["E"] for t in range(0, 13)})
    say(f"  distinct values E(tau) can take over tau = 0..12: {distinct_E}")
    out["effective_alignments"] = dict(
        E_by_tau={str(t): E_by_tau[t] for t in range(0, 13)},
        E_exact=E_exact, E_advisor_tau10=E_advisor,
        distinct_values_of_E=distinct_E,
        criterion="two offsets are the same alignment when their 12-slot "
                  "labellings agree on at least tau slots; tau = 12 is exact "
                  "equality, tau = 10 is the advisor's six-month near-symmetry")

    # ==================================================================== 64.9
    say("\n=== 64.9 the deduplicated test, and what its floor is ===")
    dedup = {}
    for tag, E, tau in (("exact", E_exact, 12), ("advisor", E_advisor, 10)):
        cl = E_by_tau[tau]["classes"]
        ident = [g for g in cl if 0 in g][0]
        members = {tuple(g): [k for k in range(N) if (k % 12) in g] for g in cl}
        rep_small, hits_small = {}, []
        rep_max, hits_max = {}, []
        for g in cl:
            ms = members[tuple(g)]
            non_zero = [k for k in ms if k != 0]
            rs = min(non_zero) if non_zero else 0
            rep_small[str(g)] = rs
            rep_max[str(g)] = max(ms, key=lambda k: n_term[k])
            if g != ident:
                if n_term[rs] >= N_OBS:
                    hits_small.append(g)
                if max(n_term[k] for k in ms) >= N_OBS:
                    hits_max.append(g)
        p_small = (1 + len(hits_small)) / E
        p_max = (1 + len(hits_max)) / E
        dedup[tag] = dict(
            tau=tau, E=E, classes=cl, identity_class=ident,
            class_members={str(g): members[tuple(g)] for g in cl},
            representative_smallest={str(g): rep_small[str(g)] for g in cl},
            n_term_at_representative={str(g): n_term[rep_small[str(g)]]
                                      for g in cl},
            max_n_term_in_class={str(g): max(n_term[k] for k in members[tuple(g)])
                                 for g in cl},
            p_dedup_smallest_rep=p_small, p_dedup_max_over_class=p_max,
            resolution_floor=1.0 / E,
            p_equals_floor=bool(p_small == 1.0 / E and p_max == 1.0 / E),
            n_nonidentity_classes_reaching_observed=len(hits_max))
        say(f"  {tag:>8} (tau {tau}): E = {E}, floor 1/{E} = {1.0 / E:.4f}; "
            f"non-identity classes reaching {N_OBS}: {len(hits_max)} "
            f"(under the most permissive representative rule)")
        say(f"           p_dedup = {p_small:.4f} (smallest-k rep) = "
            f"{p_max:.4f} (max over class)")
    out["deduplicated"] = dedup

    # ==================================================================== 64.10
    say("\n=== 64.10 the letter, item by item ===")
    shifted_term = {((m + 6 - 1) % 12) + 1 for m in TERM}
    overlap6 = sorted(set(TERM) & shifted_term)
    odd6 = [k for k in range(6, N, 12)]
    even12 = [k for k in range(12, N, 12)]
    odd6_hits = [k for k in odd6 if n_term[k] >= N_OBS]
    matching_mod12 = sorted({k % 12 for k in ge})
    matching_mod6 = sorted({k % 6 for k in ge})
    # which single clearing month breaks the six-month shift
    broken6 = [ym for ym in CLEARING if LAB[6][ym] != "term"]
    verdict = dict(
        i_overlap_six_of_seven=dict(
            claim="|TERM ∩ (TERM+6 mod 12)| = 6 of 7",
            shifted_set=sorted(shifted_term), overlap=overlap6,
            n_overlap=len(overlap6), n_term_months=len(TERM),
            holds=bool(len(overlap6) == 6 and len(TERM) == 7)),
        ii_matches_are_multiples_of_six=dict(
            claim="the three matching shifts are multiples of 6",
            shifts=ge, residues_mod_6=matching_mod6,
            holds=bool(all(k % 6 == 0 for k in ge))),
        iii_mechanism_is_the_six_month_symmetry=dict(
            claim="they match BECAUSE a six-month shift nearly reproduces the "
                  "calendar; if so the odd multiples of six should match too",
            odd_multiples_of_six=odd6,
            n_term_at_odd_multiples={str(k): n_term[k] for k in odd6},
            odd_multiples_reaching_observed=odd6_hits,
            multiples_of_twelve=even12,
            n_term_at_multiples_of_twelve={str(k): n_term[k] for k in even12},
            residues_mod_12_of_matches=matching_mod12,
            clearing_months_lost_at_shift_six=broken6,
            holds=bool(len(odd6_hits) > 0)),
        iv_effective_count_is_six=dict(
            claim="of 79 alignments only about 6 are truly independent",
            E_exact=E_exact, E_at_tau_10=E_advisor,
            distinct_values_of_E=distinct_E,
            floor_exact=1.0 / E_exact, floor_advisor=1.0 / E_advisor,
            observed_p=p_shift,
            observed_below_both_floors=bool(p_shift < 1.0 / E_exact
                                            and p_shift < 1.0 / E_advisor),
            holds=bool(E_advisor == 6)))
    for key, v in verdict.items():
        say(f"  {key}: holds = {v['holds']}")
    say(f"  the six-month shift loses exactly these clearing months: {broken6}")
    say(f"  n_term at the odd multiples of six: "
        f"{[n_term[k] for k in odd6]} (observed {N_OBS})")
    say(f"  n_term at the multiples of twelve:  "
        f"{[n_term[k] for k in even12]} (observed {N_OBS})")
    out["letter_verdict"] = verdict

    # ==================================================================== 64.11
    say("\n=== 64.11 R-ROT under the same deduplication ===")
    # The declaration says R-ROT's resolution is subject to Part B's analysis,
    # so it is applied here rather than left as an aside. The question is not
    # only how many rotations beat the observed t, but whether any of them is a
    # DIFFERENT calendar from the true one.
    rot_dedup = {}
    for name in PRIMARY_SERIES:
        ts = rot[name]["t_by_k"]
        t_obs = ts[0]
        ge_t = [k for k in range(1, N) if ts[k] >= t_obs]
        res12 = sorted({k % 12 for k in ge_t})
        entry = dict(t_observed=t_obs, shifts_ge_t=ge_t,
                     residues_mod_12=res12,
                     all_ge_t_are_offset_zero=bool(all(k % 12 == 0
                                                       for k in ge_t)))
        for tag, E, tau in (("exact", E_exact, 12), ("advisor", E_advisor, 10)):
            cl = E_by_tau[tau]["classes"]
            ident = [g for g in cl if 0 in g][0]
            hits, best = [], {}
            for g in cl:
                ms = [k for k in range(N) if (k % 12) in g]
                best[str(g)] = float(max(ts[k] for k in ms if k != 0)) \
                    if [k for k in ms if k != 0] else None
                if g != ident and max(ts[k] for k in ms) >= t_obs:
                    hits.append(g)
            entry[tag] = dict(E=E, tau=tau, identity_class=ident,
                              best_t_in_class=best,
                              n_nonidentity_classes_ge_t=len(hits),
                              p_dedup=(1 + len(hits)) / E,
                              resolution_floor=1.0 / E)
        rot_dedup[name] = entry
        say(f"  {name}: rotations with t >= observed: {ge_t} "
            f"(residues mod 12 {res12}); every one is the true calendar "
            f"displaced by the seam: {entry['all_ge_t_are_offset_zero']}")
        say(f"    deduplicated p on t: {entry['exact']['p_dedup']:.4f} at "
            f"E = {E_exact}, {entry['advisor']['p_dedup']:.4f} at "
            f"E = {E_advisor}")
    out["rotation_regression_dedup"] = rot_dedup

    branch_b = ("E_exact == 12 and E(10) == 6" if (E_exact == 12 and E_advisor == 6)
                else "E_exact == 12 and E(10) != 6" if E_exact == 12
                else "E_exact != 12")
    b_mi = primary["mi_bits_hf"]
    b_as = primary["assortativity_all"]
    pos_both = b_mi["coef"]["term"] > 0 and b_as["coef"]["term"] > 0
    t_both = (abs(b_mi["t"]["term"]) >= 2.0 and abs(b_as["t"]["term"]) >= 2.0)
    branch_a = ("b > 0 and |t| >= 2 at L = 3 on both primary series"
                if pos_both and t_both else
                "b > 0 on both but |t| < 2 at L = 3" if pos_both else
                "b <= 0 on either primary series")
    out["branch"] = dict(part_A=branch_a, part_B=branch_b)
    say(f"\n  branch A: {branch_a}")
    say(f"  branch B: {branch_b}")

    # the declared predictions, scored
    out["declaration"]["prediction_outcome"] = dict(
        sign_positive_on_both_primary=bool(pos_both),
        magnitude_vs_p41_contrast=dict(
            b_assortativity=b_as["coef"]["term"],
            p41_mean_within_year_contrast=interp["assortativity_all"][
                "p41_mean_within_year_contrast"],
            ratio=float(b_as["coef"]["term"] / interp["assortativity_all"][
                "p41_mean_within_year_contrast"])),
        E_exact_predicted_12=bool(E_exact == 12),
        E_tau10_predicted_6=bool(E_advisor == 6),
        observed_p_below_both_floors=bool(
            verdict["iv_effective_count_is_six"]["observed_below_both_floors"]))

    out["inputs"] = dict(
        series="results_p42.json rows[*], not rebuilt",
        labels="results_p42.json rows[*].semester, reproduced by p42's label()",
        floor="results_p51.json cells['202312|national'].mi_perm_median",
        anchors="results_p41.json (sign test), results_p54.json (counts and "
                "exact nulls), results_p58.json (the rotation), "
                "results_p63.json (Figure 2's level trend)",
        no_parquet=True, rng=None, deterministic=True,
        writes_only="eda/results_p64.json")

    with open(f"{ROOT}/eda/results_p64.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p64.json")


if __name__ == "__main__":
    main()
