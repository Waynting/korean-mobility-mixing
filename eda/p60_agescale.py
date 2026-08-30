#!/usr/bin/env python
"""Phase 60 — the age axis, coarsened the way the spatial axis already was.

WHY THIS EXISTS. Figure 4 pairs a spatial panel and an age panel that do not
measure the same thing: (a) is assortativity against the number of locations,
79 monthly curves out of `results_p38.json`; (b) is the March-versus-December
2020 change in the discretionary arrival share, two months out of
`results_p19.json`. The advisor's 8-27 letter, point 4, is right at the code
level -- `p47_fig34.py:245` reads p38 and `p47_fig34.py:281` reads p19/p18b --
and the consequence is that the age axis reads as bolted on. His fix was a
single 16-bin-versus-3-bin number. A single number is still a two-point
comparison and would read as bolted on for a different reason, so this phase
builds the mirror instead: the SAME estimator, assortativity, against the
number of AGE bins, one curve per month for all 79 months, exactly the shape of
panel (a).

WHAT IT COSTS. Nothing. Newman's r on the co-arrival matrix is a function of
e = A / A.sum() alone -- `p26_matrix.py:145-167` uses `pop` only for the
reciprocity assert and the eigenvector -- and `results_p37.json` already stores
all 79 16x16 matrices. No parquet, no derived tree, no population vector, no
external drive. `results_p37.json` is NOT rewritten: it is gated by
`p31_report_audit.py`, recomputed by `p36_recompute.py`, sha256-pinned in
`eda/archive/README.md` layer 2, and costs 27 minutes to rebuild. This is what
p37 did to p26, what p38 did to p34 and what p41 did to p37; p60 does the same
and writes `results_p60.json`.

WHAT THE ESTIMATOR IS. `assortativity` is imported from `p27_survey`, not
retyped, and anchor A3 asserts that the imported function reproduces every one
of p37's 79 stored dong-level values -- which came out of `p26_matrix.stats` --
at max |delta| = 0.0. That assertion is what ties this file's estimator to the
one in the paper rather than merely making it consistent with it.

WHY p19 IS PARSED AND NOT IMPORTED. Anchor A5 requires the three-band map to be
p19's and to fire if p19's edges ever move. This file reads `LIM_BANDS` out of
p19's source with `ast`, which keeps exactly that coupling -- a changed literal
in p19 fails here -- and depends on nothing except the literal itself.
`p19_bandwidth.py` is import-safe: since 2026-08-28 its whole phase sits behind
`if __name__ == "__main__"`, so `from p19_bandwidth import LIM_BANDS` executes
nothing and touches no results file. The parse is kept anyway, because it is the
narrower dependency. Importing would load duckdb, matplotlib and pandas for one
dict, and would tie A5 to p19 remaining importable rather than to the partition
p19 publishes.

    python eda/p60_agescale.py

No rng anywhere; the run is byte-deterministic by construction.
"""
import ast
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AGES, AGE_LABEL
from p27_survey import assortativity
from p41_semester import CONTROL_YEAR, TERM, VAC, contrast
from paths import ROOT

EDA = os.path.dirname(os.path.abspath(__file__))
PUBLISHED_MONTHS = [202001, 202012, 202312, 202402, 202512, 202606]
NB = len(AGES)                                   # 16

# ---------------------------------------------------------------------------
# THE DECLARATION.  Authored before any line of analysis code below it and
# before any number was produced.  CLAUDE.md: "Declare the primary cell before
# the run, in the script's `declaration` block.  Choosing after the fact is
# number-shopping."  It is stored verbatim in results_p60.json so a reader sees
# the prediction preceded the result.
# ---------------------------------------------------------------------------
DECLARATION = """\
source        results_p37.json matrices_we_dong (79 x 16 x 16, WE, dong, default day set).
              NOT rebuilt. results_p37.json is not touched and is sha256-pinned.
estimator     p26_matrix.stats's assortativity formula applied to S A S', where S is the
              n_bin x 16 aggregation matrix. Imported/asserted, not retyped.
two arms, both declared before the run:
  "lim"       the externally motivated partition, p19_bandwidth.py:86-90 LIM_BANDS, n_bin = 3
              (0-19 / 20-59 / 60+). This is the arm Claim 2's sentence uses.
  "nested"    a nested dyadic ladder 16 -> 8 -> 4 -> 2 -> 1 (adjacent pairs, then pairs of
              pairs), so every rung is a coarsening of the one above and the curve is a genuine
              ladder rather than five unrelated partitions.
primary       the "lim" 3-bin value at each of the 79 months; the nested ladder is the curve.
predicted     STATED BEFORE THE RUN: 3-bin assortativity is expected to FALL relative to 16-bin
              at every month, because aggregation cannot create between-band structure it has
              erased. A month where it RISES is a finding, not a bug."""

# The three pre-declared branches. Which one the run lands in is decided by the
# numbers, not by the author, and the branch is recorded in the results file.
BRANCHES = {
    "falls_everywhere":
        "r falls at every month: Claim 2 becomes one sentence, the new panel "
        "mirrors (a), and 4(b) is demoted to a supporting bar.",
    "falls_mostly":
        "r falls in most months but not all: report the fraction, list the "
        "exceptions by ym label, and state the mechanism (diagonal absorption).",
    "rises_systematically":
        "r rises systematically: the panel is NOT drawn as a mirror. The finding "
        "is that the two axes are not symmetric -- spatial coarsening destroys, "
        "age coarsening inflates via diagonal absorption -- and that goes in the "
        "letter as a written answer to point 4.",
}

LADDER_K = [16, 8, 4, 2, 1]


def hdr(s):
    print(f"\n=== {s} ===")


# ---------------------------------------------------------------------- input
def load(name):
    return json.load(open(f"{ROOT}/eda/results_{name}.json"))


def lim_bands_from_p19_source():
    """p19's LIM_BANDS literal, read without executing p19 (see the docstring).

    `ast.literal_eval` on the assignment's value, so anything other than a plain
    dict-of-lists-of-ints raises instead of being silently accepted.
    """
    src = open(f"{EDA}/p19_bandwidth.py", encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "LIM_BANDS"
                for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("A5: LIM_BANDS not found in p19_bandwidth.py")


# ----------------------------------------------------------------- estimator
def agg_matrix(groups):
    """S, the n_bin x 16 aggregation matrix of a partition of the 16 bands."""
    seen = sorted(i for g in groups for i in g)
    assert seen == list(range(NB)), "partition is not a partition of the 16 bands"
    S = np.zeros((len(groups), NB))
    for g, idx in enumerate(groups):
        S[g, list(idx)] = 1.0
    return S


def collapse(A, S):
    """S A S' -- the co-arrival matrix of the coarsened age partition."""
    return S @ A @ S.T


def terms(A):
    """(trace of e, sum of squared margins) for e = A / A.sum().

    r = (T - Q) / (1 - Q) with T = tr e and Q = sum_a r_a^2 -- the same two
    quantities p41 decomposes band-wise. They are stored per rung because they
    are the mechanism: coarsening raises BOTH (more pairs fall inside a bin, and
    the margins concentrate), and which of the two rises faster is the whole
    question this phase asks.
    """
    e = A / A.sum()
    a_ = e.sum(1)
    return float(np.trace(e)), float((a_ * a_).sum())


def r_from_terms(T, Q):
    """The second route to r: straight from the two scalars, no matrix algebra.

    Independent of `assortativity`'s path through e; asserted against it on every
    month and every rung, so the collapse operator is checked by two roads.
    """
    return (T - Q) / (1 - Q) if (1 - Q) else float("nan")


def nested_groups(k):
    """The dyadic rung with k groups: k adjacent blocks of 16 // k bands."""
    assert NB % k == 0, "the dyadic ladder needs k to divide 16"
    step = NB // k
    return [list(range(i * step, (i + 1) * step)) for i in range(k)]


def is_coarsening(fine, coarse):
    """True iff every group of `fine` sits inside one group of `coarse`."""
    return all(any(set(f) <= set(c) for c in coarse) for f in fine)


# --------------------------------------------------------------------- output
def summary(vals):
    """The shape of results_p37.json's summary_dong, on whatever is passed."""
    a = np.array([v for v in vals.values()], float)
    pub = np.array([vals[str(m)] for m in PUBLISHED_MONTHS if str(m) in vals],
                   float)
    return dict(n_months=int(a.size), min=float(a.min()), max=float(a.max()),
                median=float(np.median(a)),
                iqr=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
                published_six_min=float(pub.min()),
                published_six_max=float(pub.max()))


def main():
    out = {"declaration": DECLARATION, "declaration_branches": BRANCHES}
    print(DECLARATION)

    p37, p26 = load("p37"), load("p26")
    mats = p37["matrices_we_dong"]
    months = [str(m) for m in p37["months"]]            # ym LABELS, never index
    A_of = {ym: np.array(mats[ym]["A"], float) for ym in months}

    # ------------------------------------------------------------ A5, A6 first
    hdr("60.0 anchors A5 and A6 — the partition and the month set")
    lim = lim_bands_from_p19_source()
    assert lim == {"0-19": [0, 10, 15],
                   "20-59": [20, 25, 30, 35, 40, 45, 50, 55],
                   "60+": [60, 65, 70, 75, 80]}, \
        "A5: p19's LIM_BANDS is not the published 0-19 / 20-59 / 60+ map"
    lim_groups = [[AGES.index(a) for a in lim[b]] for b in ("0-19", "20-59", "60+")]
    assert sorted(i for g in lim_groups for i in g) == list(range(NB)), \
        "A5: LIM_BANDS does not partition the 16 boxes"
    S_lim = agg_matrix(lim_groups)
    print(f"  A5  LIM_BANDS from p19 source (not executed): "
          f"{ {b: [AGE_LABEL[a] for a in lim[b]] for b in lim} }")

    assert set(mats) == set(months) and len(months) == 79, \
        "A6: matrices_we_dong and p37's month list disagree"
    assert len(set(months)) == 79, "A6: duplicate ym labels"
    print(f"  A6  {len(months)} months, keyed by ym label, "
          f"{months[0]}..{months[-1]}, set-equal to p37's month list")

    # ------------------------------------------------- A1, A2 the two endpoints
    hdr("60.1 anchors A1 and A2 — the collapse operator at its two endpoints")
    S_id = agg_matrix(nested_groups(16))
    a1_mat = max(float(np.abs(collapse(A_of[ym], S_id) - A_of[ym]).max())
                 for ym in months)
    a1_r = max(abs(assortativity(collapse(A_of[ym], S_id)) - assortativity(A_of[ym]))
               for ym in months)
    assert a1_mat == 0.0 and a1_r == 0.0, \
        f"A1: the 16-group collapse is not the identity ({a1_mat}, {a1_r})"
    print(f"  A1  identity partition, 79 months: max |S A S' - A| = {a1_mat}, "
          f"max |dr| = {a1_r}")

    # A2 is the Null-1 pattern of p44_beta.py:298-305, transplanted to the age
    # axis, and it does NOT come out as zero -- it comes out as 0/0. With one age
    # bin the collapsed e is exactly [[1.0]], so tr e - sum r_a^2 = 0.0 and
    # 1 - sum r_a^2 = 0.0, both exactly. p44's null is zero because its collapse
    # is over VENUES while the matrix stays 16x16; here the collapse is over the
    # age axis itself, so at k = 1 there is no partition left to be assortative
    # with respect to. The anchor is therefore the two exact zeros, and the
    # reported value at k = 1 is null, not 0.
    S_one = agg_matrix(nested_groups(1))
    a2 = []
    for ym in months:
        T1, Q1 = terms(collapse(A_of[ym], S_one))
        a2.append((T1 - Q1, 1 - Q1))
    a2_num = max(abs(n) for n, _ in a2)
    a2_den = max(abs(d) for _, d in a2)
    assert a2_num == 0.0 and a2_den == 0.0, \
        f"A2: the one-group collapse is not 0/0 ({a2_num}, {a2_den})"
    a2_impl = assortativity(collapse(A_of[months[0]], S_one))
    print(f"  A2  one-group partition, 79 months: max |numerator| = {a2_num}, "
          f"max |denominator| = {a2_den}")
    print(f"      r at k = 1 is 0/0, not 0; the imported implementation returns "
          f"{a2_impl} and the file stores null")

    # ------------------------------------------------------- A3, A4 the 16 bins
    hdr("60.2 anchors A3 and A4 — the 16-bin values are p37's and p26's")
    rows = {(r["ym"], r["panel"], r["level"]): r for r in p37["rows"]}
    r16 = {ym: float(assortativity(A_of[ym])) for ym in months}
    a3 = max(abs(r16[ym] - rows[(int(ym), "WE", "dong")]["assortativity"])
             for ym in months)
    assert a3 == 0.0, f"A3: rebuilt 16-bin values differ from p37 by {a3}"
    print(f"  A3  79 rebuilt 16-bin values vs results_p37.json rows "
          f"(WE, dong): max |delta| = {a3}")

    p26_row = {(r["ym"], r["panel"], r["level"]): r["assortativity"]
               for r in p26["ladder"]}
    a4 = max(abs(r16[str(m)] - p26_row[(m, "WE", "dong")])
             for m in PUBLISHED_MONTHS)
    assert a4 == 0.0, f"A4: the six published months differ from p26 by {a4}"
    print(f"  A4  the six published months vs results_p26.json ladder: "
          f"max |delta| = {a4}")

    out["anchors"] = dict(
        A1=dict(what="16-group collapse is the identity, all 79 months",
                max_abs_matrix_diff=a1_mat, max_abs_r_diff=a1_r, passed=True),
        A2=dict(what="1-group collapse is 0/0, all 79 months",
                max_abs_numerator=a2_num, max_abs_denominator=a2_den,
                implementation_returns=("nan" if np.isnan(a2_impl) else a2_impl),
                reported_as=None, passed=True,
                note="p44's Null-1 gives exactly 0 because it collapses venues "
                     "with the matrix still 16x16; collapsing the age axis to "
                     "one bin leaves no partition, so r is 0/0 and is stored as "
                     "null rather than forced to 0"),
        A3=dict(what="79 rebuilt 16-bin values vs p37 rows WE|dong",
                max_abs_diff=a3, n=len(months), passed=True),
        A4=dict(what="six published months vs p26 ladder WE|dong",
                max_abs_diff=a4, n=len(PUBLISHED_MONTHS), passed=True),
        A5=dict(what="band map equals p19_bandwidth.py LIM_BANDS",
                bands=lim, read_by="ast.literal_eval on p19's source; p19 is NOT "
                                   "imported, so this anchor rests on p19's "
                                   "LIM_BANDS literal alone and not on p19 "
                                   "being importable",
                passed=True),
        A6=dict(what="month set equals p37's, keyed by ym label",
                n_months=len(months), first=months[0], last=months[-1],
                passed=True))

    # ------------------------------------------------------- 60.3 the two arms
    hdr("60.3 the two declared arms")
    per = {}
    dev_second_route = 0.0
    for ym in months:
        A = A_of[ym]
        rec = {"r16": r16[ym]}

        Tl, Ql = terms(collapse(A, S_lim))
        rl = float(assortativity(collapse(A, S_lim)))
        dev_second_route = max(dev_second_route, abs(rl - r_from_terms(Tl, Ql)))
        rec["lim"] = dict(n_bin=3, r=rl, trace_e=Tl, sum_margin_sq=Ql)
        rec["retained_frac"] = rl / r16[ym]

        lad = {}
        for k in LADDER_K:
            Ak = collapse(A, agg_matrix(nested_groups(k)))
            Tk, Qk = terms(Ak)
            if k == 1:
                lad[str(k)] = dict(r=None, trace_e=Tk, sum_margin_sq=Qk,
                                   undefined="0/0")
                continue
            rk = float(assortativity(Ak))
            dev_second_route = max(dev_second_route, abs(rk - r_from_terms(Tk, Qk)))
            lad[str(k)] = dict(r=rk, trace_e=Tk, sum_margin_sq=Qk)
        rec["nested"] = lad
        per[ym] = rec

    # The second route is written independently of `assortativity`: it never
    # forms e's off-diagonal at all, only the two scalars. It is not a substitute
    # for p36 (which owes this phase a section), but it does mean the collapse
    # operator is checked by two roads inside this file.
    assert dev_second_route < 1e-12, \
        f"second route disagrees with the imported estimator by {dev_second_route}"
    print(f"  second route (r straight from tr e and sum r_a^2) agrees with the "
          f"imported estimator to {dev_second_route:.3e}")

    # nestedness of the ladder is a property of the partitions, asserted rather
    # than assumed, so "every rung is a coarsening of the one above" is checked.
    for fine, coarse in zip(LADDER_K, LADDER_K[1:]):
        assert is_coarsening(nested_groups(fine), nested_groups(coarse)), \
            f"the dyadic ladder is not nested at {fine} -> {coarse}"
    print(f"  ladder {LADDER_K} is nested: every rung is a coarsening of the one "
          f"above")

    # --------------------------------------------- 60.4 which branch we land in
    hdr("60.4 the pre-declared prediction: does 3-bin r fall at every month?")
    against = sorted(ym for ym in months if per[ym]["lim"]["r"] >= r16[ym])
    n_against = len(against)
    if n_against == 0:
        branch = "falls_everywhere"
    elif n_against == len(months):
        branch = "rises_systematically"
    elif n_against >= len(months) / 2:
        branch = "rises_systematically"
    else:
        branch = "falls_mostly"
    kept = {ym: per[ym]["retained_frac"] for ym in months}
    ka = np.array(list(kept.values()), float)
    print(f"  months where 3-bin r >= 16-bin r: {n_against} / {len(months)}")
    print(f"  retained fraction r3 / r16: median {np.median(ka):.4f}, "
          f"min {ka.min():.4f}, max {ka.max():.4f}")
    print(f"  BRANCH: {branch} — {BRANCHES[branch]}")
    if against:
        print(f"  months moving against the prediction ({n_against}): "
              f"{', '.join(against[:12])}{' ...' if n_against > 12 else ''}")

    # The mechanism, in the two scalars, on the month with the median 16-bin r.
    med_ym = sorted(months, key=lambda y: r16[y])[len(months) // 2]
    m = per[med_ym]
    print(f"  mechanism at the median month {med_ym}: "
          f"tr e {m['nested']['16']['trace_e']:.4f} -> {m['lim']['trace_e']:.4f}, "
          f"sum r_a^2 {m['nested']['16']['sum_margin_sq']:.4f} -> "
          f"{m['lim']['sum_margin_sq']:.4f}, "
          f"r {m['r16']:.5f} -> {m['lim']['r']:.5f}")

    # ladder monotonicity, on the rungs where r is defined. Both directions are
    # counted because the declaration predicted one and the run gave the other,
    # and a count of the predicted direction that is simply absent from the file
    # would let a later reader think it was never asked.
    defined = [k for k in LADDER_K if k != 1]
    n_down = sum(all(per[ym]["nested"][str(a)]["r"] >= per[ym]["nested"][str(b)]["r"]
                     for a, b in zip(defined, defined[1:])) for ym in months)
    n_up = sum(all(per[ym]["nested"][str(a)]["r"] <= per[ym]["nested"][str(b)]["r"]
                   for a, b in zip(defined, defined[1:])) for ym in months)
    print(f"  nested ladder, 16 -> 8 -> 4 -> 2: monotone DOWN in "
          f"{n_down} / {len(months)} months (predicted), monotone UP in "
          f"{n_up} / {len(months)}")
    for k in defined:
        v = np.array([per[ym]["nested"][str(k)]["r"] for ym in months], float)
        print(f"    k = {k:>2}: median r {np.median(v):.5f}  "
              f"[{v.min():.5f}, {v.max():.5f}]")

    out["months"] = months
    out["per_month"] = per
    out["prediction"] = dict(
        text="3-bin assortativity FALLS relative to 16-bin at every month",
        declared_before_run=True,
        n_months=len(months), n_against=n_against, months_against=against,
        frac_against=n_against / len(months),
        branch=branch, branch_text=BRANCHES[branch],
        held=(n_against == 0))
    out["summary"] = dict(
        r16=summary({ym: per[ym]["r16"] for ym in months}),
        lim3=summary({ym: per[ym]["lim"]["r"] for ym in months}),
        retained_frac=summary(kept),
        nested={str(k): summary({ym: per[ym]["nested"][str(k)]["r"]
                                 for ym in months})
                for k in LADDER_K if k != 1},
        nested_k1="undefined (0/0): one bin leaves no partition",
        ladder_monotone=dict(n_monotone_down=n_down, n_monotone_up=n_up,
                             n_months=len(months), rungs=defined,
                             predicted_direction="down"),
        retained_frac_note="this ratio is ABOVE 1 at every month, so it is an "
                           "inflation factor and not a retention; it is kept "
                           "under the declared name so the prediction and its "
                           "failure are readable in the same field",
        second_route_max_dev=dev_second_route,
        mechanism_median_month=dict(
            ym=med_ym,
            trace_e_16=m["nested"]["16"]["trace_e"], trace_e_3=m["lim"]["trace_e"],
            sum_margin_sq_16=m["nested"]["16"]["sum_margin_sq"],
            sum_margin_sq_3=m["lim"]["sum_margin_sq"],
            r16=m["r16"], r3=m["lim"]["r"]))

    # ------------------------------------------- 60.5 the asymmetry, in one table
    # The whole point of the phase is a comparison, so the spatial side is put in
    # this file too rather than left for a reader to fetch from p37. It is
    # recomputed from p37's own rows and the floored version is asserted against
    # p37's stored ladder.kept_median, so the two files cannot disagree about the
    # spatial number this phase compares against.
    hdr("60.5 the asymmetry: the same estimator, coarsened on each axis")
    sp = {}
    for lev in ("gu", "city"):
        v = {ym: rows[(int(ym), "WE", lev)]["assortativity"] /
                 rows[(int(ym), "WE", "dong")]["assortativity"] for ym in months}
        sp[f"{lev}_over_dong"] = summary(v)
    floored = {ym: ((rows[(int(ym), "WE", "gu")]["assortativity"]
                     - rows[(int(ym), "WE", "city")]["assortativity"]) /
                    (rows[(int(ym), "WE", "dong")]["assortativity"]
                     - rows[(int(ym), "WE", "city")]["assortativity"]))
               for ym in months}
    fl = summary(floored)
    assert fl["median"] == p37["ladder"]["kept_median"], \
        "the spatial kept fraction does not reproduce p37's ladder.kept_median"
    sp["gu_over_dong_floored_at_city"] = fl
    sp["floored_reproduces_p37_kept_median"] = True
    print(f"  spatial  gu / dong  : median {sp['gu_over_dong']['median']:.4f}  "
          f"[{sp['gu_over_dong']['min']:.4f}, {sp['gu_over_dong']['max']:.4f}]"
          f"   (17x coarser)")
    print(f"  spatial  city / dong: median {sp['city_over_dong']['median']:.4f}  "
          f"[{sp['city_over_dong']['min']:.4f}, {sp['city_over_dong']['max']:.4f}]"
          f"   (424x coarser)")
    print(f"  spatial  p37's floored kept fraction reproduced exactly: "
          f"{fl['median']}")
    print(f"  age      3 bins / 16 : median {np.median(ka):.4f}  "
          f"[{ka.min():.4f}, {ka.max():.4f}]   (5.3x coarser)")
    print("  -> coarsening the SPATIAL axis destroys, coarsening the AGE axis "
          "inflates. The two axes are NOT the same sentence.")
    out["axis_asymmetry"] = dict(
        spatial=sp, age_3_over_16=summary(kept),
        age_nested_over_16={str(k): summary({ym: per[ym]["nested"][str(k)]["r"]
                                             / per[ym]["r16"] for ym in months})
                            for k in defined if k != 16},
        statement="the same estimator on the same 79 matrices: spatial "
                  "coarsening lowers r at every month, age coarsening raises it "
                  "at every month; the axes are not symmetric and Figure 4 must "
                  "not draw them as a mirror",
        why="space is marginalised out (locations are re-paired, the age margins "
            "do not move), age is re-partitioned (the margins themselves change "
            "and cross-bin excess becomes within-bin)")

    # ------------------------------------------------------- 60.6 the mechanism
    # The declaration predicted a fall and the run gave a rise at every month, so
    # the first question a reviewer asks is whether coarsening MANUFACTURES
    # assortativity. It does not, and the null proves it: if mixing were exactly
    # proportionate, e_ab = r_a r_b, then the collapsed matrix has
    # e_gh = r_g r_h as well, so T = Q and r = 0 at EVERY rung. The rise is
    # therefore not an artefact of the operator -- it is real excess that a coarse
    # partition relabels.
    #
    # Where the relabelled excess comes from is an exact identity:
    #
    #     T_k - Q_k = sum_g sum_{a,b in g} (e_ab - r_a r_b)
    #
    # At 16 bins only the 16 diagonal terms are counted; at 3 bins the entire
    # 8x8 within-20-59 block of near-diagonal excess is counted too. So
    #
    #     r_3 = (N_16 + absorbed) / (1 - Q_3),   r_16 = N_16 / (1 - Q_16)
    #
    # and the ratio factorises into a numerator gain and a denominator shrink,
    # both stored per month. That factorisation is the written answer to the
    # advisor's point 4.
    hdr("60.6 the mechanism — the proportionate-mixing null and the identity")
    pm_worst = 0.0
    ident_worst = 0.0
    for ym in months:
        A = A_of[ym]
        e = A / A.sum()
        rm = e.sum(1)
        E_pm = np.outer(rm, rm) * A.sum()
        for S in [S_lim] + [agg_matrix(nested_groups(k)) for k in defined]:
            pm_worst = max(pm_worst, abs(float(assortativity(collapse(E_pm, S)))))
            g_of = {i: g for g, row in enumerate(S) for i in np.nonzero(row)[0]}
            same = np.array([[g_of[i] == g_of[j] for j in range(NB)]
                             for i in range(NB)])
            lhs = float(((e - np.outer(rm, rm)) * same).sum())
            T, Q = terms(collapse(A, S))
            ident_worst = max(ident_worst, abs(lhs - (T - Q)))
    assert pm_worst < 1e-12, \
        f"proportionate mixing does not give r = 0 after collapse ({pm_worst})"
    assert ident_worst < 1e-12, \
        f"the within-group excess identity fails by {ident_worst}"
    print(f"  proportionate-mixing null, 79 months x 5 rungs: max |r| = "
          f"{pm_worst:.3e} — collapsing does NOT manufacture assortativity")
    print(f"  identity  T_k - Q_k = sum_g sum_{{a,b in g}} (e_ab - r_a r_b): "
          f"max |delta| = {ident_worst:.3e}")

    gain, shrink = {}, {}
    for ym in months:
        N16 = per[ym]["nested"]["16"]["trace_e"] - per[ym]["nested"]["16"]["sum_margin_sq"]
        N3 = per[ym]["lim"]["trace_e"] - per[ym]["lim"]["sum_margin_sq"]
        gain[ym] = N3 / N16
        shrink[ym] = (1 - per[ym]["nested"]["16"]["sum_margin_sq"]) / \
                     (1 - per[ym]["lim"]["sum_margin_sq"])
        per[ym]["numerator_gain"] = gain[ym]
        per[ym]["denominator_shrink"] = shrink[ym]
        per[ym]["absorbed_offdiag_excess"] = N3 - N16
    ga = np.array(list(gain.values()), float)
    sa = np.array(list(shrink.values()), float)
    print(f"  numerator gain  N_3 / N_16   : median {np.median(ga):.4f} "
          f"[{ga.min():.4f}, {ga.max():.4f}]")
    print(f"  denominator shrink (1-Q16)/(1-Q3): median {np.median(sa):.4f} "
          f"[{sa.min():.4f}, {sa.max():.4f}]")
    print(f"  product = the ratio above: median "
          f"{np.median(ga * sa):.4f} vs measured {np.median(ka):.4f}")
    assert max(abs(gain[ym] * shrink[ym] - kept[ym]) for ym in months) < 1e-12, \
        "the two-factor decomposition does not reproduce the measured ratio"

    out["mechanism"] = dict(
        proportionate_mixing_null_max_abs_r=pm_worst,
        proportionate_mixing_null_what="e_ab = r_a r_b collapses to e_gh = r_g "
                                       "r_h, so r = 0 at every rung; the "
                                       "measured rise is relabelled real "
                                       "excess, not an artefact of the operator",
        within_group_excess_identity_max_abs_diff=ident_worst,
        numerator_gain=summary(gain), denominator_shrink=summary(shrink),
        factorisation="r3 / r16 = (N_3 / N_16) x ((1 - Q_16) / (1 - Q_3)); both "
                      "factors exceed 1, so the two effects push the same way")

    # ------------------------------------- 60.7 what the coarse number costs
    # NOT DECLARED IN ADVANCE, and labelled as such in the file. It is reported
    # because landing in the "rises systematically" branch obliges a mechanism,
    # and because the sharpest consequence of that branch is not the level at
    # all: p41's whole external-validity argument is that the term-minus-vacation
    # contrast is positive in 6 of the 7 years and the single exception is 2020,
    # the year schools did not open in March -- the signal appears where it
    # should and vanishes where it should. Re-running that contrast on the 3-bin
    # series asks whether the coarse instrument still has the property that makes
    # the fine one credible. p41's TERM / VAC sets and its `contrast` are
    # imported, not restated, so the two files cannot drift apart.
    hdr("60.7 the semester contrast on the coarse series (post hoc, undeclared)")
    sem = {}
    for tag, ser in (("r16", {ym: per[ym]["r16"] for ym in months}),
                     ("lim3", {ym: per[ym]["lim"]["r"] for ym in months})):
        by_year = {}
        for ym, v in ser.items():
            by_year.setdefault(int(ym[:4]), {})[int(ym[4:])] = v
        d = {y: contrast(mv) for y, mv in sorted(by_year.items())}
        d = {y: v for y, v in d.items() if not np.isnan(v)}
        sem[tag] = dict(by_year={str(y): float(v) for y, v in d.items()},
                        n_positive=int(sum(v > 0 for v in d.values())),
                        n_years=len(d),
                        control_year=CONTROL_YEAR,
                        control_year_value=float(d[CONTROL_YEAR]),
                        control_year_is_the_exception=bool(
                            d[CONTROL_YEAR] < 0
                            and all(v > 0 for y, v in d.items()
                                    if y != CONTROL_YEAR)))
        print(f"  {tag:>5}: {sem[tag]['n_positive']} / {sem[tag]['n_years']} "
              f"years positive, {CONTROL_YEAR} = "
              f"{sem[tag]['control_year_value']:+.5f}, control-year exception "
              f"{'HOLDS' if sem[tag]['control_year_is_the_exception'] else 'GONE'}")
    out["semester_posthoc"] = dict(
        declared_before_run=False,
        why="branch 'rises systematically' obliges a mechanism; the level is not "
            "the only thing the coarse partition changes",
        term=list(TERM), vacation=list(VAC), convention="p41_semester.contrast, "
                                                       "imported not restated",
        **sem)

    # -------------------------------------------------------- 60.8 cross-check
    # results_p19.json .range_retained_frac = 0.506 is a DIFFERENT quantity on a
    # DIFFERENT object: the fraction of the 16-band E-share RANGE that survives
    # aggregation to three bands, measured on two months (202003, 202012). It is
    # stored beside this phase's retained fraction so the two are never quoted as
    # if they were the same number, and never averaged.
    p19 = load("p19")
    out["cross_check"] = dict(
        p19_range_retained_frac=p19["range_retained_frac"],
        p19_quantity="fraction of the 16-band E-share RANGE retained at 3 bands, "
                     "2 months (202003 vs 202012), a composition not a mixing "
                     "statistic",
        p60_quantity="ratio of 3-bin to 16-bin Newman assortativity on the WE "
                     "dong co-arrival matrix, 79 months",
        comparable="no -- same partition, different estimator and different "
                   "object; report side by side, never as one number")
    hdr("60.8 cross-check")
    print(f"  p19 range_retained_frac = {p19['range_retained_frac']} "
          f"(E-share range, 2 months) — a different quantity, not comparable")
    print(f"  p60 retained_frac median = {np.median(ka):.4f} "
          f"(assortativity ratio, 79 months)")

    with open(f"{ROOT}/eda/results_p60.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, sort_keys=True)
    print("\nwrote results_p60.json")


if __name__ == "__main__":
    main()
