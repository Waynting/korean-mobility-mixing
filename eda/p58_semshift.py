#!/usr/bin/env python
"""Phase 58 -- the semester p-value under a circular-shift null.

WHY THIS EXISTS. p54 re-counted the 79-month passive series against p51's
corrected floor and found 17 months at or above it, all 17 of them Korean
school-term months, with an exact within-year p of 4.97e-06. That p comes from
`p42_monthscope.within_year_null`, whose own docstring states the assumption in
one sentence: "Within one year the labels are exchangeable across that year's
months." The advisor's 2026-08-27 letter, point 2, rejects that null and he is
right about the premise. Semesters are contiguous blocks, not labels scattered
at random over 79 months, and the 17 clearing months are blocky too -- they sit
in 9 runs, not 17 independent draws. Exchangeable month labels therefore
overstate the effective sample size and the p is inflated.

WHAT THIS FILE DOES. It replaces the null, not the statistic and not the rule.
A month still clears iff its holiday-free mi_bits is at or above p51's corrected
floor -- p42's rule, p54's floor, both unchanged. What changes is what "by
chance" means: instead of permuting month labels, the whole school calendar is
circularly shifted along the 79-month series by k = 1..78 months, and the
clearing months are re-counted against the shifted labelling each time. Both
block structures survive: the calendar stays a run of contiguous semesters and
the clearing months stay where they are.

WHAT IT DOES NOT DO. A rotation does not hold per-year margins. within_year_null
is within-year because the level of this series trends by a factor of 1.70 from
2020 to 2026, and a null that mixes years reads that trend as a cycle. Across
the 79 rotations the per-year term count runs 3..8 against 7,7,7,7,7,7,4 at
k = 0, so the rotation null does NOT control the trend the within-year null
exists to control. Neither null dominates. Both are reported with what each
assumes, and the within-year p is re-derived here (58.0 A3) rather than
discarded -- a superseded number nobody can check is worse than the original
error.

NOTHING IS REBUILT AND NOTHING FROZEN IS REWRITTEN. The only inputs are
results_p42.json (the 79 readings and their true semester labels),
results_p51.json (the floor) and results_p54.json (the anchor). No parquet, no
matrices, no drive. The only output is results_p58.json.

FULLY DETERMINISTIC. There is no rng: 78 shifts is an exhaustive enumeration.

    python eda/p58_semshift.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from p41_semester import TERM, VAC              # (3,4,5,6,9,10,11) / (1,2,7,8)
from p42_monthscope import label, within_year_null
from paths import ROOT

# ==========================================================================
# THE DECLARATION.  Authored before any line of analysis code below it and not
# edited afterwards.  CLAUDE.md: "Declare the primary cell before the run, in
# the script's `declaration` block.  Choosing after the fact is number-
# shopping."  The resolution floor in particular is declared in advance,
# because a test over 79 rotations cannot produce a p below 1/79 no matter what
# the data say, and a reader must be told that before being shown the number.
# ==========================================================================
DECLARATION = """\
primary statistic  n_term(k) = |{clearing months whose label under rotation k is "term"}|,
                   clearing = mi_bits_hf >= p51 corrected floor; observed n_term(0) = 17 of 17
shift set          ALL k in 1..78. No exclusions. Declared in full.
p_shift            (1 + #{k : n_term(k) >= 17}) / 79
resolution floor   p_shift >= 1/79 = 0.0126582...  DECLARED IN ADVANCE. A p of 0.0127 means
                   "the smallest this test can produce", not a measurement of 0.0127.
secondary (a)      the same p over the 66 shifts with k mod 6 != 0, because the 12 shifts at
                   k in {6,12,...,72} agree with the true labelling on 59-74 of 79 months
                   (chance is 36.5) and are the alternative displaced by the 79 = 6*12+7 seam.
                   REPORTED WHICHEVER WAY IT LANDS; the ALL-78 version stays primary because
                   it is the conservative one and excluding shifts after seeing them is shopping.
secondary (b)      D(k) = median(mi_bits_hf | term_k) - median(mi_bits_hf | vacation_k), p41's
                   own contrast, rank of D(0) among the 79 rotations. Declared because the count
                   statistic is censored at 17 and cannot resolve below 1/79.
what is NOT claimed  the rotation null does not hold per-year margins (they run 3..8 across the
                   79 rotations against 7,7,7,7,7,7,4 at k=0), so it does not control the 1.70x
                   level trend that within_year_null exists to control. Both nulls are reported
                   with what each assumes. Neither dominates.
"""

# The declaration's secondary (a) carries an internal arithmetic inconsistency
# that is visible without running anything and is therefore recorded here
# rather than silently resolved: "k mod 6 != 0" over k in 1..78 excludes 13
# multiples of six (6,12,...,72,78), leaving 65 shifts, but the parenthetical
# names only the 12 shifts in {6,12,...,72}, leaving 66. The declaration is left
# exactly as written and BOTH readings are computed and both reported, so the
# discrepancy cannot become a choice made after seeing a number.
DECLARATION_NOTE = (
    "secondary (a) as written is arithmetically inconsistent: 'the 66 shifts "
    "with k mod 6 != 0' and 'the 12 shifts at k in {6,12,...,72}' cannot both "
    "hold, because 78 is also a multiple of 6, so k mod 6 != 0 leaves 65 "
    "shifts and dropping only {6..72} leaves 66. The declaration is NOT "
    "edited; both readings are computed (a_mod6 and a_drop12) and both are "
    "reported regardless of how either lands.")

BRANCH_TABLE = [
    dict(condition="p_shift <= 0.05",
         framing="Circular shift becomes the primary p. Within-year 4.97e-06 "
                 "demoted to secondary with its assumption stated and the "
                 "reason it is inflated. Claim 1 keeps its corroboration."),
    dict(condition="0.05 < p_shift <= 0.20",
         framing="Demoted from a p-value to a descriptive alignment: 'all 17 "
                 "clearing months are term months; k of the 78 alternative "
                 "calendar alignments match that.' Leaves the abstract and "
                 "Figure 1's claim box; stays in the semester section as "
                 "supporting structure."),
    dict(condition="p_shift > 0.20",
         framing="Reported as a null result. 4.97e-06 is retracted in the "
                 "08-27 letter with the reason. archive/section_semester_draft.md "
                 "retitled from a finding to a robustness note."),
]

out = {"declaration": {"text": DECLARATION, "note": DECLARATION_NOTE,
                       "branch_table": BRANCH_TABLE}}


def say(s=""):
    print(s, flush=True)


def main():
    # ==================================================================== inputs
    p42 = json.load(open(f"{ROOT}/eda/results_p42.json"))
    p51 = json.load(open(f"{ROOT}/eda/results_p51.json"))
    p54 = json.load(open(f"{ROOT}/eda/results_p54.json"))

    rows = p42["rows"]
    N = len(rows)
    FLOOR = p51["cells"]["202312|national"]["mi_perm_median"]

    # Everything downstream is keyed by the ym LABEL. The rotation needs a series
    # order, so the order is built once, asserted strictly increasing, and the
    # position of a month is only ever reached through this map -- never by
    # assuming rows[] is already in order. CLAUDE.md: "Never index months
    # positionally."
    YMS = sorted(r["ym"] for r in rows)
    POS = {ym: i for i, ym in enumerate(YMS)}
    MI = {r["ym"]: r["mi_bits_hf"] for r in rows}
    TRUE = {r["ym"]: r["semester"] for r in rows}
    YEAR = {r["ym"]: r["year"] for r in rows}

    anchors = {}

    # ==================================================================== 58.0
    say("=== 58.0 anchors: the run aborts on any failure ===")

    # ------------------------------------------------------------------- A1
    strictly_increasing = all(YMS[i] < YMS[i + 1] for i in range(N - 1))
    assert N == 79, f"expected 79 months, got {N}"
    assert strictly_increasing, "the month series is not strictly increasing"
    assert len(MI) == N and len(TRUE) == N, "duplicate ym in results_p42.json rows"
    say(f"  A1  n_months {N}, strictly increasing {strictly_increasing}, "
        f"{YMS[0]}..{YMS[-1]}")
    anchors["A1"] = dict(n_months=N, strictly_increasing=strictly_increasing,
                         first_ym=YMS[0], last_ym=YMS[-1], unique_yms=len(MI))

    # ------------------------------------------------------------------- A7
    # The float this file reads must be the float p54 read, bit for bit. p54 stores
    # it, so the comparison is exact and not a re-derivation.
    floor_p54 = p54["corrected"]["floor"]
    assert FLOOR == floor_p54, "the floor is not the one p54 counted against"
    say(f"  A7  floor {FLOOR!r} bit-equal to the one p54 read: "
        f"{FLOOR == floor_p54}")
    anchors["A7"] = dict(floor=FLOOR, floor_p54=floor_p54,
                         bit_equal=bool(FLOOR == floor_p54),
                         source="results_p51.json cells['202312|national']"
                                ".mi_perm_median")

    # ------------------------------------------------------------------- A4
    # p42's own partition, imported, applied to every ym. If label() and the stored
    # semester ever disagreed, the rotation would be rotating a different labelling
    # than the one 17-of-17 was counted under.
    assert TERM == (3, 4, 5, 6, 9, 10, 11), f"TERM moved: {TERM}"
    assert VAC == (1, 2, 7, 8), f"VAC moved: {VAC}"
    mismatch = [ym for ym in YMS if label(ym) != TRUE[ym]]
    assert not mismatch, f"label() disagrees with rows[].semester on {mismatch}"
    counts_true = {lab: sum(1 for ym in YMS if TRUE[ym] == lab)
                   for lab in ("term", "vacation", "december")}
    say(f"  A4  label() reproduces rows[].semester on {N}/{N} months; "
        f"{counts_true}")
    anchors["A4"] = dict(n_checked=N, n_mismatch=len(mismatch),
                         term=TERM, vacation=VAC, label_counts=counts_true)

    # ------------------------------------------------------------------- A2
    # The recount at k = 0, against the corrected floor, must reproduce p54's
    # corrected block exactly -- count, split, and the identical 17-ym list.
    clearing = [ym for ym in YMS if MI[ym] >= FLOOR]
    obs0 = {lab: sum(1 for ym in clearing if TRUE[ym] == lab)
            for lab in ("term", "vacation", "december")}
    want = p54["corrected"]
    assert len(clearing) == want["n_at_or_above"], "count differs from p54"
    assert obs0 == want["observed"], "label split differs from p54"
    assert clearing == sorted(want["months"]), "a different set of months clears"
    N_OBS = obs0["term"]
    say(f"  A2  k=0 recount: n {len(clearing)} (p54 {want['n_at_or_above']}), "
        f"term {obs0['term']}, vacation {obs0['vacation']}, "
        f"december {obs0['december']}, months identical: True")
    anchors["A2"] = dict(n_at_or_above=len(clearing), observed=obs0,
                         months=clearing, reproduces_p54=True)

    # ------------------------------------------------------------------- A3
    # Re-derive p54's within-year p through the imported null, from this file's own
    # per-year tally. This proves p58 applies p42's rule and p54's floor before it
    # changes the null, and it keeps the superseded number checkable.
    years = sorted({YEAR[ym] for ym in YMS})
    per_year = {}
    for y in years:
        ys = [ym for ym in YMS if YEAR[ym] == y]
        per_year[y] = (len(ys), sum(1 for ym in ys if TRUE[ym] == "term"),
                       sum(1 for ym in ys if MI[ym] >= FLOOR))
    _, p_within, _ = within_year_null(per_year, N_OBS)
    want_p = want["p_term_ge"]
    assert abs(p_within - want_p) < 1e-15, "the within-year null does not reproduce"
    say(f"  A3  within-year p re-derived {p_within:.15e}")
    say(f"      p54 says              {want_p:.15e}   |diff| "
        f"{abs(p_within - want_p):.1e}")
    anchors["A3"] = dict(p_within_year=p_within, p_p54=want_p,
                         abs_diff=abs(p_within - want_p),
                         per_year={str(y): list(per_year[y]) for y in years})
    out["within_year"] = dict(
        p_term_ge=p_within, source="p42_monthscope.within_year_null, imported",
        assumption="within one year the month labels are exchangeable across that "
                   "year's months; the years are independent, so the total is the "
                   "convolution of per-year hypergeometrics",
        why_it_is_inflated="semesters are contiguous blocks and the 17 clearing "
                           "months are blocky too, so the effective sample size is "
                           "below 17 and an exchangeable-label null overstates it",
        what_it_controls_that_rotation_does_not="the per-year margins are held "
                                                "fixed, so the 1.70x level trend "
                                                "from 2020 to 2026 cannot be read "
                                                "as a cycle")

    # ==================================================================== 58.1
    say("\n=== 58.1 the rotation, and the two structures it preserves ===")


    def rotate(k):
        """Label vector under a circular shift of the school calendar by k months.

    The month at series position i receives the label that truly belongs to the
    month k positions earlier, wrapping at the ends. Every rotation is a
    permutation of the same 79 labels, so the term margin is fixed at 46 by
    construction: this is a fixed-margin randomisation test, not a resampling.
    Returned keyed by ym, so no caller ever indexes a month positionally.
    """
        return {YMS[i]: TRUE[YMS[(i - k) % N]] for i in range(N)}


    LAB = [rotate(k) for k in range(N)]

    # ------------------------------------------------------------------- A5
    term_margin = [sum(1 for ym in YMS if LAB[k][ym] == "term") for k in range(N)]
    assert set(term_margin) == {counts_true["term"]}, (
        f"the term margin is not invariant: {sorted(set(term_margin))}")
    say(f"  A5  term margin under all {N} rotations: "
        f"{sorted(set(term_margin))} -- invariant, so the margin is fixed by "
        f"construction")
    anchors["A5"] = dict(distinct_term_margins=sorted(set(term_margin)),
                         invariant=True, margin=counts_true["term"])

    # ------------------------------------------------------------------- A6
    # The degeneracy audit. 79 is prime, so every rotation is a distinct
    # permutation; but "distinct permutation" is not "independent draw", and the
    # agreement profile is what shows it.
    is_prime = N > 1 and all(N % d for d in range(2, int(N ** 0.5) + 1))
    agreement = [sum(1 for ym in YMS if LAB[k][ym] == TRUE[ym]) for k in range(N)]
    vectors = {tuple(LAB[k][ym] for ym in YMS) for k in range(N)}
    sym_ok = all(agreement[k] == agreement[(N - k) % N] for k in range(N))
    chance = sum(v * v for v in counts_true.values()) / N
    assert is_prime, "79 is not prime -- the distinctness argument fails"
    assert len(vectors) == N, f"only {len(vectors)} distinct label vectors, not {N}"
    assert sym_ok, "agreement(k) != agreement(79-k); the rotation is not circular"
    assert agreement[0] == N, f"agreement(0) is {agreement[0]}, not {N}"
    assert agreement[12] == 74, f"agreement(12) is {agreement[12]}, not 74"
    top = sorted(range(1, N), key=lambda k: -agreement[k])[:14]
    say(f"  A6  {N} prime: {is_prime}; distinct label vectors {len(vectors)}/{N}; "
        f"agreement(k)==agreement(79-k): {sym_ok}")
    say(f"      agreement(0) {agreement[0]}, agreement(12) {agreement[12]}, "
        f"chance {chance:.3f}")
    say("      most-agreeing shifts: "
        + " ".join(f"k={k}:{agreement[k]}" for k in top))
    anchors["A6"] = dict(n_is_prime=is_prime, n_distinct_label_vectors=len(vectors),
                         symmetric_k_and_n_minus_k=sym_ok,
                         agreement_0=agreement[0], agreement_12=agreement[12],
                         chance_agreement=chance)
    out["degeneracy"] = dict(
        agreement_by_k=agreement, chance_agreement=chance,
        top_14_shifts_by_agreement={str(k): agreement[k] for k in top},
        mod6_zero_shifts={str(k): agreement[k] for k in range(6, N, 6)},
        note="79 = 6*12 + 7, so a shift that is a multiple of 6 lands the calendar "
             "nearly back on itself; those shifts are the alternative displaced by "
             "the seam, not draws from the null")

    # the block structure the advisor's premise turns on
    runs, cur = [], [clearing[0]]
    for ym in clearing[1:]:
        if POS[ym] == POS[cur[-1]] + 1:
            cur.append(ym)
        else:
            runs.append(cur)
            cur = [ym]
    runs.append(cur)
    say(f"\n  the {len(clearing)} clearing months sit in {len(runs)} runs, lengths "
        f"{[len(r) for r in runs]}")
    out["run_structure"] = dict(
        n_clearing=len(clearing), n_runs=len(runs),
        run_lengths=[len(r) for r in runs], runs=runs,
        effective_sample_note="the advisor's premise: 17 clearing months in "
                              f"{len(runs)} contiguous runs, so the effective "
                              "sample is nearer the number of runs than 17")

    # what the rotation gives up: the per-year margins move
    per_year_term = {k: {y: sum(1 for ym in YMS
                                if YEAR[ym] == y and LAB[k][ym] == "term")
                         for y in years} for k in range(N)}
    py_min = {y: min(per_year_term[k][y] for k in range(N)) for y in years}
    py_max = {y: max(per_year_term[k][y] for k in range(N)) for y in years}
    say(f"  per-year term counts at k=0: "
        f"{[per_year_term[0][y] for y in years]}")
    say(f"  across all {N} rotations they range "
        f"{min(py_min.values())}..{max(py_max.values())} -- the rotation null does "
        f"NOT hold them fixed")
    out["per_year_margins"] = dict(
        years=[str(y) for y in years],
        identity={str(y): per_year_term[0][y] for y in years},
        min_over_rotations={str(y): py_min[y] for y in years},
        max_over_rotations={str(y): py_max[y] for y in years},
        overall_min=min(py_min.values()), overall_max=max(py_max.values()),
        note="within_year_null holds these fixed by construction; the rotation "
             "null does not, so it does not control the 1.70x level trend")

    # ==================================================================== 58.2
    say("\n=== 58.2 PRIMARY: the circular-shift p, all 78 shifts ===")
    n_term = [sum(1 for ym in clearing if LAB[k][ym] == "term") for k in range(N)]
    assert n_term[0] == N_OBS, "the rotation at k=0 is not the identity"
    ge = [k for k in range(1, N) if n_term[k] >= N_OBS]
    p_shift = (1 + len(ge)) / N
    resolution_floor = 1.0 / N
    branch = ("p_shift <= 0.05" if p_shift <= 0.05 else
              "0.05 < p_shift <= 0.20" if p_shift <= 0.20 else "p_shift > 0.20")
    say(f"  observed n_term(0) = {n_term[0]} of {len(clearing)}")
    say(f"  n_term(k) over k=1..78: min {min(n_term[1:])}, max {max(n_term[1:])}, "
        f"mean {float(np.mean(n_term[1:])):.3f}")
    say(f"  shifts reaching {N_OBS}: {ge}")
    say(f"  p_shift = (1 + {len(ge)}) / {N} = {p_shift:.7f}")
    say(f"  resolution floor 1/{N} = {resolution_floor:.7f} (declared in advance)")
    say(f"  -> branch: {branch}")
    out["p_shift"] = dict(
        statistic="n_term(k)", observed=n_term[0], n_clearing=len(clearing),
        numerator=1 + len(ge), denominator=N, p=p_shift,
        shifts_at_or_above_observed=ge, n_shifts_at_or_above=len(ge),
        shift_set="all k in 1..78, no exclusions",
        resolution_floor=resolution_floor,
        at_resolution_floor=bool(p_shift == resolution_floor),
        n_term_by_k=n_term,
        n_term_over_shifts=dict(min=min(n_term[1:]), max=max(n_term[1:]),
                                mean=float(np.mean(n_term[1:]))),
        branch=branch)

    # ==================================================================== 58.3
    say("\n=== 58.3 secondary (a): the same p with the seam shifts dropped ===")
    say(f"  {DECLARATION_NOTE}")
    sec_a = {}
    for tag, keep in (("a_mod6", [k for k in range(1, N) if k % 6 != 0]),
                      ("a_drop12", [k for k in range(1, N)
                                    if k not in range(6, 73, 6)])):
        hits = [k for k in keep if n_term[k] >= N_OBS]
        sec_a[tag] = dict(n_shifts=len(keep), numerator=1 + len(hits),
                          denominator=1 + len(keep), p=(1 + len(hits)) / (1 + len(keep)),
                          hits=hits,
                          resolution_floor=1.0 / (1 + len(keep)))
        say(f"  {tag:>9}: {len(keep)} shifts kept, {len(hits)} reach {N_OBS}, "
            f"p = (1+{len(hits)})/(1+{len(keep)}) = {sec_a[tag]['p']:.7f}")
    out["secondary_a"] = dict(
        readings=sec_a,
        rationale="the shifts that are multiples of 6 agree with the true "
                  "labelling far above chance, so they are the alternative "
                  "displaced by the 79 = 6*12+7 seam rather than null draws",
        status="SECONDARY. The all-78 version stays primary because it is the "
               "conservative one and excluding shifts after seeing them is "
               "shopping.")

    # ==================================================================== 58.4
    say("\n=== 58.4 secondary (b): p41's median contrast under the same shifts ===")


    def contrast(k):
        """p41's own statistic: median(term) - median(vacation), December in
    neither set, evaluated on the labelling produced by rotation k."""
        t = [MI[ym] for ym in YMS if LAB[k][ym] == "term"]
        v = [MI[ym] for ym in YMS if LAB[k][ym] == "vacation"]
        return float(np.median(t) - np.median(v))


    D = [contrast(k) for k in range(N)]
    ge_d = [k for k in range(1, N) if D[k] >= D[0]]
    p_d = (1 + len(ge_d)) / N
    rank_d = 1 + sum(1 for k in range(N) if D[k] > D[0])
    say(f"  D(0) = {D[0]:+.7f}")
    say(f"  D(k), k=1..78: min {min(D[1:]):+.7f}, max {max(D[1:]):+.7f}, "
        f"mean {float(np.mean(D[1:])):+.7f}")
    say(f"  rank of D(0) among the {N} rotations (1 = largest): {rank_d}")
    say(f"  shifts with D(k) >= D(0): {ge_d}")
    say(f"  p_D = (1 + {len(ge_d)}) / {N} = {p_d:.7f}")
    out["secondary_b"] = dict(
        statistic="D(k) = median(mi_bits_hf | term_k) - median(mi_bits_hf | "
                  "vacation_k), December in neither set",
        observed=D[0], rank_of_observed=rank_d, numerator=1 + len(ge_d),
        denominator=N, p=p_d, shifts_at_or_above=ge_d, D_by_k=D,
        D_over_shifts=dict(min=min(D[1:]), max=max(D[1:]),
                           mean=float(np.mean(D[1:]))),
        why="the count statistic is censored at 17 -- it cannot separate the "
            "identity from any rotation that also keeps all 17 in term -- so a "
            "continuous contrast is reported alongside it")

    # ==================================================================== 58.5
    say("\n=== 58.5 does the rotation DIRECTION change anything? ===")
    say("  Shifting the calendar forward by k and backward by k enumerate the same")
    say("  78 rotations, relabelled k <-> 79-k, so the PRIMARY p cannot depend on")
    say("  the convention. Secondary (a)'s exclusion rule can, because it names k")
    say("  rather than measuring agreement, and the seam shifts sit at k = 0 mod 6")
    say("  under one convention and k = 1 mod 6 under the other.")


    def rotate_back(k):
        """The opposite convention: the month at position i takes the label k
    positions LATER. rotate_back(k) is rotate(79-k) by construction."""
        return {YMS[i]: TRUE[YMS[(i + k) % N]] for i in range(N)}


    LAB_B = [rotate_back(k) for k in range(N)]
    n_term_b = [sum(1 for ym in clearing if LAB_B[k][ym] == "term")
                for k in range(N)]
    assert all(n_term_b[k] == n_term[(N - k) % N] for k in range(N)), (
        "the two conventions are not each other's mirror")
    ge_b = [k for k in range(1, N) if n_term_b[k] >= N_OBS]
    p_shift_b = (1 + len(ge_b)) / N
    assert p_shift_b == p_shift, "the primary p depends on the rotation direction"
    a_mod6_b = [k for k in range(1, N) if k % 6 != 0 and n_term_b[k] >= N_OBS]
    say(f"  primary p under both directions: {p_shift:.7f} == {p_shift_b:.7f}")
    say(f"  shifts reaching {N_OBS}: forward {ge}, backward {ge_b}")
    say(f"  secondary (a) 'k mod 6 != 0' keeps {len(a_mod6_b)} of them backward "
        f"vs {len(sec_a['a_mod6']['hits'])} forward -- NOT convention-free")

    # The convention-free reading of the same exclusion: drop a shift AND its
    # mirror, since agreement(k) == agreement(79-k) makes them equally
    # near-degenerate. Computed and reported whichever way it lands, exactly like
    # the two readings above; the primary is untouched either way.
    keep_sym = [k for k in range(1, N) if k % 6 != 0 and (N - k) % 6 != 0]
    hits_sym = [k for k in keep_sym if n_term[k] >= N_OBS]
    sec_a["a_mirror_symmetric"] = dict(
        n_shifts=len(keep_sym), numerator=1 + len(hits_sym),
        denominator=1 + len(keep_sym),
        p=(1 + len(hits_sym)) / (1 + len(keep_sym)), hits=hits_sym,
        resolution_floor=1.0 / (1 + len(keep_sym)))
    say(f"  a_mirror_symmetric: {len(keep_sym)} shifts kept, {len(hits_sym)} reach "
        f"{N_OBS}, p = {sec_a['a_mirror_symmetric']['p']:.7f}")
    out["secondary_a"]["readings"] = sec_a
    out["secondary_a"]["convention_defect"] = (
        "the exclusion rule names k, and the seam shifts are k = 0 mod 6 under the "
        "forward convention and k = 1 mod 6 under the backward one, so 'k mod 6 != "
        "0' is not invariant to the rotation direction: forward it drops all "
        f"{len(ge)} shifts that reach {N_OBS}, backward it drops none of them. The "
        "mirror-symmetric reading is reported for that reason.")
    out["direction_check"] = dict(
        p_forward=p_shift, p_backward=p_shift_b, equal=bool(p_shift == p_shift_b),
        shifts_forward=ge, shifts_backward=ge_b,
        mirror_identity_holds=True,
        secondary_a_mod6_hits_backward=a_mod6_b)

    # ==================================================================== 58.6
    out["anchors"] = anchors
    out["inputs"] = dict(
        series="results_p42.json rows[*].mi_bits_hf, not rebuilt",
        labels="results_p42.json rows[*].semester, reproduced by p42's label()",
        floor="results_p51.json cells['202312|national'].mi_perm_median",
        anchor="results_p54.json corrected",
        no_parquet=True, rng=None, deterministic=True)

    with open(f"{ROOT}/eda/results_p58.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p58.json")


if __name__ == "__main__":
    main()
