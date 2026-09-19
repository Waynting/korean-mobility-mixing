#!/usr/bin/env python
"""Phase 63 — Figure 2: the school calendar the matrix is never given, and the
null that is allowed to say so.

WHY THIS EXISTS. Figure 2 was still `eda/fig/p41_semester.png`, which is
`p41_semester.py`'s own diagnostic: 150 dpi, phase-numbered panel titles, no
`--pdf` (its argparse takes `--mc` and nothing else), and a per-year panel whose
grey bands are the WITHIN-YEAR RELABELLING null. This round demoted that null.
Writing a caption against that figure would have described (i) a file that
cannot be submitted and (ii) a null the letter of 2026-08-27 rejects. Same
reason p56 exists rather than a `--pdf` bolted onto p39: a diagnostic and a
submission figure answer different questions, and p41 sits in
`determinism_check.sh`'s default target list with published numbers of its own,
so editing it to draw a different figure would put an unrelated phase's
byte-determinism at risk to save a file.

IT COMPUTES NOTHING THAT IS NOT A DISPLAY AGGREGATE. Every number on the page is
read at draw time from `results_p42.json` (the 79-month series and its true
semester labels), `results_p51.json` (the floor), `results_p54.json` (the
corrected recount), `results_p58.json` (the circular-shift null) and
`results_p37.json` (the spatial ladder (c) plots). The
derived quantities are named as derived: the per-year medians drawn in (a) —
p42 stores the series, not its year medians — and the two counts taken off the
stored vectors, how many of the 78 alternative rotations agree above chance and
how many carry a regression t above the observed. All are aggregations of read
values, not new estimates, and the year-median ratio is anchored against the
1.70x that `memo/phase42-monthscope.md` publishes.

WHAT THE FIGURE IS ALLOWED TO CLAIM, and it is less than the old one claimed.
p58 replaced the semester null. The within-year hypergeometric (4.97e-06) treats
month labels as exchangeable inside a year; school terms are contiguous blocks,
so it is anticonservative. Rotating the whole calendar gives p = 4/79 = 0.0506,
which is ABOVE 0.05 and FINER than the test can resolve: the school labelling is
a function of month-of-year and 79 = 6*12 + 7, so the 79 rotations realise only
TWELVE distinct calendars, the smallest p the test can resolve is 1/12 = 0.083,
and 1/6 = 0.17 once the near-duplicate six-month offset is collapsed. The
observed 0.0506 is below both floors. So the figure carries a DESCRIPTIVE
ALIGNMENT COUNT — all 17 clearing months are term months, and 3 of the 78
alternative alignments match that — and it never states a significance verdict.
The resolution strip under (b) exists to make those floors visible rather than
to be read as a verdict, and (b)'s caption exists so that "3 of 78" cannot be
read as three coincidences: the three are at k = 12, 24 and 36, every one a
multiple of twelve, and they agree with the true labelling on 74, 69 and 64 of
79 months against a chance level of 36.5. They are the same calendar displaced
by the 79 = 6*12+7 seam, not draws from a null.

BOTH NULLS APPEAR, because neither dominates. The rotation does not hold the
per-year term margins (they run 3..8 across rotations against 7,7,7,7,7,7,4 at
k = 0), so it does not control the level trend that the within-year design
exists to control -- the trend panel (a) draws. That is panel (e).

(c) IS THE 2020 NATURAL CONTROL, AND IT ARRIVED FROM FIGURE 3 (2026-09-02).
It was 3(c): `p37_timeseries`'s spatial ladder on these same 79 months, tinted
by term and annotated "2020: schools shut, the cycle is absent". Figure 3 was
the wrong page for it. Neither §3.2 nor §3.3 -- the two sections Figure 3
belongs to -- ever cited the panel; its only reader was §3.1, two subsections
earlier, and §3.4's "dong above district above city in 79 of 79 months" had no
figure to point at at all. Both of its readings now sit next to a section that
uses them: §3.1 points here locally, §3.4 points back.

The panel arrives on ONE condition, and it is (a)'s band vocabulary. Stacked
under (a) on the identical 79-month axis, a column of this figure has to mean
the same thing in both panels or the pair is worse than the split was. So (c)
drops 3(c)'s own encoding -- a red 2020 hatch and no December state at all --
and calls the same bands() helper (a) does: term tint, December hatched grey.
That leaves 2020 needing a channel that is neither fill nor texture, because
both are now spoken for, so it is a dashed rule at the 2020|2021 seam with the
annotation inside the span. A line is the third channel, it survives greyscale,
and 2020 sits against the left spine so one rule brackets it.

WHAT IT STILL DOES NOT DRAW. The per-band decomposition, and p41's per-year
panel: that one would have to redraw the within-year relabelling bands to be
itself -- putting the demoted null back on the manuscript's second figure. One
statistic, one null story, one page.

ROMANISATION. No Hangul: it renders as tofu in every font matplotlib ships and
a figure needing a locally installed CJK font breaks on the typesetter's
machine.

    python eda/p63_fig2.py --pdf
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figstyle import WIDTH, finish, plabel
from paths import FIG, ROOT

# JRSI wants figure text in Times at 9-11 pt and refuses anything under 7.5 pt,
# so 7.5 is the floor for every explicit size below. STIXGeneral is the serif:
# it is Times-metric AND it ships inside matplotlib, so the figure renders the
# same on a machine with no Times installed. A figure that depends on a locally
# installed font breaks on the typesetter's machine -- the same reason the
# Korean here is romanised rather than set in a CJK font.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Times New Roman",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})


TEAL, RED, BLUE, GOLD = "#0E7C86", "#A8434E", "#4C6E8A", "#BC8034"
INK, GREY, PALE = "#1b1b1b", "#8a8a8a", "#c9c9c9"

# The December band in (a) is hatched so that it survives a greyscale print;
# see the comment where it is drawn. A December run is one month, which is
# 20 px of the 1638 px the 79 months get, so the pattern has to be dense
# enough to put more than one stroke inside a sliver that narrow, and thin
# enough not to fill it in: at the default 1.0 pt a "///" band comes out solid.
DEC_HATCH = "/////"
plt.rcParams["hatch.linewidth"] = 0.5

# (c) reads p37's series on the weekday-evening panel, the one every other
# 79-month reading in this paper is taken on.
PANEL = "WE"

sheet = []


def note(panel, what, value):
    """Record a number the caption is allowed to quote."""
    sheet.append((panel, what, value))


def load(name):
    return json.load(open(f"{ROOT}/eda/results_{name}.json"))


def runs_of(seq):
    """[(start, end_inclusive, value), ...] over a list, merging equal runs."""
    out, i = [], 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        out.append((i, j, seq[i]))
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    p42, p51, p54, p58, p64 = (load("p42"), load("p51"), load("p54"),
                               load("p58"), load("p64"))
    p37 = load("p37")                      # the spatial ladder (c) draws

    # ====================================================================
    # GATES. p58's own anchors, re-asserted from the files before a single
    # line is drawn. A figure is the last place a stale results file should
    # be discovered, and this one draws two files against each other.
    # ====================================================================
    rows = p42["rows"]
    assert len(rows) == 79, f"results_p42.json holds {len(rows)} months, not 79"
    YMS = sorted(r["ym"] for r in rows)
    assert len(set(YMS)) == 79, "duplicate ym in results_p42.json rows"
    assert all(YMS[i] < YMS[i + 1] for i in range(78)), \
        "the month series is not strictly increasing"
    # Never index months positionally: every lookup below goes through the label.
    POS = {ym: i for i, ym in enumerate(YMS)}
    MI = {r["ym"]: r["mi_bits_hf"] for r in rows}
    LAB = {r["ym"]: r["semester"] for r in rows}
    YEAR = {r["ym"]: r["year"] for r in rows}

    anc = p58["anchors"]
    assert anc["A1"]["n_months"] == 79, "p58 did not run on 79 months"
    assert anc["A1"]["first_ym"] == YMS[0] and anc["A1"]["last_ym"] == YMS[-1], \
        "p58's span is not the span drawn here"

    # The floor. It must be the p51 float bit for bit -- not p42's own `counts`
    # floor, which is the SUPERSEDED 0.023817 that counted 13 months.
    FLOOR = p51["cells"]["202312|national"]["mi_perm_median"]
    assert FLOOR == anc["A7"]["floor"], \
        "the floor drawn here is not the float p58 counted against"
    assert FLOOR == p54["corrected"]["floor"], \
        "the floor drawn here is not the float p54 counted against"
    assert anc["A7"]["bit_equal"], "p58 itself reports the floor as not bit-equal"
    assert FLOOR != p42["counts"]["202312|national"]["floor"], \
        "p42's counts block now holds the corrected floor; re-read this script"

    # A2: the recount at k = 0 reproduces p54, and the 17 months this figure
    # marks are re-derived here from the series it draws rather than copied.
    assert anc["A2"]["reproduces_p54"], "p58 no longer reproduces p54's recount"
    CLEAR = anc["A2"]["months"]
    assert CLEAR == sorted(p54["corrected"]["months"]), "p54 and p58 disagree"
    assert CLEAR == [ym for ym in YMS if MI[ym] >= FLOOR], \
        "the clearing months do not fall out of p42's series at p51's floor"
    obs = anc["A2"]["observed"]
    assert obs == {"term": 17, "vacation": 0, "december": 0}, \
        f"the label split is no longer 17/0/0: {obs}"
    assert len(CLEAR) == 17 == obs["term"], "the clearing count is not 17"
    assert p54["corrected"]["all_clearing_are_term"], \
        "p54: the clearing months are no longer all term months"

    # A4/A5: the label counts, and the margin the rotation holds fixed.
    counts_lab = anc["A4"]["label_counts"]
    assert anc["A4"]["n_mismatch"] == 0, "p42's stored labels and label() differ"
    assert sum(counts_lab.values()) == 79, "the three labels do not tile 79"
    for lab, n in counts_lab.items():
        assert sum(1 for ym in YMS if LAB[ym] == lab) == n, \
            f"p58's {lab} count is not what p42's rows carry"
    assert anc["A5"]["invariant"] and anc["A5"]["margin"] == 46, \
        "the term margin is no longer invariant at 46"
    assert anc["A5"]["distinct_term_margins"] == [46], \
        f"the margin moved: {anc['A5']['distinct_term_margins']}"

    # A6: the degeneracy audit. 79 prime, so 79 distinct rotations -- distinct
    # is not independent, and agreement(k) is what shows it.
    assert anc["A6"]["n_is_prime"], "79 is not prime; the distinctness fails"
    assert anc["A6"]["n_distinct_label_vectors"] == 79, \
        "the 79 rotations are not 79 distinct label vectors"
    assert anc["A6"]["symmetric_k_and_n_minus_k"], "agreement is not symmetric"
    AGREE = p58["degeneracy"]["agreement_by_k"]
    assert len(AGREE) == 79, "agreement_by_k is not 79 long"
    assert AGREE[0] == anc["A6"]["agreement_0"] == 79, "agreement(0) is not 79"
    assert AGREE[12] == anc["A6"]["agreement_12"] == 74, "agreement(12) is not 74"
    CHANCE = p58["degeneracy"]["chance_agreement"]
    assert CHANCE == anc["A6"]["chance_agreement"], "two chance levels in p58"
    assert all(AGREE[k] == AGREE[(79 - k) % 79] for k in range(79)), \
        "agreement(k) != agreement(79-k) in the stored vector"

    # The primary statistic, and the three matching alignments.
    sh = p58["p_shift"]
    NT = sh["n_term_by_k"]
    assert len(NT) == 79, "n_term_by_k is not 79 long"
    assert NT[0] == sh["observed"] == 17, "the rotation at k=0 is not the identity"
    GE = sh["shifts_at_or_above_observed"]
    assert GE == [k for k in range(1, 79) if NT[k] >= 17], \
        "the stored hit list is not the hits in the stored vector"
    assert sh["n_shifts_at_or_above"] == len(GE) == 3, \
        f"the alignment count is no longer 3: {GE}"
    # Multiples of TWELVE, not of six. p64's letter_verdict shows n_term at
    # k = 6, 18, 30, 42, 54, 66, 78 reaching 16 at best, so a six-month shift
    # does not by itself reproduce the alignment; the matches are 0 mod 12.
    assert all(k % 12 == 0 for k in GE), \
        f"the hits are not all multiples of 12: {GE}"
    assert sh["denominator"] == 79 and sh["numerator"] == 1 + len(GE), \
        "p_shift's fraction is not (1 + hits)/79"
    assert abs(sh["p"] - sh["numerator"] / sh["denominator"]) < 1e-15, \
        "p_shift is not its own numerator over its own denominator"
    assert abs(sh["resolution_floor"] - 1.0 / 79) < 1e-15, \
        "the declared resolution floor is not 1/79"
    assert not sh["at_resolution_floor"], "p_shift is at its floor; re-read (d)"

    # ---- p37, the spatial ladder (c) draws -------------------------------
    # These two asserts came from p47's gates() with the panel (2026-09-02).
    # A ladder that has stopped holding is the one thing that would make (c)
    # draw three curves whose ORDER is the reading, in an order that is no
    # longer true.
    assert p37["ladder"]["n_monotone"] == p37["ladder"]["n_months"] == 79, \
        "p37: the dong > gu > city ladder no longer holds in every month"
    assert not p37["ladder"]["breaks"], "p37: the ladder has breaks"
    # And the months have to be the months (a) and (b) are drawn on, or the two
    # panels share an x axis they do not share. Compared as LABELS, never by
    # position: p10_response once reported August's residual as September's.
    assert p37["months"] == YMS, \
        "p37's 79 months are not the 79 months (a) and (b) are drawn on"
    _LADDER = {}
    for scale in ("dong", "gu", "city"):
        by = {r["ym"]: r for r in p37["rows"]
              if r["level"] == scale and r["panel"] == PANEL}
        assert set(by) == set(YMS), f"p37: {scale} is not on all 79 months"
        _LADDER[scale] = {ym: by[ym]["assortativity"] for ym in YMS}
    assert all(_LADDER["dong"][ym] > _LADDER["gu"][ym] > _LADDER["city"][ym]
               for ym in YMS), \
        "the ladder p37 reports monotone is not monotone in the drawn series"

    # The rotation direction. k is drawn under p58's forward convention (the
    # month at position i takes the label of the month k positions earlier).
    # The p does not depend on that choice, but the IDENTITY of the matching
    # shifts does -- backward they are the mirrors 43, 55, 67 -- so the axis is
    # only readable if the convention is fixed and the mirror fact is stated.
    dc = p58["direction_check"]
    assert dc["equal"] and dc["p_forward"] == sh["p"], \
        "the primary p depends on the rotation direction"
    assert dc["shifts_forward"] == GE, "p58's forward hits are not the hit list"
    assert dc["mirror_identity_holds"], "the two conventions are not mirrors"
    assert sorted(dc["shifts_backward"]) == sorted(79 - k for k in GE), \
        "the backward hits are not the forward hits mirrored"

    # The secondary contrast, and the tie the caption has to state.
    sb = p58["secondary_b"]
    D = sb["D_by_k"]
    assert len(D) == 79, "D_by_k is not 79 long"
    assert D[0] == sb["observed"], "D(0) is not the stored observed contrast"
    assert D[12] == D[0], \
        "D(12) is no longer bitwise identical to D(0); the tie claim changes"
    assert sb["rank_of_observed"] == 1, "D(0) is no longer rank 1"
    assert sb["numerator"] == 2 and sb["denominator"] == 79, \
        "p_D is no longer 2/79"

    # The runs: the advisor's premise about the effective sample size.
    rs = p58["run_structure"]
    assert rs["n_clearing"] == 17 and sum(rs["run_lengths"]) == 17, \
        "the runs do not account for the 17 clearing months"
    assert rs["n_runs"] == len(rs["run_lengths"]) == 9, \
        f"the clearing months no longer sit in 9 runs: {rs['run_lengths']}"

    # The within-year null, kept and reported rather than discarded.
    P_WITHIN = p58["within_year"]["p_term_ge"]
    assert P_WITHIN == anc["A3"]["p_within_year"] == p54["corrected"]["p_term_ge"], \
        "p58 and p54 disagree on the within-year p"

    # The per-year margins, which is what the rotation gives up.
    pym = p58["per_year_margins"]
    YEARS = [int(y) for y in pym["years"]]
    assert YEARS == sorted({YEAR[ym] for ym in YMS}), \
        "p58's years are not the years in p42's rows"
    assert sum(pym["identity"].values()) == 46 == anc["A5"]["margin"], \
        "the per-year identity margins do not sum to the fixed term margin"
    for y in YEARS:
        assert pym["identity"][str(y)] == sum(
            1 for ym in YMS if YEAR[ym] == y and LAB[ym] == "term"), \
            f"p58's identity margin for {y} is not p42's"

    # The level trend, derived here because p42 stores the series and not its
    # year medians. Anchored against the 1.70x that phase42-monthscope.md
    # publishes, so this is a redraw of a published number, not a new one.
    YMED = {y: statistics.median([MI[ym] for ym in YMS if YEAR[ym] == y])
            for y in YEARS}
    TREND = YMED[max(YEARS)] / YMED[min(YEARS)]
    assert round(TREND, 2) == 1.70, \
        f"the year-median trend is {TREND:.4f}, not the published 1.70x"

    # ====================================================================
    # p64's GATES. The regression that replaces the seven-point sign test
    # (advisor 2026-08-31, item 2.1) and the honest resolution floor
    # (item 2.2) both live in results_p64.json. Re-assert its invariants
    # from the file before drawing, and check its series IS this one.
    # ====================================================================
    assert p64["inputs"]["series"].startswith("results_p42.json"), \
        "p64 was not fitted on p42's series"
    _a64 = p64["anchors"]["A1"]
    assert (_a64["n_months"] == 79 and _a64["first_ym"] == YMS[0]
            and _a64["last_ym"] == YMS[-1] and _a64["n_label_mismatch"] == 0), \
        "p64's span or labels are not this figure's"
    REG = p64["primary"]["assortativity_all"]      # the sign test's own series
    REGM = p64["primary"]["mi_bits_hf"]            # the series (a) draws
    assert REG["n"] == 79 and REG["lag"] == 3, \
        "p64's assortativity fit is not the 79-month, lag-3 primary"
    assert REG["coef"]["term"] > 0 and REGM["coef"]["term"] > 0, \
        "the term coefficient is no longer positive on both primary series"
    assert (p64["sign_agreement"]["n_positive"]
            == p64["sign_agreement"]["n_series"] == 6), \
        "the term coefficient is no longer positive on all six variants"
    # The rotation regression: the same fit under all 79 circular calendar
    # rotations, year effects and the trend held fixed. Its observed t is the
    # statistic panel (c) draws.
    RROT = p64["rotation_regression"]["assortativity_all"]["t_by_k"]
    assert len(RROT) == 79 and abs(RROT[0] - REG["t"]["term"]) < 1e-12, \
        "p64's rotation regression does not open at the observed t"
    assert p64["rotation_regression_dedup"]["assortativity_all"][
        "shifts_ge_t"] == [], "a calendar rotation now beats the observed t"
    assert max(range(79), key=lambda k: RROT[k]) == 0, \
        "the observed regression t is no longer the tallest of the 79 rotations"
    # The honest resolution floor. The school labelling is a function of
    # month-of-year and 79 = 6*12 + 7, so the 79 rotations realise only 12
    # distinct calendars; collapsing the six-month near-symmetry leaves 6.
    E12 = p64["effective_alignments"]["E_exact"]
    E6 = p64["effective_alignments"]["E_advisor_tau10"]
    assert E12 == 12 and E6 == 6, f"the alignment count moved: {E12}, {E6}"
    FLOOR_12 = p64["deduplicated"]["exact"]["resolution_floor"]
    FLOOR_6 = p64["deduplicated"]["advisor"]["resolution_floor"]
    assert abs(FLOOR_12 - 1.0 / 12) < 1e-12 and abs(FLOOR_6 - 1.0 / 6) < 1e-12, \
        "the honest resolution floors are not 1/12 and 1/6"
    assert sh["p"] < FLOOR_12 < FLOOR_6, \
        "the observed p is no longer below both honest floors"
    assert (p64["deduplicated"]["exact"][
                "n_nonidentity_classes_reaching_observed"] == 0
            and p64["deduplicated"]["advisor"][
                "n_nonidentity_classes_reaching_observed"] == 0), \
        "an alternative calendar now reproduces the observed alignment"

    # ====================================================================
    # THE FIGURE. Three panels.
    #   (a) the observation, kept.
    #   (b) the rotation count, carrying the HONEST resolution floor that
    #       old (d) drew at 1/79, with the p-resolution strip beneath it.
    #       From 2026-08-31 to 2026-09-15 a second strip under that one
    #       carried the regression that replaces the seven-point sign test,
    #       as two lines of teal text. The advisor's 2026-09-15 letter (item
    #       5) read it as a working note rather than a figure element, so the
    #       regression now lives in the caption; the numbers are still
    #       emitted to the caption sheet below, where the gate reads them.
    #   (c) the spatial ladder on the same 79 months, arrived from Figure 3
    #       on 2026-09-02 (see the docstring). It is a plotted panel and it
    #       does take a third letter: it is a different QUANTITY on a
    #       different y axis, not a restatement of (b)'s.
    # The "distinct is not independent" point and the secondary contrast go
    # to the caption; the per-year-margin range goes to SI 5.
    #
    # Portrait, WIDTH in wide (figstyle.py). The middle block lost the
    # regression strip (0.24 of its 1.00 sub-units) and gained a gap wide
    # enough for (b)'s own x ticks and title, which the strip's removal
    # exposes: (b) used to hand its x axis to the strip below and label
    # nothing, so k had neither ticks nor a title (advisor 2026-09-15, item
    # 7). Holding (b) and the p strip at the heights they had, the block is
    # 1.25 outer units rather than 1.70. At ratios [1.24, 1.25, 1.24] and
    # hspace 0.34 the three rows and their two gaps come to 4.575 height
    # units, so holding the unit at ~1.48 in needs 6.77 in of content band,
    # plus 0.50 in under the axes for (c)'s year ticks and its new axis
    # title and 0.35 in of top margin: 7.62 in.
    # ====================================================================
    fig = plt.figure(figsize=(WIDTH, 7.62))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.24, 1.25, 1.24],
                          hspace=0.34, left=0.135, right=0.975,
                          top=0.954, bottom=0.066)
    ax_a = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[2])
    # The strip was 0.20 and its own marker labels did not fit inside it: at
    # 0.20 it is 37 px tall, which leaves 18 px below the markers for a label
    # that is 10 px tall plus the marker's own radius, so every label crossed
    # the bottom spine and the frame line struck through it. 0.26 leaves 24 px
    # and the labels sit inside the box. hspace 0.635 is the 0.55 the three-row
    # block had, widened so that (b)'s tick labels and axis title fit between
    # (b) and the strip: 0.635 x 0.38 = 0.24 sub-units, 0.30 outer, 0.44 in.
    _gsb = gs[1].subgridspec(2, 1, height_ratios=[0.50, 0.26],
                             hspace=0.635)
    ax_b = fig.add_subplot(_gsb[0])
    ax_bp = fig.add_subplot(_gsb[1])           # the p-resolution strip

    def title(ax, letter, s):
        ax.set_title(plabel(letter, s), fontsize=8.2, loc="left", pad=4.5,
                     color=INK)

    # ------------------------------------------------ (a) the observation
    a = ax_a
    x = [POS[ym] for ym in YMS]
    y = [MI[ym] for ym in YMS]
    # THE TWO BANDS ARE NOT LEFT TO COLOUR. JRSI prints black and white by
    # default, and the two shadings as first drawn -- teal at alpha .11 and
    # grey at alpha .17 -- converted to 237.5 and 235.0 out of 255. A 2.5/255
    # difference is nothing: the reader of a printed copy saw ONE band colour,
    # and this panel's whole claim is that the 17 red points all fall in the
    # first kind and none in the second. So December carries two redundant,
    # non-colour channels on top of its hue: it is darker (alpha .30 against
    # .11, which is as far below the term band as the term band is below the
    # white of a vacation month, so the three states read as three tones in a
    # row) and it is HATCHED, which no lightness conversion can flatten. The
    # hatch is on December rather than on term because December is seven
    # one-month slivers and term is thirty-nine months of the seventy-nine:
    # hatching the common state would put texture behind most of the series.
    # ONE helper, called by (a) and by (c). They are stacked on the identical
    # 79-month axis, so a column has to mean the same thing in both; two
    # copies of this that drift apart would be the worst version of that.
    def bands(ax):
        for i0, i1, lab in runs_of([LAB[ym] for ym in YMS]):
            if lab == "term":
                ax.axvspan(i0 - .5, i1 + .5, facecolor=to_rgba(TEAL, .11),
                           lw=0, zorder=0)
            elif lab == "december":
                ax.axvspan(i0 - .5, i1 + .5, facecolor=to_rgba(GREY, .30),
                           hatch=DEC_HATCH, edgecolor=(0, 0, 0, .42), lw=0,
                           zorder=0)

    bands(a)
    a.plot(x, y, "-", lw=1.15, color=INK, zorder=3)
    a.plot(x, y, "o", ms=2.5, mfc="white", mec=INK, mew=.7, zorder=4)
    a.plot([POS[ym] for ym in CLEAR], [MI[ym] for ym in CLEAR], "o", ms=5.0,
           color=RED, zorder=6)
    a.axhline(FLOOR, color=RED, lw=1.15, ls="--", zorder=5)
    for y_ in YEARS:                       # the level trend, as year medians
        xs = [POS[ym] for ym in YMS if YEAR[ym] == y_]
        a.plot([min(xs) - .5, max(xs) + .5], [YMED[y_]] * 2, "-", lw=1.6,
               color=BLUE, alpha=.75, zorder=2)
    # Two annotations only, both short: the number on the floor and the trend.
    # Everything else the old panel spelled out is in the caption now
    # (advisor 2026-08-31, item 3.1).
    a.text(0.6, FLOOR * 1.035, f"floor {FLOOR:.5f} bits", fontsize=7.5,
           color=RED, va="bottom",
           bbox=dict(fc="white", ec="none", alpha=.88, pad=1.2))
    a.text(16.0, 0.00742, f"year medians rise {TREND:.2f}\u00d7", fontsize=7.5,
           color=BLUE, ha="left", va="bottom")
    ticks = [POS[ym] for ym in YMS if ym % 100 == 1]
    a.set_xticks(ticks)
    a.set_xticklabels([str(YMS[t] // 100) for t in ticks], fontsize=7.6)
    a.set_xlim(-1.0, 79.0)
    a.set_ylim(0.0068, 0.0322)
    a.tick_params(axis="y", labelsize=7.5)
    a.set_xlabel("month, 2020\u20132026", fontsize=7.6)
    a.set_ylabel("departure from proportionate mixing\n"
                 "age-pair mutual information (bits)", fontsize=7.6)
    # The panel letter carries no title: the span and the subject of each
    # panel are already the caption's first clause, and JRSI's own figures
    # letter the panel and say the rest below the figure.
    title(a, "a", "")
    a.grid(alpha=.16, lw=.55, axis="y")
    a.legend(handles=[
        Patch(fc=to_rgba(TEAL, .11), lw=0,
              label="school term (Mar\u2013Jun, Sep\u2013Nov)"),
        Patch(fc=to_rgba(GREY, .30), hatch=DEC_HATCH, ec=(0, 0, 0, .42), lw=0,
              label="December: neither set"),
        Line2D([], [], marker="o", ls="none", ms=5, color=RED,
               label=f"reaches the national survey floor ({len(CLEAR)} of "
                     f"{len(YMS)})"),
        Line2D([], [], color=BLUE, lw=1.6, alpha=.75, label="year median")],
        fontsize=7.5, frameon=True, framealpha=.92, edgecolor="none",
        loc="upper left", ncol=2, handlelength=1.5, columnspacing=1.1,
        borderpad=.4, labelspacing=.35)
    note("a", "months in the series", len(YMS))
    note("a", "first month / last month", f"{YMS[0]} / {YMS[-1]}")
    note("a", "floor (p51 corrected national permutation median, bits)", FLOOR)
    note("a", "months at or above the floor", len(CLEAR))
    note("a", "of those, term / vacation / December",
         f"{obs['term']} / {obs['vacation']} / {obs['december']}")
    note("a", "the clearing months", CLEAR)
    note("a", "series labels: term / vacation / December",
         f"{counts_lab['term']} / {counts_lab['vacation']} / "
         f"{counts_lab['december']}")
    note("a", "the clearing months sit in this many runs", rs["n_runs"])
    note("a", "run lengths", rs["run_lengths"])
    note("a", "year medians, first to last (derived at draw time, bits)",
         f"{YMED[min(YEARS)]:.5f} -> {YMED[max(YEARS)]:.5f}")
    note("a", "level trend over the period (derived; phase42 publishes 1.70x)",
         TREND)

    # ------------------- (b) rotating the whole school calendar: the count
    b = ax_b
    ks = list(range(79))
    cols = [TEAL if k == 0 else (GOLD if k in GE else PALE) for k in ks]
    b.vlines(ks, 0, NT, colors=cols, lw=1.9, zorder=2)
    b.plot([k for k in ks if k not in GE and k != 0],
           [NT[k] for k in ks if k not in GE and k != 0], "o", ms=2.3,
           color=GREY, zorder=3)
    b.plot(GE, [NT[k] for k in GE], "o", ms=4.8, color=GOLD, zorder=4)
    b.plot([0], [NT[0]], "o", ms=5.6, color=TEAL, zorder=5)
    b.axhline(17, color=INK, lw=.8, ls=":", zorder=1)
    mean_sh = sh["n_term_over_shifts"]["mean"]
    b.axhline(mean_sh, color=GREY, lw=.9, ls="--", zorder=1)
    b.text(1.5, NT[0] + 2.2, "k = 0: the true calendar", fontsize=7.5,
           color=TEAL, va="bottom")
    b.text(37.0, NT[GE[-1]] + 2.2, "k = 12, 24, 36: the 79 = 6\u00d712 + 7 seam",
           fontsize=7.5, color="#8a5c1f", va="bottom", ha="center")
    b.text(78.0, mean_sh - 1.0, f"mean {mean_sh:.2f}", fontsize=7.5,
           color=GREY, ha="right", va="top")
    b.set_ylim(0, 22.0)
    b.set_yticks([0, 5, 10, 15, 17])
    b.set_xlim(-1.0, 79.0)
    # k is labelled on its own axis. Until 2026-09-15 the ticks were drawn
    # and their labels suppressed, on the reading that the strip below was
    # (b)'s x axis; it is not -- the strip's x is p on a log scale -- so k
    # had no ticks a reader could read and no title.
    b.set_xticks([0, 12, 24, 36, 48, 60, 72])
    b.set_xticks([6, 18, 30, 42, 54, 66, 78], minor=True)
    b.tick_params(axis="both", labelsize=7.5)
    b.set_xlabel("calendar rotation k (months)", fontsize=7.6)
    b.set_ylabel("clearing months\nlabelled term", fontsize=7.6)
    title(b, "b", "")
    b.grid(alpha=.16, lw=.55, axis="both")

    note("b", "clearing months in term, true calendar", NT[0])
    note("b", "alternative alignments that also reach 17", len(GE))
    note("b", "alternatives in total", len(ks) - 1)
    note("b", "the shifts that match (months)", GE)
    note("b", "every matching shift is a multiple of twelve",
         all(k % 12 == 0 for k in GE))
    note("b", "n_term over the 78 shifts: min / max / mean",
         f"{sh['n_term_over_shifts']['min']} / "
         f"{sh['n_term_over_shifts']['max']} / {mean_sh:.4f}")
    note("b", "term margin, invariant across all 79 rotations",
         anc["A5"]["margin"])
    note("b", "the same three shifts under the opposite convention",
         dc["shifts_backward"])
    note("b", "circular-shift p, all 78 alternatives", sh["p"])
    note("b", "circular-shift p as a fraction",
         f"{sh['numerator']}/{sh['denominator']}")

    # ------------- (b, lower strip) what the rotation test can resolve.
    # This is old panel (d), corrected: the 79 rotations realise only 12
    # distinct calendars (79 = 6x12 + 7 and the labelling is a function of
    # month-of-year), so the smallest p the test resolves is 1/12 = 0.083,
    # or 1/6 = 0.17 collapsed -- not the 1/79 = 0.0127 the old panel drew.
    # Both rotation p's sit inside the grey, i.e. below the resolution.
    bp = ax_bp
    bp.axvspan(1.3e-6, FLOOR_12, color=GREY, alpha=.22, lw=0, zorder=0)
    bp.axvspan(FLOOR_12, FLOOR_6, color=GREY, alpha=.11, lw=0, zorder=0)
    bp.axvline(FLOOR_12, color=INK, lw=.85, zorder=1)
    bp.axvline(FLOOR_6, color=INK, lw=.85, ls="--", zorder=1)
    # ha is deliberately opposite for the two labels: anchored at their own
    # axvline, each string grows AWAY from the other, which is what keeps
    # "1/12 = 0.083" and "1/6 = 0.17" from colliding at this log spacing.
    bp.text(FLOOR_12, 1.28, f"1/12 = {FLOOR_12:.3f}  ", fontsize=7.5,
            color=INK, ha="right", va="center")
    bp.text(FLOOR_6, 1.28, f"  1/6 = {FLOOR_6:.2f}", fontsize=7.5,
            color=INK, ha="left", va="center")
    # The three markers' own labels, one line each and on one row. "median
    # contrast" (p = 0.0253) and "count" (p = 0.0506) are a factor of two
    # apart, which on this log axis is 28 px, and the two strings are 60 px
    # wide together: they cannot both be centred on their own dot. The old
    # evasion was to set "median contrast" over two lines and drop "count" to a
    # second row at y = -1.85. It did not work -- the two boxes still crossed
    # (by 1.3 px at the 7.2 pt "count" was set at, 2.5 px once both were raised
    # to the 7.5 pt floor) and the second row pushed "count" down into the x
    # tick labels' band, which is the collision this strip has had before.
    # The fix is horizontal, not vertical: "median contrast" is anchored RIGHT
    # of its own dot and grows away, the same device the two floor labels above
    # use, which frees a whole line of height and lets all three labels sit
    # inside the axes rather than straddling its bottom spine.
    # "count" stays CENTRED and is not given the mirror treatment: growing it
    # rightwards would run it straight through the solid 1/12 rule at x = 0.083.
    for _p, _c, _m, _lab, _ha in ((P_WITHIN, BLUE, "D", "within-year", "center"),
                                  (sb["p"], GOLD, "o", "median contrast  ",
                                   "right"),
                                  (sh["p"], RED, "o", "count", "center")):
        bp.plot([_p], [0], _m, ms=6.0, color=_c, mfc=_c, zorder=3)
        bp.text(_p, -0.56, _lab, fontsize=7.5, color=_c, ha=_ha, va="top",
                linespacing=1.15)
    bp.text(2.0e-6, 1.28, "no p resolves left of here", fontsize=7.5,
            color=INK, ha="left", va="center", style="italic")
    bp.set_xscale("log")
    bp.set_xlim(1.3e-6, 1.0)
    bp.set_ylim(-2.0, 2.0)
    bp.set_yticks([])
    bp.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    bp.minorticks_off()
    bp.tick_params(axis="x", labelsize=7.5)
    bp.set_xlabel("p under each rotation null (log scale)", fontsize=7.5)
    bp.grid(alpha=.14, lw=.5, axis="x")

    note("b", "distinct calendars the 79 rotations realise (79 = 6x12 + 7)",
         E12)
    note("b", "resolution floor, 1 over the distinct calendars", FLOOR_12)
    note("b", "resolution floor after collapsing the six-month near-symmetry",
         FLOOR_6)
    note("b", "the observed p is below both resolution floors",
         sh["p"] < FLOOR_12)
    note("b", "within-year hypergeometric p (superseded, kept findable)",
         P_WITHIN)
    note("b", "secondary median-contrast p, on the rotations", sb["p"])
    note("b", "secondary median-contrast D(0), bits", sb["observed"])
    note("b", "secondary contrast rank among the 79 rotations "
              "(a tie: D(12) is bitwise D(0))", sb["rank_of_observed"])

    # The "not three independent draws" point (advisor 2026-08-31, item 3.2:
    # old panel (c)'s message folds into (b)'s caption). AGREE and CHANCE are
    # already anchored above against p58's own A6 invariants; this is the
    # first and only place they are put on the page.
    n_above = sum(1 for k in range(1, 79) if AGREE[k] > CHANCE)
    note("b", "agreement at k = 12", AGREE[12])
    note("b", "agreement at k = 24", AGREE[24])
    note("b", "agreement at k = 36", AGREE[36])
    note("b", "chance agreement", CHANCE)
    note("b", "shifts agreeing above chance", f"{n_above} of {len(ks) - 1}")

    # ------------------- (b), the regression that replaces the sign test.
    # Advisor item 2.1: 79 months, a sign test on 7. The fit is
    #   y_t = a + b*term_t + d*dec_t + year FE + trend, Newey-West SE,
    # on the assortativity series the sign test itself used. It is not drawn:
    # from 2026-08-31 it was two lines of text in a strip under the p strip,
    # and the advisor's 2026-09-15 letter (item 5) took the strip for a
    # working note, so the coefficient, its standard error, t, p and the rank
    # among the 79 rotations go to the caption and to the sheet below, which
    # is where the gate reads them. RROT (the observed t against its value
    # under every calendar rotation) is still computed and still gates the
    # "tallest of all 79" claim -- the caption states the rank, a reader does
    # not need the other 78 bars to believe it.
    b_term, se_term = REG["coef"]["term"], REG["se"]["term"]
    t_term, p_term = REG["t"]["term"], REG["p_t"]["term"]

    note("b", "term coefficient, dong-level assortativity", b_term)
    note("b", "Newey-West standard error", se_term)
    note("b", "t on the term coefficient", t_term)
    note("b", "p on the term coefficient (two-sided)", p_term)
    note("b", "Newey-West lag, fixed by rule before the fit", REG["lag"])
    note("b", "months in the fit", REG["n"])
    note("b", "term coefficient on the mutual information of (a), bits",
         REGM["coef"]["term"])
    note("b", "t on that coefficient", REGM["t"]["term"])
    note("b", "the term coefficient is positive on this many of six variants",
         p64["sign_agreement"]["n_positive"])
    note("b", "the trend term itself reads t", REG["t"]["trend"])
    note("b", "the observed regression t ranks first among all 79 rotations",
         1 + sum(1 for k in range(1, 79) if RROT[k] > RROT[0]))
    note("b", "calendar rotations whose t beats the observed",
         sum(1 for k in range(1, 79) if RROT[k] > RROT[0]))
    note("b", "the regression rotation p after deduplication, 1/12",
         p64["rotation_regression_dedup"]["assortativity_all"]["exact"][
             "p_dedup"])

    # --------------------- (c) the spatial ladder, and the 2020 control
    # Arrived from Figure 3 on 2026-09-02. Content unchanged: p37's three
    # scales on p37's 79 months. What changed is the band key (bands(), the
    # same call (a) makes) and the x ticks, which are now (a)'s years rather
    # than 3(c)'s every-sixth-month rotated ym labels. Two panels stacked on
    # one axis may not carry two x vocabularies, and the years win because
    # (a) needs them for its per-year medians.
    c = ax_c
    bands(c)
    for scale, lab, col in (("dong", "dong (424)", TEAL),
                            ("gu", "district (25)", BLUE),
                            ("city", "city (1)", GOLD)):
        c.plot(x, [_LADDER[scale][ym] for ym in YMS], "-", lw=1.3, color=col,
               label=lab, zorder=3)
    # 2020 gets the third channel. Fill is the term band and texture is
    # December, so the year the panel exists to mark cannot have either
    # without colliding with a state (a) has already defined -- see the
    # docstring. A dashed rule at the seam does it, and because 2020 is the
    # first year on the axis the left spine closes the bracket.
    _seam = POS[202012] + .5
    c.axvline(_seam, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=5)
    _dong_max = max(_LADDER["dong"].values())
    # Centred on the span the label would put its right edge ON the seam rule,
    # which is the one thing on the panel it must not cover. 1.6 units left of
    # centre clears it and still sits inside 2020.
    c.annotate("2020: schools shut,\nthe cycle is absent",
               xy=(_seam / 2 - 1.6, _dong_max), ha="center", va="top",
               fontsize=7.5,
               color=RED,
               bbox=dict(fc="white", ec="none", alpha=.86, pad=1.6), zorder=6)
    c.set_xticks(ticks)
    c.set_xticklabels([str(YMS[t] // 100) for t in ticks], fontsize=7.6)
    c.set_xlim(-1.0, 79.0)
    c.tick_params(axis="y", labelsize=7.5)
    c.set_xlabel("month, 2020\u20132026", fontsize=7.6)
    c.set_ylabel("assortativity", fontsize=7.6)
    # Headroom for a legend that can only get one strip across the top at this
    # width: the dong series is the top curve in all 79 months, so every other
    # corner is either on a curve or on the 2020 annotation.
    _lo, _hi = c.get_ylim()
    c.set_ylim(_lo, _hi + .26 * (_hi - _lo))
    title(c, "c", "")
    c.grid(alpha=.16, lw=.55, axis="y")
    c.legend(handles=[
        Line2D([], [], color=TEAL, lw=1.3, label="dong (424)"),
        Line2D([], [], color=BLUE, lw=1.3, label="district (25)"),
        Line2D([], [], color=GOLD, lw=1.3, label="city (1)"),
        Line2D([], [], color=RED, lw=1.0, ls=(0, (4, 2.4)),
               label="2020 ends here")],
        fontsize=7.5, frameon=True, framealpha=.92, edgecolor="none",
        loc="upper right", ncol=4, handlelength=1.5, columnspacing=1.0,
        borderpad=.4, labelspacing=.35)

    _sd = p37["summary_dong"]
    for k in ("assort_min", "assort_max", "assort_median"):
        note("c", f"dong {k}", round(float(_sd[k]), 5))
    note("c", "months", int(_sd["n_months"]))
    note("c", "months where the ladder holds", int(p37["ladder"]["n_monotone"]))
    note("c", "the panel p37's series is read on", PANEL)

    png = f"{FIG}/p63_figure2.png"
    mode = finish(fig, png, pdf=args.pdf)
    plt.close(fig)
    print(f"wrote {png} ({mode})" + (" (+ .pdf)" if args.pdf else ""))

    print("\n=== the numbers this figure's caption may quote ===")
    for panel, what, value in sheet:
        v = f"{value:.6g}" if isinstance(value, float) else str(value)
        print(f"  ({panel})  {what:<62} {v}")
    out = dict(
        caption_numbers=[dict(panel=p_, what=w, value=v)
                         for p_, w, v in sheet],
        source="results_p58.json (the null) + results_p42.json (the series) "
               "+ results_p51.json (the floor) + results_p54.json (the recount) "
               "+ results_p64.json (the regression the caption of (b) states, "
               "and the resolution floors it is read against)",
        derived_at_draw_time=[
            "the per-year medians drawn in (a): p42 stores the series, not its "
            f"year medians; their ratio {TREND:.4f} is anchored against the "
            "1.70x published in memo/phase42-monthscope.md",
            "the two counts read off the stored vectors: how many of the 78 "
            "alternative rotations agree above p58's chance level, and how "
            "many carry a regression t above the observed in p64's t_by_k"],
        refuses="the figure states no significance verdict. p_shift = 0.0506 "
                "is above 0.05 and is finer than the test can resolve: the "
                "school labelling is a function of month-of-year and "
                "79 = 6x12+7, so the 79 rotations realise only 12 distinct "
                "calendars, the smallest p the test can resolve is "
                "1/12 = 0.083, and 1/6 = 0.17 once the near-duplicate "
                "six-month offset is collapsed, so the observed 0.0506 is "
                "below both floors; the 3 matching shifts are the same "
                "calendar displaced by that seam, not three independent "
                "draws; and the secondary contrast's rank 1 is a tie with "
                "k = 12. What the figure carries is a descriptive alignment "
                "count.",
        not_drawn="the per-band decomposition: p41's per-year panel would "
                  "have to redraw the within-year relabelling bands this "
                  "round demoted. The 2020 natural control is drawn, as "
                  "(c)'s annotation, since panel (c) arrived from Figure 3 on "
                  "2026-09-02. The regression's numbers are in the sheet and "
                  "the caption, not on the page, since 2026-09-15.")
    with open(f"{ROOT}/eda/results_p63.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {ROOT}/eda/results_p63.json")


if __name__ == "__main__":
    main()
