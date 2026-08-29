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
corrected recount) and `results_p58.json` (the circular-shift null). The two
derived quantities are named as derived: the per-year medians drawn in (a) —
p42 stores the series, not its year medians — and the arithmetic p = (1 + h)/79
for h one either side of the observed, which is the resolution statement in (d).
Both are aggregations of read values, not new estimates, and the year-median
ratio is anchored against the 1.70x that `memo/phase42-monthscope.md` publishes.

WHAT THE FIGURE IS ALLOWED TO CLAIM, and it is less than the old one claimed.
p58 replaced the semester null. The within-year hypergeometric (4.97e-06) treats
month labels as exchangeable inside a year; school terms are contiguous blocks,
so it is anticonservative. Rotating the whole calendar gives p = 4/79 = 0.0506,
which is ABOVE 0.05 and a factor of four above the smallest p a 79-rotation test
can produce. So the figure carries a DESCRIPTIVE ALIGNMENT COUNT — all 17
clearing months are term months, and 3 of the 78 alternative alignments match
that — and it never states a significance verdict. Panel (d) exists to make the
resolution visible rather than to be read as a verdict, and panel (c) exists so
that "3 of 78" cannot be read as three coincidences: the three are at k = 12, 24
and 36, every one a multiple of six, and they agree with the true labelling on
74, 69 and 64 of 79 months against a chance level of 36.5. They are the same
calendar displaced by the 79 = 6*12+7 seam, not draws from a null.

BOTH NULLS APPEAR, because neither dominates. The rotation does not hold the
per-year term margins (they run 3..8 across rotations against 7,7,7,7,7,7,4 at
k = 0), so it does not control the level trend that the within-year design
exists to control -- the trend panel (a) draws. That is panel (e).

WHAT IT DOES NOT DRAW. The 2020 natural control and the per-band decomposition.
Figure 3(c) already annotates "2020: schools shut, the cycle is absent" on the
same 79 months, and p41's per-year panel would have to redraw the within-year
relabelling bands to be itself -- putting the demoted null back on the
manuscript's second figure. One statistic, one null story, one page.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

    p42, p51, p54, p58 = load("p42"), load("p51"), load("p54"), load("p58")

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
    assert all(k % 6 == 0 for k in GE), f"the hits are not all multiples of 6: {GE}"
    assert sh["denominator"] == 79 and sh["numerator"] == 1 + len(GE), \
        "p_shift's fraction is not (1 + hits)/79"
    assert abs(sh["p"] - sh["numerator"] / sh["denominator"]) < 1e-15, \
        "p_shift is not its own numerator over its own denominator"
    assert abs(sh["resolution_floor"] - 1.0 / 79) < 1e-15, \
        "the declared resolution floor is not 1/79"
    assert not sh["at_resolution_floor"], "p_shift is at its floor; re-read (d)"

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
    # THE FIGURE. Portrait, 7.00 in wide: AJE caps portrait figures there and
    # JRSI is no wider, so the constraint holds through the target change.
    # ====================================================================
    fig = plt.figure(figsize=(7.00, 8.55))
    # (d) takes a little more of the bottom row than it did at 6.4 pt: it is
    # the one panel whose content is mostly prose, so it is the one the 7.5 pt
    # floor costs the most. (e) carries four y ticks and a two-line legend and
    # gives the width up without crowding.
    gs = fig.add_gridspec(4, 2, height_ratios=[1.30, 0.94, 0.94, 1.00],
                          width_ratios=[1.24, 0.76], hspace=0.46, wspace=0.34,
                          left=0.132, right=0.985, top=0.960, bottom=0.058)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :], sharex=ax_b)
    ax_d = fig.add_subplot(gs[3, 0])
    ax_e = fig.add_subplot(gs[3, 1])

    def title(ax, s):
        ax.set_title(s, fontsize=8.2, loc="left", pad=4.5, color=INK)

    # ------------------------------------------------ (a) the observation
    a = ax_a
    x = [POS[ym] for ym in YMS]
    y = [MI[ym] for ym in YMS]
    for i0, i1, lab in runs_of([LAB[ym] for ym in YMS]):
        if lab == "term":
            a.axvspan(i0 - .5, i1 + .5, color=TEAL, alpha=.11, lw=0, zorder=0)
        elif lab == "december":
            a.axvspan(i0 - .5, i1 + .5, color=GREY, alpha=.17, lw=0, zorder=0)
    a.plot(x, y, "-", lw=1.15, color=INK, zorder=3)
    a.plot(x, y, "o", ms=2.5, mfc="white", mec=INK, mew=.7, zorder=4)
    a.plot([POS[ym] for ym in CLEAR], [MI[ym] for ym in CLEAR], "o", ms=5.0,
           color=RED, zorder=6)
    a.axhline(FLOOR, color=RED, lw=1.15, ls="--", zorder=5)
    for y_ in YEARS:                       # the level trend, as year medians
        xs = [POS[ym] for ym in YMS if YEAR[ym] == y_]
        a.plot([min(xs) - .5, max(xs) + .5], [YMED[y_]] * 2, "-", lw=1.6,
               color=BLUE, alpha=.75, zorder=2)
    # White plate, like the labels in (b) and (c): at 7.5 pt the line reaches
    # into 2023, where the series spikes through it.
    a.text(0.6, FLOOR * 1.035,
           f"floor: the national survey's own permutation median, "
           f"{FLOOR:.5f} bits", fontsize=7.5, color=RED, va="bottom",
           bbox=dict(fc="white", ec="none", alpha=.88, pad=1.2))
    a.text(16.0, 0.00742,
           f"year medians (blue rules) rise {TREND:.2f}x from "
           f"{min(YEARS)} to {max(YEARS)}", fontsize=7.5, color=BLUE,
           ha="left", va="bottom")
    ticks = [POS[ym] for ym in YMS if ym % 100 == 1]
    a.set_xticks(ticks)
    a.set_xticklabels([str(YMS[t] // 100) for t in ticks], fontsize=7.6)
    a.set_xlim(-1.0, 79.0)
    a.set_ylim(0.0068, 0.0322)
    a.tick_params(axis="y", labelsize=7.5)
    a.set_ylabel("departure from proportionate mixing\n"
                 "age-pair mutual information (bits)", fontsize=7.6)
    title(a, f"(a)  {len(YMS)} monthly matrices, {YMS[0] // 100}-"
             f"{YMS[0] % 100:02d} to {YMS[-1] // 100}-{YMS[-1] % 100:02d}: "
             f"{len(CLEAR)} clear the floor, all of them term months")
    a.grid(alpha=.16, lw=.55, axis="y")
    a.legend(handles=[
        Patch(fc=TEAL, alpha=.11, label="school term (Mar-Jun, Sep-Nov)"),
        Patch(fc=GREY, alpha=.17, label="December: in neither set"),
        Line2D([], [], marker="o", ls="none", ms=5, color=RED,
               label=f"reaches the floor ({len(CLEAR)} of {len(YMS)})"),
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

    # ------------------------------------------- (b) the circular-shift null
    b = ax_b
    ks = list(range(79))
    cols = [TEAL if k == 0 else (GOLD if k in GE else PALE) for k in ks]
    b.vlines(ks, 0, NT, colors=cols, lw=1.9, zorder=2)
    b.plot([k for k in ks if k not in GE and k != 0],
           [NT[k] for k in ks if k not in GE and k != 0], "o", ms=2.4,
           color=GREY, zorder=3)
    b.plot(GE, [NT[k] for k in GE], "o", ms=5.2, color=GOLD, zorder=4)
    b.plot([0], [NT[0]], "o", ms=6.0, color=TEAL, zorder=5)
    b.axhline(17, color=INK, lw=.8, ls=":", zorder=1)
    mean_sh = sh["n_term_over_shifts"]["mean"]
    b.axhline(mean_sh, color=GREY, lw=.9, ls="--", zorder=1)
    b.text(76.2, mean_sh + .45, f"mean over the {len(ks) - 1} shifts, "
                                f"{mean_sh:.2f}", fontsize=7.5, color=GREY,
           ha="right", va="bottom",
           bbox=dict(fc="white", ec="none", alpha=.88, pad=1.2))
    b.annotate("k = 0: the true calendar", xy=(0, NT[0]), xytext=(2.6, 21.0),
               fontsize=7.5, color=TEAL, ha="left", va="center",
               arrowprops=dict(arrowstyle="->", lw=.85, color=TEAL))
    b.annotate(f"k = {', '.join(str(k) for k in GE)} also keep all "
               f"{len(CLEAR)} in term\n(every one a multiple of 6; "
               f"{len(ks)} = 6x12 + 7)",
               xy=(GE[-1], NT[GE[-1]] + .4), xytext=(41.0, 20.4), fontsize=7.5,
               color="#8a5c1f", ha="left", va="center",
               arrowprops=dict(arrowstyle="->", lw=.85, color=GOLD))
    b.set_ylim(0, 24.6)
    b.set_yticks([0, 5, 10, 15, 17])
    b.tick_params(axis="both", labelsize=7.5)
    b.tick_params(axis="x", labelbottom=False)
    b.set_ylabel("clearing months\nlabelled term", fontsize=7.6)
    title(b, f"(b)  the same {len(CLEAR)} months under every rotation of the "
             f"school calendar: {len(GE)} of {len(ks) - 1} alternatives match")
    b.grid(alpha=.16, lw=.55, axis="both")
    note("b", "term months among the clearing months, true calendar", NT[0])
    note("b", "alternative alignments that also reach 17", len(GE))
    note("b", "alternatives in total", len(ks) - 1)
    note("b", "the shifts that match (months)", GE)
    note("b", "every matching shift is a multiple of 6", True)
    note("b", "n_term over the 78 shifts: min / max / mean",
         f"{sh['n_term_over_shifts']['min']} / "
         f"{sh['n_term_over_shifts']['max']} / {mean_sh:.4f}")
    note("b", "term margin, invariant across all 79 rotations",
         anc["A5"]["margin"])
    note("b", "k is p58's forward convention: position i takes the label "
              "of the month k earlier", True)
    note("b", "the same three shifts under the opposite convention",
         dc["shifts_backward"])
    note("b", "the p is the same under either convention", dc["p_forward"])

    # ------------------------------------------------ (c) why 3 is not 3 draws
    c = ax_c
    # Gold means the same thing here as in (b) -- the three shifts that match --
    # rather than a mod-6 rule. The comb is legible without a second colour
    # because the x gridlines already sit at the multiples of six, and a rule
    # that gilded k = 0 mod 6 would leave their mirrors (k = 1 mod 6, equally
    # near-degenerate, agreement(k) = agreement(79-k)) unexplained in grey.
    c.vlines(ks, 0, AGREE, colors=[TEAL if k == 0 else
                                   (GOLD if k in GE else PALE) for k in ks],
             lw=1.9, zorder=2)
    c.plot([k for k in ks if k not in GE and k != 0],
           [AGREE[k] for k in ks if k not in GE and k != 0], "o", ms=2.4,
           color=GREY, zorder=3)
    c.plot(GE, [AGREE[k] for k in GE], "o", ms=5.2, color=GOLD, zorder=4)
    c.plot([0], [AGREE[0]], "o", ms=6.0, color=TEAL, zorder=5)
    c.axhline(CHANCE, color=BLUE, lw=1.05, ls="--", zorder=1)
    c.text(76.2, CHANCE - 2.5, f"chance agreement, {CHANCE:.1f} of {len(ks)}",
           fontsize=7.5, color=BLUE, ha="right", va="top",
           bbox=dict(fc="white", ec="none", alpha=.88, pad=1.2))
    for k in GE:
        c.annotate(f"{AGREE[k]}", xy=(k, AGREE[k]), xytext=(0, 3.2),
                   textcoords="offset points", fontsize=7.5, color="#8a5c1f",
                   ha="center", va="bottom")
    n_above = sum(1 for k in range(1, 79) if AGREE[k] > CHANCE)
    c.text(0.015, 0.985,
           f"a shift by a multiple of six (the gridlines), or its mirror since "
           f"agreement(k) = agreement(79 - k),\nlands the calendar nearly back "
           f"on itself: {n_above} of the {len(ks) - 1} shifts agree above "
           f"chance",
           transform=c.transAxes, fontsize=7.5, color=INK, ha="left", va="top",
           linespacing=1.5)
    c.set_ylim(0, 112)
    c.set_yticks([0, 20, 40, 60, 79])
    c.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78])
    c.tick_params(axis="both", labelsize=7.5)
    c.set_xlim(-1.0, 79.0)
    c.set_xlabel("circular shift k, in months, of the whole school calendar",
                 fontsize=7.6)
    c.set_ylabel("months whose label\nsurvives the shift", fontsize=7.6)
    title(c, "(c)  the rotations are distinct but not independent: agreement "
             "with the true labelling")
    c.grid(alpha=.16, lw=.55, axis="both")
    note("c", "agreement at k = 0 (the identity)", AGREE[0])
    for k in GE:
        note("c", f"agreement at k = {k}", AGREE[k])
    note("c", "chance agreement", CHANCE)
    note("c", "shifts agreeing above chance (derived at draw time)",
         f"{n_above} of {len(ks) - 1}")
    note("c", "distinct label vectors over the 79 rotations",
         anc["A6"]["n_distinct_label_vectors"])
    note("c", "79 is prime, so every rotation is a distinct permutation",
         anc["A6"]["n_is_prime"])
    note("c", "agreement(k) == agreement(79-k)",
         anc["A6"]["symmetric_k_and_n_minus_k"])

    # ------------------------------- (d) what each null assumes, and resolves
    d = ax_d
    floor_p = sh["resolution_floor"]
    ladder = [(1 + h) / 79 for h in range(0, 6)]
    # The floor binds the two ROTATION rows and nothing else: the within-year
    # null is a different test and is not bounded by 1/79, so the band stops
    # below its row rather than running the height of the panel.
    # Top at 1.40, not 1.62: it has to sit above the rotation row it binds and
    # below the within-year row's text, and that text is three lines tall, so
    # at 7.5 pt its bottom line reaches down to y = 1.43.
    d.fill_between([1.1e-6, floor_p], -0.62, 1.40, color=GREY, alpha=.16, lw=0,
                   zorder=0)
    d.plot([floor_p, floor_p], [-0.62, 1.40], color=INK, lw=.9, zorder=2)
    d.text(1.6e-6, 0.55, f"1/{len(ks)} = {floor_p:.4f}\nno {len(ks)}-rotation "
                         f"test\ncan go below this",
           fontsize=7.5, color=INK, ha="left", va="center", linespacing=1.45)
    d.plot([P_WITHIN], [2], "D", ms=6.0, color=BLUE, zorder=4)
    d.plot([sh["p"]], [1], "o", ms=6.8, color=RED, zorder=4)
    d.plot([sb["p"]], [0], "o", ms=6.8, mfc="white", mec=GOLD, mew=1.7,
           zorder=4)
    d.text(P_WITHIN * 2.1, 2.0, f"{P_WITHIN:.2e} -- month labels\nexchangeable "
                                f"inside a year, but\nsemesters are contiguous "
                                f"blocks",
           fontsize=7.5, color=BLUE, ha="left", va="center", linespacing=1.45)
    d.text(sh["p"] * 1.7, 1.0, f"{sh['p']:.4f} = {sh['numerator']}/"
                               f"{sh['denominator']}\none alignment either\n"
                               f"way: {ladder[2]:.4f} / {ladder[4]:.4f}",
           fontsize=7.5, color=RED, ha="left", va="center", linespacing=1.45)
    d.text(sb["p"] * 1.7, 0.0, f"{sb['p']:.4f} = {sb['numerator']}/"
                               f"{sb['denominator']}, but rank 1\nis a tie: "
                               f"D(12) is bitwise D(0)",
           fontsize=7.5, color="#8a5c1f", ha="left", va="center",
           linespacing=1.45)
    d.set_xscale("log")
    # The right end is the width of the TEXT LANE, not a measurement: nothing
    # is measured past 10^0 and there is no tick out there. It was 9.0 while
    # the three labels were set at 6.4 pt; at the 7.5 pt JRSI floor the two
    # right-hand blocks -- anchored in data coordinates beside their markers --
    # ran over the right spine, so the lane is longer and they sit back inside.
    d.set_xlim(1.1e-6, 60.0)
    d.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    d.minorticks_off()      # nothing is measured past 10^0; that lane is text
    d.set_ylim(-0.85, 2.62)
    d.set_yticks([2, 1, 0])
    d.set_yticklabels(["within-year\nnull", "rotation,\ncount",
                       "rotation,\ncontrast"], fontsize=7.5)
    d.tick_params(axis="x", labelsize=7.5)
    d.set_xlabel("p under each null (log scale)", fontsize=7.6)
    title(d, "(d)  neither null dominates, and the\n      rotation has little "
             "resolution to spend")
    d.grid(alpha=.16, lw=.55, axis="x")
    note("d", "within-year hypergeometric p (superseded as the primary)",
         P_WITHIN)
    note("d", "circular-shift p, all 78 alternatives", sh["p"])
    note("d", "circular-shift p as a fraction",
         f"{sh['numerator']}/{sh['denominator']}")
    note("d", "p with one matching alignment fewer", ladder[2])
    note("d", "p with one matching alignment more", ladder[4])
    note("d", "resolution floor of any 79-rotation test", floor_p)
    note("d", "how far the observed p sits above that floor",
         sh["p"] / floor_p)
    note("d", "median-contrast D(0), bits", sb["observed"])
    note("d", "median-contrast p", sb["p"])
    note("d", "rank of D(0) among the 79 rotations", sb["rank_of_observed"])
    note("d", "the rank-1 tie: D(12) is bitwise equal to D(0)", D[12] == D[0])

    # -------------------------------------- (e) what the rotation gives up
    e = ax_e
    xs = list(range(len(YEARS)))
    lo = [pym["min_over_rotations"][str(y)] for y in YEARS]
    hi = [pym["max_over_rotations"][str(y)] for y in YEARS]
    idn = [pym["identity"][str(y)] for y in YEARS]
    e.vlines(xs, lo, hi, color=GREY, lw=5.4, alpha=.45, zorder=1)
    e.plot(xs, idn, "s", ms=4.6, color=INK, zorder=3)
    e.set_xticks(xs)
    e.set_xticklabels([f"'{str(y)[2:]}" for y in YEARS], fontsize=7.5)
    e.set_ylim(2.2, 9.6)
    e.set_yticks([3, 5, 7, 9])
    e.tick_params(axis="y", labelsize=7.5)
    e.set_ylabel("term months in that year", fontsize=7.5)
    e.set_xlabel("year", fontsize=7.6)
    title(e, "(e)  the rotation moves the\n      per-year margins")
    e.grid(alpha=.16, lw=.55, axis="y")
    e.legend(handles=[
        Line2D([], [], marker="s", ls="none", ms=5, color=INK,
               label="true calendar"),
        # "over 79" rather than "over the 79": at 7.5 pt in a panel this narrow
        # the longer string reaches the 2026 bar.
        Line2D([], [], color=GREY, lw=5.4, alpha=.45,
               label="range over 79 rotations")],
        fontsize=7.5, frameon=False, loc="lower left", handlelength=1.2,
        borderpad=.2, labelspacing=.3, bbox_to_anchor=(-0.02, -0.02))
    note("e", "per-year term margins, true calendar",
         [pym["identity"][str(y)] for y in YEARS])
    note("e", "per-year term margins across rotations, overall min / max",
         f"{pym['overall_min']} / {pym['overall_max']}")
    note("e", "the within-year null holds these fixed by construction", True)

    png = f"{FIG}/p63_figure2.png"
    fig.savefig(png, dpi=300)
    if args.pdf:
        # CreationDate omitted on purpose: matplotlib stamps the wall clock into
        # the PDF, which makes two runs of the same figure differ by bytes for a
        # reason that has nothing to do with the figure. Without it the vector
        # file is byte-reproducible, like the sheet beside it.
        fig.savefig(f"{FIG}/p63_figure2.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {png}" + (" (+ .pdf)" if args.pdf else ""))

    print("\n=== the numbers this figure's caption may quote ===")
    for panel, what, value in sheet:
        v = f"{value:.6g}" if isinstance(value, float) else str(value)
        print(f"  ({panel})  {what:<62} {v}")
    out = dict(
        caption_numbers=[dict(panel=p_, what=w, value=v)
                         for p_, w, v in sheet],
        source="results_p58.json (the null) + results_p42.json (the series) "
               "+ results_p51.json (the floor) + results_p54.json (the recount)",
        derived_at_draw_time=[
            "the per-year medians drawn in (a): p42 stores the series, not its "
            f"year medians; their ratio {TREND:.4f} is anchored against the "
            "1.70x published in memo/phase42-monthscope.md",
            "the p ladder (1 + h)/79 in (d), for h one either side of the "
            "observed 3; the denominator is p58's own"],
        refuses="the figure states no significance verdict. p_shift = 0.0506 "
                "is above 0.05 and a factor of four above 1/79, the smallest p "
                "a 79-rotation test can produce; the 3 matching shifts are the "
                "same calendar displaced by the 79 = 6x12+7 seam, not three "
                "independent draws; and the secondary contrast's rank 1 is a "
                "tie with k = 12. What the figure carries is a descriptive "
                "alignment count.",
        not_drawn="the 2020 natural control and the per-band decomposition. "
                  "Figure 3(c) already annotates 2020 on the same 79 months, "
                  "and p41's per-year panel would have to redraw the "
                  "within-year relabelling bands this round demoted.")
    with open(f"{ROOT}/eda/results_p63.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {ROOT}/eda/results_p63.json")


if __name__ == "__main__":
    main()
