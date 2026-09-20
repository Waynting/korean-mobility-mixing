#!/usr/bin/env python
"""Phase 48 — Figure 1, the study design and processing pipeline.

This was the only figure in `paper/paper_structure.md` §7 with no material at
all. It is a schematic rather than a plot, which is exactly why it is the figure
most likely to drift: a hand-drawn box saying "10.2 billion rows" keeps saying it
long after the inventory moves.

So the schematic is drawn, but EVERY NUMBER ON IT IS READ FROM A RESULTS FILE at
draw time — the span, the file and row counts, the masking rate and which bands
it touches, the measured fill, the scale ladder. Nothing here
is typed as a literal. If the inventory changes and this figure is not rerun, the
next run disagrees with itself and the assertions below stop it.

Korean terms are romanised on the figure. The Hangul renders as tofu in every
font matplotlib ships, and a submission figure that depends on a locally
installed CJK font is a figure that breaks on the typesetter's machine.

The one number that is aggregated rather than read is the overall masking rate:
`results_p16.json` stores it per age band with cell counts, not as a total, so it
is the count-weighted mean of those. That is a display aggregate of stored
values, not a new estimate.

Two things this figure says are load-bearing and easy to get wrong:

  * masking is a THREE-way fact -- 20-44 material, 45-64 a trace, 0-19 and 65+
    exactly none -- and the two outer runs are found structurally, never by
    indexing into the band list. An earlier version said "65-69 and above lose
    exactly none" because it took `zero[-4]`, which silently dropped 0-19 and
    would have pointed somewhere else the moment any rate moved. The figure now
    prints only "nearly all in 20-44", so the three-way split is carried by the
    caption and by the payload; the partition is still built and asserted here,
    because `p31` gates it as a partition and the sentence "nearly all" is only
    true while the material block holds essentially every masked cell -- which
    is now an assert, not a hope.
  * Claim 1's box quotes a COUNT, not a p. See the note beside the p58 read.

WHY THE LAYOUT WAS REBUILT (2026-09-03)
---------------------------------------
The previous version was five stacked text boxes and three paragraph-length
claim boxes: 376 words, four hues, no legend, and a reading order that only
worked if you already knew the paper. It was a slide. This one is built to the
rules a journal schematic is actually held to:

  * ONE DIRECTION. Everything flows top to bottom. Two inputs at the top merge
    once, the spine runs down, and it fans once into the three claims. No arrow
    crosses another and none doubles back.
  * ONE SHAPE, THREE COLOURS. Every node is the same rounded rectangle and the
    colour carries the category, declared in a legend: grey is a data source
    (the release AND the three external anchors -- they are data, and saying so
    in colour is the point), teal is estimation, blue is a claim. What the
    release fails to record is not a node at all, so it is drawn as the one
    thing that is not a node: a dashed, unfilled annotation hanging off the box
    it is about.
  * NOUN PHRASES, NOT SENTENCES. Every box is a title and at most two short
    lines. The detail those sentences carried -- the 10.9%-69.5% band range,
    the 0.13% trace, the 1.5 midpoint an interval censor would have used -- is
    in the caption, which is where a journal figure keeps it. 376 words down to
    about 170.
  * A GRID, IN INCHES. The layout used to be in axes fractions while the type
    was in points, so the two moved against each other and the figure height
    could not be changed without redoing the arithmetic. The axes now spans the
    canvas with `xlim`/`ylim` set to the figure size, so ONE DATA UNIT IS ONE
    INCH: leading, padding, corner radii and the claim badges are all in the
    same unit as the type, and the badges are round rather than ovals.

JRSI's figure rules are what the sizes answer to: text in Times at 9-11 pt with
an absolute 7.5 pt floor (`figstyle.finish` refuses the file otherwise), drawn
at the width it will print at so the size written here is the size on the page,
and a vector copy from the same draw via `--pdf`.

    python eda/p48_fig1.py [--pdf]
"""
import argparse
import json
import math
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
from figstyle import WIDTH, finish
from paths import FIG, ROOT

# Three categories and one accent, which is the whole palette. The three node
# colours are the project's own (p47/p55/p56/p63 draw from the same four), so
# Figure 1 does not arrive in a different colour language from Figures 2-6 and S1;
# GOLD is dropped here because the external anchors are data, not a fourth kind
# of thing. RED is never a node: it marks the one dashed annotation.
INK = "#1b1b1b"
GREY, GREY_F = "#5F6B72", "#F2F4F5"
TEAL, TEAL_F = "#0E7C86", "#E4F0F1"
BLUE, BLUE_F = "#4C6E8A", "#ECF0F4"
RED = "#A8434E"
STAT = "#4C5860"

# JRSI wants figure text in Times at 9-11 pt and refuses anything under 7.5 pt,
# so 7.5 is the floor for every size below. STIXGeneral is the serif: it is
# Times-metric AND it ships inside matplotlib, so this renders the same on a
# machine with no Times installed. A figure that depends on a locally installed
# font breaks on the typesetter's machine -- the same reason the Korean on this
# schematic is romanised rather than set in a CJK font.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Times New Roman",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

TITLE_PT, BODY_PT = 10.0, 9.0
LEAD = 0.1875         # inches between baselines: 13.5 pt against 9 pt type
PADX, PADY = 0.12, 0.11
GAPX = 0.030          # extra space before a block that starts a new idea
DASH = (0, (2.6, 1.8))
SEP = "\u2002\u00b7\u2002"   # en space, middot, en space


def load(name):
    return json.load(open(f"{ROOT}/eda/results_{name}.json"))


def thin(n):
    """Thousands separated by U+2009 THIN SPACE, never by a comma.

    JRSI: "In numbers, thousands separated by thin spaces, not commas (eg
    10 000 000)". The manuscript and the SI already carry 24 of these and no
    comma-separated number anywhere; this figure was the only place in the
    project that still printed 1,896 and 10,186,891,962.
    """
    return f"{n:,}".replace(",", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    inv, p16 = load("inventory"), load("p16")
    p29 = load("p29")
    p37, p38 = load("p37"), load("p38")
    # Claims 1 and 3 quote numbers, so they are read like everything else here.
    # p54 is the semester count against the corrected floor; p52 is the R0 sweep
    # on the corrected national vector; p58 is the circular-shift test that
    # replaced p54's within-year p as the thing Claim 1 may say out loud.
    p52, p54, p58 = load("p52"), load("p54"), load("p58")
    p64, p33 = load("p64"), load("p33")

    # ---------------------------------------------------------------- numbers
    span = f"{inv['span_first']}–{inv['span_last']}"
    months, files = inv["span_months"], inv["parquet_files"]
    rows = inv["parquet_rows"]
    assert inv["months_missing"] == 0 and inv["months_complete"] == months, \
        "inventory: the span is no longer complete, the caption cannot say 79 of 79"

    by_age = p16["masking"]["by_age"]
    n_tot = sum(b["n"] for b in by_age)
    mask_rate = sum(b["n"] * b["rate"] for b in by_age) / n_tot
    # Masking is a THREE-way fact, not an interval and not a two-way split:
    # 20-44 carry 11%-70% of cells masked, 45-64 a trace of order 1e-3 to 1e-5,
    # and 0-19 together with 65+ exactly zero. Quoting a first-to-last range
    # over the nonzero bands would say "20-24 to 60-64", which is wrong twice;
    # collapsing to hit/zero forces the zero run to be named by a positional
    # index into the band list, which is the failure this repo forbids by name.
    # So all three classes are built here and all three are rendered.
    MATERIAL = 0.01
    bands = [b["age_l"] for b in by_age]
    rate_of = {b["age_l"]: b["rate"] for b in by_age}
    n_of = {b["age_l"]: b["n"] for b in by_age}
    material = [b for b in bands if rate_of[b] >= MATERIAL]
    trace = [b for b in bands if 0.0 < rate_of[b] < MATERIAL]
    none = [b for b in bands if rate_of[b] == 0.0]

    def runs(sub):
        """`sub` as contiguous runs of positions in `bands`."""
        out = []
        for i in sorted(bands.index(s) for s in sub):
            if out and i == out[-1][-1] + 1:
                out[-1].append(i)
            else:
                out.append([i])
        return out

    assert material, "p16: no band is materially masked, the trap list is wrong"
    assert len(runs(material)) == 1, \
        "p16: the materially masked bands are no longer contiguous"
    assert len(runs(trace)) == 1, \
        "p16: the trace-masked bands are no longer contiguous"
    assert len(material) + len(trace) + len(none) == len(bands) == 16, \
        "p16: material/trace/none no longer partition the 16 age bands"
    none_runs = runs(none)
    assert len(none_runs) == 2, \
        "p16: the unmasked bands are no longer exactly two runs, one at each end"
    # Structural, never positional: the low run is the one that ends before the
    # material run begins, the high run the one that begins after the trace ends.
    none_low = [bands[i] for i in none_runs[0]]
    none_high = [bands[i] for i in none_runs[1]]
    assert (none_runs[0][-1] < bands.index(material[0])
            and bands.index(trace[-1]) < none_runs[1][0]), \
        "p16: the four runs are no longer ordered none / material / trace / none"

    def run_label(run):
        """A label for a contiguous run of bands. '45-49'..'60-64' -> '45-64';
        an open top band keeps its plus, '65-69'..'80+' -> '65+'."""
        lo = run[0].split("-")[0].rstrip("+")
        return f"{lo}+" if run[-1].endswith("+") else f"{lo}-{run[-1].split('-')[1]}"

    mat_lo = min(rate_of[b] for b in material)
    mat_hi = max(rate_of[b] for b in material)
    trace_max = max(rate_of[b] for b in trace)
    # A ceiling, so "under X%" is a true bound however the rates move. The max
    # over the trace run is 50-54's 1.28e-03, NOT 45-49's 6.95e-04: the run is
    # not monotone in age, so the first trace band is not the largest one.
    trace_cap = math.ceil(trace_max * 1e4) / 1e2
    # THE FIGURE SAYS "nearly all in 20-44" AND THIS IS WHAT MAKES THAT TRUE:
    # the share of masked CELLS (not of bands) that fall in the material block.
    # The old figure spelled the whole three-way split out on the page, which
    # cost four lines to say something the caption already says better. The
    # short form is only honest while this share is essentially one, so the
    # threshold is asserted rather than eyeballed.
    masked_cells = {b: n_of[b] * rate_of[b] for b in bands}
    mat_share = sum(masked_cells[b] for b in material) / sum(masked_cells.values())
    assert mat_share > 0.99, \
        f"p16: only {mat_share:.4f} of masked cells are in {run_label(material)}; " \
        f"the figure may no longer say 'nearly all'"
    fill = p29["202012"]["per_cell_mean"]
    fill_share = p29["202012"]["measured_share"]

    # Until 2026-09-21 this read p36's check count for a verification strip
    # under the claims ("checked three ways ... 692 of 692"). The strip was the
    # paper's own quality control advertising itself inside a data-flow figure
    # -- the same kind of thing as the "only R0 previously run" the advisor
    # struck from Figure 7 -- so it is gone, and with it the p36 read and the
    # `p36_checks` key this sheet used to carry. The count still lives where a
    # reader meets it in prose: section 2.6 and Data accessibility, both gated.

    n_dong = p38["adjacency"]["n_nodes"]
    assert p37["ladder"]["n_monotone"] == months, \
        "p37: the ladder no longer holds in every month"

    # Claim 1's corroboration: months clearing the corrected national floor, and
    # the fact that carries it -- every one of them is a school-term month.
    n_clear = p54["corrected"]["n_at_or_above"]
    assert p54["corrected"]["all_clearing_are_term"], \
        "p54: the clearing months are no longer all term months"
    p_term = p54["corrected"]["p_term_ge"]
    # THE FIGURE NO LONGER QUOTES A p FOR THIS. p54's within-year null treats the
    # month labels inside a year as exchangeable, but school terms are contiguous
    # blocks, so that null is inflated. p58 rotates the calendar instead, and its
    # smallest attainable p is 1/79 = 0.0127 -- a p of 0.0506 sits a factor of
    # four above its own floor and cannot carry a claim. What survives is the
    # alignment count, which is a description rather than a test, so that is what
    # the box says. p_term stays in the payload for traceability only.
    shift = p58["p_shift"]
    n_alt = shift["denominator"] - 1        # k in 1..78; k = 0 is the true calendar
    n_match = shift["n_shifts_at_or_above"]
    # WHAT THE BOX DRAWS IS NOT n_match OF n_alt, AND THAT WAS THE BUG. It said
    # "3 of 78 alternative calendar alignments match", contradicting 3.1 head on.
    # The 79 rotations are not 79 calendars: the school labelling is a function
    # of month-of-year, so k and k + 12 are the same calendar, and the identity
    # class {0, 12, 24, 36, 48, 60, 72} holds every rotation that reproduces the
    # result. Counting classes instead of offsets gives 11 alternatives of which
    # none match, which is what 3.1 says. The raw pair stays in the payload
    # because the 08-27 round's captions were sent quoting it.
    n_cal_alt = p64["effective_alignments"]["E_exact"] - 1
    n_cal_match = p64["deduplicated"]["exact"][
        "n_nonidentity_classes_reaching_observed"]
    cal_subj = "none" if n_cal_match == 0 else f"{n_cal_match}"
    assert p58["anchors"]["A2"]["reproduces_p54"], \
        "p58: the shift test no longer reproduces p54's clearing months"
    assert p58["anchors"]["A2"]["n_at_or_above"] == n_clear == shift["n_clearing"], \
        "p58 and p54 disagree on how many months clear the corrected floor"
    assert len(shift["n_term_by_k"]) == n_alt + 1, \
        "p58: the rotation set is not one true calendar plus n_alt offsets"
    assert p64["deduplicated"]["exact"]["E"] == n_cal_alt + 1, \
        "p64: the distinct calendars are not one true one plus n_cal_alt"
    # Claim 3's: where over R0 the consequence is largest. The grid starts at
    # 1.1, so this is an interval the grid actually covers, not "the threshold".
    argmax = sorted(set(p52["point_summary"]["argmax"].values()))
    assert p52["declaration"]["n_boot"] == 200, \
        "p52 was not run at the pinned 200 draws"
    r0_lo, r0_hi = min(argmax), max(argmax)
    n_flip = p52["answer"]["n_flips"]
    n_verdict = p52["answer"]["n_verdicts"]
    # CLAIM 3 IS ABOUT COVERAGE, AND "n of m verdicts" IS NOT A COVERAGE NUMBER.
    # The box used to print the two side by side under one "it", which reads as
    # though the coverage family moved the verdicts. It does not: n_flip counts
    # regret against its own bootstrap null (p52), while the coverage family is
    # p33's theta sweep. The two are named separately below, each against the
    # figure that carries it.
    tau_min = min(r["kendall_tau_vs_theta0"]
                  for rows in p33["R2_theta_family"].values() for r in rows)
    assert 0.0 <= tau_min <= 1.0, \
        "p33's Kendall tau against theta = 0 is out of range"

    # ------------------------------------------------------------------ canvas
    # ONE DATA UNIT IS ONE INCH. `set_position` gives the axes the whole canvas
    # and the limits are the canvas size, so every constant below -- leading,
    # padding, corner radius, badge radius, arrow length -- is in the same unit
    # as the type, and a circle drawn here is a circle rather than an oval. It
    # also means the crop at the end is computed straight from the layout
    # cursor with no scale factor in between. H is generous and the figure is
    # cropped to what the layout used; the type size is unaffected by H, which
    # was the point of moving off axes fractions.
    W, H = WIDTH, 8.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    M = 0.06
    L, R = M, W - M
    CX = W / 2                     # the spine, and the middle claim's centre
    LW = 3.90                      # the release and its caveats
    RX, RW = L + LW + 0.18, R - (L + LW + 0.18)   # the external anchors

    # TEXT IS MEASURED, NOT ESTIMATED. The first cut of this layout wrapped on
    # "Times at 9 pt averages 0.0063 in per character", which is close enough
    # to be wrong quietly: one trap line ran a word past the dashed box and
    # nothing complained, because a character count cannot know that "(hap),"
    # is wider than "in a". `_w` asks the renderer, `wrap_to` wraps on what it
    # answers, and `panel` asserts every line it draws actually fits the box it
    # draws it in -- so an overflow becomes a failed run rather than a figure
    # that looks finished.
    renderer = fig.canvas.get_renderer()

    def _w(t, size, weight):
        h = ax.text(0, 0, t, fontsize=size, fontweight=weight)
        width = h.get_window_extent(renderer=renderer).width / fig.dpi
        h.remove()
        return width

    def wrap_to(t, avail, size, weight, bullet=False):
        # ASCII spaces only: `str.split()` also breaks on the U+2009 inside
        # "10 186 891 962" and on the U+2002 that spaces the separators, so it
        # could wrap a row count in half and it flattened every gap on the
        # figure to one space.
        first, rest = ("\u00b7 ", "   ") if bullet else ("", "")
        words = [w for w in t.split(" ") if w]
        out, cur = [], first + (words or [""])[0]
        for word in words[1:]:
            trial = f"{cur} {word}"
            if _w(trial, size, weight) <= avail:
                cur = trial
            else:
                out.append(cur)
                cur = rest + word
        out.append(cur)
        return out

    def flow(t, width, size=BODY_PT, weight="normal", color=INK, gap=0.0,
             bullet=False, dx=0.0):
        """One string as wrapped rows: (line, size, weight, colour, gap, dx).

        `width` is the box, not the measure: the padding and `dx` come off it
        here, so a caller never has to keep the two in step. `dx` indents a
        whole block past the left padding, which is what lets a claim's title
        start to the right of its badge instead of underneath it.
        """
        avail = width - 2 * PADX - dx
        out = wrap_to(t, avail, size, weight, bullet)
        return [(ln, size, weight, color, gap if i == 0 else 0.0, dx)
                for i, ln in enumerate(out)]

    def height(rows):
        return 2 * PADY + LEAD * len(rows) + sum(r[4] for r in rows)

    def panel(x, top, width, rows, face, edge, h=None, lw=1.0, dashed=False):
        """One node: the same rounded rectangle every time, colour carries the
        category. Returns the bottom edge."""
        h = height(rows) if h is None else h
        ax.add_patch(FancyBboxPatch(
            (x, top - h), width, h,
            boxstyle="round,pad=0,rounding_size=0.055",
            fc=face, ec=edge, lw=lw, zorder=2,
            linestyle=DASH if dashed else "solid"))
        y = top - PADY
        for ln, size, weight, color, gap, dx in rows:
            y -= gap
            over = _w(ln, size, weight) - (width - 2 * PADX - dx)
            assert over <= 1e-9, \
                f"figure 1: {ln!r} overruns its box by {over:.3f} in"
            ax.text(x + PADX + dx, y, ln, fontsize=size, fontweight=weight,
                    color=color, ha="left", va="top", zorder=5, linespacing=1.4)
            y -= LEAD
        return top - h

    def down(x, y0, y1, head=True, color=INK, lw=1.0, style="solid"):
        if head:
            ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                         mutation_scale=8, lw=lw, color=color,
                                         zorder=3, shrinkA=0, shrinkB=0))
        else:
            ax.add_line(Line2D([x, x], [y0, y1], color=color, lw=lw, zorder=3,
                               linestyle=DASH if style == "dashed" else "solid"))

    def across(x0, x1, yy, color=INK, lw=1.0):
        ax.add_line(Line2D([x0, x1], [yy, yy], color=color, lw=lw, zorder=3))

    # An en dash for a range, because the caption sets 20-24 through 40-44 that
    # way and the band labels arrive from p16 hyphenated.
    mat_range = run_label(material).replace("-", "\u2013")

    y = H - 0.10

    # ---- input 1: the release, and what it does not record ----------------
    src = (flow("Seoul Living Migration (Saenghwal Idong)", LW, TITLE_PT, "bold")
           + flow("dong \u00d7 weekday \u00d7 arrival hour \u00d7 sex "
                  "\u00d7 age band \u00d7 trip type", LW, gap=GAPX)
           + flow(f"{span}{SEP}{months} of {months} months{SEP}{n_dong} dong", LW, weight="bold", gap=GAPX)
           + flow(f"{thin(files)} parquet files{SEP}{thin(rows)} rows",
                  LW, weight="bold"))
    # The spine leaves the release down its left margin and the caveats hang
    # off to the right of it, indented. Before this the leg started under the
    # dashed box, which drew the release's shortcomings flowing INTO the
    # estimator -- the one thing on this figure that is not an input.
    LX, TIN = L + 0.28, 0.52
    src_bot = panel(L, y, LW, src, GREY_F, GREY, lw=1.1)

    # Noun phrases, not sentences: what each of these costs the estimator is
    # the caption's job and 2.1's, not a text box's.
    TW = LW - TIN
    traps = flow("Not recorded", TW, TITLE_PT, "bold", RED)
    for t in ["no date column, one row per weekday",
              "monthly sums over like weekdays (hap), never daily means",
              f"{mask_rate:.1%} of cells masked, nearly all of it in {mat_range}",
              "age-varying expansion weights, unrecorded",
              "a code space frozen at 2020, districts since moved"]:
        traps += flow(t, TW, gap=0.018, bullet=True)
    # A dashed leader, no arrowhead: the caveats are a property of the box
    # above, not something flowing out of it.
    down(L + TIN + (LW - TIN) / 2, src_bot, src_bot - 0.10, head=False,
         color=RED, lw=0.9, style="dashed")
    left_bot = panel(L + TIN, src_bot - 0.10, LW - TIN, traps, "#FFFFFF", RED,
                     lw=1.0, dashed=True)

    # ---- input 2: the external anchors ------------------------------------
    anch = flow("External anchors", RW, TITLE_PT, "bold")
    for a in [f"official district file: {fill:.2f} per masked cell, "
              f"{fill_share:.1%} of volume",
              "contact survey: pins \u03b2 without identifying it",
              "resident registration: denominators, coverage profile"]:
        anch += flow(a, RW, gap=0.018, bullet=True)
    right_bot = panel(RX, y, RW, anch, GREY_F, GREY, lw=1.1)

    # ---- the legend, in the space the shorter input column leaves ---------
    # It goes here rather than under the figure because this is where the dead
    # space is, and a key belongs beside the things it names.
    keys = [(GREY_F, GREY, "solid", "data source"),
            (TEAL_F, TEAL, "solid", "estimation"),
            (BLUE_F, BLUE, "solid", "claim"),
            ("#FFFFFF", RED, "dashed", "what the release does not record")]
    ly = right_bot - 0.34
    for face, edge, style, label in keys:
        ax.add_patch(FancyBboxPatch((RX + 0.02, ly - 0.065), 0.15, 0.13,
                                    boxstyle="round,pad=0,rounding_size=0.035",
                                    fc=face, ec=edge, lw=1.0, zorder=4,
                                    linestyle=DASH if style == "dashed" else "solid"))
        ax.text(RX + 0.24, ly, label, fontsize=BODY_PT, color=INK,
                ha="left", va="center", zorder=5)
        ly -= 0.235

    # ---- the two inputs merge, once ---------------------------------------
    # The anchors' leg runs down the right margin rather than down the middle
    # of its own column, because the legend is in that column.
    rail = min(left_bot, right_bot) - 0.28
    assert rail < ly + 0.10, "the merge rail now runs through the legend"
    down(LX, src_bot, rail, head=False)
    down(R - 0.18, right_bot, rail, head=False)
    across(LX, R - 0.18, rail)
    y = rail - 0.24
    down(CX, rail, y)

    # ---- estimation -------------------------------------------------------
    coa = (flow("Co-arrival matrix", R - L, TITLE_PT, "bold", TEAL)
           + flow("same dong, same arrival hour, same weekday  \u2192  "
                  "an age \u00d7 age matrix A", R - L, gap=GAPX)
           + flow(f"at three spatial scales: {n_dong} dong{SEP}25 districts{SEP}1 city", R - L))
    y = panel(L, y, R - L, coa, TEAL_F, TEAL, lw=1.1)
    down(CX, y, y - 0.24)
    y -= 0.24

    exc = (flow("Excess over proportionate mixing", R - L, TITLE_PT, "bold", TEAL)
           + flow("assortativity" + SEP + "normalised mutual information" + SEP + "rank", R - L, gap=GAPX))
    y = panel(L, y, R - L, exc, TEAL_F, TEAL, lw=1.1)

    # ---- one fan into the three claims ------------------------------------
    CGAP = 0.10
    ccw = (R - L - 2 * CGAP) / 3
    cxs = [L + i * (ccw + CGAP) + ccw / 2 for i in range(3)]
    assert abs(cxs[1] - CX) < 1e-9, "the middle claim is no longer under the spine"
    fan = y - 0.17
    down(CX, y, fan, head=False)
    across(cxs[0], cxs[2], fan)
    for x in cxs:
        down(x, fan, fan - 0.15)
    y = fan - 0.15

    claims = [
        ("1", "The instrument responds",
         "it tracks a school calendar it never sees",
         f"{n_clear} clearing months, all in term; "
         f"{cal_subj} of {n_cal_alt} alternative calendars match",
         "Fig 2"),   # Fig 3 has only (a) and (b); the 2020 panel is Fig 2(c)
        ("2", "It still reads near zero",
         "at every spatial and age resolution",
         "an identity bounds what it can read at all",
         "Fig 3, Fig 4, Fig 5"),
        ("3", "The ranking moves",
         "across the coverage family",
         f"Kendall tau of the leading eigenvector falls to {tau_min:.3f} "
         f"(Fig 6); separately, {n_flip} of {n_verdict} allocation verdicts "
         f"clear their own null, worst at $R_0$ {r0_lo:g}\u2013{r0_hi:g} "
         f"(Fig S1)",
         "Fig 6, Fig S1"),   # the R0 sweep moved to the SI as Figure S1
    ]
    BADGE, BOFF = 0.085, 0.04
    bodies = []
    for _, head, sub, stat, _ in claims:
        # NOT `rows`: that name holds the parquet row count the source box
        # prints and the payload records, and shadowing it here once put a list
        # of wrapped text lines into results_p48.json's `parquet_rows`.
        # The title clears the badge by `dx` and wraps to the measure that
        # leaves, so the two can never overlap however the wording changes.
        body = flow(head, ccw, TITLE_PT, "bold", dx=2 * BADGE + BOFF)
        body += flow(sub, ccw, gap=GAPX)
        body += flow(stat, ccw, color=STAT, gap=0.024)
        bodies.append(body)
    ch = max(height(b) for b in bodies) + LEAD + 0.02   # + the figure pointer
    for i, (n, _, _, _, figs) in enumerate(claims):
        x = L + i * (ccw + CGAP)
        panel(x, y, ccw, bodies[i], BLUE_F, BLUE, h=ch, lw=1.1)
        cy = y - PADY - LEAD / 2 + 0.01
        ax.add_patch(Circle((x + PADX + BADGE, cy), BADGE, fc=BLUE, ec="none",
                            zorder=6))
        ax.text(x + PADX + BADGE, cy, n, fontsize=BODY_PT, fontweight="bold",
                color="white", ha="center", va="center", zorder=7)
        ax.text(x + PADX, y - ch + PADY, figs, fontsize=BODY_PT,
                fontweight="bold", color=BLUE, ha="left", va="bottom", zorder=5)
    y -= ch

    # The verification strip that used to hang under the claims (2026-09-03 to
    # 2026-09-21) is not drawn any more; see the note where p36 was read.

    # Crop to what the layout actually used. Data units are inches and the axes
    # fills the canvas, so the cursor IS the crop edge -- no scale factor.
    assert y > 0.20, "figure 1 overflows the canvas; raise H"
    crop = Bbox.from_extents(0, y - 0.10, W, H)
    out = f"{FIG}/p48_figure1.png"
    mode = finish(fig, out, pdf=args.pdf, bbox_inches=crop)
    plt.close(fig)
    print(f"  -> {out}   ({W:.2f} x {crop.height:.2f} in, {mode}, "
          f"content bottom at y={y:.3f} in)")
    assert abs(crop.width - WIDTH) < 1e-9, \
        "the crop is not the full reproduction width"

    payload = dict(span=span, months=months, parquet_files=files, parquet_rows=rows,
                   dong=n_dong, masking_rate_weighted=round(float(mask_rate), 6),
                   masked_bands_material=material,
                   masked_band_rate_range=[round(mat_lo, 4), round(mat_hi, 4)],
                   # The three-way split the caption renders, each class with the
                   # label it uses, so a caption can be gated against this file
                   # instead of against the PNG. The figure itself now says only
                   # "nearly all in 20-44"; `masked_material_cell_share` is what
                   # licenses that phrase and is asserted above.
                   masked_bands_trace=trace,
                   masked_band_trace_max=trace_max,
                   masked_band_trace_max_band=max(trace, key=rate_of.get),
                   masked_band_trace_cap_pct=trace_cap,
                   masked_bands_none_low=none_low,
                   masked_bands_none_high=none_high,
                   masked_span_material=f"{material[0]} to {material[-1]}",
                   masked_span_trace=run_label(trace),
                   masked_span_none=f"{run_label(none_low)} and {run_label(none_high)}",
                   masked_bands_exactly_zero=none,
                   masked_material_cell_share=round(float(mat_share), 6),
                   measured_fill=fill, measured_fill_share=fill_share,
                   # The claim boxes quote these, so they belong in the sheet
                   # too -- this file IS the record of what the figure shows,
                   # and a number on the figure that is not in the sheet is a
                   # number the caption cannot be checked against.
                   claim1_months_clearing=n_clear,
                   # Kept for traceability, NOT drawn: see the note above where
                   # p58 is read. The figure quotes the alignment count instead.
                   claim1_p_term=p_term,
                   claim1_shift_matches=n_match,
                   claim1_shift_alternatives=n_alt,
                   claim1_calendars_matching=n_cal_match,
                   claim1_calendars_alternative=n_cal_alt,
                   claim1_p_shift=shift["p"],
                   claim3_r0_argmax_range=[r0_lo, r0_hi],
                   claim3_flips=n_flip, claim3_verdicts=n_verdict,
                   claim3_kendall_tau_min=tau_min)
    with open(f"{ROOT}/eda/results_p48.json", "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
    print("\n=== the numbers on the figure, for the caption ===")
    for k, v in payload.items():
        print(f"  {k:<30} {v}")
    print("\nwrote results_p48.json")


if __name__ == "__main__":
    main()
