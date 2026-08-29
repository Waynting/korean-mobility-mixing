#!/usr/bin/env python
"""Phase 48 — Figure 1, the study design and processing pipeline.

This was the only figure in `paper/paper_structure.md` §7 with no material at
all. It is a schematic rather than a plot, which is exactly why it is the figure
most likely to drift: a hand-drawn box saying "10.2 billion rows" keeps saying it
long after the inventory moves.

So the schematic is drawn, but EVERY NUMBER ON IT IS READ FROM A RESULTS FILE at
draw time — the span, the file and row counts, the masking rate and which bands
it touches, the measured fill, the gate tallies, the scale ladder. Nothing here
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
    would have pointed somewhere else the moment any rate moved.
  * Claim 1's box quotes a COUNT, not a p. See the note beside the p58 read.

Layout is portrait and 7 inches wide because AJE caps portrait figures there
(checked 2026-08-23, see paper_structure §4).

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

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
from paths import FIG, ROOT

INK, TEAL, RED, BLUE, GOLD = "#222222", "#0E7C86", "#A8434E", "#4C6E8A", "#BC8034"
FACE, TRAP = "#F4F6F7", "#FBF1F2"

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


def load(name):
    return json.load(open(f"{ROOT}/eda/results_{name}.json"))


def box(ax, x, y, w, h, face, edge, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=face, ec=edge, lw=lw, zorder=2))


def arrow(ax, x, y0, y1, color=INK):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.1, color=color, zorder=3))


def txt(ax, x, y, s, size=7.5, weight="normal", color=INK, ha="left", va="top"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=4, linespacing=1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    inv, p16 = load("inventory"), load("p16")
    p29, p36 = load("p29"), load("p36")
    p37, p38 = load("p37"), load("p38")
    # Claims 1 and 3 quote numbers, so they are read like everything else here.
    # p54 is the semester count against the corrected floor; p52 is the R0 sweep
    # on the corrected national vector; p58 is the circular-shift test that
    # replaced p54's within-year p as the thing Claim 1 may say out loud.
    p52, p54, p58 = load("p52"), load("p54"), load("p58")

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
    fill = p29["202012"]["per_cell_mean"]
    fill_share = p29["202012"]["measured_share"]

    gates_n = p36["n_checks"]
    assert p36["n_fail"] == 0, "p36: the independent recompute has failures"
    # p31 is a pass/fail gate script and writes no results file, so the box
    # carries it without a number; p36 is the one with a tally. Neither phase
    # number is drawn: they are this repo's bookkeeping, not the reader's.

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
    assert p58["anchors"]["A2"]["reproduces_p54"], \
        "p58: the shift test no longer reproduces p54's clearing months"
    assert p58["anchors"]["A2"]["n_at_or_above"] == n_clear == shift["n_clearing"], \
        "p58 and p54 disagree on how many months clear the corrected floor"
    assert len(shift["n_term_by_k"]) == n_alt + 1, \
        "p58: the rotation set is not one true calendar plus n_alt alternatives"
    # Claim 3's: where over R0 the consequence is largest. The grid starts at
    # 1.1, so this is an interval the grid actually covers, not "the threshold".
    argmax = sorted(set(p52["point_summary"]["argmax"].values()))
    assert p52["declaration"]["n_boot"] == 200, \
        "p52 was not run at the pinned 200 draws"
    r0_lo, r0_hi = min(argmax), max(argmax)
    n_flip = p52["answer"]["n_flips"]
    n_verdict = p52["answer"]["n_verdicts"]

    # ------------------------------------------------------------------ canvas
    # Everything is laid out from a y-cursor rather than fixed coordinates, so
    # copy edits move the boxes instead of colliding with them.
    W, H = 7.0, 9.4
    fig, ax = plt.subplots(figsize=(W, H))
    # The axes fills the canvas exactly, so axes fractions are figure fractions
    # and the final crop below can be computed from the layout cursor.
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    L, R = .025, .975
    LEAD = .0175          # line height in axes units at the body size
    PAD = .012

    def block(x, y, lines, size=7.5, weight="normal", color=INK, lead=LEAD):
        for ln in lines:
            txt(ax, x, y, ln, size, weight, color)
            y -= lead
        return y

    def wrap(s, width):
        return textwrap.wrap(s, width) or [""]

    y = .985

    # ---- 1 source ---------------------------------------------------------
    top = y
    y -= PAD
    y = block(L + .015, y, ["SOURCE"], 8.0, "bold", BLUE)
    y -= .004
    y = block(L + .015, y, [
        "Seoul Living Migration (Saenghwal Idong), administrative-dong product",
        "dong x weekday x arrival hour x sex x age band x trip type x volume"], 7.5)
    y -= .004
    y = block(L + .015, y, [
        f"{span}   ·   {months} of {months} months complete   ·   {n_dong} dong",
        f"{files:,} parquet files   ·   {rows:,} rows"], 7.5, "bold")
    y -= PAD
    box(ax, L, y, R - L, top - y, FACE, BLUE, 1.2)

    y -= .026

    # ---- 2 the traps ------------------------------------------------------
    top = y
    y -= PAD
    # "does not record", not "does not tell you": the same wording as §2.1
    # and the caption, and one less instrument with intentions.
    y = block(L + .015, y, ["WHAT THE PRODUCT DOES NOT RECORD"], 8.0, "bold", RED)
    y -= .004
    traps = [
        "no date column: a month is one row per weekday, not per day",
        "the volume column is the within-month sum over like weekdays "
        "(hap), not a daily mean",
        f"cells below 3 are masked, {mask_rate:.0%} of cells overall, and the "
        f"loss is not spread evenly: {material[0]} to {material[-1]} lose "
        f"{mat_lo:.0%}-{mat_hi:.0%} of their cells, {run_label(trace)} lose "
        f"under {trace_cap:g}%, and {run_label(none_low)} and {run_label(none_high)} "
        f"lose exactly none",
        "expansion weights vary with age and are not recorded in the data",
        "the code space is frozen at 2020 while the districts have moved",
    ]
    for t in traps:
        wrapped = wrap(t, 84)
        txt(ax, L + .018, y, "\u2022", 7.5, color="#7A2E38")
        y = block(L + .033, y, wrapped, 7.5, color="#7A2E38")
    y -= PAD - .004
    box(ax, L, y, R - L, top - y, TRAP, RED, 1.2)

    arrow(ax, .5, y - .004, y - .024)
    y -= .030

    # ---- 3 external anchors ----------------------------------------------
    top = y
    y -= PAD
    # "gap", not "trap": §2.2 and the caption both say the anchors close gaps.
    y = block(L + .015, y, ["EXTERNAL ANCHORS — what closes each gap"],
              8.0, "bold", GOLD)
    y -= .004
    anchors = [
        ("official district file",
         f"measures the hidden volume: {fill:.2f} per masked cell, "
         f"{fill_share:.1%} of volume, not the 1.5 midpoint"),
        ("contact survey", "pins the transmission parameter; it does not "
                           "identify it"),
        ("resident registration", "denominators, and the coverage profile "
                                  "the data does not record"),
    ]
    col_w = (R - L - .03) / 3
    ends = []
    for i, (h, b) in enumerate(anchors):
        x = L + .015 + i * col_w
        yy = block(x, y, wrap(h, 26), 7.5, "bold", "#7A5312")
        yy = block(x, yy - .002, wrap(b, 30), 7.5, color="#7A5312")
        ends.append(yy)
    y = min(ends) - PAD + .004
    box(ax, L, y, R - L, top - y, FACE, GOLD, 1.2)

    arrow(ax, .5, y - .004, y - .024)
    y -= .030

    # ---- 4 estimator ------------------------------------------------------
    top = y
    y -= PAD
    y = block(L + .015, y, ["ESTIMATOR"], 8.0, "bold", TEAL)
    y -= .004
    y = block(L + .015, y, [
        "co-arrival in the same dong, the same hour, the same weekday",
        f"\u2192  age x age matrix A, at three spatial scales "
        f"(dong {n_dong} / district 25 / city 1)",
        "\u2192  excess over the proportionate-mixing null: assortativity, "
        "normalised MI, rank"], 7.5)
    y -= .006
    y = block(L + .015, y,
              ["THREE CHECKS, all green before any number is quoted"],
              7.5, "bold")
    y = block(L + .015, y - .002, [
        f"citation audit   ·   independent second implementation "
        f"{gates_n}/{gates_n}   ·   bit-for-bit determinism"], 7.5)
    y -= PAD - .004
    box(ax, L, y, R - L, top - y, FACE, TEAL, 1.2)

    arrow(ax, .5, y - .004, y - .030)
    y -= .038

    # ---- 5 the three claims ----------------------------------------------
    txt(ax, .5, y, "THREE CLAIMS", 8.0, "bold", INK, ha="center", va="top")
    y -= .026

    claims = [
        ("1", "the instrument responds",
         f"the co-arrival matrix tracks the Korean school year in all "
         f"{months} months, and the year schools stayed shut is flat. The "
         f"calendar is never supplied to it. Of the {n_clear} months that "
         f"reach the survey's floor, every one is a term month; {n_match} of "
         f"{n_alt} alternative calendar alignments match.",
         "Fig 2, Fig 3(c)"),
        ("2", "and it still reads near zero",
         "the departure from proportionate mixing is near zero at every "
         "spatial scale and every age resolution. An identity fixes what "
         "this estimator can read at all, and a recovery experiment turns "
         "that into a bound conditional on how much sits below the cell.",
         "Fig 3, Fig 4, Fig 5"),
        ("3", "which changes an allocation",
         f"across the coverage family the ranking an allocation rule reads "
         f"moves, and the SEIR cost of reading the wrong one depends on "
         f"$R_0$: largest at {r0_lo:g}-{r0_hi:g}, and {n_flip} of "
         f"{n_verdict} verdicts turn on it.",
         "Fig 6, Fig 7"),
    ]
    cw = (R - L - .028) / 3
    bodies = [wrap(c[2], 31) for c in claims]
    heads = [wrap(c[1], 22) for c in claims]
    depth = max(len(h) for h in heads) * LEAD + max(len(b) for b in bodies) * LEAD
    h_box = .030 + depth + .030
    for i, (n, head, _, figs) in enumerate(claims):
        x = L + i * (cw + .014)
        box(ax, x, y - h_box, cw, h_box, "#FFFFFF", INK, 1.2)
        ax.add_patch(FancyBboxPatch((x + .012, y - .026), .026, .020,
                                    boxstyle="circle,pad=0.005",
                                    fc=INK, ec="none", zorder=4))
        txt(ax, x + .025, y - .0165, n, 8.0, "bold", "white",
            ha="center", va="center")
        yy = block(x + .050, y - .008, heads[i], 7.5, "bold")
        yy = block(x + .014, min(yy, y - .034) - .002, bodies[i], 7.5)
        txt(ax, x + .014, y - h_box + .012, figs, 7.5, "bold", TEAL, va="bottom")
    y -= h_box

    # Crop to what the layout actually used rather than shrinking the canvas:
    # the font sizes are in points, so changing H would rescale the line
    # spacing against the type and undo the layout.
    assert y > 0.0, "figure 1 overflows the canvas; raise H"
    crop = Bbox.from_extents(0, max(0.0, y * H - .10), W, H)
    out = f"{FIG}/p48_figure1.png"
    fig.savefig(out, dpi=300, bbox_inches=crop)
    if args.pdf:
        # CreationDate omitted on purpose: matplotlib stamps the wall clock
        # into the PDF, so two runs of the same figure differ by bytes for a
        # reason that has nothing to do with the figure. p46 and p63 have
        # done this since they were written; these four had not, so their
        # vector files showed up in every diff whether or not the figure
        # had changed. The PNG beside them was always stable.
        fig.savefig(f"{FIG}/p48_figure1.pdf", bbox_inches=crop,
                    metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  -> {out}   ({W:.2f} x {crop.height:.2f} in, "
          f"content bottom at y={y:.3f})")
    assert crop.width <= 7.0, "figure 1 is wider than AJE's 7-inch portrait cap"

    payload = dict(span=span, months=months, parquet_files=files, parquet_rows=rows,
                   dong=n_dong, masking_rate_weighted=round(float(mask_rate), 6),
                   masked_bands_material=material,
                   masked_band_rate_range=[round(mat_lo, 4), round(mat_hi, 4)],
                   # The three-way split the sentence renders, each class with
                   # the label the figure actually prints, so a caption can be
                   # gated against this file instead of against the PNG.
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
                   measured_fill=fill, measured_fill_share=fill_share,
                   p36_checks=gates_n,
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
                   claim1_p_shift=shift["p"],
                   claim3_r0_argmax_range=[r0_lo, r0_hi],
                   claim3_flips=n_flip, claim3_verdicts=n_verdict)
    with open(f"{ROOT}/eda/results_p48.json", "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
    print("\n=== the numbers on the figure, for the caption ===")
    for k, v in payload.items():
        print(f"  {k:<26} {v}")
    print("\nwrote results_p48.json")


if __name__ == "__main__":
    main()
