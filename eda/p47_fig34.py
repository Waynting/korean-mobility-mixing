#!/usr/bin/env python
"""Phase 47 — Figures 3 and 4, the two composites the paper still owed.

`paper/paper_structure.md` §7 listed both as "materials exist, has to be
composed": Figure 3 out of `p32_excess` and `p32_spectrum`, Figure 4 out of
`p38_ksweep79` and `p19_bandwidth`.

3(c) IS GONE, AND IT WENT TO FIGURE 2 (2026-09-02). It drew `p37_timeseries`'s
spatial ladder on 79 months, tinted by school term and hatched over 2020 with
the annotation "2020: schools shut, the cycle is absent". Neither section this
figure belongs to cited it: §3.2 and §3.3 never mention the panel, and its only
reader was §3.1, two subsections EARLIER -- a forward reference to a panel
placed by neither of the sections that use it. Its 2020 reading is §3.1's
argument and its ladder reading is §3.4's, so it now sits under Figure 2's own
79-month axis in `p63_fig2.py`, where §3.1 can point at it locally and §3.4
points back. Nothing about the panel's content changed except its band key,
which had to become Figure 2(a)'s -- see the note where p63 draws it. The
`p37` gate that guarded it moved with it; p47 keeps only the p60/p37 month-set
gate, which is Figure 4's.

IT COMPUTES NO NEW QUANTITY, on the p46 rule. Every curve, bar and singular
value is read from `results_p19/p26/p32/p38/p60/p61.json`. The one thing
that is not in a results file is the excess matrix behind panel 3(a) —
`results_p32.json` stores its summary statistics, not the matrix — so that panel
reuses `symmetrise` and `pm_null` imported FROM `p32_pmix`. Imported, not
reimplemented: a second copy of that algebra is exactly what the rule exists to
prevent.

3(b) NOW CARRIES THE SURVEY'S OWN NULL (`results_p61.json`). The panel's selling
point was that it "restates (a) without reference to any null", which is also
its hole: three spectra drawn as if they were noiseless. p61 permuted the 465
Seoul respondents' ego bands 500 times — alter margins and every respondent's
contact count survive, only the pairing dies — and the band drawn behind the
spectra at k = 2 is that distribution, read from the percentiles p61 REPORTED
rather than redrawn here. A KDE violin was rejected: it is computed in linear
space and would read as a symmetric blob on a log axis.

4(b) IS NEW AND IT IS NOT A MIRROR (`results_p60.json`). The advisor asked for a
3-bin assortativity line so that both panels measure one quantity; p60 measured
it and the assumption behind the request is inverted. Coarsening geography
LOWERS r (gu/dong 0.469, city/dong 0.245); coarsening age RAISES it, by a median
factor 3.958, in 79 of 79 months. So the two panels are drawn as an asymmetry:
(a) rises to the right, (b) falls to the right, and that opposition is the
result. Flipping an axis to make them slope alike would delete the finding.
Under exact proportionate mixing the collapsed matrix returns r = 0 at every
rung (max |r| = 4.94e-16), so the rise is relabelled real excess, not an
artefact of the collapse operator. k = 1 is 0/0 and is NOT plotted.

There is no draft caption to check against yet (paper_structure §7 has every
caption at "write from scratch"), so the gate p46 got from the coverage draft is
replaced by two things: the invariants each source phase already asserted are
re-asserted here against the results files, and the run prints a numbered sheet
of every figure number so the caption can be written against it rather than from
memory. Write the captions from that sheet, not from the PNG.

    python eda/p47_fig34.py [--pdf]

SCOPE, panel 4(c) — SETTLED 2026-08-24, on the rule in phase1c-scope.md §3.
(It was panel 4(b) until the age ladder took that slot; the ruling below is
unchanged, and the panel keeps every word of it.)
The endpoint filter is not a matter of taste and the two filters are not
interchangeable, because they estimate different things:

    co-occurrence inside Seoul (the mixing matrix)  -> both endpoints in Seoul
    residents' behaviour (destination composition)  -> origin in Seoul only

Panel 4(c) is an aggregation argument about a composition, so the two readings
belong on the same axes rather than one of them being chosen silently. The bars
stay on p19's scope, and the scope pair is drawn from `results_p18b.json`.

BOTH markers come from p18b and NEITHER comes from p19, deliberately. p18b is
measured BEFORE the masking fill, so its both-endpoints series equals the bars
only outside 20-44, which is where masking occurs at all; pairing a p18b
origin-only value against a p19 bar would price the fill as if it were the
scope. gates() asserts that agreement outside 20-44 rather than assuming it.

What the filter is worth, and why it had to be settled before this panel could
travel: 80+ +0.844 -> +0.554 pp, 20-24 -3.956 -> -3.127 pp (two-endpoint
filtering inflates the gradient at BOTH ends), the zero crossing retreats one
bin from 65-69/70-74 to 70-74/75-79, and the three-band 60+ rule changes sign,
+0.075 -> -0.157 pp. The panel's own conclusion survives either filter and is
stronger under origin-in-Seoul, where three bands invert the sign of a top band
that is +0.554 -- which is the reason this figure no longer waits on the ruling.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter
from common import AGE_LABEL, AGES
from figstyle import WIDTH, finish, plabel
from p32_pmix import pm_null, symmetrise
from paths import FIG, ROOT

# JRSI wants figure text in Times at 9-11 pt and refuses anything under 7.5 pt,
# so 7.5 is the floor for every explicit size below. STIXGeneral is the serif:
# it is Times-metric AND it ships inside matplotlib, so both figures render the
# same on a machine with no Times installed. A figure that depends on a locally
# installed font breaks on the typesetter's machine -- the same reason the
# Korean here is romanised rather than set in a CJK font. mathtext.fontset =
# "stix" keeps $\sigma_k/\sigma_1$ in the same face as the prose around it.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Times New Roman",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix",
                     # One hatch left on this figure: the deficit cells in
                     # 3(a). At the default 1.0 that is a black cell rather
                     # than a texture, because a cell is 36 px across. (The
                     # 2020 band that shared this setting left with 3(c).)
                     "hatch.linewidth": 0.35})

# The six months released in earlier rounds of this work. They are no longer
# marked on any panel: to a journal reader "the months we published before" is
# internal history, not information (advisor 2026-08-31, item 3.10). The list is
# kept for traceability and because p37/p38/p60 still anchor against it.
PUBLISHED_MONTHS = [202001, 202012, 202312, 202402, 202512, 202606]
PANEL = "WE"
REF_MONTH = 202312

# Lim et al.'s three bands, in the order 4(c) draws them, with the number of
# 5-year boxes each covers so 4(c) can place its rules. ONE literal in this
# script, because gates() has to check that 4(b)'s gold markers (p60's "lim"
# arm) and 4(c)'s gold rules (p19's 3-band series) are the SAME partition.
LIM_SPAN = (("0-19", 3), ("20-59", 8), ("60+", 5))
# The rungs of p60's nested age ladder that 4(b) plots. 1 is deliberately absent:
# collapsing the age axis to one bin leaves no partition, so r is 0/0.
NESTED_K = (16, 8, 4, 2)

TEAL, RED, BLUE, GOLD = "#0E7C86", "#A8434E", "#4C6E8A", "#BC8034"

sheet = []


def note(fig, panel, what, value):
    """Record a number the caption is allowed to quote."""
    sheet.append((fig, panel, what, value))


def load(name):
    return json.load(open(f"{ROOT}/eda/results_{name}.json"))


# --------------------------------------------------------------------------
# gates: the invariants the source phases asserted, re-asserted from the files
# --------------------------------------------------------------------------
def gates(p18b, p19, p32, p37, p38, p60, p61):
    d = p32["identity_check"]
    assert d["max_abs_diff"] == 0.0, \
        "p32: assortativity is no longer exactly the diagonal excess"
    assert p38["p34_anchor"]["max_deviation"] == 0.0, \
        "p38: the 79-month sweep no longer reproduces p34 on the shared months"
    assert p38["adjacency"]["identical_to_p34"], "p38: adjacency graph drifted"
    assert p38["adjacency"]["connected"] and p38["adjacency"]["n_nodes"] == 424, \
        "p38: adjacency graph is not the connected 424-node graph"
    assert p19["p18_reproduction_max_abs_diff_pp"] == 0.0, \
        "p19: no longer reproduces p18"
    # The dong > gu > city ladder asserts moved to p63_fig2.py with the panel
    # that draws them (2026-09-02). p37 is still loaded here for the month-set
    # gate below, which belongs to Figure 4's age ladder, not to the spatial one.

    # p18b carries the scope pair panel 4(c) now draws. Two things have to hold
    # before those markers may sit on the same axes as p19's bars.
    b18 = {r["band"]: r for r in p18b["by_age"]}
    assert len(b18) == len(AGES), "p18b: not the 16-band grid"
    # (i) Masking occurs only in 20-44, so OUTSIDE 20-44 the pre-fill p18b
    #     both-ends series must equal p19's filled bars. If this ever fails,
    #     either masking has spread or the two phases stopped sharing a scope,
    #     and the marker pair silently becomes a different comparison.
    d16g = {r["age"]: r for r in p19["e_share_change_16band"]}
    masked = {"20-24", "25-29", "30-34", "35-39", "40-44"}
    worst = max(abs(d16g[AGE_LABEL[a]]["meas"] - b18[AGE_LABEL[a]]["both_ends_pp"])
                for a in AGES if AGE_LABEL[a] not in masked)
    assert worst <= 0.005, \
        f"p18b/p19: both-ends series disagree by {worst:.4f} pp outside 20-44"
    # (ii) The sign flip is what made this a ruling and not a wording choice.
    #      It is asserted, not trusted: the panel's whole point is that three
    #      bands invert a top band whose sign the filter does not change.
    assert p18b["three_band_int"]["band_pp"]["60+"] > 0 > \
        p18b["three_band_out"]["band_pp"]["60+"], \
        "p18b: the three-band 60+ rule no longer changes sign with the filter"
    assert b18["80+"]["both_ends_pp"] > b18["80+"]["origin_only_pp"] > 0, \
        "p18b: 80+ no longer positive under both filters"

    # ---- p60, the age ladder behind the new panel 4(b) --------------------
    assert p60["anchors"]["A3"]["max_abs_diff"] == 0.0, \
        "p60: the rebuilt 16-bin series is no longer p37's"
    assert p60["anchors"]["A1"]["max_abs_r_diff"] == 0.0, \
        "p60: the 16-group collapse is no longer the identity"
    assert p60["anchors"]["A4"]["max_abs_diff"] == 0.0, \
        "p60: the six published months are no longer p26's"
    # The prediction FAILED, and the panel exists because it failed. Assert the
    # failure, so a later re-run that quietly reverses it cannot leave 4(b)
    # drawn as an asymmetry while the numbers have become a mirror.
    assert p60["prediction"]["n_against"] == p60["prediction"]["n_months"] == 79, \
        "p60: age coarsening no longer raises r in all 79 months"
    assert p60["mechanism"]["proportionate_mixing_null_max_abs_r"] < 1e-12, \
        "p60: the collapse operator no longer returns r = 0 under proportionate mixing"
    assert len(p60["months"]) == len(p37["months"]) == 79, \
        "p60/p37: the month sets are no longer the same 79"
    # THE GATE THIS ROUND EXISTS FOR. 4(c)'s gold rules come from p19 and 4(b)'s
    # gold markers come from p60; both are supposed to be p19_bandwidth's
    # LIM_BANDS, and until now nothing said so. If that literal ever moves, the
    # two age panels would silently measure different partitions — which is the
    # exact defect the new panel was added to repair.
    a5 = p60["anchors"]["A5"]
    assert a5["passed"], "p60: its band map is no longer p19_bandwidth's LIM_BANDS"
    assert [(b, len(a5["bands"][b])) for b, _ in LIM_SPAN] == list(LIM_SPAN), \
        "4(b)/4(c): p60's bands are not the three bands 4(c) rules"
    assert [r["band"] for r in p19["e_share_change_3band"]] == \
        [b for b, _ in LIM_SPAN], \
        "4(c): p19's 3-band series is not the three bands this panel draws"
    assert [a for b, _ in LIM_SPAN for a in a5["bands"][b]] == list(AGES), \
        "4(b)/4(c): the three bands no longer tile the 16 boxes in order"
    assert p60["per_month"][str(REF_MONTH)]["lim"]["n_bin"] == len(LIM_SPAN) == 3, \
        "p60: the lim arm is no longer a three-bin partition"

    # ---- p61, the survey's own null behind the new band in 3(b) ----------
    assert p61["anchors"]["n_ok"] == p61["anchors"]["n"], "p61: an anchor failed"
    assert (p61["cells"][str(REF_MONTH)]["observed"]["sigma2_over_sigma1"]
            == p32["spectrum"][f"survey|{REF_MONTH}|{PANEL}|seoul"]
                  ["sigma2_over_sigma1"]), \
        "p61's null was drawn against a different observation than (b) plots"

    print(f"  gates: identity, p34 anchor, adjacency, p18 reproduction, "
          f"p18b scope pair (worst {worst:.4f} pp outside 20-44), "
          f"p60 anchors + one shared 3-band partition, p61 "
          f"({p61['anchors']['n_ok']}/{p61['anchors']['n']}) — all hold")


# --------------------------------------------------------------------------
# figure 3
# --------------------------------------------------------------------------
def figure3(p26, p27, p32, p61, pdf):
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]      # no 80+ survey egos
    lbl = [AGE_LABEL[AGES[i]] for i in sq]

    mats = []
    for key, title, tag in (
            ("202312|WE|dong|holidayfree", "passive, dong (424)", "dong"),
            ("202312|WE|gu", "passive, district (25)", "district")):
        A = np.array(p26["matrices"][key]["A"])[np.ix_(sq, sq)]
        T, _ = symmetrise(A / pops[REF_MONTH][sq][:, None], pops[REF_MONTH][sq])
        e, _, E = pm_null(T)
        mats.append((np.log10(e / E), title, tag))
    Cs = np.array(p27["survey"]["202312|WE|seoul"]["C"])[np.ix_(sq, sq)]
    Ts, _ = symmetrise(Cs, pops[REF_MONTH][sq])
    e, _, E = pm_null(Ts)
    mats.append((np.log10(np.where(e > 0, e / E, np.nan)),
                 "contact survey, Seoul", "survey"))
    # TWO EMPTY-CELL COUNTS EXIST AND THEY ARE NOT THE SAME NUMBER. p61 counts
    # 24 empty cells in the RAW 15x15 survey block, which is what its own
    # diagnostic is about. This panel draws the field AFTER symmetrise(), and
    # symmetrisation fills any cell whose transpose was reported: only the 10
    # pairs empty in BOTH directions stay empty, and those are the grey cells.
    # Both are true of the same survey and a caption that quotes one for the
    # other is wrong, so the relation is asserted here rather than left to the
    # reader of two sheets.
    _raw_zero = int((Cs == 0).sum())
    _both_zero = int(((Cs == 0) & (Cs.T == 0)).sum())
    _drawn_zero = int(np.count_nonzero(e <= 0))
    assert _raw_zero == p61["emptiness_diagnostic_post_hoc"]["cells"][
        str(REF_MONTH)]["observed_zero_cells"], \
        "3(a)'s raw survey block no longer has p61's empty-cell count"
    assert _drawn_zero == _both_zero, \
        f"3(a) draws {_drawn_zero} empty cells but {_both_zero} pairs are " \
        f"empty in both directions; symmetrise() is not what fills the rest"
    note(3, "a", "empty cells in the raw survey block (p61's count)", _raw_zero)
    note(3, "a", "of those, empty in both directions and so still empty after "
                 "symmetrisation, which is what this panel greys", _both_zero)
    vmax = max(np.nanmax(np.abs(m)) for m, _, _ in mats)
    note(3, "a", "shared colour limit, log10 ratio to the null",
         round(float(vmax), 4))
    for m, t, _ in mats:
        note(3, "a", f"max |log10 ratio to the null|, {t}",
             round(float(np.nanmax(np.abs(m))), 4))
    # What the hatch and the grey ground each cover, so a caption can name them
    # without counting cells off the PNG. Display aggregates of drawn values.
    for m, _, tag in mats:
        note(3, "a", f"{tag} matrix: cells below the null (hatched)",
             int(np.count_nonzero(m < 0)))
        note(3, "a", f"{tag} matrix: cells greyed as unobserved after "
                     "symmetrisation", int(np.count_nonzero(np.isnan(m))))
    note(3, "a", "cells in each matrix", int(mats[0][0].size))

    # WIDTH in wide, which is JRSI's own measure (figstyle.py). The old canvas
    # was 13.2 in and arrived on the page at 0.56x, so the 7.5 pt floor this
    # figure was raised to landed as 4.2 pt of ink -- the width defeated the
    # type size. The 7.00 in that replaced it was still scaled, to 0.9286.
    # Height is not capped, so the row that used to hold (b) beside (c) is
    # spent instead: (a) keeps its three-across row and (b) takes a full-width
    # row of its own. 2026-09-02: (c) left for Figure 2 (see the docstring),
    # and its 2.90 in came off H rather than being redistributed -- the two
    # panels that remain were already at the size their content needs, and
    # growing them to fill a hole would only push the figure's reproduction
    # factor back down. Nothing is merged and nothing is re-lettered; (a) and
    # (b) carry exactly the content the caption describes.
    #
    # Saved WITHOUT bbox_inches="tight" (the p46/p63 rule), so the emitted page
    # IS the figsize rather than whatever the trim happens to leave. That makes
    # the margins below load-bearing: `left` has to hold the y tick labels AND
    # the panel letter, `top` the row header.
    # The three rows are placed in INCHES rather than by height_ratios, because
    # what has to be reserved is a text height (a rotated tick label, a title)
    # and that does not scale with the panel. Each row's band is written out so
    # the arithmetic is checkable: the numbers below sum to H.
    H = 6.00
    def fy(inches):                       # inches from the bottom -> fraction
        return inches / H
    # (b) 0.28 pad | 0.30 xlabel | 2.10 axes | 0.30 title
    # (a) 0.30 gap | 0.44 rotated band labels | 1.80 axes | 0.16 titles
    #             | 0.30 header
    # The bands below sum to 5.98 against H = 6.00, the same 0.02 of slack the
    # three-row version carried at 8.88 against 8.90.
    B_LO, B_HI = fy(0.58), fy(2.68)
    A_LO, A_HI = fy(3.72), fy(5.52)
    LEFT, RIGHT = .105, .985
    # Row (a) stops short of the right margin so the colour bar and its tick
    # labels have somewhere to be. At 13.2 in the bar hung off the third matrix
    # and its labels rode the figure edge; at 7 in they fell off it.
    A_LEFT, A_RIGHT = .070, .885

    fig = plt.figure(figsize=(WIDTH, H))
    gsA = fig.add_gridspec(1, 3, left=A_LEFT, right=A_RIGHT,
                           top=A_HI, bottom=A_LO, wspace=.09)
    gsB = fig.add_gridspec(1, 1, left=LEFT, right=RIGHT, top=B_HI, bottom=B_LO)

    def panel_letter(s, y_top):
        """Panel letters flush at the left edge, on the row's title line.

        They used to be offsets in AXES fractions, which is a fixed distance
        only while the axes stays the same size; at half the width they walked
        into the tick labels. Anchoring them to the figure instead also lines
        them up with each other, which the old placement never did.
        """
        fig.text(.010, y_top + 0.09 / H, s, fontsize=10,
                 va="baseline", ha="left")

    # (a) three excess matrices on one colour scale
    axes_a = []
    for j, (m, t, _) in enumerate(mats):
        ax = fig.add_subplot(gsA[0, j])
        axes_a.append(ax)
        # Empty cells read as ABSENT, not as zero excess. The survey block has
        # age pairs no respondent reported, and imshow leaves NaN as the axes
        # background: on white that is indistinguishable from the pale middle
        # of a diverging scale, which is exactly the reading the panel must not
        # invite. A grey ground separates "no observation" from "no excess".
        ax.set_facecolor("0.86")
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        # THE SIGN IS NOT LEFT TO COLOUR. JRSI prints black and white by
        # default, and RdBu_r is symmetric in lightness: +1.82 and -1.82
        # convert to the same dark grey, so a greyscale reader could not tell
        # excess from deficit -- which is the entire content of this panel.
        # Hatching the cells below the null is a redundant encoding that
        # survives the conversion and costs nothing in colour.
        # r, c and NOT i, j: `j` is the panel index this loop sits inside,
        # and rebinding it here left `j` holding the column of the last
        # below-null cell by the time set_yticklabels tested `j == 0` --
        # so every panel took the empty branch and all three matrices lost
        # their age labels. Silent: the ticks still drew, only the text went.
        for r, c in np.argwhere(m < 0):
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False, lw=0,
                                   hatch="///", edgecolor=(0, 0, 0, .34),
                                   zorder=3))
        ax.set_title(t, fontsize=8.5, pad=3)
        ax.set_xticks(range(len(lbl)))
        ax.set_yticks(range(len(lbl)))
        ax.set_xticklabels(lbl, rotation=90, fontsize=7.5)
        # The three matrices are on the same 15 age bands, so the bands are
        # labelled once. Repeating them would cost 0.6 in of the 7 that exist,
        # and the width is the whole constraint here.
        ax.set_yticklabels(lbl if j == 0 else [], fontsize=7.5)
        # The axes carry a title as well as the band ticks (advisor 2026-09-15,
        # item 7: no panel may leave an axis untitled). "age band" once on the
        # y side, for the same width reason the tick labels appear once.
        ax.set_xlabel("age band", fontsize=7.8)
        if j == 0:
            ax.set_ylabel("age band", fontsize=7.8)
        ax.tick_params(length=2, pad=1.5)
    panel_letter(plabel("a"), A_HI + 0.16 / H)
    # What stays on the panel is the key, not the description: "hatched =
    # below the null" decodes a mark the reader cannot otherwise read,
    # while the statistic, the month and the shared limits are the
    # caption's own first sentence.
    fig.text(A_LEFT, A_HI + 0.25 / H,
             "hatched = below the null, grey = no pair observed",
             fontsize=8.5, va="baseline", ha="left")
    # One colour bar for the row, in its own axes rather than carved out of the
    # third matrix: carving made that matrix smaller than the two beside it,
    # which reads as a difference in the data.
    cax = fig.add_axes([A_RIGHT + .016, A_LO + .10 * (A_HI - A_LO),
                        .016, .80 * (A_HI - A_LO)])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7.5, length=2, pad=1.5)
    # A RATIO, NOT A DIFFERENCE. The scale runs negative, and the log of a
    # difference cannot; what is drawn is log10 of the observed cell over its
    # proportionate-mixing null. The bar carried no label at all until now,
    # which is how the caption and the results keys drifted apart.
    cb.set_label("log10 ratio to the null", fontsize=7.5, labelpad=3)

    # (b) the rank test, with the survey's own null drawn behind it
    ax = fig.add_subplot(gsB[0, 0])
    c61 = p61["cells"][str(REF_MONTH)]
    perm = c61["permutation"]["sigma2_over_sigma1"]
    boot = c61["bootstrap"]["sigma2_over_sigma1"]
    pq, bq = perm["quantiles"], boot["quantiles"]
    # The null is a distribution of ONE number, sigma2/sigma1, so it is drawn
    # where that number lives -- a window around k = 2 -- and not across the
    # panel. Every edge of it is a percentile p61 REPORTED: p61 stored all 500
    # draws so that this figure shows the distribution the text quotes rather
    # than a fresh one. A KDE violin was rejected on purpose -- it is computed
    # in linear space and reads as a symmetric blob once the axis is
    # logarithmic, which is exactly the misreading this band has to avoid.
    x0, x1 = 1.42, 2.58
    ax.fill_between([x0, x1], perm["min"], perm["max"], color="0.88", lw=0, zorder=0)
    ax.fill_between([x0, x1], pq["p2.5"], pq["p97.5"], color="0.74", lw=0, zorder=0)
    ax.hlines(perm["median"], x0, x1, color="0.28", lw=1.4, zorder=1)
    ax.hlines(pq["p95"], x0, x1, color="0.28", lw=1.2, ls=":", zorder=1)
    # The bootstrap is the observation's own spread, so it sits ON k = 2 under
    # the survey marker: the two supports do not touch, and that is the point.
    ax.vlines(2, bq["p2.5"], bq["p97.5"], color=RED, lw=7, alpha=.32, zorder=2)

    for key, style, col, lab in (
            ("passive|202312|WE|dong|holidayfree", "-o", TEAL, "passive, dong"),
            ("passive|202312|WE|gu", "-s", BLUE, "passive, district"),
            ("survey|202312|WE|seoul", "-^", RED, "contact survey")):
        sv = np.array(p32["spectrum"][key]["singular_values"], float)
        ax.semilogy(range(1, len(sv) + 1), sv / sv[0], style, color=col,
                    ms=4, lw=1.2, zorder=3, label=lab)
        note(3, "b", f"sigma2/sigma1, {lab}",
             round(float(p32["spectrum"][key]["sigma2_over_sigma1"]), 4))
        note(3, "b", f"sigma1 share, {lab}",
             round(float(p32["spectrum"][key]["sigma1_share"]), 4))
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_k/\sigma_1$")
    # Floor set below the smallest singular value (district, k = 15) so the
    # seven-entry legend has a strip of its own. Without it the legend lands on
    # the district curve, and a legend that covers a curve is worse than a
    # smaller panel. 2e-6 was that floor while the legend was set at 6.1 pt;
    # at the 7.5 pt JRSI floor the seven entries are half an inch taller and
    # the box climbed back over the district curve at k = 8, so the strip is a
    # decade and a bit deeper.
    ax.set_ylim(1e-7, 2.0)
    handles = [Line2D([], [], color=TEAL, marker="o", ms=4, lw=1.2,
                      label="passive, dong"),
               Line2D([], [], color=BLUE, marker="s", ms=4, lw=1.2,
                      label="passive, district"),
               Line2D([], [], color=RED, marker="^", ms=4, lw=1.2,
                      label="contact survey"),
               Patch(fc="0.74", label="survey null, central 95% of 500"),
               Patch(fc="0.88", label="survey null, full range of 500"),
               Line2D([], [], color="0.28", lw=1.2, ls=":",
                      label="null p95 (escape threshold)"),
               Line2D([], [], color=RED, lw=7, alpha=.32,
                      label="survey bootstrap, 95% CI")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left", framealpha=.92,
              handlelength=1.8, handletextpad=.6, labelspacing=.35,
              borderpad=.4)
    ax.grid(alpha=.3, which="both")
    panel_letter(plabel("b"), B_HI)
    ax.set_title("grey = the survey's own null at $k=2$",
                 fontsize=8.5, loc="left", pad=5)

    v2 = c61["verdict_sigma2_over_sigma1"]
    # Every edge this panel draws is on the sheet, so a caption can name the
    # band it points at without reading a percentile off the PNG.
    note(3, "b", "survey null, permutation median sigma2/sigma1",
         round(float(perm["median"]), 6))
    note(3, "b", "survey null, p2.5 (lower edge of the shaded band)",
         round(float(pq["p2.5"]), 6))
    note(3, "b", "survey null, p95 (the declared escape threshold)",
         round(float(pq["p95"]), 6))
    note(3, "b", "survey null, p97.5", round(float(pq["p97.5"]), 6))
    note(3, "b", "survey null, min over 500 ego-band shuffles",
         round(float(perm["min"]), 6))
    note(3, "b", "survey null, max over 500 ego-band shuffles",
         round(float(perm["max"]), 6))
    note(3, "b", "survey bootstrap, p2.5", round(float(bq["p2.5"]), 6))
    note(3, "b", "survey bootstrap, p97.5", round(float(bq["p97.5"]), 6))
    note(3, "b", "survey bootstrap, min over 500 respondent resamples",
         round(float(boot["min"]), 6))
    note(3, "b", "primary escape (observed above the null p95)",
         bool(v2["primary_escape"]))
    note(3, "b", "secondary escape (bootstrap p2.5 above null p97.5)",
         bool(v2["secondary_escape"]))
    note(3, "b", "permutation p-value (0 of 500 draws reach the observation)",
         float(v2["permutation_p_value"]))
    pin = p61["passive_inside_survey_null"]["cells"]
    note(3, "b", "passive dong sigma2/sigma1, percentile in the survey's null",
         float(pin[f"{REF_MONTH}|dong"]["percentile_within_survey_null"]))
    note(3, "b", "passive district sigma2/sigma1, percentile in the survey's null",
         float(pin[f"{REF_MONTH}|gu"]["percentile_within_survey_null"]))
    emp = p61["emptiness_diagnostic_post_hoc"]["cells"][str(REF_MONTH)]
    # RAW, before the symmetrisation 3(a) draws through -- see the assert in
    # (a). 24 here against the 10 cells (a) greys, and neither is the other.
    note(3, "b", "observed empty cells, 15x15 survey block (raw, before "
                 "symmetrisation)", int(emp["observed_zero_cells"]))
    note(3, "b", "null median empty cells", float(emp["null_zero_cells_median"]))

    out = f"{FIG}/p47_figure3.png"
    mode = finish(fig, out, pdf=pdf)
    plt.close(fig)
    print(f"  -> {out}   ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in, {mode})")


# --------------------------------------------------------------------------
# figure 4
# --------------------------------------------------------------------------
def figure4(p18b, p19, p38, p60, pdf):
    # WAS 1x3 at 17.6 in, which reproduces at 0.44x: the 7.5 pt floor this
    # figure was raised to arrived as 3.3 pt of ink. WIDTH is JRSI's own
    # measure (figstyle.py), so the row has to become two.
    #
    # The old comment here refused a spanning bottom row on the grounds that
    # "a full-width bottom row reads as a promotion, and that panel is being
    # demoted". The width cap overrules it, and the trade is honest: at 7 in a
    # three-across row gives (c) 2.6 in for 16 labelled bars and an eight-entry
    # legend, which is not a panel, it is a smudge. (c) is still demoted by
    # being third and by carrying the smallest claim; what it gets back is the
    # 6.4 in its content needs. (a) and (b) share the top row because they are
    # the matched pair -- the same estimator on two axes, answering in opposite
    # directions -- and reading them side by side is the point of the figure.
    # No panel is merged, dropped or re-lettered.
    #
    # Rows are placed in INCHES, not by height_ratios: what has to be reserved
    # is a text height (a rotated tick label, a two-line title), and that does
    # not scale with the panel.
    # 2026-08-31 (advisor 3.5/3.6): the third panel is gone. It was the one
    # behavioural, single-year result in a figure of 79-month measurements,
    # and it sat on neither of the two scale axes (a) and (b) plot -- see the
    # note where its notes used to be drawn, a few lines below. H drops to
    # just the top row's own height; the panel's numbers are unchanged and
    # still read at draw time, they are just no longer rendered as a panel.
    H = 2.95
    def fy(inches):
        return inches / H
    R_LO, R_HI = fy(0.55), fy(2.55)
    RIGHT = .985
    # The top row starts further in than the bottom one. Both (a) and (b) are
    # log axes whose minor labels read "2 x 10^-2", which is half an inch of
    # tick label before the rotated "assortativity" even starts; at left=.105
    # that word ran off the left edge of the page. wspace has to clear the same
    # stack twice, because (b) carries its own copy of it.
    R_LEFT, R_WSPACE = .130, .30

    fig = plt.figure(figsize=(WIDTH, H))
    gsR = fig.add_gridspec(1, 2, left=R_LEFT, right=RIGHT,
                           top=R_HI, bottom=R_LO, wspace=R_WSPACE)
    axes = [fig.add_subplot(gsR[0, 0]), fig.add_subplot(gsR[0, 1])]

    def panel_letter(ax, s, dy_in=.22):
        """Letter to the LEFT of the axes, on the title's first line.

        Placed off the axes' own box rather than in axes fractions: a fraction
        offset is a fixed distance only while the axes stays the same size, and
        these are now half the width they were.
        """
        p = ax.get_position()
        fig.text(p.x0 - .34 / WIDTH, p.y1 + dy_in / H, s,
                 fontsize=10, va="baseline", ha="left")

    # (a) spatial scale. All 79 curves are drawn alike: the six previously
    # published months are no longer picked out in red (advisor 3.10).
    ax = axes[0]
    for ym in p38["months"]:
        b = p38["band"][f"{ym}|{PANEL}|adjacency"]
        n = np.array([x["n_loc"] for x in b], float)
        r = np.array([x["assortativity"]["median"] for x in b], float)
        k = r > 0
        ax.plot(n[k], r[k], "-", lw=.7, color="0.72", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvspan(424, 1e7, color="0.85", zorder=0)
    ax.set_xlim(0.8, 1e4)
    # Anchored low in the band for the same reason as (b)'s "one bin" string:
    # the panel is 2.5 in tall now, and a 0.9 in rotated label starting at 0.4
    # of the top ran out through the axes frame.
    ax.text(700, ax.get_ylim()[1] * .16, "unsupported by these data",
            fontsize=7.5, rotation=90)
    ax.set_xlabel("number of locations")
    ax.set_ylabel("assortativity")
    ax.grid(alpha=.3, which="both")
    # (a) and (b) are the same estimator on two axes, and they answer in
    # opposite directions. The two notes are a matched pair; they are the whole
    # reason the panels are NOT drawn as a mirror.
    ax.text(.03, .96, "coarsening geography\nlowers r", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, color="#33475B", linespacing=1.3,
            bbox=dict(fc="white", ec="none", alpha=.85, pad=2.0))
    panel_letter(ax, plabel("a"))
    ax.set_title(f"spatial scale: {len(p38['months'])} monthly curves,\n"
                 "adjacency merge arm",
                 fontsize=8.5, loc="left", pad=4)
    ex = p38["extrapolation_support"]
    note(4, "a", "curves", len(p38["months"]))
    note(4, "a", "log10 span the curve supports", round(float(ex["log10_span"]), 3))
    note(4, "a", "log10 span the published extrapolation needed",
         round(float(ex["log10_span_required"]), 3))
    note(4, "a", "adjacency edges / mean degree",
         f"{p38['adjacency']['edges']} / {p38['adjacency']['mean_degree']:.3f}")

    # (b) age scale, the same estimator as (a) on the other axis. The advisor
    # asked for this line so that both panels would measure one quantity; p60
    # ran it and found the direction inverted, so the panel is drawn as an
    # ASYMMETRY. (a) rises to the right, (b) falls to the right, and neither
    # axis is flipped to make them agree: the opposition IS the result.
    ax = axes[1]
    xk = np.array(NESTED_K, float)
    for ym in p60["months"]:
        nested = p60["per_month"][ym]["nested"]
        r = np.array([nested[str(k)]["r"] for k in NESTED_K], float)
        ax.plot(xk, r, "-", lw=.7, color="0.72", zorder=1)
        # Lim et al.'s three bands are not a rung of the nested ladder (3, 8 and
        # 5 boxes, not a dyadic split), so they are a marker and not a point on
        # the curve. Gold is the 3-band colour in (c), so both age panels key
        # the same partition to the same colour; gates() asserts it IS the same
        # partition rather than trusting the two files to agree.
        ax.plot([3], [p60["per_month"][ym]["lim"]["r"]], "o",
                ms=2.0, color=GOLD, mec=GOLD, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(3, color=GOLD, lw=1.0, ls="--", zorder=2)
    # k = 1 is 0/0, not 0: collapsing the age axis to one bin leaves no
    # partition at all, so p60 stores null and this panel plots nothing there.
    # The band mirrors (a)'s "unsupported by these data" band structurally --
    # both say "the curve stops here, and here is why".
    ax.axvspan(0.8, 2, color="0.85", zorder=0)
    ax.set_xlim(0.8, 22)
    ax.set_ylim(0.008, 0.15)
    ax.set_xticks(sorted(NESTED_K + (3,)))
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    # Anchored low in the band, not at 0.30 of the top: the panel is 2.5 in
    # tall now instead of 5, and a 0.95 in string starting at 0.30 of a log
    # axis ran up into the second line of the title.
    ax.text(1.26, ax.get_ylim()[1] * .13, "one bin: r is 0/0, undefined",
            fontsize=7.5, rotation=90, ha="center", va="bottom")
    ax.text(.97, .96, "coarsening age\nraises r", transform=ax.transAxes,
            va="top", ha="right", fontsize=8.5, color="#33475B", linespacing=1.3,
            bbox=dict(fc="white", ec="none", alpha=.85, pad=2.0))
    ax.set_xlabel("number of age bins")
    ax.set_ylabel("assortativity")
    ax.grid(alpha=.3, which="major")
    handles_b = [Line2D([], [], color="0.72", lw=.9,
                        label=f"one curve per month ({len(p60['months'])})"),
                 Line2D([], [], color=GOLD, marker="o", ms=3.2, ls="--", lw=1.0,
                        label="Lim et al., 3 bands")]
    ax.legend(handles=handles_b, fontsize=7.5, loc="lower right",
              framealpha=.92, handlelength=1.9, handletextpad=.6)
    panel_letter(ax, plabel("b"))
    ax.set_title(f"age scale: the same {len(p60['months'])} matrices,\n"
                 "nested ladder, every rung coarsens the one above",
                 fontsize=8.5, loc="left", pad=4)

    aa = p60["axis_asymmetry"]
    note(4, "b", "curves", len(p60["months"]))
    note(4, "b", "median 16-bin assortativity",
         round(float(p60["summary"]["r16"]["median"]), 6))
    note(4, "b", "median 3-bin assortativity (Lim bands)",
         round(float(p60["summary"]["lim3"]["median"]), 6))
    note(4, "b", "median 3-bin / 16-bin ratio",
         round(float(aa["age_3_over_16"]["median"]), 4))
    note(4, "b", "3-bin / 16-bin ratio, range over the months",
         f"{aa['age_3_over_16']['min']:.4f}-{aa['age_3_over_16']['max']:.4f}")
    note(4, "b", "months moving against the declared prediction",
         f"{p60['prediction']['n_against']} of {p60['prediction']['n_months']}")
    for k in NESTED_K:
        note(4, "b", f"nested ladder median, {k} bins",
             round(float(p60["summary"]["nested"][str(k)]["median"]), 6))
    note(4, "b", "1 bin", p60["summary"]["nested_k1"])
    note(4, "b", "median absorbed-excess gain (numerator)",
         round(float(p60["mechanism"]["numerator_gain"]["median"]), 4))
    note(4, "b", "median margin-concentration shrink (denominator)",
         round(float(p60["mechanism"]["denominator_shrink"]["median"]), 4))
    note(4, "b", "proportionate-mixing null, max |r| over every rung",
         f"{p60['mechanism']['proportionate_mixing_null_max_abs_r']:.2e}")
    note(4, "b", "spatial median ratio, district / dong",
         round(float(aa["spatial"]["gu_over_dong"]["median"]), 4))
    note(4, "b", "spatial median ratio, city / dong",
         round(float(aa["spatial"]["city_over_dong"]["median"]), 4))

    # (c) is gone (advisor 2026-08-31, item 3.6): the March-vs-December 2020
    # discretionary-arrival composition was the one behavioural, single-year
    # result in a figure otherwise built from 79-month measurements, and it
    # sat on neither of the two scale axes (a) and (b) plot. The main text
    # now carries it as one sentence (SS 3.4) and SI 6 carries it in full.
    #
    # NOTHING BELOW HERE DRAWS ANYTHING. d16/d3/span/b18/labs/mid are kept
    # because the note() calls a few lines down still read them, and every
    # number they register is still checked by p31 and still quoted in the
    # manuscript pair -- what changed is that this page no longer renders a
    # bar chart to go with them, not that the underlying measurement went
    # anywhere.
    d16 = {r["age"]: r for r in p19["e_share_change_16band"]}
    d3 = {r["band"]: r for r in p19["e_share_change_3band"]}
    span = {r["resolution"]: r for r in p19["resolution_span"]}
    # p18b is read for the notes only, and it is measured BEFORE the masking
    # fill -- which is why none of its values may be paired with a p19 bar in
    # prose either. gates() still asserts the agreement outside 20-44.
    b18 = {r["band"]: r for r in p18b["by_age"]}
    labs = [AGE_LABEL[a] for a in AGES]
    mid = np.array([d16[l]["meas"] for l in labs], float)

    note(4, "not drawn", "16-band range, pp", span["16 bands"]["range_pp"])
    note(4, "not drawn", "3-band range, pp", span["3 bands (Lim)"]["range_pp"])
    note(4, "not drawn", "fraction of the range three bands retain",
         p19["range_retained_frac"])
    note(4, "not drawn", "largest single absorption, pp",
         max(p19["what_the_bands_absorb"], key=lambda r: abs(r["absorbed"]))["absorbed"])
    note(4, "not drawn", "3-band 60+ , pp (both-ends-in-Seoul scope)", d3["60+"]["meas"])
    note(4, "not drawn", "16-band 80+ , pp (both-ends-in-Seoul scope)", d16["80+"]["meas"])
    # The scope pair, all from p18b so the caption never pairs it with a bar.
    note(4, "not drawn", "p18b 16-band 80+, both endpoints, pp",
         round(b18["80+"]["both_ends_pp"], 4))
    note(4, "not drawn", "p18b 16-band 80+, origin only, pp",
         round(b18["80+"]["origin_only_pp"], 4))
    note(4, "not drawn", "p18b 16-band 20-24, both endpoints, pp",
         round(b18["20-24"]["both_ends_pp"], 4))
    note(4, "not drawn", "p18b 16-band 20-24, origin only, pp",
         round(b18["20-24"]["origin_only_pp"], 4))
    note(4, "not drawn", "p18b 3-band 60+, both endpoints, pp",
         round(p18b["three_band_int"]["band_pp"]["60+"], 4))
    note(4, "not drawn", "p18b 3-band 60+, origin only, pp",
         round(p18b["three_band_out"]["band_pp"]["60+"], 4))
    note(4, "not drawn", "p18b 3-band 20-59, both endpoints, pp",
         round(p18b["three_band_int"]["band_pp"]["20-59"], 4))
    note(4, "not drawn", "p18b 3-band 20-59, origin only, pp",
         round(p18b["three_band_out"]["band_pp"]["20-59"], 4))
    note(4, "not drawn", "zero crossing, both endpoints",
         p18b["claims"]["both_ends"]["zero_crossing"])
    note(4, "not drawn", "zero crossing, origin only",
         p18b["claims"]["origin_only"]["zero_crossing"])
    note(4, "not drawn", "range three bands retain, both endpoints",
         round(p18b["three_band_int"]["range_kept"], 4))
    note(4, "not drawn", "range three bands retain, origin only",
         round(p18b["three_band_out"]["range_kept"], 4))
    # Renamed with the markers that used to carry it: the two series are still
    # measured, still compared here, and still reported in SI 6 -- the panel
    # just no longer draws the second one, so "marker" would name nothing.
    note(4, "not drawn", "largest gap between p19's bars and p18b's both-endpoints "
                 "series, pp (the masking fill, not the scope)",
         round(float(max(abs(mid[i] - b18[l]["both_ends_pp"])
                         for i, l in enumerate(labs))), 4))

    out = f"{FIG}/p47_figure4.png"
    mode = finish(fig, out, pdf=pdf)
    plt.close(fig)
    print(f"  -> {out}   ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in, {mode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    p18b, p19, p26, p27 = load("p18b"), load("p19"), load("p26"), load("p27")
    p32, p37, p38 = load("p32"), load("p37"), load("p38")
    p60, p61 = load("p60"), load("p61")

    print("=== 47.0 gates ===")
    gates(p18b, p19, p32, p37, p38, p60, p61)
    print("\n=== 47.1 figure 3 ===")
    figure3(p26, p27, p32, p61, args.pdf)
    print("\n=== 47.2 figure 4 ===")
    figure4(p18b, p19, p38, p60, args.pdf)

    print("\n=== 47.3 the caption sheet — write the captions from THIS ===")
    for f, p, what, v in sheet:
        print(f"  fig {f}({p})  {what:<58} {v}")
    with open(f"{ROOT}/eda/results_p47.json", "w") as fh:
        json.dump({"caption_numbers": [
            dict(figure=f, panel=p, what=w, value=v) for f, p, w, v in sheet]},
            fh, indent=1, ensure_ascii=False, default=str)
    print("\nwrote results_p47.json")


if __name__ == "__main__":
    main()
