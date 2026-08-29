#!/usr/bin/env python
"""Phase 55 — Figure 7: the whole R0 curve, not the single point.

WHAT THIS REPLACES. `paper/paper_structure.md` §4.7 used to instruct the writer
to bind Claim 3 to R0 = 2.5. The advisor's 2026-08-24 letter changes the claim
to "the consequence of matrix choice depends on R0, and is largest near the
epidemic threshold", and asks for the full curve as a figure rather than one or
two values. This is that figure.

IT COMPUTES NOTHING, on the p46/p47/p48 rule. Every number on it is read from
`results_p52.json` and `results_p62.json` at draw time. The two panels are:

  (a) deterministic regret over the 12-point R0 grid, one line per cell. No
      Monte Carlo, so no error bars belong on it. R0 = 2.5 is marked because it
      is the only value p40 ever ran, and the point of the panel is that it is
      the friendly end.
  (b) the five bootstrapped R0 values, each cell's regret interval against its
      own resample-the-survey null, with the verdicts that flip marked. p40's
      rule: the cell flips iff regret p2.5 > null p97.5.

THE ONE CIRCLE THAT IS NOT ON THE LOW-R0 SIDE is annotated, because it is the
first thing a reader asks about. Four of the five flips sit at R0 <= 1.8; the
fifth is Feb 2024 at Seoul scope, R0 = 3.5, and it clears the null by 0.0146.
It is annotated as the thin margin it is and NOT as a multiple-comparisons
artefact: p62 measures the size of p52's flip rule under the global null at
6.2e-101, because the rule compares two bootstrap PERCENTILE RANGES, which do
not narrow with n_boot. The expected number of flips under that null is 1.2e-99
over 20 verdicts, so "one expected false positive in twenty" is wrong by about
a hundred orders of magnitude and wrong in the direction that concedes a false
positive we do not have. What is genuinely fragile there is endpoint
resolution: at 200 draws the nominal 2.5th percentile is the 6th order
statistic and the coverage it achieves runs 0.011-0.057.

WHAT IT REFUSES TO DRAW. The advisor's phrase is "largest near the epidemic
threshold". The grid starts at R0 = 1.1 and the four argmaxes are 1.5, 1.3, 1.3
and 1.2, so the maximum sits in the 1.2-1.5 region and R0 -> 1 was never
measured. The panel is annotated "1.2-1.5", not "at the threshold", and the
shaded band stops where the grid stops. A figure that implied a measurement at
the threshold would be the figure making the overstatement, not the prose.

ROMANISATION. No Hangul: it renders as tofu in every font matplotlib ships and
a figure needing a locally installed CJK font breaks on the typesetter's
machine. Cell keys are rewritten as "Dec 2023, national" rather than
"202312|national".

    python eda/p55_fig7.py --pdf
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
from paths import FIG, ROOT

# JRSI wants figure text in Times at 9-11 pt and refuses anything under 7.5 pt,
# so 7.5 is the floor for every explicit size below. STIXGeneral is the serif:
# it is Times-metric AND it ships inside matplotlib, so the figure renders the
# same on a machine with no Times installed. A figure that depends on a locally
# installed font breaks on the typesetter's machine -- the same reason the
# Korean here is romanised rather than set in a CJK font. mathtext.fontset =
# "stix" keeps the maths in the same face as the prose around it.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Times New Roman",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

TEAL, RED, BLUE, GOLD = "#0E7C86", "#A8434E", "#4C6E8A", "#BC8034"
INK = "#1b1b1b"
STYLE = {
    "202312|national": (TEAL, "-", "o", "Dec 2023, national"),
    "202402|national": (RED, "-", "s", "Feb 2024, national"),
    "202312|seoul": (BLUE, "--", "^", "Dec 2023, Seoul"),
    "202402|seoul": (GOLD, "--", "v", "Feb 2024, Seoul"),
}
ORDER = list(STYLE)

sheet = []


def note(panel, what, value):
    """Record a number the caption is allowed to quote."""
    sheet.append((panel, what, value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    p52 = json.load(open(f"{ROOT}/eda/results_p52.json"))

    # ---------------------------------------------------------------- gates
    # Re-assert p52's own invariants from the file, before anything is drawn.
    # A figure is the last place a stale results file should be discovered.
    d = p52["declaration"]
    assert d["n_boot"] == 200, \
        f"results_p52.json was produced at {d['n_boot']} draws, not the pinned 200"
    assert p52["anchor_point_curve"]["worst_abs_diff"] == 0.0, \
        "p52's replay of p45's point curve is not bit-exact"
    assert p52["anchor_bootstrap"]["bit_exact"], \
        "p52's bootstrap replay of p45 is not bit-exact"
    assert p52["anchor_deterministic"]["quantities"] == 12, \
        "p52 did not replay p45's twelve deterministic quantities"
    assert not p52["fail"], f"p52 reports failures: {p52['fail']}"
    assert p52["answer"]["n_verdicts"] == 20, "the tally is no longer 20 verdicts"
    assert set(p52["point"]) == set(ORDER), \
        f"cells changed: {sorted(p52['point'])}"

    grid = [r["r0"] for r in p52["point"][ORDER[0]]]
    assert grid == list(d["r0_point"]), "the grid does not match the declaration"
    boot_grid = list(d["r0_boot"])

    # p62's gate. This panel is about to be annotated with the flip rule's own
    # size, so it must not be drawable against a stale p62: a size measured at a
    # different n_boot is a different number by ~50 orders of magnitude (p62
    # size_vs_n_boot), and an outlier identified in a different verdict set is a
    # different point on the panel.
    p62 = json.load(open(f"{ROOT}/eda/results_p62.json"))
    a2 = p62["anchor_A2"]
    assert a2["n_flips"] == 5, \
        f"p62 replayed {a2['n_flips']} flips, not the 5 this panel circles"
    assert a2["verdicts_replayed_correctly"] == 20, \
        "p62 did not replay all 20 of p52's verdicts"
    assert a2["worst_margin_abs_diff"] == 0.0, \
        "p62's replay of p52's margins is not bit-exact"
    # The stored outlier block carries its margin alongside its identity, so the
    # two identifying fields are checked rather than the whole dict.
    assert (a2["outlier"]["cell"], a2["outlier"]["r0"]) == ("202402|seoul", 3.5), \
        f"the outlier moved to {a2['outlier']['cell']} at R0={a2['outlier']['r0']}"
    assert p62["fail"] == [], f"p62 reports failures: {p62['fail']}"
    assert p62["alpha_rule"]["n_boot"] == d["n_boot"], \
        f"p62 measured the rule's size at n_boot = " \
        f"{p62['alpha_rule']['n_boot']} and p52 ran at {d['n_boot']}"

    # 2 x 1, NOT 1 x 2. JRSI reproduces a figure at 180 mm, which is the 7.0 in
    # p48 and p63 already build to; an 11.6 in row arrives on the page at 0.60x
    # and the 7.5 pt floor set for this figure lands as 4.5 pt of ink. Stacking
    # spends height, which the page has, instead of type size, which it does
    # not. Neither panel loses width in the trade: both now get the full 7 in,
    # where the old row gave each of them 5.4.
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 8.0))

    # ------------------------------------------------- (a) the whole curve
    a = ax[0]
    argmaxes = sorted({v for v in p52["point_summary"]["argmax"].values()})
    a.axvspan(min(argmaxes), max(argmaxes), color=GOLD, alpha=0.12, zorder=0)
    a.text(np.sqrt(min(argmaxes) * max(argmaxes)), 0.985,
           f"largest here\n({min(argmaxes):g}-{max(argmaxes):g})", ha="center",
           va="top", fontsize=8, color="#7a5320")
    for k in ORDER:
        col, ls, mk, lab = STYLE[k]
        rows = p52["point"][k]
        a.plot([r["r0"] for r in rows], [r["regret"] for r in rows], ls,
               marker=mk, ms=4, lw=1.6, color=col, label=lab)
        note("a", f"{lab}: regret at R0=2.5",
             p52["point_summary"]["at_anchor"][k])
        note("a", f"{lab}: maximum over the grid",
             p52["point_summary"]["max_over_grid"][k])
        note("a", f"{lab}: R0 at that maximum", p52["point_summary"]["argmax"][k])
    a.axvline(2.5, color=INK, lw=1.0, ls=":", zorder=1)
    a.annotate("the only $R_0$\npreviously run", xy=(2.5, 0.62),
               xytext=(3.15, 0.72), fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))
    a.set_xscale("log")
    a.set_xticks(grid)
    a.set_xticklabels([f"{g:g}" for g in grid], fontsize=7.5)
    a.minorticks_off()
    a.set_xlabel("basic reproduction number $R_0$")
    a.set_ylabel("regret: share of achievable benefit forgone")
    a.set_ylim(-0.02, 1.0)
    a.set_title("(a)  the consequence of matrix choice, over $R_0$",
                fontsize=10, loc="left")
    a.legend(fontsize=8, frameon=False, loc="upper right")
    a.grid(alpha=0.18, lw=0.6)

    # ------------------------------------------- (b) the verdicts, with CIs
    b = ax[1]
    n_cell = len(ORDER)
    width = 0.19
    # The one flip that is not on the low-R0 side, read out of p62 rather than
    # named here, and LOCATED BY MATCHING (cell, R0) inside the loop below --
    # never by position. p10_response.py once reported August's residual as
    # September's because a list was indexed rather than keyed.
    out_cell, out_r0 = a2["outlier"]["cell"], a2["outlier"]["r0"]
    out_margin = a2["thinnest"]["margin"]
    _flip_r0 = sorted({r for v in a2["flips"].values() for r in v})
    r0_lo_top = max(r for r in _flip_r0 if r != out_r0)
    out_xy = None
    for i, k in enumerate(ORDER):
        col, _, mk, lab = STYLE[k]
        rows = {r["r0"]: r for r in p52["verdict"][k]}
        xs, lo, hi, med, flip = [], [], [], [], []
        for j, r0 in enumerate(boot_grid):
            r = rows[r0]
            x = j + (i - (n_cell - 1) / 2) * width
            xs.append(x)
            lo.append(r["ci"][0])
            hi.append(r["ci"][1])
            med.append(r["median"])
            flip.append(r["exceeds_null"])
            if k == out_cell and r0 == out_r0:
                out_xy = (x, r["median"])
            # the null's upper edge: the bar the regret has to clear
            b.plot([x - width * 0.42, x + width * 0.42],
                   [r["null_ci"][1]] * 2, color="#999999", lw=1.4, zorder=1)
        b.vlines(xs, lo, hi, color=col, lw=2.0, zorder=2)
        b.scatter(xs, med, s=22, color=col, marker=mk, zorder=3, label=lab)
        fx = [x for x, f in zip(xs, flip) if f]
        fy = [m for m, f in zip(med, flip) if f]
        b.scatter(fx, fy, s=110, facecolors="none", edgecolors=INK, lw=1.3,
                  zorder=4)
    assert out_xy is not None, \
        f"{out_cell} at R0={out_r0} is not among the cells this panel draws"
    # y in axes fraction, x in data: the y limits are set by the intervals, so a
    # fixed data-y here would be a guess that falls off the panel the moment one
    # of them moves. Same reason p56's (b) does it.
    b.annotate(f"only flip above $R_0$ = {r0_lo_top:g}\n"
               f"margin +{out_margin:.3f}",
               xy=out_xy, xycoords="data",
               xytext=(out_xy[0] - 1.05, 0.68), textcoords=b.get_xaxis_transform(),
               fontsize=7.5, color=INK, ha="center", va="center",
               arrowprops=dict(arrowstyle="->", lw=0.8, color=INK, shrinkB=8))
    b.set_xticks(range(len(boot_grid)))
    b.set_xticklabels([f"{g:g}" for g in boot_grid])
    b.set_xlabel("basic reproduction number $R_0$")
    b.set_ylabel("regret, 95% bootstrap interval")
    b.set_title("(b)  against the cost of simply re-running the survey",
                fontsize=10, loc="left")
    b.grid(alpha=0.18, lw=0.6, axis="y")
    handles = [Line2D([], [], color="#999999", lw=1.4,
                      label="null 97.5th pct (the bar to clear)"),
               Line2D([], [], marker="o", ls="none", ms=9,
                      markerfacecolor="none", markeredgecolor=INK,
                      label="exceeds the null")]
    b.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")

    ans = p52["answer"]
    note("b", "verdicts that exceed the null, corrected", ans["n_flips"])
    note("b", "verdicts in total", ans["n_verdicts"])
    note("b", "verdicts that exceeded before the vector was corrected",
         ans["n_flips_published"])
    for k, v in ans["flips"].items():
        note("b", f"{STYLE[k][3]}: R0 values that exceed", v)
    # p62. The rule is two non-overlapping bootstrap PERCENTILE RANGES, not a
    # test at the 5% level: the ranges do not narrow with n_boot, so the rule's
    # size collapses with it. None of the five flips is attributable to
    # multiple comparisons, and the caption may not say one of them is.
    al = p62["alpha_rule"]
    ef = p62["expected_flips"]
    note("b", "size of the flip rule under the global null (n_boot = 200)",
         al["value"])
    note("b", "that size, log10", al["log10"])
    note("b", "expected flips among 20 verdicts, treating them as independent",
         ef["reading_20_independent"]["expected"])
    note("b", "expected flips among the 4 independent surveys",
         ef["reading_4_independent"]["expected"])
    note("b", f"thinnest margin ({STYLE[out_cell][3]}, R0 = {out_r0:g})",
         out_margin)
    note("b", f"flips at or below R0 = {r0_lo_top:g}",
         a2["n_flips_inside_1p2_1p8"])
    # What IS fragile about that circle: at 200 draws the nominal 2.5th and
    # 97.5th percentiles are the 6th and 195th order statistics, and the
    # coverage they achieve is a range, not a point.
    ep = p62["endpoint_fragility"]
    note("b", f"coverage the nominal {ep['nominal_low']:g} endpoint achieves "
              f"(rank {ep['low_rank']} of {d['n_boot']})", ep["low_level_ci"])
    note("b", f"coverage the nominal {ep['nominal_high']:g} endpoint achieves "
              f"(rank {ep['high_rank']} of {d['n_boot']})", ep["high_level_ci"])

    fig.tight_layout()
    png = f"{FIG}/p55_figure7.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    if args.pdf:
        # CreationDate omitted on purpose: matplotlib stamps the wall clock
        # into the PDF, so two runs of the same figure differ by bytes for a
        # reason that has nothing to do with the figure. p46 and p63 have
        # done this since they were written; these four had not, so their
        # vector files showed up in every diff whether or not the figure
        # had changed. The PNG beside them was always stable.
        fig.savefig(f"{FIG}/p55_figure7.pdf", bbox_inches="tight",
                    metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {png}" + (" (+ .pdf)" if args.pdf else ""))

    print("\n=== the numbers this figure's caption may quote ===")
    for panel, what, value in sheet:
        v = (f"{value:.4g}" if isinstance(value, float) else str(value))
        print(f"  ({panel})  {what:<52} {v}")
    out = dict(caption_numbers=[dict(panel=p_, what=w, value=v)
                                for p_, w, v in sheet],
               source="results_p52.json + results_p62.json",
               n_boot=d["n_boot"],
               established=d["established"],
               refuses="the grid starts at 1.1, so the caption says 1.2-1.5 "
                       "and never 'at the epidemic threshold'. And the caption "
                       "does not call the R0 = 3.5 circle an expected false "
                       "positive: p62 measures the flip rule's size at "
                       "6.2e-101, so the expected number of flips under the "
                       "global null is 1.2e-99 over 20 verdicts and 2.5e-100 "
                       "over the 4 independent surveys. What is fragile about "
                       "that point is endpoint resolution at 200 draws, not "
                       "multiplicity.")
    with open(f"{ROOT}/eda/results_p55.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {ROOT}/eda/results_p55.json")


if __name__ == "__main__":
    main()
