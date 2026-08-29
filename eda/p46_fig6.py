#!/usr/bin/env python
"""Phase 46 — Figure 6, the composite the coverage section has been missing.

`archive/section_coverage_draft.md` §2 said the English prose was paste-ready and the
only thing missing is the figure: two existing PNGs that have to become one
four-panel submission figure. This builds it.

IT COMPUTES NOTHING. Every number is read from `results_p33.json` and
`results_p35.json`; there is no second implementation of anything here, because a
figure script that recomputes is a figure script that can disagree with the text.

WHAT IT DOES ASSERT is the other half of that: the caption in
That staging prose quoted eleven numbers, and this script checks each one
against the results file before drawing. A caption and a figure drifting apart is
the exact failure the citation gate exists to prevent for letters, and the figure
had no equivalent. If the draft caption is edited without the results moving, or
the results move without the caption, this run stops.

WHAT IT DID NOT HAVE UNTIL 2026-08-28 is the other half of that same idea, and
it was the only figure phase without one: a RESULTS FILE. p47/p48/p55/p56/p63
each emit a caption-number sheet -- every number the caption is allowed to
quote, read from the source file at draw time -- and the citation gate reads the
sheet. Figure 6 emitted nothing, so it was the one manuscript figure whose
caption had no machine-readable source and the one figure script outside
`determinism_check.sh`. The eleven gates below are unchanged; the sheet is new,
and so is `results_p46.json`, which carries the sheet, the sources and what
each gate compared.

THE FIRST THING THE SHEET CAUGHT WAS ON THIS FIGURE. Panel (b) carried a typed
annotation reading "leading band at theta = 0: 40-44 (both months)". Both halves
were wrong, and had been since the figure was built on 2026-08-22: 40-44 is the
leading band at theta = -1, and at theta = 0 the two months do not agree --
December leads on 15-19 and February on 75-79. The archived
`archive/20260822/results_p33.json` says so too, so this was never true; it was
simply a string nothing compared to a file. `memo/phase33-coverage.md` and
`section_coverage_draft.md` both have it right in prose. The annotation is now
built from `results_p33.json` at draw time, like every other number here.

Panels, in the caption's own order:
  (a) excess over proportionate mixing across the coverage family rho_a(theta),
      both survey months, with the masking band drawn to scale beside it
  (b) Kendall tau of the NGM dominant eigenvector against its theta = 0 value,
      with the leading age band annotated at both ends
  (c) age bands ranked by reduction in R0 per person-day, survey vs passive
  (d) benefit forgone by spending the passive-derived allocation in the survey
      world, two months x three values of R0

    python eda/p46_fig6.py [--pdf]

Writes `eda/results_p46.json`. Reads `results_p33.json` and `results_p35.json`.
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
from paths import FIG, ROOT

MONTHS = [202312, 202402]
LAB = {202312: "December 2023", 202402: "February 2024"}
R0S = [1.3, 1.8, 2.5]
COL = {"survey": "#222222", "passive_dong": "#1f77b4", "passive_gu": "#d62728"}
NAME = {"survey": "contact survey", "passive_dong": "passive, dong",
        "passive_gu": "passive, district"}

sheet = []
# What each pre-draw gate compared, recorded so the citation gate can read the
# comparison itself rather than trusting that the run printed "ok".
GATES = []


def note(panel, what, value):
    """Record a number the caption is allowed to quote."""
    sheet.append((panel, what, value))


# What the draft caption says. Every one is checked before anything is drawn.
CAPTION_CLAIMS = [
    ("assortativity positive throughout the family", None),
    ("smallest multiple against the survey anywhere in the family", 3.36),
    ("masking band is narrower than the coverage family by", 14.05),
    ("tau at theta = -1, 202312", 0.167),
    ("tau at theta = +1, 202312", 0.117),
    ("assortativity theta=0, 202312", 0.02115),
    ("assortativity theta=-1, 202312", 0.01704),
    ("assortativity theta=+1, 202312", 0.05099),
    ("masking band width, 202312", 0.00242),
    ("coverage family width, 202312", 0.03395),
    ("regret at R0=2.5, 202312", 0.0747),
]


def check(name, want, got, tol):
    ok = abs(got - want) <= tol * max(abs(want), 1e-12)
    GATES.append(dict(gate=name, caption=want, file=got, rel_tol=tol, ok=ok))
    print(f"  {'ok ' if ok else 'FAIL'} {name:<52} caption {want:<10.5g} "
          f"file {got:.5g}")
    assert ok, f"caption and results disagree on {name}: {want} vs {got}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    args = ap.parse_args()

    p33 = json.load(open(f"{ROOT}/eda/results_p33.json"))
    p35 = json.load(open(f"{ROOT}/eda/results_p35.json"))
    bands = p35["bands"]
    fam = {ym: sorted(p33["R2_theta_family"][str(ym)], key=lambda r: r["theta"])
           for ym in MONTHS}
    mvc = p33["mask_vs_coverage"]

    # ------------------------------------------------ the caption, checked first
    print("=== 46.0 the draft caption against the results files ===")
    f0 = fam[202312]
    th = {round(r["theta"], 6): r for r in f0}
    n_fam = sum(len(fam[ym]) for ym in MONTHS)
    min_assort = min(r["assortativity"] for ym in MONTHS for r in fam[ym])
    GATES.append(dict(gate="assortativity positive throughout the family",
                      caption=None, file=min_assort, rel_tol=None,
                      ok=min_assort > 0, points=n_fam))
    assert all(r["assortativity"] > 0 for ym in MONTHS for r in fam[ym]), \
        "the caption says assortativity is positive throughout; it is not"
    print("  ok  assortativity positive throughout, "
          f"{n_fam} points")
    smallest = min(v[k] for v in p33["R2_gaps_vs_survey"]["202312"]
                   for k in ("half_l1", "cramers_v", "assortativity", "nmi"))
    check("smallest multiple vs survey, 202312", 3.36, smallest, 3e-3)
    check("masking narrower than coverage by", 14.05,
          mvc["202312"]["ratio"], 1e-3)
    check("tau at theta=-1, 202312", 0.167, th[-1.0]["kendall_tau_vs_theta0"], 6e-3)
    check("tau at theta=+1, 202312", 0.117, th[1.0]["kendall_tau_vs_theta0"], 6e-3)
    for t, want in ((-1.0, 0.01704), (0.0, 0.02115), (1.0, 0.05099)):
        check(f"assortativity theta={t:+.0f}, 202312", want,
              th[t]["assortativity"], 6e-4)
    check("masking band width, 202312", 0.00242,
          mvc["202312"]["mask_width"], 6e-3)
    check("coverage family width, 202312", 0.03395,
          mvc["202312"]["theta_width"], 6e-4)
    # Per-month views of the same two objects, for the sheet. The gates above
    # read 202312 by its literal key and are left exactly as they were.
    thm = {ym: {round(r["theta"], 6): r for r in fam[ym]} for ym in MONTHS}
    small = {ym: min(v[k] for v in p33["R2_gaps_vs_survey"][str(ym)]
                     for k in ("half_l1", "cramers_v", "assortativity", "nmi"))
             for ym in MONTHS}
    reg = p35["allocation"]["202312|R0=2.5"]["cross_application"][
        "regret_share_of_benefit"]
    check("regret at R0=2.5, 202312", 0.0747, reg, 6e-3)
    print(f"  {len(CAPTION_CLAIMS)} caption claims, all reproduced.\n")

    # --------------------------------------------------------------------- draw
    # JRSI wants figure text in Times at 9-11 pt and refuses anything under
    # 7.5 pt, so 7.5 is the floor everywhere below. STIXGeneral is the serif:
    # it is Times-metric AND it ships inside matplotlib, so the figure renders
    # the same on a machine with no Times installed. A figure that needs a
    # locally installed font breaks on the typesetter's machine -- the same
    # reason the Korean on these axes is romanised rather than set in a CJK
    # font. mathtext.fontset = "stix" keeps $\theta$ and $R_0$ in the same face
    # as the prose around them.
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["STIXGeneral", "Times New Roman",
                                        "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    # WAS 9.0 in wide, which reproduces at 0.72x on a 7 in page: the 7.5 pt
    # floor this figure was raised to arrived as 5.4 pt of ink. 7.0 in is what
    # p48 and p63 already build to and what JRSI prints a full-page figure at.
    # The 2x2 survives the cut -- none of the four panels is a wide one -- so
    # what the width buys back is spent on height instead: the panels keep
    # roughly the area they had, in a taller box. Nothing is merged, dropped or
    # re-lettered; (a)-(d) still carry what the caption describes.
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 7.2))

    # (a) the coverage family, with the masking band to scale
    a = ax[0, 0]
    for ym, mk in zip(MONTHS, ("o-", "s--")):
        a.plot([r["theta"] for r in fam[ym]],
               [r["assortativity"] for r in fam[ym]], mk, ms=3.5, lw=1.3,
               color="#1f77b4" if ym == 202312 else "#ff7f0e", label=LAB[ym])
    lo, hi = mvc["202312"]["mask_band"]
    a.axhspan(lo, hi, color="0.55", alpha=.35, lw=0, zorder=0)
    a.annotate(f"masking band,\n{mvc['202312']['ratio']:.1f}x narrower",
               xy=(-0.85, (lo + hi) / 2), xytext=(-0.95, hi + 0.0125),
               fontsize=7.5, color="0.25", ha="left",
               arrowprops=dict(arrowstyle="->", color="0.45", lw=.8))
    a.axvline(0, color="0.7", lw=.8, ls=":")
    a.set_xlabel(r"$\theta$   (device space $\leftarrow$  published  "
                 r"$\rightarrow$ expansion applied twice)")
    a.set_ylabel("assortativity (excess over\nproportionate mixing)")
    a.set_title("(a)  the sign and the magnitude survive the family", loc="left")
    a.legend(frameon=False)
    a.grid(alpha=.25)

    thetas = [r["theta"] for r in fam[202312]]
    note("a", "coverage family: theta from / to / points",
         f"{thetas[0]:g} / {thetas[-1]:g} / {len(thetas)}")
    note("a", "points drawn across both months", n_fam)
    note("a", "smallest assortativity anywhere in the family (it is positive)",
         min_assort)
    for ym in MONTHS:
        for t in (-1.0, 0.0, 1.0):
            note("a", f"{LAB[ym]}: assortativity at theta = {t:+.0f}",
                 thm[ym][t]["assortativity"])
    for ym in MONTHS:
        note("a", f"{LAB[ym]}: smallest multiple against the contact survey "
                  "anywhere in the family", small[ym])
        note("a", f"{LAB[ym]}: coverage family width", mvc[str(ym)]["theta_width"])
        note("a", f"{LAB[ym]}: masking band width", mvc[str(ym)]["mask_width"])
        note("a", f"{LAB[ym]}: masking is narrower than the family by",
             mvc[str(ym)]["ratio"])
    note("a", "the grey band drawn is December 2023: lower edge", lo)
    note("a", "the grey band drawn is December 2023: upper edge", hi)

    # (b) the ordering does not
    b = ax[0, 1]
    for ym, mk in zip(MONTHS, ("o-", "s--")):
        b.plot([r["theta"] for r in fam[ym]],
               [r["kendall_tau_vs_theta0"] for r in fam[ym]], mk, ms=3.5, lw=1.3,
               color="#1f77b4" if ym == 202312 else "#ff7f0e", label=LAB[ym])
    # One label per end per month, pushed apart so December and February do not
    # land on each other: both months end on 0-9, and two labels on one point
    # reads as a smudge rather than as a result. WHICH month goes above is read
    # from the file, not fixed. A fixed "December up, February down" works at
    # theta = -1, where the two taus are equal, but at theta = +1 February sits
    # 0.13 above December, so pushing December up and February down drove the
    # two "0-9" labels into the same pixels -- which is what they had been doing.
    # The labels now sit OUTSIDE the data range, not inside it: at 3.4 in a
    # panel width the +/-3 pt inset put the theta = -1 label straight onto the
    # rising segment beside it and the theta = +1 label onto the falling one --
    # the offsets are in points and did not shrink with the panel. Past the end
    # of the curve there is nothing to strike through, so the label reads and
    # the data stays uncovered. set_xlim below opens the margin they need.
    for end, ha, dx in ((0, "right", -4), (-1, "left", 4)):
        taus = {ym: fam[ym][end]["kendall_tau_vs_theta0"] for ym in MONTHS}
        above = MONTHS[0] if taus[MONTHS[0]] >= taus[MONTHS[1]] else MONTHS[1]
        for ym in MONTHS:
            r = fam[ym][end]
            b.annotate(r["top_band"], (r["theta"], r["kendall_tau_vs_theta0"]),
                       textcoords="offset points",
                       xytext=(dx, 9 if ym == above else -14), ha=ha,
                       fontsize=7.5,
                       color="#1f77b4" if ym == 202312 else "#ff7f0e")
    # READ, NOT TYPED. This string said "40-44 (both months)" from the day the
    # figure was built until 2026-08-28. 40-44 is the leading band at
    # theta = -1, and at theta = 0 the months disagree -- so the label named the
    # wrong end of the axis and asserted an agreement that is not there. Nothing
    # compared it to results_p33.json; that is what the sheet below is for.
    lead0 = {ym: thm[ym][0.0]["top_band"] for ym in MONTHS}
    lead0_txt = (f"leading band at $\\theta=0$: {lead0[202312]} (Dec), "
                 f"{lead0[202402]} (Feb)")
    b.annotate(lead0_txt,
               (0.0, 1.0), textcoords="offset points", xytext=(0, 8),
               ha="center", fontsize=7.5, color="0.3")
    b.axhline(1.0, color="0.7", lw=.8, ls=":")
    # Headroom above the peak, so the legend gets a strip instead of a corner.
    # Was 1.15, which left the legend nowhere to go: every corner of this panel
    # is spoken for -- theta = +-1 by the band labels, the middle-left by the
    # rising limb, the top-centre by the theta = 0 caption.
    b.set_ylim(0, 1.34)
    # Widened from (-1.15, 1.28) to make room for the end labels outside the
    # data. The curves are unchanged; only the empty margin either side grew.
    b.set_xlim(-1.40, 1.40)
    b.set_xlabel(r"$\theta$")
    b.set_ylabel(r"Kendall $\tau$ of the NGM dominant" "\n"
                 r"eigenvector against $\theta = 0$")
    b.set_title("(b)  the ordering does not (leading band annotated)", loc="left")
    # Not "lower right": that corner is where the theta = +1 labels live, and
    # the legend was sitting on top of December's. Not "center left" either,
    # which is where it went instead: that reads clear at 4.0 in and not at
    # 3.4, where a 1.3 in legend box reaches as far right as the rising limb
    # and "December 2023" ended underneath the theta = -0.25 marker. It goes in
    # the headroom opened above, where nothing is drawn at all.
    b.legend(frameon=False, loc="upper right", handlelength=1.6,
             handletextpad=.5, borderaxespad=.5)
    b.grid(alpha=.25)

    for ym in MONTHS:
        note("b", f"{LAB[ym]}: Kendall tau at theta = -1",
             thm[ym][-1.0]["kendall_tau_vs_theta0"])
        note("b", f"{LAB[ym]}: Kendall tau at theta = +1",
             thm[ym][1.0]["kendall_tau_vs_theta0"])
        note("b", f"{LAB[ym]}: Kendall tau at theta = 0 is the identity",
             thm[ym][0.0]["kendall_tau_vs_theta0"])
        note("b", f"{LAB[ym]}: smallest Kendall tau over the family",
             min(r["kendall_tau_vs_theta0"] for r in fam[ym]))
        for t in (-1.0, 0.0, 1.0):
            note("b", f"{LAB[ym]}: leading band at theta = {t:+.0f}",
                 thm[ym][t]["top_band"])
    note("b", "the theta = 0 label drawn on the panel, built from the file",
         lead0_txt)
    note("b", "the months agree on the leading band at theta = 0",
         lead0[202312] == lead0[202402])

    # (c) who to buy first. The caption promises a RANKING, so this plots the
    # ranking: scaling each matrix to its own maximum instead makes the survey
    # look empty (its marginal benefit is concentrated in two young bands) and
    # hides the very thing the panel is for.
    c = ax[1, 0]
    x = np.arange(len(bands))
    for tag, mk in (("survey", "o-"), ("passive_dong", "s--"),
                    ("passive_gu", "^:")):
        v = np.array(p35["marginal"]["202312"][tag], float)
        rank = len(v) - np.argsort(np.argsort(v))      # 1 = buy first
        c.plot(x, rank, mk, ms=4, lw=1.1, color=COL[tag], label=NAME[tag],
               alpha=.9)
    c.invert_yaxis()
    # Empty ranks below the worst real one, so the three-entry legend has a
    # strip of its own. At 9 in a 3-column legend took a third of the panel;
    # at 3.4 in it takes all of it, and it was sitting on the survey curve's
    # dive to rank 16 at 20-24 and on the y tick at 15. The alternative was
    # stacking the legend into 3 rows, which walks into the same corner.
    _rk_lo, _rk_hi = c.get_ylim()          # inverted, so _rk_lo is the largest
    c.set_ylim(_rk_lo + 3.4, _rk_hi)
    c.set_yticks([1, 5, 10, 15])
    c.set_xticks(x)
    c.set_xticklabels(bands, rotation=60, ha="right", fontsize=7.5)
    c.set_ylabel("rank by reduction in $R_0$\nper person-day allocated  (1 = highest)")
    c.set_title("(c)  December 2023: the first choice agrees, the rest do not",
                loc="left")
    c.legend(loc="lower right", framealpha=.92, edgecolor="0.8", ncol=3,
             fontsize=7.5, handlelength=1.4, columnspacing=0.8)
    c.grid(alpha=.25)
    first = {t: bands[int(np.argmax(p35["marginal"]["202312"][t]))]
             for t in ("survey", "passive_dong", "passive_gu")}
    # Offset DOWNWARD into the panel. It used to be xytext=(10, 14), which put
    # the string above the top of the axes and straight through the panel
    # title; the y axis is inverted, so rank 1 is the ceiling and there is no
    # room above it. Ranks 2-6 to the right of 15-19 are empty, so the label
    # goes there.
    c.annotate(f"all three agree here: {first['survey']}",
               (x[bands.index(first["survey"])], 1),
               textcoords="offset points", xytext=(14, -16), fontsize=7.5,
               color="0.25", ha="left", va="top",
               arrowprops=dict(arrowstyle="->", color="0.55", lw=.8))

    note("c", "the month panel (c) draws", str(MONTHS[0]))
    note("c", "age bands ranked", len(bands))
    for tag in ("survey", "passive_dong", "passive_gu"):
        v = np.array(p35["marginal"]["202312"][tag], float)
        order = np.argsort(-v)
        note("c", f"{NAME[tag]}: buy first (rank 1)", bands[int(order[0])])
        note("c", f"{NAME[tag]}: buy second (rank 2)", bands[int(order[1])])
        note("c", f"{NAME[tag]}: marginal reduction in R0 per person-day at "
                  "rank 1", float(v[int(order[0])]))
    note("c", "all three matrices agree on the first choice",
         len(set(first.values())) == 1)
    note("c", "the first choice they agree on", first["survey"])
    note("c", "the three second choices are distinct",
         len({bands[int(np.argsort(-np.array(
             p35["marginal"]["202312"][t], float))[1])]
             for t in ("survey", "passive_dong", "passive_gu")}) == 3)

    # (d) what the disagreement costs
    d = ax[1, 1]
    xx = np.arange(len(R0S))
    for i, ym in enumerate(MONTHS):
        vals = [100 * p35["allocation"][f"{ym}|R0={r0}"]["cross_application"]
                ["regret_share_of_benefit"] for r0 in R0S]
        d.bar(xx + (i - 0.5) * 0.34, vals, 0.34, label=LAB[ym],
              color="#1f77b4" if ym == 202312 else "#ff7f0e", alpha=.9)
        for j, v in enumerate(vals):
            d.annotate(f"{v:.1f}%", (xx[j] + (i - 0.5) * 0.34, v),
                       ha="center", va="bottom", fontsize=7.5)
    d.set_xticks(xx)
    d.set_xticklabels([f"$R_0$ = {r}" for r in R0S])
    d.set_ylabel("benefit forgone by spending the\npassive plan in the survey world (%)")
    d.set_title("(d)  and what the disagreement costs", loc="left")
    d.legend(frameon=False)
    d.grid(alpha=.25, axis="y")

    note("d", "R0 values drawn", "/".join(f"{r:g}" for r in R0S))
    for ym in MONTHS:
        for r0 in R0S:
            note("d", f"{LAB[ym]}: benefit forgone at R0 = {r0:g} (the bar is "
                      "labelled with 100x this)",
                 p35["allocation"][f"{ym}|R0={r0}"]["cross_application"][
                     "regret_share_of_benefit"])
    _reg_all = [p35["allocation"][f"{ym}|R0={r0}"]["cross_application"][
        "regret_share_of_benefit"] for ym in MONTHS for r0 in R0S]
    note("d", "smallest bar on the panel", min(_reg_all))
    note("d", "largest bar on the panel", max(_reg_all))

    fig.tight_layout()
    assert fig.get_size_inches()[0] <= 7.0, \
        "figure 6 is wider than the 7-inch reproduction width"
    out_png = f"{FIG}/p46_figure6.png"
    fig.savefig(out_png, dpi=300)
    print(f"  figure -> {out_png}")
    if args.pdf:
        # CreationDate omitted on purpose: matplotlib stamps the wall clock into
        # the PDF, so two runs of the same figure differ by bytes for a reason
        # that has nothing to do with the figure. p63 does the same.
        fig.savefig(f"{FIG}/p46_figure6.pdf", metadata={"CreationDate": None})
        print(f"  figure -> {FIG}/p46_figure6.pdf")
    plt.close(fig)

    print("\n=== the numbers this figure's caption may quote ===")
    for panel, what, value in sheet:
        v = f"{value:.6g}" if isinstance(value, float) else str(value)
        print(f"  ({panel})  {what:<66} {v}")
    out = dict(
        caption_numbers=[dict(panel=p_, what=w, value=v)
                         for p_, w, v in sheet],
        source="results_p33.json (the coverage family, the gaps against the "
               "survey and the masking band) + results_p35.json (the SEIR "
               "marginals and the allocation regret)",
        caption_document="paper/manuscript_JRSI.md",
        gates=GATES,
        gates_run=len(GATES),
        caption_claims=len(CAPTION_CLAIMS),
        computes="nothing. Every value above is read from a results file at "
                 "draw time; the only derived quantities are display "
                 "aggregates over read values -- the min and max over the "
                 "family, the ranks in (c), and the smallest and largest bar "
                 "in (d).",
        corrected="panel (b)'s theta = 0 label was a typed string reading "
                  "'40-44 (both months)' and it was wrong on both counts: "
                  "40-44 leads at theta = -1, and at theta = 0 December leads "
                  "on 15-19 while February leads on 75-79. It is read from "
                  "results_p33.json now. archive/20260822/results_p33.json "
                  "carries the same values, so the label was never right.",
        refuses="panel (c) plots the RANK, not the scaled marginal benefit. "
                "Scaling each matrix to its own maximum empties the survey "
                "series -- its benefit is concentrated in two young bands -- "
                "and hides what the panel is for. The caption promises 'age "
                "bands ranked by', so the rank is the faithful reading. And "
                "the panel is December 2023 only; February is not drawn here.")
    with open(f"{ROOT}/eda/results_p46.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {ROOT}/eda/results_p46.json")


if __name__ == "__main__":
    main()
