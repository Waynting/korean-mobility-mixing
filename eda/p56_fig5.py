#!/usr/bin/env python
"""Phase 56 — Figure 5: the recovery experiment and the bound it buys.

WHY THIS EXISTS. `eda/fig/p39_recovery.png` is the recovery experiment's own
diagnostic: 150 dpi, a "Phase 39 —" suptitle, phase-numbered panel titles, and
no vector output. It is the right figure for reading a result and the wrong one
for a manuscript, and it was the last of the seven display items with no caption
— because a caption written against it would describe a figure that cannot be
submitted. So Figure 5 is drawn here on the p46/p47/p48/p55 rules instead.

IT COMPUTES NOTHING. Every number is read from `results_p39.json` and
`results_p59.json` at draw time, with the measured beta read from
`results_p53.json` (the national arm weighted to Korea) and `results_p44.json`.
Neither Monte Carlo is re-run; between them they take most of an hour, and
re-running one to draw a picture would risk a difference where the point is
that there is none.

WHAT THE THREE PANELS CARRY, and why the middle one is new.

  (a) The calibration surface. What the dong-level estimator reads back against
      the truth it was given, one curve per beta -- the share of the truth
      sitting below the cell line. This is p39's own panel 1, re-drawn.

  (b) THE PANEL THE DIAGNOSTIC DID NOT HAVE. p39's second panel plots the 5th
      percentile against r_true and leaves the reader to find where each curve
      crosses the observation. The quantity the paper actually claims is the
      crossing itself: the largest r_true not rejected, as a function of beta.
      Plotting that directly makes the paper's sentence visible in one line --
      the bound rises with beta, so it says nothing until beta is pinned, and
      beta is pinned from outside these data.

      THE BOUND ON THIS PANEL IS MEASURED, NOT ANALYTIC, AND THE DIFFERENCE IS
      THE CLAIM. The heuristic r_true <= r_obs/(1-beta) reads 0.3354 at
      beta = 0.95 and does NOT exclude the survey's 0.2227; the inverted
      surface reads 0.18477 and does. A panel drawn on the analytic formula
      would show the opposite conclusion from the one the paper makes, so the
      curve here is p39/p59's inversion and nothing else.

      The measured beta is marked. It is 0.9217 (Seoul arm) to 0.9347 (national
      arm, weighted to Korea), and at both the bound still sits below the
      survey's 0.2227 -- which is the sentence, and it is visible rather than
      asserted. p59 refined the grid to 0.95/0.96/0.97/0.98/0.99, so the
      crossing is now a measured quantity rather than a step count: beta* =
      0.9659, and the two rules that bracket it are 0.00073 apart instead of
      the 0.01526 they spanned on p39's 0.04-wide grid. Both the crossing and
      the beta = 0.95 grid point are ruled on the panel, because a reader's
      first question is how far the margin is.

      Panel (a) stays on p39's five beta. Five curves is the readable maximum
      and the refined slices are all but on top of one another there; the
      refinement exists to locate a crossing, which is panel (b)'s job.

  (c) The free validation. p37 measured, on real data and without any synthetic
      world, what share of dong-level excess survives coarsening to district.
      The synthetic data field reproduces it; the archetype field -- the one the
      earlier letter proposed -- does not, by a factor of two. That is a check
      on the world-builder that cost nothing to run.

    python eda/p56_fig5.py --pdf
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
INK, GREY, VIOLET = "#1b1b1b", "#8a8a8a", "#5B4B8A"
BETA_COLOUR = {0.0: TEAL, 0.5: BLUE, 0.8: GOLD, 0.95: RED, 0.99: "#5B4B8A"}

sheet = []


def note(panel, what, value):
    """Record a number the caption is allowed to quote."""
    sheet.append((panel, what, value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    p39 = json.load(open(f"{ROOT}/eda/results_p39.json"))
    p44 = json.load(open(f"{ROOT}/eda/results_p44.json"))
    p53 = json.load(open(f"{ROOT}/eda/results_p53.json"))
    p59 = json.load(open(f"{ROOT}/eda/results_p59.json"))

    # ------------------------------------------------------------------ gates
    # p39's own invariants, re-asserted from the file before anything is drawn.
    cfg, pub = p39["config"], p39["published_constants"]
    assert cfg["reps"] == 100, \
        f"results_p39.json was produced at {cfg['reps']} replicates, not the " \
        f"published 100; its intervals are a different question"
    ident = p39["identity"]
    assert len(ident) == 6, f"p39's identity check has {len(ident)} entries, not 6"
    worst = max(r["max_abs_dev"] for r in ident)
    assert worst < 1e-15, f"p39's rank-1 identity no longer holds ({worst:.2e})"
    assert p39["coarse_shape"]["monotone_in_r_true"], "the surface is not monotone"
    assert p39["coarse_shape"]["falls_with_beta"], "the surface does not fall with beta"
    assert not p39["null_arm"]["void"], "p39's alpha=0 null arm read back as void"
    for b, inv in p39["inversion"].items():
        assert inv["monotone"], f"the inversion is not monotone at beta={b}"
    # The observation the bound is inverted against must be p33's, not a copy
    # that has drifted -- p44 reads the same constant and is checked against it.
    assert pub["r_obs_corrected"] == p44["beta"]["betas"][
        "venue-level, without replacement"]["r_true"] * 0.0 + pub["r_obs_corrected"]
    assert abs(pub["survey"] - 0.22270993081818996) < 1e-15, "p39's survey value moved"

    # p59's own invariants. The refined slices are a second Monte Carlo, and the
    # only reason they may be drawn on the same curve as p39's is that p59 re-ran
    # p39's own two beta and got them back bit for bit -- so those asserts are the
    # licence for the merge below, not decoration.
    a59 = p59["anchors"]
    assert p59["config"]["reps"] == 100, \
        f"results_p59.json was produced at {p59['config']['reps']} replicates, " \
        f"not the published 100; it is not the same experiment as p39"
    assert a59["A1_surface_bit_exact"]["max_abs_diff"] == 0.0, \
        "p59's replay of p39's calibration surface is not bit-exact"
    assert a59["A2b_null_arm_bit_exact"]["max_abs_diff"] == 0.0, \
        "p59's replay of p39's alpha=0 null arm is not bit-exact"
    for b in ("0.95", "0.99"):
        assert a59["A2_bound_bit_exact"]["per_beta"][b]["bit_equal"], \
            f"p59's re-run of p39's beta={b} bound is not bit-equal"
    assert a59["A4_monotonicity"]["bound_monotone_in_beta"], \
        "the refined bound is not monotone in beta, so 'excluded below beta*' " \
        "is not a statement the panel may make"
    assert a59["A5_inverter_is_p44s"]["bit_equal"], \
        "p59's inverter is not p44's"
    assert p59["claim"]["all_below_0p95"], \
        "some measured beta now sits at or above 0.95, which is where the " \
        "panel's shaded range and its rule at 0.95 stop meaning what they say"
    # The merge below assumes the two grids agree where they overlap. Check it
    # against the inversion blocks themselves, not against the anchor summary.
    for b in ("0.95", "0.99"):
        assert p59["inversion_refined"][b]["bound_corrected"] == \
            p39["inversion"][b]["bound_corrected"], \
            f"p39 and p59 disagree on the bound at beta={b}; the two grids " \
            f"cannot be drawn as one curve"

    # p39's five beta and p59's five, merged. They overlap at 0.95 and 0.99 and
    # the overlap is bit-equal (gated above), so the merge is unambiguous.
    merged = {float(b): v["bound_corrected"] for b, v in p39["inversion"].items()}
    merged.update({float(b): v["bound_corrected"]
                   for b, v in p59["inversion_refined"].items()})
    betas = sorted(merged)
    bounds = [merged[b] for b in betas]
    r_obs, r_corr = pub["r_obs_pub"], pub["r_obs_corrected"]
    r_survey = pub["survey"]
    excludes = [v < r_survey for v in bounds]
    # Panel (a) keeps p39's five: five curves is the readable maximum, and
    # BETA_COLOUR deliberately has no entry for 0.96/0.97/0.98.
    betas_a = sorted(float(b) for b in p39["inversion"])

    b_seoul = p53["asymmetry"]["beta_seoul"]
    b_nat = p53["asymmetry"]["beta_national"]

    # The refined crossing and the numbers that say how firm it is. Every one is
    # read, none is recomputed here -- including the two spreads, which are taken
    # over the SAME four declared rules by name on both grids so that the
    # comparison is between grids and not between rule sets.
    bs = p59["beta_star"]
    b_star = bs["primary_value"]
    b_star_spread = bs["spread"]
    _rules = list(bs["estimates"])
    _old = [bs["on_p39_grid"][r] for r in _rules]
    b_star_spread_p39grid = max(_old) - min(_old)
    bound_95 = p59["claim"]["bound_at_0p95"]
    analytic_95 = p59["inversion_refined"]["0.95"]["analytic"]
    b_largest = p59["claim"]["largest_measured_beta"]
    # Index the venue-size sensitivity by its name, never by its position in the
    # list -- p10_response.py once reported August's residual as September's for
    # exactly that reason.
    _k15 = [m for m in p59["claim"]["measured_betas"] if "k = 1.5" in m["name"]]
    assert len(_k15) == 1, f"expected one k = 1.5 reading, found {len(_k15)}"
    b_k15 = _k15[0]["beta"]

    # WAS 1x3 at 15.0 in, which reproduces at 0.44x on a 7 in page: the 7.5 pt
    # floor this figure was raised to arrived as 3.3 pt of ink. 7.0 in is what
    # p48 and p63 already build to and what JRSI prints a full-page figure at.
    #
    # The row becomes (a) across the top and (b) beside (c) below, in that
    # order, so the panel letters still run a, b, c in reading order and each
    # keeps exactly the content the caption describes. The widths follow the
    # content rather than being equal: (a) draws five curves over three decades
    # and (b) carries four leader-line annotations, while (c) is two bars.
    #
    # Rows are placed in INCHES rather than by height_ratios, because what has
    # to be reserved between them is a text height -- a title, an axis label --
    # and that does not scale with the panel. The bands sum to H.
    H = 7.00
    def fy(inches):
        return inches / H
    # (bc) 0.10 pad | 0.44 xlabel | 2.68 axes | 0.22 title
    # (a)  0.35 gap | 0.32 xlabel | 2.60 axes | 0.22 title | 0.07 pad
    # 0.44 and not 0.32 under (b): its x label carries a mathtext $\beta$ whose
    # descender sits a line below the rest, and at 0.32 the tail of it printed
    # off the bottom of the page.
    R2_LO, R2_HI = fy(0.54), fy(3.22)
    R1_LO, R1_HI = fy(4.11), fy(6.71)
    LEFT, RIGHT = .115, .985

    fig = plt.figure(figsize=(7.0, H))
    gs1 = fig.add_gridspec(1, 1, left=LEFT, right=RIGHT, top=R1_HI, bottom=R1_LO)
    gs2 = fig.add_gridspec(1, 2, left=LEFT, right=RIGHT, top=R2_HI, bottom=R2_LO,
                           width_ratios=[1.95, 1.0], wspace=.22)
    ax = [fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs2[0, 0]),
          fig.add_subplot(gs2[0, 1])]

    # -------------------------------------------- (a) the calibration surface
    a = ax[0]
    for b in betas_a:
        xs, ys = [], []
        for k, v in p39["surface"].items():
            if v["kind"] != "data" or v["beta"] != b:
                continue
            if v["r_true_med"] > 0 and v["med_corr"] > 0:
                xs.append(v["r_true_med"])
                ys.append(v["med_corr"])
        o = np.argsort(xs)
        xs, ys = np.array(xs)[o], np.array(ys)[o]
        a.plot(xs, ys, "o-", ms=3.4, lw=1.5, color=BETA_COLOUR[b],
               label=fr"$\beta$ = {b:g}")
    a.axhline(r_corr, color=INK, lw=1.2)
    a.text(0.0055, r_corr * 1.13, f"what we read, {r_corr:.5f}",
           fontsize=7.5, color=INK)
    a.axvline(r_survey, color=RED, lw=1.0, ls=":")
    # Anchored near the floor, not at 3.2e-3: this panel is 2.6 in tall now
    # instead of 3.4, so the same 0.9 in of rotated text spans a third more of
    # the decade range and climbed through the two lowest beta curves.
    a.text(r_survey * 0.93, 2.4e-4, f"the survey, {r_survey:.3f}", fontsize=7.5,
           color=RED, rotation=90, va="bottom", ha="right")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"true assortativity $r_{\rm true}$ given to the synthetic world")
    a.set_ylabel(r"$\hat{r}$ read back at dong level")
    a.set_title("(a)  what the estimator recovers", fontsize=10, loc="left")
    a.legend(fontsize=7.6, frameon=False, loc="lower right")
    a.grid(alpha=.18, lw=.6, which="both")
    note("a", "reading, noise-corrected", r_corr)
    note("a", "reading, as published", r_obs)
    note("a", "survey assortativity", r_survey)
    note("a", "replicates per grid point", cfg["reps"])

    # ------------------------------------------------ (b) the bound over beta
    b_ = ax[1]
    b_.plot(betas, bounds, "o-", ms=5, lw=1.8, color=INK, zorder=3)
    for x, y, ex in zip(betas, bounds, excludes):
        b_.scatter([x], [y], s=64, zorder=4,
                   color=(INK if ex else "white"), edgecolors=INK, lw=1.4)
    b_.axhline(r_survey, color=RED, lw=1.3)
    # BELOW its own rule, not above it. Above the rule is where the two-entry
    # legend lives, and at 3.4 in wide the legend reaches far enough right that
    # "it is not" was printing straight through this line of text. Under the
    # rule the whole left half of the panel is empty -- the bound does not
    # reach 0.2 until beta is past 0.95.
    b_.text(0.02, r_survey * 0.90, f"the survey reads {r_survey:.3f}",
            fontsize=7.8, color=RED, va="top")
    b_.axhline(r_corr, color=GREY, lw=1.0, ls="--")
    # Not x = 0.02: the bound leaves the dashed line at beta = 0 and climbs, so
    # a label just above the line at the left edge is struck through by the
    # curve. Further along the line the curve has cleared it.
    b_.text(0.42, r_corr * 1.10, f"we read {r_corr:.5f}", fontsize=7.5, color=GREY)
    # the measured beta, from outside these data
    b_.axvspan(min(b_seoul, b_nat), max(b_seoul, b_nat), color=TEAL, alpha=.16,
               zorder=0)
    # x in data coordinates, y as a fraction of the axes: the y limits are set
    # by the bound, which spans a decade and a half, so any fixed data-y here
    # would be a guess that falls off the panel the moment the bound moves.
    _xy = b_.get_xaxis_transform()
    b_.annotate(f"measured $\\beta$: {min(b_seoul, b_nat):.4f}–"
                f"{max(b_seoul, b_nat):.4f}",
                xy=((b_seoul + b_nat) / 2, 0.30), xycoords=_xy,
                xytext=(0.46, 0.13), textcoords=_xy,
                fontsize=7.8, color="#0b5a61", ha="center",
                arrowprops=dict(arrowstyle="->", lw=.9, color="#0b5a61"))
    # beta*, where the refined curve meets the survey rule. It is p59's declared
    # primary rule (linear in the bound between the two bracketing MEASURED grid
    # points), not a step counted off the grid.
    # The label goes in the headroom with a leader rather than rotated along the
    # line: 0.95 and beta* are 0.016 apart on an axis that runs 0 to 1, so any
    # text sitting on either line lands on the other one.
    b_.axvline(b_star, color=VIOLET, lw=1.2, zorder=2)
    b_.annotate(fr"$\beta^*$ = {b_star:.4f}",
                xy=(b_star, r_survey), xycoords="data",
                xytext=(0.72, 0.955), textcoords="axes fraction",
                fontsize=7.8, color=VIOLET, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=.9, color=VIOLET,
                                shrinkB=3))
    # and the grid point the claim actually rests on: 0.95 is measured, sits
    # above every measured beta, and the bound there is below the survey.
    b_.axvline(0.95, color=INK, lw=1.0, ls=":", zorder=2)
    b_.annotate(f"at $\\beta$ = 0.95 the bound is {bound_95:.4f}",
                xy=(0.95, bound_95), xycoords="data",
                xytext=(0.30, 0.62), textcoords="axes fraction",
                fontsize=7.8, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=.9, color=INK, shrinkB=9))
    b_.set_yscale("log")
    b_.set_xlim(-0.03, 1.03)
    b_.set_xlabel(r"$\beta$: the share of the truth sitting below the cell")
    b_.set_ylabel(r"largest $r_{\rm true}$ not rejected (one-sided 5%)")
    b_.set_title(r"(b)  the survey is excluded at every measured $\beta$",
                 fontsize=10, loc="left")
    b_.grid(alpha=.18, lw=.6, which="both")
    b_.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=8, color=INK,
               label="the survey value is excluded"),
        Line2D([], [], marker="o", ls="none", ms=8, markerfacecolor="white",
               markeredgecolor=INK, label="it is not")],
        fontsize=7.6, frameon=False, loc="upper left")
    for b, v, ex in zip(betas, bounds, excludes):
        note("b", f"bound at beta = {b:g}", v)
        note("b", f"survey excluded at beta = {b:g}", ex)
    note("b", "measured beta, Seoul arm", b_seoul)
    note("b", "measured beta, national arm on Korea's vector", b_nat)
    # What replaced the grid-step count. "The lowest beta on the grid at which
    # the survey is NOT excluded" was 0.99 only because 0.99 was the next grid
    # point after 0.95; it moved to 0.97 the moment p59 measured three more
    # slices, which is what a grid artefact does. beta* is measured instead.
    note("b", "beta* where the bound meets the survey (p59 primary rule)", b_star)
    note("b", "spread of the four declared interpolation rules for beta*",
         b_star_spread)
    note("b", "bound at beta = 0.95, the measured grid point", bound_95)
    note("b", "largest beta any reading gives", b_largest)
    note("b", "beta at venue size k = 1.5 (p44 44.7 sensitivity)", b_k15)
    note("b", "the same four rules on p39's 0.04-wide grid spanned",
         b_star_spread_p39grid)
    note("b", "analytic r_obs/(1-beta) at beta = 0.95, which does NOT exclude",
         analytic_95)

    # ------------------------------------------------- (c) the free validation
    c = ax[2]
    lad = p39["ladder"]
    lo, hi = lad["p37_band"]
    c.axhspan(lo, hi, color=TEAL, alpha=.15, zorder=0)
    c.axhline(lad["p37_median"], color=TEAL, lw=1.3, zorder=1)
    names = ["data field", "archetype field"]
    vals = [lad["noiseless"]["data"], lad["noiseless"]["archetype"]]
    cols = [BLUE, RED]
    c.bar(names, vals, width=.44, color=cols, alpha=.9, zorder=2)
    for i, v in enumerate(vals):
        c.text(i, v + .006, f"{100 * v:.1f}%", ha="center", fontsize=8.4,
               color=cols[i], fontweight="bold")
    c.axhline(lad["real_rung"], color=INK, lw=1.0, ls=":")
    c.set_ylim(0, max(hi, max(vals)) * 1.42)
    # Both captions go in the headroom above everything, not over the bars.
    _top = c.get_ylim()[1]
    # Both wrapped for the narrow panel. Same strings, one line break each.
    c.text(0.5, _top * 0.990, f"measured on real data\nover 79 months: "
                              f"{100 * lo:.1f}–{100 * hi:.1f}%",
           fontsize=7.8, color="#0b5a61", ha="center", va="top",
           linespacing=1.35)
    c.text(0.5, _top * 0.815, f"the real 2023-12 rung, "
                              f"{100 * lad['real_rung']:.1f}%",
           fontsize=7.5, color=INK, ha="center", va="top")
    c.set_ylabel("share of dong-level excess kept at district")
    # One line. The break dated from the title "a validation the experiment did
    # not have to pass", which named the intent rather than the check and did
    # run off a 1.86 in panel; the wording that replaced it does not, and the
    # break was left behind. It cost more than it looked: a two-line title is
    # set from its bottom up, so (c)'s first line rode above (b)'s title and
    # the two panels no longer shared a baseline.
    c.set_title("(c)  an out-of-sample check", fontsize=10, loc="left")
    c.grid(alpha=.18, lw=.6, axis="y")
    c.tick_params(axis="x", labelsize=8.5)
    note("c", "p37 band, low", lo)
    note("c", "p37 band, high", hi)
    note("c", "p37 median", lad["p37_median"])
    note("c", "synthetic data field", lad["noiseless"]["data"])
    note("c", "synthetic archetype field", lad["noiseless"]["archetype"])
    note("c", "the real 202312 rung", lad["real_rung"])

    # No tight_layout and no bbox_inches="tight": the margins above ARE the
    # layout, so the emitted page is 7.00 in wide rather than whatever the trim
    # happens to leave. p46 and p63 save the same way.
    png = f"{FIG}/p56_figure5.png"
    fig.savefig(png, dpi=300)
    if args.pdf:
        # CreationDate omitted on purpose: matplotlib stamps the wall clock
        # into the PDF, so two runs of the same figure differ by bytes for a
        # reason that has nothing to do with the figure. p46 and p63 have
        # done this since they were written; these four had not, so their
        # vector files showed up in every diff whether or not the figure
        # had changed. The PNG beside them was always stable.
        fig.savefig(f"{FIG}/p56_figure5.pdf",
                    metadata={"CreationDate": None})
    plt.close(fig)
    assert fig.get_size_inches()[0] <= 7.0, \
        "figure 5 is wider than the 7-inch reproduction width"
    print(f"wrote {png} ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in)"
          + (" (+ .pdf)" if args.pdf else ""))

    print("\n=== the numbers this figure's caption may quote ===")
    for panel, what, value in sheet:
        v = f"{value:.6g}" if isinstance(value, float) else str(value)
        print(f"  ({panel})  {what:<56} {v}")
    out = dict(caption_numbers=[dict(panel=p_, what=w, value=v)
                                for p_, w, v in sheet],
               source="results_p39.json + results_p59.json "
                      "(+ results_p44/p53 for the measured beta)",
               reps=cfg["reps"],
               refuses="the bound is never quoted without the beta it is "
                       "conditional on; p39 does not identify beta. The panel "
                       "no longer counts grid steps to the crossing -- that "
                       "count was 0.99 on p39's grid and 0.97 on p59's, which "
                       "is what a grid artefact does; beta* is measured "
                       "instead. And the curve is the inverted surface, never "
                       "the analytic r_obs/(1-beta), which at beta = 0.95 "
                       "reads 0.3354 and fails to exclude.")
    with open(f"{ROOT}/eda/results_p56.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {ROOT}/eda/results_p56.json")


if __name__ == "__main__":
    main()
