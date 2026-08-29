#!/usr/bin/env python
"""Phase 47 — Figures 3 and 4, the two composites the paper still owed.

`paper/paper_structure.md` §7 listed both as "materials exist, has to be
composed": Figure 3 out of `p32_excess`, `p32_spectrum` and `p37_timeseries`,
Figure 4 out of `p38_ksweep79` and `p19_bandwidth`.

IT COMPUTES NO NEW QUANTITY, on the p46 rule. Every curve, bar and singular
value is read from `results_p19/p26/p32/p37/p38/p60/p61.json`. The one thing
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
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from common import AGE_LABEL, AGES
from p32_pmix import pm_null, symmetrise
from paths import FIG, ROOT

PUBLISHED_MONTHS = [202001, 202012, 202312, 202402, 202512, 202606]
TERM = {3, 4, 5, 6, 9, 10, 11}
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
    assert p37["ladder"]["n_monotone"] == p37["ladder"]["n_months"] == 79, \
        "p37: the dong > gu > city ladder no longer holds in every month"
    assert not p37["ladder"]["breaks"], "p37: the ladder has breaks"

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

    print(f"  gates: identity, p34 anchor, adjacency, p18 reproduction, ladder, "
          f"p18b scope pair (worst {worst:.4f} pp outside 20-44), "
          f"p60 anchors + one shared 3-band partition, p61 "
          f"({p61['anchors']['n_ok']}/{p61['anchors']['n']}) — all hold")


# --------------------------------------------------------------------------
# figure 3
# --------------------------------------------------------------------------
def figure3(p26, p27, p32, p37, p61, pdf):
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]      # no 80+ survey egos
    lbl = [AGE_LABEL[AGES[i]] for i in sq]

    mats = []
    for key, title in (("202312|WE|dong|holidayfree", "passive, dong (424)"),
                       ("202312|WE|gu", "passive, district (25)")):
        A = np.array(p26["matrices"][key]["A"])[np.ix_(sq, sq)]
        T, _ = symmetrise(A / pops[REF_MONTH][sq][:, None], pops[REF_MONTH][sq])
        e, _, E = pm_null(T)
        mats.append((np.log10(e / E), title))
    Cs = np.array(p27["survey"]["202312|WE|seoul"]["C"])[np.ix_(sq, sq)]
    Ts, _ = symmetrise(Cs, pops[REF_MONTH][sq])
    e, _, E = pm_null(Ts)
    mats.append((np.log10(np.where(e > 0, e / E, np.nan)), "contact survey, Seoul"))
    vmax = max(np.nanmax(np.abs(m)) for m, _ in mats)
    note(3, "a", "shared colour limit, log10 excess", round(float(vmax), 4))
    for m, t in mats:
        note(3, "a", f"max |log10 excess|, {t}", round(float(np.nanmax(np.abs(m))), 4))

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], hspace=.42, wspace=.28)

    # (a) three excess matrices on one colour scale
    for j, (m, t) in enumerate(mats):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(t, fontsize=9)
        ax.set_xticks(range(len(lbl)))
        ax.set_yticks(range(len(lbl)))
        ax.set_xticklabels(lbl, rotation=90, fontsize=6)
        ax.set_yticklabels(lbl, fontsize=6)
        if j == 0:
            ax.text(-0.34, 1.14, "(a)", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="top")
            ax.text(0.0, 1.075, "log$_{10}$ excess over proportionate mixing, "
                                "2023-12, shared colour limits",
                    transform=ax.transAxes, fontsize=9.5)
        if j == 2:
            fig.colorbar(im, ax=ax, fraction=.046)

    # (b) the rank test, with the survey's own null drawn behind it
    ax = fig.add_subplot(gs[1, 0])
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
    # smaller panel.
    ax.set_ylim(2e-6, 2.0)
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
    ax.legend(handles=handles, fontsize=6.1, loc="lower left", framealpha=.92,
              handlelength=1.8, handletextpad=.6, labelspacing=.35,
              borderpad=.4)
    ax.grid(alpha=.3, which="both")
    ax.text(-0.20, 1.13, "(b)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.set_title("how close to rank one? (2023-12)\n"
                 "grey = the survey's own null at $k=2$", fontsize=9.5)

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
    note(3, "b", "observed empty cells, 15x15 survey block",
         int(emp["observed_zero_cells"]))
    note(3, "b", "null median empty cells", float(emp["null_zero_cells_median"]))

    # (c) the same estimator on 79 months
    ax = fig.add_subplot(gs[1, 1:])
    rows = p37["rows"]
    months = p37["months"]

    def series(scale):
        by = {r["ym"]: r for r in rows
              if r["level"] == scale and r["panel"] == PANEL}
        return np.array([by[m]["assortativity"] for m in months], float)

    for scale, lab, c in (("dong", "dong (424)", TEAL), ("gu", "district (25)", BLUE),
                          ("city", "city (1)", GOLD)):
        ax.plot(range(len(months)), series(scale), "-", c=c, lw=1.3, label=lab)
    dong = series("dong")
    for i, m in enumerate(months):
        if m % 100 in TERM:
            ax.axvspan(i - .5, i + .5, color=TEAL, alpha=.07, lw=0)
    idx = [i for i, m in enumerate(months) if m // 100 == 2020]
    ax.axvspan(min(idx) - .5, max(idx) + .5, color=RED, alpha=.06, lw=0)
    pub = [i for i, m in enumerate(months) if m in PUBLISHED_MONTHS]
    ax.plot(pub, dong[pub], "*", c=RED, ms=12, zorder=5,
            label="the six published months")
    ax.annotate("2020: schools shut,\nthe cycle is absent",
                xy=((min(idx) + max(idx)) / 2, dong.max()),
                ha="center", va="top", fontsize=7, color=RED)
    ax.set_xticks(range(0, len(months), 6))
    ax.set_xticklabels([str(m) for m in months][::6], rotation=90, fontsize=7)
    ax.set_ylabel("assortativity")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=.3)
    ax.text(-0.085, 1.13, "(c)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.set_title(f"the ladder holds in all {len(months)} months; "
                 "shading = school term", fontsize=9.5)

    s = p37["summary_dong"]
    for k in ("assort_min", "assort_max", "assort_median",
              "published_six_min", "published_six_max"):
        note(3, "c", f"dong {k}", round(float(s[k]), 5))
    note(3, "c", "months", int(s["n_months"]))
    note(3, "c", "months where the ladder holds", int(p37["ladder"]["n_monotone"]))

    out = f"{FIG}/p47_figure3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(f"{FIG}/p47_figure3.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# --------------------------------------------------------------------------
# figure 4
# --------------------------------------------------------------------------
def figure4(p18b, p19, p38, p60, pdf):
    # 1x3, and deliberately NOT 2x2 with the March-vs-December panel spanning
    # the bottom row: a full-width bottom row reads as a promotion, and that
    # panel is being demoted. It keeps the widest of the three ratios only
    # because it carries 16 labelled bars and eight legend entries.
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.3]})

    # (a) spatial scale
    ax = axes[0]
    for ym in p38["months"]:
        b = p38["band"][f"{ym}|{PANEL}|adjacency"]
        n = np.array([x["n_loc"] for x in b], float)
        r = np.array([x["assortativity"]["median"] for x in b], float)
        k = r > 0
        pub = ym in PUBLISHED_MONTHS
        ax.plot(n[k], r[k], "-", lw=1.6 if pub else .7,
                color=RED if pub else "0.72", zorder=3 if pub else 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvspan(424, 1e7, color="0.85", zorder=0)
    ax.set_xlim(0.8, 1e4)
    ax.text(700, ax.get_ylim()[1] * .4, "unsupported by these data",
            fontsize=7, rotation=90)
    ax.set_xlabel("number of locations")
    ax.set_ylabel("assortativity")
    ax.grid(alpha=.3, which="both")
    # (a) and (b) are the same estimator on two axes, and they answer in
    # opposite directions. The two notes are a matched pair; they are the whole
    # reason the panels are NOT drawn as a mirror.
    ax.text(.03, .96, "coarsening geography\nlowers r", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, color="#33475B", linespacing=1.3,
            bbox=dict(fc="white", ec="none", alpha=.85, pad=2.0))
    ax.text(-0.17, 1.10, "(a)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.set_title(f"spatial scale: {len(p38['months'])} monthly curves, "
                 "adjacency arm\n(red = the six published months)", fontsize=9.5)
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
        pub = int(ym) in PUBLISHED_MONTHS
        ax.plot(xk, r, "-", lw=1.6 if pub else .7,
                color=RED if pub else "0.72", zorder=3 if pub else 1)
        # Lim et al.'s three bands are not a rung of the nested ladder (3, 8 and
        # 5 boxes, not a dyadic split), so they are a marker and not a point on
        # the curve. Gold is the 3-band colour in (c), so both age panels key
        # the same partition to the same colour; gates() asserts it IS the same
        # partition rather than trusting the two files to agree.
        ax.plot([3], [p60["per_month"][ym]["lim"]["r"]], "o",
                ms=3.2 if pub else 2.0, color=GOLD, mec=GOLD,
                zorder=5 if pub else 4)
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
    ax.text(1.26, ax.get_ylim()[1] * .30, "one bin: r is 0/0, undefined",
            fontsize=7, rotation=90, ha="center", va="bottom")
    ax.text(.97, .96, "coarsening age\nraises r", transform=ax.transAxes,
            va="top", ha="right", fontsize=8.5, color="#33475B", linespacing=1.3,
            bbox=dict(fc="white", ec="none", alpha=.85, pad=2.0))
    ax.set_xlabel("number of age bins")
    ax.set_ylabel("assortativity")
    ax.grid(alpha=.3, which="major")
    handles_b = [Line2D([], [], color="0.72", lw=.9,
                        label=f"one curve per month ({len(p60['months'])})"),
                 Line2D([], [], color=RED, lw=1.6,
                        label="the six published months"),
                 Line2D([], [], color=GOLD, marker="o", ms=3.2, ls="--", lw=1.0,
                        label="Lim et al., 3 bands")]
    ax.legend(handles=handles_b, fontsize=7.0, loc="lower right",
              framealpha=.92, handlelength=1.9, handletextpad=.6)
    ax.text(-0.17, 1.10, "(b)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.set_title(f"age scale: the same {len(p60['months'])} matrices, nested "
                 "ladder\n(red = the six published months)", fontsize=9.5)

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

    # (c) age scale, the March-vs-December composition. Demoted from (b) when
    # the ladder above took that slot; every word of the endpoint ruling, and
    # every number it prints, travels with it unchanged.
    ax = axes[2]
    d16 = {r["age"]: r for r in p19["e_share_change_16band"]}
    d3 = {r["band"]: r for r in p19["e_share_change_3band"]}
    labs = [AGE_LABEL[a] for a in AGES]
    x = np.arange(len(labs))
    mid = np.array([d16[l]["meas"] for l in labs], float)
    lo = np.array([d16[l]["lo"] for l in labs], float)
    hi = np.array([d16[l]["hi"] for l in labs], float)
    # The endpoint filter, settled 2026-08-24. Both markers are p18b's, because
    # p18b is measured before the masking fill: pairing one of its values with a
    # p19 bar would charge the fill to the scope. Outside 20-44 the filled
    # marker lands on its bar (gates() asserts it); inside 20-44 the offset the
    # reader sees between bar and filled marker IS the fill, not the filter.
    b18 = {r["band"]: r for r in p18b["by_age"]}
    both = np.array([b18[l]["both_ends_pp"] for l in labs], float)
    orig = np.array([b18[l]["origin_only_pp"] for l in labs], float)

    ax.bar(x, mid, width=.68, color=[RED if v > 0 else TEAL for v in mid])
    ax.vlines(x, lo, hi, color="#333", lw=1.1)
    ax.axhline(0, color="k", lw=.8)
    ax.vlines(x, np.minimum(both, orig), np.maximum(both, orig),
              color=BLUE, lw=.9, alpha=.9, zorder=8)
    ax.plot(x, both, "D", ms=3.4, color=BLUE, mec=BLUE, ls="none", zorder=9)
    ax.plot(x, orig, "D", ms=5.0, mfc="white", mec=BLUE, mew=1.2, ls="none",
            zorder=9)
    floor = min(lo.min(), both.min(), orig.min())
    ceil = max(hi.max(), both.max(), orig.max())
    ax.set_ylim(floor - .5, ceil + 1.6)
    top = ax.get_ylim()[1]
    start = 0
    for band, n in LIM_SPAN:
        end = start + n
        v = d3[band]["meas"]
        ax.hlines(v, start - .45, end - .55, color=GOLD, lw=3.4, zorder=6)
        ax.annotate(f"{v:+.2f}", (end - .55, v), xytext=(4, 8),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color="#8A5A18", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=.85, pad=.8))
        # The same three bands under both filters, and BOTH from p18b -- the
        # same rule the markers follow. Pairing the gold rule (p19, after the
        # fill) against the origin-only rule (p18b, before it) would have made
        # the 20-59 gap read as scope when most of it is fill: gold -1.42,
        # p18b both-ends -1.47, p18b origin-only -1.18. So the scope is read
        # solid-blue against dashed-blue, and the fill is read gold against
        # solid-blue, exactly as bar against filled marker reads it above.
        vi = p18b["three_band_int"]["band_pp"][band]
        vo = p18b["three_band_out"]["band_pp"][band]
        ax.hlines(vi, start - .45, end - .55, color=BLUE, lw=1.5, zorder=6)
        ax.hlines(vo, start - .45, end - .55, color=BLUE, lw=1.9, ls=(0, (3, 2)),
                  zorder=6)
        ax.annotate(f"{vo:+.2f}", (end - .55, vo), xytext=(4, -10),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=7, color=BLUE, zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=.85, pad=.8))
        ax.text((start + end - 1) / 2, top - .28, band, ha="center", va="top",
                fontsize=9, color="#8A5A18", weight="bold")
        if end < len(labs):
            ax.axvline(end - .5, color="#999", lw=.7, ls=":")
        start = end
    ax.axhspan(top - .95, top, color=GOLD, alpha=.07)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("change in E (discretionary) arrival share, pp")
    ax.grid(axis="y", alpha=.3)
    # The bars are sign-coded, so the legend has to name both colours or it
    # implies the 16-band series is one colour and the reader mistrusts the rest.
    handles = [Patch(fc=RED, label="16 bands, share rises"),
               Patch(fc=TEAL, label="16 bands, share falls"),
               Line2D([], [], color=GOLD, lw=3.4, label="3 bands (Lim et al.)"),
               Line2D([], [], color="#333", lw=1.1,
                      label="masking range, 0 to 3 per cell"),
               Line2D([], [], color=BLUE, marker="D", ms=3.4, ls="none",
                      label="both endpoints in Seoul (p18b)"),
               Line2D([], [], color=BLUE, marker="D", ms=5.0, mfc="white",
                      mew=1.2, ls="none", label="origin in Seoul only (p18b)"),
               Line2D([], [], color=BLUE, lw=1.5,
                      label="3 bands, both endpoints"),
               Line2D([], [], color=BLUE, lw=1.9, ls=(0, (3, 2)),
                      label="3 bands, origin in Seoul")]
    ax.legend(handles=handles, fontsize=7.0, loc="lower left", framealpha=.92,
              ncol=2, columnspacing=1.0, handletextpad=.6)
    ax.text(-0.12, 1.10, "(c)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.set_title("age scale: Dec vs Mar 2020, within-age share\n"
                 "bars = measured masking fill, whiskers = 0..3", fontsize=9.5)
    # The stamp used to read NOT SETTLED, and before that it was clipped to
    # "...ope:" by the legend -- a warning that is present but unreadable is
    # worse than none, because it still looks discharged. The question is
    # settled now (phase1c-scope.md §3), so what the panel has to carry is no
    # longer a warning but the rule: which filter answers which question. It
    # keeps the same empty band above the small right-hand bars.
    ax.text(.985, .885,
            "endpoint filter, settled: co-occurrence -> both endpoints;\n"
            "composition -> origin only. Both shown; the conclusion\n"
            "holds under either, and is stronger under origin-only.",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.9,
            color="#33475B", linespacing=1.35,
            bbox=dict(fc="white", ec=BLUE, lw=.8, alpha=.95, pad=3.0))

    span = {r["resolution"]: r for r in p19["resolution_span"]}
    note(4, "c", "16-band range, pp", span["16 bands"]["range_pp"])
    note(4, "c", "3-band range, pp", span["3 bands (Lim)"]["range_pp"])
    note(4, "c", "fraction of the range three bands retain",
         p19["range_retained_frac"])
    note(4, "c", "largest single absorption, pp",
         max(p19["what_the_bands_absorb"], key=lambda r: abs(r["absorbed"]))["absorbed"])
    note(4, "c", "3-band 60+ , pp (both-ends-in-Seoul scope)", d3["60+"]["meas"])
    note(4, "c", "16-band 80+ , pp (both-ends-in-Seoul scope)", d16["80+"]["meas"])
    # The scope pair, all from p18b so the caption never pairs it with a bar.
    note(4, "c", "p18b 16-band 80+, both endpoints, pp",
         round(b18["80+"]["both_ends_pp"], 4))
    note(4, "c", "p18b 16-band 80+, origin only, pp",
         round(b18["80+"]["origin_only_pp"], 4))
    note(4, "c", "p18b 16-band 20-24, both endpoints, pp",
         round(b18["20-24"]["both_ends_pp"], 4))
    note(4, "c", "p18b 16-band 20-24, origin only, pp",
         round(b18["20-24"]["origin_only_pp"], 4))
    note(4, "c", "p18b 3-band 60+, both endpoints, pp",
         round(p18b["three_band_int"]["band_pp"]["60+"], 4))
    note(4, "c", "p18b 3-band 60+, origin only, pp",
         round(p18b["three_band_out"]["band_pp"]["60+"], 4))
    note(4, "c", "p18b 3-band 20-59, both endpoints, pp",
         round(p18b["three_band_int"]["band_pp"]["20-59"], 4))
    note(4, "c", "p18b 3-band 20-59, origin only, pp",
         round(p18b["three_band_out"]["band_pp"]["20-59"], 4))
    note(4, "c", "zero crossing, both endpoints",
         p18b["claims"]["both_ends"]["zero_crossing"])
    note(4, "c", "zero crossing, origin only",
         p18b["claims"]["origin_only"]["zero_crossing"])
    note(4, "c", "range three bands retain, both endpoints",
         round(p18b["three_band_int"]["range_kept"], 4))
    note(4, "c", "range three bands retain, origin only",
         round(p18b["three_band_out"]["range_kept"], 4))
    note(4, "c", "largest bar-vs-marker gap, pp (the masking fill, not the scope)",
         round(float(max(abs(mid[i] - both[i]) for i in range(len(labs)))), 4))

    out = f"{FIG}/p47_figure4.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(f"{FIG}/p47_figure4.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


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
    figure3(p26, p27, p32, p37, p61, args.pdf)
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
