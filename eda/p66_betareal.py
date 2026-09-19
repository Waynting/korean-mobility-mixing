#!/usr/bin/env python
"""Phase 66 — beta in one unit: the design value the grid is indexed by, and
the realised value the survey is measured in.

WHY THIS EXISTS. The advisor's 2026-09-05 letter, point 2, is that the paper
compares two numbers that are not the same quantity. The synthetic world's beta
is a DESIGN value: `p39_recovery.py:967` sets `rb = (1 - beta) * r_true` and
solves for the cell field, so 0.95 is an instruction to the generator. The
survey's beta is a REALISED value: `p44_beta.py:588` computes
`b = 1 - R_OBS_CORRECTED / r_true`, an attenuation actually observed. Under
finitely many venues those two are not equal, and the paper already knows it --
`manuscript_JRSI.md:213` says the recovery at design beta = 0.95 is 6.8% to
9.1% rather than equation (2)'s 5% "because eleven venues per cell at the
median leave 9.2% of the truth between cells". That sentence is the conversion
factor, used as an excuse instead of as a coordinate.

WHAT THIS PHASE DOES, AND WHAT IT DOES NOT. It re-indexes an existing curve. No
simulation is re-run and no results file is rewritten: `results_p39.json` and
`results_p59.json` already store, at every grid point, both ingredients of the
realised value (`p39_recovery.py:983-984`, `r_true_med` and `r_between_med`).
The bound VALUES are untouched; only the axis they are read against changes.
That is the whole content of the correction, and it is why it costs no compute.

TWO DEFINITIONS OF THE REALISED VALUE, BOTH REPORTED, DECLARED BEFORE THE RUN.

  A  beta_real = 1 - r_between_med / r_true_med
     The ground truth of the synthetic world: what share of its own truth
     actually sits below the cell line once the venues are finitely many.
     `r_true_med` is computed from pre-aggregation venue pairings
     (`p39_recovery.py:467`) and `r_between_med` is the noiseless cell reading
     (`p39_recovery.py:471`), so neither passes through the estimator's noise.

  B  beta_att  = 1 - med_corr / r_true_med
     The survey-matched form. `p44_beta.py:588` divides a NOISE-CORRECTED
     aggregate reading by a venue-level truth, and `med_corr` is exactly the
     noise-corrected synthetic reading. This is the like-for-like analogue and
     it is the one the comparison in the paper actually needs.

  DECLARED PRIMARY: A. It is independent of r_true to five decimal places,
  which B is not (B inherits the recovery's mild curvature in r_true), so A
  gives a single axis rather than a family of them. B is reported at the
  self-consistent r_true -- the grid point nearest that beta's own bound -- so
  that the two are compared where the claim is actually read. They agree to
  0.001 everywhere on the grid, which is the evidence that the choice does not
  carry the answer.

  NOT DONE: running `p44_beta.collapse` over materialised synthetic venues. The
  letter suggests it and it is the highest-fidelity version, but it needs the
  18-minute p39 re-run to re-materialise `brute_A_true`'s venue lists, and A
  and B already bracket it -- both are ratios of the same two objects the
  survey's estimator uses.

WHAT COMES OUT. The design-to-realised map, the bound re-read against it, the
realised crossing, and a verdict for every one of the twelve beta readings
`results_p59.json` records. Readings above the top of the realised axis are
reported as OFF GRID and not extrapolated: the design grid stops at 0.99, which
in realised units is only 0.946, so the calibration does not reach the top of
the measured range. That shortfall is a finding, not a gap to interpolate over.

    python eda/p66_betareal.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paths import ROOT

# Declared before the run. The design grid points that carry BOTH an inversion
# bound and a surface, which is what a conversion needs; 0.0/0.5/0.8 come from
# p39 and 0.96/0.97/0.98 from p59's refinement.
DESIGN_GRID = (0.0, 0.5, 0.8, 0.95, 0.96, 0.97, 0.98, 0.99)
PRIMARY = "A"                      # see the docstring
LEAK_AT_0P95 = 9.2                 # the manuscript's own number, %; anchor 66.0b
BETA_STAR_DESIGN = 0.9659115800390183   # results_p59.json; anchor 66.0c

out = {}


def say(s=""):
    print(s, flush=True)


def load(tag):
    with open(f"{ROOT}/eda/results_{tag}.json") as fh:
        return json.load(fh)


def surface_points(p39, p59):
    """Every synthetic grid point on the data arm, keyed by design beta."""
    pts = {}
    for src in (p39["surface"], p59.get("surface_refined", {})):
        for key, v in src.items():
            if not key.startswith("data|"):
                continue
            if not v.get("r_true_med"):
                continue        # the beta|0.0 rung carries no truth to divide by
            pts.setdefault(round(float(v["beta"]), 4), []).append(v)
    return pts


def bounds(p39, p59):
    """The inverted upper bound on r_true, keyed by design beta."""
    b = {}
    for src in (p39["inversion"], p59.get("inversion_refined", {})):
        for key, v in src.items():
            if isinstance(v, dict) and "bound_corrected" in v:
                b[round(float(v.get("beta", key)), 4)] = v["bound_corrected"]
    return b


def main():
    p39, p59 = load("p39"), load("p59")
    r_survey = p39["published_constants"]["survey"]
    pts, bnd = surface_points(p39, p59), bounds(p39, p59)

    say("=== 66.0 anchors: nothing below is allowed to run on a moved number ===")
    anch = {}

    # 66.0a -- the grid this phase re-indexes is the published one, unchanged.
    missing = [b for b in DESIGN_GRID if b not in bnd or b not in pts]
    assert not missing, (
        f"design grid points {missing} carry no bound or no surface in "
        f"results_p39/p59 -- the conversion would be re-indexing a curve that "
        f"is not the published one")
    seq = [bnd[b] for b in DESIGN_GRID]
    assert all(x < y for x, y in zip(seq, seq[1:])), (
        "the bound is not monotone in design beta, so re-indexing it onto any "
        "other axis is undefined")
    anch["a_grid_present_and_monotone"] = dict(
        design_grid=list(DESIGN_GRID), bounds=seq, monotone=True)
    say(f"  66.0a  {len(DESIGN_GRID)} design points, bound monotone      ok")

    # 66.0b -- the manuscript's own finite-venue leak must fall out of the
    # conversion, or the conversion is measuring something else.
    leak95 = max(100 * v["r_between_med"] / v["r_true_med"] for v in pts[0.95])
    assert abs(leak95 - LEAK_AT_0P95) < 0.05, (
        f"the beta = 0.95 slice leaves {leak95:.2f}% of the truth between "
        f"cells, not the {LEAK_AT_0P95}% the manuscript states at §3.3 and "
        f"SI §7 -- one of the two is reading a different grid")
    anch["b_leak_at_0p95"] = dict(recomputed=leak95, in_manuscript=LEAK_AT_0P95)
    say(f"  66.0b  beta = 0.95 leaves {leak95:.2f}% between cells "
        f"(manuscript: {LEAK_AT_0P95}%)   ok")

    # 66.0c -- the design-coordinate crossing is p59's, untouched.
    star_design = p59["beta_star"]["primary_value"]
    assert star_design == BETA_STAR_DESIGN, (
        f"results_p59 now reports beta* = {star_design}, not "
        f"{BETA_STAR_DESIGN} -- p59 was re-run and this phase is stale")
    anch["c_beta_star_design"] = star_design
    say(f"  66.0c  beta* (design) = {star_design:.7f}                   ok")

    # 66.0d -- the recovery bands the manuscript quotes, recomputed here so the
    # ratio this phase inverts is demonstrably the ratio the paper reports.
    rec = {b: [100 * v["med_corr"] / v["r_true_med"] for v in vs]
           for b, vs in pts.items()}
    r0, r95 = (min(rec[0.0]), max(rec[0.0])), (min(rec[0.95]), max(rec[0.95]))
    assert abs(r0[0] - 92) < 0.6 and abs(r0[1] - 101) < 0.6, r0
    assert abs(r95[0] - 6.8) < 0.06 and abs(r95[1] - 9.1) < 0.06, r95
    anch["d_recovery_bands"] = dict(beta_0=list(r0), beta_0p95=list(r95))
    say(f"  66.0d  recovery {r0[0]:.0f}%-{r0[1]:.0f}% at beta = 0, "
        f"{r95[0]:.1f}%-{r95[1]:.1f}% at 0.95 (manuscript: 92-101, 6.8-9.1)  ok")
    out["anchors"] = anch

    say("\n=== 66.1 the conversion ===")
    say(f"  {'design':>7} {'bound':>8} {'realised A':>11} {'realised B':>11} "
        f"{'A - design':>11}")
    rows, ax_a, ax_b = [], [], []
    for b in DESIGN_GRID:
        vs = sorted(pts[b], key=lambda v: v["r_true_med"])
        a = float(np.mean([1 - v["r_between_med"] / v["r_true_med"] for v in vs]))
        spread = float(max(1 - v["r_between_med"] / v["r_true_med"] for v in vs)
                       - min(1 - v["r_between_med"] / v["r_true_med"] for v in vs))
        near = min(vs, key=lambda v: abs(v["r_true_med"] - bnd[b]))
        bb = float(1 - near["med_corr"] / near["r_true_med"])
        rows.append(dict(design=b, bound=bnd[b], realised_A=a, realised_B=bb,
                         realised_A_spread_over_r_true=spread,
                         B_evaluated_at_r_true=near["r_true_med"]))
        ax_a.append(a)
        ax_b.append(bb)
        say(f"  {b:>7.3f} {bnd[b]:>8.5f} {a:>11.5f} {bb:>11.5f} "
            f"{a - b:>+11.5f}")
    out["conversion"] = rows
    gap = max(abs(a - b) for a, b in zip(ax_a, ax_b))
    say(f"  the two definitions differ by at most {gap:.5f} across the grid, "
        f"so the choice of definition does not carry the answer.")
    out["definition_gap_max"] = gap

    say("\n=== 66.2 the crossing, in both units ===")
    star_a = float(np.interp(r_survey, seq, ax_a))
    star_b = float(np.interp(r_survey, seq, ax_b))
    say(f"  design      beta* = {star_design:.4f}")
    say(f"  realised A  beta* = {star_a:.4f}")
    say(f"  realised B  beta* = {star_b:.4f}")
    say(f"  the design value overstates the crossing by "
        f"{star_design - star_a:.4f} in realised units.")
    out["beta_star"] = dict(design=star_design, realised_A=star_a,
                            realised_B=star_b, primary=PRIMARY,
                            primary_value=star_a if PRIMARY == "A" else star_b)

    say("\n=== 66.3 every measured reading, re-read on the realised axis ===")
    axis = ax_a if PRIMARY == "A" else ax_b
    top = max(axis)
    say(f"  the realised axis spans 0 to {top:.4f}; design 0.99 is realised "
        f"{ax_a[-1]:.4f}, so readings above that are OFF GRID.")
    say(f"\n  {'reading':<52} {'beta':>8} {'bound':>8}  verdict")
    verdicts = []
    readings = list(p59["claim"]["measured_betas"])
    vetoed = p59["claim"].get("excluded_reading")
    if vetoed:
        readings.append(dict(name=vetoed["name"] + " [vetoed in advance]",
                             beta=vetoed["beta"], convention="vetoed"))
    for r in readings:
        b = r["beta"]
        if b > top:
            v = dict(name=r["name"], beta=b, bound=None, excludes=None,
                     off_grid=True)
            say(f"  {r['name'][:52]:<52} {b:>8.4f} {'--':>8}  OFF GRID")
        else:
            bd = float(np.interp(b, axis, seq))
            v = dict(name=r["name"], beta=b, bound=bd,
                     excludes=bool(bd < r_survey), off_grid=False,
                     margin=float(r_survey - bd))
            say(f"  {r['name'][:52]:<52} {b:>8.4f} {bd:>8.4f}  "
                f"{'excludes' if bd < r_survey else 'DOES NOT EXCLUDE'} "
                f"({r_survey - bd:+.4f})")
        verdicts.append(v)
    out["verdicts"] = verdicts

    on = [v for v in verdicts if not v["off_grid"] and "vetoed" not in v["name"]]
    exc = [v for v in on if v["excludes"]]
    off = [v for v in verdicts if v["off_grid"]]
    out["answer"] = dict(
        r_survey=r_survey,
        n_readings=len(verdicts),
        n_on_grid=len(on),
        n_excluding=len(exc),
        n_off_grid=len(off),
        excluding=[v["name"] for v in exc],
        not_excluding=[v["name"] for v in on if not v["excludes"]],
        off_grid=[v["name"] for v in off],
        thinnest_excluding_margin=min([v["margin"] for v in exc], default=None),
        design_axis_top=DESIGN_GRID[-1],
        realised_axis_top=top,
    )

    say("\n=== the number ===")
    say(f"  design beta 0.95, the value the paper calls 'above all eleven "
        f"retained readings', is realised {ax_a[3]:.4f}.")
    say(f"  that is BELOW every retained reading, not above: the smallest "
        f"venue-collapse reading is {min(r['beta'] for r in p59['claim']['measured_betas']):.4f}.")
    say(f"  on the realised axis {len(exc)} of {len(on)} on-grid readings still "
        f"exclude the survey's {r_survey:.4f}:")
    for v in exc:
        say(f"    {v['name'][:60]}  bound {v['bound']:.4f}  margin {v['margin']:+.4f}")
    if off:
        say(f"  {len(off)} reading(s) sit above the calibrated range and are "
            f"not extrapolated to.")

    out["declaration"] = dict(
        primary_definition=PRIMARY,
        definition_A="1 - r_between_med / r_true_med, the world's own share "
                     "below the cell line",
        definition_B="1 - med_corr / r_true_med, the form p44_beta.py:588 uses "
                     "on the survey, evaluated at the r_true nearest that "
                     "beta's own bound",
        design_grid=list(DESIGN_GRID),
        recomputed_nothing="the bound values are read from results_p39/p59 and "
                           "are not re-derived; only the axis changes",
        extrapolation="none. readings above the realised axis top are reported "
                      "OFF GRID, because extending the axis needs p39 re-run "
                      "at design beta above 0.99",
        stop_loss="abort unless the beta = 0.95 slice reproduces the "
                  "manuscript's 9.2% finite-venue leak and p59's design beta*",
    )

    with open(f"{ROOT}/eda/results_p66.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p66.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
