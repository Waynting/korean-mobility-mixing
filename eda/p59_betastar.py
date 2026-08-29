#!/usr/bin/env python
"""Phase 59 — beta*, the critical share below the cell line, MEASURED on a
refined grid rather than interpolated across a 0.04-wide gap.

WHY NOW. The advisor's 2026-08-27 letter, point 1, asks for the beta passage to
become a one-sided argument with a critical value. The reasoning is right and it
runs in our favour: the bound on r_true is monotone increasing in beta, the
direction we need is an upper bound, so an unidentified LOWER end of beta costs
us nothing. What we owe the reader is (a) the bound evaluated at a beta above
every beta we measure, and (b) a single critical value beta* at which the bound
reaches the survey's 0.2227.

THE LETTER'S ALGEBRA IS A HEURISTIC AND THE REPO'S BOUND IS NOT IT. The letter
writes r_true <= r_obs / (1 - beta). That is what you get if the estimator read
back exactly r_between = (1 - beta) r_true. p39's bound is not that: it is an
EMPIRICALLY INVERTED SURFACE (p39_recovery.py:1133-1180, section 39.8) -- the
largest r_true whose 5th-percentile reading still sits at or below the observed
value. Writing phi(beta) = bound(beta) (1 - beta) / r_obs_corrected, the stored
inversion gives

    beta      0.00      0.50      0.80      0.95      0.99
    phi     0.9955    0.9556    0.8519    0.5510    0.1911

so the measured bound is far TIGHTER than the analytic one, and increasingly so
as beta rises (at beta = 0.99: 0.3205 measured against 1.6768 analytic). Tighter
is the direction we want -- it pushes the crossing UP. But it also means beta*
is interpolation-dependent at the second decimal on a 0.04-wide grid: linear in
bound (what p44_beta.py:272 already does) gives 0.96118, linear in phi gives
0.97640, linear in log-bound 0.96357. Handing a reviewer "beta* = 0.961" invites
"your grid is 0.04 wide, where did the third digit come from?" -- the very
objection the letter is trying to close. So this phase measures three new beta
slices and quotes beta* off a 0.01-wide grid.

THE CLAIM DOES NOT DEPEND ON beta* AND MUST NOT BE WRITTEN AS IF IT DID. The
bound is monotone increasing in beta and is MEASURED at beta = 0.95 to be
0.18477, below the survey's 0.22271. Every beta reading we have is below 0.95.
So the sentence is "at beta = 0.95, above every beta we measure, the bound is
still 0.185 and the survey's 0.223 is excluded" -- one measured grid point, no
interpolation, no grid-step counting. beta* is the margin, not the test.

WHY A NEW GRID CAN BE RUN AT ALL, AND WHY IT COMES WITH A FREE ANCHOR.
p39_recovery.py:445 seeds each replicate from (SEED, arm, round(r_true*1e6),
round(beta*1e6), rep) "and from nothing else, so extending the grid in either
direction cannot move a point already computed". Therefore new beta values can
be simulated without touching results_p39.json, AND re-running the beta = 0.95
and beta = 0.99 slices must reproduce results_p39.json["surface"] bit for bit.
That is anchor A1, and it is the only thing that licenses the rest of the file.

WHY THE SURFACE CANNOT BE COLLAPSED IN r_between, so this cannot be had for
free by reparametrising what is already stored. If q_lo_corr depended on
(beta, r_true) only through r_between = (1 - beta) r_true, then
c(beta) = (1 - beta) bound(beta) would be constant. It is not: it falls
0.01669 -> 0.01602 -> 0.01428 -> 0.00924 -> 0.00320 across p39's grid, a factor
of 5.2. The finite-venue leak section 39.8 mentions -- delta enters the realised
cell margin through phi = multinomial(m, P)/m, which does not vanish at finite m
-- is what breaks the collapse. q_lo_corr is genuinely two-dimensional, so new
beta values need new simulation. This is a fact about the surface, not an
implementation note.

NOTHING IS RE-IMPLEMENTED. p39_recovery is imported unchanged and its World,
invert and quantile_curve are called, the way p53_natbeta imports p44_beta and
sets state from outside. The one thing that cannot be imported is p39's
`point()`: it is a closure inside p39's main() over `knobs`, `ARM`, `W`, `NULL`
and `args`, so it is not reachable from outside. `grid_point()` below is that
closure's orchestration re-expressed against the same three World methods
(solve_knob / field / draw); no estimator, no field, no measurement layer and no
seed is redefined here. A1 is what proves the re-expression is faithful.

    python eda/p59_betastar.py                 # the full refined grid, ~15 min
    python eda/p59_betastar.py --reps 3 --smoke # wiring check, partial file
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p39_recovery as P39
import p44_beta as P44
from p39_recovery import World, invert, quantile_curve
from common import AGE_LABEL, AGES
from paths import DERIVED, RESULTS, ROOT, require

# ============================================================ THE DECLARATION
# Written before any analysis code in this file and before the run. CLAUDE.md:
# "Declare the primary cell before the run... Choosing after the fact is
# number-shopping." It is copied verbatim into results_p59.json.
DECLARATION = {
    "grid": "BETAS_STAR = (0.95, 0.96, 0.97, 0.98, 0.99) declared before the "
            "run. 0.95 and 0.99 are p39's own and exist ONLY as anchors; "
            "0.96/0.97/0.98 are new. The grid brackets the crossing at 0.01 "
            "and is NOT chosen after seeing it.",
    "rtrue grid": "p39's RTRUE, unchanged, imported.",
    "reps": "100, matching p39's published --reps 100. Any other value is a "
            "different question.",
    "primary beta*": "linear interpolation of bound_corrected between the two "
                     "bracketing MEASURED grid points, i.e. p39/p44's own "
                     "`invert` convention. Declared primary because the bound "
                     "is convex in beta, so linear-in-bound UNDER-states "
                     "beta*, which is the conservative direction for a claim "
                     "of the form \"beta < beta*\".",
    "reported also": "phi-linear (linear in bound*(1-beta)) and the two "
                     "bound_ci crossings. All four are reported whichever way "
                     "they land.",
    "the claim": "does NOT depend on beta*. It is: bound(0.95) = 0.18477 < "
                 "0.22271 = r_survey, and 0.95 exceeds every measured beta "
                 "(0.93467 national, 0.93258 primary, 0.92167 Seoul, 0.90370 "
                 "at k=1.25, 0.85776 at k=1.5). beta* is reported as the "
                 "margin, not the test.",
    "failure mode": "if the refined grid moves the crossing BELOW 0.95, the "
                    "claim is dead and is reported as dead. If p44's existing "
                    "assert |crossing - 0.96| < 0.01 (p44_beta.py:274) fires "
                    "against the refined value, that is a FINDING: it means "
                    "the 5-point interpolation was materially wrong, and it "
                    "goes in the letter.",
}

BETAS_STAR = (0.95, 0.96, 0.97, 0.98, 0.99)
ANCHOR_BETAS = (0.95, 0.99)            # the two p39 already ran
NULL_REPS = 25                         # p39's --coarse-reps default, its null arm
BOOT = 300                             # p39's 39.8 replicate bootstrap, unchanged
ARM_DATA = 3                           # p39's ARM["data"], the sweep's arm index
NULL_TAGS = {"clean": 0, "mask only": 1, "noise only": 2, "both": 3}

out = {}


def say(s=""):
    print(s, flush=True)


# ------------------------------------------------------------------ the loop
def grid_point(W, knobs, NULL, beta, r_true, reps):
    """One grid point, exactly p39's `point("data", beta, r_true, reps)`.

    p39's version is a closure inside main() and cannot be imported. Every line
    below calls a p39 object: solve_knob, field and draw are World methods and
    the seed tag is p39's, so the only thing re-expressed here is the four-line
    orchestration. A1 checks the re-expression against p39's stored output.
    """
    rb = (1 - beta) * r_true
    ky = ("data", round(rb, 9))
    if ky not in knobs:
        knobs[ky] = W.solve_knob("data", rb)
    if knobs[ky] is None:
        return None
    Pf = W.field("data", knobs[ky])
    delta = float(np.sqrt(max(r_true - rb, 0.0) / max(1 - rb, 1e-12)))
    rows = [W.draw(Pf, delta, i, (ARM_DATA, int(round(r_true * 1e6)),
                                  int(round(beta * 1e6))))
            for i in range(reps)]
    h = np.array([x["r_hat_dong"] for x in rows])
    return dict(kind="data", beta=beta, r_true_target=r_true, knob=knobs[ky],
                delta=delta, n_rep=reps,
                r_true_med=float(np.median([x["r_true"] for x in rows])),
                r_between_med=float(np.median([x["r_between"] for x in rows])),
                med_raw=float(np.median(h)), med_corr=float(np.median(h) - NULL),
                q_lo_raw=float(np.percentile(h, P39.Q_LO)),
                q_lo_corr=float(np.percentile(h, P39.Q_LO) - NULL),
                q_hi_raw=float(np.percentile(h, 100 - P39.Q_LO)),
                r_hat=h.tolist(),
                clipped=float(np.median([x["clipped"] for x in rows])))


def rebuild_null(W, reps):
    """p39's 39.4 alpha = 0 null arm, re-run. Same field, same four arms, same
    tags, so the medians are bit-exact against results_p39.json or the whole
    measurement layer differs and nothing below it is comparable."""
    Pnull = W.field("data", 0.0)
    assert np.abs(Pnull - W.r[None, :]).max() < 1e-12, \
        "the alpha = 0 field is not proportionate mixing"
    arms = {}
    for tag, (noise, mask_err) in (("clean", (False, False)),
                                   ("mask only", (False, True)),
                                   ("noise only", (True, False)),
                                   ("both", (True, True))):
        v = [W.draw(Pnull, 0.0, i, (0, 2, NULL_TAGS[tag]),
                    noise=noise, mask_error=mask_err)["r_hat_dong"]
             for i in range(reps)]
        arms[tag] = dict(median=float(np.median(v)), lo=float(np.min(v)),
                         hi=float(np.max(v)), n=len(v))
    return arms


def bound_ci(surface, beta, keys, xs, NULL, r_obs_corr, n_boot=BOOT):
    """p39's 39.8 replicate bootstrap on the bound, unchanged, including the
    order in which the generator is consumed -- keys arrive from
    quantile_curve() sorted by r_true_med, and the alpha = 0 record consumes
    nothing because its r_hat list is empty."""
    boot = []
    rb = np.random.default_rng([P39.SEED, 7, int(round(beta * 1e6))])
    for _ in range(n_boot):
        q = []
        for t in keys:
            h = surface[t]["r_hat"]
            q.append(float(np.percentile(rb.choice(h, len(h)), P39.Q_LO) - NULL)
                     if h else surface[t]["q_lo_corr"])
        v = invert(xs, q, r_obs_corr)
        if v is not None:
            boot.append(v)
    ci = ([float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
          if len(boot) > 30 else None)
    return ci, len(boot)


# --------------------------------------------------------------- interpolators
def cross_linear(betas, vals, target):
    """p44_beta.py:272's convention: np.interp of beta against the bound. Also
    reports whether the target actually lies inside the measured range, because
    np.interp clamps silently and a clamped value is not a crossing."""
    lo, hi = min(vals), max(vals)
    inside = lo <= target <= hi
    return float(np.interp(target, vals, betas)), bool(inside)


def cross_phi_linear(betas, bounds, target, r_obs):
    """The same crossing, but with phi(beta) = bound (1 - beta) / r_obs taken
    linear between the bracketing grid points instead of the bound itself.
    Closed form: with u = (beta - b0)/(b1 - b0) and phi = p0 + (p1 - p0) u,
    solving phi r_obs / (1 - beta) = target gives
        u = (target (1 - b0) - p0 r_obs) / ((p1 - p0) r_obs + target (b1 - b0)).
    """
    phis = [b_ * (1 - be) / r_obs for be, b_ in zip(betas, bounds)]
    for i in range(1, len(betas)):
        if bounds[i - 1] <= target < bounds[i]:
            b0, b1 = betas[i - 1], betas[i]
            p0, p1 = phis[i - 1], phis[i]
            num = target * (1 - b0) - p0 * r_obs
            den = (p1 - p0) * r_obs + target * (b1 - b0)
            if den == 0:
                return None, phis
            return float(b0 + (b1 - b0) * num / den), phis
    return None, phis


def cross_log(betas, bounds, target):
    lb = [float(np.log(v)) for v in bounds]
    return float(np.interp(float(np.log(target)), lb, betas))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100,
                    help="replicates per grid point; 100 is the published value")
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check: two r_true values, no bit-exact anchors, "
                         "writes results_p59.partial.json")
    args = ap.parse_args()
    t0 = time.time()
    full = (args.reps == 100) and not args.smoke
    require(DERIVED / f"arrivals_dow_{P39.YM}.parquet",
            f"p26 arrival cache for {P39.YM} (run eda/p26_matrix.py)")

    p39 = json.load(open(f"{ROOT}/eda/results_p39.json"))
    p44 = json.load(open(f"{ROOT}/eda/results_p44.json"))
    p53 = json.load(open(f"{ROOT}/eda/results_p53.json"))
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))
    p33 = json.load(open(f"{ROOT}/eda/results_p33.json"))

    say("=== 59.0 the declaration, as written before the run ===")
    for k, v in DECLARATION.items():
        say(f"  {k:<14} {v}")
    out["declaration"] = DECLARATION

    # ---------------------------------------------------- A3 the constants
    # Re-read from the file that owns them, the way p44_beta.py:258-266 does.
    # Nothing in this file retypes a published number as a literal.
    say("\n=== 59.1 anchor A3: the published constants, re-read not retyped ===")
    pub = p39["published_constants"]
    R_OBS_PUB = pub["r_obs_pub"]
    NOISE_BIAS = pub["noise_bias"]
    R_OBS_CORR = pub["r_obs_corrected"]
    R_SURVEY = pub["survey"]
    a3 = {}
    for nm, got, want in (("r_obs_pub", R_OBS_PUB, P39.R_OBS_PUB),
                          ("noise_bias", NOISE_BIAS, P39.P33_NOISE_BIAS),
                          ("r_obs_corrected", R_OBS_CORR, P39.R_OBS_CORRECTED),
                          ("survey", R_SURVEY, P39.R_SURVEY)):
        a3[nm] = dict(from_results_p39=got, in_p39_module=want,
                      bit_equal=bool(got == want))
        assert got == want, f"{nm}: p39 module has {want!r}, its file has {got!r}"
        say(f"  {nm:<17} {got!r}   bit-equal to the p39 module")
    assert abs(R_OBS_PUB - NOISE_BIAS - R_OBS_CORR) < 1e-15
    say(f"  identity  {R_OBS_PUB:.8f} - {NOISE_BIAS:.8f} = {R_OBS_CORR:.8f}")
    # p33 owns the first three; assert against p33 itself, not only against p39.
    st = p33["R1_effective_sample"]["202312"]["stats"]["assortativity"]
    for nm, got, want in (("r_obs_pub", R_OBS_PUB, st["point"]),
                          ("noise_bias", NOISE_BIAS, st["noise_bias"]),
                          ("r_obs_corrected", R_OBS_CORR, st["bias_corrected"])):
        assert got == want, f"{nm} disagrees with results_p33.json"
        a3[nm]["from_results_p33"] = want
    say("  the same three re-read a second time from results_p33.json: 3/3 equal")
    out.setdefault("anchors", {})["A3_published_constants"] = a3

    # ---------------------------------------------- A5 the inverter is p44's
    say("\n=== 59.2 anchor A5: this file's inverter is p44's inverter ===")
    inv39 = sorted((float(b), v) for b, v in p39["inversion"].items())
    b39 = [b for b, _ in inv39]
    d39 = [v["bound_corrected"] for _, v in inv39]
    x39, _in39 = cross_linear(b39, d39, R_SURVEY)
    want44 = p44["beta"]["p39_crossing"]
    say(f"  interpolated from p39's original 5 points : {x39!r}")
    say(f"  results_p44.json .beta.p39_crossing       : {want44!r}")
    say(f"  |diff| {abs(x39 - want44):.1e}")
    assert x39 == want44, (
        "this file's beta* inverter does not reproduce p44's stored crossing; "
        "every beta* below would be a different convention")
    assert want44 == p44["anchors"]["p39_crossing_interpolated"]
    out["anchors"]["A5_inverter_is_p44s"] = dict(
        recomputed=x39, results_p44=want44, bit_equal=True,
        also_matches_anchor_block=True)

    # the p39 grid's phi, for the memo's "the analytic bound is not this bound"
    _, phi39 = cross_phi_linear(b39, d39, R_SURVEY, R_OBS_CORR)
    p39_table = [dict(beta=b, bound_corrected=v["bound_corrected"],
                      bound_raw=v["bound_raw"], bound_ci=v["bound_ci"],
                      excludes_survey=v["excludes_survey"], phi=ph,
                      analytic=R_OBS_CORR / (1 - b),
                      c_of_beta=(1 - b) * v["bound_corrected"])
                 for (b, v), ph in zip(inv39, phi39)]
    say("\n  p39's stored grid, with phi = bound (1-beta) / r_obs_corrected:")
    say(f"  {'beta':>5} {'bound':>10} {'analytic':>10} {'phi':>8} "
        f"{'(1-b)*bound':>12}")
    for row in p39_table:
        say(f"  {row['beta']:>5.2f} {row['bound_corrected']:>10.5f} "
            f"{row['analytic']:>10.5f} {row['phi']:>8.4f} "
            f"{row['c_of_beta']:>12.5f}")
    _c = [r["c_of_beta"] for r in p39_table]
    say(f"  (1-beta)*bound falls {_c[0]:.5f} -> {_c[-1]:.5f}, a factor of "
        f"{_c[0] / _c[-1]:.2f}: the surface does NOT collapse in r_between, so "
        f"new beta values need new simulation.")
    out["p39_stored_grid"] = p39_table
    out["collapse_check"] = dict(
        c_of_beta=_c, ratio_max_over_min=float(max(_c) / min(_c)),
        collapses=False,
        reading="if q_lo_corr depended on (beta, r_true) only through "
                "r_between = (1-beta) r_true then (1-beta)*bound would be "
                "constant; it falls by a factor of "
                f"{max(_c) / min(_c):.2f} across p39's grid. The finite-venue "
                "leak (delta enters the realised cell margin through "
                "phi = multinomial(m, P)/m) is what breaks the collapse.")

    # ------------------------------------------------------ the world, p39's
    say("\n=== 59.3 the synthetic world, built by p39's own World ===")
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}
    w = np.array([cov[AGE_LABEL[a]] for a in AGES], float)
    assert np.abs(w - np.array(p33["profile"]["w"])).max() == 0.0, \
        "the coverage profile differs from the one p33 pushed through the matrix"
    W = World(P39.YM, w, pops[P39.YM])
    W.build_base_field()
    # build_archetype_field() is deliberately NOT called. It sets only c_arch /
    # share_arch, which the "data" field and draw() never read, and it consumes
    # no random state (assign_prices is a deterministic price loop). A1 below is
    # what proves the omission cannot have moved anything.

    # ------------------------------------------------------- A2b the null arm
    say("\n=== 59.4 anchor A2b: p39's alpha = 0 null arm, re-run ===")
    if full:
        arms = rebuild_null(W, NULL_REPS)
        stored = p39["null_arm"]["arms"]
        worst = 0.0
        for k in ("clean", "mask only", "noise only", "both"):
            for f in ("median", "lo", "hi"):
                worst = max(worst, abs(arms[k][f] - stored[k][f]))
            say(f"  {k:<11} median {arms[k]['median']:+.9f}   stored "
                f"{stored[k]['median']:+.9f}")
        NULL = arms["both"]["median"]
        assert worst == 0.0, (
            f"the null arm does not reproduce results_p39.json (max |diff| "
            f"{worst:.2e}); the measurement layer differs and A1 cannot hold")
        say(f"  12 quantities (4 arms x median/lo/hi), max |diff| = {worst:.1e}")
        out["anchors"]["A2b_null_arm_bit_exact"] = dict(
            max_abs_diff=worst, bit_exact=True, null_median=NULL,
            arms=arms, reps=NULL_REPS)
    else:
        NULL = p39["null_arm"]["null_median"]
        say(f"  --smoke / --reps != 100: null read from results_p39.json "
            f"({NULL!r}), not re-run")
        out["anchors"]["A2b_null_arm_bit_exact"] = dict(
            skipped="partial run", null_median=NULL)

    # ------------------------------------------------------- the refined grid
    rtrue = list(P39.RTRUE) if full else [0.1, 0.22, 0.35]
    say(f"\n=== 59.5 the refined grid: beta in {BETAS_STAR}, "
        f"{len(rtrue)} r_true values, {args.reps} replicates a point ===")
    knobs, surface = {}, {}
    null_rec = dict(kind="data", beta=0.0, r_true_target=0.0, knob=0.0,
                    delta=0.0, n_rep=p39["null_arm"]["arms"]["both"]["n"],
                    r_true_med=0.0, r_between_med=0.0, med_raw=NULL,
                    med_corr=0.0,
                    q_lo_raw=p39["null_arm"]["arms"]["both"]["lo"],
                    q_lo_corr=p39["null_arm"]["arms"]["both"]["lo"] - NULL,
                    q_hi_raw=p39["null_arm"]["arms"]["both"]["hi"],
                    r_hat=[], clipped=0.0)
    for beta in BETAS_STAR:
        say(f"  beta = {beta}"
            + ("   [p39 anchor slice]" if beta in ANCHOR_BETAS else "   [new]"))
        surface[f"data|{beta}|0.0"] = dict(null_rec, beta=beta)
        for rt in rtrue:
            rec = grid_point(W, knobs, NULL, beta, rt, args.reps)
            if rec is None:
                say(f"    r_true {rt}: r_between {(1 - beta) * rt:.4f} beyond "
                    f"the field's reach, skipped")
                continue
            surface[f"data|{beta}|{rt}"] = rec
            say(f"    r_true {rt:>6.3f} -> r_between {rec['r_between_med']:.5f}"
                f", r_hat {rec['med_raw']:.5f} [q05 {rec['q_lo_raw']:.5f}]"
                f", recovered {rec['med_corr'] / rec['r_true_med']:.1%}"
                f"  ({time.time() - t0:.0f}s)")
    out["surface_refined"] = surface

    # ---------------------------------------------------------- A1 bit-exact
    say("\n=== 59.6 anchor A1: the two p39 slices, bit for bit ===")
    a1 = {}
    if full:
        worst_all = 0.0
        for beta in ANCHOR_BETAS:
            worst = 0.0
            ncmp = 0
            for rt in rtrue:
                k = f"data|{beta}|{rt}"
                mine, theirs = surface[k], p39["surface"][k]
                assert len(mine["r_hat"]) == len(theirs["r_hat"]) == 100
                d = max(abs(a - b) for a, b in zip(mine["r_hat"],
                                                   theirs["r_hat"]))
                for f in ("q_lo_corr", "med_corr", "q_lo_raw", "med_raw",
                          "r_true_med", "r_between_med", "knob", "delta"):
                    d = max(d, abs(mine[f] - theirs[f]))
                ncmp += len(mine["r_hat"]) + 8
                worst = max(worst, d)
            a1[str(beta)] = dict(max_abs_diff=worst, n_compared=ncmp,
                                 bit_exact=bool(worst == 0.0))
            say(f"  beta {beta}: {ncmp} quantities, max |diff| = {worst:.1e}")
            worst_all = max(worst_all, worst)
        assert worst_all == 0.0, (
            "the refined grid does not reproduce results_p39.json's surface at "
            "beta = 0.95 / 0.99; this World is not p39's World and nothing else "
            "in this file means anything")
        say(f"  A1 PASSES: max |diff| over both slices = {worst_all:.1e}")
        out["anchors"]["A1_surface_bit_exact"] = dict(
            per_beta=a1, max_abs_diff=worst_all, bit_exact=True)
    else:
        say("  --smoke / --reps != 100: A1 not applicable (p39 stored 100 reps)")
        out["anchors"]["A1_surface_bit_exact"] = dict(skipped="partial run")

    # -------------------------------------------------------- the inversion
    say("\n=== 59.7 the inversion on the refined grid ===")
    inv = {}
    for beta in BETAS_STAR:
        xs, qlo_c, med_c, keys = quantile_curve(surface, beta, "corr")
        _, qlo_r, med_r, _ = quantile_curve(surface, beta, "raw")
        mono = all(qlo_c[i] <= qlo_c[i + 1] for i in range(len(qlo_c) - 1))
        b_corr = invert(xs, qlo_c, R_OBS_CORR)
        b_raw = invert(xs, qlo_r, R_OBS_PUB)
        ci, nb = bound_ci(surface, beta, keys, xs, NULL, R_OBS_CORR)
        inv[str(beta)] = dict(beta=beta, r_true=xs, q_lo_corr=qlo_c,
                              med_corr=med_c, monotone=bool(mono),
                              bound_corrected=b_corr, bound_raw=b_raw,
                              bound_ci=ci, n_boot=nb,
                              excludes_survey=(b_corr is not None
                                               and b_corr < R_SURVEY),
                              phi=(b_corr * (1 - beta) / R_OBS_CORR
                                   if b_corr is not None else None),
                              analytic=R_OBS_CORR / (1 - beta),
                              is_p39_anchor=beta in ANCHOR_BETAS)
        s_c = f"[{ci[0]:.5f}, {ci[1]:.5f}]" if ci else "n/a"
        s_b = f"{b_corr:.5f}" if b_corr is not None else "none in grid"
        s_r = f"{b_raw:.5f}" if b_raw is not None else "none in grid"
        s_p = (f"{inv[str(beta)]['phi']:.4f}"
               if inv[str(beta)]["phi"] is not None else "n/a")
        say(f"  beta {beta:>4.2f}: monotone {str(mono):>5} | bound "
            f"{s_b:>13} {s_c} | raw {s_r:>13} | phi {s_p:>6} | "
            f"{'EXCLUDES' if inv[str(beta)]['excludes_survey'] else 'does NOT exclude'}"
            f" the survey's {R_SURVEY:.4f}")
    assert all(v["bound_corrected"] is not None for v in inv.values()), (
        "a refined beta slice never crosses the observation inside the r_true "
        "grid, so its bound is undefined and beta* cannot be interpolated")
    out["inversion_refined"] = inv

    # ------------------------------------------------------------ A2, A4
    say("\n=== 59.8 anchors A2 and A4 ===")
    a2 = {}
    if full:
        for beta in ANCHOR_BETAS:
            got = inv[str(beta)]["bound_corrected"]
            want = p39["inversion"][str(beta)]["bound_corrected"]
            a2[str(beta)] = dict(recomputed=got, results_p39=want,
                                 abs_diff=abs(got - want),
                                 bit_equal=bool(got == want))
            say(f"  bound at beta {beta}: {got!r} vs stored {want!r}  "
                f"|diff| {abs(got - want):.1e}")
            assert got == want, f"bound at beta={beta} moved"
            gci = inv[str(beta)]["bound_ci"]
            wci = p39["inversion"][str(beta)]["bound_ci"]
            a2[str(beta)]["ci_recomputed"] = gci
            a2[str(beta)]["ci_results_p39"] = wci
            a2[str(beta)]["ci_bit_equal"] = bool(gci == wci)
            say(f"    its bootstrap CI {gci} vs stored {wci}  "
                f"{'bit-equal' if gci == wci else 'DIFFERS'}")
            assert gci == wci, (
                f"the bootstrap CI at beta={beta} does not reproduce; the "
                f"generator is being consumed in a different order")
        say("  A2 PASSES: both bounds and both CIs are bit-equal to p39")
        out["anchors"]["A2_bound_bit_exact"] = dict(per_beta=a2, bit_exact=True)
    else:
        say("  --smoke / --reps != 100: A2 not applicable")
        out["anchors"]["A2_bound_bit_exact"] = dict(skipped="partial run")

    betas_r = list(BETAS_STAR)
    bounds_r = [inv[str(b)]["bound_corrected"] for b in betas_r]
    mono_beta = all(bounds_r[i] < bounds_r[i + 1]
                    for i in range(len(bounds_r) - 1))
    mono_rt = {str(b): inv[str(b)]["monotone"] for b in betas_r}
    say(f"  A4 bound monotone increasing in beta: {mono_beta}")
    say(f"  A4 each slice monotone in r_true    : {all(mono_rt.values())}")
    assert mono_beta, "the bound is not monotone increasing in beta"
    assert all(mono_rt.values()), "a slice is not monotone in r_true"
    out["anchors"]["A4_monotonicity"] = dict(
        bound_monotone_in_beta=bool(mono_beta), bounds=bounds_r,
        slice_monotone_in_rtrue=mono_rt)

    # ------------------------------------------------------------- beta star
    say("\n=== 59.9 beta*: four rules, one grid ===")
    ci_lo = [inv[str(b)]["bound_ci"][0] for b in betas_r]
    ci_hi = [inv[str(b)]["bound_ci"][1] for b in betas_r]
    bs_lin, in_lin = cross_linear(betas_r, bounds_r, R_SURVEY)
    bs_phi, phis_r = cross_phi_linear(betas_r, bounds_r, R_SURVEY, R_OBS_CORR)
    bs_clo, in_clo = cross_linear(betas_r, ci_lo, R_SURVEY)
    bs_chi, in_chi = cross_linear(betas_r, ci_hi, R_SURVEY)
    bs_log = cross_log(betas_r, bounds_r, R_SURVEY)
    dead = bounds_r[0] > R_SURVEY
    above_grid = bounds_r[-1] < R_SURVEY
    ests = dict(
        linear_in_bound=dict(
            value=bs_lin, rule="np.interp(r_survey, bound_corrected, beta) "
                               "between the two bracketing measured grid "
                               "points -- p44_beta.py:272's convention",
            primary=True, target_inside_measured_range=in_lin),
        phi_linear=dict(
            value=bs_phi, rule="phi(beta) = bound (1-beta)/r_obs_corrected taken "
                               "linear between the bracketing points, then "
                               "bound = phi r_obs/(1-beta) solved for beta",
            primary=False, target_inside_measured_range=in_lin),
        ci_low=dict(value=bs_clo, rule="the same linear rule applied to the LOW "
                                       "end of the bound's replicate bootstrap CI",
                    primary=False, target_inside_measured_range=in_clo),
        ci_high=dict(value=bs_chi, rule="the same linear rule applied to the HIGH "
                                        "end of that CI",
                     primary=False, target_inside_measured_range=in_chi),
    )
    vals = [v["value"] for v in ests.values() if v["value"] is not None]
    spread = float(max(vals) - min(vals))
    for k, v in ests.items():
        say(f"  {k:<16} beta* = {v['value']!r}"
            + ("   <- PRIMARY (declared)" if v["primary"] else ""))
    say(f"  {'log_in_bound':<16} beta* = {bs_log!r}   (reported, not one of "
        f"the declared four)")
    say(f"  spread over the four declared rules: {spread:.5f}")
    say(f"  on p39's 0.04-wide grid the same four were "
        f"{x39:.5f} / {cross_phi_linear(b39, d39, R_SURVEY, R_OBS_CORR)[0]:.5f}"
        f" / {cross_linear(b39, [v['bound_ci'][0] for _, v in inv39], R_SURVEY)[0]:.5f}"
        f" / {cross_linear(b39, [v['bound_ci'][1] for _, v in inv39], R_SURVEY)[0]:.5f}")
    out["beta_star"] = dict(
        estimates=ests, log_in_bound=bs_log, spread=spread,
        primary="linear_in_bound", primary_value=bs_lin,
        grid_width=0.01, claim_dead=bool(dead), crossing_above_grid=bool(above_grid),
        on_p39_grid=dict(
            linear_in_bound=x39,
            phi_linear=cross_phi_linear(b39, d39, R_SURVEY, R_OBS_CORR)[0],
            ci_low=cross_linear(b39, [v["bound_ci"][0] for _, v in inv39],
                                R_SURVEY)[0],
            ci_high=cross_linear(b39, [v["bound_ci"][1] for _, v in inv39],
                                 R_SURVEY)[0],
            log_in_bound=cross_log(b39, d39, R_SURVEY),
            grid_width=0.04),
        moved_by=float(bs_lin - x39))

    # p44's live assert, evaluated against the refined value. The constant is
    # read off the p44 module itself (p44_beta.py:87), not retyped or rounded
    # back out of its own output.
    _p44c = P44.P39_BETA_AT_SURVEY
    p44_assert = abs(bs_lin - _p44c) < 0.01
    say(f"\n  p44_beta.py:274 asserts |crossing - {_p44c:.2f}| < 0.01 "
        f"(the constant at p44_beta.py:87). Against the refined "
        f"beta* = {bs_lin:.5f} that is |{bs_lin - _p44c:.5f}|"
        f" -> {'STILL HOLDS' if p44_assert else 'WOULD FIRE'}.")
    out["p44_assert"] = dict(
        constant=_p44c, refined_beta_star=bs_lin,
        abs_diff=float(abs(bs_lin - _p44c)),
        tolerance=0.01, still_holds=bool(p44_assert),
        note="p44 asserts against ITS OWN 5-point interpolation, which this "
             "phase does not change; this row asks the counterfactual question "
             "'would the assert survive being pointed at the refined value'.")

    # ------------------------------------------------------------- the claim
    say("\n=== 59.10 the claim, which does not use beta* ===")
    measured = [
        ("national pooled, Korea weighting (p53, the reported national arm)",
         p53["scope"]["national pooled, Korea weighting (corrected)"]["beta"],
         "venue-level, without replacement"),
        ("national pooled, Seoul weighting (p44, published primary)",
         p44["beta"]["betas"]["venue-level, without replacement"]["beta"],
         "venue-level, without replacement"),
        ("Seoul 202312 (p44/p53)",
         p53["scope"]["seoul 202312, Seoul weighting (unchanged)"]["beta"],
         "venue-level, without replacement"),
        ("survey's own contact matrix, star pairs (p44)",
         p44["beta"]["betas"]["survey contact matrix (star pairs)"]["beta"],
         "star pairs, not the venue collapse"),
        ("national, one-convention (p53 53.2 sensitivity)",
         p53["convention_sensitivity"]["arms"]["national (korea)"][
             "beta_one_convention"],
         "venue-level WOR, passive side put on p49's convention"),
        ("Seoul, one-convention (p53 53.2 sensitivity)",
         p53["convention_sensitivity"]["arms"]["seoul"]["beta_one_convention"],
         "venue-level WOR, passive side put on p49's convention"),
    ] + [(f"venue size k = {r['k']} (p44 44.7 sensitivity)", r["beta"],
          "venue-level WOR, declared venue-size curve")
         for r in p44["sensitivity"]["venue_size"]["rows"]]
    withheld = ("venue-level, WITH replacement (p44 44.2)",
                p44["beta"]["betas"]["venue-level, with replacement"]["beta"])
    b095 = inv["0.95"]["bound_corrected"]
    assert b095 is not None, "the bound at beta = 0.95 is undefined"

    dominated = [(n, v) for n, v, _ in measured if v < 0.95]
    not_dominated = [(n, v) for n, v, _ in measured if v >= 0.95]
    say(f"  bound at the MEASURED beta = 0.95 : {b095!r}")
    say(f"  the survey reads                  : {R_SURVEY!r}")
    say(f"  excluded                          : {b095 < R_SURVEY}")
    say(f"  beta readings strictly below 0.95 : {len(dominated)} of "
        f"{len(measured)}"
        + (f"   (largest {max(v for _, v in dominated):.5f})" if dominated else ""))
    for n, v, _c in sorted(measured, key=lambda z: -z[1]):
        say(f"      {v:.5f}  {n}")
    if not_dominated:
        for n, v in not_dominated:
            say(f"  ⚠ NOT below 0.95: {v:.5f}  {n}")
    say(f"  withheld from the list, because p44 44.2's shuffle null rejects the "
        f"convention: {withheld[1]:.5f}  {withheld[0]}")
    out["claim"] = dict(
        bound_at_0p95=b095, r_survey=R_SURVEY,
        excluded=bool(b095 < R_SURVEY),
        margin=float(R_SURVEY - b095),
        measured_betas=[dict(name=n, beta=v, convention=c)
                        for n, v, c in measured],
        largest_measured_beta=float(max(v for _, v, _ in measured)),
        all_below_0p95=bool(not not_dominated),
        excluded_reading=dict(name=withheld[0], beta=withheld[1],
                              why="p44 44.1-44.2: with-replacement pairing puts "
                                  "1/6 of a four-person venue's mass on the "
                                  "diagonal and survives a within-place shuffle "
                                  "at 62%, so it is an artefact and is not a "
                                  "measurement of beta. It is the only reading "
                                  "anywhere in results_p44 that sits above the "
                                  "crossing, so it is named rather than "
                                  "quietly dropped."),
        sentence="at beta = 0.95, above every beta we measure, the bound is "
                 f"{b095:.4f} and the survey's {R_SURVEY:.4f} is excluded. No "
                 "interpolation is used and no grid step is counted.")

    # ------------------------------------------------------------------- phi
    out["phi_refined"] = [dict(beta=b, bound_corrected=inv[str(b)]["bound_corrected"],
                               phi=inv[str(b)]["phi"],
                               analytic=inv[str(b)]["analytic"],
                               ratio_measured_over_analytic=(
                                   inv[str(b)]["bound_corrected"]
                                   / inv[str(b)]["analytic"]))
                          for b in betas_r]
    say("\n=== 59.11 phi on the refined grid ===")
    say(f"  {'beta':>5} {'bound':>10} {'analytic':>10} {'phi':>8} {'meas/anal':>10}")
    for row in out["phi_refined"]:
        say(f"  {row['beta']:>5.2f} {row['bound_corrected']:>10.5f} "
            f"{row['analytic']:>10.5f} {row['phi']:>8.4f} "
            f"{row['ratio_measured_over_analytic']:>10.4f}")

    out["config"] = dict(ym=P39.YM, panel=P39.PANEL, reps=args.reps,
                         seed=P39.SEED, betas=list(BETAS_STAR),
                         r_true_grid=rtrue, q_lo=P39.Q_LO, n_boot=BOOT,
                         null_reps=NULL_REPS, imp_true=P39.IMP_MEASURED,
                         imp_estimator=P39.IMP_PUB, full=bool(full))
    # No timing goes into the results file. determinism_check.sh compares the
    # whole file by sha256, so a wall-clock field makes a pair of runs report
    # DIFFERS on the one key that is supposed to differ; every other phase the
    # script checks writes no timing for exactly this reason. The elapsed time
    # is printed instead. (results_p39.json does carry runtime_seconds and
    # coarse_shape.seconds, which is why `determinism_check.sh p39_recovery`
    # cannot pass as written -- reported, not fixed here: p39 is frozen.)
    out["config"]["timing"] = "printed, not stored; see the comment at the write"
    name = "results_p59.json" if full else "results_p59.partial.json"
    with open(f"{RESULTS}/{name}", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    say(f"\nwrote {RESULTS}/{name}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
