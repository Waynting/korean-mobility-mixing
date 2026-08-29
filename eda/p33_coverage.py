#!/usr/bin/env python
"""Phase 33 — what the age-varying coverage profile does to the four indicators.

THE INSTRUCTION AND WHY IT CANNOT BE FOLLOWED LITERALLY. The 8-19 letter says the
coverage profile must be pushed through the matrix rather than reported beside it:
0-9 is covered about 28x and 45-49 about 3.2x, that is an age-varying bias, it
changes the row weights and therefore assortativity and the NGM eigenvalue, and
masking is zero above 65 while coverage bias is in every cell.

Every clause of that is right except the direction of the fix. The product's
`pop` column is ALREADY the vendor-expanded population -- `pop = w_a x devices`
is exactly how p9 recovers w_a from the masking threshold in the first place.
Multiplying by w_a again applies the expansion twice: the 0-9 row moves by 28x in
LEVEL while assortativity, being scale-free in a way that hides it, still looks
plausible. That is the failure mode the same letter warns about in its last
paragraph, and it would have been produced by following the letter's own step.

SO THE PROFILE IS PUSHED THROUGH AS A FAMILY, NOT AS A POINT. Three readings,
all of them computed here, none of them requiring a ruling before the work can
be done:

  R1  w_a as an EFFECTIVE SAMPLE SIZE. n_a / w_a is the device count behind the
      published number, so it fixes the noise, not the level. A parametric
      bootstrap in device space puts a confidence interval on all four
      indicators. This reading is unambiguously correct and it is the one that
      belongs in the precision sentence.

  R2  w_a as a FAMILY OF RESIDUAL ERROR AMPLIFICATIONS, rho_a(theta) =
      (w_a/wbar)^theta for theta in [-1, +1]. theta = 0 is the published matrix
      (the vendor's expansion is right); theta = -1 is device space (the age
      structure is entirely vendor-manufactured, the honest lower end);
      theta = +1 is the letter's literal reading (the vendor corrected nothing).
      "The conclusion does not change" then becomes "sign and ordering are
      invariant over the whole family", which is a far stronger statement than
      re-weighting at one point.

  R3  w_a as a WITHIN-BAND SELECTION BOUND. The 1 in 28 children who are observed
      are not a random 1 in 28; they are the older ones. Moving a share phi of
      0-9 arrivals into 10-14 and re-measuring gives a bound that no amount of
      re-weighting can produce, because it is a composition error inside a band
      rather than a level error between bands.

RE-WEIGHTING IS NOT SEPARABLE, and this is why the work has to go back to cell
level rather than to the stored A. With A[a,a'] = sum_l n_a(l) n_a'(l) / n_.(l),
scaling row a scales the DENOMINATOR of every cell too, so A' != D A D for any
diagonal D. The 424-dong x 24-hour x 7-dow cell table is therefore rebuilt here
and the weights are applied before the ratio is formed.

ONE FACT RUNS IN OUR FAVOUR and it is worth stating in the paper rather than in a
footnote: p9's coverage profile for the five bands from 20 to 44 is censored from
below at 3.00 (the masking threshold cannot reveal a weight below the value it
suppresses), so the true coverage gradient is STEEPER than the one used here.
A distortion computed from a profile we know to be flattened is a conservative
underestimate of the distortion.

THE ANCHOR. At theta = 0, with the same panel, day set and imputation, this
script's cell-level rebuild must reproduce the assortativity and lead eigenvalue
already published in results_p26.json. It asserts that rather than trusting it:
the whole point of a re-weighting exercise is that a wrong rebuild would move
every number by a plausible-looking amount.

    python eda/p33_coverage.py [--months 202312,202402] [--boot 400]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calendar_kr as K
from common import AGE_LABEL, AGES
from paths import DERIVED, FIG, ROOT, require

NA = len(AGES)
DEFAULT_MONTHS = [202312, 202402]
THETAS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
PHIS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
SEED = 20260819
IMP = 1.5                       # p26's published imputation; p30's measured
IMP_MEASURED = 2.27             # fill is carried as a sensitivity, not a switch

out = {}


# ------------------------------------------------------------ the statistics
def pm(T):
    s = T.sum()
    e = T / s if s else T
    r = e.sum(1)
    return e, r, np.outer(r, r)


def four(T):
    """The four indicators the letter names, on one matrix.

    Identical formulas to p32.excess_stats, restated rather than imported so the
    two scripts can disagree and be caught by the audit gate instead of agreeing
    because they share a bug.
    """
    e, r, E = pm(T)
    d = e - E
    m = e > 0
    mi = float((e[m] * np.log2(e[m] / E[m])).sum())
    h = -float((r[r > 0] * np.log2(r[r > 0])).sum())
    chi2_n = float(np.nansum(np.where(E > 0, d * d / E, 0.0)))
    k = min(int((r > 0).sum()), int((e.sum(0) > 0).sum()))
    den = 1 - float((r * r).sum())
    return dict(assortativity=float((np.trace(e) - (r * r).sum()) / den) if den else np.nan,
                half_l1=float(np.abs(d).sum() / 2),
                cramers_v=float(np.sqrt(chi2_n / (k - 1))) if k > 1 else np.nan,
                nmi=float(mi / h) if h else np.nan,
                mi_bits=mi)


def ngm(A, pop):
    """Dominant eigenvalue and eigenvector of C = A / N, the next-generation
    functional the letter names beside assortativity."""
    C = A / pop[:, None]
    w, v = np.linalg.eig(C)
    k = int(np.argmax(w.real))
    vec = np.abs(v[:, k].real)
    s = vec.sum()
    return float(w[k].real), (vec / s if s else vec)


def kendall_tau(x, y):
    """Rank agreement without pulling scipy into requirements.txt for one line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            c += s > 0
            d += s < 0
    return float((c - d) / (0.5 * n * (n - 1))) if n > 1 else np.nan


# ------------------------------------------------------- the cell-level rebuild
def cell_matrix(X, tot_scale, rho=None, shift=None):
    """A = sum_l n_a n_a' / n_.  with the weights applied BEFORE the ratio.

    `rho` multiplies the age columns; `shift` is a list of (from, to, phi) moves
    of a share of one band's arrivals into another. Both change the row sums AND
    the per-cell denominator, which is precisely why neither can be applied to a
    stored A afterwards.
    """
    Y = X
    if shift:
        Y = Y.copy()
        for i, j, phi in shift:
            moved = phi * Y[:, i]
            Y[:, i] -= moved
            Y[:, j] += moved
    if rho is not None:
        Y = Y * rho[None, :]
    tot = Y.sum(1)
    keep = tot > 0
    Y, tot = Y[keep], tot[keep]
    return (Y / tot[:, None]).T @ Y / tot_scale


def build_cells(ym, panel="WE", level="dong", imp=IMP, dows=None):
    """The (place, hour, dow) x age table in per-day units, exactly as p26 forms
    it, but returned before the ratio so the weights can get inside."""
    path = DERIVED / f"arrivals_dow_{ym}.parquet"
    require(path, f"p26 arrival cache for {ym} (run eda/p26_matrix.py)")
    df = pd.read_parquet(path)
    dows = list(range(1, 8)) if dows is None else list(dows)
    nd = {d: K.cell_exposure(ym, d)["n_days"] for d in dows}
    d = df if panel == "WE" else df[df.dest_attr == panel]
    d = d[d.dest_attr != "H"]
    d = d[d.dow_n.isin(dows)]
    d = d.assign(n=(d.v_obs + imp * d.n_masked) / d.dow_n.map(nd))
    loc = {"dong": d.d_dong, "gu": d.d_dong // 1000,
           "city": pd.Series(0, index=d.index)}[level]
    g = (d.assign(loc=loc).groupby(["loc", "arr_hour", "dow_n", "age"],
                                   as_index=False)["n"].sum())
    wide = g.pivot_table(index=["loc", "arr_hour", "dow_n"], columns="age",
                         values="n", fill_value=0.0).reindex(columns=AGES,
                                                             fill_value=0.0)
    return wide.to_numpy(float), float(len(dows))


def holiday_free_dows(ym):
    return [d for d in range(1, 8) if K.cell_exposure(ym, d)["n_holiday"] == 0]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    yms = ([int(x) for x in args.months.split(",")] if args.months
           else DEFAULT_MONTHS)
    rng = np.random.default_rng(SEED)

    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p32 = json.load(open(f"{ROOT}/eda/results_p32.json"))
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}
    censored = {c["band"]: bool(c["censored_band"]) for c in p9["coverage_profile"]}
    w = np.array([cov[AGE_LABEL[a]] for a in AGES], float)
    sq = [i for i, a in enumerate(AGES) if a < 80]

    print("=== 33.0 the profile being pushed through ===")
    print(f"  {'band':>6} {'w_a':>7} {'w_a/wbar':>9}  censored from below at 3.00")
    wbar = float(np.exp(np.log(w).mean()))
    for i, a in enumerate(AGES):
        print(f"  {AGE_LABEL[a]:>6} {w[i]:>7.2f} {w[i] / wbar:>9.3f}"
              f"  {'yes' if censored[AGE_LABEL[a]] else ''}")
    print(f"  geometric mean wbar = {wbar:.3f}; "
          f"{sum(censored.values())} of {NA} bands censored, so the true "
          f"gradient is STEEPER than this one.")
    out["profile"] = dict(bands=[AGE_LABEL[a] for a in AGES], w=w.tolist(),
                          wbar=wbar, censored=[censored[AGE_LABEL[a]] for a in AGES])

    # ------------------------------------------------------------- the anchor
    print("\n=== 33.1 anchor: theta = 0 must reproduce results_p26.json ===")
    cells, anchors = {}, []
    for ym in yms:
        hf = holiday_free_dows(ym)
        for tag, dows in (("holidayfree", hf), ("alldays", None)):
            X, ns = build_cells(ym, dows=dows)
            cells[(ym, tag)] = (X, ns)
            A = cell_matrix(X, ns)
            key = f"{ym}|WE|dong" + ("|holidayfree" if tag == "holidayfree" else "")
            pub = p26["matrices"].get(key)
            if pub is None:
                print(f"  {key}: not in results_p26.json, skipped")
                continue
            r_ = four(A)["assortativity"]
            lam, _ = ngm(A, pops[ym])
            dr = abs(r_ - pub["assortativity"]) / abs(pub["assortativity"])
            dl = abs(lam - pub["lead_eigenvalue"]) / abs(pub["lead_eigenvalue"])
            print(f"  {key:>34}  r {r_:.8f} vs {pub['assortativity']:.8f} "
                  f"(rel {dr:.2e});  lambda rel {dl:.2e}")
            anchors.append(dict(key=key, rel_r=dr, rel_lambda=dl))
            assert dr < 1e-9 and dl < 1e-9, (
                f"cell-level rebuild does not reproduce p26 for {key}; "
                f"every number below would be wrong by a plausible amount")
    out["anchor"] = anchors
    print(f"  {len(anchors)} anchors, max relative deviation "
          f"{max(max(a['rel_r'], a['rel_lambda']) for a in anchors):.2e}")

    # ------------------------------------------ R1 effective sample size / CIs
    # Devices, not arrivals, are what was sampled. n_a = w_a * d_a, so a Poisson
    # count of devices carries variance w_a^2 * d_a = w_a * n_a -- the published
    # number is over-dispersed relative to its own magnitude by exactly w_a, and
    # the 0-9 row is over-dispersed 28-fold. That is a precision statement and it
    # is the only reading of the profile that needs no ruling.
    # A percentile interval would be wrong here and wrong in an instructive
    # direction. All four indicators are non-negative quadratic-ish functionals
    # of the cell counts, so INDEPENDENT NOISE MANUFACTURES EXCESS: the
    # resampled matrices sit systematically above the observed one. The gap
    # between the resample mean and the point estimate is therefore an estimate
    # of the bias already sitting in the point estimate, and the basic (rather
    # than percentile) bootstrap interval is the one that removes it. The
    # bias-corrected value is a result in its own right -- it says how much of
    # the measured departure from proportionate mixing is device sampling noise.
    print(f"\n=== 33.2 R1: parametric bootstrap in device space ({args.boot}) ===")
    r1 = {}
    KEYS = ("assortativity", "half_l1", "cramers_v", "nmi")
    for ym in yms:
        X, ns = cells[(ym, "holidayfree")]
        A0 = cell_matrix(X, ns)
        base = four(A0)
        lam0, _ = ngm(A0, pops[ym])
        base["lead_eigenvalue"] = lam0
        draws = {k: [] for k in KEYS + ("lead_eigenvalue",)}
        # X already carries the deterministic masked-cell imputation, so those
        # few cells get Poisson noise they did not earn. Masking is confined to
        # 20-44 and is worth ~11% of volume, and the effect is to overstate the
        # noise slightly -- i.e. to overstate the correction and understate the
        # excess, which is the conservative direction for the claim.
        D = X / w[None, :]                       # device-equivalent cell counts
        for _ in range(args.boot):
            Xb = rng.poisson(np.maximum(D, 0)) * w[None, :]
            Ab = cell_matrix(Xb, ns)
            st = four(Ab)
            for k in KEYS:
                draws[k].append(st[k])
            draws["lead_eigenvalue"].append(ngm(Ab, pops[ym])[0])
        rep = {}
        for k, v in draws.items():
            v = np.asarray(v, float)
            bias = float(v.mean() - base[k])
            rep[k] = dict(
                point=float(base[k]), boot_mean=float(v.mean()),
                noise_bias=bias, bias_corrected=float(base[k] - bias),
                # basic bootstrap: [2t - q975, 2t - q025]
                ci=[float(2 * base[k] - np.percentile(v, 97.5)),
                    float(2 * base[k] - np.percentile(v, 2.5))],
                boot_sd=float(v.std(ddof=1)))
        r1[str(ym)] = dict(stats=rep, n_boot=args.boot,
                           n_device_equiv=float((pops[ym][sq] / w[sq]).sum()))
        print(f"  {ym}  n_device_equiv {r1[str(ym)]['n_device_equiv']:,.0f}")
        print(f"    {'':>14} {'published':>10} {'noise bias':>11} "
              f"{'corrected':>10} {'95% CI':>22} {'noise share':>12}")
        for k in KEYS + ("lead_eigenvalue",):
            d_ = rep[k]
            share = d_["noise_bias"] / d_["point"] if d_["point"] else np.nan
            print(f"    {k:>14} {d_['point']:>10.5f} {d_['noise_bias']:>+11.5f} "
                  f"{d_['bias_corrected']:>10.5f} "
                  f"[{d_['ci'][0]:>9.5f},{d_['ci'][1]:>9.5f}] {share:>11.1%}")
        print("    'noise share' is how much of the published departure from "
              "proportionate mixing is device sampling noise. Devices recur "
              "across cells, so the real noise is larger and this is a floor.")
    out["R1_effective_sample"] = r1

    # ------------------------------------------------------- R2 the theta family
    print("\n=== 33.3 R2: rho_a(theta) = (w_a/wbar)^theta, theta in [-1, +1] ===")
    r2 = {}
    for ym in yms:
        X, ns = cells[(ym, "holidayfree")]
        rows, base_vec = [], None
        for th in THETAS:
            rho = (w / wbar) ** th
            A = cell_matrix(X, ns, rho=rho)
            st = four(A)
            lam, vec = ngm(A, pops[ym])
            if th == 0.0:
                base_vec = vec
            rows.append(dict(theta=th, **{k: st[k] for k in
                             ("assortativity", "half_l1", "cramers_v", "nmi")},
                             lead_eigenvalue=lam,
                             top_band=AGE_LABEL[AGES[int(np.argmax(vec))]],
                             eigvec=vec.tolist()))
        for r_ in rows:
            r_["kendall_tau_vs_theta0"] = kendall_tau(r_["eigvec"], base_vec)
        r2[str(ym)] = rows
        print(f"\n  {ym}   {'theta':>6} {'assort':>9} {'half-L1':>9} "
              f"{'CramerV':>9} {'NMI':>9} {'lambda':>9} {'tau':>6} {'top band':>9}")
        for r_ in rows:
            mark = "  <- published" if r_["theta"] == 0 else ""
            print(f"        {r_['theta']:>6.2f} {r_['assortativity']:>9.5f} "
                  f"{r_['half_l1']:>9.5f} {r_['cramers_v']:>9.5f} "
                  f"{r_['nmi']:>9.5f} {r_['lead_eigenvalue']:>9.4f} "
                  f"{r_['kendall_tau_vs_theta0']:>6.3f} {r_['top_band']:>9}{mark}")
        sgn = {np.sign(r_["assortativity"]) for r_ in rows}
        print(f"    sign of assortativity over the family: "
              f"{'invariant (+)' if sgn == {1.0} else f'NOT invariant: {sgn}'}")
    out["R2_theta_family"] = r2

    # ----------------------------------------- the gap against the survey side
    # The letter's test is not whether the indicators move but whether the
    # conclusion moves, and the conclusion is a ratio against the survey. So the
    # family is carried all the way to the ratio rather than stopping at the
    # passive matrix.
    print("\n=== 33.4 the four gaps, over the whole theta family ===")
    gaps = {}
    for ym in yms:
        sk = f"survey|{ym}|WE|seoul"
        if sk not in p32["excess"]:
            continue
        surv = p32["excess"][sk]
        rows = []
        for r_ in r2[str(ym)]:
            g = {k: (surv[k] / r_[k] if r_[k] else np.nan)
                 for k in ("assortativity", "half_l1", "cramers_v", "nmi")}
            rows.append(dict(theta=r_["theta"], **g))
        gaps[str(ym)] = rows
        print(f"\n  {ym}   {'theta':>6} {'assort x':>9} {'half-L1 x':>10} "
              f"{'CramerV x':>10} {'NMI x':>8}")
        for g in rows:
            print(f"        {g['theta']:>6.2f} {g['assortativity']:>9.1f} "
                  f"{g['half_l1']:>10.1f} {g['cramers_v']:>10.1f} {g['nmi']:>8.1f}")
        lo = min(min(g[k] for k in ("assortativity", "half_l1", "cramers_v", "nmi"))
                 for g in rows)
        print(f"    smallest gap anywhere in the family: {lo:.1f}x "
              f"({'still >1, conclusion holds' if lo > 1 else 'CONCLUSION BREAKS'})")
    out["R2_gaps_vs_survey"] = gaps

    # ------------------------------------------- R3 within-band selection bound
    # The bound the profile cannot give. Being observed at 1-in-28 is not the
    # same as being sampled at 1-in-28: the observed children are the older ones,
    # and no row multiplier can express that, because it is a composition error
    # inside a band rather than a level error between bands.
    print("\n=== 33.5 R3: within-band selection, 0-9 arrivals re-labelled 10-14 ===")
    i09, i1014 = AGES.index(0), AGES.index(10)
    r3 = {}
    for ym in yms:
        X, ns = cells[(ym, "holidayfree")]
        rows = []
        for phi in PHIS:
            A = cell_matrix(X, ns, shift=[(i09, i1014, phi)])
            st = four(A)
            lam, _ = ngm(A, pops[ym])
            rows.append(dict(phi=phi, **{k: st[k] for k in
                             ("assortativity", "half_l1", "cramers_v", "nmi")},
                             lead_eigenvalue=lam))
        # the generalised version: every band leaks upward in proportion to how
        # far its coverage weight sits above the best-covered band
        lift = np.log(w / w.min()) / np.log(w.max() / w.min())
        gen = []
        for phi in PHIS:
            sh = [(i, i + 1, float(phi * lift[i])) for i in range(NA - 1)]
            A = cell_matrix(X, ns, shift=sh)
            st = four(A)
            lam, _ = ngm(A, pops[ym])
            gen.append(dict(phi=phi, **{k: st[k] for k in
                            ("assortativity", "half_l1", "cramers_v", "nmi")},
                            lead_eigenvalue=lam))
        r3[str(ym)] = dict(zero_to_nine=rows, all_bands=gen, lift=lift.tolist())
        print(f"\n  {ym}   {'phi':>5} {'assort':>9} {'half-L1':>9} {'CramerV':>9} "
              f"{'NMI':>9} {'lambda':>9}   | generalised: assort / NMI")
        for a_, b_ in zip(rows, gen):
            print(f"        {a_['phi']:>5.2f} {a_['assortativity']:>9.5f} "
                  f"{a_['half_l1']:>9.5f} {a_['cramers_v']:>9.5f} "
                  f"{a_['nmi']:>9.5f} {a_['lead_eigenvalue']:>9.4f}   | "
                  f"{b_['assortativity']:>9.5f} {b_['nmi']:>9.5f}")
    out["R3_within_band"] = r3

    # ------------------------------------------------------ masking sensitivity
    # Masking is zero above 65 and the coverage profile is not, which is the
    # letter's reason for ranking coverage above masking. Both are moved here so
    # the ranking is measured rather than asserted.
    print("\n=== 33.6 coverage against masking, on the same scale ===")
    comp = {}
    for ym in yms:
        X, ns = cells[(ym, "holidayfree")]
        base = four(cell_matrix(X, ns))["assortativity"]
        Xm, nm = build_cells(ym, imp=IMP_MEASURED, dows=holiday_free_dows(ym))
        meas = four(cell_matrix(Xm, nm))["assortativity"]
        Xl, nl = build_cells(ym, imp=0.0, dows=holiday_free_dows(ym))
        Xh, nh = build_cells(ym, imp=3.0, dows=holiday_free_dows(ym))
        # imp=0 and imp=3 are the two ENDS of the masking band, not its low and
        # high value: dropping masked volume RAISES assortativity here, because
        # the masked cells are the small mixed ones. Ordering them by value
        # rather than by imputation keeps the width positive and the comparison
        # against the coverage family meaningful.
        ends = sorted((four(cell_matrix(Xl, nl))["assortativity"],
                       four(cell_matrix(Xh, nh))["assortativity"]))
        th = sorted(r_["assortativity"] for r_ in r2[str(ym)])
        mask_w, cov_w = ends[1] - ends[0], th[-1] - th[0]
        comp[str(ym)] = dict(base=base, mask_band=ends, mask_width=mask_w,
                             mask_measured=meas, mask_imp0=ends[1] if
                             four(cell_matrix(Xl, nl))["assortativity"] == ends[1]
                             else ends[0],
                             theta_span=[th[0], th[-1]], theta_width=cov_w,
                             ratio=cov_w / mask_w if mask_w else np.nan)
        print(f"  {ym}  masking band [{ends[0]:.5f}, {ends[1]:.5f}] "
              f"(width {mask_w:.5f}, measured fill {meas:.5f}); "
              f"coverage family [{th[0]:.5f}, {th[-1]:.5f}] "
              f"(width {cov_w:.5f}) "
              f"-> coverage moves the indicator {cov_w / mask_w:.1f}x further "
              f"than masking does")
    out["mask_vs_coverage"] = comp

    # ------------------------------------------------------------------ figure
    ym0 = yms[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.bar(range(NA), w, color=["#A8434E" if censored[AGE_LABEL[a]] else "#0E7C86"
                                for a in AGES])
    ax.axhline(3.0, ls=":", c="k", lw=1)
    ax.set_xticks(range(NA))
    ax.set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=90, fontsize=6)
    ax.set_ylabel("coverage weight $w_a$")
    ax.set_title("the profile (red = censored at 3.00,\nso the true gradient is steeper)",
                 fontsize=9)
    ax = axes[1]
    for k, c in (("assortativity", "#0E7C86"), ("nmi", "#A8434E"),
                 ("half_l1", "#BC8034"), ("cramers_v", "#4C6E8A")):
        v = [r_[k] for r_ in r2[str(ym0)]]
        ax.plot(THETAS, np.array(v) / v[THETAS.index(0.0)], "o-", c=c, label=k)
    ax.axvline(0, ls=":", c="k", lw=1)
    ax.set_xlabel(r"$\theta$  (device space $\leftarrow$  published  "
                  r"$\rightarrow$ literal re-weighting)")
    ax.set_ylabel("indicator, relative to published")
    ax.legend(fontsize=7)
    ax.set_title(f"{ym0}: the whole family, not one point", fontsize=9)
    ax = axes[2]
    v = [r_["assortativity"] for r_ in r3[str(ym0)]["zero_to_nine"]]
    g = [r_["assortativity"] for r_ in r3[str(ym0)]["all_bands"]]
    ax.plot(PHIS, v, "o-", c="#0E7C86", label="0-9 leaks into 10-14")
    ax.plot(PHIS, g, "s--", c="#A8434E", label="every band leaks upward")
    ax.set_xlabel(r"$\varphi$, share re-labelled")
    ax.set_ylabel("assortativity")
    ax.legend(fontsize=7)
    ax.set_title("within-band selection: the bound\nre-weighting cannot give", fontsize=9)
    fig.suptitle("Phase 33 — the coverage profile pushed through the matrix, "
                 "three readings", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p33_coverage.png", dpi=150)
    plt.close(fig)
    print(f"\n  -> fig/p33_coverage.png")

    with open(f"{ROOT}/eda/results_p33.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p33.json")


if __name__ == "__main__":
    main()
