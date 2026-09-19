#!/usr/bin/env python
"""Phase 69 — beta measured on the synthetic world the way it is measured on the
survey: the venue-level truth under the SURVEY'S pairing convention, and the
merge ladder that runs from one venue to the whole cell.

WHY THIS EXISTS. The advisor's 2026-09-05 letter, point 2.4: "you already have
the full venue structure at every synthetic grid point, so run the same beta
estimation procedure the survey uses -- the same venue-merge decay -- and print
the realised beta at every grid point." p66 did not do that. It re-indexed the
curve with two definitions built out of numbers p39 had already stored:

  A  1 - r_between_med / r_true_med   the world's own share below the cell line
  B  1 - med_corr      / r_true_med   the form p44_beta.py:588 uses on the survey

Both divide by `r_true_med`, and `r_true_med` is p39's venue-level truth summed
as sum_v v v' / n_v -- WITH replacement. The survey's r_true is not that. p44's
primary is `collapse(U, wg, replacement=False)`, sum_v (v v' - diag v)/(n_v - 1),
and 44.1 is a whole section arguing which of the two the survey should use. So A
and B answer "what share of this world sits below the cell line", and the survey
answers "what does the pairing rule I actually run read at venue level" -- and
the paper compares the second to a grid indexed by the first.

  C  1 - med_corr / r_venue_wor       this phase's primary

C is B with one thing changed: the denominator is the venue-level truth under
the survey's own pairing convention. That is the whole of 2.4 as a number, and
the size of the change is the answer to whether the letter's worry was material.

THE NUMERATOR STAYS WITH REPLACEMENT, and that is not an oversight. On the
survey side the numerator is the PASSIVE reading, which p26's published
estimator computes as n_a n_a' / n_. -- with replacement -- while the
denominator is the survey's own venue-level r_true without it. The survey's beta
therefore mixes the two conventions, and a like-for-like synthetic beta has to
mix them the same way. 69.2 reports the all-with-replacement and
all-without-replacement variants beside it so the mixing is visible rather than
hidden.

WHAT IS RE-RUN AND WHAT IS NOT. No simulation is re-designed and no published
file is rewritten. Every grid point's field comes from the `knob` p39/p59 already
stored, every replicate is drawn from p39's own seed tag, and the readings come
back through `p39_recovery.World.draw`. What is new is computed alongside:

  * the venue-level matrix in the without-replacement convention, in closed
    form, by the same trick p39 uses for the with-replacement one (a sum over
    (cell, venue type) pairs collapses to three 16 x 16 matmuls -- see
    `venue_matrices`), and
  * the merge ladder, run by MATERIALISING venues and calling p44's own
    `collapse`, so the ladder is the survey's function and not a restatement of
    it (69.1).

THE DIAGONAL CORRECTION MEETS FRACTIONAL PEOPLE, and this is the one place where
"the same procedure" cannot be literally the same. The survey's U are integer
headcounts, so v_a^2 - v_a >= 0 always. A synthetic venue carries fractional
arrivals -- a band with 0.4 of a person in it contributes v_a^2 - v_a = -0.24 --
so the correction can push a diagonal entry negative where the survey's never
could. The formula is applied as written (that is what "the same convention"
means) and 69.1c reports the size of the effect by also running an INTEGERISED
arm, where every venue's count vector is rounded to whole people before the
convention is applied. If the two disagree, the disagreement belongs in SI 7 and
not in a comment here.

ANCHORS, all of them against files this phase does not write.

  A1  every grid point's replicates reproduce the stored `r_true_med` and
      `r_between_med` bit for bit. The seeds and the knob are p39's, so anything
      else means the world moved under us and nothing below is comparable.
  A2  the closed-form WITH-replacement venue matrix equals `draw`'s own A_true,
      which p39 builds by a different algebra, to 1e-12 -- which is what licenses
      the without-replacement one built by the same route.
  A3  on the ladder's declared cell subsample, the explicit materialisation fed
      to p44's `collapse` reproduces both closed forms, and the within-cell arm
      at full merge reproduces the cell reading.
  A4  p66's conversion table is reproduced from the same stored numbers, so the
      A and C axes this phase compares are the published A and this phase's C.

    python eda/p69_betacollapse.py                 # the full union grid, ~5 min
    python eda/p69_betacollapse.py --smoke         # wiring only, writes .partial
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p39_recovery as P39
from p39_recovery import World, newman, AGES, NA, AGE_LABEL
from p44_beta import collapse
from paths import ROOT

# ------------------------------------------------------- declared before the run
LADDER_SEED = 20260907          # the ladder's cell subsample and permutations
LADDER_CELLS = 4000             # cells materialised for the explicit ladder
LADDER_REPS = 5                 # replicates per ladder point
LADDER_G = (1, 2, 4, 8, 16, 32, 64)   # venues per merged group; "cell" is added
ARM_DATA = 3                    # p39's seed tag for the data arm
DESIGN_GRID = (0.0, 0.5, 0.8, 0.95, 0.96, 0.97, 0.98, 0.99)

DECLARATION = {
    "primary": "C = 1 - med_corr / r_venue_wor, evaluated at the r_true grid "
               "point nearest that beta's own bound -- the same point p66 "
               "evaluates B at, so C and B differ in the denominator's pairing "
               "convention and in nothing else",
    "secondary": "C_noiseless = 1 - r_between_med / r_venue_wor (comparable to "
                 "p66's A), and the two single-convention variants, reported "
                 "beside the primary and not instead of it",
    "conventions": "numerator with replacement (p26's published passive "
                   "estimator), denominator without (p44's survey primary). "
                   "The survey's beta mixes them in exactly this way.",
    "grid": "the union of results_p39's and results_p59's surfaces, which is "
            "the eight design points p66's conversion table is built on",
    "reads_not_writes": "knobs, deltas and replicate counts come from "
                        "results_p39/p59; bounds come from their inversions; "
                        "nothing in either file is recomputed or rewritten",
    "ladder": f"{LADDER_CELLS} cells drawn once from seed {LADDER_SEED}, "
              f"{LADDER_REPS} replicates, groups of {LADDER_G} venues plus the "
              f"whole cell, both arms of p44 44.3 (random pooling and "
              f"within-cell pooling), fed to p44's own collapse()",
    "stop_loss": "abort unless every grid point reproduces p39/p59's stored "
                 "r_true_med and r_between_med, and unless the closed-form "
                 "with-replacement venue matrix reproduces draw()'s A_true",
    "prediction": "C sits BELOW B. The without-replacement correction removes "
                  "diagonal mass, so r_venue_wor < r_venue_wr, so the ratio "
                  "rises and beta falls. Recorded before the run; if C comes "
                  "out above B the reasoning is wrong and that is the finding.",
}

out = {"declaration": DECLARATION}


def say(s=""):
    print(s, flush=True)


def load(tag):
    with open(f"{ROOT}/eda/results_{tag}.json") as fh:
        return json.load(fh)


# ------------------------------------------------------------ the venue algebra
def venue_matrices(W, P, phi, delta, f=None, cells=None):
    """(A_wr, A_wor, f): the venue-level matrix in both pairing conventions.

    A venue of type k in cell l has count vector v = s_l * ((1-d) p_l + d e_k)*f
    (p39_recovery.py:509-512, materialised there one venue at a time). Writing
    u_l = s_l (1-d) p_l * f and n_lk = v.sum(), every sum over venues of a
    quadratic in v collapses to matmuls over the (cell, type) grid:

        sum_lk c_lk/D_lk v v'  =  U' diag(alpha) U
                                  + d [ (U diag(s))' M + transpose ]
                                  + d^2 diag( sum_l s_l^2 a_lk f_k^2 )

    with a_lk = c_lk / D_lk, alpha_l = sum_k a_lk and M_lk = a_lk f_k. D_lk is
    n_lk with replacement and n_lk - 1 without, and the without-replacement arm
    additionally subtracts diag(sum_lk a_lk v).

    p39 computes the WITH-replacement case by a different arrangement of the
    same algebra (draw()'s Pt/gm/H block). 69.0's A2 compares the two.
    """
    if f is None:
        # The rescaling is a property of the WHOLE world -- it is what puts the
        # measured age margin back on the cell table -- so a caller restricting
        # to a subsample of cells passes the world's f in rather than letting a
        # subsample compute its own.
        Xc = W.S[:, None] * ((1 - delta) * P + delta * phi)
        f = W.N / Xc.sum(0)
    idx = slice(None) if cells is None else cells
    S_l, m_l, P_l, phi_l = W.S[idx], W.m[idx], P[idx], phi[idx]
    c = m_l[:, None] * phi_l                     # venues of type k in cell l
    s_v = S_l / m_l                              # venue size
    U = (1 - delta) * s_v[:, None] * (P_l * f[None, :])         # u_l, per cell
    n = U.sum(1)[:, None] + delta * s_v[:, None] * f[None, :]

    def build(D, subtract_diag):
        a = np.divide(c, D, out=np.zeros_like(c), where=D > 0)
        alpha = a.sum(1)
        M = a * f[None, :]
        s = s_v
        A = (U * alpha[:, None]).T @ U
        T2 = (U * s[:, None]).T @ M
        A = A + delta * (T2 + T2.T)
        A[np.diag_indices(NA)] += (delta ** 2) * (
            (s ** 2)[:, None] * a * (f ** 2)[None, :]).sum(0)
        if subtract_diag:
            D1 = (U * alpha[:, None]).sum(0) + delta * (s[:, None] * M).sum(0)
            A = A - np.diag(D1)
        return A

    return build(n, False), build(n - 1.0, True), f


def materialise(W, P, phi, delta, f, cells):
    """Every venue of `cells` as an explicit count vector, p39 39.3b's loop.

    Returns (V, cell_index): V[i] is one venue's 16-vector of arrivals and
    cell_index[i] is the cell it belongs to, so a caller can pool within cells
    or across them. This is the object the survey's `collapse` was written for.
    """
    rows, owner = [], []
    for l in cells:
        counts = np.rint(phi[l] * W.m[l]).astype(int)
        s_l = W.S[l] / W.m[l]
        for k in range(NA):
            if counts[k] == 0:
                continue
            q = (1 - delta) * P[l] + delta * np.eye(NA)[k]
            rows.append(np.repeat((s_l * q * f)[None, :], counts[k], axis=0))
            owner.append(np.full(counts[k], l))
    if not rows:
        return np.zeros((0, NA)), np.zeros(0, int)
    return np.vstack(rows), np.concatenate(owner)


def pooled_r(V, gid, W, replacement):
    """p44's collapse over groups defined by `gid`, read by p39's newman."""
    ng = gid.max() + 1 if len(gid) else 0
    U = np.zeros((ng, NA))
    np.add.at(U, gid, V)
    return newman(collapse(U, np.ones(ng), replacement=replacement), W.pop)


# --------------------------------------------------------------------- the grid
def surface_points(p39, p59):
    """Every stored grid point, keyed by design beta. p66's own reader."""
    pts = {}
    for src in (p39["surface"], p59["surface_refined"]):
        for k, v in src.items():
            if not k.startswith("data|") or v.get("r_true_med", 0) <= 0:
                continue
            pts.setdefault(round(v["beta"], 6), {})[round(v["r_true_target"], 9)] = v
    return pts


def bounds(p39, p59):
    b = {round(float(k), 6): v["bound_corrected"]
         for k, v in p39["inversion"].items()}
    b.update({round(float(k), 6): v["bound_corrected"]
              for k, v in p59["inversion_refined"].items()})
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="two grid points, two replicates, no ladder; writes "
                         "results_p69.partial.json")
    args = ap.parse_args()
    t0 = time.time()

    p39, p59, p66 = load("p39"), load("p59"), load("p66")
    p26, p9, p33 = load("p26"), load("p9"), load("p33")
    r_survey = p39["published_constants"]["survey"]
    pts, bnd = surface_points(p39, p59), bounds(p39, p59)

    say("=== 69.0 the declaration, as written before the run ===")
    for k, v in DECLARATION.items():
        say(f"  {k:<18} {v}")

    # ------------------------------------------------------------- the world
    say("\n=== 69.1 the world, built by p39's own World ===")
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}
    w = np.array([cov[AGE_LABEL[a]] for a in AGES], float)
    assert np.abs(w - np.array(p33["profile"]["w"])).max() == 0.0, \
        "the coverage profile differs from the one p33 pushed through the matrix"
    W = World(P39.YM, w, pops[P39.YM])
    W.build_base_field()

    # ------------------------------------------------- A4: p66's own conversion
    say("\n=== 69.2 anchor A4: p66's conversion, rebuilt from the same points ===")
    conv66 = {round(r["design"], 6): r for r in p66["conversion"]}
    worst_a = 0.0
    for b in DESIGN_GRID:
        vs = list(pts[b].values())
        a = float(np.mean([1 - v["r_between_med"] / v["r_true_med"] for v in vs]))
        worst_a = max(worst_a, abs(a - conv66[b]["realised_A"]))
    assert worst_a < 1e-12, f"p66's A axis does not rebuild: {worst_a:.3e}"
    say(f"  eight design points, worst deviation from p66's A axis "
        f"{worst_a:.2e}   ok")
    out["anchors"] = {"a4_p66_axis_rebuilt": worst_a}

    # ------------------------------------------------------------- the sweep
    grid = [(b, rt) for b in DESIGN_GRID for rt in sorted(pts[b])]
    if args.smoke:
        grid = [(0.95, sorted(pts[0.95])[-2]), (0.99, sorted(pts[0.99])[-2])]
    say(f"\n=== 69.3 the sweep: {len(grid)} grid points, the venue level in "
        f"both conventions ===")
    say(f"  {'design':>7} {'r_true':>8} {'venue wr':>10} {'venue wor':>10} "
        f"{'cell':>9} {'wor/wr':>8}")
    rows, a1_worst, a2_worst = [], 0.0, 0.0
    for b, rt in grid:
        rec = pts[b][rt]
        reps = 2 if args.smoke else rec["n_rep"]
        Pf = W.field("data", rec["knob"])
        delta, tag = rec["delta"], (ARM_DATA, int(round(rt * 1e6)),
                                    int(round(b * 1e6)))
        wr, wor, tru, btw = [], [], [], []
        for i in range(reps):
            d = W.draw(Pf, delta, i, tag, levels=(), noise=False, brute=True)
            A_wr, A_wor, f = venue_matrices(W, Pf, d["phi"], delta)
            a2 = np.abs(A_wr - d["A_true"]).max() / max(np.abs(d["A_true"]).max(), 1e-300)
            a2_worst = max(a2_worst, a2)
            wr.append(newman(A_wr, W.pop))
            wor.append(newman(A_wor, W.pop))
            tru.append(d["r_true"])
            btw.append(d["r_between"])
        med = lambda x: float(np.median(x))
        if not args.smoke:
            for name, mine, theirs in (("r_true_med", med(tru), rec["r_true_med"]),
                                       ("r_between_med", med(btw),
                                        rec["r_between_med"])):
                dev = abs(mine - theirs) / abs(theirs) if theirs else abs(mine)
                a1_worst = max(a1_worst, dev)
                assert dev == 0.0 or dev < 1e-12, (
                    f"{b}|{rt} {name}: {mine!r} against the stored {theirs!r} -- "
                    f"the world moved, and nothing below this line is comparable")
        rows.append(dict(beta=b, r_true_target=rt, n_rep=reps,
                         r_venue_wr=med(wr), r_venue_wor=med(wor),
                         r_true_med=med(tru), r_between_med=med(btw),
                         med_corr=rec["med_corr"],
                         wor_over_wr=med(wor) / med(wr)))
        say(f"  {b:>7.3f} {rt:>8.3f} {med(wr):>10.5f} {med(wor):>10.5f} "
            f"{med(btw):>9.5f} {med(wor) / med(wr):>8.5f}")
    out["surface"] = rows
    out["anchors"]["a1_worst_relative_deviation"] = a1_worst
    out["anchors"]["a2_closed_form_vs_draw"] = a2_worst
    say(f"\n  A1 stored r_true_med / r_between_med reproduced, worst relative "
        f"deviation {a1_worst:.2e}")
    say(f"  A2 closed-form venue matrix against draw()'s own A_true, worst "
        f"relative deviation {a2_worst:.2e}")

    # ------------------------------------------------------ the ladder's grid
    ladder_betas = DESIGN_GRID
    n_cells, n_reps = LADDER_CELLS, LADDER_REPS
    if args.smoke:
        # The ladder is the half of this phase a two-point sweep does not touch,
        # so the wiring check runs it too, small: one beta, one replicate, 200
        # cells. A ladder bug found here costs seconds instead of eight minutes.
        ladder_betas, n_cells, n_reps = (0.95,), 200, 1

    # --------------------------------------------------------------- the ladder
    say("\n=== 69.4 the merge ladder, materialised and fed to p44's collapse ===")
    rng = np.random.default_rng(LADDER_SEED)
    cells = np.sort(rng.choice(W.n_cell, size=n_cells, replace=False))
    say(f"  {n_cells} cells, {int(W.m[cells].sum()):,} venues, "
        f"{n_reps} replicates per point")
    ladder, a3_worst = [], 0.0
    for b in ladder_betas:
        rt = min(pts[b], key=lambda r: abs(pts[b][r]["r_true_med"] - bnd[b]))
        rec = pts[b][rt]
        Pf = W.field("data", rec["knob"])
        delta = rec["delta"]
        tag = (ARM_DATA, int(round(rt * 1e6)), int(round(b * 1e6)))
        curves = {"random": {}, "within-cell": {}}
        ks = {"fractional": [], "integerised": [], "wr_subsample": [],
              "int_people_dropped": []}
        for i in range(n_reps):
            d = W.draw(Pf, delta, i, tag, levels=(), noise=False, brute=True)
            A_wr, A_wor, f = venue_matrices(W, Pf, d["phi"], delta)
            V, owner = materialise(W, Pf, d["phi"], delta, f, cells)
            # A3: the explicit venues ARE the closed form, on these cells and
            # with the world's own f. One group per venue is the g = 1 rung.
            for conv in (True, False):
                ref = venue_matrices(W, Pf, d["phi"], delta, f=f,
                                     cells=cells)[0 if conv else 1]
                ref_r = newman(ref, W.pop)
                dev = abs(pooled_r(V, np.arange(len(V)), W, conv) - ref_r) / abs(ref_r)
                a3_worst = max(a3_worst, dev)
            # THE CONVENTION'S RATIO, k = r_venue_wor / r_venue_wr, measured on
            # these cells in two arms. The survey's U are integer headcounts, so
            # v_a^2 - v_a >= 0 always; a synthetic venue carries fractional
            # arrivals and the subtraction can exceed the square. The integerised
            # arm rounds every venue to whole people first, which is the state
            # the survey's data is actually in, and the fractional arm applies
            # the formula as written. Both are reported; 69.5 carries the axis
            # each one implies.
            one = np.arange(len(V))
            V_int = np.rint(V)
            k_frac = (pooled_r(V, one, W, False) / pooled_r(V, one, W, True))
            k_int = (pooled_r(V_int, one, W, False) / pooled_r(V_int, one, W, True))
            ks["fractional"].append(float(k_frac))
            ks["integerised"].append(float(k_int))
            ks["wr_subsample"].append(float(pooled_r(V, one, W, True)))
            ks["int_people_dropped"].append(
                float(1 - V_int.sum() / V.sum()))
            # the cell each venue sits in, and its position inside that cell
            order = np.argsort(owner, kind="stable")
            Vo, o = V[order], owner[order]
            starts = np.flatnonzero(np.r_[True, o[1:] != o[:-1]])
            sizes = np.diff(np.r_[starts, len(o)])
            pos = np.arange(len(o)) - np.repeat(starts, sizes)
            cell_of = np.repeat(np.arange(len(starts)), sizes)
            perm = rng.permutation(len(V))
            for g in LADDER_G:
                nc = len(V) // g
                gid = np.repeat(np.arange(nc), g)
                curves["random"].setdefault(g, []).append(
                    pooled_r(V[perm[:nc * g]], gid, W, False))
                gid2 = cell_of * (int(sizes.max()) // g + 1) + pos // g
                _, gid2 = np.unique(gid2, return_inverse=True)
                curves["within-cell"].setdefault(g, []).append(
                    pooled_r(Vo, gid2, W, False))
            # the top rung: the whole cell, in both conventions. The
            # with-replacement one is p39's own r_between restricted to these
            # cells, which is what makes the ladder end where the paper reads.
            _, gcell = np.unique(owner, return_inverse=True)
            curves["within-cell"].setdefault("cell", []).append(
                pooled_r(V, gcell, W, False))
            curves["random"].setdefault("cell", []).append(
                pooled_r(V, gcell, W, True))
        row = dict(beta=b, r_true_target=rt,
                   arms={k: {str(g): float(np.median(v)) for g, v in c.items()}
                         for k, c in curves.items()},
                   k_wor_over_wr={k: float(np.median(v)) for k, v in ks.items()})
        ladder.append(row)
        say(f"  beta {b:>5.2f}: venue {row['arms']['within-cell']['1']:.5f}"
            f" -> g=8 {row['arms']['within-cell']['8']:.5f}"
            f" -> cell {row['arms']['within-cell']['cell']:.5f}"
            f"   k = {row['k_wor_over_wr']['fractional']:+.4f} fractional, "
            f"{row['k_wor_over_wr']['integerised']:+.4f} integerised")
    out["ladder"] = ladder
    out["anchors"]["a3_explicit_vs_closed_form"] = a3_worst
    say(f"  A3 explicit materialisation against the closed form, worst "
        f"relative deviation {a3_worst:.2e}")

    if args.smoke:
        with open(f"{ROOT}/eda/results_p69.partial.json", "w") as fh:
            json.dump(out, fh, indent=1)
        say("\nwrote results_p69.partial.json (smoke run)")
        return 0

    # ------------------------------------------------------------- the C axis
    say("\n=== 69.5 the C axis, beside p66's A and B ===")
    say(f"  {'design':>7} {'bound':>8} {'A':>9} {'B':>9} {'C frac':>9} "
        f"{'C int':>9} {'C int - B':>10}")
    ax_a, ax_b, ax_c, ax_ci, conv = [], [], [], [], []
    for b in DESIGN_GRID:
        near = min([r for r in rows if r["beta"] == b],
                   key=lambda r: abs(r["r_true_med"] - bnd[b]))
        a = conv66[b]["realised_A"]
        bb = conv66[b]["realised_B"]
        # C = 1 - med_corr / r_venue_wor = 1 - (1 - B) / k, with
        # k = r_venue_wor / r_venue_wr. Writing it through k is what lets the
        # integerised arm -- which only the explicit venues can measure -- share
        # the full world's numerator instead of needing a second noisy pipeline.
        kk = {r["beta"]: r["k_wor_over_wr"] for r in ladder}[b]
        c = float(1 - near["med_corr"] / near["r_venue_wor"])
        c_int = float(1 - (1 - bb) / kk["integerised"])
        c_sub = float(1 - (1 - bb) / kk["fractional"])
        c_noiseless = float(1 - near["r_between_med"] / near["r_venue_wor"])
        # HOW FLAT IS THE AXIS IN r_true. p66 made A its primary because A is
        # constant in r_true to five places and B is not, so a B axis is a
        # family of axes rather than one. C inherits whatever the without-
        # replacement correction does to that, and the correction is a function
        # of venue OCCUPANCY, not of beta -- so the spread is reported here for
        # C the way p66 reports it for A, rather than being hidden by the choice
        # of evaluation point.
        slice_c = [1 - r["r_between_med"] / r["r_venue_wor"]
                   for r in rows if r["beta"] == b and r["r_between_med"] > 0]
        conv.append(dict(design=b, bound=bnd[b], realised_A=a, realised_B=bb,
                         realised_C=c, realised_C_noiseless=c_noiseless,
                         C_noiseless_spread_over_r_true=(
                             float(max(slice_c) - min(slice_c)) if slice_c else None),
                         C_noiseless_over_r_true=[
                             dict(r_true=r["r_true_med"],
                                  C=1 - r["r_between_med"] / r["r_venue_wor"],
                                  wor_over_wr=r["wor_over_wr"])
                             for r in rows if r["beta"] == b],
                         realised_C_integerised=c_int,
                         realised_C_fractional_subsample=c_sub,
                         k_fractional_full=near["wor_over_wr"],
                         k_fractional_subsample=kk["fractional"],
                         k_integerised_subsample=kk["integerised"],
                         int_people_dropped=kk["int_people_dropped"],
                         r_venue_wr=near["r_venue_wr"],
                         r_venue_wor=near["r_venue_wor"],
                         wor_over_wr=near["wor_over_wr"],
                         evaluated_at_r_true=near["r_true_med"]))
        ax_a.append(a); ax_b.append(bb); ax_c.append(c); ax_ci.append(c_int)
        say(f"  {b:>7.3f} {bnd[b]:>8.5f} {a:>9.5f} {bb:>9.5f} {c:>9.5f} "
            f"{c_int:>9.5f} {c_int - bb:>+10.5f}")
    out["conversion"] = conv
    gap_cb = max(abs(c - bb) for c, bb in zip(ax_c, ax_b))
    gap_ca = max(abs(c - a) for c, a in zip(ax_c, ax_a))
    out["definition_gap"] = dict(C_vs_B=gap_cb, C_vs_A=gap_ca,
                                 A_vs_B=p66["definition_gap_max"])
    say(f"  C differs from B by at most {gap_cb:.5f} and from A by at most "
        f"{gap_ca:.5f}; p66 measured A against B at "
        f"{p66['definition_gap_max']:.5f}.")

    # ------------------------------------------------------ the verdicts on C
    say("\n=== 69.6 every measured reading, re-read on the C axis ===")
    seq = [bnd[b] for b in DESIGN_GRID]
    # WHERE THE C AXIS EXISTS AT ALL. An axis has to be increasing in design beta
    # to be inverted, and the without-replacement convention does not deliver
    # that everywhere: k = r_venue_wor / r_venue_wr is a function of venue
    # OCCUPANCY, not of beta, and each design point is evaluated at the r_true
    # nearest its own bound -- so the low-beta points are read where the venues
    # hold too few people for the correction to leave a positive reading. That is
    # a definedness rule and not a choice between answers: the segment kept is
    # the longest run ENDING AT THE TOP of the design grid on which the axis is
    # positive and strictly increasing, every design point below it is reported
    # as undefined, and no reading is extrapolated to either side. The rule is
    # applied to both arms and both results are printed.
    def usable(ax):
        i = len(ax) - 1
        while i > 0 and ax[i - 1] > 0 and ax[i - 1] < ax[i]:
            i -= 1
        return i
    mono_i, mono_f = usable(ax_ci), usable(ax_c)
    say(f"  the axis is increasing and positive from design "
        f"{DESIGN_GRID[mono_i]} upwards (integerised) and from "
        f"{DESIGN_GRID[mono_f]} upwards (fractional)")
    out["axis_defined_from"] = dict(
        integerised=DESIGN_GRID[mono_i], fractional=DESIGN_GRID[mono_f],
        rule="longest run ending at the top of the design grid on which the "
             "axis is positive and strictly increasing")
    # The integerised arm is the survey-shaped one -- the survey's headcounts
    # are whole people -- so it is the axis the verdicts are read on whenever it
    # is defined at least as far down as the fractional one.
    if mono_i <= mono_f:
        axis, axis_name, lo = ax_ci[mono_i:], "integerised", mono_i
    else:
        axis, axis_name, lo = ax_c[mono_f:], "fractional", mono_f
    seq_used = seq[lo:]
    out["axis_used"] = axis_name
    top = max(axis)
    bottom = min(axis)
    readings = list(p59["claim"]["measured_betas"])
    vetoed = p59["claim"].get("excluded_reading")
    if vetoed:
        readings.append(dict(name=vetoed["name"] + " [vetoed in advance]",
                             beta=vetoed["beta"], convention="vetoed"))
    say(f"  the C axis ({axis_name}) spans {bottom:.4f} to {top:.4f}, over "
        f"design {DESIGN_GRID[lo]} to {DESIGN_GRID[-1]}")
    say(f"\n  {'reading':<52} {'beta':>8} {'bound C':>9} {'bound A':>9}  verdict")
    verdicts = []
    for r in readings:
        b = r["beta"]
        if b > top or b < bottom:
            side = "ABOVE" if b > top else "BELOW"
            verdicts.append(dict(name=r["name"], beta=b, bound=None,
                                 excludes=None, off_grid=True, side=side))
            say(f"  {r['name'][:52]:<52} {b:>8.4f} {'--':>9} {'--':>9}  "
                f"OFF GRID ({side})")
            continue
        bd = float(np.interp(b, axis, seq_used))
        bd_a = float(np.interp(b, ax_a, seq)) if b <= max(ax_a) else float("nan")
        verdicts.append(dict(name=r["name"], beta=b, bound=bd,
                             bound_on_A=bd_a, excludes=bool(bd < r_survey),
                             off_grid=False, margin=float(r_survey - bd)))
        say(f"  {r['name'][:52]:<52} {b:>8.4f} {bd:>9.4f} {bd_a:>9.4f}  "
            f"{'excludes' if bd < r_survey else 'DOES NOT EXCLUDE'} "
            f"({r_survey - bd:+.4f})")
    out["verdicts"] = verdicts
    on = [v for v in verdicts if not v["off_grid"] and "vetoed" not in v["name"]]
    exc = [v for v in on if v["excludes"]]
    star_c = (float(np.interp(r_survey, seq_used, axis))
              if seq_used[0] <= r_survey <= seq_used[-1] else float("nan"))
    out["beta_star"] = dict(design=p59["beta_star"]["primary_value"],
                            realised_A=p66["beta_star"]["realised_A"],
                            realised_B=p66["beta_star"]["realised_B"],
                            realised_C=star_c)
    out["answer"] = dict(
        r_survey=r_survey, n_readings=len(verdicts), n_on_grid=len(on),
        n_excluding=len(exc),
        n_off_grid=len([v for v in verdicts if v["off_grid"]]),
        excluding=[v["name"] for v in exc],
        not_excluding=[v["name"] for v in on if not v["excludes"]],
        thinnest_excluding_margin=min([v["margin"] for v in exc], default=None),
        c_axis_top=top, c_axis_bottom=bottom, axis_used=axis_name,
    )
    # The one comparison the paper leans on, named rather than left to be found
    # among twelve rows: the Seoul arm, which is the primary reading.
    seoul = next((v for v in verdicts if v["name"].startswith("Seoul 202312")),
                 None)
    if seoul:
        out["answer"]["primary_reading"] = dict(
            name=seoul["name"], beta=seoul["beta"], bound_C=seoul["bound"],
            bound_A=seoul.get("bound_on_A"), excludes=seoul["excludes"],
            margin_C=seoul["margin"],
            margin_A=float(r_survey - seoul["bound_on_A"]))

    say("\n=== the number ===")
    say(f"  beta* is {star_c:.4f} on the C axis, against "
        f"{p66['beta_star']['realised_A']:.4f} on A and "
        f"{p59['beta_star']['primary_value']:.4f} in design units.")
    say(f"  {len(exc)} of {len(on)} on-grid readings exclude the survey's "
        f"{r_survey:.4f}; the thinnest margin is "
        f"{out['answer']['thinnest_excluding_margin']:+.4f}.")
    say(f"  the run took {time.time() - t0:.0f} s")

    with open(f"{ROOT}/eda/results_p69.json", "w") as fh:
        json.dump(out, fh, indent=1)
    say(f"\nwrote {ROOT}/eda/results_p69.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
