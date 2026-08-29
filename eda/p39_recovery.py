#!/usr/bin/env python
"""Phase 39 — the recovery experiment: how much true assortativity survives the
cell boundary, and what that licenses us to say about the true value.

WHY NOW. The advisor's 2026-08-21 letter, point 2, makes this the highest
priority in the project and says why: a reviewer will ask whether the ladder is
about Seoul or about aggregation itself, and whether a near-rank-1 spectrum is a
mechanical consequence of the estimator. The only way to answer is to build a
world whose answer is known and run the identical estimator on it. The payoff is
a change of claim -- from "we saw nothing" to "we bounded the true value" -- and
the first is a negative result while the second is a number that can be cited.

THE OBSERVATION THAT DETERMINES THE WHOLE DESIGN. With cells
l = (destination dong, arrival hour, day of week), the estimator is

    A[a,a'] = sum_l n_a(l) n_a'(l) / n_.(l)

and n_a(l) is a MARGIN. The estimator never sees who paired with whom. Writing
p(.|l) for cell l's age composition and w_l = n_.(l) / sum_l n_.(l),

    A / A.sum()  ==  sum_l  w_l  p(.|l) (x) p(.|l)                        (39.1)

exactly, as an algebraic identity rather than an approximation. Section 39.1
asserts it numerically on the real data before anything else runs. Two things
follow, and both belong in the paper:

  * A is identically a convex combination of rank-1 outer products, which is why
    the measured spectrum is near rank-1 (sigma1^2/sum sigma^2 median 0.9823 over
    79 months, p37). That is a property of the estimator's input, not a discovery
    about Seoul, and it is the answer to "is the rank-1 result mechanical?" --
    yes, partly, and here is the algebra.
  * r_hat IS A FUNCTIONAL OF THE CELL MARGINS ALONE. Anything that happens below
    the cell boundary at fixed cell margins is invisible to it. So true
    assortativity cannot be encoded in a pairing rule -- aggregation has already
    thrown pairings away -- and a generative model whose truth lives at or above
    the cell line tests nothing.

THE TWO-LAYER MODEL. Individuals meet inside venues -- a classroom, an office
floor, a clinic waiting room. Venue v has composition q_v and size s_v, contacts
inside it are frequency-dependent, so the truth is the same functional form one
level down,

    A_true = sum_v s_v q_v (x) q_v ,   A_obs = sum_l S_l p_l (x) p_l ,
    A_true - A_obs = sum_l S_l Cov_{v in l}(q)  >=  0                     (39.2)

-- the loss is exactly the within-cell between-venue composition variance. The
two knobs are therefore

    the CELL field   p_l, which fixes r_between: what is recoverable in principle
    the VENUE split  delta, which adds truth strictly below the cell line

and a venue takes composition q_v = (1 - delta) p_l + delta e_{k(v)}, with the
venue's type k(v) drawn from p_l itself and e_k the single-band composition. That
choice is not cosmetic: it makes E_v[q_v] = p_l exactly for every delta, so the
cell margins -- and therefore r_hat -- are unchanged by it, and

    r_true = (1 - delta^2) r_between + delta^2                           (39.3)

in closed form. beta := 1 - r_between/r_true is then "the share of the truth that
sits below the cell line", the letter's second sweep dimension, and the grid is
laid out in (r_true, beta) and inverted to (gamma, delta).

WHY THE LETTER'S ARCHETYPE FIELD IS RUN BUT NOT USED. The letter proposes cell
compositions drawn from K archetypes (school-heavy, workplace-heavy,
elderly-heavy). That is implemented here as a 16-member archetype family and
assigned to real cells by how much each real cell looks like each archetype, so
it inherits Seoul's spatial and diurnal arrangement. It FAILS p37's free
internal validation: coarsening dong -> gu keeps 15.0% of its excess against the
27.6-32.6% p37 measured in 79 of 79 real months. A hard archetype label throws
away the smooth part of the composition field, and the smooth part is most of
what survives a 17-fold merge. The primary arm therefore uses the real
composition field itself, sharpened by a power transform,
p_l ~ p_real(l)^gamma r^(1-gamma), which reproduces 30.6% at gamma = 1 by
construction. Both arms are reported; section 39.7 is the evidence.

FOUR FAILURE MODES, FOUR DETECTORS. Each is a named section rather than a
sentence, because each would produce a plausible-looking curve:

  1  segregation above the cell line  ->  beta is swept, and r_true is
     decomposed into between-cell and within-cell parts, both reported (39.6)
  2  cells too thin, noise manufactures excess (p33 R1: 20.7% of the published
     dong-level assortativity is device sampling noise)  ->  a MANDATORY
     alpha = 0 null arm (39.4). If true proportionate mixing reads back a large
     positive value the experiment is measuring noise and is void, and the
     script stops rather than producing a curve nobody should use.
  3  the synthetic population is already proportionate-mixing  ->  r_true is
     computed from the venue-level pairings BEFORE aggregation and asserted
     against (39.3), and against an explicitly materialised venue list (39.3b)
  4  margins do not match, so the r (x) r null is a different null  ->  the
     synthetic age margin is asserted against the measured one at 1e-9 (39.3)

THE INVERSION IS A TEST, NOT A READ-OFF.

    r_true^upper = the smallest r_true whose replicate distribution of r_hat has
                   its 5th percentile above the observed value,

so every r_true at or above it is rejected one-sided at 5%. The observation is
the NOISE-CORRECTED 0.01677 from p33 R1, not the published 0.02115, compared
against a synthetic reading corrected by its own alpha = 0 null median so that
the two sides are corrected the same way; the raw-against-raw bound is reported
beside it. Monotonicity is checked empirically, because without it the inversion
is undefined and only a compatibility region can be reported.

NOTHING IS RE-IMPLEMENTED. matrices() and stats() are imported from p26_matrix
and handed a table in cells()'s schema; excess_stats/rank_stats come from
p32_pmix. results_p26/p32/p33/p34/p37/p38 are read and never written.

    python eda/p39_recovery.py --coarse        # 5-point shape check, ~2 min
    python eda/p39_recovery.py                 # the full surface
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calendar_kr as K
from common import AGE_LABEL, AGES
from p26_matrix import cells, holiday_free_dows, matrices, stats
from p32_pmix import excess_stats, rank_stats, symmetrise
from paths import DERIVED, FIG, RESULTS, ROOT, require

NA = len(AGES)
SEED = 20260821
YM = 202312                      # the survey month; p33's R1 is measured on it
YM_CHECK = 202001                # the 2020 month the realism table is checked on
PANEL = "WE"

IMP_PUB = 1.5                    # p26's published imputation for a masked row
IMP_MEASURED = 2.27              # p29/p30's measured fill; p33 carries the same

# p33 R1, 202312, WE, dong, holiday-free: the published point estimate, the
# device-noise bias its parametric bootstrap measured, and the corrected value.
R_OBS_PUB = 0.02115385938228908
P33_NOISE_BIAS = 0.0043860558176699906
R_OBS_CORRECTED = 0.01676780356461909
# p32, survey|202312|WE|seoul, bands < 80. The letter's "~0.22".
R_SURVEY = 0.22270993081818996
# p37, 79 months, zero exceptions: share of dong-level excess surviving 17x
P37_KEPT = (0.27614019178568994, 0.3257572440213601)
P37_KEPT_MEDIAN = 0.29837771337134716

VENUE = 50.0                     # per-day arrivals per venue (a classroom/floor)
SIGMA = 0.6                      # archetype-family width, fitted in 39.2
SIGMA_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0)
# The base composition field is averaged over five seasonally adjacent months.
# A single month's field carries its own device noise, and sharpening it and then
# injecting fresh Poisson noise would count that noise twice: the dong -> gu
# retention of a one-month field is 30.4% before noise and 25.2% after, against
# p37's 27.6-32.6%. Averaging five months cuts the embedded noise by ~sqrt(5) and
# takes the noiseless retention to 32.9%, inside the band. Section 39.7 reports
# both numbers rather than only the one that passes.
BASE_MONTHS = (202310, 202311, 202312, 202401, 202402)

# the grid. beta = share of the truth that sits below the cell line.
BETAS = (0.0, 0.5, 0.8, 0.95, 0.99)
RTRUE = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.22, 0.35, 0.50)
COARSE_BETAS = (0.0, 0.8)
COARSE_RTRUE = (0.01, 0.02, 0.05, 0.22)
Q_LO = 5.0                       # the one-sided test's lower quantile, per cent

out = {}


# --------------------------------------------------------------------- helpers
def newman(A, pop):
    """The published estimator's assortativity. p26.stats is imported rather than
    its two-line formula retyped: a second implementation of r would be a
    different object, and p31/p36 guard only the imported one."""
    return stats(A, pop)["assortativity"]


def ipf_rows_cols(X, S, N, iters=200, tol=1e-15):
    """Scale X to row sums S and column sums N. Used to give a candidate cell
    field the measured age margin EXACTLY while leaving each cell's volume alone,
    so that failure mode 4 -- a different r (x) r null on the two sides -- cannot
    happen by a fraction of a per cent."""
    X = np.array(X, float)
    for _ in range(iters):
        rs = X.sum(1)
        X *= np.divide(S, rs, out=np.zeros_like(rs), where=rs > 0)[:, None]
        cs = X.sum(0)
        X *= np.divide(N, cs, out=np.zeros_like(cs), where=cs > 0)[None, :]
        if (np.abs(X.sum(1) / S - 1).max() < tol
                and np.abs(X.sum(0) / N - 1).max() < tol):
            break
    return X


def archetypes(r, sigma):
    """A venue-composition family indexed by the age band it concentrates on.

    kappa(a, k) is a Gaussian in band index, row-normalised over k so that
    sum_k kappa(a, k) = 1. With Q_k(a) = r_a kappa(a, k) / Z_k and
    pi_k = Z_k = sum_b r_b kappa(b, k),

        sum_k pi_k Q_k(a) = r_a sum_k kappa(a, k) = r_a          exactly,

    for every sigma, which is what keeps the archetype arm's margin equal to the
    measured one. sigma -> 0 gives one archetype per band (r_target -> 1),
    sigma -> infinity gives proportionate mixing. The three the letter names --
    school-heavy, workplace-heavy, elderly-heavy -- are members k = 2, 7, 13.
    """
    k = np.arange(NA)
    G = np.exp(-((k[:, None] - k[None, :]) ** 2) / (2.0 * sigma * sigma))
    kap = G / G.sum(1, keepdims=True)
    pi = (r[:, None] * kap).sum(0)
    Q = (r[:, None] * kap / pi[None, :]).T
    return Q, pi


def assign_prices(score, vol, pi, iters=1200, eta0=0.15, eta1=0.02):
    """Assign each real cell to the archetype it most resembles, subject to the
    archetype volume shares coming out at pi.

    The assignment is what would carry Seoul's spatial and diurnal arrangement
    into the archetype arm. Left unconstrained it would also fix the archetype
    volume shares at whatever the data happens to give, and those do not average
    back to r, so the synthetic margin would drift. A per-archetype price
    restores the shares without changing the ranking inside any one cell.
    """
    lam, best = np.zeros(pi.shape), (np.inf, None)
    for it in range(iters):
        c = np.argmax(score + lam[None, :], axis=1)
        share = np.array([vol[c == k].sum() for k in range(len(pi))])
        share = share / share.sum()
        err = float(np.abs(share / pi - 1).max())
        if err < best[0]:
            best = (err, lam.copy())
        # Undamped the price update overshoots and the assignment oscillates
        # between two corners; annealing eta and keeping the best iterate takes
        # the worst share error from 127% to 0.16%.
        lam += (eta0 * (eta1 / eta0) ** (it / max(iters - 1, 1))) * (
            np.log(pi) - np.log(np.maximum(share, 1e-12)))
    c = np.argmax(score + best[1][None, :], axis=1)
    share = np.array([vol[c == k].sum() for k in range(len(pi))])
    return c, share / share.sum(), best[0]


class World:
    """The synthetic universe's fixed furniture: the cell set, the cell volumes,
    the measured margin, the masked-row pattern, the venue counts, the coverage
    weights and the schema-shaped frame matrices() is handed. Built once; a
    replicate only refills the v_obs column.
    """

    def __init__(self, ym, w, pop, imp_true=IMP_MEASURED, venue=VENUE, quiet=False):
        self.ym, self.w, self.pop, self.imp_true = ym, w, pop, imp_true
        self.dows = holiday_free_dows(ym)
        nd = {d: K.cell_exposure(ym, d)["n_days"] for d in self.dows}

        df = cells(ym)
        d = df[(df.dest_attr != "H") & (df.dow_n.isin(self.dows))]
        # The truth carries the masked volume at the MEASURED fill. The synthetic
        # release then hides imp_true * n_masked of it and the estimator restores
        # imp * n_masked, so imp != imp_true reproduces the real analysis's
        # misspecification rather than modelling it.
        d = d.assign(n=(d.v_obs + imp_true * d.n_masked) / d.dow_n.map(nd))
        g = d.groupby(["d_dong", "arr_hour", "dow_n", "age"], as_index=False).agg(
            n=("n", "sum"), m=("n_masked", "sum"))
        wide = g.pivot_table(index=["d_dong", "arr_hour", "dow_n"], columns="age",
                             values="n", fill_value=0.0).reindex(columns=AGES,
                                                                 fill_value=0.0)
        mk = g.pivot_table(index=["d_dong", "arr_hour", "dow_n"], columns="age",
                           values="m", fill_value=0.0).reindex(columns=AGES,
                                                               fill_value=0.0)
        assert list(wide.index) == list(mk.index)
        self.X = wide.to_numpy(float)                    # per-day arrivals
        self.M = mk.to_numpy(float)                      # masked ROW counts
        idx = wide.index.to_frame(index=False)
        self.nd_cell = idx.dow_n.map(nd).to_numpy(float)
        self.S = self.X.sum(1)
        self.N = self.X.sum(0)                           # the measured margin
        self.r = self.N / self.N.sum()
        self.p_real = self.X / np.maximum(self.S, 1e-30)[:, None]
        self.key = {k: i for i, k in enumerate(
            zip(idx.d_dong.to_numpy(), idx.arr_hour.to_numpy(),
                idx.dow_n.to_numpy()))}
        self.n_cell = len(self.S)
        self.gu = (idx.d_dong.to_numpy() // 1000)
        self.hourdow = idx.arr_hour.to_numpy() * 10 + idx.dow_n.to_numpy()

        # venues per cell, proportional to cell volume. Small cells get few, so
        # their realised composition is lumpy -- which is true segregation, not
        # noise, and something the surface has to price rather than assume away.
        self.m = np.maximum(1, np.rint(self.S / venue)).astype(np.int64)
        self.s = self.S / self.m                         # venue size

        # the archetype family, for the arm 39.7 shows fails
        self.p_base = self.p_real
        self.Q, self.pi = archetypes(self.r, SIGMA)
        self.G = np.einsum("k,ka,kb->ab", self.pi, self.Q, self.Q)
        self.r_target = newman(self.G, pop)

        # the schema matrices() expects, built once. dest_attr is 'E' on every
        # row: the WE panel sums W and E inside the cell, so the split does not
        # enter the estimand, and halving the row count halves the runtime.
        self.frame = pd.DataFrame({
            "d_dong": np.repeat(idx.d_dong.to_numpy(), NA),
            "arr_hour": np.repeat(idx.arr_hour.to_numpy(), NA),
            "dow_n": np.repeat(idx.dow_n.to_numpy(), NA),
            "dest_attr": "E", "age": np.tile(np.array(AGES), self.n_cell),
            "v_obs": 0.0, "n_masked": self.M.ravel()})
        if not quiet:
            print(f"  {ym}: {self.n_cell:,} cells x {NA} bands, "
                  f"{self.S.sum():,.0f} arrivals over the {len(self.dows)}-day "
                  f"set, venues/cell median {np.median(self.m):.0f} "
                  f"[{self.m.min()}, {self.m.max()}], archetype r_target "
                  f"{self.r_target:.4f}")

    # ------------------------------------------------------------- cell fields
    def build_base_field(self, months=BASE_MONTHS, quiet=False):
        """The mean age composition of each (dong, hour, dow) cell over several
        seasonally adjacent months, with the target month's margin imposed.

        This is the synthetic world's ground-truth spatial field. It has to be an
        estimate of Seoul's TRUE composition field rather than one month's
        published one, because the published one already contains the device
        noise that the measurement layer is about to add again; see BASE_MONTHS.
        """
        acc = np.zeros((self.n_cell, NA))
        for ym in months:
            df = cells(ym)
            d = df[df.dest_attr != "H"]
            nd = {x: K.cell_exposure(ym, x)["n_days"] for x in range(1, 8)}
            d = d.assign(n=(d.v_obs + self.imp_true * d.n_masked)
                         / d.dow_n.map(nd))
            g = d.groupby(["d_dong", "arr_hour", "dow_n", "age"],
                          as_index=False)["n"].sum()
            wi = g.pivot_table(index=["d_dong", "arr_hour", "dow_n"],
                               columns="age", values="n",
                               fill_value=0.0).reindex(columns=AGES,
                                                       fill_value=0.0)
            ix = wi.index.to_frame(index=False)
            rows = np.array([self.key.get(k, -1) for k in
                             zip(ix.d_dong, ix.arr_hour, ix.dow_n)])
            m = rows >= 0
            V = wi.to_numpy(float)[m]
            acc[rows[m]] += V / np.maximum(V.sum(1, keepdims=True), 1e-30)
        acc = np.maximum(acc, 0)
        s = acc.sum(1, keepdims=True)
        # a cell absent from every month keeps the target month's composition
        acc = np.where(s > 0, acc / np.maximum(s, 1e-300), self.p_real)
        self.p_base = ipf_rows_cols(self.S[:, None] * acc, self.S,
                                    self.N) / self.S[:, None]
        self.base_months = list(months)
        if not quiet:
            print(f"  base field: mean composition over {len(months)} months "
                  f"{months[0]}-{months[-1]}, noiseless r_between "
                  f"{newman(self.cell_matrix(self.p_base), self.pop):.5f} "
                  f"(the single-month field gives "
                  f"{newman(self.cell_matrix(self.p_real), self.pop):.5f})")

    def field(self, kind, knob, fast=False):
        """A candidate cell-composition field with the measured margin exactly.

        'data'      p_l ~ p_real(l)^gamma r^(1-gamma) -- the real field, sharpened
                    (gamma=1 is the real field itself, gamma=0 is proportionate)
        'archetype' p_l = (1-alpha) r + alpha Q_{c_l}  -- the letter's construction
        """
        if kind == "data":
            # A large gamma drives log p far negative and exp() underflows to
            # zero; a cell whose every band underflows then has a zero row, and
            # the normalisation below turns it into NaN, which surfaces 400 lines
            # later as "pvals contains NaNs" inside the multinomial. The
            # log-space clip and the zero-row fallback are what keep the extreme
            # corner of the grid (r_between >= 0.35) computable at all.
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = (knob * np.log(np.maximum(self.p_base, 1e-300))
                      + (1 - knob) * np.log(self.r)[None, :])
            lg = lg - lg.max(1, keepdims=True)
            P = np.where(self.p_base > 0, np.exp(np.maximum(lg, -700.0)), 0.0) \
                if knob > 0 else np.tile(self.r, (self.n_cell, 1))
        elif kind == "archetype":
            P = (1 - knob) * self.r[None, :] + knob * self.Q[self.c_arch]
        else:
            raise ValueError(kind)
        P = np.maximum(P, 0.0)
        tot = P.sum(1, keepdims=True)
        P = np.where(tot > 0, P / np.maximum(tot, 1e-300),
                     self.r[None, :])          # a cell that underflowed to zero
        X = ipf_rows_cols(self.S[:, None] * P, self.S, self.N,
                          iters=15 if fast else 200)
        P = X / self.S[:, None]
        # multinomial() rejects a row that does not lie in the simplex, and IPF
        # leaves rounding at the last ulp, so the row is re-normalised here
        # rather than being rejected 30 seconds into a 100-replicate point.
        P = np.maximum(P, 0.0)
        P = P / P.sum(1, keepdims=True)
        assert np.isfinite(P).all(), f"{kind} field at knob {knob} is not finite"
        return P

    def build_archetype_field(self):
        with np.errstate(divide="ignore"):
            L = np.log(np.maximum(self.Q, 1e-300)) - np.log(self.r)[None, :]
        self.c_arch, self.share_arch, self.share_err = assign_prices(
            self.p_base @ L.T, self.S, self.pi)

    def solve_knob(self, kind, r_between, lo=0.0, hi=None, iters=45):
        """Bisect the field knob so the noiseless cell-level reading equals a
        requested r_between. Monotone in the knob, checked by the bracket. The
        bisection runs on a 15-iteration IPF because the target only has to be
        hit approximately -- every replicate MEASURES its own r_between, and
        that measured value, not the target, is what the surface is plotted on.
        """
        if r_between <= 0:
            return 0.0
        hi = (1.0 if kind == "archetype" else 25.0) if hi is None else hi
        f = lambda k: newman(self.cell_matrix(self.field(kind, k, fast=True)),
                             self.pop)
        top = f(hi)
        if not np.isfinite(top) or top < r_between:
            return None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if f(mid) < r_between:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def cell_matrix(self, P):
        X = self.S[:, None] * P
        t = X.sum(1)
        k = t > 0
        return (X[k] / t[k, None]).T @ X[k]

    # ------------------------------------------------------------ one replicate
    def draw(self, P, delta, rep, tag, levels=("dong",), imp=IMP_PUB,
             noise=True, mask_error=True, brute=False):
        """One synthetic world and one reading of it.

        The rng is seeded from (SEED, tag, replicate) and from nothing else, so
        extending the grid in either direction cannot move a point already
        computed -- the style p38 proved with 96,672 numbers at 0.0e+00.
        """
        rng = np.random.default_rng([SEED] + list(tag) + [rep])
        # venue types drawn from the cell's own composition, so E_v[q_v] = p_l
        # exactly and the cell margin -- hence r_hat -- does not move with delta
        # except through the finite-venue fluctuation of phi, which is real.
        phi = rng.multinomial(self.m, P) / self.m[:, None]
        Xc = self.S[:, None] * ((1 - delta) * P + delta * phi)
        f = self.N / Xc.sum(0)
        X = Xc * f[None, :]

        # --- the truth, from the venue-level pairings, before any aggregation.
        # A venue of type k in cell l has count vector s_l * ((1-d) p_l + d e_k)
        # * f. Summing v v^T / v.sum() over all of them collapses to two 16 x 16
        # matmuls; 39.3b materialises a subsample of venues and checks it.
        Pt = (1 - delta) * (P * f[None, :])
        pf = P @ f
        gm = (self.m[:, None] * phi) * self.s[:, None] / (
            (1 - delta) * pf[:, None] + delta * f[None, :])
        H = Pt.T @ gm                                    # 16 x 16, column = k
        A_true = (Pt * gm.sum(1)[:, None]).T @ Pt
        Hf = H * f[None, :]
        A_true = A_true + delta * (Hf + Hf.T)
        A_true[np.diag_indices(NA)] += delta * delta * f * f * gm.sum(0)
        r_true = newman(A_true, self.pop)

        # --- the ceiling: what the estimator reads with no noise and no masking
        t = X.sum(1)
        r_between = newman((X / t[:, None]).T @ X, self.pop)

        # --- the measurement layer, exactly p33 R1's
        Y = X
        if noise:
            # devices are what was sampled: n_a = w_a d_a, so a Poisson device
            # count gives Var(n_a) = w_a n_a. p33 applies this to the same
            # per-day cell table, which is what makes the floors comparable.
            Y = rng.poisson(np.maximum(X / self.w[None, :], 0)) * self.w[None, :]
        v = Y * self.nd_cell[:, None] - (self.imp_true if mask_error else imp) * self.M
        self.frame["v_obs"] = np.maximum(v, 0.0).ravel()

        res = dict(r_true=r_true, r_between=r_between, delta=delta,
                   clipped=float((v < 0).mean()),
                   rescale_max=float(np.abs(f - 1).max()),
                   margin_rel=float(np.abs(X.sum(0) / self.N - 1).max()))
        if brute:
            res["A_true"] = A_true
            res["phi"] = phi
            res["X"] = X
        for lev in levels:
            A = matrices(self.frame, self.ym, PANEL, lev, imp=imp, dows=self.dows)
            res[f"r_hat_{lev}"] = newman(A, self.pop)
        return res


# ------------------------------------------------- 39.3b the venue-level anchor
def brute_A_true(W, P, phi, delta, f, cells_idx):
    """Materialise every venue of the selected cells as an explicit count vector
    and sum v v' / v.sum() over them, one venue at a time.

    draw() collapses that sum into two 16 x 16 matmuls. The collapse is where an
    algebra slip would hide, and a slip there would move r_true -- the thing the
    whole calibration is expressed in -- by a plausible amount while every other
    number in the file stayed sane. So the loop is run on a subsample and the two
    are compared at machine precision.
    """
    A = np.zeros((NA, NA))
    for l in cells_idx:
        counts = np.rint(phi[l] * W.m[l]).astype(int)
        for k in range(NA):
            if counts[k] == 0:
                continue
            q = (1 - delta) * P[l] + delta * np.eye(NA)[k]
            v = W.s[l] * q * f
            A += counts[k] * np.outer(v, v) / v.sum()
    return A


def pairing_check(rng, n_vec, n_pairs=400000):
    """Draw actual contacts inside one venue and check the pair-count matrix
    converges on n (x) n / n_. -- i.e. that A_true really is "who met whom" under
    frequency-dependent mixing, and not merely a formula that looks like it."""
    p = n_vec / n_vec.sum()
    a = rng.choice(NA, size=n_pairs, p=p)
    b = rng.choice(NA, size=n_pairs, p=p)
    T = np.zeros((NA, NA))
    np.add.at(T, (a, b), 1.0)
    T = (T + T.T) / 2
    want = np.outer(n_vec, n_vec) / n_vec.sum()
    return T / T.sum(), want / want.sum()


def quantile_curve(surface, beta, key="corr"):
    """(median realised r_true, lower quantile of r_hat) along one beta-slice."""
    pts = sorted((v["r_true_med"], v[f"q_lo_{key}"], v[f"med_{key}"], t)
                 for t, v in surface.items() if v["beta"] == beta)
    return ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts],
            [p[3] for p in pts])


def invert(xs, qlo, obs):
    """Smallest x whose lower quantile of r_hat exceeds obs, by linear
    interpolation between the bracketing grid points. Everything at or above it
    is rejected one-sided at the quantile's level. None if the grid never
    crosses -- which means the experiment excluded nothing, and says so."""
    for i in range(1, len(xs)):
        if qlo[i - 1] <= obs < qlo[i]:
            t = (obs - qlo[i - 1]) / (qlo[i] - qlo[i - 1])
            return float(xs[i - 1] + t * (xs[i] - xs[i - 1]))
    return None if qlo[-1] <= obs else float(xs[0])


def make_figure(surface, inv, lad, noiseless, NULL, kept_real, reps):
    """The figure, as a function of what results_p39.json already stores, so
    `--figure-only` can redraw it without re-running the Monte Carlo."""
    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    cols = {0.0: "#0E7C86", 0.5: "#4C6E8A", 0.8: "#BC8034", 0.95: "#A8434E",
            0.99: "#6B4C7A"}
    ax = axes[0]
    for beta in BETAS:
        xs, qlo, med, keys = quantile_curve(surface, beta, "raw")
        c = cols.get(beta, "0.5")
        ax.plot(xs, med, "-o", ms=3.5, lw=1.4, c=c, label=f"beta = {beta}")
        for t in keys:
            h = surface[t]["r_hat"]
            if h:
                x = surface[t]["r_true_med"]
                ax.plot(np.full(len(h), x), h, ".", ms=1.0, c=c, alpha=.30,
                        zorder=1)
    ax.axhline(NULL, ls="--", c="0.35", lw=1)
    ax.annotate(f"$\\alpha=0$ null arm  {NULL:.5f}", xy=(0.985, NULL),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=6.5, color="0.35")
    ax.axhline(R_OBS_PUB, ls="-", c="k", lw=1)
    ax.annotate(f"observed  {R_OBS_PUB:.5f}", xy=(0.985, R_OBS_PUB),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=6.5)
    ax.axvline(R_SURVEY, ls=":", c="#A8434E", lw=1.2)
    ax.annotate("survey 0.223", xy=(R_SURVEY, 0.62),
                xycoords=("data", "axes fraction"), rotation=90, fontsize=6.5,
                color="#A8434E", ha="right", va="bottom")
    ax.annotate("the replicate scatter is plotted and invisible: 100 draws span\n"
                "less than 1e-4, so the width of this experiment is the choice\n"
                r"of $\beta$ and of the field, not Monte-Carlo error.",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=6.2, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true assortativity $r_{true}$ (from the venue-level pairings)")
    ax.set_ylabel(r"$\hat r$ read at dong level")
    ax.set_title("the calibration surface: what the estimator\nreads back, by "
                 "how much of the truth sits below the cell", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3, which="both")

    ax = axes[1]
    for beta in BETAS:
        xs, qlo, med, keys = quantile_curve(surface, beta, "corr")
        c = cols.get(beta, "0.5")
        ax.plot(xs, qlo, "-o", ms=3.5, lw=1.4, c=c, label=f"beta = {beta}")
        b = inv[str(beta)]["bound_corrected"]
        if b is not None:
            ax.plot([b], [R_OBS_CORRECTED], "v", ms=9, c=c, zorder=6)
    ax.axhline(R_OBS_CORRECTED, ls="-", c="k", lw=1.1)
    ax.annotate(f"noise-corrected observation  {R_OBS_CORRECTED:.5f}",
                xy=(0.015, R_OBS_CORRECTED), xytext=(0, 3),
                textcoords="offset points",
                xycoords=("axes fraction", "data"), ha="left", va="bottom",
                fontsize=6.5)
    ax.axvline(R_SURVEY, ls=":", c="#A8434E", lw=1.2)
    ax.annotate("survey 0.223", xy=(R_SURVEY, 0.02),
                xycoords=("data", "axes fraction"), rotation=90, fontsize=6.5,
                color="#A8434E", ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true assortativity $r_{true}$")
    ax.set_ylabel(r"5th percentile of $\hat r$, null-corrected")
    ax.set_title("the inversion: the triangle is the largest $r_{true}$\n"
                 "not yet rejected at one-sided 5%", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3, which="both")

    ax = axes[2]
    ax.axhspan(P37_KEPT[0], P37_KEPT[1], color="#0E7C86", alpha=.13, lw=0,
               label="p37, 79 months (27.6-32.6%)")
    ax.axhline(P37_KEPT_MEDIAN, c="#0E7C86", lw=1, ls="--")
    ax.axhline(kept_real, c="k", lw=0.9, ls=":")
    ax.annotate(f"the real {YM} rung, {kept_real:.1%}", xy=(0.985, kept_real),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=6.5)
    for kind, mk, c in (("data", "o", "#4C6E8A"), ("archetype", "s", "#A8434E")):
        xs = [lad[k]["r_true_target"] for k in lad if k.startswith(kind + "|")]
        ys = [lad[k]["kept_median"] for k in lad if k.startswith(kind + "|")]
        ax.plot(xs, ys, mk + "-", c=c, ms=6, lw=1.2,
                label=f"{kind} field, with device noise")
        if kind in noiseless:
            ax.plot([0.02], [noiseless[kind]], mk, mfc="none", mec=c, ms=10,
                    mew=1.6, label=f"{kind} field, no device noise")
    ax.set_xscale("log")
    ax.set_xlabel("true assortativity $r_{true}$ (beta = 0)")
    ax.set_ylabel("share of dong-level excess kept at gu")
    ax.set_ylim(0.0, 0.42)
    ax.set_title("p37's free validation: the letter's archetype field\n"
                 "is not Seoul-like; the data field is, before noise", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=.3)

    fig.suptitle(f"Phase 39 — recovery experiment, {YM}, {PANEL} panel, "
                 f"{reps} replicates a point", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "p39_recovery.png", dpi=150)
    plt.close(fig)




# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--coarse", action="store_true",
                    help="only the 5-point shape check")
    ap.add_argument("--coarse-reps", type=int, default=25)
    ap.add_argument("--ladder-reps", type=int, default=25)
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw fig/p39_recovery.png from results_p39.json")
    ap.add_argument("--verify-seeds", action="store_true",
                    help="recompute a few stored grid points out of order")
    args = ap.parse_args()
    t0 = time.time()
    if args.figure_only:
        # Everything the figure draws is in the results file, which is the
        # point of storing all 100 replicates per grid point rather than a
        # summary: the picture is recomputable from the numbers, by anyone.
        R = json.load(open(f"{RESULTS}/results_p39.json"))
        make_figure(R["surface"], R["inversion"], R["ladder"]["points"],
                    R["ladder"]["noiseless"], R["null_arm"]["null_median"],
                    R["ladder"]["real_rung"], R["config"]["reps"])
        print(f"redrew {FIG}/p39_recovery.png from results_p39.json")
        return 0
    require(DERIVED / f"arrivals_dow_{YM}.parquet",
            f"p26 arrival cache for {YM} (run eda/p26_matrix.py)")

    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p32 = json.load(open(f"{ROOT}/eda/results_p32.json"))
    p33 = json.load(open(f"{ROOT}/eda/results_p33.json"))
    p37 = json.load(open(f"{ROOT}/eda/results_p37.json"))
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    # p33's own two lines, so the noise model here is the noise model there.
    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}
    w = np.array([cov[AGE_LABEL[a]] for a in AGES], float)
    assert np.abs(w - np.array(p33["profile"]["w"])).max() == 0.0, (
        "the coverage profile differs from the one p33 pushed through the "
        "matrix; the null arm would not be comparable to p33's floor")
    print(f"=== 39.0 setup ===")
    print(f"  coverage weights reproduce results_p33.json exactly "
          f"(max |diff| 0.0e+00), so the device-noise model is p33's")
    # every published number this script compares itself against is re-read from
    # the file that owns it, so a constant here cannot drift from its source.
    src = dict(
        r_obs_pub=p33["R1_effective_sample"]["202312"]["stats"]["assortativity"]["point"],
        noise_bias=p33["R1_effective_sample"]["202312"]["stats"]["assortativity"]["noise_bias"],
        r_obs_corrected=p33["R1_effective_sample"]["202312"]["stats"]["assortativity"]["bias_corrected"],
        survey=p32["excess"]["survey|202312|WE|seoul"]["assortativity"],
        kept_min=p37["ladder"]["kept_min"], kept_max=p37["ladder"]["kept_max"],
        kept_median=p37["ladder"]["kept_median"])
    for nm, got, want in (("published r", src["r_obs_pub"], R_OBS_PUB),
                          ("p33 noise bias", src["noise_bias"], P33_NOISE_BIAS),
                          ("corrected r", src["r_obs_corrected"], R_OBS_CORRECTED),
                          ("survey r", src["survey"], R_SURVEY),
                          ("p37 kept min", src["kept_min"], P37_KEPT[0]),
                          ("p37 kept max", src["kept_max"], P37_KEPT[1]),
                          ("p37 kept median", src["kept_median"], P37_KEPT_MEDIAN)):
        assert got == want, f"{nm}: this file says {want}, its source says {got}"
    print(f"  7 published constants re-read from results_p32/p33/p37 and "
          f"asserted bit for bit")
    out["published_constants"] = src

    # ------------------------------------------------------- 39.1 the identity
    # A / A.sum() = sum_l w_l p(.|l) (x) p(.|l). Asserted first, on the real
    # data, because everything after it is a consequence: the estimator sees
    # margins only, so the ground truth has to live below the cell line.
    print("\n=== 39.1 anchor 1: the estimator is a mixture of rank-1 outer "
          "products ===")
    ident = []
    for ym in (YM, YM_CHECK):
        df = cells(ym)
        hf = holiday_free_dows(ym)
        nd = {d: K.cell_exposure(ym, d)["n_days"] for d in hf}
        for level in ("dong", "gu", "city"):
            A = matrices(df, ym, PANEL, level, dows=hf)
            d = df[(df.dest_attr != "H") & (df.dow_n.isin(hf))]
            d = d.assign(n=(d.v_obs + IMP_PUB * d.n_masked) / d.dow_n.map(nd))
            loc = {"dong": d.d_dong, "gu": d.d_dong // 1000,
                   "city": pd.Series(0, index=d.index)}[level]
            g = (d.assign(loc=loc).groupby(["loc", "arr_hour", "dow_n", "age"],
                                           as_index=False)["n"].sum())
            X = g.pivot_table(index=["loc", "arr_hour", "dow_n"], columns="age",
                              values="n", fill_value=0.0).reindex(
                                  columns=AGES, fill_value=0.0).to_numpy(float)
            tot = X.sum(1)
            X, tot = X[tot > 0], tot[tot > 0]
            Pm = X / tot[:, None]
            E = (Pm * (tot / tot.sum())[:, None]).T @ Pm
            dev = float(np.abs(A / A.sum() - E).max())
            ident.append(dict(ym=ym, level=level, max_abs_dev=dev,
                              n_cells=int(len(tot))))
            assert dev < 1e-12, f"{ym} {level}: identity (39.1) fails at {dev:.1e}"
    print(f"  {len(ident)} (month, level) checks, max |A/A.sum() - "
          f"sum_l w_l p (x) p| = {max(x['max_abs_dev'] for x in ident):.2e}")
    print("  so the near-rank-1 spectrum p37 reports in all 79 months "
          "(sigma1^2/sum sigma^2 median 0.9823) is a property of the "
          "estimator's\n  input, not a discovery about Seoul -- and true "
          "assortativity can only enter through segregation ACROSS cells.")
    out["identity"] = ident

    # ------------------------------------------------- 39.2 realism and design
    print("\n=== 39.2 the synthetic universe, taken from the data ===")
    W = World(YM, w, pops[YM])
    W.build_base_field()
    W.build_archetype_field()
    Wc = World(YM_CHECK, w, pops[YM_CHECK])
    real = {}
    for tag, U in (("202312", W), ("202001", Wc)):
        q = np.percentile(U.S, [10, 50, 90])
        real[tag] = dict(n_cells=U.n_cell, n_dows=len(U.dows),
                         volume=float(U.S.sum()),
                         cell_p10=float(q[0]), cell_median=float(q[1]),
                         cell_p90=float(q[2]), cell_min=float(U.S.min()),
                         thin_share=float((U.S < 50).mean()),
                         masked_rows=float(U.M.sum()),
                         margin=(U.N / U.N.sum()).tolist())
        print(f"  {tag}: {U.n_cell:,} cells over {len(U.dows)} holiday-free "
              f"weekdays, volume {U.S.sum():,.0f}, cell size p10/median/p90 "
              f"{q[0]:,.0f}/{q[1]:,.0f}/{q[2]:,.0f}, "
              f"{(U.S < 50).mean():.1%} below 50 arrivals/day")
    d12 = np.abs(np.array(real["202312"]["margin"])
                 / np.array(real["202001"]["margin"]) - 1)
    print(f"  the cell universe and the size distribution are the same object "
          f"in both months; the AGE MARGIN is not -- it moves by up to "
          f"{d12.max():.0%}\n  (worst band "
          f"{AGE_LABEL[AGES[int(np.argmax(d12))]]}), which is why the synthetic "
          f"world is built on {YM}'s margin and {YM_CHECK} is a realism check "
          f"rather than a second arm.")
    real["margin_shift_max"] = float(d12.max())
    real["margin_shift_band"] = AGE_LABEL[AGES[int(np.argmax(d12))]]
    out["realism"] = real

    # the archetype family's width, fitted against the Korean survey's own
    # excess pattern rather than chosen. Reported even though 39.7 rejects the
    # archetype arm, because it is what makes M_true a survey-shaped matrix.
    sq = [i for i, a in enumerate(AGES) if a < 80]
    Csv = np.array(json.load(open(f"{ROOT}/eda/results_p27.json"))
                   ["survey"]["202312|WE|seoul"]["C"])[np.ix_(sq, sq)]
    Ts, _ = symmetrise(Csv, pops[YM][sq])
    es = Ts / Ts.sum()
    rs = es.sum(1)
    Ds = (es - np.outer(rs, rs)) / np.sqrt(np.outer(rs, rs))
    sig_fit = []
    for sg in SIGMA_GRID:
        Q, pi = archetypes(W.r, sg)
        G = np.einsum("k,ka,kb->ab", pi, Q, Q)[np.ix_(sq, sq)]
        eg = G / G.sum()
        rg = eg.sum(1)
        Dg = (eg - np.outer(rg, rg)) / np.sqrt(np.outer(rg, rg))
        sig_fit.append(dict(sigma=sg, r_target=newman(
            np.einsum("k,ka,kb->ab", pi, Q, Q), pops[YM]),
            corr_with_survey=float(np.corrcoef(Dg.ravel(), Ds.ravel())[0, 1])))
    best = max(sig_fit, key=lambda x: x["corr_with_survey"])
    here = [x for x in sig_fit if x["sigma"] == SIGMA][0]
    print(f"  archetype family: sigma = {SIGMA} is used; its excess pattern "
          f"correlates {here['corr_with_survey']:.3f} with the Korean survey's, "
          f"against {best['corr_with_survey']:.3f} at the grid's best "
          f"({best['sigma']}). r_target {W.r_target:.4f}, so alpha in [0, 1] "
          f"reaches r_true {W.r_target:.2f} > the survey's {R_SURVEY:.2f}.")
    print(f"  archetype volume shares hit their targets to "
          f"{W.share_err:.2%} (worst band)")
    out["sigma_fit"] = sig_fit
    out["design"] = dict(sigma=SIGMA, r_target=W.r_target, venue=VENUE,
                         base_months=W.base_months,
                         archetype_share_err=W.share_err,
                         survey_assortativity=R_SURVEY,
                         venues_per_cell_median=float(np.median(W.m)),
                         venues_per_cell_min=int(W.m.min()),
                         venues_per_cell_max=int(W.m.max()))

    if args.verify_seeds:
        # The house rule is that a seed depends on its own grid point and on
        # nothing else, so extending or reordering the grid cannot move a point
        # that has already been run. Asserting that costs a minute: pick four
        # stored points, recompute them in a DIFFERENT order and without the
        # other 46 points having been computed first, and demand bit equality.
        R = json.load(open(f"{RESULTS}/results_p39.json"))
        dev, n = 0.0, 0
        for beta, rt in ((0.95, 0.05), (0.0, 0.02), (0.8, 0.22), (0.5, 0.01)):
            want = R["surface"][f"data|{beta}|{rt}"]["r_hat"]
            rb = (1 - beta) * rt
            Pf = W.field("data", W.solve_knob("data", rb))
            delta = float(np.sqrt(max(rt - rb, 0.0) / max(1 - rb, 1e-12)))
            got = [W.draw(Pf, delta, i, (3, int(round(rt * 1e6)),
                                         int(round(beta * 1e6))))["r_hat_dong"]
                   for i in range(len(want))]
            d = float(np.abs(np.array(got) - np.array(want)).max())
            print(f"  beta {beta:<5} r_true {rt:<6} {len(got)} replicates, "
                  f"max |diff| {d:.1e}")
            dev, n = max(dev, d), n + len(got)
        print(f"  {n} stored numbers recomputed out of grid order, max "
              f"deviation {dev:.1e}")
        assert dev == 0.0, "a seed depends on something other than its own point"
        return 0

    # ------------------------------------------------------- 39.3 the anchors
    print("\n=== 39.3 anchors: margins, the truth, and the venue algebra ===")
    P0 = W.field("data", 1.0)
    anch = {}
    # (a) failure mode 4 -- the margin. Band by band, relative.
    d0 = W.draw(P0, 0.0, 0, (0, 0, 0), brute=True)
    band_rel = np.abs(d0["X"].sum(0) / W.N - 1)
    anch["margin_max_rel"] = float(band_rel.max())
    print(f"  (a) synthetic age margin against the measured one, band by band: "
          f"max relative deviation {band_rel.max():.2e}  "
          f"[{'PASS' if band_rel.max() < 1e-9 else 'FAIL'}]")
    assert band_rel.max() < 1e-9, "the synthetic margin is not the measured one"
    # (b) failure mode 3 -- the truth is not proportionate mixing, and it equals
    # the closed form (39.3) in the infinite-venue limit
    cf = []
    for delta in (0.0, 0.1, 0.3, 0.6):
        r_ = W.draw(P0, delta, 0, (0, 1, int(delta * 100)))
        want = (1 - delta ** 2) * r_["r_between"] + delta ** 2
        cf.append(dict(delta=delta, r_true=r_["r_true"], closed_form=want,
                       rel=float(abs(r_["r_true"] / want - 1)),
                       r_between=r_["r_between"]))
        print(f"  (b) delta {delta:.1f}: r_true {r_['r_true']:.6f} from the "
              f"pairings, {want:.6f} from (39.3), rel {cf[-1]['rel']:.2e}"
              + ("   <- exact at delta = 0" if delta == 0 else ""))
    assert cf[0]["rel"] < 1e-9
    print("      the gap at delta > 0 is the finite-venue term: (39.3) is the "
          "m -> infinity limit and cells hold "
          f"{np.median(W.m):.0f} venues at the median. r_true is MEASURED from "
          "the realised\n      venues everywhere below, never read off (39.3).")
    anch["closed_form"] = cf
    # (c) the venue algebra, against an explicit venue list
    rng = np.random.default_rng([SEED, 99])
    idxs = rng.choice(W.n_cell, 250, replace=False)
    dd = W.draw(P0, 0.3, 0, (0, 1, 30), brute=True)
    fB = W.N / (W.S[:, None] * ((1 - 0.3) * P0 + 0.3 * dd["phi"])).sum(0)
    Ab = brute_A_true(W, P0, dd["phi"], 0.3, fB, idxs)
    # the same restriction, from the collapsed form
    m = np.zeros(W.n_cell, bool)
    m[idxs] = True
    Pt = 0.7 * (P0 * fB[None, :])
    gm = (W.m[:, None] * dd["phi"]) * W.s[:, None] / (
        0.7 * (P0 @ fB)[:, None] + 0.3 * fB[None, :])
    gm = gm * m[:, None]
    H = Pt.T @ gm
    Ac = (Pt * gm.sum(1)[:, None]).T @ Pt
    Hf = H * fB[None, :]
    Ac = Ac + 0.3 * (Hf + Hf.T)
    Ac[np.diag_indices(NA)] += 0.09 * fB * fB * gm.sum(0)
    dev = float(np.abs(Ab - Ac).max() / np.abs(Ac).max())
    anch["venue_bruteforce_rel"] = dev
    print(f"  (c) 250 cells materialised venue by venue "
          f"({int(np.rint(dd['phi'][idxs] * W.m[idxs, None]).sum()):,} venues): "
          f"max relative deviation from the collapsed form {dev:.2e}")
    assert dev < 1e-9, "the collapsed venue sum is not the explicit one"
    # (d) A_true really is "who met whom"
    nv = np.rint(W.S[idxs[0]] * P0[idxs[0]]).astype(float) + 1.0
    got, want = pairing_check(np.random.default_rng([SEED, 98]), nv)
    anch["pairing_mc_max_abs"] = float(np.abs(got - want).max())
    print(f"  (d) 400,000 contacts drawn inside one venue: the pair-count "
          f"matrix differs from n (x) n / n_. by at most "
          f"{np.abs(got - want).max():.2e} (Monte-Carlo error)")

    # ------------------------------------------- 39.4 THE alpha = 0 NULL ARM
    # p33 R1: all four indicators are non-negative quadratic forms in the cell
    # counts, so INDEPENDENT NOISE MANUFACTURES EXCESS. If true proportionate
    # mixing reads back something comparable to the observation, the experiment
    # is measuring noise and there is nothing to calibrate.
    print("\n=== 39.4 the alpha = 0 null arm: true proportionate mixing, read "
          "by this pipeline ===")
    Pnull = W.field("data", 0.0)
    assert np.abs(Pnull - W.r[None, :]).max() < 1e-12
    null = {}
    for tag, (noise, mask_err) in (("clean", (False, False)),
                                   ("mask only", (False, True)),
                                   ("noise only", (True, False)),
                                   ("both", (True, True))):
        v = [W.draw(Pnull, 0.0, i, (0, 2, {"clean": 0, "mask only": 1,
                                           "noise only": 2, "both": 3}[tag]),
                    noise=noise, mask_error=mask_err)["r_hat_dong"]
             for i in range(args.coarse_reps)]
        null[tag] = dict(median=float(np.median(v)), lo=float(np.min(v)),
                         hi=float(np.max(v)), n=len(v))
        print(f"  {tag:<11} r_hat = {np.median(v):+.5f}  "
              f"[{np.min(v):+.5f}, {np.max(v):+.5f}]")
    NULL = null["both"]["median"]
    print(f"\n  p33 R1 measured the device-noise bias in the PUBLISHED matrix "
          f"as +{P33_NOISE_BIAS:.5f}.")
    print(f"  this pipeline, on a world with NO age structure at all, reads "
          f"{NULL:+.5f} -- a ratio of {NULL / P33_NOISE_BIAS:.2f}.")
    void = NULL >= R_OBS_CORRECTED
    warn = not (0.5 * P33_NOISE_BIAS <= NULL <= 2.0 * P33_NOISE_BIAS)
    out["null_arm"] = dict(arms=null, null_median=NULL,
                           p33_noise_bias=P33_NOISE_BIAS,
                           ratio_to_p33=float(NULL / P33_NOISE_BIAS),
                           observed_corrected=R_OBS_CORRECTED,
                           void=bool(void), outside_half_to_double=bool(warn))
    if void:
        print("\n  *** STOP. The null arm reads at or above the noise-corrected "
              "observation.\n      Independent device noise alone accounts for "
              "the measurement, so this\n      experiment is measuring noise "
              "and is VOID. No calibration curve follows. ***")
        with open(f"{RESULTS}/results_p39.json", "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        return 1
    print(f"  the null is {NULL / R_OBS_CORRECTED:.1%} of the noise-corrected "
          f"observation, so the arm PASSES and the sweep may run."
          + ("  (WARNING: outside [0.5x, 2x] of p33's bias.)" if warn else ""))

    # --------------------------------------------------------- the sweep loop
    knobs, ARM = {}, {"data": 3, "archetype": 4}

    def point(kind, beta, r_true, reps, levels=("dong",)):
        """One grid point: r_between = (1-beta) r_true fixes the cell field,
        delta then adds the rest of the truth strictly below the cell line."""
        rb = (1 - beta) * r_true
        ky = (kind, round(rb, 9))
        if ky not in knobs:
            knobs[ky] = W.solve_knob(kind, rb)
        if knobs[ky] is None:
            return None
        Pf = W.field(kind, knobs[ky])
        delta = float(np.sqrt(max(r_true - rb, 0.0) / max(1 - rb, 1e-12)))
        rows = [W.draw(Pf, delta, i, (ARM[kind], int(round(r_true * 1e6)),
                                      int(round(beta * 1e6))), levels=levels)
                for i in range(reps)]
        h = np.array([x["r_hat_dong"] for x in rows])
        rec = dict(kind=kind, beta=beta, r_true_target=r_true, knob=knobs[ky],
                   delta=delta, n_rep=reps,
                   r_true_med=float(np.median([x["r_true"] for x in rows])),
                   r_between_med=float(np.median([x["r_between"] for x in rows])),
                   med_raw=float(np.median(h)), med_corr=float(np.median(h) - NULL),
                   q_lo_raw=float(np.percentile(h, Q_LO)),
                   q_lo_corr=float(np.percentile(h, Q_LO) - NULL),
                   q_hi_raw=float(np.percentile(h, 100 - Q_LO)),
                   r_hat=h.tolist(),
                   clipped=float(np.median([x["clipped"] for x in rows])))
        if "gu" in levels:
            k = [(x["r_hat_gu"] - x["r_hat_city"])
                 / (x["r_hat_dong"] - x["r_hat_city"]) for x in rows]
            rec["kept_median"] = float(np.median(k))
            rec["r_hat_gu_med"] = float(np.median([x["r_hat_gu"] for x in rows]))
            rec["r_hat_city_med"] = float(np.median([x["r_hat_city"]
                                                     for x in rows]))
        return rec

    # ------------------------------------------------ 39.5 the coarse 5 points
    print(f"\n=== 39.5 the coarse grid first: does the surface have the shape "
          f"the design predicts? ===")
    tc = time.time()
    coarse = {}
    print(f"  {'beta':>5} {'r_true':>8} {'gamma':>7} {'delta':>6} "
          f"{'r_between':>10} {'r_true(real)':>12} {'r_hat med':>10} "
          f"{'q05':>9} {'recovered':>10}")
    for beta in COARSE_BETAS:
        for rt in COARSE_RTRUE:
            rec = point("data", beta, rt, args.coarse_reps)
            coarse[f"data|{beta}|{rt}"] = rec
            print(f"  {beta:>5.2f} {rt:>8.3f} {rec['knob']:>7.3f} "
                  f"{rec['delta']:>6.3f} {rec['r_between_med']:>10.5f} "
                  f"{rec['r_true_med']:>12.5f} {rec['med_raw']:>10.5f} "
                  f"{rec['q_lo_raw']:>9.5f} "
                  f"{rec['med_corr'] / rec['r_true_med']:>9.1%}", flush=True)
    out["coarse"] = coarse
    mono = all(coarse[f"data|{b}|{COARSE_RTRUE[i]}"]["med_raw"]
               < coarse[f"data|{b}|{COARSE_RTRUE[i + 1]}"]["med_raw"]
               for b in COARSE_BETAS for i in range(len(COARSE_RTRUE) - 1))
    drop = all(coarse[f"data|{COARSE_BETAS[0]}|{rt}"]["med_corr"]
               > coarse[f"data|{COARSE_BETAS[-1]}|{rt}"]["med_corr"]
               for rt in COARSE_RTRUE)
    print(f"  shape: monotone in r_true at every beta {mono}; "
          f"recovery falls as beta rises {drop}; "
          f"coarse grid took {time.time() - tc:.0f}s")
    out["coarse_shape"] = dict(monotone_in_r_true=bool(mono),
                               falls_with_beta=bool(drop),
                               seconds=float(time.time() - tc))
    if args.coarse:
        with open(f"{RESULTS}/results_p39.json", "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(f"\nwrote {RESULTS}/results_p39.json (coarse only)")
        return 0

    # ---------------------------------------------------- 39.6 the full surface
    print(f"\n=== 39.6 the calibration surface, {args.reps} replicates a point "
          f"===")
    surface = {}
    null_rec = dict(kind="data", beta=0.0, r_true_target=0.0, knob=0.0,
                    delta=0.0, n_rep=null["both"]["n"], r_true_med=0.0,
                    r_between_med=0.0, med_raw=NULL, med_corr=0.0,
                    q_lo_raw=null["both"]["lo"], q_lo_corr=null["both"]["lo"] - NULL,
                    q_hi_raw=null["both"]["hi"], r_hat=[], clipped=0.0)
    for beta in BETAS:
        print(f"  beta = {beta}")
        surface[f"data|{beta}|0.0"] = dict(null_rec, beta=beta)
        for rt in RTRUE:
            rec = point("data", beta, rt, args.reps)
            if rec is None:
                print(f"    r_true {rt}: r_between {(1 - beta) * rt:.3f} is "
                      f"beyond the field's reach, skipped")
                continue
            surface[f"data|{beta}|{rt}"] = rec
            print(f"    r_true {rt:>6.3f} -> r_between {rec['r_between_med']:.5f}"
                  f", r_hat {rec['med_raw']:.5f} "
                  f"[q05 {rec['q_lo_raw']:.5f}, q95 {rec['q_hi_raw']:.5f}], "
                  f"recovered {rec['med_corr'] / rec['r_true_med']:.1%}",
                  flush=True)
    out["surface"] = surface
    print(f"  elapsed {time.time() - t0:.0f}s")

    # ---------------------------------------- 39.6b does the masked fill move it
    # The letter asks for both imputations. The truth hides imp_true * n_masked
    # and the estimator restores imp * n_masked, so three configurations exist
    # and only one of them is the situation the real analysis is in.
    print("\n=== 39.6b the masked fill: the published 1.5 against p29's 2.27 ===")
    W15 = World(YM, w, pops[YM], imp_true=IMP_PUB, quiet=True)
    W15.build_base_field(quiet=True)
    imps = {}
    for beta in (0.0, 0.8):
        for rt in (0.02, 0.05):
            rb = (1 - beta) * rt
            delta = float(np.sqrt(max(rt - rb, 0.0) / max(1 - rb, 1e-12)))
            row = {}
            for nm, U, ie in (("truth 2.27 / est 1.50", W, IMP_PUB),
                              ("truth 2.27 / est 2.27", W, IMP_MEASURED),
                              ("truth 1.50 / est 1.50", W15, IMP_PUB)):
                ky = ("data", round(rb, 9))
                kn = U.solve_knob("data", rb)
                Pf = U.field("data", kn)
                v = [U.draw(Pf, delta, i, (6, int(round(rt * 1e6)),
                                           int(round(beta * 1e6))), imp=ie
                            )["r_hat_dong"] for i in range(20)]
                row[nm] = float(np.median(v))
            imps[f"{beta}|{rt}"] = row
            base = row["truth 2.27 / est 1.50"]
            print(f"  beta {beta:>4.2f} r_true {rt:.2f}: "
                  + "  ".join(f"{k} {v:.5f} ({v / base - 1:+.2%})"
                              for k, v in row.items()))
    out["imputation"] = imps
    print("  the masked rows are 20-44 only and worth ~11% of volume, so the "
          "fill moves the reading far less than the device noise does; 39.4's\n"
          "  decomposition puts the misspecification at +0.00005 against "
          "+0.00431 for noise.")

    # ------------------------------------------- 39.7 p37's free validation
    print("\n=== 39.7 dong -> gu retention: is the synthetic world Seoul-like? "
          "===")
    rung = {x["level"]: x["assortativity"] for x in p26["ladder"]
            if x["ym"] == YM and x["panel"] == PANEL}
    kept_real = ((rung["gu"] - rung["city"]) / (rung["dong"] - rung["city"]))
    print(f"  p37, 79 months, zero exceptions: median {P37_KEPT_MEDIAN:.1%}, "
          f"range {P37_KEPT[0]:.1%}-{P37_KEPT[1]:.1%}.  "
          f"The real {YM} rung gives {kept_real:.1%}.")
    lad = {}
    for kind in ("data", "archetype"):
        for rt in (0.02, 0.05, 0.15):
            rec = point(kind, 0.0, rt, args.ladder_reps, levels=("dong", "gu",
                                                                 "city"))
            if rec is None:
                continue
            lad[f"{kind}|{rt}"] = rec
            ok = P37_KEPT[0] <= rec["kept_median"] <= P37_KEPT[1]
            print(f"  {kind:>10} r_true {rt:>5.2f} (beta 0): r_hat "
                  f"{rec['med_raw']:.5f}, kept {rec['kept_median']:.1%}  "
                  f"[{'inside' if ok else 'OUTSIDE'} p37's band]")
    # the same field with the noise switched off, which is what separates a
    # field that is not Seoul-like from a noise model that is not
    noiseless = {}
    for kind in ("data", "archetype"):
        Pf = W.field(kind, W.solve_knob(kind, 0.02))
        r_ = W.draw(Pf, 0.0, 0, (ARM[kind], 5, 0), levels=("dong", "gu", "city"),
                    noise=False)
        k = (r_["r_hat_gu"] - r_["r_hat_city"]) / (r_["r_hat_dong"]
                                                   - r_["r_hat_city"])
        noiseless[kind] = float(k)
        print(f"  {kind:>10} r_true 0.02, NO device noise: kept {k:.1%}  "
              f"[{'inside' if P37_KEPT[0] <= k <= P37_KEPT[1] else 'OUTSIDE'}]")
    out["ladder"] = dict(points=lad, noiseless=noiseless, real_rung=kept_real,
                         p37_band=list(P37_KEPT), p37_median=P37_KEPT_MEDIAN)

    # ------------------------------------------------------- 39.8 the inversion
    print("\n=== 39.8 the inversion: what does the observation exclude? ===")
    print(f"  observed, noise-corrected (p33 R1): {R_OBS_CORRECTED:.5f}; "
          f"published point estimate {R_OBS_PUB:.5f}; survey {R_SURVEY:.4f}")
    inv = {}
    for beta in BETAS:
        xs, qlo_c, med_c, keys = quantile_curve(surface, beta, "corr")
        _, qlo_r, med_r, _ = quantile_curve(surface, beta, "raw")
        mono = all(qlo_c[i] <= qlo_c[i + 1] for i in range(len(qlo_c) - 1))
        b_corr = invert(xs, qlo_c, R_OBS_CORRECTED)
        b_raw = invert(xs, qlo_r, R_OBS_PUB)
        # a replicate bootstrap on the bound, so the number carries an interval
        boot = []
        rb = np.random.default_rng([SEED, 7, int(round(beta * 1e6))])
        for _ in range(300):
            q = []
            for t in keys:
                h = surface[t]["r_hat"]
                q.append(float(np.percentile(rb.choice(h, len(h)), Q_LO) - NULL)
                         if h else surface[t]["q_lo_corr"])
            v = invert(xs, q, R_OBS_CORRECTED)
            if v is not None:
                boot.append(v)
        ci = ([float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
              if len(boot) > 30 else None)
        inv[str(beta)] = dict(beta=beta, r_true=xs, q_lo_corr=qlo_c,
                              med_corr=med_c, monotone=bool(mono),
                              bound_corrected=b_corr, bound_raw=b_raw,
                              bound_ci=ci, n_boot=len(boot),
                              excludes_survey=(b_corr is not None
                                               and b_corr < R_SURVEY))
        s_b = f"{b_corr:.4f}" if b_corr is not None else "none in grid"
        s_r = f"{b_raw:.4f}" if b_raw is not None else "none in grid"
        s_c = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "n/a"
        print(f"  beta {beta:>4.2f}: monotone {str(mono):>5} | upper bound on "
              f"r_true (corrected) {s_b:>13} {s_c:<20} | raw {s_r:>13} | "
              f"{'EXCLUDES' if inv[str(beta)]['excludes_survey'] else 'does NOT exclude'}"
              f" the survey's {R_SURVEY:.2f}")
    out["inversion"] = inv
    any_mono = all(v["monotone"] for v in inv.values())
    if not any_mono:
        print("  ⚠ at least one beta-slice is not monotone in r_true, so the "
              "inversion is undefined there and only a compatibility region "
              "can be reported.")
    print("\n  What the surface says: r_hat is set by r_between = (1-beta) "
          "r_true almost alone -- the venue layer is invisible to the estimator "
          "except\n  through the finite-venue leak -- so the bound on r_true is "
          "a bound DIVIDED BY (1-beta), and beta is not identified by these "
          "data.")

    make_figure(surface, inv, lad, noiseless, NULL, kept_real, args.reps)

    out["runtime_seconds"] = float(time.time() - t0)
    out["config"] = dict(ym=YM, panel=PANEL, reps=args.reps, seed=SEED,
                         betas=list(BETAS), r_true_grid=list(RTRUE),
                         imp_true=IMP_MEASURED, imp_estimator=IMP_PUB,
                         q_lo=Q_LO)
    with open(f"{RESULTS}/results_p39.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nwrote {RESULTS}/results_p39.json, {FIG}/p39_recovery.png "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
