#!/usr/bin/env python
"""Phase 35 — the fourth pillar: three matrices, one SEIR, does the age-targeted
allocation change order?

WHY THIS PILLAR EXISTS. The 8-19 letter is explicit about the stakes: the first
three pillars (the estimator hierarchy, the resolution scaling law, the sign flip
under age-bin width) make a measurement paper, and a measurement paper about a
matrix nobody has to use is an AJE or IJE paper. The fourth pillar asks the only
question that makes the measurement consequential: feed the survey matrix, the
passive dong-level matrix and the passive gu-level matrix into the SAME
age-structured SEIR, and see whether the optimal age-targeted allocation of a
fixed intervention budget comes out in a different order.

THE ONE DESIGN DECISION THAT MAKES THIS LEGITIMATE. All three matrices are
normalised to the SAME R0 before anything is integrated. That removes the level
difference entirely -- and the level difference is exactly what directive 1
forbids us to claim as ours, because Di Domenico et al. published it for France
this year. What survives normalisation is structure, and structure is the whole
claim of this paper. A reader who suspects the ranking differences are just "the
passive matrix has fewer contacts" is answered by construction, not by argument.

WHAT IS DELIBERATELY NOT DONE. This is not a fitted model and no calibration to
Seoul case data happens here: the project's scope decision stops at measurement,
and a fitted model would turn the paper into a modelling paper whose
identification conditions we cannot meet. The SEIR is a FUNCTIONAL of the matrix,
used the way a next-generation eigenvalue is used, and the paper has to say so in
those words. Nothing here is a forecast.

FIFTEEN BANDS, NOT SIXTEEN. The survey has no 80+ egos, so every square-matrix
statistic in this project runs on the 0-79 block, and so does this. That also
settles the objective function: with the oldest band absent, a severity- or
death-weighted objective would be weighting the one band whose contacts were
never measured, so the objective is INFECTIONS.

THE CHECK THAT COMES FIRST. Seoul contributes 465 respondents, i.e. 11 to 55 egos
per band, and a ranking estimated from 11 egos may have a confidence interval so
wide that no between-matrix difference could clear it. That is a real possibility
and it would kill the pillar, so section 35.5 measures it BEFORE anything is
built on the ranking: the survey respondent bootstrap is propagated all the way
through to the allocation, and the between-matrix distance is compared with the
within-survey distance. If the second swallows the first, the honest report is
that the pillar cannot be delivered on 465 respondents.

TWO RANKINGS, because they can disagree and the disagreement is informative:

  * the MARGINAL ranking -- the derivative of R0 with respect to a small uniform
    reduction in band a's contacts, per unit of budget. Analytic, from the left
    and right dominant eigenvectors, so it has no solver in it to go wrong.
  * the GREEDY ranking -- spend the budget in small increments on whichever band
    buys the largest reduction in the final epidemic size, integrating the full
    SEIR each time. This is the allocation an actual planner would produce, and
    it accounts for depletion of susceptibles, which the marginal ranking cannot.

Contact reductions are applied as C'[a,a'] = sqrt((1-u_a)(1-u_a')) C[a,a'], which
is the standard symmetric form and is the only one that keeps reciprocity, i.e.
keeps the matrix usable in a transmission model at all.

    python eda/p35_seir.py [--boot 200] [--r0 2.5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import AGE_LABEL, AGES
from paths import FIG, ROOT

SURVEY_MONTHS = [202312, 202402]
SEED = 20260819
# Natural history. Deliberately generic and identical across the three matrices:
# every number below cancels out of a comparison, and the point of the exercise
# is the comparison.
LATENT_DAYS = 3.0
INFECTIOUS_DAYS = 5.0
DT = 0.25
T_END = 720.0
SEED_FRAC = 1e-5

out = {}


# ------------------------------------------------------------- matrix plumbing
def symmetrise(C, pop):
    """T = N_a C[a,a'] symmetrised, then back to C. Reciprocity holds exactly on
    the passive side and only approximately in a survey, so it is imposed here
    once, identically, on all three."""
    T = pop[:, None] * C
    asym = float(np.abs(T - T.T).sum() / T.sum()) if T.sum() else np.nan
    T = (T + T.T) / 2
    return T / pop[:, None], asym


def ngm(C, q=1.0):
    """K = q C / gamma, everyone susceptible. Returns rho and both eigenvectors."""
    K = q * C * INFECTIOUS_DAYS
    w, V = np.linalg.eig(K)
    k = int(np.argmax(w.real))
    r = float(w[k].real)
    right = np.abs(V[:, k].real)
    wl, Vl = np.linalg.eig(K.T)
    kl = int(np.argmax(wl.real))
    left = np.abs(Vl[:, kl].real)
    return r, left / left.sum(), right / right.sum()


def scale_to_r0(C, r0):
    """The susceptibility q that puts this matrix at the target R0. This is the
    step that removes the level difference the letter forbids us to claim."""
    rho, _, _ = ngm(C, 1.0)
    return r0 / rho if rho else np.nan


def reduce_contacts(C, u):
    """C'[a,a'] = sqrt((1-u_a)(1-u_a')) C[a,a'] -- symmetric, reciprocity-safe."""
    f = np.sqrt(np.clip(1.0 - u, 0.0, None))
    return C * np.outer(f, f)


# --------------------------------------------------------------------- the SEIR
def simulate(C, q, pop, u=None, t_end=T_END, dt=DT):
    """Age-structured SEIR, RK4, frequency-dependent force of infection.

    lambda_a = q * sum_a' C[a,a'] * I_a' / N_a'  -- the same frequency-dependent
    convention the matrix was built under in p26. Returns the attack rate by band
    and overall.
    """
    Cr = C if u is None else reduce_contacts(C, u)
    N = pop.astype(float)
    n = len(N)
    sig, gam = 1.0 / LATENT_DAYS, 1.0 / INFECTIOUS_DAYS
    y = np.zeros(3 * n)
    y[:n] = 1.0 - SEED_FRAC                     # S/N
    y[2 * n:] = SEED_FRAC                       # I/N (E starts empty)

    def deriv(y):
        S, E, I = y[:n], y[n:2 * n], y[2 * n:]
        lam = q * (Cr @ I)
        dS = -lam * S
        dE = lam * S - sig * E
        dI = sig * E - gam * I
        return np.concatenate([dS, dE, dI])

    steps = int(t_end / dt)
    for _ in range(steps):
        k1 = deriv(y)
        k2 = deriv(y + dt / 2 * k1)
        k3 = deriv(y + dt / 2 * k2)
        k4 = deriv(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.clip(y, 0.0, 1.0)
    attack = np.clip(1.0 - y[:n], 0.0, 1.0)     # ever-infected share per band
    return attack, float((attack * N).sum() / N.sum())


# -------------------------------------------------------------- the two rankings
def marginal_ranking(C, q, pop):
    """BENEFIT per person-day of reducing band a's contacts, at u = 0.

    Standard eigenvalue perturbation: for K(u) with K'[a,b] =
    sqrt((1-u_a)(1-u_b)) K[a,b], d K/d u_a at u=0 has -K[a,b]/2 in row a and
    -K[b,a]/2 in column a, so

        d rho / d u_a = -(v_a (K w)_a + w_a (v^T K)_a) / (2 v.w)

    with v, w the left and right dominant eigenvectors. Cost of reducing band a
    by du is N_a du person-days.

    SIGN. d rho / d u_a is NEGATIVE for every band -- cutting contacts always
    lowers R0 -- so the band a planner buys first is the one with the LARGEST
    MAGNITUDE, not the largest value. The function therefore returns the negated
    derivative, `benefit`, and every ranking downstream is a plain descending
    sort of it. Returning the raw derivative and sorting it descending picks the
    LEAST useful band while printing a table that looks entirely reasonable,
    which is exactly the class of error this project keeps gating for.
    """
    K = q * C * INFECTIOUS_DAYS
    _, v, w = ngm(C, q)
    denom = float(v @ w)
    d = -(v * (K @ w) + w * (v @ K)) / (2 * denom)
    assert (d <= 1e-12).all(), "d rho / d u should be non-positive for every band"
    return -d / pop, -d


def check_marginal(C, q, pop, du=1e-4):
    """Finite-difference check of marginal_ranking, since the whole pillar rests
    on it. Returns the worst relative disagreement over the bands."""
    benefit, raw = marginal_ranking(C, q, pop)
    rho0, _, _ = ngm(C, q)
    worst = 0.0
    for a in range(len(pop)):
        u = np.zeros(len(pop))
        u[a] = du
        rho1, _, _ = ngm(reduce_contacts(C, u), q)
        fd = -(rho1 - rho0) / du
        if abs(raw[a]) > 0:
            worst = max(worst, abs(fd - raw[a]) / abs(raw[a]))
    return float(worst)


def final_size(C, q, u=None, tol=1e-13, iters=20000):
    """Exact attack rate by band, with no integrator in it.

    Dividing dS_a/dt = -lambda_a S_a by S_a and integrating to infinity gives the
    classical final-size system, because the latent period drops out and
    integral I_b dt is just z_b / gamma:

        x_a = exp( -(q/gamma) * sum_b C[a,b] (1 - x_b) ),   z_a = 1 - x_a

    with x_a the never-infected share of band a. Solved by fixed point from
    x = 0, which converges monotonically from below to the largest root -- the
    epidemic outcome, not the disease-free one.

    This replaces the ODE for every optimisation call. The gain is not only
    speed: DT and T_END stop being tuning knobs that could quietly change an
    allocation. simulate() is kept as the anchor that this agrees with.
    """
    Cr = C if u is None else reduce_contacts(C, u)
    M = (q * INFECTIOUS_DAYS) * Cr
    x = np.zeros(len(M))
    for _ in range(iters):
        xn = np.exp(-M @ (1.0 - x))
        if np.max(np.abs(xn - x)) < tol:
            x = xn
            break
        x = xn
    return np.clip(1.0 - x, 0.0, 1.0)


def optimal_allocation(C, q, pop, budget_frac=0.10, u_max=0.9, seeds=None):
    """Minimise total infections subject to sum_a N_a u_a <= budget.

    Fifteen decision variables and a smooth objective, so this is a small
    constrained problem rather than a search: SLSQP from several starts, best
    kept. A myopic greedy was tried first and rejected -- it produced NEGATIVE
    regret in the cross-application check, i.e. the plan derived from the other
    matrix beat the one it called optimal, which is impossible for a real
    optimum and is a property of the heuristic rather than of the matrices.
    """
    from scipy.optimize import minimize
    n = len(pop)
    B = budget_frac * pop.sum()

    def obj(u):
        return float((final_size(C, q, u) * pop).sum() / pop.sum())

    cons = [{"type": "ineq", "fun": lambda u: B - float(pop @ u)}]
    bnds = [(0.0, u_max)] * n
    starts = [np.full(n, min(budget_frac, u_max))]
    ben, _ = marginal_ranking(C, q, pop)
    w_ = np.clip(ben, 0, None)
    if w_.sum() > 0:                       # spend it all where the margin is best
        starts.append(np.clip(B * w_ / (w_ @ pop), 0, u_max))
    top = int(np.argmax(ben))              # and the corner solution
    corner = np.zeros(n)
    corner[top] = min(u_max, B / pop[top])
    starts.append(corner)
    starts += list(seeds or [])
    best_u, best_v = None, np.inf
    for u0 in starts:
        r = minimize(obj, np.clip(u0, 0, u_max), method="SLSQP", bounds=bnds,
                     constraints=cons, options={"maxiter": 120, "ftol": 1e-10})
        u = np.clip(r.x, 0, u_max)
        if pop @ u > B * 1.000001:         # project back onto the budget
            u *= B / (pop @ u)
        v = obj(u)
        if v < best_v:
            best_u, best_v = u, v
    return best_u, best_v, best_u * pop


def kendall_tau(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            c += s > 0
            d += s < 0
    return float((c - d) / (0.5 * n * (n - 1))) if n > 1 else np.nan


def top_k(score, k=3):
    return [int(i) for i in np.argsort(score)[::-1][:k]]


# ------------------------------------- the retreats, declared before being run
# The 8-19 letter fixes the trigger, both merge maps, the top-weighted
# statistics and the reporting rule IN ADVANCE, so that a result which does not
# flatter the paper cannot be read as a choice made after seeing it. Nothing
# here replaces the fifteen-band block: every retreat is an additional row.
#
#   trigger  the fifteen-band within-survey bootstrap tau's 95% lower bound
#            falls at or below the between-matrix tau. 202402 tripped it.
#   A        nine bands, adults merged into decades, the school bands left
#            alone. Declared with its limit already measured: it does NOT widen
#            the smallest cell, because the bottleneck is 15-19 at 11 egos.
#   B        eight bands, 10-19 merged as well. The only map that lifts the
#            minimum (11 -> 22 egos), and the one expected to move the answer
#            AGAINST us, since the survey's top two bands and the passive top
#            band all fall into the merged one.
#   C        top-weighted statistics instead of the full-ranking tau, because a
#            budget only buys the first few bands.
RETREAT_SCHEMES = {
    "identity15": None,                                  # the 15-band baseline
    "A9": [["0-9"], ["10-14"], ["15-19"], ["20-24", "25-29"],
           ["30-34", "35-39"], ["40-44", "45-49"], ["50-54", "55-59"],
           ["60-64", "65-69"], ["70-74", "75-79"]],
    "B8": [["0-9"], ["10-14", "15-19"], ["20-24", "25-29"],
           ["30-34", "35-39"], ["40-44", "45-49"], ["50-54", "55-59"],
           ["60-64", "65-69"], ["70-74", "75-79"]],
}


def merge_groups(groups, lbl):
    """Label groups -> index groups, asserted to be a partition of the bands.

    A map that drops or double-counts a band would still produce a matrix and a
    ranking, and the ranking would look entirely reasonable.
    """
    idx = [[lbl.index(b) for b in g] for g in groups]
    flat = sorted(i for g in idx for i in g)
    assert flat == list(range(len(lbl))), "merge map is not a partition"
    return idx


def merge_matrix(C, pop, idx):
    """Population-weighted on the ego side, summed on the alter side:

        C'[A,B] = sum_{a in A} (N_a / N_A) * sum_{b in B} C[a,b]

    This is the aggregation that conserves contacts: N_A * sum_B C'[A,B] equals
    sum_{a in A} N_a sum_b C[a,b] exactly, which 35.6 checks rather than
    asserts in a comment. Any other weighting silently invents or destroys
    contacts and would change R0 for reasons that have nothing to do with the
    merge.
    """
    n = len(idx)
    popm = np.array([pop[g].sum() for g in idx], float)
    Cm = np.zeros((n, n))
    for A, ga in enumerate(idx):
        w = pop[ga] / pop[ga].sum()
        for B, gb in enumerate(idx):
            Cm[A, B] = float(w @ C[np.ix_(ga, gb)].sum(1))
    return Cm, popm


def kendall_tau_w(x, y, flat=False):
    """Top-weighted Kendall tau -- Vigna's (2015) additive hyperbolic weighting,
    symmetrised over the two rankings.

    Each band carries weight 1/(1 + its rank) within a ranking; a pair carries
    the sum of its two bands' weights; the weight actually used is the average
    of what the two rankings assign that pair, which makes the statistic
    symmetric in its arguments. `flat=True` sets every weight to one, which must
    reproduce kendall_tau exactly -- checked in 35.6, because a weighting scheme
    that quietly fails to reduce to the unweighted case is indistinguishable
    from a bug.

    WHY IT IS THE RIGHT STATISTIC HERE. The budget buys the first few bands, so
    swapping 45-49 with 50-54 is not a planning error; the plain tau charges it
    exactly what it charges a swap at the top.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)

    def hw(v):
        r = np.empty(n)
        r[np.argsort(v)[::-1]] = np.arange(n)
        return 1.0 / (1.0 + r)

    wx, wy = (np.ones(n), np.ones(n)) if flat else (hw(x), hw(y))
    num = den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = 0.5 * ((wx[i] + wx[j]) + (wy[i] + wy[j]))
            num += w * np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            den += w
    return float(num / den) if den else np.nan


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--r0", type=float, default=2.5)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--r0-sweep", default="1.3,1.8,2.5",
                    help="R0 values for the final-size allocation (35.3). The "
                         "marginal ranking in 35.1 is R0-invariant.")
    ap.add_argument("--skip-greedy", action="store_true",
                    help="skip 35.3, the part that needs the optimiser")
    args = ap.parse_args()
    args.r0_sweep = [float(x) for x in args.r0_sweep.split(",")]
    rng = np.random.default_rng(SEED)

    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]
    lbl = [AGE_LABEL[AGES[i]] for i in sq]
    out["bands"] = lbl
    out["settings"] = dict(r0=args.r0, r0_sweep=args.r0_sweep,
                           latent_days=LATENT_DAYS,
                           infectious_days=INFECTIOUS_DAYS, dt=DT, t_end=T_END,
                           budget_frac=args.budget, n_boot=args.boot)

    print(f"=== 35.0 setup: {len(sq)} bands, R0 fixed at {args.r0} for all "
          f"three matrices ===")

    # ------------------------------------------------------- the three matrices
    mats = {}
    for ym in SURVEY_MONTHS:
        pop = pops[ym][sq]
        for tag, key in (("passive_dong", f"{ym}|WE|dong|holidayfree"),
                         ("passive_gu", f"{ym}|WE|gu")):
            if key not in p26["matrices"]:
                continue
            A = np.array(p26["matrices"][key]["A"])[np.ix_(sq, sq)]
            C, asym = symmetrise(A / pops[ym][sq][:, None], pop)
            mats[(ym, tag)] = (C, asym)
        sk = f"{ym}|WE|seoul"
        Cs = np.array(p27["survey"][sk]["C"])[np.ix_(sq, sq)]
        C, asym = symmetrise(Cs, pop)
        mats[(ym, "survey")] = (C, asym)

    print(f"  {'month':>7} {'matrix':>13} {'mean contacts/day':>18} "
          f"{'rho(C)':>8} {'q for R0':>9} {'asymmetry':>10}")
    scaled = {}
    for (ym, tag), (C, asym) in sorted(mats.items()):
        q = scale_to_r0(C, args.r0)
        rho, _, _ = ngm(C, 1.0)
        scaled[(ym, tag)] = (C, q)
        print(f"  {ym:>7} {tag:>13} {C.sum(1).mean():>18.3f} {rho:>8.3f} "
              f"{q:>9.4f} {asym:>10.4f}")
    out["matrices"] = {f"{ym}|{tag}": dict(
        mean_contacts=float(C.sum(1).mean()), rho_unscaled=float(ngm(C, 1.0)[0]),
        q_for_r0=float(q), asymmetry=float(mats[(ym, tag)][1]))
        for (ym, tag), (C, q) in scaled.items()}

    # ---------------------------------------------------- 35.1 marginal ranking
    print(f"\n=== 35.1 marginal ranking: R0 bought per person-day spent ===")
    marg, fdchk = {}, []
    for ym in SURVEY_MONTHS:
        print(f"\n  {ym}  {'band':>6} " + " ".join(f"{t:>14}" for t in
              ("survey", "passive_dong", "passive_gu")))
        sc = {}
        for tag in ("survey", "passive_dong", "passive_gu"):
            if (ym, tag) not in scaled:
                continue
            C, q = scaled[(ym, tag)]
            per_cost, _ = marginal_ranking(C, q, pops[ym][sq])
            fd = check_marginal(C, q, pops[ym][sq])
            assert fd < 1e-3, (f"analytic marginal disagrees with the finite "
                               f"difference by {fd:.2e} for {ym}|{tag}")
            fdchk.append(dict(ym=ym, matrix=tag, max_rel_err=fd))
            sc[tag] = per_cost
        for i, b in enumerate(lbl):
            print(f"       {b:>6} " + " ".join(
                f"{sc[t][i]:>14.3e}" for t in sc))
        marg[str(ym)] = {t: v.tolist() for t, v in sc.items()}
        print(f"       {'buy 1st':>6} " + " ".join(
            f"{lbl[int(np.argmax(sc[t]))]:>14}" for t in sc))
    out["marginal"] = marg
    out["marginal_finite_difference_check"] = fdchk
    print(f"\n  anchor: analytic vs finite-difference marginal, worst relative "
          f"error over {len(fdchk)} matrices = "
          f"{max(r['max_rel_err'] for r in fdchk):.2e}")

    # ------------------------------------------------- 35.2 do the orders differ
    print("\n=== 35.2 Kendall tau between matrices, and the top three ===")
    pairs = [("survey", "passive_dong"), ("survey", "passive_gu"),
             ("passive_dong", "passive_gu")]
    cmp_ = {}
    for ym in SURVEY_MONTHS:
        sc = {t: np.array(v) for t, v in marg[str(ym)].items()}
        rows = []
        for a, b in pairs:
            if a not in sc or b not in sc:
                continue
            ta, tb = top_k(sc[a]), top_k(sc[b])
            rows.append(dict(pair=f"{a} vs {b}", tau=kendall_tau(sc[a], sc[b]),
                             top3_a=[lbl[i] for i in ta],
                             top3_b=[lbl[i] for i in tb],
                             top3_overlap=len(set(ta) & set(tb)),
                             rank1_same=bool(ta[0] == tb[0])))
        cmp_[str(ym)] = rows
        print(f"\n  {ym}")
        for r_ in rows:
            print(f"    {r_['pair']:>28}  tau {r_['tau']:>+6.3f}  "
                  f"top3 overlap {r_['top3_overlap']}/3  "
                  f"rank1 {'same' if r_['rank1_same'] else 'DIFFERENT'}")
            print(f"    {'':>28}  {', '.join(r_['top3_a'])}  |  "
                  f"{', '.join(r_['top3_b'])}")
    out["comparison_marginal"] = cmp_

    # --------------------------------------------------- 35.3 optimal allocation
    greedy = {}
    if not args.skip_greedy:
        # The ODE and the closed form have to agree before either is used. If
        # they do not, one of them is wrong and the allocation built on top would
        # be wrong by a plausible-looking amount.
        Cc, qc = scaled[(SURVEY_MONTHS[0], "passive_dong")]
        popc = pops[SURVEY_MONTHS[0]][sq]
        att_ode, _ = simulate(Cc, qc, popc)
        att_fs = final_size(Cc, qc)
        worst = float(np.max(np.abs(att_ode - att_fs)))
        print(f"\n=== 35.3 optimal allocation of a budget worth "
              f"{args.budget:.0%} of population-days ===")
        print(f"  anchor: closed-form final size vs the RK4 SEIR, worst band "
              f"disagreement {worst:.2e}")
        assert worst < 5e-3, "final-size solve disagrees with the integrator"
        out["final_size_anchor"] = dict(max_abs_diff=worst)

        # One R0 is not enough. At R0 = 2.5 these matrices put the attack rate
        # above 0.8, and in that regime almost any allocation looks alike --
        # a comparison made only there would understate the difference and would
        # not be the regime a planner is ever in. The ranking from 35.1 is
        # R0-invariant by construction (q enters linearly), so the sweep is a
        # statement about the FINAL-SIZE objective only.
        for r0 in args.r0_sweep:
            print(f"\n  --- R0 = {r0} ---")
            for ym in SURVEY_MONTHS:
                got = {}
                for tag in ("survey", "passive_dong", "passive_gu"):
                    if (ym, tag) not in mats:
                        continue
                    C = mats[(ym, tag)][0]
                    q = scale_to_r0(C, r0)
                    pop = pops[ym][sq]
                    u, val, spend = optimal_allocation(C, q, pop,
                                                       budget_frac=args.budget)
                    z0 = float((final_size(C, q) * pop).sum() / pop.sum())
                    got[tag] = dict(u=u.tolist(), spend=spend.tolist(),
                                    attack_no_intervention=z0, attack_with=val,
                                    averted_frac=float((z0 - val) / z0),
                                    top_spend=[lbl[i] for i in
                                               np.argsort(spend)[::-1][:3]])
                    print(f"    {ym} {tag:>13}: spends on "
                          f"{', '.join(got[tag]['top_spend'])};  attack "
                          f"{z0:.4f} -> {val:.4f} "
                          f"({got[tag]['averted_frac']:.1%} averted)")
                # The consequence sentence: run each matrix's plan in the other
                # matrix's world. Reported as achieved attack rate rather than as
                # a share of benefit, so a suboptimal solve cannot manufacture a
                # negative number.
                if "survey" in got and "passive_dong" in got:
                    Cs = mats[(ym, "survey")][0]
                    qs = scale_to_r0(Cs, r0)
                    pop = pops[ym][sq]
                    u_p = np.array(got["passive_dong"]["u"])
                    z_p = float((final_size(Cs, qs, u_p) * pop).sum() / pop.sum())
                    z_s = got["survey"]["attack_with"]
                    z0 = got["survey"]["attack_no_intervention"]
                    got["cross_application"] = dict(
                        attack_survey_plan=z_s, attack_passive_plan=z_p,
                        excess_attack_pp=float(100 * (z_p - z_s)),
                        regret_share_of_benefit=float((z_p - z_s) / (z0 - z_s))
                        if z0 > z_s else np.nan)
                    print(f"    {ym} {'cross-check':>13}: the passive-derived "
                          f"plan, spent in the survey world, leaves attack "
                          f"{z_p:.4f} against {z_s:.4f} -- "
                          f"{100 * (z_p - z_s):+.2f} pp, i.e. "
                          f"{(z_p - z_s) / (z0 - z_s):+.1%} of the achievable "
                          f"benefit given up")
                sc = {t: np.array(got[t]["spend"]) for t in got
                      if t != "cross_application"}
                for a, b in pairs:
                    if a in sc and b in sc:
                        print(f"    {ym} tau({a}, {b}) on spend = "
                              f"{kendall_tau(sc[a], sc[b]):+.3f}")
                greedy[f"{ym}|R0={r0}"] = got
    out["allocation"] = greedy

    # ------------------------------------ 35.4 is 465 respondents enough for this
    # The check that had to come first. If the survey's own sampling noise moves
    # the ranking more than the choice of matrix does, the pillar cannot be
    # delivered on this survey and the honest thing is to say so.
    print(f"\n=== 35.4 does the ranking survive 465 respondents? "
          f"({args.boot} bootstraps) ===")
    boot_out = {}
    try:
        from p27_survey import (contact_matrix, ego_cubes, holiday_free_dows,
                                load_survey)
        ego, d = load_survey()
        d = d[d.panel.isin(["W", "E"])]
        for ym, scope, seoul_only in [(y, "seoul", True) for y in SURVEY_MONTHS] \
                                     + [(y, "national", False) for y in SURVEY_MONTHS]:
            cube, eb, nd = ego_cubes(ego, d, seoul_only, ym=ym,
                                     dows=holiday_free_dows(ym))
            pop = pops[ym][sq]
            key = str(ym) if scope == "seoul" else f"{ym}|national"
            # The national arm exists to answer one question: is the pillar
            # blocked by the METHOD or by the 465 Seoul respondents? It is a
            # power statement, not a second result -- the passive matrix is
            # Seoul-only, so a national survey is not a valid comparator, only a
            # valid way to see how the interval shrinks with n.
            #
            # Each arm is resampled against ITS OWN point estimate. Comparing
            # national bootstrap draws with the Seoul point ranking would measure
            # the Seoul/national difference with bootstrap noise painted on top,
            # and would report it as a sampling interval.
            if scope == "seoul":
                base = np.array(marg[str(ym)]["survey"])
            else:
                Cn, _ = contact_matrix(cube, eb, nd)
                Cn, _ = symmetrise(Cn[np.ix_(sq, sq)], pop)
                base, _ = marginal_ranking(Cn, scale_to_r0(Cn, args.r0), pop)
            taus, top1, top3 = [], [], []
            for _ in range(args.boot):
                idx = rng.integers(0, len(eb), len(eb))
                C, _ = contact_matrix(cube[idx], eb[idx], nd)
                C, _ = symmetrise(C[np.ix_(sq, sq)], pop)
                q = scale_to_r0(C, args.r0)
                if not np.isfinite(q):
                    continue
                s, _ = marginal_ranking(C, q, pop)
                taus.append(kendall_tau(base, s))
                top1.append(int(np.argmax(s)))
                top3.append(top_k(s))
            taus = np.array(taus)
            n_ego = np.bincount(eb, minlength=len(AGES))[sq]
            base_top1 = int(np.argmax(base))
            same1 = float(np.mean([t == base_top1 for t in top1]))
            base_t3 = set(top_k(base))
            ov = float(np.mean([len(set(t) & base_t3) for t in top3]))
            # cmp_ holds the Seoul-vs-passive taus. They are the thing the
            # Seoul arm has to clear; for the national arm they are carried only
            # so the two intervals can be read on one axis, and the verdict is
            # explicitly labelled as a power statement rather than a result.
            between = [r_["tau"] for r_ in cmp_[str(ym)]
                       if r_["pair"].startswith("survey vs")]
            boot_out[key] = dict(
                n_ego_per_band=n_ego.tolist(), n_ego_min=int(n_ego.min()),
                n_ego_max=int(n_ego.max()),
                tau_within_survey=[float(np.percentile(taus, 2.5)),
                                   float(np.percentile(taus, 97.5))],
                tau_within_median=float(np.median(taus)),
                tau_between_matrices=between,
                top1_stability=same1, top3_mean_overlap=ov,
                scope=scope, n_respondents=int(len(eb)),
                verdict=("separable" if between and
                         max(between) < np.percentile(taus, 2.5)
                         else "NOT separable"))
            b = boot_out[key]
            print(f"\n  {ym} [{scope}, {len(eb)} respondents]: "
                  f"{n_ego.min()}-{n_ego.max()} egos per band")
            print(f"    within-survey tau (respondent bootstrap): median "
                  f"{b['tau_within_median']:+.3f}, 95% "
                  f"[{b['tau_within_survey'][0]:+.3f}, "
                  f"{b['tau_within_survey'][1]:+.3f}]")
            print(f"    between-matrix tau: "
                  f"{', '.join(f'{t:+.3f}' for t in between)}")
            print(f"    the survey's own top band survives resampling "
                  f"{same1:.0%} of the time; mean top-3 overlap {ov:.2f}/3")
            print(f"    VERDICT{' (power check only)' if scope == 'national' else ''}"
                  f": the between-matrix difference is "
                  f"{'LARGER than' if b['verdict'] == 'separable' else 'INSIDE'}"
                  f" this arm's own sampling noise -> {b['verdict']}")
            # 35.5 The sharper version of the same question. The ranking is one
            # summary of the allocation; the quantity a planner actually loses is
            # the regret, and it has its own sampling distribution. Each
            # bootstrap survey world is re-optimised and the FIXED passive plan
            # is evaluated in it, so the interval below is "how much the passive
            # matrix costs you, given that the survey itself is only 465 people".
            if not args.skip_greedy and str(ym) in {k.split("|")[0]
                                                    for k in greedy}:
                r0 = args.r0_sweep[-1]
                gk = f"{ym}|R0={r0}"
                if gk in greedy and "passive_dong" in greedy[gk]:
                    u_p = np.array(greedy[gk]["passive_dong"]["u"])
                    regs, nulls = [], []
                    for _ in range(args.boot):
                        idx = rng.integers(0, len(eb), len(eb))
                        Cb, _ = contact_matrix(cube[idx], eb[idx], nd)
                        Cb, _ = symmetrise(Cb[np.ix_(sq, sq)], pop)
                        qb = scale_to_r0(Cb, r0)
                        if not np.isfinite(qb):
                            continue
                        ub, vb, _ = optimal_allocation(Cb, qb, pop,
                                                       budget_frac=args.budget)
                        z0 = float((final_size(Cb, qb) * pop).sum() / pop.sum())
                        zp = float((final_size(Cb, qb, u_p) * pop).sum()
                                   / pop.sum())
                        regs.append((zp - vb) / (z0 - vb) if z0 > vb else np.nan)
                        # the null this has to clear: a plan built from ANOTHER
                        # draw of the same survey, evaluated the same way. Any
                        # optimiser has some slack, and that slack must not be
                        # read as the cost of using passive data.
                        jdx = rng.integers(0, len(eb), len(eb))
                        Cj, _ = contact_matrix(cube[jdx], eb[jdx], nd)
                        Cj, _ = symmetrise(Cj[np.ix_(sq, sq)], pop)
                        qj = scale_to_r0(Cj, r0)
                        uj, _, _ = optimal_allocation(Cj, qj, pop,
                                                      budget_frac=args.budget)
                        zj = float((final_size(Cb, qb, uj) * pop).sum()
                                   / pop.sum())
                        nulls.append((zj - vb) / (z0 - vb) if z0 > vb else np.nan)
                    regs = np.array([r for r in regs if np.isfinite(r)])
                    nulls = np.array([r for r in nulls if np.isfinite(r)])
                    boot_out[key]["regret"] = dict(
                        r0=r0, median=float(np.median(regs)),
                        ci=[float(np.percentile(regs, 2.5)),
                            float(np.percentile(regs, 97.5))],
                        null_median=float(np.median(nulls)),
                        null_ci=[float(np.percentile(nulls, 2.5)),
                                 float(np.percentile(nulls, 97.5))],
                        exceeds_null=bool(np.percentile(regs, 2.5)
                                          > np.percentile(nulls, 97.5)))
                    b2 = boot_out[key]["regret"]
                    print(f"    regret of the passive plan at R0={r0}: median "
                          f"{b2['median']:.1%}, 95% "
                          f"[{b2['ci'][0]:.1%}, {b2['ci'][1]:.1%}]")
                    print(f"    the same measured for a SECOND survey draw "
                          f"(the null): median {b2['null_median']:.1%}, 95% "
                          f"[{b2['null_ci'][0]:.1%}, {b2['null_ci'][1]:.1%}]")
                    print(f"    -> the cost of using passive data "
                          f"{'EXCEEDS' if b2['exceeds_null'] else 'does NOT exceed'}"
                          f" the cost of re-running the survey on "
                          f"{len(eb)} new people")
    except Exception as exc:                                     # noqa: BLE001
        print(f"  skipped: {exc}")
        boot_out = {"status": f"skipped: {exc}"}
    out["survey_bootstrap"] = boot_out

    # --------------------------------------- 35.6 the retreats declared in 8-19
    # Declared in the letter BEFORE this ran; see RETREAT_SCHEMES. Everything
    # here is ADDITIONAL -- 35.1-35.5 above are untouched, and this section uses
    # its own random stream so that they come out byte-identical whether or not
    # this section runs.
    print("\n=== 35.6 declared retreats: A (9 bands), B (8 bands), "
          "C (top-weighted statistics) ===")
    retreats = {"declaration": dict(
        trigger="15-band within-survey bootstrap tau 95% lower bound <= "
                "between-matrix tau (202402 tripped it)",
        maps={k: v for k, v in RETREAT_SCHEMES.items() if v},
        merge="ego side population-weighted, alter side summed",
        reporting_rule="additional to the 15-band block, never replacing it")}
    try:
        from p27_survey import (contact_matrix, ego_cubes, holiday_free_dows,
                                load_survey)
        rng6 = np.random.default_rng(SEED + 6)
        ego, d = load_survey()
        d = d[d.panel.isin(["W", "E"])]
        schemes = {k: merge_groups(v or [[b] for b in lbl], lbl)
                   for k, v in RETREAT_SCHEMES.items()}
        scheme_lbl = {k: ["-".join([lbl[g[0]].split("-")[0],
                                    lbl[g[-1]].split("-")[-1]])
                          for g in gs] for k, gs in schemes.items()}

        # --- three anchors, before any retreat number is read
        C0, _ = scaled[(SURVEY_MONTHS[0], "passive_dong")]
        pop0 = pops[SURVEY_MONTHS[0]][sq]
        Ci, popi = merge_matrix(C0, pop0, schemes["identity15"])
        a_identity = float(np.abs(Ci - C0).max())
        a_pop = float(np.abs(popi - pop0).max())
        a_cons = 0.0
        for g in schemes.values():
            Cm, popm = merge_matrix(C0, pop0, g)
            tot0 = float(pop0 @ C0.sum(1))
            a_cons = max(a_cons, abs(float(popm @ Cm.sum(1)) - tot0) / tot0)
            assert abs(popm.sum() - pop0.sum()) / pop0.sum() < 1e-12
        u_, v_ = rng6.normal(size=len(sq)), rng6.normal(size=len(sq))
        a_flat = float(abs(kendall_tau_w(u_, v_, flat=True) - kendall_tau(u_, v_)))
        assert a_identity < 1e-12, "the identity map does not reproduce C"
        assert a_pop < 1e-9, "the identity map does not reproduce the population"
        assert a_cons < 1e-12, "the merge does not conserve contacts"
        assert a_flat < 1e-12, "flat-weighted tau_w is not the plain tau"
        retreats["anchors"] = dict(identity_max_abs_diff=a_identity,
                                   identity_pop_max_abs_diff=a_pop,
                                   contacts_conserved_max_rel=a_cons,
                                   tau_w_flat_equals_tau=a_flat)
        print(f"  anchors: identity map reproduces C to {a_identity:.2e}, "
              f"merge conserves contacts to {a_cons:.2e}, flat tau_w equals "
              f"tau to {a_flat:.2e}")

        for ym in SURVEY_MONTHS:
            pop = pops[ym][sq]
            for scope, seoul_only in (("seoul", True), ("national", False)):
                cube, eb, nd = ego_cubes(ego, d, seoul_only, ym=ym,
                                         dows=holiday_free_dows(ym))
                n_ego = np.bincount(eb, minlength=len(AGES))[sq]
                if scope == "seoul":
                    arms = {"survey": mats[(ym, "survey")][0]}
                    for tag in ("passive_dong", "passive_gu"):
                        if (ym, tag) in mats:
                            arms[tag] = mats[(ym, tag)][0]
                else:
                    # power reference only: the passive matrix is Seoul-only, so
                    # a national survey is not a valid comparator -- it is here
                    # to show how the intervals move with n, nothing else.
                    Cn, _ = contact_matrix(cube, eb, nd)
                    Cn, _ = symmetrise(Cn[np.ix_(sq, sq)], pop)
                    arms = {"survey": Cn}

                point, between = {}, {}
                for sname, g in schemes.items():
                    sc = {}
                    for tag, C in arms.items():
                        Cm, popm = merge_matrix(C, pop, g)
                        ben, _ = marginal_ranking(Cm, scale_to_r0(Cm, args.r0),
                                                  popm)
                        sc[tag] = ben
                    point[sname] = sc
                    rows = []
                    for a, b in pairs:
                        if a in sc and b in sc:
                            ta, tb = top_k(sc[a]), top_k(sc[b])
                            rows.append(dict(
                                pair=f"{a} vs {b}",
                                tau=kendall_tau(sc[a], sc[b]),
                                tau_w=kendall_tau_w(sc[a], sc[b]),
                                top3_overlap=len(set(ta) & set(tb)),
                                rank1_same=bool(ta[0] == tb[0]),
                                top3_a=[scheme_lbl[sname][i] for i in ta],
                                top3_b=[scheme_lbl[sname][i] for i in tb]))
                    between[sname] = rows

                # One draw feeds every scheme, so the schemes are compared on
                # the SAME resampled surveys rather than on independent ones --
                # otherwise a difference between A and B could be draw noise.
                acc = {k: dict(tau=[], tau_w=[], top1=[], top3=[])
                       for k in schemes}
                for _ in range(args.boot):
                    idx = rng6.integers(0, len(eb), len(eb))
                    Cb, _ = contact_matrix(cube[idx], eb[idx], nd)
                    Cb, _ = symmetrise(Cb[np.ix_(sq, sq)], pop)
                    for sname, g in schemes.items():
                        Cm, popm = merge_matrix(Cb, pop, g)
                        qm = scale_to_r0(Cm, args.r0)
                        if not np.isfinite(qm):
                            continue
                        s_, _ = marginal_ranking(Cm, qm, popm)
                        base = point[sname]["survey"]
                        acc[sname]["tau"].append(kendall_tau(base, s_))
                        acc[sname]["tau_w"].append(kendall_tau_w(base, s_))
                        acc[sname]["top1"].append(int(np.argmax(s_)))
                        acc[sname]["top3"].append(top_k(s_))

                print(f"\n  {ym} [{scope}, {len(eb)} respondents]")
                for sname, g in schemes.items():
                    t = np.array(acc[sname]["tau"])
                    tw = np.array(acc[sname]["tau_w"])
                    base = point[sname]["survey"]
                    b1 = int(np.argmax(base))
                    b3 = set(top_k(base))
                    merged_ego = [int(n_ego[gg].sum()) for gg in g]
                    bt = [r_["tau"] for r_ in between[sname]
                          if r_["pair"].startswith("survey vs")]
                    btw = [r_["tau_w"] for r_ in between[sname]
                           if r_["pair"].startswith("survey vs")]
                    lo, low = float(np.percentile(t, 2.5)), float(np.percentile(tw, 2.5))
                    ent = dict(
                        scheme=sname, scope=scope, n_bands=len(g),
                        bands=scheme_lbl[sname],
                        n_ego_per_band=merged_ego,
                        n_ego_min=int(min(merged_ego)),
                        n_ego_max=int(max(merged_ego)),
                        n_respondents=int(len(eb)),
                        between=between[sname],
                        tau_within=[lo, float(np.percentile(t, 97.5))],
                        tau_within_median=float(np.median(t)),
                        tau_w_within=[low, float(np.percentile(tw, 97.5))],
                        tau_w_within_median=float(np.median(tw)),
                        top1_stability=float(np.mean(
                            [i == b1 for i in acc[sname]["top1"]])),
                        top3_mean_overlap=float(np.mean(
                            [len(set(v) & b3) for v in acc[sname]["top3"]])),
                        verdict_tau=("separable" if bt and max(bt) < lo
                                     else "NOT separable") if scope == "seoul"
                        else "power reference",
                        verdict_tau_w=("separable" if btw and max(btw) < low
                                       else "NOT separable") if scope == "seoul"
                        else "power reference")
                    retreats[f"{ym}|{scope}|{sname}"] = ent
                    print(f"    {sname:>10} {len(g):>2} bands, egos "
                          f"{ent['n_ego_min']:>3}-{ent['n_ego_max']:>3}  "
                          f"within tau {ent['tau_within_median']:+.3f} "
                          f"[{lo:+.3f}, {ent['tau_within'][1]:+.3f}]  "
                          f"tau_w {ent['tau_w_within_median']:+.3f} "
                          f"[{low:+.3f}, {ent['tau_w_within'][1]:+.3f}]  "
                          f"top1 {ent['top1_stability']:.3f}  "
                          f"ov3 {ent['top3_mean_overlap']:.2f}")
                    if scope == "seoul":
                        print(f"    {'':>10} between: tau "
                              f"{', '.join(f'{x:+.3f}' for x in bt)}  "
                              f"tau_w {', '.join(f'{x:+.3f}' for x in btw)}"
                              f"  -> tau {ent['verdict_tau']}, "
                              f"tau_w {ent['verdict_tau_w']}")
    except Exception as exc:                                     # noqa: BLE001
        print(f"  skipped: {exc}")
        retreats["status"] = f"skipped: {exc}"
    out["retreats"] = retreats

    # ------------------------------------------------------------------ figure
    ym0 = SURVEY_MONTHS[0]
    sc = {t: np.array(v) for t, v in marg[str(ym0)].items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for t, c in (("survey", "#A8434E"), ("passive_dong", "#0E7C86"),
                 ("passive_gu", "#4C6E8A")):
        if t in sc:
            v = sc[t] / np.abs(sc[t]).max()
            ax.plot(range(len(sq)), v, "o-", c=c, label=t)
    ax.set_xticks(range(len(sq)))
    ax.set_xticklabels(lbl, rotation=90, fontsize=6)
    ax.set_ylabel("$R_0$ bought per person-day\n(scaled to its own max)")
    ax.legend(fontsize=7)
    ax.set_title(f"{ym0}: where a planner would spend first", fontsize=9)
    ax = axes[1]
    gk = f"{ym0}|R0={args.r0_sweep[-1]}" if greedy else None
    if gk and greedy.get(gk):
        g = greedy[gk]
        wid = 0.27
        for k, (t, c) in enumerate((("survey", "#A8434E"),
                                    ("passive_dong", "#0E7C86"),
                                    ("passive_gu", "#4C6E8A"))):
            if t in g:
                ax.bar(np.arange(len(sq)) + (k - 1) * wid,
                       np.array(g[t]["spend"]) / sum(g[t]["spend"]),
                       width=wid, color=c, label=t)
        ax.set_xticks(range(len(sq)))
        ax.set_xticklabels(lbl, rotation=90, fontsize=6)
        ax.set_ylabel("share of budget")
        ax.legend(fontsize=7)
        ax.set_title(f"optimal allocation of the same budget "
                     f"($R_0$={args.r0_sweep[-1]})", fontsize=9)
    ax = axes[2]
    if isinstance(boot_out.get(str(ym0)), dict) and "tau_within_median" in boot_out[str(ym0)]:
        b = boot_out[str(ym0)]
        ax.axhspan(b["tau_within_survey"][0], b["tau_within_survey"][1],
                   color="#A8434E", alpha=0.18,
                   label="survey resampled against itself (95%)")
        for i, t in enumerate(b["tau_between_matrices"]):
            ax.plot([i], [t], "o", c="#0E7C86", ms=9)
        ax.axhline(b["tau_within_median"], ls=":", c="#A8434E")
        ax.set_ylabel(r"Kendall $\tau$ of the allocation ranking")
        ax.set_xticks(range(len(b["tau_between_matrices"])))
        ax.set_xticklabels(["vs passive\ndong", "vs passive\ngu"][
            :len(b["tau_between_matrices"])], fontsize=7)
        ax.legend(fontsize=7)
        ax.set_title("is the difference bigger than the noise?", fontsize=9)
    fig.suptitle("Phase 35 — three matrices, one SEIR, same $R_0$: does the "
                 "age-targeted allocation change order?", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p35_seir.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p35_seir.png")

    with open(f"{ROOT}/eda/results_p35.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p35.json")


if __name__ == "__main__":
    main()
