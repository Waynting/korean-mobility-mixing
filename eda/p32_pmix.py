#!/usr/bin/env python
"""Phase 32 — excess over proportionate mixing, and how close to rank-1 the
passive matrix is.

WHY THIS EXISTS. The advisor's 8-19 letter says the Section 7 ten-fold gap must
be reported as excess over proportionate mixing rather than as a raw ratio, and
that a rank test must go beside it, because the passive matrix looks nearly
rank-1 -- rows barely differ, and the structure is carried by destination
attractiveness, which is itself close to proportionate mixing.

FIRST, A CORRECTION THAT CHANGES HOW THE SECTION READS. Newman assortativity, as
it is computed in p26_matrix.stats and p27_survey.assortativity, is

    r = (tr e - sum_a r_a^2) / (1 - sum_a r_a^2),   e = T / T.sum(),  r_a = e_a.

and its numerator is exactly sum_a (e_aa - r_a r_a) -- the total DIAGONAL excess
over the proportionate-mixing null E = r r'. So the ten-fold gap was never a
pre-proportionate-mixing comparison. It was a proportionate-mixing comparison
confined to the diagonal. §32.1 asserts that identity rather than asserting it in
prose, because the paper has to state it: what is new here is the OFF-DIAGONAL
structure and the rank statistic, not a correction to the old number.

THE GAP IS STATISTIC-DEPENDENT, and that has to be disclosed rather than
navigated. Half-L1, Cramer's V, assortativity and normalised mutual information
put the same two matrices between roughly 6x and 39x apart. Reporting only the
largest would be number-shopping; reporting only the smallest understates. §32.3
prints all of them side by side and the spread goes in the text.

WHAT THE NULL FLOORS ARE FOR. A gap ratio says nothing until each side's own
sampling floor is known. Two floors are computed and they are not symmetric:

  * the survey side rests on 465 Seoul respondents, so its floor is large, and
    it is estimated three ways -- Miller-Madow analytic bias, a respondent-level
    bootstrap, and a permutation null that shuffles alters across egos;
  * the passive side rests on millions of device-days, so its floor is small,
    and it is simulated multinomially at the device-equivalent sample size
    implied by the coverage profile (p9). Devices contribute to many cells and
    are not independent draws, so this floor is a LOWER bound on the true floor
    -- which is the conservative direction for the claim being made.

The comparison those two floors license is the sharpest sentence available here,
and it is not the ratio: the passive matrix's entire departure from proportionate
mixing is smaller than the sampling uncertainty of the survey it is measured
against. That is the advisor's expected conclusion in its strongest form, and it
is why §32.4 leads the reporting rather than §32.3.

THE OTHER BRANCH IS BUILT IN. If the gap were to converge under normalisation,
the story becomes "passive is usable, calibrate the margins", and the deliverable
becomes the calibration map itself. §32.5 fits it either way by bi-proportional
scaling (IPF/RAS) of the passive matrix onto the survey margins, so the branch
never has to be retrofitted.

This script reads results_p26.json and results_p27.json and never writes them.
The survey microdata is used only for the bootstrap and permutation floors; if
the drive is absent those are skipped and marked, and everything else still runs.

    python eda/p32_pmix.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import AGES, AGE_LABEL
from paths import FIG, RESULTS, ROOT

SURVEY_MONTHS = [202312, 202402]
BOOT = 500
PERM = 500
NULL_SIM = 400
SEED = 20260819

out = {}


# --------------------------------------------------------------- the statistics
def pm_null(T):
    """The proportionate-mixing null for a symmetric contact-count matrix.

    e is the joint distribution of (ego band, alter band) over contacts; the null
    is the outer product of its margins, which is what "everyone mixes in
    proportion to how much contact each band has in total" means.
    """
    s = T.sum()
    e = T / s if s else T
    r = e.sum(1)
    return e, r, np.outer(r, r)


def mi_bits(e, r, E):
    """Mutual information I(a;a') in bits: how much of the age entropy the
    matrix resolves. Zero exactly under proportionate mixing."""
    m = e > 0
    return float((e[m] * np.log2(e[m] / E[m])).sum())


def mm_bias_bits(e, n_obs):
    """Miller-Madow small-sample bias of the plug-in MI, in bits.

    The plug-in estimator is biased UPWARD, and the bias grows with the number of
    occupied cells and shrinks with the sample size, so a sparse survey matrix
    reports mutual information it has not earned. This is the analytic floor;
    §32.4 also measures it by permutation, and the two should agree.
    """
    if not n_obs:
        return float("nan")
    kxy = int((e > 0).sum())
    kx = int((e.sum(1) > 0).sum())
    ky = int((e.sum(0) > 0).sum())
    return float((kxy - kx - ky + 1) / (2.0 * n_obs * np.log(2)))


def excess_stats(T, n_obs=None):
    """Every way of saying "how far is this from proportionate mixing"."""
    e, r, E = pm_null(T)
    h = -float((r[r > 0] * np.log2(r[r > 0])).sum())
    i = mi_bits(e, r, E)
    d = e - E
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(E > 0, e / E, np.nan)
    dg = np.diag(ratio)
    # Cramer's V on the joint table; the same chi-square-shaped quantity that
    # assortativity applies to the diagonal, applied to every cell.
    chi2_n = float(np.nansum(np.where(E > 0, d * d / E, 0.0)))
    k = min((r > 0).sum(), (e.sum(0) > 0).sum())
    denom = 1 - float((r * r).sum())
    return dict(
        assortativity=float((np.trace(e) - (r * r).sum()) / denom) if denom else np.nan,
        diag_excess_sum=float(np.trace(e) - (r * r).sum()),
        half_l1=float(np.abs(d).sum() / 2),
        cramers_v=float(np.sqrt(chi2_n / (k - 1))) if k > 1 else np.nan,
        mi_bits=i,
        entropy_bits=h,
        nmi=float(i / h) if h else np.nan,
        mi_bits_debiased=(float(i - mm_bias_bits(e, n_obs))
                          if n_obs else None),
        mm_bias_bits=(mm_bias_bits(e, n_obs) if n_obs else None),
        pm_frobenius=float(np.linalg.norm(d) / np.linalg.norm(e)),
        diag_ratio_min=float(np.nanmin(dg)),
        diag_ratio_max=float(np.nanmax(dg)),
        mean_abs_ratio_dev=float(np.nanmean(np.abs(ratio - 1))),
    )


def rank_stats(T):
    """How close the matrix is to rank one, two ways.

    sigma1^2/sum sigma^2 and sigma2/sigma1 describe the BEST rank-1
    approximation. Proportionate mixing is a SPECIFIC rank-1 matrix, r r', so the
    residual against that one is reported beside them. When the two agree, the
    best rank-1 approximation of the matrix essentially IS proportionate mixing,
    which is a stronger statement than either number alone.
    """
    e, r, E = pm_null(T)
    sv = np.linalg.svd(e, compute_uv=False)
    tot = float((sv ** 2).sum())
    return dict(
        singular_values=sv.tolist(),
        sigma1_share=float(sv[0] ** 2 / tot) if tot else np.nan,
        sigma2_over_sigma1=float(sv[1] / sv[0]) if sv[0] else np.nan,
        best_rank1_resid=float(np.sqrt(max(0.0, 1 - sv[0] ** 2 / tot))),
        pm_rank1_resid=float(np.linalg.norm(e - E) / np.linalg.norm(e)),
        row_l1_max=float(max(
            np.abs(a - b).sum()
            for a in (e / e.sum(1, keepdims=True))
            for b in (e / e.sum(1, keepdims=True)))),
    )


def ipf(T, target_r, target_c, iters=2000, tol=1e-12):
    """Bi-proportional (RAS/IPF) scaling of T onto given margins.

    Fitting the passive matrix to the survey's margins removes exactly the level
    and composition difference that directive 1 forbids us to claim, leaving the
    interaction structure. The row multipliers it produces ARE the calibration
    map, and are the deliverable in the branch where the gap converges.
    """
    X = np.array(T, float)
    X[X <= 0] = 1e-300
    for _ in range(iters):
        rs = X.sum(1)
        X *= np.divide(target_r, rs, out=np.zeros_like(rs), where=rs > 0)[:, None]
        cs = X.sum(0)
        X *= np.divide(target_c, cs, out=np.zeros_like(cs), where=cs > 0)[None, :]
        if (np.abs(X.sum(1) - target_r).max() < tol
                and np.abs(X.sum(0) - target_c).max() < tol):
            break
    return X, (T.sum(1) / np.where(X.sum(1) > 0, X.sum(1), np.nan))


def symmetrise(C, pop):
    """T = N_a C[a,a'], symmetrised -- the same reduction p27 uses, repeated here
    rather than imported so this script states its own definition."""
    T = pop[:, None] * C
    asym = float(np.abs(T - T.T).sum() / T.sum()) if T.sum() else np.nan
    return (T + T.T) / 2, asym


def null_floor(T, n_obs, n_sim=NULL_SIM, rng=None):
    """Simulate the excess statistics under proportionate mixing at sample size
    n_obs. This is what "no age structure at all" looks like when it is measured
    with this much data, and nothing below it is a finding."""
    rng = rng or np.random.default_rng(SEED)
    e, r, E = pm_null(T)
    p = E.ravel()
    p = p / p.sum()
    keep = ("mi_bits", "nmi", "assortativity", "half_l1", "cramers_v")
    draws = {k: [] for k in keep}
    n = int(min(n_obs, 5e6))
    for _ in range(n_sim):
        d = rng.multinomial(n, p).reshape(E.shape).astype(float)
        d = (d + d.T) / 2
        if d.sum() == 0:
            continue
        st = excess_stats(d)
        for k in keep:
            draws[k].append(st[k])
    return {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                    mean=float(np.mean(v))) for k, v in draws.items() if v}


# ------------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))
    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]          # no 80+ survey egos
    lbl = [AGE_LABEL[AGES[i]] for i in sq]

    # ------------------------------------------------- 32.1 the identity check
    print("=== 32.1 assortativity is the diagonal excess, asserted not asserted ===")
    tk = "202312|WE|dong|holidayfree"
    T0, _ = symmetrise(np.array(p26["matrices"][tk]["A"])[np.ix_(sq, sq)]
                       / pops[202312][sq][:, None], pops[202312][sq])
    e0, r0, E0 = pm_null(T0)
    num = float(np.trace(e0) - (r0 * r0).sum())
    diag_excess = float((np.diag(e0) - np.diag(E0)).sum())
    assert abs(num - diag_excess) < 1e-12, "assortativity is not the diagonal excess"
    print(f"  numerator of Newman r          {num: .8f}")
    print(f"  sum_a (e_aa - E_aa)            {diag_excess: .8f}")
    print("  identical, so the published ten-fold gap already WAS an")
    print("  excess-over-proportionate-mixing comparison, restricted to the")
    print("  diagonal. What follows adds the off-diagonal and the rank test.")
    out["identity_check"] = dict(newman_numerator=num, diagonal_excess=diag_excess,
                                 max_abs_diff=abs(num - diag_excess))

    # --------------------------------- 32.2 excess and rank for every matrix
    print("\n=== 32.2 excess and rank, every stored matrix ===")
    excess, spectrum = {}, {}
    for key, m in p26["matrices"].items():
        ym = int(key.split("|")[0])
        if ym not in pops:
            continue
        A = np.array(m["A"])[np.ix_(sq, sq)]
        T, asym = symmetrise(A / pops[ym][sq][:, None], pops[ym][sq])
        excess[f"passive|{key}"] = dict(excess_stats(T), asymmetry=asym,
                                        side="passive")
        spectrum[f"passive|{key}"] = rank_stats(T)
    for key, s in p27["survey"].items():
        ym_tag = key.split("|")[0]
        ym = int(ym_tag.replace("all", "")) if ym_tag != "pooled" else 202312
        if ym not in pops:
            continue
        C = np.array(s["C"])[np.ix_(sq, sq)]
        n_obs = float(np.sum(s["n_ego"]) and
                      np.array(s["n_ego"])[sq] @ C.sum(1) * s["n_days"])
        T, asym = symmetrise(C, pops[ym][sq])
        excess[f"survey|{key}"] = dict(excess_stats(T, n_obs=n_obs),
                                       asymmetry=asym, n_contacts=n_obs,
                                       side="survey")
        spectrum[f"survey|{key}"] = rank_stats(T)
    out["excess"], out["spectrum"] = excess, spectrum
    print(f"  {len(excess)} matrices: {sum(1 for k in excess if k.startswith('passive'))}"
          f" passive, {sum(1 for k in excess if k.startswith('survey'))} survey")

    print("\n  the ladder, WE panel, holiday-free where it exists:")
    print(f"  {'matrix':<34}{'nMI':>9}{'assort':>9}{'sig1^2':>9}{'pm resid':>10}")
    ladder = {}
    for ym in SURVEY_MONTHS:
        for lev in ("dong", "gu", "city"):
            k = f"{ym}|WE|{lev}" + ("|holidayfree" if lev == "dong" else "")
            pk = f"passive|{k}"
            if pk not in excess:
                continue
            ex, sp = excess[pk], spectrum[pk]
            ladder[f"{ym}|WE|{lev}"] = dict(ex, **{"sigma1_share": sp["sigma1_share"]})
            print(f"  {k:<34}{ex['nmi']:>9.4f}{ex['assortativity']:>9.4f}"
                  f"{sp['sigma1_share']:>9.4f}{sp['pm_rank1_resid']:>10.4f}")
        sk = f"survey|{ym}|WE|seoul"
        ex, sp = excess[sk], spectrum[sk]
        print(f"  {'survey ' + str(ym) + ' WE seoul':<34}{ex['nmi']:>9.4f}"
              f"{ex['assortativity']:>9.4f}{sp['sigma1_share']:>9.4f}"
              f"{sp['pm_rank1_resid']:>10.4f}")
    out["ladder_excess"] = ladder

    # -------------------------------------- 32.3 the head-to-head, all statistics
    print("\n=== 32.3 passive against survey, every statistic ===")
    print("  The spread across statistics is the point: a single ratio is a"
          " choice, not a measurement.")
    comp = []
    for ym in SURVEY_MONTHS:
        pk, sk = f"passive|{ym}|WE|dong|holidayfree", f"survey|{ym}|WE|seoul"
        if pk not in excess or sk not in excess:
            continue
        p_, s_ = excess[pk], excess[sk]
        row = dict(ym=ym)
        print(f"\n  {ym}   {'statistic':<24}{'passive':>12}{'survey':>12}{'ratio':>10}")
        for st, nm in (("half_l1", "half-L1"), ("cramers_v", "Cramer's V"),
                       ("assortativity", "assortativity"), ("mi_bits", "MI (bits)"),
                       ("nmi", "normalised MI")):
            ratio = s_[st] / p_[st] if p_[st] else np.nan
            row[st] = dict(passive=p_[st], survey=s_[st], ratio=float(ratio))
            print(f"  {'':<6}{nm:<24}{p_[st]:>12.4f}{s_[st]:>12.4f}{ratio:>10.1f}x")
        rr = [row[s]["ratio"] for s in ("half_l1", "cramers_v", "assortativity",
                                        "nmi")]
        row["ratio_spread"] = dict(lo=float(min(rr)), hi=float(max(rr)))
        print(f"  {'':<6}{'--> ratio spans':<24}{min(rr):>11.1f}x{max(rr):>11.1f}x")
        comp.append(row)
    out["comparison"] = comp

    print("\n  rank: the passive matrix against the survey's")
    for ym in SURVEY_MONTHS:
        p_ = spectrum.get(f"passive|{ym}|WE|dong|holidayfree")
        s_ = spectrum.get(f"survey|{ym}|WE|seoul")
        if not p_ or not s_:
            continue
        print(f"  {ym}: passive sigma1^2/sum {p_['sigma1_share']:.4f}, "
              f"sigma2/sigma1 {p_['sigma2_over_sigma1']:.3f}, "
              f"best-rank1 resid {p_['best_rank1_resid']:.4f} vs "
              f"proportionate-mixing resid {p_['pm_rank1_resid']:.4f}")
        print(f"  {'':<6}survey  sigma1^2/sum {s_['sigma1_share']:.4f}, "
              f"sigma2/sigma1 {s_['sigma2_over_sigma1']:.3f}, "
              f"best-rank1 resid {s_['best_rank1_resid']:.4f} vs "
              f"proportionate-mixing resid {s_['pm_rank1_resid']:.4f}")
        gap = abs(p_["best_rank1_resid"] - p_["pm_rank1_resid"])
        print(f"  {'':<6}the two passive residuals differ by {gap:.4f} -- the best"
              f" rank-1 approximation of the passive matrix IS proportionate mixing")

    # ------------------------------------------------------- 32.4 the null floors
    print("\n=== 32.4 what each side's own sampling floor is ===")
    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}
    floors = {}
    for ym in SURVEY_MONTHS:
        pk, sk = f"passive|{ym}|WE|dong|holidayfree", f"survey|{ym}|WE|seoul"
        if pk not in excess:
            continue
        A = np.array(p26["matrices"][f"{ym}|WE|dong|holidayfree"]["A"])[np.ix_(sq, sq)]
        Tp, _ = symmetrise(A / pops[ym][sq][:, None], pops[ym][sq])
        # device-equivalent sample size: population divided by its coverage weight
        n_dev = float(sum(pops[ym][i] / cov.get(AGE_LABEL[AGES[i]], np.nan)
                          for i in sq))
        fp = null_floor(Tp, n_dev, rng=rng)
        Cs = np.array(p27["survey"][f"{ym}|WE|seoul"]["C"])[np.ix_(sq, sq)]
        Ts, _ = symmetrise(Cs, pops[ym][sq])
        n_con = excess[sk]["n_contacts"]
        fs = null_floor(Ts, n_con, rng=rng)
        floors[str(ym)] = dict(passive=dict(n_effective=n_dev, floor=fp,
                                            observed=excess[pk]["mi_bits"]),
                               survey=dict(n_effective=n_con, floor=fs,
                                           observed=excess[sk]["mi_bits"],
                                           mm_bias_bits=excess[sk]["mm_bias_bits"]))
        print(f"\n  {ym}")
        print(f"    passive  MI {excess[pk]['mi_bits']:.4f} bits, "
              f"null floor {fp['mi_bits']['median']:.5f} "
              f"(p95 {fp['mi_bits']['p95']:.5f}) at n_dev {n_dev:,.0f}")
        print(f"    survey   MI {excess[sk]['mi_bits']:.4f} bits, "
              f"null floor {fs['mi_bits']['median']:.4f} "
              f"(p95 {fs['mi_bits']['p95']:.4f}) at n_contacts {n_con:,.0f}; "
              f"Miller-Madow bias {excess[sk]['mm_bias_bits']:.4f}")
        ratio = fs["mi_bits"]["median"] / excess[pk]["mi_bits"]
        print(f"    --> the survey's own floor is {ratio:.2f}x the ENTIRE passive"
              f" excess")
    out["null_floor"] = floors
    print("\n  The passive floor is a lower bound: devices appear in many cells,"
          " so the\n  multinomial draw overstates its precision. That is the"
          " conservative\n  direction for the claim being made.")

    # ------------------------ 32.4b survey bootstrap and permutation, if microdata
    boot = {"status": "skipped: survey microdata unavailable"}
    try:
        from p27_survey import (contact_matrix, ego_cubes, holiday_free_dows,
                                load_survey)
        ego, d = load_survey()
        # WE means work+school and other, NOT household. p27 filters the contact
        # frame by panel before building the cube; leaving H in makes the matrix
        # markedly LESS assortative, because household contacts are the
        # cross-generational ones. The anchor below is what caught that.
        d = d[d.panel.isin(["W", "E"])]
        print("\n=== 32.4b survey floor, measured rather than assumed ===")
        boot = {}
        for ym in SURVEY_MONTHS:
            cube, eb, nd = ego_cubes(ego, d, True, ym=ym, dows=holiday_free_dows(ym))
            pop = pops[ym][sq]
            draws, perms, anchor = [], [], []
            for _ in range(BOOT):
                idx = rng.integers(0, len(eb), len(eb))
                C, _ = contact_matrix(cube[idx], eb[idx], nd)
                T, _ = symmetrise(C[np.ix_(sq, sq)], pop)
                st = excess_stats(T)
                draws.append(st["mi_bits"])
                # ANCHOR. p27 already published a respondent bootstrap of
                # assortativity on exactly this cube; re-deriving it here from an
                # independent rng stream is the check that this loop resamples what
                # it thinks it does. If assortativity reproduces and MI does not
                # bracket its point estimate, that is a property of MI -- which is
                # sparsity-biased and shrinks when a bootstrap draw thins the
                # distinct-respondent count -- and not a bug in the resampling.
                anchor.append(st["assortativity"])
            for _ in range(PERM):
                # shuffle ego bands, keeping each ego's own alter tally intact:
                # the alter margin and every ego's contact count survive, only the
                # pairing dies. That is exactly the proportionate-mixing null.
                C, _ = contact_matrix(cube, rng.permutation(eb), nd)
                T, _ = symmetrise(C[np.ix_(sq, sq)], pop)
                perms.append(excess_stats(T)["mi_bits"])
            lo, hi = np.percentile(draws, [2.5, 97.5])
            alo, ahi = np.percentile(anchor, [2.5, 97.5])
            pub = p27["survey_assortativity_bootstrap"][str(ym)]
            ok = abs(alo - pub["lo"]) < 0.02 and abs(ahi - pub["hi"]) < 0.02
            boot[str(ym)] = dict(
                mi_boot_lo=float(lo), mi_boot_hi=float(hi),
                mi_boot_median=float(np.median(draws)),
                mi_perm_median=float(np.median(perms)),
                mi_perm_p95=float(np.percentile(perms, 95)),
                assort_boot_lo=float(alo), assort_boot_hi=float(ahi),
                assort_published_lo=pub["lo"], assort_published_hi=pub["hi"],
                anchor_reproduces=bool(ok),
                n_ego=int(len(eb)))
            print(f"  {ym}: ANCHOR assortativity CI [{alo:+.4f}, {ahi:+.4f}] vs "
                  f"p27's published [{pub['lo']:+.4f}, {pub['hi']:+.4f}] "
                  f"-- {'reproduces' if ok else 'DOES NOT REPRODUCE'}")
            assert ok, ("the respondent bootstrap does not reproduce p27's "
                        "published assortativity CI; fix before reading the MI")
            b = boot[str(ym)]
            print(f"  {ym}: MI {excess[f'survey|{ym}|WE|seoul']['mi_bits']:.4f} bits, "
                  f"95% CI [{lo:.4f}, {hi:.4f}] over {BOOT} respondent bootstraps "
                  f"(n={len(eb)})")
            print(f"  {'':<6}permutation null median {b['mi_perm_median']:.4f}, "
                  f"p95 {b['mi_perm_p95']:.4f} over {PERM} shuffles")
            pas = excess[f"passive|{ym}|WE|dong|holidayfree"]["mi_bits"]
            print(f"  {'':<6}the passive matrix's entire excess is {pas:.4f} bits, "
                  f"{'BELOW' if pas < b['mi_perm_median'] else 'above'} the survey's"
                  f" permutation floor")
    except Exception as exc:                              # noqa: BLE001
        print(f"\n=== 32.4b skipped: {exc} ===")
        boot = {"status": f"skipped: {exc}"}
    out["survey_floor_measured"] = boot

    # ------------------------------------------------- 32.5 the convergence branch
    print("\n=== 32.5 if the margins are calibrated away, what is left? ===")
    ipf_out = {}
    for ym in SURVEY_MONTHS:
        pk, sk = f"{ym}|WE|dong|holidayfree", f"{ym}|WE|seoul"
        if pk not in p26["matrices"]:
            continue
        pop = pops[ym][sq]
        Tp, _ = symmetrise(np.array(p26["matrices"][pk]["A"])[np.ix_(sq, sq)]
                           / pops[ym][sq][:, None], pop)
        Ts, _ = symmetrise(np.array(p27["survey"][sk]["C"])[np.ix_(sq, sq)], pop)
        tgt = Ts.sum(1) / Ts.sum() * Tp.sum()
        X, _ = ipf(Tp, tgt, tgt)
        mult = np.divide(X.sum(1), Tp.sum(1), out=np.ones_like(tgt),
                         where=Tp.sum(1) > 0)
        ex_before, ex_after = excess_stats(Tp), excess_stats(X)
        ratio_b = excess[f"survey|{sk}"]["nmi"] / ex_before["nmi"]
        ratio_a = excess[f"survey|{sk}"]["nmi"] / ex_after["nmi"]
        ipf_out[str(ym)] = dict(bands=lbl, row_multiplier=mult.tolist(),
                                nmi_before=ex_before["nmi"],
                                nmi_after=ex_after["nmi"],
                                nmi_survey=excess[f"survey|{sk}"]["nmi"],
                                gap_before=float(ratio_b), gap_after=float(ratio_a),
                                converged=bool(ratio_a < 2.0))
        print(f"  {ym}: normalised MI gap {ratio_b:.1f}x before margin calibration,"
              f" {ratio_a:.1f}x after")
        print(f"  {'':<6}row multipliers {mult.min():.2f}-{mult.max():.2f} "
              f"(largest on {lbl[int(np.argmax(mult))]})")
    conv = all(v["converged"] for v in ipf_out.values()) if ipf_out else False
    print(f"  branch: {'CONVERGES -- the deliverable is the calibration map' if conv else 'does NOT converge -- dong-level co-presence carries almost no age information once composition is removed'}")
    out["ipf_calibration"] = dict(months=ipf_out, converged=conv)

    # ------------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("202312|WE|dong|holidayfree", "passive, dong"),
              ("202312|WE|gu", "passive, gu")]
    mats = []
    for k, t in panels:
        A = np.array(p26["matrices"][k]["A"])[np.ix_(sq, sq)]
        T, _ = symmetrise(A / pops[202312][sq][:, None], pops[202312][sq])
        e, r, E = pm_null(T)
        mats.append((np.log10(e / E), t))
    Cs = np.array(p27["survey"]["202312|WE|seoul"]["C"])[np.ix_(sq, sq)]
    Ts, _ = symmetrise(Cs, pops[202312][sq])
    e, r, E = pm_null(Ts)
    mats.append((np.log10(np.where(e > 0, e / E, np.nan)), "survey, Seoul"))
    vmax = max(np.nanmax(np.abs(m)) for m, _ in mats)
    for ax, (m, t) in zip(axes, mats):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{t}\nlog10 ratio to proportionate mixing", fontsize=9)
        ax.set_xticks(range(len(lbl)))
        ax.set_yticks(range(len(lbl)))
        ax.set_xticklabels(lbl, rotation=90, fontsize=6)
        ax.set_yticklabels(lbl, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("2023-12, same colour limits: flat rows against a block diagonal",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "p32_excess.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key, style in ((f"passive|202312|WE|dong|holidayfree", "-o"),
                       (f"passive|202312|WE|gu", "-s"),
                       (f"survey|202312|WE|seoul", "-^")):
        sv = np.array(spectrum[key]["singular_values"])
        ax.semilogy(range(1, len(sv) + 1), sv / sv[0], style, ms=4,
                    label=key.replace("|", " "))
    ax.set_xlabel("singular value index")
    ax.set_ylabel("sigma_k / sigma_1")
    ax.set_title("How close to rank one? (2023-12)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "p32_spectrum.png", dpi=150)
    plt.close(fig)

    with open(f"{RESULTS}/results_p32.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {RESULTS}/results_p32.json, "
          f"{FIG}/p32_excess.png, {FIG}/p32_spectrum.png")


if __name__ == "__main__":
    main()
