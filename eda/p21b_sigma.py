#!/usr/bin/env python
"""Phase 1d — sigma_d, and how far it can move the headline gradient.

The manual settled that the expansion weight is a function of (행정동 x 성 x 연령),
so Appendix 1 row 3's condition — weights homogeneous across dong within an age
band — is refuted by construction. What is left is a magnitude question, and the
plan branches on it:

    induced movement in 80+ < 0.10 pp   the city-wide aggregate stands as is
    induced movement in 80+ > 0.25 pp   every headline has to be estimated dong
                                        by dong and reweighted, Phase 7 grows a
                                        week, and Phase 5 becomes more urgent

This script computes that movement. Two things have to be got right for the
answer to mean anything.

WHICH ESTIMATOR CAN EVEN BE CORRECTED. The weight attaches to a device, and the
device is placed by its 야간상주지 — its residence dong. A trip record gives the
residence dong only when the origin attribute is H. For a W-origin or E-origin
trip there is no way to recover whose weight it carried. So p18's headline, which
pools all nine mtypes, cannot be corrected for dong-level weight heterogeneity at
all: its exposure is real but unquantifiable from the data alone. The H-origin
share can be corrected, and that is what is measured here. This is an argument
for reporting the H-origin variant as the defensible one, not a technicality.

WHETHER THE ERROR CANCELS IN A DIFFERENCE. The headline is a Dec-minus-March
change, not a level. results_p8.json shows the quantisation frozen across months
(CV <= 0.0123, median exactly 0), so the weight error is the same draw in both
months and largely cancels in the difference. That is a big effect and it has to
be modelled rather than assumed away, so both are reported: frozen weights (one
draw shared by both months) and, as the pessimistic alternative, an independent
draw per month.

sigma_d itself is bounded, not identified. The cross-dong spread of log rho-hat
contains real behavioural variation as well as weight error, so using all of it
as if it were weight error is deliberately conservative: the true induced
movement is smaller than what this reports. If the conservative number already
clears the "small" threshold, the covariate regression that would tighten it is
not needed for the branch decision.

    python eda/p21_rho.py        # first — rho-hat
    python eda/p21b_sigma.py     # then — sigma_d and the propagation
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import AGE_LABEL, AGES, YMS, YM_LABEL, connect
from paths import DERIVED, FIG, PARQUET_GLOB, ROOT, require

os.makedirs(FIG, exist_ok=True)
SEOUL = "BETWEEN 1101000 AND 1125999"
CACHE = "dest_attr_by_dong.parquet"
MONTHS = (202003, 202012)          # the headline comparison, as in p18/p19
N_DRAWS = 4000
RNG = np.random.default_rng(20260817)
out = {}


def dest_by_dong(refresh=False):
    """(ym, origin dong, age, destination attribute) for H-origin trips.

    H-origin only, because that is the subset whose residence dong — and so whose
    weight — is knowable. Deduplicated on the natural key, as everywhere else.
    """
    if not refresh and (DERIVED / CACHE).exists():
        df = pd.read_parquet(DERIVED / CACHE)
        print(f"  reusing {DERIVED / CACHE} ({len(df):,} rows)")
        return df
    con = connect(threads=6)
    con.execute("PRAGMA disable_progress_bar")
    frames = []
    for ym in YMS:
        frames.append(con.execute(f"""
            WITH g AS (
              SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                     min(pop) AS pop, bool_or(masked) AS masked
              FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
              WHERE ym = {ym} AND o_dong {SEOUL} AND mtype LIKE 'H%'
              GROUP BY ALL)
            SELECT ym, o_dong, age, substr(mtype, 2, 1) AS dest_attr,
                   coalesce(sum(pop), 0)                        AS v_obs,
                   count(*) FILTER (masked)                     AS n_masked,
                   coalesce(sum(pop) FILTER (d_dong {SEOUL}), 0) AS v_obs_int,
                   count(*) FILTER (masked AND d_dong {SEOUL})   AS n_masked_int
            FROM g GROUP BY 1,2,3,4""").df())
        print(f"  {ym} done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(DERIVED / CACHE, index=False)
    return df


def robust_sd(x):
    """1.4826 * MAD. The young bands have fat tails (p21 section 1.7: CV 2-2.9
    against a p05-p95 that is narrow), so a plain SD there describes a handful of
    dong rather than the spread, and would inflate every number downstream."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def main():
    require(DERIVED / "rho_numerator.parquet", "rho-hat (run eda/p21_rho.py)")

    # ------------------------------------------------- 1 sigma_d(a,t) from rho-hat
    print("=== 1 sigma_d(a,t): cross-dong spread of log rho-hat ===")
    r = pd.read_parquet(DERIVED / "rho_numerator.parquet")
    d = (r.groupby(["ym", "o_dong", "age"], as_index=False)
         .agg(V=("V_mid", "sum"), regpop=("regpop", "sum")))
    # 주민등록 counts Korean nationals only while KT counts every device, so a
    # dong with many foreign residents gets a denominator that is too small and a
    # rho-hat that is too high. The foreign share ranges from 0.2% to 35% ACROSS
    # dong, so leaving it out injects spurious cross-dong variance straight into
    # sigma_d — the one statistic that has to be an honest measure of dispersion.
    fpath = DERIVED / "foreign.parquet"
    if fpath.exists():
        fo = pd.read_parquet(fpath)
        fo = (fo[fo.ym.isin(YMS)].rename(columns={"dong_code": "o_dong"})
              .groupby(["ym", "o_dong", "age"], as_index=False)["pop"].sum()
              .rename(columns={"pop": "foreign"}))
        fo["o_dong"] = fo.o_dong.astype("int64")
        d = d.merge(fo, on=["ym", "o_dong", "age"], how="left")
        d["foreign"] = d.foreign.fillna(0.0)
        print(f"  denominator = 주민등록 + 등록외국인 "
              f"(foreign is {d.foreign.sum() / (d.regpop.sum() + d.foreign.sum()):.2%} "
              f"of Seoul overall, but 0.2%-35% across dong)")
    else:
        d["foreign"] = 0.0
        print("  denominator = 주민등록 only (no foreign.parquet)")
    d["den"] = d.regpop + d.foreign
    d = d[(d.V > 0) & (d.den > 0)].copy()
    d["lr"] = np.log(d.V / d.den)
    d["lr_kor"] = np.log(d.V / d.regpop.where(d.regpop > 0))
    sig = (d.groupby(["ym", "age"])
           .agg(sd_plain=("lr", "std"), sd_robust=("lr", robust_sd),
                sd_robust_kor=("lr_kor", robust_sd), n=("lr", "size")).reset_index())
    korsd = sig.pivot(index="age", columns="ym", values="sd_robust_kor").median(axis=1)
    piv = sig.pivot(index="age", columns="ym", values="sd_robust")
    print("  robust sigma_d by age x month (MAD-based):")
    show = piv.copy()
    show.index = [AGE_LABEL[a] for a in show.index]
    show.columns = [YM_LABEL[y] for y in show.columns]
    print(show.round(3).to_string())
    uncond = piv.median(axis=1)       # frozen weights -> one sigma per age band
    plain = sig.pivot(index="age", columns="ym", values="sd_plain").median(axis=1)

    # ------------------------- 1b conditional sigma_d: regress out what dong are
    # The plan's decision statistic is the residual after regressing log rho-hat
    # on dong-level covariates, not the raw spread: a dong that is denser or more
    # commercial genuinely has more departures per resident, and that is
    # behaviour, not weight error. Conditioning removes what the covariates can
    # explain and leaves a tighter — still upper — bound.
    #
    # area_km2_ref, not area_km2. p20c found that SGIS redrew the 한강 allocation
    # between its 2024 and 2025 vintages, moving 42 riverside dong by up to 55%
    # with no administrative change behind it. A vintage-tracking area would inject
    # that cartographic revision into the covariate as if it were real.
    cov = pd.read_parquet(DERIVED / "dong_covariates.parquet")
    cov = cov[cov.year == 2020][["dong_code", "area_km2_ref", "worker",
                                 "estab", "worker_retail_food"]]
    cov = cov.rename(columns={"dong_code": "o_dong"})
    dc = d.merge(cov, on="o_dong", how="left")
    miss = dc.area_km2_ref.isna().sum()
    if miss:
        raise SystemExit(f"{miss} rows have no covariates — check the crosswalk")
    dc["x_popden"] = np.log(dc.regpop / dc.area_km2_ref)
    dc["x_workden"] = np.log((dc.worker + 1) / dc.area_km2_ref)
    dc["x_retail"] = dc.worker_retail_food / dc.worker.clip(lower=1)
    XCOLS = ["x_popden", "x_workden", "x_retail"]

    def resid_sd(g):
        X = np.column_stack([np.ones(len(g))] + [g[c].values for c in XCOLS])
        y = g.lr.values
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        if ok.sum() <= X.shape[1] + 2:
            return np.nan
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        return robust_sd(y[ok] - X[ok] @ beta)

    cond = (dc.groupby(["ym", "age"])[["lr"] + XCOLS]
            .apply(resid_sd).rename("sd").reset_index()
            .pivot(index="age", columns="ym", values="sd").median(axis=1))
    print("\n  sigma_d pooled over months:")
    print(f"    {'band':<7} {'kor-only':>9} {'uncond':>7} {'cond':>7} "
          f"{'cond/uncond':>12} {'plain':>7}")
    for a in AGES:
        print(f"    {AGE_LABEL[a]:<7} {korsd[a]:>9.3f} {uncond[a]:>7.3f} "
              f"{cond[a]:>7.3f} {cond[a] / uncond[a]:>12.3f} {plain[a]:>7.3f}")
    print(f"  adding 등록외국인 to the denominator alone cuts sigma_d by "
          f"{1 - (uncond / korsd).median():.1%} at the median band — spurious "
          f"dispersion, not behaviour")
    print(f"  covariates: log(registered density), log(worker density), "
          f"retail+food share of workers")
    print(f"  conditioning removes least where the paper's claims are: "
          f"70-74 {1 - cond[70] / uncond[70]:.1%}, 80+ {1 - cond[80] / uncond[80]:.1%}; "
          f"most at 20-49 ({1 - (cond[[20,25,30,35,40,45]] / uncond[[20,25,30,35,40,45]]).min():.1%} at best)")
    sigma_a = cond                    # the decision statistic the plan specifies
    out["sigma_d"] = {"robust_by_age_month": sig.to_dict("records"),
                      "pooled_unconditional": {AGE_LABEL[a]: float(uncond[a]) for a in AGES},
                      "pooled_conditional": {AGE_LABEL[a]: float(cond[a]) for a in AGES},
                      "pooled_plain": {AGE_LABEL[a]: float(plain[a]) for a in AGES},
                      "covariates": XCOLS}

    # --------------------------------------- 2 the dong-level E share to perturb
    print("\n=== 2 H-origin E share by dong ===")
    da = dest_by_dong()
    da["v"] = da.v_obs + 1.5 * da.n_masked
    wide = (da.pivot_table(index=["ym", "o_dong", "age"], columns="dest_attr",
                           values="v", aggfunc="sum", fill_value=0.0)
            .reset_index())
    wide["tot"] = wide[["E", "H", "W"]].sum(axis=1)
    wide = wide[wide.tot > 0].copy()
    wide["s"] = wide.E / wide.tot

    city = (wide.groupby(["ym", "age"], as_index=False)
            .agg(E=("E", "sum"), tot=("tot", "sum"), n_dong=("o_dong", "size")))
    city["E_share"] = city.E / city.tot
    w2 = wide.merge(city[["ym", "age", "E_share"]], on=["ym", "age"])
    w2["dev2"] = w2.tot * (w2.s - w2.E_share) ** 2
    var = (w2.groupby(["ym", "age"], as_index=False)
           .agg(num=("dev2", "sum"), den=("tot", "sum")))
    var["sd_m"] = np.sqrt(var.num / var.den)
    city = city.merge(var[["ym", "age", "sd_m"]], on=["ym", "age"])
    print("  city-wide H-origin E share, and its volume-weighted cross-dong SD:")
    hdr = city[city.ym.isin(MONTHS)].pivot(index="age", columns="ym",
                                           values="E_share")
    sdm = city[city.ym.isin(MONTHS)].pivot(index="age", columns="ym", values="sd_m")
    for a in AGES:
        print(f"    {AGE_LABEL[a]:<7} Mar {hdr.loc[a, 202003]:.4f}  "
              f"Dec {hdr.loc[a, 202012]:.4f}  "
              f"change {(hdr.loc[a, 202012] - hdr.loc[a, 202003]) * 100:+6.3f} pp"
              f"   cross-dong SD(s) Mar {sdm.loc[a, 202003]:.4f}")
    out["e_share_h_origin"] = city.to_dict("records")

    # ---------------------------------- 3 closed-form worst case, on the LEVEL
    # First order, the shift in a weight-perturbed weighted mean is the
    # m-weighted covariance of eps with s. Maximising that under a variance
    # budget gives, by Cauchy-Schwarz, exactly sigma * SD_m(s) — no simulation
    # needed, and it is an upper bound rather than a typical value.
    print("\n=== 3 adversarial upper bound on the LEVEL shift (sigma x SD_m(s)) ===")
    bound = {}
    for a in AGES:
        b = sigma_a[a] * sdm.loc[a, 202003] * 100
        bound[AGE_LABEL[a]] = float(b)
        print(f"    {AGE_LABEL[a]:<7} sigma {sigma_a[a]:.3f} x SD_m(s) "
              f"{sdm.loc[a, 202003]:.4f} = {b:6.3f} pp")
    out["level_bound_pp"] = bound

    # ------------------------------- 4 Monte Carlo on the CHANGE, the real target
    print(f"\n=== 4 Monte Carlo, {N_DRAWS} draws: induced movement in the "
          f"Dec-minus-March change ===")
    print("  frozen  = one eps per dong shared by both months (what p8 shows)")
    print("  unfrozen= an independent eps per month (pessimistic alternative)")
    res = []
    for a in AGES:
        sub = wide[(wide.age == a) & (wide.ym.isin(MONTHS))]
        p = sub.pivot(index="o_dong", columns="ym", values=["E", "tot"]).dropna()
        m_mar, m_dec = p[("tot", 202003)].values, p[("tot", 202012)].values
        s_mar = p[("E", 202003)].values / m_mar
        s_dec = p[("E", 202012)].values / m_dec
        base = (m_dec @ s_dec) / m_dec.sum() - (m_mar @ s_mar) / m_mar.sum()
        n = len(m_mar)
        sd = sigma_a[a]

        def shift(e_mar, e_dec):
            wm, wd = np.exp(e_mar) * m_mar, np.exp(e_dec) * m_dec
            return (wd @ s_dec) / wd.sum(1) - (wm @ s_mar) / wm.sum(1)

        e = RNG.normal(0, sd, size=(N_DRAWS, n))
        froz = shift(e, e) - base
        e2 = RNG.normal(0, sd, size=(N_DRAWS, n))
        unfroz = shift(e, e2) - base
        res.append(dict(
            age=a, band=AGE_LABEL[a], sigma=float(sd), n_dong=n,
            base_change_pp=float(base * 100),
            frozen_sd_pp=float(froz.std() * 100),
            frozen_p975_pp=float(np.percentile(np.abs(froz), 97.5) * 100),
            unfrozen_sd_pp=float(unfroz.std() * 100),
            unfrozen_p975_pp=float(np.percentile(np.abs(unfroz), 97.5) * 100),
            level_bound_pp=bound[AGE_LABEL[a]]))
    mc = pd.DataFrame(res)
    print(f"  {'band':<7} {'sigma':>6} {'change':>8} {'frozen SD':>10} "
          f"{'frozen 97.5':>12} {'unfrozen SD':>12} {'unfroz 97.5':>12}")
    for _, x in mc.iterrows():
        print(f"  {x.band:<7} {x.sigma:>6.3f} {x.base_change_pp:>+7.3f}p "
              f"{x.frozen_sd_pp:>10.4f} {x.frozen_p975_pp:>12.4f} "
              f"{x.unfrozen_sd_pp:>12.4f} {x.unfrozen_p975_pp:>12.4f}")
    out["monte_carlo"] = mc.to_dict("records")

    # ---------------------------------------------------- 5 the branch decision
    #
    # The plan's thresholds are absolute — 0.10 pp small, 0.25 pp large — but they
    # were set against p18's all-mtype E-share, whose 80+ change is +0.84 pp. The
    # H-origin estimator measured here has a 80+ change roughly four times that,
    # so the absolute numbers are not comparable and reading them straight off
    # would be a category error in whichever direction happened to flatter the
    # result. The transferable quantity is the RATIO of induced movement to
    # signal, and the plan's own thresholds convert to 0.10/0.84 = 11.9% and
    # 0.25/0.84 = 29.8%. Both readings are printed; the ratio is the one that
    # answers the question the thresholds were written to answer.
    P18_SIGNAL_PP = 0.844          # p18/p19's 80+ E-share change, Dec - Mar
    R_SMALL, R_LARGE = 0.10 / P18_SIGNAL_PP, 0.25 / P18_SIGNAL_PP
    print("\n=== 5 branch decision ===")
    print(f"  plan thresholds: <0.10 pp small, >0.25 pp large, set against p18's "
          f"80+ signal of {P18_SIGNAL_PP:+.3f} pp")
    print(f"  as ratios of signal that is <{R_SMALL:.1%} small, >{R_LARGE:.1%} large")

    def verdict(ind, sig):
        r = ind / abs(sig) if sig else np.inf
        return r, ("SMALL" if r < R_SMALL else "LARGE" if r > R_LARGE else "MIDDLE")

    mc["frozen_ratio"], mc["frozen_verdict"] = zip(
        *[verdict(i, s) for i, s in zip(mc.frozen_p975_pp, mc.base_change_pp)])
    mc["unfrozen_ratio"], mc["unfrozen_verdict"] = zip(
        *[verdict(i, s) for i, s in zip(mc.unfrozen_p975_pp, mc.base_change_pp)])
    print(f"\n  {'band':<7} {'signal':>8} {'induced':>8} {'ratio':>7} {'verdict':>8}"
          f"   {'unfrozen':>8} {'ratio':>7} {'verdict':>8}")
    for _, x in mc.iterrows():
        print(f"  {x.band:<7} {x.base_change_pp:>+7.3f}p {x.frozen_p975_pp:>8.4f} "
              f"{x.frozen_ratio:>7.1%} {x.frozen_verdict:>8}   "
              f"{x.unfrozen_p975_pp:>8.4f} {x.unfrozen_ratio:>7.1%} "
              f"{x.unfrozen_verdict:>8}")
    top = mc[mc.age == 80].iloc[0]
    print(f"\n  80+ (the band the thresholds were written about): induced "
          f"{top.frozen_p975_pp:.4f} pp on a signal of {top.base_change_pp:+.3f} pp "
          f"= {top.frozen_ratio:.1%} -> {top.frozen_verdict}")
    # How much bigger would the weight error have to be to change the answer?
    # Induced movement is very nearly linear in sigma, so this is a ratio.
    need = R_LARGE / top.frozen_ratio
    print(f"  sigma_d would have to be {need:.1f}x larger than this already "
          f"conservative bound to push 80+ into LARGE")
    bad = mc[mc.frozen_verdict != "SMALL"]
    print(f"\n  bands NOT in the small zone under frozen weights: "
          f"{', '.join(f'{x.band} ({x.frozen_ratio:.0%})' for _, x in bad.iterrows())
             or 'none'}")
    print("  These are the bands whose Dec-minus-March change is near zero, so a "
          "small absolute movement is a large share of it. The elderly bands "
          "carrying the headline are not among them.")
    out["branch"] = {
        "p18_signal_pp": P18_SIGNAL_PP,
        "ratio_thresholds": {"small_below": R_SMALL, "large_above": R_LARGE},
        "per_band": mc[["band", "base_change_pp", "frozen_p975_pp",
                        "frozen_ratio", "frozen_verdict", "unfrozen_p975_pp",
                        "unfrozen_ratio", "unfrozen_verdict"]].to_dict("records"),
        "eighty_plus_verdict": top.frozen_verdict,
        "sigma_multiple_to_flip": float(need),
        "caveats": [
            "sigma_d is the entire cross-dong spread of log rho-hat, so it "
            "contains real behavioural variation as well as weight error and "
            "overstates the latter. Every number here is an upper bound.",
            "The estimator is the H-origin E share, not p18's all-mtype share, "
            "because only H-origin trips reveal the residence dong the weight "
            "attaches to. Absolute pp are therefore not comparable to p18's.",
            "For W- and E-origin trips the travellers in one origin dong come "
            "from many residence dong, so their weight errors average and the "
            "perturbation attenuates. That makes the H-origin figure a "
            "conservative proxy for the all-mtype estimator rather than an "
            "unrelated one — an argument, not a proof, since it fails to the "
            "extent that people work near where they live.",
        ],
    }

    # ------------------------------- 6 route R4: the two derivations, multiplied
    # rho-hat and the expansion weight come from completely different places —
    # rho-hat from registered population, the weight from inverting the masking
    # threshold (p9_quantum 9.6) — and neither used the other. Dividing one by the
    # other backs out the quantity KT actually observed:
    #
    #     rho-hat / w  =  (expanded trips / residents) / (people per device)
    #                  =  observed moving KT devices per resident per day
    #
    # which has an external ceiling nothing in this project controls: it cannot
    # exceed KT's market share, roughly 0.3. That makes this a falsification test
    # rather than a consistency check — if either derivation were badly wrong the
    # product would run through the ceiling, and there is no free parameter to
    # absorb it.
    print("\n=== 6 R4: rho-hat / w = observed devices per resident per day ===")
    with open(f"{ROOT}/eda/results_p9.json") as fh:
        prof = {r["age"]: r for r in json.load(fh)["coverage_profile"]}
    KT_SHARE = 0.30
    lvl = (r[r.ym == 202001].groupby("age", as_index=False)
           .agg(V=("V_mid", "sum"), regpop=("regpop", "sum")))
    lvl["rho"] = lvl.V / lvl.regpop
    rows = []
    for _, x in lvl.iterrows():
        a = int(x.age)
        w = prof[a]["p10"]
        rows.append(dict(band=AGE_LABEL[a], rho=float(x.rho), w=float(w),
                         devices_per_resident=float(x.rho / w),
                         censored=bool(prof[a]["censored_band"])))
    r4 = pd.DataFrame(rows)
    print(f"  {'band':<7} {'rho':>7} {'w':>7} {'devices/resident':>18}")
    for _, x in r4.iterrows():
        flag = "  (w censored, upper bound)" if x.censored else ""
        print(f"  {x.band:<7} {x.rho:>7.3f} {x.w:>7.2f} "
              f"{x.devices_per_resident:>18.4f}{flag}")
    clean = r4[~r4.censored]
    over = clean[clean.devices_per_resident > KT_SHARE]
    print(f"\n  ceiling is KT's market share, ~{KT_SHARE:.0%}. Bands over it: "
          f"{list(over.band) if len(over) else 'none'} -> "
          f"{'INCONSISTENT' if len(over) else 'both derivations survive'}")
    print(f"  max among uncensored bands: {clean.devices_per_resident.max():.3f} "
          f"({clean.loc[clean.devices_per_resident.idxmax(), 'band']}), "
          f"which implies P(moves | has a KT phone) = "
          f"{clean.devices_per_resident.max() / KT_SHARE:.2f}")
    out["r4_device_rate"] = {"kt_share_assumed": KT_SHARE,
                             "by_age": r4.to_dict("records"),
                             "bands_over_ceiling": list(over.band)}

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(AGES))
    axes[0].bar(x, [sigma_a[a] for a in AGES], color="#3a6ea5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=45, fontsize=8)
    axes[0].set_ylabel(r"$\sigma_d$  (robust SD of $\log\hat\rho$ across dong)")
    axes[0].set_title("cross-dong spread, upper bound on weight error", fontsize=10)
    axes[1].bar(x - 0.2, mc.base_change_pp.abs(), 0.4, color="#c9c9c9",
                label="|E-share change| (the signal)")
    axes[1].bar(x + 0.2, mc.frozen_p975_pp, 0.4, color="#b03a48",
                label="induced movement, frozen (97.5th pct)")
    axes[1].plot(x + 0.2, mc.unfrozen_p975_pp, "k.", ms=5,
                 label="induced movement, unfrozen")
    axes[1].axhline(0.10, color="#3a6ea5", lw=.8, ls="--")
    axes[1].axhline(0.25, color="#b03a48", lw=.8, ls="--")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=45, fontsize=8)
    axes[1].set_ylabel("pp (log scale)")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("signal vs weight-induced movement, Dec - Mar", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/p21b_sigma.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p21b_sigma.png")

    with open(f"{ROOT}/eda/results_p21b.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p21b.json")


if __name__ == "__main__":
    main()
