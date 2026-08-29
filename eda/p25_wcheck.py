#!/usr/bin/env python
"""Phase 4 — can W be called a work panel, and for which age bands?

주간상주지 (W) is glossed as "workplace" everywhere in the current draft. The
manual (memo/phase1a-manual.md section 5) says it is 근무지 **또는 학교** — the
place with the most accumulated daytime stay, 05:00 to 05:00. So the label is
wrong in two independent directions and they must not be pooled:

  10-19   School is INSIDE W, not excluded. If the panel behaves like schools
          for these bands that is the product working as documented, and the fix
          is the panel's NAME, not its contents.
  65+     The advisor's attack: for a retiree the most-stayed-in daytime place
          can be a 경로당, a market or a hospital, and nothing in the definition
          stops it. That would be contamination, not mislabelling — the rows
          would not be workplaces under any name.

Six tests, each of which the two contaminations answer differently:

  A  Does W track jobs?      log HW arrivals per dong against 종사자수, with log
                             registered population as the competing predictor.
                             A workplace panel loads on jobs; a stay-near-home
                             panel loads on residents; a school panel loads on
                             residents too — which is why A alone cannot separate
                             10-19 from 65+ and B exists.
  B  Whose jobs?             Arrival-weighted KSIC division mix of the
                             destination dong. Education (P) high for 10-19 is
                             the school signature; health/social (Q) and retail
                             (G) high for 65+ is the advisor's signature.
  C  Hour of arrival         출근 peaks at 08. A daytime-hangout does not.
  D  Weekend persistence     Workplaces and schools empty out on Sunday.
  E  Same-dong "commute"     If 주간상주지 == 야간상주지 the trip never left the
                             residence dong. That is a defensible workplace only
                             by accident.
  F  Level against employment  W's share of home-origin departures, by age,
                             against the 고용률 age profile. Within an age band
                             the coverage factor cancels in a share, which is why
                             the share and not the per-resident rate is the
                             comparable object.

G turns C, D, E and B into a number. If the 65+ panel is a mixture
(1-c)·(work, distributed like the reference band) + c·(something else), then for
any feature the total variation distance between the 65+ and reference
distributions is c·TV(contaminant, reference) <= c. So TV is a LOWER bound on
contamination — but only if prime-age and elderly *workers* commute alike, which
they do not (part-time, later starts, different industries). It therefore bounds
"excess non-prime-age-commute-like volume", which is contamination plus genuine
elderly-work difference, and it is reported that way.

Months: 202606 is the primary — post-pandemic, school in session, and the
question is what W is NOW, not what it was under 거리두기. 202512 and 202001
are carried as robustness. Only holiday-free weekdays are used: 이동인구(합) is a
monthly SUM over each occurrence of a weekday, so a 공휴일 inside a (ym, dow)
cell cannot be removed after the fact — the whole weekday has to go. 202606's
Wednesday holds the 지방선거 임시공휴일 and 202001's Monday/Wednesday/Friday hold
설 연휴, which is why the clean-weekday sets differ by month.

Employment covariates stop at the 2024 survey year, so 202512 and 202606 are
joined to 2024 and 202001 to 2020. The parquet deliberately leaves 2025-2026
null rather than carrying 2024 forward (memo/phase1d-covariates.md section 5), so
the join year is named explicitly here rather than picked by nearest-year logic.

    python eda/p25_wcheck.py              # reuses the scan cache
    python eda/p25_wcheck.py --refresh    # rescans the parquet
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calendar_kr as K
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import AGE_LABEL, AGES
from paths import DERIVED, FIG, PARQUET, PARQUET_GLOB, ROOT, require
from scipy import stats

import duckdb

os.makedirs(FIG, exist_ok=True)

MONTHS = [202606, 202512, 202001]
PRIMARY = 202606
COV_YEAR = {202606: 2024, 202512: 2024, 202001: 2020}
SEOUL = "BETWEEN 1101000 AND 1125999"
IMPUTATIONS = {"lo": 0.0, "mid": 1.5, "hi": 3.0}
REF_AGE = 45                      # 45-49: prime working age, and p21b's lowest sigma_d
ELDERLY = [65, 70, 75, 80]
SCHOOL = [10, 15]
COMMUTE_HOURS = range(6, 10)      # 06:00-09:59 arrivals

# J 정보통신, K 금융보험, M 전문과학기술, N 사업시설관리·사업지원. L 부동산업 is
# left out on purpose: it is dominated by small neighbourhood agencies and tracks
# housing, not office floorspace, so including it would blunt exactly the
# contrast this index exists to draw.
OFFICE_DIV = ["J", "K", "M", "N"]
DIV_LABEL = {"G": "도소매", "I": "숙박·음식", "P": "교육", "Q": "보건·사회복지",
             "O": "공공행정", "R": "예술·스포츠", "S": "협회·수리·기타",
             "J": "정보통신", "K": "금융·보험", "M": "전문·과학·기술",
             "N": "사업시설·지원", "C": "제조", "F": "건설", "H": "운수·창고"}
WATCH_DIV = ["G", "I", "P", "Q", "R", "S", "O"]

ARR_CACHE = "p25_w_arrivals.parquet"
DEP_CACHE = "p25_h_departures.parquet"
out = {}


def clean_weekdays(ym):
    """Weekdays with no 공휴일 in any of that month's occurrences of them.

    A holiday cannot be netted out of a (ym, dow) cell because the published
    value is already summed over the month's occurrences of that weekday. The
    only honest move is to drop the weekday, so the usable set is per month.
    """
    return [d for d in range(1, 6) if K.cell_exposure(ym, d)["n_holiday"] == 0]


def scan(refresh=False):
    """Two aggregates per month off the same deduplicated rows.

    Dedup is on the FULL natural key (ym, dow_n, arr_hour, o_dong, d_dong, sex,
    age, mtype). The file carries whole-row duplicates (p16), and a short key
    silently collapses distinct cells instead of duplicates — an error that
    leaves every ratio looking right and only the level wrong, which is the way
    it survives review.

    Arrivals keep `same_dong` and the origin attribute because sections C/E need
    the 출근 flow (HW) separated from EW and WW, and the o_dong itself is not
    otherwise needed on the destination side.
    """
    a_path, d_path = DERIVED / ARR_CACHE, DERIVED / DEP_CACHE
    if not refresh and a_path.exists() and d_path.exists():
        arr, dep = pd.read_parquet(a_path), pd.read_parquet(d_path)
        if sorted(arr.ym.unique()) == sorted(MONTHS):
            print(f"  reusing {a_path.name} ({len(arr):,} rows) and "
                  f"{d_path.name} ({len(dep):,} rows); --refresh to rescan")
            return arr, dep
        print("  cache covers different months — rescanning")
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")      # the drive is shared with other jobs
    con.execute("PRAGMA disable_progress_bar")
    arr_f, dep_f = [], []
    for ym in MONTHS:
        dows = ",".join(map(str, clean_weekdays(ym) + [6, 7]))
        dedup = f"""
            SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                   min(pop) AS pop, bool_or(masked) AS masked
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
            WHERE ym = {ym} AND dow_n IN ({dows})"""
        arr_f.append(con.execute(f"""
            WITH g AS ({dedup} AND d_dong {SEOUL} GROUP BY ALL)
            SELECT ym, dow_n, arr_hour, d_dong, age,
                   substr(mtype, 1, 1) AS o_attr, substr(mtype, 2, 1) AS d_attr,
                   (o_dong = d_dong)   AS same_dong,
                   coalesce(sum(pop), 0)    AS v_obs,
                   count(*) FILTER (masked) AS n_masked
            FROM g GROUP BY 1,2,3,4,5,6,7,8""").df())
        dep_f.append(con.execute(f"""
            WITH g AS ({dedup} AND o_dong {SEOUL} AND mtype LIKE 'H%' GROUP BY ALL)
            SELECT ym, dow_n, o_dong, age, substr(mtype, 2, 1) AS d_attr,
                   (o_dong = d_dong)   AS same_dong,
                   coalesce(sum(pop), 0)    AS v_obs,
                   count(*) FILTER (masked) AS n_masked
            FROM g GROUP BY 1,2,3,4,5,6""").df())
        print(f"  {ym}  arrivals {len(arr_f[-1]):>9,} rows   "
              f"H-departures {len(dep_f[-1]):>8,} rows", flush=True)
    arr = pd.concat(arr_f, ignore_index=True)
    dep = pd.concat(dep_f, ignore_index=True)
    # The drive has unmounted mid-run once already this session; writing the
    # cache to a path that no longer exists must not look like a successful run.
    require(PARQUET, "parquet tree (did the drive unmount mid-scan?)")
    DERIVED.mkdir(parents=True, exist_ok=True)
    arr.to_parquet(a_path, index=False)
    dep.to_parquet(d_path, index=False)
    return arr, dep


def impute(df, tag="mid"):
    return df.v_obs + IMPUTATIONS[tag] * df.n_masked


def covariates(ym):
    """Dong-level 종사자수, 사업체수 and the KSIC division split for one month."""
    cov = pd.read_parquet(DERIVED / "dong_covariates.parquet")
    cov = cov[cov.year == COV_YEAR[ym]].copy()
    assert cov.worker.notna().all(), (
        f"employment is null for survey year {COV_YEAR[ym]} — the parquet leaves "
        f"2025-2026 empty on purpose, so COV_YEAR must name a surveyed year")
    return cov


def regpop_dong(ym):
    rp = pd.read_parquet(DERIVED / "regpop.parquet")
    rp = rp[(rp.ym == ym) & rp.dong_code.notna()].copy()
    rp["dong_code"] = rp.dong_code.astype("int64")
    return rp


def tv(p, q):
    """Total variation between two distributions over the same support."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    return 0.5 * np.abs(p - q).sum()


def std_ols(y, X):
    """Standardised coefficients, so 'jobs or residents' is a fair comparison.

    Both regressors are log counts on the same 424 dong and are correlated with
    each other; raw slopes are not comparable across age bands because the
    dispersion of log arrivals changes with age. Standardising puts every band on
    the same footing, which is the only thing this regression is asked to decide.
    """
    y = (y - y.mean()) / y.std(ddof=0)
    Z = (X - X.mean(0)) / X.std(0, ddof=0)
    Z = np.column_stack([np.ones(len(Z)), Z])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    r2 = 1 - ((y - Z @ b) ** 2).sum() / (y ** 2).sum()
    return b[1:], r2


def main():
    refresh = "--refresh" in sys.argv
    require(PARQUET, "parquet tree")
    require(DERIVED / "dong_covariates.parquet", "covariates (run eda/p20c_covariates.py)")
    require(DERIVED / "regpop.parquet", "registered population (run eda/p20_regpop.py)")
    require(DERIVED / "dong_crosswalk.parquet", "dong crosswalk")

    print("=== 0 scan: W-attribute arrivals and H-origin departures ===")
    for ym in MONTHS:
        print(f"  {ym}: clean weekdays {clean_weekdays(ym)} "
              f"(dropped {sorted(set(range(1, 6)) - set(clean_weekdays(ym)))} for 공휴일)")
    arr, dep = scan(refresh)
    days = {}
    for ym in MONTHS:
        wd = clean_weekdays(ym)
        days[(ym, "wd")] = sum(K.cell_exposure(ym, d)["n_days"] for d in wd)
        days[(ym, "we")] = sum(K.cell_exposure(ym, d)["n_days"] for d in (6, 7))
    arr["mid"] = impute(arr)
    dep["mid"] = impute(dep)
    wd_set = {ym: set(clean_weekdays(ym)) for ym in MONTHS}
    arr["is_wd"] = [d in wd_set[y] for y, d in zip(arr.ym, arr.dow_n)]
    dep["is_wd"] = [d in wd_set[y] for y, d in zip(dep.ym, dep.dow_n)]
    print(f"  arr_hour range {arr.arr_hour.min()}-{arr.arr_hour.max()}, "
          f"{arr.d_dong.nunique()} destination dong, {arr.age.nunique()} age bands")

    # ---------------------------------------------------------- 0b level check
    # Two anchors before anything is believed. p21 measured H-origin departures
    # per registered resident at 0.69 for 45-49 in 2020; if this script's
    # numerator is built wrong the level moves and the ratios do not, which is
    # exactly the failure mode that has passed review twice in this project.
    print("\n=== 0b level anchors ===")
    d20 = dep[(dep.ym == 202001) & dep.is_wd]
    rp20 = regpop_dong(202001)
    v45 = d20[d20.age == REF_AGE]["mid"].sum() / days[(202001, "wd")]
    n45 = rp20[rp20.age == REF_AGE]["pop"].sum()
    rho45 = v45 / n45
    print(f"  H-origin departures per registered resident per weekday, 45-49, "
          f"2020-01: {rho45:.3f}   (p21 reports 0.691 over all days of 2020)")
    assert 0.4 < rho45 < 1.2, f"rho-hat level {rho45:.3f} is not credible"
    cov_chk = covariates(PRIMARY)
    print(f"  Seoul 종사자수 {COV_YEAR[PRIMARY]}: {cov_chk.worker.sum():,} over "
          f"{len(cov_chk)} dong   (사업체조사 published total 5,772,456)")
    assert 4.5e6 < cov_chk.worker.sum() < 7e6, "employment total is off"
    hw_day = (arr[(arr.ym == PRIMARY) & arr.is_wd & (arr.o_attr == "H")
                  & (arr.d_attr == "W")]["mid"].sum() / days[(PRIMARY, "wd")])
    print(f"  HW arrivals into Seoul per weekday, {PRIMARY}: {hw_day:,.0f} "
          f"(Seoul has ~5.0-5.3M jobs; H->W is one leg of one day)")
    out["level_anchors"] = {"rho_h_origin_45_49_202001": float(rho45),
                            "seoul_workers": int(cov_chk.worker.sum()),
                            "hw_arrivals_per_weekday_primary": float(hw_day)}

    # ------------------------------------- 0c is the W place a single dong at all?
    # A person has one 주간상주지 per month, so a W->W trip should not change dong.
    # If it routinely does, the attribute is being assigned per trip endpoint and
    # every reading below that treats W as a person-level label is weaker.
    ww = arr[(arr.ym == PRIMARY) & (arr.o_attr == "W") & (arr.d_attr == "W")]
    ww_cross = 1 - ww[ww.same_dong]["mid"].sum() / ww["mid"].sum()
    print(f"\n=== 0c W->W trips that change dong: {ww_cross:.1%} ===")
    print("  A month has one 주간상주지 per device, so this should be near zero if "
          "the label were person-level; it is not, so W marks a trip endpoint "
          "class, and 'the W panel' means 'trips ending at a W-classified place'.")
    out["ww_cross_dong_share"] = float(ww_cross)

    # =========================================================== A  W vs jobs
    print("\n=== A does W track 종사자수? (HW arrivals by destination dong) ===")
    rows_a, scat = [], {}
    for ym in MONTHS:
        cov = covariates(ym)[["dong_code", "worker", "estab", "area_km2_ref"]]
        rp = (regpop_dong(ym).groupby("dong_code", as_index=False)["pop"].sum()
              .rename(columns={"pop": "regpop"}))
        base = (arr[(arr.ym == ym) & arr.is_wd & (arr.o_attr == "H")
                    & (arr.d_attr == "W")]
                .groupby(["d_dong", "age"], as_index=False)["mid"].sum())
        for a in AGES:
            g = (base[base.age == a].rename(columns={"d_dong": "dong_code"})
                 .merge(cov, on="dong_code", how="right")
                 .merge(rp, on="dong_code", how="left"))
            g["mid"] = g["mid"].fillna(0.0)
            g = g[(g.worker > 0) & (g.regpop > 0)]
            y = np.log(g["mid"] + 1.0)
            lw, lr = np.log(g.worker.astype(float)), np.log(g.regpop.astype(float))
            b, r2 = std_ols(y.values, np.column_stack([lw, lr]))
            rows_a.append({
                "ym": ym, "age": AGE_LABEL[a], "n_dong": int(len(g)),
                "r_worker": float(stats.pearsonr(y, lw)[0]),
                "r_regpop": float(stats.pearsonr(y, lr)[0]),
                "beta_worker": float(b[0]), "beta_regpop": float(b[1]),
                "r2": float(r2),
                "jobs_lead": float(b[0] - b[1])})
            if ym == PRIMARY:
                scat[a] = (lw.values, y.values)
    A = pd.DataFrame(rows_a)
    ap = A[A.ym == PRIMARY].set_index("age")
    print(f"  {PRIMARY}, 424 dong, standardised coefficients on log 종사자수 and "
          f"log 등록인구")
    print(f"  {'band':<7} {'r(job)':>7} {'r(pop)':>7} {'b(job)':>7} {'b(pop)':>7} "
          f"{'R2':>6} {'jobs lead':>10}")
    for a in AGES:
        r = ap.loc[AGE_LABEL[a]]
        print(f"  {AGE_LABEL[a]:<7} {r.r_worker:>7.3f} {r.r_regpop:>7.3f} "
              f"{r.beta_worker:>7.3f} {r.beta_regpop:>7.3f} {r.r2:>6.3f} "
              f"{r.jobs_lead:>+10.3f}")
    ref_lead = ap.loc[AGE_LABEL[REF_AGE], "jobs_lead"]
    print(f"\n  reference {AGE_LABEL[REF_AGE]}: jobs lead residents by "
          f"{ref_lead:+.3f} sd")
    for a in ELDERLY + SCHOOL:
        print(f"    {AGE_LABEL[a]:<7} {ap.loc[AGE_LABEL[a], 'jobs_lead']:+.3f} "
              f"({ap.loc[AGE_LABEL[a], 'jobs_lead'] - ref_lead:+.3f} vs reference)")
    out["A_jobs_vs_residents"] = A.to_dict("records")

    # ======================================================= B  whose jobs
    print("\n=== B arrival-weighted KSIC division mix of the destination dong ===")
    cov = covariates(PRIMARY)
    divs = [c[len("worker_"):] for c in cov.columns
            if c.startswith("worker_") and len(c) == len("worker_") + 1]
    share = cov[["dong_code"]].copy()
    for d in divs:
        share[d] = cov[f"worker_{d}"].astype(float) / cov.worker.astype(float)
    share["OFFICE"] = share[OFFICE_DIV].sum(1)
    city = {d: float(cov[f"worker_{d}"].sum() / cov.worker.sum()) for d in divs}
    city["OFFICE"] = float(sum(city[d] for d in OFFICE_DIV))
    base = (arr[(arr.ym == PRIMARY) & arr.is_wd & (arr.d_attr == "W")]
            .groupby(["d_dong", "age"], as_index=False)["mid"].sum()
            .rename(columns={"d_dong": "dong_code"}))
    expo = []
    for a in AGES:
        g = base[base.age == a].merge(share, on="dong_code", how="inner")
        w = g["mid"].values
        row = {"age": AGE_LABEL[a], "volume": float(w.sum())}
        for d in WATCH_DIV + ["OFFICE"]:
            row[d] = float(np.average(g[d].values, weights=w))
        expo.append(row)
    B = pd.DataFrame(expo).set_index("age")
    cols = ["OFFICE", "G", "I", "P", "Q", "R", "S", "O"]
    print("  exposure = share of the destination dong's 종사자 in that division, "
          "averaged over W arrivals")
    print(f"  {'band':<7} " + " ".join(f"{c:>7}" for c in cols))
    print(f"  {'CITY':<7} " + " ".join(f"{city[c]:>7.3f}" for c in cols)
          + "   <- if arrivals were spread like Seoul's jobs")
    for a in AGES:
        print(f"  {AGE_LABEL[a]:<7} " + " ".join(f"{B.loc[AGE_LABEL[a], c]:>7.3f}"
                                                 for c in cols))
    ref = B.loc[AGE_LABEL[REF_AGE]]
    print(f"\n  deviation from {AGE_LABEL[REF_AGE]} (pp):")
    print(f"  {'band':<7} " + " ".join(f"{c:>7}" for c in cols))
    dev = {}
    for a in SCHOOL + ELDERLY:
        dev[AGE_LABEL[a]] = {c: float((B.loc[AGE_LABEL[a], c] - ref[c]) * 100)
                             for c in cols}
        print(f"  {AGE_LABEL[a]:<7} " + " ".join(
            f"{dev[AGE_LABEL[a]][c]:>+7.2f}" for c in cols))
    out["B_division_exposure"] = {"city_job_mix": city,
                                  "by_age": B.reset_index().to_dict("records"),
                                  "deviation_from_ref_pp": dev,
                                  "office_divisions": OFFICE_DIV}

    # ================================================= C  hour of arrival
    print("\n=== C hour-of-arrival profile of HW (출근) arrivals ===")
    hr = (arr[(arr.ym == PRIMARY) & arr.is_wd & (arr.o_attr == "H")
              & (arr.d_attr == "W")]
          .groupby(["age", "arr_hour"], as_index=False)["mid"].sum())
    H = hr.pivot(index="arr_hour", columns="age", values="mid").fillna(0.0)
    H = H.reindex(range(24), fill_value=0.0)
    Hn = H / H.sum(0)
    rows_c = []
    for a in AGES:
        s = Hn[a]
        rows_c.append({"age": AGE_LABEL[a], "peak_hour": int(s.idxmax()),
                       "peak_share": float(s.max()),
                       "share_06_09": float(s.loc[list(COMMUTE_HOURS)].sum()),
                       "share_10_15": float(s.loc[list(range(10, 16))].sum()),
                       "median_hour": float(np.interp(
                           0.5, s.cumsum().values, np.arange(24)))})
    C = pd.DataFrame(rows_c).set_index("age")
    print(f"  {'band':<7} {'peak':>5} {'peak sh':>8} {'06-09':>7} {'10-15':>7} "
          f"{'median h':>9}")
    for a in AGES:
        r = C.loc[AGE_LABEL[a]]
        print(f"  {AGE_LABEL[a]:<7} {int(r.peak_hour):>5} {r.peak_share:>8.3f} "
              f"{r.share_06_09:>7.3f} {r.share_10_15:>7.3f} {r.median_hour:>9.2f}")
    out["C_hour_profile"] = C.reset_index().to_dict("records")
    out["C_hour_shares"] = {AGE_LABEL[a]: Hn[a].round(5).tolist() for a in AGES}

    # ================================================ D  weekend persistence
    print("\n=== D weekend / weekday ratio of W arrivals ===")
    rows_d = []
    for a in AGES:
        g = arr[(arr.ym == PRIMARY) & (arr.age == a) & (arr.d_attr == "W")]
        wdv = g[g.is_wd]["mid"].sum() / days[(PRIMARY, "wd")]
        wev = g[~g.is_wd]["mid"].sum() / days[(PRIMARY, "we")]
        gs = g[g.o_attr == "H"]
        wdh = gs[gs.is_wd]["mid"].sum() / days[(PRIMARY, "wd")]
        weh = gs[~gs.is_wd]["mid"].sum() / days[(PRIMARY, "we")]
        rows_d.append({"age": AGE_LABEL[a], "weekday_W": float(wdv),
                       "weekend_W": float(wev), "ratio_W": float(wev / wdv),
                       "ratio_HW": float(weh / wdh)})
    D = pd.DataFrame(rows_d).set_index("age")
    print(f"  {'band':<7} {'wd/day':>12} {'we/day':>12} {'we:wd all W':>12} "
          f"{'we:wd HW':>10}")
    for a in AGES:
        r = D.loc[AGE_LABEL[a]]
        print(f"  {AGE_LABEL[a]:<7} {r.weekday_W:>12,.0f} {r.weekend_W:>12,.0f} "
              f"{r.ratio_W:>12.3f} {r.ratio_HW:>10.3f}")
    out["D_weekend"] = D.reset_index().to_dict("records")

    # ================================================ E  same-dong "commute"
    print("\n=== E share of HW arrivals that never left the residence dong ===")
    rows_e = []
    for ym in MONTHS:
        g = arr[(arr.ym == ym) & arr.is_wd & (arr.o_attr == "H")
                & (arr.d_attr == "W")]
        tot = g.groupby("age")["mid"].sum()
        same = g[g.same_dong].groupby("age")["mid"].sum()
        for a in AGES:
            rows_e.append({"ym": ym, "age": AGE_LABEL[a],
                           "same_dong_share": float(same.get(a, 0.0) / tot[a])})
    E = pd.DataFrame(rows_e)
    ep = E.pivot(index="age", columns="ym", values="same_dong_share")
    print(f"  {'band':<7} " + " ".join(f"{y:>9}" for y in MONTHS))
    for a in AGES:
        print(f"  {AGE_LABEL[a]:<7} " + " ".join(
            f"{ep.loc[AGE_LABEL[a], y]:>9.3f}" for y in MONTHS))
    out["E_same_dong"] = E.to_dict("records")

    # =========================================== F  level against employment
    # Within an age band the expansion weight and the observable-and-moving share
    # multiply BOTH the numerator and the denominator of a destination share, so
    # they cancel. The per-resident rate does not have that property, which is why
    # the share is the object compared to 고용률 and the rate is only printed
    # alongside it.
    print("\n=== F W's share of home-origin departures, against 고용률 ===")
    rows_f = []
    for ym in MONTHS:
        g = dep[(dep.ym == ym) & dep.is_wd]
        tot = g.groupby("age")["mid"].sum()
        w = g[g.d_attr == "W"].groupby("age")["mid"].sum()
        rp = regpop_dong(ym).groupby("age")["pop"].sum()
        for a in AGES:
            rows_f.append({"ym": ym, "age": AGE_LABEL[a],
                           "w_share_of_H_departures": float(w[a] / tot[a]),
                           "hw_per_resident_per_day": float(
                               w[a] / days[(ym, "wd")] / rp[a])})
    F = pd.DataFrame(rows_f)
    fp = F[F.ym == PRIMARY].set_index("age")
    # 고용률 by age, 통계청 경제활동인구조사, 2024 annual, nationwide. Held here as a
    # SHAPE reference only: it is national rather than Seoul, and 2024 rather than
    # 2026, so any claim rests on the age ordering and not on the individual
    # levels. The school bands have no employment analogue at all — their
    # counterpart is 취학률, which is ~99% for 10-17.
    EMP = {15: 0.077, 20: 0.463, 25: 0.716, 30: 0.803, 35: 0.813, 40: 0.812,
           45: 0.804, 50: 0.784, 55: 0.740, 60: 0.640, 65: 0.487, 70: 0.400,
           75: 0.264, 80: 0.130}
    print(f"  {'band':<7} {'W share':>9} {'HW/resident':>12} {'고용률 2024':>11} "
          f"{'W share / 고용률':>16}")
    for a in AGES:
        r = fp.loc[AGE_LABEL[a]]
        e = EMP.get(a)
        rel = f"{r.w_share_of_H_departures / e:>16.2f}" if e else f"{'—':>16}"
        print(f"  {AGE_LABEL[a]:<7} {r.w_share_of_H_departures:>9.3f} "
              f"{r.hw_per_resident_per_day:>12.3f} "
              f"{(f'{e:.3f}' if e else '—'):>11} {rel}")
    corr = stats.spearmanr(
        [fp.loc[AGE_LABEL[a], "w_share_of_H_departures"] for a in EMP],
        [EMP[a] for a in EMP])
    print(f"\n  Spearman(W share, 고용률) over the 14 bands with an employment "
          f"analogue: {corr[0]:.3f} (p={corr[1]:.4f})")
    work_age = [a for a in EMP if 25 <= a <= 59]
    corr_w = stats.spearmanr(
        [fp.loc[AGE_LABEL[a], "w_share_of_H_departures"] for a in work_age],
        [EMP[a] for a in work_age])
    print(f"  restricted to 25-59, where employment actually varies: "
          f"{corr_w[0]:.3f} (p={corr_w[1]:.4f})")
    out["F_employment"] = {"by_month": F.to_dict("records"),
                           "employment_rate_2024_national": EMP,
                           "spearman_all": [float(corr[0]), float(corr[1])],
                           "spearman_25_59": [float(corr_w[0]), float(corr_w[1])]}

    # ========================================== G  bounding the contamination
    print("\n=== G total-variation lower bounds against the "
          f"{AGE_LABEL[REF_AGE]} commute ===")
    off = share[["dong_code", "OFFICE"]].copy()
    off["q"] = pd.qcut(off.OFFICE, 5, labels=False)
    rows_g = []
    for tag in IMPUTATIONS:
        g = arr[(arr.ym == PRIMARY) & arr.is_wd & (arr.o_attr == "H")
                & (arr.d_attr == "W")].copy()
        g["v"] = impute(g, tag)
        g = g.merge(off[["dong_code", "q"]], left_on="d_dong",
                    right_on="dong_code", how="left")
        hour = g.pivot_table(index="arr_hour", columns="age", values="v",
                             aggfunc="sum").reindex(range(24)).fillna(0.0)
        joint = g.assign(cell=g.arr_hour.astype(str) + "|"
                         + g.same_dong.astype(int).astype(str)).pivot_table(
            index="cell", columns="age", values="v", aggfunc="sum").fillna(0.0)
        dest = g.pivot_table(index="q", columns="age", values="v",
                             aggfunc="sum").fillna(0.0)
        for a in AGES:
            rows_g.append({
                "imputation": tag, "age": AGE_LABEL[a],
                "tv_hour": float(tv(hour[a], hour[REF_AGE])),
                "tv_hour_x_samedong": float(tv(joint[a], joint[REF_AGE])),
                "tv_dest_officeness": float(tv(dest[a], dest[REF_AGE]))})
    G = pd.DataFrame(rows_g)
    gm = G[G.imputation == "mid"].set_index("age")
    print(f"  {'band':<7} {'TV hour':>8} {'TV hour x same':>15} {'TV dest mix':>12} "
          f"{'TV hour lo..hi':>16}")
    for a in AGES:
        r = gm.loc[AGE_LABEL[a]]
        lo = G[(G.imputation == "lo") & (G.age == AGE_LABEL[a])].tv_hour.iloc[0]
        hi = G[(G.imputation == "hi") & (G.age == AGE_LABEL[a])].tv_hour.iloc[0]
        print(f"  {AGE_LABEL[a]:<7} {r.tv_hour:>8.3f} {r.tv_hour_x_samedong:>15.3f} "
              f"{r.tv_dest_officeness:>12.3f} {f'{lo:.3f}..{hi:.3f}':>16}")
    out["G_tv_bounds"] = G.to_dict("records")

    # The bound is on "volume that does not look like the reference commute". For
    # 10-19 that is the school signature and is expected; for 65+ it is what the
    # advisor asked to be quantified. Reporting them in the same table with the
    # same statistic is the point — the number does not discriminate, section B's
    # division mix and section C's direction do.
    print(f"\n  as a share of each band's W panel volume ({PRIMARY}, weekdays):")
    vol = (arr[(arr.ym == PRIMARY) & arr.is_wd & (arr.d_attr == "W")]
           .groupby("age")["mid"].sum())
    verdict = {}
    for a in SCHOOL + ELDERLY:
        r = gm.loc[AGE_LABEL[a]]
        verdict[AGE_LABEL[a]] = {
            "panel_volume_per_weekday": float(vol[a] / days[(PRIMARY, "wd")]),
            "tv_hour": float(r.tv_hour),
            "tv_hour_x_samedong": float(r.tv_hour_x_samedong),
            "tv_dest_officeness": float(r.tv_dest_officeness),
            "implied_min_atypical_volume_per_weekday": float(
                vol[a] / days[(PRIMARY, "wd")] * r.tv_hour_x_samedong)}
        v = verdict[AGE_LABEL[a]]
        print(f"    {AGE_LABEL[a]:<7} panel {v['panel_volume_per_weekday']:>10,.0f}/day"
              f"   >= {v['tv_hour_x_samedong']:.1%} atypical = "
              f"{v['implied_min_atypical_volume_per_weekday']:>9,.0f}/day")
    out["G_verdict"] = verdict

    # ------------------------------------------------------------------ figures
    cmap = plt.get_cmap("viridis")
    col = {a: cmap(i / (len(AGES) - 1)) for i, a in enumerate(AGES)}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for a, mk in [(15, "o"), (REF_AGE, "s"), (80, "^")]:
        lw, y = scat[a]
        axes[0].scatter(lw, y, s=7, alpha=.45, color=col[a], marker=mk,
                        label=f"{AGE_LABEL[a]}  r={ap.loc[AGE_LABEL[a], 'r_worker']:.2f}")
    axes[0].set_xlabel("log 종사자수 (dong)")
    axes[0].set_ylabel("log HW arrivals per weekday")
    axes[0].set_title(f"W arrivals against jobs, {PRIMARY} weekdays, 424 dong",
                      fontsize=10)
    axes[0].legend(fontsize=8)
    x = np.arange(len(AGES))
    axes[1].bar(x - .2, [ap.loc[AGE_LABEL[a], "beta_worker"] for a in AGES],
                .4, color="#3a6ea5", label="log 종사자수")
    axes[1].bar(x + .2, [ap.loc[AGE_LABEL[a], "beta_regpop"] for a in AGES],
                .4, color="#b03a48", label="log 등록인구")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=45, fontsize=8)
    axes[1].set_ylabel("standardised coefficient")
    axes[1].set_title("what W arrivals load on, by age", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/p25_w_vs_employment.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p25_w_vs_employment.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for a in AGES:
        axes[0].plot(range(24), Hn[a], lw=1.4, color=col[a], label=AGE_LABEL[a])
    axes[0].axvspan(6, 10, color="k", alpha=.06)
    axes[0].set_xlabel("arrival hour"); axes[0].set_ylabel("share of HW arrivals")
    axes[0].set_title(f"출근 arrival hour by age, {PRIMARY} weekdays", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=len(AGES) - 1))
    cb = fig.colorbar(sm, ax=axes[0], ticks=range(0, len(AGES), 3))
    cb.ax.set_yticklabels([AGE_LABEL[AGES[i]] for i in range(0, len(AGES), 3)],
                          fontsize=8)
    axes[1].bar(x, [C.loc[AGE_LABEL[a], "share_06_09"] for a in AGES],
                color=[col[a] for a in AGES])
    axes[1].axhline(C.loc[AGE_LABEL[REF_AGE], "share_06_09"], color="k", lw=.8,
                    ls="--", label=f"{AGE_LABEL[REF_AGE]} reference")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=45, fontsize=8)
    axes[1].set_ylabel("share arriving 06:00-09:59")
    axes[1].set_title("commute-window share", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/p25_hour_profile.png", dpi=150)
    plt.close(fig)
    print("  -> fig/p25_hour_profile.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    show = ["OFFICE", "G", "I", "P", "Q"]
    lab = {"OFFICE": "office (J,K,M,N)", **{d: DIV_LABEL[d] for d in show[1:]}}
    for j, d in enumerate(show):
        axes[0].plot(x, [B.loc[AGE_LABEL[a], d] for a in AGES], marker="o", ms=3,
                     lw=1.3, label=lab[d])
        axes[0].axhline(city[d], color=f"C{j}", lw=.6, ls=":")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=45, fontsize=8)
    axes[0].set_ylabel("arrival-weighted division share")
    axes[0].set_title("whose jobs the W arrivals land among\n"
                      "(dotted = Seoul's own job mix)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [gm.loc[AGE_LABEL[a], "tv_hour"] for a in AGES], marker="o",
                 ms=3, lw=1.3, label="TV(hour)")
    axes[1].plot(x, [gm.loc[AGE_LABEL[a], "tv_hour_x_samedong"] for a in AGES],
                 marker="s", ms=3, lw=1.3, label="TV(hour x same-dong)")
    axes[1].plot(x, [ep.loc[AGE_LABEL[a], PRIMARY] for a in AGES], marker="^",
                 ms=3, lw=1.3, label="same-dong share")
    axes[1].plot(x, [D.loc[AGE_LABEL[a], "ratio_HW"] for a in AGES], marker="v",
                 ms=3, lw=1.3, label="weekend : weekday")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=45, fontsize=8)
    axes[1].set_title(f"distance from the {AGE_LABEL[REF_AGE]} commute", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/p25_contamination.png", dpi=150)
    plt.close(fig)
    print("  -> fig/p25_contamination.png")

    with open(f"{ROOT}/eda/results_p25.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p25.json")


if __name__ == "__main__":
    main()
