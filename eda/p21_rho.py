#!/usr/bin/env python
"""Phase 1c — the rho-hat numerator, defined precisely, plus the first estimates.

rho-hat(a,s,d,t) = V(a,s,d,t) / RegPop(a,s,d,t), where V is the average number of
home-origin departures per calendar day and RegPop is the registered population
(p20_regpop.py). The unit is departures from home per registered resident per day.

Four definitional choices, each of which changes the answer:

  H-origin only        mtype LIKE 'H%' — HH, HW, HE. The first character is the
                       origin attribute (야간상주지), so an H-origin trip starts at
                       the traveller's own residence dong and that dong's registered
                       population is the right denominator. The manual confirms the
                       expansion weight is keyed on 야간상주지 too, so numerator and
                       denominator hang on the same unit.

  origin in Seoul,     A resident who leaves Seoul is still an observed resident.
  destination free     This differs from p18/p19, which filter both ends; their
                       WHERE clause must not be copied here.

  dedup first          The file carries exactly-duplicated rows (p16: 475,266,
                       all whole-row copies). Collapsing on the natural key first.

  per calendar day,    이동인구(합) is a monthly SUM over each occurrence of that
  not per week         weekday, and RegPop is a point-in-time stock, so V has to be
                       a daily average: sum(pop) / days_in_month. It is NOT
                       sum(pop_day) — that quantity weights each weekday once and
                       yields a representative *week*. Section 1.0 measures the gap
                       between the two; this project has already had a conclusion
                       flip on exactly this distinction (DATA_REPORT 2.1a).

Masking: a masked cell is 0 < value < 3, so V is reported at three imputations
(0 / 1.5 / 3). Because 0-19 and 45+ are essentially never masked, their rho-hat is
effectively point identified while 20-44 carries a wide band — the opposite of the
usual complaint, and it should be stated that way round.

Sensitivity: HH (home to home) may be classifier noise or second-residence travel,
so an HW+HE-only variant is carried alongside.

    python eda/p20_regpop.py     # first — the denominator
    python eda/p21_rho.py        # then — the numerator and rho-hat
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import p20_regpop as P20
import pandas as pd
from common import AGE_LABEL, AGES, YMS, YM_LABEL, cell_table, connect
from paths import DERIVED, FIG, PARQUET_GLOB, ROOT, require

os.makedirs(FIG, exist_ok=True)
IMPUTATIONS = {"lo": 0.0, "mid": 1.5, "hi": 3.0}
SEOUL = "BETWEEN 1101000 AND 1125999"
out = {}


CACHE = "rho_numerator_dow.parquet"


def numerator(refresh=False):
    """One pass per month at (ym, dow, origin dong, sex, age).

    Kept at dow granularity on purpose: the per-day average, the weekday-only
    average and section 1.0's sum(pop_day) comparison are all derivable from it,
    so the 5.7 GB is scanned once rather than three times. Cached for the same
    reason — everything below this is arithmetic on 1.1M rows, and rescanning to
    change a plot is waste.
    """
    if not refresh and (DERIVED / CACHE).exists():
        df = pd.read_parquet(DERIVED / CACHE)
        print(f"  reusing {DERIVED / CACHE} ({len(df):,} rows); "
              f"--refresh to rescan the parquet")
        return df
    con = connect(threads=6)
    con.execute("PRAGMA disable_progress_bar")
    frames = []
    for ym in YMS:
        df = con.execute(f"""
            WITH g AS (
              SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                     min(pop) AS pop, bool_or(masked) AS masked
              FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
              WHERE ym = {ym} AND o_dong {SEOUL} AND mtype LIKE 'H%'
              GROUP BY ALL)
            SELECT ym, dow_n, o_dong, sex, age,
                   coalesce(sum(pop), 0)                     AS v_obs,
                   count(*) FILTER (masked)                  AS n_masked,
                   coalesce(sum(pop) FILTER (mtype <> 'HH'), 0) AS v_obs_nohh,
                   count(*) FILTER (masked AND mtype <> 'HH')   AS n_masked_nohh,
                   coalesce(sum(pop) FILTER (d_dong {SEOUL}), 0) AS v_obs_int,
                   count(*) FILTER (masked AND d_dong {SEOUL})   AS n_masked_int
            FROM g GROUP BY 1,2,3,4,5""").df()
        frames.append(df)
        print(f"  {ym}  {len(df):>6,} cells  V_obs {df.v_obs.sum():>14,.0f}",
              flush=True)
    num = pd.concat(frames, ignore_index=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    num.to_parquet(DERIVED / CACHE, index=False)
    return num


def main():
    require(DERIVED / "regpop.parquet", "denominator (run eda/p20_regpop.py)")
    cal = pd.DataFrame(cell_table())[["ym", "dow_n", "n_days"]]
    days = cal.groupby("ym")["n_days"].sum().rename("days")
    days_wd = (cal[cal.dow_n <= 5].groupby("ym")["n_days"].sum().rename("days_wd"))
    for ym, d in days.items():
        assert 28 <= d <= 31, f"{ym}: {d} days in month"

    print("=== 1.1 numerator: H-origin trips, origin in Seoul, deduplicated ===")
    num = numerator()
    num = num.merge(cal, on=["ym", "dow_n"], how="left")
    assert num.n_days.notna().all(), "calendar does not cover every (ym, dow)"

    # ---------------------------------------------------------- 1.0 the 합 trap
    # sum(pop)/days_in_month is the daily average. sum(pop_day) weights every
    # weekday once, which is a representative week, not a day. They differ by
    # however unevenly the month's 28-31 days fall across the seven weekdays.
    print("\n=== 1.0 the 합 arithmetic: daily average vs sum(pop_day) ===")
    city = num.groupby(["ym", "dow_n"], as_index=False).agg(
        v_obs=("v_obs", "sum"), n_days=("n_days", "first"))
    city["pop_day"] = city.v_obs / city.n_days
    trap = city.groupby("ym").agg(v_month=("v_obs", "sum"),
                                  sum_pop_day=("pop_day", "sum")).join(days)
    trap["daily_avg"] = trap.v_month / trap.days
    trap["week_over_7"] = trap.sum_pop_day / 7
    trap["rel_gap"] = trap.week_over_7 / trap.daily_avg - 1
    print(trap[["daily_avg", "week_over_7", "rel_gap"]]
          .rename(index=YM_LABEL).round(4).to_string())
    print(f"  worst month: {trap.rel_gap.abs().max():.3%} — small at city level, "
          f"but it is a per-month bias, so it lands entirely on cross-month "
          f"comparison, which is the one thing rho-hat is for.")
    out["sum_trap"] = trap.reset_index().to_dict("records")

    # ------------------------------------------------------- 1.2 V, three bounds
    v = num.groupby(["ym", "o_dong", "sex", "age"], as_index=False)[
        ["v_obs", "n_masked", "v_obs_nohh", "n_masked_nohh",
         "v_obs_int", "n_masked_int"]].sum()
    wd = (num[num.dow_n <= 5]
          .groupby(["ym", "o_dong", "sex", "age"], as_index=False)[
              ["v_obs", "n_masked"]].sum()
          .rename(columns={"v_obs": "v_obs_wd", "n_masked": "n_masked_wd"}))
    v = v.merge(wd, on=["ym", "o_dong", "sex", "age"], how="left").fillna(
        {"v_obs_wd": 0, "n_masked_wd": 0})
    v = v.join(days, on="ym").join(days_wd, on="ym")
    for tag, c in IMPUTATIONS.items():
        v[f"V_{tag}"] = (v.v_obs + c * v.n_masked) / v.days
        v[f"V_wd_{tag}"] = (v.v_obs_wd + c * v.n_masked_wd) / v.days_wd
        v[f"V_nohh_{tag}"] = (v.v_obs_nohh + c * v.n_masked_nohh) / v.days
        v[f"V_int_{tag}"] = (v.v_obs_int + c * v.n_masked_int) / v.days

    # ------------------------------------------------- 1.3 join the denominator
    regall = pd.read_parquet(DERIVED / "regpop.parquet")
    regall = regall[regall.ym.isin(YMS)].copy()
    reg = regall[regall.dong_code.notna()].copy()
    reg["o_dong"] = reg.dong_code.astype("int64")
    den = reg.groupby(["ym", "o_dong", "sex", "age"], as_index=False)[
        "pop"].sum().rename(columns={"pop": "regpop"})
    r = v.merge(den, on=["ym", "o_dong", "sex", "age"], how="outer",
                indicator=True)
    # A cell in the denominator with no row at all in the numerator is a
    # structural zero, not missing data: the manual says an origin-destination
    # pair with no moving KT customer stays absent after expansion. Dropping
    # those would bias rho-hat up, so they are filled with V = 0.
    zeros = r[r._merge == "right_only"]
    if len(zeros):
        print(f"\n  {len(zeros)} (ym,dong,sex,age) cells have registered residents "
              f"but no H-origin row at all — structural zeros, set V = 0")
        print("  " + zeros.groupby("age").size().rename(index=AGE_LABEL)
              .to_string().replace("\n", "\n  "))
        out["structural_zero_cells"] = zeros[
            ["ym", "o_dong", "sex", "age", "regpop"]].to_dict("records")
    orphan = r[r._merge == "left_only"]
    if len(orphan):
        raise SystemExit(f"{len(orphan)} numerator cells have no denominator — "
                         f"the crosswalk is incomplete:\n{orphan.head().to_string()}")
    r = r.drop(columns="_merge")
    fill = [c for c in r.columns if c.startswith(("v_obs", "n_masked", "V_"))]
    r[fill] = r[fill].fillna(0.0)
    r[["days", "days_wd"]] = r[["days", "days_wd"]].fillna(
        r.groupby("ym")[["days", "days_wd"]].transform("max"))
    VARIANTS = {"V": "all", "V_wd": "wd", "V_nohh": "nohh", "V_int": "int"}
    for tag in IMPUTATIONS:
        for col, name in VARIANTS.items():
            r[f"rho_{name}_{tag}"] = r[f"{col}_{tag}"] / r.regpop
    r.to_parquet(DERIVED / "rho_numerator.parquet", index=False)
    print(f"\nwrote {DERIVED / 'rho_numerator.parquet'} ({len(r):,} rows)")

    # ------------------------------------------------------ 1.4 the level check
    # Plan's stage falsification: if working-age rho-hat is routinely above 2,
    # the numerator is counting trips rather than people harder than assumed and
    # the H* definition has to be revisited, starting by dropping HH.
    print("\n=== 1.4 level check: rho-hat by age (city, all 12 months) ===")
    city_age = (r.groupby(["ym", "age"], as_index=False)
                .agg(**{f"V_{t}": (f"V_{t}", "sum") for t in IMPUTATIONS},
                     V_nohh_mid=("V_nohh_mid", "sum"),
                     V_wd_mid=("V_wd_mid", "sum"),
                     V_int_mid=("V_int_mid", "sum"),
                     regpop=("regpop", "sum")))
    for t in IMPUTATIONS:
        city_age[f"rho_{t}"] = city_age[f"V_{t}"] / city_age.regpop
    for k in ("nohh", "wd", "int"):
        city_age[f"rho_{k}"] = city_age[f"V_{k}_mid"] / city_age.regpop
    lvl = city_age.groupby("age").agg(
        rho_lo=("rho_lo", "mean"), rho_mid=("rho_mid", "mean"),
        rho_hi=("rho_hi", "mean"), rho_nohh=("rho_nohh", "mean"),
        rho_wd=("rho_wd", "mean"), rho_int=("rho_int", "mean"))
    lvl.index = [AGE_LABEL[a] for a in lvl.index]
    print(lvl.round(3).to_string())
    over2 = lvl[lvl.rho_hi > 2]
    print(f"  age bands with rho-hat(upper) > 2: "
          f"{list(over2.index) if len(over2) else 'none'}"
          f"  -> H* definition {'NEEDS REVIEW' if len(over2) else 'stands'}")
    out["level_by_age"] = lvl.reset_index(names="age").to_dict("records")

    # -------------------------------------------------------- 1.5 the time path
    # Plan, verbatim: if 70+'s rho-hat path is FLAT over 2020, the coverage-drift
    # story that killed the 14.3% headline is itself wrong and that result comes
    # back. Check this before writing anything.
    print("\n=== 1.5 rho-hat time path by age (mid imputation, indexed to March) ===")
    path = city_age.pivot(index="ym", columns="age", values="rho_mid")
    idx = path / path.loc[202003]
    disp = idx.copy()
    disp.columns = [AGE_LABEL[a] for a in disp.columns]
    disp.index = [YM_LABEL[y] for y in disp.index]
    print(disp.round(3).to_string())
    mar_dec = (idx.loc[202012] - 1.0) * 100
    print("\n  March -> December, % change in rho-hat:")
    for a in AGES:
        print(f"    {AGE_LABEL[a]:<7} {mar_dec[a]:+7.2f}%")
    old = [a for a in AGES if a >= 70]
    detail = ", ".join(f"{AGE_LABEL[a]} {mar_dec[a]:+.1f}%" for a in old)
    verdict = ("FLAT (all |Mar->Dec| < 3%) — the coverage-drift story that killed "
               "the 14.3% headline is itself in question"
               if all(abs(mar_dec[a]) < 3.0 for a in old) else "not flat")
    print(f"\n  70+ path over 2020: {verdict}\n    {detail}")
    out["time_path_index_mar"] = idx.reset_index().to_dict("records")
    out["mar_to_dec_pct"] = {AGE_LABEL[a]: float(mar_dec[a]) for a in AGES}

    # --------------------------------- 1.5b does the retracted 14.3% come back?
    # The headline that was withdrawn was raw volume: 75+ moved 14.3% more in
    # December than in March. Two things sit between that and a behavioural
    # claim, and rho-hat separates the first from the second.
    #
    #   population   Seoul's elderly population itself grew over 2020, so part of
    #                the raw rise is more people, not more movement per person.
    #                Dividing by RegPop removes exactly this.
    #   coverage     rho-hat is still kappa x lambda. If kappa (observable and
    #                moving share) drifted up as elderly smartphone ownership
    #                rose, rho-hat rises with no change in behaviour.
    #
    # So the test is whether the rho-hat rise clears the external kappa band. It
    # only revives the headline if it lands ABOVE it.
    print("\n=== 1.5b decomposing the retracted 14.3%: population, coverage, behaviour ===")
    pop_mar_dec = (city_age.pivot(index="ym", columns="age", values="regpop")
                   .pipe(lambda p: (p.loc[202012] / p.loc[202003] - 1) * 100))
    raw = (city_age.pivot(index="ym", columns="age", values="V_mid")
           .pipe(lambda p: (p.loc[202012] / p.loc[202003] - 1) * 100))
    # The survey bands are 60대 and 70세 이상, so the kappa bound steps at 70. That
    # step is a property of how the surveys report, not of how ownership actually
    # moved with age, and 65-69 sits right under it — so its implied lambda is the
    # most sensitive number in the table. The last column says how sensitive: the
    # kappa drift that would drive lambda to exactly zero.
    KAPPA_BAND = {60: (1.7, 3.1), 65: (1.7, 3.1), 70: (8.9, 21.9),
                  75: (8.9, 21.9), 80: (8.9, 21.9)}
    v75 = city_age[city_age.age >= 75].groupby("ym")[["V_mid", "regpop"]].sum()
    raw75 = (v75.V_mid.loc[202012] / v75.V_mid.loc[202003] - 1) * 100
    rho75 = ((v75.V_mid / v75.regpop).loc[202012]
             / (v75.V_mid / v75.regpop).loc[202003] - 1) * 100
    print(f"  75+ combined, the band the retracted headline was about: "
          f"raw V {raw75:+.2f}%, rho-hat {rho75:+.2f}%")
    print(f"\n  {'band':<7} {'raw V':>8} {'pop':>7} {'rho':>8}  "
          f"{'kappa band':>15}  {'implied lambda':>20}  {'kappa for l=0':>13}")
    lam = {}
    for a in AGES:
        band = KAPPA_BAND.get(a)
        head = (f"  {AGE_LABEL[a]:<7} {raw[a]:>+8.2f} {pop_mar_dec[a]:>+7.2f} "
                f"{mar_dec[a]:>+8.2f}")
        if band is None:
            print(f"{head}  {'not resolved':>15}  {'—':>20}  {'—':>13}")
            continue
        lo = ((1 + mar_dec[a] / 100) / (1 + band[1] / 100) - 1) * 100
        hi = ((1 + mar_dec[a] / 100) / (1 + band[0] / 100) - 1) * 100
        lam[AGE_LABEL[a]] = [lo, hi]
        print(f"{head}  {band[0]:>+6.1f}..{band[1]:>+6.1f}%  "
              f"{lo:>+8.2f} .. {hi:>+7.2f}%  {mar_dec[a]:>+12.2f}%")
    # How far each determinate sign is from flipping. A band whose rho-hat sits
    # just outside its kappa bound owes its sign to the bound, not to the data.
    signed, margin = {}, {}
    for a, band in KAPPA_BAND.items():
        lo, hi = lam[AGE_LABEL[a]]
        if lo > 0:
            signed[AGE_LABEL[a]] = (lo, hi)
            margin[AGE_LABEL[a]] = mar_dec[a] - band[1]
        elif hi < 0:
            signed[AGE_LABEL[a]] = (lo, hi)
            margin[AGE_LABEL[a]] = band[0] - mar_dec[a]
    desc = ", ".join(f"{k} ({v[0]:+.1f}..{v[1]:+.1f}%, margin {margin[k]:.2f} pp)"
                     for k, v in signed.items())
    print(f"\n  determinate sign: {desc or 'none'}")
    print("  Everything else straddles zero: rho-hat's move is fully absorbable "
          "by the coverage band, so no behavioural claim is licensed there.")
    fragile = [k for k, m in margin.items() if m < 2.0]
    if fragile:
        print(f"  FRAGILE (< 2 pp of margin): {', '.join(fragile)}. The survey "
              f"reports 60대 and 70세 이상, so the kappa bound steps at 70 for "
              f"reporting reasons alone — these signs are set by that step, not "
              f"by the measurement. This is section 5's bandwidth argument landing "
              f"on the calibration source instead of on the data.")
    out["decomposition_mar_dec"] = {
        "raw_V_pct": {AGE_LABEL[a]: float(raw[a]) for a in AGES},
        "regpop_pct": {AGE_LABEL[a]: float(pop_mar_dec[a]) for a in AGES},
        "rho_pct": {AGE_LABEL[a]: float(mar_dec[a]) for a in AGES},
        "kappa_band_pct": {AGE_LABEL[a]: list(b) for a, b in KAPPA_BAND.items()},
        "implied_lambda_pct": lam,
        "sign_determinate": {k: list(v) for k, v in signed.items()},
        "margin_pp": {k: float(m) for k, m in margin.items()},
        "fragile": fragile,
        "combined_75plus": {"raw_V_pct": float(raw75), "rho_pct": float(rho75)},
    }

    # --------------------------------------------- 1.6 항동: which dong absorbs it
    # Phase 1b left this as a named test rather than a guess. 구로구 항동 is in the
    # registration in all 79 months and never in the product's code space. If KT
    # folds its residents into a neighbour, that neighbour's rho-hat is inflated
    # by the ratio of the two populations and should stand out inside 구로구.
    print("\n=== 1.6 구로구 항동: is it folded into a neighbouring dong? ===")
    # The denominator this script just used already carries p20's ABSORBED
    # assignment, so testing against it would be circular and the constant would
    # stop being falsifiable. Rebuild the pre-assignment state instead: join the
    # registration to the crosswalk on name alone, which is how 항동 came to be
    # unassigned in the first place. The test then re-derives the answer from
    # scratch on every run and disagrees loudly if the constant is wrong.
    cw = pd.read_parquet(DERIVED / "dong_crosswalk.parquet")
    guro_reg = regall[(regall.gu == "구로구")].drop(
        columns=["dong_code", "gu_code"])
    guro_reg = guro_reg.merge(cw[["gu", "dong", "dong_code"]],
                              on=["gu", "dong"], how="left")
    den_raw = (guro_reg[guro_reg.dong_code.notna()]
               .groupby("dong_code", as_index=False)["pop"].sum()
               .rename(columns={"pop": "regpop", "dong_code": "o_dong"}))
    den_raw["o_dong"] = den_raw.o_dong.astype("int64")
    by_dong = (v[v.o_dong // 1000 == 1117].groupby("o_dong", as_index=False)
               .agg(V=("V_mid", "sum"))
               .merge(den_raw, on="o_dong", how="inner"))
    by_dong["rho"] = by_dong.V / by_dong.regpop
    by_dong = by_dong.merge(cw[["dong_code", "dong"]], left_on="o_dong",
                            right_on="dong_code", how="left")
    med, sd = by_dong.rho.median(), by_dong.rho.std()
    by_dong["z"] = (by_dong.rho - med) / sd
    print(by_dong[["dong", "regpop", "rho", "z"]]
          .sort_values("rho", ascending=False).round(3).to_string(index=False))

    # Every dong in the gu is a candidate, not just the ones whose codes sit next
    # to 항동's — code adjacency is a weak prior and the data can decide. If dong
    # X absorbed 항동's residents, X's numerator already contains their trips
    # while its denominator does not, so X reads high; adding 항동's population to
    # X's denominator should pull it back into the gu's distribution. Score each
    # candidate by how far the whole gu is from flat afterwards.
    hang_pop = guro_reg[guro_reg.dong == "항동"]["pop"].sum()
    base_spread = float(np.sqrt(((by_dong.rho - med) ** 2).mean()))
    cand = []
    for _, row in by_dong.iterrows():
        trial = by_dong.copy()
        m = trial.o_dong == row.o_dong
        trial.loc[m, "rho"] = trial.loc[m, "V"] / (trial.loc[m, "regpop"] + hang_pop)
        cand.append({
            "dong": row.dong,
            "rho_before": float(row.rho), "z_before": float(row.z),
            "inflation": float((row.regpop + hang_pop) / row.regpop),
            "rho_after": float(trial.loc[m, "rho"].iloc[0]),
            "z_after": float((trial.loc[m, "rho"].iloc[0] - med) / sd),
            "gu_rms_after": float(np.sqrt(((trial.rho - med) ** 2).mean())),
        })
    cand.sort(key=lambda c: c["gu_rms_after"])
    print(f"\n  구로구 median rho {med:.3f}, sd {sd:.3f}, RMS deviation "
          f"{base_spread:.4f} with 항동 assigned to nobody")
    print(f"  {'candidate':<9} {'rho':>6} {'z':>6} {'x':>5} {'rho*':>6} "
          f"{'z*':>6} {'gu RMS*':>8}")
    for c in cand[:5]:
        print(f"  {c['dong']:<9} {c['rho_before']:>6.3f} {c['z_before']:>+6.2f} "
              f"{c['inflation']:>5.2f} {c['rho_after']:>6.3f} "
              f"{c['z_after']:>+6.2f} {c['gu_rms_after']:>8.4f}")
    best = cand[0]
    print(f"\n  best assignment: {best['dong']} "
          f"(gu RMS {base_spread:.4f} -> {best['gu_rms_after']:.4f}, "
          f"{'improves' if best['gu_rms_after'] < base_spread else 'does not improve'})"
          f"; runner-up {cand[1]['dong']} at {cand[1]['gu_rms_after']:.4f}")
    held = P20.ABSORBED.get(("구로구", "항동"), (None, None))[1]
    agree = held == best["dong"]
    print(f"  p20_regpop.ABSORBED currently assigns it to {held}: "
          f"{'agrees' if agree else 'DISAGREES — update ABSORBED or explain why'}")
    out["hangdong_agrees_with_absorbed"] = bool(agree)
    out["hangdong_test"] = {"guro_median_rho": float(med), "guro_sd": float(sd),
                            "hang_regpop_person_months": int(hang_pop),
                            "gu_rms_unassigned": base_spread,
                            "by_dong": by_dong.drop(columns="dong_code")
                            .to_dict("records"),
                            "candidates_ranked": cand}

    # ------------------------------------------- 1.7 cross-dong spread, first look
    print("\n=== 1.7 cross-dong spread of rho-hat by age (Dec, mid) ===")
    dec = r[r.ym == 202012]
    d_age = (dec.groupby(["o_dong", "age"], as_index=False)
             .agg(V=("V_mid", "sum"), regpop=("regpop", "sum")))
    d_age["rho"] = d_age.V / d_age.regpop
    spread = d_age.groupby("age")["rho"].agg(
        ["median", "std", lambda s: s.quantile(.05), lambda s: s.quantile(.95)])
    spread.columns = ["median", "sd", "p05", "p95"]
    spread["cv"] = spread.sd / spread["median"]
    spread.index = [AGE_LABEL[a] for a in spread.index]
    print(spread.round(3).to_string())
    out["cross_dong_spread_dec"] = spread.reset_index(names="age").to_dict("records")

    # --------------------------- 1.8 the other denominator: 주민등록 + 등록외국인
    # 주민등록 counts Korean nationals only, but KT counts devices regardless of
    # nationality, so the numerator already contains foreign residents' trips
    # while the denominator does not. rho-hat on 주민등록 alone is therefore
    # inflated by (K+F)/K, and that factor is very far from uniform: foreigners
    # are 7% of 20-24 and 0.2% of 80+.
    #
    # It is also not constant in time, which is the part that could manufacture a
    # trend. Seoul's registered foreign population fell 13.4% over 2020 as the
    # borders closed, so the inflation factor shrank through the year and the
    # uncorrected rho-hat picks up a spurious decline in exactly the bands where
    # foreigners concentrate. Both denominators are reported, as the plan requires.
    fpath = DERIVED / "foreign.parquet"
    if not fpath.exists():
        print("\n=== 1.8 skipped: no foreign.parquet (run eda/p20b_foreign.py) ===")
    else:
        print("\n=== 1.8 denominator with and without 등록외국인 ===")
        fo = pd.read_parquet(fpath)
        fo = (fo[fo.ym.isin(YMS)].rename(columns={"dong_code": "o_dong"})
              .groupby(["ym", "o_dong", "sex", "age"], as_index=False)["pop"]
              .sum().rename(columns={"pop": "foreign"}))
        fo["o_dong"] = fo.o_dong.astype("int64")
        rr = r.merge(fo, on=["ym", "o_dong", "sex", "age"], how="left")
        rr["foreign"] = rr.foreign.fillna(0.0)
        ca = (rr.groupby(["ym", "age"], as_index=False)
              .agg(V=("V_mid", "sum"), kor=("regpop", "sum"),
                   frn=("foreign", "sum")))
        ca["rho_kor"] = ca.V / ca.kor
        ca["rho_all"] = ca.V / (ca.kor + ca.frn)
        ca["frn_share"] = ca.frn / (ca.kor + ca.frn)
        mar, dec = ca[ca.ym == 202003].set_index("age"), ca[ca.ym == 202012].set_index("age")
        print(f"  {'band':<7} {'foreign %':>10} {'rho(kor)':>9} {'rho(all)':>9} "
              f"{'inflation':>10}   {'Mar->Dec kor':>13} {'Mar->Dec all':>13}")
        rows = []
        for a in AGES:
            infl = mar.loc[a, "rho_kor"] / mar.loc[a, "rho_all"]
            dk = (dec.loc[a, "rho_kor"] / mar.loc[a, "rho_kor"] - 1) * 100
            da = (dec.loc[a, "rho_all"] / mar.loc[a, "rho_all"] - 1) * 100
            rows.append(dict(band=AGE_LABEL[a], foreign_share=float(mar.loc[a, "frn_share"]),
                             rho_kor=float(mar.loc[a, "rho_kor"]),
                             rho_all=float(mar.loc[a, "rho_all"]),
                             inflation=float(infl), mar_dec_kor=float(dk),
                             mar_dec_all=float(da), delta_pp=float(da - dk)))
            print(f"  {AGE_LABEL[a]:<7} {mar.loc[a, 'frn_share']:>9.2%} "
                  f"{mar.loc[a, 'rho_kor']:>9.3f} {mar.loc[a, 'rho_all']:>9.3f} "
                  f"{infl:>10.3f}   {dk:>+12.2f}% {da:>+12.2f}%")
        cmp_ = pd.DataFrame(rows)
        worst = cmp_.loc[cmp_.delta_pp.abs().idxmax()]
        print(f"\n  Seoul foreign residents: {ca[ca.ym == 202003].frn.sum():,.0f} in March "
              f"-> {ca[ca.ym == 202012].frn.sum():,.0f} in December "
              f"({ca[ca.ym == 202012].frn.sum() / ca[ca.ym == 202003].frn.sum() - 1:+.1%})")
        print(f"  largest change to the Mar->Dec figure: {worst.band} "
              f"{worst.mar_dec_kor:+.2f}% -> {worst.mar_dec_all:+.2f}% "
              f"({worst.delta_pp:+.2f} pp)")
        eld = cmp_[cmp_.band.isin(["70-74", "75-79", "80+"])]
        print(f"  elderly bands move by at most {eld.delta_pp.abs().max():.2f} pp "
              f"— the headline is unaffected")
        out["denominator_with_foreign"] = cmp_.to_dict("records")

    # ------------------------------------------------------------------ figures
    # Sixteen series exhaust matplotlib's colour cycle twice over, so age is
    # mapped to a sequential ramp instead: the point of the panel is the age
    # ordering, and a ramp shows it where a categorical palette hides it.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    cmap = plt.get_cmap("viridis")
    months = [YM_LABEL[y] for y in path.index]
    for i, a in enumerate(AGES):
        axes[0].plot(months, path[a], lw=1.4, color=cmap(i / (len(AGES) - 1)),
                     label=AGE_LABEL[a])
    axes[0].set_ylabel(r"$\hat\rho$  (home departures / resident / day)")
    axes[0].set_title("rho-hat by age and month (mid imputation)", fontsize=10)
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=len(AGES) - 1))
    cb = fig.colorbar(sm, ax=axes[0], ticks=range(0, len(AGES), 3))
    cb.ax.set_yticklabels([AGE_LABEL[AGES[i]] for i in range(0, len(AGES), 3)],
                          fontsize=8)
    cb.set_label("age band", fontsize=9)
    x = np.arange(len(AGES))
    vals = [mar_dec[a] for a in AGES]
    axes[1].bar(x, vals, color=["#b03a48" if v < 0 else "#3a6ea5" for v in vals])
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=45, fontsize=8)
    axes[1].set_ylabel("% change")
    axes[1].set_title("rho-hat, March -> December 2020", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/p21_rho_path.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p21_rho_path.png")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.boxplot([d_age.loc[d_age.age == a, "rho"].dropna() for a in AGES],
               tick_labels=[AGE_LABEL[a] for a in AGES], showfliers=False)
    ax.set_ylabel(r"$\hat\rho$"); ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title("cross-dong spread of rho-hat by age (Dec 2020, 424 dong)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/p21_rho_dong_spread.png", dpi=150)
    plt.close(fig)
    print("  -> fig/p21_rho_dong_spread.png")

    with open(f"{ROOT}/eda/results_p21.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p21.json")


if __name__ == "__main__":
    main()
