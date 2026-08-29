#!/usr/bin/env python
"""2020 vs 2026 — who came back, net of coverage drift.

The advisor's framing question for the expanded window is distributional: after
the pandemic, did age-specific mixing return, and who did not come back. Two 2026
months are now on disk (Phase 2a), so a first answer is available without waiting
for the full 78-month ingest.

The comparison is 2020-06 against 2026-06 and 2020-07 against 2026-07 — the same
calendar months, so season is held fixed by construction rather than modelled,
and the pair gives an independent replicate of every number.

WHAT MAKES THIS IDENTIFIABLE, AND WHERE IT ISN'T. rho-hat is kappa x lambda, so a
six-year change in rho-hat is only a behavioural claim once the coverage change
is bounded. Two facts make that bound tight for most of the age range:

  * ownership was ALREADY saturated for 20-59 in 2020 (98.7-99.9% on both
    surveys), so kappa cannot have moved more than about a point;
  * ownership cannot exceed 100%, so the unpublished 2026 wave is pinned between
    the observed 2025 level and 100 — which for 60대 (99.8% NIA, 98.2% KCC in
    2025) is a band about two points wide.

For 70+ it stays wide: 2020 was 53.8% (NIA) / 50.8% (KCC), so the drift band is
tens of points and only a very large rho-hat move could clear it. That asymmetry
is the result, not a defect — it says exactly which bands this data can speak to.

FRAMING CAVEAT, which must not be lost: 2020-06 and 2020-07 are pandemic months,
not a pre-pandemic baseline. This project has no 2019. So a positive lambda change
means recovery relative to the pandemic year, NOT a return to a 2019 level, and
the paper must say so wherever this number appears.

    python eda/p23_baseline.py
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
from common import AGE_LABEL, AGES, months_on_disk
import calendar_kr as K
import duckdb
from paths import DERIVED, FIG, PARQUET_GLOB, ROOT, require

PAIRS = [(202006, 202606), (202007, 202607)]
SEOUL = "BETWEEN 1101000 AND 1125999"
CACHE = "rho_2026_numerator.parquet"
IMP = {"lo": 0.0, "mid": 1.5, "hi": 3.0}

# Survey band -> our 16 bands. 0-9 is deliberately absent: both surveys start at
# age 3 or 6, so under-6 ownership is not measured and that band gets no bound.
BAND_MAP = {
    "6-19": [10, 15], "20s": [20, 25], "30s": [30, 35], "40s": [40, 45],
    "50s": [50, 55], "60s": [60, 65], "70+": [70, 75, 80],
}
# Published levels, from derived/ownership.parquet (p20d). 2026 is unpublished,
# so its value is bounded by [2025 observed, 100].
LEVELS_2020 = {"nia": {"6-19": 94.7, "20s": 99.6, "30s": 99.5, "40s": 99.5,
                       "50s": 99.9, "60s": 93.1, "70+": 53.8},
               "kcc": {"6-19": 98.4, "20s": 99.5, "30s": 99.4, "40s": 99.5,
                       "50s": 98.7, "60s": 91.7, "70+": 50.8}}
LEVELS_2025 = {"nia": {"6-19": 99.1, "20s": 100.0, "30s": 100.0, "40s": 100.0,
                       "50s": 100.0, "60s": 99.8, "70+": 92.6},
               "kcc": {"6-19": 97.6, "20s": 100.0, "30s": 99.8, "40s": 99.8,
                       "50s": 99.4, "60s": 98.2, "70+": 76.3}}
out = {}


def numerator(refresh=False):
    """H-origin, origin-in-Seoul, deduplicated, for the four months compared."""
    if not refresh and (DERIVED / CACHE).exists():
        return pd.read_parquet(DERIVED / CACHE)
    yms = [y for p in PAIRS for y in p]
    on_disk = months_on_disk()
    missing = [y for y in yms if y not in on_disk]
    if missing:
        raise SystemExit(f"months not ingested: {missing}\n  on disk: {on_disk}")
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA disable_progress_bar")
    frames = []
    for ym in yms:
        frames.append(con.execute(f"""
            WITH g AS (
              -- The dedup key is the FULL natural key. Dropping arr_hour,
              -- d_dong or mtype from it would make min() collapse across hours
              -- and destinations instead of deduplicating, which silently
              -- divides the numerator by ~30 and still returns plausible ratios.
              SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                     min(pop) AS pop, bool_or(masked) AS masked
              FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
              WHERE ym = {ym} AND o_dong {SEOUL} AND mtype LIKE 'H%'
              GROUP BY ALL)
            SELECT ym, dow_n, o_dong, sex, age,
                   coalesce(sum(pop), 0) AS v_obs,
                   count(*) FILTER (masked) AS n_masked
            FROM g GROUP BY 1,2,3,4,5""").df())
        print(f"  {ym} scanned", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(DERIVED / CACHE, index=False)
    return df


def main():
    require(DERIVED / "regpop.parquet", "denominator")
    print("=== 1 numerator for 2020-06/07 and 2026-06/07 ===")
    num = numerator()
    days = {ym: sum(K.cell_exposure(ym, d)["n_days"] for d in range(1, 8))
            for ym in num.ym.unique()}
    v = num.groupby(["ym", "sex", "age"], as_index=False)[
        ["v_obs", "n_masked"]].sum()
    v["days"] = v.ym.map(days)
    for k, c in IMP.items():
        v[f"V_{k}"] = (v.v_obs + c * v.n_masked) / v.days

    reg = pd.read_parquet(DERIVED / "regpop.parquet")
    reg = reg[reg.ym.isin(v.ym.unique())].groupby(
        ["ym", "sex", "age"], as_index=False)["pop"].sum().rename(
        columns={"pop": "kor"})
    fo = pd.read_parquet(DERIVED / "foreign.parquet")
    fo = fo[fo.ym.isin(v.ym.unique())].groupby(
        ["ym", "sex", "age"], as_index=False)["pop"].sum().rename(
        columns={"pop": "frn"})
    d = v.merge(reg, on=["ym", "sex", "age"]).merge(fo, on=["ym", "sex", "age"])
    d["den"] = d.kor + d.frn
    a = d.groupby(["ym", "age"], as_index=False).agg(
        **{f"V_{k}": (f"V_{k}", "sum") for k in IMP}, den=("den", "sum"))
    for k in IMP:
        a[f"rho_{k}"] = a[f"V_{k}"] / a.den
    # Sanity gate. rho-hat is departures per resident per day and p21 measured it
    # at 0.24-0.71 across all sixteen bands in 2020. Anything far outside that is
    # an arithmetic error, not a finding, and it must stop the run rather than be
    # read as a result -- the ratios stay plausible even when the level is wrong.
    lo, hi = a.rho_mid.min(), a.rho_mid.max()
    if not (0.05 < lo and hi < 3.0):
        raise SystemExit(f"rho-hat out of range [{lo:.4f}, {hi:.4f}] — expected "
                         f"roughly 0.2-0.8; the numerator or denominator is wrong")

    # -------------------------------------------------- 2 rho-hat, 2020 vs 2026
    print("\n=== 2 rho-hat by age, same calendar month six years apart ===")
    print(f"  {'band':<7} {'Jun20':>7} {'Jun26':>7} {'d%':>8}   "
          f"{'Jul20':>7} {'Jul26':>7} {'d%':>8}")
    rows = []
    piv = a.pivot(index="age", columns="ym", values="rho_mid")
    for ag in AGES:
        r = {"band": AGE_LABEL[ag], "age": int(ag)}
        for y20, y26 in PAIRS:
            r[f"rho_{y20}"] = float(piv.loc[ag, y20])
            r[f"rho_{y26}"] = float(piv.loc[ag, y26])
            r[f"d_{y20 % 100}"] = float(piv.loc[ag, y26] / piv.loc[ag, y20] - 1)
        rows.append(r)
        print(f"  {AGE_LABEL[ag]:<7} {r['rho_202006']:>7.3f} {r['rho_202606']:>7.3f} "
              f"{r['d_6'] * 100:>+7.2f}%   {r['rho_202007']:>7.3f} "
              f"{r['rho_202607']:>7.3f} {r['d_7'] * 100:>+7.2f}%")
    rho = pd.DataFrame(rows)
    rho["d_mean"] = (rho.d_6 + rho.d_7) / 2
    rho["replicate_gap"] = (rho.d_6 - rho.d_7).abs()
    print(f"  June/July replicates agree to within "
          f"{rho.replicate_gap.max() * 100:.2f} pp at worst "
          f"({rho.loc[rho.replicate_gap.idxmax(), 'band']})")
    out["rho_by_age"] = rho.to_dict("records")

    # ------------------------------------------ 3 bound kappa, then read lambda
    # The 2026 wave is unpublished, so kappa(2026) is bounded below by the 2025
    # level and above by 100. Both surveys are used and the UNION of their drift
    # bands is taken, so the answer is never better than the more pessimistic one.
    print("\n=== 3 coverage drift 2020 -> 2026, and the behaviour it leaves ===")
    band_of = {ag: b for b, aa in BAND_MAP.items() for ag in aa}
    kap = {}
    for b in BAND_MAP:
        los, his = [], []
        for s in ("nia", "kcc"):
            v20, v25 = LEVELS_2020[s][b], LEVELS_2025[s][b]
            los.append(v25 / v20 - 1)
            his.append(100.0 / v20 - 1)
        kap[b] = (min(los), max(his))
    print(f"  {'band':<7} {'survey band':>11} {'kappa drift':>18} "
          f"{'rho change':>11}   {'implied lambda':>22} {'verdict':>9}")
    res = []
    for ag in AGES:
        b = band_of.get(ag)
        dr = float(rho.loc[rho.age == ag, "d_mean"].iloc[0])
        if b is None:
            print(f"  {AGE_LABEL[ag]:<7} {'not covered':>11} {'—':>18} "
                  f"{dr * 100:>+10.2f}%   {'—':>22} {'—':>9}")
            res.append(dict(band=AGE_LABEL[ag], survey_band=None,
                            rho_change=dr, lam_lo=None, lam_hi=None,
                            verdict="no bound"))
            continue
        klo, khi = kap[b]
        lo = (1 + dr) / (1 + khi) - 1
        hi = (1 + dr) / (1 + klo) - 1
        # A band counts as sign-determinate only if BOTH calendar-month
        # replicates say so on their own. Averaging them first would let one
        # month carry a claim the other contradicts -- 20-24 moves +21.0% in June
        # and -1.9% in July, and their mean looks like a clean positive.
        per = {}
        for y20, _ in PAIRS:
            dj = float(rho.loc[rho.age == ag, f"d_{y20 % 100}"].iloc[0])
            per[y20] = ((1 + dj) / (1 + khi) - 1, (1 + dj) / (1 + klo) - 1)
        signs = {"positive" if l > 0 else "negative" if h < 0 else "straddles 0"
                 for l, h in per.values()}
        verdict = signs.pop() if len(signs) == 1 else "replicates disagree"
        res.append(dict(band=AGE_LABEL[ag], survey_band=b, rho_change=dr,
                        kappa_lo=klo, kappa_hi=khi, lam_lo=lo, lam_hi=hi,
                        jun_lam=per[202006], jul_lam=per[202007],
                        verdict=verdict))
        print(f"  {AGE_LABEL[ag]:<7} {b:>11} {klo * 100:>+7.1f}..{khi * 100:>+6.1f}% "
              f"{dr * 100:>+10.2f}%   {lo * 100:>+9.2f} .. {hi * 100:>+7.2f}% "
              f"{verdict:>9}")
    R = pd.DataFrame(res)
    out["decomposition"] = R.to_dict("records")

    signed = R[R.verdict.isin(["positive", "negative"])]
    print(f"\n  sign-determinate in BOTH calendar-month replicates: "
          f"{len(signed)} of {len(R)}")
    for _, x in signed.iterrows():
        print(f"    {x.band:<7} {x.verdict:<8} {x.lam_lo * 100:+.1f} .. "
              f"{x.lam_hi * 100:+.1f}%")
    bad = R[R.verdict == "replicates disagree"]
    if len(bad):
        print(f"\n  replicates disagree (claim NOT supported): "
              f"{', '.join(bad.band)}")
        for _, x in bad.iterrows():
            print(f"    {x.band:<7} Jun {x.jun_lam[0] * 100:+.1f}..{x.jun_lam[1] * 100:+.1f}%"
                  f"   Jul {x.jul_lam[0] * 100:+.1f}..{x.jul_lam[1] * 100:+.1f}%")
    print("\n  NOTE: 2020-06/07 are pandemic months, not a pre-pandemic baseline "
          "(this project has no 2019). A positive lambda is recovery relative to "
          "the pandemic year, not a return to 2019.")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(AGES))
    ax.bar(x, R.rho_change * 100, color="#c9c9c9", label=r"$\hat\rho$ change (raw)")
    ok = R.lam_lo.notna()
    ax.errorbar(x[ok.values], ((R.lam_lo[ok] + R.lam_hi[ok]) / 2) * 100,
                yerr=[((R.lam_lo[ok] + R.lam_hi[ok]) / 2 - R.lam_lo[ok]) * 100,
                      (R.lam_hi[ok] - (R.lam_lo[ok] + R.lam_hi[ok]) / 2) * 100],
                fmt="o", ms=4, color="#b03a48", capsize=3, lw=1.2,
                label=r"implied behavioural change $\lambda$ (bounds)")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([AGE_LABEL[ag] for ag in AGES],
                                         rotation=45, fontsize=8)
    ax.set_ylabel("% change, 2020 -> 2026")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("who came back: same calendar months, six years apart "
                 "(Jun+Jul mean; 2020 is itself a pandemic year)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/p23_baseline.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p23_baseline.png")

    with open(f"{ROOT}/eda/results_p23.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p23.json")


if __name__ == "__main__":
    main()
