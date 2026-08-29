#!/usr/bin/env python
"""Phase 1d routes R2 and R3, now that saturated-era and normal-term months exist.

Both routes were blocked on the time window rather than on method, and both are
unblocked by the same two months (2025-03 and 2025-12).

R2 — SATURATION ANCHORING. The plan's premise was that 70+ ownership saturates by
2024-2026; p20d showed that is false (92.6% and still climbing). But the useful
property was never saturation itself, it was that kappa stops moving. Two things
deliver that without 70+ saturating:

  * 60대 IS saturated (99.1% in 2024, 99.8% in 2025), so its within-year drift is
    capped by the ceiling at a fraction of a point;
  * the bound is one-sided in the right direction. In 2020 the survey band
    EXCLUDED zero (+1.7 to +3.1%), so a small rho-hat move was fully absorbable
    by coverage. From a saturated base the band starts at zero, so a rho-hat move
    larger than the ceiling is a behavioural move.

Bounding a within-year (March to December) drift needs the waves either side of
it. Under monotone ownership, March sits at or above the previous wave and
December at or below 100, so the drift is bounded by [0, 100/v_prev - 1]. The
monotonicity is not free — KCC's 70+ fell 60.1 to 59.2 across 2021-2022 — so the
lower end is reported as an assumption, and any band whose verdict depends on it
is marked.

R3 — INSTITUTIONAL ANCHORING. 15-19's departure rate in a normal school month is
pinned by compulsory attendance, which makes it a lambda reference. p21 showed
2020 has no normal school month: the weekday rho-hat(15-19)/rho-hat(45-49) ratio
swings 0.689 to 1.065 across the year, measuring closure policy rather than an
anchor. With 2025-03, 2025-12 and 2026-06 (term) against 2026-07 (vacation) the
anchor can finally be tested for the stability it needs to be an anchor at all.

    python eda/p24_r2r3.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb
import numpy as np
import pandas as pd
from common import AGE_LABEL, AGES, months_on_disk
import calendar_kr as K
from paths import DERIVED, FIG, PARQUET_GLOB, ROOT, require

SEOUL = "BETWEEN 1101000 AND 1125999"
NEW = [202503, 202512]
CACHE = "rho_2025_numerator.parquet"

# Published 스마트폰 보유율 by survey band (p20d / derived/ownership.parquet).
LV = {
    "nia": {2019: {"60s": 92.0, "70+": 40.2}, 2020: {"60s": 93.1, "70+": 53.8},
            2024: {"60s": 99.1, "70+": 82.2}, 2025: {"60s": 99.8, "70+": 92.6}},
    "kcc": {2019: {"60s": 85.4, "70+": 39.7}, 2020: {"60s": 91.7, "70+": 50.8},
            2024: {"60s": 96.9, "70+": 73.0}, 2025: {"60s": 98.2, "70+": 76.3}},
}
BAND_MAP = {"60s": [60, 65], "70+": [70, 75, 80]}
# The letter's interpolated 2020 March-to-December bands, for side-by-side.
KAPPA_2020 = {"60s": (0.017, 0.031), "70+": (0.089, 0.219)}
# School regime per month, from the Korean academic calendar. 2020 is the
# pandemic year: term was delayed to 4/9 online and phased in-person from 5/20.
REGIME = {202001: "winter break", 202002: "winter break", 202003: "closed",
          202004: "remote", 202005: "phased", 202006: "phased",
          202007: "phased/summer", 202008: "summer", 202009: "phased",
          202010: "phased", 202011: "phased", 202012: "remote (wave 3)",
          202503: "term (start)", 202512: "term", 202606: "term",
          202607: "summer"}
out = {}


def scan(yms, cache):
    if (DERIVED / cache).exists():
        return pd.read_parquet(DERIVED / cache)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA disable_progress_bar")
    fr = []
    for ym in yms:
        fr.append(con.execute(f"""
            WITH g AS (
              SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                     min(pop) AS pop, bool_or(masked) AS masked
              FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
              WHERE ym = {ym} AND o_dong {SEOUL} AND mtype LIKE 'H%'
              GROUP BY ALL)
            SELECT ym, dow_n, sex, age,
                   coalesce(sum(pop), 0) AS v_obs,
                   count(*) FILTER (masked) AS n_masked
            FROM g GROUP BY 1,2,3,4""").df())
        print(f"  {ym} scanned", flush=True)
    df = pd.concat(fr, ignore_index=True)
    df.to_parquet(DERIVED / cache, index=False)
    return df


def rho_table():
    """(ym, age) -> rho-hat, all days and weekday-only, all months available."""
    # The caches OVERLAP: rho_numerator_dow has all of 2020 and rho_2026_numerator
    # also carries 2020-06/07 as its comparison base. Concatenating and summing
    # would double those two months -- and the ratio in R3 would survive it
    # unscathed, since numerator and denominator double together, so the error
    # would only ever show up in a level. Each month is taken from the first
    # cache that has it.
    parts, seen = [], set()
    for f in ("rho_numerator_dow.parquet", "rho_2026_numerator.parquet"):
        if not (DERIVED / f).exists():
            continue
        d = pd.read_parquet(DERIVED / f)
        d = d[~d.ym.isin(seen)]
        seen |= set(d.ym.unique())
        parts.append(d[["ym", "dow_n", "sex", "age", "v_obs", "n_masked"]])
    missing = [y for y in NEW if y not in months_on_disk()]
    if missing:
        raise SystemExit(f"months not ingested: {missing}")
    fresh = scan(NEW, CACHE)
    parts.append(fresh[~fresh.ym.isin(seen)])
    n = (pd.concat(parts, ignore_index=True)
         .groupby(["ym", "dow_n", "sex", "age"], as_index=False).sum())

    def agg(sub, dows):
        days = {ym: sum(K.cell_exposure(ym, d)["n_days"] for d in dows)
                for ym in sub.ym.unique()}
        g = sub.groupby(["ym", "age"], as_index=False)[["v_obs", "n_masked"]].sum()
        g["V"] = (g.v_obs + 1.5 * g.n_masked) / g.ym.map(days)
        return g[["ym", "age", "V"]]

    alld = agg(n, range(1, 8)).rename(columns={"V": "V_all"})
    wk = agg(n[n.dow_n <= 5], range(1, 6)).rename(columns={"V": "V_wd"})
    reg = pd.read_parquet(DERIVED / "regpop.parquet").groupby(
        ["ym", "age"], as_index=False)["pop"].sum().rename(columns={"pop": "kor"})
    fo = pd.read_parquet(DERIVED / "foreign.parquet").groupby(
        ["ym", "age"], as_index=False)["pop"].sum().rename(columns={"pop": "frn"})
    t = alld.merge(wk, on=["ym", "age"]).merge(reg, on=["ym", "age"]).merge(
        fo, on=["ym", "age"])
    t["den"] = t.kor + t.frn
    t["rho"] = t.V_all / t.den
    t["rho_wd"] = t.V_wd / t.den
    lo, hi = t.rho.min(), t.rho.max()
    if not (0.05 < lo and hi < 3.0):
        raise SystemExit(f"rho-hat out of range [{lo:.4f}, {hi:.4f}]")
    return t


def main():
    require(DERIVED / "regpop.parquet", "denominator")
    print("=== 0 rho-hat across every month on disk ===")
    t = rho_table()
    print(f"  months: {sorted(t.ym.unique())}")

    # ---------------------------------------------- R2: same contrast, two eras
    print("\n=== R2 March->December, 2020 (pre-saturation) vs 2025 (saturated) ===")
    piv = t.pivot(index="ym", columns="age", values="rho")
    rows = []
    for band, ages in BAND_MAP.items():
        # 2025 kappa: monotone ownership puts March at or above the 2024 wave and
        # December at or below 100, so drift lies in [0, 100/v_2024 - 1]. The
        # union over surveys is taken, so the looser survey governs.
        # Two readings. The union over surveys is the conservative one. But the
        # project's stated source policy (letter section 3.1) is that NIA is
        # primary -- ten times the sample, a fixed 7/1 reference date, exactly
        # twelve months between waves -- with KCC as cross-validation, and that
        # the post-2021 divergence is a suspected questionnaire-base change. At
        # 70+ the two disagree by up to 28.5 pp, so which one governs decides
        # whether this route delivers anything, and both are reported.
        hi25 = max(100.0 / LV[s][2024][band] - 1 for s in LV)
        hi25_nia = 100.0 / LV["nia"][2024][band] - 1
        lo25 = 0.0
        lo20, hi20 = KAPPA_2020[band]
        for ag in ages:
            d20 = piv.loc[202012, ag] / piv.loc[202003, ag] - 1
            d25 = piv.loc[202512, ag] / piv.loc[202503, ag] - 1
            def lam(d, kl, kh):
                return ((1 + d) / (1 + kh) - 1, (1 + d) / (1 + kl) - 1)
            l20, l25 = lam(d20, lo20, hi20), lam(d25, lo25, hi25)
            l25n = lam(d25, lo25, hi25_nia)
            v = lambda p: ("positive" if p[0] > 0 else
                           "negative" if p[1] < 0 else "straddles 0")
            rows.append(dict(band=AGE_LABEL[ag], survey_band=band,
                             rho_2020=float(d20), rho_2025=float(d25),
                             kappa_2020=[lo20, hi20], kappa_2025=[lo25, hi25],
                             kappa_2025_nia=[lo25, hi25_nia],
                             lam_2020=list(l20), lam_2025=list(l25),
                             lam_2025_nia=list(l25n), verdict_2020=v(l20),
                             verdict_2025=v(l25), verdict_2025_nia=v(l25n)))
    R = pd.DataFrame(rows)
    print(f"  {'band':<7} | {'2020 rho':>9} {'kappa':>14} {'lambda':>19} {'':>12}"
          f" | {'2025 rho':>9} {'kappa':>13} {'lambda':>19} {'':>12}")
    for _, x in R.iterrows():
        print(f"  {x.band:<7} | {x.rho_2020 * 100:>+8.2f}% "
              f"{x.kappa_2020[0] * 100:>+5.1f}..{x.kappa_2020[1] * 100:>+5.1f}% "
              f"{x.lam_2020[0] * 100:>+8.2f}..{x.lam_2020[1] * 100:>+7.2f}% "
              f"{x.verdict_2020:>12} | {x.rho_2025 * 100:>+8.2f}% "
              f"{x.kappa_2025[0] * 100:>+4.1f}..{x.kappa_2025[1] * 100:>+5.1f}% "
              f"{x.lam_2025[0] * 100:>+8.2f}..{x.lam_2025[1] * 100:>7.2f}% "
              f"{x.verdict_2025:>12}")
    print(f"\n  under the survey UNION, bands gaining a determinate sign in the "
          f"saturated era: "
          f"{', '.join(R[(R.verdict_2020 == 'straddles 0') & (R.verdict_2025 != 'straddles 0')].band) or 'none'}")
    print("  under NIA alone (the project's primary source):")
    for _, x in R.iterrows():
        print(f"    {x.band:<7} 2025 kappa +0.0..{x.kappa_2025_nia[1] * 100:>5.2f}%  "
              f"lambda {x.lam_2025_nia[0] * 100:>+7.2f}..{x.lam_2025_nia[1] * 100:>+6.2f}%  "
              f"{x.verdict_2025_nia:<12} margin "
              f"{(x.rho_2025 - x.kappa_2025_nia[1]) * 100:>+6.2f} pp")
    stuck = R[R.verdict_2025_nia == "straddles 0"]
    print(f"  still unidentified even on NIA alone: "
          f"{', '.join(stuck.band) or 'none'}")
    if len(stuck):
        print("  -> 70+ does NOT become identified by waiting. Its 2024 base is "
              "73-82% depending on survey, so the ceiling still leaves a 22-37% "
              "gap. It needs the surveys to agree, or a different kappa source.")
    # Which assumption each verdict actually rests on, stated precisely: the
    # CEILING (kappa <= 100/v_prev - 1) sets the lower end of lambda and is what
    # a positive verdict needs; MONOTONICITY (kappa >= 0) sets the upper end and
    # is what a negative verdict would need. No 2025 verdict is negative, so
    # monotonicity -- the assumption KCC's 2021-2022 fall from 60.1 to 59.2 would
    # violate -- carries no weight here at all. Every 2025 conclusion stands on
    # the ceiling, which is arithmetic rather than an assumption.
    neg = R[R.verdict_2025.isin(["negative"]) | R.verdict_2025_nia.eq("negative")]
    print(f"\n  monotonicity (kappa >= 0) is load-bearing only for a negative "
          f"verdict; 2025 has {len(neg)}, so it carries no weight here. The "
          f"positive verdicts rest on the ceiling, which is arithmetic.")
    out["r2"] = R.to_dict("records")

    # -------------------------------------- R3: is 15-19 an anchor or a policy?
    print("\n=== R3 the 15-19 institutional anchor, by school regime ===")
    w = t.pivot(index="ym", columns="age", values="rho_wd")
    ratio = (w[15] / w[45]).rename("ratio").reset_index()
    ratio["regime"] = ratio.ym.map(REGIME)
    print(f"  {'ym':>8} {'regime':<16} {'rho_wd(15-19)':>13} {'rho_wd(45-49)':>13} "
          f"{'ratio':>7}")
    for _, x in ratio.sort_values("ym").iterrows():
        print(f"  {int(x.ym):>8} {x.regime:<16} {w.loc[x.ym, 15]:>13.3f} "
              f"{w.loc[x.ym, 45]:>13.3f} {x.ratio:>7.3f}")
    term = ratio[ratio.regime.str.startswith("term")]
    midterm = ratio[ratio.regime == "term"]
    dis = ratio[~ratio.regime.isin(["term", "summer", "winter break"])]
    print(f"\n  normal-term months: n={len(term)}, ratio "
          f"{term.ratio.min():.3f}-{term.ratio.max():.3f}, "
          f"spread {term.ratio.max() - term.ratio.min():.3f}")
    print(f"  pandemic-disrupted months: n={len(dis)}, ratio "
          f"{dis.ratio.min():.3f}-{dis.ratio.max():.3f}, "
          f"spread {dis.ratio.max() - dis.ratio.min():.3f}")
    # March is the first month of the Korean school year and behaves differently
    # from mid-term months, so it is reported separately rather than averaged in.
    print(f"  of which term-start (March): "
          f"{', '.join(f'{int(r.ym)} {r.ratio:.3f}' for _, r in ratio[ratio.regime == 'term (start)'].iterrows())}")
    print(f"  mid/late-term months only: n={len(midterm)}, ratio "
          f"{midterm.ratio.min():.3f}-{midterm.ratio.max():.3f}, "
          f"spread {midterm.ratio.max() - midterm.ratio.min():.4f}")
    usable = (midterm.ratio.max() - midterm.ratio.min()) < 0.05 and len(midterm) >= 2
    print(f"  -> 15-19 is {'USABLE' if usable else 'NOT yet usable'} as a lambda "
          f"anchor on mid/late-term months (needs >=2 spreading < 0.05)")
    out["r3"] = {"by_month": ratio.to_dict("records"),
                 "term_spread": float(term.ratio.max() - term.ratio.min()),
                 "midterm_spread": float(midterm.ratio.max() - midterm.ratio.min()),
                 "disrupted_spread": float(dis.ratio.max() - dis.ratio.min()),
                 "usable": bool(usable)}

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(R))
    axes[0].errorbar(x - 0.12, R.lam_2020.apply(lambda p: sum(p) / 2) * 100,
                     yerr=[(R.lam_2020.apply(lambda p: (p[1] - p[0]) / 2)) * 100] * 2,
                     fmt="o", ms=4, capsize=3, color="#b03a48", label="2020 Mar->Dec")
    axes[0].errorbar(x + 0.12, R.lam_2025.apply(lambda p: sum(p) / 2) * 100,
                     yerr=[(R.lam_2025.apply(lambda p: (p[1] - p[0]) / 2)) * 100] * 2,
                     fmt="s", ms=4, capsize=3, color="#3a6ea5", label="2025 Mar->Dec")
    axes[0].axhline(0, color="k", lw=.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(R.band, rotation=45, fontsize=8)
    axes[0].set_ylabel(r"implied $\lambda$ change (%)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("R2: the same contrast before and after saturation", fontsize=10)
    col = {"term": "#3a6ea5", "summer": "#c9a227", "winter break": "#c9a227"}
    rs = ratio.sort_values("ym")
    axes[1].bar(range(len(rs)), rs.ratio,
                color=[col.get(r, "#b03a48") for r in rs.regime])
    axes[1].set_xticks(range(len(rs)))
    axes[1].set_xticklabels([str(int(v)) for v in rs.ym], rotation=90, fontsize=7)
    axes[1].axhline(1.0, color="k", lw=.6, ls=":")
    axes[1].set_ylabel(r"$\hat\rho$(15-19) / $\hat\rho$(45-49), weekdays")
    axes[1].set_title("R3: blue = normal term, red = pandemic-disrupted",
                      fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/p24_r2r3.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p24_r2r3.png")
    with open(f"{ROOT}/eda/results_p24.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p24.json")


if __name__ == "__main__":
    main()
