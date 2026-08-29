#!/usr/bin/env python
"""Phase 12 — separate policy stringency from perceived epidemic risk.

With 8 months the residual from the dose line looked like monotone decay: Feb and
Mar below it, everything after above it. December breaks that. December carries
the highest dose of the year AND the deepest mobility trough (0.720 against
March's 0.767), and it is the month Seoul's own epidemic finally arrived
(328 cases/day against 12 in March). So the residual may be tracking risk, not
time, and the two are separable now that the case series is in hand.

Case exposure per cell uses the same trick as dose: the file has no date column,
but the dates each (대상연월, 요일) cell aggregates are known exactly, so the
mean case count over precisely those days is computable. Risk is measured on the
7 days BEFORE each date, which is what a person deciding whether to go out could
have read.
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
import calendar_kr as K
from common import DERIVED, ROOT, YMS, YM_LABEL, AGE_LABEL, AGES, DOW_LABEL

FIG = f"{ROOT}/eda/fig"
out = {}


def main():
    cases = pd.read_parquet(f"{ROOT}/cases/daily_panel.parquet").set_index("date")
    S = cases["seoul_cases"].astype(float)
    N = cases["nat_cases"].astype(float)


    def cell_risk(ym, dow_n, series, lag_days=7):
        """mean of `series` over the 7 days preceding each date the cell covers"""
        days = pd.to_datetime([d for d in K.cell_exposure(ym, dow_n)["dates"]])
        vals = []
        for d in days:
            w = series.loc[(series.index >= d - pd.Timedelta(days=lag_days))
                           & (series.index < d)]
            vals.append(w.mean() if len(w) else 0.0)
        return float(np.mean(vals))


    cal = pd.DataFrame(K.panel(YMS))[["ym", "dow_n", "n_days", "n_holiday", "dose", "dose_alt"]]
    cal["seoul_risk"] = [cell_risk(r.ym, r.dow_n, S) for r in cal.itertuples()]
    cal["nat_risk"] = [cell_risk(r.ym, r.dow_n, N) for r in cal.itertuples()]
    CLEAN = cal[cal.n_holiday == 0]

    # ------------------------------------------------------------- mobility panel
    core = pd.read_parquet(f"{DERIVED}/panel_core.parquet")
    WD = core[core.dow_n <= 5].merge(CLEAN[["ym", "dow_n"]], on=["ym", "dow_n"])
    ref = WD[WD.ym.isin([202002, 202007])].groupby(["ym", "dow_n", "arr_hour"],
                                                   as_index=False)["mid"].sum()
    ref["f"] = ref["mid"] / ref.groupby(["ym", "arr_hour"])["mid"].transform("mean")
    FAC = ref.groupby(["dow_n", "arr_hour"], as_index=False)["f"].mean()
    d = WD.merge(FAC, on=["dow_n", "arr_hour"])
    d["mid"] = d["mid"] / d["f"]

    GROUPS = {"20-24": [20], "25-44": [25, 30, 35, 40], "45-64": [45, 50, 55, 60],
              "65-74": [65, 70], "75+": [75, 80]}

    print("=== 12.1 monthly picture: dose, local risk, mobility ===")
    cellsum = d.groupby(["ym", "dow_n"], as_index=False)["mid"].sum()
    tot = cellsum.groupby("ym")["mid"].mean()
    M = pd.DataFrame({
        "dose": CLEAN[CLEAN.dow_n <= 5].groupby("ym")["dose"].mean(),
        "seoul_cases_per_day": CLEAN[CLEAN.dow_n <= 5].groupby("ym")["seoul_risk"].mean(),
        "nat_cases_per_day": CLEAN[CLEAN.dow_n <= 5].groupby("ym")["nat_risk"].mean(),
        "mobility": tot / tot[202001]})
    M.index = [YM_LABEL[i] for i in M.index]
    print(M.round(3).to_string())
    out["monthly"] = M.reset_index().to_dict("records")

    # --------------------------------------------- 12.2 does dose survive the risk?
    print("\n=== 12.2 cell-level regression, log(mobility) on dose and log risk ===")
    print("    (weekday cells only, holiday-free, weekday-composition balanced)")


    def ols(X, y):
        n = len(X[0])
        X = np.column_stack([np.ones(n)] + [np.asarray(c, float) for c in X])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        n, k = X.shape
        s2 = r @ r / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
        r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
        return b, se, r2


    rows = []
    for g, ages in {"all": AGES, **GROUPS}.items():
        sub = d[d.age.isin(ages)].groupby(["ym", "dow_n"], as_index=False)["mid"].sum()
        sub = sub.merge(CLEAN, on=["ym", "dow_n"])
        y = np.log(sub["mid"].to_numpy())
        dose, risk = sub["dose"].to_numpy(), np.log1p(sub["seoul_risk"].to_numpy())
        for label, X in [("dose only", [dose]), ("risk only", [risk]),
                         ("both", [dose, risk])]:
            b, se, r2 = ols(X, y)
            rows.append(dict(group=g, spec=label, n=len(y), r2=round(r2, 3),
                             b_dose=round(b[1], 4) if "dose" in label or label == "both" else None,
                             se_dose=round(se[1], 4) if "dose" in label or label == "both" else None,
                             b_risk=round(b[-1], 4) if "risk" in label or label == "both" else None,
                             se_risk=round(se[-1], 4) if "risk" in label or label == "both" else None))
    R = pd.DataFrame(rows)
    print(R.to_string(index=False))
    out["regression"] = rows

    print("\n  interpretation: the coefficient that survives 'both' is the one doing the work.")
    for g in ["all", "25-44", "75+"]:
        r = R[(R.group == g) & (R.spec == "both")].iloc[0]
        print(f"    {g:>6}: dose {r.b_dose:+.4f} (se {r.se_dose:.4f})   "
              f"log-risk {r.b_risk:+.4f} (se {r.se_risk:.4f})   R2={r.r2}")

    # ------------------------------------- 12.3 the age gradient of risk sensitivity
    print("\n=== 12.3 who responds to what: coefficients by age band ===")
    rows = []
    for a in AGES:
        sub = d[d.age == a].groupby(["ym", "dow_n"], as_index=False)["mid"].sum()
        sub = sub.merge(CLEAN, on=["ym", "dow_n"])
        y = np.log(sub["mid"].to_numpy())
        b, se, r2 = ols([sub["dose"].to_numpy(), np.log1p(sub["seoul_risk"].to_numpy())], y)
        rows.append(dict(age=AGE_LABEL[a], b_dose=round(b[1], 4), se_dose=round(se[1], 4),
                         b_risk=round(b[2], 4), se_risk=round(se[2], 4), r2=round(r2, 3)))
    A = pd.DataFrame(rows)
    print(A.to_string(index=False))
    out["by_age"] = rows

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].plot(M.index, M["mobility"], marker="o", color="#0E7C86", lw=1.8, label="mobility (Jan=1)")
    ax2 = axes[0].twinx()
    ax2.plot(M.index, M["seoul_cases_per_day"], marker="s", color="#A8434E", lw=1.5,
             label="Seoul cases/day")
    ax2.set_yscale("log"); ax2.set_ylabel("Seoul cases/day (log)")
    axes[0].set_ylabel("mobility, Jan = 1"); axes[0].grid(alpha=.3)
    axes[0].set_title("mobility against Seoul's own epidemic", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    axes[1].errorbar(A["age"], A["b_dose"], yerr=1.96 * A["se_dose"], marker="o",
                     color="#0E7C86", capsize=3)
    axes[1].axhline(0, color="k", lw=.6)
    axes[1].set_title("policy-dose coefficient, conditional on risk", fontsize=10)
    axes[1].tick_params(axis="x", rotation=90); axes[1].grid(alpha=.3)

    axes[2].errorbar(A["age"], A["b_risk"], yerr=1.96 * A["se_risk"], marker="o",
                     color="#9A6C15", capsize=3)
    axes[2].axhline(0, color="k", lw=.6)
    axes[2].set_title("log-risk coefficient, conditional on dose", fontsize=10)
    axes[2].tick_params(axis="x", rotation=90); axes[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/p12_dose_vs_risk.png", dpi=150); plt.close(fig)
    print("\n  -> fig/p12_dose_vs_risk.png")

    # ------------------------------------ 12.4 March vs December: the decisive pair
    print("\n=== 12.4 March vs December: same-ish dose, 27x the local risk ===")
    pair = M.loc[["Mar", "Dec"]]
    print(pair.round(3).to_string())
    print(f"\n  Dec/Mar dose ratio {pair.dose['Dec']/pair.dose['Mar']:.2f}, "
          f"Seoul-risk ratio {pair.seoul_cases_per_day['Dec']/pair.seoul_cases_per_day['Mar']:.1f}x, "
          f"mobility ratio {pair.mobility['Dec']/pair.mobility['Mar']:.3f}")
    byg = {}
    for g, ages in GROUPS.items():
        sub = d[d.age.isin(ages)].groupby(["ym", "dow_n"], as_index=False)["mid"].sum()
        s = sub.groupby("ym")["mid"].mean()
        byg[g] = dict(Mar=s[202003], Dec=s[202012], ratio=round(s[202012] / s[202003], 3))
    print(pd.DataFrame(byg).T.to_string())
    out["mar_dec"] = byg

    with open(f"{ROOT}/eda/results_p12.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p12.json")


if __name__ == "__main__":
    main()
