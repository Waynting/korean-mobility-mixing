#!/usr/bin/env python
"""Phase 15 — S4 of the screen: the pre-specified specification family.

§S4 of the screening protocol stated the hard gate as a sign-stability
requirement over a family of specifications fixed BEFORE estimation. The
protocol itself has been retired with the causal design it served; the spec is
in git history (`git show 095ea05:PROTOCOL.md`). p12 only runs three of them
(dose / risk / both), so the family — and the sign flip that fails the gate —
lived in the memo rather than in code. This runs the whole family, on the
12-month panel, and reports the range of every coefficient across it.

Family (pre-specified in §S4, unchanged here):
    {dose}
    {dose, log local risk}
    {dose, log national risk}
    {dose, log local risk, log national risk}
    {dose, month trend}
    {dose, log local risk, month trend}

Risk is the mean daily case count over the 7 days BEFORE each date the cell
covers, i.e. what a person deciding whether to go out could have read.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import calendar_kr as K
from common import DERIVED, ROOT, YMS, YM_LABEL, AGES, AGE_LABEL

out = {}


def main():
    cases = pd.read_parquet(f"{ROOT}/cases/daily_panel.parquet").set_index("date")
    S = cases["seoul_cases"].astype(float)
    N = cases["nat_cases"].astype(float)


    def cell_risk(ym, dow_n, series, lag_days=7):
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
    cal["trend"] = [YMS.index(y) for y in cal["ym"]]
    CLEAN = cal[(cal.n_holiday == 0) & (cal.dow_n <= 5)]
    print(f"clean weekday cells: n = {len(CLEAN)}")

    core = pd.read_parquet(f"{DERIVED}/panel_core.parquet")
    WD = core[core.dow_n <= 5].merge(CLEAN[["ym", "dow_n"]], on=["ym", "dow_n"])
    ref = WD[WD.ym.isin([202002, 202007])].groupby(["ym", "dow_n", "arr_hour"],
                                                   as_index=False)["mid"].sum()
    ref["f"] = ref["mid"] / ref.groupby(["ym", "arr_hour"])["mid"].transform("mean")
    FAC = ref.groupby(["dow_n", "arr_hour"], as_index=False)["f"].mean()
    d = WD.merge(FAC, on=["dow_n", "arr_hour"])
    d["mid"] = d["mid"] / d["f"]

    GROUPS = {"all": AGES, "20-24": [20], "25-44": [25, 30, 35, 40],
              "45-64": [45, 50, 55, 60], "65-74": [65, 70], "75+": [75, 80]}
    FAMILY = [
        ("dose",                    ["dose"]),
        ("+ local risk",            ["dose", "lrisk"]),
        ("+ national risk",         ["dose", "nrisk"]),
        ("+ both risks",            ["dose", "lrisk", "nrisk"]),
        ("+ month trend",           ["dose", "trend"]),
        ("+ local risk + trend",    ["dose", "lrisk", "trend"]),
    ]


    def ols(X, y):
        X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        n, k = X.shape
        se = np.sqrt(np.diag((r @ r / (n - k)) * np.linalg.pinv(X.T @ X)))
        return b, se, 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))


    rows = []
    for g, ages in GROUPS.items():
        sub = (d[d.age.isin(ages)].groupby(["ym", "dow_n"], as_index=False)["mid"].sum()
               .merge(CLEAN, on=["ym", "dow_n"]))
        y = np.log(sub["mid"].to_numpy())
        col = {"dose": sub["dose"].to_numpy(),
               "lrisk": np.log1p(sub["seoul_risk"].to_numpy()),
               "nrisk": np.log1p(sub["nat_risk"].to_numpy()),
               "trend": sub["trend"].to_numpy()}
        print(f"\n=== {g}  (n={len(y)}) ===")
        print(f"{'spec':>22} {'R2':>6} {'dose':>18} {'local risk':>18} "
              f"{'nat risk':>18} {'trend':>18}")
        for label, terms in FAMILY:
            b, se, r2 = ols([col[t] for t in terms], y)
            est = {t: (b[i + 1], se[i + 1]) for i, t in enumerate(terms)}
            rows.append(dict(group=g, spec=label, n=len(y), r2=round(float(r2), 3),
                             **{f"b_{t}": round(float(est[t][0]), 4) for t in terms},
                             **{f"se_{t}": round(float(est[t][1]), 4) for t in terms}))
            cells = "".join(
                f"{est[t][0]:+9.4f} ({est[t][1]:.4f})" if t in est else f"{'':>18}"
                for t in ("dose", "lrisk", "nrisk", "trend"))
            print(f"{label:>22} {r2:>6.3f} {cells}")

    R = pd.DataFrame(rows)
    out["family"] = rows

    print("\n" + "=" * 78)
    print("S4 gate: does any coefficient change sign inside the family?")
    print("=" * 78)
    verdict = []
    for g in GROUPS:
        sub = R[R.group == g]
        line = {"group": g}
        for t, name in (("b_dose", "dose"), ("b_lrisk", "local risk"),
                        ("b_nrisk", "national risk")):
            v = sub[t].dropna() if t in sub else pd.Series(dtype=float)
            if len(v) == 0:
                continue
            flip = bool(v.min() < 0 < v.max())
            line[name] = f"[{v.min():+.4f}, {v.max():+.4f}]" + ("  FLIPS" if flip else "")
            line[name + "_flips"] = flip
        verdict.append(line)
        print(f"  {g:>6}  dose {line['dose']:>28}   local risk {line['local risk']:>28}")
    out["verdict"] = verdict
    n_flip = sum(v.get("local risk_flips", False) for v in verdict)
    print(f"\n  groups whose LOCAL RISK coefficient changes sign inside the family: "
          f"{n_flip} of {len(verdict)}")
    print(f"  groups whose DOSE coefficient changes sign inside the family: "
          f"{sum(v.get('dose_flips', False) for v in verdict)} of {len(verdict)}")

    print("\n  dose coefficient, first vs last spec of the family:")
    for g in GROUPS:
        s = R[R.group == g]
        a = s[s.spec == "dose"].iloc[0]
        z = s[s.spec == "+ local risk + trend"].iloc[0]
        print(f"    {g:>6}: {a.b_dose:+.4f} (se {a.se_dose:.4f})  ->  "
              f"{z.b_dose:+.4f} (se {z.se_dose:.4f})"
              f"{'   [loses significance]' if abs(z.b_dose) < 2*z.se_dose else ''}")

    with open(f"{ROOT}/eda/results_p15.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p15.json")


if __name__ == "__main__":
    main()
