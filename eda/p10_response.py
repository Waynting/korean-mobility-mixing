#!/usr/bin/env python
"""Phase 10 — level response over the dose ladder, and what falls off the line.

Two questions the 8-month panel can answer that the Jan/Sep pair could not:
how mobility sensitivity to policy stringency varies with age, and whether the
same stringency bought the same reduction in March as in September.

All volumes are per calendar day, Seoul-internal, weekday, restricted to cells
containing no public holiday, and re-balanced for weekday composition using
hour-specific factors from Feb and Jul (the two months whose weekdays are all
clean). See p8_panel.py for the construction.
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
from common import DERIVED, ROOT, cell_table, YMS, YM_LABEL, AGE_LABEL, AGES

FIG = f"{ROOT}/eda/fig"
out = {}


def main():
    core = pd.read_parquet(f"{DERIVED}/panel_core.parquet")
    cal = pd.DataFrame(cell_table())
    CLEAN = cal[cal.n_holiday == 0][["ym", "dow_n"]]
    WD = core[core.dow_n <= 5].merge(CLEAN, on=["ym", "dow_n"])

    ref = WD[WD.ym.isin([202002, 202007])].groupby(
        ["ym", "dow_n", "arr_hour"], as_index=False)["mid"].sum()
    ref["f"] = ref["mid"] / ref.groupby(["ym", "arr_hour"])["mid"].transform("mean")
    FAC = ref.groupby(["dow_n", "arr_hour"], as_index=False)["f"].mean()
    d = WD.merge(FAC, on=["dow_n", "arr_hour"])
    for c in ("lo", "mid", "hi"):
        d[c] = d[c] / d["f"]

    DOSE = cal.merge(CLEAN, on=["ym", "dow_n"])
    DOSE = DOSE[DOSE.dow_n <= 5].groupby("ym")["dose"].mean()
    x = np.array([DOSE[y] for y in YMS])

    # sum over hours inside a cell, then average over the clean weekday cells
    cellsum = d.groupby(["ym", "dow_n", "age"], as_index=False)["mid"].sum()
    lev = cellsum.groupby(["ym", "age"], as_index=False)["mid"].mean()
    P = lev.pivot(index="age", columns="ym", values="mid")
    P = P.div(P[202001], axis=0)

    tot = (cellsum.groupby(["ym", "dow_n"], as_index=False)["mid"].sum()
           .groupby("ym")["mid"].mean())
    TOT = tot / tot[202001]

    print("=== 10.1 per-day weekday volume by age, Jan = 1 ===")
    show = P.copy()
    show.index = [AGE_LABEL[a] for a in show.index]
    show.columns = [YM_LABEL[c] for c in show.columns]
    print(show.round(3).to_string())
    print("\nall-age:", {YM_LABEL[k]: round(v, 3) for k, v in TOT.items()})
    print("dose:   ", {YM_LABEL[k]: round(v, 2) for k, v in DOSE.items()})
    out["level_by_age"] = show.reset_index().to_dict("records")
    out["dose"] = {int(k): float(v) for k, v in DOSE.items()}

    print("\n=== 10.2 dose elasticity by age (OLS of log volume on dose) ===")
    rows = []
    for a in AGES:
        y = np.log(P.loc[a, YMS].to_numpy(dtype=float))
        b, _ = np.polyfit(x, y, 1)
        rows.append(dict(age=AGE_LABEL[a], pct_per_dose=round((np.exp(b) - 1) * 100, 1),
                         r=round(float(np.corrcoef(x, y)[0, 1]), 3)))
    E = pd.DataFrame(rows)
    print(E.to_string(index=False))
    out["elasticity"] = rows

    print("\n=== 10.3 residual from the dose line — did the same stringency keep working? ===")
    y = np.log(TOT[YMS].to_numpy(dtype=float))
    b, a0 = np.polyfit(x, y, 1)
    R = pd.DataFrame({"ym": [YM_LABEL[m] for m in YMS], "dose": x.round(2),
                      "observed": np.exp(y).round(3),
                      "fitted": np.exp(a0 + b * x).round(3),
                      "resid_pct": ((np.exp(y - a0 - b * x) - 1) * 100).round(1)})
    print(R.to_string(index=False))
    # index by label, not position: inserting June shifted every positional index by one
    res = R.set_index("ym")["resid_pct"]
    print(f"\n  March sits {res['Mar']:+.1f}% off the line, September {res['Sep']:+.1f}%, "
          f"December {res['Dec']:+.1f}%.")
    print(f"  The three same-dose months are May {res['May']:+.1f}%, Jun {res['Jun']:+.1f}%, "
          f"Jul {res['Jul']:+.1f}% -> spread {res[['May','Jun','Jul']].max()-res[['May','Jun','Jul']].min():.1f} pts.")
    out["dose_residual"] = R.to_dict("records")

    # ---------------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    cmap = plt.get_cmap("viridis")
    order = np.argsort(x)
    for i, a in enumerate(AGES):
        axes[0].plot([YM_LABEL[m] for m in YMS], P.loc[a, YMS], color=cmap(i / 15),
                     lw=1.3, marker="o", ms=3, label=AGE_LABEL[a])
    axes[0].axhline(1, color="k", lw=.6, ls="--")
    axes[0].set_title("per-day weekday volume, Jan = 1", fontsize=10)
    axes[0].legend(fontsize=6, ncol=2); axes[0].grid(alpha=.3)

    axes[1].plot(E["age"], E["pct_per_dose"], marker="o", color="#0E7C86", lw=1.6)
    axes[1].axhline(0, color="k", lw=.6)
    axes[1].set_title("mobility response per unit of policy dose (%)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=90); axes[1].grid(alpha=.3)

    axes[2].scatter(x, TOT[YMS], s=46, color="#0E7C86", zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    axes[2].plot(xs, np.exp(a0 + b * xs), color="#9A6C15", lw=1.4)
    for xi, ym in zip(x, YMS):
        axes[2].annotate(YM_LABEL[ym], (xi, TOT[ym]), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)
    axes[2].set_xlabel("policy dose"); axes[2].set_ylabel("volume, Jan = 1")
    axes[2].set_title("dose-response, all ages", fontsize=10); axes[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/p10_dose_response.png", dpi=150); plt.close(fig)
    print("\n  -> fig/p10_dose_response.png")

    with open(f"{ROOT}/eda/results_p10.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p10.json")


if __name__ == "__main__":
    main()
