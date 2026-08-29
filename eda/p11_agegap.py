#!/usr/bin/env python
"""Phase 11 — the protective age gradient inverts over 2020.

In February and March the oldest groups cut mobility harder than prime-age
adults; by September they cut less, even though September carried the higher
policy dose. This is the age decomposition of the aggregate efficacy decay in
p10_response.py, and it lands on the highest-IFR group.

The month-to-month statement is reported two ways: relative to January (readable
but baseline-dependent) and as a March-to-September ratio of ratios, which needs
no baseline at all.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import connect, ROOT, YMS, YM_LABEL, GU_NAMES

FIG = f"{ROOT}/eda/fig"


def main():
    con = connect()
    out = {}

    GROUPS = {"20-24": [20], "25-44": [25, 30, 35, 40], "45-64": [45, 50, 55, 60],
              "65-74": [65, 70], "75+": [75, 80]}

    # per-day weekday volume, holiday-free cells only, averaged over the clean cells
    series = {}
    for k, ages in GROUPS.items():
        v = con.execute(f"""
        SELECT ym, dow_n, sum(pop_day) AS v FROM m
        WHERE o_seoul AND d_seoul AND dow_n <= 5 AND n_holiday = 0
          AND age IN ({','.join(map(str, ages))})
        GROUP BY 1,2""").df().groupby("ym", as_index=False)["v"].mean().set_index("ym")["v"]
        series[k] = v
    V = pd.DataFrame(series)
    R = V / V.loc[202001]
    R.index = [YM_LABEL[i] for i in R.index]
    R["gap 65-74"] = R["65-74"] - R["25-44"]
    R["gap 75+"] = R["75+"] - R["25-44"]

    print("=== 11.1 per-day weekday volume, Jan = 1 ===")
    print(R.round(3).to_string())
    out["indexed"] = R.reset_index().to_dict("records")

    print("\n=== 11.2 baseline-free: March -> September, by group ===")
    print("   (September carries the HIGHER policy dose: 2.21 vs 1.81)")
    ms = pd.DataFrame({"Mar": V.loc[202003], "Sep": V.loc[202009]})
    ms["Sep/Mar"] = ms["Sep"] / ms["Mar"]
    print(ms.round(3).to_string())
    print(f"\n   65-74 rose {(ms.loc['65-74','Sep/Mar']-1)*100:+.1f}% while 25-44 rose "
          f"{(ms.loc['25-44','Sep/Mar']-1)*100:+.1f}% — under a stricter regime.")
    print(f"   75+   rose {(ms.loc['75+','Sep/Mar']-1)*100:+.1f}%.")
    out["mar_to_sep"] = ms.reset_index().to_dict("records")

    print("\n=== 11.3 same reversal at gu level? ===")
    g = con.execute("""
    SELECT d_gu, ym, dow_n,
           sum(pop_day) FILTER (age >= 65)                AS old,
           sum(pop_day) FILTER (age BETWEEN 25 AND 44)    AS work
    FROM m WHERE o_seoul AND d_seoul AND dow_n <= 5 AND n_holiday = 0
    GROUP BY 1,2,3""").df().groupby(["d_gu", "ym"], as_index=False)[["old", "work"]].mean()
    w = g.pivot(index="d_gu", columns="ym", values=["old", "work"])
    gu = pd.DataFrame({
        "gu": [GU_NAMES.get(int(i), i) for i in w.index],
        "old Sep/Mar": (w[("old", 202009)] / w[("old", 202003)]).round(3),
        "work Sep/Mar": (w[("work", 202009)] / w[("work", 202003)]).round(3)})
    gu["excess"] = (gu["old Sep/Mar"] - gu["work Sep/Mar"]).round(3)
    gu = gu.sort_values("excess")
    print(pd.concat([gu.head(4), gu.tail(4)]).to_string(index=False))
    print(f"\n   all 25 gu positive: {bool((gu.excess > 0).all())}   "
          f"range {gu.excess.min():+.3f} to {gu.excess.max():+.3f}")
    out["gu_reversal"] = gu.to_dict("records")

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    lab = [YM_LABEL[m] for m in YMS]
    for k, c in zip(GROUPS, ["#7A5195", "#0E7C86", "#4C9F70", "#BC8034", "#A8434E"]):
        axes[0].plot(lab, R[k], marker="o", ms=4, lw=1.5, color=c, label=k)
    axes[0].axhline(1, color="k", lw=.6, ls="--")
    axes[0].set_title("per-day weekday volume, Jan = 1", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

    axes[1].axhline(0, color="k", lw=.8)
    axes[1].plot(lab, R["gap 65-74"], marker="o", ms=5, lw=1.8, color="#BC8034",
                 label="65-74  minus  25-44")
    axes[1].plot(lab, R["gap 75+"], marker="s", ms=5, lw=1.8, color="#A8434E",
                 label="75+  minus  25-44")
    axes[1].fill_between(lab, R["gap 75+"], 0, where=R["gap 75+"] < 0, alpha=.12,
                         color="#A8434E")
    axes[1].set_title("age gap in mobility retention\n(negative = elderly cut more)",
                      fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.suptitle("Phase 11 — the protective age gradient inverts across 2020", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/p11_age_gap.png", dpi=150); plt.close(fig)
    print("\n  -> fig/p11_age_gap.png")

    with open(f"{ROOT}/eda/results_p11.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p11.json")


if __name__ == "__main__":
    main()
