#!/usr/bin/env python
"""Phase 5 — baseline validity and week structure.

The plan in ../reference/EDA.md assumed a daily series ("2020-01 逐日總量, 標出農曆新年窗,
baseline = 1月非假日週的星期別均值"). This file has no date column: the unit is
대상연월 x 요일 x 도착시간, so no day inside January can be included or excluded.
What can still be done is (a) measure the week structure, (b) bound how much the
holidays contaminate each weekday aggregate, and (c) identify which weekdays are
holiday-free in both months and can therefore carry the comparison.

Calendar: Jan 2020 holidays 01-01 (Wed, 신정) and 01-24..27 (Fri/Sat/Sun/Mon, 설).
Sep 2020 holidays 09-30 (Wed, 추석 eve). Tue and Thu are clean in both months.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import connect, ROOT, OCCURRENCES, HOLIDAYS, AGE_LABEL

FIG = f"{ROOT}/eda/fig"
con = connect()
out = {}
DOW = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# -------------------------------------------- 5.1 the week structure, per day
q = con.execute("""
    SELECT ym, dow_n, any_value(n_days) AS occ, sum(pop_day) AS pop_day
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1,2 ORDER BY 1,2""").df()
q["dow"] = q["dow_n"].map(DOW)
q["holidays_in_cell"] = [HOLIDAYS.get(r.ym, {}).get(r.dow_n, 0) for r in q.itertuples()]
q["holiday_frac"] = q["holidays_in_cell"] / q["occ"]
print("=== 5.1 Seoul-internal volume per calendar day, by weekday ===")
print(q[["ym", "dow", "occ", "holidays_in_cell", "holiday_frac", "pop_day"]]
      .round(4).to_string(index=False))
out["week_structure"] = q.to_dict("records")

# ------------------------------- 5.2 how deep is the holiday dip? (identified
# only through the contaminated cells, so this is a joint fit, not a measurement)
print("\n=== 5.2 implied holiday depression delta ===")
jan = q[q.ym == 202001].set_index("dow_n")
sep = q[q.ym == 202009].set_index("dow_n")
# Tue and Thu are clean in both months and bracket the week, so use their mean as
# the clean weekday level; solve pop_day = clean * (1 - holiday_frac * delta).
for name, tab in (("202001", jan), ("202009", sep)):
    clean = (tab.loc[2, "pop_day"] + tab.loc[4, "pop_day"]) / 2
    rows = []
    for d in (1, 3, 5):
        hf = tab.loc[d, "holiday_frac"]
        if hf == 0:
            continue
        delta = (1 - tab.loc[d, "pop_day"] / clean) / hf
        rows.append((DOW[d], hf, tab.loc[d, "pop_day"], clean, delta))
    for r in rows:
        print(f"  {name} {r[0]}: holiday_frac={r[1]:.3f}  observed={r[2]:,.0f}  "
              f"clean_ref={r[3]:,.0f}  implied delta={r[4]:.3f}")
    out[f"holiday_delta_{name}"] = rows

print("\n  NOTE: delta conflates the holiday dip with the genuine Mon/Wed/Fri-vs-"
      "Tue/Thu weekday effect; both are only separable with a daily series.")

# ---------------------------- 5.3 how much of January is already post-first-case
print("\n=== 5.3 January contamination that cannot be excluded ===")
print(f"  Korea's first confirmed case 2020-01-20; 2020-01-20..31 = 12 of 31 days"
      f" = {12/31:.1%} of the month.")
print(f"  Every weekday cell in 202001 mixes pre- and post-01-20 days:")
for d in range(1, 8):
    days = [x for x in range(1, 32) if (x + 2) % 7 == d % 7]  # 2020-01-01 = Wed
    post = [x for x in days if x >= 20]
    print(f"    {DOW[d]}: {len(days)} occurrences, {len(post)} on/after 01-20 "
          f"({len(post)/len(days):.0%}), days={days}")
    out.setdefault("jan_post_share", []).append(
        dict(dow=DOW[d], n=len(days), post=len(post), days=days))

# ----------------------------------- 5.4 school-vacation confound on young bands
print("\n=== 5.4 Jan = school winter vacation, Sep = term: young-band day profile ===")
d = con.execute("""
    SELECT age, arr_hour, ym, sum(pop_day) AS pop_day
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4) AND age IN (0,10,15,20)
    GROUP BY 1,2,3""").df()
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharex=True)
for ax, a in zip(axes, (0, 10, 15, 20)):
    for ym, g in d[d.age == a].groupby("ym"):
        g = g.sort_values("arr_hour")
        ax.plot(g["arr_hour"], g["pop_day"] / 1e3, marker="o", ms=2.5, label=str(ym))
    ax.set_title(f"age {AGE_LABEL[a]}", fontsize=10)
    ax.set_xlabel("arrival hour"); ax.grid(alpha=.3)
axes[0].set_ylabel("thousand arrivals/day"); axes[0].legend(fontsize=8)
fig.suptitle("Seoul-internal arrivals, Tue+Thu — school-age bands", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/p5_school_bands.png", dpi=150); plt.close(fig)
print("  -> fig/p5_school_bands.png")
out["school_bands"] = d.to_dict("records")

# ------------------------------------------------------ 5.5 week-shape stability
print("\n=== 5.5 weekday shape: is the day-of-week pattern stable across months? ===")
w = q.pivot(index="dow_n", columns="ym", values="pop_day")
w = w / w.mean()
w.index = [DOW[i] for i in w.index]
w["change"] = w[202009] - w[202001]
print(w.round(4).to_string())
out["week_shape"] = w.reset_index().to_dict("records")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for ym in (202001, 202009):
    ax.plot(list(DOW.values()), q[q.ym == ym].sort_values("dow_n")["pop_day"] / 1e6,
            marker="o", label=str(ym))
for d in (1, 3, 5, 6, 7):
    ax.axvline(DOW[d], color="0.9", zorder=0)
ax.set_ylabel("million Seoul-internal moves / day"); ax.grid(alpha=.3); ax.legend()
ax.set_title("week structure (grey = weekday cells containing a holiday in Jan)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/p5_week_structure.png", dpi=150); plt.close(fig)
print("  -> fig/p5_week_structure.png")

with open(f"{ROOT}/eda/results_p5.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("\nwrote results_p5.json")
