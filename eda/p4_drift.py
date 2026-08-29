#!/usr/bin/env python
"""Phase 4 — H/W/E label drift diagnostics (the WFH endogeneity landmine).

Rival hypothesis: part of the measured commute decline is the classifier moving
a WFH worker's daytime anchor from W to H, i.e. a transfer between type labels
rather than a fall in total movement. Three observable signatures:
 (a) HW share vs HH/HE share moving in opposite directions at constant total;
 (b) weekend HW share, which should be near-invariant if labels are stable;
 (c) travel time by type — a reclassified 'HE' should still carry a commute-like
     travel-time distribution.
All volumes are per calendar day. Tue+Thu are the only holiday-free weekdays in
both months, so they carry the weekday comparison.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import connect, ROOT

FIG = f"{ROOT}/eda/fig"
os.makedirs(FIG, exist_ok=True)


def main():
    con = connect()
    out = {}
    MT = ["HW", "HE", "HH", "WH", "WE", "WW", "EH", "EW", "EE"]

    # --------------------------------- 4a type composition, clean weekdays vs weekend
    print("=== 4a type composition of Seoul-internal volume (per day) ===")
    df = con.execute("""
    SELECT CASE WHEN dow_n IN (2,4) THEN 'weekday(Tue,Thu)'
                WHEN dow_n >= 6 THEN 'weekend' END AS daytype,
           ym, mtype, sum(pop_day) AS pop_day
    FROM m WHERE o_seoul AND d_seoul AND (dow_n IN (2,4) OR dow_n >= 6)
    GROUP BY 1,2,3""").df()
    piv = df.pivot_table(index=["daytype", "mtype"], columns="ym", values="pop_day")
    piv["ratio_sep_jan"] = piv[202009] / piv[202001]
    tot = df.groupby(["daytype", "ym"])["pop_day"].sum()
    piv["share_jan"] = [r[202001] / tot[(i[0], 202001)] for i, r in piv.iterrows()]
    piv["share_sep"] = [r[202009] / tot[(i[0], 202009)] for i, r in piv.iterrows()]
    piv["share_pt_change"] = piv["share_sep"] - piv["share_jan"]
    print(piv.round(4).to_string())
    out["type_composition"] = piv.reset_index().to_dict("records")

    print("\n  totals per day:")
    print(tot.round(0).to_string())
    out["totals"] = tot.reset_index().to_dict("records")

    # ----------------------------------------- 4a' does the change look like transfer?
    print("\n=== 4a' transfer test: H-origin volume redistributes, or shrinks? ===")
    print(con.execute("""
    SELECT ym, substr(mtype,1,1) AS origin_anchor,
           round(sum(pop_day),0) AS pop_day
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
    GROUP BY 1,2 ORDER BY 2,1""").df().to_string(index=False))
    print(con.execute("""
    SELECT ym, substr(mtype,2,1) AS dest_anchor,
           round(sum(pop_day),0) AS pop_day
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
    GROUP BY 1,2 ORDER BY 2,1""").df().to_string(index=False))

    # ------------------------------------- 4b weekend HW share as an anomaly detector
    print("\n=== 4b weekend HW share (labels stable => near-invariant) ===")
    q = con.execute("""
    SELECT dow_n, ym,
           round(sum(pop_day) FILTER (mtype='HW')/sum(pop_day),4) AS hw_share,
           round(sum(pop_day) FILTER (mtype='HE')/sum(pop_day),4) AS he_share,
           round(sum(pop_day) FILTER (substr(mtype,1,1)='H')/sum(pop_day),4) AS h_origin_share,
           round(sum(pop_day),0) AS pop_day
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1,2 ORDER BY 1,2""").df()
    print(q.to_string(index=False))
    out["hw_share_by_dow"] = q.to_dict("records")

    # ------------------------------------------ 4b' HW share by hour, weekday/weekend
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (label, filt) in zip(axes, [("weekday (Tue,Thu)", "dow_n IN (2,4)"),
                                        ("weekend", "dow_n >= 6")]):
        d = con.execute(f"""
        SELECT arr_hour, ym, sum(pop_day) FILTER (mtype='HW')/sum(pop_day) AS hw_share
        FROM m WHERE o_seoul AND d_seoul AND {filt} GROUP BY 1,2 ORDER BY 1,2""").df()
        for ym, g in d.groupby("ym"):
            ax.plot(g["arr_hour"], g["hw_share"], marker="o", ms=3, label=str(ym))
        ax.set_title(f"HW share of arrivals — {label}", fontsize=10)
        ax.set_xlabel("arrival hour"); ax.grid(alpha=.3); ax.legend()
        out[f"hw_share_hour_{label}"] = d.to_dict("records")
    axes[0].set_ylabel("HW share of volume")
    fig.tight_layout(); fig.savefig(f"{FIG}/p4_hw_share_by_hour.png", dpi=150); plt.close(fig)
    print("  -> fig/p4_hw_share_by_hour.png")

    # ---------------------------------------------- 4c travel time by type and month
    print("\n=== 4c volume-weighted mean travel time by type (Tue+Thu, Seoul-internal) ===")
    q = con.execute("""
    SELECT mtype,
           round(sum(pop*mean_min) FILTER (ym=202001)/sum(pop) FILTER (ym=202001),1) AS min_jan,
           round(sum(pop*mean_min) FILTER (ym=202009)/sum(pop) FILTER (ym=202009),1) AS min_sep,
           round(sum(pop*mean_min) FILTER (ym=202009)/sum(pop) FILTER (ym=202009)
                 - sum(pop*mean_min) FILTER (ym=202001)/sum(pop) FILTER (ym=202001),1) AS delta
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
    GROUP BY 1 ORDER BY 1""").df()
    print(q.to_string(index=False))
    out["traveltime_by_type"] = q.to_dict("records")

    print("\n=== 4c' travel-time distribution of HE arrivals in the commute window (7-9h) ===")
    q = con.execute("""
    SELECT mtype, ym,
           round(quantile_cont(mean_min,0.25),0) AS p25,
           round(quantile_cont(mean_min,0.50),0) AS p50,
           round(quantile_cont(mean_min,0.75),0) AS p75,
           round(quantile_cont(mean_min,0.90),0) AS p90
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
      AND arr_hour BETWEEN 7 AND 9 AND mtype IN ('HW','HE','HH')
    GROUP BY 1,2 ORDER BY 1,2""").df()
    print(q.to_string(index=False))
    out["traveltime_commute_window"] = q.to_dict("records")

    with open(f"{ROOT}/eda/results_p4.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p4.json")


if __name__ == "__main__":
    main()
