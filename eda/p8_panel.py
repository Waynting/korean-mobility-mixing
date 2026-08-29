#!/usr/bin/env python
"""Phase 8 — the 8-month panel: what the monthly series unlocks.

Four things could not be answered with two months and can be now:
  T1.1  does the per-device expansion weight move across months?  (assumption A1)
  T2.1  is the H/W/E composition drifting smoothly or jumping at policy dates?
        (assumption A2 — the decisive label-drift test)
  P6    the real dose ladder: is the diurnal response SHAPE invariant to dose,
        and is it common across strata?
  PLB   May vs Jul sit at essentially the same policy dose in different seasons,
        which is a free placebo for the season/schooling confound.

D3 is generalised: with 8 months no weekday is holiday-free everywhere, so the
exclusion moves from weekday level to CELL level (12 of 56 cells dropped), and
weekday composition is re-balanced with hour-specific weekday factors estimated
from the two months whose weekdays are all clean (Feb, Jul).
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
from common import (DERIVED, connect, cell_table, ROOT, YMS, YM_LABEL,
                    DOW_LABEL, AGE_LABEL, AGES)

FIG = f"{ROOT}/eda/fig"


def main():
    con = connect()
    out = {}

    cal = pd.DataFrame(cell_table())[
        ["ym", "dow_n", "n_days", "n_holiday", "holiday_frac", "dose", "dose_alt"]]
    CLEAN = cal[cal.n_holiday == 0][["ym", "dow_n"]]
    print(f"clean cells: {len(CLEAN)} of {len(cal)}")
    print(cal.assign(clean=cal.n_holiday == 0)
          .pivot(index="ym", columns="dow_n", values="clean")
          .rename(columns=DOW_LABEL, index=YM_LABEL).to_string())

    # ------------------------------------------------------------ core aggregation
    print("\naggregating Seoul-internal flows by ym x dow x hour x age x sex ...")
    core = con.execute("""
    SELECT ym, dow_n, arr_hour, age, sex,
           sum(pop_day)                                          AS lo,
           coalesce(sum(1.0/n_days) FILTER (masked), 0)          AS n_masked_day,
           count(*)                                              AS cells,
           count(*) FILTER (masked)                              AS masked_cells
    FROM m WHERE o_seoul AND d_seoul
    GROUP BY 1,2,3,4,5""").df()
    core["mid"] = core["lo"] + 1.5 * core["n_masked_day"]
    core["hi"] = core["lo"] + 3.0 * core["n_masked_day"]
    core = core.merge(cal, on=["ym", "dow_n"])
    core.to_parquet(f"{DERIVED}/panel_core.parquet")
    print(f"  {len(core):,} rows -> derived/panel_core.parquet")

    WD = core[(core.dow_n <= 5)].merge(CLEAN, on=["ym", "dow_n"])
    WE = core[(core.dow_n >= 6)].merge(CLEAN, on=["ym", "dow_n"])

    # ------------------------------- weekday re-balancing factors from Feb and Jul
    ref = WD[WD.ym.isin([202002, 202007])].groupby(["ym", "dow_n", "arr_hour"],
                                                   as_index=False)["mid"].sum()
    ref["f"] = ref["mid"] / ref.groupby(["ym", "arr_hour"])["mid"].transform("mean")
    FAC = ref.groupby(["dow_n", "arr_hour"], as_index=False)["f"].mean()
    print("\nweekday factors (mean over hours), estimated from Feb+Jul:")
    print(FAC.groupby("dow_n")["f"].mean().rename(index=DOW_LABEL).round(4).to_string())
    out["weekday_factors"] = FAC.to_dict("records")


    def balanced(df, keys=("age", "sex")):
        """per-day flow, weekday-composition-adjusted, averaged over clean cells."""
        d = df.merge(FAC, on=["dow_n", "arr_hour"])
        for c in ("lo", "mid", "hi"):
            d[c] = d[c] / d["f"]
        g = d.groupby(["ym", "arr_hour", *keys], as_index=False)[["lo", "mid", "hi"]].mean()
        return g


    BW = balanced(WD)
    BW_all = balanced(WD, keys=())

    # =========================================================== T1.1  assumption A1
    print("\n" + "=" * 78)
    print("T1.1  per-device expansion weight by month (assumption A1)")
    print("=" * 78)
    w = con.execute("""
    SELECT ym, age, sex, min(pop) AS min_pop
    FROM m WHERE o_seoul AND NOT masked GROUP BY 1,2,3""").df()
    piv = w.pivot_table(index=["age", "sex"], columns="ym", values="min_pop")
    piv.index = [f"{AGE_LABEL[a]}{s}" for a, s in piv.index]
    piv.columns = [YM_LABEL[c] for c in piv.columns]
    piv["cv"] = piv.std(axis=1) / piv.mean(axis=1)
    print(piv.round(3).to_string())
    out["weight_by_month"] = piv.reset_index().to_dict("records")
    print(f"\n  coefficient of variation across months: max={piv['cv'].max():.4f} "
          f"(stratum {piv['cv'].idxmax()}), median={piv['cv'].median():.4f}")

    # =========================================================== T2.1  assumption A2
    print("\n" + "=" * 78)
    print("T2.1  H/W/E composition trajectory (assumption A2)")
    print("=" * 78)
    mt = con.execute("""
    SELECT ym, dow_n, mtype, sum(pop_day) AS pop_day
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1,2,3""").df()
    mt = mt.merge(CLEAN, on=["ym", "dow_n"])
    mtw = mt[mt.dow_n <= 5].groupby(["ym", "mtype"], as_index=False)["pop_day"].mean()
    mtw["share"] = mtw["pop_day"] / mtw.groupby("ym")["pop_day"].transform("sum")
    sh = mtw.pivot(index="ym", columns="mtype", values="share")
    lv = mtw.pivot(index="ym", columns="mtype", values="pop_day")
    sh.index = [YM_LABEL[i] for i in sh.index]
    lv.index = [YM_LABEL[i] for i in lv.index]
    print("\nshare of Seoul-internal weekday volume:")
    print(sh.round(4).to_string())
    print("\nlevel, indexed to Jan = 1.00:")
    print((lv / lv.iloc[0]).round(3).to_string())
    out["mtype_share"] = sh.reset_index().to_dict("records")
    out["mtype_level"] = lv.reset_index().to_dict("records")

    anchor = mt[mt.dow_n <= 5].assign(o=lambda d: d.mtype.str[0]).groupby(
        ["ym", "o"], as_index=False)["pop_day"].mean()
    ap = anchor.pivot(index="ym", columns="o", values="pop_day")
    ap.index = [YM_LABEL[i] for i in ap.index]
    print("\norigin anchor level, indexed to Jan = 1.00:")
    print((ap / ap.iloc[0]).round(3).to_string())
    out["anchor_level"] = ap.reset_index().to_dict("records")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    x = range(len(YMS))
    lab = [YM_LABEL[y] for y in YMS]
    for k in ["HW", "WH", "HE", "HH", "EE", "WE"]:
        axes[0].plot(x, sh[k], marker="o", ms=4, label=k)
    axes[0].set_title("type share of weekday volume", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    for k in ["HW", "WH", "HE", "HH", "EE", "WE"]:
        axes[1].plot(x, lv[k] / lv[k].iloc[0], marker="o", ms=4, label=k)
    axes[1].set_title("type level, Jan = 1", fontsize=10)
    axes[1].axhline(1, color="k", lw=.6, ls="--")
    for k in ["E", "H", "W"]:
        axes[2].plot(x, ap[k] / ap[k].iloc[0], marker="o", ms=4, label=f"{k} anchor")
    axes[2].set_title("origin anchor level, Jan = 1", fontsize=10)
    axes[2].axhline(1, color="k", lw=.6, ls="--"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(list(x)); ax.set_xticklabels(lab); ax.grid(alpha=.3)
    fig.suptitle("T2.1 — label drift diagnostics over the 8-month panel", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/p8_label_drift.png", dpi=150); plt.close(fig)
    print("  -> fig/p8_label_drift.png")

    # ============================================================ dose ladder setup
    DOSE = cal.merge(CLEAN, on=["ym", "dow_n"])
    DOSE = DOSE[DOSE.dow_n <= 5].groupby("ym", as_index=False)["dose"].mean()
    DOSE["label"] = [YM_LABEL[y] for y in DOSE.ym]
    print("\n" + "=" * 78)
    print("dose ladder (mean over clean weekday cells)")
    print("=" * 78)
    print(DOSE.sort_values("dose").to_string(index=False))
    out["dose_ladder"] = DOSE.to_dict("records")

    # =========================================== P6  shape invariance across doses
    BASE = 202001
    print("\n" + "=" * 78)
    print("P6  diurnal response shapes over the dose ladder")
    print("=" * 78)


    def shapes_by(df, keys):
        b = df[df.ym == BASE].set_index([*keys, "arr_hour"])["mid"]
        d = df[df.ym != BASE].set_index(["ym", *keys, "arr_hour"])["mid"]
        r = (d / b).rename("r").reset_index()
        sh = {}
        for k, g in r.groupby(["ym", *keys]):
            v = g.sort_values("arr_hour")["r"].to_numpy()
            sh[k] = v / v.mean()
        return sh


    sh_strat = shapes_by(BW, ["age", "sex"])
    sh_agg = shapes_by(BW_all, [])

    # (a) across-strata, within dose  — the mixture-artefact test, per month
    print("\n(a) across-strata shape correlation, computed separately at each dose")
    rows = []
    for ym in YMS:
        if ym == BASE:
            continue
        for lab_sub, ages in [("all 16", AGES), ("25+", [a for a in AGES if a >= 25]),
                              ("45+", [a for a in AGES if a >= 45])]:
            keys = [(ym, a, s) for s in ("F", "M") for a in ages]
            M = np.array([sh_strat[k] for k in keys])
            C = np.corrcoef(M)
            iu = np.triu_indices(len(keys), 1)
            v = C[iu]
            rows.append(dict(ym=YM_LABEL[ym], dose=float(DOSE.set_index("ym").dose[ym]),
                             subset=lab_sub, n=len(v), min=v.min(),
                             median=float(np.median(v)), frac95=float((v > .95).mean())))
    t = pd.DataFrame(rows)
    print(t.pivot(index=["ym", "dose"], columns="subset",
                  values=["median", "min", "frac95"]).round(3).to_string())
    out["across_strata_by_dose"] = rows

    # (b) across-dose, within stratum — is the shape a system property?
    print("\n(b) across-dose shape correlation, within each stratum")
    rows = []
    for a in AGES:
        for s in ("F", "M"):
            M = np.array([sh_strat[(ym, a, s)] for ym in YMS if ym != BASE])
            C = np.corrcoef(M)
            iu = np.triu_indices(M.shape[0], 1)
            v = C[iu]
            rows.append(dict(age=AGE_LABEL[a], sex=s, n=len(v), min=v.min(),
                             median=float(np.median(v)), frac95=float((v > .95).mean())))
    t2 = pd.DataFrame(rows)
    print(t2.pivot(index="age", columns="sex",
                   values=["median", "min", "frac95"]).reindex(
        [AGE_LABEL[a] for a in AGES]).round(3).to_string())
    out["across_dose_by_stratum"] = rows
    Ma = np.array([sh_agg[(ym,)] for ym in YMS if ym != BASE])
    Ca = np.corrcoef(Ma)
    iu = np.triu_indices(Ma.shape[0], 1)
    print(f"\n  aggregate curve, across-dose: min={Ca[iu].min():.3f} "
          f"median={np.median(Ca[iu]):.3f}  frac>0.95={(Ca[iu]>.95).mean():.1%}")
    out["across_dose_aggregate"] = dict(min=float(Ca[iu].min()),
                                        median=float(np.median(Ca[iu])),
                                        frac95=float((Ca[iu] > .95).mean()))

    # curves figure
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, ym in zip(axes.ravel(), [y for y in YMS if y != BASE]):
        for i, a in enumerate(AGES):
            ax.plot(range(24), sh_strat[(ym, a, "F")], color=cmap(i / 15), lw=1.1)
        ax.plot(range(24), sh_agg[(ym,)], color="crimson", lw=2.2, ls="--")
        ax.set_title(f"{YM_LABEL[ym]}  dose={DOSE.set_index('ym').dose[ym]:.2f}", fontsize=10)
        ax.axhline(1, color="k", lw=.5); ax.grid(alpha=.3)
    axes[-1][-1].axis("off")
    fig.suptitle("normalised diurnal response vs Jan, by age band (F). red dashed = aggregate",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/p8_shapes_by_dose.png", dpi=150); plt.close(fig)
    print("  -> fig/p8_shapes_by_dose.png")

    # ============================================ PLB  May vs Jul placebo (same dose)
    print("\n" + "=" * 78)
    print("PLB  May vs Jul — same policy dose (1.08 vs 1.00), different season")
    print("=" * 78)
    lev = BW.groupby(["ym", "age", "sex"], as_index=False)["mid"].sum()
    lp = lev.pivot_table(index=["age", "sex"], columns="ym", values="mid")
    plb = pd.DataFrame({
        "May/Jan": lp[202005] / lp[202001],
        "Jul/Jan": lp[202007] / lp[202001],
        "Jul/May": lp[202007] / lp[202005],
    })
    plb.index = [f"{AGE_LABEL[a]}{s}" for a, s in plb.index]
    print(plb.round(3).to_string())
    print(f"\n  Jul/May over all strata: min={plb['Jul/May'].min():.3f} "
          f"median={plb['Jul/May'].median():.3f} max={plb['Jul/May'].max():.3f}")
    out["placebo_may_jul"] = plb.reset_index().to_dict("records")

    # ============================= within-month identifying variation after two-way FE
    print("\n" + "=" * 78)
    print("identifying variation in dose after month + weekday fixed effects")
    print("=" * 78)
    d = cal[(cal.dow_n <= 5)].copy()
    d = d.merge(CLEAN, on=["ym", "dow_n"])
    d["resid"] = (d["dose"] - d.groupby("ym")["dose"].transform("mean")
                  - d.groupby("dow_n")["dose"].transform("mean") + d["dose"].mean())
    print(f"  raw dose      sd = {d['dose'].std():.4f}  range = "
          f"[{d['dose'].min():.2f}, {d['dose'].max():.2f}]")
    print(f"  after 2-way FE sd = {d['resid'].std():.4f}  range = "
          f"[{d['resid'].min():+.3f}, {d['resid'].max():+.3f}]")
    print(f"  share of dose variance surviving both FE: "
          f"{d['resid'].var()/d['dose'].var():.2%}")
    out["identifying_variation"] = dict(raw_sd=float(d["dose"].std()),
                                        resid_sd=float(d["resid"].std()),
                                        share=float(d["resid"].var() / d["dose"].var()))

    with open(f"{ROOT}/eda/results_p8.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p8.json")


if __name__ == "__main__":
    main()
