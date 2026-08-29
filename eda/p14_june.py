#!/usr/bin/env python
"""Phase 14 — what the June file adds, once the year is complete.

June was the last hole in the series. Three things it makes possible that the
11-month panel could not support:

  PLB3  A same-dose TRIANGLE. May, June and July all sit at policy dose ~1.0
        (1.06 / 1.00 / 1.00) and span spring to midsummer, so the seasonal
        placebo goes from one pair to three, and June is the only interior
        month of the three, i.e. the one a seasonal trend would have to pass
        THROUGH rather than end at.
  FAC   June is a third month whose five weekday cells are all holiday-free
        (Feb and Jul were the other two, and the weekday re-balancing factors
        of Phase 8 are estimated from those two). So the factors now have an
        out-of-sample month to be checked against.
  RHO   The S1 criterion rho* was a single odd/even split of the 424 dong. The
        split is arbitrary and dong codes are spatially ordered, so it is
        re-estimated here over repeated random splits, on 12 months.

Also fixes a calendar omission that only mattered once June existed: 현충일
(2020-06-06, a Saturday) was missing from calendar_kr.HOLIDAYS_2020.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import DERIVED, connect, cell_table, ROOT, YMS, YM_LABEL, DOW_LABEL, AGE_LABEL, AGES


def main():
    con = connect()
    out = {}

    cal = pd.DataFrame(cell_table())[["ym", "dow_n", "n_days", "n_holiday", "dose", "dose_alt"]]
    CLEAN = cal[cal.n_holiday == 0][["ym", "dow_n"]]
    WD_CLEAN = cal[(cal.n_holiday == 0) & (cal.dow_n <= 5)]
    print(f"clean cells: {len(CLEAN)} of {len(cal)};  clean weekday cells: {len(WD_CLEAN)}")
    print("months whose five weekday cells are ALL clean: "
          + ", ".join(YM_LABEL[y] for y in YMS
                      if (WD_CLEAN.ym == y).sum() == 5))
    out["n_clean_cells"] = int(len(CLEAN))
    out["n_clean_weekday_cells"] = int(len(WD_CLEAN))
    out["dose_by_month"] = (cal[cal.dow_n <= 5].groupby("ym")["dose"].mean().round(4)
                            .to_dict())

    core = pd.read_parquet(f"{DERIVED}/panel_core.parquet")
    WD = core[core.dow_n <= 5].merge(CLEAN, on=["ym", "dow_n"])

    # --------------------------------------------------- weekday re-balancing factors
    def factors(ref_yms):
        r = WD[WD.ym.isin(ref_yms)].groupby(["ym", "dow_n", "arr_hour"],
                                            as_index=False)["mid"].sum()
        r["f"] = r["mid"] / r.groupby(["ym", "arr_hour"])["mid"].transform("mean")
        return r.groupby(["dow_n", "arr_hour"], as_index=False)["f"].mean()


    print("\n" + "=" * 78)
    print("FAC  weekday factors: is June (unused in fitting them) consistent with Feb+Jul?")
    print("=" * 78)
    F_old = factors([202002, 202007])                      # Phase 8's choice
    F_jun = factors([202006])
    F_new = factors([202002, 202006, 202007])
    cmp = (F_old.rename(columns={"f": "feb_jul"})
           .merge(F_jun.rename(columns={"f": "jun"}), on=["dow_n", "arr_hour"])
           .merge(F_new.rename(columns={"f": "all3"}), on=["dow_n", "arr_hour"]))
    cmp["d_jun"] = cmp["jun"] - cmp["feb_jul"]
    cmp["d_all3"] = cmp["all3"] - cmp["feb_jul"]
    print(cmp.groupby("dow_n")[["feb_jul", "jun", "all3", "d_jun", "d_all3"]].mean()
          .rename(index=DOW_LABEL).round(4).to_string())
    print(f"\n  |June - (Feb,Jul)| over all 120 (dow,hour) cells: "
          f"max={cmp['d_jun'].abs().max():.4f}  median={cmp['d_jun'].abs().median():.4f}")
    print(f"  |adding June to the fit| : max={cmp['d_all3'].abs().max():.4f}  "
          f"median={cmp['d_all3'].abs().median():.4f}")
    out["weekday_factor_check"] = dict(
        max_abs_dev_june=float(cmp["d_jun"].abs().max()),
        median_abs_dev_june=float(cmp["d_jun"].abs().median()),
        max_abs_shift_if_added=float(cmp["d_all3"].abs().max()))

    FAC = F_old  # keep Phase 8's specification; June is a check, not a re-fit


    def balanced(df, keys=("age",)):
        d = df.merge(FAC, on=["dow_n", "arr_hour"])
        for c in ("lo", "mid", "hi"):
            d[c] = d[c] / d["f"]
        return d.groupby(["ym", "arr_hour", *keys], as_index=False)[["lo", "mid", "hi"]].mean()


    BW = balanced(WD, keys=("age", "sex"))

    # ------------------------------------------------ PLB3  May / Jun / Jul triangle
    print("\n" + "=" * 78)
    print("PLB3  same-dose triangle: May(1.06) / Jun(1.00) / Jul(1.00), spring->summer")
    print("=" * 78)
    lev = BW.groupby(["ym", "age"], as_index=False)["mid"].sum()
    lp = lev.pivot_table(index="age", columns="ym", values="mid")
    tri = pd.DataFrame({
        "May/Jan": lp[202005] / lp[202001],
        "Jun/Jan": lp[202006] / lp[202001],
        "Jul/Jan": lp[202007] / lp[202001],
        "Jun/May": lp[202006] / lp[202005],
        "Jul/Jun": lp[202007] / lp[202006],
        "Jul/May": lp[202007] / lp[202005],
    })
    tri.index = [AGE_LABEL[a] for a in tri.index]
    print(tri.round(3).to_string())
    adult = tri.loc[[AGE_LABEL[a] for a in AGES if a >= 25]]
    for c in ("Jun/May", "Jul/Jun", "Jul/May"):
        print(f"\n  {c:>8}  all ages 25+:  min={adult[c].min():.3f}  "
              f"median={adult[c].median():.3f}  max={adult[c].max():.3f}"
              f"   max |dev from 1| = {(adult[c]-1).abs().max():.3f}")
    out["placebo_triangle"] = tri.round(4).reset_index(names="age").to_dict("records")
    out["placebo_triangle_25plus"] = {
        c: dict(min=float(adult[c].min()), median=float(adult[c].median()),
                max=float(adult[c].max()), max_abs_dev=float((adult[c] - 1).abs().max()))
        for c in ("Jun/May", "Jul/Jun", "Jul/May")}

    # ---------------------------------------------------- RHO  S1 criterion, redone
    print("\n" + "=" * 78)
    print("RHO  S1 criterion from repeated random split-half (12 months)")
    print("=" * 78)
    BASE = 202001
    dongs = con.execute("""
    SELECT DISTINCT d_dong FROM m WHERE d_seoul ORDER BY 1""").df()["d_dong"].to_numpy()
    print(f"  {len(dongs)} destination dong")

    cell = con.execute("""
    SELECT ym, age, arr_hour, d_dong,
           sum(pop_day) + 1.5*coalesce(sum(1.0/n_days) FILTER (masked), 0) AS v
    FROM m WHERE o_seoul AND d_seoul AND dow_n <= 5 AND n_holiday = 0
    GROUP BY 1,2,3,4""").df()
    yms = [y for y in YMS if y != BASE]


    def shapes_from(df, keys):
        b = df[df.ym == BASE].groupby(keys + ["arr_hour"])["v"].sum()
        x = df[df.ym != BASE].groupby(["ym"] + keys + ["arr_hour"])["v"].sum()
        r = (x / b).rename("r").reset_index()
        o = {}
        for k, g in r.groupby(["ym"] + keys):
            vec = g.sort_values("arr_hour")["r"].to_numpy()
            o[k] = vec / vec.mean()
        return o


    rho_draws = []
    rng = np.random.default_rng(0)
    for it in range(200):
        half = pd.Series(rng.integers(0, 2, len(dongs)), index=dongs)
        d = cell.assign(half=cell["d_dong"].map(half))
        sh = shapes_from(d, ["age", "half"])
        rel = [np.corrcoef(sh[(y, a, 0)], sh[(y, a, 1)])[0, 1] for y in yms for a in AGES]
        rho_draws.append(np.quantile(rel, 0.05))
        if it % 50 == 0:
            print(f"  draw {it:>3}  rho*={rho_draws[-1]:.3f}", flush=True)

    rho_rand = float(np.median(rho_draws))
    print(f"\n  rho* over 200 random splits: median={rho_rand:.3f}  "
          f"p05={np.quantile(rho_draws,.05):.3f}  p95={np.quantile(rho_draws,.95):.3f}")

    odd = pd.Series(dongs % 2, index=dongs)
    sh_oe = shapes_from(cell.assign(half=cell["d_dong"].map(odd)), ["age", "half"])
    rel_oe = [np.corrcoef(sh_oe[(y, a, 0)], sh_oe[(y, a, 1)])[0, 1] for y in yms for a in AGES]
    rho_oe = float(np.quantile(rel_oe, 0.05))
    print(f"  rho* from the odd/even split (Phase 13's choice), 12 months: {rho_oe:.3f}"
          f"   (n={len(rel_oe)}, median={np.median(rel_oe):.3f})")

    sh_age = shapes_from(cell, ["age"])
    sh_tot = shapes_from(cell, [])
    vs = {a: [np.corrcoef(sh_age[(y, a)], sh_tot[(y,)])[0, 1] for y in yms] for a in AGES}
    tab = pd.DataFrame({"median": {AGE_LABEL[a]: np.median(vs[a]) for a in AGES},
                        "min": {AGE_LABEL[a]: min(vs[a]) for a in AGES}})
    for name, rho in (("odd/even", rho_oe), ("random", rho_rand)):
        tab[f"pass_med_{name}"] = tab["median"] >= rho
        tab[f"pass_min_{name}"] = tab["min"] >= rho
    print("\nstratum-vs-aggregate shape correlation, 12 months:")
    print(tab.round(3).to_string())
    allvs = [v for a in AGES for v in vs[a]]
    print(f"\n  n={len(allvs)}  share >= rho*(odd/even {rho_oe:.3f}): "
          f"{np.mean(np.array(allvs) >= rho_oe):.1%}"
          f"   >= rho*(random {rho_rand:.3f}): {np.mean(np.array(allvs) >= rho_rand):.1%}")
    out["rho_star"] = dict(odd_even_12mo=rho_oe, random_median=rho_rand,
                           random_p05=float(np.quantile(rho_draws, .05)),
                           random_p95=float(np.quantile(rho_draws, .95)),
                           n_within=len(rel_oe), n_vs_agg=len(allvs),
                           share_pass_oe=float(np.mean(np.array(allvs) >= rho_oe)),
                           share_pass_rand=float(np.mean(np.array(allvs) >= rho_rand)))
    out["vs_aggregate"] = tab.round(4).reset_index(names="age").to_dict("records")

    with open(f"{ROOT}/eda/results_p14.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p14.json")


if __name__ == "__main__":
    main()
