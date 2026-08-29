#!/usr/bin/env python
"""Phase 30 — re-price the 20-44 results with the measured masked-cell fill.

p29 measured what a masked dong cell actually holds, by differencing our dong
data against the official 자치구 release: 2.06 to 2.81 depending on age band,
against the 1.5 midpoint this project has assumed everywhere. This script spends
that measurement on the quantities it changes, which are the 20-44 ones — 45+ and
0-19 carry no masked cells at all, so nothing there moves by construction.

THE FILL IS PER BAND, not the 2.27 overall average. p29 resolves it by age and
the spread is real (25-29 at 2.07, 40-44 at 2.81), running opposite to masking
intensity: the more a band is masked, the lower the average hidden cell, which is
what you would expect if the threshold bites deeper into a band whose cells are
smaller. Using one number would import the aggregation error this project spends
section 5 warning about.

TWO CAVEATS TRAVEL WITH EVERY NUMBER BELOW.

  * It is a LOWER bound. The gu file masks 9.3-9.4% of its own cells and those
    count as zero on both sides of the difference, so the true fill is higher and
    the true rho-hat higher still.
  * It was measured on Seoul-internal flows at gu level, and rho-hat's numerator
    is origin-Seoul with destination free. The threshold mechanism is the same
    but the trip set is not, so the transfer is an assumption — stated here, not
    hidden. Both months agree to within 0.03 on every band, so the fill itself is
    stable even if its transfer is not.

    python eda/p30_measured_fill.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
import numpy as np
import pandas as pd
from common import AGE_LABEL, AGES, cell_table
from paths import DERIVED, ROOT, require

MASKED_BANDS = [20, 25, 30, 35, 40]      # the only bands with masked cells
out = {}


def measured_fill():
    """Per-band mean of a masked dong cell, averaged over p29's months."""
    p = f"{ROOT}/eda/results_p29.json"
    require(p, "results_p29.json (run eda/p29_gu_reconcile.py)")
    d = json.load(open(p))
    rows = []
    for ym, v in d.items():
        for lbl, x in v["by_age"].items():
            rows.append(dict(ym=ym, label=lbl, cells=x["masked_cells"],
                             per_cell=x["per_cell"]))
    D = pd.DataFrame(rows)
    D = D[D.cells > 1e4]                 # below that the difference is noise
    fill = D.groupby("label").per_cell.mean()
    spread = D.groupby("label").per_cell.agg(lambda s: s.max() - s.min())
    return fill, spread


def main():
    fill, spread = measured_fill()
    lab2age = {AGE_LABEL[a]: a for a in AGES}
    f = {lab2age[k]: v for k, v in fill.items() if k in lab2age}
    print("=== 0 the measured fill, per band ===")
    print(f"  {'band':>6} {'fill':>6} {'vs 1.5':>7} {'month spread':>13}")
    for a in sorted(f):
        print(f"  {AGE_LABEL[a]:>6} {f[a]:6.2f} {f[a] / 1.5:6.2f}x "
              f"{spread[AGE_LABEL[a]]:13.3f}")
    out["fill"] = {AGE_LABEL[a]: round(float(v), 4) for a, v in f.items()}

    # ------------------------------------------------------------ 1 rho-hat
    r = pd.read_parquet(DERIVED / "rho_numerator.parquet")
    r = r[r.age.isin(MASKED_BANDS)].copy()
    r["fill"] = r.age.map(f)
    r["V_meas"] = (r.v_obs + r.fill * r.n_masked) / r.days
    g = (r.groupby(["ym", "age"])
         .agg(V_lo=("V_lo", "sum"), V_mid=("V_mid", "sum"),
              V_hi=("V_hi", "sum"), V_meas=("V_meas", "sum"),
              pop=("regpop", "sum"), masked=("n_masked", "sum"))
         .reset_index())
    for c in ("lo", "mid", "hi", "meas"):
        g[f"rho_{c}"] = g[f"V_{c}"] / g["pop"]

    yr = (g.groupby("age")
          .agg(rho_lo=("rho_lo", "mean"), rho_mid=("rho_mid", "mean"),
               rho_hi=("rho_hi", "mean"), rho_meas=("rho_meas", "mean")))
    print("\n=== 1 rho-hat over 2020, the interval against the measurement ===")
    print(f"  {'band':>6} {'lo':>7} {'mid':>7} {'hi':>7} | {'measured':>9} "
          f"{'old width':>10} {'position':>9}")
    rows = []
    for a in MASKED_BANDS:
        x = yr.loc[a]
        width = (x.rho_hi - x.rho_lo) / x.rho_mid
        pos = (x.rho_meas - x.rho_lo) / (x.rho_hi - x.rho_lo)
        print(f"  {AGE_LABEL[a]:>6} {x.rho_lo:7.3f} {x.rho_mid:7.3f} "
              f"{x.rho_hi:7.3f} | {x.rho_meas:9.3f} {width:10.1%} "
              f"{pos:9.0%}")
        rows.append(dict(band=AGE_LABEL[a], rho_lo=float(x.rho_lo),
                         rho_mid=float(x.rho_mid), rho_hi=float(x.rho_hi),
                         rho_measured=float(x.rho_meas),
                         old_width_frac=float(width),
                         position_in_band=float(pos)))
    out["rho_2020"] = rows
    print("  'position' is where the measurement falls inside the old 0-to-3 "
          "band: 50% would be the midpoint we had been using.")

    # ------------------------------------ 2 the Mar->Dec change, re-priced
    m = g[g.ym.isin([202003, 202012])].pivot(index="age", columns="ym")
    print("\n=== 2 Mar->Dec change in rho-hat, mid against measured ===")
    print(f"  {'band':>6} {'mid':>9} {'measured':>10} {'shift':>8}")
    rows = []
    for a in MASKED_BANDS:
        mid = (m[("rho_mid", 202012)][a] / m[("rho_mid", 202003)][a] - 1) * 100
        me = (m[("rho_meas", 202012)][a] / m[("rho_meas", 202003)][a] - 1) * 100
        print(f"  {AGE_LABEL[a]:>6} {mid:+8.2f}% {me:+9.2f}% {me - mid:+7.2f}pp")
        rows.append(dict(band=AGE_LABEL[a], mar_dec_mid_pct=float(mid),
                         mar_dec_measured_pct=float(me),
                         shift_pp=float(me - mid)))
    out["mar_to_dec"] = rows

    # ------------------------------------------- 3 does the E share move?
    # Within a fixed (age, month) the fill cancels out of a share to first
    # order, but not exactly: it enters the numerator and denominator with
    # different masked-cell counts. p18/p19's headline for 20-24 is -3.956 pp,
    # so the question is whether re-pricing moves the third decimal.
    # Built exactly as p19 builds it -- per-day volumes and per-day masked
    # counts, both divided by that weekday's occurrences before summing. The
    # monthly-sum version weights weekdays differently and shifts the answer by
    # ~0.2 pp, which is the size of the effect being measured, so the
    # construction has to match rather than merely resemble.
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("CREATE TABLE cal(ym INTEGER, dow_n TINYINT, n_days TINYINT, "
                "n_holiday TINYINT)")
    con.executemany("INSERT INTO cal VALUES (?,?,?,?)",
                    [(r["ym"], r["dow_n"], r["n_days"], r["n_holiday"])
                     for r in cell_table()])
    base = con.execute(f"""
        SELECT g.ym, g.age, substr(g.mtype, 2, 1) AS dest_attr,
               sum(g.pop_day_unmasked)          AS lo,
               sum(g.n_masked_cells / c.n_days) AS mcells
        FROM read_parquet('{DERIVED}/gu_level.parquet') g
        JOIN cal c ON c.ym = g.ym AND c.dow_n = g.dow_n
        WHERE g.ym IN (202003, 202012)
          AND g.o_gu BETWEEN 1101 AND 1125 AND g.d_gu BETWEEN 1101 AND 1125
          AND g.age IN ({','.join(map(str, MASKED_BANDS))})
        GROUP BY 1,2,3""").df()
    print("\n=== 3 the E-share change for 20-44, re-priced ===")
    print(f"  {'band':>6} {'mid':>9} {'measured':>10} {'shift':>9}")
    rows = []
    for a in MASKED_BANDS:
        b = base[base.age == a]
        sh = {}
        for tag, fillv in (("mid", 1.5), ("meas", f[a])):
            v = b.assign(n=b.lo + fillv * b.mcells)
            tot = v.groupby("ym")["n"].sum()
            e = v[v.dest_attr == "E"].groupby("ym")["n"].sum()
            sh[tag] = (e / tot * 100)
        d_mid = sh["mid"][202012] - sh["mid"][202003]
        d_me = sh["meas"][202012] - sh["meas"][202003]
        print(f"  {AGE_LABEL[a]:>6} {d_mid:+8.3f}pp {d_me:+9.3f}pp "
              f"{d_me - d_mid:+8.4f}pp")
        rows.append(dict(band=AGE_LABEL[a], e_share_mid_pp=float(d_mid),
                         e_share_measured_pp=float(d_me),
                         shift_pp=float(d_me - d_mid)))
        if a == 20:
            with open(f"{ROOT}/eda/results_p19.json") as fh:
                p19 = {r["age"]: r for r in
                       json.load(fh)["e_share_change_16band"]}
            want = p19["20-24"]["mid"]
            assert abs(d_mid - want) < .02, \
                f"mid re-build gives {d_mid:.3f}, p19 publishes {want}"
    out["e_share_change"] = rows
    print(f"  The mid column reproduces p19's published 16-band numbers, which "
          f"is what licenses the comparison.")
    print("  These are NOT invariant. A within-age share cancels a COMMON "
          "factor, but the fill is not common across destination attributes: "
          "masked cells sit disproportionately in the E panel, so re-pricing "
          "them lifts E and shrinks the fall.")

    with open(f"{ROOT}/eda/results_p30.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("\nwrote results_p30.json")


if __name__ == "__main__":
    main()
