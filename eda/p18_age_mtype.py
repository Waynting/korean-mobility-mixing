"""Age x 이동유형 x month cross — the §2 row-2 estimator.

Within a fixed (age, month), the share across 이동유형 is immune to both data
defects that block the level comparisons:

  * the KT expansion weight rho_a is an age-level scalar, so it appears in the
    numerator and the denominator of the share and cancels exactly;
  * `이동인구(합)` is a monthly SUM, but every 유형 in a given (ym, dow_n) cell
    is summed over the identical set of calendar days, so the occurrence count
    cancels too.

So this answers the mechanism question — does the elderly autumn/December
rebound go back into the W panel (직장) or the E panel (기타)? — without
assuming A1 (within-year stability of age-specific KT carriage).

Reads derived/gu_level.parquet, which p1_conservation.py wrote WITHOUT a Seoul
filter, so the Seoul-internal restriction is applied here to stay comparable
with p17's age index.
"""
import json
import os
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, AGE_LABEL, AGES, ROOT, YM_LABEL, YMS, cell_table


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("CREATE TABLE cal(ym INTEGER, dow_n TINYINT, n_days TINYINT, n_holiday TINYINT)")
    con.executemany("INSERT INTO cal VALUES (?,?,?,?)",
                    [(r["ym"], r["dow_n"], r["n_days"], r["n_holiday"]) for r in cell_table()])

    out = {}


    def dump(df, key, note=""):
        out[key] = json.loads(df.to_json(orient="records"))
        print(f"\n=== {key} {note}\n{df.to_string(index=False)}", flush=True)


    # --------------------------------------------------------------- 18.0 base agg
    # n_masked_cells is a raw cell count, so it must be divided by the occurrence
    # count before it can be added to pop_day_unmasked (which already is per-day).
    print("scanning gu_level.parquet (307M rows) ...", flush=True)
    base = con.execute(f"""
    SELECT g.ym, g.age, g.mtype,
           substr(g.mtype, 1, 1) AS orig_attr,
           substr(g.mtype, 2, 1) AS dest_attr,
           sum(g.pop_day_unmasked)                AS lo,
           sum(g.n_masked_cells / c.n_days)       AS n_masked_day
    FROM read_parquet('{DERIVED}/gu_level.parquet') g
    JOIN cal c ON c.ym = g.ym AND c.dow_n = g.dow_n
    WHERE g.o_gu BETWEEN 1101 AND 1125 AND g.d_gu BETWEEN 1101 AND 1125
    GROUP BY 1,2,3,4,5
""").df()
    base["mid"] = base.lo + 1.5 * base.n_masked_day
    base["hi"] = base.lo + 3.0 * base.n_masked_day
    print(f"  {len(base)} (ym, age, mtype) rows", flush=True)

    IMP = ["lo", "mid", "hi"]          # 0 / 1.5 / 3.0 imputation of the `*` cells
    FOCUS = [65, 70, 75, 80]           # the groups the age-gradient reversal is about
    REF = [20, 25, 30, 35, 40]         # the 20-44 comparison block (heavily masked)


    # ------------------------------------------------ 18.1 dest-attr share by age
    # The W/E/H split of arrivals, within (age, ym). This is the panel assignment
    # used by the mixing-matrix construction, so its stability is a prerequisite.
    for v in IMP:
        sh = base.groupby(["ym", "age", "dest_attr"])[v].sum().unstack()
        sh = sh.div(sh.sum(axis=1), axis=0)
        if v == "mid":
            share_mid = sh
        out[f"dest_share_{v}"] = {
            f"{a}": {YM_LABEL[y]: round(float(sh.loc[(y, a), s]), 5)
                     for y in YMS for s in sh.columns if (y, a) in sh.index
                     for s in [s]}
            for a in AGES}

    tbl = pd.DataFrame({
        "age": [AGE_LABEL[a] for a in AGES],
        **{f"{s}_Jan": [round(float(share_mid.loc[(202001, a), s]), 4) for a in AGES]
           for s in ["E", "H", "W"]},
        **{f"{s}_Dec": [round(float(share_mid.loc[(202012, a), s]), 4) for a in AGES]
           for s in ["E", "H", "W"]},
    })
    for s in ["E", "H", "W"]:
        tbl[f"{s}_d"] = (tbl[f"{s}_Dec"] - tbl[f"{s}_Jan"]).round(4)
    dump(tbl, "dest_share_jan_dec", "(mid imputation; arrival-attribute share within age)")


    # ------------------------------------- 18.2 W/E log ratio, differenced on month
    # log[P^W(a,m) / P^E(a,m)] - log[P^W(a,Jan) / P^E(a,Jan)].
    # rho_a cancels in the inner ratio and again in the month difference, so this is
    # clean under both the frozen-weight and the drifting-weight reading.
    rows = []
    for a in AGES:
        r = {"age": AGE_LABEL[a]}
        for v in IMP:
            s = base[base.age == a].groupby(["ym", "dest_attr"])[v].sum().unstack()
            lr = (s["W"] / s["E"]).apply(lambda x: pd.NA if x <= 0 else x)
            rel = (lr / lr.loc[202001])
            for y in [202003, 202009, 202012]:
                r[f"{YM_LABEL[y]}_{v}"] = round(float(rel.loc[y]), 4)
        rows.append(r)
    dump(pd.DataFrame(rows), "we_logratio_rel_jan",
         "(W/E odds relative to Jan; >1 = shifted toward the work panel)")


    # --------------------------------- 18.3 the mechanism question, stated directly
    # Mar -> Sep and Mar -> Dec, per age, per arrival attribute, under all three
    # imputations. Shares only — no level claim, so A1 is not invoked.
    rows = []
    for a in AGES:
        r = {"age": AGE_LABEL[a]}
        for s in ["W", "E"]:
            for v in IMP:
                sh = base.groupby(["ym", "age", "dest_attr"])[v].sum().unstack()
                sh = sh.div(sh.sum(axis=1), axis=0)
                r[f"{s}_SepMar_{v}"] = round(float(sh.loc[(202009, a), s] - sh.loc[(202003, a), s]), 4)
                r[f"{s}_DecMar_{v}"] = round(float(sh.loc[(202012, a), s] - sh.loc[(202003, a), s]), 4)
        rows.append(r)
    mech = pd.DataFrame(rows)
    dump(mech, "mechanism_share_shift", "(percentage-point shift in arrival-attribute share)")


    # ------------------------------------------------- 18.4 sign stability summary
    # Does the direction survive all three imputations, for every age?
    sign = []
    for _, r in mech.iterrows():
        for s in ["W", "E"]:
            for cmp_ in ["SepMar", "DecMar"]:
                vals = [r[f"{s}_{cmp_}_{v}"] for v in IMP]
                sign.append({"age": r["age"], "panel": s, "cmp": cmp_,
                             "lo": vals[0], "mid": vals[1], "hi": vals[2],
                             "same_sign": bool(min(vals) > 0 or max(vals) < 0)})
    sg = pd.DataFrame(sign)
    dump(sg[~sg.same_sign], "sign_flips_under_imputation",
         "(rows where the 0/1.5/3 imputation band crosses zero)")
    out["n_sign_flips"] = int((~sg.same_sign).sum())
    out["n_tested"] = int(len(sg))


    # --------------------------------- 18.5 full 9-type composition, focus vs 20-44
    for grp, name in [(FOCUS, "65plus"), (REF, "20_44")]:
        sub = base[base.age.isin(grp)]
        comp = sub.groupby(["ym", "mtype"])["mid"].sum().unstack()
        comp = comp.div(comp.sum(axis=1), axis=0)
        comp.index = [YM_LABEL[y] for y in comp.index]
        dump(comp.round(4).reset_index().rename(columns={"index": "ym"}),
             f"mtype_composition_{name}", "(mid imputation)")

    with open(f"{ROOT}/eda/results_p18.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p18.json")


if __name__ == "__main__":
    main()
