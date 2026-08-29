#!/usr/bin/env python
"""Phase 6b — is the shape heterogeneity real, or an artefact?

Three ways it could be fake: (i) it is driven entirely by the two tiny school-age
bands, which carry a schooling confound rather than a policy response; (ii) it is
an artefact of the 1.5 imputation for '*' cells, which matters only for 20-44;
(iii) it is noise in the low-volume bands. Each gets a check.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from common import connect, ROOT, AGE_LABEL


def main():
    con = connect()
    AGES = sorted(AGE_LABEL)
    out = {}

    df = con.execute("""
    SELECT ym, age, sex, arr_hour,
      sum(pop_day)                                                   AS lo,
      sum(pop_day) + 1.5*coalesce(sum(1.0/n_days) FILTER (masked),0)  AS mid,
      sum(pop_day) + 3.0*coalesce(sum(1.0/n_days) FILTER (masked),0)  AS hi
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
    GROUP BY 1,2,3,4""").df()
    piv = df.pivot_table(index=["age", "sex", "arr_hour"], columns="ym",
                         values=["lo", "mid", "hi"])


    def shapes_for(imp):
        """imp='mid' point estimate; 'lo'/'hi' use the same bound in both months so
    the imputation moves the level, not the between-month comparison."""
        r = (piv[(imp, 202009)] / piv[(imp, 202001)]).reset_index(name="ratio")
        d = {}
        for (a, s), g in r.groupby(["age", "sex"]):
            v = g.sort_values("arr_hour")["ratio"].to_numpy()
            d[(a, s)] = v / v.mean()
        return d


    def corr_stats(shapes, ages, label):
        keys = [(a, s) for s in ("F", "M") for a in ages]
        C = np.corrcoef(np.array([shapes[k] for k in keys]))
        iu = np.triu_indices(len(keys), 1)
        v = C[iu]
        print(f"  {label:<34} n={len(v):>4}  min={v.min():+.3f}  p05={np.quantile(v,.05):+.3f}  "
              f"median={np.median(v):+.3f}  >0.95={(v>.95).mean():5.1%}  >0.90={(v>.90).mean():5.1%}")
        return dict(label=label, n=len(v), min=float(v.min()),
                    p05=float(np.quantile(v, .05)), median=float(np.median(v)),
                    frac95=float((v > .95).mean()), frac90=float((v > .90).mean()))


    print("=== 6b.1 sensitivity to the '*' imputation (all 16 bands) ===")
    res = []
    for imp in ("lo", "mid", "hi"):
        res.append(corr_stats(shapes_for(imp), AGES, f"impute '*' at {imp}"))

    print("\n=== 6b.2 sensitivity to which age bands are included (mid imputation) ===")
    sh = shapes_for("mid")
    subsets = {
        "all 16 bands": AGES,
        "drop 0-14 (school confound)": [a for a in AGES if a >= 15],
        "adults 25-64 only": [a for a in AGES if 25 <= a <= 64],
        "adults 25+ only": [a for a in AGES if a >= 25],
        "elderly 65+ only": [a for a in AGES if a >= 65],
        "low-censoring bands only (45+)": [a for a in AGES if a >= 45],
    }
    for lab, ages in subsets.items():
        res.append(corr_stats(sh, ages, lab))
    out["corr_stats"] = res

    print("\n=== 6b.3 daytime hours only (06-22), where volume and precision are highest ===")


    def corr_stats_hours(shapes, ages, h0, h1, label):
        keys = [(a, s) for s in ("F", "M") for a in ages]
        M = np.array([shapes[k][h0:h1 + 1] for k in keys])
        C = np.corrcoef(M)
        iu = np.triu_indices(len(keys), 1)
        v = C[iu]
        print(f"  {label:<34} n={len(v):>4}  min={v.min():+.3f}  median={np.median(v):+.3f}  "
              f">0.95={(v>.95).mean():5.1%}")
        return dict(label=label, min=float(v.min()), median=float(np.median(v)),
                    frac95=float((v > .95).mean()))


    for lab, ages in [("all 16 bands, 06-22h", AGES),
                      ("25+ only, 06-22h", [a for a in AGES if a >= 25]),
                      ("45+ only, 06-22h", [a for a in AGES if a >= 45])]:
        res.append(corr_stats_hours(sh, ages if "all" in lab else ages, 6, 22, lab))

    print("\n=== 6b.4 what actually differs: peak-hour vs midday response by age (F+M) ===")
    q = con.execute("""
    WITH t AS (
      SELECT age, arr_hour,
             sum(pop_day) FILTER (ym=202001) AS jan,
             sum(pop_day) FILTER (ym=202009) AS sep
      FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4) GROUP BY 1,2)
    SELECT age,
      round(sum(sep) FILTER (arr_hour BETWEEN 7 AND 9)/sum(jan) FILTER (arr_hour BETWEEN 7 AND 9),3) AS am_peak_7_9,
      round(sum(sep) FILTER (arr_hour BETWEEN 11 AND 15)/sum(jan) FILTER (arr_hour BETWEEN 11 AND 15),3) AS midday_11_15,
      round(sum(sep) FILTER (arr_hour BETWEEN 17 AND 19)/sum(jan) FILTER (arr_hour BETWEEN 17 AND 19),3) AS pm_peak_17_19,
      round(sum(sep) FILTER (arr_hour BETWEEN 21 AND 23)/sum(jan) FILTER (arr_hour BETWEEN 21 AND 23),3) AS night_21_23,
      round(sum(sep) FILTER (arr_hour <= 4)/sum(jan) FILTER (arr_hour <= 4),3) AS deep_night_0_4
    FROM t GROUP BY age ORDER BY age""").df()
    q.index = [AGE_LABEL[a] for a in q["age"]]
    print(q.drop(columns="age").to_string())
    out["window_response"] = q.reset_index().to_dict("records")

    print("\n=== 6b.5 same, for the elderly at gu granularity: is 65+ stable across gu? ===")
    g = con.execute("""
    WITH t AS (
      SELECT d_gu,
             sum(pop_day) FILTER (ym=202001) AS jan,
             sum(pop_day) FILTER (ym=202009) AS sep
      FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4) AND age >= 65 GROUP BY 1)
    SELECT round(min(sep/jan),3) AS min_gu, round(quantile_cont(sep/jan,0.25),3) AS p25,
           round(median(sep/jan),3) AS median, round(quantile_cont(sep/jan,0.75),3) AS p75,
           round(max(sep/jan),3) AS max_gu, count(*) AS n_gu
    FROM t""").df()
    print(g.to_string(index=False))
    print(con.execute("""
    WITH t AS (
      SELECT d_gu,
             sum(pop_day) FILTER (ym=202001 AND age>=65) AS jan65,
             sum(pop_day) FILTER (ym=202009 AND age>=65) AS sep65,
             sum(pop_day) FILTER (ym=202001 AND age BETWEEN 25 AND 44) AS jan25,
             sum(pop_day) FILTER (ym=202009 AND age BETWEEN 25 AND 44) AS sep25
      FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4) GROUP BY 1)
    SELECT d_gu, round(sep65/jan65,3) AS r_65plus, round(sep25/jan25,3) AS r_25_44,
           round(sep65/jan65 - sep25/jan25,3) AS gap
    FROM t ORDER BY gap""").df().to_string(index=False))

    with open(f"{ROOT}/eda/results_p6b.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p6b.json")


if __name__ == "__main__":
    main()
