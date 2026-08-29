#!/usr/bin/env python
"""Phase 0 — data contract verification, and the unit test Phase 1 depends on.

Everything here answers one question: can the 설명서 be taken at face value, and
what is the actual unit of `이동인구(합)`?
"""
import json
import os
import sys
import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PARQUET, PARQUET_GLOB, ROOT, require

PQ = PARQUET_GLOB


def main():
    require(PARQUET, "parquet tree")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{PQ}', hive_partitioning=false)")

    out = {}
    def show(title, sql, key=None):
        df = con.execute(sql).df()
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
        if key:
            out[key] = df.to_dict("records")
        return df

    # ---------------------------------------------------------------- 0.1 volume
    show("0.1 rows per month", """
    SELECT ym, count(*) AS rows, count(DISTINCT arr_hour) AS n_hours,
           sum(pop) AS pop_unmasked, count(*) FILTER (masked) AS n_masked
    FROM m GROUP BY ym ORDER BY ym""", "rows_per_month")

    # --------------------------------------------------------- 0.2 column domains
    show("0.2a dow domain (rows per weekday)", """
    SELECT ym, dow_n, count(*) AS rows, sum(pop) AS pop
    FROM m GROUP BY ym, dow_n ORDER BY ym, dow_n""", "dow_domain")
    show("0.2b sex domain", "SELECT ym, sex, count(*) AS rows FROM m GROUP BY 1,2 ORDER BY 1,2",
         "sex_domain")
    show("0.2c age domain", """
    SELECT age, count(*) AS rows, count(DISTINCT ym) AS n_ym
    FROM m GROUP BY age ORDER BY age""", "age_domain")
    show("0.2d mtype domain (all 9 H/W/E combos?)", """
    SELECT mtype, count(*) FILTER (ym=202001) AS rows_jan,
           count(*) FILTER (ym=202009) AS rows_sep,
           sum(pop) FILTER (ym=202001) AS pop_jan,
           sum(pop) FILTER (ym=202009) AS pop_sep
    FROM m GROUP BY mtype ORDER BY mtype""", "mtype_domain")

    # ------------------------------------------------------------- 0.3 code space
    show("0.3a dong code digit-length", """
    SELECT len(o_dong::VARCHAR) AS digits,
           count(DISTINCT o_dong) AS n_origin_codes,
           count(*) AS rows
    FROM m GROUP BY 1 ORDER BY 1""", "code_lengths")

    show("0.3b non-7-digit codes (external areas)", """
    SELECT code, sum(as_origin) AS rows_as_origin, sum(as_dest) AS rows_as_dest FROM (
      SELECT o_dong AS code, 1 AS as_origin, 0 AS as_dest FROM m
      WHERE len(o_dong::VARCHAR) <> 7
      UNION ALL
      SELECT d_dong, 0, 1 FROM m WHERE len(d_dong::VARCHAR) <> 7)
    GROUP BY code ORDER BY code""", "external_codes")

    show("0.3c Seoul gu prefixes (expect 25)", """
    SELECT substr(o_dong::VARCHAR,1,4) AS gu, count(DISTINCT o_dong) AS n_dong
    FROM m WHERE len(o_dong::VARCHAR)=7
    GROUP BY 1 ORDER BY 1""", "gu_prefixes")

    show("0.3d code universe: origin vs destination vs month", """
    WITH o AS (SELECT DISTINCT ym, o_dong AS c FROM m),
         d AS (SELECT DISTINCT ym, d_dong AS c FROM m)
    SELECT ym,
           (SELECT count(*) FROM o WHERE o.ym=x.ym) AS n_origin,
           (SELECT count(*) FROM d WHERE d.ym=x.ym) AS n_dest,
           (SELECT count(*) FROM o WHERE o.ym=x.ym AND c NOT IN (SELECT c FROM d WHERE d.ym=x.ym)) AS origin_only,
           (SELECT count(*) FROM d WHERE d.ym=x.ym AND c NOT IN (SELECT c FROM o WHERE o.ym=x.ym)) AS dest_only
    FROM (SELECT DISTINCT ym FROM m) x ORDER BY ym""", "code_universe")

    show("0.3e codes present in one month only (boundary change detector)", """
    WITH a AS (SELECT DISTINCT ym, c FROM (
                 SELECT ym, o_dong AS c FROM m UNION ALL SELECT ym, d_dong FROM m))
    SELECT c AS code,
           bool_or(ym=202001) AS in_jan, bool_or(ym=202009) AS in_sep
    FROM a GROUP BY c HAVING NOT (bool_or(ym=202001) AND bool_or(ym=202009))
    ORDER BY c""", "code_month_diff")

    # ------------------------------------------------------ 0.4 numeric integrity
    show("0.4a mean_min range", """
    SELECT ym, min(mean_min) AS min, max(mean_min) AS max,
           count(*) FILTER (mean_min IS NULL) AS n_null,
           count(*) FILTER (mean_min = 0) AS n_zero,
           quantile_cont(mean_min, 0.5) AS median
    FROM m GROUP BY ym ORDER BY ym""", "mean_min_range")

    show("0.4b pop range / masking threshold", """
    SELECT ym, min(pop) AS min_unmasked, max(pop) AS max,
           quantile_cont(pop, 0.5) AS median,
           count(*) FILTER (pop < 3) AS n_below_3
    FROM m WHERE NOT masked GROUP BY ym ORDER BY ym""", "pop_range")

    # ------------------------------- 1.0 THE UNIT TEST: is 이동인구(합) sum or mean?
    # Jan 2020 has 5 Wed/Thu/Fri and 4 Mon/Tue/Sat/Sun.
    # Sep 2020 has 5 Tue/Wed and 4 Mon/Thu/Fri/Sat/Sun.
    # If the column is a monthly SUM, weekdays with 5 occurrences carry ~1.25x the
    # volume of otherwise-comparable weekdays with 4. If it is a per-day mean, they
    # carry ~1.0x.
    show("1.0 sum-vs-mean test (weekday occurrences in month)", """
    SELECT ym, dow_n,
           CASE WHEN (ym=202001 AND dow_n IN (3,4,5)) OR (ym=202009 AND dow_n IN (2,3))
                THEN 5 ELSE 4 END AS n_occurrences,
           sum(pop) AS pop_unmasked, count(*) AS cells
    FROM m WHERE dow_n BETWEEN 1 AND 5
    GROUP BY ym, dow_n ORDER BY ym, dow_n""", "unit_test")

    with open(f"{ROOT}/eda/results_p0.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p0.json")


if __name__ == "__main__":
    main()
