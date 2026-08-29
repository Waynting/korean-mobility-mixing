#!/usr/bin/env python
"""Phase 1 — conservation identities and the censored-mass envelope.

(a) daily closure: over a full 24h a dong's inflow should ~= its outflow.
(b) gu aggregation: emitted for the free cross-check against the official
    자치구 file (not on disk yet).
(c) censored mass: every '*' cell is in [0,3), so the true total is bracketed.
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from common import DERIVED, connect, ROOT, GU_NAMES, YMS


def main():
    con = connect()
    out = {}


    def show(title, sql, key=None, n=40):
        df = con.execute(sql).df()
        print(f"\n=== {title} ===")
        print(df.head(n).to_string(index=False))
        if key:
            out[key] = df.to_dict("records")
        return df


    # --------------------------------------------------- 1a daily closure per dong
    show("1a Seoul dong daily closure  inflow vs outflow (per calendar day)", """
    WITH o AS (SELECT ym, o_dong AS dong, sum(pop_day) AS outflow
               FROM m WHERE o_seoul GROUP BY 1,2),
         i AS (SELECT ym, d_dong AS dong, sum(pop_day) AS inflow
               FROM m WHERE d_seoul GROUP BY 1,2),
         j AS (SELECT ym, dong, outflow, inflow,
                      (inflow-outflow)/((inflow+outflow)/2) AS rel_imb
               FROM o JOIN i USING (ym, dong))
    SELECT ym, count(*) AS n_dong,
           sum(inflow) AS tot_in, sum(outflow) AS tot_out,
           (sum(inflow)-sum(outflow))/sum(outflow) AS global_rel_gap,
           min(rel_imb) AS min_rel, quantile_cont(rel_imb,0.05) AS p05,
           quantile_cont(rel_imb,0.5) AS p50, quantile_cont(rel_imb,0.95) AS p95,
           max(rel_imb) AS max_rel,
           count(*) FILTER (abs(rel_imb) > 0.10) AS n_over_10pct
    FROM j GROUP BY ym ORDER BY ym""", "closure_summary")

    show("1a' worst-imbalanced Seoul dongs (Jan)", """
    WITH o AS (SELECT o_dong AS dong, sum(pop_day) AS outflow FROM m
               WHERE o_seoul AND ym=202001 GROUP BY 1),
         i AS (SELECT d_dong AS dong, sum(pop_day) AS inflow FROM m
               WHERE d_seoul AND ym=202001 GROUP BY 1)
    SELECT dong, round(inflow,0) AS inflow, round(outflow,0) AS outflow,
           round((inflow-outflow)/((inflow+outflow)/2),4) AS rel_imb
    FROM o JOIN i USING (dong)
    ORDER BY abs((inflow-outflow)/((inflow+outflow)/2)) DESC LIMIT 12""",
         "closure_worst")

    # ----------------------------------------------- 1b Seoul in/out vs rest of ROK
    show("1b flows crossing the Seoul boundary (per day)", """
    SELECT ym,
           sum(pop_day) FILTER (o_seoul AND d_seoul)         AS internal,
           sum(pop_day) FILTER (o_seoul AND NOT d_seoul)     AS out_of_seoul,
           sum(pop_day) FILTER (NOT o_seoul AND d_seoul)     AS into_seoul,
           sum(pop_day) FILTER (NOT o_seoul AND NOT d_seoul) AS external_only,
           sum(pop_day) AS total
    FROM m GROUP BY ym ORDER BY ym""", "boundary_flows")

    # --------------------------------------------------- 1c censored-mass envelope
    show("1c censored mass envelope, whole file", """
    SELECT ym,
           sum(pop) AS lower_bound, count(*) FILTER (masked) AS n_masked_cells,
           sum(pop) + 3*count(*) FILTER (masked) AS upper_bound,
           3*count(*) FILTER (masked) / (sum(pop) + 3*count(*) FILTER (masked))
             AS max_censored_share,
           1.5*count(*) FILTER (masked) / (sum(pop) + 1.5*count(*) FILTER (masked))
             AS mid_censored_share
    FROM m GROUP BY ym ORDER BY ym""", "censored_total")

    show("1c' censored share by age band (Jan/Sep, whole day)", """
    SELECT age,
           round(3*count(*) FILTER (masked AND ym=202001)
                 / (sum(pop) FILTER (ym=202001)
                    + 3*count(*) FILTER (masked AND ym=202001)), 4) AS max_share_jan,
           round(3*count(*) FILTER (masked AND ym=202009)
                 / (sum(pop) FILTER (ym=202009)
                    + 3*count(*) FILTER (masked AND ym=202009)), 4) AS max_share_sep
    FROM m GROUP BY age ORDER BY age""", "censored_by_age")

    show("1c'' censored share by movement type", """
    SELECT mtype,
           round(3*count(*) FILTER (masked AND ym=202001)
                 / (sum(pop) FILTER (ym=202001)
                    + 3*count(*) FILTER (masked AND ym=202001)), 4) AS max_share_jan,
           round(3*count(*) FILTER (masked AND ym=202009)
                 / (sum(pop) FILTER (ym=202009)
                    + 3*count(*) FILTER (masked AND ym=202009)), 4) AS max_share_sep
    FROM m GROUP BY mtype ORDER BY mtype""", "censored_by_mtype")

    # ----------------------------------------- 1b' gu-level file for external check
    # Written month by month. At 11 months the single GROUP BY ALL still fit; at 12
    # it spills more temp than the disk has free, and DuckDB aborts the whole COPY.
    # `ym` is a grouping key, so per-month parts concatenate to the identical table.
    PARTS = f"{DERIVED}/gu_parts"
    os.makedirs(PARTS, exist_ok=True)
    for ym in YMS:
        con.execute(f"""
        COPY (SELECT ym, dow_n, arr_hour, o_gu, d_gu, sex, age, mtype,
                     sum(pop) AS pop_sum_unmasked,
                     sum(pop_day) AS pop_day_unmasked,
                     count(*) FILTER (masked) AS n_masked_cells,
                     sum(pop*mean_min)/nullif(sum(pop),0) AS mean_min_wtd
              FROM m WHERE ym = {ym} GROUP BY ALL)
        TO '{PARTS}/{ym}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
        print(f"  gu_level part {ym}", flush=True)
    con.execute(f"""
    COPY (SELECT * FROM read_parquet('{PARTS}/*.parquet'))
    TO '{DERIVED}/gu_level.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    for ym in YMS:
        os.remove(f"{PARTS}/{ym}.parquet")
    os.rmdir(PARTS)
    print("\nwrote derived/gu_level.parquet")

    with open(f"{ROOT}/eda/results_p1.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p1.json")


if __name__ == "__main__":
    main()
