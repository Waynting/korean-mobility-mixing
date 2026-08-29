#!/usr/bin/env python
"""Phase 2 — structure of the '*' mask.

The headline is that the mask is NOT a small-cell rule in the demographic sense.
It is a value threshold (<3) applied to an expansion-weighted count whose weight
per observed device varies by age, so the age bands with a coarse weight (0-19,
65+) can never fall under 3 and are never masked, while 20-44 loses half its
cells. Section 2.5 estimates that weight.

Section 2.7 is the number that goes in the paper: the share of *volume* lost, not
the share of cells, deduplicated and scoped. They differ by a factor of three and
only the pair of them is honest.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import connect, ROOT, AGE_LABEL, PARQUET_GLOB, YMS

FIG = f"{ROOT}/eda/fig"
os.makedirs(FIG, exist_ok=True)


def main():
    con = connect()
    out = {}

    AGES = sorted(AGE_LABEL)


    def heat(df, val, title, fname, fmt="{:.0%}", cmap="magma"):
        p = df.pivot(index="age", columns="arr_hour", values=val).reindex(AGES)
        fig, ax = plt.subplots(figsize=(11, 5.2))
        im = ax.imshow(p.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(24)); ax.set_xticklabels(range(24), fontsize=8)
        ax.set_yticks(range(len(AGES)))
        ax.set_yticklabels([AGE_LABEL[a] for a in AGES], fontsize=8)
        ax.set_xlabel("arrival hour"); ax.set_ylabel("age band")
        ax.set_title(title, fontsize=11)
        for i in range(len(AGES)):
            for j in range(24):
                v = p.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.3,
                            color="white" if v < 0.6 else "black")
        fig.colorbar(im, ax=ax, label=val)
        fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=150); plt.close(fig)
        print(f"  -> fig/{fname}")


    # ------------------------------------------------- 2.1 cell mask rate age x hour
    for ym in (202001, 202009):
        df = con.execute(f"""
        SELECT age, arr_hour,
               count(*) FILTER (masked)/count(*)::DOUBLE AS mask_rate
        FROM m WHERE ym={ym} AND o_seoul AND d_seoul
        GROUP BY 1,2""").df()
        heat(df, "mask_rate", f"cell mask rate, age x arrival hour ({ym}, Seoul-internal)",
             f"p2_maskrate_age_hour_{ym}.png")
        out[f"maskrate_age_hour_{ym}"] = df.to_dict("records")

    # ---------------------------------- 2.2 censored VOLUME share (upper bound) heat
    for ym in (202001, 202009):
        df = con.execute(f"""
        SELECT age, arr_hour,
               3*count(*) FILTER (masked)
                 /(sum(pop) + 3*count(*) FILTER (masked)) AS max_vol_share
        FROM m WHERE ym={ym} AND o_seoul AND d_seoul
        GROUP BY 1,2""").df()
        heat(df, "max_vol_share",
             f"upper bound on censored share of volume, age x hour ({ym})",
             f"p2_volshare_age_hour_{ym}.png", cmap="viridis")
        out[f"volshare_age_hour_{ym}"] = df.to_dict("records")

    # ------------------------------------------------------- 2.3 age x gu mask rate
    df = con.execute("""
    SELECT age, o_gu,
           count(*) FILTER (masked AND ym=202001)/count(*) FILTER (ym=202001)::DOUBLE AS jan,
           count(*) FILTER (masked AND ym=202009)/count(*) FILTER (ym=202009)::DOUBLE AS sep
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1,2""").df()
    out["maskrate_age_gu"] = df.to_dict("records")
    p = df.pivot(index="age", columns="o_gu", values="jan").reindex(AGES)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(p.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(AGES))); ax.set_yticklabels([AGE_LABEL[a] for a in AGES], fontsize=8)
    ax.set_title("cell mask rate, age x origin gu (202001)", fontsize=11)
    ax.set_xlabel("gu code"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(f"{FIG}/p2_maskrate_age_gu.png", dpi=150); plt.close(fig)
    print("  -> fig/p2_maskrate_age_gu.png")
    print("\n=== 2.3 spread of gu-level mask rate within each age (Jan) ===")
    print(df.groupby("age")["jan"].agg(["min", "median", "max"]).round(3).to_string())

    # --------------------------------- 2.4 the cell ../reference/EDA.md worried about: 65+ x night
    print("\n=== 2.4 elderly x late night: is the design-critical cell censored? ===")
    q = con.execute("""
    SELECT ym, age, arr_hour,
           count(*) AS cells, count(*) FILTER (masked) AS masked_cells,
           round(sum(pop_day),0) AS pop_day
    FROM m WHERE age >= 65 AND (arr_hour >= 22 OR arr_hour <= 4)
      AND o_seoul AND d_seoul
    GROUP BY 1,2,3 ORDER BY 1,2,3""").df()
    print(q.groupby(["ym", "age"]).agg(cells=("cells", "sum"),
                                       masked=("masked_cells", "sum"),
                                       pop_day=("pop_day", "sum")).to_string())
    out["elderly_night"] = q.to_dict("records")

    # --------------------------------- 2.5 implied per-device expansion weight by age
    print("\n=== 2.5 implied expansion weight: smallest published values by age x sex (Jan) ===")
    rows = []
    for a in AGES:
        for s in ("F", "M"):
            v = con.execute(f"""
            SELECT DISTINCT pop FROM m
            WHERE ym=202001 AND age={a} AND sex='{s}' AND o_seoul
            ORDER BY pop LIMIT 6""").df()["pop"].tolist()
            mn = v[0]
            rows.append(dict(age=a, sex=s, min_pop=round(mn, 3),
                             implied_weight=round(mn / max(1, int(np.ceil(3 / mn * 0.999))), 3)
                             if mn < 3 else round(mn / max(1, round(mn / (mn if mn >= 3 else 1))), 3),
                             smallest_values=[round(x, 3) for x in v]))
    for r in rows:
        print(f"  age {r['age']:>2} {r['sex']}  min={r['min_pop']:>6}  "
              f"first values={r['smallest_values']}")
    out["weight_probe"] = rows

    # -------------------------------------------------- 2.6 masking by hour, overall
    print("\n=== 2.6 mask rate and censored-volume bound by hour (all ages, Seoul-internal) ===")
    q = con.execute("""
    SELECT arr_hour,
           round(count(*) FILTER (masked AND ym=202001)/count(*) FILTER (ym=202001)::DOUBLE,3) AS cellrate_jan,
           round(count(*) FILTER (masked AND ym=202009)/count(*) FILTER (ym=202009)::DOUBLE,3) AS cellrate_sep,
           round(3*count(*) FILTER (masked AND ym=202001)
                 /(sum(pop) FILTER (ym=202001)+3*count(*) FILTER (masked AND ym=202001)),3) AS volbound_jan,
           round(3*count(*) FILTER (masked AND ym=202009)
                 /(sum(pop) FILTER (ym=202009)+3*count(*) FILTER (masked AND ym=202009)),3) AS volbound_sep
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1 ORDER BY 1""").df()
    print(q.to_string(index=False))
    out["mask_by_hour"] = q.to_dict("records")

    # ------------------- 2.7 cell rate vs VOLUME rate by age, deduplicated (Phase 0b)
    # The 27.68% that has been quoted everywhere is a *cell* rate over the whole file:
    # no Seoul filter, and no dedup. It is the number that makes the mask sound fatal.
    # The volume rate is the number that says what is actually lost, and the two are
    # far apart, so neither may ever be reported without the other.
    #
    # Two corrections the earlier figure needs before it can be quoted:
    #   dedup      — the file carries exactly-duplicated rows, which inflate the cell
    #                count. Collapsing on the natural key removes them.
    #   scope      — 'whole file' includes trips with neither end in Seoul. Both the
    #                origin-in-Seoul scope (what the Phase 1c numerator uses) and the
    #                both-ends-in-Seoul scope (what every other section here uses) are
    #                reported, because they give different answers.
    #
    # A masked cell is 0 < value < 3 (manual p.17: 3명 *미만*), so its contribution is
    # bounded but not known: 0 is the floor, 3 the ceiling, 1.5 the midpoint. Three
    # bounds, not one estimate.
    print("\n=== 2.7 cell rate vs volume rate by age, deduplicated ===")
    SEOUL = "BETWEEN 1101000 AND 1125999"
    SCOPES = {"all": "TRUE", "o_seoul": "os", "seoul_internal": "os AND ds"}
    per_month = []
    for ym in YMS:
        sel = ",\n           ".join(
            f"count(*) FILTER ({w}) AS cells_{k}, "
            f"count(*) FILTER (({w}) AND masked) AS masked_{k}, "
            f"coalesce(sum(pop) FILTER ({w}), 0) AS vol_{k}"
            for k, w in SCOPES.items())
        df = con.execute(f"""
        WITH g AS (
          SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                 count(*) AS n_raw, min(pop) AS pop, max(pop) AS pop_hi,
                 bool_or(masked) AS masked
          FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
          WHERE ym = {ym}
          GROUP BY ALL),
        f AS (SELECT *, o_dong {SEOUL} AS os, d_dong {SEOUL} AS ds FROM g)
        SELECT {ym} AS ym, age,
               sum(n_raw) AS rows_raw,
               count(*) FILTER (n_raw > 1) AS dup_keys,
               count(*) FILTER (pop IS DISTINCT FROM pop_hi) AS conflict_keys,
               {sel}
        FROM f GROUP BY age ORDER BY age""").df()
        per_month.append(df)
        print(f"  {ym} done", flush=True)
    dedup = pd.concat(per_month, ignore_index=True)
    out["cell_vs_volume_by_month_age"] = dedup.to_dict("records")

    tot = dedup.groupby("age", as_index=False).sum(numeric_only=True).drop(columns="ym")


    def shares(d):
        """Paired cell rate and the three volume bounds, for one scope."""
        r = {}
        for k in SCOPES:
            cells, masked, vol = d[f"cells_{k}"], d[f"masked_{k}"], d[f"vol_{k}"]
            r[f"{k}_cell_rate"] = masked / cells
            r[f"{k}_vol_lower"] = 0.0 * masked          # a masked cell may be ~0
            r[f"{k}_vol_mid"] = 1.5 * masked / (vol + 1.5 * masked)
            r[f"{k}_vol_upper"] = 3.0 * masked / (vol + 3.0 * masked)
        return r


    for k, v in shares(tot).items():
        tot[k] = v
    out["cell_vs_volume_by_age"] = tot.to_dict("records")

    grand = tot.sum(numeric_only=True).drop(labels=[c for c in tot.columns
                                                    if c.endswith(("_rate", "_lower",
                                                                   "_mid", "_upper"))]
                                            + ["age"])
    g = {**grand.to_dict(), **{k: float(v) for k, v in shares(grand).items()}}
    out["cell_vs_volume_total"] = g
    out["dedup"] = {"rows_raw": int(grand["rows_raw"]),
                    "cells_after_dedup": int(grand["cells_all"]),
                    "duplicate_rows": int(grand["rows_raw"] - grand["cells_all"]),
                    "duplicate_keys": int(grand["dup_keys"]),
                    "keys_with_conflicting_pop": int(grand["conflict_keys"])}

    print(f"  rows in file        {int(grand['rows_raw']):>14,}")
    print(f"  distinct cells      {int(grand['cells_all']):>14,}"
          f"  ({int(grand['rows_raw'] - grand['cells_all']):,} duplicate rows on "
          f"{int(grand['dup_keys']):,} keys, {int(grand['conflict_keys'])} with "
          f"conflicting values)")
    for k in SCOPES:
        print(f"  {k:<15} cell {g[k + '_cell_rate']:6.2%}   "
              f"volume 0% / {g[k + '_vol_mid']:.2%} / {g[k + '_vol_upper']:.2%}")
    print("\n  by age band (both ends in Seoul):")
    print(f"  {'band':<7} {'cells':>12} {'cell rate':>10} {'vol mid':>9} {'vol upper':>10}")
    for _, r in tot.iterrows():
        print(f"  {AGE_LABEL[int(r['age'])]:<7} {int(r['cells_seoul_internal']):>12,} "
              f"{r['seoul_internal_cell_rate']:>9.1%} "
              f"{r['seoul_internal_vol_mid']:>8.2%} "
              f"{r['seoul_internal_vol_upper']:>9.2%}")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(tot))
    ax.bar(x - 0.2, tot["seoul_internal_cell_rate"], 0.4, label="cells masked",
           color="#b03a48")
    ax.bar(x + 0.2, tot["seoul_internal_vol_mid"], 0.4, label="volume lost (midpoint)",
           color="#3a6ea5")
    ax.errorbar(x + 0.2, tot["seoul_internal_vol_mid"],
                yerr=[tot["seoul_internal_vol_mid"],
                      tot["seoul_internal_vol_upper"] - tot["seoul_internal_vol_mid"]],
                fmt="none", ecolor="#1d3557", capsize=2, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([AGE_LABEL[a] for a in tot["age"]],
                                         rotation=45, fontsize=8)
    ax.set_ylabel("share"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("masking: share of cells vs share of volume, by age "
                 "(2020, both ends in Seoul; bars 0-3 bounds)", fontsize=10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout(); fig.savefig(f"{FIG}/p2_cell_vs_volume.png", dpi=150)
    plt.close(fig)
    print("  -> fig/p2_cell_vs_volume.png")

    with open(f"{ROOT}/eda/results_p2.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p2.json")


if __name__ == "__main__":
    main()
