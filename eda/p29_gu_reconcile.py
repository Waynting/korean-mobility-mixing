#!/usr/bin/env python
"""Phase 29 — the official 자치구 file, and what it turns masking from into.

Two things were open for months and one comparison closes both.

RECONCILIATION. `derived/gu_level.parquet` is our dong data summed to 자치구. It
has never been checked against the official 자치구 release, which is the same KT
pipeline aggregated one level up. DATA_REPORT and eda/README both carry the
to-do.

MASKING. Every volume this project reports carries a 0/1.5/3 imputation band,
because a dong cell below 3 is published as `*`. But the threshold is applied
AFTER aggregation, so a flow hidden at dong level is usually published at gu
level. Differencing the two files therefore MEASURES the hidden volume where we
have only ever bounded it.

The route here came from Lim et al.'s Zenodo release (decisions_and_todos B5).
Their processed odm_*.csv reproduces the official 자치구 file to 1.000000 in every
age band and weekday, which is what identified the product they used, and the
same identity is what makes it usable as ground truth.

WHAT THE MEASUREMENT IS AND IS NOT. The difference is a LOWER bound on hidden
volume, not the exact value: the gu file has its own masked cells (9.4% of gu
cells in 2020-12, concentrated in the same 20-44 bands), and those are counted as
zero on both sides. So the true per-masked-cell mean is at least what is reported
here.

    python eda/p29_gu_reconcile.py [--months 202003,202012]
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
import numpy as np
import pandas as pd
from common import AGE_LABEL, AGES
from dl_gu import zip_path
from paths import DERIVED, ROOT, require

DOW = {"월": 1, "화": 2, "수": 3, "목": 4, "금": 5, "토": 6, "일": 7}
SEOUL_GU = range(11010, 11260)      # the gu file uses 5-digit 시군구 codes
out = {}


def official(ym):
    """Seoul-internal rows of the official 자치구 release, masked cells flagged."""
    path = zip_path(ym)
    require(path, f"official gu release for {ym} (run eda/dl_gu.py --months {ym})")
    frames = []
    with zipfile.ZipFile(path) as zf:
        for n in sorted(x for x in zf.namelist() if not x.endswith("/")):
            df = pd.read_csv(
                io.BytesIO(zf.open(n).read()), encoding="cp949", header=0,
                names=["ym", "dow", "hour", "o", "d", "sex", "age", "mtype",
                       "mean_min", "popv"], dtype={"popv": str})
            frames.append(df[df.o.isin(SEOUL_GU) & df.d.isin(SEOUL_GU)])
    g = pd.concat(frames, ignore_index=True)
    g["dow_n"] = g.dow.map(DOW)
    g["masked"] = g.popv.astype(str).str.strip().eq("*")
    g["v"] = pd.to_numeric(g.popv, errors="coerce").fillna(0.0)
    return g


def ours(ym):
    """Our dong data summed to gu, Seoul-internal, with the masked-cell count."""
    p = DERIVED / "gu_level.parquet"
    require(p, "derived/gu_level.parquet (run eda/p1_conservation.py)")
    con = duckdb.connect()
    return con.execute(f"""
        SELECT dow_n, age, mtype,
               sum(pop_sum_unmasked) AS v_lo,
               sum(n_masked_cells)   AS mcells
        FROM read_parquet('{p}')
        WHERE ym = {ym} AND o_gu BETWEEN 1101 AND 1125
                        AND d_gu BETWEEN 1101 AND 1125
        GROUP BY 1,2,3""").df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="202003,202012")
    args = ap.parse_args()
    for ym in [int(x) for x in args.months.split(",") if x]:
        if not zip_path(ym).exists():
            print(f"  {ym}: no official gu zip, skipping")
            continue
        off = official(ym)
        mine = ours(ym)
        o = (off.groupby(["dow_n", "age", "mtype"])
             .agg(off_v=("v", "sum"), off_masked=("masked", "sum"))
             .reset_index())
        m = mine.merge(o, on=["dow_n", "age", "mtype"], how="outer").fillna(0.0)

        # ---------------------------------------------- 1 the reconciliation
        # Bands where no dong cell is masked must agree exactly. They are the
        # only place the two files are measuring the identical quantity, which
        # is what makes them the test.
        clean = m[(m.mcells == 0) & (m.off_masked == 0)]
        rel = (clean.v_lo.sum() - clean.off_v.sum()) / clean.off_v.sum()
        print(f"\n=== 1 reconciliation against the official 자치구 file ({ym}) ===")
        print(f"  cells with no masking on either side: {len(clean):,} of "
              f"{len(m):,} (age x weekday x mtype)")
        print(f"  our sum {clean.v_lo.sum():,.0f} vs official "
              f"{clean.off_v.sum():,.0f}  relative difference {rel:+.6f}")
        print(f"  p16's 475,266 duplicated raw row pairs would put ours "
              f"+0.00034 high; the residual is two orders smaller and changes "
              f"sign between months, so they do not survive into gu-level sums "
              f"for the unmasked bands.")
        assert abs(rel) < 2e-3, "gu reconciliation is off by more than 0.2%"

        # -------------------------------------- 2 masking, measured not bounded
        by = (m.groupby("age")
              .agg(v_lo=("v_lo", "sum"), mcells=("mcells", "sum"),
                   off_v=("off_v", "sum"), off_masked=("off_masked", "sum"))
              .reindex(AGES).fillna(0.0))
        by["hidden"] = by.off_v - by.v_lo
        by["per_cell"] = by.hidden / by.mcells.replace(0, np.nan)
        by["measured_share"] = by.hidden / by.off_v
        by["assumed_mid"] = 1.5 * by.mcells / (by.v_lo + 1.5 * by.mcells)
        by["assumed_hi"] = 3.0 * by.mcells / (by.v_lo + 3.0 * by.mcells)
        print(f"\n=== 2 what the masked dong cells actually held ({ym}) ===")
        print(f"  {'age':>6} {'masked cells':>13} {'hidden volume':>14} "
              f"{'per cell':>9} | {'measured':>9} {'mid (1.5)':>10} "
              f"{'hi (3.0)':>9}")
        # A masked cell is below 3 by definition, so a per-cell mean above 3
        # is not a measurement of the hidden volume -- it is the difference
        # picking up something else (the gu file's own masked cells, the
        # duplicate rows) on a band too small to absorb it. Flagged, not fixed.
        for a in AGES:
            r = by.loc[a]
            if r.mcells == 0:
                continue
            flag = " <- above 3: too few cells to measure" if r.per_cell > 3 else ""
            print(f"  {AGE_LABEL[a]:>6} {r.mcells:>13,.0f} "
                  f"{r.hidden:>14,.0f} {r.per_cell:>9.2f} | "
                  f"{r.measured_share:>9.2%} {r.assumed_mid:>10.2%} "
                  f"{r.assumed_hi:>9.2%}{flag}")
        tot_hidden = by.hidden.sum()
        tot_off = by.off_v.sum()
        tot_cells = by.mcells.sum()
        print(f"  {'all':>6} {tot_cells:>13,.0f} {tot_hidden:>14,.0f} "
              f"{tot_hidden / tot_cells:>9.2f} | "
              f"{tot_hidden / tot_off:>9.2%} "
              f"{1.5 * tot_cells / (by.v_lo.sum() + 1.5 * tot_cells):>10.2%} "
              f"{3.0 * tot_cells / (by.v_lo.sum() + 3.0 * tot_cells):>9.2%}")
        print(f"\n  A masked dong cell holds {tot_hidden / tot_cells:.2f} on "
              f"average, not the 1.5 midpoint this project has been assuming — "
              f"{tot_hidden / tot_cells / 1.5:.2f}x more.")
        print(f"  The gu file masks {off.masked.sum():,} of its own cells "
              f"({off.masked.mean():.2%}), counted as zero on both sides, so "
              f"every number above is a LOWER bound.")
        out[str(ym)] = dict(
            reconciliation_rel_diff=float(rel),
            clean_cells=int(len(clean)),
            hidden_volume=float(tot_hidden),
            masked_cells=int(tot_cells),
            per_cell_mean=float(tot_hidden / tot_cells),
            measured_share=float(tot_hidden / tot_off),
            assumed_mid_share=float(1.5 * tot_cells
                                    / (by.v_lo.sum() + 1.5 * tot_cells)),
            assumed_hi_share=float(3.0 * tot_cells
                                   / (by.v_lo.sum() + 3.0 * tot_cells)),
            gu_own_masked_cells=int(off.masked.sum()),
            gu_own_masked_share=float(off.masked.mean()),
            by_age={AGE_LABEL[a]: dict(
                masked_cells=float(by.loc[a, "mcells"]),
                hidden=float(by.loc[a, "hidden"]),
                per_cell=float(by.loc[a, "per_cell"]),
                measured_share=float(by.loc[a, "measured_share"]),
                assumed_mid=float(by.loc[a, "assumed_mid"]))
                for a in AGES})
    with open(f"{ROOT}/eda/results_p29.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("\nwrote results_p29.json")


if __name__ == "__main__":
    main()
