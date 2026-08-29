#!/usr/bin/env python
"""Phase 43 — the three Data Records blockers for the Scientific Data descriptor.

Three items, one script, three subcommands. They are unrelated to each other
except that all three block the same section, so they share the disk-safety
discipline and nothing else.

    python eda/p43_datarecords.py manifest      # item 1 -- the 2020 sha256 gap
    python eda/p43_datarecords.py dedup         # item 2 -- what DISTINCT moves
    python eda/p43_datarecords.py core          # item 3 -- build matrix_core

ITEM 1, AND WHY IT IS HALF-IMPOSSIBLE. `raw/mobility/manifest.json` carries a
`zip_sha256` for 67 of 79 months. The twelve without are 202001-202012, fetched
by hand before dl_mobility.py existed. Those zips are NOT on disk -- the drive
holds exactly two 행정동 zips (202606, 202607) because dl_mobility_fill.sh runs
with --delete-zip and the sha is taken before the delete. So `zip_sha256` for
2020 cannot be computed retroactively and re-downloading is blocked (the file
endpoint breaks TLS at 100-300 MB with no Range support: a break restarts at
zero). That half is dead.

What is NOT dead: the extracted CSVs are all still there, for all 79 months. So
the integrity claim can be made one level down, on the bytes the ETL actually
read, and it can be made UNIFORMLY across all 79 months -- which the zip-level
claim never could. See the memo for why this is arguably the stronger primitive
and precisely which claim is lost.

This subcommand writes a NEW sidecar, `raw/mobility/manifest_csv.json`. It does
not touch manifest.json.

ITEM 2, AND THE ORDER IT HAS TO BE DONE IN. p16 found 475,266 duplicate key
groups, each exactly two byte-identical rows, present in the raw CSVs. p26's
cells() already collapses them (min(pop) over GROUP BY ALL); p1_conservation's
gu_level.parquet does not, and p19/p29/p30 read gu_level. So the question is not
"should we dedupe" but "which published numbers move", and that is answered on a
sample BEFORE anything is rebuilt, because p31 gates those numbers and p36
recomputes them.

The two months measured are 202003 and 202012 -- not a convenience sample, they
are the only two months with an official 자치구 file on disk, which is what
p29's measurement differences against, so they are the only two months where the
question is even answerable.

ITEM 3. matrix_core.parquet, built by concatenating what p26's cells() already
computes per month. Nothing is re-implemented: cells() is imported and called,
so the core table is deduplicated by construction and byte-identical to what the
published matrices were built from. The consequence is that p26's A is
reproducible from the released table exactly, which is checked here on two
months.

DISK. REVISION_PLAN records that a single GROUP BY ALL over the whole tree spilled
past the remaining 14 GB and aborted the COPY (p1_conservation.py:99-101). The
dataset is 6.5x larger now. Everything here is per-month; free space on the
target volume is checked before every shard against MIN_FREE_GB and the run
aborts cleanly rather than filling the disk.
"""
import argparse
import hashlib
import json
import unicodedata
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
import numpy as np
import pandas as pd

from common import AGE_LABEL, AGES, months_on_disk
from p26_matrix import cells, matrices, stats
from paths import DATA_ROOT, DERIVED, PARQUET, ROOT, require

# Abort margin, stated in advance. The 2020 abort happened with 14 GB free, so
# 50 GB is a 3.5x margin over the failure that motivated the rule. Checked on
# the volume that is actually being written to, before every shard.
MIN_FREE_GB = 50
SEOUL_LO, SEOUL_HI = 1101000, 1125999
out = {}


def free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def check_space(path, what):
    g = free_gb(path)
    if g < MIN_FREE_GB:
        raise SystemExit(
            f"ABORT before {what}: only {g:.1f} GB free on {path}, "
            f"margin is {MIN_FREE_GB} GB. Nothing was written.")
    return g


def con_for(tmpdir, mem="6GB", threads=4):
    """DuckDB with spill and memory bounded explicitly, both set before use.

    The 2020 abort was an unbounded spill to a temp directory DuckDB chose for
    itself. Both are pinned here, and temp goes to the internal disk rather than
    to the volume being written.
    """
    os.makedirs(tmpdir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA disable_progress_bar")
    con.execute(f"PRAGMA memory_limit='{mem}'")
    con.execute(f"PRAGMA temp_directory='{tmpdir}'")
    return con


# ===================================================================== item 1
def sha256_file(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cmd_manifest(args):
    """Per-CSV sha256 for the months the zip-level manifest cannot cover.

    Default scope is the twelve 2020 months -- the gap. `--all` extends it to
    every month on disk, which is the version that makes the claim uniform;
    the memo reports what that costs.
    """
    man_path = DATA_ROOT / "raw" / "mobility" / "manifest.json"
    zipman = json.loads(man_path.read_text()) if man_path.exists() else {}
    disk = months_on_disk()
    yms = disk if args.all else [y for y in disk if 202001 <= y <= 202012]

    sidecar = DATA_ROOT / "raw" / "mobility" / "manifest_csv.json"
    prev = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    # Resumable: a 57 GB read over USB is long enough that an interrupted run
    # must not start from zero.
    rec = dict(prev.get("months", {}))

    etl = {}
    p_etl = f"{ROOT}/eda/etl_report.json"
    if os.path.exists(p_etl):
        for r in json.load(open(p_etl)):
            etl[(str(r["ym"]), str(r["hour"]))] = r

    t0 = time.time()
    total_bytes = 0
    for ym in yms:
        key = str(ym)
        if key in rec and not args.force:
            print(f"  {ym}: already hashed, skipping", flush=True)
            continue
        d = DATA_ROOT / f"생활이동_행정동_{ym}"
        if not d.exists():
            print(f"  {ym}: NO CSV DIRECTORY -- cannot hash", flush=True)
            rec[key] = dict(ym=ym, error="csv_dir_missing", csv_dir=str(d))
            continue
        files = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() == ".csv")
        per = {}
        tb = 0
        ts = time.time()
        for p in files:
            # The hour is the last underscore-separated field: ..._00시.csv.
            # NFC-normalise FIRST: macOS stores these names decomposed, so a
            # literal "시" typed in this file (composed) does not match the
            # filename's two-codepoint form and the strip silently no-ops.
            # eda/README.md lists NFD directory names as a known ETL trap; it
            # bites the same way here. Found by the etl_rows join coming back 0.
            stem = unicodedata.normalize("NFC", p.stem)
            hh = stem.rsplit("_", 1)[-1].replace("시", "").zfill(2)
            n = p.stat().st_size
            per[hh] = dict(file=p.name, bytes=n, sha256=sha256_file(p))
            tb += n
        dt = time.time() - ts
        total_bytes += tb
        z = zipman.get(key, {})
        rec[key] = dict(
            ym=ym, csv_dir=str(d), files=len(files), csv_bytes=tb,
            per_hour=per,
            # If the zip-level manifest covers this month, the two must agree on
            # the expanded size. A mismatch means the CSVs on disk are not the
            # ones the recorded zip expanded to, which would invalidate the
            # whole substitute -- so it is recorded, per month, not assumed.
            zip_sha256=z.get("zip_sha256"),
            zip_expanded_bytes=z.get("zip_expanded_bytes"),
            bytes_match_zip_manifest=(
                None if not z.get("zip_expanded_bytes")
                else tb == z["zip_expanded_bytes"]),
            hashed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        # ETL row counts are the other half of the substitute claim: the parquet
        # is verified against the CSV row by row at conversion time, so a CSV
        # hash plus an ETL row count chains the released parquet back to bytes.
        rows = sum(etl[(key, h)]["rows"] for h in per if (key, h) in etl)
        if rows:
            rec[key]["etl_rows"] = rows
        mark = ("" if rec[key]["bytes_match_zip_manifest"] is None
                else ("  bytes==zip manifest"
                      if rec[key]["bytes_match_zip_manifest"]
                      else "  *** BYTES DISAGREE WITH ZIP MANIFEST ***"))
        print(f"  {ym}: {len(files)} files, {tb / 2**30:6.2f} GB, "
              f"{dt:6.1f}s ({tb / dt / 1e6:5.1f} MB/s){mark}", flush=True)
        sidecar.write_text(json.dumps(
            dict(note=("per-CSV sha256 of the extracted 생활이동_행정동 files. "
                       "This is NOT a zip checksum: the 2020 zips no longer "
                       "exist (dl_mobility_fill.sh runs --delete-zip) and "
                       "cannot be recovered. See eda/memo/phase43-datarecords.md."),
                 min_free_gb_margin=MIN_FREE_GB,
                 months=rec), ensure_ascii=False, indent=1))
    dt = time.time() - t0
    print(f"\n  hashed {total_bytes / 2**30:.1f} GB in {dt / 60:.1f} min"
          f"{'' if not total_bytes else f' ({total_bytes / dt / 1e6:.1f} MB/s)'}")
    print(f"  wrote {sidecar}")
    n_ok = sum(1 for v in rec.values() if "per_hour" in v)
    print(f"  months with per-CSV sha256: {n_ok}")
    return 0


# ===================================================================== item 2
def gu_shape(con, ym, dedup):
    """(dow_n, age, mtype) -> unmasked volume, masked-cell count, Seoul-internal.

    This is the shape p29 pulls out of gu_level.parquet, read straight from the
    month's parquet partition instead. `dedup=False` is what gu_level.parquet
    actually contains (raw rows, duplicates included); `dedup=True` collapses
    whole-row duplicates the way p26's cells() does.

    The month's own directory is globbed rather than the whole tree: the answer
    is identical and it does not open 1,896 files to read one month's.
    """
    glob = f"{PARQUET}/ym={ym}/*.parquet"
    src = (f"""
      SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
             min(pop) AS pop, bool_or(masked) AS masked
      FROM read_parquet('{glob}', hive_partitioning=false)
      WHERE o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
      GROUP BY ALL""" if dedup else f"""
      SELECT ym, dow_n, age, mtype, pop, masked
      FROM read_parquet('{glob}', hive_partitioning=false)
      WHERE o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}""")
    rows = con.execute(f"""
        WITH src AS ({src})
        SELECT dow_n, age, mtype,
               coalesce(sum(pop), 0)          AS v_lo,
               count(*) FILTER (WHERE masked) AS mcells
        FROM src GROUP BY 1,2,3""").fetchall()
    return {(int(d), int(a), m): (float(v), int(c)) for d, a, m, v, c in rows}


def dup_census(con, ym):
    """How many duplicate rows there are, and how much volume they carry."""
    glob = f"{PARQUET}/ym={ym}/*.parquet"
    r = con.execute(f"""
        WITH k AS (
          SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                 count(*) AS n, min(pop) AS pop, bool_or(masked) AS masked
          FROM read_parquet('{glob}', hive_partitioning=false)
          GROUP BY 1,2,3,4,5,6,7,8)
        SELECT sum(n)                                   AS n_rows,
               count(*)                                 AS n_keys,
               count(*) FILTER (WHERE n > 1)            AS n_dup_groups,
               max(n)                                   AS max_group,
               sum(n - 1)                               AS n_extra_rows,
               sum((n - 1) * coalesce(pop, 0))          AS extra_pop,
               sum(coalesce(pop, 0) * n)                AS pop_raw,
               sum(coalesce(pop, 0))                    AS pop_dedup,
               sum(CASE WHEN masked THEN n ELSE 0 END)  AS mrows_raw,
               count(*) FILTER (WHERE masked)           AS mcells_dedup
        FROM k""").fetchone()
    return dict(zip(["n_rows", "n_keys", "n_dup_groups", "max_group",
                     "n_extra_rows", "extra_pop", "pop_raw", "pop_dedup",
                     "mrows_raw", "mcells_dedup"], [float(x) for x in r]))


def cmd_dedup(args):
    """Which published numbers move if DISTINCT is applied. Sample only."""
    from p36_recompute import official_gu          # zipfile+csv reader, no pandas

    tmp = args.tmp
    con = con_for(tmp)
    p29 = json.load(open(f"{ROOT}/eda/results_p29.json"))
    yms = [int(x) for x in args.months.split(",") if x]

    print("=== 43.2a duplicate census, whole month, both ends unrestricted ===")
    print(f"  {'ym':>7} {'rows':>14} {'keys':>14} {'dup groups':>12} "
          f"{'max/grp':>8} {'extra rows':>11} {'extra vol':>13} {'vol share':>10}")
    census = {}
    for ym in yms:
        c = dup_census(con, ym)
        census[str(ym)] = c
        print(f"  {ym:>7} {c['n_rows']:>14,.0f} {c['n_keys']:>14,.0f} "
              f"{c['n_dup_groups']:>12,.0f} {c['max_group']:>8,.0f} "
              f"{c['n_extra_rows']:>11,.0f} {c['extra_pop']:>13,.0f} "
              f"{c['extra_pop'] / c['pop_raw']:>10.5%}", flush=True)
    out["dup_census"] = census

    print("\n=== 43.2b p29's two gated numbers, computed both ways ===")
    print("  'as published' must reproduce results_p29.json, or this whole "
          "comparison is measuring my SQL rather than the dedup.")
    moves = {}
    for ym in yms:
        got = official_gu(ym)
        if got is None:
            print(f"  {ym}: no official 자치구 archive on disk -- p29's "
                  f"measurement is not defined for this month, skipping")
            continue
        off, off_rows, off_masked_rows = got
        row = {}
        for dedup in (False, True):
            mine = gu_shape(con, ym, dedup)
            keys = set(off) | set(mine)
            v_lo = sum(mine.get(k, (0.0, 0))[0] for k in keys)
            mcells = sum(mine.get(k, (0.0, 0))[1] for k in keys)
            off_v = sum(off.get(k, (0.0, 0))[0] for k in keys)
            hidden = off_v - v_lo
            per_cell = hidden / mcells if mcells else float("nan")
            # The reconciliation p29 asserts on: bands masked on neither side.
            clean = [k for k in keys
                     if mine.get(k, (0.0, 0))[1] == 0 and off.get(k, (0.0, 0))[1] == 0]
            cv = sum(mine.get(k, (0.0, 0))[0] for k in clean)
            co = sum(off.get(k, (0.0, 0))[0] for k in clean)
            rel = (cv - co) / co if co else float("nan")
            row["dedup" if dedup else "raw"] = dict(
                v_lo=v_lo, mcells=float(mcells), hidden=hidden,
                per_cell=per_cell, measured_share=hidden / off_v,
                recon_rel=rel, n_clean=len(clean))
        pub = p29[str(ym)]
        r, d = row["raw"], row["dedup"]
        print(f"\n  --- {ym} ---")
        print(f"  {'quantity':<28} {'published':>12} {'as published':>13} "
              f"{'deduplicated':>13} {'move':>10}")
        for lab, key, pk, fmt in [
                ("masked cell holds", "per_cell", "per_cell_mean", "{:.4f}"),
                ("measured masked share", "measured_share", "measured_share", "{:.5f}"),
                ("masked cell count", "mcells", "masked_cells", "{:.0f}"),
                ("gu reconciliation rel", "recon_rel", "reconciliation_rel_diff", "{:+.6f}")]:
            pv = pub[pk]
            print(f"  {lab:<28} {fmt.format(pv):>12} {fmt.format(r[key]):>13} "
                  f"{fmt.format(d[key]):>13} "
                  f"{(d[key] / r[key] - 1) if r[key] else float('nan'):>+9.4%}")
        # The gate rounds to 2 decimals. What matters is not whether the number
        # moves but whether it moves enough to change the printed value.
        print(f"  ROUNDED AS THE GATE PRINTS THEM (p31 section 4):")
        print(f"    per-cell   published {pub['per_cell_mean']:.2f}  ->  "
              f"deduplicated {d['per_cell']:.2f}   "
              f"{'GATE BREAKS' if round(d['per_cell'], 2) != round(pub['per_cell_mean'], 2) else 'gate holds'}")
        print(f"    masked %   published {pub['measured_share'] * 100:.2f}  ->  "
              f"deduplicated {d['measured_share'] * 100:.2f}   "
              f"{'GATE BREAKS' if round(d['measured_share'] * 100, 2) != round(pub['measured_share'] * 100, 2) else 'gate holds'}")
        moves[str(ym)] = dict(published=pub, raw=r, dedup=d)
    out["p29_fork"] = moves
    con.close()
    with open(f"{ROOT}/eda/results_p43_dedup.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p43_dedup.json  (a NEW file; no results_* was touched)")
    return 0



def eshare_fork(con, ym):
    """p19's inputs, raw and deduplicated, in ONE scan of the month.

    p19 reads gu_level.parquet's `pop_day_unmasked` and `n_masked_cells/n_days`.
    pop_day divides by that weekday's occurrence count per row, before summing,
    so the division is inside the aggregate here too. Grouping to the full row
    key with count(*) gives both versions from one pass: multiply by n for the
    raw (what gu_level holds), take it once for the deduplicated.
    """
    from p36_recompute import weekday_days
    nd = weekday_days(ym)
    case = "CASE dow_n " + " ".join(f"WHEN {d} THEN {n}" for d, n in
                                    sorted(nd.items())) + " END"
    glob = f"{PARQUET}/ym={ym}/*.parquet"
    rows = con.execute(f"""
        WITH k AS (
          SELECT dow_n, age, mtype, count(*) AS n,
                 min(pop) AS pop, bool_or(masked) AS masked
          FROM read_parquet('{glob}', hive_partitioning=false)
          WHERE o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
            AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
          GROUP BY ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype)
        SELECT age, substr(mtype, 2, 1) AS dest_attr,
               sum(coalesce(pop, 0) * n / ({case}))                    AS lo_raw,
               sum(coalesce(pop, 0)     / ({case}))                    AS lo_ded,
               sum(CASE WHEN masked THEN n   ELSE 0 END / ({case}))    AS nm_raw,
               sum(CASE WHEN masked THEN 1.0 ELSE 0 END / ({case}))    AS nm_ded
        FROM k GROUP BY 1,2""").fetchall()
    return {(int(a), d): tuple(float(x) for x in r)
            for a, d, *r in [list(x) for x in rows]}


def cmd_eshare(args):
    """Does the dedup move p19's gated E-share numbers? Two months, one scan each."""
    con = con_for(args.tmp)
    p19 = json.load(open(f"{ROOT}/eda/results_p19.json"))
    p29 = json.load(open(f"{ROOT}/eda/results_p29.json"))
    # p19's measured fill: mean over months of p29's per-band per-cell value,
    # for bands with more than 10,000 masked cells. Same rule, read not retyped.
    fills = {}
    for v in p29.values():
        if not isinstance(v, dict) or "by_age" not in v:
            continue
        for lab, x in v["by_age"].items():
            if x["masked_cells"] > 1e4:
                fills.setdefault(lab, []).append(x["per_cell"])
    fill = {a: sum(fills[AGE_LABEL[a]]) / len(fills[AGE_LABEL[a]])
            for a in AGES if AGE_LABEL[a] in fills}

    agg = {}
    for ym in (202003, 202012):
        t = time.time()
        agg[ym] = eshare_fork(con, ym)
        print(f"  scanned {ym} in {time.time() - t:.0f}s", flush=True)

    def share(ym, ages, c, which):
        i = 0 if which == "raw" else 1
        def vol(a, d):
            r = agg[ym].get((a, d))
            if r is None:
                return 0.0
            lo, nm = r[i], r[2 + i]
            return lo + (fill.get(a, 0.0) if c == "meas" else c) * nm
        e = sum(vol(a, "E") for a in ages)
        tot = sum(vol(a, d) for a in ages for d in ("H", "W", "E"))
        return e / tot if tot else float("nan")

    LIM = {"0-19": [0, 10, 15],
           "20-59": [20, 25, 30, 35, 40, 45, 50, 55],
           "60+": [60, 65, 70, 75, 80]}
    pub16 = {r["age"]: r for r in p19["e_share_change_16band"]}
    pub3 = {r["band"]: r for r in p19["e_share_change_3band"]}

    print("\n=== 43.2c p19's E-share change, raw vs deduplicated ===")
    print("  'raw' must reproduce results_p19.json to 2 dp or the comparison "
          "is measuring my SQL.")
    print(f"  {'band':>7} {'published':>10} {'raw':>9} {'dedup':>9} "
          f"{'move pp':>9} {'rounds to':>11}")
    rec, worst = {}, 0.0
    for a in AGES:
        d = {w: (share(202012, [a], "meas", w) - share(202003, [a], "meas", w)) * 100
             for w in ("raw", "dedup")}
        lab = AGE_LABEL[a]
        pv = pub16[lab]["meas"]
        mv = d["dedup"] - d["raw"]
        worst = max(worst, abs(mv))
        flag = ("same" if round(d["dedup"], 2) == round(pv, 2)
                else f"{round(d['dedup'], 2):+.2f} != {pv:+.2f}  MOVES")
        print(f"  {lab:>7} {pv:>+10.2f} {d['raw']:>+9.3f} {d['dedup']:>+9.3f} "
              f"{mv:>+9.4f} {flag:>11}")
        rec[lab] = dict(published=pv, raw=d["raw"], dedup=d["dedup"], move_pp=mv)
    for b, ags in LIM.items():
        d = {w: (share(202012, ags, "meas", w) - share(202003, ags, "meas", w)) * 100
             for w in ("raw", "dedup")}
        pv = pub3[b]["meas"]
        mv = d["dedup"] - d["raw"]
        worst = max(worst, abs(mv))
        flag = ("same" if round(d["dedup"], 2) == round(pv, 2)
                else f"{round(d['dedup'], 2):+.2f} != {pv:+.2f}  MOVES")
        print(f"  {b:>7} {pv:>+10.2f} {d['raw']:>+9.3f} {d['dedup']:>+9.3f} "
              f"{mv:>+9.4f} {flag:>11}")
        rec[b] = dict(published=pv, raw=d["raw"], dedup=d["dedup"], move_pp=mv)
    r16 = {w: max((share(202012, [a], "meas", w) - share(202003, [a], "meas", w)) * 100
                  for a in AGES)
              - min((share(202012, [a], "meas", w) - share(202003, [a], "meas", w)) * 100
                    for a in AGES) for w in ("raw", "dedup")}
    r3 = {w: max((share(202012, g, "meas", w) - share(202003, g, "meas", w)) * 100
                 for g in LIM.values())
             - min((share(202012, g, "meas", w) - share(202003, g, "meas", w)) * 100
                   for g in LIM.values()) for w in ("raw", "dedup")}
    print(f"\n  16-band spread   published 5.79   raw {r16['raw']:.3f}   "
          f"dedup {r16['dedup']:.3f}")
    print(f"  3-band spread    raw {r3['raw']:.3f}   dedup {r3['dedup']:.3f}")
    print(f"  three bands retain  published 50.6%   "
          f"raw {r3['raw'] / r16['raw']:.1%}   "
          f"dedup {r3['dedup'] / r16['dedup']:.1%}")
    print(f"\n  largest move of any gated E-share number: {worst:.4f} pp")
    out["p19_fork"] = dict(bands=rec, spread16=r16, spread3=r3,
                           retain_raw=r3["raw"] / r16["raw"],
                           retain_dedup=r3["dedup"] / r16["dedup"],
                           worst_move_pp=worst)
    con.close()
    prev = {}
    pj = f"{ROOT}/eda/results_p43_dedup.json"
    if os.path.exists(pj):
        prev = json.load(open(pj))
    prev.update(out)
    with open(pj, "w") as fh:
        json.dump(prev, fh, indent=1, default=str)
    print("\nwrote results_p43_dedup.json  (a NEW file; no results_* was touched)")
    return 0


# ===================================================================== item 3
CORE_COLS = ["ym", "dow_n", "weekend", "d_dong", "arr_hour", "dest_attr",
             "age", "v_obs", "n_masked"]


def cmd_core(args):
    """Build derived/matrix_core.parquet, per month, then concatenate.

    The per-month shard is exactly what p26's cells() returns -- imported and
    called, not re-implemented -- so the released table is the same object the
    published matrices were built from, deduplicated by cells()' min(pop)
    GROUP BY ALL. `ym` and a weekend flag are the only columns added.

    GRANULARITY, AND A DEPARTURE FROM THE SPEC THAT HAD TO BE MADE. A9 in
    ../release/data_descriptor_plan.md keys on (ym, weekday/weekend, ...). That key CANNOT
    reproduce the published matrices: p26 forms A per (place, hour, DAY OF WEEK)
    cell and A is quadratic in the cell composition, so collapsing seven
    weekdays into two before the product is not the same number. dow_n is
    therefore kept and `weekend` is carried alongside as a derived column, so a
    user who wants the spec's key can group to it and a user who wants the
    published matrices can reproduce them. The cost of the collapse is measured
    in 43.3c rather than asserted.
    """
    tmp = args.tmp
    parts = DERIVED / "matrix_core_parts"
    os.makedirs(parts, exist_ok=True)
    disk = months_on_disk()
    yms = ([int(x) for x in args.months.split(",")] if args.months else disk)
    print(f"=== 43.3a building {len(yms)} monthly shards ===")
    print(f"  target volume {DERIVED}, {free_gb(DERIVED):.1f} GB free, "
          f"abort margin {MIN_FREE_GB} GB")
    t0 = time.time()
    nrow = 0
    for i, ym in enumerate(yms, 1):
        g0 = check_space(DERIVED, f"shard {ym}")
        p = parts / f"{ym}.parquet"
        if p.exists() and not args.force:
            nrow += duckdb.connect().execute(
                f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]
            continue
        df = cells(ym)                      # p26's own function; cached per month
        df.insert(0, "ym", np.int32(ym))
        df["weekend"] = df.dow_n >= 6
        # ROW ORDER IS LOAD-BEARING, so it is preserved rather than sorted.
        # p26's matrices() sums within (loc, hour, dow, age) groups in ROW
        # order, and float addition is not associative, so at gu (34 terms) and
        # city (848 terms) level a re-sorted table reproduces the published A
        # only to the last ulp, not exactly. Sorting would compress better; the
        # trade was taken the other way because "the published matrix rebuilds
        # bit for bit" is the claim the descriptor is making.
        df = df[CORE_COLS]
        df.to_parquet(p, index=False, compression="zstd")
        nrow += len(df)
        g1 = free_gb(DERIVED)
        print(f"  [{i:>2}/{len(yms)}] {ym}  {len(df):>9,} rows  "
              f"{p.stat().st_size / 1e6:6.1f} MB  free {g1:.1f} GB "
              f"(-{g0 - g1:.2f})", flush=True)
    print(f"  shards done: {nrow:,} rows in {(time.time() - t0) / 60:.1f} min")

    # ---------------------------------------------------------- concatenate
    check_space(DERIVED, "concatenation")
    dest = DERIVED / "matrix_core.parquet"
    if dest.exists():
        raise SystemExit(f"ABORT: {dest} already exists. This script never "
                         f"overwrites a data file; move it aside first.")
    con = con_for(args.tmp, mem="4GB")
    tc = time.time()
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{parts}/*.parquet'))
        TO '{dest}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
    sz = dest.stat().st_size
    n = con.execute(f"SELECT count(*) FROM read_parquet('{dest}')").fetchone()[0]
    print(f"\n=== 43.3b the file ===")
    print(f"  {dest}")
    print(f"  rows {n:,}   size {sz / 2**30:.3f} GiB ({sz / 1e6:.0f} MB)   "
          f"concat {time.time() - tc:.0f}s   total {(time.time() - t0) / 60:.1f} min")
    assert n == nrow, f"concatenated {n:,} but shards held {nrow:,}"
    months = con.execute(
        f"SELECT count(DISTINCT ym) FROM read_parquet('{dest}')").fetchone()[0]
    print(f"  months {months}   (shards left in {parts} until the memo is read)")

    # A9's stated estimate was 92.6% fill / ~7.3M rows, measured on ONE year and
    # on a two-panel (W, E) grid. Both halves are checked against 79 months here
    # rather than carried forward.
    grid_all = 424 * 24 * 7 * 16 * 3
    we = con.execute(f"SELECT count(*) FROM read_parquet('{dest}') "
                     f"WHERE dest_attr <> 'H'").fetchone()[0]
    print(f"\n=== 43.3c the fill estimate, checked on 79 months ===")
    print(f"  full (dong x hour x dow x age x attr) grid   "
          f"{grid_all:,}/month, observed {n / months:,.0f}  "
          f"fill {n / months / grid_all:.1%}")
    coll = con.execute(f"""
        SELECT count(*) FROM (SELECT ym, weekend, d_dong, arr_hour, dest_attr, age
                              FROM read_parquet('{dest}') GROUP BY ALL)""").fetchone()[0]
    coll_we = con.execute(f"""
        SELECT count(*) FROM (SELECT ym, weekend, d_dong, arr_hour, dest_attr, age
                              FROM read_parquet('{dest}') WHERE dest_attr <> 'H'
                              GROUP BY ALL)""").fetchone()[0]
    print(f"  A9's stated key (ym, weekend, dong, hour, age, attr):")
    print(f"    all three attrs  {coll:,} rows   "
          f"fill {coll / months / (424 * 24 * 2 * 16 * 3):.1%}")
    print(f"    W+E only         {coll_we:,} rows   "
          f"fill {coll_we / months / (424 * 24 * 2 * 16 * 2):.1%}   "
          f"<- the shape the 92.6% / 7.3M estimate was for")
    print(f"  dow_n retained     {n:,} rows (all attrs), {we:,} (W+E)")
    out["shape"] = dict(rows=int(n), bytes=int(sz), months=int(months),
                        rows_we=int(we), rows_collapsed=int(coll),
                        rows_collapsed_we=int(coll_we),
                        fill_dow=n / months / grid_all,
                        fill_collapsed_we=coll_we / months / (424 * 24 * 2 * 16 * 2))
    con.close()

    # ------------------------------------------------- 43.3d the real validation
    print(f"\n=== 43.3d rebuild p26's published A from matrix_core ===")
    pub = json.load(open(f"{ROOT}/eda/results_p26.json"))
    have = {k.split("|")[0] for k in pub["matrices"]}
    want = [y for y in (args.check.split(",") if args.check else
                        ["202001", "202312"]) if y in have]
    print(f"  checking {want} (published months: {sorted(have)})")
    dev = {}
    for ymk in want:
        ym = int(ymk)
        core = pd.read_parquet(dest, filters=[("ym", "==", ym)])
        for panel in ("W", "E", "WE"):
            for level in ("dong", "gu", "city"):
                key = f"{ym}|{panel}|{level}"
                if key not in pub["matrices"]:
                    continue
                A = matrices(core, ym, panel, level)
                B = np.array(pub["matrices"][key]["A"], float)
                d = float(np.max(np.abs(A - B)))
                rel = float(np.max(np.abs(A - B) / np.maximum(np.abs(B), 1e-300)))
                dev[key] = dict(max_abs=d, max_rel=rel, exact=bool(d == 0.0))
                print(f"  {key:<22} max|A-A_pub| {d:.3e}   "
                      f"max rel {rel:.3e}   "
                      f"{'BIT FOR BIT' if d == 0.0 else 'DIFFERS'}")
    out["p26_rebuild"] = dev

    # And what the spec's own key costs, measured.
    print(f"\n=== 43.3e what collapsing dow_n to weekday/weekend would cost ===")
    print("  A is quadratic in the cell composition, so this is not a rounding "
          "question. Reported because A9's stated key is the collapsed one.")
    cost = {}
    for ymk in want:
        ym = int(ymk)
        core = pd.read_parquet(dest, filters=[("ym", "==", ym)])
        c2 = (core.assign(dow_n=np.where(core.weekend, 7, 1))
              .groupby(["ym", "dow_n", "weekend", "d_dong", "arr_hour",
                        "dest_attr", "age"], as_index=False)
              [["v_obs", "n_masked"]].sum())
        for panel in ("WE",):
            key = f"{ym}|{panel}|dong"
            if key not in pub["matrices"]:
                continue
            B = np.array(pub["matrices"][key]["A"], float)
            A = matrices(c2, ym, panel, "dong")
            r = float(np.max(np.abs(A - B) / np.maximum(np.abs(B), 1e-300)))
            sB = stats(B, np.ones(len(AGES)))["assortativity"]
            sA = stats(A, np.ones(len(AGES)))["assortativity"]
            cost[key] = dict(max_rel=r, assort_pub=sB, assort_collapsed=sA)
            print(f"  {key:<22} max rel deviation {r:.3%}   "
                  f"assortativity {sB:+.4f} -> {sA:+.4f}")
    out["collapse_cost"] = cost

    with open(f"{ROOT}/eda/results_p43_core.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p43_core.json  (a NEW file; no results_* was touched)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    scratch = os.environ.get("P43_TMP", "/tmp/p43")

    m = sub.add_parser("manifest")
    m.add_argument("--all", action="store_true")
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_manifest)

    d = sub.add_parser("dedup")
    d.add_argument("--months", default="202003,202012")
    d.add_argument("--tmp", default=scratch)
    d.set_defaults(fn=cmd_dedup)

    e = sub.add_parser("eshare")
    e.add_argument("--tmp", default=scratch)
    e.set_defaults(fn=cmd_eshare)

    c = sub.add_parser("core")
    c.add_argument("--months", default="")
    c.add_argument("--check", default="")
    c.add_argument("--force", action="store_true")
    c.add_argument("--tmp", default=scratch)
    c.set_defaults(fn=cmd_core)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
