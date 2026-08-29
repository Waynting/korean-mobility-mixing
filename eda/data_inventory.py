#!/usr/bin/env python
"""What is actually on the drive, in a form the report gate can check.

Section 8 of the advisor report quotes the state of the download -- how many
months, how many rows, how many of them carry a zip checksum. Those were drive
facts with no results file behind them, so p31 could only list them under "not
covered here" and they were verified by hand each time they changed. They
changed four times in two days and went stale twice.

This writes the inventory to results_inventory.json so the gate can check the
letter against it like any other number. It is deliberately dumb: it counts
files and rows and reports gaps, and it asserts nothing about whether the data
is correct -- etl_report.json and dl_mobility.py --verify do that.

    python eda/data_inventory.py            # count, write, print
    python eda/data_inventory.py --fast     # skip the full row count
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb

from paths import DATA_ROOT, PARQUET, PARQUET_GLOB, ROOT

FIRST, LAST = 202001, 202607


def span(first=FIRST, last=LAST):
    return [y * 100 + m for y in range(first // 100, last // 100 + 1)
            for m in range(1, 13) if first <= y * 100 + m <= last]


def ranges(months):
    """Collapse a sorted month list into printable runs."""
    runs, start, prev = [], None, None
    for m in months:
        if start is None:
            start = prev = m
            continue
        y, mo = divmod(prev, 100)
        nxt = (y + 1) * 100 + 1 if mo == 12 else prev + 1
        if m == nxt:
            prev = m
        else:
            runs.append((start, prev))
            start = prev = m
    if start is not None:
        runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    have = {int(p.name.split("=")[1]): len(list(p.glob("*.parquet")))
            for p in PARQUET.glob("ym=*")}
    full = sorted(k for k, v in have.items() if v == 24)
    partial = {k: v for k, v in have.items() if v != 24}
    want = span()
    missing = [w for w in want if w not in full]

    man_path = DATA_ROOT / "raw" / "mobility" / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man = {k: v for k, v in man.items() if isinstance(v, dict)}
    with_sha = sum(1 for v in man.values() if v.get("zip_sha256"))

    rows = None
    if not args.fast:
        con = duckdb.connect()
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA disable_progress_bar")
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{PARQUET_GLOB}', "
            f"hive_partitioning=false)").fetchone()[0]

    out = dict(
        span_first=FIRST, span_last=LAST, span_months=len(want),
        months_complete=len(full), months_partial=partial,
        months_missing=len(missing), missing_list=missing,
        missing_ranges=ranges(missing),
        parquet_files=sum(have.values()),
        parquet_rows=rows,
        manifest_months=len(man), manifest_with_sha256=with_sha,
        manifest_files=sum(v.get("files", 0) for v in man.values()),
        # the 2020 twelve were fetched by hand before dl_mobility.py existed, so
        # they carry ETL row-count verification but no zip checksum. Stating the
        # arithmetic here stops the gap reading as a missing-data problem.
        months_without_zip_sha=len(full) - len(man))

    print(f"span            {FIRST}..{LAST}  ({len(want)} months)")
    print(f"complete        {len(full)}  ({len(full) / len(want):.0%})")
    print(f"partial         {partial if partial else 'none'}")
    print(f"missing         {len(missing)}" + (f"  {out['missing_ranges']}"
                                               if missing else ""))
    print(f"parquet files   {out['parquet_files']:,}")
    if rows is not None:
        print(f"parquet rows    {rows:,}")
    print(f"manifest        {len(man)} months, {with_sha} with zip sha256, "
          f"{out['manifest_files']:,} files")
    print(f"no zip sha256   {out['months_without_zip_sha']} months "
          f"(the hand-fetched 2020 twelve; ETL-verified instead)")

    with open(f"{ROOT}/eda/results_inventory.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results_inventory.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
