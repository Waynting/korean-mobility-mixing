#!/usr/bin/env python
"""CP949 CSV -> parquet ETL for 서울 생활이동 (행정동).

Two traps in the raw files, both silent:

1. DuckDB 1.5.5's CSV reader drops ~12.5% of rows on these files without raising
   (verified: pandas 5,634,192 vs DuckDB 4,929,770 on 2020.01_08시). So the read
   side is pandas; DuckDB is used only downstream on the parquet.
2. The header row alone ends in CRLF while every data row ends in LF, and the
   macOS filesystem stores the Hangul directory names in NFD, so any glob or
   .replace() written with NFC Korean matches nothing. All path handling here
   stays ASCII.

Every file's row count is checked against `wc -l` - 1; a mismatch aborts.
"""
import glob
import os
import re
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PARQUET, RAW, RESULTS, require

# `require` is called inside the guard, not here: it raises SystemExit when the
# drive is not mounted, and an import that kills the importing process is not a
# thing an unrelated script should be able to trip over. `convert()` takes a
# full path, so nothing at module level needs the checked RAW.
OUT = str(PARQUET)

DOW_NUM = {"월": 1, "화": 2, "수": 3, "목": 4, "금": 5, "토": 6, "일": 7}

READ_DTYPES = {
    0: "int32", 1: "string", 2: "int8", 3: "int32", 4: "int32",
    5: "string", 6: "int16", 7: "string", 8: "float32", 9: "string",
}
NAMES = ["ym", "dow", "arr_hour", "o_dong", "d_dong",
         "sex", "age", "mtype", "mean_min", "pop_raw"]

SCHEMA = pa.schema([
    ("ym", pa.int32()), ("dow_n", pa.int8()), ("arr_hour", pa.int8()),
    ("o_dong", pa.int32()), ("d_dong", pa.int32()),
    ("sex", pa.dictionary(pa.int8(), pa.string())),
    ("age", pa.int16()),
    ("mtype", pa.dictionary(pa.int8(), pa.string())),
    ("mean_min", pa.float32()), ("pop", pa.float32()), ("masked", pa.bool_()),
])


def convert(path):
    m = re.search(r"(\d{4})\.(\d{2})_(\d{2})", os.path.basename(path))
    ym, hh = m.group(1) + m.group(2), m.group(3)
    dst = os.path.join(OUT, f"ym={ym}", f"h{hh}.parquet")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    expected = int(subprocess.run(["wc", "-l", path], capture_output=True,
                                  text=True).stdout.split()[0]) - 1
    t0 = time.time()
    writer = pq.ParquetWriter(dst, SCHEMA, compression="zstd")
    rows = masked = 0
    hours, ysm, bad_dow = set(), set(), 0
    try:
        for ch in pd.read_csv(path, encoding="cp949", header=0, names=NAMES,
                              dtype=READ_DTYPES, engine="c", chunksize=1_000_000,
                              na_filter=False):
            dn = ch["dow"].map(DOW_NUM)
            bad_dow += int(dn.isna().sum())
            mask = ch["pop_raw"] == "*"
            tbl = pa.Table.from_pydict({
                "ym": ch["ym"], "dow_n": dn.fillna(0).astype("int8"),
                "arr_hour": ch["arr_hour"], "o_dong": ch["o_dong"],
                "d_dong": ch["d_dong"], "sex": ch["sex"], "age": ch["age"],
                "mtype": ch["mtype"], "mean_min": ch["mean_min"],
                "pop": pd.to_numeric(ch["pop_raw"], errors="coerce").astype("float32"),
                "masked": mask,
            }, schema=SCHEMA)
            writer.write_table(tbl)
            rows += len(ch)
            masked += int(mask.sum())
            hours |= set(ch["arr_hour"].unique().tolist())
            ysm |= set(ch["ym"].unique().tolist())
    finally:
        writer.close()

    if rows != expected:
        raise RuntimeError(f"{path}: wrote {rows} rows, file has {expected}")
    return dict(ym=ym, hour=hh, rows=rows, expected=expected, masked=masked,
                bad_dow=bad_dow, hours_in_file=sorted(hours),
                ym_in_file=sorted(ysm), secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    RAW = require(RAW, "raw CSV directory (KOREAN_RAW_DIR)")
    files = sorted(glob.glob(os.path.join(RAW, "*", "*.csv")))
    print(f"{len(files)} files", flush=True)
    report = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(convert, files):
            report.append(r)
            print(f"{r['ym']} {r['hour']}시  rows={r['rows']:>10,}  "
                  f"masked={r['masked']:>9,} ({r['masked']/r['rows']:.1%})  "
                  f"bad_dow={r['bad_dow']}  hours={r['hours_in_file']}  "
                  f"{r['secs']}s", flush=True)
    with open(os.path.join(RESULTS, "etl_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("TOTAL rows", f"{sum(r['rows'] for r in report):,}")
    print("done")
