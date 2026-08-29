#!/usr/bin/env python
"""Pull the 자치구-level 서울 생활이동 release (infSeq=3) — the ground truth for
what dong-level masking hides.

The dong product masks any cell below 3 and 27.68% of its cells are masked, so
every volume this project reports carries a 0/1.5/3 imputation band. The gu
product is the SAME pipeline aggregated one level up before the threshold is
applied, so a cell that is masked at dong level is usually published at gu level.
Summing our dong data to gu and differencing against the official gu file
therefore measures the hidden volume instead of bounding it.

That comparison was a standing to-do ("official 자치구 file", eda/README next
steps 7; DATA_REPORT 'gu_level.parquet 待與官方자치구檔對帳"). It became urgent
when Lim et al.'s Zenodo release turned out to be built on this exact product:
their odm_*.csv reproduces it to 1.000000 in every age band.

Same three-request flow and the same content-first verification as
dl_mobility.py, whose session and transfer helpers this reuses; only the product
constants differ (infSeq, member names, 시군구 columns, size range).

    python eda/dl_gu.py --months 202003,202012
    python eda/dl_gu.py --verify
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_mobility import BadFile, FILE, PAGE, UA  # noqa: E402
from paths import DATA_ROOT  # noqa: E402

ZIPDIR = DATA_ROOT / "raw" / "gu"
MANIFEST = ZIPDIR / "manifest.json"
INF_SEQ = "3"                    # 2 = 행정동, 3 = 자치구
HEADER = ["대상연월", "요일", "도착시간", "출발 시군구 코드", "도착 시군구 코드",
          "성별", "나이", "이동유형", "평균 이동 시간(분)", "이동인구(합)"]
MEMBER_RE = re.compile(r"생활이동_자치구_(\d{4})\.(\d{2})_(\d{2})시\.csv$")
N_HOURS = 24
ZIP_BYTES_RANGE = (80_000_000, 400_000_000)


def zip_path(ym):
    return ZIPDIR / f"생활이동_자치구_{ym}.zip"


def opener():
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
        urllib.request.HTTPRedirectHandler())
    op.addheaders = [("User-Agent", UA), ("Referer", PAGE)]
    op.open(PAGE, timeout=60).read()
    return op


def download(op, ym, dest, tries=4):
    form = urllib.parse.urlencode({"infId": "DOWNLOAD", "infSeq": INF_SEQ,
                                   "seqNo": "", "seq": str(ym)}).encode()
    last = None
    for attempt in range(1, tries + 1):
        tmp = dest.with_suffix(".part")
        try:
            req = urllib.request.Request(FILE, data=form, method="POST")
            req.add_header("Content-Type",
                           "application/x-www-form-urlencoded; charset=UTF-8")
            r = op.open(req, timeout=300)
            expect = int(r.headers.get("Content-Length") or 0)
            h, got = hashlib.sha256(), 0
            with r, open(tmp, "wb") as fh:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    if got == 0 and buf[:4] != b"PK\x03\x04":
                        raise BadFile(f"not a zip: {buf[:80]!r}")
                    fh.write(buf)
                    h.update(buf)
                    got += len(buf)
            if expect and got != expect:
                raise BadFile(f"truncated: {got:,} of {expect:,}")
            if not ZIP_BYTES_RANGE[0] <= got <= ZIP_BYTES_RANGE[1]:
                raise BadFile(f"{got:,} bytes outside {ZIP_BYTES_RANGE}")
            os.replace(tmp, dest)
            return got, h.hexdigest()
        except (urllib.error.URLError, BadFile, TimeoutError, OSError) as e:
            last = e
            tmp.unlink(missing_ok=True)
            if attempt < tries:
                print(f"    attempt {attempt} failed ({e}); retrying",
                      flush=True)
                time.sleep(15 * attempt)
    raise BadFile(f"{ym}: gave up after {tries} attempts — {last}")


def check_zip(ym, path):
    """24 hours of the month asked for, cp949, the 시군구 header, 5-digit codes."""
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise BadFile(f"unreadable zip: {e}")
    with zf:
        hours, rows = {}, 0
        for n in [x for x in zf.namelist() if not x.endswith("/")]:
            m = MEMBER_RE.search(os.path.basename(n))
            if not m:
                raise BadFile(f"unexpected member {n!r}")
            if int(m.group(1) + m.group(2)) != ym:
                raise BadFile(f"member {n!r} is not month {ym}")
            hours[m.group(3)] = n
        if set(hours) != {f"{h:02d}" for h in range(N_HOURS)}:
            raise BadFile(f"hours {sorted(hours)} != 00..23")
        for hh in sorted(hours):
            text = zf.open(hours[hh]).read().decode("cp949")
            rd = list(csv.reader(io.StringIO(text)))
            if [c.strip() for c in rd[0]] != HEADER:
                raise BadFile(f"{hh}시: header is {rd[0]}")
            first = rd[1]
            if first[0].strip() != str(ym) or int(first[2]) != int(hh):
                raise BadFile(f"{hh}시: first row is {first[:3]}")
            if len(first[3].strip()) != 5:
                raise BadFile(f"{hh}시: 시군구 code {first[3]!r} is not 5 digits")
            rows += len(rd) - 1
    return len(hours), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ZIPDIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    if args.verify:
        for ym in sorted(int(k) for k in man):
            p = zip_path(ym)
            if not p.exists():
                print(f"  {ym}  zip gone (manifest only)")
                continue
            try:
                h, rows = check_zip(ym, p)
                print(f"  {ym}  ok  {h} hours  {rows:,} rows")
            except BadFile as e:
                print(f"  {ym}  BAD  {e}")
        return

    yms = [int(x) for x in args.months.split(",") if x]
    if not yms:
        print("nothing to do: pass --months or --verify")
        return
    op = opener()
    for ym in yms:
        dest = zip_path(ym)
        if dest.exists() and not args.force:
            print(f"  {ym} already on disk")
            continue
        print(f"  {ym} downloading ...", flush=True)
        size, sha = download(op, ym, dest)
        hours, rows = check_zip(ym, dest)
        man[str(ym)] = dict(bytes=size, sha256=sha, hours=hours, rows=rows,
                            fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()))
        MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        print(f"    {size/1e6:.1f} MB, {hours} hours, {rows:,} rows, "
              f"sha {sha[:16]}")
        time.sleep(3)


if __name__ == "__main__":
    main()
