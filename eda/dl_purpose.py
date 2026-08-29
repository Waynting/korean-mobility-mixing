#!/usr/bin/env python
"""Pull 수도권 생활이동 (성 연령별, 도착지 기준) — 내국인 — from data.seoul.go.kr.

This is the SECOND KT product, and Phase 4b needs it: where 서울 생활이동 labels a
trip endpoint H/W/E, this one labels the trip's PURPOSE. The advisor's 65+ attack
("주간상주지 for a retiree may be a 경로당, a market or a hospital") turns from a
plausibility argument into a measurement the moment both classifiers can be run
on the same city, month and pipeline.

OA-22298, one zip per month, 2023-01 .. 2026-07, about 75 MB compressed and
250 MB expanded — two orders of magnitude smaller than the dong product, because
it is destination-side only. The download is the same three-request dance as
dl_mobility.py except that the form posts infId=OA-22298 and infSeq=1 rather than
infId=DOWNLOAD, so the two cannot share a fetcher.

WHAT IS INSIDE, and how it differs from the H/W/E product — every one of these
differences has to be carried into any comparison:

  * DAILY files. One CSV per calendar date, so this product HAS the date column
    that the dong product lacks. That is what makes a holiday-clean comparison
    possible without dropping whole weekdays.
  * 10-YEAR age bands (00,10,...,70+), 8 per sex, against our sixteen 5-year
    bands. Ours aggregate up to theirs exactly, so the comparison happens at
    their resolution and section 5's bandwidth argument applies to the loss.
  * MIXED time resolution: `time_cd` is two digits for an off-peak hour and four
    for a 20-minute bin inside 07:00-09:59 and 17:00-19:59. Reading it as an
    integer silently sends the entire morning peak to hour 7 (0700 -> 700).
  * MOIS 8-digit destination codes, not the dong product's own 7-digit space,
    and destinations outside Seoul appear at 시군구 level (codes ending 0000).
    Seoul is about a third of the volume.
  * 내국인 only. Our product does not separate nationality, so the denominators
    differ by the registered-foreigner share — 6.26% at 20-24, 0.25% at 80+
    (phase1b-foreign), i.e. concentrated exactly where it hurts least here.

    python eda/dl_purpose.py --list
    python eda/dl_purpose.py --months 202606,202312
    python eda/dl_purpose.py --verify
    python eda/dl_purpose.py --months 202606 --etl
"""
import argparse
import calendar
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
from paths import DATA_ROOT, DERIVED  # noqa: E402

ZIPDIR = DATA_ROOT / "raw" / "purpose"
MANIFEST = ZIPDIR / "manifest.json"
PAGE = "https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do"
FILE = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
INF_ID, INF_SEQ = "OA-22298", "1"
FIRST_YM, LAST_YM = 202301, 202607
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

SEXES = ["male", "feml"]
AGE10 = ["00", "10", "20", "30", "40", "50", "60", "70"]
HEADER = (["d_admdong_cd", "time_cd", "move_purpose"]
          + [f"{s}_{a}_cnt" for s in SEXES for a in AGE10]
          + ["total_cnt", "etl_ymd"])
MEMBER_RE = re.compile(r"seoul_purpose_admdong1_in_(\d{8})\.csv$")
ZIP_BYTES_RANGE = (20_000_000, 300_000_000)


class BadFile(Exception):
    """The bytes on disk are not the month we asked for."""


def months(first=FIRST_YM, last=LAST_YM):
    y, m = divmod(first, 100)
    while y * 100 + m <= last:
        yield y * 100 + m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def zip_path(ym):
    return ZIPDIR / f"seoul_purpose_admdong1_in_{ym}.zip"


def opener():
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
        urllib.request.HTTPRedirectHandler())
    op.addheaders = [("User-Agent", UA), ("Referer", PAGE)]
    op.open(PAGE, timeout=60).read()          # the endpoint wants a session
    return op


def download(op, ym, dest, tries=4):
    """Stream one month's zip. Returns (bytes, sha256). No Range support."""
    form = {"infId": INF_ID, "infSeq": INF_SEQ, "seqNo": "", "seq": str(ym)}
    body = urllib.parse.urlencode(form).encode()
    last = None
    for attempt in range(1, tries + 1):
        tmp = dest.with_suffix(".part")
        try:
            req = urllib.request.Request(FILE, data=body, method="POST")
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
                wait = 15 * attempt
                print(f"    attempt {attempt} failed ({e}); retry in {wait}s",
                      flush=True)
                time.sleep(wait)
    raise BadFile(f"{ym}: gave up after {tries} attempts — {last}")


def check_zip(ym, path):
    """Content, not transfer: every day of the month, cp949, exact header.

    A Korean portal answers a timed-out session with a 200 and a
    Content-Disposition header, so the only trustworthy check is whether the
    bytes decode into the month that was asked for.
    """
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise BadFile(f"unreadable zip: {e}")
    with zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        days = {}
        for n in names:
            m = MEMBER_RE.search(os.path.basename(n))
            if not m:
                raise BadFile(f"unexpected member {n!r}")
            days[m.group(1)] = n
        y, mo = divmod(ym, 100)
        want = {f"{ym}{d:02d}" for d in
                range(1, calendar.monthrange(y, mo)[1] + 1)}
        if set(days) != want:
            raise BadFile(f"days {sorted(set(days) ^ want)} missing or extra")
        rows = 0
        for d in sorted(days):
            raw = zf.open(days[d]).read()
            try:
                text = raw.decode("cp949")
            except UnicodeDecodeError as e:
                raise BadFile(f"{d}: not cp949 ({e})")
            rd = list(csv.reader(io.StringIO(text)))
            if [c.strip() for c in rd[0]] != HEADER:
                raise BadFile(f"{d}: header is {rd[0]}")
            if not rd[1:]:
                raise BadFile(f"{d}: no data rows")
            if rd[1][-1] != d:
                raise BadFile(f"{d}: first row carries etl_ymd {rd[1][-1]}")
            rows += len(rd) - 1
    return len(days), rows


def etl(ym, path):
    """Long parquet: one row per (date, time_cd, purpose, dong, sex, age band).

    Zeros are dropped rather than stored. They are not observations — the file
    is a sparse list of cells that occurred — and keeping them would triple the
    file for no information, the same reasoning as p16's finding that 99% of the
    logical grid has no row.
    """
    import pandas as pd
    frames = []
    with zipfile.ZipFile(path) as zf:
        for n in sorted(zf.namelist()):
            if n.endswith("/"):
                continue
            df = pd.read_csv(io.BytesIO(zf.open(n).read()), encoding="cp949",
                             dtype={"d_admdong_cd": str, "time_cd": str,
                                    "etl_ymd": str})
            cnt = [c for c in df.columns if c.endswith("_cnt")
                   and c != "total_cnt"]
            # total_cnt is the row sum by construction; assert rather than
            # trust, because a silently rescaled column would move every share.
            assert (df[cnt].sum(axis=1) - df.total_cnt).abs().max() < 1e-6, \
                f"{n}: total_cnt is not the row sum"
            long = df.melt(id_vars=["d_admdong_cd", "time_cd", "move_purpose",
                                    "etl_ymd"], value_vars=cnt,
                           var_name="col", value_name="cnt")
            long = long[long.cnt > 0].copy()
            long["sex"] = long.col.str[:4].map({"male": "M", "feml": "F"})
            long["age10"] = long.col.str[5:7].astype("int8")
            long["hour"] = long.time_cd.str[:2].astype("int8")
            long["min20"] = long.time_cd.str[2:].replace("", "0").astype("int8")
            frames.append(long.drop(columns="col"))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out.etl_ymd
    out = out.drop(columns="etl_ymd")
    dest = DERIVED / f"purpose_{ym}.parquet"
    DERIVED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    return dest, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--etl", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sleep", type=float, default=3.0)
    args = ap.parse_args()

    ZIPDIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    if args.list:
        for ym in months():
            p = zip_path(ym)
            print(f"  {ym}  {'on disk' if p.exists() else '-':>8}  "
                  f"{p.stat().st_size / 1e6:8.1f} MB" if p.exists()
                  else f"  {ym}  {'-':>8}")
        return

    if args.verify:
        for ym in sorted(int(k) for k in man):
            p = zip_path(ym)
            if not p.exists():
                print(f"  {ym}  zip gone (manifest only)")
                continue
            try:
                d, rows = check_zip(ym, p)
                print(f"  {ym}  ok  {d} days  {rows:,} rows")
            except BadFile as e:
                print(f"  {ym}  BAD  {e}")
        return

    yms = [int(x) for x in args.months.split(",") if x] or []
    if not yms:
        print("nothing to do: pass --months, --list or --verify")
        return
    op = opener()
    for ym in yms:
        dest = zip_path(ym)
        if dest.exists() and not args.force:
            print(f"  {ym} already on disk")
        else:
            print(f"  {ym} downloading ...", flush=True)
            size, sha = download(op, ym, dest)
            days, rows = check_zip(ym, dest)
            man[str(ym)] = dict(bytes=size, sha256=sha, days=days, rows=rows,
                                fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime()))
            MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
            print(f"    {size/1e6:.1f} MB, {days} days, {rows:,} rows, "
                  f"sha {sha[:16]}")
            time.sleep(args.sleep)
        if args.etl:
            p, n = etl(ym, dest)
            print(f"    -> {p.name}  {n:,} rows")


if __name__ == "__main__":
    main()
