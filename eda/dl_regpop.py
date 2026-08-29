#!/usr/bin/env python
"""L2 — pull 주민등록 연령별 인구현황, monthly, 읍면동 x 1세 x 성별, 2020-01..2026-07.

This is the denominator for rho-hat in Phase 1. The advisor's route hangs on it,
so the download has to be reproducible and each file has to be checked at the
moment it lands rather than trusted because the HTTP status was 200.

The site (jumin.mois.go.kr) has no API. The 통계표 page posts a hidden form to
`downloadCsvAge.do`; the parameters below are that form, with three changes from
its defaults:

    sltArgTypes = 1     single-year ages, not the default 10-year bands
    xlsStats    = 3     전체읍면동현황, not the currently displayed screen
    state       = 3     the radio that xlsStats mirrors

`searchYearMonth=month` selects the monthly table. Start and end month are set
equal, one request per month: the national 읍면동 table for one month is already
310 columns wide, and a range multiplies columns, not rows.

Output is one cp949 CSV per month under $KOREAN_DATA_ROOT/raw/regpop/, plus
manifest.json holding sha256, shape and the national total of every file. Files
that already verify are skipped, so re-running costs nothing and a partial run
can be resumed.

    python eda/dl_regpop.py              # fetch what is missing, verify all
    python eda/dl_regpop.py --verify     # verify what is on disk, fetch nothing
    python eda/dl_regpop.py --force      # re-fetch everything

Verification is deliberately about the *content*, not the transfer: a Korean
government portal answering with a session-timeout HTML page carries a 200 and a
Content-Disposition header just as happily as a real CSV does.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT  # noqa: E402

DEST = DATA_ROOT / "raw" / "regpop"
MANIFEST = DEST / "manifest.json"

BASE = "https://jumin.mois.go.kr"
PAGE = f"{BASE}/ageStatMonth.do"
ENDPOINT = f"{BASE}/downloadCsvAge.do?searchYearMonth=month&xlsStats=3"

FIRST_YM, LAST_YM = 202001, 202607

# 101 single-year columns (0..99, 100 이상) plus 총인구수 and 연령구간인구수,
# once for each of 계 / 남 / 여, plus the 행정구역 label column.
N_AGE_COLS = 101
N_COLS = 1 + 3 * (2 + N_AGE_COLS)
SEXES = ("계", "남", "여")

# Seoul is 25 gu and ~424-426 dong over the window; the national file is ~3,800
# rows. Both bounds are loose on purpose — they catch a truncated or wrong-scope
# file, not a boundary revision, which is Phase 1b's job to find and explain.
MIN_ROWS, MIN_SEOUL_ROWS = 3000, 400
# 17 시도 for most of the window, 16 from 2026-07 when 광주광역시 and 전라남도
# merged into 전남광주통합특별시 (code 1200000000). The count is therefore a
# range, not a constant; what has to hold is that the 시도 rows still partition
# the country, which the national-total check below tests. Any change in the
# count is reported by the run summary rather than silently accepted.
N_SIDO_RANGE = (16, 17)
CODE_RE = re.compile(r"\((\d{10})\)\s*$")
# 2020-01 national 주민등록: 51,849,861. The window's drift is well inside this.
NAT_TOTAL_RANGE = (50_000_000, 53_000_000)


def months(first=FIRST_YM, last=LAST_YM):
    y, m = divmod(first, 100)
    while y * 100 + m <= last:
        yield y * 100 + m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _form(ym):
    y, m = divmod(ym, 100)
    return {
        "sltOrgType": "1", "sltOrgLvl1": "A", "sltOrgLvl2": "",
        "gender": "gender", "sum": "sum", "sltUndefType": "",
        "searchYearStart": str(y), "searchMonthStart": f"{m:02d}",
        "searchYearEnd": str(y), "searchMonthEnd": f"{m:02d}",
        "sltOrderType": "1", "sltOrderValue": "ASC",
        "sltArgTypes": "1", "sltArgTypeA": "0", "sltArgTypeB": "100",
        "category": "month", "state": "3",
    }


def path_for(ym):
    return DEST / f"regpop_{ym}.csv"


class BadFile(Exception):
    """The bytes on disk are not the month we asked for."""


def verify(ym, raw):
    """Parse and check one month. Returns the facts worth recording.

    Raises BadFile with a reason rather than returning a flag, so a caller that
    forgets to look still fails.
    """
    try:
        text = raw.decode("cp949")
    except UnicodeDecodeError as e:
        raise BadFile(f"not cp949 ({e}); first bytes {raw[:60]!r}")
    if "<html" in text[:2000].lower():
        raise BadFile("server returned HTML, not a CSV (session or scope lost)")

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise BadFile("empty file")
    header, body = rows[0], [r for r in rows[1:] if r and r[0].strip()]

    if len(header) != N_COLS:
        raise BadFile(f"{len(header)} columns, expected {N_COLS}")
    if header[0].strip() != "행정구역":
        raise BadFile(f"first column is {header[0]!r}, expected 행정구역")

    y, m = divmod(ym, 100)
    tag = f"{y}년{m:02d}월"
    off = {s: 1 + i * (2 + N_AGE_COLS) for i, s in enumerate(SEXES)}
    for s in SEXES:
        want = f"{tag}_{s}_총인구수"
        got = header[off[s]].strip()
        if got != want:
            raise BadFile(f"column {off[s]} is {got!r}, expected {want!r} "
                          f"— the file is for a different month or layout")
    if not header[-1].strip().endswith("100세 이상"):
        raise BadFile(f"last column is {header[-1]!r}, expected the 100세 이상 bin")

    if len(body) < MIN_ROWS:
        raise BadFile(f"{len(body)} data rows, expected >= {MIN_ROWS} "
                      f"— looks like a 시군구 file, not 읍면동")

    def num(cell):
        cell = cell.strip().replace(",", "")
        return int(cell) if cell and cell != "-" else 0

    seoul = [r for r in body if r[0].startswith("서울특별시")]
    if len(seoul) < MIN_SEOUL_ROWS:
        raise BadFile(f"{len(seoul)} Seoul rows, expected >= {MIN_SEOUL_ROWS}")

    # Every label carries its MOIS 행정기관코드 in parentheses, and the code says
    # the level: 시도 is XX00000000, 시군구 XXXXX00000, 읍면동 the full ten digits.
    # Read the level off the code, never off the name — 세종특별자치시 has no 시군구
    # tier and 제주 has 행정시, so name shape does not determine depth.
    codes = [CODE_RE.search(r[0]) for r in body]
    if any(c is None for c in codes):
        bad = next(r[0] for r, c in zip(body, codes) if c is None)
        raise BadFile(f"row label carries no 10-digit code: {bad!r}")
    sido = [r for r, c in zip(body, codes) if c.group(1).endswith("00000000")]
    nat = sum(num(r[off["계"]]) for r in sido)
    if not N_SIDO_RANGE[0] <= len(sido) <= N_SIDO_RANGE[1]:
        raise BadFile(f"{len(sido)} 시도 rows, expected {N_SIDO_RANGE}")
    if not NAT_TOTAL_RANGE[0] <= nat <= NAT_TOTAL_RANGE[1]:
        raise BadFile(f"national total {nat:,} from {len(sido)} 시도 rows is "
                      f"outside {NAT_TOTAL_RANGE} — scope is wrong")
    dong = [r for r, c in zip(body, codes)
            if not c.group(1).endswith("00000") and c.group(1).startswith("11")]

    # 계 = 남 + 여, on every row. Cheap, and it catches a column-offset shift
    # that every check above would sail past.
    for r in body:
        if num(r[off["계"]]) != num(r[off["남"]]) + num(r[off["여"]]):
            raise BadFile(f"계 != 남+여 on row {r[0]!r}")

    # 총인구수 must equal the 101 single-year columns, or the age range asked for
    # (sltArgTypeA/B) did not survive the round trip.
    for s in SEXES:
        base = off[s]
        head = num(body[0][base])
        by_age = sum(num(c) for c in body[0][base + 2: base + 2 + N_AGE_COLS])
        if head != by_age:
            raise BadFile(f"{s}: 총인구수 {head:,} != sum of single years "
                          f"{by_age:,} on row {body[0][0]!r}")

    return {"ym": ym, "bytes": len(raw), "rows": len(body),
            "cols": len(header), "seoul_rows": len(seoul),
            "seoul_dong": len(dong), "n_sido": len(sido),
            "national_total": nat,
            "sha256": hashlib.sha256(raw).hexdigest()}


def fetch(ym, opener, tries=3):
    body = urllib.parse.urlencode(_form(ym), encoding="utf-8").encode()
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, method="POST")
            req.add_header("Referer", PAGE)
            req.add_header("Content-Type",
                           "application/x-www-form-urlencoded; charset=UTF-8")
            with opener.open(req, timeout=300) as r:
                raw = r.read()
            return verify(ym, raw), raw
        except (urllib.error.URLError, BadFile, TimeoutError, OSError) as e:
            last = e
            if attempt < tries:
                wait = 5 * attempt
                print(f"    attempt {attempt} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
    raise BadFile(f"{ym}: gave up after {tries} attempts — {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check the files already on disk; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download even months that already verify")
    ap.add_argument("--first", type=int, default=FIRST_YM)
    ap.add_argument("--last", type=int, default=LAST_YM)
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="seconds between requests (default 1.5)")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST.exists():
        manifest = {int(k): v for k, v in json.load(open(MANIFEST)).items()}

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
        urllib.request.HTTPRedirectHandler())
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                                        "Chrome/120.0 Safari/537.36")]
    if not args.verify:
        opener.open(PAGE, timeout=60).read()   # take a session cookie first

    wanted = list(months(args.first, args.last))
    ok, failed, fetched = [], [], 0
    for ym in wanted:
        p = path_for(ym)
        if p.exists() and not args.force:
            try:
                rec = verify(ym, p.read_bytes())
                manifest[ym] = rec
                ok.append(ym)
                continue
            except BadFile as e:
                print(f"  {ym} on disk is bad: {e}")
                if args.verify:
                    failed.append((ym, str(e)))
                    continue
        if args.verify:
            if not p.exists():
                failed.append((ym, "not downloaded"))
            continue

        print(f"  {ym} downloading ...", flush=True)
        try:
            rec, raw = fetch(ym, opener)
        except BadFile as e:
            print(f"  {ym} FAILED: {e}")
            failed.append((ym, str(e)))
            continue
        p.write_bytes(raw)
        manifest[ym] = rec
        ok.append(ym)
        fetched += 1
        print(f"  {ym} ok  {rec['bytes']:>9,} B  {rec['rows']:>5} rows  "
              f"{rec['seoul_dong']:>4} Seoul dong  national {rec['national_total']:,}")
        time.sleep(args.sleep)

    with open(MANIFEST, "w") as fh:
        json.dump({str(k): manifest[k] for k in sorted(manifest)}, fh,
                  indent=1, ensure_ascii=False)

    print(f"\n{len(ok)}/{len(wanted)} months verified"
          f"{f', {fetched} newly downloaded' if fetched else ''}")
    if ok:
        counts = sorted({manifest[y]["seoul_dong"] for y in ok})
        print(f"  Seoul dong counts across the window: {counts}"
              f"{'  <- boundary revisions, Phase 1b' if len(counts) > 1 else ''}")
        sido = sorted({manifest[y]["n_sido"] for y in ok})
        if len(sido) > 1:
            first = {n: min(y for y in ok if manifest[y]["n_sido"] == n)
                     for n in sido}
            print(f"  시도 counts across the window: {sido}"
                  f"  (first month at each: {first})")
        dupes = {}
        for y in ok:
            dupes.setdefault(manifest[y]["sha256"], []).append(y)
        for h, ys in dupes.items():
            if len(ys) > 1:
                print(f"  identical bytes for {ys} — the site ignored the month")
                failed.append((ys[1], f"duplicate of {ys[0]}"))
    if failed:
        print(f"\n{len(failed)} MISSING OR BAD:")
        for ym, why in failed:
            print(f"  {ym}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
