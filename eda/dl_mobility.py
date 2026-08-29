#!/usr/bin/env python
"""Pull 서울 생활이동 인구 데이터 (행정동 단위) monthly archives from data.seoul.go.kr.

One month = one zip of 24 cp949 CSVs (one per arrival hour), ~1.07 GB compressed
and ~4.5 GB expanded. The 2020 tree on disk was fetched by hand; this script is
the scripted version, so the remaining months of 2020-01..2026-07 can be pulled
the same way.

The site has no API. `seoulLivingMigration.do` renders a month grid whose buttons
call `downloadFile(infSeq, seq)`; that is three requests, and this script makes
the same three:

    seoulLivingYear.do          POST infSeq, year   -> the months that exist
    getSeoulLivingDownloadYN.do POST infSeq, seq    -> whether one is servable
    nio_download.do             POST infId=DOWNLOAD, infSeq, seq  -> the bytes

`infSeq=2` is the 행정동 (dong) product this project uses; `infSeq=3` is the
coarser 자치구 (gu) one. `seq` is just YYYYMM.

Verification is deliberately about the *content*, not the transfer, exactly as in
`dl_regpop.py`: a Korean government portal answering with a session-timeout HTML
page carries a 200 and a Content-Disposition header just as happily as a real
archive does. So a month counts as present only when the zip opens, holds 24
members named for the 24 hours of the month asked for, and every extracted CSV
decodes as cp949 with the 10-column header and the right 대상연월 in its first
data row. The server does not honour Range, so a broken transfer is retried from
zero rather than resumed.

    python eda/dl_mobility.py --list                   # what exists, how big
    python eda/dl_mobility.py --months 202606,202607   # fetch + verify + expand
    python eda/dl_mobility.py --verify                 # check disk, fetch nothing
    python eda/dl_mobility.py --months 202607 --etl    # ... then write parquet

Files land as $KOREAN_DATA_ROOT/생활이동_행정동_YYYYMM/*.csv, matching the 2020
tree, so `eda/etl.py`'s glob picks them up unchanged. The zip is kept next to
them under raw/mobility/ unless --delete-zip is given; its sha256 is recorded
either way in raw/mobility/manifest.json.
"""
import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT  # noqa: E402

ZIPDIR = DATA_ROOT / "raw" / "mobility"
MANIFEST = ZIPDIR / "manifest.json"

SITE = "https://data.seoul.go.kr"
PAGE = f"{SITE}/dataVisual/seoul/seoulLivingMigration.do"
LIST = f"{SITE}/dataVisual/seoul/seoulLivingYear.do"
READY = f"{SITE}/dataVisual/seoul/getSeoulLivingDownloadYN.do"
FILE = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"

INF_SEQ = 2                     # 2 = 행정동 단위, 3 = 자치구 단위
FIRST_YM, LAST_YM = 202001, 202607

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# The 10 columns of the product, in order. Checked verbatim against every file:
# a renamed or reordered column has to stop the ingest, not flow into parquet
# under the old name.
HEADER = ["대상연월", "요일", "도착시간", "출발 행정동 코드", "도착 행정동 코드",
          "성별", "나이", "이동유형", "평균 이동 시간(분)", "이동인구(합)"]
N_COLS = len(HEADER)
N_HOURS = 24
MEMBER_RE = re.compile(r"생활이동_행정동_(\d{4})\.(\d{2})_(\d{2})시\.csv$")

# One month of the dong product is 0.9-1.3 GB zipped. Anything far outside that
# is a landing page, an error stub, or a different product.
ZIP_BYTES_RANGE = (500_000_000, 3_000_000_000)

# See download() for why this is not 5. Overridable with --tries.
DOWNLOAD_TRIES = 15


def months(first=FIRST_YM, last=LAST_YM):
    y, m = divmod(first, 100)
    while y * 100 + m <= last:
        yield y * 100 + m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def zip_path(ym):
    return ZIPDIR / f"생활이동_행정동_{ym}.zip"


def csv_dir(ym):
    return DATA_ROOT / f"생활이동_행정동_{ym}"


class BadFile(Exception):
    """The bytes on disk are not the month we asked for."""


def opener():
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
        urllib.request.HTTPRedirectHandler())
    op.addheaders = [("User-Agent", UA), ("Referer", PAGE)]
    return op


def _retry(what, fn, tries=5, base=10):
    """Retry a small HTTP call with linear backoff.

    The portal is intermittently slow enough to time out mid-body even when it
    answered the handshake in seconds, and the bulk download already retries. The
    small calls did not, so a single slow read on the session-cookie fetch could
    kill a multi-gigabyte job before it started. Everything retried here is
    idempotent: a cookie fetch and a month listing.
    """
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as e:
            last = e
            if attempt < tries:
                wait = base * attempt
                print(f"    {what}: attempt {attempt} failed ({type(e).__name__}: "
                      f"{e}); retrying in {wait}s", flush=True)
                time.sleep(wait)
    raise BadFile(f"{what}: gave up after {tries} attempts — {last}")


def _post(op, url, form, timeout=60):
    body = urllib.parse.urlencode(form, encoding="utf-8").encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type",
                   "application/x-www-form-urlencoded; charset=UTF-8")
    req.add_header("X-Requested-With", "XMLHttpRequest")
    return op.open(req, timeout=timeout)


def available(op, years):
    """The months the site says it has, per year, as a sorted list of ints."""
    out = []
    for y in years:
        with _post(op, LIST, {"infSeq": str(INF_SEQ), "year": str(y)}) as r:
            payload = json.loads(r.read().decode("utf-8"))
        for row in payload.get("result") or []:
            if row:
                out.append(int(row["seq"]))
    return sorted(out)


def servable(op, ym):
    with _post(op, READY, {"infSeq": str(INF_SEQ), "seq": str(ym)}) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return bool(payload.get("result"))


def probe(op, ym):
    """Content-Length and filename without pulling the body.

    There is no HEAD on this endpoint, so the request is a normal POST whose
    response is closed after the headers arrive. Costs one connection.
    """
    r = _post(op, FILE, {"infId": "DOWNLOAD", "infSeq": str(INF_SEQ),
                         "seqNo": "", "seq": str(ym)}, timeout=120)
    try:
        size = int(r.headers.get("Content-Length") or 0)
        disp = urllib.parse.unquote(r.headers.get("Content-Disposition") or "")
    finally:
        r.close()
    name = re.search(r'filename="?([^";]+)', disp)
    return size, (name.group(1) if name else "")


def download(op, ym, dest, tries=DOWNLOAD_TRIES):
    """Stream one month's zip to `dest`. Returns (bytes, sha256).

    The server ignores Range -- a `Range: bytes=N-` request comes back 200 with
    the whole file and no Content-Range -- so there is nothing to resume: a
    truncated transfer is thrown away and re-requested.

    WHY THE RETRY BUDGET IS LARGE. The portal does not fail cleanly under load.
    It degrades to a few hundred kB/s, then starts dropping the TLS connection
    outright (`SSL: RECORD_LAYER_FAILURE`), sometimes after 300 MB of a 750 MB
    body. But the failure is intermittent, not a ban: a probe answers in under a
    second throughout, and a retry a minute later can run the whole file at
    3.5 MB/s. Five attempts with linear backoff gave up inside four minutes and
    threw away eleven months of progress; the cost of an extra attempt is a few
    minutes, and the cost of giving up early is a re-run of the whole month.

    Each attempt opens a FRESH session. The dropped connections leave the old
    cookie jar in a state the server keeps refusing, so reusing the opener turns
    one bad transfer into a run of them.
    """
    last = None
    for attempt in range(1, tries + 1):
        tmp = dest.with_suffix(".part")
        try:
            if attempt > 1:
                op = opener()
                _retry("session cookie", lambda: op.open(PAGE, timeout=120).read(),
                       tries=3)
            r = _post(op, FILE, {"infId": "DOWNLOAD", "infSeq": str(INF_SEQ),
                                 "seqNo": "", "seq": str(ym)}, timeout=300)
            expect = int(r.headers.get("Content-Length") or 0)
            h, got, t0, tick = hashlib.sha256(), 0, time.time(), 0
            with r, open(tmp, "wb") as fh:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    if got == 0 and buf[:4] != b"PK\x03\x04":
                        raise BadFile(f"response does not start with a zip "
                                      f"header: {buf[:80]!r}")
                    fh.write(buf)
                    h.update(buf)
                    got += len(buf)
                    if time.time() - tick > 20:
                        tick = time.time()
                        rate = got / max(time.time() - t0, 1e-9) / 1e6
                        print(f"      {got/1e9:5.2f}/{expect/1e9:.2f} GB "
                              f"{rate:5.1f} MB/s", flush=True)
            if expect and got != expect:
                raise BadFile(f"truncated: {got:,} of {expect:,} bytes")
            if not ZIP_BYTES_RANGE[0] <= got <= ZIP_BYTES_RANGE[1]:
                raise BadFile(f"{got:,} bytes is outside {ZIP_BYTES_RANGE} "
                              f"— wrong product or an error page")
            os.replace(tmp, dest)
            return got, h.hexdigest()
        except (urllib.error.URLError, BadFile, TimeoutError, OSError) as e:
            last = e
            tmp.unlink(missing_ok=True)
            if attempt < tries:
                # linear to a minute, then flat: the server recovers on its own
                # schedule and doubling past a couple of minutes only wastes the
                # window in which it is willing to serve again.
                wait = min(15 * attempt, 120)
                print(f"    attempt {attempt}/{tries} failed ({e}); "
                      f"retrying in {wait}s", flush=True)
                time.sleep(wait)
    raise BadFile(f"{ym}: gave up after {tries} attempts — {last}")


def check_zip(ym, path):
    """The archive holds the 24 hours of the month we asked for, and no more."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"PK\x03\x04":
            fh.seek(0)
            raise BadFile(f"not a zip; first bytes {fh.read(80)!r}")
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise BadFile(f"unreadable zip: {e}")
    with zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        hours = {}
        for n in names:
            m = MEMBER_RE.search(os.path.basename(n))
            if not m:
                raise BadFile(f"unexpected member {n!r}")
            if int(m.group(1) + m.group(2)) != ym:
                raise BadFile(f"member {n!r} is not month {ym}")
            hours[m.group(3)] = n
        want = {f"{h:02d}" for h in range(N_HOURS)}
        if set(hours) != want:
            raise BadFile(f"hours {sorted(set(hours))} != 00..23")
        if len(names) != N_HOURS:
            raise BadFile(f"{len(names)} members, expected {N_HOURS}")
        expanded = sum(i.file_size for i in zf.infolist())
    return hours, expanded


def check_csv(ym, path):
    """Header, first data row and row count of one extracted hour file.

    Reads the head as cp949 (a switch to UTF-8 upstream must fail loudly, not
    land as mojibake) and takes the row count from `wc -l`, which is the number
    `etl.py` asserts its parquet against.
    """
    with open(path, "rb") as fh:
        head = fh.read(1 << 16)
    # Cut the block back to the last newline before decoding: a fixed-size read
    # lands mid-character often enough that decoding the raw block reports a
    # perfectly good cp949 file as broken. 0x0A is never a cp949 trail byte
    # (trail bytes are 0x41-0x5A, 0x61-0x7A, 0x81-0xFE), so a newline is always
    # a safe boundary.
    cut = head.rfind(b"\n")
    head = head[:cut + 1] if cut > 0 else head
    try:
        text = head.decode("cp949")
    except UnicodeDecodeError as e:
        raise BadFile(f"{path.name}: not cp949 ({e}); first bytes {head[:60]!r}")
    if "<html" in text[:2000].lower():
        raise BadFile(f"{path.name}: HTML, not a CSV")
    lines = text.splitlines()
    if len(lines) < 2:
        raise BadFile(f"{path.name}: fewer than two lines")

    header = [c.strip().strip('"') for c in lines[0].split(",")]
    if header != HEADER:
        raise BadFile(f"{path.name}: header is {header!r}, expected {HEADER!r}")

    row = lines[1].split(",")
    if len(row) != N_COLS:
        raise BadFile(f"{path.name}: first data row has {len(row)} fields")
    if row[0].strip() != str(ym):
        raise BadFile(f"{path.name}: 대상연월 {row[0]!r}, expected {ym}")
    hour = int(MEMBER_RE.search(path.name).group(3))
    if int(row[2]) != hour:
        raise BadFile(f"{path.name}: 도착시간 {row[2]!r}, expected {hour}")
    if row[5].strip() not in ("F", "M"):
        raise BadFile(f"{path.name}: 성별 {row[5]!r}")
    pop = row[9].strip()
    if pop != "*":
        try:
            float(pop)
        except ValueError:
            raise BadFile(f"{path.name}: 이동인구 {pop!r} is neither * nor a number")

    n = int(subprocess.run(["wc", "-l", str(path)], capture_output=True,
                           text=True).stdout.split()[0]) - 1
    if n < 100_000:
        raise BadFile(f"{path.name}: {n} data rows — implausibly short")
    return {"hour": hour, "bytes": path.stat().st_size, "rows": n,
            "header": ",".join(header)}


def verify_month(ym):
    """Everything on disk for one month. Raises BadFile, never returns a flag."""
    d = csv_dir(ym)
    if not d.is_dir():
        raise BadFile("not extracted")
    files = sorted(p for p in d.iterdir() if p.suffix == ".csv")
    if len(files) != N_HOURS:
        raise BadFile(f"{len(files)} CSVs in {d.name}, expected {N_HOURS}")
    per_hour = [check_csv(ym, p) for p in files]
    if sorted(f["hour"] for f in per_hour) != list(range(N_HOURS)):
        raise BadFile(f"hours on disk are {[f['hour'] for f in per_hour]}")
    heads = {f["header"] for f in per_hour}
    if len(heads) != 1:
        raise BadFile(f"the 24 files disagree on the header: {heads}")
    return {"ym": ym, "csv_dir": str(d), "files": N_HOURS,
            "csv_bytes": sum(f["bytes"] for f in per_hour),
            "rows": sum(f["rows"] for f in per_hour),
            "header": per_hour[0]["header"],
            "per_hour": {f"{f['hour']:02d}": {"rows": f["rows"],
                                              "bytes": f["bytes"]}
                         for f in per_hour}}


def extract(ym, zpath, hours):
    """Expand the 24 members, flat, into 생활이동_행정동_YYYYMM/.

    zipfile checks each member's CRC as it is read, so a silently corrupted
    archive fails here rather than in the ETL.
    """
    d = csv_dir(ym)
    d.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(d).free
    with zipfile.ZipFile(zpath) as zf:
        need = sum(i.file_size for i in zf.infolist())
        if free < need * 1.1:
            raise BadFile(f"{free/1e9:.1f} GB free, need {need/1e9:.1f} GB")
        for hh in sorted(hours):
            member = hours[hh]
            dst = d / os.path.basename(member)
            with zf.open(member) as src, open(dst, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 22)
    return d


def run_etl(ym):
    """Hand the month's 24 CSVs to eda/etl.py, unchanged, one file at a time.

    Imported rather than shelled out because etl.py's `__main__` globs every
    month on the drive; `convert()` itself is per-file, already does the wc -l
    reconciliation, and is the one place the parquet schema is defined. Six
    workers, the same width etl.py uses.
    """
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    from etl import convert                             # noqa: PLC0415
    d = csv_dir(ym)
    files = sorted(str(p) for p in d.iterdir() if p.suffix == ".csv")
    out = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(convert, files):
            out.append(r)
            print(f"    {r['ym']} {r['hour']}시  rows={r['rows']:>10,}  "
                  f"masked={r['masked']:>9,} ({r['masked']/r['rows']:.1%})  "
                  f"bad_dow={r['bad_dow']}  {r['secs']}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--months", default="",
                    help="comma-separated YYYYMM; default is every month the "
                         "site lists in 2020-01..2026-07")
    ap.add_argument("--list", action="store_true",
                    help="report availability and size per month, download nothing")
    ap.add_argument("--verify", action="store_true",
                    help="check what is already on disk, download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download months that already verify")
    ap.add_argument("--delete-zip", action="store_true",
                    help="drop the archive once its CSVs verify (saves ~1 GB/month)")
    ap.add_argument("--etl", action="store_true",
                    help="run eda/etl.py on each month once it verifies")
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--tries", type=int, default=DOWNLOAD_TRIES,
                    help="attempts per month before giving up (see download())")
    args = ap.parse_args()

    ZIPDIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST.exists():
        manifest = {int(k): v for k, v in json.load(open(MANIFEST)).items()}

    wanted = ([int(x) for x in args.months.split(",") if x.strip()]
              or list(months()))

    if args.verify:
        ok, failed = [], []
        for ym in wanted:
            try:
                rec = verify_month(ym)
                manifest.setdefault(ym, {}).update(rec)
                ok.append(ym)
                print(f"  {ym} ok  {rec['files']} files  {rec['rows']:>12,} rows  "
                      f"{rec['csv_bytes']/1e9:5.2f} GB")
            except BadFile as e:
                failed.append((ym, str(e)))
        _save(manifest)
        return _summary(ok, failed, wanted, 0)

    op = opener()
    _retry("session cookie", lambda: op.open(PAGE, timeout=120).read())

    years = sorted({ym // 100 for ym in wanted})
    have = set(_retry("month listing", lambda: available(op, years)))
    missing = [ym for ym in wanted if ym not in have]
    if missing:
        print(f"not published: {missing}")
    wanted = [ym for ym in wanted if ym in have]

    if args.list:
        print(f"{'month':>8}  {'servable':>8}  {'zip bytes':>15}  filename")
        for ym in wanted:
            yn = servable(op, ym)
            size, name = probe(op, ym) if yn else (0, "")
            print(f"{ym:>8}  {str(yn):>8}  {size:>15,}  {name}")
            time.sleep(args.sleep)
        return 0

    ok, failed, fetched = [], [], 0
    for ym in wanted:
        if not args.force:
            try:
                rec = verify_month(ym)
                manifest.setdefault(ym, {}).update(rec)
                # A month whose CSVs are already here but whose archive was
                # never recorded (an interrupted earlier run) still gets its
                # checksum, so the manifest is rebuildable from the disk alone.
                z = zip_path(ym)
                if z.exists() and not manifest[ym].get("zip_sha256"):
                    manifest[ym].update(
                        zip_bytes=z.stat().st_size,
                        zip_sha256=hashlib.sha256(z.read_bytes()).hexdigest(),
                        zip_expanded_bytes=check_zip(ym, z)[1],
                        zip_kept=True)
                ok.append(ym)
                print(f"  {ym} already on disk  {rec['rows']:,} rows")
                _save(manifest)
                if args.etl:
                    run_etl(ym)
                continue
            except BadFile as e:
                if csv_dir(ym).exists():
                    print(f"  {ym} on disk is incomplete: {e}")

        try:
            if not servable(op, ym):
                raise BadFile("site reports the month as not downloadable")
            z = zip_path(ym)
            if z.exists() and not args.force:
                nbytes = z.stat().st_size
                sha = hashlib.sha256(z.read_bytes()).hexdigest()
                print(f"  {ym} zip already here  {nbytes/1e9:.2f} GB")
            else:
                size, name = probe(op, ym)
                print(f"  {ym} downloading {size/1e9:.2f} GB  ({name}) ...",
                      flush=True)
                nbytes, sha = download(op, ym, z, tries=args.tries)
                fetched += 1
            hours, expanded = check_zip(ym, z)
            print(f"    zip ok: 24 members, {expanded/1e9:.2f} GB expanded; "
                  f"extracting ...", flush=True)
            extract(ym, z, hours)
            rec = verify_month(ym)
            rec.update({"zip_bytes": nbytes, "zip_sha256": sha,
                        "zip_expanded_bytes": expanded,
                        "zip_kept": not args.delete_zip,
                        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())})
            if args.delete_zip:
                z.unlink()
            manifest[ym] = rec
            ok.append(ym)
            print(f"  {ym} ok  {rec['files']} files  {rec['rows']:,} rows  "
                  f"{rec['csv_bytes']/1e9:.2f} GB  sha256 {sha[:16]}")
            _save(manifest)
            if args.etl:
                run_etl(ym)
        except BadFile as e:
            print(f"  {ym} FAILED: {e}")
            failed.append((ym, str(e)))
        time.sleep(args.sleep)

    _save(manifest)
    return _summary(ok, failed, wanted, fetched)


def _save(manifest):
    with open(MANIFEST, "w") as fh:
        json.dump({str(k): manifest[k] for k in sorted(manifest)}, fh,
                  indent=1, ensure_ascii=False)


def _summary(ok, failed, wanted, fetched):
    print(f"\n{len(ok)}/{len(wanted)} months verified"
          f"{f', {fetched} newly downloaded' if fetched else ''}")
    if failed:
        print(f"{len(failed)} MISSING OR BAD:")
        for ym, why in failed:
            print(f"  {ym}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
