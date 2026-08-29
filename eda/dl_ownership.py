#!/usr/bin/env python
"""L4 — pull the two national smartphone-ownership surveys that bound kappa.

rho-hat = kappa * lambda. Only lambda is behaviour; kappa is the share of
residents observable-and-moving in KT's data, and it is bounded from outside
this project by how smartphone ownership drifted. `../archive/letter_to_advisor.md` bounds
it with two independent national surveys whose published waves stopped at 2021,
and whose top age band is the open-ended 70세 이상. Route R2 (saturation
anchoring) needs the later waves. This script fetches them.

    A. 과기정통부 · NIA 「인터넷이용실태조사」 — the 통계표 (statistical tables)
       volume, from nia.or.kr. The quantity is 스마트폰 **보유율** (ownership),
       read off 「휴대형 정보통신기기 보유현황(복수응답)」, BASE 만6세 이상 전체,
       PERIOD 단시점 7.1 기준. The same volume also carries 스마트폰 **이용률**
       (「최근 1개월 이내 스마트폰 이용여부」) — a different quantity. p20d reads
       only the ownership table; do not substitute one for the other.

    B. 방송통신위원회(현 방송미디어통신위원회) 「방송매체 이용행태조사」 — the
       annual 보고서, from kmcc.go.kr. The quantity is 스마트폰 **보유**, read
       off the 개인 설문 통계표 item 「[문1-2] 스마트폰 보유」. Also ownership.

Both are PDF-only at source; neither agency publishes these tables as a
spreadsheet. Some waves are scanned images with no text layer at all — those
are recorded as such rather than guessed at, because a wrong ownership number
moves the kappa bound and therefore changes which behavioural claims survive.

Output is one PDF per (source, year, language) under
$KOREAN_DATA_ROOT/raw/ownership/, plus manifest.json holding sha256, byte
count, page count, text-layer density and the marker strings found in each
file. Files that already verify are skipped, so re-running costs nothing.

    python eda/dl_ownership.py              # fetch what is missing, verify all
    python eda/dl_ownership.py --verify     # verify what is on disk, fetch nothing
    python eda/dl_ownership.py --force      # re-fetch everything

Verification is about the *content*. Three failure modes actually observed
while building this, none of which an HTTP status would have caught:

  * kmcc.go.kr closes the connection mid-body and still leaves a 200 in the
    access log; the PDF is truncated and pdfinfo reports zero pages. The
    trailing `%%EOF` check catches it.
  * nia.or.kr answers a request whose bcIdx did not survive the round trip
    with an HTML error page carrying `Content-Type: application/octet-stream`.
    The `%PDF` magic check catches it.
  * Several waves are 240-page scans with ~2 characters of extractable text
    per page. Nothing about the transfer is wrong; the file simply cannot be
    parsed. `text_layer=False` records it as a named gap.
"""
import argparse
import hashlib
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

DEST = DATA_ROOT / "raw" / "ownership"
MANIFEST = DEST / "manifest.json"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

NIA_BASE = "https://www.nia.or.kr"
NIA_BOARD = 99870                      # 지식정보 > 통계·실태조사 > 인터넷이용실태조사
KCC_BASE = "https://www.kmcc.go.kr"

# ---------------------------------------------------------------------------
# What to fetch.
#
# `bcIdx`/`fileNo` are the NIA board post and its attachment index; `fileSeq`
# is the kmcc attachment id. Both were read off the board pages rather than
# guessed, and both are stable identifiers on those portals.
#
# `lang` matters more than it looks. NIA published 2018-2020, 2024 and 2025 as
# scans in Korean, but the 2024 and 2025 *English* editions of the very same
# 통계표 carry a full text layer. Those two are therefore not redundant copies
# — they are the only machine-readable form of those waves. The English tables
# are a translation of the identical survey output, so a number read from them
# is the same number, but p20d records which language each value came from.
# ---------------------------------------------------------------------------
NIA_FILES = [
    # (year, bcIdx, fileNo, lang, note)
    (2018, 21014, 1, "ko", "2018_인터넷이용실태조사_국문_통계표.pdf"),
    (2018, 21014, 2, "en", "2018_인터넷이용실태조사_통계표(영문).pdf"),
    (2019, 22082, 1, "ko", "인터넷실태조사_통계표(최종).pdf"),
    (2019, 22082, 2, "en", "2019_인터넷이용실태조사_통계표(영문).pdf"),
    (2020, 23270, 3, "ko", "2020년도_인터넷이용실태조사_통계표(최종).pdf"),
    (2020, 23270, 4, "en", "2020년도_인터넷이용실태조사_통계표(영문).pdf"),
    (2021, 24456, 3, "ko", "2021년_인터넷이용실태조사_통계표_국문(최종).pdf"),
    (2022, 25604, 3, "ko", "2022년_인터넷이용실태조사_통계표(국문).pdf"),
    (2023, 26741, 3, "ko", "2023인터넷이용실태조사_통계표(국문).pdf"),
    (2024, 27870, 3, "ko", "7._2024_인터넷이용실태조사_통계표_수정_250902.pdf"),
    (2024, 28317, 4, "en", "3._2024_인터넷이용실태조사_통계표(영문)_수정_250828.pdf"),
    (2025, 29199, 3, "ko", "2025_인터넷이용실태조사_통계표_수정_260512.pdf"),
    (2025, 29580, 2, "en", "2025_인터넷이용실태조사_통계표(영문).pdf"),
]

# The 최종보고서 (main report) volumes of the same survey, same board.
#
# These are here for one reason: NIA published the 2018, 2019 and 2020 통계표
# as scans, so the three waves the letter's kappa bound actually rests on are
# unreadable in their primary form. The report volume carries the same numbers
# in its narrative and in 「성·연령별 스마트폰 보유율」, and each edition plots the
# current year *against the previous one*, so one readable report recovers two
# waves. The 2018 Korean report is a text PDF; p20d reads 2017 and 2018 out of
# it. Editions that turn out to be scans are recorded as such and skipped.
#
# A number from here is the same published statistic, but it is read off a
# chart and a sentence rather than a table cell, so p20d tags it with a
# different `route` and cross-checks the two against each other before
# emitting it. Never silently merged with the 통계표 route.
NIA_REPORT_FILES = [
    # (year, bcIdx, fileNo, lang, note)
    (2018, 21013, 2, "ko", "2018_인터넷이용실태조사_국문_최종보고서.pdf"),
    (2019, 21930, 7, "ko", "2019_인터넷이용실태조사_본보고서(최종).pdf"),
    (2020, 23213, 5, "ko", "2020년도_인터넷이용실태조사_보고서(최종).pdf"),
    (2021, 24378, 4, "ko", "2021년_인터넷이용실태조사_보고서_국문(최종).pdf"),
    (2022, 25521, 2, "ko", "2022년_인터넷이용실태조사_보고서(국문).pdf"),
]

# 방송통계 board (boardId=1027). 2018 has no post: the board jumps straight
# from 2017 (boardSeq 45461) to 2019 (48358). That is a real gap in the
# publisher's own archive, not a failed download — see the memo.
KCC_FILES = [
    # (year, boardSeq, fileSeq)
    (2017, 45461, 46073),
    (2019, 48358, 49639),
    (2020, 50589, 51187),
    (2021, 52581, 53087),
    (2022, 54472, 55464),
    (2023, 59157, 57564),
    (2024, 65111, 60258),
    (2025, 67915, 62435),
]

# Loose on purpose: these catch a truncated or wrong-scope file, not an
# edition that grew by a few pages.
NIA_PAGES = (150, 450)
NIAREP_PAGES = (100, 400)
KCC_PAGES = (350, 800)
PAGE_RANGE = {"nia": NIA_PAGES, "niarep": NIAREP_PAGES, "kcc": KCC_PAGES}

# Marker strings that must appear in a file that has a text layer at all.
# They are the table titles p20d navigates by, so a file that verifies here is
# a file p20d can actually parse.
NIA_MARKERS = {
    "ko": ["휴대형 정보통신기기 보유현황", "만6세 이상 전체"],
    "en": ["Portable ICT device ownership", "Entire population aged 6 or older"],
}
KCC_MARKERS = ["스마트폰 보유", "개인 설문 통계표"]
# The report volume's section heading and the figure p20d reads.
NIAREP_MARKERS = ["스마트폰 보유율", "스마트폰"]

# A page averaging fewer than this many extractable characters is a scan.
# Observed: parseable waves run 1,500-1,600 chars/page; scans run 0-31.
TEXT_LAYER_MIN_CPP = 200


class BadFile(Exception):
    """The bytes on disk are not the document we asked for."""


def spec_key(source, year, lang):
    return f"{source}_{year}_{lang}"


def path_for(source, year, lang):
    return DEST / f"{spec_key(source, year, lang)}.pdf"


def _pdf_facts(path, sample_pages=24):
    """Page count and text density, without reading every page of a 146 MB scan.

    Returns (n_pages, chars_per_page, sampled_text). The text is sampled from
    pages spread through the document rather than the first N, because the
    front matter of these volumes is a table of contents that carries text even
    when every table page is an image — which is exactly how the 2018 Korean
    edition fools a naive check.
    """
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        n = len(pdf.pages)
        if n == 0:
            raise BadFile("0 pages")
        step = max(1, n // sample_pages)
        idx = list(range(0, n, step))[:sample_pages]
        chunks = []
        for i in idx:
            try:
                chunks.append(pdf.pages[i].extract_text() or "")
            except Exception as e:              # noqa: BLE001 - a broken page
                chunks.append("")               # is data, not a crash
                if os.environ.get("DL_OWNERSHIP_DEBUG"):
                    print(f"      page {i}: {e}")
        text = "\n".join(chunks)
    cpp = len(re.sub(r"\s", "", text)) / max(1, len(idx))
    return n, cpp, text


def verify(spec, raw, path):
    """Check one downloaded file. Raises BadFile with a reason, never a flag."""
    source, year, lang = spec["source"], spec["year"], spec["lang"]

    if not raw.startswith(b"%PDF"):
        head = raw[:200]
        if b"<html" in head.lower() or b"<!DOCTYPE" in head:
            raise BadFile("server returned HTML, not a PDF "
                          "(session lost, or the post id did not survive)")
        raise BadFile(f"not a PDF; first bytes {head[:60]!r}")

    # Truncation. kmcc.go.kr does this often enough that it is the single most
    # important check in this file.
    if b"%%EOF" not in raw[-4096:]:
        raise BadFile(f"no %%EOF in the last 4 KB of {len(raw):,} bytes "
                      f"— the transfer was cut short")

    n_pages, cpp, text = _pdf_facts(path)
    lo, hi = PAGE_RANGE[source]
    if not lo <= n_pages <= hi:
        raise BadFile(f"{n_pages} pages, expected {lo}-{hi}")

    text_layer = cpp >= TEXT_LAYER_MIN_CPP
    markers = {"nia": NIA_MARKERS.get(lang, []),
               "niarep": NIAREP_MARKERS,
               "kcc": KCC_MARKERS}[source]
    found = [m for m in markers if m in text]
    if text_layer and not found:
        raise BadFile(
            f"{cpp:.0f} chars/page of text but none of the expected markers "
            f"{markers} — this is not the 통계표/보고서 volume we asked for")

    return {
        "source": source, "year": year, "lang": lang,
        "bytes": len(raw), "pages": n_pages,
        "chars_per_page": round(cpp, 1),
        "text_layer": text_layer,
        "markers_found": found,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "url": spec["url"], "referer": spec["referer"],
        "published_filename": spec.get("note", ""),
    }


def build_specs():
    specs = []
    for year, bc, fno, lang, note in NIA_FILES:
        specs.append({
            "source": "nia", "year": year, "lang": lang, "note": note,
            "url": (f"{NIA_BASE}/common/board/Download.do"
                    f"?bcIdx={bc}&cbIdx={NIA_BOARD}&fileNo={fno}"),
            "referer": (f"{NIA_BASE}/site/nia_kor/ex/bbs/View.do"
                        f"?cbIdx={NIA_BOARD}&bcIdx={bc}&parentSeq={bc}"),
        })
    for year, bc, fno, lang, note in NIA_REPORT_FILES:
        specs.append({
            "source": "niarep", "year": year, "lang": lang, "note": note,
            "url": (f"{NIA_BASE}/common/board/Download.do"
                    f"?bcIdx={bc}&cbIdx={NIA_BOARD}&fileNo={fno}"),
            "referer": (f"{NIA_BASE}/site/nia_kor/ex/bbs/View.do"
                        f"?cbIdx={NIA_BOARD}&bcIdx={bc}&parentSeq={bc}"),
        })
    for year, seq, fs in KCC_FILES:
        specs.append({
            "source": "kcc", "year": year, "lang": "ko",
            "note": f"{year} 방송매체 이용행태 조사 보고서.pdf",
            "url": f"{KCC_BASE}/download.do?fileSeq={fs}",
            "referer": (f"{KCC_BASE}/user.do?mode=view&page=A02060100"
                        f"&dc=K02060100&boardId=1027&boardSeq={seq}"),
        })
    return specs


def fetch_one(spec, path, tries=4, timeout=900):
    """Download to `path`, verify, return the record.

    Deliberately no Range/resume: kmcc.go.kr ignores `Range` and answers a
    resume request with the whole body again, so `curl -C -` silently produces
    a file that is the document concatenated onto its own prefix. Every attempt
    here starts from zero.
    """
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(spec["url"], method="GET")
            req.add_header("User-Agent", UA)
            req.add_header("Referer", spec["referer"])
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            path.write_bytes(raw)
            return verify(spec, raw, path)
        except (urllib.error.URLError, BadFile, TimeoutError, OSError) as e:
            last = e
            if attempt < tries:
                wait = 5 * attempt
                print(f"    attempt {attempt} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
    raise BadFile(f"gave up after {tries} attempts — {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check the files already on disk; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download even files that already verify")
    ap.add_argument("--source", choices=["nia", "niarep", "kcc"],
                    help="restrict to one document set")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between requests (default 2.0)")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if MANIFEST.exists() else {}

    specs = [s for s in build_specs()
             if not args.source or s["source"] == args.source]
    ok, failed, fetched = [], [], 0

    for spec in specs:
        key = spec_key(spec["source"], spec["year"], spec["lang"])
        p = path_for(spec["source"], spec["year"], spec["lang"])
        if p.exists() and not args.force:
            try:
                manifest[key] = verify(spec, p.read_bytes(), p)
                ok.append(key)
                continue
            except BadFile as e:
                print(f"  {key} on disk is bad: {e}")
                if args.verify:
                    failed.append((key, str(e)))
                    continue
        if args.verify:
            if not p.exists():
                failed.append((key, "not downloaded"))
            continue

        print(f"  {key} downloading ...", flush=True)
        try:
            rec = fetch_one(spec, p)
        except BadFile as e:
            print(f"  {key} FAILED: {e}")
            failed.append((key, str(e)))
            p.unlink(missing_ok=True)
            continue
        manifest[key] = rec
        ok.append(key)
        fetched += 1
        print(f"  {key} ok  {rec['bytes']:>11,} B  {rec['pages']:>4} pages  "
              f"{rec['chars_per_page']:>7.1f} chars/page  "
              f"{'text' if rec['text_layer'] else 'SCAN (no text layer)'}")
        time.sleep(args.sleep)

    with open(MANIFEST, "w") as fh:
        json.dump({k: manifest[k] for k in sorted(manifest)}, fh,
                  indent=1, ensure_ascii=False)

    print(f"\n{len(ok)}/{len(specs)} files verified"
          f"{f', {fetched} newly downloaded' if fetched else ''}")

    scans = [k for k in ok if not manifest[k]["text_layer"]]
    if scans:
        print(f"\n{len(scans)} file(s) are image-only scans — no text to parse.")
        for k in scans:
            print(f"  {k}: {manifest[k]['chars_per_page']:.1f} chars/page")
        print("  These are named gaps, not download failures. p20d skips them.")

    # A wave is recoverable if ANY edition of it has a text layer — either
    # language of the 통계표, or the 최종보고서 that carries the same figure.
    # 'nia' and 'niarep' are the same survey, so they collapse to one wave.
    survey_of = {"nia": "nia", "niarep": "nia", "kcc": "kcc"}
    if args.source:
        # With a filter on, "no parseable edition" would mean "none among the
        # subset I just looked at", which reads as a much worse finding than
        # it is. Only report it over the whole set.
        return 1 if failed else 0
    parseable = {}
    for k in ok:
        r = manifest[k]
        parseable.setdefault((survey_of[r["source"]], r["year"]), []).append(
            (k, r["text_layer"]))
    dead = sorted(sy for sy, v in parseable.items()
                  if not any(t for _, t in v))
    if dead:
        print(f"\n{len(dead)} wave(s) have NO parseable edition at all:")
        for s, y in dead:
            print(f"  {s} {y}: " +
                  ", ".join(k for k, _ in parseable[(s, y)]))
    recovered = sorted(sy for sy, v in parseable.items()
                       if any(t for k, t in v if k.startswith("niarep"))
                       and not any(t for k, t in v if k.startswith("nia_")))
    if recovered:
        print(f"\n{len(recovered)} wave(s) readable ONLY via the 최종보고서:")
        for s, y in recovered:
            print(f"  {s} {y}")

    if failed:
        print(f"\n{len(failed)} MISSING OR BAD:")
        for k, why in failed:
            print(f"  {k}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
