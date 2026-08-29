#!/usr/bin/env python
"""L3 — pull 등록외국인 (registered foreigners), the other half of the rho-hat
denominator. Seoul, 2020-01..2026-07.

주민등록 (dl_regpop.py) counts Korean nationals only. Registered foreigners are
~2.5-3% of Seoul but they are not spread evenly: they pile into a handful of
dong and they skew hard to 20-49, which is exactly the age band where the
mobility data's masking bites hardest. A denominator that omits them is wrong
in a spatially and demographically structured way, not by a flat 3%.

The plan (../archive/REVISION_PLAN.md, L3) expected three sources and warned that none of
them gives month x dong x age. Two of the three turned out to be dead ends for a
script, and the third source family — which the plan did not name — turns out to
give dong x age x sex directly. What this script pulls, and why:

  A  data.go.kr 15100022, 법무부 "월별 등록외국인 시군구별 거주 현황"
     month x 시군구, one number per cell. No age, no sex.
     VERIFIED: the file covers 2022-01..2026-06 only — its own internal name is
     "... (2022-2026).csv". The plan's window starts 2020-01, so this source
     alone leaves 24 months uncovered at the front and 2026-07 at the back.
     It is the only *monthly* source, so it sets the time axis and nothing else.

  B  moj.go.kr board 227 (출입국·외국인정책 통계월보 자료실), attachment #25 of
     each quarterly "등록외국인 지역별 현황" post:
     "등록외국인(읍면동, 연령별) 현황.xlsx"
     시도 x 시군구 x 행정동 x 성별(총계/남성/여성) x 17 five-year age bands.
     VERIFIED: attachment #25 first appears in the 2020년 12월말 post and runs
     quarterly to 2026년 6월말 — 23 quarters. The three 2020 quarters before it
     have only attachments #21-24, so the dong x age structure genuinely does
     not exist before 2020-12.

  C  the same posts, attachment #22, "등록외국인(지역별, 연령별) 현황.xlsx"
     시도 x 시군구 x 성별 x the same 17 age bands, quarterly, and it does go back
     before 2020-12. It is fetched only for the four quarters 2019-12..2020-09,
     where B does not exist: from 2020-12 on, the gu numbers are just B summed
     over dong, and taking them from B instead guarantees the two levels agree.

All three are the same 법무부 register, so they are the same universe: a person
who has been in Korea 91+ days and holds an 외국인등록번호. 단기체류 and
외국국적동포 거소신고 are outside all three; that gap is a property of the
denominator, not of the download, and it is written up in the memo.

Two sources the plan named that this script does NOT use, with the reason:

  - KOSIS DT_110025_A033_A "읍면동별 외국인주민현황" (행안부): kosis.kr redirects
    statHtml.do to sso.kosis.kr, so the table is not reachable without a login
    or an API key. It is also a *different universe* — 외국인주민 counts
    naturalised citizens and the Korean-born children of foreign residents,
    which are not what a 등록외국인 denominator wants — and it is annual, where
    source B is quarterly. Nothing is lost by not having it.
  - 서울 열린데이터광장 DT_201004_A020002 "등록외국인 현황(연령별/동별)":
    exists, quarterly 2014Q1-2026Q2, 440 dong x 22 age classes x 계/남자/여자.
    Its values sit behind an OLAP session on stat.eseoul.go.kr that would not
    drive from a script (statHtml/html.do answers every well-formed request with
    a generic error). It is a re-publication of the same 법무부 register that B
    already gives us at the same granularity, so it is a cross-check we can
    live without rather than a gap.

    python eda/dl_foreign.py            # fetch what is missing, verify all
    python eda/dl_foreign.py --verify   # verify what is on disk, fetch nothing
    python eda/dl_foreign.py --force    # re-fetch everything

Verification is about the *content*, not the transfer. data.go.kr answers a
lost session with a 200 and a Content-Disposition header; moj.go.kr answers a
cookie-less download with a 307 that redirects to itself forever. Neither is
detectable from the status line, so every file is parsed and checked against
what it claims to be before it is allowed to stay on disk.
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
import openpyxl  # noqa: E402
from paths import DATA_ROOT  # noqa: E402

DEST = DATA_ROOT / "raw" / "foreign"
MANIFEST = DEST / "manifest.json"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# --- source A -------------------------------------------------------------
DGK_PK = "15100022"
DGK_PAGE = f"https://www.data.go.kr/data/{DGK_PK}/fileData.do"
DGK_META = "https://www.data.go.kr/tcs/dss/selectFileDataDownload.do"
DGK_FILE = "https://www.data.go.kr/cmm/cmm/fileDownload.do"
# The uddi is the dataset's detail key; it is stable across the monthly
# re-publication (the atchFileId is not, so that one is looked up every run).
DGK_UDDI = "uddi:0918620b-57e4-4a99-9bc2-fe657c9f8b97"
A_NAME = "moj_gu_month.csv"
A_HEADER = ["년", "월", "시도", "시군구", "등록외국인수"]
A_MIN_ROWS = 12_000

# --- sources B and C ------------------------------------------------------
MOJ = "https://www.moj.go.kr"
BOARD = f"{MOJ}/bbs/immigration/227/artclList.do"
POST = f"{MOJ}/bbs/immigration/227/{{ntt}}/artclView.do"
DOWN = f"{MOJ}/bbs/immigration/227/{{fid}}/download.do"
TITLE_RE = re.compile(r"등록외국인 지역별 현황\s*\((\d{4})년\s*(\d{1,2})월말")
# 2019-12 is kept as the anchor immediately before the window: p20b interpolates
# between quarter-ends, and 2020-01/02 need a left anchor to interpolate from.
FIRST_Q, LAST_Q = 201912, 202606
# The quarter the 읍면동 attachment first appears. Before it, only C exists.
FIRST_DONG_Q = 202012

# Seoul is 25 gu. Both spreadsheets carry 총계 roll-up rows at every level, so
# the row counts are dominated by the whole country; these bounds only have to
# separate "a national table" from "a truncated or wrong file".
B_MIN_ROWS, C_MIN_ROWS = 6_000, 500
AGE_HDR = ["0~4", "5~9", "10~14", "15~19", "20~24", "25~29", "30~34", "35~39",
           "40~44", "45~49", "50~54", "55~59", "60~64", "65~69", "70~74",
           "75~79", "80+"]
SEXES = ("총계", "남성", "여성")
# The roll-up rows. Every level of the hierarchy is repeated as a subtotal row,
# and the label for it changed in 2026-06 from 총계 to 시도별총계 — reading only
# for 총계 would double count a whole 시도 into the dong level.
ROLLUP = {"총계", "총합계", "시도별총계", ""}
# 2026-06 also renamed the dong column from 행정동 to 읍면동.
DONG_COL = ("행정동", "읍면동")


class BadFile(Exception):
    """The bytes on disk are not the thing we asked for."""


def _norm_age(s):
    """'0세~4세', '0~4세', '80세이상', '80세 이상' -> a single canonical label.

    The two spreadsheets spell the same 17 bands differently, and the spelling
    drifts across quarters. Normalising here means the header check is a real
    check rather than a string-equality accident.
    """
    s = re.sub(r"\s+", "", str(s or ""))
    if "이상" in s:
        n = re.search(r"(\d+)", s)
        return f"{n.group(1)}+" if n else s
    n = re.findall(r"(\d+)", s)
    return f"{n[0]}~{n[1]}" if len(n) >= 2 else s


def kinds_for(ym):
    """Which spreadsheet a quarter needs.

    From 2020-12 the 읍면동 file exists and the gu numbers are it, summed —
    downloading the separate gu file as well would only create a second version
    of the same total that could disagree. Before 2020-12 the 읍면동 file does
    not exist at all, so the gu file is the only thing there is. That edge is a
    property of the source, not a download failure, and p20b_foreign.py is where
    the consequence (back-extrapolated dong shares for 2020-01..2020-11) is
    handled and priced.
    """
    return ["B"] if ym >= FIRST_DONG_Q else ["C"]


def quarters(first=FIRST_Q, last=LAST_Q):
    y, m = divmod(first, 100)
    while y * 100 + m <= last:
        yield y * 100 + m
        y, m = (y + 1, 3) if m == 12 else (y, m + 3)


def opener():
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
        urllib.request.HTTPRedirectHandler())
    op.addheaders = [("User-Agent", UA)]
    return op


def _get(op, url, referer=None, timeout=120):
    req = urllib.request.Request(url)
    if referer:
        req.add_header("Referer", referer)
    with op.open(req, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------------------
# source A
# --------------------------------------------------------------------------
def verify_a(raw):
    """month x 시군구 registered-foreigner counts. Checks the shape, the month
    axis, and that Seoul is present with 25 gu in every month."""
    try:
        text = raw.decode("cp949")
    except UnicodeDecodeError as e:
        raise BadFile(f"not cp949 ({e}); first bytes {raw[:60]!r}")
    if "<html" in text[:2000].lower():
        raise BadFile("server returned HTML, not a CSV (session lost)")

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise BadFile("empty file")
    header, body = [c.strip() for c in rows[0]], [r for r in rows[1:] if r and r[0].strip()]
    if header != A_HEADER:
        raise BadFile(f"header is {header}, expected {A_HEADER}")
    if len(body) < A_MIN_ROWS:
        raise BadFile(f"{len(body)} rows, expected >= {A_MIN_ROWS}")

    seoul, months = {}, set()
    for r in body:
        ym = int(r[0]) * 100 + int(r[1])
        months.add(ym)
        if r[2].strip().startswith("서울"):
            seoul.setdefault(ym, {})[r[3].strip()] = int(r[4].replace(",", ""))
    if not seoul:
        raise BadFile("no 서울특별시 rows at all")
    bad = {ym: len(g) for ym, g in seoul.items() if len(g) != 25}
    if bad:
        raise BadFile(f"months where Seoul does not have 25 gu: {bad}")

    # Seoul's registered foreigners ran ~220k-290k over the window. A file for
    # the wrong scope (one province, or 체류외국인 instead of 등록외국인) fails
    # this by an order of magnitude, which is the point.
    tot = {ym: sum(g.values()) for ym, g in seoul.items()}
    lo, hi = min(tot.values()), max(tot.values())
    if not (150_000 <= lo and hi <= 500_000):
        raise BadFile(f"Seoul totals range {lo:,}..{hi:,} — scope looks wrong")

    # The gu names are free text and they are not clean: 2025-02 carries 도붕구
    # for 도봉구. Report it rather than repair it here; p20b normalises.
    names = sorted({n for g in seoul.values() for n in g})
    return {"bytes": len(raw), "rows": len(body),
            "months": sorted(months), "n_months": len(months),
            "seoul_gu_names": names,
            "seoul_total_first": tot[min(tot)], "seoul_total_last": tot[max(tot)],
            "sha256": hashlib.sha256(raw).hexdigest()}


def fetch_a(op):
    _get(op, DGK_PAGE)                       # session cookie
    body = urllib.parse.urlencode({
        "publicDataPk": DGK_PK, "publicDataDetailPk": DGK_UDDI,
        "atchFileId": "", "fileDetailSn": "1", "publicDataTyCode": "PR0051",
    }).encode()
    req = urllib.request.Request(DGK_META, data=body, method="POST")
    req.add_header("Content-Type",
                   "application/x-www-form-urlencoded; charset=UTF-8")
    req.add_header("X-Requested-With", "XMLHttpRequest")
    req.add_header("Referer", DGK_PAGE)
    with op.open(req, timeout=120) as r:
        meta = json.load(r)
    if not meta.get("status"):
        raise BadFile(f"data.go.kr refused the download handshake: {meta!r}"[:300])
    q = urllib.parse.urlencode({"atchFileId": meta["atchFileId"],
                                "fileDetailSn": meta["fileDetailSn"],
                                "dataNm": "foreign"})
    raw = _get(op, f"{DGK_FILE}?{q}", referer=DGK_PAGE)
    rec = verify_a(raw)
    rec["published_name"] = (meta.get("fileDataRegistVO") or {}).get("orginlFileNm")
    return rec, raw


# --------------------------------------------------------------------------
# sources B and C — the quarterly spreadsheets
# --------------------------------------------------------------------------
def board_index(op, pages=45):
    """(year, month) -> nttId for every '등록외국인 지역별 현황' post.

    The board is the 통계월보 자료실 and mixes several series, so the index is
    built by matching the title, not by position on the page.
    """
    out = {}
    for page in range(1, pages + 1):
        t = _get(op, f"{BOARD}?page={page}").decode("utf-8", "replace")
        for m in re.finditer(
                r'/bbs/immigration/227/(\d+)/artclView\.do[^"]*"[^>]*>(.*?)</a>',
                t, re.S):
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            q = TITLE_RE.search(title)
            if q:
                out[int(q.group(1)) * 100 + int(q.group(2))] = m.group(1)
        if out and min(out) <= FIRST_Q:
            break
        time.sleep(0.4)
    return out


def attachments(op, ntt):
    """[(fileId, filename)] for one post, in the order the page lists them.

    The filenames are '21.xlsx'..'25.xlsx' up to 2023 and descriptive from 2026,
    and the ids are not in filename order (a file added later gets a later id),
    so the pair has to be read together.
    """
    t = _get(op, POST.format(ntt=ntt)).decode("utf-8", "replace")
    out = []
    for m in re.finditer(
            r'/bbs/immigration/227/(\d+)/download\.do[^>]*>(.{0,300}?)</a>',
            t, re.S):
        name = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        out.append((m.group(1), name))
    return out, t


def order(atts, kind):
    """Attachment ids, likeliest first. The kind is decided by content, not by
    this ordering — this only keeps the usual case down to one download.

    Two naming eras: bare '21.xlsx'..'25.xlsx' up to 2023, descriptive
    ('2.5. 등록외국인(읍면동, 연령별) 현황(6월말).xlsx') from 2024. The file ids
    are not in filename order either — a file added to a post after the fact
    gets a later id — so the id and the name have to be read as a pair.
    """
    num, word = {"B": (r"(^|\D)2\.?5(\D|$)", "읍면동"),
                 "C": (r"(^|\D)2\.?2(\D|$)", "지역")}[kind]
    hits = [f for f, n in atts if re.search(num, n) or word in n]
    return hits + [f for f, n in atts if f not in hits]


def read_sheet(raw):
    """(column index map, age columns, stray columns, their names, body rows).

    Nothing here may key off a column *position*. The 2020-12 읍면동 file ships
    with 시도 duplicated into two columns, later ones do not; the number of
    title rows above the header changes between quarters; and the age labels
    are spelled '0세~4세' in one file and '0~4세' in the other. So the header
    row is found by content (it is the row that carries 성별 and 총합계), the
    five leading fields are located by name, and the age block is whatever
    columns after 총합계 normalise to the 17 expected bands.
    """
    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    head, body = None, []
    for row in ws.iter_rows(values_only=True):
        cells = [str(c).strip() if c is not None else "" for c in row]
        if head is None:
            if "성별" in cells and "총합계" in cells and "시도" in cells:
                head = cells
            continue
        if cells and cells[0]:
            body.append(list(row))
    wb.close()
    if head is None:
        raise BadFile("no header row carrying 시도/성별/총합계 — wrong attachment")

    idx = {}
    for name in ("시군구", "성별", "총합계"):
        if name in head:
            idx[name] = head.index(name)
    for name in DONG_COL:
        if name in head:
            idx["행정동"] = head.index(name)
            break
    # 2020-12 ships two columns both labelled 시도, and they do not agree: the
    # first says 서울특별시 on rows whose second says 부산광역시. The one that
    # matches the 시군구 beside it is the right-hand one, so take the last 시도
    # before 시군구 rather than the first. Reading position 0 here would file
    # 해운대구 in Seoul, which is precisely the bug this rule removes.
    sido = [i for i, h in enumerate(head) if h == "시도"]
    if sido:
        cut = idx.get("시군구", len(head))
        idx["시도"] = max([i for i in sido if i < cut] or sido)
    for name in ("시도", "성별", "총합계"):
        if name not in idx:
            raise BadFile(f"header has no {name} column: {head[:8]}")
    cols = [i for i in range(idx["총합계"] + 1, len(head)) if head[i]]
    got = [_norm_age(head[i]) for i in cols]
    if got[:len(AGE_HDR)] != AGE_HDR:
        raise BadFile(f"age columns {got} != {AGE_HDR}")
    # Some quarters append stray columns past 80세이상 — 2022-03 through 2022-12
    # carry '111세' and '120세', ten people nationally. They are returned
    # separately rather than dropped, because "there is a column here I did not
    # expect" is exactly the kind of thing that must not be silent.
    extra = cols[len(AGE_HDR):]
    return idx, cols[:len(AGE_HDR)], extra, [head[i] for i in extra], body


def verify_bc(raw, kind, ym):
    """Parse and check one quarterly spreadsheet.

    The kind is read off the header — a 행정동 column means the 읍면동 file —
    and then compared with what was asked for, so an attachment that is the
    wrong table fails here rather than being silently ingested as the right one.
    """
    if raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        # 2024-03 is published as a real BIFF .xls while every other quarter is
        # .xlsx. openpyxl cannot read BIFF and the venv has no xlrd, so this
        # quarter has no dong file the pipeline can open. Named explicitly so it
        # reads as one quarter in one legacy format, not as a broken download.
        raise BadFile("legacy OLE2 .xls (BIFF), not .xlsx — openpyxl cannot "
                      "read it and the venv has no xlrd")
    if raw[:2] != b"PK":
        raise BadFile(f"not a zip/xlsx; first bytes {raw[:16]!r}")
    idx, ages, extra, extra_nm, body = read_sheet(raw)
    got = "B" if "행정동" in idx else "C"
    if got != kind:
        raise BadFile(f"this attachment is kind {got}, not {kind}")
    if "시군구" not in idx:
        raise BadFile("header has no 시군구 column")
    floor = B_MIN_ROWS if kind == "B" else C_MIN_ROWS
    if len(body) < floor:
        raise BadFile(f"{len(body)} data rows, expected >= {floor}")

    sx, tot = idx["성별"], idx["총합계"]

    def num(c):
        if c is None or str(c).strip() in ("", "-"):
            return 0
        return int(float(str(c).replace(",", "")))

    def txt(r, i):
        return str(r[i]).strip() if i < len(r) and r[i] is not None else ""

    sexes = {txt(r, sx) for r in body}
    if not set(SEXES) <= sexes:
        raise BadFile(f"성별 values {sorted(sexes)} do not contain {SEXES}")

    # 총합계 must equal the age bands, on every row. This is the check that
    # catches a shifted column block, which every check above sails past. The
    # published total is allowed to be taken over the 17 canonical bands alone
    # or over those plus the stray columns — which of the two holds is recorded,
    # because it decides whether the strays are already inside the total.
    def rowsum(r, cols):
        return sum(num(r[i]) if i < len(r) else 0 for i in cols)

    # Only the 남성/여성 rows are checked cell by cell: from 2026-06 the 총계
    # roll-up rows carry the total with the age cells left blank, so including
    # them would flag a third of the file as broken when nothing is.
    detail = [r for r in body if txt(r, sx) in ("남성", "여성")]
    if not detail:
        raise BadFile("no 남성/여성 rows — the file is roll-ups only")
    strict = [r for r in detail if num(r[tot]) != rowsum(r, ages + extra)]
    loose = [r for r in detail if num(r[tot]) != rowsum(r, ages)]
    off = min(strict, loose, key=len)
    seoul_off = [r for r in off if txt(r, idx["시도"]).startswith("서울")]
    # A handful of rows in the source do not add up — 2022-03 has two rows in
    # 강원도 강릉시 교2동 that are off by two people in opposite directions, a sex
    # misfiling at source. Tolerated in bulk, recorded, and fatal if it ever
    # touches Seoul, because Seoul is the only part of this file we use.
    if seoul_off or len(off) > 20:
        raise BadFile(f"총합계 != sum of age bands on {len(off)} row(s) "
                      f"({len(seoul_off)} in Seoul), first {off[0][:5]!r}")

    seoul = [r for r in body if txt(r, idx["시도"]).startswith("서울")]
    if not seoul:
        raise BadFile("no 서울특별시 rows")
    gu = {txt(r, idx["시군구"]) for r in seoul} - ROLLUP
    live = {g for g in gu if any(txt(r, idx["시군구"]) == g and num(r[tot])
                                 for r in seoul)}
    if not len(live) >= 25:
        raise BadFile(f"{len(live)} Seoul 시군구 with population, expected 25: "
                      f"{sorted(live)}")
    s_tot = next((num(r[tot]) for r in seoul
                  if txt(r, idx["시군구"]) in ROLLUP - {""}
                  and txt(r, sx) in ROLLUP - {""}), None)
    if s_tot is None:
        raise BadFile("no 서울 총계/총계 row")
    if not 150_000 <= s_tot <= 500_000:
        raise BadFile(f"Seoul total {s_tot:,} outside the plausible band")

    # End to end: the 남성/여성 rows at the finest level present must add back up
    # to the 서울 roll-up. This is the check that would catch a roll-up row being
    # read as a detail row, which is how a hierarchical spreadsheet usually goes
    # wrong, and it is the property p20b actually depends on.
    fine = idx.get("행정동", idx["시군구"])
    leaf = sum(num(r[tot]) for r in seoul if txt(r, sx) in ("남성", "여성")
               and txt(r, fine) not in ROLLUP
               and txt(r, idx["시군구"]) not in ROLLUP)
    if leaf != s_tot:
        raise BadFile(f"Seoul leaf rows sum to {leaf:,}, roll-up says {s_tot:,}")

    rec = {"ym": ym, "kind": kind, "bytes": len(raw), "rows": len(body),
           "seoul_total": s_tot, "seoul_gu": len(live),
           "extra_gu_under_seoul": sorted(gu - live) if len(gu) > 25 else [],
           "rows_not_adding_up": len(off),
           "extra_age_cols": extra_nm,
           "extra_age_total": sum(rowsum(r, extra) for r in body),
           "total_covers_extras": bool(extra) and len(strict) <= len(loose),
           "sha256": hashlib.sha256(raw).hexdigest()}
    if kind == "B":
        dong = {(txt(r, idx["시군구"]), txt(r, idx["행정동"])) for r in seoul
                if txt(r, idx["행정동"]) not in ROLLUP}
        rec["seoul_dong"] = len(dong)
        if len(dong) < 400:
            raise BadFile(f"{len(dong)} Seoul dong, expected >= 400")
    return rec


def path_bc(kind, ym):
    tag = {"B": "dong_age", "C": "gu_age"}[kind]
    return DEST / f"moj_{tag}_{ym}.xlsx"


def fetch_bc(op, ntt, kind, ym):
    """Download the right attachment, deciding by content when the name lies."""
    atts, _ = attachments(op, ntt)
    if not atts:
        raise BadFile(f"post {ntt} has no attachments")
    last = None
    for fid in order(atts, kind):
        try:
            raw = _get(op, DOWN.format(fid=fid),
                       referer=POST.format(ntt=ntt), timeout=240)
            return verify_bc(raw, kind, ym), raw
        except (BadFile, urllib.error.URLError, OSError) as e:
            last = e
    raise BadFile(f"no attachment of post {ntt} verifies as kind {kind}: {last}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check the files already on disk; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download even files that already verify")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
    op = opener()
    ok, failed, fetched = [], [], 0

    # --- A ----------------------------------------------------------------
    pa = DEST / A_NAME
    done = False
    if pa.exists() and not args.force:
        try:
            manifest["A"] = verify_a(pa.read_bytes())
            ok.append("A")
            done = True
        except BadFile as e:
            print(f"  {A_NAME} on disk is bad: {e}")
    if not done:
        if args.verify:
            failed.append((A_NAME, "not downloaded" if not pa.exists() else "bad"))
        else:
            print(f"  {A_NAME} downloading ...", flush=True)
            try:
                rec, raw = fetch_a(op)
                pa.write_bytes(raw)
                manifest["A"] = rec
                ok.append("A")
                fetched += 1
                print(f"    ok  {rec['bytes']:,} B  {rec['n_months']} months "
                      f"{rec['months'][0]}..{rec['months'][-1]}  "
                      f"Seoul {rec['seoul_total_first']:,} -> "
                      f"{rec['seoul_total_last']:,}")
                print(f"    published as {rec['published_name']!r}")
            except (BadFile, urllib.error.URLError, OSError) as e:
                print(f"    FAILED: {e}")
                failed.append((A_NAME, str(e)))

    # --- B and C ----------------------------------------------------------
    index = {}
    need = [(k, ym) for ym, ks in ((q, kinds_for(q)) for q in quarters())
            for k in ks if args.force or not path_bc(k, ym).exists()]
    if need and not args.verify:
        print("  indexing moj.go.kr board 227 ...", flush=True)
        index = board_index(op)
        print(f"    {len(index)} quarterly posts, "
              f"{min(index) if index else '-'}..{max(index) if index else '-'}")
        manifest["board_index"] = {str(k): v for k, v in sorted(index.items())}

    for ym in quarters():
        for kind in kinds_for(ym):
            key = f"{kind}:{ym}"
            p = path_bc(kind, ym)
            if p.exists() and not args.force:
                try:
                    manifest[key] = verify_bc(p.read_bytes(), kind, ym)
                    ok.append(key)
                    continue
                except BadFile as e:
                    print(f"  {p.name} on disk is bad: {e}")
            if args.verify:
                failed.append((p.name, "not downloaded" if not p.exists() else "bad"))
                continue
            ntt = index.get(ym)
            if not ntt:
                print(f"  {key} FAILED: no board post for {ym}")
                failed.append((p.name, "no board post"))
                continue
            print(f"  {key} downloading (post {ntt}) ...", flush=True)
            try:
                rec, raw = fetch_bc(op, ntt, kind, ym)
            except (BadFile, urllib.error.URLError, OSError) as e:
                print(f"    FAILED: {e}")
                failed.append((p.name, str(e)))
                continue
            p.write_bytes(raw)
            manifest[key] = rec
            ok.append(key)
            fetched += 1
            extra = f"  {rec['seoul_dong']} Seoul dong" if kind == "B" else ""
            print(f"    ok  {rec['bytes']:,} B  {rec['rows']:,} rows  "
                  f"Seoul {rec['seoul_total']:,}{extra}")
            time.sleep(args.sleep)

    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)

    want = 1 + sum(len(kinds_for(ym)) for ym in quarters())
    print(f"\n{len(ok)}/{want} files verified"
          f"{f', {fetched} newly downloaded' if fetched else ''}")

    b = {int(k.split(':')[1]): v for k, v in manifest.items()
         if k.startswith("B:") and isinstance(v, dict)}
    if b:
        counts = sorted({v["seoul_dong"] for v in b.values()})
        print(f"  Seoul dong in the 읍면동 files: {counts}"
              f"{'  <- boundary revisions, p20b_foreign' if len(counts) > 1 else ''}")
        dupes = {}
        for ym, v in b.items():
            dupes.setdefault(v["sha256"], []).append(ym)
        for h, ys in dupes.items():
            if len(ys) > 1:
                print(f"  identical bytes for quarters {sorted(ys)} — "
                      f"the board served the same file twice")
                failed.append((f"B:{sorted(ys)[1]}", f"duplicate of {sorted(ys)[0]}"))
    if failed:
        print(f"\n{len(failed)} MISSING OR BAD:")
        for n, why in failed:
            print(f"  {n}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
