#!/usr/bin/env python
"""Phase 1d — pull the dong-level covariates rho-hat has to be regressed on.

Phase 1d's decision statistic is the cross-dong SD of the *residual* from
regressing log rho-hat on dong-level covariates inside each (age, month) cell.
`p21_rho.py` can only report the unconditional spread today because the only
covariate on disk is population. This script fetches the two that are actually
obtainable, and says plainly which one is not.

    1. 면적 (area, km^2) per 행정동 — the one that turns population into density.
       No open table gives it: 지적통계's 읍면동 rows are 법정동, not 행정동, and
       서울 열린데이터광장's 행정구역(동별) table (DT_201004_K060005) only comes out
       of stat.eseoul.go.kr's KOSIS-clone grid, which needs a browser session that
       could not be reproduced from a script. So area is computed from the official
       행정동 boundary geometry instead, one vintage per year of the window.

       Provenance travels with the number: the boundaries are 통계청 SGIS's 행정동
       경계, opened under 공공누리 제1유형 (KOGL Type 1, attribution required),
       redistributed with corrections as a 1975-2026 series by vuski/admdongkor
       under CC BY 4.0. SGIS's own download needs a logged-in session, which is
       why the mirror is used rather than the source. The attribution is written
       into the manifest so it cannot be lost downstream.

    2. 종사자수 / 사업체수 per 행정동 — 서울시 사업체조사 결과 정보 (data.seoul.go.kr
       OA-20326), the establishment-level microdata behind 전국사업체조사, one CSV
       per year. Aggregating it beats any published table: it carries the industry
       division, so the Phase 4 W-panel check gets a retail/food split for free.

    3. 상업 연면적 per 행정동 — NOT OBTAINED, and the reason is structural rather
       than a missing key: 서울시 건축물 연면적 (DT_201004_K030006) is published
       "서울시 및 자치구" only, and the underlying 건축물대장 표제부 is keyed on
       법정동, whose mapping to 행정동 is many-to-many at the parcel level. There is
       no dong-level commercial floor area to download. `p20c_covariates.py` carries
       the retail/food employment split as the nearest available stand-in and labels
       it as such; it is not the same quantity and must not be reported as one.

Output lands in $KOREAN_DATA_ROOT/raw/covariates/ as

    hjd_YYYYMMDD.geojson     행정동 boundaries, one vintage per year
    estab_YYYY.zip           사업체조사 microdata + its own 행정구역 code table
    manifest.json            sha256, shape and the totals every file was checked on

Files that already verify are skipped, so re-running costs nothing and a partial
run resumes.

    python eda/dl_covariates.py             # fetch what is missing, verify all
    python eda/dl_covariates.py --verify    # verify what is on disk, fetch nothing
    python eda/dl_covariates.py --force     # re-fetch everything
    python eda/dl_covariates.py --only estab

Verification is about the *content*, never the transfer. data.seoul.go.kr answers
a lost session with an HTML page carrying a 200 and a Content-Disposition header,
and raw.githubusercontent.com answers a rate limit with a 199-byte text file that
a naive downloader happily writes over 34 MB of geometry. Both are caught here by
parsing the bytes and checking that Seoul is the right size and the right shape.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT  # noqa: E402

DEST = DATA_ROOT / "raw" / "covariates"
MANIFEST = DEST / "manifest.json"

# ---------------------------------------------------------------- 1. boundaries
# vuski/admdongkor republishes the 행정동 boundary set quarterly, each vintage
# carrying both code spaces we need: `adm_cd` is the 7-digit 통계청 행정동 코드 —
# which is what 생활이동's own code table turns out to be, 423 of 424 codes
# identical — and `adm_cd2` is the MOIS 10-digit 행정기관코드 the registration file
# uses. One vintage per year of the data window, so area can move with a boundary
# revision instead of being frozen at one cross-section.
#
# Fetched through the GitHub *API* rather than raw.githubusercontent.com:
# raw.* rate-limits 34 MB files after two or three of them and then answers with
# a 199-byte "429: Too Many Requests" text body, which a downloader that trusts
# the status code writes straight over the geometry. The API's blob endpoint has
# its own, much roomier bucket (60 unauthenticated requests an hour against the
# 14 this needs) and returns the bytes verbatim under the raw media type.
GEO_REPO = "https://api.github.com/repos/vuski/admdongkor"
GEO_RAW = "https://raw.githubusercontent.com/vuski/admdongkor/master"
VINTAGES = ["20200101", "20210101", "20220101", "20230101",
            "20240101", "20250101", "20260701"]

# South Korea is 100,443 km^2 and Seoul 605.24 km^2 by 지적통계. The boundary set
# runs ~0.5-1% high against both because it follows the coast and the Han river
# edge rather than the cadastral line. The bands are loose on purpose: they catch
# a truncated file, a rate-limit stub or a projection blunder, not a revision.
NAT_AREA_RANGE = (95_000.0, 108_000.0)
SEOUL_AREA_RANGE = (590.0, 620.0)
GU_AREA_RANGE = (8.0, 55.0)         # 중구 ~10, 서초구 ~48
N_FEAT_RANGE = (3_300, 3_700)
N_SEOUL_RANGE = (420, 440)
# 17 시도 for most of the window, 16 from 2026-07 when 광주광역시 and 전라남도 merged.
N_SIDO_RANGE = (16, 17)

# ------------------------------------------------------------- 2. 사업체조사
# OA-20326 is a FILE dataset: one zip per survey year, each holding the
# establishment-level CSV, the field dictionary, and — the part that makes the
# join tractable — that year's official 한국행정구역분류 sheet, so the 7- or
# 8-digit survey code can be turned into (구, 동) names without a second source.
EST_PAGE = "https://data.seoul.go.kr/dataList/OA-20326/F/1/datasetView.do"
EST_ENDPOINT = ("https://datafile.seoul.go.kr/bigfile/iot/inf/"
                "nio_download.do?&useCache=false")
EST_INF_ID, EST_INF_SEQ = "OA-20326", "3"
EST_YEARS = [2020, 2021, 2022, 2023, 2024]   # 2025 not yet published (2026-08)

# The column names drift across years — 2020 says WOKE_ALL_SUM, 2021 WOKE_SUM,
# 2022 onward SURV_PHS_WOKE_SUM — so every consumer picks by alias, never by
# position, and fails loudly when none of the aliases is present.
COL_YEAR = ("SURV_BASE_YEAR",)
COL_DONG = ("AD_CD",)
COL_INDUSTRY = ("INDST_LCLS_CD",)
COL_WORKERS = ("WOKE_ALL_SUM", "WOKE_SUM", "SURV_PHS_WOKE_SUM")
COL_WORKERS_M = ("MWK_SUM", "MAN_SUM", "SURV_PHS_MAN_SUM")
COL_WORKERS_F = ("WWK_SUM", "WMAN_SUM", "SURV_PHS_WMAN_SUM")

# Seoul had ~1.21 m establishments and ~5.87 m workers in 2020. Scope checks, not
# accuracy checks: they catch a national file or a truncated one.
N_ESTAB_RANGE = (1_000_000, 1_400_000)
N_WORKER_RANGE = (4_500_000, 7_000_000)


class BadFile(Exception):
    """The bytes on disk are not the thing we asked for."""


# ---------------------------------------------------------------- geodesy
# Area on the WGS84 ellipsoid, without geopandas/shapely/pyproj — none of which
# are in .venv, and adding one would make the pipeline depend on a package that
# requirements.txt does not name. The polygon is mapped onto the *authalic*
# sphere (the sphere of equal total area, R = 6,371,007.181 m) by replacing
# geodetic latitude with authalic latitude, then closed by the spherical excess
# formula of Chamberlain & Duquette. That is exact for the sphere and accurate to
# ~1e-5 relative for a city-scale polygon on the ellipsoid — three orders of
# magnitude finer than the 0.4% by which the boundary set and 지적통계 disagree
# about Seoul in the first place.
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_E = math.sqrt(_E2)


def _q(sin_phi):
    """Authalic-latitude numerator; q(pi/2) normalises it to sin(beta)."""
    return (1 - _E2) * (sin_phi / (1 - _E2 * sin_phi * sin_phi)
                        - (1 / (2 * _E)) * np.log((1 - _E * sin_phi)
                                                  / (1 + _E * sin_phi)))


_QP = float(_q(np.array(1.0)))
R_AUTHALIC = _A * math.sqrt(_QP / 2)


def ring_area_m2(coords):
    """Signed area of one closed lon/lat ring, in m^2."""
    c = np.asarray(coords, dtype=float)
    lam = np.radians(c[:, 0])
    sb = _q(np.sin(np.radians(c[:, 1]))) / _QP
    return R_AUTHALIC * R_AUTHALIC / 2.0 * float(
        np.sum((lam[1:] - lam[:-1]) * (sb[1:] + sb[:-1])))


def polygon_area_m2(rings):
    """First ring is the exterior, the rest are holes (RFC 7946).

    Taken on absolute values rather than on the signed sum, because plenty of
    published GeoJSON ignores the winding-order rule and a sign convention read
    off the data would silently turn a hole into extra land.
    """
    a = abs(ring_area_m2(rings[0]))
    for hole in rings[1:]:
        a -= abs(ring_area_m2(hole))
    return a


def feature_area_m2(geom):
    if geom["type"] == "Polygon":
        return polygon_area_m2(geom["coordinates"])
    if geom["type"] == "MultiPolygon":
        return sum(polygon_area_m2(p) for p in geom["coordinates"])
    raise BadFile(f"unexpected geometry type {geom['type']!r}")


# ---------------------------------------------------------------- verification
def verify_boundary(vintage, raw):
    """Parse and check one boundary vintage. Returns the facts worth recording."""
    if raw[:1] not in (b"{", b"\xef"):
        head = raw[:120].decode("utf-8", "replace")
        raise BadFile(f"not JSON — starts {head!r} "
                      f"(a rate-limit or error stub, {len(raw)} bytes)")
    try:
        gj = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BadFile(f"unparseable JSON ({e}); {len(raw)} bytes")

    if gj.get("type") != "FeatureCollection":
        raise BadFile(f"type is {gj.get('type')!r}, expected FeatureCollection")
    if gj.get("name") != f"HangJeongDong_ver{vintage}":
        raise BadFile(f"name is {gj.get('name')!r}, expected "
                      f"HangJeongDong_ver{vintage} — wrong vintage")
    crs = json.dumps(gj.get("crs", {}))
    if "CRS84" not in crs and "4326" not in crs:
        raise BadFile(f"CRS is {crs} — the area formula assumes lon/lat WGS84")

    feats = gj["features"]
    if not N_FEAT_RANGE[0] <= len(feats) <= N_FEAT_RANGE[1]:
        raise BadFile(f"{len(feats)} features, expected {N_FEAT_RANGE}")

    sido, nat_m2, widths = {}, 0.0, set()
    seoul_area, gu_area, seoul_names = 0.0, {}, {}
    for f in feats:
        p = f["properties"]
        cd, cd2, nm = str(p.get("adm_cd", "")), str(p.get("adm_cd2", "")), p.get("adm_nm")
        # Width is checked everywhere, digits only inside Seoul: a handful of 읍
        # elsewhere carry an alphanumeric 통계청 code (대구 달성군 현풍읍 is 22310AB),
        # and rejecting those would fail a file that is perfectly good for us.
        # 7 characters through the 2023 vintage, 8 from 2024 — 통계청 widened the
        # 시군구 field (사직동 1101053 -> 11010530), the same change the 사업체조사
        # made in 2022. Both widths are legal; what is not legal is a file mixing
        # them, which the width set below catches.
        if len(cd) not in (7, 8):
            raise BadFile(f"adm_cd {cd!r} is not 7 or 8 characters ({nm})")
        widths.add(len(cd))
        if len(cd2) != 10:
            raise BadFile(f"adm_cd2 {cd2!r} is not 10 characters ({nm})")
        if cd.startswith("11") and not (cd.isdigit() and cd2.isdigit()):
            raise BadFile(f"Seoul codes must be numeric: {cd!r} / {cd2!r} ({nm})")
        if not nm or len(nm.split()) < 2:
            raise BadFile(f"adm_nm {nm!r} is not '시도 시군구 읍면동'")
        # A ring that does not close makes the shoelace sum meaningless, and it
        # is the one geometric defect the area itself would not reveal.
        for poly in ([f["geometry"]["coordinates"]]
                     if f["geometry"]["type"] == "Polygon"
                     else f["geometry"]["coordinates"]):
            for ring in poly:
                if ring[0] != ring[-1]:
                    raise BadFile(f"unclosed ring in {nm}")
        a = feature_area_m2(f["geometry"])
        nat_m2 += a
        sido[nm.split()[0]] = sido.get(nm.split()[0], 0.0) + a
        if cd.startswith("11"):
            seoul_area += a
            gu = nm.split()[1]
            gu_area[gu] = gu_area.get(gu, 0.0) + a
            seoul_names[cd] = nm

    nat_km2 = nat_m2 / 1e6
    seoul_km2 = seoul_area / 1e6
    if len(widths) != 1:
        raise BadFile(f"mixed adm_cd widths {sorted(widths)} in one vintage")
    if not NAT_AREA_RANGE[0] <= nat_km2 <= NAT_AREA_RANGE[1]:
        raise BadFile(f"national area {nat_km2:,.0f} km2 outside {NAT_AREA_RANGE}")
    if not N_SIDO_RANGE[0] <= len(sido) <= N_SIDO_RANGE[1]:
        raise BadFile(f"{len(sido)} 시도, expected {N_SIDO_RANGE}")
    if not N_SEOUL_RANGE[0] <= len(seoul_names) <= N_SEOUL_RANGE[1]:
        raise BadFile(f"{len(seoul_names)} Seoul dong, expected {N_SEOUL_RANGE}")
    if len(gu_area) != 25:
        raise BadFile(f"{len(gu_area)} Seoul 자치구, expected 25: "
                      f"{sorted(gu_area)}")
    if not SEOUL_AREA_RANGE[0] <= seoul_km2 <= SEOUL_AREA_RANGE[1]:
        raise BadFile(f"Seoul area {seoul_km2:.2f} km2 outside {SEOUL_AREA_RANGE}")
    for gu, a in gu_area.items():
        if not GU_AREA_RANGE[0] <= a / 1e6 <= GU_AREA_RANGE[1]:
            raise BadFile(f"{gu} is {a / 1e6:.2f} km2, outside {GU_AREA_RANGE}")

    return {"vintage": vintage, "bytes": len(raw), "features": len(feats),
            "adm_cd_width": widths.pop(),
            "n_sido": len(sido), "seoul_dong": len(seoul_names),
            "seoul_area_km2": round(seoul_km2, 4),
            "national_area_km2": round(nat_km2, 1),
            "gu_area_km2": {g: round(a / 1e6, 4) for g, a in sorted(gu_area.items())},
            "sha256": hashlib.sha256(raw).hexdigest()}


def _zip_names(z):
    """Zip entries whose names are cp437-mangled cp949, which these are."""
    out = []
    for i in z.infolist():
        if i.flag_bits & 0x800:
            out.append((i, i.filename))
            continue
        try:
            out.append((i, i.orig_filename.encode("cp437").decode("cp949")))
        except (UnicodeEncodeError, UnicodeDecodeError):
            out.append((i, i.orig_filename))
    return out


def pick(header, aliases, what):
    """The one column of `aliases` present in `header`, or a loud failure."""
    for a in aliases:
        if a in header:
            return a
    raise BadFile(f"no {what} column in {header[:6]}... — "
                  f"tried {aliases}; the layout changed")


def verify_estab(year, raw):
    """Parse and check one 사업체조사 zip."""
    if raw[:2] != b"PK":
        head = raw[:200].decode("utf-8", "replace")
        raise BadFile(f"not a zip — starts {head!r} (a session-timeout page "
                      f"carries a 200 and a Content-Disposition just as happily)")
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as e:
        raise BadFile(f"corrupt zip ({e})")

    members = _zip_names(z)
    csvs = [(i, n) for i, n in members if n.lower().endswith(".csv")]
    xlsx = [(i, n) for i, n in members if n.lower().endswith((".xlsx", ".xls"))]
    if len(csvs) != 1:
        raise BadFile(f"{len(csvs)} CSV members, expected 1: "
                      f"{[n for _, n in members]}")
    if not xlsx:
        raise BadFile("no field-dictionary workbook — the 행정구역 code table "
                      "the join needs lives in it")

    info, csv_name = csvs[0]
    body = z.read(info)
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = body.decode("cp949")
        except UnicodeDecodeError as e:
            raise BadFile(f"{csv_name}: neither utf-8 nor cp949 ({e})")

    rd = csv.reader(io.StringIO(text))
    header = next(rd)
    c_year = header.index(pick(header, COL_YEAR, "survey year"))
    c_dong = header.index(pick(header, COL_DONG, "dong code"))
    c_ind = header.index(pick(header, COL_INDUSTRY, "industry division"))
    c_tot = header.index(pick(header, COL_WORKERS, "total workers"))
    c_m = header.index(pick(header, COL_WORKERS_M, "male workers"))
    c_f = header.index(pick(header, COL_WORKERS_F, "female workers"))

    def num(cell):
        cell = (cell or "").strip().replace(",", "")
        return int(cell) if cell and cell != "-" else 0

    n_rows, workers, codes, bad_sum, code_len = 0, 0, {}, 0, set()
    industries = set()
    for r in rd:
        if not r or not r[c_dong].strip():
            continue
        n_rows += 1
        if r[c_year].strip() != str(year):
            raise BadFile(f"row {n_rows} says year {r[c_year]!r}, expected {year} "
                          f"— the portal's file ids moved under us")
        cd = r[c_dong].strip()
        code_len.add(len(cd))
        codes[cd] = codes.get(cd, 0) + 1
        industries.add(r[c_ind].strip())
        t = num(r[c_tot])
        workers += t
        if num(r[c_m]) + num(r[c_f]) != t:
            bad_sum += 1

    if not N_ESTAB_RANGE[0] <= n_rows <= N_ESTAB_RANGE[1]:
        raise BadFile(f"{n_rows:,} establishments, expected {N_ESTAB_RANGE} "
                      f"— wrong scope (national file?) or truncated")
    if not N_WORKER_RANGE[0] <= workers <= N_WORKER_RANGE[1]:
        raise BadFile(f"{workers:,} workers, expected {N_WORKER_RANGE}")
    if not N_SEOUL_RANGE[0] <= len(codes) <= N_SEOUL_RANGE[1]:
        raise BadFile(f"{len(codes)} distinct dong codes, expected {N_SEOUL_RANGE}")
    if len(code_len) != 1:
        raise BadFile(f"mixed dong-code widths {sorted(code_len)} in one file")
    if not all(c.startswith("11") for c in codes):
        outside = sorted(c for c in codes if not c.startswith("11"))[:5]
        raise BadFile(f"dong codes outside Seoul: {outside}")
    # 남 + 여 = 계 on every establishment. Cheap, and it is the one check that
    # catches a column-offset shift after a layout revision.
    if bad_sum:
        raise BadFile(f"{bad_sum:,} rows where 남+여 != 계")

    return {"year": year, "bytes": len(raw), "csv": csv_name,
            "establishments": n_rows, "workers": workers,
            "dong": len(codes), "dong_code_width": code_len.pop(),
            "industry_divisions": "".join(sorted(industries)),
            "members": [n for _, n in members],
            "sha256": hashlib.sha256(raw).hexdigest()}


# ---------------------------------------------------------------- fetching
def _opener():
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
        urllib.request.HTTPRedirectHandler())
    op.addheaders = [("User-Agent",
                      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36")]
    return op


def _get(op, url, tries, wait0, accept=None, referer=None):
    """GET with backoff. A 429 is waited out, doubling; anything else retries flat."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url)
            if accept:
                req.add_header("Accept", accept)
            if referer:
                req.add_header("Referer", referer)
            with op.open(req, timeout=900) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            wait = wait0 * (2 ** (attempt - 1)) if e.code == 429 else wait0
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            wait = wait0
        if attempt < tries:
            print(f"    attempt {attempt} failed ({last}); retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise BadFile(f"gave up after {tries} attempts — {last}")


def fetch_boundary(vintage, op, tries, wait0):
    """Contents endpoint under the raw media type, with raw.* as the fallback.

    `contents` returns the bytes verbatim for files up to 100 MB when asked for
    `application/vnd.github.raw` — the 1 MB ceiling people remember applies to
    the base64 JSON representation, not to this. One request per file.
    """
    fname = f"HangJeongDong_ver{vintage}.geojson"
    try:
        raw = _get(op, f"{GEO_REPO}/contents/ver{vintage}/{fname}", tries, wait0,
                   accept="application/vnd.github.raw")
    except BadFile as e:
        print(f"    API route failed ({e}); falling back to raw.githubusercontent")
        raw = _get(op, f"{GEO_RAW}/ver{vintage}/{fname}", tries, wait0)
    return verify_boundary(vintage, raw), raw


def discover_estab(op):
    """(year -> portal file id) scraped from the dataset page.

    The ids are the portal's own row numbers and there is nothing stopping them
    from moving, so they are read off the page every run rather than pinned in
    the source. `verify_estab` then checks the survey year *inside* the file, so
    a renumbering fails loudly instead of quietly saving the wrong year.
    """
    page = _get(op, EST_PAGE, 3, 5).decode("utf-8", "replace")
    pairs = re.findall(r"downloadFile\('(\d+)'\);\">([^<]+)</span>", page)
    if not pairs:
        raise BadFile("no file rows on the OA-20326 page — layout changed or "
                      "the portal returned an error page")
    found = {}
    for seq, name in pairs:
        m = re.search(r"(\d{4})\s*년", name)
        if m:
            found.setdefault(int(m.group(1)), (seq, name.strip()))
    return found


def fetch_estab(year, seq, op, tries, wait0):
    body = urllib.parse.urlencode({"infId": EST_INF_ID, "seqNo": "",
                                   "seq": seq, "infSeq": EST_INF_SEQ}).encode()
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(EST_ENDPOINT, data=body, method="POST")
            req.add_header("Referer", EST_PAGE)
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with op.open(req, timeout=900) as r:
                raw = r.read()
            return verify_estab(year, raw), raw
        except (urllib.error.URLError, BadFile, TimeoutError, OSError) as e:
            last = e
            if attempt < tries:
                print(f"    attempt {attempt} failed ({e}); "
                      f"retrying in {wait0 * attempt}s", flush=True)
                time.sleep(wait0 * attempt)
    raise BadFile(f"{year}: gave up after {tries} attempts — {last}")


def geo_path(v):
    return DEST / f"hjd_{v}.geojson"


def est_path(y):
    return DEST / f"estab_{y}.zip"


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check the files already on disk; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download even files that already verify")
    ap.add_argument("--only", choices=("area", "estab"), default=None,
                    help="restrict to one source")
    ap.add_argument("--sleep", type=float, default=20.0,
                    help="seconds between boundary requests (default 20; "
                         "raw.githubusercontent rate-limits 34 MB files hard)")
    ap.add_argument("--tries", type=int, default=6)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
    manifest.setdefault("boundaries", {})
    manifest.setdefault("establishments", {})
    manifest["attribution"] = {
        "boundaries":
            "통계청 통계지리정보서비스(SGIS, https://sgis.kostat.go.kr)에서 공공누리 "
            "제1유형으로 개방한 행정동 경계를 가공한 것이며(가공: vuski/admdongkor, "
            "https://github.com/vuski/admdongkor), CC BY 4.0으로 배포됩니다.",
        "establishments":
            "서울시 사업체조사 결과 정보 (서울 열린데이터광장 OA-20326, "
            "공공누리 제1유형). 원 조사는 통계청 전국사업체조사.",
    }
    manifest["not_obtained"] = {
        "상업 연면적 (commercial gross floor area) per 행정동":
            "no dong-level source exists. 서울시 건축물 연면적 (DT_201004_K030006) "
            "is published 시/자치구 only; 건축물대장 표제부 is keyed on 법정동, whose "
            "map to 행정동 is many-to-many at parcel level. p20c carries the "
            "retail/food employment split as a stand-in, not as a substitute.",
        "official 행정동 면적 table":
            "서울 열린데이터광장 DT_201004_K060005 (행정구역(동별), 매년 12월 기준) "
            "does hold it, but it is only served through stat.eseoul.go.kr's "
            "KOSIS-clone grid, whose html.do/downGrid.do endpoints refused every "
            "reconstruction of the browser form. Area here is computed from the "
            "boundary geometry instead and cross-checked against the published "
            "605.24 km2 for Seoul.",
    }

    op = _opener()
    ok, failed, fetched = [], [], 0

    wanted_geo = [] if args.only == "estab" else VINTAGES
    for v in wanted_geo:
        p = geo_path(v)
        if p.exists() and not args.force:
            try:
                manifest["boundaries"][v] = verify_boundary(v, p.read_bytes())
                ok.append(f"geo {v}")
                continue
            except BadFile as e:
                print(f"  boundary {v} on disk is bad: {e}")
                if args.verify:
                    failed.append((f"geo {v}", str(e)))
                    continue
        if args.verify:
            if not p.exists():
                failed.append((f"geo {v}", "not downloaded"))
            continue
        print(f"  boundary {v} downloading ...", flush=True)
        try:
            rec, raw = fetch_boundary(v, op, args.tries, args.sleep)
        except BadFile as e:
            print(f"  boundary {v} FAILED: {e}")
            failed.append((f"geo {v}", str(e)))
            continue
        p.write_bytes(raw)
        manifest["boundaries"][v] = rec
        ok.append(f"geo {v}")
        fetched += 1
        print(f"  boundary {v} ok  {rec['bytes']:>10,} B  "
              f"{rec['seoul_dong']:>3} Seoul dong  "
              f"{rec['seoul_area_km2']:.2f} km2  "
              f"national {rec['national_area_km2']:,.0f} km2")
        time.sleep(args.sleep)

    wanted_est = [] if args.only == "area" else EST_YEARS
    seqs = None
    for y in wanted_est:
        p = est_path(y)
        if p.exists() and not args.force:
            try:
                manifest["establishments"][str(y)] = verify_estab(y, p.read_bytes())
                ok.append(f"estab {y}")
                continue
            except BadFile as e:
                print(f"  estab {y} on disk is bad: {e}")
                if args.verify:
                    failed.append((f"estab {y}", str(e)))
                    continue
        if args.verify:
            if not p.exists():
                failed.append((f"estab {y}", "not downloaded"))
            continue
        if seqs is None:
            seqs = discover_estab(op)
            print(f"  OA-20326 offers {len(seqs)} years: "
                  f"{sorted(seqs)[:3]}...{sorted(seqs)[-3:]}")
        if y not in seqs:
            failed.append((f"estab {y}", "no file for that year on the page"))
            continue
        seq, name = seqs[y]
        print(f"  estab {y} downloading seq={seq} ({name}) ...", flush=True)
        try:
            rec, raw = fetch_estab(y, seq, op, 3, 10)
        except BadFile as e:
            print(f"  estab {y} FAILED: {e}")
            failed.append((f"estab {y}", str(e)))
            continue
        p.write_bytes(raw)
        rec["portal_seq"], rec["portal_name"] = seq, name
        manifest["establishments"][str(y)] = rec
        ok.append(f"estab {y}")
        fetched += 1
        print(f"  estab {y} ok  {rec['bytes']:>10,} B  "
              f"{rec['establishments']:>9,} establishments  "
              f"{rec['workers']:>9,} workers  {rec['dong']} dong  "
              f"code width {rec['dong_code_width']}")
        time.sleep(2)

    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False, sort_keys=True)

    n_want = len(wanted_geo) + len(wanted_est)
    print(f"\n{len(ok)}/{n_want} files verified"
          f"{f', {fetched} newly downloaded' if fetched else ''}")

    b = manifest["boundaries"]
    if b:
        counts = sorted({r["seoul_dong"] for r in b.values()})
        print(f"  Seoul dong across the boundary vintages: {counts}"
              f"{'  <- boundary revisions inside the window' if len(counts) > 1 else ''}")
        areas = [r["seoul_area_km2"] for r in b.values()]
        print(f"  Seoul area across vintages: {min(areas):.2f}-{max(areas):.2f} km2 "
              f"(지적통계 publishes 605.24)")
        gw = sorted({r["adm_cd_width"] for r in b.values()})
        if len(gw) > 1:
            per = {v: r["adm_cd_width"] for v, r in sorted(b.items())}
            print(f"  adm_cd width changes across vintages: {per} "
                  f"— join on names, not on codes")
    e = manifest["establishments"]
    if e:
        widths = sorted({r["dong_code_width"] for r in e.values()})
        if len(widths) > 1:
            per = {r["year"]: r["dong_code_width"] for r in e.values()}
            print(f"  dong-code width changes across survey years: {per} "
                  f"— join on names, not on codes")
    if failed:
        print(f"\n{len(failed)} MISSING OR BAD:")
        for what, why in failed:
            print(f"  {what}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
