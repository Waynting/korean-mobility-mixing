#!/usr/bin/env python
"""L4 — tidy the two ownership surveys into one traceable series.

Reads the PDFs `eda/dl_ownership.py` put under $KOREAN_DATA_ROOT/raw/ownership/
and writes $KOREAN_DATA_ROOT/derived/ownership.parquet plus
`eda/results_p20d.json`.

This series sets the external bound on kappa, and the kappa bound decides which
behavioural claims survive in Phase 1c. Three of them hang on it by less than
1.6 pp. So the extraction is built to fail loudly rather than to produce a
plausible-looking number:

  * **Columns are found by x-coordinate, never by position.** The NIA 통계표
    reorders its columns between waves — 2021 runs 이동전화 · 모바일기기 · 스마트폰,
    and some 2022/2023 pages run 모바일기기 · 이동전화 · 스마트폰. Taking "the
    third number" would silently return 이동전화 for some waves. The header word
    `스마트폰` is located, and each row's value is the number whose x-interval
    contains that word's centre.

  * **The age block is found by structure, not by y-offset.** Every one of
    these tables repeats the same age labels under 성*연령 immediately below
    the 연령 block, and the 2025 English edition repeats them with no prefix at
    all. Rows are grouped into runs, a run ending where a band repeats, and
    only the first run is the 연령 block.

  * **Two quantities are never mixed.** The NIA volume carries both 스마트폰
    보유율 (「휴대형 정보통신기기 보유현황」, ownership) and 스마트폰 이용률
    (「최근 1개월 이내 스마트폰 이용여부」, usage in the last month). Only the
    first is read. It also carries a *household* 스마트폰 보유율 (표 02, BASE
    가구 전체) which is a different population again — the 2022 press release
    headline of 98.6% is that one, not the 98.3% individual figure here.

  * **The 최종보고서 route is cross-checked before it is trusted.** NIA
    published the 2018-2020 통계표 as image-only scans, so those waves — the
    three the letter's bound rests on — exist only in the report volume, where
    the number is a data label on a bar chart. The chart is read by the same
    coordinate machinery, and independently the narrative sentence
    (「70세 이상(53.8%)은 13.6%p 상승」) is parsed. A value is emitted only if
    both agree; disagreements are reported, not averaged.

Every emitted row carries `route`, `source_file`, `page` and the published band
label, so any number can be walked back to a page of a named PDF.

    python eda/p20d_ownership.py                # extract everything, write outputs
    python eda/p20d_ownership.py --source kcc   # one survey
    python eda/p20d_ownership.py --no-write     # print only

Runtime is a few minutes: these are 200-600 page volumes and the page scan is
linear. Located pages are cached in results_p20d.json and reused.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT, RESULTS  # noqa: E402

RAW = DATA_ROOT / "raw" / "ownership"
MANIFEST = RAW / "manifest.json"
OUT_PARQUET = DATA_ROOT / "derived" / "ownership.parquet"
OUT_JSON = RESULTS / "results_p20d.json"

NUM = re.compile(r"^\d{1,3}(?:\.\d)?$")

# ---------------------------------------------------------------------------
# Age bands, as published and as normalised.
#
# The published label is kept verbatim on every row. The normalisation is only
# a join key; it deliberately does NOT paper over the fact that 「70세 이상」 and
# 「70대 이상」 are the same open-ended band under two names, or that the band is
# open-ended at all. `open_top` marks it, because that openness is the whole
# problem this table exists to document.
# ---------------------------------------------------------------------------
BANDS = {
    # published label (whitespace-stripped) -> canonical
    "3-9세": "3-9", "3~9세": "3-9",
    "6-19세": "6-19", "6~19세": "6-19", "6-19": "6-19", "6~19": "6-19",
    "10대": "10-19", "10s": "10-19",
    "20대": "20-29", "20s": "20-29",
    "30대": "30-39", "30s": "30-39",
    "40대": "40-49", "40s": "40-49",
    "50대": "50-59", "50s": "50-59",
    "60대": "60-69", "60s": "60-69",
    "60대이상": "60+", "60orolder": "60+",
    "70세이상": "70+", "70대이상": "70+", "70대": "70+",
    "70+": "70+", "70sorolder": "70+", "70orolder": "70+", "70s": "70+",
}
TOP_BAND = "70+"

SURVEY = {
    "nia": "과기정통부·NIA 인터넷이용실태조사",
    "niarep": "과기정통부·NIA 인터넷이용실태조사",
    "kcc": "방송통신위원회/방송미디어통신위원회 방송매체 이용행태조사",
}

# What the letter already quotes, for the reproduction check. Source key is the
# survey, not the document, because the letter cites the survey.
LETTER = {
    ("nia", 2018): 36.3, ("nia", 2019): 40.2,
    ("nia", 2020): 53.8, ("nia", 2021): 67.9,
    ("kcc", 2018): 37.8, ("kcc", 2019): 39.7,
    ("kcc", 2020): 50.8, ("kcc", 2021): 60.1,
}

# Methodology breaks. A difference taken across one of these measures the
# revision, not the ownership, so `analyse` refuses to emit it as a number.
#
# NIA 2023: the whole survey steps down between the 2022 and 2023 waves — all
# ages 98.3 -> 96.1, 이동전화 100.0 -> 99.4, 70+ 87.7 -> 76.9 — and the table
# numbering jumps from 표 93 to 표 37. Population-level ownership of a durable
# good does not fall, so what moved is the instrument. The value is kept under
# a `_uncomparable` key so the memo can still show it with its label attached,
# but `change_pct` / `mar_to_dec_pct` go to None: a caveat in a memo does not
# survive being copied into a letter, and a None does.
BREAKS = {
    "nia": {2023: "2023 questionnaire/table redesign; all-ages ownership "
                  "falls 98.3 -> 96.1 (-2.2 pp) across the same boundary"},
}


def break_crossed(src, years):
    """Break years with waves on both sides inside `years`."""
    return sorted(b for b in BREAKS.get(src, {}) if b in years and b - 1 in years)


def waves_touched(series, anchor, year, f0, f1):
    """The waves an interpolated window actually reads."""
    ys = sorted(series)
    touched = set()
    for t in (year + f0, year + f1):
        for y0, y1 in zip(ys, ys[1:]):
            if y0 + anchor <= t <= y1 + anchor:
                touched.update((y0, y1))
    return touched


# ---------------------------------------------------------------------------
# page geometry helpers
# ---------------------------------------------------------------------------
def page_rows(page, tol=2.0):
    """Words grouped into visual rows, each sorted left to right."""
    ws = page.extract_words(x_tolerance=1.5, y_tolerance=2.0)
    buckets = {}
    for w in ws:
        buckets.setdefault(round(w["top"] / tol), []).append(w)
    return [(min(x["top"] for x in buckets[k]),
             sorted(buckets[k], key=lambda w: w["x0"]))
            for k in sorted(buckets)]


def header_centre(rows, token, before_top=None):
    """x-centre of the column header word `token`, matched as a whole word.

    Whole-word matching matters: the continuation pages of the NIA table carry
    a column called 「일반 이동전화(스마트폰 제외)」, and a substring match would
    happily point the extractor at it.
    """
    for top, row in rows:
        if before_top is not None and top >= before_top:
            continue
        for w in row:
            if w["text"].strip().rstrip("*").rstrip("(②+④+") == token:
                return top, (w["x0"] + w["x1"]) / 2, w
    return None


CASE_N = re.compile(r"\(\d[\d,]*\)")


def split_label(row):
    """(label, numeric words) for one visual row.

    The label is everything left of the leftmost number. Rows whose labels sit
    to the *right* of the numbers — the mirrored continuation pages — therefore
    come back with an empty label and are ignored, which is what we want: those
    pages do not carry the 스마트폰 column.

    Two things live in that left region besides the label itself, and both had
    to be handled explicitly because each silently produced *zero* extracted
    rows rather than a wrong one:

      * the 방송매체 tables put the unweighted case count there, so the raw
        label of the 10-19 row is `10대(597)`;
      * the 2025 English NIA edition puts the *block* name in a column further
        left, vertically centred, so one row of the age block reads
        `Age 40s` and one row of the sex block reads `Male 40s`.

    So leading words are dropped one at a time until what is left is a band we
    recognise. Dropping a leading `남자`/`Male` turns a 성*연령 row into a bare
    band, which is harmless: `first_age_run` separates the blocks by where a
    band repeats, not by the prefix.
    """
    nums = [w for w in row if NUM.match(w["text"])]
    if not nums:
        return "", []
    first_x = min(w["x0"] for w in nums)
    parts = [w["text"] for w in row if w["x1"] <= first_x + 1]
    full = CASE_N.sub("", re.sub(r"\s+", "", "".join(parts)))
    for i in range(len(parts)):
        cand = CASE_N.sub("", re.sub(r"\s+", "", "".join(parts[i:])))
        if cand in BANDS:
            return cand, nums
    return full, nums


def first_age_run(rows):
    """The 연령 block: the first maximal run of age-band rows.

    A run ends as soon as a band repeats. Every one of these tables follows the
    연령 block immediately with 성*연령, which restates the same bands, so
    "until a band repeats" is exactly the block boundary.
    """
    runs, cur, seen = [], [], set()
    for top, row in rows:
        label, nums = split_label(row)
        band = BANDS.get(label)
        if band is None:
            continue
        if band in seen:
            runs.append(cur)
            cur, seen = [], set()
        cur.append((top, label, band, row, nums))
        seen.add(band)
    if cur:
        runs.append(cur)
    for run in runs:
        if any(b == TOP_BAND for _, _, b, _, _ in run) and len(run) >= 4:
            return run
    return []


def value_at(nums, xc, slack=6.0):
    """The single number whose x-interval contains `xc`. None if 0 or >1 do."""
    hit = [w for w in nums if w["x0"] - slack <= xc <= w["x1"] + slack]
    return float(hit[0]["text"]) if len(hit) == 1 else None


# ---------------------------------------------------------------------------
# extractor 1 — the NIA 통계표 ownership table
# ---------------------------------------------------------------------------
NIA_TABLE = {
    "ko": {"title": "휴대형 정보통신기기 보유현황",
           "base": "만6세 이상 전체", "col": "스마트폰",
           "total": "만6세이상전체"},
    "en": {"title": "Portable ICT device ownership",
           "base": "Entire population aged 6 or older", "col": "Smartphone",
           "total": "Entirepopulationaged6orolder"},
}
PERIOD_RE = re.compile(r"PERIOD\s*[:：]\s*([^\n]{0,60})")


def extract_nia_table(pdf, lang, log):
    cfg = NIA_TABLE[lang]
    out = []
    for pi, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        flat = re.sub(r"\s+", " ", text)
        if cfg["title"] not in flat or cfg["base"] not in flat:
            continue
        # 「- 인터넷 이용자」/「- 인터넷 비이용자」 are different populations.
        if "이용자" in flat.split(cfg["title"])[1][:40] or \
           "Internet users" in flat.split(cfg["title"])[1][:60] or \
           "Internet non-users" in flat.split(cfg["title"])[1][:60]:
            continue
        rows = page_rows(page)
        hdr = header_centre(rows, cfg["col"])
        if hdr is None:
            continue
        htop, xc, _ = hdr
        run = first_age_run([(t, r) for t, r in rows if t > htop])
        if not run:
            continue
        m = PERIOD_RE.search(flat)
        period = m.group(1).strip() if m else ""
        # the all-ages row, as an internal consistency check
        total = None
        for t, r in rows:
            if t <= htop:
                continue
            lab, nums = split_label(r)
            # prefix match: the English all-ages label is "Entire population
            # aged 6 or older", and the bare "6" in it parses as a number, so
            # the label is truncated at it. The prefix is still unambiguous.
            if lab.startswith(cfg["total"][:18]):
                total = value_at(nums, xc)
                break
        got = []
        for _, label, band, _, nums in run:
            v = value_at(nums, xc)
            if v is None or not 0.0 <= v <= 100.0:
                log.append(f"    p{pi+1}: {label!r} gave no unique value")
                continue
            got.append({"age_band": band, "age_band_published": label,
                        "value": v, "page": pi + 1, "period": period,
                        "total_all_ages": total})
        if got:
            log.append(f"    p{pi+1}: {len(got)} bands, "
                       f"all-ages {total}, PERIOD {period!r}")
            out.extend(got)
            break        # the first such page is the main table
    return out


# ---------------------------------------------------------------------------
# extractor 2 — the 방송매체 report's 개인 설문 통계표 item
# ---------------------------------------------------------------------------
KCC_HEAD = re.compile(r"\[문[^\]]{0,10}\]\s*\x07?\s*스마트폰\s*보유\s*$", re.M)
# The "has one" column is 있음 up to 2023 and 보유 from 2024.
KCC_YES = ("있음", "보유")
CASES = re.compile(r"^\(([\d,]+)\)$")


def extract_kcc_table(pdf, log):
    for pi, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        head = text[:600]
        if not KCC_HEAD.search(head) or "대수" in head[:400]:
            continue
        rows = page_rows(page)
        # Anchor on the 사례수 row, not on a bare search for 있음/보유.
        # From 2024 the "has one" column is headed 보유 — and the page title is
        # 「[문1-2] 스마트폰 **보유**」, higher up and further left. A page-wide
        # search finds the title first and silently points every row at the
        # wrong x. The column header always shares a line with 사례수.
        hrow = next(((t, r) for t, r in rows
                     if any(w["text"].startswith("사례수") for w in r)), None)
        if hrow is None:
            log.append(f"    p{pi+1}: no 사례수 header row")
            continue
        htop, hrow_words = hrow
        hdr = None
        for token in KCC_YES:
            for w in hrow_words:
                if w["text"].strip() == token:
                    hdr = (htop, (w["x0"] + w["x1"]) / 2, w)
                    break
            if hdr:
                break
        if hdr is None:
            log.append(f"    p{pi+1}: 사례수 row carries no 있음/보유 column")
            continue
        _, xc, hw = hdr
        yes_label = hw["text"].strip()
        run = first_age_run([(t, r) for t, r in rows if t > htop])
        if not run:
            continue
        out = []
        for _, label, band, row, nums in run:
            v = value_at(nums, xc)
            n = None
            for w in row:
                m = CASES.match(w["text"])
                if m and w["x1"] < xc:
                    n = int(m.group(1).replace(",", ""))
            if v is None or not 0.0 <= v <= 100.0:
                log.append(f"    p{pi+1}: {label!r} gave no unique value")
                continue
            out.append({"age_band": band, "age_band_published": label,
                        "value": v, "n_cases": n, "page": pi + 1,
                        "yes_column": yes_label})
        if out:
            log.append(f"    p{pi+1}: {len(out)} bands, column {yes_label!r}")
            return out
    return []


# ---------------------------------------------------------------------------
# extractor 2b — the 방송매체 report's summary paragraph, which carries the
# previous wave alongside the current one
# ---------------------------------------------------------------------------
# The 방송통계 board has no 2018 report at all: it jumps from 2017 (boardSeq
# 45461) to 2019 (48358). 2018 is one of the four values the letter quotes for
# this survey, so it cannot simply be dropped. The 2019 report's 요약 states it:
#
#   스마트폰(91.1%)의 60대와 70세 이상의 보유 비율이 각각 85.4%, 39.7%로
#   전년(80.3%, 37.8%) 대비 증가
#
# — which is the 2018 pair, and in the same breath restates 2019, which the
# cross-tab on p.307 of the same report also gives. That overlap is a free
# check of the narrative route against the table route; `route_agreement`
# reports it. Later editions word the sentence differently and do not match
# this pattern, which is why 2018 is the only wave this route contributes.
KCC_SUMMARY = re.compile(
    r"60대와\s*70세\s*이상의\s*보유\s*비율이\s*각각\s*"
    r"(\d{1,3}\.\d)\s*%?\s*,\s*(\d{1,3}\.\d)\s*%?\s*로\s*"
    r"전년\s*[\(（]\s*(\d{1,3}\.\d)\s*%?\s*,\s*(\d{1,3}\.\d)\s*%?\s*[\)）]")


def extract_kcc_summary(pdf, year, log):
    """(current-year pairs, previous-year pairs) for 60대 and 70세 이상."""
    for pi, page in enumerate(pdf.pages):
        flat = re.sub(r"\s+", " ", page.extract_text() or "")
        m = KCC_SUMMARY.search(flat)
        if not m:
            continue
        cur60, cur70, prev60, prev70 = (float(m.group(i)) for i in range(1, 5))
        log.append(f"    p{pi+1}: summary gives {year} 60s={cur60} 70+={cur70}, "
                   f"{year-1} 60s={prev60} 70+={prev70}")
        return ({year: {"60-69": cur60, TOP_BAND: cur70},
                 year - 1: {"60-69": prev60, TOP_BAND: prev70}}, pi + 1)
    return {}, None


# ---------------------------------------------------------------------------
# extractor 3 — the NIA 최종보고서 chart, for the waves published as scans
# ---------------------------------------------------------------------------
# The figure plots two years side by side, so one readable report yields the
# current wave and the one before it. The axis category row is found first;
# the data labels are the numbers sitting above it.
REP_SECTION = re.compile(r"성·연령별 스마트폰 보유율")
# 「60대의 스마트폰 보유율(93.1%)은 전년 대비 1.1%p, 70세 이상(53.8%)은 13.6%p 상승하여」
# Both editions word it identically, and it carries four numbers: the two bands
# for the current wave plus the change from the previous one, which recovers
# the previous wave too. This sentence is the only unambiguous machine-readable
# form of the 2017-2020 values — everything else on the page is a bar chart.
REP_NARRATIVE = re.compile(
    r"60대의\s*스마트폰\s*보유율\s*[\(（]\s*(\d{1,3}\.\d)\s*%?\s*[\)）]\s*은?\s*"
    r"전년\s*대비\s*(\d{1,3}\.\d)\s*%?p\s*,\s*"
    r"70세\s*이상\s*[\(（]\s*(\d{1,3}\.\d)\s*%?\s*[\)）]\s*은?\s*"
    r"(\d{1,3}\.\d)\s*%?p\s*상승")
REP_YEARS = re.compile(r"(20\d\d)\s+(20\d\d)")


def extract_nia_report(pdf, year, log):
    for pi, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if not REP_SECTION.search(text):
            continue
        rows = page_rows(page)

        # the axis: the row carrying the most band labels
        best, best_hits = None, 0
        for top, row in rows:
            labs = [re.sub(r"\s+", "", w["text"]) for w in row]
            hits = sum(1 for x in labs if x in BANDS)
            if hits > best_hits:
                best, best_hits = (top, row), hits
        if best is None or best_hits < 5:
            continue
        axis_top, axis_row = best

        # which two years are plotted
        flat = re.sub(r"\s+", " ", text)
        ym = REP_YEARS.search(flat)
        years = (int(ym.group(1)), int(ym.group(2))) if ym else (year - 1, year)
        if years[1] != year:
            log.append(f"    p{pi+1}: legend says {years}, expected .. {year}")

        # numbers above the axis, assigned to the nearest category by x
        cats = [(w, BANDS[re.sub(r'\s+', '', w['text'])]) for w in axis_row
                if re.sub(r"\s+", "", w["text"]) in BANDS]
        labels = [w for top, row in rows if top < axis_top
                  for w in row if NUM.match(w["text"])]
        out = []
        for w, band in cats:
            cx = (w["x0"] + w["x1"]) / 2
            near = sorted(labels,
                          key=lambda v: abs((v["x0"] + v["x1"]) / 2 - cx))[:2]
            if len(near) < 2:
                continue
            near.sort(key=lambda v: v["x0"])
            # a label belongs to this category only if it is closer to it than
            # to any other; otherwise the pairing is ambiguous and we drop it
            ok = True
            for v in near:
                vx = (v["x0"] + v["x1"]) / 2
                if min(abs(vx - (c["x0"] + c["x1"]) / 2) for c, _ in cats) \
                        != abs(vx - cx):
                    ok = False
            if not ok:
                continue
            for yr, v in zip(years, near):
                val = float(v["text"])
                if 0.0 <= val <= 100.0:
                    out.append({"age_band": band,
                                "age_band_published": re.sub(r"\s+", "",
                                                             w["text"]),
                                "value": val, "year": yr, "page": pi + 1})
        # narrative cross-check for the top band
        nm = REP_NARRATIVE.search(re.sub(r"\s+", " ", text))
        narrative = None
        if nm:
            c60, d60, c70, d70 = (float(nm.group(i)) for i in range(1, 5))
            narrative = {
                year: {"60-69": c60, TOP_BAND: c70},
                years[0]: {"60-69": round(c60 - d60, 1),
                           TOP_BAND: round(c70 - d70, 1)},
            }
            log.append(
                f"    p{pi+1}: narrative gives {years[0]} "
                f"60s={narrative[years[0]]['60-69']} "
                f"70+={narrative[years[0]][TOP_BAND]}, {year} "
                f"60s={c60} 70+={c70}")
        if out:
            log.append(f"    p{pi+1}: chart gives {len(out)} points "
                       f"over years {years}")
            return out, narrative, years
    return [], None, None


# ---------------------------------------------------------------------------
def run(sources, cache):
    import pdfplumber

    manifest = json.load(open(MANIFEST))
    rows, logs, notes = [], {}, []

    for key in sorted(manifest):
        rec = manifest[key]
        src, year, lang = rec["source"], rec["year"], rec["lang"]
        if sources and src not in sources:
            continue
        if not rec["text_layer"]:
            notes.append({"key": key, "why": "image-only scan, no text layer",
                          "chars_per_page": rec["chars_per_page"]})
            continue
        path = RAW / f"{key}.pdf"
        log = logs.setdefault(key, [])
        t0 = time.time()
        print(f"  {key} ...", flush=True)
        try:
            with pdfplumber.open(path) as pdf:
                if src == "nia":
                    got = extract_nia_table(pdf, lang, log)
                    for g in got:
                        rows.append(dict(
                            source="nia", survey=SURVEY["nia"], year=year,
                            lang=lang, metric="스마트폰 보유율",
                            metric_en="smartphone ownership rate",
                            population="만6세 이상 개인",
                            route="통계표 table", source_file=key, **g))
                elif src == "kcc":
                    got = extract_kcc_table(pdf, log)
                    for g in got:
                        rows.append(dict(
                            source="kcc", survey=SURVEY["kcc"], year=year,
                            lang=lang, metric="스마트폰 보유",
                            metric_en="smartphone ownership rate",
                            population="만13세 이상 개인",
                            route="보고서 개인 설문 통계표", source_file=key, **g))
                    summ, spage = extract_kcc_summary(pdf, year, log)
                    for sy, bands in summ.items():
                        for band, v in bands.items():
                            rows.append(dict(
                                source="kcc", survey=SURVEY["kcc"], year=sy,
                                lang=lang, metric="스마트폰 보유",
                                metric_en="smartphone ownership rate",
                                population="만13세 이상 개인",
                                route="보고서 요약 narrative", source_file=key,
                                age_band=band, age_band_published=(
                                    "60대" if band == "60-69" else "70세 이상"),
                                value=v, n_cases=None, page=spage,
                                yes_column="보유"))
                elif src == "niarep":
                    got, narrative, years = extract_nia_report(pdf, year, log)
                    charted = set()
                    for g in got:
                        gy = g.pop("year")
                        charted.add((gy, g["age_band"]))
                        rec_ = dict(
                            source="nia", survey=SURVEY["nia"], year=gy,
                            lang=lang, metric="스마트폰 보유율",
                            metric_en="smartphone ownership rate",
                            population="만6세 이상 개인",
                            route="최종보고서 chart", source_file=key,
                            period="", total_all_ages=None, **g)
                        nv = (narrative or {}).get(gy, {}).get(g["age_band"])
                        if nv is not None:
                            rec_["narrative_value"] = nv
                            rec_["cross_checked"] = abs(nv - g["value"]) < 0.051
                        rows.append(rec_)
                    # The chart route drops a category whenever its two data
                    # labels are not unambiguously nearer to it than to a
                    # neighbour — which is what happens to 70세 이상 in both
                    # readable reports, because the neighbouring 60대 bars
                    # carry labels at a similar x. The narrative names the band
                    # and the value in words, so it is emitted in its own
                    # right rather than only as a check on a value the chart
                    # route failed to produce. Without this the three waves the
                    # letter's kappa bound rests on come out empty.
                    for gy, bands in (narrative or {}).items():
                        for band, v in bands.items():
                            if (gy, band) in charted:
                                continue
                            rows.append(dict(
                                source="nia", survey=SURVEY["nia"], year=gy,
                                lang=lang, metric="스마트폰 보유율",
                                metric_en="smartphone ownership rate",
                                population="만6세 이상 개인",
                                route="최종보고서 narrative", source_file=key,
                                period="", total_all_ages=None,
                                age_band=band, age_band_published=(
                                    "60대" if band == "60-69" else "70세 이상"),
                                value=v, page=None))
                            log.append(f"    narrative-only: {gy} {band} = {v}")
        except Exception as e:                                # noqa: BLE001
            log.append(f"    FAILED: {type(e).__name__}: {e}")
            notes.append({"key": key, "why": f"extraction failed: {e}"})
        log.append(f"    [{time.time() - t0:.0f}s]")
        for line in log:
            print(line)
    return rows, logs, notes


def reconcile(rows):
    """Compare against what ../archive/letter_to_advisor.md already quotes.

    A mismatch here is a finding about the existing analysis, so it is reported
    rather than corrected.
    """
    out = []
    for (src, year), quoted in sorted(LETTER.items()):
        cand = [r for r in rows if r["source"] == src and r["year"] == year
                and r["age_band"] == TOP_BAND]
        if not cand:
            out.append({"source": src, "year": year, "letter": quoted,
                        "extracted": None, "status": "NOT AVAILABLE"})
            continue
        vals = sorted({r["value"] for r in cand})
        # the highest-ranked route's value, not the smallest number present
        got = min(cand, key=lambda r: ROUTE_RANK.get(r["route"], 9))["value"]
        out.append({
            "source": src, "year": year, "letter": quoted, "extracted": got,
            "all_extracted": vals,
            "routes": sorted({r["route"] for r in cand}),
            "diff_pp": round(got - quoted, 2),
            "status": "REPRODUCED" if abs(got - quoted) < 0.051
            else "DISAGREES",
        })
    return out


# ---------------------------------------------------------------------------
# what the series does to the kappa bound
# ---------------------------------------------------------------------------
# Each wave is a single point in the year, so a within-year window has to be
# interpolated between two waves. NIA states a fixed reference date; the 방송매체
# survey has no reference date at all, only a field period (2020: 7/6-9/18,
# 2025: 6/16-9/5), so its anchor is the field midpoint and is approximate.
ANCHOR = {"nia": 7.0 / 12, "kcc": 8.5 / 12}   # position within the year


def interp(series, anchor, t, mode="linear"):
    """Ownership at fractional year `t`, between the two waves it falls
    between. Returns None outside the observed range rather than extrapolating:
    extrapolating a saturating curve is exactly how you manufacture drift.

    `mode` is "linear" for the reported series; the "geometric" arm exists only
    so that `letter_band_reconstruction` can search the convention family
    rather than assume one member of it.
    """
    pts = sorted((y + anchor, v) for y, v in series.items())
    if not pts or t < pts[0][0] or t > pts[-1][0]:
        return None
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            if mode == "linear" or v0 <= 0 or v1 <= 0:
                return v0 + (v1 - v0) * f
            return v0 * (v1 / v0) ** f
    return None


def kappa_window(series, anchor, year, m0=3, m1=12, mode="linear", frac=None):
    """Ratio of ownership at month m1 to month m0 of `year`, as a percentage.

    This is the quantity the letter calls the coverage drift: kappa is
    proportional to ownership, so the ratio is what survives into rho-hat.

    `frac` overrides the month-to-fraction convention (default: month
    midpoints). It is a parameter rather than a constant because which
    convention was used is exactly the question `letter_band_reconstruction`
    has to answer.
    """
    f0, f1 = frac if frac else ((m0 - 0.5) / 12, (m1 - 0.5) / 12)
    a = interp(series, anchor, year + f0, mode)
    b = interp(series, anchor, year + f1, mode)
    if a is None or b is None or a == 0:
        return None
    return round((b / a - 1) * 100, 2)


def subband_floor(mean_pct, share):
    """Lower bound on any sub-band's ownership, given the band mean.

    mean = s*x + (1-s)*y with y <= 100 gives x >= 1 - (1-mean)/s. This is the
    step that makes saturation useful without finer age resolution: once a
    published band is near 100%, every sub-band inside it is pinned too,
    because ownership cannot exceed 100% anywhere else in the band.
    """
    lo = 100.0 - (100.0 - mean_pct) / share
    return round(max(0.0, lo), 2)


# Where the same (survey, year, band) is reachable by more than one route, the
# published cross-tab wins and the narrative/chart is a check on it. Ranked so
# the choice is explicit rather than "whichever the loop saw last".
ROUTE_RANK = {"통계표 table": 0, "보고서 개인 설문 통계표": 0,
              "최종보고서 chart": 1, "보고서 요약 narrative": 2}


def route_agreement(rows):
    """Every (survey, year, band) reached by two or more routes, compared."""
    seen = {}
    for r in rows:
        seen.setdefault((r["source"], r["year"], r["age_band"]), []).append(
            (r["route"], r["value"], r["source_file"]))
    out = []
    for k, v in sorted(seen.items()):
        if len({rt for rt, _, _ in v}) < 2:
            continue
        vals = {val for _, val, _ in v}
        out.append({
            "source": k[0], "year": k[1], "age_band": k[2],
            "by_route": [{"route": rt, "value": val, "file": f}
                         for rt, val, f in sorted(v)],
            "spread_pp": round(max(vals) - min(vals), 2),
            "agree": max(vals) - min(vals) < 0.051,
        })
    return out


# ---------------------------------------------------------------------------
# Can the kappa band the letter quotes actually be rebuilt from these series?
#
# `../archive/letter_to_advisor.md` says: "內插到 3 月→12 月，70+ 的覆蓋率漂移是
# +8.9% 至 +21.9%，中位 +16.4%；60 대只有 +1.7% 至 +3.1%". `phase1c-rho.md` and
# `phase1d-r2r3.md` both use those two bands as the 2020 kappa bound, and the
# 60-64 margin of 0.57 pp is the distance from rho-hat to the *lower* edge of
# the 60s band. `phase1d-ownership.md` asked whether that lower edge is the
# same quantity this script computes (+1.94% for NIA) — because if it is, the
# margin rewrites to 0.81 pp, and if it is not, the two must never be
# subtracted from each other.
#
# The letter records no derivation and no memo reconstructs one, so the
# question cannot be answered by reading. It is answered from the other side
# instead: enumerate the whole family of conventions a hand interpolation could
# have used, and ask which members land on the quoted edges. The free
# parameters are the anchor (where in the year a wave is deemed to sit), the
# month-to-fraction convention, and linear vs geometric interpolation.
# ---------------------------------------------------------------------------
LETTER_KAPPA = {"70+": (8.9, 21.9), "60-69": (1.7, 3.1)}
LETTER_KAPPA_MEDIAN = {"70+": 16.4}
LETTER_KAPPA_YEAR = 2020

# The anchor is bounded by documents rather than taste: NIA states 단시점 7.1,
# and the 방송매체 field period has run from mid-June to mid-September depending
# on the wave. So [6/1, 9/1] is defensible for either survey. The rest of the
# year is searched as well, but reported apart, because reaching a quoted edge
# only by anchoring a July survey in February is not reaching it.
DEFENSIBLE_ANCHOR = (6.0 / 12, 9.0 / 12)
ANCHOR_GRID = [i / 48.0 for i in range(48)]
MONTH_CONV = {"midpoint": (2.5 / 12, 11.5 / 12),
              "month-start": (2.0 / 12, 11.0 / 12),
              "month-end": (3.0 / 12, 12.0 / 12)}
INTERP_MODES = ("linear", "geometric")
MAX_HITS_RECORDED = 24


def convention_family(series, year):
    """Every Mar->Dec window value this series can produce."""
    out = []
    for a in ANCHOR_GRID:
        for cname, frac in MONTH_CONV.items():
            for mode in INTERP_MODES:
                v = kappa_window(series, a, year, mode=mode, frac=frac)
                if v is None:
                    continue
                out.append({"anchor_month": round(a * 12, 2),
                            "months": cname, "interp": mode, "value": v,
                            "defensible": (DEFENSIBLE_ANCHOR[0] - 1e-9 <= a
                                           <= DEFENSIBLE_ANCHOR[1] + 1e-9)})
    return out


def letter_band_reconstruction(by, rho=None):
    year = LETTER_KAPPA_YEAR
    fam = {}
    for band in LETTER_KAPPA:
        for src in ("nia", "kcc"):
            ser = by.get((src, band), {})
            fam[(src, band)] = convention_family(ser, year) if ser else []

    bands = []
    for band, (lo, hi) in sorted(LETTER_KAPPA.items()):
        rec = {"age_band": band, "quoted_band": [lo, hi], "per_source": [],
               "joint_hits": [], "n_joint_hits": 0}
        for src in ("nia", "kcc"):
            f = fam[(src, band)]
            if not f:
                continue
            allv = [d["value"] for d in f]
            defv = [d["value"] for d in f if d["defensible"]]
            rec["per_source"].append({
                "source": src,
                "documented_anchor_value": kappa_window(
                    by[(src, band)], ANCHOR[src], year),
                "reachable_defensible_anchor": [min(defv), max(defv)],
                "reachable_any_anchor": [min(allv), max(allv)],
                "quoted_low_reachable": any(round(v, 1) == lo for v in defv),
                "quoted_high_reachable": any(round(v, 1) == hi for v in defv),
            })

        # A hand calculation would hold one month convention and one
        # interpolation rule across both surveys, but may legitimately anchor
        # them differently, since their reference dates differ. So the joint
        # search fixes (months, interp) and lets the two anchors move.
        kidx = {}
        for d in fam[("kcc", band)]:
            kidx.setdefault((d["months"], d["interp"]), []).append(d)
        for dn in fam[("nia", band)]:
            for dk in kidx.get((dn["months"], dn["interp"]), []):
                if {round(dn["value"], 1), round(dk["value"], 1)} != {lo, hi}:
                    continue
                rec["n_joint_hits"] += 1
                if len(rec["joint_hits"]) < MAX_HITS_RECORDED:
                    rec["joint_hits"].append({
                        "months": dn["months"], "interp": dn["interp"],
                        "nia_anchor_month": dn["anchor_month"],
                        "kcc_anchor_month": dk["anchor_month"],
                        "nia_value": dn["value"], "kcc_value": dk["value"],
                        "both_anchors_defensible": (dn["defensible"]
                                                    and dk["defensible"]),
                        "nia_at_documented_anchor": abs(
                            dn["anchor_month"] - ANCHOR["nia"] * 12) < 0.01,
                    })

        # A band quoted with a median has more than two members behind it: for
        # two numbers the median IS the midpoint. This is checkable without any
        # convention at all, so it is reported whether or not the search hits.
        if band in LETTER_KAPPA_MEDIAN:
            mid = round((lo + hi) / 2, 2)
            rec["quoted_median"] = LETTER_KAPPA_MEDIAN[band]
            rec["band_midpoint"] = mid
            rec["median_implies_more_than_two_estimates"] = (
                abs(LETTER_KAPPA_MEDIAN[band] - mid) > 0.05)

        hits_doc = [h for h in rec["joint_hits"]
                    if h["both_anchors_defensible"]
                    and h["nia_at_documented_anchor"]]
        unreachable = []
        for ps in rec["per_source"]:
            rmin, rmax = ps["reachable_any_anchor"]
            for edge in (lo, hi):
                if not rmin - 0.05 <= edge <= rmax + 0.05:
                    unreachable.append(
                        f"{ps['source']}: quoted {edge:+.1f}% is outside "
                        f"[{rmin:+.2f}%, {rmax:+.2f}%], the full range this "
                        f"series can produce at any anchor")
        if hits_doc:
            rec["verdict"] = "reproduced at the documented anchors"
        elif rec["n_joint_hits"]:
            rec["verdict"] = ("reproducible ONLY off the documented anchors — "
                              "same quantity, different convention")
        else:
            rec["verdict"] = ("NOT reproducible from these two series under "
                              "any convention in the family")
        rec["edges_out_of_reach"] = unreachable
        bands.append(rec)

    out = {"year": year, "family_size_per_series": len(ANCHOR_GRID)
           * len(MONTH_CONV) * len(INTERP_MODES),
           "defensible_anchor_months": [round(DEFENSIBLE_ANCHOR[0] * 12, 2),
                                        round(DEFENSIBLE_ANCHOR[1] * 12, 2)],
           "bands": bands}

    # What it means for the margin Phase 1c reports.
    #
    # rho-hat = kappa * lambda, so lambda lies in [rho - kappa_hi, rho - kappa_lo]
    # once kappa is only known to a band. WHICH edge binds depends on the sign
    # of the claim: a negative lambda (60-64) is pinned by kappa_lo and a
    # positive one (65-69) by kappa_hi. So the margin is the distance from zero
    # of whichever endpoint is nearer it, and it is only a margin at all if the
    # interval stays on one side.
    if rho:
        kappa = {}
        doc_pair = [kappa_window(by[(sc, "60-69")], ANCHOR[sc], year)
                    for sc in ("nia", "kcc")]
        kappa["letter band"] = list(LETTER_KAPPA["60-69"])
        kappa["documented anchors"] = [min(doc_pair), max(doc_pair)]
        defv = [d["value"] for sc in ("nia", "kcc")
                for d in fam[(sc, "60-69")] if d["defensible"]]
        kappa["defensible family"] = [min(defv), max(defv)]

        imp = []
        for sub, r in sorted(rho.items()):
            row = {"subband": sub, "rho_mar_to_dec_pct": round(r, 4),
                   "under": []}
            for name, (klo, khi) in kappa.items():
                lam = [round(r - khi, 2), round(r - klo, 2)]
                straddles = lam[0] <= 0 <= lam[1]
                row["under"].append({
                    "kappa_band": name,
                    "kappa_pp": [round(klo, 2), round(khi, 2)],
                    "lambda_pp": lam,
                    "sign": "0 in band" if straddles else
                            ("positive" if lam[0] > 0 else "negative"),
                    "margin_pp": None if straddles else
                                 round(min(abs(lam[0]), abs(lam[1])), 2),
                })
            imp.append(row)
        out["phase1c_margin_implication"] = imp
    return out


def analyse(rows, rho=None):
    by = {}
    best = {}
    for r in rows:
        k = (r["source"], r["age_band"], r["year"])
        rank = ROUTE_RANK.get(r["route"], 9)
        if k not in best or rank < best[k]:
            best[k] = rank
            by.setdefault((r["source"], r["age_band"]), {})[r["year"]] = \
                r["value"]

    annual, windows, sat = [], [], []
    for (src, band), series in sorted(by.items()):
        ys = sorted(series)
        for y0, y1 in zip(ys, ys[1:]):
            if y1 - y0 != 1:
                continue
            v0, v1 = series[y0], series[y1]
            row = {
                "source": src, "age_band": band, "from": y0, "to": y1,
                "from_value": v0, "to_value": v1,
                "change_pp": round(v1 - v0, 2),
                "change_pct": round((v1 / v0 - 1) * 100, 2) if v0 else None,
            }
            crossed = break_crossed(src, {y0, y1})
            if crossed:
                row["change_pp_uncomparable"] = row["change_pp"]
                row["change_pct_uncomparable"] = row["change_pct"]
                row["change_pp"] = None
                row["change_pct"] = None
                row["crosses_break"] = crossed
                row["break_reason"] = BREAKS[src][crossed[0]]
            annual.append(row)
        for y in ys:
            w = kappa_window(series, ANCHOR[src], y)
            if w is None:
                continue
            row = {"source": src, "age_band": band, "year": y,
                   "mar_to_dec_pct": w}
            # A window is exposed if the *interpolation* reads both sides of a
            # break, which happens for the year before it as well as the year
            # of it: with a mid-year anchor, December of year b-1 is already
            # being interpolated towards the wave at b.
            crossed = break_crossed(
                src, waves_touched(series, ANCHOR[src], y,
                                   (3 - 0.5) / 12, (12 - 0.5) / 12))
            if crossed:
                row["mar_to_dec_pct"] = None
                row["mar_to_dec_pct_uncomparable"] = w
                row["crosses_break"] = crossed
                row["break_reason"] = BREAKS[src][crossed[0]]
            windows.append(row)
    # Is the band flat enough that R2 (saturation anchoring) actually works?
    for (src, band), series in sorted(by.items()):
        ys = sorted(series)
        if len(ys) < 2:
            continue
        last = ys[-1]
        prev = last - 1
        if prev not in series:
            continue
        v0, v1 = series[prev], series[last]
        row = {
            "source": src, "age_band": band, "latest_year": last,
            "latest_value": v1, "prev_value": v0,
            "annual_change_pct": round((v1 / v0 - 1) * 100, 2) if v0 else None,
            "nine_month_equiv_pct": round(((v1 / v0) ** 0.75 - 1) * 100, 2)
            if v0 else None,
        }
        crossed = break_crossed(src, {prev, last})
        if crossed:
            row["annual_change_pct_uncomparable"] = row["annual_change_pct"]
            row["nine_month_equiv_pct_uncomparable"] = \
                row["nine_month_equiv_pct"]
            row["annual_change_pct"] = None
            row["nine_month_equiv_pct"] = None
            row["crosses_break"] = crossed
            row["break_reason"] = BREAKS[src][crossed[0]]
        sat.append(row)

    # The sub-band bound for 60-64 / 65-69, which is where Phase 1c is fragile.
    floors = []
    for src in sorted({r["source"] for r in rows}):
        s60 = by.get((src, "60-69"), {})
        for y in sorted(s60):
            for share in (0.40, 0.50):
                floors.append({
                    "source": src, "year": y, "band": "60-69",
                    "band_mean": s60[y], "assumed_subband_share": share,
                    "subband_floor_pct": subband_floor(s60[y], share)})

    # The bound that does not need the next wave.
    #
    # Ownership cannot exceed 100%. So once a band is observed at v, every
    # later value is in [v, 100] (ownership of a durable good does not fall at
    # the population level over a year), and the coverage drift after that
    # observation is bounded above by 100/v - 1 whatever the next wave turns
    # out to say. For a saturated band this is a *certificate*: it closes the
    # kappa bound now, rather than in 2027 when the next wave publishes.
    # Applied to a sub-band via its floor, it is the argument that retires the
    # 60-64 / 65-69 fragility without any finer age resolution.
    ceilings = []
    for (src, band), series in sorted(by.items()):
        if not series:
            continue
        last = max(series)
        v = series[last]
        if v <= 0:
            continue
        # The load-bearing bound. It differences nothing: it reads the single
        # latest wave and the arithmetic fact that ownership cannot exceed
        # 100%, so it is the one result here that the 2023 break cannot touch.
        # Recorded rather than asserted in prose, so a future edit that made it
        # read two waves would show up in the results file.
        row = {"source": src, "age_band": band, "as_of_year": last,
               "latest_value": v, "reads_waves": [last],
               "differences_across_break": False,
               "max_further_drift_pct": round((100.0 / v - 1) * 100, 2)}
        if band == "60-69":
            for share in (0.40, 0.50):
                f = subband_floor(v, share)
                row[f"subband_floor_s{int(share*100)}"] = f
                row[f"max_subband_drift_pct_s{int(share*100)}"] = (
                    round((100.0 / f - 1) * 100, 2) if f > 0 else None)
        ceilings.append(row)

    return {"annual_change": annual, "mar_to_dec_window": windows,
            "saturation_check": sat, "subband_floor_60s": floors,
            "ceiling_bound": ceilings,
            "series_breaks": {k: {str(y): why for y, why in v.items()}
                              for k, v in BREAKS.items()},
            "letter_band_reconstruction": letter_band_reconstruction(by, rho),
            "route_agreement": route_agreement(rows),
            "preferred_series": {f"{s}|{b}": dict(sorted(v.items()))
                                 for (s, b), v in sorted(by.items())}}


def load_rho_60s():
    """rho-hat's Mar->Dec change for the two 60s sub-bands, from p21.

    Read rather than hardcoded, and optional rather than required: p20d must
    still run in a checkout where p21 has not been run. The band used is the
    one with 등록외국인 in the denominator (`mar_dec_all`), because that is the
    one `phase1c-rho.md` §4b declares live and the one phase1d-r2r3 quotes as
    +1.13% for 60-64.
    """
    src = RESULTS / "results_p21.json"
    if not src.exists():
        return None
    try:
        with open(src) as fh:
            rows = json.load(fh).get("denominator_with_foreign", [])
    except (ValueError, OSError):
        return None
    out = {r["band"]: r["mar_dec_all"] for r in rows
           if r.get("band") in ("60-64", "65-69") and "mar_dec_all" in r}
    return out or None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append",
                    choices=["nia", "niarep", "kcc"])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST} — run eda/dl_ownership.py first")

    rows, logs, notes = run(set(args.source or []), {})
    if not rows:
        raise SystemExit("no rows extracted — refusing to write an empty series")

    recon = reconcile(rows)

    top = sorted([r for r in rows if r["age_band"] == TOP_BAND],
                 key=lambda r: (r["source"], r["year"]))
    print("\n70+ smartphone ownership, as extracted")
    print(f"  {'survey':<7} {'year':<6} {'value':>7}  {'route':<22} file")
    for r in top:
        print(f"  {r['source']:<7} {r['year']:<6} {r['value']:>7.1f}  "
              f"{r['route']:<22} {r['source_file']}")

    print("\nreproduction of the values ../archive/letter_to_advisor.md quotes")
    for c in recon:
        got = "n/a" if c["extracted"] is None else f"{c['extracted']:.1f}"
        print(f"  {c['source']} {c['year']}: letter {c['letter']:.1f}  "
              f"extracted {got:>5}  -> {c['status']}")

    analysis = analyse(rows, rho=load_rho_60s())

    ra = [a for a in analysis["route_agreement"]]
    if ra:
        print("\nvalues reachable by more than one route")
        for a in ra:
            mark = "agree" if a["agree"] else f"DISAGREE {a['spread_pp']:+.2f}pp"
            vals = ", ".join(f"{d['route']}={d['value']:.1f}"
                             for d in a["by_route"])
            print(f"  {a['source']} {a['year']} {a['age_band']:<6} {vals}  "
                  f"-> {mark}")

    print("\nMar->Dec coverage drift implied by the series (%)")
    for w in analysis["mar_to_dec_window"]:
        if w["age_band"] not in (TOP_BAND, "60-69"):
            continue
        if w["mar_to_dec_pct"] is None:
            print(f"  {w['source']} {w['year']} {w['age_band']:<6} "
                  f"{'REFUSED':>8}   ({w['mar_to_dec_pct_uncomparable']:+.2f}% "
                  f"would cross the {w['crosses_break'][0]} break)")
        else:
            print(f"  {w['source']} {w['year']} {w['age_band']:<6} "
                  f"{w['mar_to_dec_pct']:+7.2f}%")

    crossed = [a for a in analysis["annual_change"]
               if a.get("crosses_break") and a["age_band"] in (TOP_BAND,
                                                               "60-69")]
    if crossed:
        print("\nyear-on-year changes refused for crossing a break")
        for a in crossed:
            print(f"  {a['source']} {a['from']}->{a['to']} "
                  f"{a['age_band']:<6} {a['from_value']:.1f} -> "
                  f"{a['to_value']:.1f}  "
                  f"({a['change_pct_uncomparable']:+.2f}% withheld)")
            print(f"      {a['break_reason']}")

    print("\nsaturation check on the latest year pair")
    for s in analysis["saturation_check"]:
        if s["age_band"] in (TOP_BAND, "60-69"):
            print(f"  {s['source']} {s['age_band']:<6} "
                  f"{s['prev_value']:.1f} -> {s['latest_value']:.1f} "
                  f"({s['annual_change_pct']:+.2f}%/yr, "
                  f"{s['nine_month_equiv_pct']:+.2f}% over 9 months)")

    print("\nceiling bound — max further drift given ownership <= 100%")
    for c in analysis["ceiling_bound"]:
        if c["age_band"] not in (TOP_BAND, "60-69"):
            continue
        extra = ""
        if "max_subband_drift_pct_s40" in c:
            extra = (f"   sub-band(s=0.40) >= "
                     f"{c['subband_floor_s40']:.2f}%, "
                     f"drift <= {c['max_subband_drift_pct_s40']:+.2f}%")
        print(f"  {c['source']} {c['age_band']:<6} as of {c['as_of_year']} "
              f"= {c['latest_value']:.1f}%  -> further drift "
              f"<= {c['max_further_drift_pct']:+.2f}%{extra}")

    lbr = analysis["letter_band_reconstruction"]
    print(f"\ncan the letter's {lbr['year']} kappa band be rebuilt from these "
          f"series?")
    print(f"  family searched per series: {lbr['family_size_per_series']} "
          f"conventions; anchors called defensible in months "
          f"{lbr['defensible_anchor_months'][0]:.1f}-"
          f"{lbr['defensible_anchor_months'][1]:.1f}")
    for b in lbr["bands"]:
        print(f"\n  {b['age_band']}  letter quotes "
              f"{b['quoted_band'][0]:+.1f}% .. {b['quoted_band'][1]:+.1f}%")
        for ps in b["per_source"]:
            print(f"    {ps['source']:<4} at its documented anchor: "
                  f"{ps['documented_anchor_value']:+.2f}%   "
                  f"reachable {ps['reachable_defensible_anchor'][0]:+.2f}"
                  f"..{ps['reachable_defensible_anchor'][1]:+.2f}% "
                  f"(defensible) / "
                  f"{ps['reachable_any_anchor'][0]:+.2f}"
                  f"..{ps['reachable_any_anchor'][1]:+.2f}% (any anchor)")
        if "quoted_median" in b:
            print(f"    quoted median {b['quoted_median']:+.1f}% vs band "
                  f"midpoint {b['band_midpoint']:+.2f}% -> "
                  f"{'more than two estimates behind the band' if b['median_implies_more_than_two_estimates'] else 'consistent with two'}")
        for u in b["edges_out_of_reach"]:
            print(f"    ** {u}")
        print(f"    joint hits: {b['n_joint_hits']}")
        for h in b["joint_hits"][:4]:
            print(f"      nia anchor {h['nia_anchor_month']:.2f} / kcc "
                  f"{h['kcc_anchor_month']:.2f}, {h['months']}, {h['interp']}"
                  f" -> {h['nia_value']:+.2f}% / {h['kcc_value']:+.2f}%"
                  f"{'  [nia at documented 7.0]' if h['nia_at_documented_anchor'] else ''}")
        print(f"    VERDICT: {b['verdict']}")

    for imp in lbr.get("phase1c_margin_implication", []):
        print(f"\n  {imp['subband']}: rho-hat "
              f"{imp['rho_mar_to_dec_pct']:+.2f}%")
        for u in imp["under"]:
            m = "-" if u["margin_pp"] is None else f"{u['margin_pp']:.2f} pp"
            print(f"      kappa {u['kappa_band']:<19} "
                  f"[{u['kappa_pp'][0]:+.2f}, {u['kappa_pp'][1]:+.2f}]%  ->  "
                  f"lambda [{u['lambda_pp'][0]:+.2f}, "
                  f"{u['lambda_pp'][1]:+.2f}] pp, {u['sign']:<10} "
                  f"margin {m}")

    result = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": len(rows),
        "sources": sorted({r["source"] for r in rows}),
        "years": sorted({r["year"] for r in rows}),
        "top_band_series": [
            {k: r[k] for k in ("source", "year", "value", "route", "lang",
                               "age_band_published", "source_file", "page")}
            for r in top],
        "reconciliation_with_letter": recon,
        "analysis": analysis,
        "skipped": notes,
        "extraction_log": logs,
    }

    if not args.no_write:
        import pandas as pd
        df = pd.DataFrame(rows)
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"\nwrote {OUT_PARQUET}  ({len(df)} rows, "
              f"{df['source'].nunique()} sources, "
              f"{df['year'].nunique()} years)")
        with open(OUT_JSON, "w") as fh:
            json.dump(result, fh, indent=1, ensure_ascii=False)
        print(f"wrote {OUT_JSON}")

    bad = [c for c in recon if c["status"] == "DISAGREES"]
    if bad:
        print(f"\n{len(bad)} quoted value(s) DID NOT reproduce — see the memo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
