"""Shared DuckDB setup: per-day normalisation, gu aggregation, policy exposure.

`이동인구(합)` is a monthly SUM over every occurrence of that weekday (verified in
p0_contract.py section 1.0), so nothing may be compared across months or weekdays
without dividing by the occurrence count. `pop_day` is the per-calendar-day flow.

The occurrence counts, holiday counts and policy doses are computed from the real
calendar in calendar_kr.py rather than hard-coded, so adding months is free.
"""
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calendar_kr as K
# ROOT is the repo (results, figures, cases); the parquet is wherever
# KOREAN_DATA_ROOT points. See paths.py.
from paths import (DATA_ROOT, DERIVED, FIG, PARQUET, PARQUET_GLOB,  # noqa: F401
                   RAW, REPO, ROOT, require)

# The 2020 analysis window. NOT "what is on disk" — the parquet tree also holds
# 2026 months (Phase 2a), and every existing results_pNN.json was computed on
# these twelve. Widening this silently would rewrite published numbers, so a
# script that wants more months has to ask for them: connect(ym_filter=...).
YMS = [202001, 202002, 202003, 202004, 202005, 202006, 202007,
       202008, 202009, 202010, 202011, 202012]


def months_on_disk():
    """ym partitions actually present in the parquet tree, ascending."""
    if not PARQUET.exists():
        return []
    out = []
    for p in PARQUET.glob("ym=*"):
        try:
            out.append(int(p.name.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(out)
YM_LABEL = {202001: "Jan", 202002: "Feb", 202003: "Mar", 202004: "Apr",
            202005: "May", 202006: "Jun", 202007: "Jul", 202008: "Aug",
            202009: "Sep", 202010: "Oct", 202011: "Nov", 202012: "Dec"}
DOW_LABEL = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

GU_NAMES = {  # verified 2026-08-17 against the official 행정동코드_20210907.xlsx:
              # all 25 names match, 424 Seoul dong, counts match (송파 27, 금천 10).
              # Table is at $KOREAN_DATA_ROOT/docs/; see memo/phase1a-manual.md.
    1101: "종로구", 1102: "중구", 1103: "용산구", 1104: "성동구", 1105: "광진구",
    1106: "동대문구", 1107: "중랑구", 1108: "성북구", 1109: "강북구", 1110: "도봉구",
    1111: "노원구", 1112: "은평구", 1113: "서대문구", 1114: "마포구", 1115: "양천구",
    1116: "강서구", 1117: "구로구", 1118: "금천구", 1119: "영등포구", 1120: "동작구",
    1121: "관악구", 1122: "서초구", 1123: "강남구", 1124: "송파구", 1125: "강동구",
}

AGE_LABEL = {0: "0-9", 10: "10-14", 15: "15-19", 20: "20-24", 25: "25-29",
             30: "30-34", 35: "35-39", 40: "40-44", 45: "45-49", 50: "50-54",
             55: "55-59", 60: "60-64", 65: "65-69", 70: "70-74", 75: "75-79",
             80: "80+"}
AGES = sorted(AGE_LABEL)

# 두 달 모두에서 공휴일이 없는 요일만 쓰던 D3 의 8개월 버전은 clean_dows() 참조.


def cell_table(yms=None):
    """(ym, dow_n) -> 일수, 공휴일수, 정책 dose. calendar_kr 에서 계산."""
    return [K.cell_exposure(ym, d) for ym in (yms or YMS) for d in range(1, 8)]


def clean_dows():
    """모든 월에서 공휴일을 하나도 포함하지 않는 요일 (D3 의 일반화)."""
    bad = {r["dow_n"] for r in cell_table() if r["n_holiday"] > 0}
    return sorted(set(range(1, 8)) - bad)


def connect(threads=8, ym_filter=None):
    """DuckDB with a view `m` over the parquet, joined to the calendar.

    `ym_filter` defaults to YMS — the 2020 window — and the restriction is an
    explicit WHERE, not a side effect of the calendar join. That distinction is
    the whole point of this function's shape: `cal` used to be built from YMS
    alone and the view INNER JOINed against it, so once 2026 landed in the tree
    those rows were dropped with no error and no row count to notice. A scan that
    silently covers a different period than the caller believes is the worst
    failure mode this project has, because the output still looks right.

    Pass `ym_filter="all"` for every month on disk, or an explicit list.
    """
    require(PARQUET, "parquet tree")
    disk = months_on_disk()
    if ym_filter == "all":
        ym_filter = disk
    elif ym_filter is None:
        ym_filter = YMS
    ym_filter = sorted(ym_filter)
    missing = [y for y in ym_filter if y not in disk]
    if missing:
        raise SystemExit(f"months requested but not in the parquet tree: {missing}\n"
                         f"  on disk: {disk}")
    # cal covers everything on disk, so the join can never be what excludes a
    # month; only the explicit WHERE below can.
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("""CREATE TABLE cal(
        ym INTEGER, dow_n TINYINT, n_days TINYINT, n_holiday TINYINT,
        holiday_frac DOUBLE, dose DOUBLE, dose_alt DOUBLE)""")
    con.executemany("INSERT INTO cal VALUES (?,?,?,?,?,?,?)",
                    [(r["ym"], r["dow_n"], r["n_days"], r["n_holiday"],
                      r["holiday_frac"], r["dose"], r["dose_alt"])
                     for r in cell_table(disk or YMS)])
    where = f"WHERE r.ym IN ({','.join(map(str, ym_filter))})"
    con.execute(f"""
        CREATE VIEW m AS
        SELECT r.*, c.n_days, c.n_holiday, c.holiday_frac, c.dose, c.dose_alt,
               r.pop / c.n_days                       AS pop_day,
               r.o_dong // 1000                       AS o_gu,
               r.d_dong // 1000                       AS d_gu,
               (r.o_dong BETWEEN 1101000 AND 1125999) AS o_seoul,
               (r.d_dong BETWEEN 1101000 AND 1125999) AS d_seoul,
               r.dow_n >= 6                           AS weekend
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false) r
        JOIN cal c ON c.ym = r.ym AND c.dow_n = r.dow_n
        {where}
    """)
    return con
