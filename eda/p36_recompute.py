#!/usr/bin/env python
"""Phase 36 — an independent second implementation of the headline numbers.

WHY. The 8-19 letter's closing paragraph: three times in one round a level came
out wrong while the ratio built on it still looked reasonable, and the answer to
that is not a bigger gate. p31 checks that the report quotes results_*.json
correctly; it cannot check that results_*.json is correct, because it reads the
same file the report does. So the headline numbers are computed a SECOND time
here, by a different route, and the two are compared.

WHAT "INDEPENDENT" MEANS HERE, precisely, because a vague claim of independence
is worse than none:

  * the passive matrix is built in SQL, as a self-join that forms
    sum_l n_a n_a' / n_. inside the database. p26 builds it by pivoting a cached
    pandas frame and multiplying two numpy arrays. Neither line of code is shared
    and neither intermediate file is read.
  * the arrival cache in derived/ is NOT used. This script goes back to the
    parquet every time, so a stale or mis-keyed cache cannot pass.
  * the calendar is re-derived from the `holidays` package and Python's own
    `calendar` module, not from calendar_kr.py. Weekday-occurrence counts and the
    holiday-free weekday set are the two inputs that silently rescale everything
    downstream -- per-day normalisation and the survey day-set alignment -- so
    they are the last thing that should be taken on trust.
  * the survey matrix is read with the csv module into dictionaries and counted
    with plain loops. p27 uses pandas merges and np.add.at.
  * every statistic -- assortativity, mutual information, the dominant eigenvalue
    -- is re-implemented in plain Python from its definition. Assortativity is
    computed from the definition sum_a (e_aa - r_a^2) / (1 - sum r_a^2) with
    explicit loops, and the dominant eigenvalue by power iteration rather than
    by numpy.linalg.eig.

COVERAGE. Sections 36.1-36.5 cover the matrix ladder, the survey comparison, the
four headline ratios, the coverage profile and the calendar; 36.3 also carries
p32's survey spectrum and, in 36.3c, p61's permutation null. Sections 36.6-36.12
extend the same treatment to the seven other scripts the report quotes:

  36.6  p21's rho-hat        -- from raw parquet, with regpop read in SQL rather
                                than through pandas
  36.7  p29's measured fill  -- our side rebuilt from RAW parquet rather than
                                from derived/gu_level.parquet, and the official
                                gu archive re-read with csv/dicts rather than
                                pandas. This is the one that tests gu_level.parquet
                                itself, which nothing else does.
  36.8  p19's E-share        -- also from raw parquet rather than gu_level.parquet
  36.9  p34's arithmetic     -- the Monte Carlo cannot be re-simulated, so its
                                two deterministic anchors are checked against the
                                matrices section 36.2 built, and every quantity
                                DERIVED from the stored sweep (local slopes, the
                                polygon interval, the arm-gap ratios, the log10
                                spans) is recomputed. That splits "the simulation"
                                from "the arithmetic on top of it", and the
                                arithmetic is where transcription errors live.
  36.10 p58's semester p     -- the month labels come out of the ym instead of
                                out of p41's tuples and p42's stored column, and
                                the 79 rotations are enumerated as string
                                rotations counted with collections.Counter
  36.11 p59's beta*          -- the crossing is found by bisecting a forward
                                monotone interpolant instead of by p39's
                                closed-form `invert`, and the claim that uses no
                                interpolation at all is checked separately by
                                re-scanning results_p44/p53 for every beta
  36.12 p60's age collapse   -- the 3-bin matrix is summed by a pandas groupby
                                on band labels rather than by the matrix product
                                S A S', and the mechanism claim (proportionate
                                mixing must collapse to r = 0) is rebuilt from
                                its own definition
  36.13 the two abstract numbers that had no second route at all -- p37's
                                rank-one gap without an SVD, both the difference
                                of norms SI 3 quotes ("at most 0.00202") and the
                                distance the abstract quotes since 2026-09-05
                                ("at most 0.0258", p67's object), and
                                data_inventory's 79 months and 10.19 billion
                                rows off the ym COLUMN rather than off the file
                                names and the parquet footers. data_inventory is
                                also the one producer in this project with no
                                assert of its own, so until 36.13 nothing but
                                the drive itself stood behind those counts.

WHAT IS SHARED, and therefore not checked here: DuckDB's parquet reader, and the
ETL that wrote the parquet. The ETL already checks itself file by file against
`wc -l` on the source CSVs (etl_report.json), which is a stronger check than a
second implementation would be. Section 36.9's sweep medians are also taken as
given; re-running 200 replicates x 424 merge levels x six months in a second
implementation would test the random number generator, not the result.

A DISAGREEMENT IS A FAILURE, not a diagnostic. Every comparison below is an
assert with a stated tolerance, and the tolerances are RELATIVE -- an absolute
tolerance on a quantity of order 0.005 waves through a 60% error, which is the
mistake the 8-19 round found in this project's own gate.

36.10, 36.12 and 36.13 read their month series from results_p42.json and
results_p37.json and do NOT rebuild them: at 79 months section 36.2's route is a
billion-row scan. All three say so in their own output rather than leaving it to
be inferred, and 36.13 additionally recomputes its gap for the two months 36.2
does rebuild from the parquet, so the abstract's 0.00202 has one leg on raw rows
even though seventy-seven of the months do not. Section 36.3c is the one block here that is not an anchor at
all -- a permutation null on a different stream cannot reproduce a median bit
for bit -- so it compares within Monte Carlo error and asserts only the verdict,
the way 45.4 does.

    python eda/p36_recompute.py [--months 202312,202402]
    python eda/p36_recompute.py --sections 1,2,3,4,9    # skip the full scans

REPRODUCIBILITY. This script runs DuckDB single-threaded, because at more than
one thread its own answers drift in the last ulp; the measurement and the reason
are in main(). `eda/determinism_check.sh p36_recompute` is the check.
"""
import argparse
import calendar as pycal
import collections
import csv
import datetime as dt
import io
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb

from paths import DATA_ROOT, DERIVED, PARQUET_GLOB, ROOT, require

# Restated here rather than imported: an age-label table that drifted between
# two scripts would line the matrices up wrongly and every statistic would still
# be computable.
AGES = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
LBL = ["0-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
       "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
NA = len(AGES)
IMP = 1.5
SEOUL_LO, SEOUL_HI = 1101000, 1125999
ALL_SECTIONS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                "12", "13"}
# 36.3c draws its own permutation null. The seed is deliberately NOT p61's
# 20260827: two runs of the same stream agreeing says nothing, and the point of
# that block is to see whether a different stream reaches the same verdict.
PERM_MC = 500
PERM_SEED = 20260828
# Monte Carlo tolerances for 36.3c, and only for 36.3c. The reasoning is in the
# block itself; they are stated as constants so nobody has to hunt for the two
# numbers that were loosened and why.
MC_RTOL_MED = 0.05
MC_RTOL_P95 = 0.10
CHAE = DATA_ROOT / "raw" / "chae2026"
DIARY_WEEKS = {202312: (dt.date(2023, 12, 6), dt.date(2023, 12, 12)),
               202402: (dt.date(2024, 2, 7), dt.date(2024, 2, 13))}

fails, checks = [], []


def _record(label, mine, theirs, rel, ok, rtol):
    """The one place a check enters the list, so every entry has the same shape.

    There used to be two appends with different arities; the summary indexes
    position 5 and the short one raised IndexError only after the twelve-month
    scan in 36.6 had finished.
    """
    checks.append((label, mine, theirs, rel, ok, rtol))
    if not ok:
        fails.append(label)


def compare(label, mine, theirs, rtol=1e-9):
    """Relative comparison, always. See the module docstring for why."""
    if theirs in (None, 0) and mine in (None, 0):
        rel = 0.0
    elif theirs in (None, 0):
        rel = float("inf")
    else:
        rel = abs(mine - theirs) / abs(theirs)
    ok = rel <= rtol
    _record(label, mine, theirs, rel, ok, rtol)
    return ok


# ------------------------------------------------------- the calendar, re-derived
def kr_holidays(year):
    """Public holidays, from the `holidays` package rather than calendar_kr."""
    import holidays as H
    return set(H.country_holidays("KR", years=year).keys())


def weekday_days(ym):
    """{isoweekday: how many times it occurs in this month}, from `calendar`."""
    y, m = divmod(ym, 100)
    n = pycal.monthrange(y, m)[1]
    out = defaultdict(int)
    for day in range(1, n + 1):
        out[dt.date(y, m, day).isoweekday()] += 1
    return dict(out)


def holiday_free(ym):
    """Weekdays of `ym` on which no public holiday falls."""
    y, m = divmod(ym, 100)
    hol = kr_holidays(y)
    bad = {d.isoweekday() for d in hol if d.year == y and d.month == m}
    return sorted(set(range(1, 8)) - bad)


# ------------------------------------------------- the statistics, from definition
def assortativity(T):
    """(tr e - sum r_a^2) / (1 - sum r_a^2), by explicit summation."""
    s = 0.0
    for row in T:
        for v in row:
            s += v
    if s == 0:
        return float("nan")
    e = [[v / s for v in row] for row in T]
    r = [sum(row) for row in e]
    tr = sum(e[i][i] for i in range(len(e)))
    rr = sum(x * x for x in r)
    return (tr - rr) / (1.0 - rr) if rr != 1.0 else float("nan")


def mutual_information_bits(T):
    s = sum(sum(row) for row in T)
    if s == 0:
        return float("nan"), float("nan")
    e = [[v / s for v in row] for row in T]
    r = [sum(row) for row in e]
    c = [sum(e[i][j] for i in range(len(e))) for j in range(len(e[0]))]
    mi = 0.0
    for i in range(len(e)):
        for j in range(len(e[0])):
            if e[i][j] > 0 and r[i] > 0 and c[j] > 0:
                mi += e[i][j] * math.log2(e[i][j] / (r[i] * c[j]))
    h = -sum(x * math.log2(x) for x in r if x > 0)
    return mi, (mi / h if h else float("nan"))


def dominant_eigenvalue(M, iters=20000, tol=1e-14):
    """Power iteration. A non-negative matrix has a non-negative dominant
    eigenvector, so no deflation or sorting is needed and there is nothing for a
    library's eigenvalue ordering convention to get wrong."""
    n = len(M)
    v = [1.0 / n] * n
    lam = 0.0
    for _ in range(iters):
        w = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
        nrm = math.sqrt(sum(x * x for x in w))
        if nrm == 0:
            return 0.0
        w = [x / nrm for x in w]
        new = sum(w[i] * sum(M[i][j] * w[j] for j in range(n)) for i in range(n))
        if abs(new - lam) < tol * max(1.0, abs(new)):
            return new
        lam, v = new, w
    return lam


def symmetrise(C, pop):
    n = len(C)
    T = [[pop[i] * C[i][j] for j in range(n)] for i in range(n)]
    return [[(T[i][j] + T[j][i]) / 2 for j in range(n)] for i in range(n)]


# ------------------------------------------------------- the passive matrix, in SQL
def passive_matrix(con, ym, level, dows, imp=IMP):
    """A[a,a'] = sum_l n_a(l) n_a'(l) / n_.(l), averaged over `dows`.

    The whole reduction happens in the database as a self-join on the cell key.
    The dedupe is the one thing that has to match p26's intent rather than its
    code: the product's primary key is not unique (Phase 16 found 475,266 exact
    duplicate pairs), so rows are collapsed to one per full key first. min(pop)
    on identical rows is the identity, and doing it any later double-counts.
    """
    nd = weekday_days(ym)
    loc = {"dong": "d_dong", "gu": "d_dong // 1000", "city": "0"}[level]
    dow_in = ",".join(str(d) for d in dows)
    nd_case = " ".join(f"WHEN {d} THEN {nd[d]}" for d in dows)
    sql = f"""
    WITH dedup AS (
      SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
             min(pop) AS pop, bool_or(masked) AS masked
      FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
      WHERE ym = {ym}
        AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND dow_n IN ({dow_in})
        AND substr(mtype, 2, 1) <> 'H'
      GROUP BY ALL),
    cell AS (
      SELECT {loc} AS loc, arr_hour, dow_n, age,
             (coalesce(sum(pop), 0)
              + {imp} * count(*) FILTER (WHERE masked))
             / (CASE dow_n {nd_case} END) AS n
      FROM dedup GROUP BY 1,2,3,4),
    tot AS (
      SELECT loc, arr_hour, dow_n, sum(n) AS t
      FROM cell GROUP BY 1,2,3)
    SELECT x.age AS a, y.age AS b, sum(x.n * y.n / tot.t) AS v
    FROM cell x
      JOIN cell y ON x.loc = y.loc AND x.arr_hour = y.arr_hour
                 AND x.dow_n = y.dow_n
      JOIN tot ON tot.loc = x.loc AND tot.arr_hour = x.arr_hour
              AND tot.dow_n = x.dow_n
    WHERE tot.t > 0
    GROUP BY 1,2"""
    A = [[0.0] * NA for _ in range(NA)]
    idx = {a: i for i, a in enumerate(AGES)}
    for a, b, v in con.execute(sql).fetchall():
        A[idx[a]][idx[b]] = v / len(dows)
    return A


def coverage_p10(con, ym):
    """The per-cell weekday minimum, 10th percentile by age -- p9's estimator,
    but with the order statistic taken in Python instead of by quantile_cont."""
    rows = con.execute(f"""
        WITH c AS (
          SELECT age, min(pop) AS w,
                 bit_count(bit_or(1::UBIGINT << dow_n)) AS n_dow
          FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
          WHERE ym = {ym}
            AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
            AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
            AND NOT masked
          GROUP BY o_dong, d_dong, arr_hour, mtype, age, sex)
        SELECT age, w FROM c WHERE n_dow >= 4""").fetchall()
    by = defaultdict(list)
    for a, w in rows:
        by[a].append(w)
    out = {}
    for a, v in by.items():
        v.sort()
        # linear interpolation between order statistics, the same convention
        # quantile_cont uses, written out so the two cannot agree by sharing it
        h = 0.10 * (len(v) - 1)
        lo = int(math.floor(h))
        hi = min(lo + 1, len(v) - 1)
        out[a] = v[lo] + (h - lo) * (v[hi] - v[lo])
    return out


# --------------------------------------------------- the survey matrix, in dicts
def survey_matrix(ym, dows, seoul_only=True):
    """Mean alters of band b per respondent of band a per day, from the CSVs.

    Read with the csv module into dictionaries and counted with loops; no pandas
    and no numpy anywhere in this path.
    """
    require(CHAE, "Chae microdata (run eda/dl_chae.py)")

    def rd(name):
        txt = (CHAE / name).read_bytes().decode("utf-8-sig")
        r = csv.reader(io.StringIO(txt))
        head = [h.strip() for h in next(r)]
        return [dict(zip(head, row)) for row in r]

    def parse_date(v):
        """The file carries MMDD with no year: '1206' is 2023-12-06 and '207' is
        2024-02-07. Restated rather than imported, since a wrong year here would
        silently empty one month and leave the other looking fine."""
        v = str(v).strip()
        if len(v) == 4 and v.startswith("12"):
            return dt.date(2023, 12, int(v[2:]))
        if len(v) == 3 and v.startswith("2"):
            return dt.date(2024, 2, int(v[1:]))
        raise ValueError(f"unparseable diary date {v!r}")

    def band_idx(age):
        a = 0 if age < 10 else (80 if age >= 80 else (age // 5) * 5)
        if 10 <= a < 15:
            a = 10
        return AGES.index(a)

    ego = {}
    for row in rd("results_preliminary_survey.csv"):
        try:
            age = int(row["SQ1"])
        except (ValueError, KeyError):
            continue
        if seoul_only and row.get("SQ3") != "1":
            continue
        ego[row["ID"]] = band_idx(age)

    # place code -> panel, restated: 1 household, 2 work, 3 school, 4-8 other
    panel_of_place = {1: "H", 2: "W", 3: "W", 4: "E", 5: "E", 6: "E", 7: "E",
                      8: "E"}
    counts = defaultdict(lambda: [0.0] * NA)
    for row in rd("results_main_survey.csv"):
        eid = row["ID"]
        if eid not in ego:
            continue
        try:
            aage = int(row["Q1"])
        except (ValueError, KeyError):
            continue
        panel = None
        for k in range(1, 9):
            v = (row.get(f"Q5_{k}") or "").strip()
            if v.isdigit() and int(v) in panel_of_place:
                panel = panel_of_place[int(v)]
                break
        if panel not in ("W", "E"):
            continue
        try:
            d = parse_date(row["Date"])
        except (ValueError, KeyError):
            continue
        if d.year * 100 + d.month != ym or d.isoweekday() not in dows:
            continue
        counts[eid][band_idx(aage)] += 1.0

    lo, hi = DIARY_WEEKS[ym]
    n_days = sum(1 for k in range((hi - lo).days + 1)
                 if (lo + dt.timedelta(days=k)).isoweekday() in dows)
    # One row per ego: (its own band, its alter tally). The matrix is built
    # FROM these rows rather than accumulated in place, because 36.3c's
    # permutation null needs exactly this decomposition -- it re-pairs the bands
    # with the tallies and rebuilds through the same function, so the null and
    # the observation cannot be assembled two different ways.
    ego_rows = [(eb, counts.get(eid) or [0.0] * NA) for eid, eb in ego.items()]
    C, n_ego = survey_C(ego_rows, n_days)
    n_contact = sum(sum(v) for v in counts.values())
    assert n_contact > 0, (
        f"the dict reader found no contacts for {ym} on weekdays {dows} -- an "
        f"empty month reads as a zero matrix, not as an error, so this is an "
        f"assert rather than a warning")
    return C, n_ego, n_days, n_contact, ego_rows


def survey_C(ego_rows, n_days):
    """Mean alters of band b per respondent of band a per day, from ego rows."""
    tot = [[0.0] * NA for _ in range(NA)]
    n_ego = [0] * NA
    for eb, c in ego_rows:
        n_ego[eb] += 1
        for j in range(NA):
            tot[eb][j] += c[j]
    C = [[(tot[i][j] / (n_ego[i] * n_days) if n_ego[i] and n_days else 0.0)
          for j in range(NA)] for i in range(NA)]
    return C, n_ego



# ------------------------------------------------- the other four scripts' data
# p19, p21 and p29 all reduce the raw product to a small table. Each is rebuilt
# here from the parquet with SQL written for this file, and -- importantly -- p19
# and p29 are rebuilt WITHOUT going through derived/gu_level.parquet, which is
# what they actually read. Nothing else in this project ever recomputes that
# file, so this is the only place its contents are tested.
#
# DEDUPLICATION IS A REAL FORK, not a detail. The product carries 475,266
# whole-row duplicate pairs (p16). p26's cells() collapses them with a
# min(pop) GROUP BY ALL, so sections 36.2-36.4 dedupe. p1_conservation's
# gu_level.parquet does NOT -- it sums the raw rows -- so p19 and p29 inherit the
# doubled rows. Matching them therefore means NOT deduping, and the honest way to
# handle that is to compute both and report the gap rather than to pick one
# quietly: `dedup=False` reproduces what they published, `dedup=True` says what
# the number would be without the duplicates.
def gu_agg(con, ym, dedup=False):
    """(dow_n, age, mtype) -> unmasked volume and masked-cell count, Seoul-internal.

    The shape p29 gets from gu_level.parquet, taken straight from the parquet.
    """
    src = f"""
      SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
             min(pop) AS pop, bool_or(masked) AS masked
      FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
      WHERE ym = {ym} AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
      GROUP BY ALL""" if dedup else f"""
      SELECT ym, dow_n, age, mtype, pop, masked
      FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
      WHERE ym = {ym} AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}"""
    rows = con.execute(f"""
        WITH src AS ({src})
        SELECT dow_n, age, mtype,
               coalesce(sum(pop), 0)            AS v_lo,
               count(*) FILTER (WHERE masked)   AS mcells
        FROM src GROUP BY 1,2,3""").fetchall()
    return {(int(d), int(a), m): (float(v), int(c)) for d, a, m, v, c in rows}


def official_gu(ym):
    """The official 자치구 archive, read with zipfile + csv + dicts.

    p29 reads it with pandas.read_csv into a frame and groups it. Here it is a
    csv.reader over a cp949 decode, counted into a dict -- no pandas in the path.
    """
    import zipfile
    path = DATA_ROOT / "raw" / "gu" / f"생활이동_자치구_{ym}.zip"
    if not path.exists():
        return None
    dow = {"월": 1, "화": 2, "수": 3, "목": 4, "금": 5, "토": 6, "일": 7}
    seoul = set(range(11010, 11260))
    agg = defaultdict(lambda: [0.0, 0])
    n_rows = n_masked_rows = 0
    with zipfile.ZipFile(path) as zf:
        for name in sorted(n for n in zf.namelist() if not n.endswith("/")):
            text = zf.open(name).read().decode("cp949", errors="replace")
            r = csv.reader(io.StringIO(text))
            next(r, None)                       # header
            for row in r:
                if len(row) < 10:
                    continue
                try:
                    o, d = int(row[3]), int(row[4])
                except ValueError:
                    continue
                if o not in seoul or d not in seoul:
                    continue
                raw = row[9].strip()
                masked = raw == "*"
                try:
                    v = 0.0 if masked else float(raw)
                except ValueError:
                    v = 0.0
                key = (dow.get(row[1].strip(), 0), int(row[6]), row[7].strip())
                cell = agg[key]
                cell[0] += v
                cell[1] += 1 if masked else 0
                n_rows += 1
                n_masked_rows += 1 if masked else 0
    # The masked SHARE p29 reports is over ROWS of the official file, not over
    # the (weekday, age, mtype) groups those rows collapse into -- the two differ
    # by three orders of magnitude, which is how this was caught.
    return {k: (v, m) for k, (v, m) in agg.items()}, n_rows, n_masked_rows


def eshare_agg(con, ym, dedup=False):
    """(age, dest_attr) -> per-DAY unmasked volume and per-day masked-cell count.

    p19 gets this from gu_level.parquet's pop_day_unmasked and
    n_masked_cells / n_days. pop_day is pop divided by that weekday's occurrence
    count in the month, so the division happens per row, before any summing.
    """
    nd = weekday_days(ym)
    nd_case = " ".join(f"WHEN {d} THEN {n}" for d, n in sorted(nd.items()))
    src = f"""
      SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
             min(pop) AS pop, bool_or(masked) AS masked
      FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
      WHERE ym = {ym} AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
      GROUP BY ALL""" if dedup else f"""
      SELECT ym, dow_n, age, mtype, pop, masked
      FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
      WHERE ym = {ym} AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
        AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}"""
    rows = con.execute(f"""
        WITH src AS ({src})
        SELECT age, substr(mtype, 2, 1) AS dest_attr,
               coalesce(sum(pop / (CASE dow_n {nd_case} END)), 0) AS lo,
               coalesce(sum(CASE WHEN masked
                                 THEN 1.0 / (CASE dow_n {nd_case} END)
                                 ELSE 0 END), 0)                  AS nmask_day
        FROM src GROUP BY 1,2""").fetchall()
    return {(int(a), d): (float(lo), float(nm)) for a, d, lo, nm in rows}


def rho_numerator(con, ym, imp=1.5):
    """Daily-average H-origin departures by age band.

    Two definitional points, both of which change the answer and both of which
    are restated here rather than imported: origin in Seoul with the destination
    UNRESTRICTED (a resident who leaves Seoul is still a resident), and the
    monthly sum divided by DAYS IN THE MONTH, not by each weekday's occurrence
    count -- sum(pop_day) would give a representative week, not a day.
    """
    y, m = divmod(ym, 100)
    days = pycal.monthrange(y, m)[1]
    rows = con.execute(f"""
        WITH dedup AS (
          SELECT ym, dow_n, arr_hour, o_dong, d_dong, sex, age, mtype,
                 min(pop) AS pop, bool_or(masked) AS masked
          FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
          WHERE ym = {ym} AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
            AND mtype LIKE 'H%'
          GROUP BY ALL)
        SELECT age,
               (coalesce(sum(pop), 0)
                + {imp} * count(*) FILTER (WHERE masked)) / {days} AS v
        FROM dedup GROUP BY 1""").fetchall()
    return {int(a): float(v) for a, v in rows}


def regpop_by_age(con, ym):
    """The denominator, read in SQL. p21 reads it with pandas and groups there."""
    path = DERIVED / "regpop.parquet"
    require(path, "regpop (run eda/p20_regpop.py)")
    rows = con.execute(f"""
        SELECT age, sum(pop) AS n FROM read_parquet('{path}')
        WHERE ym = {ym} AND dong_code IS NOT NULL GROUP BY 1""").fetchall()
    return {int(a): float(n) for a, n in rows}


# ------------------------------------------- the spectrum, without an SVD
def _power_top(M, iters=4000, rtol=1e-16):
    """Dominant eigenpair of a symmetric matrix, by power iteration.

    p32 takes its singular values from numpy.linalg.svd. Nothing on this path
    calls an SVD: the matrices here are symmetric, so the singular values are
    the square roots of the eigenvalues of M'M, and those are reached by
    iteration and deflation instead.

    A NEGATIVE rtol DISABLES THE EARLY EXIT and is a deliberate caller option,
    not a degenerate argument. The stopping rule watches the Rayleigh quotient,
    which is stationary at the eigenvector: it returns once the EIGENVALUE has
    settled, leaving the VECTOR at about the square root of that tolerance. That
    is all spectrum_2 needs and it is not enough for rank1_pm_gap(), which uses
    the vector itself -- see the note there.
    """
    n = len(M)
    v = [1.0 / math.sqrt(n)] * n
    lam = None
    for _ in range(iters):
        w = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
        nrm = math.sqrt(sum(x * x for x in w))
        if nrm == 0:
            return 0.0, v
        v = [x / nrm for x in w]
        new = sum(v[i] * sum(M[i][j] * v[j] for j in range(n)) for i in range(n))
        if lam is not None and abs(new - lam) <= rtol * abs(new):
            return new, v
        lam = new
    return lam, v


def spectrum_2(T):
    """(sigma1^2 / sum sigma^2, sigma2 / sigma1) for a symmetric matrix.

    Both are scale-free, so M = e'e is divided by its own trace -- which is
    sum_i sigma_i^2 -- before anything else. The largest eigenvalue of the
    scaled matrix then IS sigma1_share, and the power iteration's stopping rule
    becomes relative rather than absolute, which is what makes the agreement
    below meaningful at 1e-16 rather than at 1e-14.
    """
    s = sum(sum(row) for row in T)
    if s == 0:
        return float("nan"), float("nan")
    e = [[v / s for v in row] for row in T]
    n = len(e)
    M = [[sum(e[k][i] * e[k][j] for k in range(n)) for j in range(n)]
         for i in range(n)]
    tr = sum(M[i][i] for i in range(n))
    if tr == 0:
        return float("nan"), float("nan")
    M = [[M[i][j] / tr for j in range(n)] for i in range(n)]
    l1, v1 = _power_top(M)
    D = [[M[i][j] - l1 * v1[i] * v1[j] for j in range(n)] for i in range(n)]
    l2, _ = _power_top(D)
    return l1, (math.sqrt(max(0.0, l2) / l1) if l1 else float("nan"))


def frobenius(M):
    """||M||_F by explicit double summation. p32 calls numpy.linalg.norm."""
    return math.sqrt(sum(v * v for row in M for v in row))


def rank1_residuals(T):
    """(sigma1^2/sum sigma^2, best rank-1 residual, proportionate-mixing residual).

    These are the three quantities behind the abstract's "departs from rank one
    by at most 0.00202": the gap is |best - pm|, a DIFFERENCE of two numbers near
    0.134, so the fourth significant figure of the answer is the thirteenth of
    its inputs and a route that shares an SVD with p32 would not be testing much.
    Nothing here calls one: the share comes out of spectrum_2's power iteration
    and deflation on e'e, and the proportionate-mixing residual is a Frobenius
    norm written as an explicit sum over cells.
    """
    s = sum(sum(row) for row in T)
    if s == 0:
        return float("nan"), float("nan"), float("nan")
    e = [[v / s for v in row] for row in T]
    share, _ = spectrum_2(T)
    best = math.sqrt(max(0.0, 1.0 - share))
    r = [sum(row) for row in e]
    n = len(e)
    D = [[e[i][j] - r[i] * r[j] for j in range(n)] for i in range(n)]
    return share, best, frobenius(D) / frobenius(e)


def rank1_pm_gap(T):
    """||sigma_1 u_1 u_1' - q (x) q||_F / ||e||_F: how far the best rank-one
    approximation is from the proportionate-mixing null. The abstract's 0.0257.

    NOT rank1_residuals()'s third value, and the difference between them is the
    correction the 2026-09-05 round made. That one is |best - pm|, a DIFFERENCE
    OF NORMS, which the triangle inequality makes a LOWER bound on this distance;
    the abstract used to quote it as an upper one. Both are kept: SI 3 still
    prints the difference, labelled as the lower bound it is.

    THE SECOND ROUTE, which is the reason this lives here. p67_rank1gap.py takes
    the leading pair off numpy.linalg.svd. Here it comes out of the same power
    iteration spectrum_2 uses -- e is symmetric and non-negative, so its dominant
    eigenpair IS its top singular pair -- and both norms are explicit sums.

    THE ITERATION IS RUN TO A FIXED COUNT, deliberately, which is the one place
    this file asks _power_top for something spectrum_2 never needed. Its stopping
    rule leaves the vector at about sqrt(1e-16); the eigenvalue is all a share is
    made of, but this quantity depends on the vector linearly. Measured against
    p67 over the 79 months: the early exit costs 7.8e-08, a fixed hundred
    iterations costs 2.1e-13, and 400 and 1000 iterations do not improve on it.
    """
    s = sum(sum(row) for row in T)
    if s == 0:
        return float("nan")
    e = [[v / s for v in row] for row in T]
    n = len(e)
    l1, u = _power_top(e, iters=100, rtol=-1.0)
    q = [sum(row) for row in e]
    D = [[l1 * u[i] * u[j] - q[i] * q[j] for j in range(n)] for i in range(n)]
    return frobenius(D) / frobenius(e)


def below_80(A):
    """p37's T for the rank statistics: drop the 80+ band, then symmetrise.

    p37 forms T as pop * (A[sq, sq] / pop) and averages it with its transpose.
    The population enters and leaves in the same order, so it cancels exactly and
    is never introduced here at all -- which is the point rather than a shortcut:
    the rank statistics cannot be sensitive to a denominator that divides out,
    and a route that carries one would hide that.
    """
    n = NA - 1                       # AGES[-1] is 80, the band p32's `sq` drops
    return [[(A[i][j] + A[j][i]) / 2 for j in range(n)] for i in range(n)]


def month_span(first, last):
    """The months from `first` to `last`, marched one real calendar month at a
    time with `calendar.monthrange`.

    data_inventory builds the same list by modular arithmetic on y * 100 + m.
    Walking the actual dates instead is what makes "79 CONSECUTIVE months" -- the
    abstract's word -- a checked claim rather than a restatement of the same
    two-line comprehension.
    """
    d = dt.date(first // 100, first % 100, 1)
    out = []
    while d.year * 100 + d.month <= last:
        out.append(d.year * 100 + d.month)
        d += dt.timedelta(days=pycal.monthrange(d.year, d.month)[1])
    return out


def percentile(v, q):
    """The q-th percentile with numpy's default linear convention, written out
    so the two implementations cannot agree by sharing one."""
    s = sorted(v)
    h = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def median(v):
    """The middle order statistic. numpy.median returns exactly this for an odd
    n, and both 79 and 500 are handled here the same way numpy handles them."""
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# -------------------------------------- a forward interpolant and a root finder
def pw_linear(xs, ys):
    """The piecewise-linear function through (xs, ys), xs increasing.

    p39's `invert` solves this same interpolant algebraically and in closed
    form, on whichever segment brackets the target. Here it is only ever
    evaluated FORWARDS and the crossing is found by bisection, so the two share
    a definition and no arithmetic.
    """
    def g(x):
        if x <= xs[0]:
            return ys[0]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + t * (ys[i] - ys[i - 1])
        return ys[-1]
    return g


def bisect_root(f, lo, hi, iters=200):
    """The sign change of f on [lo, hi], bisected to the last representable bit."""
    flo, fhi = f(lo), f(hi)
    assert (flo <= 0.0 <= fhi) or (fhi <= 0.0 <= flo), (
        f"bisection was handed a bracket that does not straddle zero: "
        f"f({lo}) = {flo}, f({hi}) = {fhi}")
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if mid == lo or mid == hi:
            break
        fm = f(mid)
        if (fm < 0.0) == (flo < 0.0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return (lo + hi) / 2.0


# ------------------------------------------- the age collapse, by groupby
LIM_BANDS = [[0, 1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]


def nested_groups(k):
    """The dyadic rung with k groups: k adjacent blocks of 16 // k bands."""
    assert NA % k == 0, "the dyadic ladder needs k to divide 16"
    step = NA // k
    return [list(range(i * step, (i + 1) * step)) for i in range(k)]


def collapse_groupby(A, groups):
    """A summed into the coarse partition `groups`, by a pandas groupby.

    p60 collapses with the matrix product S A S'. Here the 256 entries go into a
    long frame keyed by (group of i, group of j) and are summed by group. Same
    arithmetic, different route -- and with no S there is nothing for a wrongly
    built aggregation matrix to be quietly right about.
    """
    import pandas as pd
    of = {i: g for g, gr in enumerate(groups) for i in gr}
    n = len(groups)
    frame = pd.DataFrame({
        "g": [of[i] for i in range(NA) for _ in range(NA)],
        "h": [of[j] for _ in range(NA) for j in range(NA)],
        "v": [A[i][j] for i in range(NA) for j in range(NA)]})
    s = frame.groupby(["g", "h"], sort=True)["v"].sum()
    return [[float(s.loc[(g, h)]) for h in range(n)] for g in range(n)]


def proportionate_mixing(A):
    """E[i][j] = R_i R_j / R_., the proportionate-mixing matrix with A's margins.

    p60 builds it as outer(e.sum(1), e.sum(1)) * A.sum() after normalising; this
    forms it straight from the raw row sums, so the two do not share the
    normalise-then-rescale step whose cancellation is exactly what the null is
    being used to test.
    """
    R = [sum(row) for row in A]
    S = sum(R)
    return [[R[i] * R[j] / S for j in range(len(R))] for i in range(len(R))]


# ------------------------------------------- the school calendar, from the ym
TERM_MONTHS = frozenset((3, 4, 5, 6, 9, 10, 11))
VAC_MONTHS = frozenset((1, 2, 7, 8))


def semester_letter(ym):
    """'T' / 'V' / 'D' from the ym itself.

    p58 imports TERM and VAC from p41_semester and reads rows[]["semester"] out
    of results_p42.json. Neither happens here: the month comes out of the ym by
    the same divmod(ym, 100) section 36.1 uses on the calendar, and the two
    Korean school-term sets are restated -- a table that drifted between two
    scripts would relabel the calendar and every rotation built on it, and the
    rotation would still run. December is in neither set.
    """
    _, m = divmod(ym, 100)
    return "T" if m in TERM_MONTHS else ("V" if m in VAC_MONTHS else "D")


def json_leaves(obj, key=None):
    """Every scalar leaf of a decoded JSON document, with the key it sits under."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from json_leaves(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from json_leaves(v, key)
    else:
        yield key, obj


def pw_linear_shift(xs, ys, target):
    """pw_linear(xs, ys) minus a constant -- the function whose root is wanted."""
    g = pw_linear(xs, ys)
    return lambda x: g(x) - target


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="202312,202402")
    ap.add_argument("--skip-coverage", action="store_true",
                    help="9.6's recompute is a full-month scan; skip for a fast run")
    ap.add_argument("--sections", default="1,2,3,4,5,6,7,8,9,10,11,12,13",
                    help="which sections to run; 6/7/8 are full-month parquet "
                         "scans and 6 covers all twelve months of 2020, while "
                         "10/11/12 read only results files and are seconds. 13 "
                         "is mostly a results-file section but ends in a GROUP "
                         "BY over the ym column of all 79 months, which is the "
                         "only place the 10.19 billion rows are counted rather "
                         "than read out of the parquet footers")
    ap.add_argument("--threads", type=int, default=1,
                    help="DuckDB threads. 1 by default and deliberately -- see "
                         "the comment in main(); >1 makes this script's own "
                         "output drift in the last ulp.")
    args = ap.parse_args()
    yms = [int(x) for x in args.months.split(",")]
    sections = {x.strip() for x in args.sections.split(",") if x.strip()}
    unknown = sections - ALL_SECTIONS
    if unknown:
        raise SystemExit(f"unknown section(s): {sorted(unknown)}; "
                         f"valid are {sorted(ALL_SECTIONS)}")

    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))
    p32 = json.load(open(f"{ROOT}/eda/results_p32.json"))
    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))
    p61 = json.load(open(f"{ROOT}/eda/results_p61.json"))
    pops = {int(k): v for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]

    # SINGLE-THREADED ON PURPOSE, and the reason is a finding rather than a
    # preference. DuckDB's parallel hash aggregate combines partial sums in
    # whatever order the threads finish, and floating-point addition is not
    # associative, so the 18-million-row self-join in passive_matrix() returns
    # answers that differ in the last one or two units in the last place from run
    # to run. Measured: at threads=6 three runs of the same join disagree by
    # ~3e-16 per cell; at threads=1 three runs are bit-identical. Nothing about
    # any conclusion moves at 1e-15 -- every check here passes either way with
    # room to spare -- but a script whose job is to be the reproducible second
    # opinion cannot itself be irreproducible. It costs about four seconds per
    # matrix.
    #
    # This applies to p26 as well, whose arrivals cache is built by a
    # multi-threaded DuckDB aggregate: results_p26.json carries one particular
    # realisation of that jitter, and deleting the cache and re-running could
    # move the published assortativity in its fifteenth digit. Worth knowing;
    # not worth acting on.
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA disable_progress_bar")

    matrices_36_2 = {}

    # ------------------------------------------------- 36.1 the calendar itself
    if "1" in sections:
        print("=== 36.1 the calendar, re-derived from `holidays` and `calendar` ===")
    import calendar_kr as K
    for ym in (yms + [202001, 202012]) if "1" in sections else []:
        mine_nd = weekday_days(ym)
        for d in range(1, 8):
            compare(f"{ym} dow {d} occurrences", mine_nd[d],
                    K.cell_exposure(ym, d)["n_days"])
        mine_hf = holiday_free(ym)
        theirs_hf = [d for d in range(1, 8)
                     if K.cell_exposure(ym, d)["n_holiday"] == 0]
        ok = mine_hf == theirs_hf
        # 0.0 tolerance: the weekday set is either the same list or it is not.
        # Six fields, like every other entry -- the summary and the JSON both
        # index position 5, and a five-tuple here took out a 25-minute run.
        _record(f"{ym} holiday-free weekdays", mine_hf, theirs_hf,
                0.0 if ok else 1.0, ok, 0.0)
        print(f"  {ym}: occurrences {[mine_nd[d] for d in range(1, 8)]}, "
              f"holiday-free {mine_hf} "
              f"{'(agrees)' if ok else f'(calendar_kr says {theirs_hf})'}")

    # -------------------------------------------- 36.2 the passive matrix in SQL
    if "2" in sections:
        print("\n=== 36.2 the passive co-arrival matrix, rebuilt in SQL ===")
    mine = matrices_36_2
    for ym in (yms if "2" in sections else []):
        hf = holiday_free(ym)
        for level, dows, key in (
                ("dong", hf, f"{ym}|WE|dong|holidayfree"),
                ("dong", list(range(1, 8)), f"{ym}|WE|dong"),
                ("gu", list(range(1, 8)), f"{ym}|WE|gu"),
                ("city", list(range(1, 8)), f"{ym}|WE|city")):
            A = passive_matrix(con, ym, level, dows)
            mine[key] = A
            pub = p26["matrices"][key]
            r = assortativity(A)
            pop = pops[ym]
            C = [[A[i][j] / pop[i] for j in range(NA)] for i in range(NA)]
            lam = dominant_eigenvalue(C)
            compare(f"{key} assortativity", r, pub["assortativity"], 1e-6)
            compare(f"{key} lead eigenvalue", lam, pub["lead_eigenvalue"], 1e-6)
            print(f"  {key:>34}  r {r:+.8f} (published {pub['assortativity']:+.8f})"
                  f"  lambda {lam:.6f} (published {pub['lead_eigenvalue']:.6f})")

    # ------------------------------------------------- 36.3 the survey matrix
    if "3" in sections:
        print("\n=== 36.3 the survey matrix, rebuilt from the CSVs with dicts ===")
    surv = {}
    for ym in (yms if "3" in sections else []):
        hf = holiday_free(ym)
        C, n_ego, n_days, n_contact, ego_rows = survey_matrix(ym, hf)
        surv[ym] = C
        pub = p27["survey"][f"{ym}|WE|seoul"]
        compare(f"{ym} survey mean contacts", sum(sum(r) for r in C) / NA,
                pub["mean_contacts"], 2e-3)
        compare(f"{ym} survey diary days", float(n_days), float(pub["n_days"]))
        compare(f"{ym} survey egos", float(sum(n_ego)), float(sum(pub["n_ego"])))
        print(f"  {ym}: {sum(n_ego)} Seoul egos, {n_days} diary days, "
              f"{int(n_contact):,} contacts, mean contacts "
              f"{sum(sum(r) for r in C) / NA:.4f} "
              f"(published {pub['mean_contacts']:.4f})")

        # --- 36.3b the spectrum, by power iteration rather than an SVD
        Tsq = symmetrise([[C[i][j] for j in sq] for i in sq],
                         [pops[ym][i] for i in sq])
        s1, s21 = spectrum_2(Tsq)
        sp = p32["spectrum"][f"survey|{ym}|WE|seoul"]
        compare(f"{ym} survey sigma1 share", s1, sp["sigma1_share"], 1e-6)
        compare(f"{ym} survey sigma2/sigma1", s21, sp["sigma2_over_sigma1"],
                1e-6)
        print(f"  {'':>6}  sigma1 share {s1:.12f} (published "
              f"{sp['sigma1_share']:.12f}), sigma2/sigma1 {s21:.12f} "
              f"(published {sp['sigma2_over_sigma1']:.12f})")

        # --- 36.3c p61's permutation null, on this script's OWN stream.
        # NOT AN ANCHOR, and it must not be dressed as one. A permutation null
        # drawn from a different generator under a different seed cannot
        # reproduce a median bit for bit, and reusing p61's seed would prove
        # only that two copies of one draw agree. The tolerances are therefore
        # Monte Carlo tolerances, declared with their arithmetic: p61 reports
        # sd 0.0178 over 500 draws, so the standard error of the median is
        # about 1.25 sd / sqrt(500) = 0.0010, which is 0.8% of it; two
        # independent runs differ by sqrt(2) of that, so 5% is a comfortable
        # three-sigma band, and the p95 moves more than the median so it gets
        # 10%. What is ASSERTED is the verdict -- whether the observed value
        # escapes its own null -- because that is what the paper reads and it
        # does not depend on the stream. Same treatment p45 gives its bootstrap
        # in 45.4.
        v21 = p61["cells"][str(ym)]["verdict_sigma2_over_sigma1"]
        v1s = p61["cells"][str(ym)]["verdict_sigma1_share"]
        rnd = random.Random(PERM_SEED + ym)
        bands = [eb for eb, _ in ego_rows]
        tallies = [c for _, c in ego_rows]
        draw21, draw1s = [], []
        for _ in range(PERM_MC):
            rnd.shuffle(bands)
            Cp, _ = survey_C(list(zip(bands, tallies)), n_days)
            Tp = symmetrise([[Cp[i][j] for j in sq] for i in sq],
                            [pops[ym][i] for i in sq])
            a, b = spectrum_2(Tp)
            draw1s.append(a)
            draw21.append(b)
        med21, p95_21 = median(draw21), percentile(draw21, 95)
        med1s, p05_1s = median(draw1s), percentile(draw1s, 5)
        compare(f"{ym} perm null median sigma2/sigma1 [MC, not an anchor]",
                med21, v21["null_median"], MC_RTOL_MED)
        compare(f"{ym} perm null p95 sigma2/sigma1 [MC, not an anchor]",
                p95_21, v21["null_threshold"], MC_RTOL_P95)
        compare(f"{ym} perm null median sigma1 share [MC, not an anchor]",
                med1s, v1s["null_median"], MC_RTOL_MED)
        compare(f"{ym} perm null p5 sigma1 share [MC, not an anchor]",
                p05_1s, v1s["null_threshold"], MC_RTOL_P95)
        esc21, want21 = bool(s21 > p95_21), bool(v21["primary_escape"])
        esc1s, want1s = bool(s1 < p05_1s), bool(v1s["primary_escape"])
        _record(f"{ym} sigma2/sigma1 escapes its own permutation null",
                esc21, want21, 0.0 if esc21 == want21 else 1.0,
                esc21 == want21, 0.0)
        _record(f"{ym} sigma1 share escapes its own permutation null",
                esc1s, want1s, 0.0 if esc1s == want1s else 1.0,
                esc1s == want1s, 0.0)
        print(f"  {'':>6}  own-stream permutation null: {PERM_MC} shuffles, "
              f"seed {PERM_SEED + ym}, random.Random rather than numpy")
        print(f"  {'':>6}    sigma2/sigma1 null median {med21:.6f} p95 "
              f"{p95_21:.6f}  (p61 {v21['null_median']:.6f} / "
              f"{v21['null_threshold']:.6f}, apart by "
              f"{abs(med21 - v21['null_median']) / v21['null_median']:.2%} / "
              f"{abs(p95_21 - v21['null_threshold']) / v21['null_threshold']:.2%})")
        print(f"  {'':>6}    observed {s21:.6f} escapes it: {esc21} "
              f"(p61 says {want21}); sigma1 share {s1:.6f} below the null's "
              f"5th percentile {p05_1s:.6f}: {esc1s} (p61 says {want1s})")
        print(f"  {'':>6}    the VERDICTS are asserted; the medians are "
              f"compared within Monte Carlo error and are not anchors")

    # --------------------------------------- 36.4 the headline gap, end to end
    if "4" in sections and not ("2" in sections and "3" in sections):
        print("\n  36.4 needs sections 2 and 3; skipped")
    if "4" in sections and "2" in sections and "3" in sections:
        print("\n=== 36.4 the headline: passive vs survey, 0-79 block ===")
    for ym in (yms if {"2", "3", "4"} <= sections else []):
        pop = pops[ym]
        A = mine[f"{ym}|WE|dong|holidayfree"]
        Cp = [[A[i][j] / pop[i] for j in sq] for i in sq]
        Tp = symmetrise(Cp, [pop[i] for i in sq])
        Ts = symmetrise([[surv[ym][i][j] for j in sq] for i in sq],
                        [pop[i] for i in sq])
        rp, rs = assortativity(Tp), assortativity(Ts)
        _, np_ = mutual_information_bits(Tp)
        _, ns = mutual_information_bits(Ts)
        pk, sk = f"passive|{ym}|WE|dong|holidayfree", f"survey|{ym}|WE|seoul"
        compare(f"{ym} passive assortativity (0-79)", rp,
                p32["excess"][pk]["assortativity"], 1e-6)
        compare(f"{ym} survey assortativity (0-79)", rs,
                p32["excess"][sk]["assortativity"], 2e-3)
        compare(f"{ym} passive NMI", np_, p32["excess"][pk]["nmi"], 1e-6)
        compare(f"{ym} survey NMI", ns, p32["excess"][sk]["nmi"], 2e-3)
        pub_gap_r = p32["excess"][sk]["assortativity"] / p32["excess"][pk]["assortativity"]
        pub_gap_n = p32["excess"][sk]["nmi"] / p32["excess"][pk]["nmi"]
        compare(f"{ym} assortativity gap x", rs / rp, pub_gap_r, 2e-3)
        compare(f"{ym} NMI gap x", ns / np_, pub_gap_n, 2e-3)
        print(f"  {ym}: assortativity passive {rp:.6f} vs survey {rs:.6f} "
              f"-> {rs / rp:.2f}x (published {pub_gap_r:.2f}x)")
        print(f"  {'':>6}  NMI          passive {np_:.6f} vs survey {ns:.6f} "
              f"-> {ns / np_:.2f}x (published {pub_gap_n:.2f}x)")

    # ------------------------------------------------ 36.5 the coverage profile
    if "5" in sections and not args.skip_coverage:
        print("\n=== 36.5 the coverage profile (p9 section 9.6), recomputed ===")
        cov_pub = {c["age"]: c["p10"] for c in p9["coverage_profile"]}
        mine_cov = coverage_p10(con, 202001)
        for a in AGES:
            compare(f"coverage w_a {LBL[AGES.index(a)]}", mine_cov.get(a),
                    cov_pub.get(a), 1e-4)
        print(f"  {'band':>6} {'mine':>8} {'published':>10}")
        for a in AGES:
            print(f"  {LBL[AGES.index(a)]:>6} {mine_cov.get(a, float('nan')):>8.3f} "
                  f"{cov_pub.get(a, float('nan')):>10.3f}")


    # -------------------------------------------------- 36.6 p21's rho-hat
    if "6" in sections:
        print("\n=== 36.6 p21's rho-hat, rebuilt from the parquet ===")
        p21 = json.load(open(f"{ROOT}/eda/results_p21.json"))
        yms12 = list(range(202001, 202013))
        rho = {}
        for ym in yms12:
            num = rho_numerator(con, ym)
            den = regpop_by_age(con, ym)
            rho[ym] = {a: num.get(a, 0.0) / den[a] for a in AGES if den.get(a)}
            print(f"  {ym} done", flush=True)
        # p21 reports the mean over the twelve months of each month's ratio,
        # which is not the ratio of the sums; getting that backwards moves the
        # third decimal and nothing else, which is why it is worth stating.
        lvl = {a: sum(rho[y][a] for y in yms12) / len(yms12) for a in AGES}
        pub = {r["age"]: r["rho_mid"] for r in p21["level_by_age"]}
        worst = max(abs(lvl[a] - pub[LBL[AGES.index(a)]]) / abs(pub[LBL[AGES.index(a)]])
                    for a in AGES)
        for a in AGES:
            compare(f"rho-hat level {LBL[AGES.index(a)]}", lvl[a],
                    pub[LBL[AGES.index(a)]], 1e-6)
        print(f"  16 band levels, worst relative deviation {worst:.2e}")
        compare("rho-hat level low end", round(min(lvl.values()), 2),
                round(min(pub.values()), 2), 1e-9)
        compare("rho-hat level high end", round(max(lvl.values()), 2),
                round(max(pub.values()), 2), 1e-9)
        mar_dec = {a: (rho[202012][a] / rho[202003][a] - 1) * 100 for a in AGES}
        for a in AGES:
            compare(f"rho-hat Mar->Dec {LBL[AGES.index(a)]}", mar_dec[a],
                    p21["mar_to_dec_pct"][LBL[AGES.index(a)]], 1e-6)
        print(f"  Mar->Dec 80+ {mar_dec[80]:+.2f}% "
              f"(published {p21['mar_to_dec_pct']['80+']:+.2f}%)")
        print(f"  level range {min(lvl.values()):.3f}-{max(lvl.values()):.3f} "
              f"(published {min(pub.values()):.3f}-{max(pub.values()):.3f})")

    # ------------------------------------------- 36.7 p29's measured masked fill
    if "7" in sections:
        print("\n=== 36.7 p29's measured fill, our side rebuilt from raw parquet ===")
        p29 = json.load(open(f"{ROOT}/eda/results_p29.json"))
        for ym in (202003, 202012):
            got = official_gu(ym)
            if got is None:
                print(f"  {ym}: no official gu archive on disk, skipped")
                continue
            off, off_rows, off_masked_rows = got
            for dedup in (False, True):
                mine = gu_agg(con, ym, dedup=dedup)
                keys = set(off) | set(mine)
                v_lo = sum(mine.get(k, (0.0, 0))[0] for k in keys)
                mcells = sum(mine.get(k, (0.0, 0))[1] for k in keys)
                off_v = sum(off.get(k, (0.0, 0))[0] for k in keys)
                hidden = off_v - v_lo
                per_cell = hidden / mcells if mcells else float("nan")
                tag = "dedup" if dedup else "as p29"
                print(f"  {ym} [{tag:>6}]  per-cell {per_cell:6.3f}  "
                      f"masked share {hidden / off_v:7.4%}  "
                      f"gu own masked {off_masked_rows / off_rows:7.4%} "
                      f"({off_masked_rows:,} of {off_rows:,} rows)")
                if not dedup:
                    compare(f"{ym} masked cell holds", per_cell,
                            p29[str(ym)]["per_cell_mean"], 2e-3)
                    compare(f"{ym} measured masked share", hidden / off_v,
                            p29[str(ym)]["measured_share"], 2e-3)
                    compare(f"{ym} gu own masked share",
                            off_masked_rows / off_rows,
                            p29[str(ym)]["gu_own_masked_share"], 2e-3)
                    compare(f"{ym} gu own masked cells", float(off_masked_rows),
                            float(p29[str(ym)]["gu_own_masked_cells"]), 1e-9)
                    compare(f"{ym} masked cell count", float(mcells),
                            float(p29[str(ym)]["masked_cells"]), 2e-3)
                else:
                    d = per_cell / p29[str(ym)]["per_cell_mean"] - 1
                    print(f"  {'':>6}  {'':>8}  deduplicating the raw rows moves "
                          f"the per-cell fill by {d:+.4%}")

    # ------------------------------------------------------- 36.8 p19's E-share
    if "8" in sections:
        print("\n=== 36.8 p19's E-share, rebuilt from raw parquet ===")
        p19 = json.load(open(f"{ROOT}/eda/results_p19.json"))
        p29 = json.load(open(f"{ROOT}/eda/results_p29.json"))
        # p19's measured fill: the mean over months of p29's per-band per-cell
        # value, for bands with more than 10,000 masked cells.
        fills = defaultdict(list)
        for v in p29.values():
            if not isinstance(v, dict) or "by_age" not in v:
                continue
            for lab, x in v["by_age"].items():
                if x["masked_cells"] > 1e4:
                    fills[lab].append(x["per_cell"])
        fill = {AGES[LBL.index(k)]: sum(v) / len(v) for k, v in fills.items()}
        agg = {ym: eshare_agg(con, ym) for ym in (202003, 202012)}
        LIM = {"0-19": [0, 10, 15],
               "20-59": [20, 25, 30, 35, 40, 45, 50, 55],
               "60+": [60, 65, 70, 75, 80]}

        def share(ym, ages, c):
            """E-arrival share within a set of age bands, at imputation c."""
            def vol(a, d):
                lo, nm = agg[ym].get((a, d), (0.0, 0.0))
                return lo + (fill.get(a, 0.0) if c == "meas" else c) * nm
            e = sum(vol(a, "E") for a in ages)
            tot = sum(vol(a, d) for a in ages for d in ("H", "W", "E"))
            return e / tot if tot else float("nan")

        d16 = {a: {c: (share(202012, [a], c) - share(202003, [a], c)) * 100
                   for c in (0.0, 1.5, "meas")} for a in AGES}
        pub16 = {r["age"]: r for r in p19["e_share_change_16band"]}
        for a in AGES:
            lab = LBL[AGES.index(a)]
            compare(f"E-share 16band {lab} mid", round(d16[a][1.5], 2),
                    pub16[lab]["mid"], 3e-3)
            compare(f"E-share 16band {lab} meas", round(d16[a]["meas"], 2),
                    pub16[lab]["meas"], 3e-3)
        d3 = {b: {c: (share(202012, ags, c) - share(202003, ags, c)) * 100
                  for c in (1.5, "meas")} for b, ags in LIM.items()}
        pub3 = {r["band"]: r for r in p19["e_share_change_3band"]}
        for b in LIM:
            compare(f"E-share 3band {b} meas", round(d3[b]["meas"], 2),
                    pub3[b]["meas"], 3e-3)
        rng16 = max(d16[a]["meas"] for a in AGES) - min(d16[a]["meas"] for a in AGES)
        rng3 = max(d3[b]["meas"] for b in LIM) - min(d3[b]["meas"] for b in LIM)
        span = {r["resolution"]: r for r in p19["resolution_span"]}
        compare("16-band spread pp", round(rng16, 2),
                round(span["16 bands"]["range_pp"], 2), 3e-3)
        compare("3-band spread pp", round(rng3, 2),
                round(span["3 bands (Lim)"]["range_pp"], 2), 3e-3)
        compare("three bands retain", round(rng3 / rng16, 4),
                round(p19["range_retained_frac"], 4), 3e-3)
        print(f"  16-band spread {rng16:.2f} pp, 3-band {rng3:.2f} pp, "
              f"retained {rng3 / rng16:.1%} "
              f"(published {p19['range_retained_frac']:.1%})")
        print(f"  20-24 E-share change (measured fill) {d16[20]['meas']:+.2f} pp "
              f"(published {pub16['20-24']['meas']:+.2f})")

    # ------------------------------------------------- 36.9 p34's arithmetic
    if "9" in sections:
        print("\n=== 36.9 p34: the two deterministic anchors, and every number "
              "derived from the sweep ===")
        p34 = json.load(open(f"{ROOT}/eda/results_p34.json"))
        sw = {k: {x["n_loc"]: x for x in v} for k, v in p34["sweep"].items()}
        # The anchors are the part of p34 that is NOT Monte Carlo: with 424
        # groups nothing is merged and with 1 group there is only one partition,
        # so both ends must equal the matrices section 36.2 built. That ties the
        # sweep to an independently recomputed value rather than to p26's file.
        if not matrices_36_2:
            print("  (section 2 did not run, so the k=424 and k=1 anchors are "
                  "not checked against an independently rebuilt matrix here; "
                  "run --sections 2,9 for those)")
        for ym in (yms if matrices_36_2 else []):
            for arm in ("adjacency", "random"):
                key = f"{ym}|WE|{arm}"
                if key not in sw or f"{ym}|WE|dong" not in matrices_36_2:
                    continue
                compare(f"{key} k=424 anchor",
                        sw[key][424]["assortativity"]["median"],
                        assortativity(matrices_36_2[f"{ym}|WE|dong"]), 1e-6)
                compare(f"{key} k=1 anchor",
                        sw[key][1]["assortativity"]["median"],
                        assortativity(matrices_36_2[f"{ym}|WE|city"]), 1e-6)
        # everything downstream of the sweep is arithmetic, and arithmetic is
        # where transcription errors live
        for ym_s, poly in p34["polygon_prediction_interval"].items():
            ls = p34["local_slope"][ym_s]["adjacency|assortativity"]
            near = [x["slope"] for x in ls if x["n"] >= 200]
            compare(f"{ym_s} slope_lo from stored slopes", min(near),
                    poly["slope_lo"], 1e-9)
            compare(f"{ym_s} slope_hi from stored slopes", max(near),
                    poly["slope_hi"], 1e-9)
            r424 = sw[f"{ym_s}|WE|adjacency"][424]["assortativity"]["median"]
            compare(f"{ym_s} r_dong from sweep", r424, poly["r_dong"], 1e-12)
            pred = sorted(r424 * (1831.0 / 424.0) ** t for t in (min(near), max(near)))
            compare(f"{ym_s} r_polygon_lo", pred[0], poly["r_polygon_lo"], 1e-12)
            compare(f"{ym_s} r_polygon_hi", pred[1], poly["r_polygon_hi"], 1e-12)
        es = p34["extrapolation_support"]
        compare("log10 span supported", math.log10(424), es["log10_span"], 1e-12)
        compare("log10 span required", math.log10(6.3e6),
                es["log10_span_required"], 1e-12)
        for ym_s in p34["polygon_prediction_interval"]:
            a = sw.get(f"{ym_s}|WE|adjacency")
            r = sw.get(f"{ym_s}|WE|random")
            if not a or not r:
                continue
            for k in (1, 25, 424):
                if k in a and k in r:
                    ratio = (a[k]["assortativity"]["median"]
                             / r[k]["assortativity"]["median"])
                    if k in (1, 424):
                        # both arms must coincide where there is only one
                        # possible partition; a ratio of 1 is not a coincidence
                        compare(f"{ym_s} arms coincide at k={k}", ratio, 1.0, 1e-9)
        print(f"  anchors and derived arithmetic checked for "
              f"{len(p34['polygon_prediction_interval'])} months")

    # ------------------------------- 36.10 p58's circular-shift semester p
    if "10" in sections:
        print("\n=== 36.10 p58's circular-shift p, relabelled from the ym and "
              "rotated as a string ===")
        p58 = json.load(open(f"{ROOT}/eda/results_p58.json"))
        p42 = json.load(open(f"{ROOT}/eda/results_p42.json"))
        p51 = json.load(open(f"{ROOT}/eda/results_p51.json"))
        # WHAT IS TAKEN ON TRUST HERE, said out loud rather than left implied:
        # the 79 monthly MI readings themselves are p42's, not rebuilt. Section
        # 36.2's route costs about four seconds a matrix on two months; at 79
        # months over the full product it is a billion-row scan, and this
        # section is not where that is affordable. What IS independent is
        # everything built ON that series: the labels, the clearing rule, the
        # rotation, and the count.
        mi = {r["ym"]: r["mi_bits_hf"] for r in p42["rows"]}
        floor = p51["cells"]["202312|national"]["mi_perm_median"]
        order = sorted(mi)
        n = len(order)
        print(f"  series: {n} months {order[0]}..{order[-1]}, mi_bits_hf taken "
              f"from results_p42.json and NOT rebuilt (a 79-month SQL rebuild "
              f"is a billion-row scan); floor {floor!r} from results_p51.json")
        # The labels come out of the ym, and the rotation is a string rotation,
        # not p58's modular indexing into an array of dicts.
        letters = "".join(semester_letter(ym) for ym in order)
        mask = "".join("1" if mi[ym] >= floor else "0" for ym in order)

        def rotate(k):
            k %= n
            return letters[n - k:] + letters[:n - k]

        lab_counts = collections.Counter(letters)
        for tag, name in (("T", "term"), ("V", "vacation"), ("D", "december")):
            compare(f"p58 {name} months in the 79-month calendar",
                    float(lab_counts[tag]),
                    float(p58["anchors"]["A4"]["label_counts"][name]))
        n_term = [collections.Counter(ch for ch, m in zip(rotate(k), mask)
                                      if m == "1")["T"] for k in range(n)]
        margins = sorted({collections.Counter(rotate(k))["T"] for k in range(n)})
        ok = margins == [p58["anchors"]["A5"]["margin"]]
        _record("p58 term margin invariant under all 79 rotations", margins,
                [p58["anchors"]["A5"]["margin"]], 0.0 if ok else 1.0, ok, 0.0)
        n_clear = mask.count("1")
        compare("p58 clearing months", float(n_clear),
                float(p58["run_structure"]["n_clearing"]))
        compare("p58 observed n_term(0)", float(n_term[0]),
                float(p58["p_shift"]["observed"]))
        ge = [k for k in range(1, n) if n_term[k] >= n_term[0]]
        okg = ge == p58["p_shift"]["shifts_at_or_above_observed"]
        _record("p58 shifts reaching the observed count", ge,
                p58["p_shift"]["shifts_at_or_above_observed"],
                0.0 if okg else 1.0, okg, 0.0)
        compare("p58 n_shifts_at_or_above", float(len(ge)),
                float(p58["p_shift"]["n_shifts_at_or_above"]))
        p_shift = (1.0 + len(ge)) / n
        compare("p58 p_shift", p_shift, p58["p_shift"]["p"])
        compare("p58 resolution floor 1/79", 1.0 / n,
                p58["p_shift"]["resolution_floor"])
        theirs_v = p58["p_shift"]["n_term_by_k"]
        bad_k = [k for k in range(n)
                 if k >= len(theirs_v) or n_term[k] != theirs_v[k]]
        okv = not bad_k and len(theirs_v) == n
        _record(f"p58 n_term(k) over all {n} rotations, k that disagree",
                bad_k, [], 0.0 if okv else 1.0, okv, 0.0)
        over = p58["p_shift"]["n_term_over_shifts"]
        compare("p58 n_term over shifts, min", float(min(n_term)),
                float(over["min"]))
        compare("p58 n_term over shifts, max", float(max(n_term)),
                float(over["max"]))
        compare("p58 n_term over shifts, mean",
                sum(n_term[1:]) / float(n - 1), float(over["mean"]))
        runs = [len(list(g)) for v, g in itertools.groupby(mask) if v == "1"]
        compare("p58 clearing runs", float(len(runs)),
                float(p58["run_structure"]["n_runs"]))
        okr = runs == p58["run_structure"]["run_lengths"]
        _record("p58 clearing run lengths", runs,
                p58["run_structure"]["run_lengths"], 0.0 if okr else 1.0,
                okr, 0.0)
        print(f"  labels from the ym: {lab_counts['T']} term, "
              f"{lab_counts['V']} vacation, {lab_counts['D']} december; "
              f"term margin under all {n} rotations {margins}")
        print(f"  {n_clear} clearing months in {len(runs)} runs {runs}; "
              f"n_term(0) = {n_term[0]}, shifts reaching it {ge}")
        print(f"  p_shift = (1 + {len(ge)})/{n} = {p_shift!r} "
              f"(published {p58['p_shift']['p']!r}), "
              f"floor 1/{n} = {1.0 / n:.6f}")

    # -------------------------------------------------- 36.11 p59's beta star
    if "11" in sections:
        print("\n=== 36.11 p59's beta*, by monotone interpolation and "
              "bisection ===")
        p59 = json.load(open(f"{ROOT}/eda/results_p59.json"))
        const = p59["anchors"]["A3_published_constants"]
        r_obs = const["r_obs_corrected"]["from_results_p39"]
        r_survey = const["survey"]["from_results_p39"]
        # The survey constant the whole claim is measured against is the same
        # number section 36.4 rebuilt from the CSVs, so tie the two files
        # together rather than trusting p59's copy of it.
        compare("p59 survey r ties to p32's survey cell", r_survey,
                p32["excess"]["survey|202312|WE|seoul"]["assortativity"])
        betas = sorted(float(b) for b in p59["inversion_refined"])
        bounds, cis = [], ([], [])
        for b in betas:
            d = p59["inversion_refined"][f"{b}"]
            xs, ys = d["r_true"], d["q_lo_corr"]
            assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)), (
                f"q_lo_corr is not monotone at beta = {b}; bisection needs a "
                f"single crossing and the interpolation would not be invertible")
            x = bisect_root(pw_linear_shift(xs, ys, r_obs), xs[0], xs[-1])
            bounds.append(x)
            cis[0].append(d["bound_ci"][0])
            cis[1].append(d["bound_ci"][1])
            compare(f"p59 bound at beta = {b}", x, d["bound_corrected"], 1e-9)
            phi = x * (1.0 - b) / r_obs
            pub_phi = [r for r in p59["phi_refined"] if r["beta"] == b][0]["phi"]
            compare(f"p59 phi at beta = {b}", phi, pub_phi, 1e-9)
            print(f"  beta {b}: bound {x:.10f} (published "
                  f"{d['bound_corrected']:.10f})   phi {phi:.10f} "
                  f"(published {pub_phi:.10f})")

        def cross(vals):
            """The beta at which a bound curve reaches the survey value."""
            return bisect_root(pw_linear_shift(betas, vals, r_survey),
                               betas[0], betas[-1])

        est = p59["beta_star"]["estimates"]
        mine = {"linear_in_bound": cross(bounds),
                "ci_low": cross(cis[0]), "ci_high": cross(cis[1])}
        # phi-linear: phi is interpolated, and the bound it implies is
        # phi r_obs / (1 - beta), so the crossing is a root find rather than an
        # interpolation and cannot be done by np.interp at all.
        phis = [x * (1.0 - b) / r_obs for b, x in zip(betas, bounds)]
        gphi = pw_linear(betas, phis)
        mine["phi_linear"] = bisect_root(
            lambda t: gphi(t) * r_obs / (1.0 - t) - r_survey,
            betas[0], betas[-1])
        for k in ("linear_in_bound", "phi_linear", "ci_low", "ci_high"):
            compare(f"p59 beta* {k}", mine[k], est[k]["value"], 1e-9)
        logs = [math.log(x) for x in bounds]
        b_log = bisect_root(pw_linear_shift(betas, logs, math.log(r_survey)),
                            betas[0], betas[-1])
        compare("p59 beta* log_in_bound", b_log, p59["beta_star"]["log_in_bound"],
                1e-9)
        spread = max(mine.values()) - min(mine.values())
        compare("p59 spread of the four declared rules", spread,
                p59["beta_star"]["spread"], 1e-9)
        print(f"  beta* primary (linear in bound) {mine['linear_in_bound']!r}")
        print(f"  {'':>8}phi-linear {mine['phi_linear']:.10f}, ci_low "
              f"{mine['ci_low']:.10f}, ci_high {mine['ci_high']:.10f}, "
              f"log-in-bound {b_log:.10f}; spread {spread:.7f}")

        # ---- the part of the claim that uses no interpolation at all
        b95 = bounds[betas.index(0.95)]
        excl = b95 < r_survey
        _record("p59 bound(0.95) excludes the survey value", b95, r_survey,
                0.0 if excl else 1.0, excl, 0.0)
        compare("p59 claim margin", r_survey - b95, p59["claim"]["margin"], 1e-9)
        # Every beta reading in the two files, found by a mechanical rule
        # instead of by reading p59's curated list of eleven. The strict rule is
        # "the leaf's key is exactly beta"; the broad rule adds p53's
        # beta_one_convention and beta_mixed and excludes the derived
        # quantities (the CI, the range, the p39 crossing, beta_upper,
        # beta_primary, the withdrawn bracket), which are not measurements.
        derived = {"beta_ci", "beta_range", "beta_primary", "beta_upper",
                   "beta_bracket_withdrawn", "beta_star"}
        strict, broad = set(), set()
        for tag in ("p44", "p53"):
            doc = json.load(open(f"{ROOT}/eda/results_{tag}.json"))
            for key, val in json_leaves(doc):
                if not isinstance(val, float) or key is None:
                    continue
                if key == "beta":
                    strict.add(val)
                if key.startswith("beta") and key not in derived:
                    broad.add(val)
        curated = sorted(m["beta"] for m in p59["claim"]["measured_betas"])
        all_below = all(v < 0.95 for v in curated)
        _record("p59 claim.all_below_0p95 recomputed from its own list",
                all_below, bool(p59["claim"]["all_below_0p95"]),
                0.0 if all_below == bool(p59["claim"]["all_below_0p95"]) else 1.0,
                all_below == bool(p59["claim"]["all_below_0p95"]), 0.0)
        hi_s = sorted(v for v in strict if v >= 0.95)
        hi_b = sorted(v for v in broad if v >= 0.95)
        wr = [p59["claim"]["excluded_reading"]["beta"]]
        ok_s, ok_b = hi_s == wr, hi_b == wr
        _record("p59 the only reading at or above 0.95 (strict rule)",
                hi_s, wr, 0.0 if ok_s else 1.0, ok_s, 0.0)
        _record("p59 the only reading at or above 0.95 (broad rule)",
                hi_b, wr, 0.0 if ok_b else 1.0, ok_b, 0.0)
        compare("p59 largest reading below 0.95",
                max(v for v in strict if v < 0.95),
                p59["claim"]["largest_measured_beta"])
        print(f"  bound(0.95) {b95:.5f} < survey {r_survey:.5f}: {excl}, "
              f"margin {r_survey - b95:.6f} -- no interpolation used")
        print(f"  beta readings in results_p44/p53: {len(strict)} under the "
              f"strict rule, {len(broad)} under the broad one; at or above "
              f"0.95: {hi_s} / {hi_b}")
        # The membership is NOT p59's, and that is worth printing rather than
        # hiding, because the verdict survives it. p59's inventory is the eleven
        # measured readings plus the one it names and rejects; the strict rule
        # here finds eleven leaves keyed exactly `beta`, which is a different
        # set that happens to be the same size.
        inventory = sorted(set(curated) | {p59["claim"]["excluded_reading"]["beta"]})
        only_mine = sorted(set(strict) - set(inventory))
        only_p59 = sorted(set(inventory) - set(strict))
        print(f"  p59's inventory is {len(inventory)} values ({len(curated)} "
              f"measured + 1 named and rejected); the strict rule finds "
              f"{len(strict)} and the broad rule {len(broad)}")
        print(f"  the sets are not the same: only in mine {only_mine}, only in "
              f"p59's {only_p59} -- and every rule gives ONE reading at or "
              f"above 0.95, the same one")

    # ------------------------------ 36.12 p60's 3-bin over 16-bin assortativity
    if "12" in sections:
        print("\n=== 36.12 p60's age collapse, summed by groupby instead of "
              "S A S' ===")
        p60 = json.load(open(f"{ROOT}/eda/results_p60.json"))
        p37 = json.load(open(f"{ROOT}/eda/results_p37.json"))
        mats = p37["matrices_we_dong"]
        # As in 36.10, the 79 matrices themselves are p37's and are not rebuilt;
        # what is rebuilt is every statistic taken from them. Section 36.2 does
        # rebuild two of these matrices from the parquet, so the estimator and
        # the source are both covered, just not at the same width.
        months = sorted(mats)
        print(f"  {len(months)} matrices from results_p37.json, NOT rebuilt; "
              f"36.2 rebuilds two of them from the parquet")
        r16s, r3s, ratios = [], [], []
        for ym in months:
            A = mats[ym]["A"]
            r16 = assortativity(A)
            r3 = assortativity(collapse_groupby(A, LIM_BANDS))
            pub = p60["per_month"][ym]
            compare(f"p60 {ym} r 16-bin", r16, pub["r16"], 1e-6)
            compare(f"p60 {ym} r 3-bin (lim)", r3, pub["lim"]["r"], 1e-6)
            r16s.append(r16)
            r3s.append(r3)
            ratios.append(r3 / r16)
        rising = sum(1 for x in ratios if x > 1.0)
        compare("p60 median 16-bin r", median(r16s),
                p60["summary"]["r16"]["median"], 1e-6)
        compare("p60 median 3-bin r", median(r3s),
                p60["summary"]["lim3"]["median"], 1e-6)
        compare("p60 median 3-bin over 16-bin", median(ratios),
                p60["summary"]["retained_frac"]["median"], 1e-6)
        compare("p60 minimum 3-bin over 16-bin", min(ratios),
                p60["summary"]["retained_frac"]["min"], 1e-6)
        compare("p60 maximum 3-bin over 16-bin", max(ratios),
                p60["summary"]["retained_frac"]["max"], 1e-6)
        compare("p60 months where the ratio rises", float(rising),
                float(p60["prediction"]["n_against"]))
        print(f"  16-bin median {median(r16s):.6f} (published "
              f"{p60['summary']['r16']['median']:.6f}), 3-bin median "
              f"{median(r3s):.6f} (published "
              f"{p60['summary']['lim3']['median']:.6f})")
        print(f"  ratio median {median(ratios):.4f} (published "
              f"{p60['summary']['retained_frac']['median']:.4f}), range "
              f"{min(ratios):.4f}-{max(ratios):.4f} (published "
              f"{p60['summary']['retained_frac']['min']:.4f}-"
              f"{p60['summary']['retained_frac']['max']:.4f}), rises in "
              f"{rising} of {len(months)} months")
        # THE MECHANISM, which is what makes the rise a result rather than a
        # bug: if the operator manufactured assortativity, it would manufacture
        # it out of a matrix that has none. Under exact proportionate mixing
        # e_ab = r_a r_b the collapsed matrix must return r = 0 at every rung,
        # because a partition of an outer product is an outer product of the
        # partitioned margins.
        rungs = [LIM_BANDS] + [nested_groups(k) for k in (16, 8, 4, 2)]
        worst_pm = 0.0
        for ym in months:
            E = proportionate_mixing(mats[ym]["A"])
            for gr in rungs:
                worst_pm = max(worst_pm,
                               abs(assortativity(collapse_groupby(E, gr))))
        pub_pm = p60["mechanism"]["proportionate_mixing_null_max_abs_r"]
        okpm = worst_pm < 1e-12
        # Not a bit comparison: two float summations of the same cancelling sum
        # land on different sides of zero, and the claim is that both are zero,
        # not that they are equal. The threshold is p60's own assert.
        _record(f"p60 proportionate mixing collapses to r = 0 "
                f"({len(months)} months x {len(rungs)} rungs)",
                worst_pm, pub_pm, 0.0 if okpm else 1.0, okpm, 0.0)
        print(f"  proportionate-mixing null, {len(months)} months x "
              f"{len(rungs)} rungs: max |r| = {worst_pm:.3e} "
              f"(p60 measured {pub_pm:.3e}) -- collapsing does not manufacture "
              f"assortativity")

    # -------------- 36.13 p37's rank-one gap, and the inventory the abstract counts
    if "13" in sections:
        print("\n=== 36.13 p37's departure from rank one, and the inventory "
              "the abstract counts ===")
        # WHY THIS SECTION EXISTS. Two numbers in the abstract had one
        # implementation each and no second opinion anywhere: p37's
        # `spectrum_dong.gap_to_pm_max`, quoted as "at most 0.00202", and
        # data_inventory's 79 months / 10.2 billion rows. data_inventory also
        # asserts nothing about itself, so until this block it was the only
        # producer of a quoted number with neither an anchor nor a second route.
        p37s = json.load(open(f"{ROOT}/eda/results_p37.json"))
        # p67 is the object the ABSTRACT quotes. It reads the same stored
        # matrices p37 wrote, so a second route to its number has to come from
        # somewhere else: the 79 months below are recomputed by power iteration
        # rather than by p67's SVD, and 202312 and 202402 are recomputed off the
        # matrix 36.2 built from the parquet, which owes p37 nothing at all.
        p67s = json.load(open(f"{ROOT}/eda/results_p67.json"))
        per67 = {int(r["ym"]): r for r in p67s["per_month"]}
        inv = json.load(open(f"{ROOT}/eda/results_inventory.json"))
        mats = p37s["matrices_we_dong"]
        pub37 = {int(r["ym"]): r for r in p37s["rows"]
                 if r["panel"] == "WE" and r["level"] == "dong"}
        months = sorted(mats)
        # As in 36.10 and 36.12: the 79 matrices are p37's and are not rebuilt.
        # What is rebuilt is every statistic taken off them -- and, below, the
        # gap itself for the two months 36.2 does build from the parquet, which
        # is the only place this abstract number is carried back to raw rows.
        print(f"  {len(months)} matrices from results_p37.json, NOT rebuilt; "
              f"36.2 rebuilds {', '.join(str(y) for y in yms)} from the parquet "
              f"and the gap is recomputed off those too")
        gaps, shares, worst = [], [], (0.0, None)
        grels = []
        for ym in months:
            share, best, pm = rank1_residuals(below_80(mats[ym]["A"]))
            p = pub37[int(ym)]
            g = abs(best - pm)
            compare(f"p37 {ym} sigma1 share", share, p["sigma1_share"])
            compare(f"p37 {ym} rank-one gap", g,
                    abs(p["best_rank1_resid"] - p["pm_rank1_resid"]))
            # The abstract's quantity, at the tolerance the two routes actually
            # meet at: a fixed-count power iteration against LAPACK's SVD, worst
            # 2.1e-13 over the 79 months when this was written.
            gr = rank1_pm_gap(below_80(mats[ym]["A"]))
            compare(f"p67 {ym} distance to the null", gr,
                    per67[int(ym)]["gap_rel"], 1e-11)
            grels.append(gr)
            gaps.append(g)
            shares.append(share)
            if g > worst[0]:
                worst = (g, ym)
        sd = p37s["spectrum_dong"]
        compare("p37 rank-one gap, worst month (the abstract's 0.00202)",
                max(gaps), sd["gap_to_pm_max"])
        compare("p37 rank-one gap, median month", median(gaps),
                sd["gap_to_pm_median"])
        compare("p37 sigma1 share, median month", median(shares),
                sd["sigma1_share_median"])
        compare("p37 sigma1 share, minimum month", min(shares),
                sd["sigma1_share_min"])
        # "AT MOST 0.00202" IS A CLAIM ABOUT ALL 79, not about the largest one.
        # A stored max that reproduces while some other month sits above the
        # quoted bound would leave the sentence false and every check above
        # green, so the months over the bound are counted rather than inferred.
        over = [m for m, g in zip(months, gaps) if g > 0.00202]
        _record("p37 no month exceeds the quoted 0.00202", float(len(over)),
                0.0, 0.0 if not over else 1.0, not over, 0.0)
        print(f"  worst month {worst[1]} at {worst[0]:.11f} (p37 stored "
              f"{sd['gap_to_pm_max']:.11f}), median {median(gaps):.11f} "
              f"(p37 {sd['gap_to_pm_median']:.11f}); {len(over)} of "
              f"{len(months)} months above SI 3's 0.00202")
        # ...and the same three questions asked of the distance itself, which is
        # what the abstract, 3.2 and 4.1 now say.
        s67 = p67s["summary"]
        compare("p67 distance to the null, worst month (the abstract's 0.0258)",
                max(grels), s67["gap_rel_max"], 1e-11)
        compare("p67 distance to the null, median month", median(grels),
                s67["gap_rel_median"], 1e-11)
        # 0.0258, not 0.0257: the measured maximum is 0.02573, so the four-
        # decimal quote has to round UP to be an upper bound at all. Quoting
        # 0.0257 was the same defect this file already records for "at most
        # 0.002" against 0.00202, and this row is what found it.
        over67 = [m for m, g in zip(months, grels) if g > 0.0258]
        _record("p67 no month exceeds the quoted 0.0258", float(len(over67)),
                0.0, 0.0 if not over67 else 1.0, not over67, 0.0)
        print(f"  distance to the null: worst {max(grels):.11f} (p67 stored "
              f"{s67['gap_rel_max']:.11f}), median {median(grels):.11f} "
              f"(p67 {s67['gap_rel_median']:.11f}); {len(over67)} of "
              f"{len(months)} months above the abstract's 0.0258")
        # The parquet-side leg. For these months the gap owes results_p37.json
        # nothing: the matrix underneath it was built by 36.2's SQL self-join.
        # The tolerance is looser than the 1e-9 above and has to be: the two
        # matrices are the same object summed in two different orders, and the
        # gap is a difference of two residuals near 0.134, so whatever they
        # disagree by is amplified about ninetyfold on the way out.
        for ym in (yms if matrices_36_2 else []):
            key = f"{ym}|WE|dong"
            if key not in matrices_36_2 or ym not in pub37:
                continue
            _, best, pm = rank1_residuals(below_80(matrices_36_2[key]))
            theirs = abs(pub37[ym]["best_rank1_resid"]
                         - pub37[ym]["pm_rank1_resid"])
            compare(f"p37 {ym} rank-one gap, from 36.2's SQL matrix",
                    abs(best - pm), theirs, 1e-6)
            # The abstract's number, on the same parquet-side leg, and it gets
            # three orders more tolerance than it needs rather than its sibling's
            # 1e-6: a norm of a difference has none of the cancellation that
            # amplifies the line above about ninetyfold, so this one tracks how
            # well the two matrices themselves agree. Measured on 2026-09-07,
            # 6.3e-14 for 202312 and 2.1e-14 for 202402.
            g67 = rank1_pm_gap(below_80(matrices_36_2[key]))
            compare(f"p67 {ym} distance to the null, from 36.2's SQL matrix",
                    g67, per67[ym]["gap_rel"], 1e-9)
            print(f"  {ym} gap from the SQL matrix {abs(best - pm):.11f} "
                  f"(p37 {theirs:.11f}); distance {g67:.11f} "
                  f"(p67 {per67[ym]['gap_rel']:.11f})")
        if not matrices_36_2:
            print("  36.2 did not run, so the parquet-side leg is skipped; the "
                  "79 months above still ran off results_p37.json")

        # ---------- the inventory, from the data rather than the directory names
        first, last = inv["span_first"], inv["span_last"]
        span = month_span(first, last)
        compare("inventory months in the span", float(len(span)),
                float(inv["span_months"]))
        # DuckDB's `glob` lists the tree; data_inventory walks it with pathlib.
        # Same files, no shared line.
        by_dir = {int(y): n for y, n in con.execute(
            f"SELECT regexp_extract(file, 'ym=([0-9]+)', 1) AS ym, count(*) "
            f"FROM glob('{PARQUET_GLOB}') GROUP BY 1").fetchall() if y}
        compare("inventory parquet files", float(sum(by_dir.values())),
                float(inv["parquet_files"]))
        compare("inventory months holding a full 24 files",
                float(sum(1 for n in by_dir.values() if n == 24)),
                float(inv["months_complete"]))
        # ...and now off the ym COLUMN. This is the one that costs something and
        # the one worth paying for: `count(*)` over the glob -- what
        # data_inventory asks for -- is answered out of the parquet footers
        # without a row being decoded, so the published 10,186,891,962 is a
        # metadata sum. Grouping on ym decodes the column, which both counts the
        # rows for real and is the only thing here that would catch a file whose
        # rows disagree with the ym= directory it was written into.
        by_col = {int(y): n for y, n in con.execute(f"""
            SELECT ym, count(*) AS n
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=false)
            GROUP BY 1 ORDER BY 1""").fetchall()}
        compare("inventory rows in the parquet (the abstract's 10.2 billion)",
                float(sum(by_col.values())), float(inv["parquet_rows"]))
        compare("inventory distinct months in the ym column",
                float(len(by_col)), float(inv["months_complete"]))
        stray = sorted(set(by_col) ^ set(by_dir))
        _record("inventory the ym column and the ym= directories name the same "
                "months", float(len(stray)), 0.0, 0.0 if not stray else 1.0,
                not stray, 0.0)
        missing = sorted(set(span) - set(by_col))
        _record(f"inventory {len(span)} consecutive months, none missing",
                float(len(missing)), float(inv["months_missing"]),
                0.0 if not missing else 1.0, not missing, 0.0)
        print(f"  {len(span)} consecutive months {first}..{last}, marched with "
              f"calendar.monthrange; {len(by_dir)} ym= directories holding "
              f"{sum(by_dir.values()):,} files")
        print(f"  the ym column carries {sum(by_col.values()):,} rows over "
              f"{len(by_col)} months (data_inventory: {inv['parquet_rows']:,} "
              f"over {inv['months_complete']}), and {len(missing)} of the "
              f"{len(span)} span months are absent")

    # ------------------------------------------------------------------ verdict
    print(f"\n=== {len(checks) - len(fails)} of {len(checks)} independent "
          f"recomputations agree ===")
    # Report the two separately: the checks that compare UNROUNDED quantities
    # should agree to machine precision, and the ones that compare against a
    # value p19 stored rounded to two decimals cannot do better than that
    # rounding. Lumping them together would either hide a real drift or demand
    # an impossible tolerance.
    exact = [c for c in checks if isinstance(c[3], float) and c[5] <= 1e-6]
    loose = [c for c in checks if isinstance(c[3], float) and c[5] > 1e-6]
    worst = max(exact, key=lambda c: c[3], default=None)
    if loose:
        wl = max(loose, key=lambda c: c[3])
        print(f"  checks against rounded published values: {len(loose)}, "
              f"worst {wl[0]} at {wl[3]:.2e} (tolerance {wl[5]:.0e})")
    if worst:
        print(f"  worst relative disagreement: {worst[0]} at {worst[3]:.2e}")
    if fails:
        print("\nDISAGREEMENTS -- the second implementation does not reproduce "
              "the published number:")
        for lb in fails:
            for c in checks:
                if c[0] == lb:
                    print(f"  {lb}: mine {c[1]}, published {c[2]}, "
                          f"relative {c[3]:.3e}")
                    break
    # A PARTIAL RUN MUST NOT OVERWRITE THE GATED ARTEFACT. p31 checks
    # results_p36.json's check count, so a `--sections 9` run that wrote to the
    # same path would quietly replace the full tally with a partial one and the
    # gate would then be verifying a file that covers a fraction of what it
    # claims. Found the hard way. Partial runs write beside it instead, and say
    # so.
    full = sections == ALL_SECTIONS
    name = "results_p36.json" if full else "results_p36.partial.json"
    with open(f"{ROOT}/eda/{name}", "w") as fh:
        json.dump(dict(checks=[dict(label=c[0], mine=c[1], published=c[2],
                                    rel=c[3], ok=c[4], rtol=c[5])
                               for c in checks],
                       n_checks=len(checks), n_fail=len(fails),
                       sections=sorted(sections), full_run=full),
                  fh, indent=1, default=str)
    print(f"wrote {name}" + ("" if full else
          f"  (partial run: sections {sorted(sections)}; "
          f"results_p36.json left alone)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
