#!/usr/bin/env python
"""Phase 2a — schema and classification drift, 2020 versus 2026.

The plan budgets three days for this and calls it the highest-value three days
of Phase 2: if the H/W/E classifier, the 16 age bands or the dong code space
break somewhere in the middle of the series, the whole longitudinal frame has to
be rebuilt, and week 2 is a far better time to learn that than week 10. This is
that audit, run early on the two 2026 months pulled forward for Phase 1d route
R2 (`eda/dl_mobility.py`).

Six questions, each answered with a number:

  1. mtype   — are the 9 H/W/E crossings still there, with the same codes?
  2. age     — 16 bands, 80+ still the top open band?
  3. dong    — how big is the code space, and does Seoul still have 424 dong?
               This is the one that decides how the longitudinal analysis is
               built: `memo/phase1b-regpop.md` §4 found the registration at
               425/426/427 Seoul dong over the window (상일동 split 2021-07,
               일원2동 → 개포3동 rename 2022-12, 용신동 split 2025-07) while
               the official 2021-09 code table still lists the pre-split name.
               Frozen code space => 424 everywhere and clean longitudinal
               comparability; live code space => a crosswalk and stable units.
  4. masking — still a hard floor at exactly 3.00?
  5. mean_min— 평균 이동 시간 still never null?
  6. volume  — person-trips per day 2026 versus 2020, as a structural sanity
               check on the ingest.

Reads the parquet tree directly. It deliberately does not go through
`common.connect()`: `common.YMS` is hard-coded to the twelve months of 2020 and
`calendar_kr` raises outside that window, so the shared view cannot see a 2026
month at all. Per-day volume needs only the length of the month, which comes
from `calendar.monthrange` here.

    .venv/bin/python eda/p22_schema_audit.py            # every month on disk
    .venv/bin/python eda/p22_schema_audit.py --months 202001,202607

Writes eda/results_p22.json.
"""
import argparse
import calendar
import json
import os
import re
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT, PARQUET, RESULTS, require  # noqa: E402

MTYPES = ["HH", "HW", "HE", "WH", "WW", "WE", "EH", "EW", "EE"]
AGES = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
SEXES = ["F", "M"]
SEOUL_LO, SEOUL_HI = 1101000, 1125999
MASK_FLOOR = 3.0
KEY = ["ym", "dow_n", "arr_hour", "o_dong", "d_dong", "sex", "age", "mtype"]

CODE_TABLE = DATA_ROOT / "docs" / "서울생활이동데이터_행정동코드_20210907.xlsx"
CROSSWALK = DATA_ROOT / "derived" / "dong_crosswalk.parquet"


def months_on_disk():
    p = require(PARQUET, "parquet tree")
    out = []
    for d in sorted(p.iterdir()):
        m = re.fullmatch(r"ym=(\d{6})", d.name)
        if m and any(f.suffix == ".parquet" for f in d.iterdir()):
            out.append(int(m.group(1)))
    return out


def glob_for(ym):
    return f"{PARQUET}/ym={ym}/*.parquet"


def code_names():
    """dong code -> full name, from the official 2021-09 table if it is here.

    Only used to say *which* codes moved. A code minted after 2021-09 will not
    be in the table, and that absence is itself informative.
    """
    names = {}
    try:
        import pandas as pd
        t = pd.read_excel(CODE_TABLE)
        for _, r in t.iterrows():
            names[int(r["읍면동"])] = str(r["full_name"])
    except Exception:
        pass
    if not names and CROSSWALK.exists():
        try:
            t = duckdb.sql(f"SELECT dong_code, full_name "
                           f"FROM '{CROSSWALK}'").df()
            names = {int(a): str(b) for a, b in
                     zip(t["dong_code"], t["full_name"])}
        except Exception:
            pass
    return names


def per_file_keys(con, ym):
    """Exact duplicate accounting, one hour file at a time.

    arr_hour is part of the natural key and constant within a file, so a
    duplicated key can only ever sit inside one file. Doing it per file keeps
    the GROUP BY at ~5M rows instead of ~100M, which is the difference between
    seconds and a spill.
    """
    d = PARQUET / f"ym={ym}"
    files = sorted(str(f) for f in d.iterdir() if f.suffix == ".parquet")
    tot = {"files": len(files), "rows": 0, "keys": 0, "distinct_full_rows": 0,
           "dup_keys": 0, "extra_rows": 0, "conflicting_keys": 0}
    allc = KEY + ["mean_min", "pop", "masked"]
    for f in files:
        r = con.execute(f"""
            WITH k AS (
              SELECT count(*) AS c FROM read_parquet('{f}')
              GROUP BY {', '.join(KEY)}
            )
            SELECT sum(c), count(*),
                   count(*) FILTER (WHERE c > 1),
                   sum(c - 1) FILTER (WHERE c > 1)
            FROM k""").fetchone()
        tot["rows"] += int(r[0])
        tot["keys"] += int(r[1])
        tot["dup_keys"] += int(r[2] or 0)
        tot["extra_rows"] += int(r[3] or 0)
        # Whole-row copy or a genuine key collision? Counting the distinct full
        # rows answers it without a per-group DISTINCT aggregate, which over
        # ~5M groups is orders of magnitude slower than the grouping itself.
        if int(r[2] or 0):
            full = con.execute(f"""
                SELECT count(*) FROM (
                  SELECT DISTINCT {', '.join(allc)} FROM read_parquet('{f}'))
                """).fetchone()[0]
            tot["distinct_full_rows"] += int(full)
            tot["conflicting_keys"] += int(full) - int(r[1])
        else:
            tot["distinct_full_rows"] += int(r[1])
    return tot


def audit_month(con, ym, names, dups=True):
    """Everything about one month, in two passes over its parquet (three with
    the duplicate check). The tree sits on an external drive, so the number of
    scans is the whole cost model: pass 1 is every scalar and every volume
    aggregate, pass 2 is one GROUPING SETS query that yields the mtype, age and
    dong-code marginals together instead of one scan each."""
    g = glob_for(ym)
    src = f"read_parquet('{g}')"
    ndays = calendar.monthrange(ym // 100, ym % 100)[1]
    out = {"ym": ym, "days_in_month": ndays}
    out.update(per_file_keys(con, ym) if dups else
               {"files": len(list((PARQUET / f"ym={ym}").iterdir())),
                "rows": None, "keys": None, "dup_keys": None,
                "extra_rows": None, "conflicting_keys": None})

    row = con.execute(f"""
        SELECT count(*),
               count(DISTINCT ym), min(ym), max(ym),
               count(DISTINCT dow_n), min(dow_n), max(dow_n),
               count(*) FILTER (WHERE dow_n = 0),
               count(DISTINCT arr_hour), min(arr_hour), max(arr_hour),
               count(*) FILTER (WHERE mean_min IS NULL),
               min(mean_min), max(mean_min),
               count(*) FILTER (WHERE masked),
               min(pop) FILTER (WHERE NOT masked),
               max(pop),
               count(*) FILTER (WHERE NOT masked AND pop = {MASK_FLOOR}),
               count(*) FILTER (WHERE NOT masked AND pop < {MASK_FLOOR}),
               count(*) FILTER (WHERE masked AND pop IS NOT NULL),
               count(*) FILTER (WHERE NOT masked AND pop IS NULL),
               sum(CASE WHEN masked THEN 0 ELSE pop END),
               sum(CASE WHEN masked THEN 1.5 ELSE pop END),
               sum(CASE WHEN masked THEN 3.0 ELSE pop END),
               sum(CASE WHEN masked THEN 1.5 ELSE pop END)
                 FILTER (WHERE o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}),
               sum(CASE WHEN masked THEN 1.5 ELSE pop END)
                 FILTER (WHERE o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}
                           AND d_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI}),
               sum(CASE WHEN masked THEN 1.5 ELSE pop END)
                 FILTER (WHERE mtype LIKE 'H%'
                           AND o_dong BETWEEN {SEOUL_LO} AND {SEOUL_HI})
        FROM {src}""").fetchone()
    (n, n_ym, ym_lo, ym_hi, n_dow, dow_lo, dow_hi, dow_bad,
     n_hour, hour_lo, hour_hi, mean_null, mean_lo, mean_hi,
     n_masked, pop_min, pop_max, n_at_floor, n_below_floor,
     masked_with_pop, unmasked_without_pop,
     vol_lo, vol_mid, vol_hi, vol_o_seoul, vol_both, vol_h) = row

    out["scan_rows"] = int(n)
    out["ym_values"] = [int(ym_lo), int(ym_hi)] if n_ym == 1 else "MIXED"
    out["dow"] = {"n_values": int(n_dow), "min": int(dow_lo), "max": int(dow_hi),
                  "unparseable_rows": int(dow_bad)}
    out["arr_hour"] = {"n_values": int(n_hour), "min": int(hour_lo),
                       "max": int(hour_hi)}
    out["mean_min"] = {"nulls": int(mean_null), "min": float(mean_lo),
                       "max": float(mean_hi)}
    out["masking"] = {
        "masked_rows": int(n_masked),
        "masked_cell_share": int(n_masked) / int(n),
        "min_unmasked_pop": float(pop_min),
        "max_pop": float(pop_max),
        "rows_exactly_at_floor": int(n_at_floor),
        "share_exactly_at_floor": int(n_at_floor) / int(n),
        "rows_below_floor_unmasked": int(n_below_floor),
        "masked_rows_carrying_a_value": int(masked_with_pop),
        "unmasked_rows_missing_a_value": int(unmasked_without_pop),
        "hard_floor_at_3.00": bool(float(pop_min) == MASK_FLOOR
                                   and int(n_below_floor) == 0),
    }

    # One scan, five marginals. mtype, age, sex, o_dong and d_dong are never
    # null in the data, so the grouping set a row came from is simply whichever
    # column is not null.
    marg = con.execute(f"""
        SELECT mtype, age, sex, o_dong, d_dong,
               count(*) AS n, sum(pop) AS p,
               count(*) FILTER (WHERE masked) AS k
        FROM {src}
        GROUP BY GROUPING SETS ((mtype), (age), (sex), (o_dong), (d_dong))
        """).fetchall()

    out["mtype"] = {m: {"rows": int(c), "pop_sum": float(p or 0),
                        "masked_rows": int(k)}
                    for m, a, s, o, d, c, p, k in marg if m is not None}
    out["mtype_codes"] = sorted(out["mtype"])
    out["age"] = {int(a): {"rows": int(c), "pop_sum": float(p or 0),
                           "masked_rows": int(k)}
                  for m, a, s, o, d, c, p, k in marg if a is not None}
    out["age_codes"] = sorted(out["age"])
    out["sex"] = {s: int(c) for m, a, s, o, d, c, p, k in marg if s is not None}

    as_o = {int(o) for m, a, s, o, d, c, p, k in marg if o is not None}
    as_d = {int(d) for m, a, s, o, d, c, p, k in marg if d is not None}
    all_codes = sorted(as_o | as_d)
    o_only = sorted(as_o - as_d)
    d_only = sorted(as_d - as_o)
    seoul = [c for c in all_codes if SEOUL_LO <= c <= SEOUL_HI]
    by_gu = {}
    for c in seoul:
        by_gu.setdefault(c // 1000, []).append(c)
    out["codes"] = {
        "n_total": len(all_codes),
        "n_seoul_dong": len(seoul),
        "n_seoul_gu": len(by_gu),
        "origin_only": o_only,
        "dest_only": d_only,
        "seoul_dong_per_gu": {str(g): len(v) for g, v in sorted(by_gu.items())},
        "all": all_codes,
    }

    # `이동인구(합)` is a monthly SUM over each occurrence of that weekday, so
    # the only correct per-day figure divides by the length of the month — not
    # by the occurrence count of each weekday, which yields a representative
    # week (../reference/DATA_REPORT.md 2.1a, memo/phase1c-rho.md 1).
    lo, mid, hi = float(vol_lo), float(vol_mid), float(vol_hi)
    out["volume"] = {
        "month_total_lo": lo, "month_total_mid": mid, "month_total_hi": hi,
        "per_day_lo": lo / ndays, "per_day_mid": mid / ndays,
        "per_day_hi": hi / ndays,
        "per_day_mid_o_seoul": float(vol_o_seoul) / ndays,
        "per_day_mid_both_seoul": float(vol_both) / ndays,
        # The numerator Phase 1c/1d actually uses, so a break here is felt by
        # rho-hat immediately rather than three phases later.
        "per_day_mid_h_origin_seoul": float(vol_h) / ndays,
    }

    out["names_unknown_codes"] = sorted(
        c for c in all_codes if c not in names)
    return out


def compare(base, new, names):
    """Every audit answer, as a pass/fail with the number behind it."""
    b_mtype = sorted({m for r in base for m in r["mtype_codes"]})
    n_mtype = sorted({m for r in new for m in r["mtype_codes"]})
    b_age = sorted({a for r in base for a in r["age_codes"]})
    n_age = sorted({a for r in new for a in r["age_codes"]})
    b_codes = sorted({c for r in base for c in r["codes"]["all"]})
    n_codes = sorted({c for r in new for c in r["codes"]["all"]})
    b_seoul = [c for c in b_codes if SEOUL_LO <= c <= SEOUL_HI]
    n_seoul = [c for c in n_codes if SEOUL_LO <= c <= SEOUL_HI]

    appeared = [c for c in n_codes if c not in set(b_codes)]
    vanished = [c for c in b_codes if c not in set(n_codes)]

    def named(cs):
        return [{"code": c, "name": names.get(c, "(not in the 2021-09 table)")}
                for c in cs]

    out = {
        "mtype": {
            "expected": MTYPES,
            "in_2020": b_mtype, "in_2026": n_mtype,
            "n_2020": len(b_mtype), "n_2026": len(n_mtype),
            "identical": b_mtype == n_mtype,
            "all_nine_present_2026": sorted(MTYPES) == n_mtype,
        },
        "age": {
            "expected": AGES,
            "in_2020": b_age, "in_2026": n_age,
            "n_2020": len(b_age), "n_2026": len(n_age),
            "identical": b_age == n_age,
            "top_band_2020": max(b_age), "top_band_2026": max(n_age),
            "top_band_still_80_open": max(n_age) == 80,
            "no_band_above_80": max(n_age) <= 80,
        },
        "codes": {
            "n_2020_union": len(b_codes), "n_2026_union": len(n_codes),
            "n_seoul_2020": len(b_seoul), "n_seoul_2026": len(n_seoul),
            "seoul_still_424": len(n_seoul) == 424,
            "code_space_identical": b_codes == n_codes,
            "appeared_in_2026": named(appeared),
            "vanished_since_2020": named(vanished),
            "appeared_seoul": named([c for c in appeared
                                     if SEOUL_LO <= c <= SEOUL_HI]),
            "per_month_seoul": {str(r["ym"]): r["codes"]["n_seoul_dong"]
                                for r in base + new},
            "per_month_total": {str(r["ym"]): r["codes"]["n_total"]
                                for r in base + new},
        },
        "masking": {
            "min_unmasked_pop_2020": sorted({r["masking"]["min_unmasked_pop"]
                                             for r in base}),
            "min_unmasked_pop_2026": sorted({r["masking"]["min_unmasked_pop"]
                                             for r in new}),
            "rows_below_floor_2026": sum(r["masking"]["rows_below_floor_unmasked"]
                                         for r in new),
            "hard_floor_all_months": all(r["masking"]["hard_floor_at_3.00"]
                                         for r in base + new),
            "cell_share_2020": _wmean(base, lambda r: r["masking"]["masked_cell_share"],
                                      lambda r: r["scan_rows"]),
            "cell_share_2026": _wmean(new, lambda r: r["masking"]["masked_cell_share"],
                                      lambda r: r["scan_rows"]),
            "share_at_floor_2020": _wmean(base,
                                          lambda r: r["masking"]["share_exactly_at_floor"],
                                          lambda r: r["scan_rows"]),
            "share_at_floor_2026": _wmean(new,
                                          lambda r: r["masking"]["share_exactly_at_floor"],
                                          lambda r: r["scan_rows"]),
        },
        "mean_min": {
            "nulls_2020": sum(r["mean_min"]["nulls"] for r in base),
            "nulls_2026": sum(r["mean_min"]["nulls"] for r in new),
            "never_null": all(r["mean_min"]["nulls"] == 0 for r in base + new),
            "range_2020": [min(r["mean_min"]["min"] for r in base),
                           max(r["mean_min"]["max"] for r in base)],
            "range_2026": [min(r["mean_min"]["min"] for r in new),
                           max(r["mean_min"]["max"] for r in new)],
        },
        "duplicates": {
            "extra_rows_2020": _osum(base, "extra_rows"),
            "extra_rows_2026": _osum(new, "extra_rows"),
            "dup_keys_2026": _osum(new, "dup_keys"),
            "conflicting_keys_2026": _osum(new, "conflicting_keys"),
        },
    }

    # Volume, season-matched where the same calendar month exists in 2020.
    vol = {}
    by_ym = {r["ym"]: r for r in base + new}
    for r in new:
        same = 202000 + r["ym"] % 100
        row = {"per_day_mid": r["volume"]["per_day_mid"],
               "per_day_mid_o_seoul": r["volume"]["per_day_mid_o_seoul"],
               "per_day_mid_both_seoul": r["volume"]["per_day_mid_both_seoul"],
               "per_day_mid_h_origin_seoul":
                   r["volume"]["per_day_mid_h_origin_seoul"],
               "rows": r["scan_rows"]}
        if same in by_ym:
            b = by_ym[same]
            row["vs_" + str(same)] = {
                "per_day_mid_ratio": r["volume"]["per_day_mid"]
                                     / b["volume"]["per_day_mid"],
                "both_seoul_ratio": r["volume"]["per_day_mid_both_seoul"]
                                    / b["volume"]["per_day_mid_both_seoul"],
                "h_origin_ratio": r["volume"]["per_day_mid_h_origin_seoul"]
                                  / b["volume"]["per_day_mid_h_origin_seoul"],
                "rows_ratio": r["scan_rows"] / b["scan_rows"],
            }
        vol[str(r["ym"])] = row
    if base:
        vol["_2020_mean_per_day_mid"] = (sum(r["volume"]["per_day_mid"]
                                             for r in base) / len(base))
        vol["_2020_per_month_per_day_mid"] = {
            str(r["ym"]): r["volume"]["per_day_mid"] for r in base}
    out["volume"] = vol

    # Age composition, share of monthly volume, so a re-based expansion factor
    # shows up as a level shift rather than hiding inside the totals.
    def age_share(rows):
        tot = {}
        for r in rows:
            for a, v in r["age"].items():
                tot[int(a)] = tot.get(int(a), 0.0) + v["pop_sum"]
        s = sum(tot.values())
        return {str(a): tot[a] / s for a in sorted(tot)}
    out["age_volume_share"] = {"2020": age_share(base), "2026": age_share(new)}

    def mtype_share(rows):
        tot = {}
        for r in rows:
            for m, v in r["mtype"].items():
                tot[m] = tot.get(m, 0.0) + v["pop_sum"]
        s = sum(tot.values())
        return {m: tot[m] / s for m in sorted(tot)}
    out["mtype_volume_share"] = {"2020": mtype_share(base),
                                 "2026": mtype_share(new)}
    return out


def _wmean(rows, val, weight):
    w = sum(weight(r) for r in rows)
    return sum(val(r) * weight(r) for r in rows) / w if w else None


def _osum(rows, key):
    """Sum, but None if the check was skipped for these months."""
    vals = [r[key] for r in rows]
    return None if any(v is None for v in vals) else sum(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--months", default="",
                    help="comma-separated YYYYMM; default is every month on disk")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dups", choices=("all", "new", "none"), default="all",
                    help="exact-duplicate check: every month, only the audited "
                         "months, or skip. It is the one check that needs a "
                         "third pass over the tree, so `new` is the cheap "
                         "option — 2020's figure is 475,266 extra rows "
                         "(../reference/DATA_REPORT.md 2.1b)")
    args = ap.parse_args()

    yms = ([int(x) for x in args.months.split(",") if x.strip()]
           or months_on_disk())
    base = [y for y in yms if y // 100 == 2020]
    new = [y for y in yms if y // 100 != 2020]
    if not new:
        raise SystemExit("no non-2020 month on disk — nothing to audit against. "
                         "Run eda/dl_mobility.py first.")
    if not base:
        raise SystemExit("no 2020 month selected — 2020 is the baseline the "
                         "verified field contract (../reference/DATA_REPORT.md 2, 3) is "
                         "written against, so the audit needs at least one.")
    print(f"baseline {base}\naudited  {new}", flush=True)

    names = code_names()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute("PRAGMA disable_progress_bar")

    per_month = []
    for ym in yms:
        want_dups = (args.dups == "all"
                     or (args.dups == "new" and ym // 100 != 2020))
        r = audit_month(con, ym, names, dups=want_dups)
        per_month.append(r)
        print(f"  {ym}  rows={r['scan_rows']:>12,}  codes={r['codes']['n_total']:>5}"
              f"  seoul={r['codes']['n_seoul_dong']:>4}"
              f"  mtype={len(r['mtype_codes'])}  age={len(r['age_codes'])}"
              f"  masked={r['masking']['masked_cell_share']:6.2%}"
              f"  min_pop={r['masking']['min_unmasked_pop']:.2f}"
              f"  mean_min_null={r['mean_min']['nulls']}", flush=True)

    b = [r for r in per_month if r["ym"] // 100 == 2020]
    n = [r for r in per_month if r["ym"] // 100 != 2020]
    verdict = compare(b, n, names)

    slim = []
    for r in per_month:
        r = dict(r)
        r["codes"] = {k: v for k, v in r["codes"].items() if k != "all"}
        slim.append(r)

    out = {"generated": "p22_schema_audit.py",
           "baseline_months": b and [r["ym"] for r in b],
           "audited_months": [r["ym"] for r in n],
           "verdict": verdict, "per_month": slim}
    path = os.path.join(RESULTS, "results_p22.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)

    v = verdict
    print("\n--- verdict ---")
    print(f"mtype    {v['mtype']['n_2026']} codes {v['mtype']['in_2026']}  "
          f"identical to 2020: {v['mtype']['identical']}")
    print(f"age      {v['age']['n_2026']} bands, top {v['age']['top_band_2026']}  "
          f"identical to 2020: {v['age']['identical']}")
    print(f"codes    2020 {v['codes']['n_2020_union']} "
          f"(Seoul {v['codes']['n_seoul_2020']}) -> "
          f"2026 {v['codes']['n_2026_union']} "
          f"(Seoul {v['codes']['n_seoul_2026']});  "
          f"appeared {len(v['codes']['appeared_in_2026'])}, "
          f"vanished {len(v['codes']['vanished_since_2020'])}")
    for c in v["codes"]["appeared_in_2026"]:
        print(f"           + {c['code']}  {c['name']}")
    for c in v["codes"]["vanished_since_2020"]:
        print(f"           - {c['code']}  {c['name']}")
    print(f"masking  min unmasked pop 2026 {v['masking']['min_unmasked_pop_2026']}"
          f", below floor {v['masking']['rows_below_floor_2026']}, "
          f"hard floor everywhere: {v['masking']['hard_floor_all_months']}")
    print(f"mean_min nulls 2026 {v['mean_min']['nulls_2026']}, "
          f"range {v['mean_min']['range_2026']}")
    for ym, r in v["volume"].items():
        if ym.startswith("_"):
            continue
        k = next((x for x in r if x.startswith("vs_")), None)
        tail = (f"  vs {k[3:]}: x{r[k]['per_day_mid_ratio']:.3f} all, "
                f"x{r[k]['both_seoul_ratio']:.3f} both-Seoul, "
                f"x{r[k]['h_origin_ratio']:.3f} H-origin") if k else ""
        print(f"volume   {ym} {r['per_day_mid']:>14,.0f}/day{tail}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
