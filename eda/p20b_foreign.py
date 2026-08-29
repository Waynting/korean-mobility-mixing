#!/usr/bin/env python
"""Phase 1b (L3) — registered foreigners as the second half of the rho-hat
denominator, at (ym, dong, sex, age), 2020-01..2026-07.

`p20_regpop.py` built the 주민등록 side: Korean nationals only. This builds the
등록외국인 side and writes `derived/foreign.parquet` on exactly the same keys, so
the two can be added — or not added, deliberately — one cell at a time.

WHAT THE PLAN EXPECTED, AND WHAT IS ACTUALLY THERE

../archive/REVISION_PLAN.md L3 says: "no source gives month x dong x age for foreigners",
and prescribes month-gu totals x year-dong shares with the error written down.
The first half of that is right and the second half is now unnecessary. The
법무부 출입국·외국인정책본부 publishes, as an attachment to its quarterly
"등록외국인 지역별 현황" post, a spreadsheet at
시도 x 시군구 x 행정동 x 성별 x 17 five-year age bands. So the real shape is

    month  x  gu                    (data.go.kr 15100022, 2022-01..2026-06)
    quarter x dong x sex x age      (moj.go.kr board 227 #25, 2020-12..2026-06)
    quarter x gu   x sex x age      (moj.go.kr board 227 #22, 2019-12..2020-09)

and the mismatch that has to be priced is *quarterly vs monthly*, not annual vs
monthly and not "no age at all". Two further things the age bands buy for free:

  - The 17 bands fold onto common.AGE_LABEL's 16 bins exactly — 0~4 and 5~9 sum
    to 0-9, 10~14 through 75~79 are 1:1, and 80세이상 is 80+. No interpolation
    across ages anywhere, in either half of the denominator.
  - Sex is 남성/여성, so the foreign side carries the same M/F split as regpop
    rather than being smeared across it.

WHAT IS STILL ALLOCATED, AND THE PRICE

  1. Months between quarter-ends are linearly interpolated, cell by cell, on
     (dong, sex, age). The error this makes is *measured*, not assumed: source A
     gives the true monthly gu totals from 2022-01, so the interpolated gu total
     can be compared with the published one in every mid-quarter month. Section
     4 reports that comparison; section 5 then rescales each (ym, gu) block onto
     A where A exists, which is the plan's "month-gu x quarter-dong" allocation
     with the composition coming from a quarter rather than a year.
  2. 2020-01..2020-11 has no dong file at all. The dong composition is carried
     back from 2020-12 and applied to the quarterly gu x sex x age totals that
     do exist (source C). Levels are right; the within-gu geography is
     2020-12's. Eleven months, flagged in the output as `back`.
  3. 2024-03 is published as a legacy BIFF .xls that openpyxl cannot open, so
     that quarter is simply absent and the interpolation spans 2023-12..2024-06.
  4. 2026-07 is past the last anchor (2026-06) and is held flat, flagged `held`.

  Every month carries a `basis` column saying which of these it is, so nothing
  downstream has to guess and no month is quietly treated as observed.

    python eda/dl_foreign.py      # first — fetch and verify the sources
    python eda/p20b_foreign.py    # then — allocate, check, write
"""
import csv
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from common import AGE_LABEL  # noqa: E402
from paths import DATA_ROOT, DERIVED, ROOT, require  # noqa: E402

SRC = DATA_ROOT / "raw" / "foreign"
FIRST_YM, LAST_YM = 202001, 202607
ROLLUP = {"총계", "총합계", "시도별총계", ""}
DONG_COL = ("행정동", "읍면동")
SEX_KO = {"남성": "M", "여성": "F"}

# The 17 published bands -> the 16 bins of common.AGE_LABEL. Only the first two
# merge; everything else is 1:1 and 80세이상 is already the open bin. This is a
# fold, not an interpolation — the one direction in which coarse bins are free.
BAND_TO_BIN = {"0~4": 0, "5~9": 0, "10~14": 10, "15~19": 15, "20~24": 20,
               "25~29": 25, "30~34": 30, "35~39": 35, "40~44": 40,
               "45~49": 45, "50~54": 50, "55~59": 55, "60~64": 60,
               "65~69": 65, "70~74": 70, "75~79": 75, "80+": 80}

out = {}


def norm_age(s):
    s = re.sub(r"\s+", "", str(s or ""))
    if "이상" in s:
        n = re.search(r"(\d+)", s)
        return f"{n.group(1)}+" if n else s
    n = re.findall(r"(\d+)", s)
    return f"{n[0]}~{n[1]}" if len(n) >= 2 else s


def norm_dong(s):
    """The 법무부 spelling -> the 생활이동 spelling.

    Same two differences p20_regpop.py found between the registration file and
    the product, and the same asymmetric fix: '.' -> '·' on both sides, but the
    제 of an ordinal is stripped only from the government side and only anchored
    at the end, because 홍제1동's 제 is part of its name.
    """
    s = str(s).strip().replace(".", "·")
    return re.sub(r"제(\d[\d·]*동)$", r"\1", s)


def months(first=FIRST_YM, last=LAST_YM):
    y, m = divmod(first, 100)
    while y * 100 + m <= last:
        yield y * 100 + m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def midx(ym):
    """Month index, so arithmetic on ym is arithmetic and not string surgery."""
    y, m = divmod(ym, 100)
    return y * 12 + (m - 1)


def read_moj(path):
    """One 법무부 spreadsheet -> long rows for Seoul, folded onto the 16 bins.

    Layout drifts across quarters and every drift here was found by a file that
    failed verification, not by reading the metadata: 2020-12 ships two columns
    both labelled 시도 whose values disagree (the right one is the one next to
    시군구), 2022 appends stray '111세'/'120세' columns, and 2026-06 renames
    행정동 to 읍면동 and 총계 to 시도별총계. So nothing may key off a position.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
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
        raise SystemExit(f"{path.name}: no header row")

    idx = {n: head.index(n) for n in ("시군구", "성별", "총합계") if n in head}
    for n in DONG_COL:
        if n in head:
            idx["행정동"] = head.index(n)
            break
    sido = [i for i, h in enumerate(head) if h == "시도"]
    idx["시도"] = max([i for i in sido if i < idx["시군구"]] or sido)
    band = {i: BAND_TO_BIN[norm_age(head[i])]
            for i in range(idx["총합계"] + 1, len(head))
            if head[i] and norm_age(head[i]) in BAND_TO_BIN}
    if sorted(set(band.values())) != sorted(AGE_LABEL):
        raise SystemExit(f"{path.name}: age bands do not cover the 16 bins")

    def num(r, i):
        if i >= len(r) or r[i] is None or str(r[i]).strip() in ("", "-"):
            return 0
        return int(float(str(r[i]).replace(",", "")))

    def txt(r, i):
        return str(r[i]).strip() if i < len(r) and r[i] is not None else ""

    has_dong = "행정동" in idx
    recs = []
    for r in body:
        if not txt(r, idx["시도"]).startswith("서울"):
            continue
        gu = txt(r, idx["시군구"])
        if gu in ROLLUP:
            continue
        dong = norm_dong(txt(r, idx["행정동"])) if has_dong else ""
        if has_dong and (dong in ROLLUP or not dong):
            continue
        sex = SEX_KO.get(txt(r, idx["성별"]))
        if sex is None:          # the 총계 roll-up; 남성 + 여성 covers it
            continue
        for i, age in band.items():
            v = num(r, i)
            if v:
                recs.append((gu, dong, sex, age, v))
    cols = ["gu", "dong", "sex", "age", "pop"]
    df = pd.DataFrame(recs, columns=cols)
    return df.groupby(cols[:-1], as_index=False)["pop"].sum()


def read_month_gu(path):
    """Source A: month x 시군구 totals, Seoul only."""
    text = path.read_bytes().decode("cp949")
    rows = list(csv.reader(io.StringIO(text)))
    head, body = [c.strip() for c in rows[0]], rows[1:]
    c = {n: head.index(n) for n in ("년", "월", "시도", "시군구", "등록외국인수")}
    recs = []
    for r in body:
        if not r or not r[0].strip() or not r[c["시도"]].strip().startswith("서울"):
            continue
        recs.append((int(r[c["년"]]) * 100 + int(r[c["월"]]),
                     r[c["시군구"]].strip(),
                     int(r[c["등록외국인수"]].replace(",", ""))))
    return pd.DataFrame(recs, columns=["ym", "gu", "pop"])


def main():
    require(SRC, "raw/foreign (run eda/dl_foreign.py first)")
    require(DERIVED / "regpop.parquet", "derived/regpop.parquet (run p20_regpop.py)")

    # ---- 1. the dong name -> product code map, taken from regpop -----------
    # regpop.parquet already carries every registration dong with the 생활이동
    # 7-digit code attached, including the ones created inside the window and
    # 구로구 항동, which p20_regpop resolved to 오류2동. Reusing that map rather
    # than rebuilding it means the two halves of the denominator cannot end up
    # keyed differently, which is the only way they are ever added.
    reg = pd.read_parquet(DERIVED / "regpop.parquet")
    codes = (reg[["gu", "dong", "dong_code", "gu_code", "stable_unit"]]
             .drop_duplicates(subset=["gu", "dong"]))
    name2code = {(g, d): (c, gc, su) for g, d, c, gc, su in
                 zip(codes.gu, codes.dong, codes.dong_code,
                     codes.gu_code, codes.stable_unit)}
    print(f"code map from regpop: {len(name2code)} (gu, dong) names, "
          f"{codes.dong_code.nunique()} product dong")

    # ---- 2. the quarterly anchors ------------------------------------------
    dong_files = sorted(SRC.glob("moj_dong_age_*.xlsx"))
    gu_files = sorted(SRC.glob("moj_gu_age_*.xlsx"))
    if not dong_files:
        raise SystemExit(f"no moj_dong_age_*.xlsx under {SRC} — run dl_foreign.py")
    anchors, unmatched = {}, []
    for p in dong_files:
        q = int(p.stem.split("_")[-1])
        df = read_moj(p)
        hit = pd.Series([name2code.get((g, d)) for g, d in
                         zip(df.gu, df.dong)], index=df.index)
        bad = df[hit.isna()]
        if len(bad):
            unmatched.append({"ym": q, "n_names": bad.groupby(["gu", "dong"]).ngroups,
                              "pop": int(bad["pop"].sum()),
                              "share": float(bad["pop"].sum() / df["pop"].sum()),
                              "names": sorted({f"{g} {d}" for g, d in
                                               zip(bad.gu, bad.dong)})})
        # Legacy dong names — 중구 신당1동, 종로구 청운동 and the like, abolished
        # before the window but still sitting in some registered addresses — are
        # spread over the matched dong of the same gu in the same (sex, age)
        # cell rather than dropped. They are 0.04% of Seoul; dropping them would
        # be defensible and keeping them is cheap, but silently dropping them
        # would not be.
        keep = df[hit.notna()].copy()
        if len(bad):
            w = keep.groupby(["gu", "sex", "age"])["pop"].transform("sum")
            extra = bad.groupby(["gu", "sex", "age"])["pop"].sum()
            key = list(zip(keep.gu, keep.sex, keep.age))
            add = pd.Series([extra.get(k, 0.0) for k in key], index=keep.index)
            keep["pop"] = keep["pop"] + add * keep["pop"] / w.where(w > 0, 1)
        keep = keep.assign(
            dong_code=[name2code[(g, d)][0] for g, d in zip(keep.gu, keep.dong)],
            gu_code=[name2code[(g, d)][1] for g, d in zip(keep.gu, keep.dong)])
        anchors[q] = (keep.groupby(["dong_code", "gu_code", "gu", "sex", "age"],
                                   as_index=False)["pop"].sum())
        print(f"  {q}  {len(anchors[q]):>6,} cells  "
              f"{anchors[q]['pop'].sum():>9,.0f} people  "
              f"{anchors[q].dong_code.nunique()} product dong")

    first_dong_q = min(anchors)

    # ---- 3. 2020 before the dong file existed ------------------------------
    # Source C has the gu x sex x age totals for 2019-12..2020-09. The dong
    # split inside each (gu, sex, age) cell is taken from the earliest quarter
    # that has one. That is the one genuinely assumed thing in this script.
    shares = anchors[first_dong_q].copy()
    denom = shares.groupby(["gu", "sex", "age"])["pop"].transform("sum")
    gu_denom = shares.groupby("gu")["pop"].transform("sum")
    # Where a (gu, sex, age) cell is empty in the reference quarter there is no
    # composition to borrow, so the gu's overall dong split stands in.
    shares["w"] = (shares["pop"] / denom).where(
        denom > 0, shares["pop"] / gu_denom)
    back = []
    for p in gu_files:
        q = int(p.stem.split("_")[-1])
        if q in anchors:
            continue
        g = read_moj(p)[["gu", "sex", "age", "pop"]]
        g = g.groupby(["gu", "sex", "age"], as_index=False)["pop"].sum()
        m = shares.merge(g, on=["gu", "sex", "age"], how="left",
                         suffixes=("_ref", "_gu"))
        m["pop"] = m["pop_gu"].fillna(0) * m["w"]
        anchors[q] = m[["dong_code", "gu_code", "gu", "sex", "age", "pop"]]
        back.append(q)
        print(f"  {q}  back-extrapolated from {first_dong_q} shares  "
              f"{anchors[q]['pop'].sum():>9,.0f} people")

    qs = sorted(anchors)
    want_q = [y * 100 + m for y in range(qs[0] // 100, qs[-1] // 100 + 1)
              for m in (3, 6, 9, 12) if qs[0] <= y * 100 + m <= qs[-1]]
    gap_q = [q for q in want_q if q not in anchors]
    print(f"\n{len(qs)} quarterly anchors: {qs[0]}..{qs[-1]}"
          f"{f'  MISSING: {gap_q}' if gap_q else ''}")

    # ---- 4. monthly interpolation ------------------------------------------
    # Every anchor is a stock at the last day of its month, so a month between
    # two anchors is a straight line in month index. A month past the last
    # anchor is held, never extrapolated on a trend — one held month is an
    # honest gap, an extrapolated one is a number nobody published.
    wide = None
    for q in qs:
        a = anchors[q].set_index(["dong_code", "gu_code", "gu", "sex", "age"])["pop"]
        a.name = q
        wide = a.to_frame() if wide is None else wide.join(a, how="outer")
    wide = wide.fillna(0.0)

    frames, basis = [], {}
    for ym in months():
        lo = max([q for q in qs if q <= ym], default=None)
        hi = min([q for q in qs if q >= ym], default=None)
        if lo is None:                      # before the first anchor
            v, how = wide[hi], "held"
        elif hi is None:                    # after the last anchor
            v, how = wide[lo], "held"
        elif lo == hi:
            v, how = wide[lo], "anchor"
        else:
            t = (midx(ym) - midx(lo)) / (midx(hi) - midx(lo))
            v, how = wide[lo] * (1 - t) + wide[hi] * t, "interp"
        if ym < first_dong_q:
            how = "back"                    # dong split is 2020-12's, section 3
        basis[ym] = how
        f = v.rename("pop").reset_index()
        f.insert(0, "ym", ym)
        f["basis"] = how
        frames.append(f)
    monthly = pd.concat(frames, ignore_index=True)

    # ---- 5. calibrate to the published monthly gu totals --------------------
    A = read_month_gu(SRC / "moj_gu_month.csv")
    # 2025-02 spells 도봉구 as 도붕구. Repair by nearest known gu name rather
    # than by hand, and report it, because the same class of typo will recur.
    known = set(monthly["gu"].unique())
    fix = {g: next(k for k in known if sum(a != b for a, b in zip(g, k)) <= 1
                   and len(g) == len(k))
           for g in set(A["gu"]) - known}
    if fix:
        print(f"\ngu names in source A not in the product: repaired {fix}")
        A["gu"] = A["gu"].replace(fix)
    A = A[A["gu"].isin(known)]

    got = (monthly.groupby(["ym", "gu"], as_index=False)["pop"].sum()
           .rename(columns={"pop": "interp"}))
    cmp = A.merge(got, on=["ym", "gu"], how="inner")
    cmp["rel"] = cmp["interp"] / cmp["pop"] - 1
    cmp["qend"] = cmp["ym"].isin(qs)
    val = cmp[cmp.qend]
    mid = cmp[~cmp.qend]
    print(f"\ncalibration check against source A, {cmp['ym'].nunique()} months "
          f"x 25 gu:")
    print(f"  quarter-end months (the two publications should agree exactly): "
          f"n={len(val)}  median |rel| {val['rel'].abs().median():.5f}  "
          f"max |rel| {val['rel'].abs().max():.5f}")
    print(f"  mid-quarter months (this is the interpolation error): "
          f"n={len(mid)}  median |rel| {mid['rel'].abs().median():.5f}  "
          f"p95 {mid['rel'].abs().quantile(.95):.5f}  "
          f"max {mid['rel'].abs().max():.5f}")

    # ---- 4b. what interpolation costs the *composition* ---------------------
    # Section 4 prices the gu totals, and section 5 then removes that error
    # wherever source A exists. Nothing removes the error inside a gu — the
    # dong x sex x age split of a mid-quarter month is interpolated and stays
    # interpolated. Leave-one-anchor-out prices it: predict each interior
    # quarter from its two neighbours and compare with what was published.
    # Reported as total variation distance, i.e. the share of Seoul's foreign
    # population that the prediction puts in the wrong cell.
    hold = []
    for i in range(1, len(qs) - 1):
        q, a, b = qs[i], qs[i - 1], qs[i + 1]
        t = (midx(q) - midx(a)) / (midx(b) - midx(a))
        pred = wide[a] * (1 - t) + wide[b] * t
        obs = wide[q]
        p, o = pred / pred.sum(), obs / obs.sum()
        by_dong_p = p.groupby(level="dong_code").sum()
        by_dong_o = o.groupby(level="dong_code").sum()
        hold.append({"ym": q, "months_apart": midx(b) - midx(a),
                     "tvd_cell": float((p - o).abs().sum() / 2),
                     "tvd_dong": float((by_dong_p - by_dong_o).abs().sum() / 2),
                     "back_extrapolated": q < first_dong_q})
    real = [h for h in hold if not h["back_extrapolated"]]
    if real:
        tv = sorted(h["tvd_cell"] for h in real)
        td = sorted(h["tvd_dong"] for h in real)
        print(f"\nleave-one-anchor-out, {len(real)} interior quarters "
              f"(the within-gu allocation error, which nothing corrects):")
        print(f"  cell-level (dong x sex x age) TVD  median {tv[len(tv)//2]:.4f}"
              f"  max {tv[-1]:.4f}")
        print(f"  dong-level TVD                     median {td[len(td)//2]:.4f}"
              f"  max {td[-1]:.4f}")

    sc = cmp.assign(f=cmp["pop"] / cmp["interp"].where(cmp["interp"] > 0))[
        ["ym", "gu", "f"]]
    monthly = monthly.merge(sc, on=["ym", "gu"], how="left")
    calibrated = monthly["f"].notna()
    monthly["pop"] = monthly["pop"] * monthly["f"].fillna(1.0)
    monthly.loc[calibrated, "basis"] = monthly.loc[calibrated, "basis"] + "+A"
    monthly = monthly.drop(columns="f")

    # ---- 6. write ----------------------------------------------------------
    monthly["age"] = monthly["age"].astype("int64")
    monthly["dong_code"] = monthly["dong_code"].astype("int64")
    monthly["gu_code"] = monthly["gu_code"].astype("int64")
    monthly = monthly[monthly["pop"] > 0].reset_index(drop=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(DERIVED / "foreign.parquet", index=False)
    print(f"\nwrote {DERIVED / 'foreign.parquet'} ({len(monthly):,} rows, "
          f"{monthly.ym.nunique()} months, {monthly.dong_code.nunique()} dong)")

    # ---- 7. what it does to the denominator --------------------------------
    kor = (reg.groupby(["ym", "dong_code", "sex", "age"], as_index=False)["pop"]
           .sum().rename(columns={"pop": "korean"}))
    frn = (monthly.groupby(["ym", "dong_code", "sex", "age"], as_index=False)["pop"]
           .sum().rename(columns={"pop": "foreign"}))
    both = kor.merge(frn, on=["ym", "dong_code", "sex", "age"], how="outer").fillna(0)
    city = both.groupby("ym")[["korean", "foreign"]].sum()
    city["share"] = city["foreign"] / (city["korean"] + city["foreign"])
    print(f"\nSeoul foreign share of the denominator: "
          f"{city['share'].min():.3%}..{city['share'].max():.3%} "
          f"(2020-01 {city['share'].iloc[0]:.3%}, "
          f"2026-07 {city['share'].iloc[-1]:.3%})")

    work = both[both["age"].between(20, 45)]
    by_dong = (work.groupby("dong_code")[["korean", "foreign"]].sum()
               .assign(share=lambda d: d.foreign / (d.korean + d.foreign)))
    names = codes.drop_duplicates("dong_code").set_index("dong_code")
    by_dong = by_dong.join(names[["gu", "dong"]])
    top = by_dong.sort_values("share", ascending=False).head(15)
    print("\nmost affected dong, ages 20-49, window mean:")
    for c, r in top.iterrows():
        print(f"  {r['gu']} {r['dong']:<12} {r['share']:6.2%}  "
              f"foreign {r['foreign']/79:>8,.0f}  korean {r['korean']/79:>8,.0f}")

    by_age = both.groupby("age")[["korean", "foreign"]].sum()
    by_age["share"] = by_age["foreign"] / (by_age["korean"] + by_age["foreign"])

    out["sources"] = {
        "A_month_gu": {"file": "moj_gu_month.csv",
                       "granularity": "month x 시군구, no age, no sex",
                       "months": sorted(int(x) for x in A["ym"].unique())},
        "B_quarter_dong_age_sex": {
            "files": [p.name for p in dong_files],
            "granularity": "quarter x 행정동 x 성별 x 17 five-year bands",
            "quarters": sorted(int(p.stem.split('_')[-1]) for p in dong_files)},
        "C_quarter_gu_age_sex": {
            "files": [p.name for p in gu_files],
            "granularity": "quarter x 시군구 x 성별 x 17 five-year bands",
            "quarters": sorted(int(p.stem.split('_')[-1]) for p in gu_files)},
    }
    out["age_fold"] = {k: AGE_LABEL[v] for k, v in BAND_TO_BIN.items()}
    out["unmatched_dong_names"] = unmatched[:3] + ([{"...": len(unmatched)}]
                                                   if len(unmatched) > 3 else [])
    out["unmatched_share_max"] = max((u["share"] for u in unmatched), default=0.0)
    out["anchors"] = {str(q): float(anchors[q]["pop"].sum()) for q in qs}
    out["back_extrapolated_quarters"] = sorted(back)
    out["basis_by_month"] = {str(k): v for k, v in basis.items()}
    out["calibrated_months"] = sorted(int(x) for x in cmp["ym"].unique())
    out["calibration"] = {
        "n_month_gu": int(len(cmp)),
        "quarter_end": {"n": int(len(val)),
                        "median_abs_rel": float(val["rel"].abs().median()),
                        "max_abs_rel": float(val["rel"].abs().max())},
        "mid_quarter": {"n": int(len(mid)),
                        "median_abs_rel": float(mid["rel"].abs().median()),
                        "p95_abs_rel": float(mid["rel"].abs().quantile(.95)),
                        "max_abs_rel": float(mid["rel"].abs().max())},
        "gu_name_repairs": fix,
    }
    out["leave_one_anchor_out"] = hold
    out["seoul_share_by_month"] = {str(k): float(v)
                                   for k, v in city["share"].items()}
    out["share_by_age"] = {AGE_LABEL[int(a)]: float(s)
                           for a, s in by_age["share"].items()}
    out["top_dong_20_49"] = [
        {"dong_code": int(c), "gu": r["gu"], "dong": r["dong"],
         "foreign_share": float(r["share"]),
         "foreign_mean": float(r["foreign"] / 79),
         "korean_mean": float(r["korean"] / 79)}
        for c, r in by_dong.sort_values("share", ascending=False).head(25).iterrows()]
    out["rows"] = int(len(monthly))
    with open(f"{ROOT}/eda/results_p20b.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p20b.json")


if __name__ == "__main__":
    main()
