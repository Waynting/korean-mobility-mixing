#!/usr/bin/env python
"""Phase 1d — dong-level covariates, keyed on 생활이동's own 7-digit dong code.

`p21_rho.py` can only report the *unconditional* cross-dong spread of rho-hat,
because the plan's sigma_d(a,t) is the SD of the residual after regressing
log rho-hat on dong-level covariates and the only covariate on disk was
population. This writes the rest of them:

    derived/dong_covariates.parquet   (dong_code, year) -> 면적 km^2, 사업체수,
                                      종사자수, and the same two split by KSIC
                                      division so Phase 4 can pick out 보건 (Q)
                                      and 도소매 (G) without a second ingest.

Run `eda/dl_covariates.py` first; it fetches and verifies the raw files.

THE JOIN IS THE WHOLE DIFFICULTY. Neither source speaks 생활이동's code space:

  * the boundary file carries `adm_cd`, the 7-digit 통계청 행정동 코드. That is
    *nearly* the product's code — 423 of 424 are identical, which is a finding in
    itself — but 구로구 is renumbered between the two (the product's 오류2동 is
    1117068, 통계청's is 1117073, and 통계청 additionally has 항동 at 1117074 which
    the product has never had). Joining on `adm_cd` therefore looks like it works
    and silently loses a dong. It is used here only as a cross-check.
  * the 사업체조사 CSV carries a bare code that is 7 digits up to 2021 and 8 from
    2022, with no names at all — the names come from the 한국행정구역분류 sheet
    packed inside the same zip.

So both are joined on (구, 동) *names*, and the normalisation has to be
side-specific. `p20_regpop.py` strips a 제 that the registration file writes
before an ordinal (제1동 -> 1동), anchored at the end of the string because
홍제1동's own 제 is part of the name. Neither source here writes that 제, so that
step must NOT be applied to them; all they need is the separator, which they
write as '.' where the product writes '·'.

Names left over after the crosswalk join are resolved through `regpop.parquet`,
whose (구, 동) -> dong_code map already encodes p20's two extra resolutions:
dong created inside the window inherit their parent's product code through the
stable unit (상일1/2동, 개포3동, 신설동/용두동), and 항동 — which has no product
code at all — was assigned to 오류2동 by p21 section 1.6. Both matter here: 항동's
*population* is already inside 오류2동's denominator, so its *area* has to be
inside 오류2동's area too or the density covariate is wrong for that dong.

Nothing is dropped silently. Every unmatched name is printed and written to
results_p20c.json.

WHAT IS MISSING. 상업 연면적 (commercial gross floor area) per 행정동 is not here
and could not be obtained: 서울시 건축물 연면적 is published 자치구-only and the
건축물대장 behind it is keyed on 법정동. `worker_G` (도소매) and `worker_I`
(숙박·음식점) are the nearest available stand-ins and are a different quantity —
employment, not floor space. The plan's covariate list should be read as
satisfied on 면적 and 종사자, open on 연면적.

    python eda/dl_covariates.py
    python eda/p20c_covariates.py
"""
import csv
import io
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpyxl                                          # noqa: E402
import pandas as pd                                      # noqa: E402
from dl_covariates import (COL_DONG, COL_INDUSTRY, COL_WORKERS,  # noqa: E402
                           COL_WORKERS_F, COL_WORKERS_M, COL_YEAR,
                           feature_area_m2, pick, _zip_names)
from paths import DATA_ROOT, DERIVED, ROOT, require      # noqa: E402

RAW = DATA_ROOT / "raw" / "covariates"

# Which boundary vintage stands for which year of the data window. 2026 has no
# January release, so the window's last year uses the July one — which is also
# the only vintage that sees the 2025-07 용신동 split.
AREA_VINTAGE = {2020: "20200101", 2021: "20210101", 2022: "20220101",
                2023: "20230101", 2024: "20240101", 2025: "20250101",
                2026: "20260701"}
YEARS = sorted(AREA_VINTAGE)
# The vintage `area_km2_ref` is frozen at. 2020 because that is the year the
# pipeline is actually built on and the one the 424-dong cross-section describes.
REF_YEAR = 2020

# KSIC divisions that make up street-level commerce. Named rather than inlined
# because this grouping is a judgement, not a definition, and Phase 4 will want
# to argue with it.
RETAIL_FOOD = ("G", "I")     # 도매 및 소매업 / 숙박 및 음식점업

SEOUL_CODE = re.compile(r"^11\d{5,6}$")

out = {}


def norm(name):
    """Source spelling -> product spelling: the separator, and nothing else.

    The code tables and the boundary file write 종로1.2.3.4가동, the product
    writes 종로1·2·3·4가동.
    """
    return name.replace(".", "·").strip()


# `p20_regpop._drop_ordinal`, anchored at the end of the string for the same
# reason. It is NOT applied blanket here, and the reason is a real trap: SGIS is
# inconsistent with itself. It writes 강동구 상일제1동 — 제 + ordinal, like the
# registration — but 서대문구 홍제1동, where the 제 belongs to the name. A blanket
# strip would turn 홍제1동 into 홍1동 and lose the dong; not stripping at all loses
# 상일제1동. So it is used only as a *fallback*, after the literal name has failed
# to match, and every name that needed it is reported.
_ORDINAL = re.compile(r"제(\d[\d·]*동)$")


def load_targets():
    """The 424 product dong, plus every name that is known to map into one."""
    cwp = require(DERIVED / "dong_crosswalk.parquet",
                  "dong_crosswalk.parquet (run eda/p20_regpop.py first)")
    rpp = require(DERIVED / "regpop.parquet",
                  "regpop.parquet (run eda/p20_regpop.py first)")
    cw = pd.read_parquet(cwp)
    reg = pd.read_parquet(rpp, columns=["gu", "dong", "dong_code"])
    fallback = reg.drop_duplicates()
    dup = fallback[fallback.duplicated(["gu", "dong"], keep=False)]
    if len(dup):
        raise SystemExit(f"regpop maps a name to more than one product code:\n{dup}")
    primary = {(r.gu, r.dong): int(r.dong_code) for r in cw.itertuples()}
    extra = {(r.gu, r.dong): int(r.dong_code) for r in fallback.itertuples()
             if (r.gu, r.dong) not in primary}
    return cw, primary, extra


def resolve(rows, primary, extra, what):
    """[(구, 동)] -> dong_code, recording how each name had to be matched.

    A left join that reports its failures rather than filtering them away: an
    unmatched dong is area or employment that has fallen out of the covariate,
    and neither the regression nor its residual SD would ever notice.

    Four ways in, in order, each one weaker than the last:
      crosswalk           the name is one of the 424 product dong
      regpop              the name is a registration dong p20 already folded into
                          a product dong (항동, 상일1/2동, 개포3동, 신설동/용두동)
      ...+ordinal         the same, after dropping a 제 the source put in front of
                          an ordinal that the product does not
      unmatched           reported by name, never dropped quietly
    """
    codes, how = [], {"crosswalk": [], "regpop": [], "crosswalk+ordinal": [],
                      "regpop+ordinal": [], "unmatched": []}
    for gu, dong in rows:
        stripped = _ORDINAL.sub(r"\1", dong)
        tries = [("", dong)] + ([("+ordinal", stripped)] if stripped != dong else [])
        code = None
        for suffix, name in tries:
            if (gu, name) in primary:
                code = primary[(gu, name)]
                how["crosswalk" + suffix].append(f"{gu} {dong}")
                break
            if (gu, name) in extra:
                code = extra[(gu, name)]
                how["regpop" + suffix].append(f"{gu} {dong}")
                break
        if code is None:
            how["unmatched"].append(f"{gu} {dong}")
        codes.append(code)
    n = len(rows)
    print(f"    {what}: {len(how['crosswalk'])}/{n} on the crosswalk"
          + "".join(f", {len(v)} via {k} ({', '.join(sorted(v))})"
                    for k, v in how.items()
                    if k not in ("crosswalk", "unmatched") and v)
          + f", {len(how['unmatched'])} unmatched")
    for name in how["unmatched"]:
        print(f"      UNMATCHED  {name}")
    rep = {"n_source": n, "matched_crosswalk": len(how["crosswalk"])}
    rep.update({k: sorted(v) for k, v in how.items() if k != "crosswalk"})
    return codes, rep


# ---------------------------------------------------------------- 면적
def read_boundary(vintage, primary, extra):
    """One vintage -> (dong_code -> km^2), plus how the codes lined up."""
    p = require(RAW / f"hjd_{vintage}.geojson",
                f"boundary vintage {vintage} (run eda/dl_covariates.py)")
    gj = json.loads(p.read_bytes().decode("utf-8-sig"))
    feats = [f for f in gj["features"]
             if str(f["properties"]["adm_cd"]).startswith("11")]
    rows, area, adm = [], [], []
    for f in feats:
        parts = f["properties"]["adm_nm"].split()
        rows.append((parts[1], norm(parts[2])))
        area.append(feature_area_m2(f["geometry"]) / 1e6)
        adm.append(str(f["properties"]["adm_cd"]))
    codes, rep = resolve(rows, primary, extra, f"boundary {vintage}")

    # The 통계청 code is *almost* the product code, up to the 2023 vintage: 423 of
    # 424 are identical and the 424th is 구로구, which 통계청 renumbered when 항동
    # was created. Reported rather than used, because a reader who joins on it
    # gets 423 dong and is never told about the one that vanished. From the 2024
    # vintage 통계청 widens the code to 8 digits and the resemblance ends.
    width = len(adm[0]) if adm else 0
    rep["adm_cd_width"] = width
    if width == 7:
        pairs = [(int(a), c, f"{g} {d}") for a, c, (g, d) in zip(adm, codes, rows)
                 if c is not None]
        rep["adm_cd_equals_dong_code"] = sum(1 for a, c, _ in pairs if a == c)
        rep["adm_cd_differs"] = [{"adm_cd": a, "dong_code": c, "name": n}
                                 for a, c, n in sorted(pairs) if a != c]
    else:
        rep["adm_cd_equals_dong_code"] = None
        rep["adm_cd_differs"] = "not comparable: 통계청 widened adm_cd to 8 digits"

    df = pd.DataFrame({"dong_code": codes, "area_km2": area,
                       "gu": [g for g, _ in rows], "dong": [d for _, d in rows]})
    unmatched_km2 = float(df.loc[df.dong_code.isna(), "area_km2"].sum())
    g = (df.dropna(subset=["dong_code"])
           .groupby("dong_code")
           .agg(area_km2=("area_km2", "sum"), n_polygons=("area_km2", "size")))
    g.index = g.index.astype("int64")
    rep.update({"seoul_area_km2": round(float(df.area_km2.sum()), 4),
                "assigned_area_km2": round(float(g.area_km2.sum()), 4),
                "unassigned_area_km2": round(unmatched_km2, 4),
                "dong_code_covered": int(len(g)),
                "pooled": {int(k): int(v) for k, v in
                           g.loc[g.n_polygons > 1, "n_polygons"].items()}})
    return g, rep


# ------------------------------------------------------- 사업체 / 종사자
def read_code_table(z):
    """(survey dong code -> (구, 동)) from whichever layout this zip carries.

    The workbook is 한국행정구역분류 in 2020/21/23/24 and a flat YEAR/ZONE_CD/…
    table in 2022, and the sheet is called three different things across the
    five years, so the rows are recognised by their content rather than by a
    header: a Seoul dong row is the one holding a 11xxxxx(x) code, a 시군구 name
    ending in 구, and a 읍면동 name ending in 동.
    """
    xs = [(i, n) for i, n in _zip_names(z) if n.lower().endswith((".xlsx", ".xls"))]
    table = {}
    for info, name in xs:
        wb = openpyxl.load_workbook(io.BytesIO(z.read(info)), read_only=True,
                                    data_only=True)
        for sn in wb.sheetnames:
            for row in wb[sn].iter_rows(values_only=True):
                cells = [("" if c is None else str(c)).strip() for c in row]
                code = next((c for c in cells if SEOUL_CODE.match(c)), None)
                if code is None:
                    continue
                gu = next((c for c in cells if c.endswith("구") and len(c) >= 2), None)
                dong = next((c for c in reversed(cells)
                             if c.endswith("동") and c != gu), None)
                if gu and dong:
                    table.setdefault(code, (gu, norm(dong)))
        wb.close()
    if not table:
        raise SystemExit("no 행정구역 code table found in the establishment zip")
    return table


def build_code_table(years):
    """survey/boundary dong code -> (구, 동), unioned over every file on disk.

    A zip's own classification sheet is not enough. The 2021 zip ships the
    2020-12-31 table, so it cannot name the three 강동구 codes the 2021-07 상일동
    split created (1125075 강일동, 1125076/7 상일제1/2동) — 3,597 establishments
    that a per-zip lookup drops on the floor. The boundary vintages supply them.

    Safe to union only because 통계청 does not recycle a code: 강일동 moved
    1125051 -> 1125075 -> 11250750 across the window and nothing took over the
    codes it left. That is asserted rather than assumed.
    """
    table, seen = {}, {}
    for y in years:
        z = zipfile.ZipFile(io.BytesIO((RAW / f"estab_{y}.zip").read_bytes()))
        for cd, nm in read_code_table(z).items():
            seen.setdefault(cd, {}).setdefault(nm, []).append(f"estab {y}")
    for p in sorted(RAW.glob("hjd_*.geojson")):
        gj = json.loads(p.read_bytes().decode("utf-8-sig"))
        for f in gj["features"]:
            cd = str(f["properties"]["adm_cd"])
            if not cd.startswith("11"):
                continue
            parts = f["properties"]["adm_nm"].split()
            seen.setdefault(cd, {}).setdefault(
                (parts[1], norm(parts[2])), []).append(p.stem)
    for cd, names in seen.items():
        # Two spellings of the same dong are fine (상일1동 / 상일제1동); two
        # different dong under one code would mean the union is unsafe.
        bases = {(g, _ORDINAL.sub(r"\1", d)) for g, d in names}
        if len(bases) > 1:
            raise SystemExit(f"code {cd} names more than one dong across "
                             f"sources: {names} — the union is not safe, key "
                             f"the lookup by year instead")
        table[cd] = sorted(names, key=len)[0]
    return table


def read_establishments(year, code_table, primary, extra):
    """One survey year -> per-dong_code counts, total and by KSIC division."""
    p = require(RAW / f"estab_{year}.zip",
                f"establishment survey {year} (run eda/dl_covariates.py)")
    z = zipfile.ZipFile(io.BytesIO(p.read_bytes()))
    info = [i for i, n in _zip_names(z) if n.lower().endswith(".csv")][0]
    raw = z.read(info)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    rd = csv.reader(io.StringIO(text))
    header = next(rd)
    ic = {k: header.index(pick(header, v, k)) for k, v in
          (("year", COL_YEAR), ("dong", COL_DONG), ("ind", COL_INDUSTRY),
           ("tot", COL_WORKERS), ("m", COL_WORKERS_M), ("f", COL_WORKERS_F))}

    def num(cell):
        cell = (cell or "").strip().replace(",", "")
        return int(cell) if cell and cell != "-" else 0

    acc, unknown = {}, {}
    for r in rd:
        if not r or not r[ic["dong"]].strip():
            continue
        cd = r[ic["dong"]].strip()
        if cd not in code_table:
            unknown[cd] = unknown.get(cd, 0) + 1
            continue
        key = code_table[cd]
        d = acc.setdefault(key, {"estab": 0, "worker": 0, "worker_m": 0,
                                 "worker_f": 0, "by": {}})
        d["estab"] += 1
        d["worker"] += num(r[ic["tot"]])
        d["worker_m"] += num(r[ic["m"]])
        d["worker_f"] += num(r[ic["f"]])
        div = (r[ic["ind"]] or "").strip() or "?"
        b = d["by"].setdefault(div, [0, 0])
        b[0] += 1
        b[1] += num(r[ic["tot"]])
    if unknown:
        raise SystemExit(f"{len(unknown)} survey dong codes named by no source on "
                         f"disk: {sorted(unknown)[:10]} "
                         f"({sum(unknown.values()):,} establishments)")

    names = sorted(acc)
    codes, rep = resolve(names, primary, extra, f"사업체조사 {year}")
    divisions = sorted({d for v in acc.values() for d in v["by"]})
    recs = []
    for (gu, dong), code in zip(names, codes):
        if code is None:
            continue
        d = acc[(gu, dong)]
        rec = {"dong_code": code, "estab": d["estab"], "worker": d["worker"],
               "worker_m": d["worker_m"], "worker_f": d["worker_f"]}
        for div in divisions:
            n, w = d["by"].get(div, (0, 0))
            rec[f"estab_{div}"] = n
            rec[f"worker_{div}"] = w
        recs.append(rec)
    df = pd.DataFrame(recs).groupby("dong_code", as_index=True).sum()
    df.index = df.index.astype("int64")
    rep.update({"establishments": int(df.estab.sum()),
                "workers": int(df.worker.sum()),
                "divisions": divisions,
                "dong_code_covered": int(len(df))})
    return df, rep, divisions


def main():
    require(RAW, "raw/covariates (run eda/dl_covariates.py first)")
    cw, primary, extra = load_targets()
    print(f"targets: {len(cw)} product dong, {len(extra)} extra registration "
          f"names that fold into them")

    print("\n면적 (행정동 boundary geometry, one vintage per year):")
    areas, area_rep = {}, {}
    for y in YEARS:
        v = AREA_VINTAGE[y]
        g, rep = read_boundary(v, primary, extra)
        areas[y] = g
        area_rep[v] = rep
        print(f"    -> {rep['dong_code_covered']} dong_code, "
              f"{rep['assigned_area_km2']:.2f} km2 assigned, "
              f"{rep['unassigned_area_km2']:.2f} km2 unassigned")

    print("\n사업체 / 종사자 (서울시 사업체조사 microdata):")
    est_years = sorted(int(p.stem.split("_")[1]) for p in RAW.glob("estab_*.zip"))
    code_table = build_code_table(est_years)
    print(f"    code table: {len(code_table)} Seoul dong codes unioned over "
          f"{len(est_years)} survey years and "
          f"{len(list(RAW.glob('hjd_*.geojson')))} boundary vintages")
    est, est_rep, all_div = {}, {}, set()
    for y in est_years:
        df, rep, divs = read_establishments(y, code_table, primary, extra)
        est[y] = df
        est_rep[str(y)] = rep
        all_div |= set(divs)
        print(f"    -> {rep['dong_code_covered']} dong_code, "
              f"{rep['establishments']:,} establishments, "
              f"{rep['workers']:,} workers")

    # Assemble: one row per (product dong, year of the window). Employment is
    # left null where the survey does not reach, never carried forward — a
    # covariate silently held constant would look like a real regressor.
    base = cw[["dong_code", "gu_code", "gu", "dong"]].copy()
    # One column set for every year, so a KSIC division that only appears once
    # (the 11th revision arrives with the 2024 survey) does not silently make a
    # column object-typed in the years before it existed. Counts are nullable
    # integers: null means "the survey does not reach this year", 0 means "none".
    count_cols = (["estab", "worker", "worker_m", "worker_f"]
                  + [f"{k}_{d}" for d in sorted(all_div)
                     for k in ("estab", "worker")])
    frames = []
    for y in YEARS:
        f = base.copy()
        f["year"] = y
        f["area_vintage"] = AREA_VINTAGE[y]
        a = areas[y]
        f["area_km2"] = f.dong_code.map(a.area_km2)
        f["area_n_polygons"] = f.dong_code.map(a.n_polygons).astype("Int64")
        e = est.get(y)
        f["estab_year"] = y if e is not None else pd.NA
        for c in count_cols:
            f[c] = (f.dong_code.map(e[c]).astype("Int64")
                    if e is not None and c in e.columns else pd.NA)
        f["estab_year"] = f["estab_year"].astype("Int64")
        for c in count_cols:
            f[c] = f[c].astype("Int64")
        frames.append(f)
    cov = pd.concat(frames, ignore_index=True)

    for k in ("estab", "worker"):
        cols = [f"{k}_{d}" for d in RETAIL_FOOD if f"{k}_{d}" in cov]
        cov[f"{k}_retail_food"] = cov[cols].sum(axis=1, min_count=len(cols))

    # A second, *fixed* area column, and it is the one Phase 1d should use. The
    # per-vintage series below turns out to move for cartographic reasons as well
    # as administrative ones — SGIS redraws the 한강 allocation between the 2024
    # and 2025 vintages and 42 riverside dong change area by up to 55% with
    # Seoul's total unchanged. Density built on the moving column would jump at a
    # year boundary for no reason on the ground, and the regression would read
    # that as behaviour. The moving column is kept because knowing which dong the
    # geometry is fragile for is worth having.
    ref = areas[REF_YEAR].area_km2
    cov["area_km2_ref"] = cov.dong_code.map(ref)
    cov["area_ref_vintage"] = AREA_VINTAGE[REF_YEAR]
    lead = ["dong_code", "gu_code", "gu", "dong", "year",
            "area_km2_ref", "area_ref_vintage", "area_km2",
            "area_vintage", "area_n_polygons", "estab_year", "estab", "worker",
            "worker_m", "worker_f", "estab_retail_food", "worker_retail_food"]
    cov = cov[lead + [c for c in cov.columns if c not in lead]]

    holes = cov[cov.area_km2.isna()]
    if len(holes):
        print(f"\n{len(holes)} (dong, year) cells with no area:")
        for r in holes.itertuples():
            print(f"  {r.year} {r.gu} {r.dong} ({r.dong_code})")

    DERIVED.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(DERIVED / "dong_covariates.parquet", index=False)
    print(f"\nwrote {DERIVED / 'dong_covariates.parquet'} "
          f"({len(cov):,} rows, {cov.dong_code.nunique()} dong, "
          f"{cov.year.nunique()} years, {len(cov.columns)} columns)")

    # Area is meant to be a near-constant covariate; anything that moves by more
    # than rounding across vintages is either a real boundary shift or a join
    # that has gone wrong, and either way it has to be visible.
    wide = cov.pivot_table(index="dong_code", columns="year", values="area_km2")
    drift = ((wide.max(axis=1) - wide.min(axis=1)) / wide.min(axis=1)).sort_values()
    moved = drift[drift > 0.01]
    print(f"\narea drift across vintages: median {drift.median():.2%}, "
          f"{len(moved)} dong move more than 1%")
    steps = []
    for a, b in zip(YEARS[:-1], YEARS[1:]):
        rel = (wide[b] - wide[a]).abs() / wide[a]
        steps.append({"from": a, "to": b, "dong_over_1pct": int((rel > 0.01).sum()),
                      "max_rel": round(float(rel.max()), 4),
                      "sum_abs_km2": round(float((wide[b] - wide[a]).abs().sum()), 3),
                      "net_km2": round(float((wide[b] - wide[a]).sum()), 3)})
        print(f"  {a}->{b}: {steps[-1]['dong_over_1pct']:>3} dong move >1%, "
              f"max {steps[-1]['max_rel']:.1%}, "
              f"|delta| {steps[-1]['sum_abs_km2']:.2f} km2, "
              f"net {steps[-1]['net_km2']:+.2f} km2")
    print("  a step with |delta| large and net ~0 is a redraw, not a boundary "
          "change — Seoul's total is conserved and area moves between neighbours")
    for code, d in moved.tail(8).items():
        nm = base.loc[base.dong_code == code, ["gu", "dong"]].iloc[0]
        print(f"    {nm.gu} {nm.dong} ({code})  {d:.1%}  "
              f"{wide.loc[code].min():.3f} -> {wide.loc[code].max():.3f} km2")

    y0 = YEARS[0]
    s = cov[cov.year == y0]
    w = s.worker.dropna()
    print(f"\n{y0}: area {s.area_km2.min():.3f}-{s.area_km2.max():.3f} km2 "
          f"(median {s.area_km2.median():.3f}); "
          f"workers {int(w.min()):,}-{int(w.max()):,} "
          f"(median {int(w.median()):,}) over {len(w)} dong")
    dens = (s.worker / s.area_km2).dropna()
    print(f"      worker density {dens.min():,.0f}-{dens.max():,.0f} /km2 "
          f"(median {dens.median():,.0f}); "
          f"that spread is what the sigma_d regression is being asked to absorb")

    out["inputs"] = {
        "boundary_vintages": AREA_VINTAGE,
        "establishment_years": sorted(est),
        "product_dong": int(len(cw)),
        "extra_registration_names": sorted(f"{g} {d}" for g, d in extra),
    }
    out["area"] = area_rep
    out["establishments"] = est_rep
    out["output"] = {
        "path": str(DERIVED / "dong_covariates.parquet"),
        "rows": int(len(cov)), "dong": int(cov.dong_code.nunique()),
        "years": [int(y) for y in sorted(cov.year.unique())],
        "columns": list(cov.columns),
        "cells_without_area": int(cov.area_km2.isna().sum()),
        "cells_without_employment": int(cov.estab.isna().sum()),
        "area_drift_median": float(drift.median()),
        "area_drift_over_1pct": int(len(moved)),
        "area_drift_steps": steps,
        "area_reference_vintage": AREA_VINTAGE[REF_YEAR],
        "area_column_to_use": (
            "area_km2_ref — frozen at the "
            f"{AREA_VINTAGE[REF_YEAR]} vintage. area_km2 follows the vintage of "
            "its year and moves for cartographic reasons (the 한강 allocation is "
            "redrawn between the 2024 and 2025 vintages, moving up to 55% of a "
            "riverside dong's area with Seoul's total unchanged), which a "
            "density covariate must not inherit."),
    }
    out["still_missing"] = {
        "상업 연면적 (commercial gross floor area) per 행정동": (
            "not obtained. 서울시 건축물 연면적 (DT_201004_K030006) is published "
            "시/자치구 only and 건축물대장 표제부 is keyed on 법정동, whose map to "
            "행정동 is many-to-many at parcel level. worker_G (도소매) and worker_I "
            "(숙박·음식점) are stand-ins for commercial intensity and are a "
            "different quantity — employment, not floor space."),
        "employment for 2025-2026": (
            "서울시 사업체조사 is published to the 2024 reference year as of "
            "2026-08. The 2025 and 2026 rows carry area only; employment is left "
            "null rather than carried forward."),
    }
    with open(f"{ROOT}/eda/results_p20c.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p20c.json")


if __name__ == "__main__":
    main()
