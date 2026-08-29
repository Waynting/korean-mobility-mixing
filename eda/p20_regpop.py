#!/usr/bin/env python
"""Phase 1b — registered population as the rho-hat denominator, and the crosswalk.

Two products, in dependency order:

  derived/dong_crosswalk.parquet   생활이동's own 7-digit dong code  <->  the MOIS
                                   10-digit 행정기관코드 the registration file uses.
                                   The two code spaces are unrelated by construction,
                                   so the join is on (gu name, dong name) with the
                                   orthographic difference between the two sources
                                   normalised away.

  derived/regpop.parquet           (ym, dong, sex, age band) -> registered population,
                                   single years folded into common.AGE_LABEL's 16 bins.
                                   That direction is exact and lossless, which is the
                                   one place in this project where the coarse-bin
                                   problem does not bite.

and a boundary-change log, because 424 dong is a fact about 2020, not about the
window. Any dong that enters or leaves is reported rather than absorbed.

Nothing here is allowed to drop a row silently. A registration dong with no
product counterpart is carried through labelled, not filtered out: the whole
point of the denominator is that it accounts for everybody.

    python eda/dl_regpop.py       # first — fetch and verify the CSVs
    python eda/p20_regpop.py      # then — crosswalk, ingest, check
"""
import csv
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpyxl
import pandas as pd
from common import AGE_LABEL, GU_NAMES  # noqa: E402
from paths import DATA_ROOT, DERIVED, ROOT, require  # noqa: E402

REGDIR = DATA_ROOT / "raw" / "regpop"
CODEBOOK = DATA_ROOT / "docs" / "서울생활이동데이터_행정동코드_20210907.xlsx"
CODE_RE = re.compile(r"\((\d{10})\)")
N_AGE_COLS = 101
SEX_KO = {"남": "M", "여": "F"}

# One dong is in the registration for all 79 months and has never been in
# 생활이동's code space, so no crosswalk row and no stable-unit edge can reach it:
# it predates the window rather than appearing inside it. Which product dong (if
# any) carries its residents is therefore not derivable from the code tables, and
# p21_rho.py section 1.6 decides it from the data instead — 오류2동 is the only
# 구로구 dong whose rho-hat outlier (z +2.19) is corrected rather than created by
# taking 항동 into its denominator, and it is the only assignment that lowers the
# gu's dispersion (RMS 0.1192 -> 0.0977).
#
# This is an inference, not a lookup, and it is mildly circular: rho-hat picked
# the assignment and then uses it. It is recorded here rather than buried because
# it is one 0.17%-of-Seoul decision made on a stated criterion, and anyone who
# disagrees should be able to find and change it in one place.
ABSORBED = {("구로구", "항동"): ("구로구", "오류2동")}

out = {}


def age_bin(a):
    """Single year -> the 16-bin key used everywhere else in this project."""
    if a < 10:
        return 0
    if a < 15:
        return 10
    if a < 20:
        return 15
    return 80 if a >= 80 else (a // 5) * 5


AGE_BIN = [age_bin(a) for a in range(N_AGE_COLS)]   # index == single year


def _dots(s):
    """The registration file writes the separator as '.', the product as '·'."""
    return s.replace(".", "·")


def _drop_ordinal(s):
    """'홍제제1동' -> '홍제1동'. Only the registration side spells the ordinal
    with 제, and only as a suffix — 홍제1동's own 제 is part of the name, so
    anchoring on the end is what keeps that dong from being mangled."""
    return re.sub(r"제(\d[\d·]*동)$", r"\1", s)


def classify(n_gone, n_new):
    """What a month's appearances and disappearances amount to.

    A rename costs nothing — the area is unchanged and only the label moved. A
    split or a merge does cost something: no single dong is comparable across
    the event, so both sides have to be pooled into one longitudinal unit.
    """
    if n_gone == 1 and n_new == 1:
        return "rename"          # same area, new label and code
    if n_gone == 1 and n_new > 1:
        return "split"
    if n_gone > 1 and n_new == 1:
        return "merge"
    return "reorganisation"


def stable_units(boundary, all_keys):
    """Coarsest partition of dong that is constant over the whole window.

    Every split/merge/rename event welds its before-side and after-side into one
    unit; everything untouched stays a unit of one. Longitudinal quantities are
    only comparable on these, which is why they exist — 424 dong is a fact about
    2020, not about 2020-2026.

    Union-find over dong names (gu + name), since the MOIS code changes across
    exactly the events we are trying to bridge.
    """
    parent = {k: k for k in all_keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for ev in boundary:
        names = [n for _, n in ev["dropped"]] + [n for _, n in ev["added"]]
        names = [n for n in names if n in parent]
        for n in names[1:]:
            union(names[0], n)
    groups = {}
    for k in all_keys:
        groups.setdefault(find(k), []).append(k)
    return groups


def load_crosswalk():
    """생활이동 7-digit dong code -> (gu, dong name), Seoul only."""
    require(CODEBOOK, "행정동 코드 정보 (data.seoul.go.kr)")
    wb = openpyxl.load_workbook(CODEBOOK, read_only=True)
    rows = list(wb[wb.sheetnames[0]].iter_rows(values_only=True))[1:]
    wb.close()
    recs = []
    for r in rows:
        if r[2] is None:
            continue
        code = str(r[2])
        if len(code) != 7 or not code.startswith("11"):
            continue      # 23xx 인천 / 31xx 경기 and the 5-digit rest-of-country
        gu_code = int(code[:4])
        recs.append({"dong_code": int(code), "gu_code": gu_code,
                     "gu": r[4].split()[1], "dong": _dots(r[3]),
                     "full_name": r[4]})
    cw = pd.DataFrame(recs)
    bad = set(cw["gu_code"]) - set(GU_NAMES)
    if bad:
        raise SystemExit(f"gu codes in the official table that common.GU_NAMES "
                         f"does not know: {sorted(bad)}")
    mismatch = [(c, g, GU_NAMES[c]) for c, g in
                zip(cw["gu_code"], cw["gu"]) if GU_NAMES[c] != g]
    if mismatch:
        raise SystemExit(f"gu name mismatch against common.GU_NAMES: {mismatch[:5]}")
    return cw


def read_month(path, ym):
    """One registration CSV -> long rows, Seoul dong only.

    The file is one row per area and 3 x (2 + 101) columns; the 계 block is
    redundant with 남 + 여 (dl_regpop verifies that) so only 남/여 are read.
    """
    text = path.read_bytes().decode("cp949")
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], [r for r in rows[1:] if r and r[0].strip()]
    off = {}
    for i, ko in enumerate(("계", "남", "여")):
        base = 1 + i * (2 + N_AGE_COLS)
        y, m = divmod(ym, 100)
        want = f"{y}년{m:02d}월_{ko}_총인구수"
        if header[base].strip() != want:
            raise SystemExit(f"{path.name}: column {base} is "
                             f"{header[base]!r}, expected {want!r}")
        off[ko] = base + 2                      # first single-year column

    def num(c):
        c = c.strip().replace(",", "")
        return int(c) if c and c != "-" else 0

    recs, gu_totals, city_total = [], {}, None
    for r in body:
        m = CODE_RE.search(r[0])
        if not m:
            raise SystemExit(f"{path.name}: no code in label {r[0]!r}")
        code, label = m.group(1), r[0][:m.start()].strip()
        if not code.startswith("11"):
            continue                            # not Seoul
        parts = label.split()
        if code.endswith("00000000"):
            city_total = num(r[1])
            continue
        if code.endswith("00000"):
            gu_totals[parts[1]] = num(r[1])
            continue
        gu, dong = parts[1], _drop_ordinal(_dots(parts[2]))
        for ko, sex in SEX_KO.items():
            acc = {}
            base = off[ko]
            for a in range(N_AGE_COLS):
                v = num(r[base + a])
                if v:
                    acc[AGE_BIN[a]] = acc.get(AGE_BIN[a], 0) + v
            for age, pop in acc.items():
                recs.append((ym, code, gu, dong, sex, age, pop))
    return recs, gu_totals, city_total


def main():
    require(REGDIR, "registration CSVs (run eda/dl_regpop.py first)")
    files = sorted(REGDIR.glob("regpop_*.csv"))
    if not files:
        raise SystemExit(f"no regpop_*.csv under {REGDIR} — run eda/dl_regpop.py")
    cw = load_crosswalk()
    print(f"crosswalk: {len(cw)} Seoul dong in the official 생활이동 code table")

    frames, checks, boundary = [], [], []
    seen_prev, name_of = None, {}
    for p in files:
        ym = int(p.stem.split("_")[1])
        recs, gu_totals, city_total = read_month(p, ym)
        df = pd.DataFrame(recs, columns=["ym", "mois_code", "gu", "dong",
                                         "sex", "age", "pop"])
        frames.append(df)

        codes = set(df["mois_code"])
        # name_of accumulates and is never pruned, so a dong that leaves keeps
        # the name it had while it existed — this month's file has no row for it.
        name_of.update(zip(df["mois_code"], df["gu"] + " " + df["dong"]))
        if seen_prev is not None:
            gone, new = seen_prev - codes, codes - seen_prev
            if gone or new:
                boundary.append({
                    "ym": ym,
                    "added": sorted((c, name_of[c]) for c in new),
                    "dropped": sorted((c, name_of[c]) for c in gone),
                    "kind": classify(len(gone), len(new)),
                })
        seen_prev = codes

        # Mass conservation: the dong must add up to the gu row, and the gu rows
        # to the city row, or a level of the hierarchy is being double counted.
        by_gu = df.groupby("gu")["pop"].sum()
        gu_gap = {g: int(by_gu.get(g, 0) - t) for g, t in gu_totals.items()
                  if int(by_gu.get(g, 0) - t) != 0}
        checks.append({"ym": ym, "n_dong": len(codes),
                       "total": int(df["pop"].sum()),
                       "city_row": city_total,
                       "city_gap": int(df["pop"].sum() - city_total),
                       "gu_gaps": gu_gap})
        print(f"  {ym}  {len(codes):>3} dong  {int(df['pop'].sum()):>9,}"
              f"  gap vs city row {int(df['pop'].sum() - city_total):+,}"
              f"{'  GU GAPS: ' + str(gu_gap) if gu_gap else ''}")

    reg = pd.concat(frames, ignore_index=True)

    # Attach the product code by name. Left join, and count what fails: a
    # registration dong with no product counterpart is a real gap in the
    # denominator's coverage, not a bug to be filtered away.
    reg = reg.merge(cw[["dong_code", "gu_code", "gu", "dong"]],
                    on=["gu", "dong"], how="left")
    unmatched = reg[reg["dong_code"].isna()]
    u = (unmatched.groupby(["gu", "dong"])
         .agg(months=("ym", "nunique"), pop=("pop", "sum")).reset_index())
    tot = reg["pop"].sum()
    print(f"\nunmatched on the name join: {len(u)} registration dong "
          f"(resolved below, via stable units and ABSORBED)")
    for _, r in u.iterrows():
        share = r["pop"] / tot
        print(f"  {r['gu']} {r['dong']:<12} {r['months']:>3} months  "
              f"{int(r['pop']):>12,} person-months  {share:.3%} of Seoul")
    matched_codes = set(reg.loc[reg["dong_code"].notna(), "dong_code"])
    missing = set(cw["dong_code"]) - matched_codes
    if missing:
        names = cw.set_index("dong_code").loc[sorted(missing), "full_name"]
        print(f"\nproduct dong never matched by the registration: {len(missing)}")
        for n in names:
            print(f"  {n}")

    # Stable units: the analysis unit for anything compared across months.
    reg["key"] = reg["gu"] + " " + reg["dong"]
    groups = stable_units(boundary, sorted(set(reg["key"])))
    unit_of = {k: rep for rep, ks in groups.items() for k in ks}
    reg["stable_unit"] = reg["key"].map(unit_of)
    multi = {rep: ks for rep, ks in groups.items() if len(ks) > 1}
    n_dong = reg["key"].nunique()
    pooled = sum(len(ks) for ks in multi.values())
    print(f"\nstable units: {len(groups)} for {n_dong} distinct dong names; "
          f"{len(multi)} unit(s) pool more than one dong, covering {pooled} "
          f"({pooled / n_dong:.1%} of names)")
    for rep, ks in sorted(multi.items()):
        print(f"  {rep}  <-  {', '.join(sorted(ks))}")

    # A dong created inside the window has no row in the 2021-09 code table, but
    # its stable unit contains the parent that does, so the product code follows
    # from the unit — no inference needed. That closes five of the six unmatched.
    filled = (reg.groupby("stable_unit")["dong_code"].transform("first"))
    gained = reg["dong_code"].isna() & filled.notna()
    inherited = sorted(reg.loc[gained, "key"].unique())
    if gained.any():
        print(f"\n  product code inherited through the stable unit for "
              f"{reg.loc[gained, 'key'].nunique()} dong created inside the window:")
        for k in sorted(reg.loc[gained, "key"].unique()):
            print(f"    {k}  ->  {int(filled[reg.key == k].iloc[0])}")
        reg["dong_code"] = reg["dong_code"].fillna(filled)
        reg["gu_code"] = reg["gu_code"].fillna(
            reg.groupby("stable_unit")["gu_code"].transform("first"))

    # The sixth needs the data to decide; see ABSORBED.
    for (gu_s, dong_s), (gu_t, dong_t) in ABSORBED.items():
        src = (reg.gu == gu_s) & (reg.dong == dong_s)
        tgt = cw[(cw.gu == gu_t) & (cw.dong == dong_t)]
        if not src.any() or not len(tgt):
            continue
        reg.loc[src, ["dong_code", "gu_code"]] = [tgt.dong_code.iloc[0],
                                                  tgt.gu_code.iloc[0]]
        reg.loc[src, "stable_unit"] = f"{gu_t} {dong_t}"
        print(f"  {gu_s} {dong_s} assigned to {gu_t} {dong_t} "
              f"(p21 section 1.6; see ABSORBED)")

    still = reg[reg.dong_code.isna()]
    print(f"  registration dong still without a product code: "
          f"{still['key'].nunique()}"
          f"{' — ' + ', '.join(sorted(still['key'].unique())) if len(still) else ''}")

    DERIVED.mkdir(parents=True, exist_ok=True)
    cw.to_parquet(DERIVED / "dong_crosswalk.parquet", index=False)
    reg.to_parquet(DERIVED / "regpop.parquet", index=False)
    print(f"\nwrote {DERIVED / 'dong_crosswalk.parquet'} ({len(cw):,} rows)")
    print(f"wrote {DERIVED / 'regpop.parquet'} ({len(reg):,} rows, "
          f"{reg['ym'].nunique()} months)")

    out["months"] = sorted(int(x) for x in reg["ym"].unique())
    out["conservation"] = checks
    out["boundary_changes"] = boundary
    out["crosswalk"] = {
        "product_dong": len(cw),
        "matched_on_name": len(matched_codes),
        "product_dong_never_matched": sorted(int(c) for c in missing),
        # The name join alone leaves six unmatched; five are resolved by
        # inheriting the product code through the stable unit and the sixth by
        # ABSORBED. Both states are recorded so the JSON is not read as if the
        # denominator still had holes in it.
        "unmatched_on_name": u.to_dict("records") if len(u) else [],
        "resolved_via_stable_unit": sorted(inherited),
        "resolved_via_absorbed": {f"{a} {b}": f"{c} {d}"
                                  for (a, b), (c, d) in ABSORBED.items()},
        "still_without_product_code": sorted(still["key"].unique().tolist()),
    }
    out["age_bins"] = {str(k): AGE_LABEL[k] for k in sorted(set(AGE_BIN))}
    out["stable_units"] = {
        "n_units": len(groups), "n_dong_names": n_dong,
        "n_pooled_units": len(multi), "n_names_pooled": pooled,
        "share_pooled": pooled / n_dong,
        "pooled": {rep: sorted(ks) for rep, ks in sorted(multi.items())},
    }
    with open(f"{ROOT}/eda/results_p20.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p20.json")

    if boundary:
        print(f"\n{len(boundary)} boundary event(s) over the window:")
        for b in boundary:
            out_ = ", ".join(n for _, n in b["dropped"])
            in_ = ", ".join(n for _, n in b["added"])
            print(f"  {b['ym']}  {b['kind']:<8} {out_}  ->  {in_}")
    else:
        print("\nno dong entered or left over the window")


if __name__ == "__main__":
    main()
