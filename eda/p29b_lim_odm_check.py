#!/usr/bin/env python
"""Check Lim et al.'s published odm_*.csv against the official 자치구 release.

phase29 asserted three things about [LIM]'s Zenodo deposit — that they carry the
자치구 product and not a dong-level aggregation, that the values are the monthly
`합` and not a daily mean, and that masked cells enter as zero. Those three are
the opening evidence of the data descriptor (release/data_descriptor_plan.md
§2.1), and they are the reason we can say a published reuse already tripped on
the trap this dataset sets.

They were established by hand in the 2026-08-18 session and never scripted, so
until now nothing in eda/ read odm_*.csv at all and results_p29.json carried no
number from that comparison. This script closes that: it is the file-by-file
check the 8-21 letter said had to happen before §2.1 could be written.

It also settles which release to cite. v1.0.0 (10.5281/zenodo.19020433) holds 56
files and no CSV whatsoever; the odm_*.csv that phase29 read live only in v1.0.1
(10.5281/zenodo.19097294, 112 files). phase29's own "111 files" matches v1.0.1,
so the content it described was right and only the DOI was wrong.

The weekday check is the sharp one. Their README documents `dow` as 0=Mon..6=Sun
and the data is 0=Sun. phase29 argued this from a correlation; here it is a
clean split — under 0=Sun every cell agrees to the cent, under 0=Mon 2.9% do.

    python eda/p29b_lim_odm_check.py [--months 202012]

Only months present in BOTH releases can be checked: their processed/odm/ starts
at 202005, so phase29's other month (202003) has no counterpart and the earlier
memo should not be read as having checked it.
"""
import argparse
import hashlib
import io
import json
import sys
import zipfile

import pandas as pd
from common import AGE_LABEL
from dl_gu import zip_path
from p29_gu_reconcile import official
from paths import DATA_ROOT, ROOT, require

LIM_ZIP = DATA_ROOT / "raw" / "papers" / "lim2026_zenodo_v1.0.1.zip"
LIM_SHA256 = "52347be55429e11d72623e7502f8aba933ac978c4a8ac78ff0f122c7a731110e"
LIM_DOI = "10.5281/zenodo.19097294"

# Their README says 0=Mon..6=Sun; the data says 0=Sun. official() maps the
# Korean weekday names to 월=1..일=7, so both readings become that scale here.
DOW_DATA = {0: 7, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
DOW_README = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}

KEY = ["dow_n", "o", "d", "age", "mtype"]
EXACT = 0.005          # their csv carries two decimals


def lim_release():
    """The deposit, verified against the hash recorded in raw/papers/manifest."""
    require(LIM_ZIP, f"Lim et al. {LIM_DOI} (see raw/papers/manifest.json)")
    h = hashlib.sha256()
    with open(LIM_ZIP, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != LIM_SHA256:
        sys.exit(f"{LIM_ZIP} does not match the recorded sha256 — refusing to "
                 f"compare against an unknown version of their release")
    return zipfile.ZipFile(LIM_ZIP)


def lim_odm(zf, ym):
    """Their processed/odm/odm_YYYYMM.csv, or None when the month is absent."""
    hit = [n for n in zf.namelist() if n.endswith(f"processed/odm/odm_{ym}.csv")]
    if not hit:
        return None
    return pd.read_csv(io.BytesIO(zf.open(hit[0]).read()))


def compare(off, lim, mapping):
    """Cell-by-cell join of their monthly totals onto the official file."""
    l = lim.copy()
    l["dow_n"] = l.dow.map(mapping)
    l = (l.groupby(["dow_n", "o_sgg", "d_sgg", "age", "mtype"])
          .agg(lim_v=("mpop", "sum")).reset_index()
          .rename(columns={"o_sgg": "o", "d_sgg": "d"}))
    o = (off.groupby(KEY).agg(off_v=("v", "sum"),
                              off_masked=("masked", "sum")).reset_index())
    m = o.merge(l, on=KEY, how="inner")
    m["diff"] = (m.lim_v - m.off_v).abs()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="202012")
    args = ap.parse_args()
    zf = lim_release()
    out = {"doi": LIM_DOI, "sha256": LIM_SHA256, "months": {}}

    for ym in [int(x) for x in args.months.split(",") if x]:
        lim = lim_odm(zf, ym)
        if lim is None:
            print(f"  {ym}: not in their release (processed/odm/ starts 202005)")
            continue
        if not zip_path(ym).exists():
            print(f"  {ym}: no official gu zip, skipping "
                  f"(run eda/dl_gu.py --months {ym})")
            continue
        off = official(ym)

        # ------------------------------------------- 1 which weekday convention
        data = compare(off, lim, DOW_DATA)
        readme = compare(off, lim, DOW_README)
        print(f"\n=== 1 weekday convention ({ym}) ===")
        for name, m in [("data 0=Sun", data), ("README 0=Mon", readme)]:
            ok = int(m["diff"].le(EXACT).sum())
            print(f"  {name:>14}: {ok:>9,} of {len(m):>9,} cells exact "
                  f"({ok / len(m):>8.4%})  max|diff| {m['diff'].max():>12,.2f}")
        print("  Their README documents 0=Mon..6=Sun and their data is 0=Sun. "
              "Cite the file, not the README.")
        assert data["diff"].le(EXACT).all(), "0=Sun no longer reproduces exactly"

        # ---------------------------- 2 what product, and at what aggregation
        by = (data.groupby("age")
              .agg(lim_v=("lim_v", "sum"), off_v=("off_v", "sum")).reset_index())
        by["ratio"] = by.lim_v / by.off_v
        print(f"\n=== 2 their mpop over the official monthly sum ({ym}) ===")
        for r in by.itertuples():
            print(f"  {AGE_LABEL[r.age]:>6}  {r.lim_v:>16,.2f} / "
                  f"{r.off_v:>16,.2f}  {r.ratio:.6f}")
        exact_bands = int((by.ratio.round(6) == 1.0).sum())
        print(f"  {exact_bands} of {len(by)} bands at 1.000000 — they carry the "
              f"자치구 product and the monthly 합, not a dong aggregation and "
              f"not a daily mean.")

        # --------------------------------------- 3 how they took masked cells
        masked = data[data.off_masked > 0]
        ok_masked = int(masked["diff"].le(EXACT).sum())
        print(f"\n=== 3 masked cells ({ym}) ===")
        print(f"  {len(masked):,} cells have at least one masked official "
              f"sub-row; {ok_masked:,} still agree with masked read as zero "
              f"({ok_masked / len(masked):.4%}).")

        out["months"][str(ym)] = dict(
            cells=int(len(data)),
            exact_cells_data_dow=int(data["diff"].le(EXACT).sum()),
            exact_cells_readme_dow=int(readme["diff"].le(EXACT).sum()),
            readme_dow_max_abs_diff=float(readme["diff"].max()),
            bands_at_unity=exact_bands,
            total_ratio=float(by.lim_v.sum() / by.off_v.sum()),
            cells_with_masked_subrow=int(len(masked)),
            cells_with_masked_subrow_exact=int(ok_masked),
            by_age={AGE_LABEL[r.age]: float(r.ratio) for r in by.itertuples()})

    with open(f"{ROOT}/eda/results_p29b.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("\nwrote results_p29b.json")


if __name__ == "__main__":
    main()
