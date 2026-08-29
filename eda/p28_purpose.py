#!/usr/bin/env python
"""Phase 4b — the second classifier, and what it can and cannot settle.

p25 bounded the 65+ W panel from inside the data (TV distance against the
prime-age commute, 20.9-27.7%). The plan called 4b the stronger test: run the
parallel 수도권 product, whose classifier labels a trip's PURPOSE rather than its
endpoint, over the same city and month, and read the contamination off directly.

Two things had to be established before any of that could be measured, and both
turned out to be findings in their own right.

1. THE PUBLISHED PURPOSE ORDER IS NOT THE CODE ORDER. The portal lists
   "출근, 등교, 쇼핑, 관광, 병원, 귀가, 기타", which reads as codes 1..7. The data
   says otherwise, and three independent fingerprints agree: code 3 carries 38%
   of all volume, peaks at 20시 and correlates 0.95 with registered population by
   destination dong — that is 귀가, not 쇼핑. The actual order is

       1 출근   2 등교   3 귀가   4 쇼핑   5 관광   6 병원   7 기타

   identified by behaviour, not by the description: 출근 loads on 종사자수
   (r=0.85), 등교 is 56% ten-to-nineteen, and 쇼핑/관광/병원 appear in only 15-24
   Seoul dong each, whose names read as a roll-call — 병원 lands on 신촌동
   (세브란스), 풍납2동 (아산), 일원본동 (삼성), 흑석동 (중앙대), 안암동 (고대).
   The script asserts all of this rather than trusting the page.

2. 쇼핑, 관광 AND 병원 ARE POI-ATTACHED, NOT A PARTITION. Together they are ~0.4%
   of volume and exist in a couple of dozen dong. So the plan's falsification
   line — "if 병원+쇼핑 exceeds ~40% of elderly W arrivals, the panel cannot be
   called work" — CANNOT BE EVALUATED as written, and no amount of extra months
   fixes it. This is a limit of the instrument, and it is reported as one.

What 4b CAN do is the comparison that matters anyway: 출근 and 등교 and 귀가 are
complete categories covering every destination, so the share of arrivals that the
purpose classifier calls 출근+등교 can be set against the share the H/W/E
classifier calls W, per age band, same month, same pipeline. The gap is a
measured statement about the W label, obtained from an independent classifier.

SCOPE MATCHING is the part to get right, and one thing helps: the purpose product
has a DATE column. So both sides are restricted to the same holiday-free weekdays
of the month (2026-06 drops Wednesday for the 지방선거 임시공휴일), and the H/W/E
side comes from p25's arrival cache, which filters the DESTINATION to Seoul and
leaves the origin free — the same scope the purpose file has. Residual
mismatches, both stated rather than corrected: the purpose file is 내국인 only
while ours does not separate nationality (foreigners are 6.26% of 20-24 and 0.25%
of 80+), and its age bands are ten years wide against our five.

    python eda/p28_purpose.py [--months 202606,202512]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar_kr as K
from common import AGE_LABEL, AGES
from paths import DERIVED, FIG, ROOT, require

PURPOSE = {1: "출근", 2: "등교", 3: "귀가", 4: "쇼핑", 5: "관광", 6: "병원",
           7: "기타"}
PUBLISHED_ORDER = ["출근", "등교", "쇼핑", "관광", "병원", "귀가", "기타"]
# Hospitals whose dong must appear at the top of the purpose the fingerprint
# calls 병원. Named in advance so the test can fail.
HOSPITAL_DONG = {"신촌동", "풍납2동", "흑석동", "일원본동", "안암동"}
AGE10 = [0, 10, 20, 30, 40, 50, 60, 70]
LBL10 = {0: "0-9", 10: "10-19", 20: "20-29", 30: "30-39", 40: "40-49",
         50: "50-59", 60: "60-69", 70: "70+"}
out = {}


def band10(age5):
    """Our sixteen 5-year bands onto their eight 10-year ones. Exact, upward."""
    return 70 if age5 >= 70 else (age5 // 10) * 10


def dong_keys(ym):
    """MOIS 8-digit code -> product 7-digit code, plus names and covariates."""
    reg = pd.read_parquet(DERIVED / "regpop.parquet")
    r = reg[reg.ym == ym]
    if r.empty:
        r = reg[reg.ym == reg.ym.max()]
    k = (r.drop_duplicates("mois_code")[["mois_code", "dong_code", "gu", "dong"]]
         .copy())
    k["mois8"] = k.mois_code.astype(str).str[:8]
    pop = (r.assign(mois8=r.mois_code.astype(str).str[:8])
           .groupby("mois8")["pop"].sum())
    cov = pd.read_parquet(DERIVED / "dong_covariates.parquet")
    # The covariate parquet deliberately leaves 2025-2026 null rather than
    # carrying 2024 forward (phase1d-covariates section 5), so the latest year
    # with data is picked explicitly instead of by max().
    have = cov[cov.worker.notna()]
    cov = have[have.year == have.year.max()][["dong_code", "worker"]]
    cov = cov.merge(k[["mois8", "dong_code"]], on="dong_code", how="inner")
    return k.set_index("mois8"), pop, cov.set_index("mois8").worker


def clean_dows(ym):
    """Weekdays with no 공휴일. p25's cache is built on exactly this set."""
    return [d for d in range(1, 6) if K.cell_exposure(ym, d)["n_holiday"] == 0]


def load_purpose(ym, keys, dows):
    p = DERIVED / f"purpose_{ym}.parquet"
    require(p, f"purpose month {ym} (run eda/dl_purpose.py --months {ym} --etl)")
    df = pd.read_parquet(p)
    df["dow"] = pd.to_datetime(df.date, format="%Y%m%d").dt.weekday + 1
    df["seoul_dong"] = df.d_admdong_cd.isin(keys.index)
    out.setdefault("scope", {})[str(ym)] = dict(
        rows=int(len(df)),
        seoul_dong_volume_share=float(
            df.loc[df.seoul_dong, "cnt"].sum() / df.cnt.sum()),
        distinct_dest_codes=int(df.d_admdong_cd.nunique()),
        seoul_dong_codes=int(df.loc[df.seoul_dong, "d_admdong_cd"].nunique()),
        min_positive_cnt=float(df.cnt.min()),
        peak_20min_share=float(df.loc[df.min20 > 0, "cnt"].sum() / df.cnt.sum()))
    return df[df.seoul_dong & df.dow.isin(dows)].copy()


def identify(ym, df_all, keys, pop, job):
    """Reproduce the purpose codes from behaviour, and assert what we claim."""
    d = df_all[df_all.seoul_dong]
    rows = []
    for p, g in d.groupby("move_purpose"):
        v = g.groupby("d_admdong_cd")["cnt"].sum()
        idx = v.index.intersection(pop.index).intersection(job.index)
        lv, lp = np.log(v[idx]), np.log(pop[idx])
        lj = np.log(job[idx].replace(0, np.nan))
        ok = lj.notna()
        h = g.groupby("hour")["cnt"].sum()
        age = g.groupby("age10")["cnt"].sum()
        rows.append(dict(
            code=int(p), label=PURPOSE[int(p)], volume=float(v.sum()),
            share=float(v.sum() / d.cnt.sum()), n_dong=int(len(v)),
            corr_pop=float(np.corrcoef(lv[ok], lp[ok])[0, 1]),
            corr_jobs=float(np.corrcoef(lv[ok], lj[ok])[0, 1]),
            peak_hour=int(h.idxmax()),
            share_07_09=float(h.reindex(range(7, 10)).sum() / h.sum()),
            share_10_19=float(age.get(10, 0) / age.sum()),
            top_dong=[keys.dong.get(c, c) for c in v.nlargest(5).index]))
    R = pd.DataFrame(rows).set_index("code").sort_index()
    print(f"\n=== 1 what the seven codes actually are ({ym}) ===")
    print(f"  {'code':>4} {'label':>6} {'share':>7} {'dong':>5} {'r(pop)':>7} "
          f"{'r(jobs)':>8} {'peak':>5} {'07-09':>6} {'10-19':>6}")
    for c, r in R.iterrows():
        print(f"  {c:>4} {r.label:>6} {r.share:7.2%} {r.n_dong:>5} "
              f"{r.corr_pop:+7.3f} {r.corr_jobs:+8.3f} {r.peak_hour:>5} "
              f"{r.share_07_09:6.1%} {r.share_10_19:6.1%}")
    for c in (4, 5, 6):
        print(f"    {PURPOSE[c]} top dong: {', '.join(R.loc[c, 'top_dong'])}")

    # The claims, as tests. Each is what distinguishes the code from the
    # published order, so a failure here means the mapping has to be redone.
    assert R.loc[3, "corr_pop"] > 0.9 > R.loc[1, "corr_pop"], \
        "code 3 does not behave like 귀가 (residential destination)"
    assert R.loc[1, "corr_jobs"] > 0.8 > R.loc[3, "corr_jobs"], \
        "code 1 does not behave like 출근 (job destination)"
    # A rank test, not a level: 등교's ten-to-nineteen concentration falls from
    # 51.5% in June to 49.2% in December (winter vacation), so an absolute
    # threshold would be calibrated to whichever month was looked at first.
    assert R.loc[2, "share_10_19"] > 3 * R.drop(2).share_10_19.max(), \
        "code 2 is not the school-age purpose"
    assert R.loc[1, "peak_hour"] in (7, 8, 9), "출근 does not peak in the morning"
    assert HOSPITAL_DONG & set(R.loc[6, "top_dong"]), \
        f"code 6 top dong {R.loc[6, 'top_dong']} are not hospital dong"
    assert PUBLISHED_ORDER != [PURPOSE[c] for c in sorted(PURPOSE)], \
        "published order now matches the codes — rewrite this section"
    poi = R.loc[[4, 5, 6], "share"].sum()
    print(f"\n  쇼핑+관광+병원 = {poi:.2%} of Seoul-destination volume, present in "
          f"{R.loc[[4,5,6],'n_dong'].max()} dong at most.")
    print("  They are POI-attached labels, not a partition of trips, so the "
          "plan's '병원+쇼핑 above 40%' line cannot be evaluated as written.")
    out.setdefault("codes", {})[str(ym)] = R.reset_index().to_dict("records")
    return R


def compare(ym, pur, dows):
    """W share (H/W/E classifier) against 출근+등교 share (purpose classifier)."""
    cache = DERIVED / "p25_w_arrivals.parquet"
    require(cache, "p25 arrival cache (run eda/p25_wcheck.py)")
    arr = pd.read_parquet(cache)
    arr = arr[(arr.ym == ym) & arr.dow_n.isin(dows)].copy()
    if arr.empty:
        print(f"  {ym}: not in the p25 cache — skipped")
        return None
    arr["b10"] = arr.age.map(band10)
    rows = []
    for imp, tag in ((0.0, "lo"), (1.5, "mid"), (3.0, "hi")):
        a = arr.assign(n=arr.v_obs + imp * arr.n_masked)
        tot = a.groupby("b10")["n"].sum()
        w = a[a.d_attr == "W"].groupby("b10")["n"].sum()
        e = a[a.d_attr == "E"].groupby("b10")["n"].sum()
        for b in AGE10:
            rows.append(dict(ym=ym, band=b, imp=tag,
                             w_share=float(w.get(b, 0) / tot.get(b, np.nan)),
                             e_share=float(e.get(b, 0) / tot.get(b, np.nan))))
    H = pd.DataFrame(rows)

    tot_p = pur.groupby("age10")["cnt"].sum()
    comm = (pur[pur.move_purpose.isin([1, 2])].groupby("age10")["cnt"].sum()
            / tot_p)
    home = pur[pur.move_purpose == 3].groupby("age10")["cnt"].sum() / tot_p
    poi = (pur[pur.move_purpose.isin([4, 5, 6])].groupby("age10")["cnt"].sum()
           / tot_p)
    etc = pur[pur.move_purpose == 7].groupby("age10")["cnt"].sum() / tot_p

    mid = H[H.imp == "mid"].set_index("band")
    lo = H[H.imp == "lo"].set_index("band").w_share
    hi = H[H.imp == "hi"].set_index("band").w_share
    # The classifiers disagree by a fixed amount even in prime age, where nobody
    # claims contamination, so the level of the gap is not the quantity of
    # interest -- the EXCESS over the reference band is. Same logic as p25's TV
    # bound, which is measured against the 45-49 commute rather than against zero.
    ref = 40
    res = []
    for b in AGE10:
        gap = mid.w_share[b] - comm.get(b, np.nan)
        res.append(dict(ym=ym, band=b, label=LBL10[b],
                        w_share=float(mid.w_share[b]),
                        w_share_lo=float(lo[b]), w_share_hi=float(hi[b]),
                        e_share=float(mid.e_share[b]),
                        commute_share=float(comm.get(b, np.nan)),
                        home_share=float(home.get(b, np.nan)),
                        poi_share=float(poi.get(b, np.nan)),
                        other_share=float(etc.get(b, np.nan)),
                        gap_pp=float(gap * 100),
                        non_commute_frac=float(gap / mid.w_share[b])))
    R = pd.DataFrame(res).set_index("band")
    R["excess_over_ref"] = R.non_commute_frac - R.non_commute_frac[ref]
    print(f"\n=== 2 the two classifiers on the same arrivals ({ym}, "
          f"weekdays {dows}) ===")
    print(f"  {'band':>6} | {'W share (H/W/E)':>21} | {'출근+등교':>8} "
          f"{'귀가':>6} {'POI':>6} {'기타':>6} | {'gap':>7} {'of W':>6} "
          f"{'vs ref':>7}")
    for b in AGE10:
        r = R.loc[b]
        print(f"  {LBL10[b]:>6} | {r.w_share:7.1%} [{lo[b]:.1%},{hi[b]:.1%}] "
              f"| {r.commute_share:8.1%} {r.home_share:6.1%} "
              f"{r.poi_share:6.2%} {r.other_share:6.1%} | {r.gap_pp:+6.1f}pp "
              f"{r.non_commute_frac:6.0%} {r.excess_over_ref:+7.0%}")
    print(f"  'of W' = the gap as a fraction of that band's W panel; 'vs ref' "
          f"= excess over {LBL10[ref]}, where no contamination is claimed.")
    return R.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="202606,202512")
    args = ap.parse_args()
    yms = [int(x) for x in args.months.split(",") if x]
    allres = []
    for ym in yms:
        p = DERIVED / f"purpose_{ym}.parquet"
        if not p.exists():
            print(f"  {ym}: no purpose parquet, skipping")
            continue
        keys, pop, job = dong_keys(ym)
        dows = clean_dows(ym)
        df = pd.read_parquet(p)
        df["dow"] = pd.to_datetime(df.date, format="%Y%m%d").dt.weekday + 1
        df["seoul_dong"] = df.d_admdong_cd.isin(keys.index)
        sc = dict(rows=int(len(df)),
                  seoul_dong_volume_share=float(
                      df.loc[df.seoul_dong, "cnt"].sum() / df.cnt.sum()),
                  distinct_dest_codes=int(df.d_admdong_cd.nunique()),
                  seoul_dong_codes=int(df.loc[df.seoul_dong,
                                              "d_admdong_cd"].nunique()),
                  min_positive_cnt=float(df.cnt.min()),
                  peak_20min_volume_share=float(
                      df.loc[df.min20 > 0, "cnt"].sum() / df.cnt.sum()))
        out.setdefault("scope", {})[str(ym)] = sc
        print(f"=== 0 scope ({ym}) ===")
        print(f"  {sc['rows']:,} non-zero cells, {sc['distinct_dest_codes']} "
              f"destination codes of which {sc['seoul_dong_codes']} are Seoul "
              f"dong ({sc['seoul_dong_volume_share']:.1%} of volume)")
        print(f"  smallest positive cell {sc['min_positive_cnt']:.2f} "
              f"(no '*' marker in this product); "
              f"{sc['peak_20min_volume_share']:.1%} of volume sits in the "
              f"20-minute peak bins")
        identify(ym, df, keys, pop, job)
        pur = df[df.seoul_dong & df.dow.isin(dows)]
        r = compare(ym, pur, dows)
        if r is not None:
            allres.append(r)
    if allres:
        R = pd.concat(allres, ignore_index=True)
        out["comparison"] = R.to_dict("records")
        old = R[R.band >= 60]
        print("\n=== 3 what this says about the W label ===")
        for _, x in old.iterrows():
            print(f"  {x.ym} {x.label}: H/W/E calls {x.w_share:.1%} of arrivals "
                  f"W, the purpose classifier calls {x.commute_share:.1%} "
                  f"출근+등교 — {x.non_commute_frac:.0%} of the W panel, "
                  f"{x.excess_over_ref:+.0f} points against the 40-49 reference."
                  .replace(f"{x.excess_over_ref:+.0f}",
                           f"{100*x.excess_over_ref:+.0f}"))
        print("  The gap is an upper bound on agreement, not a contamination "
              "rate: the two classifiers are not applied to identical trip sets "
              "(내국인 only on the purpose side), and 출근+등교 excludes the "
              "commute of anyone whose workplace KT could not anchor.")

        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        for ym, c in zip(sorted(R.ym.unique()), ["#A8434E", "#4C6E8A"]):
            g = R[R.ym == ym].set_index("band")
            x = range(len(AGE10))
            ax.plot(x, [g.w_share[b] for b in AGE10], "-o", ms=4, color=c,
                    label=f"W share, H/W/E classifier ({ym})")
            # Romanised in the figure only: matplotlib's bundled font has no
            # Hangul, so Korean labels render as tofu boxes in the PNG.
            ax.plot(x, [g.commute_share[b] for b in AGE10], "--s", ms=4,
                    color=c, alpha=.65,
                    label=f"chulgeun+deunggyo share, purpose ({ym})")
        ax.set_xticks(range(len(AGE10)))
        ax.set_xticklabels([LBL10[b] for b in AGE10])
        ax.set_ylabel("share of arrivals")
        ax.set_title("Phase 4b — two classifiers on the same arrivals\n"
                     "the gap is near-constant; the elderly excess sits at 60-69",
                     fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{FIG}/p28_purpose.png", dpi=150)
        plt.close(fig)
        print("\n  -> fig/p28_purpose.png")
    with open(f"{ROOT}/eda/results_p28.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p28.json")


if __name__ == "__main__":
    main()
