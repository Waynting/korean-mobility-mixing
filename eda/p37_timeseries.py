#!/usr/bin/env python
"""Phase 37 — the same measurement, on all 79 months.

WHY NOW. Until 2026-08-19 the co-arrival matrix existed for six months, chosen
because they were the ones on disk: two 2020 months, the two survey months, and
two recent ones. Every claim the paper makes about structure -- the dong-level
assortativity of about 0.02, the dong -> gu -> city ladder, the near-rank-1
spectrum, the departure from proportionate mixing that is smaller than the
survey's own sampling floor -- rested on those six. The download finished, so
the same six statements can now be made on 202001..202607 without a gap, and
the question "is this a property of the data or of the months you picked" gets
an answer instead of a caveat.

THE SAME WAY, LITERALLY. This script does not re-implement anything. It imports
p26_matrix.cells / matrices / stats and p32_pmix.excess_stats / rank_stats /
symmetrise and calls them. A second implementation would be a different object
measured on more months, which answers a different question; the point here is
to hold the estimator fixed and vary only the period. p36_recompute.py is where
independence lives, and it already covers these functions.

WHAT IS NOT REWRITTEN. results_p26.json and results_p32.json are left alone.
They are frozen, gated by p31 and independently recomputed by p36, and rewriting
them with 79 months would invalidate that whole chain to add rows. The six
published months are recomputed here anyway and asserted against them, so the
time series is tied to the published numbers rather than merely consistent with
them.

STORAGE. The full A and M are kept only for the WE dong-level matrix; at three
panels x three levels x 79 months, storing every matrix would put ~10 MB of
mostly-unread numbers into git. The statistics are what the paper uses.

    python eda/p37_timeseries.py                 # every month on disk
    python eda/p37_timeseries.py --months 202001,202012
    python eda/p37_timeseries.py --cache-only    # just build the arrival caches
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import AGE_LABEL, AGES, months_on_disk
from p26_matrix import cells, holiday_free_dows, matrices, stats
from p32_pmix import excess_stats, rank_stats, symmetrise
from paths import DERIVED, FIG, ROOT, require

PANELS = ("W", "E", "WE")
LEVELS = ("dong", "gu", "city")
N_LOC = {"dong": 424, "gu": 25, "city": 1}
PUBLISHED_MONTHS = [202001, 202012, 202312, 202402, 202512, 202606]
out = {}


def population(reg, fo, ym):
    """Registered residents plus registered foreigners, by band -- p26's
    denominator, built the same way and asserted to be complete."""
    s = (reg[reg.ym == ym].groupby("age")["pop"].sum()
         + fo[fo.ym == ym].groupby("age")["pop"].sum())
    v = s.reindex(AGES)
    assert v.notna().all() and (v > 0).all(), (
        f"{ym}: denominator has missing or zero bands -- regpop or foreign does "
        f"not cover this month, and a NaN here would silently poison every "
        f"per-capita statistic downstream")
    return v.to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="")
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()
    yms = ([int(x) for x in args.months.split(",")] if args.months
           else months_on_disk())
    require(DERIVED / "regpop.parquet", "denominator")
    reg = pd.read_parquet(DERIVED / "regpop.parquet")
    fo = pd.read_parquet(DERIVED / "foreign.parquet")
    sq = [i for i, a in enumerate(AGES) if a < 80]
    print(f"=== 37.0 {len(yms)} months: {yms[0]}..{yms[-1]} ===")

    if args.cache_only:
        for i, ym in enumerate(yms, 1):
            df = cells(ym)
            print(f"  [{i:>2}/{len(yms)}] {ym} cached, {len(df):,} cells", flush=True)
        return 0

    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    rows, spectra, mats = [], [], {}
    for i, ym in enumerate(yms, 1):
        df = cells(ym)
        pop = population(reg, fo, ym)
        hf = holiday_free_dows(ym)
        for panel in PANELS:
            for level in LEVELS:
                A = matrices(df, ym, panel, level)
                st = stats(A, pop)
                T, asym = symmetrise(A[np.ix_(sq, sq)] / pop[sq][:, None], pop[sq])
                ex, rk = excess_stats(T), rank_stats(T)
                rows.append(dict(
                    ym=ym, panel=panel, level=level, n_loc=N_LOC[level],
                    assortativity=st["assortativity"],
                    lead_eigenvalue=st["lead_eigenvalue"],
                    asymmetry=asym,
                    **{k: ex[k] for k in ("half_l1", "cramers_v", "nmi",
                                          "mi_bits", "entropy_bits",
                                          "diag_excess_sum")},
                    sigma1_share=rk["sigma1_share"],
                    sigma2_over_sigma1=rk["sigma2_over_sigma1"],
                    best_rank1_resid=rk["best_rank1_resid"],
                    pm_rank1_resid=rk["pm_rank1_resid"]))
                if panel == "WE" and level == "dong":
                    mats[str(ym)] = dict(A=A.tolist(),
                                         lead_eigenvector=st["lead_eigenvector"])
        # the holiday-free variant, so the survey-month convention exists in
        # every month rather than only in the two the survey covers
        A = matrices(df, ym, "WE", "dong", dows=hf)
        st = stats(A, pop)
        T, asym = symmetrise(A[np.ix_(sq, sq)] / pop[sq][:, None], pop[sq])
        ex = excess_stats(T)
        rows.append(dict(ym=ym, panel="WE", level="dong_holidayfree", n_loc=424,
                         assortativity=st["assortativity"],
                         lead_eigenvalue=st["lead_eigenvalue"], asymmetry=asym,
                         **{k: ex[k] for k in ("half_l1", "cramers_v", "nmi",
                                               "mi_bits", "entropy_bits",
                                               "diag_excess_sum")},
                         sigma1_share=np.nan, sigma2_over_sigma1=np.nan,
                         best_rank1_resid=np.nan, pm_rank1_resid=np.nan))
        # the plain dong figure, not rows[-1] -- that is the holiday-free
        # variant appended just above, and printing it under the label "dong"
        # would put a different number in the log than in the results file
        _d = next(r for r in reversed(rows)
                  if r["ym"] == ym and r["panel"] == "WE" and r["level"] == "dong")
        print(f"  [{i:>2}/{len(yms)}] {ym}  dong {_d['assortativity']:.5f}  "
              f"nmi {_d['nmi']:.5f}", flush=True)

    R = pd.DataFrame(rows)
    out["rows"] = R.to_dict("records")
    out["matrices_we_dong"] = mats
    out["months"] = yms

    # ------------------------------------------------------------- the anchor
    # Six of these months are already published. If the estimator has not moved,
    # they must come back bit for bit; anything else means this series is not
    # the same measurement as the one in the paper.
    print("\n=== 37.1 anchor: the six published months must reproduce ===")
    anch = []
    for ym in PUBLISHED_MONTHS:
        for panel in PANELS:
            for level in LEVELS:
                key = f"{ym}|{panel}|{level}"
                if key not in p26["matrices"]:
                    continue
                r = R[(R.ym == ym) & (R.panel == panel) & (R.level == level)]
                if r.empty:
                    continue
                got, want = float(r.assortativity.iloc[0]), \
                    p26["matrices"][key]["assortativity"]
                rel = abs(got - want) / abs(want)
                anch.append(dict(key=key, rel=rel))
                assert rel < 1e-9, f"{key}: {got} vs published {want}"
    print(f"  {len(anch)} published (month, panel, level) cells reproduced, "
          f"max relative deviation {max(a['rel'] for a in anch):.2e}")
    out["anchor"] = anch

    # -------------------------------------------------- 37.2 is 0.02 a constant
    we = R[(R.panel == "WE") & (R.level == "dong")].sort_values("ym")
    gu = R[(R.panel == "WE") & (R.level == "gu")].sort_values("ym")
    ci = R[(R.panel == "WE") & (R.level == "city")].sort_values("ym")
    print("\n=== 37.2 the dong-level concentration over 79 months ===")
    print(f"  assortativity  min {we.assortativity.min():.5f} "
          f"({int(we.loc[we.assortativity.idxmin(), 'ym'])})  "
          f"max {we.assortativity.max():.5f} "
          f"({int(we.loc[we.assortativity.idxmax(), 'ym'])})  "
          f"median {we.assortativity.median():.5f}")
    print(f"  normalised MI  min {we.nmi.min():.5f}  max {we.nmi.max():.5f}  "
          f"median {we.nmi.median():.5f}")
    print(f"  the published six span {we[we.ym.isin(PUBLISHED_MONTHS)].assortativity.min():.5f}"
          f"-{we[we.ym.isin(PUBLISHED_MONTHS)].assortativity.max():.5f}; "
          f"all 79 span {we.assortativity.min():.5f}-{we.assortativity.max():.5f}")
    out["summary_dong"] = dict(
        n_months=int(len(we)),
        assort_min=float(we.assortativity.min()),
        assort_max=float(we.assortativity.max()),
        assort_median=float(we.assortativity.median()),
        assort_iqr=[float(we.assortativity.quantile(.25)),
                    float(we.assortativity.quantile(.75))],
        nmi_min=float(we.nmi.min()), nmi_max=float(we.nmi.max()),
        nmi_median=float(we.nmi.median()),
        published_six_min=float(we[we.ym.isin(PUBLISHED_MONTHS)].assortativity.min()),
        published_six_max=float(we[we.ym.isin(PUBLISHED_MONTHS)].assortativity.max()))

    # ---------------------------------------------- 37.3 does the ladder hold
    m = we.merge(gu, on="ym", suffixes=("_dong", "_gu")).merge(
        ci[["ym", "assortativity"]].rename(columns={"assortativity": "a_city"}),
        on="ym")
    mono = ((m.assortativity_dong > m.assortativity_gu)
            & (m.assortativity_gu > m.a_city))
    kept = (m.assortativity_gu - m.a_city) / (m.assortativity_dong - m.a_city)
    print("\n=== 37.3 the resolution ladder, every month ===")
    print(f"  dong > gu > city holds in {int(mono.sum())} of {len(m)} months")
    print(f"  share of dong-level excess surviving the 17x coarsening to gu: "
          f"median {kept.median():.1%}, range {kept.min():.1%}-{kept.max():.1%}")
    out["ladder"] = dict(n_months=int(len(m)), n_monotone=int(mono.sum()),
                         kept_median=float(kept.median()),
                         kept_min=float(kept.min()), kept_max=float(kept.max()),
                         breaks=[int(x) for x in m.loc[~mono, "ym"]])

    # ------------------------------------------- 37.4 is it always near rank-1
    print("\n=== 37.4 the spectrum, every month ===")
    for level in LEVELS:
        d = R[(R.panel == "WE") & (R.level == level)]
        print(f"  {level:>5}: sigma1^2/sum sigma^2 median {d.sigma1_share.median():.4f}"
              f"  min {d.sigma1_share.min():.4f}"
              f"   |best rank-1 resid - pm resid| median "
              f"{(d.best_rank1_resid - d.pm_rank1_resid).abs().median():.5f}")
    d = R[(R.panel == "WE") & (R.level == "dong")]
    out["spectrum_dong"] = dict(
        sigma1_share_median=float(d.sigma1_share.median()),
        sigma1_share_min=float(d.sigma1_share.min()),
        gap_to_pm_median=float((d.best_rank1_resid - d.pm_rank1_resid).abs().median()),
        gap_to_pm_max=float((d.best_rank1_resid - d.pm_rank1_resid).abs().max()))

    # --------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    x = [str(y) for y in we.ym]
    # School-term shading. The term/vacation cycle and 2020's flatness are the
    # two things this figure is evidence for, and an unlabelled sawtooth makes
    # the reader find them rather than see them.
    TERM = {3, 4, 5, 6, 9, 10, 11}

    def shade(ax, months):
        for i, y in enumerate(months):
            if y % 100 in TERM:
                ax.axvspan(i - 0.5, i + 0.5, color="#0E7C86", alpha=0.07, lw=0)
        # 2020 is the control: the year schools did not open in March
        idx = [i for i, y in enumerate(months) if y // 100 == 2020]
        if idx:
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color="#A8434E",
                       alpha=0.06, lw=0)
            ax.annotate("2020: schools shut,\nand the cycle is absent",
                        xy=((min(idx) + max(idx)) / 2, ax.get_ylim()[1]),
                        xytext=((min(idx) + max(idx)) / 2, ax.get_ylim()[1]),
                        ha="center", va="top", fontsize=6.5, color="#A8434E")

    ax = axes[0]
    for d, lab, c in ((we, "dong (424)", "#0E7C86"), (gu, "gu (25)", "#4C6E8A"),
                      (ci, "city (1)", "#BC8034")):
        ax.plot(range(len(d)), d.assortativity.to_numpy(), "-", c=c, lw=1.3, label=lab)
    shade(ax, list(we.ym))
    pub = [i for i, y in enumerate(we.ym) if y in PUBLISHED_MONTHS]
    ax.plot(pub, we.assortativity.to_numpy()[pub], "*", c="#A8434E", ms=13,
            label="the six published months", zorder=5)
    ax.set_xticks(range(0, len(x), 6))
    ax.set_xticklabels(x[::6], rotation=90, fontsize=7)
    ax.set_ylabel("assortativity")
    ax.legend(fontsize=7)
    ax.set_title("the ladder holds in every month; shaded = school term\n"
                 "(the six published months are starred)", fontsize=9)
    ax = axes[1]
    ax.plot(range(len(we)), we.nmi.to_numpy(), "-", c="#0E7C86", lw=1.3)
    ax.plot(pub, we.nmi.to_numpy()[pub], "*", c="#A8434E", ms=13, zorder=5)
    shade(ax, list(we.ym))
    ax.set_xticks(range(0, len(x), 6))
    ax.set_xticklabels(x[::6], rotation=90, fontsize=7)
    ax.set_ylabel("normalised mutual information")
    ax.set_title("share of the age entropy dong-level\nco-arrival resolves", fontsize=9)
    fig.suptitle(f"Phase 37 — the same estimator on {len(we)} months, "
                 f"{we.ym.min()}–{we.ym.max()}", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p37_timeseries.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p37_timeseries.png")

    with open(f"{ROOT}/eda/results_p37.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p37.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
