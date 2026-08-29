#!/usr/bin/env python
"""Phase 6 — first look at the diurnal response, and the shape-correlation test.

With only 202001 and 202009 on disk this is a two-window contrast, not a dose
ladder, so it cannot settle the go/no-go. What it can do is run the Taipei
statistic once: build the age x sex stratified diurnal response curve
r(h) = Sep(h)/Jan(h), normalise each to its own mean, and ask whether the shapes
agree across strata. Agreement is the "shape is a system property" branch;
disagreement is the "aggregation is hiding a mixture" branch.

Restricted to Tue+Thu (holiday-free in both months) and Seoul-internal flows.
Every quantity is per calendar day. Censoring bounds accompany the point
estimates because the mask rate is not equal across the two months.
"""
import itertools
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import connect, ROOT, AGE_LABEL

FIG = f"{ROOT}/eda/fig"


def main():
    con = connect()
    out = {}
    AGES = sorted(AGE_LABEL)

    # ------------------------------------------------------------ 6.0 the raw panel
    df = con.execute("""
    SELECT ym, age, sex, arr_hour,
           sum(pop_day)                                             AS lo,
           sum(pop_day) + 1.5*coalesce(sum(1.0/n_days) FILTER (masked),0) AS mid,
           sum(pop_day) + 3.0*coalesce(sum(1.0/n_days) FILTER (masked),0) AS hi
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
    GROUP BY 1,2,3,4""").df()
    piv = df.pivot_table(index=["age", "sex", "arr_hour"], columns="ym",
                         values=["lo", "mid", "hi"])
    r = pd.DataFrame(index=piv.index)
    r["ratio"] = piv[("mid", 202009)] / piv[("mid", 202001)]
    r["ratio_lo"] = piv[("lo", 202009)] / piv[("hi", 202001)]
    r["ratio_hi"] = piv[("hi", 202009)] / piv[("lo", 202001)]
    r = r.reset_index()
    out["response_panel"] = r.to_dict("records")

    # ------------------------------------------------- 6.1 level response by stratum
    lev = con.execute("""
    SELECT age, sex,
      sum(pop_day) FILTER (ym=202001) AS jan_lo,
      sum(pop_day) FILTER (ym=202001)
        + 3.0*coalesce(sum(1.0/n_days) FILTER (masked AND ym=202001),0) AS jan_hi,
      sum(pop_day) FILTER (ym=202001)
        + 1.5*coalesce(sum(1.0/n_days) FILTER (masked AND ym=202001),0) AS jan_mid,
      sum(pop_day) FILTER (ym=202009) AS sep_lo,
      sum(pop_day) FILTER (ym=202009)
        + 3.0*coalesce(sum(1.0/n_days) FILTER (masked AND ym=202009),0) AS sep_hi,
      sum(pop_day) FILTER (ym=202009)
        + 1.5*coalesce(sum(1.0/n_days) FILTER (masked AND ym=202009),0) AS sep_mid
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
    GROUP BY 1,2 ORDER BY 1,2""").df()
    lev["ratio"] = lev["sep_mid"] / lev["jan_mid"]
    lev["ratio_unmasked_only"] = lev["sep_lo"] / lev["jan_lo"]
    lev["ratio_lo"] = lev["sep_lo"] / lev["jan_hi"]
    lev["ratio_hi"] = lev["sep_hi"] / lev["jan_lo"]
    print("=== 6.1 whole-day level response Sep/Jan by age x sex (Tue+Thu) ===")
    w = lev.pivot(index="age", columns="sex",
                  values=["ratio", "ratio_unmasked_only", "ratio_lo", "ratio_hi"]
                  ).reindex(AGES).round(4)
    w.index = [AGE_LABEL[a] for a in w.index]
    print(w.to_string())
    out["level_response"] = lev.to_dict("records")

    # ------------------------------------------------------ 6.2 curves and shapes
    shapes = {}
    for (a, s), g in r.groupby(["age", "sex"]):
        g = g.sort_values("arr_hour")
        v = g["ratio"].to_numpy()
        shapes[(a, s)] = v / np.nanmean(v)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, s in zip(axes, ("F", "M")):
        for i, a in enumerate(AGES):
            ax.plot(range(24), shapes[(a, s)], color=cmap(i / (len(AGES) - 1)),
                    lw=1.3, label=AGE_LABEL[a] if s == "F" else None)
        ax.axhline(1, color="k", lw=.6, ls="--")
        ax.set_title(f"sex = {s}", fontsize=10); ax.set_xlabel("arrival hour")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("normalised response  r(h)/mean r")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Phase 6 — diurnal response shape, Sep2020 / Jan2020, Seoul-internal, Tue+Thu",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/p6_response_shapes.png", dpi=150); plt.close(fig)
    print("\n  -> fig/p6_response_shapes.png")

    # raw (un-normalised) response curves too
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, s in zip(axes, ("F", "M")):
        for i, a in enumerate(AGES):
            g = r[(r.age == a) & (r.sex == s)].sort_values("arr_hour")
            ax.plot(g["arr_hour"], g["ratio"], color=cmap(i / (len(AGES) - 1)), lw=1.3,
                    label=AGE_LABEL[a] if s == "F" else None)
        ax.axhline(1, color="k", lw=.6, ls="--")
        ax.set_title(f"sex = {s}", fontsize=10); ax.set_xlabel("arrival hour"); ax.grid(alpha=.3)
    axes[0].set_ylabel("r(h) = Sep / Jan"); axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG}/p6_response_raw.png", dpi=150); plt.close(fig)
    print("  -> fig/p6_response_raw.png")

    # --------------------------------------------- 6.3 the shape-correlation statistic
    keys = [(a, s) for s in ("F", "M") for a in AGES]
    M = np.array([shapes[k] for k in keys])
    C = np.corrcoef(M)
    labels = [f"{AGE_LABEL[a]}{s}" for a, s in keys]
    print("\n=== 6.3 pairwise shape correlation across the 32 age x sex strata ===")
    iu = np.triu_indices(len(keys), 1)
    vals = C[iu]
    print(f"  n pairs = {len(vals)}   min = {vals.min():.4f}   p05 = {np.quantile(vals,.05):.4f}"
          f"   median = {np.median(vals):.4f}   max = {vals.max():.4f}")
    print(f"  pairs with rho > 0.95: {(vals > .95).mean():.1%}    > 0.90: {(vals > .90).mean():.1%}")
    worst = sorted(zip(vals, [(labels[i], labels[j]) for i, j in zip(*iu)]))[:10]
    print("  10 least-similar pairs:")
    for v, (x, y) in worst:
        print(f"    {x:>8} vs {y:<8} rho={v:.4f}")
    out["shape_corr"] = dict(min=float(vals.min()), median=float(np.median(vals)),
                             max=float(vals.max()),
                             frac_gt_095=float((vals > .95).mean()),
                             frac_gt_090=float((vals > .90).mean()),
                             worst=[[float(v), x, y] for v, (x, y) in worst])

    fig, ax = plt.subplots(figsize=(9, 7.6))
    im = ax.imshow(C, cmap="RdYlBu_r", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("shape correlation of the diurnal response, age x sex strata", fontsize=10)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(f"{FIG}/p6_shape_corr.png", dpi=150); plt.close(fig)
    print("  -> fig/p6_shape_corr.png")

    # --------------------------- 6.4 does the aggregate hide the strata? (mixture test)
    agg = con.execute("""
    SELECT arr_hour,
           sum(pop_day) FILTER (ym=202009)/sum(pop_day) FILTER (ym=202001) AS ratio
    FROM m WHERE o_seoul AND d_seoul AND dow_n IN (2,4)
    GROUP BY 1 ORDER BY 1""").df()
    agg_shape = agg["ratio"].to_numpy() / agg["ratio"].mean()
    cors = {f"{AGE_LABEL[a]}{s}": float(np.corrcoef(agg_shape, shapes[(a, s)])[0, 1])
            for a, s in keys}
    print("\n=== 6.4 each stratum's shape vs the aggregate shape ===")
    print(f"  min={min(cors.values()):.4f}  median={np.median(list(cors.values())):.4f}"
          f"  max={max(cors.values()):.4f}")
    for k, v in sorted(cors.items(), key=lambda x: x[1])[:6]:
        print(f"    worst: {k:>8} rho={v:.4f}")
    out["vs_aggregate"] = cors

    # ----------------------------------------------- 6.5 how wide are censoring bands
    band = r.assign(width=lambda d: d.ratio_hi - d.ratio_lo)
    print("\n=== 6.5 width of the censoring interval on r(h) ===")
    print(band.groupby("age")["width"].agg(["min", "median", "max"]).round(3)
          .rename(index=AGE_LABEL).to_string())
    out["censor_band"] = band.groupby("age")["width"].median().to_dict()

    with open(f"{ROOT}/eda/results_p6.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p6.json")


if __name__ == "__main__":
    main()
