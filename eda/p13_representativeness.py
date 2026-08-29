#!/usr/bin/env python
"""Phase 13 — S1 of the screen: what does the aggregate index actually proxy?

The 0.95 threshold used earlier was borrowed, not derived. This computes the
criterion from the dataset itself: split each stratum into two disjoint halves of
the 424 dong (odd/even code), correlate the two halves' response shapes, and take
the 5th percentile as rho*. That is the correlation noise alone can produce here,
so a between-stratum correlation below it is a real difference, not sampling.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import connect, ROOT, AGES, AGE_LABEL

FIG = f"{ROOT}/eda/fig"
BASE = 202001


def main():
    con = connect()

    df = con.execute("""
    SELECT ym, age, arr_hour, (d_dong % 2) AS half,
           sum(pop_day) + 1.5*coalesce(sum(1.0/n_days) FILTER (masked), 0) AS v
    FROM m WHERE o_seoul AND d_seoul AND dow_n <= 5 AND n_holiday = 0
    GROUP BY 1,2,3,4""").df()


    def shapes(d, keys):
        b = d[d.ym == BASE].set_index(keys + ["arr_hour"])["v"]
        x = d[d.ym != BASE].set_index(["ym"] + keys + ["arr_hour"])["v"]
        r = (x / b).rename("r").reset_index()
        out = {}
        for k, g in r.groupby(["ym"] + keys):
            v = g.sort_values("arr_hour")["r"].to_numpy()
            out[k] = v / v.mean()
        return out


    sh_half = shapes(df, ["age", "half"])
    sh_age = shapes(df.groupby(["ym", "age", "arr_hour"], as_index=False)["v"].sum(), ["age"])
    sh_tot = shapes(df.groupby(["ym", "arr_hour"], as_index=False)["v"].sum(), [])
    yms = sorted({k[0] for k in sh_age})

    rel = [np.corrcoef(sh_half[(y, a, 0)], sh_half[(y, a, 1)])[0, 1] for y in yms for a in AGES]
    btw = [np.corrcoef(sh_age[(y, a)], sh_age[(y, b)])[0, 1]
           for y in yms for i, a in enumerate(AGES) for b in AGES[i + 1:]]
    vs = {a: [np.corrcoef(sh_age[(y, a)], sh_tot[(y,)])[0, 1] for y in yms] for a in AGES}
    allvs = [v for a in AGES for v in vs[a]]
    rho = float(np.quantile(rel, 0.05))

    print("=== S1 criterion, self-calibrated from split-half reliability ===")
    print(f"  within-stratum split-half   n={len(rel):>5}  p05={rho:.3f}  median={np.median(rel):.3f}")
    print(f"  between-stratum, same month n={len(btw):>5}  p05={np.quantile(btw,.05):+.3f}"
          f"  median={np.median(btw):.3f}")
    print(f"  stratum vs aggregate        n={len(allvs):>5}  p05={np.quantile(allvs,.05):+.3f}"
          f"  median={np.median(allvs):.3f}")
    print(f"\n  rho* = {rho:.3f}")
    print(f"  between-stratum pairs at or above rho*: {np.mean(np.array(btw) >= rho):.1%}")
    print(f"  stratum-vs-aggregate at or above rho*:  {np.mean(np.array(allvs) >= rho):.1%}")

    print("\n=== which strata does the aggregate actually proxy? ===")
    tab = pd.DataFrame({"age": [AGE_LABEL[a] for a in AGES],
                        "median": [round(float(np.median(vs[a])), 3) for a in AGES],
                        "min": [round(float(min(vs[a])), 3) for a in AGES]})
    tab["proxied"] = np.where(tab["median"] >= rho, "yes", "NO")
    print(tab.to_string(index=False))
    ok = tab[tab.proxied == "yes"]["age"].tolist()
    print(f"\n  aggregate is a proxy for: {', '.join(ok)}")

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(AGES))
    med = [np.median(vs[a]) for a in AGES]
    lo = [np.min(vs[a]) for a in AGES]
    hi = [np.max(vs[a]) for a in AGES]
    ax.vlines(x, lo, hi, color="#A9B9C0", lw=2)
    ax.scatter(x, med, s=44, color="#0E7C86", zorder=3, label="stratum vs aggregate (median over months)")
    ax.axhline(rho, color="#9A6C15", lw=1.6, ls="--",
               label=f"ρ* = {rho:.3f}  (5th pct of within-stratum split-half)")
    ax.axhline(np.median(rel), color="#A9B9C0", lw=1.1, ls=":",
               label=f"split-half median = {np.median(rel):.3f}  (noise ceiling)")
    ax.set_xticks(x); ax.set_xticklabels([AGE_LABEL[a] for a in AGES], rotation=90, fontsize=8)
    ax.set_ylabel("shape correlation with the aggregate curve")
    ax.set_title("S1 — the aggregate mobility index proxies 40-64 and little else", fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/p13_s1_representativeness.png", dpi=150); plt.close(fig)
    print("\n  -> fig/p13_s1_representativeness.png")

    with open(f"{ROOT}/eda/results_p13.json", "w") as fh:
        json.dump(dict(rho_star=rho, split_half_median=float(np.median(rel)),
                       between_median=float(np.median(btw)),
                       vs_aggregate=tab.to_dict("records")), fh, indent=1)
    print("wrote results_p13.json")


if __name__ == "__main__":
    main()
