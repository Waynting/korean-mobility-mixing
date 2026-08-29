#!/usr/bin/env python
"""Phase 9 — can the raw device counts be recovered?

T1.1 showed the per-device expansion weight is frozen to the cent across all 8
months. If the published number really is weight x (integer device-days), then
dividing by the weight recovers the unweighted sample size, and that changes two
things: censored cells stop being "somewhere in [0,3)" and become "exactly 1 or 2
devices", and every cell gains a usable sampling-variance.

The weight is not directly observable — the smallest published value in a stratum
is weight x ceil(3/weight), not the weight itself — so it is recovered as the
largest g that divides essentially all published values in the stratum.

The answer is no: the weight is a per-cell quantity, not a per-stratum constant,
so no stratum-wide divisor exists (9.1). But the same fact makes section 9.6
possible, and 9.6 is what this file is cited for — the U-shaped coverage profile
by age, which the manual independently corroborates (memo/phase1a-manual.md, 3).
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
from common import connect, ROOT, AGE_LABEL, AGES, FIG, YM_LABEL


def main():
    con = connect()
    out = {}


    def frac_int(vals, g, tol=0.02):
        """share of values that are within tol of an integer multiple of g"""
        q = vals / g
        return float(np.mean(np.abs(q - np.round(q)) <= tol))


    print("=" * 80)
    print("9.1  recover the quantum: largest g dividing the published values")
    print("=" * 80)
    rows = []
    for a in AGES:
        for s in ("F", "M"):
            v = con.execute(f"""
            SELECT DISTINCT pop FROM m
            WHERE ym=202001 AND age={a} AND sex='{s}' AND o_seoul AND NOT masked
            ORDER BY pop LIMIT 4000""").df()["pop"].to_numpy()
            mn = v.min()
            best = None
            for k in range(1, 9):                    # min published = g * k
                g = mn / k
                if g < 0.3:
                    break
                f = frac_int(v, g)
                if f > 0.98:
                    best = (g, k, f)                 # keep going: larger k -> smaller g
            # prefer the LARGEST g that still explains the values
            cands = []
            for k in range(1, 9):
                g = mn / k
                if g < 0.3:
                    break
                cands.append((g, k, frac_int(v, g)))
            ok = [c for c in cands if c[2] > 0.98]
            best = max(ok, key=lambda c: c[0]) if ok else None
            rows.append(dict(age=AGE_LABEL[a], sex=s, min_pop=round(float(mn), 3),
                             quantum=round(best[0], 4) if best else None,
                             k=best[1] if best else None,
                             explained=round(best[2], 4) if best else None,
                             n_distinct=len(v)))
    q = pd.DataFrame(rows)
    print(q.to_string(index=False))
    out["quantum"] = rows

    good = q.dropna(subset=["quantum"])
    print(f"\n  strata with a quantum explaining >98% of values: {len(good)}/{len(q)}")

    if len(good):
        print("\n" + "=" * 80)
        print("9.2  validate on held-out months: does the same quantum divide Sep too?")
        print("=" * 80)
        val = []
        for r in good.itertuples():
            a = [k for k, v in AGE_LABEL.items() if v == r.age][0]
            for ym in (202005, 202009):
                v = con.execute(f"""
                SELECT DISTINCT pop FROM m
                WHERE ym={ym} AND age={a} AND sex='{r.sex}' AND o_seoul AND NOT masked
                ORDER BY pop LIMIT 3000""").df()["pop"].to_numpy()
                val.append(dict(age=r.age, sex=r.sex, ym=YM_LABEL[ym],
                                quantum=r.quantum, explained=round(frac_int(v, r.quantum), 4)))
        v = pd.DataFrame(val)
        print(v.pivot_table(index=["age", "sex"], columns="ym", values="explained")
              .round(4).to_string())
        print(f"\n  held-out explained: min={v.explained.min():.4f} "
              f"median={v.explained.median():.4f}")
        out["quantum_validation"] = val

        print("\n" + "=" * 80)
        print("9.3  what censoring becomes: masked cell = 1..(ceil(3/g)-1) devices")
        print("=" * 80)
        g2 = good.copy()
        g2["max_hidden_devices"] = np.ceil(3 / g2["quantum"]).astype(int) - 1
        g2["old_upper_bound"] = 3.0
        g2["new_upper_bound"] = g2["max_hidden_devices"] * g2["quantum"]
        g2["mid_new"] = (g2["max_hidden_devices"] + 1) / 2 * g2["quantum"]
        print(g2[["age", "sex", "quantum", "max_hidden_devices",
                  "old_upper_bound", "new_upper_bound", "mid_new"]].to_string(index=False))
        out["censoring_refined"] = g2.to_dict("records")

    print("\n" + "=" * 80)
    print("9.4  Phase 0 re-check across all 8 months")
    print("=" * 80)
    print(con.execute("""
    SELECT ym, count(*) AS rows,
           count(DISTINCT o_dong) AS n_origin_codes,
           count(DISTINCT d_dong) AS n_dest_codes,
           count(DISTINCT o_dong) FILTER (o_seoul) AS n_seoul_dong,
           count(DISTINCT mtype) AS n_mtype, count(DISTINCT age) AS n_age,
           count(DISTINCT arr_hour) AS n_hour, count(DISTINCT dow_n) AS n_dow,
           round(count(*) FILTER (masked)/count(*)::DOUBLE, 4) AS mask_rate,
           round(min(pop), 2) AS min_pop, round(max(mean_min), 0) AS max_mean_min
    FROM m GROUP BY ym ORDER BY ym""").df().to_string(index=False))

    print("\n  codes appearing in some months but not others:")
    print(con.execute("""
    WITH a AS (SELECT DISTINCT ym, c FROM (
                 SELECT ym, o_dong AS c FROM m UNION ALL SELECT ym, d_dong FROM m))
    SELECT c AS code, count(*) AS n_months, string_agg(ym::VARCHAR, ',' ORDER BY ym) AS months
    FROM a GROUP BY c HAVING count(*) < 8 ORDER BY n_months DESC, c""").df().to_string(index=False))

    print("\n" + "=" * 80)
    print("9.5  censoring rate by month x age (does it move with volume?)")
    print("=" * 80)
    mk = con.execute("""
    SELECT ym, age, round(count(*) FILTER (masked)/count(*)::DOUBLE, 3) AS mask_rate
    FROM m WHERE o_seoul AND d_seoul GROUP BY 1,2""").df()
    p = mk.pivot(index="age", columns="ym", values="mask_rate")
    p.index = [AGE_LABEL[a] for a in p.index]
    p.columns = [YM_LABEL[c] for c in p.columns]
    print(p.to_string())
    out["mask_by_month_age"] = mk.to_dict("records")

    # --------------- 9.6 the U-shaped coverage profile by age (restored 2026-08-17)
    # This query used to exist only in the prose of memo/phase9-coverage.md section
    # 9.2, which left headline 4 of eda/README.md and ../reference/DATA_REPORT.md 3.4 flagged as
    # not reproducible. It is route R4 of Phase 1d (the independent check on the
    # kappa ordering), so it has to run from source.
    #
    # 9.1 kills the stratum-level quantum, but in doing so it locates where the
    # quantum does live: inside one OD cell the same weight is reused across the
    # seven weekdays, so the SMALLEST of those seven published values is that cell's
    # weight times the smallest device count the cell ever saw. On a cell quiet
    # enough to have had a one-device day, that minimum IS the weight.
    #
    # Three choices fix the numbers, and each of them would move them:
    #
    #   cell key = (o_dong, d_dong, arr_hour, mtype, age, sex), minimum over dow_n.
    #       Anything coarser pools cells whose weights differ (9.1 measured +-1-2%
    #       jitter even within a cell) and every minimum slides down toward the 3.00
    #       floor; anything finer has no repeated draws to take a minimum over.
    #
    #   at least 4 of the 7 weekdays present. A cell seen on one weekday gets one
    #       draw, and one draw is almost never the one-device day, so its minimum is
    #       an upper bound with nothing pulling it down. Requiring a majority of the
    #       week is what makes the minimum informative: drop the filter and the cell
    #       counts inflate ~7x while the 80+ median doubles to 18.86, i.e. two
    #       devices rather than one.
    #
    #   p10, not the median, is read as the weight. Both biases point up — busy
    #       cells never see a one-device day, and using unmasked values only
    #       truncates 20-44 from below at 3.00 — so the lowest order statistic that
    #       is still stable at these cell counts is the closest to the truth. For
    #       20-44 even p10 is an over-estimate; that band is not identified here.
    #
    # Duplicate rows (Phase 16: 475,266 exact copies) can only bite through the
    # weekday-coverage filter, since min() is idempotent, so n_dow is popcounted off
    # a weekday bitmask instead of taken from count(*). The memo's cell counts are
    # the undeduplicated ones and are carried alongside for the reproduction check:
    # dedup moves 0.07% of cells and not one quantile.
    print("\n" + "=" * 80)
    print("9.6  coverage profile: per-cell minimum across weekdays, by age")
    print("=" * 80)
    MIN_DOW = 4                                  # majority of the 7 weekdays


    def coverage(ym):
        return con.execute(f"""
        WITH c AS (
          SELECT age, min(pop) AS w, count(*) AS n_row,
                 bit_count(bit_or(1::UBIGINT << dow_n)) AS n_dow
          FROM m
          WHERE ym={ym} AND o_seoul AND d_seoul AND NOT masked
          GROUP BY o_dong, d_dong, arr_hour, mtype, age, sex)
        SELECT {ym} AS ym, age,
               count(*) FILTER (n_dow >= {MIN_DOW})              AS n_cells,
               count(*) FILTER (n_row >= {MIN_DOW})              AS n_cells_raw,
               quantile_cont(w, 0.10) FILTER (n_dow >= {MIN_DOW}) AS p10,
               quantile_cont(w, 0.50) FILTER (n_dow >= {MIN_DOW}) AS median,
               quantile_cont(w, 0.90) FILTER (n_dow >= {MIN_DOW}) AS p90
        FROM c GROUP BY age ORDER BY age""").df()


    cov = coverage(202001)
    cov["band"] = [AGE_LABEL[a] for a in cov["age"]]
    # p10 is the estimate; 20-44 is censoring-truncated at 3.00 and reads too high.
    cov["censored_band"] = cov["age"].between(20, 44)
    print(cov[["band", "p10", "median", "p90", "n_cells", "n_cells_raw",
               "censored_band"]].round(2).to_string(index=False))
    out["coverage_profile"] = cov.to_dict("records")

    lo = cov.loc[~cov.censored_band, "p10"].min()
    print(f"\n  best-covered band (censoring-free): {cov.loc[cov.p10.eq(lo), 'band'].iloc[0]}"
          f" at {lo:.2f}x")
    for a in (0, 80):
        r = cov[cov.age == a].iloc[0]
        print(f"  {r.band:<5} {r.p10:6.2f}x  -> 1 observed device stands for "
              f"{r.p10:.0f} people")
    print("  manual p.13 admits '열 배 가까이' for under-10 and over-80: "
          f"80+ lands at {cov[cov.age == 80].p10.iloc[0]:.1f}x")

    # Held-out months, same test as 9.2: a profile that is really a property of the
    # expansion weights must not move when the month changes. If it did, the U shape
    # would be a January artefact and R4 would be reading noise.
    val = pd.concat([coverage(ym) for ym in (202005, 202009)], ignore_index=True)
    val["band"] = [AGE_LABEL[a] for a in val["age"]]
    allm = pd.concat([cov, val], ignore_index=True)
    piv = allm.pivot(index="band", columns="ym", values="p10").reindex(
        [AGE_LABEL[a] for a in AGES])
    piv.columns = [YM_LABEL[c] for c in piv.columns]
    print("\n  p10 across held-out months:")
    print(piv.round(2).to_string())
    print(f"\n  max spread across the three months: "
          f"{(piv.max(axis=1) - piv.min(axis=1)).max():.3f}")
    out["coverage_profile_months"] = allm.drop(columns="censored_band",
                                               errors="ignore").to_dict("records")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(cov))
    ax.fill_between(x, cov["p10"], cov["p90"], color="#3a6ea5", alpha=0.16,
                    label="p10-p90 of cell minima")
    ax.plot(x, cov["median"], color="#3a6ea5", lw=1.0, ls="--", label="median")
    ax.plot(x, cov["p10"], color="#b03a48", lw=1.8, marker="o", ms=4.5,
            label="p10 = expansion weight")
    ax.axhline(10, color="#555", lw=0.9, ls=":")
    # Label kept in ASCII: matplotlib's default font has no CJK glyphs, so the
    # manual's own wording renders as tofu boxes in the saved PNG.
    ax.text(len(cov) - 0.1, 10.4, "manual p.13: 'nearly tenfold' for <10 and 80+",
            ha="right", fontsize=7.5, color="#555")
    for i, a in enumerate(cov["age"]):
        if 20 <= a <= 44:
            ax.axvspan(i - 0.5, i + 0.5, color="#999", alpha=0.10, lw=0)
    ax.text(np.mean(x[cov.censored_band.values]), 1.6,
            "censoring-truncated at 3.00\n(true weight is lower)", ha="center",
            fontsize=7.5, color="#444")
    ax.set_yscale("log")
    ax.minorticks_off()                          # else the log locator relabels 3/40
    ax.set_yticks([3, 4, 6, 10, 20, 40])
    ax.set_yticklabels(["3", "4", "6", "10", "20", "40"])
    ax.set_ylim(1.4, 55)
    ax.set_xticks(x); ax.set_xticklabels(cov["band"], rotation=45, fontsize=8)
    ax.set_ylabel("people per observed device")
    ax.set_title("KT coverage is U-shaped in age: expansion weight inverted from the "
                 "masking threshold\n(2020-01, both ends in Seoul, cells seen on >=4 "
                 "weekdays)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(f"{FIG}/p9_coverage_profile.png", dpi=150)
    plt.close(fig)
    print("  -> fig/p9_coverage_profile.png")

    with open(f"{ROOT}/eda/results_p9.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote results_p9.json")


if __name__ == "__main__":
    main()
