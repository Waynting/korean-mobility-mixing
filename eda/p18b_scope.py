#!/usr/bin/env python
"""Does p18/p19's both-ends-in-Seoul filter bend the headline gradient?

Phase 1c had to choose a spatial scope for rho-hat and chose origin-in-Seoul with
the destination unrestricted: a resident who leaves Seoul is still an observed
resident, and the expansion weight hangs on the residence dong either way. p18
and p19 use a different scope — both ends in Seoul — and Phase 1c measured that
the two differ by 15-20% on the LEVEL of rho-hat. That was flagged as a named
follow-up rather than assumed away, and this script closes it.

The levels differing does not by itself mean the headline is wrong. p18 and p19
report a SHARE within a fixed (age, month) — what fraction of arrivals land on
the E (기타) attribute — and a share can be immune to a filter that moves the
level a lot. Whether it is depends on one thing: does the propensity to leave
Seoul correlate with the destination attribute, differently across ages? If
leaving Seoul is disproportionately an E-attribute trip for the elderly and a
W-attribute trip for the young, then dropping those trips tilts the gradient.
That is an empirical question and it is answered here rather than argued.

Three things are reported:

  the reproduction   the both-ends numbers must match results_p18.json before any
                     comparison means anything, so that is asserted, not eyeballed.
  the gradient       E-share change Dec - Mar under both scopes, by age.
  the mechanism      how much volume the destination filter removes per age and
                     per destination attribute, which is what would drive any gap.

Reads derived/gu_level.parquet, which p1_conservation.py wrote WITHOUT a Seoul
filter — that is why both scopes are available from one file. It also inherits
that file's lack of DISTINCT, so the 475,266 duplicated row-pairs are double
counted; that is 0.034% and it lands on both scopes identically, so it cannot
create the difference being measured.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb
import numpy as np
import pandas as pd
from common import AGE_LABEL, AGES, cell_table
from paths import DERIVED, FIG, ROOT, require

IMP = {"lo": 0.0, "mid": 1.5, "hi": 3.0}
MONTHS = (202003, 202012)
# Lim, Lee & Jung's three bands, as in p19.
LIM_BANDS = {"0-19": [0, 10, 15], "20-59": [20, 25, 30, 35, 40, 45, 50, 55],
             "60+": [60, 65, 70, 75, 80]}
out = {}


def main():
    require(DERIVED / "gu_level.parquet", "gu_level.parquet")
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")          # agents are using the same drive
    con.execute("PRAGMA disable_progress_bar")
    con.execute("CREATE TABLE cal(ym INTEGER, dow_n TINYINT, n_days TINYINT)")
    con.executemany("INSERT INTO cal VALUES (?,?,?)",
                    [(r["ym"], r["dow_n"], r["n_days"]) for r in cell_table()])

    print("scanning gu_level.parquet under both scopes ...", flush=True)
    base = con.execute(f"""
        SELECT g.ym, g.age, substr(g.mtype, 2, 1) AS dest_attr,
               sum(g.pop_day_unmasked) FILTER (g.d_gu BETWEEN 1101 AND 1125)
                                                        AS lo_int,
               sum(g.n_masked_cells / c.n_days) FILTER (g.d_gu BETWEEN 1101 AND 1125)
                                                        AS msk_int,
               sum(g.pop_day_unmasked)                  AS lo_out,
               sum(g.n_masked_cells / c.n_days)         AS msk_out
        FROM read_parquet('{DERIVED}/gu_level.parquet') g
        JOIN cal c ON c.ym = g.ym AND c.dow_n = g.dow_n
        WHERE g.o_gu BETWEEN 1101 AND 1125
        GROUP BY 1,2,3""").df()
    for s in ("int", "out"):
        for k, c in IMP.items():
            base[f"{s}_{k}"] = base[f"lo_{s}"] + c * base[f"msk_{s}"]

    def e_share(keys, scope, imp):
        col = f"{scope}_{imp}"
        g = base.groupby(["ym"] + keys + ["dest_attr"])[col].sum()
        return (g / g.groupby(level=list(range(1 + len(keys)))).sum()) \
            .xs("E", level="dest_attr")

    # ------------------------------------------------------- 1 the reproduction
    print("\n=== 1 does the both-ends scope reproduce results_p18.json? ===")
    s_int = e_share(["age"], "int", "mid")
    d_int = (s_int.loc[202012] - s_int.loc[202003]) * 100
    with open(f"{ROOT}/eda/results_p18.json") as fh:
        ref = {r["age"]: r["E_DecMar_mid"] * 100
               for r in json.load(fh)["mechanism_share_shift"]}
    worst = max(abs(d_int[a] - ref[AGE_LABEL[a]]) for a in AGES)
    for a in AGES:
        print(f"  {AGE_LABEL[a]:<7} here {d_int[a]:+7.3f} pp   p18 "
              f"{ref[AGE_LABEL[a]]:+7.3f} pp   diff {d_int[a] - ref[AGE_LABEL[a]]:+.4f}")
    print(f"  worst |difference| = {worst:.4f} pp")
    assert worst < 0.02, f"does not reproduce p18 ({worst:.4f} pp) — stop here"
    print("  reproduces p18 to rounding; the comparison below is meaningful")

    # --------------------------------------------------------- 2 the comparison
    print("\n=== 2 E-share change Dec - Mar, both scopes ===")
    s_out = e_share(["age"], "out", "mid")
    d_out = (s_out.loc[202012] - s_out.loc[202003]) * 100
    rows = []
    print(f"  {'band':<7} {'both ends':>10} {'origin only':>12} {'shift':>8} "
          f"{'level Mar':>10} {'level Mar':>10}")
    for a in AGES:
        rows.append(dict(band=AGE_LABEL[a], both_ends_pp=float(d_int[a]),
                         origin_only_pp=float(d_out[a]),
                         shift_pp=float(d_out[a] - d_int[a]),
                         level_mar_int=float(s_int.loc[202003, a]),
                         level_mar_out=float(s_out.loc[202003, a])))
        print(f"  {AGE_LABEL[a]:<7} {d_int[a]:>+10.3f} {d_out[a]:>+12.3f} "
              f"{d_out[a] - d_int[a]:>+8.3f} {s_int.loc[202003, a]:>10.4f} "
              f"{s_out.loc[202003, a]:>10.4f}")
    cmp_ = pd.DataFrame(rows)
    out["by_age"] = cmp_.to_dict("records")

    # Does the gradient survive as a gradient? Three claims are load-bearing:
    # its sign at the top, where it crosses zero, and its monotonicity in the
    # elderly bands. Each is checked under both scopes rather than assumed.
    def crossing(d):
        old = [a for a in AGES if a >= 60]
        for lo, hi in zip(old, old[1:]):
            if d[lo] < 0 <= d[hi]:
                return f"{AGE_LABEL[lo]}/{AGE_LABEL[hi]}"
        return "none in 60+"

    def monotone_elderly(d):
        v = [d[a] for a in AGES if a >= 60]
        return all(x <= y + 1e-9 for x, y in zip(v, v[1:]))

    print()
    for name, d in (("both ends", d_int), ("origin only", d_out)):
        print(f"  {name:<12} 80+ = {d[80]:+.3f} pp,  20-24 = {d[20]:+.3f} pp,  "
              f"zero crossing {crossing(d)},  60+ monotone {monotone_elderly(d)}")
    out["claims"] = {
        s: {"top_band_pp": float(d[80]), "young_pp": float(d[20]),
            "zero_crossing": crossing(d), "elderly_monotone": bool(monotone_elderly(d))}
        for s, d in (("both_ends", d_int), ("origin_only", d_out))}

    # ---------------------------------------------------------- 3 the mechanism
    # Any gap has to come from the destination filter removing trips unevenly
    # across (age, destination attribute). If the dropped share were flat, the
    # share estimator could not move at all.
    print("\n=== 3 what the destination filter removes, March ===")
    m = base[base.ym == 202003].groupby(["age", "dest_attr"], as_index=False)[
        ["int_mid", "out_mid"]].sum()
    m["dropped"] = 1 - m.int_mid / m.out_mid
    piv = m.pivot(index="age", columns="dest_attr", values="dropped")
    print(f"  {'band':<7} {'-> E':>8} {'-> H':>8} {'-> W':>8}   {'E minus W':>10}")
    for a in AGES:
        print(f"  {AGE_LABEL[a]:<7} {piv.loc[a, 'E']:>8.2%} {piv.loc[a, 'H']:>8.2%} "
              f"{piv.loc[a, 'W']:>8.2%}   {piv.loc[a, 'E'] - piv.loc[a, 'W']:>+10.2%}")
    spread = (piv["E"] - piv["W"]).abs().max()
    print(f"  largest |E minus W| gap in dropped share: {spread:.2%} "
          f"({AGE_LABEL[(piv['E'] - piv['W']).abs().idxmax()]})")
    out["dropped_share_march"] = piv.reset_index().assign(
        band=[AGE_LABEL[a] for a in piv.index]).to_dict("records")

    # Section 3 explains the LEVEL gap. The gap in the CHANGE needs one more step,
    # because the origin-only share is a mixture:
    #     s_out = (1 - w) * s_in + w * s_ext
    # where w is the out-of-Seoul share of volume and s_ext the E-share among
    # trips that leave. So the two scopes' Dec-minus-March changes differ by
    # exactly the change in w * (s_ext - s_in), and both factors have to be shown.
    print("\n=== 3b why the CHANGE differs: leaving-Seoul weight x E-share gap ===")
    ext = base.copy()
    ext["ext_mid"] = ext.out_mid - ext.int_mid
    tot = ext.groupby(["ym", "age"], as_index=False).agg(
        out=("out_mid", "sum"), ext=("ext_mid", "sum"))
    tot["w"] = tot.ext / tot.out
    eo = (ext[ext.dest_attr == "E"].groupby(["ym", "age"], as_index=False)
          .agg(e_ext=("ext_mid", "sum"), e_int=("int_mid", "sum")))
    den = (ext.groupby(["ym", "age"], as_index=False)
           .agg(t_ext=("ext_mid", "sum"), t_int=("int_mid", "sum")))
    mix = tot.merge(eo, on=["ym", "age"]).merge(den, on=["ym", "age"])
    mix["s_ext"] = mix.e_ext / mix.t_ext
    mix["s_in"] = mix.e_int / mix.t_int
    mix["term"] = mix.w * (mix.s_ext - mix.s_in)
    mm = mix[mix.ym.isin(MONTHS)].pivot(index="age", columns="ym",
                                        values=["w", "s_ext", "s_in", "term"])
    print(f"  {'band':<7} {'w Mar':>7} {'w Dec':>7} {'s_ext-s_in Mar':>15} "
          f"{'Dec':>7}   {'d(term)':>9} {'observed':>9}")
    for a in AGES:
        d_term = (mm[("term", 202012)][a] - mm[("term", 202003)][a]) * 100
        obs = d_out[a] - d_int[a]
        print(f"  {AGE_LABEL[a]:<7} {mm[('w', 202003)][a]:>7.3f} "
              f"{mm[('w', 202012)][a]:>7.3f} "
              f"{mm[('s_ext', 202003)][a] - mm[('s_in', 202003)][a]:>15.3f} "
              f"{mm[('s_ext', 202012)][a] - mm[('s_in', 202012)][a]:>7.3f}   "
              f"{d_term:>+9.3f} {obs:>+9.3f}")
    out["mixture_decomposition"] = mix[mix.ym.isin(MONTHS)].to_dict("records")

    # ------------------------------------------------- 4 does p19 survive too?
    print("\n=== 4 the 3-band collapse (p19's argument) under both scopes ===")
    band_of = {a: b for b, aa in LIM_BANDS.items() for a in aa}
    base["band"] = base.age.map(band_of)
    for scope, label in (("int", "both ends"), ("out", "origin only")):
        s3 = e_share(["band"], scope, "mid")
        d3 = (s3.loc[202012] - s3.loc[202003]) * 100
        s16 = e_share(["age"], scope, "mid")
        d16 = (s16.loc[202012] - s16.loc[202003]) * 100
        rng = d16.max() - d16.min()
        rng3 = d3.max() - d3.min()
        print(f"  {label:<12} 60+ band = {d3['60+']:+.3f} pp "
              f"(16-band 60-64 {d16[60]:+.2f} .. 80+ {d16[80]:+.2f}); "
              f"3-band keeps {rng3 / rng:.1%} of the 16-band range")
        out[f"three_band_{scope}"] = {
            "band_pp": {k: float(v) for k, v in d3.items()},
            "range_kept": float(rng3 / rng)}

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(AGES))
    ax.bar(x - 0.2, [d_int[a] for a in AGES], 0.4, label="both ends in Seoul (p18/p19)",
           color="#3a6ea5")
    ax.bar(x + 0.2, [d_out[a] for a in AGES], 0.4, label="origin in Seoul only (Phase 1c)",
           color="#b03a48")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([AGE_LABEL[a] for a in AGES],
                                         rotation=45, fontsize=8)
    ax.set_ylabel("E-share change, Dec - Mar (pp)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("does the spatial scope bend the headline gradient?", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/p18b_scope.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p18b_scope.png")

    with open(f"{ROOT}/eda/results_p18b.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print("wrote results_p18b.json")


if __name__ == "__main__":
    main()
