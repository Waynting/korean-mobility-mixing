"""Phase 19 — what three age bands cost you.

Lim, Lee & Jung (arXiv:2603.16064) model Seoul on the same 생활이동 data with
three age bands (0-19, 20-59, 60+) and report that restricting the non-routine
(O / our E) panel yields per-capita case reductions near zero. Our 16-band
result (p18, 18.3) is a monotone gradient in exactly that panel, running from
-3.96 pp at 20-24 to +0.84 pp at 80+ and crossing zero between 65-69 and 70-74.

Both can be true. This script shows why, and it makes two separate points that
should not be conflated:

  19.2  MAGNITUDE.  Collapsing 16 bands onto their 3 averages 20-24 (-3.96) with
        55-59 (near zero) inside one stratum, and 60-64 (-0.38) with 80+ (+0.84)
        inside another. The gradient is destroyed by construction, before any
        modelling choice is made.

  19.3  IDENTIFICATION.  This one is sharper and is the reason the aggregation is
        not merely lossy. A share taken WITHIN a fixed (age, month) is immune to
        the unknown KT expansion weight rho_a, because rho_a sits in both the
        numerator and the denominator (letter appendix, rows 1-2). A share taken
        over a band of ages is a rho-weighted mixture of the within-age shares,
        so it moves when rho_a drifts differentially across the ages inside the
        band. We re-price the December counts by that differential drift and show
        the 16-band numbers do not move at all while the 60+ band number does.

        The drift is now MEASURED rather than assumed, which is Phase 3's rerun
        obligation in ../archive/REVISION_PLAN.md. p21 estimates rho-hat = V / registered
        population per (age, dong, month), so its Mar->Dec change exists per
        5-year band, and the registered denominator has already taken out the
        demographic channel: 75+ raw volume rose 13.41% but rho-hat only 10.41%,
        and the difference is Seoul's elderly population growing, which is real
        and must NOT be re-priced away.

        What that measurement can and cannot carry has to be said exactly.
        rho-hat = kappa * lambda, and the 2020 window has no saturation anchor
        (p20d: the ownership ceiling is worth only <= +7.4% for 60-dae in 2020),
        so kappa's drift is NOT point identified here. The measured scenario is
        an ATTRIBUTION endpoint -- all of the measured per-capita rise charged to
        coverage -- and not a bound. It is not even a one-sided bound: if lambda
        fell over 2020, which December's controls make plausible, then kappa rose
        by more than rho-hat did. The two survey rows stay beside it for that
        reason, and because the letter already quotes them.

        Two things the measured scenario buys over the survey bounds it replaces.
        It is internal to this dataset, so the identification point no longer
        rests on any external source. And it is resolved to 5-year bands, where
        the surveys publish "60-dae" and "70-and-over" and so impose a step at 70
        that is reporting granularity, not a real age profile -- p21 section 3
        flags that step as an instance of this section's own argument, landing on
        the calibration source instead of on us.

Reads derived/gu_level.parquet. Two inherited caveats, both shared with p18 so
the numbers stay comparable with what the letter already quotes:
  * p1_conservation.py wrote that file WITHOUT a Seoul filter, so the
    Seoul-internal restriction is re-applied here;
  * it was written without DISTINCT, so the 475,266 duplicated raw row-pairs
    (0.034% of 1.389e9 rows) are double counted. Second-order for a share.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb
import pandas as pd

from common import DERIVED, AGE_LABEL, AGES, ROOT, cell_table

FIG = f"{ROOT}/eda/fig"
IMP = ["lo", "mid", "hi", "meas"]  # 0 / 1.5 / 3.0, plus p29's measured fill

# p29 measured what a masked cell holds by differencing our dong data against the
# official 자치구 release. That measurement was taken on Seoul-internal flows at
# gu level -- which is exactly this script's scope, the same file and the same
# filter -- so it transfers here with no assumption at all, unlike its use in
# p30 on the origin-Seoul numerator. `meas` is therefore the central case and
# lo/hi stay as the outer band; the measurement is a lower bound (the gu file
# masks 9.4% of its own cells), so the truth sits between meas and hi.
MEASURED_FILL = None               # loaded from results_p29.json at run time
MONTHS = [202003, 202012]          # the letter's headline comparison

# Lim, Lee & Jung (2026) age bands, mapped onto our 5-year boxes.
LIM_BANDS = {
    "0-19":  [0, 10, 15],
    "20-59": [20, 25, 30, 35, 40, 45, 50, 55],
    "60+":   [60, 65, 70, 75, 80],
}

# Coverage-drift scenarios, March -> December 2020: the fractional increase in
# the observable share rho_a, applied as n_corr = n_obs / (1 + d).
#
# SURVEY_DRIFT is the interpolated pair of bounds from two national surveys
# (letter section 3.1 / appendix). Superseded as of Phase 1 and kept as
# comparison rows only: they are what the letter currently quotes, and they are
# the loosest external statement available for a window the data cannot pin down.
SURVEY_DRIFT = {
    "survey_lower": {60: .017, 65: .017, 70: .089, 75: .089, 80: .089},
    "survey_upper": {60: .031, 65: .031, 70: .219, 75: .219, 80: .219},
}

# The measured scenario. Values are read from results_p21.json rather than
# re-derived, and asserted against the five figures memo/phase1c-rho.md section 3
# publishes, so a change upstream in p21 fails here loudly instead of silently
# re-pricing this section.
RHO_PUBLISHED_PCT = {"60-64": 0.40, "65-69": 4.51, "70-74": 4.38,
                     "75-79": 9.99, "80+": 11.12}

# Below 60 the measured rho-hat change is charged to behaviour and the scenario
# sets d = 0. Not a convenience: ownership in those bands was already 98.7-99.5%
# in the 2020 wave (derived/ownership.parquet), so there is no headroom for
# coverage to have grown into, while their rho-hat FELL by 3-15% -- charging that
# to coverage would mean KT lost a seventh of its 20-24 panel in nine months. It
# is also the conservative direction for the 20-59 band, whose movement any
# positive drift inside it would only widen.
HEADROOM_FROM = 60


def measured_drift():
    """rho-hat's own Mar->Dec change per 5-year band, as fractions."""
    with open(f"{ROOT}/eda/results_p21.json") as fh:
        pct = json.load(fh)["mar_to_dec_pct"]
    for lbl, want in RHO_PUBLISHED_PCT.items():
        got = pct[lbl]
        assert abs(got - want) < .006, f"p21 {lbl}: {got:.4f} vs published {want}"
    return {a: max(0.0, pct[AGE_LABEL[a]] / 100) if a >= HEADROOM_FROM else 0.0
            for a in AGES}


out = {}


def dump(df, key, note=""):
    out[key] = json.loads(df.to_json(orient="records"))
    print(f"\n=== {key} {note}\n{df.to_string(index=False)}", flush=True)


def main():
    global MEASURED_FILL
    # ------------------------------------------------------------- 19.0 base scan
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("CREATE TABLE cal(ym INTEGER, dow_n TINYINT, n_days TINYINT, n_holiday TINYINT)")
    con.executemany("INSERT INTO cal VALUES (?,?,?,?)",
                    [(r["ym"], r["dow_n"], r["n_days"], r["n_holiday"]) for r in cell_table()])

    print(f"scanning gu_level.parquet for ym in {MONTHS} ...", flush=True)
    base = con.execute(f"""
    SELECT g.ym, g.age,
           substr(g.mtype, 2, 1)             AS dest_attr,
           sum(g.pop_day_unmasked)           AS lo,
           sum(g.n_masked_cells / c.n_days)  AS n_masked_day
    FROM read_parquet('{DERIVED}/gu_level.parquet') g
    JOIN cal c ON c.ym = g.ym AND c.dow_n = g.dow_n
    WHERE g.ym IN ({','.join(map(str, MONTHS))})
      AND g.o_gu BETWEEN 1101 AND 1125 AND g.d_gu BETWEEN 1101 AND 1125
    GROUP BY 1,2,3
""").df()
    base["mid"] = base.lo + 1.5 * base.n_masked_day
    base["hi"] = base.lo + 3.0 * base.n_masked_day

    with open(f"{ROOT}/eda/results_p29.json") as fh:
        _p29 = json.load(fh)
    _fills = {}
    for _ym, _v in _p29.items():
        for _lbl, _x in _v["by_age"].items():
            if _x["masked_cells"] > 1e4:          # below that the difference is noise
                _fills.setdefault(_lbl, []).append(_x["per_cell"])
    MEASURED_FILL = {a: sum(_fills[AGE_LABEL[a]]) / len(_fills[AGE_LABEL[a]])
                     for a in AGES if AGE_LABEL[a] in _fills}
    base["meas"] = base.lo + base.age.map(MEASURED_FILL).fillna(0.0) * base.n_masked_day
    print("  measured fill per band: "
          + ", ".join(f"{AGE_LABEL[a]} {MEASURED_FILL[a]:.2f}"
                      for a in sorted(MEASURED_FILL)))
    out["measured_fill"] = {AGE_LABEL[a]: round(v, 4)
                            for a, v in MEASURED_FILL.items()}
    print(f"  {len(base)} (ym, age, dest_attr) rows", flush=True)


    def e_share(df, keys):
        """E-arrival share within each (ym, *keys) cell, per imputation."""
        g = df.groupby(["ym"] + keys + ["dest_attr"])[IMP].sum()
        tot = g.groupby(level=list(range(1 + len(keys)))).transform("sum")
        return (g / tot).xs("E", level="dest_attr")


    # ------------------------------------------------ 19.1 reproduce p18 (16 bands)
    s16 = e_share(base, ["age"])
    d16 = (s16.loc[202012] - s16.loc[202003]) * 100          # percentage points
    d16.index = [AGE_LABEL[a] for a in d16.index]
    d16 = d16.round(2)
    dump(d16.reset_index().rename(columns={"index": "age"}), "e_share_change_16band",
         "(Dec - Mar, pp; must match p18 mechanism_share_shift E_DecMar x 100)")

    with open(f"{ROOT}/eda/results_p18.json") as fh:
        p18 = {r["age"]: r for r in json.load(fh)["mechanism_share_shift"]}
    delta = max(abs(d16.loc[AGE_LABEL[a], "mid"] - p18[AGE_LABEL[a]]["E_DecMar_mid"] * 100)
                for a in AGES)
    print(f"\n  max |p19 - p18| over 16 ages, mid imputation: {delta:.3f} pp")
    out["p18_reproduction_max_abs_diff_pp"] = round(float(delta), 4)
    assert delta < 0.02, "p19 does not reproduce p18 -- stop and reconcile before using"

    # ---------------------------------------------------- 19.2 collapse to 3 bands
    band = base.copy()
    band["band"] = band.age.map({a: b for b, ages in LIM_BANDS.items() for a in ages})
    s3 = e_share(band, ["band"])
    d3 = ((s3.loc[202012] - s3.loc[202003]) * 100).reindex(LIM_BANDS).round(2)
    dump(d3.reset_index().rename(columns={"index": "band"}), "e_share_change_3band",
         "(Dec - Mar, pp; Lim/Lee/Jung bands, volume-weighted)")

    # Span is reported on the measured central case; p18's own numbers are the mid
    # column, which section 19.1 has already asserted against.
    rng16 = d16.meas.max() - d16.meas.min()
    rng3 = d3.meas.max() - d3.meas.min()
    rng16_mid = d16.mid.max() - d16.mid.min()
    span = pd.DataFrame([{
        "resolution": "16 bands", "min_pp": d16.meas.min(), "max_pp": d16.meas.max(),
        "range_pp": round(rng16, 2), "range_pp_mid_imputation": round(rng16_mid, 2),
        "sign_change": bool(d16.meas.min() < 0 < d16.meas.max())},
        {"resolution": "3 bands (Lim)", "min_pp": d3.meas.min(), "max_pp": d3.meas.max(),
         "range_pp": round(rng3, 2),
         "range_pp_mid_imputation": round(d3.mid.max() - d3.mid.min(), 2),
         "sign_change": bool(d3.meas.min() < 0 < d3.meas.max())}])
    dump(span, "resolution_span", "(the gradient the two resolutions can see)")
    print(f"\n  three bands retain {rng3 / rng16:.1%} of the 16-band spread.")
    out["range_retained_frac"] = round(float(rng3 / rng16), 4)

    # what the 20-59 band averages over, spelled out
    inside = pd.DataFrame({
        "band": [b for b, ages in LIM_BANDS.items() for a in ages],
        "age": [AGE_LABEL[a] for ages in LIM_BANDS.values() for a in ages],
        "pp_16band": [d16.meas.loc[AGE_LABEL[a]] for ages in LIM_BANDS.values() for a in ages]})
    inside["pp_3band"] = inside.band.map(d3.meas)
    inside["absorbed"] = (inside.pp_16band - inside.pp_3band).round(2)
    dump(inside, "what_the_bands_absorb", "(per-age deviation from its own band's value)")

    # --------------------------------------- 19.3 the identification point, priced
    # Correct December counts for documented coverage drift: n_corr = n_obs/(1+d).
    SCEN = {"none": None, "rho_measured": measured_drift(), **SURVEY_DRIFT}
    out["drift_scenarios_pct"] = {
        k: ({AGE_LABEL[a]: round(100 * v.get(a, 0.0), 3) for a in AGES}
            if v is not None else None) for k, v in SCEN.items()}

    rows = []
    for scen, dmap in SCEN.items():
        b = base.copy()
        if dmap is not None:
            f = b.age.map(dmap).fillna(0.0)
            m = b.ym == 202012
            for v in IMP:
                b.loc[m, v] = b.loc[m, v] / (1 + f[m])
        a16 = (e_share(b, ["age"]).loc[202012] - e_share(b, ["age"]).loc[202003]) * 100
        bb = b.copy()
        bb["band"] = bb.age.map({a: bd for bd, ages in LIM_BANDS.items() for a in ages})
        a3 = (e_share(bb, ["band"]).loc[202012] - e_share(bb, ["band"]).loc[202003]) * 100
        rows.append({"scenario": scen,
                     "16band_80+": round(float(a16.loc[80, "meas"]), 3),
                     "16band_70-74": round(float(a16.loc[70, "meas"]), 3),
                     "16band_20-24": round(float(a16.loc[20, "meas"]), 3),
                     "3band_60+": round(float(a3.loc["60+", "meas"]), 3),
                     "3band_20-59": round(float(a3.loc["20-59", "meas"]), 3)})
    drift = pd.DataFrame(rows)
    dump(drift, "coverage_drift_sensitivity",
         "(Dec - Mar E-share, pp, after correcting Dec counts for rho drift)")

    # Scenario-labelled, never positional: this project has already reported August
    # as September once by indexing a month by position (p10_response.py).
    D = drift.set_index("scenario")
    b3 = D.loc["none", "3band_60+"]
    mv16 = max(abs(D[c] - D.loc["none", c]).max() for c in
               ["16band_80+", "16band_70-74", "16band_20-24"])
    mv3_rho = abs(D.loc["rho_measured", "3band_60+"] - b3)
    mv3_all = abs(D["3band_60+"] - b3).max()
    flip_rho = (b3 > 0) != (D.loc["rho_measured", "3band_60+"] > 0)
    print(f"\n  16-band estimates move at most {mv16:.4f} pp across every scenario.")
    print(f"  the 3-band 60+ estimate moves {mv3_rho:.3f} pp under the MEASURED drift "
          f"alone ({'sign flips' if flip_rho else 'same sign'}); {mv3_all:.3f} pp over "
          f"all four scenarios.")
    out["drift_move_16band_max_pp"] = round(float(mv16), 5)
    out["drift_move_3band_60plus_pp"] = round(float(mv3_all), 4)
    out["drift_move_3band_60plus_measured_pp"] = round(float(mv3_rho), 4)
    out["sign_flip_under_measured_drift"] = bool(flip_rho)

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6),
                             gridspec_kw={"width_ratios": [1.85, 1]})
    ax = axes[0]
    labs = [AGE_LABEL[a] for a in AGES]
    x = range(len(labs))
    lo = [d16.lo.loc[l] for l in labs]
    hi = [d16.hi.loc[l] for l in labs]
    mid = [d16.meas.loc[l] for l in labs]
    ax.bar(x, mid, color=["#A8434E" if v > 0 else "#0E7C86" for v in mid],
           width=.68, label="16 bands (this work)")
    ax.vlines(x, lo, hi, color="#333", lw=1.1)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylim(min(lo) - .5, max(hi) + 1.6)
    top = ax.get_ylim()[1]
    start = 0
    for b, ages in LIM_BANDS.items():
        end = start + len(ages)
        ax.hlines(d3.meas.loc[b], start - .45, end - .55, color="#BC8034", lw=3.4,
                  zorder=6, label="3 bands (Lim et al.)" if start == 0 else None)
        ax.annotate(f"{d3.meas.loc[b]:+.2f}", (end - .55, d3.meas.loc[b]),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color="#8A5A18", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=.85, pad=.8))
        ax.text((start + end - 1) / 2, top - .28, b, ha="center", va="top",
                fontsize=9, color="#8A5A18", weight="bold")
        if end < len(labs):
            ax.axvline(end - .5, color="#999", lw=.7, ls=":")
        start = end
    ax.axhspan(top - .95, top, color="#BC8034", alpha=.07)
    ax.set_xticks(list(x)); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("change in E (discretionary) arrival share, pp")
    ax.set_title("Dec - Mar 2020, within-age share\n"
                 "bars = 16 bands at the measured masking fill, whiskers = 0..3", fontsize=10)
    ax.legend(fontsize=8, loc="lower left"); ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    w, xs = .2, [0, 1]
    LEG = {"none": "no correction", "rho_measured": "measured drift (rho-hat, p21)",
           "survey_lower": "survey bound, lower", "survey_upper": "survey bound, upper"}
    COL = {"none": "#4C6E8A", "rho_measured": "#BC8034",
           "survey_lower": "#7FA6C0", "survey_upper": "#C3D5E2"}
    vals = []
    for i, scen in enumerate(SCEN):
        r = D.loc[scen]
        for xc, v in zip(xs, [r["16band_80+"], r["3band_60+"]]):
            vals.append(v)
            ax.bar(xc + (i - 1.5) * w, v, width=w, color=COL[scen], edgecolor="#456",
                   lw=.5, label=LEG[scen] if xc == 0 else None)
            ax.annotate(f"{v:+.2f}", (xc + (i - 1.5) * w, v),
                        xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=7, color="#233")
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylim(min(vals) - .3, max(vals) + .35)
    ax.set_xticks(xs)
    ax.set_xticklabels(["16-band 80+\n(invariant)",
                        "3-band 60+\n(%s)" % ("sign flips" if flip_rho else "moves")],
                       fontsize=9)
    ax.set_ylabel("change in E arrival share, pp")
    ax.set_title("re-priced by measured drift; survey bounds for scale\n"
                 "within-age shares are invariant; band shares are not", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=.3)

    fig.suptitle("Phase 19 — three age bands cannot see the gradient, and cannot be "
                 "made weight-free", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p19_bandwidth.png", dpi=150)
    plt.close(fig)
    print("\n  -> fig/p19_bandwidth.png")

    with open(f"{ROOT}/eda/results_p19.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p19.json")


if __name__ == "__main__":
    main()
