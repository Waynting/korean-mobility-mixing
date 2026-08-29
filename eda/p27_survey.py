#!/usr/bin/env python
"""Phase 7 — the passive matrix against Chae et al.'s survey matrix.

The advisor calls this the Q2-to-Q1 dividing line, and the reason is the timing:
the survey ran 2023-12-06..12 and 2024-02-07..13, and this project's mobility
data covers those exact months, so the comparison is on a genuinely overlapping
window rather than across a four-year gap.

He is also specific about what to report: dominant-eigenvector correlation,
assortativity index, and the ratio of next-generation-matrix dominant
eigenvalues -- NOT Pearson correlation of the cells, which he calls too
forgiving. Pearson is computed anyway, precisely so the gap between it and the
strict measures is visible.

REPRODUCE BEFORE USING. Section 7a of the plan requires the descriptor's own
published summary to come back out of the microdata before anything is built on
it: 1,987 respondents, 133,776 contacts, 4.81 contacts per person per day. The
third of those is the one that matters, because it fixes the DENOMINATOR: 4.81 =
133776 / 1987 / 14, so the diary is 14 days per respondent -- both weeks, not one
-- and every respondent did record in both months. Halving that denominator would
double the survey's per-capita scale and the NGM eigenvalue ratio with it.

THE PANEL MAPPING is the part that needs care, and the manual settled it. Our W
is 근무지 또는 학교 -- work and school together -- while Mossong/CoMix-style
surveys separate them. Chae records the place of each contact, so the survey can
be collapsed to OUR definition rather than the other way round:

    W  <- Q5 place 2 (workplace) + 3 (educational facility)
    E  <- places 4-8 (religious, restaurant/cafe/bar, hospital, outdoor, other)
    H  <- place 1 (household), excluded from both, as in the mobility panels

Q5_1..Q5_8 hold the place code when selected and are blank otherwise, so a
contact can carry several. 10.5% do; each is assigned to its first selected
place, which is household for 3.4% of all contacts. That rule never double counts
-- inflating the total would corrupt the per-capita scale the eigenvalue ratio
depends on -- but it does push multi-place contacts towards H, i.e. out of both
panels, so it is conservative on level and reported as a number rather than a
footnote.

DAY-SET ALIGNMENT. Both diary weeks run Wed-Tue, so each covers every weekday
exactly once and no re-weighting for day composition is needed. What does need
handling is 설 연휴: 2024-02-09..12 falls INSIDE the February diary week. This
data has no date column, so a holiday can only be removed by dropping the whole
weekday -- and then the same weekdays must go from the survey side, which is
possible there because the survey does have dates. The primary comparison
therefore runs on holiday-free weekdays only: Tue-Sun for 2023-12 (Christmas is a
Monday) and Tue-Thu for 2024-02. The unrestricted version is reported beside it,
and the two differ enough in February to make the point on their own.

WHAT REMAINS MISMATCHED, and is not fixable here: the passive side is a monthly
average over 4-5 occurrences of each weekday, the survey side is one specific
week; the survey is a national sample (465 of 1,987 respondents live in Seoul,
and the Seoul subsample is the primary); and respondent age tops out at 79, so
there are no 80+ egos and every square-matrix statistic is computed on the 0-79
block.

    python eda/p27_survey.py
"""
import csv
import datetime as dt
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import AGE_LABEL, AGES
from p26_matrix import holiday_free_dows
from paths import DATA_ROOT, FIG, ROOT, require

CHAE = DATA_ROOT / "raw" / "chae2026"
SURVEY_MONTHS = [202312, 202402]
PLACE_PANEL = {2: "W", 3: "W", 4: "E", 5: "E", 6: "E", 7: "E", 8: "E", 1: "H"}
PUBLISHED = dict(respondents=1987, contacts=133776, per_person_day=4.81)
DIARY_DAYS = 14
DIARY_WEEK = {202312: [dt.date(2023, 12, d) for d in range(6, 13)],
              202402: [dt.date(2024, 2, d) for d in range(7, 14)]}
NA = len(AGES)
BOOT = 500
out = {}


def band(age):
    if age < 10:
        return 0
    if age < 15:
        return 10
    if age < 20:
        return 15
    return 80 if age >= 80 else (age // 5) * 5


def parse_date(v):
    """'1206' -> 2023-12-06, '207' -> 2024-02-07. The file carries no year."""
    v = str(v).strip()
    if v.startswith("12") and len(v) == 4:
        return dt.date(2023, 12, int(v[2:]))
    return dt.date(2024, 2, int(v[1:]))


def load_survey():
    """Ego x alter contacts, with panel, diary date, month and weekday attached."""
    require(CHAE, "Chae microdata (run eda/dl_chae.py)")

    def rd(name):
        t = (CHAE / name).read_bytes().decode("utf-8-sig")
        r = list(csv.reader(io.StringIO(t)))
        return pd.DataFrame(r[1:], columns=[c.strip() for c in r[0]])

    ego = rd("results_preliminary_survey.csv")
    alt = rd("results_main_survey.csv")
    ego = ego[["TYPE", "ID", "SQ1", "SQ2", "SQ3"]].copy()
    ego["ego_age"] = pd.to_numeric(ego.SQ1, errors="coerce")
    ego["seoul"] = ego.SQ3 == "1"
    ego["ego_band"] = ego.ego_age.map(band)
    alt["alter_age"] = pd.to_numeric(alt.Q1, errors="coerce")

    q5 = [c for c in alt.columns if c.startswith("Q5_") and c[3:].isdigit()]
    sel = alt[q5].apply(lambda c: c.str.strip() != "")
    n_place = sel.sum(axis=1)

    def panel_of(row):
        for c in q5:
            v = str(row[c]).strip()
            if v.isdigit() and int(v) in PLACE_PANEL:
                return PLACE_PANEL[int(v)]
        return None

    alt["panel"] = alt[q5].apply(panel_of, axis=1)
    alt["date"] = alt.Date.map(parse_date)
    alt["ym"] = alt.date.map(lambda x: x.year * 100 + x.month)
    alt["dow"] = alt.date.map(lambda x: x.isoweekday())
    out["place_multiplicity"] = dict(
        contacts=int(len(alt)), no_place=int((n_place == 0).sum()),
        multi_place_share=float((n_place > 1).mean()),
        household_plus_other_share=float((sel["Q5_1"] & (n_place > 1)).mean()))

    d = alt.merge(ego[["ID", "ego_age", "ego_band", "seoul"]], on="ID", how="inner")
    d = d[d.alter_age.notna() & d.ego_age.notna()].copy()
    d["alter_band"] = d.alter_age.astype(int).map(band)
    return ego, d


def reproduce_published(ego, d):
    """Section 7a: the descriptor's own summary, before anything is built on it."""
    n_resp, n_contact = len(ego), len(d)
    rate = n_contact / n_resp / DIARY_DAYS
    print(f"=== 7a reproduce the published summary ===")
    print(f"  respondents        {n_resp:,}   (published {PUBLISHED['respondents']:,})")
    print(f"  contacts           {n_contact:,}   (published {PUBLISHED['contacts']:,})")
    print(f"  per person per day {rate:.3f}   (published {PUBLISHED['per_person_day']})")
    assert n_resp == PUBLISHED["respondents"], "respondent count does not reproduce"
    assert n_contact == PUBLISHED["contacts"], "contact count does not reproduce"
    assert abs(rate - PUBLISHED["per_person_day"]) < .005, \
        f"per-person-day {rate:.4f} does not reproduce {PUBLISHED['per_person_day']}"
    days = d.groupby("ID").date.nunique()
    print(f"  diary dates per respondent: median {days.median():.0f}, "
          f"min {days.min()} (days with no contact leave no row)")
    print(f"  every respondent appears in both months: "
          f"{bool((d.groupby('ID').ym.nunique() == 2).all())}")
    out["reproduction"] = dict(respondents=n_resp, contacts=n_contact,
                               per_person_day=round(rate, 4), diary_days=DIARY_DAYS)
    return rate


def ego_cubes(ego, d, seoul_only, ym=None, dows=None):
    """Per-respondent alter-band counts, their own band, and the days covered.

    Kept at respondent level so the bootstrap can resample whole respondents --
    resampling contacts would understate the CI, because contacts within a
    respondent are anything but independent.

    The day count comes from the DESIGN (the diary week intersected with the
    requested weekdays), not from how many dates happen to appear in the data: a
    respondent who reported nothing on a Sunday still spent that Sunday in the
    study, and dividing by observed dates would silently inflate the rate.
    """
    e = ego[ego.ego_age.notna()]
    if seoul_only:
        e = e[e.seoul]
    ids = e.ID.to_numpy()
    pos = {i: k for k, i in enumerate(ids)}
    dd = d[d.ID.isin(pos)]
    if ym is not None:
        dd = dd[dd.ym == ym]
    if dows is not None:
        dd = dd[dd.dow.isin(dows)]
    if ym is None:
        n_days = DIARY_DAYS
    else:
        n_days = len([x for x in DIARY_WEEK[ym]
                      if dows is None or x.isoweekday() in dows])
    ai = {a: k for k, a in enumerate(AGES)}
    counts = np.zeros((len(ids), NA))
    np.add.at(counts, (dd.ID.map(pos).to_numpy(),
                       dd.alter_band.map(ai).to_numpy()), 1.0)
    return counts, e.ego_band.map(ai).to_numpy(), n_days


def contact_matrix(counts, eb, n_days):
    """Mean alters of band a' per respondent of band a per day."""
    tot = np.zeros((NA, NA))
    np.add.at(tot, eb, counts)
    n_ego = np.bincount(eb, minlength=NA).astype(float)
    den = n_ego[:, None] * n_days
    C = np.divide(tot, den, out=np.zeros_like(tot), where=den > 0)
    return C, n_ego


def assortativity(T):
    """Newman assortativity of a symmetric contact-count matrix."""
    e = T / T.sum() if T.sum() else T
    a_ = e.sum(1)
    denom = 1 - (a_ * a_).sum()
    return float((np.trace(e) - (a_ * a_).sum()) / denom) if denom else np.nan


def lead(C):
    """Dominant eigenvalue and normalised eigenvector of a contact matrix."""
    w, v = np.linalg.eig(C)
    k = int(np.argmax(w.real))
    vec = np.abs(v[:, k].real)
    return float(w[k].real), vec / vec.sum() if vec.sum() else vec


def symmetrise(C, pop):
    """T = N_a C[a,a'], symmetrised. Reciprocity holds by construction on the
    passive side and only approximately in a survey, so the asymmetry is
    measured rather than assumed away."""
    T = pop[:, None] * C
    asym = np.abs(T - T.T).sum() / T.sum() if T.sum() else np.nan
    return (T + T.T) / 2, float(asym)


def main():
    ego, d = load_survey()
    reproduce_published(ego, d)
    mp = out["place_multiplicity"]
    print(f"\n  places per contact: {mp['multi_place_share']:.1%} carry more than "
          f"one; {mp['household_plus_other_share']:.1%} are household plus "
          f"something else and so land in H")
    print(f"  panel mix: {d.panel.value_counts().to_dict()}")
    print(f"  Seoul-resident respondents: {int(ego.seoul.sum())} of {len(ego)}")

    mob_path = f"{ROOT}/eda/results_p26.json"
    if not os.path.exists(mob_path):
        print("\n=== stop: run eda/p26_matrix.py first ===")
        return
    mob = json.load(open(mob_path))
    pops = {int(k): np.array(v) for k, v in mob.get("population", {}).items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]      # no 80+ egos exist
    lbl = [AGE_LABEL[AGES[i]] for i in sq]

    # ------------------------------------------------------- survey matrices
    print("\n=== survey side ===")
    # Each month is built twice: on the holiday-free weekdays (primary, matched
    # to the passive side) and on all seven (the sensitivity that shows what
    # 설 연휴 does). The pooled fortnight is kept for the published-summary scale.
    res, cubes = {}, {}
    for ym, key_ym, dows in ([(None, "pooled", None)]
                             + [(y, str(y), holiday_free_dows(y)) for y in SURVEY_MONTHS]
                             + [(y, f"{y}all", None) for y in SURVEY_MONTHS]):
        for scope, so in (("seoul", True), ("national", False)):
            for tag, panels in (("WE", ["W", "E"]), ("W", ["W"]), ("E", ["E"])):
                dd = d[d.panel.isin(panels)]
                cube, eb, nd = ego_cubes(ego, dd, so, ym, dows)
                C, n_ego = contact_matrix(cube, eb, nd)
                key = f"{key_ym}|{tag}|{scope}"
                cubes[key] = (cube, eb, nd)
                res[key] = dict(C=C.tolist(), n_ego=n_ego.tolist(), n_days=int(nd),
                                mean_contacts=float(C.sum(1).mean()))
        if dows is not None:
            print(f"  {ym}: holiday-free weekdays {dows}, "
                  f"{cubes[f'{ym}|WE|seoul'][2]} diary days kept")
    for key in [f"pooled|WE|seoul", f"pooled|WE|national"]:
        print(f"  {key}: mean contacts/person/day (W+E, home excluded) "
              f"{res[key]['mean_contacts']:.2f}")
    out["survey"] = res

    # ------------------------------------------------------ the comparison
    print("\n=== passive vs survey, holiday-free weekdays, dong level, W+E ===")
    rows = []
    for ym in SURVEY_MONTHS:
        for restrict, mkey in (("holiday-free", f"{ym}|WE|dong|holidayfree"),
                               ("all days", f"{ym}|WE|dong")):
            if mkey not in mob["matrices"] or ym not in pops:
                print(f"  missing {mkey} in results_p26.json -- rerun p26")
                continue
            pop = pops[ym]
            A = np.array(mob["matrices"][mkey]["A"])
            Cp = A / pop[:, None]                       # per resident per day
            skey = f"{ym}|WE|seoul" if restrict == "holiday-free" \
                else f"{ym}all|WE|seoul"
            Cs = np.array(res[skey]["C"])
            Tp, asym_p = symmetrise(Cp[np.ix_(sq, sq)], pop[sq])
            Ts, asym_s = symmetrise(Cs[np.ix_(sq, sq)], pop[sq])
            lam_p, vec_p = lead(Cp[np.ix_(sq, sq)])
            lam_s, vec_s = lead(Cs[np.ix_(sq, sq)])
            rows.append(dict(
                ym=ym, restriction=restrict,
                eigvec_pearson=float(np.corrcoef(vec_s, vec_p)[0, 1]),
                eigvec_spearman=float(pd.Series(vec_s).corr(pd.Series(vec_p),
                                                            method="spearman")),
                cell_pearson=float(np.corrcoef(Cs[np.ix_(sq, sq)].ravel(),
                                               Cp[np.ix_(sq, sq)].ravel())[0, 1]),
                assort_passive=assortativity(Tp), assort_survey=assortativity(Ts),
                eig_passive=lam_p, eig_survey=lam_s, eig_ratio=lam_p / lam_s,
                survey_asymmetry=asym_s, passive_asymmetry=asym_p,
                mean_passive=float(Cp[np.ix_(sq, sq)].sum(1).mean()),
                mean_survey=float(Cs[np.ix_(sq, sq)].sum(1).mean()),
                # the advisor asked for the vectors themselves, not just the
                # scalar correlation, so both travel with the row
                eigvec_survey=vec_s.tolist(), eigvec_passive=vec_p.tolist(),
                bands=lbl))
            r = rows[-1]
            print(f"  {ym} {restrict:<12} eigenvector r={r['eigvec_pearson']:+.3f} "
                  f"(spearman {r['eigvec_spearman']:+.3f}) | assortativity "
                  f"{r['assort_passive']:+.4f} vs {r['assort_survey']:+.4f} | "
                  f"NGM eigenvalue ratio {r['eig_ratio']:.2f} | cell Pearson "
                  f"{r['cell_pearson']:+.3f}")
    out["comparison"] = rows
    print("  Cell Pearson is reported only to show how much more forgiving it is "
          "than the eigenvector and assortativity measures.")

    # ------------------------------- how much of the gap can resolution explain
    # The obvious reading of a ten-fold assortativity gap is "dong is too coarse",
    # which is the reviewer attack of Phase 5 arriving from the survey side. p26
    # measured the slope of assortativity in the number of locations, so the
    # question has a number attached: how fine would the partition have to be for
    # co-arrival assortativity to reach the survey's value? If the answer is a
    # cell holding a couple of people, then resolution is not the explanation --
    # the definition of contact is -- and no feasible data release closes it.
    pred = {(x["ym"], x["panel"]): x for x in mob.get("resolution_prediction", [])}
    gap = []
    for r in [x for x in rows if x["restriction"] == "holiday-free"]:
        pr = pred.get((r["ym"], "WE"))
        if not pr:
            continue
        need = r["assort_survey"] / r["assort_passive"]
        n_loc = 424 * need ** (1 / pr["slope"])
        gap.append(dict(ym=r["ym"], ratio=need, slope=pr["slope"],
                        r_polygon=pr["r_pred_polygon"],
                        polygon_closes=pr["r_pred_polygon"] / r["assort_survey"],
                        n_loc_needed=n_loc, persons_per_loc=9.6e6 / n_loc))
    if gap:
        print("\n=== could resolution explain the gap? ===")
        for g in gap:
            print(f"  {g['ym']}: survey is {g['ratio']:.1f}x the dong-level value. "
                  f"The 1,831-polygon prediction reaches "
                  f"{g['polygon_closes']:.1%} of it. Matching it on the measured "
                  f"slope ({g['slope']:.3f}) would need {g['n_loc_needed']:,.0f} "
                  f"locations -- about {g['persons_per_loc']:.1f} residents each.")
        print("  A cell of a few people is not a coarser or finer neighbourhood, "
              "it is an individual encounter. So the gap is a difference in what "
              "is being measured, not a resolution deficit, and the Phase 5 "
              "refinement cannot be expected to close it.")
        out["resolution_gap"] = gap

    # -------------------------------------- bootstrap CI on the survey side
    if rows:
        print(f"\n=== survey assortativity, {BOOT} respondent-level bootstraps ===")
        rng = np.random.default_rng(20260818)
        boot = {}
        for ym in SURVEY_MONTHS:
            cube, eb, nd = cubes[f"{ym}|WE|seoul"]
            pop = pops[ym]
            draws = []
            for _ in range(BOOT):
                idx = rng.integers(0, len(eb), len(eb))
                C, n_ego = contact_matrix(cube[idx], eb[idx], nd)
                T, _ = symmetrise(C[np.ix_(sq, sq)], pop[sq])
                draws.append(assortativity(T))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            boot[str(ym)] = dict(lo=float(lo), hi=float(hi),
                                 median=float(np.median(draws)))
            point = [r for r in rows if r["ym"] == ym
                     and r["restriction"] == "holiday-free"][0]["assort_survey"]
            print(f"  {ym}: {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"(n={int(ego.seoul.sum())} Seoul respondents)")
        out["survey_assortativity_bootstrap"] = boot

        print("\n=== who the dominant eigenvector lands on (top 3 bands) ===")
        for ym in SURVEY_MONTHS:
            _, vs = lead(np.array(res[f"{ym}|WE|seoul"]["C"])[np.ix_(sq, sq)])
            k = f"{ym}|WE|dong|holidayfree"
            _, vp = lead((np.array(mob["matrices"][k]["A"])
                          / pops[ym][:, None])[np.ix_(sq, sq)])
            for nm, v in (("survey ", vs), ("passive", vp)):
                top = np.argsort(v)[::-1][:3]
                print(f"  {ym} {nm}: " + ", ".join(
                    f"{lbl[i]} {v[i]:.3f}" for i in top))

        # -------------------------------- passive masking band, for symmetry
        bandrows = [b for b in mob.get("imputation_band", [])
                    if b["ym"] in SURVEY_MONTHS and b["panel"] == "WE"]
        if bandrows:
            print("\n=== passive assortativity, masking band (0 / 1.5 / 3) ===")
            for ym in SURVEY_MONTHS:
                mid = mob["matrices"].get(f"{ym}|WE|dong", {}).get("assortativity")
                b = {x["imp"]: x["assortativity"] for x in bandrows if x["ym"] == ym}
                if mid is not None and b:
                    print(f"  {ym}: {b.get('lo', float('nan')):+.4f} .. "
                          f"{mid:+.4f} .. {b.get('hi', float('nan')):+.4f}")

    # ------------------------------------------------------------- figures
    if rows:
        ym0 = SURVEY_MONTHS[0]
        Cs = np.array(res[f"{ym0}|WE|seoul"]["C"])[np.ix_(sq, sq)]
        Cp = (np.array(mob["matrices"][f"{ym0}|WE|dong|holidayfree"]["A"])
              / pops[ym0][:, None])[np.ix_(sq, sq)]
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        for ax, (M, ttl) in zip(axes[:2], [
                (Cs / Cs.sum(1, keepdims=True), f"survey (Chae et al.), {ym0}"),
                (Cp / Cp.sum(1, keepdims=True), f"passive co-arrival, {ym0}")]):
            im = ax.imshow(M, cmap="magma", vmin=0, vmax=max(M.max(), 1e-9))
            ax.set_xticks(range(len(sq))); ax.set_yticks(range(len(sq)))
            ax.set_xticklabels(lbl, rotation=90, fontsize=6)
            ax.set_yticklabels(lbl, fontsize=6)
            ax.set_title(ttl + "\nrow-normalised", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=.046)
        ax = axes[2]
        ax.scatter(Cs.ravel(), Cp.ravel(), s=9, alpha=.5, color="#4C6E8A")
        ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("survey contacts per person per day")
        ax.set_ylabel("passive co-arrivals per resident per day")
        r0 = [x for x in rows if x["ym"] == ym0 and x["restriction"] == "holiday-free"][0]
        ax.set_title(f"cell by cell (Pearson {r0['cell_pearson']:+.3f})", fontsize=10)
        ax.grid(alpha=.3)
        fig.suptitle("Phase 7 — passive co-arrival against the survey contact "
                     "matrix, holiday-free weekdays", fontsize=11)
        fig.tight_layout(); fig.savefig(f"{FIG}/p27_matrix_compare.png", dpi=150)
        plt.close(fig)
        print("\n  -> fig/p27_matrix_compare.png")

        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        x = range(len(sq))
        for ym, c in zip(SURVEY_MONTHS, ["#A8434E", "#BC8034"]):
            k = f"{ym}|WE|dong|holidayfree"
            if k not in mob["matrices"]:
                continue
            _, vp = lead((np.array(mob["matrices"][k]["A"])
                          / pops[ym][:, None])[np.ix_(sq, sq)])
            ax.plot(x, vp, "-o", ms=3.5, color=c, label=f"passive {ym}")
        for ym, c in zip(SURVEY_MONTHS, ["#0E7C86", "#4C6E8A"]):
            _, vs = lead(np.array(res[f"{ym}|WE|seoul"]["C"])[np.ix_(sq, sq)])
            ax.plot(x, vs, "--s", ms=3.5, color=c, label=f"survey {ym}")
        ax.set_xticks(list(x)); ax.set_xticklabels(lbl, rotation=45, ha="right",
                                                   fontsize=8)
        ax.set_ylabel("dominant eigenvector (normalised)")
        ax.set_title("who the next generation lands on: passive vs survey",
                     fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{FIG}/p27_eigenvector.png", dpi=150)
        plt.close(fig)
        print("  -> fig/p27_eigenvector.png")

    with open(f"{ROOT}/eda/results_p27.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote results_p27.json")


if __name__ == "__main__":
    main()
