#!/usr/bin/env python
"""Phase 44 — beta, measured: how much age segregation sits below the cell line.

`p39` left the paper's main number conditional. The estimator reads
`r_between = (1 - beta) * r_true`, where beta is the share of true age
segregation sitting below the (dong x hour x day-of-week) line, and beta is not
identified by the mobility data: one equation, two unknowns. The family of upper
bounds indexed by beta is therefore not a bound at all until beta is pinned from
outside, and the one empirical route we had -- measuring the same quantity at
1,831 traffic polygons -- is closed by the Big Data Campus eligibility rule.

This phase pins it from a source already on disk and already md5-verified: Chae
et al.'s national contact survey. The survey records, for every one of its
133,776 contacts, WHERE it happened (Q5, eight place codes) and HOW BIG the
gathering was (Q9, the number of people present including the respondent). That
is enough to run the mobility estimator one level down, on real contact data:

    A[a,a'] = sum_g n_a(g) n_a'(g) / n_.(g)          the SAME functional form,
                                                     with g = a venue-visit
                                                     instead of a dong-hour cell

and then to coarsen g by pooling venue-visits and watch the reading fall. The
decay law that comes out is measured over four decades of group size, on real
contacts, and it says what fraction of venue-level segregation survives at any
cell size -- which is exactly beta.

WHY THE PAIRING RULE HAD TO CHANGE, and it is not cosmetic. `n_a n_a' / n_.` is
sampling WITH replacement: it pairs every person with themselves. At n = 4,000
that is a 1/n rounding error and `p26` is right to ignore it. At a venue-visit of
4 people it puts a sixth of the mass on the diagonal, and since respondents
differ in age band, that manufactures assortativity out of nothing. The
with-replacement venue reading is 0.436; the without-replacement reading
(n_a n_a' - delta_aa' n_a) / (n_. - 1), which has the same total mass n_. and is
the finite-population estimand the large-n form approximates, is 0.249. The
permutation null in 44.1 shows which of those two is the artefact: shuffling
alter ages within place type leaves the with-replacement number at 0.271 (62% of
it survives a world with no within-venue sorting at all) and collapses the
without-replacement number to 0.034. Everything below is without replacement,
and both are reported.

WHAT IS NOT CIRCULAR HERE, and what is. Pinning beta needs a truth measured
outside the mobility data, and the survey is that truth under exactly the
assumption that already licenses the passive-vs-survey comparison in the paper --
no new one. What this phase cannot do is settle whether a national contact survey
describes Seoul co-arrivals; that is the open scope question, and both scopes are
reported side by side.

PRE-DECLARED BEFORE THE RUN, because the project has already been burned by
choosing after: the primary arm is the NATIONAL sample, both diary weeks pooled,
W+E panel, without replacement, venue size winsorised at its 99th percentile.
National for the shape (beta is a shape, and 1,987 respondents give four times
the venue count); pooled for the same reason; W+E because the passive matrix
drops household destinations. The Seoul / 202312 arm is the anchor, because that
is the cell `p32`'s 0.2227 and `p33`'s 0.01677 were measured in.

    python eda/p44_beta.py
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
import calendar_kr as K
from common import AGE_LABEL, AGES
from p26_matrix import cells, holiday_free_dows, matrices
from p27_survey import (assortativity, contact_matrix, ego_cubes, load_survey,
                        symmetrise)
from paths import FIG, ROOT

SEED = 20260822
YM = 202312                        # the anchor month
NA = len(AGES)
SQ = [i for i, a in enumerate(AGES) if a < 80]     # no 80+ egos exist
IMP_PUB = 1.5                      # p26's published imputation for a masked row

# Read back, not retyped. Every one of these is asserted against its source file
# in 44.0 and the run aborts on a mismatch.
R_SURVEY = 0.22270993081818996             # p32, survey|202312|WE|seoul, <80
R_OBS_PUB = 0.02115385938228908            # p33 R1, passive dong WE holidayfree
P33_NOISE_BIAS = 0.0043860558176699906     # its parametric-bootstrap device bias
R_OBS_CORRECTED = 0.01676780356461909      # published minus bias
P39_BETA_AT_SURVEY = 0.96                  # p39's interpolated crossing

PLACE_LABEL = {1: "household", 2: "workplace", 3: "educational", 4: "religious",
               5: "restaurant/cafe/bar", 6: "hospital", 7: "outdoor", 8: "other"}
POOL_M = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 1000, 1600,
          2600, 4400, 7000)
REPS = 40                          # pooling replicates per m (12 above m=1000)
MIN_CELLS = 30                     # below this a between-cell variance is noise
BOOT = 300                         # respondent-level bootstrap replicates
WINSOR = 99.0                      # primary venue-size winsorisation percentile
WINSOR_SWEEP = (90.0, 95.0, 99.0, 99.9, 100.0)
# 44.7's replacement for the withdrawn Q9 padding arm: venue size as a
# declared multiple of what the respondent recorded. Declared here, before
# the run, because choosing the grid after seeing beta would be shopping.
K_VENUE = (1.25, 1.5, 2.0, 3.0, 5.0)

out = {}


# --------------------------------------------------------------------- helpers
def r_of(A):
    """Newman assortativity on the 0-79 block, the published convention."""
    return assortativity(A[np.ix_(SQ, SQ)])


def collapse(U, wg, replacement=False):
    """The p26 estimator over arbitrary groups.

    U[g, a] is the HEADCOUNT of band a in group g -- counts, not weights, because
    the without-replacement correction is a statement about people. wg[g] is the
    group's post-stratification weight, which multiplies its whole contribution;
    the operator is homogeneous of degree one in n, so a weight and a group-size
    multiplier are the same thing and the composition is untouched.

        with replacement     sum_g w_g n_a n_a' / n_.
        without replacement  sum_g w_g (n_a n_a' - delta_aa' n_a) / (n_. - 1)

    Both have total mass sum_g w_g n_.(g), so they are the same object at
    different pairing conventions rather than different normalisations.
    """
    tot = U.sum(1)
    keep = tot > (1.0 + 1e-9) if not replacement else tot > 1e-12
    Uk, tk, wk = U[keep], tot[keep], wg[keep]
    den = (tk - 1.0) if not replacement else tk
    S = Uk * (wk / den)[:, None]
    A = S.T @ Uk
    if not replacement:
        A = A - np.diag(S.sum(0))
    return A


def n_eff(sizes):
    """Effective number of independent groups in a pool: (sum s)^2 / sum s^2.

    This, not the raw count, is what the composition variance of a pooled group
    scales with, and it is what makes the decay law parameter-free.
    """
    s = np.asarray(sizes, float)
    ss = (s * s).sum()
    return float((s.sum() ** 2) / ss) if ss > 0 else 0.0


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------- survey assembly
def build(ego, d, scope, ym, panels=("W", "E")):
    """Venue-visit table for one scope/month: headcounts, weights, sizes.

    A venue-visit g = (respondent, diary date, place code) -- one respondent's
    stay at one place on one day. Its members are the respondent plus the alters
    they recorded there, so the composition is a SAMPLE of the venue, taken by
    the respondent. 44.6 reports which way that bias runs.
    """
    e = ego[ego.ego_age.notna()]
    if scope == "seoul":
        e = e[e.seoul]
    dd = d[d.ID.isin(set(e.ID)) & d.panel.isin(list(panels))]
    if ym is not None:
        dd = dd[(dd.ym == ym) & (dd.dow.isin(holiday_free_dows(ym)))]
    dd = dd.copy()

    ai = {a: k for k, a in enumerate(AGES)}
    nego = np.bincount(e.ego_band.map(ai).to_numpy(), minlength=NA).astype(float)
    # Post-stratification to Seoul's registered population, band by band. This is
    # algebraically p27's own convention: pop_a * (contacts per ego of band a),
    # which is what makes 44.0's third anchor an identity rather than a check.
    w = np.divide(POP, nego, out=np.zeros_like(POP), where=nego > 0)
    dd["wt"] = w[dd.ego_band.map(ai).to_numpy()]
    dd["ebi"] = dd.ego_band.map(ai)
    dd["abi"] = dd.alter_band.map(ai)
    dd["q9"] = pd.to_numeric(dd.Q9, errors="coerce")

    gid, uniq = pd.factorize(pd.MultiIndex.from_frame(dd[["ID", "date", "place"]]))
    G = len(uniq)
    egos = dd.drop_duplicates(subset=["ID", "date", "place"])
    gid_ego = gid[dd.index.get_indexer(egos.index)]

    U = np.zeros((G, NA))
    np.add.at(U, (gid, dd.abi.to_numpy()), 1.0)
    np.add.at(U, (gid_ego, egos.ebi.to_numpy()), 1.0)      # the respondent
    wg = np.zeros(G)
    wg[gid_ego] = egos.wt.to_numpy()
    place = np.zeros(G, int)
    place[gid_ego] = egos.place.to_numpy()
    who = np.empty(G, object)
    who[gid_ego] = egos.ID.to_numpy()
    q9 = dd.groupby(gid).q9.max().reindex(range(G)).to_numpy()
    return dict(e=e, dd=dd, egos=egos, gid=gid, gid_ego=gid_ego, G=G, U=U,
                wg=wg, place=place, who=who, q9=q9, obs=U.sum(1))


def star(dd):
    """The survey's own contact matrix under the same weights: p27's T."""
    P = np.zeros((NA, NA))
    np.add.at(P, (dd.ebi.to_numpy(), dd.abi.to_numpy()), dd.wt.to_numpy())
    return (P + P.T) / 2


# ------------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    global POP
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p33 = json.load(open(f"{ROOT}/eda/results_p33.json"))
    p39 = json.load(open(f"{ROOT}/eda/results_p39.json"))
    p38 = json.load(open(f"{ROOT}/eda/results_p38.json"))
    POP = np.array(p26["population"][str(YM)], float)

    # ===================================================== 44.0 five anchors
    say("=== 44.0 anchors: the same data, the same estimator, the same numbers ===")
    ego, d = load_survey()
    q5 = [c for c in d.columns if c.startswith("Q5_") and c[3:].isdigit()]

    def place_of(row):
        for c in q5:
            v = str(row[c]).strip()
            if v.isdigit():
                return int(v)
        return 0
    d = d.assign(place=d[q5].apply(place_of, axis=1))
    assert (d.place > 0).all(), "a contact with no place code"

    a1 = dict(respondents=len(ego), contacts=len(d))
    assert a1 == dict(respondents=1987, contacts=133776), a1
    say(f"  A1 published summary      {a1['respondents']:,} respondents, "
        f"{a1['contacts']:,} contacts")

    dows = holiday_free_dows(YM)
    dd = d[d.panel.isin(["W", "E"])]
    cube, eb, nd = ego_cubes(ego, dd, True, YM, dows)
    C, _ = contact_matrix(cube, eb, nd)
    T, _ = symmetrise(C[np.ix_(SQ, SQ)], POP[SQ])
    a2 = assortativity(T)
    assert abs(a2 - R_SURVEY) < 1e-15, f"p27 path gives {a2!r}, not {R_SURVEY!r}"
    say(f"  A2 p27's own path         r = {a2!r}")

    anchor = build(ego, d, "seoul", YM)
    a3 = r_of(star(anchor["dd"]))
    say(f"  A3 weighted-star route    r = {a3!r}   |delta| = {abs(a3 - R_SURVEY):.2e}")
    assert abs(a3 - R_SURVEY) < 1e-12, "the weighting is not p27's convention"

    # A4: the collapse operator IS p26's, checked against the published matrix.
    cl = cells(YM)
    A_pub = np.array(p26["matrices"][f"{YM}|WE|dong|holidayfree"]["A"])
    A_mine = matrices(cl, YM, "WE", "dong", imp=IMP_PUB, dows=dows)
    a4 = float(np.abs(A_pub - A_mine).max() / np.abs(A_pub).max())
    say(f"  A4 p26 matrix rebuilt     max relative |delta| = {a4:.2e}")
    assert a4 < 1e-12, "the mobility estimator does not reproduce p26"

    a5 = dict(r_survey=R_SURVEY, r_obs_published=R_OBS_PUB,
              noise_bias=P33_NOISE_BIAS, r_obs_corrected=R_OBS_CORRECTED)
    pc = p39["published_constants"]
    for k, v in (("r_obs_pub", R_OBS_PUB), ("noise_bias", P33_NOISE_BIAS),
                 ("r_obs_corrected", R_OBS_CORRECTED), ("survey", R_SURVEY)):
        assert pc[k] == v, f"p39 carries {k} = {pc[k]!r}, this file has {v!r}"
    assert abs(R_OBS_PUB - P33_NOISE_BIAS - R_OBS_CORRECTED) < 1e-15
    say(f"  A5 p39 constants, 4/4 bit-equal   passive {R_OBS_PUB:.5f} - noise "
        f"{P33_NOISE_BIAS:.5f} = {R_OBS_CORRECTED:.5f}")

    # p39's crossing, interpolated from its own stored bounds rather than quoted.
    inv = sorted((float(b), v["bound_corrected"]) for b, v in p39["inversion"].items())
    bs = [b for b, _ in inv]
    bd = [x for _, x in inv]
    crossing = float(np.interp(R_SURVEY, bd, bs))
    assert abs(crossing - P39_BETA_AT_SURVEY) < 0.01, f"crossing moved: {crossing}"
    say(f"  A5b p39's bound reaches the survey value at beta = {crossing:.4f}"
        f"   (the letter's '~{P39_BETA_AT_SURVEY:.2f}')")
    out["anchors"] = dict(a1, survey_r_p27=a2, survey_r_weighted=a3,
                          p26_rebuild_maxrel=a4, constants=a5,
                          p39_crossing_interpolated=crossing)

    # ============================================ 44.1 the pairing rule, tested
    say("\n=== 44.1 with or without replacement, and the null that decides it ===")
    P = build(ego, d, "national", None)
    say(f"  national W+E: {P['G']:,} venue-visits from {len(P['e']):,} respondents, "
        f"{len(P['dd']):,} contacts")
    r_star = r_of(star(P["dd"]))
    r_wr = r_of(collapse(P["U"], P["wg"], replacement=True))
    r_wor = r_of(collapse(P["U"], P["wg"]))
    say(f"  survey's own contact matrix (star pairs)   r = {r_star:.5f}")
    say(f"  venue-visit collapse, with replacement     r = {r_wr:.5f}"
        f"   ({r_wr / r_star:.2f}x the survey's own)")
    say(f"  venue-visit collapse, without replacement  r = {r_wor:.5f}"
        f"   ({r_wor / r_star:.2f}x the survey's own)")

    # Null 1: everyone in one group. Both readings have a closed form and the run
    # aborts if either misses it. WR gives exactly zero -- proportionate mixing
    # against its own margins. WOR gives exactly -1/(n-1): with the self-pairs
    # removed, a single finite group is slightly DISassortative relative to its
    # own margins, because nobody can be their own contact. That offset is the
    # whole difference between the two conventions, and it is why WR manufactures
    # assortativity in a four-person venue while WOR does not.
    # The closed form is a statement about the whole matrix, so it is asserted on
    # all 16 bands; the 0-79 block used everywhere else is reported beside it and
    # differs only because dropping the 80+ alters drops n.
    one_U, one_w = P["U"].sum(0, keepdims=True), np.array([P["wg"].mean()])
    n_one = float(one_U.sum())
    n_blk = float(one_U[:, SQ].sum())
    n1_wr = float(assortativity(collapse(one_U, one_w, replacement=True)))
    n1_wor = float(assortativity(collapse(one_U, one_w)))
    say(f"  N1 everyone in one group ({n_one:,.0f} people)  r = {n1_wr:.2e} (WR, "
        f"exactly 0), {n1_wor:.4e} (WOR, exactly -1/(n-1) = {-1 / (n_one - 1):.4e})")
    say(f"     on the 0-79 block: {r_of(collapse(one_U, one_w)):.4e}, which is "
        f"-1/(n-1) for the {n_blk:,.0f} people it keeps ({-1 / (n_blk - 1):.4e})")
    assert abs(n1_wr) < 1e-12, "WR null is not zero"
    assert abs(n1_wor + 1 / (n_one - 1)) < 1e-12, "WOR null misses -1/(n-1)"
    assert abs(r_of(collapse(one_U, one_w)) + 1 / (n_blk - 1)) < 1e-12

    # Null 2: shuffle alter bands within place type. Destroys within-venue
    # sorting; keeps every place type's age margin and every ego's own band.
    nulls = {"wr": [], "wor": []}
    for _ in range(20):
        ab = P["dd"].groupby("place").abi.transform(
            lambda s: rng.permutation(s.to_numpy())).to_numpy()
        U2 = np.zeros_like(P["U"])
        np.add.at(U2, (P["gid"], ab), 1.0)
        np.add.at(U2, (P["gid_ego"], P["egos"].ebi.to_numpy()), 1.0)
        nulls["wr"].append(r_of(collapse(U2, P["wg"], replacement=True)))
        nulls["wor"].append(r_of(collapse(U2, P["wg"])))
    n2_wr, n2_wor = float(np.median(nulls["wr"])), float(np.median(nulls["wor"]))
    say(f"  N2 alter ages shuffled within place type   r = {n2_wr:.5f} (WR, "
        f"{n2_wr / r_wr:.0%} survives), {n2_wor:.5f} (WOR, {n2_wor / r_wor:.0%})")
    say("     WR keeps most of its reading in a world with no within-venue")
    say("     sorting at all, so WR is the artefact. WOR is primary from here.")
    out["pairing"] = dict(r_star=r_star, r_venue_wr=r_wr, r_venue_wor=r_wor,
                          null_one_group=dict(wr=n1_wr, wor=n1_wor),
                          null_shuffled=dict(wr=n2_wr, wor=n2_wor,
                                             share_wr=n2_wr / r_wr,
                                             share_wor=n2_wor / r_wor))

    # ==================================================== 44.2 what beta sees
    say("\n=== 44.2 the observable ladder: coarsen the group, watch it go ===")
    ladders = {}
    for scope, ym in (("national", None), ("national", YM),
                      ("seoul", None), ("seoul", YM)):
        B = build(ego, d, scope, ym)
        rs = r_of(star(B["dd"]))
        rows = []
        for name, keys in (("venue-visit (ego,date,place)", ["ID", "date", "place"]),
                           ("ego-day (ego,date)", ["ID", "date"]),
                           ("place x date x region", ["place", "date", "region"]),
                           ("place x date", ["place", "date"]),
                           ("region x date", ["region", "date"]),
                           ("date", ["date"])):
            dd2 = B["dd"]
            if "region" in keys:
                reg = dict(zip(B["e"].ID, B["e"].SQ3))
                dd2 = dd2.assign(region=dd2.ID.map(reg))
            g2, u2 = pd.factorize(pd.MultiIndex.from_frame(dd2[keys]))
            eg = dd2.drop_duplicates(subset=keys + ["ID"])
            ge = g2[dd2.index.get_indexer(eg.index)]
            U2 = np.zeros((len(u2), NA))
            np.add.at(U2, (g2, dd2.abi.to_numpy()), 1.0)
            np.add.at(U2, (ge, eg.ebi.to_numpy()), 1.0)
            w2 = np.zeros(len(u2))
            np.add.at(w2, ge, eg.wt.to_numpy())
            cnt = np.zeros(len(u2))
            np.add.at(cnt, ge, 1.0)
            w2 = np.divide(w2, cnt, out=np.zeros_like(w2), where=cnt > 0)
            r2 = r_of(collapse(U2, w2))
            rows.append(dict(grouping=name, n_groups=int(len(u2)),
                             mean_persons=float(U2.sum(1).mean()),
                             r=r2, keep=r2 / rs))
        key = f"{scope}|{ym or 'pooled'}"
        ladders[key] = dict(r_star=rs, rows=rows)
        if key in ("national|pooled", f"seoul|{YM}"):
            say(f"  {key}   survey's own contact matrix r = {rs:.5f}")
            for r in rows:
                say(f"    {r['grouping']:<30}{r['n_groups']:>8,} groups"
                    f"{r['mean_persons']:>9.1f} ppl   r={r['r']:.5f}"
                    f"   keep={r['keep']:6.1%}")
    out["ladder"] = ladders

    # ======================================= 44.3 the decay law, four decades
    say("\n=== 44.3 pooling venue-visits: the decay law ===")
    U, wg, G, place = P["U"], P["wg"], P["G"], P["place"]
    obs = P["obs"]
    r1 = r_wor
    curves = {}
    for arm in ("random", "within-place"):
        rows = []
        for m in POOL_M:
            vals, neffs, ncell = [], [], []
            reps = REPS if m <= 1000 else 12
            for _ in range(reps):
                if arm == "random":
                    perm = rng.permutation(G)
                    nc = G // m
                    if nc < 1:
                        break
                    idx = perm[:nc * m].reshape(nc, m)
                    Up, wp, sz = U[idx].sum(1), wg[idx].mean(1), obs[idx]
                else:
                    Us, ws, szs = [], [], []
                    for pl in np.unique(place):
                        ix = np.where(place == pl)[0]
                        rng.shuffle(ix)
                        nc = len(ix) // m
                        if nc < 1:
                            continue
                        j = ix[:nc * m].reshape(nc, m)
                        Us.append(U[j].sum(1))
                        ws.append(wg[j].mean(1))
                        szs.append(obs[j])
                    if not Us:
                        break
                    Up, wp, sz = np.vstack(Us), np.concatenate(ws), np.vstack(szs)
                vals.append(r_of(collapse(Up, wp)))
                neffs.append(float(np.mean([n_eff(s) for s in sz])))
                ncell.append(len(Up))
            if not vals:
                continue
            v = np.array(vals)
            C = float(np.mean(ncell))
            # A between-group variance read off C groups is short by (1 - 1/C):
            # the grand mean is estimated from the same C groups. At C = 3 that
            # is a third of the signal, which is exactly where the raw curve
            # bends away below. Corrected here, and both are stored.
            keep = float(np.median(v)) / r1
            rows.append(dict(m=m, n_eff=float(np.mean(neffs)), n_cells=C,
                             r=float(np.median(v)), keep=keep,
                             keep_corrected=keep / (1 - 1 / C) if C > 1 else keep,
                             lo=float(np.percentile(v, 2.5)),
                             hi=float(np.percentile(v, 97.5)), reps=len(vals)))
        curves[arm] = rows
        say(f"  {arm} pooling (r at m=1 is {r1:.5f}):")
        say(f"    {'m':>6}{'cells':>8}{'n_eff':>8}{'r':>10}{'keep':>9}"
            f"{'x n_eff':>9}{'corrected':>11}")
        for r in rows:
            flag = "" if r["n_cells"] >= MIN_CELLS else "   (too few cells)"
            say(f"    {r['m']:>6}{r['n_cells']:>8.0f}{r['n_eff']:>8.2f}"
                f"{r['r']:>10.5f}{r['keep']:>9.4f}"
                f"{r['keep'] * r['n_eff']:>9.3f}"
                f"{r['keep_corrected'] * r['n_eff']:>11.3f}{flag}")
    fit = [r for r in curves["random"] if r["m"] >= 8 and r["n_cells"] >= MIN_CELLS]
    c_law = float(np.median([r["keep_corrected"] * r["n_eff"] for r in fit]))
    say(f"  random arm: keep x n_eff is flat at {c_law:.3f} over "
        f"m = 8..{fit[-1]['m']} (the {len(curves['random']) - len(fit) - 1} "
        f"points with fewer than {MIN_CELLS} cells are excluded from the fit,")
    say(f"     not from the table). The law is keep = {c_law:.3f} / n_eff, and "
        "1.000 is what independent groups predict.")
    plateau = [r for r in curves["within-place"]
               if r["m"] >= 55 and r["n_cells"] >= MIN_CELLS]
    floor = float(np.median([r["r"] for r in plateau]))
    say(f"  within-place arm SATURATES at r = {floor:.5f} over m = "
        f"{plateau[0]['m']}..{plateau[-1]['m']}, and 44.1's shuffled null sits")
    say(f"     at {n2_wor:.5f} -- the same floor within {abs(floor / n2_wor - 1):.0%}, "
        "reached two ways. Pooling inside a place")
    say("     type can never destroy the contrast BETWEEN place types, so this is")
    say("     the ceiling for any partition that separates the eight settings.")
    out["pooling"] = dict(curves=curves, law_constant=c_law,
                          min_cells_for_fit=MIN_CELLS,
                          within_place_floor=floor,
                          floor_vs_shuffled_null=float(floor / n2_wor))

    # ================================== 44.4 where the dong-hour cell actually is
    say("\n=== 44.4 the cell, measured (and the letter's 20,000 is wrong) ===")
    nd_map = {dw: K.cell_exposure(YM, dw)["n_days"] for dw in dows}
    cm = cl[(cl.dest_attr != "H") & (cl.dow_n.isin(dows))].copy()
    cm["n"] = (cm.v_obs + IMP_PUB * cm.n_masked) / cm.dow_n.map(nd_map)
    per_cell = cm.groupby(["d_dong", "arr_hour", "dow_n"], as_index=False)["n"].sum()
    per_dongday = cm.groupby(["d_dong", "dow_n"], as_index=False)["n"].sum()
    nl = per_cell["n"].to_numpy()
    N_day = float(nl.sum()) / len(dows)
    L_day = len(nl) / len(dows)
    cell = dict(n_cells_total=int(len(nl)), n_dows=len(dows),
                cells_per_day=L_day, arrivals_per_day=N_day,
                median=float(np.median(nl)), mean=float(nl.mean()),
                n_weighted_mean=float((nl * nl).sum() / nl.sum()),
                dong_day_median=float(per_dongday["n"].median()))
    say(f"  dong x hour x dow cell: median {cell['median']:,.0f} arrivals, "
        f"mean {cell['mean']:,.0f}, arrival-weighted mean "
        f"{cell['n_weighted_mean']:,.0f}")
    say(f"  dong x day (no hour):   median {cell['dong_day_median']:,.0f}"
        "   <- this is the ~20,000 the letter quotes; the letter attributes it")
    say("     to the dong x HOUR cell, which is 34x smaller. Correct the letter.")

    q9 = P["q9"]
    sizes = {}
    for p in WINSOR_SWEEP:
        cap = np.percentile(q9, p)
        s = np.minimum(q9, cap)
        sizes[p] = dict(cap=float(cap), mean=float(s.mean()),
                        size_weighted=float((s * s).sum() / s.sum()))
    sw = sizes[WINSOR]["size_weighted"]
    say(f"  venue size from Q9 (people present, respondent included):")
    for p in WINSOR_SWEEP:
        z = sizes[p]
        say(f"    winsorised at p{p:<5} cap {z['cap']:>7.0f}   mean "
            f"{z['mean']:>6.2f}   size-weighted E[s^2]/E[s] {z['size_weighted']:>8.2f}")
    say(f"  observed venue-visit (respondent + recorded alters): mean "
        f"{obs.mean():.2f}, size-weighted {(obs * obs).sum() / obs.sum():.2f}")

    # The prediction. r is the arrival-weighted variance of cell composition over
    # (1 - sum p^2), so pooling n_eff independent venues into cell l gives
    # keep_l = c / n_eff(l) with n_eff(l) = n_l / (E[s^2]/E[s]); the arrival
    # weights then collapse the whole sum to c * E[s^2]/E[s] * L / N.
    pred = {}
    for p in WINSOR_SWEEP:
        s_eff = sizes[p]["size_weighted"]
        keep_pred = c_law * s_eff * L_day / N_day
        pred[p] = dict(size_eff=s_eff, keep=keep_pred, r=keep_pred * r1,
                       measured_over_predicted=R_OBS_CORRECTED / (keep_pred * r1))
    pr = pred[WINSOR]
    say(f"\n  predicted passive reading, independent venues, winsor p{WINSOR}:")
    say(f"    n_eff per cell = n_l / {sw:.2f};  sum_l w_l / n_eff(l) = "
        f"{sw * L_day / N_day:.5f}")
    say(f"    keep = {c_law:.3f} x {sw:.2f} x {L_day:,.0f} / {N_day:,.0f} = "
        f"{pr['keep']:.5f}   ->   r = {pr['r']:.5f}")
    say(f"    measured (noise-corrected)                              "
        f"r = {R_OBS_CORRECTED:.5f}")
    say(f"    the real cell reads {pr['measured_over_predicted']:.2f}x a random "
        "pool of the same number of real venues.")
    say(f"    winsorisation sweep: "
        + ", ".join(f"p{p:g} {pred[p]['measured_over_predicted']:.2f}x"
                    for p in WINSOR_SWEEP))
    # 44.4b The same shortfall, from the mobility data alone. Random pooling says
    # r scales with the NUMBER of cells, so dong (424) -> gu (25) should cost a
    # factor of 424/25 = 17. p37 measured what it actually costs, on 79 months
    # with no exceptions, and it is far less: real geography merges correlated
    # places. The excess that shows up there is the same quantity as the 6.8x
    # above, reached without the survey and without any venue-size mapping.
    kept = float(p39["published_constants"]["kept_median"])
    kept_lo = float(p39["published_constants"]["kept_min"])
    kept_hi = float(p39["published_constants"]["kept_max"])
    rand_predict = 25 / 424
    clus_gu = kept / rand_predict
    clus_cell = pr["measured_over_predicted"]
    say(f"\n  44.4b the same shortfall, measured inside the mobility data:")
    say(f"    dong -> gu, random pooling would keep {rand_predict:.4f} of the "
        f"excess (25/424)")
    say(f"    p37 measures {kept:.4f} kept ({kept_lo:.4f}-{kept_hi:.4f} over 79 "
        f"months, no exceptions)")
    say(f"    so real merging retains {clus_gu:.2f}x what random pooling predicts,")
    say(f"    against {clus_cell:.2f}x for the cell-vs-venue comparison above --")
    say(f"    two routes to the spatial-clustering excess, agreeing within "
        f"{abs(clus_cell / clus_gu - 1):.0%}. The first of them")
    say("    never touches the survey or the venue-size mapping, so it is the")
    say("    one to quote if a reviewer attacks the Q9 winsorisation.")
    out["clustering"] = dict(kept_dong_to_gu=kept, kept_range=[kept_lo, kept_hi],
                             random_prediction=rand_predict,
                             excess_from_gu=clus_gu, excess_from_venues=clus_cell,
                             disagreement=abs(clus_cell / clus_gu - 1))

    out["cell"] = dict(cell, venue_sizes=sizes, prediction=pred,
                       observed_venue_mean=float(obs.mean()),
                       observed_venue_size_weighted=float((obs * obs).sum() / obs.sum()))

    # ================================================ 44.5 beta, and what it costs
    say("\n=== 44.5 the passive reading, bracketed by two survey-derived predictions ===")
    # The bracket is the non-circular part of this phase. Neither end uses the
    # passive matrix: both are computed from the survey alone, and the passive
    # reading is then dropped in to see where it lands.
    lo_pred = pred[WINSOR]["r"]
    say(f"  floor   independent venues, no spatial structure   r = {lo_pred:.5f}")
    say(f"          (sweep over winsorisation: "
        + ", ".join(f"p{p:g} {pred[p]['r']:.5f}" for p in WINSOR_SWEEP) + ")")
    say(f"  MEASURED passive dong x hour x dow, noise-corrected r = "
        f"{R_OBS_CORRECTED:.5f}")
    say(f"  ceiling a partition that perfectly separates the eight")
    say(f"          place types (the within-place plateau)      r = {floor:.5f}")
    say(f"\n  The measurement lands INSIDE the bracket: "
        f"{R_OBS_CORRECTED / lo_pred:.1f}x the floor "
        f"({R_OBS_CORRECTED / pred[100.0]['r']:.1f}x to "
        f"{R_OBS_CORRECTED / pred[90.0]['r']:.1f}x across the sweep) and "
        f"{R_OBS_CORRECTED / floor:.0%} of the ceiling.")

    say("")
    say("  Reading it forwards: a dong-hour cell is not a random bag of venues --")
    say("  it holds several times more age structure than one -- but it resolves")
    say("  only about half of what separating home from school from work would.")
    say("  The ceiling needs no venue-size mapping, so it is the robust half of")
    say("  the bracket; the floor moves by 7x across the winsorisation sweep.")

    say("  beta itself, four readings of the truth:")
    betas = {}
    for name, r_true in (("survey contact matrix (star pairs)", r_star),
                         ("venue-level, without replacement", r_wor),
                         ("venue-level, with replacement", r_wr),
                         ("Seoul 202312 anchor (p32's 0.2227)", R_SURVEY)):
        b = 1 - R_OBS_CORRECTED / r_true
        betas[name] = dict(r_true=r_true, beta=b, recovery=R_OBS_CORRECTED / r_true)
        say(f"  vs {name:<38} r_true={r_true:.5f}  beta={b:.4f}  "
            f"recovery={R_OBS_CORRECTED / r_true:.2%}")
    b_star = betas["venue-level, without replacement"]["beta"]
    b_lo = min(v["beta"] for v in betas.values())
    b_hi = max(v["beta"] for v in betas.values())
    say(f"\n  So beta = {b_star:.3f} (range {b_lo:.3f}-{b_hi:.3f} over the four")
    say(f"  readings of the truth), against p39's crossing at "
        f"{crossing:.3f}. The measured value")
    say("  sits BELOW the crossing, so p39's sentence survives -- but the reason")
    say("  it survives has to be stated, because it is not an independent test:")
    say("  beta and r_true are one equation, so pinning beta from the survey and")
    say("  then testing whether the bound excludes the survey is circular. What")
    say("  the survey actually supplies is r_true itself, and with r_true in hand")
    say("  the bound is not needed: the instrument recovers "
        f"{R_OBS_CORRECTED / r_wor:.1%} of it. That is the")
    say("  sentence to write -- a measured sensitivity, not a conditional bound.")

    # Two measured decay laws, and what each says a finer release would buy.
    slope_survey = 1.0                        # 44.3, unstructured venue pooling
    need_unstructured = R_SURVEY / R_OBS_CORRECTED
    loc_unstructured = 424 * need_unstructured
    slope_ladder = float(p38["local_slope"][str(YM)]["adjacency|assortativity"]
                         ["slope_at"]["424"])
    assert 0.178 <= slope_ladder <= 0.478, f"p38 slope {slope_ladder} off the record"
    loc_ladder = 424 * need_unstructured ** (1 / slope_ladder)
    say(f"\n  Two measured decay laws disagree, and the gap between them IS the")
    say(f"  spatial correlation. Reaching p32's {R_SURVEY:.4f} from "
        f"{R_OBS_CORRECTED:.5f} needs {need_unstructured:.1f}x:")
    say(f"    slope 1.00 (44.3, pooling independent venues)  ->"
        f" {loc_unstructured:>12,.0f} locations")
    say(f"    slope {slope_ladder:.2f} (p38's merge ladder, real geography) ->"
        f" {loc_ladder:>12,.0f} locations")
    b075 = 1831
    say(f"  B075's {b075:,} traffic polygons sit below BOTH targets. Under the")
    say(f"  optimistic law they would have reached r = "
        f"{R_OBS_CORRECTED * b075 / 424:.4f} "
        f"({R_OBS_CORRECTED * b075 / 424 / R_SURVEY:.0%} of the survey's); under")
    say(f"  p38's measured slope, r = "
        f"{R_OBS_CORRECTED * (b075 / 424) ** slope_ladder:.4f} "
        f"({R_OBS_CORRECTED * (b075 / 424) ** slope_ladder / R_SURVEY:.0%}).")
    say("  Neither closes it, so the Big Data Campus route -- now shut by the")
    say("  eligibility rule -- would not have settled beta either. That is worth")
    say("  saying plainly: the closed door cost less than section 6 assumed.")
    out["beta"] = dict(
        betas=betas, p39_crossing=crossing, beta_primary=b_star,
        beta_range=[b_lo, b_hi], recovery_primary=R_OBS_CORRECTED / r_wor,
        bracket=dict(floor=lo_pred, measured=R_OBS_CORRECTED, ceiling=floor,
                     measured_over_floor=R_OBS_CORRECTED / lo_pred,
                     measured_over_floor_sweep={str(p): R_OBS_CORRECTED / pred[p]["r"]
                                                for p in WINSOR_SWEEP},
                     measured_over_ceiling=R_OBS_CORRECTED / floor),
        reach=dict(factor_needed=need_unstructured,
                   locations_slope1=loc_unstructured,
                   locations_ladder=loc_ladder, ladder_slope=slope_ladder,
                   b075_polygons=b075,
                   b075_r_slope1=R_OBS_CORRECTED * b075 / 424,
                   b075_r_ladder=R_OBS_CORRECTED * (b075 / 424) ** slope_ladder))

    # ============================================ 44.6 bootstrap and the limits
    say("\n=== 44.6 respondent bootstrap, and four things running against us ===")
    ids = P["egos"].ID.to_numpy()
    uniq_ids = np.unique(ids)
    id_groups = {i: np.where(ids == i)[0] for i in uniq_ids}
    boots = []
    for _ in range(BOOT):
        pick = rng.choice(uniq_ids, size=len(uniq_ids), replace=True)
        idx = np.concatenate([id_groups[i] for i in pick])
        boots.append(r_of(collapse(U[idx], wg[idx])))
    bo = np.array(boots)
    ci = (float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)))
    say(f"  r_venue (WOR) = {r_wor:.5f}, respondent bootstrap 95% CI "
        f"[{ci[0]:.5f}, {ci[1]:.5f}]")
    say(f"  beta = {b_star:.4f}, CI [{min(1 - R_OBS_CORRECTED / ci[0], 1 - R_OBS_CORRECTED / ci[1]):.4f}, "
        f"{max(1 - R_OBS_CORRECTED / ci[0], 1 - R_OBS_CORRECTED / ci[1]):.4f}] -- the respondent")
    say("  bootstrap is fourth-decimal narrow and is NOT this number's real")
    say("  uncertainty; the four readings of r_true above are.")
    beta_ci = sorted([1 - R_OBS_CORRECTED / ci[0], 1 - R_OBS_CORRECTED / ci[1]])
    out["bootstrap"] = dict(r_venue=r_wor, ci=ci, n_boot=BOOT, beta_ci=beta_ci)

    lim = [
        "the venue composition is the RESPONDENT'S sample of the venue, not the "
        "venue: alters are the people they spoke to, so the composition is more "
        "age-concentrated than the room. r_venue is an over-estimate, which "
        "makes beta an over-estimate, which runs against the paper's sentence.",
        "the pooling arm pools venues at random, so it carries no spatial "
        "correlation at all. A real dong-hour cell does, which is why it reads "
        f"{pr['measured_over_predicted']:.1f}x the prediction; the random arm is "
        "a floor on retention, not an estimate of it.",
        "Q9 has a 1,000-person tail and E[s^2]/E[s] is tail-dominated, so the "
        "size mapping moves by a factor of "
        f"{sizes[100.0]['size_weighted'] / sizes[90.0]['size_weighted']:.1f} "
        "across the winsorisation sweep. The sweep is reported, not just p99.",
        "the survey is national and the passive matrix is Seoul; this phase "
        "does not settle that, it inherits it. Both scopes are in 44.2 and the "
        "venue-level reading differs between them.",
    ]
    for i, t in enumerate(lim, 1):
        say(f"  {i}. {t}")
    out["limitations"] = lim

    # ================================== 44.7 the two sensitivities 44.6 only named
    hdr = lambda t: say(f"\n=== {t} ===")
    hdr("44.7 the two limitations from 44.6, turned into numbers")

    # (1) The respondent's sample of the venue vs the venue itself.
    #
    # WITHDRAWN 2026-08-24, and the withdrawal is the point of this block. This
    # arm used to pad every visit up to its reported Q9. The arithmetic was
    # right and the field was wrong: the questionnaire's own worked example
    # (survey_offline_v3_ENG.pdf, A9 -- four people playing basketball and then
    # three for coffee is a contact size of six, 4 + 3 - 1) settles that Q9
    # counts THE PEOPLE THE RESPONDENT CONTACTED, self included. It does not
    # count the people present at the venue. This arm exists to add the people
    # at the venue the respondent did NOT talk to, which is precisely the set Q9
    # excludes, so it never bracketed r_venue and the beta bracket it produced
    # -- [0.9096, 0.9326], quoted in the 08-21 letter -- does not hold.
    #
    # The old number is still computed, because a retraction nobody can check is
    # not better than the error. What replaces it is below: the same "unobserved
    # members carry no within-venue sorting" assumption, with venue size left as
    # a DECLARED free parameter instead of read out of the wrong field. That is
    # the same shape as the paper's own beta story -- a sensitivity in something
    # the data does not identify, reported as such, rather than a bracket that
    # looks like knowledge.
    rng2 = np.random.default_rng(SEED + 1)
    place_dist = {}
    for pl in np.unique(place):
        m = P["dd"].place.to_numpy() == pl
        c = np.bincount(P["dd"].abi.to_numpy()[m], minlength=NA).astype(float)
        place_dist[pl] = c / c.sum() if c.sum() else np.full(NA, 1.0 / NA)

    def pad_to(target):
        """Median r over 20 draws, padding each visit up to `target` members."""
        need = np.maximum(0, np.round(target - obs)).astype(int)
        reps = []
        for _ in range(20):
            Up = U.copy()
            for g in np.nonzero(need)[0]:
                Up[g] += rng2.multinomial(need[g], place_dist[place[g]])
            reps.append(r_of(collapse(Up, wg)))
        return float(np.median(reps)), need

    q9raw = P["q9"]
    q9c = np.minimum(np.nan_to_num(q9raw, nan=1.0), np.percentile(q9raw, WINSOR))
    r_pad, pad_needed = pad_to(q9c)
    # The diagnosis, in the two numbers that show Q9 is not a headcount: if it
    # were, it would exceed the recorded contacts at essentially every visit.
    _fin = np.isfinite(q9raw)
    q9_above = float((q9raw[_fin] > obs[_fin]).mean())
    q9_below = float((q9raw[_fin] < obs[_fin]).mean())
    say(f"  venue composition, respondent's sample only : r = {r_wor:.5f}  "
        f"(beta {1 - R_OBS_CORRECTED / r_wor:.4f})")
    say(f"  WITHDRAWN -- padded to Q9                   : r = {r_pad:.5f}  "
        f"(beta {1 - R_OBS_CORRECTED / r_pad:.4f})")
    say(f"  Q9 is the respondent's contact count, not the venue headcount, so")
    say(f"  this arm pads with the wrong target. It shows: Q9 exceeds the")
    say(f"  recorded contacts at only {q9_above:.1%} of visits and is SMALLER at")
    say(f"  {q9_below:.1%}; only {int((pad_needed > 0).sum()):,} of {G:,} visits move at all.")
    say(f"  The bracket this produced, [{1 - R_OBS_CORRECTED / min(r_wor, r_pad):.4f}, "
        f"{1 - R_OBS_CORRECTED / max(r_wor, r_pad):.4f}], is withdrawn.")

    # The replacement: venue size as a declared multiple of what the respondent
    # recorded. k is NOT identified by the released data -- nothing in it counts
    # the people present -- so the curve is reported and no point on it is
    # adopted as the estimate.
    say(f"\n  venue size as a free multiple k of the respondent's own record")
    say(f"  (k is not identified by the released files; this is a curve, not a bound):")
    k_rows = []
    for k in K_VENUE:
        rk, needk = pad_to(k * obs)
        k_rows.append(dict(k=k, r_venue=rk, beta=1 - R_OBS_CORRECTED / rk,
                           people_added=int(needk.sum())))
        say(f"    k = {k:<4}  r_venue {rk:.5f}   beta {1 - R_OBS_CORRECTED / rk:.4f}   "
            f"+{int(needk.sum()):,} people")
    say(f"  r_venue falls monotonically in k, so every unobserved venue member")
    say(f"  pushes beta DOWN: the respondent-only reading {1 - R_OBS_CORRECTED / r_wor:.4f} is an")
    say(f"  UPPER bound on beta, and that direction is all the released data supports.")

    # (2) The national/Seoul scope, which 44.6 also only named.
    seo = ladders[f"seoul|{YM}"]["rows"][0]["r"]
    scope_rows = {}
    for nm, rv in (("national pooled (primary)", r_wor), (f"seoul {YM} (anchor)", seo)):
        scope_rows[nm] = dict(r_venue=rv, beta=1 - R_OBS_CORRECTED / rv,
                              recovery=R_OBS_CORRECTED / rv,
                              floor=c_law * sw * L_day / N_day * rv,
                              measured_over_floor=R_OBS_CORRECTED
                              / (c_law * sw * L_day / N_day * rv))
        z = scope_rows[nm]
        say(f"  {nm:<26} r_venue {rv:.5f}  beta {z['beta']:.4f}  "
            f"floor {z['floor']:.5f}  measured/floor {z['measured_over_floor']:.1f}x")
    n_seoul = int(build(ego, d, "seoul", YM)["e"].shape[0])
    say(f"  The Seoul arm is {n_seoul} respondents against {len(P['e']):,}, so it is the")
    say("  noisier reading, not the more relevant one -- but it is the arm the")
    say("  passive matrix's own scope matches, and it moves beta by less than")
    say("  the padding does.")
    # `pad` is deliberately NOT the key any more. Anything still reading it was
    # reading a withdrawn bracket, and a KeyError is the right way to find out.
    out["sensitivity"] = dict(
        pad_withdrawn=dict(
            withdrawn="2026-08-24",
            why="Q9 counts the people the respondent contacted (self included), "
                "not the venue headcount, so padding to Q9 targets the wrong "
                "quantity; the arm never bracketed r_venue",
            source="survey_offline_v3_ENG.pdf A9 worked example",
            r_padded_to_q9=r_pad, r_unpadded=r_wor,
            n_added=int(pad_needed.sum()),
            n_visits_padded=int((pad_needed > 0).sum()),
            q9_above_recorded_share=q9_above, q9_below_recorded_share=q9_below,
            beta_bracket_withdrawn=[1 - R_OBS_CORRECTED / min(r_wor, r_pad),
                                    1 - R_OBS_CORRECTED / max(r_wor, r_pad)]),
        venue_size=dict(
            k_grid=list(K_VENUE), rows=k_rows,
            identified=False,
            reading="r_venue falls monotonically in k, so unobserved venue "
                    "members can only push beta down; the respondent-only "
                    "reading is an upper bound on beta",
            beta_upper=1 - R_OBS_CORRECTED / r_wor),
        scope=scope_rows, n_seoul_respondents=n_seoul)

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for arm, sty in (("random", "o-"), ("within-place", "s--")):
        # Only the points a between-cell variance can actually be read off. The
        # excluded tail is in the table and in the JSON; plotting it would draw a
        # cliff that is small-sample bias, not decay.
        ok = [r for r in curves[arm] if r["n_cells"] >= MIN_CELLS]
        ax[0].loglog([r["n_eff"] for r in ok], [r["r"] for r in ok], sty, ms=4,
                     label=f"{arm} pooling")
    ax[0].axhline(R_OBS_CORRECTED, color="crimson", lw=1.4,
                  label=f"passive dong x hour x dow ({R_OBS_CORRECTED:.4f})")
    ax[0].axhline(r_star, color="0.4", ls=":", lw=1.2,
                  label=f"survey contact matrix ({r_star:.3f})")
    ax[0].axhline(n2_wor, color="seagreen", ls="-.", lw=1.0,
                  label=f"place-type floor ({n2_wor:.4f})")
    ax[0].set_xlabel("effective number of venues pooled")
    ax[0].set_ylabel("assortativity")
    ax[0].set_title("44.3  the decay law, measured on real contacts")
    ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].grid(alpha=.3, which="both")

    rows = ladders["national|pooled"]["rows"]
    names = ["contacts"] + [r["grouping"].split(" (")[0] for r in rows]
    vals = [r_star] + [r["r"] for r in rows]
    ax[1].barh(range(len(vals)), vals, color=["0.35"] + ["steelblue"] * len(rows))
    ax[1].axvline(R_OBS_CORRECTED, color="crimson", lw=1.4)
    ax[1].set_yticks(range(len(vals)))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xscale("log")
    ax[1].set_xlabel("assortativity (log)")
    ax[1].set_title("44.2  coarsen the group, watch it go")
    ax[1].grid(alpha=.3, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(f"{FIG}/p44_beta.png", dpi=150)
    say(f"\n  figure -> {FIG}/p44_beta.png")

    with open(f"{ROOT}/eda/results_p44.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"  results -> {ROOT}/eda/results_p44.json")


if __name__ == "__main__":
    main()
