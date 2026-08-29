#!/usr/bin/env python
"""Phase 51 — the national arm, symmetrised with Korea's population.

WHY THIS EXISTS. `p40_natfloor.py` builds the national arm by reducing the
national survey's per-capita matrix with `symmetrise(C, pop)` where `pop` is
`results_p26.json["population"]` -- Seoul's registered population
(`p26_matrix.py:46` says so in as many words). 1,987 respondents drawn from the
whole country are therefore reweighted onto Seoul's age structure, and
`results_p50.json` measures how different those structures are: the Seoul share
of 25-29 is 1.263x Korea's, of 10-14 it is 0.835x, and the largest band-share
gap is 1.81 pp. The arm is neither Seoul nor national. The advisor's 2026-08-24
letter: 不修的話那一臂稱不上全國臂.

This phase is the corrected arm. It writes `results_p51.json` and mutates
nothing -- `results_p40.json` in particular is not re-run and cannot move,
which is HOW the Seoul arm is held bit-identical rather than merely hoped to be.

THE ARCHITECTURE, and why it is a new phase rather than an edit to p40.

`p40_natfloor.py:60-67` states its own design: "Both scopes are drawn from this
one stream so the Seoul and national arms of 40.2 are directly comparable."
40.2 is a single `for ym: for scope:` loop over one `default_rng(SEED)`.
Inserting a third scope, or changing what any cell draws, shifts every
subsequent draw and moves the Seoul arm. The most natural edit destroys the one
thing that must not move. So p40 is left alone.

THE PAIRING TRICK THAT MAKES THIS A CONTROLLED COMPARISON. The bootstrap
indices and the ego-band permutations depend on the rng and on the ego array --
NOT on the population vector, which enters only afterwards inside
`symmetrise`. So one pass over p40's exact stream can produce BOTH readings of
every draw: reduce each drawn contact matrix twice, once with Seoul's vector and
once with Korea's. The Seoul-vector half must then reproduce `results_p40.json`
bit for bit, and the Korea-vector half differs from it by the population vector
and by nothing else -- same draws, same permutations, same order. A re-run and
a diff could not establish that; it could only show two numbers that happen to
agree.

  ONE EXCEPTION, HANDLED EXPLICITLY. `null_floor(T0, n_con, rng=rng)` draws
  FROM the matrix, so it consumes the shared stream in a way that depends on
  which vector built T0. Calling it twice would desynchronise the replay. It is
  therefore called once on the Seoul-vector matrix, inside the shared stream, to
  keep the replay exact; the corrected arm's multinomial floor is drawn from its
  own declared stream (SEED_CORRECTED) and is reported as such rather than
  pretended to be paired.

DECLARED BEFORE THE RUN.

  * PRIMARY CELL: 202312 | national -- the same cell `results_p40.json`
    declares, so the corrected number replaces a number rather than answering a
    different question.
  * The survival rule is copied verbatim from p40's declaration, not restated:
    "headline survives iff passive mi_bits < the primary cell's permutation
    median".
  * COUNTS ARE p40's: BOOT and PERM from p32, so nothing but the vector differs.
  * BOTH national vectors are `established=True` in results_p50.json. 202402's
    foreigner component still has its dong-level age composition interpolated
    (2024-03 is an unreadable legacy .xls), but its level is anchored 시군구 by
    시군구 onto the monthly national file, and the Seoul restriction reproduces
    foreign.parquet to 3.6e-16. An earlier build of p50 interpolated without
    that anchor, declared a 1e-3 tolerance and fired at 3.79%; that declaration
    and its number are kept in results_p50.json["superseded_declaration"].
    202312 still carries the claim, because it is the declared primary cell.
  * THE REPORTED NATIONAL ARM IS THE CORRECTED ONE. The Seoul-vector national
    arm is retained only so the 08-19 and 08-22 letters stay checkable against
    what they were written from. Keeping both and choosing later is the
    number-shopping the declaration blocks exist to prevent.

THE CONTROL THAT p40 DOES NOT HAVE, AND WHY IT IS MANDATORY HERE. 40.4 builds
the IPF target margin as `Ts.sum(1)/Ts.sum() * Tp.sum()` from the survey matrix
symmetrised with `pop`. Once `pop` is Korea's, that target differs from Seoul's
passive margin by AGE COMPOSITION as well as by mixing -- and the advisor now
wants "IPF absorbs only 19.1% / 23.0%" promoted to a load-bearing, survey-free
claim. Without a control, part of that number is demography. So a third arm is
run: the SEOUL survey matrix reduced with KOREA's vector. It shares the
national arm's demography and the Seoul arm's mixing, so the difference between
it and the Seoul arm is the demographic contribution, isolated.

    python eda/p51_natsym.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import AGES, AGE_LABEL
from paths import ROOT

import p32_pmix as P32
from p32_pmix import excess_stats, ipf, null_floor, rank_stats

SURVEY_MONTHS = [202312, 202402]
SCOPES = [("national", False), ("seoul", True)]     # p40's order, national first
SEED = 20260821                                     # p40's, so the replay lines up
SEED_CORRECTED = 20260824                           # this phase's own draws
PRIMARY = dict(statistic="mi_perm_median", ym=202312, scope="national",
               panel="WE", days="holiday-free", block="0-79",
               vector="korea")

out = {}
FAIL = []


def hdr(s):
    print(f"\n=== {s} ===", flush=True)


def say(s=""):
    print(s, flush=True)


def main():
    # ------------------------------------------------------------------ 51.0 load
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))
    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))
    p32 = json.load(open(f"{ROOT}/eda/results_p32.json"))
    p40 = json.load(open(f"{ROOT}/eda/results_p40.json"))
    p50 = json.load(open(f"{ROOT}/eda/results_p50.json"))
    p9 = json.load(open(f"{ROOT}/eda/results_p9.json"))

    sq = [i for i, a in enumerate(AGES) if a < 80]
    lbl = [AGE_LABEL[AGES[i]] for i in sq]
    cov = {c["band"]: c["p10"] for c in p9["coverage_profile"]}

    # The two vectors. `seoul` is what every arm has used; `korea` is p50's.
    POP_SEOUL = {ym: np.array(p26["population"][str(ym)], float)
                 for ym in SURVEY_MONTHS}
    POP_KOREA = {ym: np.array(p50["vectors"][str(ym)]["national"], float)
                 for ym in SURVEY_MONTHS}
    ESTABLISHED = {ym: bool(p50["vectors"][str(ym)]["established"])
                   for ym in SURVEY_MONTHS}

    # p50's Seoul column must BE p26's, or the two phases disagree about the thing
    # they share and nothing below can be compared to anything above.
    for ym in SURVEY_MONTHS:
        d = np.abs(np.array(p50["vectors"][str(ym)]["seoul"], float)
                   - POP_SEOUL[ym]).max()
        assert d == 0.0, f"p50's Seoul vector is not p26's for {ym} (max {d})"
    say(f"51.0  p50's Seoul column reproduces p26's population, both months, "
        f"max abs diff 0.0")
    say(f"      202312 national vector established={ESTABLISHED[202312]}, "
        f"202402 established={ESTABLISHED[202402]}")

    # The passive side is Seoul and STAYS Seoul under both readings -- that is the
    # asymmetry the paper already declares as a limitation, not something this
    # phase changes. Only the SURVEY side was being mis-reduced.
    passive = {}
    for ym in SURVEY_MONTHS:
        A = np.array(p26["matrices"][f"{ym}|WE|dong|holidayfree"]["A"])[np.ix_(sq, sq)]
        Tp, _ = P32.symmetrise(A / POP_SEOUL[ym][sq][:, None], POP_SEOUL[ym][sq])
        n_dev = float(sum(POP_SEOUL[ym][i] / cov.get(AGE_LABEL[AGES[i]], np.nan)
                          for i in sq))
        passive[ym] = dict(T=Tp, n_dev=n_dev, ex=excess_stats(Tp), sp=rank_stats(Tp))

    from p27_survey import (contact_matrix, ego_cubes, holiday_free_dows,
                            load_survey)

    ego, d = load_survey()
    d = d[d.panel.isin(["W", "E"])]          # NOT household -- p32 32.4b's rule
    cubes = {}
    for ym in SURVEY_MONTHS:
        for scope, so in SCOPES:
            cubes[(ym, scope)] = ego_cubes(ego, d, so, ym=ym,
                                           dows=holiday_free_dows(ym))

    # ==================================================== 51.1 the replay anchor
    hdr("51.1 REPLAY ANCHOR: p40's 40.2 loop, same stream, both vectors at once")
    say("  The bootstrap indices and permutations do not depend on the population")
    say("  vector, so one pass produces both readings of every draw. The Seoul-")
    say("  vector half must reproduce results_p40.json bit for bit.")

    rng = np.random.default_rng(SEED)
    cells_replay, cells_korea = {}, {}
    for ym in SURVEY_MONTHS:
        for scope, _ in SCOPES:
            cube, eb, nd = cubes[(ym, scope)]
            p_s = POP_SEOUL[ym][sq]
            p_k = POP_KOREA[ym][sq]
            # For the Seoul SCOPE the corrected vector is not a question -- a Seoul
            # survey reduced with Seoul's population is already right -- so both
            # halves use the Seoul vector there and the arm is untouched by design.
            p_alt = p_k if scope == "national" else p_s

            draws_s, perms_s, assort_s, aperm_s = [], [], [], []
            draws_k, perms_k, assort_k, aperm_k = [], [], [], []
            for _ in range(P32.BOOT):
                idx = rng.integers(0, len(eb), len(eb))
                C, _ = contact_matrix(cube[idx], eb[idx], nd)
                Cq = C[np.ix_(sq, sq)]
                st = excess_stats(P32.symmetrise(Cq, p_s)[0])
                draws_s.append(st["mi_bits"])
                assort_s.append(st["assortativity"])
                stk = excess_stats(P32.symmetrise(Cq, p_alt)[0])
                draws_k.append(stk["mi_bits"])
                assort_k.append(stk["assortativity"])
            for _ in range(P32.PERM):
                C, _ = contact_matrix(cube, rng.permutation(eb), nd)
                Cq = C[np.ix_(sq, sq)]
                st = excess_stats(P32.symmetrise(Cq, p_s)[0])
                perms_s.append(st["mi_bits"])
                aperm_s.append(st["assortativity"])
                stk = excess_stats(P32.symmetrise(Cq, p_alt)[0])
                perms_k.append(stk["mi_bits"])
                aperm_k.append(stk["assortativity"])

            C0, _ = contact_matrix(cube, eb, nd)
            C0q = C0[np.ix_(sq, sq)]
            n_con = float(np.bincount(eb, minlength=len(AGES))[sq]
                          @ C0q.sum(1) * nd)
            T0_s = P32.symmetrise(C0q, p_s)[0]
            T0_k = P32.symmetrise(C0q, p_alt)[0]
            ex_s = excess_stats(T0_s, n_obs=n_con)
            ex_k = excess_stats(T0_k, n_obs=n_con)
            # Consumes the shared stream, and depends on T0 -- so it is drawn ONCE,
            # on the Seoul-vector matrix, exactly where p40 draws it.
            mfl_s = null_floor(T0_s, n_con, rng=rng)
            pas = passive[ym]["ex"]["mi_bits"]

            def pack(perms, draws, assort, aperm, ex, mfloor, rr):
                pv = np.array(perms, float)
                med = float(np.median(pv))
                p95 = float(np.percentile(pv, 95))
                meds = [float(np.median(rr.choice(pv, len(pv), replace=True)))
                        for _ in range(1000)]
                se = float(np.std(meds, ddof=1))
                return dict(
                    ym=ym, scope=scope, n_ego=int(len(eb)), n_contacts=n_con,
                    survey_mi_bits=ex["mi_bits"], passive_mi_bits=pas,
                    mi_perm_median=med, mi_perm_p95=p95, mi_perm_median_se=se,
                    mi_perm_median_margin_in_se=float((med - pas) / se) if se
                    else np.nan,
                    passive_frac_below_perm_null=float((pv < pas).mean()),
                    mi_perm_lo=float(np.percentile(pv, 2.5)),
                    mi_perm_hi=float(np.percentile(pv, 97.5)),
                    mi_boot_median=float(np.median(draws)),
                    mi_boot_lo=float(np.percentile(draws, 2.5)),
                    mi_boot_hi=float(np.percentile(draws, 97.5)),
                    assort_boot_lo=float(np.percentile(assort, 2.5)),
                    assort_boot_hi=float(np.percentile(assort, 97.5)),
                    assort_perm_median=float(np.median(aperm)),
                    assort_observed=ex["assortativity"],
                    mm_bias_bits=ex["mm_bias_bits"],
                    multinomial_floor_median=mfloor["mi_bits"]["median"],
                    multinomial_floor_p95=mfloor["mi_bits"]["p95"],
                    floor_over_passive_median=float(med / pas),
                    floor_over_passive_p95=float(p95 / pas),
                    survives_median=bool(pas < med), survives_p95=bool(pas < p95))

            # The Seoul-vector reading uses the shared stream for its median SE,
            # exactly as p40 does, so the replay stays bit-exact.
            cells_replay[f"{ym}|{scope}"] = pack(perms_s, draws_s, assort_s,
                                                 aperm_s, ex_s, mfl_s, rng)
            cells_korea[f"{ym}|{scope}"] = dict(
                _pending=True, perms=perms_k, draws=draws_k, assort=assort_k,
                aperm=aperm_k, ex=ex_k, n_con=n_con, eb=len(eb))

    # --- the anchor itself ------------------------------------------------------
    KEYS = ["survey_mi_bits", "passive_mi_bits", "mi_perm_median", "mi_perm_p95",
            "mi_perm_median_se", "passive_frac_below_perm_null", "mi_perm_lo",
            "mi_perm_hi", "mi_boot_median", "mi_boot_lo", "mi_boot_hi",
            "assort_boot_lo", "assort_boot_hi", "assort_perm_median",
            "assort_observed", "mm_bias_bits", "multinomial_floor_median",
            "multinomial_floor_p95", "floor_over_passive_median",
            "floor_over_passive_p95"]
    anchor = []
    worst = 0.0
    for k, got in cells_replay.items():
        want = p40["cells"][k]
        for f in KEYS:
            d_ = abs(float(got[f]) - float(want[f]))
            worst = max(worst, d_)
            if d_ != 0.0:
                FAIL.append(f"replay {k}.{f}: {got[f]!r} vs p40 {want[f]!r}")
        anchor.append(dict(cell=k, n_keys=len(KEYS),
                           max_abs_diff=max(abs(float(got[f]) - float(want[f]))
                                            for f in KEYS)))
    say(f"  replayed {len(cells_replay)} cells x {len(KEYS)} quantities; "
        f"worst |diff| vs results_p40.json = {worst:.3e}")
    assert worst == 0.0, (
        f"the replay does not reproduce p40 ({worst:.3e}). The corrected arm would "
        f"then differ from the published one by the machinery as well as by the "
        f"population vector, and nothing below could be attributed to the vector.")
    out["replay_anchor"] = dict(cells=anchor, n_quantities=len(KEYS),
                                worst_abs_diff=float(worst), bit_exact=True)

    # ================================================ 51.2 the corrected cells
    hdr("51.2 THE CORRECTED NATIONAL ARM")
    rng_c = np.random.default_rng(SEED_CORRECTED)
    cells = {}
    for k, st in cells_korea.items():
        ym, scope = int(k.split("|")[0]), k.split("|")[1]
        pv = np.array(st["perms"], float)
        pas = passive[ym]["ex"]["mi_bits"]
        med = float(np.median(pv))
        p95 = float(np.percentile(pv, 95))
        meds = [float(np.median(rng_c.choice(pv, len(pv), replace=True)))
                for _ in range(1000)]
        se = float(np.std(meds, ddof=1))
        p_alt = POP_KOREA[ym][sq] if scope == "national" else POP_SEOUL[ym][sq]
        C0, _ = contact_matrix(*[cubes[(ym, scope)][0], cubes[(ym, scope)][1],
                                 cubes[(ym, scope)][2]])
        T0 = P32.symmetrise(C0[np.ix_(sq, sq)], p_alt)[0]
        mfl = null_floor(T0, st["n_con"], rng=rng_c)
        cells[k] = dict(
            ym=ym, scope=scope, vector=("korea" if scope == "national" else "seoul"),
            established=ESTABLISHED[ym] if scope == "national" else True,
            n_ego=st["eb"], n_contacts=st["n_con"],
            survey_mi_bits=st["ex"]["mi_bits"], passive_mi_bits=pas,
            mi_perm_median=med, mi_perm_p95=p95, mi_perm_median_se=se,
            passive_frac_below_perm_null=float((pv < pas).mean()),
            mi_boot_median=float(np.median(st["draws"])),
            mi_boot_lo=float(np.percentile(st["draws"], 2.5)),
            mi_boot_hi=float(np.percentile(st["draws"], 97.5)),
            assort_observed=st["ex"]["assortativity"],
            assort_perm_median=float(np.median(st["aperm"])),
            multinomial_floor_median=mfl["mi_bits"]["median"],
            floor_over_passive_median=float(med / pas),
            floor_over_passive_p95=float(p95 / pas),
            survives_median=bool(pas < med), survives_p95=bool(pas < p95))
    out["cells"] = cells

    say(f"  {'cell':<20}{'passive':>10}{'perm med':>10}{'ratio':>8}"
        f"{'was':>8}{'move':>9}  survives")
    for ym in SURVEY_MONTHS:
        for scope, _ in SCOPES:
            k = f"{ym}|{scope}"
            c, w = cells[k], p40["cells"][k]
            mv = c["floor_over_passive_median"] / w["floor_over_passive_median"] - 1
            flag = "" if c["established"] else "  (202402: vector not established)"
            say(f"  {k:<20}{c['passive_mi_bits']:>10.5f}"
                f"{c['mi_perm_median']:>10.5f}"
                f"{c['floor_over_passive_median']:>8.2f}"
                f"{w['floor_over_passive_median']:>8.2f}{100 * mv:>+8.1f}%"
                f"  {'yes' if c['survives_median'] else 'NO'}{flag}")

    prim_k = f"{PRIMARY['ym']}|{PRIMARY['scope']}"
    out["verdict_primary"] = dict(
        cell=prim_k, rule=p40["declaration"]["survives_rule"],
        ratio=cells[prim_k]["floor_over_passive_median"],
        ratio_published=p40["cells"][prim_k]["floor_over_passive_median"],
        survives=cells[prim_k]["survives_median"],
        survives_published=p40["cells"][prim_k]["survives_median"])

    # ============================================ 51.3 the point comparison
    hdr("51.3 the four statistics, national arm, corrected vector")
    comp = {}
    for ym in SURVEY_MONTHS:
        rows = {}
        for scope, _ in SCOPES:
            C = np.array(p27["survey"][f"{ym}|WE|{scope}"]["C"])[np.ix_(sq, sq)]
            p_alt = POP_KOREA[ym][sq] if scope == "national" else POP_SEOUL[ym][sq]
            T, asym = P32.symmetrise(C, p_alt)
            rows[scope] = dict(excess_stats(T, n_obs=cells[f"{ym}|{scope}"]
                                            ["n_contacts"]),
                               asymmetry=asym, **rank_stats(T))
        pas = passive[ym]["ex"]
        comp[str(ym)] = dict(passive=pas, seoul=rows["seoul"],
                             national=rows["national"])
        say(f"\n  {ym}  {'statistic':<16}{'passive':>10}{'national':>10}"
            f"{'x now':>9}{'x before':>10}")
        for st, nm in (("half_l1", "half-L1"), ("cramers_v", "Cramer's V"),
                       ("assortativity", "assortativity"),
                       ("mi_bits", "MI (bits)"), ("nmi", "normalised MI")):
            rn = rows["national"][st] / pas[st]
            was = p40["point_comparison"][str(ym)]["national"][st] / pas[st]
            say(f"  {'':<6}{nm:<16}{pas[st]:>10.4f}{rows['national'][st]:>10.4f}"
                f"{rn:>9.2f}{was:>10.2f}")
    out["comparison"] = comp
    spans = {}
    for ym in SURVEY_MONTHS:
        rs = [comp[str(ym)]["national"][s] / comp[str(ym)]["passive"][s]
              for s in ("half_l1", "cramers_v", "assortativity", "mi_bits", "nmi")]
        was_rs = [p40["point_comparison"][str(ym)]["national"][s_]
                  / comp[str(ym)]["passive"][s_]
                  for s_ in ("half_l1", "cramers_v", "assortativity", "mi_bits",
                             "nmi")]
        spans[str(ym)] = dict(lo=float(min(rs)), hi=float(max(rs)),
                              lo_published=float(min(was_rs)),
                              hi_published=float(max(was_rs)))
    out["national_span"] = spans
    say(f"\n  corrected national span: "
        + ", ".join(f"{ym} {spans[str(ym)]['lo']:.1f}x-{spans[str(ym)]['hi']:.1f}x "
                    f"(was {spans[str(ym)]['lo_published']:.1f}x-"
                    f"{spans[str(ym)]['hi_published']:.1f}x)"
                    for ym in SURVEY_MONTHS))

    # ==================================================== 51.4 IPF, with control
    hdr("51.4 IPF margin calibration, corrected -- with the demography control")
    say("  Three arms. seoul: Seoul survey, Seoul vector (unchanged, anchored to")
    say("  p32). national: national survey, Korea vector (the corrected number).")
    say("  demography_control: SEOUL survey, KOREA vector -- national demography,")
    say("  Seoul mixing, so what it absorbs is demography and not mixing.")
    ipf_out = {}
    for ym in SURVEY_MONTHS:
        Tp = passive[ym]["T"]
        row = {}
        arms = [("seoul", f"{ym}|WE|seoul", POP_SEOUL[ym][sq]),
                ("national", f"{ym}|WE|national", POP_KOREA[ym][sq]),
                ("demography_control", f"{ym}|WE|seoul", POP_KOREA[ym][sq])]
        for name, key, pv_ in arms:
            C = np.array(p27["survey"][key]["C"])[np.ix_(sq, sq)]
            Ts, _ = P32.symmetrise(C, pv_)
            tgt = Ts.sum(1) / Ts.sum() * Tp.sum()
            X, _ = ipf(Tp, tgt, tgt)
            mult = np.divide(X.sum(1), Tp.sum(1), out=np.ones_like(tgt),
                             where=Tp.sum(1) > 0)
            eb_, ea_ = excess_stats(Tp), excess_stats(X)
            nmi_s = comp[str(ym)]["national" if name == "national" else "seoul"]["nmi"]
            row[name] = dict(
                bands=lbl, row_multiplier=mult.tolist(),
                nmi_before=eb_["nmi"], nmi_after=ea_["nmi"], nmi_survey=nmi_s,
                gap_before=float(nmi_s / eb_["nmi"]),
                gap_after=float(nmi_s / ea_["nmi"]),
                absorbed_frac=float(1 - (nmi_s / ea_["nmi"]) / (nmi_s / eb_["nmi"])),
                converged=bool(nmi_s / ea_["nmi"] < 2.0))
            say(f"  {ym} {name:<19}: gap {row[name]['gap_before']:.1f}x -> "
                f"{row[name]['gap_after']:.1f}x "
                f"({row[name]['absorbed_frac']:.1%} absorbed); "
                f"{'CONVERGES' if row[name]['converged'] else 'does NOT converge'}")
        # anchor: the Seoul arm must still reproduce p32's stored 32.5 exactly
        stored = p32["ipf_calibration"]["months"].get(str(ym))
        dd = abs(row["seoul"]["gap_after"] - stored["gap_after"])
        assert dd < 1e-9, f"51.4's Seoul arm no longer reproduces p32 32.5: {dd:.2e}"
        say(f"       anchor: Seoul gap_after reproduces p32 "
            f"({stored['gap_after']:.6f}), |diff| {dd:.1e}")
        was = p40["ipf_calibration"][str(ym)]["national"]["absorbed_frac"]
        dem = row["demography_control"]["absorbed_frac"]
        nat = row["national"]["absorbed_frac"]
        say(f"       national absorbed {was:.1%} (published, Seoul vector) -> "
            f"{nat:.1%} (corrected)")
        say(f"       of which demography alone accounts for {dem:.1%} "
            f"(Seoul survey on Korea's vector)")
        row["published_absorbed_frac"] = was
        ipf_out[str(ym)] = row
    out["ipf_calibration"] = ipf_out

    out["declaration"] = dict(
        primary_cell=PRIMARY, seed_replay=SEED, seed_corrected=SEED_CORRECTED,
        n_boot=P32.BOOT, n_perm=P32.PERM,
        survives_rule=p40["declaration"]["survives_rule"],
        reported_arm="corrected (Korea vector); the Seoul-vector national arm is "
                     "retained only so the 08-19/08-22 letters stay checkable",
        established={str(k): v for k, v in ESTABLISHED.items()},
        touches_nothing=["results_p26.json", "results_p27.json", "results_p32.json",
                         "results_p40.json"])
    out["fail"] = FAIL

    with open(f"{ROOT}/eda/results_p51.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p51.json")
    if FAIL:
        say("\nFAILURES:")
        for f in FAIL:
            say(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
