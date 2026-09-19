#!/usr/bin/env python
"""Phase 65 — NIMS's reply to the 2026-08-24 enquiry, and the three things it moves.

WHY THIS EXISTS. Jonggul Lee replied on 2026-09-01 (the reply is pasted under
`Data_Questions_Prof.md`). It answers all three questions, and two of the answers
are not the ones the enquiry assumed:

  Q1  Q5 IS a multiple-response item. The Sci Data descriptor's "single-response"
      sentence is wrong by the authors' own account -- and that sentence was
      `p57`'s stated reason for marking the recoding-aware reading primary. They
      decline to endorse either convention: code 8 beside another code "may
      reflect recoding ... but it may also represent a genuine multiple-location
      contact", and the handling is to be chosen by analytic objective and
      reported. So 10.53% is not an error and 4.48% is not a correction; they
      bracket a quantity the released fields cannot resolve.

      They also confirm a SECOND coding error, in the opposite direction, that
      `p57` did not model: wherever `Q5_8_etc` carries text, `Q5_8` should be
      considered set. Section 65.2 is that convention -- the literal rule applied
      to the file the authors say they meant to release -- and it is the reason
      the recoding-aware reading survives its own justification being withdrawn:
      on the corrected file the literal rule counts a free-text FLAG, not a place.

  Q2  Code 1 is any private household, the respondent's own or someone else's,
      and the released data cannot separate them. Section 65.4 is what that costs
      us. Our panel rule sends a contact to the panel of its LOWEST set code, so
      code 1 always wins and every visit to a relative's home lands in H and is
      dropped from W+E -- while on the passive side the destination attribute is
      the traveller's OWN 야간상주지, so a trip to a relative's home is an E
      destination that the passive W+E panel keeps. The two sides classify the
      same event differently, and the reply is what makes that certain.

  Q3  Q9 = 0 and Q9 = 1 are errors; no cleaning and no top-coding were applied;
      and the descriptor's "maximum 329, minimum 2" describes records per
      participant, not Q9. This CONFIRMS the 2026-08-24 withdrawal of `p44`
      section 44.7's padding arm (`eda/memo/phase44-beta.md`), so nothing is
      withdrawn here. Section 65.3 gates the counts instead: they are quoted in a
      letter that has been sent and answered, and no results file produces them.

WHAT IS ANCHORED. Three anchors, each read from the file it anchors to rather
than typed here, so the two cannot drift apart:

  * `p57`'s two conventions must come back bit-for-bit off the same CSV (65.0).
    If they do not, this file is not reading the rows `p57` read and nothing
    below is a statement about `p57`'s numbers.
  * `p27`'s published contact count, 133,776 (65.0).
  * `p27`'s published survey assortativity for the primary cell must come back
    bit-for-bit from an unmodified panel assignment before any contact is moved
    (65.4). That anchor checks the CALL, not the estimator -- section 65.4
    imports `p27_survey`'s own functions on purpose, because the quantity wanted
    is a delta under one changed input and a second implementation would confound
    the delta with the rewrite. `p36` is where independent recomputation lives.

WHAT THIS PHASE DOES NOT DO. It does not edit `p27` or `p57`, and it adopts
nothing downstream. `results_p57.json` records what was measured BEFORE the reply
and its declaration is left exactly as it was; rewriting a pre-registration after
the answer arrives is the failure it exists to prevent.

    python eda/p65_chaereply.py
"""
import csv
import io as _io
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import AGES
from p26_matrix import holiday_free_dows
from p27_survey import (assortativity, contact_matrix, ego_cubes, load_survey,
                        symmetrise)
from paths import DATA_ROOT, ROOT, require

CHAE = DATA_ROOT / "raw" / "chae2026"
ETC = 8
OUT = f"{ROOT}/eda/results_p65.json"
SURVEY_MONTHS = [202312, 202402]

# ---------------------------------------------------------------------------
# DECLARED BEFORE THE RUN. The household arms are the part that could be tuned
# after seeing the answer, so all three are fixed here and all three are
# reported. They are nested (letter_four subset of kinship subset of
# any_free_text), which is what makes them a bracket rather than three guesses.
#
#   letter_four   the four strings the SENT letter quotes. Pre-registered by a
#                 document that left before the reply arrived, so it cannot have
#                 been chosen to produce an outcome. Narrowest.
#   kinship       substring match on a declared kinship vocabulary. Written from
#                 Korean kinship terms, not from the file's value frequencies;
#                 the script prints what it did NOT match so the coverage is
#                 auditable rather than asserted.
#   any_free_text ANY code-1 row carrying free text. Needs no judgement about
#                 Korean strings at all: per the reply, free text means the
#                 respondent did not take the plain "household" option and the
#                 answer was back-mapped, so a code-1 row with free text is by
#                 construction not the ordinary own-home answer. Widest, and the
#                 one to quote as the upper end.
#
# Table S6 of the descriptor's Supplementary Information holds the authors' own
# location mapping and would replace all three. It is not in the released
# deposit, and the reply notes that only part of the outdoor list was published.
# ADDED 2026-09-01, AFTER the first run, and the reason is in the file it reads:
# Table S6 is the AUTHORS' OWN mapping, obtained from the descriptor's
# Supplementary Information (which is not in the figshare deposit) once the reply
# named it. Its membership is not ours to choose, which is what makes adding an
# arm here different from widening one. The three arms below it are kept exactly
# as they were declared and reported beside it -- in particular `kinship`, which
# Table S6 shows undercounts. A declared arm that turned out to be a poor guess
# is evidence about the guess and is not deleted.
#
# The reply also says only PART of the outdoor list was published, so a value's
# ABSENCE from S6 is not evidence that it was not mapped.
S6 = json.loads((pathlib.Path(__file__).parent / "chae_tableS6.json").read_text())
S6_HOUSEHOLD = frozenset(S6["entries"])

LETTER_FOUR = ("친척집", "할아버지댁", "할머니집", "시댁")
KINSHIP = ("친척", "할아버지", "할머니", "조부모", "외가", "친정", "시댁", "처가",
           "이모", "고모", "삼촌", "외삼촌", "큰집", "작은집", "본가", "부모님",
           "언니", "누나", "형", "오빠", "동생", "사촌", "손자", "손녀",
           "친구집", "지인집", "이웃집")

declaration = dict(
    primary_convention="recoding_aware",
    why=("the descriptor's single-response statement -- p57's stated reason -- "
         "was withdrawn by the authors on 2026-09-01. The reason is now that on "
         "the file they say they meant to release, every free-text row carries "
         "code 8, so the literal rule counts a flag and not a place"),
    reported_as="a bracket: recoding_aware is the lower end, corrected_literal the upper",
    household_arms=["letter_four", "kinship", "any_free_text",
                    "s6_household", "s6_other"],
    household_arm_primary="any_free_text",
    s6_decomposition=("s6_household and s6_other partition any_free_text by the "
                      "authors' own Table S6. s6_household is a LOWER BOUND on "
                      "'a residence that is not the respondent's own home': the "
                      "run shows S6's household list is incomplete, so s6_other "
                      "is a mixture of unlisted residences and genuine second "
                      "places, not the second category alone"),
    household_destination_panel="E",
    household_destination_reason=(
        "the passive dest_attr is the traveller's own 야간상주지, so another "
        "household is an E destination there and the two sides must agree"),
    primary_cell="202312|WE|seoul, holiday-free -- the cell p32's 0.2227 was measured in",
    touches_nothing=["results_p26.json", "results_p27.json", "results_p44.json",
                     "results_p57.json"])


def rows():
    require(CHAE, "Chae microdata (run eda/dl_chae.py)")
    t = (CHAE / "results_main_survey.csv").read_bytes().decode("utf-8-sig")
    r = list(csv.reader(_io.StringIO(t)))
    return [c.strip() for c in r[0]], r[1:]


def conventions(head, data):
    """The three readings of Q5, on all 133,776 rows.

    literal            every set flag is a place (p27's published rule)
    recoding_aware     code 8 beside a real code is the residue of recoding
    corrected_literal  literal, on the file the authors say they meant to
                       release: Q5_8 is set wherever Q5_8_etc carries text
    """
    q5i = [(i, int(c[3:])) for i, c in enumerate(head)
           if c.startswith("Q5_") and c[3:].isdigit()]
    etc_i = head.index("Q5_8_etc")
    n = len(data)
    lit = aware = corr = 0
    hh_lit = hh_aware = hh_corr = 0
    # the confirmed CSV error: free text present, Q5_8 absent
    err = 0
    err_by_lowest = {}
    for row in data:
        codes = {c for i, c in q5i if row[i].strip() != ""}
        has_text = row[etc_i].strip() != ""
        if has_text and ETC not in codes:
            err += 1
            low = min(codes) if codes else 0
            err_by_lowest[low] = err_by_lowest.get(low, 0) + 1
        ccodes = codes | ({ETC} if has_text else set())
        others = codes - {ETC}
        eff = others if (ETC in codes and others) else codes
        if len(codes) > 1:
            lit += 1
            hh_lit += 1 in codes
        if len(eff) > 1:
            aware += 1
            hh_aware += 1 in eff
        if len(ccodes) > 1:
            corr += 1
            hh_corr += 1 in ccodes
    mk = lambda k, h: dict(multi_place_share=k / n, n_multi=k,
                           household_plus_other_share=h / n)
    return n, dict(literal=mk(lit, hh_lit), recoding_aware=mk(aware, hh_aware),
                   corrected_literal=mk(corr, hh_corr)), err, err_by_lowest


def q9_profile(head, data):
    """Question 3's counts. Every one of them is quoted in the sent letter and
    produced by no script until now."""
    i = head.index("Q9")
    v = []
    for row in data:
        s = row[i].strip()
        v.append(float(s) if s not in ("", "NA") else np.nan)
    a = np.array(v)
    fin = np.isfinite(a)
    return dict(n=len(a), n_missing=int((~fin).sum()),
                n_zero=int((a[fin] == 0).sum()), n_one=int((a[fin] == 1).sum()),
                share_one=float((a[fin] == 1).mean()),
                n_below_two=int((a[fin] < 2).sum()),
                share_below_two=float((a[fin] < 2).mean()),
                maximum=float(a[fin].max()), n_at_100=int((a[fin] == 100).sum()),
                n_at_max=int((a[fin] == a[fin].max()).sum()),
                median=float(np.median(a[fin])))


def household_arms(ego, d, p26, p27):
    """Section 65.4: move another-household contacts from H into E and remeasure.

    The panel rule keeps the LOWEST set code, so `panel == "H"` is exactly
    `Q5_1 is set`. The three declared arms select inside that set on the free
    text alone; nothing else about the estimator changes.
    """
    pops = {int(k): np.array(v, float) for k, v in p26["population"].items()}
    sq = [i for i, a in enumerate(AGES) if a < 80]     # no 80+ egos exist
    base_row = {(r["ym"], r["restriction"]): r for r in p27["comparison"]}

    def r_of(dd_all, ym, seoul):
        dows = holiday_free_dows(ym)
        dd = dd_all[dd_all.panel.isin(["W", "E"])]
        cube, eb, nd = ego_cubes(ego, dd, seoul, ym, dows)
        C, _ = contact_matrix(cube, eb, nd)
        T, _asym = symmetrise(C[np.ix_(sq, sq)], pops[ym][sq])
        return assortativity(T)

    txt = d.Q5_8_etc.fillna("").astype(str).str.strip()
    is_h = d.panel == "H"
    # Table S6 matches on the whitespace-stripped string, because the PDF the
    # table was read from wraps entries across lines: "그 친구 집" and "그 친구집"
    # are one entry there and must be one entry here.
    flat = txt.map(lambda s: re.sub(r"\s+", "", s))
    in_s6 = flat.isin(S6_HOUSEHOLD)
    sel = {
        "letter_four": is_h & txt.apply(lambda s: any(k in s for k in LETTER_FOUR)),
        "kinship": is_h & txt.apply(lambda s: any(k in s for k in KINSHIP)),
        "any_free_text": is_h & (txt != ""),
        # a residence that is not the respondent's own home, per the authors
        "s6_household": is_h & (txt != "") & in_s6,
        # code 1 and a free text S6 does not list. MEASURED, NOT ASSUMED, and
        # the measurement went against the obvious reading: the top of this arm
        # is 큰집 / 외할아버지댁 / 친가 / 외할머니집, which are plainly residences
        # that S6 simply does not list. So this is NOT "the genuine two-place
        # contacts" -- it is those PLUS the residences S6 left out, and the
        # completeness diagnostic below puts a number on the second part.
        "s6_other": is_h & (txt != "") & ~in_s6,
    }
    assert int(sel["s6_household"].sum()) + int(sel["s6_other"].sum()) \
        == int(sel["any_free_text"].sum()), \
        "p65: the S6 split does not partition the free-text rows"

    base = {}
    for ym in SURVEY_MONTHS:
        for scope, so in (("seoul", True), ("national", False)):
            base[f"{ym}|{scope}"] = r_of(d, ym, so)

    # THE ANCHOR. Nothing has been moved yet, so this must be p27's own number.
    pub = base_row[(202312, "holiday-free")]["assort_survey"]
    assert abs(base["202312|seoul"] - pub) < 1e-15, (
        f"p65: the unmodified panel assignment gives {base['202312|seoul']!r}, "
        f"not p27's {pub!r} -- this is not p27's cell and the deltas below are "
        "deltas from something else")
    pub2 = base_row[(202402, "holiday-free")]["assort_survey"]
    assert abs(base["202402|seoul"] - pub2) < 1e-15, "p65: 202402 anchor failed"

    arms = {}
    for name, mask in sel.items():
        d2 = d.copy()
        d2.loc[mask, "panel"] = "E"
        cells = {}
        for ym in SURVEY_MONTHS:
            for scope, so in (("seoul", True), ("national", False)):
                k = f"{ym}|{scope}"
                r = r_of(d2, ym, so)
                p = base_row[(ym, "holiday-free")]["assort_passive"]
                cells[k] = dict(assort_survey=r, delta=r - base[k],
                                rel_delta=(r - base[k]) / base[k],
                                assort_passive=p, ratio=r / p,
                                ratio_base=base[k] / p)
        arms[name] = dict(n_moved=int(mask.sum()),
                          share_of_contacts=float(mask.mean()),
                          share_of_H=float(mask.sum() / max(int(is_h.sum()), 1)),
                          cells=cells)

    unmatched = sorted(
        txt[is_h & (txt != "") & ~sel["kinship"]].value_counts()
        .head(25).items(), key=lambda kv: -kv[1])
    off_s6 = sorted(txt[sel["s6_other"]].value_counts().head(25).items(),
                    key=lambda kv: -kv[1])
    # How incomplete is S6's household list? A residence-shaped string is one
    # ending in 집 (house), 댁 (honorific house) or 자택 (residence). This is a
    # DIAGNOSTIC and feeds no reported assortativity; it exists so "S6 is
    # incomplete" is a count rather than an impression.
    resid = flat.str.endswith(("집", "댁", "자택"))
    s6_missing = int((sel["s6_other"] & resid).sum())
    miss_top = sorted(txt[sel["s6_other"] & resid].value_counts().head(15).items(),
                      key=lambda kv: -kv[1])
    return dict(
        baseline=base, anchor=dict(p27_published=pub, recomputed=base["202312|seoul"],
                                   bit_exact=True),
        contacts_in_d=int(len(d)), n_household_panel=int(is_h.sum()),
        n_household_with_text=int((is_h & (txt != "")).sum()),
        arms=arms,
        s6=dict(n_entries=S6["n_entries"], pdf_sha256=S6["pdf_sha256"],
                retrieved=S6["retrieved"],
                residence_shaped_but_unlisted=s6_missing,
                residence_shaped_but_unlisted_top=[dict(text=k, n=int(v))
                                                   for k, v in miss_top]),
        unmatched_by_kinship=[dict(text=k, n=int(v)) for k, v in unmatched],
        not_in_s6_household=[dict(text=k, n=int(v)) for k, v in off_s6])


def main():
    head, data = rows()
    n, conv, err, err_by_lowest = conventions(head, data)
    p57 = json.load(open(f"{ROOT}/eda/results_p57.json"))
    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))
    p26 = json.load(open(f"{ROOT}/eda/results_p26.json"))

    print("=== 65.0 anchors: p57's two conventions, off the same rows ===")
    for k in ("literal", "recoding_aware"):
        mine, theirs = conv[k]["multi_place_share"], p57["conventions"][k]["multi_place_share"]
        print(f"  {k:<16} {mine:.17f}  vs p57 {theirs:.17f}")
        assert abs(mine - theirs) < 1e-15, f"p65: {k} does not reproduce p57"
    assert n == p27["place_multiplicity"]["contacts"] == 133776, n
    print(f"  contacts {n:,} -- both anchors hold")

    print("\n=== 65.1 the second coding error the reply confirms ===")
    print(f"  rows with free text in Q5_8_etc and Q5_8 NOT set: {err:,}")
    for low in sorted(err_by_lowest):
        print(f"    of which lowest set code {low or '(none)'}: "
              f"{err_by_lowest[low]:,}")
    print("  Per the reply these rows should carry code 8; the authors intend to")
    print("  correct the released CSV, so this is v1's error and not a reading.")

    print("\n=== 65.2 the three conventions ===")
    for k in ("recoding_aware", "literal", "corrected_literal"):
        c = conv[k]
        print(f"  {k:<18} multi-place {c['multi_place_share']:8.4%}  "
              f"(n = {c['n_multi']:,})  household+other "
              f"{c['household_plus_other_share']:.4%}")
    assert conv["recoding_aware"]["n_multi"] == p57["conventions"]["recoding_aware"]["n_multi"], \
        "p65: the correction moved the recoding-aware reading, which it cannot"
    print("  The correction leaves recoding_aware untouched by construction, and")
    print("  that is the point: on the corrected file the literal rule counts a")
    print("  free-text flag rather than a place, so the bracket is")
    print(f"  {conv['recoding_aware']['multi_place_share']:.2%} to "
          f"{conv['corrected_literal']['multi_place_share']:.2%}.")

    q9 = q9_profile(head, data)
    print("\n=== 65.3 Q9, whose 0 and 1 the reply calls errors ===")
    print(f"  Q9 = 1: {q9['n_one']:,} ({q9['share_one']:.1%})   Q9 = 0: "
          f"{q9['n_zero']:,}   below the definition's minimum of 2: "
          f"{q9['n_below_two']:,} ({q9['share_below_two']:.1%})")
    print(f"  maximum {q9['maximum']:.0f} ({q9['n_at_max']:,} record(s)); "
          f"{q9['n_at_100']:,} records sit at exactly 100; median {q9['median']:.0f}")
    print("  No cleaning and no top-coding were applied, so the tail is reported")
    print("  and not editorial. p44 section 44.7's padding arm stays withdrawn.")

    ego, d = load_survey()
    hh = household_arms(ego, d, p26, p27)
    print("\n=== 65.4 what code 1 covering other households costs the comparison ===")
    print(f"  anchor: unmodified panels reproduce p27's survey assortativity "
          f"{hh['anchor']['p27_published']:.17f}")
    print(f"  contacts in panel H: {hh['n_household_panel']:,}, of which "
          f"{hh['n_household_with_text']:,} carry free text")
    print(f"  {'arm':<15} {'moved':>7} {'202312 seoul':>14} {'delta':>9} "
          f"{'survey/passive':>15}")
    b = hh["baseline"]["202312|seoul"]
    bp = hh["arms"]["any_free_text"]["cells"]["202312|seoul"]["assort_passive"]
    print(f"  {'(baseline)':<15} {0:>7} {b:>14.5f} {0.0:>9.5f} {b / bp:>15.2f}")
    for name, a in hh["arms"].items():
        c = a["cells"]["202312|seoul"]
        print(f"  {name:<15} {a['n_moved']:>7,} {c['assort_survey']:>14.5f} "
              f"{c['delta']:>+9.5f} {c['ratio']:>15.2f}")
    print("  The passive side already counts these trips: its destination")
    print("  attribute is the traveller's own 야간상주지, so another household is")
    print("  an E destination there and an H contact here. Moving them to E is")
    print("  what makes the two sides classify the same event the same way.")
    print(f"\n  Table S6 ({hh['s6']['n_entries']} household mappings, retrieved "
          f"{hh['s6']['retrieved']}) splits the primary arm into what the")
    print("  authors' own table lists as a residence, and what it does not:")
    print(f"    s6_household {hh['arms']['s6_household']['n_moved']:>6,}"
          f"    s6_other {hh['arms']['s6_other']['n_moved']:>6,}")
    print("\n  free text on code-1 rows that the kinship arm does NOT match, top 25")
    print("  (kept as declared: Table S6 shows this arm undercounts, and that is")
    print("  evidence about the guess, not a reason to delete it):")
    for u in hh["unmatched_by_kinship"]:
        print(f"    {u['n']:>5,}  {u['text']}")
    print(f"\n  ...but S6's household list is INCOMPLETE: {hh['s6']['residence_shaped_but_unlisted']:,} "
          f"of the {hh['arms']['s6_other']['n_moved']:,} rows it does")
    print("  not list end in 집 / 댁 / 자택, i.e. name a residence. s6_household is")
    print("  therefore a LOWER bound and s6_other is a mixture, not a category.")
    print("  Residence-shaped and unlisted, top 15:")
    for u in hh["s6"]["residence_shaped_but_unlisted_top"]:
        print(f"    {u['n']:>5,}  {u['text']}")
    print("\n  free text on code-1 rows that Table S6 does not list, top 25 --")
    print("  a mixture: unlisted residences above, and genuine second places")
    print("  (마트, 병원, 놀이터, 영화관, PC방, 산) below them:")
    for u in hh["not_in_s6_household"]:
        print(f"    {u['n']:>5,}  {u['text']}")

    out = dict(
        declaration=declaration,
        anchor=dict(contacts=n, p57_literal=conv["literal"]["multi_place_share"],
                    p57_recoding_aware=conv["recoding_aware"]["multi_place_share"],
                    p57_published_literal=p57["conventions"]["literal"]["multi_place_share"],
                    p57_published_aware=p57["conventions"]["recoding_aware"]["multi_place_share"],
                    bit_exact=True),
        conventions=conv,
        csv_error=dict(rows_text_without_flag=err,
                       by_lowest_code={str(k): v for k, v in sorted(err_by_lowest.items())}),
        q9=q9,
        household=hh)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
