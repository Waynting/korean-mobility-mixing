#!/usr/bin/env python
"""Phase 57 — Q5's retained "etc." flag, and what it did to p27's multiplicity.

WHY THIS EXISTS. `results_p27.json` reports
`place_multiplicity.multi_place_share = 0.1053`, quoted in
`eda/memo/phase7-survey.md` as "10.5% of contacts carry more than one place".
Reading [CHAE]'s released files closely enough to write the 2026-08-24 enquiry
to NIMS turned that number up as wrong, and wrong in a way that is invisible
from the JSON: every one of the 2,016 rows coded 6 (hospital) and every one of
the 3,210 rows coded 7 (outdoor) ALSO has `Q5_8` set with Korean free text
beside it. That is researcher recoding of an "etc." answer with the original
code 8 left in place rather than cleared, and p27 counts the residue as a second
place. Two entire location categories are therefore 100% classified as
multi-place, which is not a property of the data but of the flag.

WHY IT IS A NEW PHASE AND NOT AN EDIT TO p27. `results_p27.json` is read-only
for this round (CLAUDE.md, `eda/archive/README.md`), and p27's matrices are
anchored by p32, p44 and p53. This phase touches nothing: it re-reads the same
CSV and writes its own file.

WHAT IS ANCHORED. Under the literal convention -- any row with two or more Q5
flags is multi-place -- this file must reproduce p27's published
`multi_place_share` and `household_plus_other_share` to the last bit. If it does
not, it is not measuring the same thing p27 measured and the corrected number
below means nothing. The anchor is read from `results_p27.json` itself rather
than typed here, so the two can never drift apart.

WHAT IS NOT SETTLED. Which convention is RIGHT is question 1 of the enquiry sent
to NIMS on 2026-08-24 (`Email_Discussion/nims_enquiry_20260824.md`). Both are
computed and both are reported; the recoding-aware reading is marked primary
because the descriptor calls Q5 a single-response item and the Technical
Validation describes exactly this recoding, but that is a reading and it is
labelled as one. Nothing downstream consumes this file yet -- it exists so the
number is measured before the reply rather than after it.

    python eda/p57_placecode.py
"""
import csv
import io as _io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT, ROOT, require

CHAE = DATA_ROOT / "raw" / "chae2026"
ETC = 8                       # the "etc." code that recoding leaves behind
OUT = f"{ROOT}/eda/results_p57.json"

declaration = dict(
    primary_convention="recoding_aware",
    rule="a row whose Q5 flags are {8, x} with x != 8 is one place, namely x",
    settled_by="question 1 of the 2026-08-24 NIMS enquiry; unanswered at run time",
    touches_nothing=["results_p26.json", "results_p27.json", "results_p44.json"])


def rows():
    require(CHAE, "Chae microdata (run eda/dl_chae.py)")
    t = (CHAE / "results_main_survey.csv").read_bytes().decode("utf-8-sig")
    r = list(csv.reader(_io.StringIO(t)))
    head = [c.strip() for c in r[0]]
    return head, r[1:]


def main():
    head, data = rows()
    q5i = [(i, int(c[3:])) for i, c in enumerate(head)
           if c.startswith("Q5_") and c[3:].isdigit()]
    etc_txt = head.index("Q5_8_etc") if "Q5_8_etc" in head else None

    literal, aware, hh_lit, hh_aware = 0, 0, 0, 0
    per_code_with_etc = {}          # code c AND code 8, however many codes
    per_code_etc_only_other = {}    # code c AND code 8, and nothing else
    etc_with_exactly_one_other = 0
    n = 0
    for row in data:
        n += 1
        codes = {c for i, c in q5i if row[i].strip() != ""}
        others = codes - {ETC}
        # literal: p27's rule, every set flag is a place
        if len(codes) > 1:
            literal += 1
            if 1 in codes:
                hh_lit += 1
        # recoding-aware: the etc residue is dropped whenever a real code is set
        eff = others if (ETC in codes and others) else codes
        if len(eff) > 1:
            aware += 1
            if 1 in eff:
                hh_aware += 1
        if ETC in codes:
            # The claim the enquiry letter makes is about code c carrying the
            # flag AT ALL, not about c being the only other code, so both are
            # counted: the letter says "all 2,016 hospital rows", and a
            # narrower statistic would understate it and read as a correction.
            for c in others:
                per_code_with_etc[c] = per_code_with_etc.get(c, 0) + 1
            if len(others) == 1:
                etc_with_exactly_one_other += 1
                per_code_etc_only_other[sorted(others)[0]] = \
                    per_code_etc_only_other.get(sorted(others)[0], 0) + 1

    # per-code totals, so "100% of hospital and outdoor rows" is a measured claim
    totals = {}
    for row in data:
        for i, c in q5i:
            if row[i].strip() != "":
                totals[c] = totals.get(c, 0) + 1

    p27 = json.load(open(f"{ROOT}/eda/results_p27.json"))["place_multiplicity"]
    lit_share, aware_share = literal / n, aware / n

    print("=== 57.0 anchor: p27's published multiplicity, on the literal rule ===")
    print(f"  contacts                      {n:,}  vs p27 {p27['contacts']:,}")
    print(f"  multi_place_share  literal    {lit_share:.17f}")
    print(f"                     p27        {p27['multi_place_share']:.17f}")
    assert n == p27["contacts"], "p57: not the same row set p27 read"
    assert abs(lit_share - p27["multi_place_share"]) < 1e-15, \
        "p57: the literal rule no longer reproduces p27 -- this file is not " \
        "measuring what p27 measured, so its correction means nothing"
    assert abs(hh_lit / n - p27["household_plus_other_share"]) < 1e-15, \
        "p57: household_plus_other_share no longer reproduces p27"
    print("  household_plus_other_share reproduced too -- anchor holds")

    print("\n=== 57.1 what the retained etc. flag is worth ===")
    print(f"  rows carrying code 8 with exactly one other code: "
          f"{etc_with_exactly_one_other:,}")
    print("  by code -- 'with flag' is code c and code 8 both set:")
    for c in sorted(per_code_with_etc):
        tot = totals.get(c, 0)
        print(f"    code {c}: {per_code_with_etc[c]:,} of {tot:,} rows "
              f"({per_code_with_etc[c] / tot:7.2%}), of which "
              f"{per_code_etc_only_other.get(c, 0):,} carry no third code")
    print(f"\n  multi-place share, literal (p27, published) : {lit_share:.4%}")
    print(f"  multi-place share, recoding-aware (primary) : {aware_share:.4%}")
    print(f"  the flag alone accounts for {lit_share - aware_share:.4%} of contacts")
    print(f"  household + other, literal / aware          : "
          f"{hh_lit / n:.4%} / {hh_aware / n:.4%}")
    print("\n  Which convention is right is question 1 of the NIMS enquiry sent")
    print("  2026-08-24. Both are reported; neither is adopted downstream yet.")

    out = dict(
        declaration=declaration,
        anchor=dict(contacts=n, literal_share=lit_share,
                    p27_published=p27["multi_place_share"],
                    household_literal=hh_lit / n,
                    p27_household_published=p27["household_plus_other_share"],
                    bit_exact=True),
        conventions=dict(
            literal=dict(multi_place_share=lit_share,
                         household_plus_other_share=hh_lit / n,
                         n_multi=literal),
            recoding_aware=dict(multi_place_share=aware_share,
                                household_plus_other_share=hh_aware / n,
                                n_multi=aware)),
        etc_flag=dict(rows_with_etc_and_exactly_one_other=etc_with_exactly_one_other,
                      by_code={str(k): dict(with_etc=v, total=totals.get(k, 0),
                                            share=v / totals[k],
                                            etc_only_other=per_code_etc_only_other.get(k, 0))
                               for k, v in sorted(per_code_with_etc.items())},
                      share_of_contacts_explained=lit_share - aware_share))
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
