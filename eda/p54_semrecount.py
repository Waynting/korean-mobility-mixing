#!/usr/bin/env python
"""Phase 54 — the 13-of-79 count, re-counted against the corrected floor.

WHY THIS EXISTS. `p42_monthscope.py` counts how many of the 79 months carry a
passive reading at or above the national permutation floor, and finds 13, all of
them school-term months, p = 2.24e-04. The advisor's 2026-08-24 letter promotes
that from a wound to Claim 1's independent corroboration.

But the floor it counts against is `results_p40.json`'s national permutation
median -- `p31_report_audit.py:1317` asserts they are the same float, bit for
bit, precisely so the two can never drift. `p51_natsym.py` corrects that median,
because p40 built the national arm on Seoul's population vector. A corrected
floor is a different floor, and 13 is a count against a floor. So the sentence
the advisor wants moved into Claim 1 cannot be written until it is re-counted.

THE COUNT IS CHEAP AND THE MATRICES ARE NOT REBUILT. `results_p42.json` already
stores all 79 monthly readings (`rows[*].mi_bits_hf`), each on the holiday-free
day set that the floor is attached to -- aligning those two day sets is the
whole reason p42 exists. Re-counting is therefore a comparison against a
different scalar, not a new measurement, and re-deriving the matrices would risk
introducing a difference where the point is that there is none.

WHAT IS ANCHORED. Re-counting against p42's OWN floor must reproduce every
number p42 published: the count of 13, the term/vacation/December split, the
base rates, and the exact within-year p-value. If it does not, this file is
not applying p42's rule and its corrected count means nothing. The null itself
is imported from p42 rather than rewritten -- it is an exact hypergeometric
convolution, no seed, no simulation error, and re-deriving it here would be a
second chance to get it wrong.

    python eda/p54_semrecount.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from paths import ROOT
from p42_monthscope import within_year_null

out = {}


def say(s=""):
    print(s, flush=True)


def main():
    p42 = json.load(open(f"{ROOT}/eda/results_p42.json"))
    p51 = json.load(open(f"{ROOT}/eda/results_p51.json"))
    p50 = json.load(open(f"{ROOT}/eda/results_p50.json"))

    rows = p42["rows"]
    assert len(rows) == 79, f"expected 79 months, got {len(rows)}"


    def tally(floor):
        """p42's rule: a month clears iff its holiday-free mi_bits >= the floor."""
        clearing = [r for r in rows if r["mi_bits_hf"] >= floor]
        obs = {lab: sum(1 for r in clearing if r["semester"] == lab)
               for lab in ("term", "vacation", "december")}
        # Counted explicitly per year, because the null is within-year: the level
        # trends across 2020-2026 and a pooled shuffle would read that trend as a
        # cycle. See within_year_null's docstring in p42.
        per_year_term, per_year_vac = {}, {}
        years = sorted({r["year"] for r in rows})
        for y in years:
            yr = [r for r in rows if r["year"] == y]
            cl = [r for r in yr if r["mi_bits_hf"] >= floor]
            per_year_term[y] = (len(yr), sum(1 for r in yr
                                             if r["semester"] == "term"), len(cl))
            per_year_vac[y] = (len(yr), sum(1 for r in yr
                                            if r["semester"] == "vacation"), len(cl))
        _, p_term, _ = within_year_null(per_year_term, obs["term"])
        _, _, p_novac = within_year_null(per_year_vac, obs["vacation"])
        return dict(floor=float(floor), n_at_or_above=len(clearing), observed=obs,
                    months=[r["ym"] for r in clearing],
                    p_term_ge=float(p_term), p_vacation_le=float(p_novac),
                    all_clearing_are_term=bool(obs["term"] == len(clearing)))


    # ==================================================================== 54.0
    say("=== 54.0 ANCHOR: p42's own floor must give back p42's own numbers ===")
    floor_pub = p42["counts"]["202312|national"]["floor"]
    a = tally(floor_pub)
    want_n = p42["counts"]["202312|national"]["n_at_or_above"]
    want_obs = p42["semester"]["observed"]
    want_p = p42["semester"]["within_year"]["p_term_ge"]
    say(f"  floor {floor_pub:.9f}")
    say(f"  count      {a['n_at_or_above']:>4}   p42 says {want_n:>4}")
    say(f"  term       {a['observed']['term']:>4}   p42 says {want_obs['term']:>4}")
    say(f"  vacation   {a['observed']['vacation']:>4}   p42 says "
        f"{want_obs['vacation']:>4}")
    say(f"  december   {a['observed']['december']:>4}   p42 says "
        f"{want_obs['december']:>4}")
    say(f"  p(term)    {a['p_term_ge']:.6e}   p42 says {want_p:.6e}   "
        f"|diff| {abs(a['p_term_ge'] - want_p):.1e}")
    assert a["n_at_or_above"] == want_n, "the recount does not reproduce p42's count"
    assert a["observed"] == want_obs, "the label split does not reproduce p42's"
    assert abs(a["p_term_ge"] - want_p) < 1e-15, "the null does not reproduce p42's"
    assert sorted(a["months"]) == sorted(
        p42["counts"]["202312|national"]["months"]), "different months clear"
    out["anchor"] = dict(floor=floor_pub, reproduced=True, count=want_n,
                         p_term_ge=want_p)

    # ==================================================================== 54.1
    say("\n=== 54.1 the corrected floor ===")
    cell = p51["cells"]["202312|national"]
    floor_new = cell["mi_perm_median"]
    se_new = cell["mi_perm_median_se"]
    say(f"  p40 published floor : {floor_pub:.9f}")
    say(f"  p51 corrected floor : {floor_new:.9f}   "
        f"({100 * (floor_new / floor_pub - 1):+.2f}%)")
    say(f"  the corrected floor is LOWER, so at least as many months clear it.")
    b = tally(floor_new)
    say(f"\n  count      {b['n_at_or_above']:>4}   (was {want_n})")
    say(f"  term       {b['observed']['term']:>4}")
    say(f"  vacation   {b['observed']['vacation']:>4}")
    say(f"  december   {b['observed']['december']:>4}")
    say(f"  p(term)    {b['p_term_ge']:.6e}   (was {want_p:.6e})")
    say(f"  every clearing month a term month: {b['all_clearing_are_term']}")
    out["corrected"] = b
    out["published"] = a

    # ==================================================================== 54.2
    say("\n=== 54.2 the count against the floor's own uncertainty ===")
    say("  13 was never a natural break, so p42 reported the count at +/- 2 SE of")
    say("  the floor and this does the same with p51's SE.")
    sens = {}
    for k, off in (("-2se", -2), ("-1se", -1), ("+0se", 0), ("+1se", 1),
                   ("+2se", 2)):
        t = tally(floor_new + off * se_new)
        sens[k] = dict(floor=floor_new + off * se_new, n=t["n_at_or_above"],
                       term=t["observed"]["term"],
                       all_term=t["all_clearing_are_term"])
        say(f"  {k:>5}  floor {sens[k]['floor']:.9f}  count {sens[k]['n']:>3}  "
            f"term {sens[k]['term']:>3}  all term: {sens[k]['all_term']}")
    out["floor_sensitivity"] = dict(se=se_new, counts_at_offsets=sens)
    out["published_floor_sensitivity"] = p42["floor_sensitivity"]["counts_at_offsets"]

    # ==================================================================== 54.3
    say("\n=== 54.3 the two survey months, which is the honest half ===")
    say("  The advisor's letter keeps this and so does this file: the months the")
    say("  survey was run in are NOT among the ones that clear, and they rank")
    say("  below most term months. That is the sentence that costs something.")
    term_rows = sorted([r for r in rows if r["semester"] == "term"],
                       key=lambda r: r["mi_bits_hf"])
    n_term = len(term_rows)
    survey_ranks = {}
    for ym in (202312, 202402):
        r = [x for x in rows if x["ym"] == ym][0]
        below = sum(1 for x in term_rows if x["mi_bits_hf"] > r["mi_bits_hf"])
        survey_ranks[str(ym)] = dict(
            mi_bits=r["mi_bits_hf"], semester=r["semester"],
            clears_corrected=bool(r["mi_bits_hf"] >= floor_new),
            clears_published=bool(r["mi_bits_hf"] >= floor_pub),
            n_term_months_above=below, n_term_months=n_term)
        say(f"  {ym}: {r['mi_bits_hf']:.6f} ({r['semester']}), below "
            f"{below} of the {n_term} term months; clears the corrected floor: "
            f"{r['mi_bits_hf'] >= floor_new}")
    out["survey_months"] = survey_ranks

    out["declaration"] = dict(
        rule="a month clears iff its holiday-free mi_bits >= the floor -- p42's, "
             "unchanged",
        floor_published=floor_pub, floor_corrected=floor_new,
        floor_source="results_p51.json cells['202312|national'].mi_perm_median",
        series_source="results_p42.json rows[*].mi_bits_hf, not rebuilt",
        null="p42's exact within-year hypergeometric convolution, imported",
        established_202312=bool(p50["vectors"]["202312"]["established"]))

    with open(f"{ROOT}/eda/results_p54.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p54.json")


if __name__ == "__main__":
    main()
