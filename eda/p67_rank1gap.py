#!/usr/bin/env python
"""Phase 67 — the distance between the best rank-one approximation and the
proportionate-mixing null, measured as a distance rather than as a difference
of two distances.

WHY THIS EXISTS. The advisor's 2026-09-05 letter, point 3.2. Until 2026-09-07
the abstract said the best rank-one approximation "departs from the outer
product of its own margins by at most 0.00202". That number is produced at
`p37_timeseries.py:226-227` as

    gap_to_pm_max = max_over_months | best_rank1_resid - pm_rank1_resid |

which is the difference of two NORMS, not the norm of a difference. Writing
X = sigma_1 u_1 u_1' and E = r (x) r, the triangle inequality gives

    | ||e - X|| - ||e - E|| |  <=  ||X - E||

so the published quantity is a LOWER bound on the distance the sentence claims
as an UPPER bound. The direction of the inequality is the whole defect: the
sentence is not merely imprecise, it points the wrong way.

WHAT THIS PHASE COMPUTES. ||X - E||_F / ||e||_F on all 79 monthly dong-level
matrices, in the same relative units as the two residuals it replaces, so the
new number can be read beside the old ones without a change of scale. The
absolute form is reported too, because a relative Frobenius distance divided by
a small ||e|| is easy to misread.

WHY NO PARQUET IS NEEDED, WHICH IS NOT OBVIOUS. `p37_timeseries.py:104` builds
its matrix as `symmetrise(A[sq,sq] / pop[sq][:,None], pop[sq])`, and
`p32_pmix.py:symmetrise` multiplies by `pop[:,None]` again on the way in. The
population divides out exactly. So the whole rank-one calculation is a function
of the stored `A` alone, and `results_p37.json` already stores all 79 of them
(`p37_timeseries.py:118`). Anchor 67.0 exploits that: this phase reproduces
every stored `best_rank1_resid` and `pm_rank1_resid` to 1e-12 before it
computes anything new. The tolerance is not decoration and the claim is not bit
equality: the worst deviation over the 316 quantities is 9.6e-16, one ulp of the
SVD, which is why the results file records `bit_equal` alongside `max_abs_dev`
instead of asserting the stronger thing. If the cancellation were wrong, 158
anchors would fire.

WHAT THIS PHASE DOES NOT DO. It does not touch `results_p37.json`, and it does
not re-derive the residuals themselves -- `rank_stats` and `pm_null` are
imported from `p32_pmix` unchanged, so the object being measured is the same
object the paper reports, at the same 15 bands below 80 that `p37` uses.

    python eda/p67_rank1gap.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import AGES
from p32_pmix import pm_null, rank_stats
from paths import ROOT

PANEL, LEVEL = "WE", "dong"          # p37's stored panel; declared, not chosen
GAP_MEDIAN_PUB = 0.001503531770104094     # results_p37 spectrum_dong; anchor
GAP_MAX_PUB = 0.00201621420069259         # results_p37 spectrum_dong; anchor

out = {}


def say(s=""):
    print(s, flush=True)


def main():
    with open(f"{ROOT}/eda/results_p37.json") as fh:
        p37 = json.load(fh)
    sq = [i for i, a in enumerate(AGES) if a < 80]
    stored = {r["ym"]: r for r in p37["rows"]
              if r["panel"] == PANEL and r["level"] == LEVEL}
    mats = p37["matrices_we_dong"]

    say(f"=== 67.0 anchor: reproduce every stored residual from the stored A ===")
    rows, worst = [], 0.0
    for ym in sorted(mats, key=int):
        A = np.array(mats[ym]["A"], float)[np.ix_(sq, sq)]
        T = (A + A.T) / 2                       # the population cancels; see above
        rs = rank_stats(T)
        st = stored[int(ym)]
        for k in ("best_rank1_resid", "pm_rank1_resid", "sigma1_share",
                  "sigma2_over_sigma1"):
            d = abs(rs[k] - st[k])
            worst = max(worst, d)
            assert d < 1e-12, (
                f"{ym} {k}: recomputed {rs[k]!r} against stored {st[k]!r}. "
                f"p37's population weighting does not cancel after all, and "
                f"every number below would be computed on a different matrix")

        e, r, E = pm_null(T)
        u, s, vt = np.linalg.svd(e)
        X = s[0] * np.outer(u[:, 0], vt[0])
        ne = float(np.linalg.norm(e))
        rows.append(dict(
            ym=int(ym),
            best_rank1_resid=rs["best_rank1_resid"],
            pm_rank1_resid=rs["pm_rank1_resid"],
            resid_difference=abs(rs["best_rank1_resid"] - rs["pm_rank1_resid"]),
            gap_rel=float(np.linalg.norm(X - E) / ne),
            gap_abs=float(np.linalg.norm(X - E)),
            e_norm=ne))
    say(f"  {len(rows)} months x 4 quantities reproduced, worst deviation "
        f"{worst:.3e}   ok")
    out["anchor"] = dict(n_months=len(rows), n_quantities=4 * len(rows),
                         max_abs_dev=worst, bit_equal=worst == 0.0)

    diff = np.array([r["resid_difference"] for r in rows])
    rel = np.array([r["gap_rel"] for r in rows])
    ab = np.array([r["gap_abs"] for r in rows])

    # 67.0b -- and the published summary of those differences, so the object
    # being corrected is demonstrably the published one.
    assert abs(float(np.median(diff)) - GAP_MEDIAN_PUB) < 1e-12
    assert abs(float(diff.max()) - GAP_MAX_PUB) < 1e-12
    say(f"  67.0b  published gap_to_pm median {np.median(diff):.5f} and max "
        f"{diff.max():.5f} reproduced   ok")

    say("\n=== 67.1 the difference of norms against the norm of the difference ===")
    say(f"  {'quantity':<46} {'median':>9} {'max':>9} {'min':>9}")
    say(f"  {'| ||e-X|| - ||e-E|| |  (published)':<46} "
        f"{np.median(diff):>9.5f} {diff.max():>9.5f} {diff.min():>9.5f}")
    say(f"  {'||X - E|| / ||e||     (the distance)':<46} "
        f"{np.median(rel):>9.5f} {rel.max():>9.5f} {rel.min():>9.5f}")
    say(f"  {'||X - E||             (absolute)':<46} "
        f"{np.median(ab):>9.6f} {ab.max():>9.6f} {ab.min():>9.6f}")
    say(f"\n  the published maximum understates the distance by a factor of "
        f"{rel.max() / diff.max():.1f}.")
    say(f"  every month satisfies the triangle inequality "
        f"|d1 - d2| <= ||X - E||: "
        f"{'yes' if bool((diff <= rel + 1e-15).all()) else 'NO'}")

    out["per_month"] = rows
    out["summary"] = dict(
        n_months=len(rows), panel=PANEL, level=LEVEL, bands="under 80 (15 of 16)",
        resid_difference_median=float(np.median(diff)),
        resid_difference_max=float(diff.max()),
        gap_rel_median=float(np.median(rel)),
        gap_rel_max=float(rel.max()),
        gap_rel_min=float(rel.min()),
        gap_abs_median=float(np.median(ab)),
        gap_abs_max=float(ab.max()),
        understatement_factor=float(rel.max() / diff.max()),
        triangle_inequality_holds=bool((diff <= rel + 1e-15).all()),
        argmax_month=int(rows[int(np.argmax(rel))]["ym"]),
        argmin_month=int(rows[int(np.argmin(rel))]["ym"]),
    )

    say("\n=== the number ===")
    say(f"  across all 79 months the best rank-one approximation departs from "
        f"the outer")
    say(f"  product of its own margins by at most {rel.max():.5f} in relative "
        f"Frobenius norm")
    say(f"  ({rows[int(np.argmax(rel))]['ym']}), with a median of "
        f"{np.median(rel):.5f}.")
    say(f"  the manuscript quotes this number. until 2026-09-07 it quoted "
        f"0.00202 instead,")
    say(f"  which is the difference of two norms and so a LOWER bound on this "
        f"quantity, then")
    say(f"  quoted as an upper bound. SI 3 now names it as the lower bound it "
        f"is.")

    out["declaration"] = dict(
        quantity="||sigma_1 u_1 u_1' - r (x) r||_F, relative to ||e||_F",
        why_relative="the two residuals it replaces are relative to ||e||_F "
                     "(p32_pmix.py:167-168), so the replacement is quoted in "
                     "the same units and can be read beside them",
        source="results_p37.json matrices_we_dong, all 79 months; no parquet, "
               "no re-run, no results file rewritten",
        bands="p37's own sub-80 block, so the object is p37's object",
        stop_loss="abort unless all 79 months reproduce p37's stored "
                  "best_rank1_resid, pm_rank1_resid, sigma1_share and "
                  "sigma2_over_sigma1 to 1e-12",
    )

    with open(f"{ROOT}/eda/results_p67.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {ROOT}/eda/results_p67.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
