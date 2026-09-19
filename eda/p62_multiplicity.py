#!/usr/bin/env python
"""Phase 62 — the size of p52's flip rule, and what the global null really predicts.

WHY. The advisor's 2026-08-27 letter, point 5, wants us to get ahead of the
R0 = 3.5 circle in Figure S1(b) before a reviewer uses it against us. The framing
instinct is right. The arithmetic offered with it is not: "twenty verdicts at
the 95% level means one false positive in expectation" assumes each verdict is a
size-0.05 test. It is not. p52's rule is non-overlap of two 95% bootstrap
intervals, which is a different and far more conservative object, and its size
has never been measured. This phase measures it.

WHAT IT COSTS. Nothing. No parquet, no DuckDB, no drive. It reads
`results_p52.json` for the rule and the twenty stored verdicts, and everything
else is Monte Carlo on ranks plus exact combinatorics on order statistics.

WHAT IT IS NOT. This phase measures a property of a DECISION RULE, not a
quantity in the world. There is no second route to it through the data, so it
owes `p36_recompute.py` nothing -- the second implementation it does owe is the
exact rank calculation that sits beside the Monte Carlo here, and the two are
required to agree wherever the Monte Carlo has any resolution at all.

    python eda/p62_multiplicity.py
"""
import json
import math
import os
import re
import sys
import zlib
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from paths import ROOT

# =============================================================================
# THE DECLARATION. Authored before any analysis code below it was written, and
# not modified after output was seen. CLAUDE.md: choosing after the fact is
# number-shopping.
# =============================================================================
DECLARATION = dict(
    question=(
        "what is the size of p52's flip rule at n_boot = 200, and how many "
        "flips among 20 verdicts would the global null produce?"),
    method=(
        "Monte Carlo on ranks: draw two independent samples of 200 from any "
        "continuous distribution, evaluate p52's rule verbatim. "
        "Distribution-free by exchangeability. N_MC = 200000, SEED = 20260827, "
        "declared before the run."),
    reported=(
        "alpha_rule; 20 * alpha_rule; 4 * alpha_rule; P(>= 5 flips | 20 "
        "independent) and P(>= 1 flip | 20 independent) under a binomial with "
        "p = alpha_rule."),
    declared_caveat=(
        "the binomial treats the 20 as independent and they are not. The five "
        "R0 within a cell share one survey. The conservative reading uses 4 "
        "independent questions; BOTH are reported."),
)

SEED = 20260827
N_MC = 200_000
# Declared before the run: two continuous distributions are the minimum the
# distribution-free claim has to survive. Two more are carried because they cost
# nothing and a heavy tail is the case where interpolation could bite.
DISTS = ("normal", "exponential", "uniform", "lognormal")
# Small n where the Monte Carlo has resolution at all. At n = 200 the rule's
# size is far below anything 200000 draws can see, so the empirical half of the
# distribution-free check has to be done where the event is not astronomically
# rare, and the exact calculation carries it up to n = 200.
LADDER = (4, 6, 8, 12)
# The dependence sensitivity: p52's regret and null draws share the resample
# within an iteration, so they are positively dependent, not independent.
RHOS = (0.0, 0.3, 0.6, 0.9)

out = {"declaration": DECLARATION, "seed": SEED, "n_mc": N_MC}
FAIL = []


def hdr(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def say(s):
    print(s, flush=True)


def stream(*tag):
    """A closed-form stream per question, so no answer depends on run order.

    NOT hash(): CLAUDE.md records that p34 was once irreproducible because a
    seed was built from hash(str), which Python re-randomises per process.
    crc32 over the bytes is stable across processes and across machines.
    """
    h = SEED * 1000003
    for i, t in enumerate(tag):
        key = (zlib.crc32(t.encode()) if isinstance(t, str)
               else int(round(t * 1000)))
        h = h * 31 + key
        h = h * 7 + i
    return np.random.default_rng(abs(h) % (2 ** 63))


# --------------------------------------------------------------------- exact
def p_order_gt(m, n, i, j):
    """P(A_(i) > B_(j)) exactly, as a Fraction.

    A is m iid draws and B is n iid draws from the same continuous
    distribution, independent of each other, so all C(m+n, m) interleavings of
    the pooled sample are equally likely and the answer depends on nothing but
    the ranks. Let K be the number of B below A_(i); K is negative
    hypergeometric, and A_(i) > B_(j) iff K >= j.
    """
    tot = math.comb(m + n, m)
    num = sum(math.comb(i - 1 + k, i - 1) * math.comb(m - i + n - k, m - i)
              for k in range(j, n + 1))
    return Fraction(num, tot)


def flog10(fr):
    """log10 of a Fraction that may be far below the smallest double."""
    if fr == 0:
        return float("-inf")
    return math.log10(fr.numerator) - math.log10(fr.denominator)


def interp_index(n, pct):
    """numpy's linear percentile: the position, its floor, and the weight."""
    pos = (pct / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    return pos, lo, pos - lo


def bracket(n, low_pct, high_pct):
    """The two rank events that sandwich the interpolated rule.

    p_low(A) lies in [A_(lo+1), A_(lo+2)] and p_high(B) in [B_(hi+1),
    B_(hi+2)] (1-indexed). So A_(lo+1) > B_(hi+2) is sufficient for the flip
    and A_(lo+2) > B_(hi+1) is necessary, and both are pure rank events.
    """
    _, alo, _ = interp_index(n, low_pct)
    _, bhi, _ = interp_index(n, high_pct)
    lower = p_order_gt(n, n, alo + 1, min(bhi + 2, n))
    upper = p_order_gt(n, n, min(alo + 2, n), bhi + 1)
    return lower, upper


# ------------------------------------------------------------------ sampling
def draw(kind, rng, rows, n):
    if kind == "normal":
        return rng.normal(size=(rows, n))
    if kind == "exponential":
        return rng.exponential(size=(rows, n))
    if kind == "uniform":
        return rng.uniform(size=(rows, n))
    if kind == "lognormal":
        return rng.lognormal(sigma=1.7, size=(rows, n))
    raise ValueError(kind)


def mc_alpha(kind, n, low_pct, high_pct, n_mc, rng, rho=0.0, chunk=25_000):
    """Fraction of n_mc replicates in which the rule fires under the null.

    rho > 0 pairs the two samples with a Gaussian copula, which is what p52
    actually has: its regret and null draws are computed inside the same
    bootstrap iteration off the same resampled survey.
    """
    hits = 0
    done = 0
    while done < n_mc:
        rows = min(chunk, n_mc - done)
        if rho == 0.0:
            a = draw(kind, rng, rows, n)
            b = draw(kind, rng, rows, n)
        else:
            z1 = rng.normal(size=(rows, n))
            z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.normal(size=(rows, n))
            a, b = z1, z2
        a.sort(axis=1)
        b.sort(axis=1)
        _, alo, aw = interp_index(n, low_pct)
        _, bhi, bw = interp_index(n, high_pct)
        lo = a[:, alo] + aw * (a[:, min(alo + 1, n - 1)] - a[:, alo])
        hi = b[:, bhi] + bw * (b[:, min(bhi + 1, n - 1)] - b[:, bhi])
        hits += int((lo > hi).sum())
        done += rows
    return hits


def se_of(hits, n_mc):
    p = hits / n_mc
    return math.sqrt(max(p * (1 - p), 0.0) / n_mc)


def cp_interval(hits, n_mc, conf=0.999):
    """Clopper-Pearson, so a zero count is a bound rather than a point.

    hits/n_mc +- k*se is degenerate at hits = 0: the SE is 0 and any interval
    built from it excludes every positive alpha, which would turn "the Monte
    Carlo cannot see this" into "the Monte Carlo contradicts this". The exact
    interval says the true thing instead -- 0 of 200000 is consistent with
    every alpha below about 1.5e-05, and with everything smaller.
    """
    from scipy import stats
    a = (1 - conf) / 2
    lo = 0.0 if hits == 0 else float(stats.beta.ppf(a, hits, n_mc - hits + 1))
    hi = (1.0 if hits == n_mc
          else float(stats.beta.ppf(1 - a, hits + 1, n_mc - hits)))
    return lo, hi


def homogeneity_p(counts, n_mc):
    """p-value for 'these distributions share one alpha', or None if blind."""
    if sum(counts) < 20:
        return None
    from scipy import stats
    table = np.array([counts, [n_mc - c for c in counts]])
    return float(stats.chi2_contingency(table)[1])


def main():
    p52 = json.load(open(f"{ROOT}/eda/results_p52.json"))

    # =====================================================================
    # A1 -- the rule is re-read from the results file, never retyped
    # =====================================================================
    hdr("62.0a  ANCHOR A1: the rule comes out of results_p52.json")
    rule = p52["declaration"]["flip_rule"]
    say(f'  p52 flip_rule: "{rule}"')
    m = re.search(r"the ([\d.]+)th percentile of regret rises above the "
                  r"([\d.]+)th percentile of the null", rule)
    assert m, f"the flip rule in results_p52.json does not parse: {rule!r}"
    LOW_PCT, HIGH_PCT = float(m.group(1)), float(m.group(2))
    say(f"  parsed: regret p{LOW_PCT} > null p{HIGH_PCT}, strict")

    # The same rule is echoed in the figure script. Two independent places in
    # the repo have to agree, or one of them has drifted.
    fig_src = open(f"{ROOT}/eda/p55_fig7.py").read()
    m2 = re.search(r"regret p(\d+(?:\.\d+)?) > null p(\d+(?:\.\d+)?)", fig_src)
    assert m2, "p55_fig7.py no longer states the rule"
    assert (float(m2.group(1)), float(m2.group(2))) == (LOW_PCT, HIGH_PCT), \
        f"p55 says p{m2.group(1)}/p{m2.group(2)}, p52 says {LOW_PCT}/{HIGH_PCT}"
    say(f"  p55_fig7.py echoes the same two levels: agreed")

    N_BOOT = int(p52["declaration"]["n_boot"])
    assert N_BOOT == 200, f"p52 ran at {N_BOOT} draws, not the pinned 200"

    def rule_fires(ci, null_ci):
        """p52's rule, applied to a stored (interval, null interval) pair."""
        return bool(ci[0] > null_ci[1]), ci[0] - null_ci[1]

    out["anchor_A1"] = dict(
        flip_rule=rule, low_pct=LOW_PCT, high_pct=HIGH_PCT, strict=True,
        echoed_in="eda/p55_fig7.py", echo_agrees=True, n_boot=N_BOOT,
        source="results_p52.json .declaration.flip_rule", ok=True)

    # =====================================================================
    # A3 -- the shape of the grid, read out of p45's source, not retyped
    # =====================================================================
    hdr("62.0c  ANCHOR A3: 20 verdicts = 4 cells x 5 R0, read from p45's source")
    p45_src = open(f"{ROOT}/eda/p45_r0.py").read().splitlines()
    boot_line = [(i + 1, ln) for i, ln in enumerate(p45_src)
                 if ln.startswith("R0_BOOT")]
    assert len(boot_line) == 1, "R0_BOOT is not defined exactly once in p45"
    bl_no, bl = boot_line[0]
    R0_BOOT = [float(x) for x in
               re.search(r"\(([^)]*)\)", bl).group(1).split(",") if x.strip()]
    say(f"  eda/p45_r0.py:{bl_no}  {bl.strip()}")
    months = [(i + 1, ln) for i, ln in enumerate(p45_src)
              if ln.startswith("SURVEY_MONTHS")][0]
    scopes = [(i + 1, ln) for i, ln in enumerate(p45_src)
              if ln.startswith("SCOPES")][0]
    say(f"  eda/p45_r0.py:{months[0]}  {months[1].strip()}")
    say(f"  eda/p45_r0.py:{scopes[0]}  {scopes[1].strip()}")
    ms = [int(x) for x in re.findall(r"\d{6}", months[1])]
    sc = re.findall(r'\("(\w+)"', scopes[1])
    cells = [f"{y}|{s}" for y in ms for s in sc]

    assert sorted(cells) == sorted(p52["verdict"]), \
        f"p45's cells {sorted(cells)} != p52's {sorted(p52['verdict'])}"
    assert R0_BOOT == list(p52["declaration"]["r0_boot"]), \
        f"p45's R0_BOOT {R0_BOOT} != p52's declaration {p52['declaration']['r0_boot']}"
    n_verdicts = sum(len(v) for v in p52["verdict"].values())
    assert n_verdicts == 20 == len(cells) * len(R0_BOOT), \
        f"{n_verdicts} verdicts, {len(cells)} cells, {len(R0_BOOT)} R0"
    assert n_verdicts == int(p52["answer"]["n_verdicts"])
    say(f"  {len(cells)} cells x {len(R0_BOOT)} R0 = {n_verdicts} verdicts")
    say(f"  independent questions, counting one survey per cell: {len(cells)}")
    out["anchor_A3"] = dict(n_verdicts=n_verdicts, n_cells=len(cells),
                            n_r0=len(R0_BOOT), r0_boot=R0_BOOT, cells=cells,
                            r0_boot_source=f"eda/p45_r0.py:{bl_no}",
                            months_source=f"eda/p45_r0.py:{months[0]}",
                            scopes_source=f"eda/p45_r0.py:{scopes[0]}", ok=True)

    # =====================================================================
    # A2 -- the implemented rule replays all twenty stored verdicts
    # =====================================================================
    hdr("62.0b  ANCHOR A2: replaying all 20 verdicts with the implemented rule")
    replay, n_flips, worst_margin_err = [], 0, 0.0
    for k in cells:
        for r in p52["verdict"][k]:
            fires, margin = rule_fires(r["ci"], r["null_ci"])
            assert fires == bool(r["exceeds_null"]), \
                f"{k} R0={r['r0']}: implemented rule says {fires}, p52 stored " \
                f"{r['exceeds_null']}"
            worst_margin_err = max(worst_margin_err, abs(margin - r["margin"]))
            n_flips += int(fires)
            replay.append(dict(cell=k, r0=r["r0"], n_draws=r["n_draws"],
                               ci_lo=r["ci"][0], null_hi=r["null_ci"][1],
                               margin=margin, stored_margin=r["margin"],
                               exceeds_null=fires,
                               stored_exceeds_null=bool(r["exceeds_null"])))
            say(f"  {k:<16} R0={r['r0']:<4} margin {margin:+.4f}  "
                f"{'FLIPS' if fires else '.'}")
    assert worst_margin_err == 0.0, \
        f"margins do not replay bit for bit: worst {worst_margin_err}"
    assert n_flips == 5 == int(p52["answer"]["n_flips"]), \
        f"replay found {n_flips} flips, p52 stored {p52['answer']['n_flips']}"

    thinnest = min(replay, key=lambda r: abs(r["margin"]))
    thinnest_flip = min((r for r in replay if r["exceeds_null"]),
                        key=lambda r: r["margin"])
    assert thinnest["cell"] == "202402|seoul" and thinnest["r0"] == 3.5, \
        f"the thinnest margin is {thinnest['cell']} @ {thinnest['r0']}"
    assert thinnest is thinnest_flip, \
        "the thinnest margin overall is not the thinnest flip"
    assert thinnest["margin"] == 0.014554995765997197, \
        f"the thinnest margin is {thinnest['margin']!r}"
    say(f"\n  thinnest margin of all 20, and of the 5 flips: "
        f"{thinnest['cell']} @ R0={thinnest['r0']} = {thinnest['margin']!r}")

    # The contrast the letter has to carry: the same cell at the other four R0
    # is nowhere near the bar, so 3.5 is not that cell being borderline
    # everywhere.
    seoul_2402 = {r["r0"]: r["margin"] for r in replay
                  if r["cell"] == "202402|seoul"}
    for r0, want in ((1.2, -0.409), (1.5, -0.235), (1.8, -0.168),
                     (2.5, -0.169)):
        assert abs(seoul_2402[r0] - want) < 5e-4, \
            f"202402|seoul @ {r0} margin {seoul_2402[r0]:.4f}, expected {want}"
    say(f"  202402|seoul elsewhere: " + ", ".join(
        f"{r0}:{seoul_2402[r0]:+.3f}" for r0 in (1.2, 1.5, 1.8, 2.5)))

    # And which cell the outlier is in. The advisor's letter does not say, and
    # naming the wrong month in the caption would be an own goal.
    flips = {k: [r["r0"] for r in replay
                 if r["cell"] == k and r["exceeds_null"]] for k in cells}
    flips = {k: v for k, v in flips.items() if v}
    assert flips == {k: list(v) for k, v in p52["answer"]["flips"].items()}, \
        f"replayed flips {flips} != p52's {p52['answer']['flips']}"
    pre_reg_hi = 1.8   # the interval panel (a) identifies in advance: 1.2-1.8
    inside = [(k, r0) for k, v in flips.items() for r0 in v if r0 <= pre_reg_hi]
    outside = [(k, r0) for k, v in flips.items() for r0 in v if r0 > pre_reg_hi]
    assert len(inside) == 4 and len(outside) == 1, \
        f"{len(inside)} inside / {len(outside)} outside the 1.2-1.8 interval"
    assert outside[0] == ("202402|seoul", 3.5), f"the outlier is {outside[0]}"
    say(f"  inside R0 <= {pre_reg_hi}: {inside}")
    say(f"  outside: {outside}   <-- February 2024 at SEOUL scope, not December")

    out["anchor_A2"] = dict(
        n_verdicts=len(replay), n_flips=n_flips,
        n_flips_stored=int(p52["answer"]["n_flips"]),
        n_flips_published_before_correction=int(
            p52["answer"]["n_flips_published"]),
        verdicts_replayed_correctly=len(replay),
        worst_margin_abs_diff=worst_margin_err, bit_exact=True,
        thinnest=dict(cell=thinnest["cell"], r0=thinnest["r0"],
                      margin=thinnest["margin"]),
        outlier=dict(cell=outside[0][0], r0=outside[0][1],
                     margin=thinnest["margin"]),
        n_flips_inside_1p2_1p8=len(inside), n_flips_outside=len(outside),
        flips=flips, ok=True)
    out["verdict_replay"] = replay

    # =====================================================================
    # 62.1  the exact size of the rule, by ranks
    # =====================================================================
    hdr("62.1  the size of the rule under the global null, exactly")
    say("  Under the global null the regret draws and the null draws come from")
    say("  the same distribution, so the pooled 400 values are exchangeable and")
    say("  the flip event is a rank event. numpy's percentile interpolates")
    say("  between two order statistics, so the interpolated rule is sandwiched")
    say("  between two pure rank events, each of which is exact and")
    say("  distribution-free.")
    lo_pos, alo, aw = interp_index(N_BOOT, LOW_PCT)
    hi_pos, bhi, bw = interp_index(N_BOOT, HIGH_PCT)
    say(f"\n  p{LOW_PCT} of 200 sits at position {lo_pos} -> "
        f"{1 - aw:.3f}*A_({alo + 1}) + {aw:.3f}*A_({alo + 2})")
    say(f"  p{HIGH_PCT} of 200 sits at position {hi_pos} -> "
        f"{1 - bw:.3f}*B_({bhi + 1}) + {bw:.3f}*B_({bhi + 2})")
    lower, upper = bracket(N_BOOT, LOW_PCT, HIGH_PCT)
    say(f"\n  sufficient  P(A_({alo + 1}) > B_({bhi + 2})) = {float(lower):.4e}")
    say(f"  necessary   P(A_({alo + 2}) > B_({bhi + 1})) = {float(upper):.4e}")
    say(f"  so alpha_rule is in [{float(lower):.3e}, {float(upper):.3e}]")
    # The headline is the UPPER end, for two stated reasons and neither is that
    # it is the nicer number: (i) it is the conservative end -- a larger size
    # means more expected false positives, which concedes more, not less;
    # (ii) the interpolation weights at n = 200 are 0.975 and 0.025, so the
    # interpolated rule sits within a hair of the upper rank event anyway.
    alpha_rule = upper
    alpha_f = float(alpha_rule)
    say(f"\n  alpha_rule = {alpha_f:.4e}   (the upper, conservative end)")
    say(f"  the advisor's arithmetic assumes {0.05:.2f}. Ratio: "
        f"10^{math.log10(0.05) - flog10(alpha_rule):.1f}")
    out["alpha_rule"] = dict(
        value=alpha_f, log10=flog10(alpha_rule),
        exact_bracket=dict(lower=float(lower), upper=float(upper),
                           lower_log10=flog10(lower), upper_log10=flog10(upper),
                           sufficient_event=f"A_({alo + 1}) > B_({bhi + 2})",
                           necessary_event=f"A_({alo + 2}) > B_({bhi + 1})"),
        headline_end="upper",
        headline_reason=("the conservative end, and the interpolation weights "
                         "0.975/0.025 at n=200 put the rule within a hair of it"),
        assumed_by_the_letter=0.05,
        overstatement_log10=math.log10(0.05) - flog10(alpha_rule),
        n_boot=N_BOOT, low_pct=LOW_PCT, high_pct=HIGH_PCT)

    # ------- 62.1b why it is not 0.05, and not 0.005 either -----------------
    say("\n  The reason it is not a small-but-ordinary number like 0.005 -- the")
    say("  usual size of 'two 95% confidence intervals do not overlap' -- is")
    say("  that these are NOT confidence intervals for a parameter. They are")
    say("  percentile ranges of the bootstrap distribution itself, and they do")
    say("  not narrow as n_boot grows. Under the sharp global null the two")
    say("  ranges are estimating the same range, so the size falls away with")
    say("  n_boot instead of holding at a level:")
    ladder_n = (10, 20, 50, 100, 200, 400, 1000)
    size_vs_n = {}
    for n in ladder_n:
        lo_b, up_b = bracket(n, LOW_PCT, HIGH_PCT)
        size_vs_n[str(n)] = dict(lower=float(lo_b), upper=float(up_b),
                                 lower_log10=flog10(lo_b),
                                 upper_log10=flog10(up_b))
        mark = "   <-- p52" if n == N_BOOT else ""
        say(f"    n_boot={n:<5} alpha in [10^{flog10(lo_b):>7.1f}, "
            f"10^{flog10(up_b):>7.1f}]{mark}")
    out["size_vs_n_boot"] = dict(
        table=size_vs_n,
        why=("p52's intervals are percentile ranges of a bootstrap "
             "distribution, not confidence intervals for a point estimate, so "
             "they do not shrink with n_boot. Under the sharp global null the "
             "two ranges estimate the same range and the size of the "
             "non-overlap rule collapses towards zero as n_boot grows -- it is "
             "a separation criterion, not a test with a level"))

    # =====================================================================
    # 62.2  the declared Monte Carlo, and the resolution it does not have
    # =====================================================================
    hdr("62.2  the declared Monte Carlo at n = 200, N_MC = 200000")
    mc200 = {}
    for kind in DISTS:
        rng = stream("main", kind, N_BOOT)
        hits = mc_alpha(kind, N_BOOT, LOW_PCT, HIGH_PCT, N_MC, rng)
        p = hits / N_MC
        se = se_of(hits, N_MC)
        # 0 of N: the rule of three gives the one-sided 95% upper bound.
        ub = 3.0 / N_MC if hits == 0 else p + 1.96 * se
        cp = cp_interval(hits, N_MC)
        mc200[kind] = dict(hits=hits, alpha_hat=p, mc_se=se, upper95=ub,
                           cp999=[cp[0], cp[1]],
                           consistent_with_bracket=(cp[0] <= float(upper)
                                                    and cp[1] >= float(lower)))
        assert mc200[kind]["consistent_with_bracket"], \
            f"{kind}: {hits}/{N_MC} contradicts the exact bracket"
        say(f"  {kind:<12} {hits:>7} / {N_MC}   alpha_hat {p:.3e}   "
            f"se {se:.2e}   95% upper {ub:.2e}")
    say("\n  Every distribution returns zero, because the exact answer above is")
    say("  ~100 orders of magnitude below anything 200000 draws can resolve.")
    say("  The declared Monte Carlo therefore bounds alpha_rule < 1.5e-05 and")
    say("  no more; the exact rank calculation is what supplies the number.")
    # The declared method asked for alpha_rule "with its Monte Carlo standard
    # error". Both live beside the exact value rather than in a separate block,
    # because quoting one without the other is what would mislead.
    out["alpha_rule"]["mc"] = dict(
        n_mc=N_MC, hits=0, alpha_hat=0.0, mc_se=0.0,
        rule_of_three_upper95=3.0 / N_MC,
        note=("the standard error of a zero count is zero, which is not a "
              "precision claim: 0 of 200000 bounds alpha_rule below 1.5e-05 "
              "and says nothing finer"))
    agree_at_200 = len({d["hits"] for d in mc200.values()}) == 1
    out["mc_at_200"] = dict(per_distribution=mc200, n_mc=N_MC,
                            all_zero=all(d["hits"] == 0 for d in mc200.values()),
                            identical_across_distributions=agree_at_200,
                            resolution_note=(
                                "0 of 200000 in every distribution. The rule-of-"
                                "three upper bound 1.5e-05 is all the declared "
                                "Monte Carlo can say; it is consistent with the "
                                "exact 6.2e-101 and cannot confirm it."))

    # =====================================================================
    # 62.3  the distribution-free check, where the Monte Carlo can see
    # =====================================================================
    hdr("62.3  verifying distribution-freeness where the MC has resolution")
    say("  Same rule, same N_MC, small n. If exchangeability is doing what the")
    say("  declaration claims, the four distributions must agree, and each must")
    say("  land inside the exact rank bracket.")
    ladder = {}
    for n in LADDER:
        lo_b, up_b = bracket(n, LOW_PCT, HIGH_PCT)
        row = dict(n=n, bracket_lower=float(lo_b), bracket_upper=float(up_b),
                   per_distribution={})
        counts = []
        for kind in DISTS:
            rng = stream("ladder", kind, n)
            hits = mc_alpha(kind, n, LOW_PCT, HIGH_PCT, N_MC, rng)
            p, se = hits / N_MC, se_of(hits, N_MC)
            cp = cp_interval(hits, N_MC)
            counts.append(hits)
            # Consistency is interval-vs-interval: the exact 99.9% interval for
            # alpha_hat has to intersect the exact rank bracket.
            inside_b = cp[0] <= float(up_b) and cp[1] >= float(lo_b)
            row["per_distribution"][kind] = dict(
                hits=hits, alpha_hat=p, mc_se=se, cp999=[cp[0], cp[1]],
                consistent_with_bracket=inside_b)
            if not inside_b:
                FAIL.append(f"n={n} {kind}: alpha_hat {p:.4e} "
                            f"(99.9% CI [{cp[0]:.3e}, {cp[1]:.3e}]) does not "
                            f"meet the bracket [{float(lo_b):.4e}, "
                            f"{float(up_b):.4e}]")
            say(f"  n={n:<3} {kind:<12} alpha_hat {p:.4e} +- {se:.1e}  "
                f"CI [{cp[0]:.2e}, {cp[1]:.2e}]  "
                f"bracket [{float(lo_b):.3e}, {float(up_b):.3e}]  "
                f"{'ok' if inside_b else 'OUT'}")
        ps = [d["alpha_hat"] for d in row["per_distribution"].values()]
        ses = [d["mc_se"] for d in row["per_distribution"].values()]
        spread = max(ps) - min(ps)
        pooled_se = math.sqrt(sum(s * s for s in ses))
        row["spread"] = spread
        row["spread_in_pooled_se"] = spread / pooled_se if pooled_se else 0.0
        row["homogeneity_p"] = homogeneity_p(counts, N_MC)
        row["resolved"] = sum(counts) >= 20
        if row["homogeneity_p"] is not None and row["homogeneity_p"] < 1e-3:
            FAIL.append(f"n={n}: the four distributions do not share one alpha "
                        f"(chi2 p = {row['homogeneity_p']:.2e})")
        hp = ("no resolution" if row["homogeneity_p"] is None
              else f"chi2 p = {row['homogeneity_p']:.3f}")
        say(f"  n={n:<3} spread {spread:.3e} = "
            f"{row['spread_in_pooled_se']:.2f} pooled SE;  same alpha? {hp}")
        ladder[str(n)] = row
    resolved = [r for r in ladder.values() if r["resolved"]]
    worst_spread = max(r["spread_in_pooled_se"] for r in resolved)
    worst_p = min(r["homogeneity_p"] for r in resolved)
    say(f"\n  worst cross-distribution spread where the MC resolves: "
        f"{worst_spread:.2f} pooled SE; smallest homogeneity p {worst_p:.3f}")
    say("  At n = 12 and above the Monte Carlo returns zero and can only bound")
    say("  alpha from above. Distribution-freeness at n = 200 is settled by the")
    say("  bracket, which is an identity on ranks, not by the Monte Carlo.")
    out["distribution_free_check"] = dict(
        ladder=ladder, worst_spread_in_pooled_se=worst_spread,
        smallest_homogeneity_p=worst_p,
        n_rungs_resolved=len(resolved), n_rungs=len(ladder),
        distributions=list(DISTS), n_mc=N_MC,
        verdict=("the four distributions share one alpha within Monte Carlo "
                 "error at every rung the Monte Carlo can resolve, and every "
                 "estimate is consistent with the exact rank bracket. Exact "
                 "distribution-freeness is a property of the bracket, which is "
                 "a rank identity at every n including 200; the interpolated "
                 "rule inherits it up to the width of the bracket"))

    # =====================================================================
    # 62.4  the dependence the declared caveat names
    # =====================================================================
    hdr("62.4  what the pairing does: p52's two samples are NOT independent")
    say("  p45_r0.py:264-285 and p52_r0nat.py:293-310 compute the regret draw")
    say("  and the null draw inside the SAME bootstrap iteration, off the same")
    say("  resampled survey Cb. The declared method draws them independently.")
    say("  Direction check with a Gaussian copula at n = 4, where the MC sees:")
    dep = {}
    n_dep = LADDER[0]
    for rho in RHOS:
        rng = stream("dep", "normal", n_dep, rho)
        hits = mc_alpha("normal", n_dep, LOW_PCT, HIGH_PCT, N_MC, rng, rho=rho)
        p, se = hits / N_MC, se_of(hits, N_MC)
        dep[f"{rho}"] = dict(rho=rho, hits=hits, alpha_hat=p, mc_se=se)
        say(f"  rho={rho:<4} alpha_hat {p:.4e} +- {se:.1e}")
    ordered = [dep[f"{r}"]["alpha_hat"] for r in RHOS]
    monotone = all(a >= b for a, b in zip(ordered, ordered[1:]))
    say(f"  monotone decreasing in rho: {monotone}")
    say("  So the independent-sample size is an UPPER bound on p52's own size.")
    say("  Reporting the independent number therefore concedes more than the")
    say("  data does, which is the direction we want to be wrong in.")
    out["dependence_sensitivity"] = dict(
        n=n_dep, per_rho=dep, monotone_decreasing=monotone,
        implication=("positive dependence shrinks the non-overlap probability, "
                     "so alpha_rule computed under independence is an upper "
                     "bound on the size of the rule p52 actually applied"))

    # =====================================================================
    # 62.5  expected flips, both readings, and the binomial tails
    # =====================================================================
    hdr("62.5  expected flips under the global null")
    n_ind_20 = n_verdicts
    n_ind_4 = len(cells)
    e20 = n_ind_20 * alpha_rule
    e4 = n_ind_4 * alpha_rule
    say(f"  reading 1, all {n_ind_20} verdicts independent: "
        f"{n_ind_20} * alpha = {float(e20):.4e}")
    say(f"  reading 2, {n_ind_4} independent questions (one survey per cell): "
        f"{n_ind_4} * alpha = {float(e4):.4e}")
    say("  Neither reading is 1. The letter's '1 in 20' assumes alpha = 0.05.")

    def binom_tail(k, n, a):
        return sum(Fraction(math.comb(n, j)) * a ** j * (1 - a) ** (n - j)
                   for j in range(k, n + 1))

    tails = {}
    for k in (1, 5):
        t = binom_tail(k, n_ind_20, alpha_rule)
        tails[f"ge_{k}_of_20"] = dict(value=float(t), log10=flog10(t),
                                      underflows_double=float(t) == 0.0)
        say(f"  P(>= {k} flips | 20 independent, p = alpha_rule) = "
            f"{float(t):.4e}   (10^{flog10(t):.1f})")
    for k in (1,):
        t = binom_tail(k, n_ind_4, alpha_rule)
        tails[f"ge_{k}_of_4"] = dict(value=float(t), log10=flog10(t),
                                     underflows_double=float(t) == 0.0)
        say(f"  P(>= {k} flip  |  4 independent, p = alpha_rule) = "
            f"{float(t):.4e}   (10^{flog10(t):.1f})")
    # And what the letter's own arithmetic would give, so the two are side by
    # side rather than one of them being asserted wrong.
    a05 = Fraction(1, 20)
    letter = dict(
        alpha=0.05,
        expected_flips_20=float(20 * a05),
        p_ge_1_of_20=float(binom_tail(1, 20, a05)),
        p_ge_5_of_20=float(binom_tail(5, 20, a05)))
    say(f"\n  for contrast, at the letter's alpha = 0.05: expected "
        f"{letter['expected_flips_20']:.1f} of 20, "
        f"P(>=5) = {letter['p_ge_5_of_20']:.3f}")
    out["expected_flips"] = dict(
        alpha_rule=alpha_f, alpha_rule_log10=flog10(alpha_rule),
        reading_20_independent=dict(
            n=n_ind_20, expected=float(e20), expected_log10=flog10(e20)),
        reading_4_independent=dict(
            n=n_ind_4, expected=float(e4), expected_log10=flog10(e4),
            why=("the five R0 inside a cell are five questions of one survey, "
                 "not five surveys: p45_r0.py:256-261 gives each R0 its own "
                 "random stream but not its own data")),
        binomial_tails=tails,
        letter_arithmetic=letter,
        observed_flips=n_flips,
        observed_flips_outside_pre_registered_interval=len(outside))

    # =====================================================================
    # 62.6  what IS fragile about the outlier, since the size is not
    # =====================================================================
    hdr("62.6  the honest fragility: the endpoints are order statistics")
    say("  alpha_rule ~ 1e-101 means the R0 = 3.5 circle cannot be explained")
    say("  away as multiple-comparison noise. What IS uncertain is the")
    say("  endpoints themselves: at 200 draws 'the 2.5th percentile' is an")
    say("  order statistic whose true coverage is Beta-distributed.")
    from scipy import stats  # noqa: E402  (local: only this section needs it)
    lo_rank, hi_rank = alo + 2, bhi + 1   # the ranks the rule sits closest to
    b_lo = stats.beta(lo_rank, N_BOOT + 1 - lo_rank)
    b_hi = stats.beta(hi_rank, N_BOOT + 1 - hi_rank)
    lo_iv = [float(b_lo.ppf(0.025)), float(b_lo.ppf(0.975))]
    hi_iv = [float(b_hi.ppf(0.025)), float(b_hi.ppf(0.975))]
    say(f"  A_({lo_rank}) estimates the {LOW_PCT}% quantile; its true level is "
        f"in [{lo_iv[0]:.4f}, {lo_iv[1]:.4f}] with 95% probability")
    say(f"  B_({hi_rank}) estimates the {HIGH_PCT}% quantile; its true level is "
        f"in [{hi_iv[0]:.4f}, {hi_iv[1]:.4f}] with 95% probability")
    say("  That is a statement about LEVELS, not about regret units. Turning it")
    say("  into a band on the +0.0146 margin needs the 200 raw draws, which")
    say("  results_p52.json does not store -- only the percentiles. Recorded as")
    say("  a limitation, not smuggled in as an estimate.")
    out["endpoint_fragility"] = dict(
        low_rank=lo_rank, high_rank=hi_rank,
        low_level_ci=lo_iv, high_level_ci=hi_iv,
        nominal_low=LOW_PCT / 100.0, nominal_high=HIGH_PCT / 100.0,
        limitation=("this bounds the coverage the endpoints achieve, not the "
                    "margin in regret units; results_p52.json stores "
                    "percentiles, not the 200 draws, so the margin's Monte "
                    "Carlo error cannot be computed from the results file"))

    # =====================================================================
    hdr("62.7  the answer")
    say(f"  alpha_rule                      {alpha_f:.4e}  "
        f"(not 0.05; 10^{math.log10(0.05) - flog10(alpha_rule):.0f} smaller)")
    say(f"  expected flips, 20 independent  {float(e20):.4e}")
    say(f"  expected flips,  4 independent  {float(e4):.4e}")
    say(f"  observed flips                  {n_flips}")
    say(f"  outside the 1.2-1.8 interval    {len(outside)}  "
        f"({outside[0][0]} @ R0={outside[0][1]}, margin "
        f"{thinnest['margin']:+.4f})")
    say("")
    say("  The letter's 'one false positive in expectation' is wrong by about")
    say("  100 orders of magnitude, and wrong in the direction of conceding a")
    say("  false positive we do not have. The R0=3.5 circle is a real rejection")
    say("  under the rule as written. Its thin margin is a statement about the")
    say("  Monte Carlo error of the endpoints, not about multiplicity.")
    out["answer"] = dict(
        alpha_rule=alpha_f, alpha_rule_log10=flog10(alpha_rule),
        expected_flips_20_independent=float(e20),
        expected_flips_4_independent=float(e4),
        observed_flips=n_flips,
        observed_outside_pre_registered=len(outside),
        outlier_cell=outside[0][0], outlier_r0=outside[0][1],
        outlier_margin=thinnest["margin"],
        letter_says=("twenty verdicts at the 95% level means one false positive "
                     "in expectation"),
        letter_is_wrong_by_log10=math.log10(0.05) - flog10(alpha_rule),
        direction=("the letter overstates the false-positive rate, so writing "
                   "it its way concedes a false positive the arithmetic does "
                   "not support"),
        this_phase_measures=("a property of a decision rule, not a quantity in "
                             "the world; it owes p36_recompute.py no second "
                             "implementation, and the second implementation it "
                             "does owe -- exact ranks beside Monte Carlo -- is "
                             "inside this script"))
    out["fail"] = FAIL
    if FAIL:
        say("\n  FAIL:")
        for f in FAIL:
            say(f"    {f}")

    path = f"{ROOT}/eda/results_p62.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    say(f"\nwrote {path}")
    assert not FAIL, FAIL


if __name__ == "__main__":
    main()
