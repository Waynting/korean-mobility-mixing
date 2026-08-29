#!/bin/zsh
# Run a script twice and compare its results file byte for byte.
#
# WHY THIS EXISTS. On 2026-08-19 p34 turned out to be irreproducible: the seed
# was built as `SEED + hash(arm) % 1000 + ym`, and Python re-randomises the hash
# of a string in every process, so the same code on the same data gave different
# confidence bands on two runs. Nothing downstream was wrong -- the anchors are
# deterministic and only the third decimal moved -- but a script that drifts
# cannot be published, and "I ran it twice and diffed the JSON" is a claim in a
# letter until it is a command anyone can run.
#
#   eda/determinism_check.sh p33_coverage p35_seir p36_recompute
#   eda/determinism_check.sh                       # the default set below
#
# While it runs, results_pNN.json is deleted and rewritten twice, so do not take
# a git commit in that window -- it would record a deleted results file. If the
# run is interrupted, results_pNN.json.determinism-backup holds the version that
# was there when it started.
#
# Each named script is run twice with identical arguments; between runs its
# results_*.json is moved aside, and the two are compared by sha256. Anything
# non-deterministic -- an unseeded rng, a set iteration order, a timestamp
# written into the output -- shows up as a mismatch.
#
# NOTE ON WHAT THIS DOES NOT CATCH: a script that is deterministic but WRONG.
# That is p36_recompute.py's job (an independent second implementation) and the
# per-script anchors' job. This one only answers "does it give the same answer
# twice".
set -u

REPO=${0:a:h:h}
PY=$REPO/.venv/bin/python
TMP=${TMPDIR:-/tmp}/determinism.$$
mkdir -p $TMP

# name:args  -- the args have to match how the script is actually run, since a
# different --boot is a different (and equally valid) deterministic answer.
typeset -A ARGS
ARGS=(
  p33_coverage  "--boot 400"
  p35_seir      "--boot 200"
  p36_recompute ""
  p34_ksweep    "--months 202001,202012,202312,202402,202512,202606"
  p32_pmix      ""
  p37_timeseries ""
  # 79 months x 2 arms x 200 replicates is ~27 minutes per run, so the pair is
  # most of an hour and this one is named explicitly rather than sitting in the
  # default set:
  #   eda/determinism_check.sh p38_ksweep79
  p38_ksweep79  ""
  # Reads results_p37.json and never touches the parquet, so the pair is a few
  # seconds and it sits in the default set. Its only rng is the 41.4b draw.
  p41_semester  ""
  # p42 has NO rng at all: one matrix per month and closed-form hypergeometric
  # nulls, so there is nothing here that a seed could move. It is listed anyway
  # because "deterministic by inspection" is a claim and this is a command, and
  # because the thing it would actually catch is not a seed but a day set --
  # the bug it was written to settle. About 80 seconds a run.
  p42_monthscope ""
  # The recovery experiment is a Monte Carlo over a 5 x 10 grid; a pair of runs
  # is most of an hour, so like p38 it is named explicitly rather than sitting
  # in the default set:
  #   eda/determinism_check.sh p39_recovery      # the published --reps 100
  # For the cheap version of the same question -- does a seed depend on
  # anything but its own grid point -- p39 carries its own one-minute check:
  #   .venv/bin/python eda/p39_recovery.py --verify-seeds
  p39_recovery  "--reps 100"
  # p44 reads results_p26/p32/p33/p38/p39.json and the survey micro-data, and
  # writes only its own file. One global rng stream feeds the 20 shuffled nulls,
  # the pooling replicates and the 300-draw bootstrap IN THAT ORDER, so it is
  # deterministic as written but reorder-fragile -- which is precisely what a
  # pair of runs is for. About 12 seconds a run, so it sits in the default set.
  # ...which it said for three rounds while NOT being in the `targets` list
  # below -- the comment described an intention and nothing enforced it, so the
  # one script here whose stated risk is rng CONSUMPTION ORDER was the one never
  # run twice. Added on 2026-08-27; the pair is byte-identical
  # (sha256 5e240f1c14413f25).
  p44_beta      ""
  # p45's args are not cosmetic: --boot35 IS the answer. The file first written
  # on 2026-08-21 came from a 4-draw smoke test whose interval verdicts were
  # meaningless, and only `declaration.n_boot` recorded it. Pinned here at the
  # published 200 so that "run it twice" cannot silently mean two different
  # questions. Each draw is two SLSQP solves x 5 R0 x 4 cells, so a pair of runs
  # is long and this one is named explicitly rather than sitting in the default
  # set:  eda/determinism_check.sh p45_r0
  p45_r0        "--boot35 200"
  # --- the 2026-08-24 round --------------------------------------------------
  # p49 rebuilds the 202312 cell table twice over and pairs it two ways. Its
  # only rng is the 25 x 16 synthetic table 49.0c checks `collapse` against, and
  # that is seeded -- but the thing a pair of runs would actually catch here is
  # the pandas groupby/pivot ordering underneath `wide_counts`, which is not
  # seeded by anything and which the whole phase rests on. About 3 minutes a run.
  p49_dongwor   ""
  # p50 has no rng at all: it reads two registration CSVs, the monthly 시군구 CSV
  # and up to two quarterly xlsx, and folds them onto 16 bins. Listed for the
  # same reason p42 is -- "deterministic by inspection" is a claim, and neither
  # openpyxl's row order nor dict insertion order into the per-gu accumulator is
  # something this file should assume.
  # Seconds a run.
  p50_natpop    ""
  # p51 replays p40's stream and draws its own; a pair of runs is the only thing
  # that would catch the replay silently depending on dict iteration order in
  # `cubes`. About 8 minutes a run, so it is named explicitly rather than
  # sitting in the default set:  eda/determinism_check.sh p51_natsym
  p51_natsym    ""
  # p53 sets p44's POP global from outside. If p44 ever acquires module-level
  # state that survives between build() calls, the second run is where it shows.
  # About a minute a run.
  p53_natbeta   ""
  # p54 is arithmetic over a stored series plus an exact hypergeometric
  # convolution -- no rng, no data access. Seconds.
  p54_semrecount ""
  # p52's --boot35 is the answer, exactly as p45's is, and for exactly the same
  # reason: the first file written on 2026-08-24 came from a 4-draw smoke test.
  # A pair of runs is ~2 hours, so it is named explicitly:
  #   eda/determinism_check.sh p52_r0nat
  p52_r0nat     "--boot35 200"
  # The figure phases compute nothing, but they DO write a results file (the
  # caption-number sheet), and a sheet that drifts between runs would let a
  # caption quote a number the next run does not produce. Seconds each.
  p55_fig7      ""
  p56_fig5      ""
  # p47 writes a sheet too, and it was left out of this list when p55/p56 were
  # added. Same reason as theirs, plus one of its own: 4(b) now reads a second
  # results file (p18b) and prints ten more numbers, so a drift between runs
  # would show up in a caption that is already written against the sheet.
  p47_fig34     ""
  # p57 re-reads the survey CSV and writes a sheet of counts; no rng, seconds.
  p57_placecode ""
  # p48 writes results_p48.json -- the number sheet Figure 1 is labelled from --
  # and until the 08-27 round it was in NO list here at all. That is exactly the
  # gap p55/p56/p47 were added to close, left open one file longer: a sheet that
  # drifted between runs would put a different number under Figure 1 and nothing
  # would say so. Seconds.
  p48_fig1      ""
  # --- the 2026-08-27 round --------------------------------------------------
  # p58 has no rng at all: 78 shifts is an exhaustive enumeration over a stored
  # series. Listed for the reason p42 and p54 are -- "deterministic by
  # inspection" is a claim and this is a command -- and because what a pair of
  # runs would actually catch here is not a seed but the label vector it rotates.
  # Seconds.
  p58_semshift  ""
  # p60 reads results_p37's 79 stored matrices and collapses the age axis; no
  # rng, and the thing a pair of runs catches is the dict iteration order the
  # nested ladder is accumulated in. Seconds.
  p60_agescale  ""
  # p61 draws 500 ego-band permutations, a 500-draw respondent bootstrap, a
  # 500-draw multinomial null and a 500-draw passive null off ONE rng stream, in
  # that order -- deterministic as written but reorder-fragile in precisely the
  # way p44 is, which is what a pair of runs is for. About 6 seconds a run.
  p61_survspec  ""
  # p62 is Monte Carlo on ranks at N = 200,000 plus exact combinatorics. The
  # seed is declared; a pair of runs is what says the seed is the only thing the
  # draws depend on. About 6 seconds a run.
  p62_multiplicity ""
  # p63 writes Figure 2's caption sheet. Same reason as p55/p56/p47/p48. Seconds.
  p63_fig2      ""
  # --- the 2026-08-28 round --------------------------------------------------
  # p46 is the LAST figure phase to get here, and until 2026-08-28 it was the
  # only one that could not be here at all: it wrote no results file, so there
  # was nothing to compare. Figure 6's caption numbers now come off a sheet like
  # every other figure's, and this is what says the sheet is the same sheet
  # twice. Reads results_p33/p35 only; seconds.
  p46_fig6      ""
  # p59's --reps is the answer, exactly as p45's --boot35 is: it re-simulates
  # three new beta slices at p39's published 100 replicates, and any other value
  # is a different question. Pinned here so that "run it twice" cannot silently
  # mean two different questions. A pair of runs is about 31 minutes, so like
  # p38/p39/p45/p51/p52 this one is named explicitly rather than sitting in the
  # default set:   eda/determinism_check.sh p59_betastar
  p59_betastar  "--reps 100"
)

# NOT `targets=(${@:-a b c})`: with an empty $@ that yields ONE element holding
# the whole default string, so the loop would look for a script named
# "p33_coverage p35_seir p36_recompute" and report it as a failure.
if (( $# )); then
  targets=($@)
else
  targets=(p33_coverage p35_seir p36_recompute p41_semester p42_monthscope
           p49_dongwor p50_natpop p53_natbeta p54_semrecount
           p44_beta p55_fig7 p56_fig5 p47_fig34 p48_fig1 p57_placecode
           p58_semshift p60_agescale p61_survspec p62_multiplicity p63_fig2
           p46_fig6)
fi
fail=0

for name in $targets; do
  num=${name%%_*}                       # p33_coverage -> p33
  out=$REPO/eda/results_${num}.json
  args=${ARGS[$name]:-}
  print -- "=== $name ${args:+($args)} ==="
  # The results file is backed up NEXT TO ITSELF, not into $TMP. $TMP is
  # rm -rf'd at the end, so a run that is interrupted -- killed, ^C, a reboot --
  # used to leave results_pNN.json deleted (this script rm's it between runs)
  # and its only copy inside a directory nobody would think to look in. That
  # happened once on 2026-08-19 and the file had to come back from git. A
  # sibling backup makes an interrupted run recoverable with a mv.
  backup=$out.determinism-backup
  [[ -f $out ]] && cp $out $backup
  for run in 1 2; do
    rm -f $out
    if ! $PY $REPO/eda/${name}.py ${=args} > $TMP/${name}.run${run}.log 2>&1; then
      print -- "  run $run FAILED, see below; restoring the committed results file"
      tail -5 $TMP/${name}.run${run}.log | sed 's/^/    /'
      cp $TMP/${name}.run${run}.log $REPO/eda/${name}.determinism-failure.log
      [[ -f $backup ]] && mv $backup $out
      fail=1
      break
    fi
    cp $out $TMP/${name}.run${run}.json
  done
  # Both runs finished, so $out holds run 2 and the backup has done its job.
  [[ -f $TMP/${name}.run2.json && -f $backup ]] && rm -f $backup
  if [[ -f $TMP/${name}.run1.json && -f $TMP/${name}.run2.json ]]; then
    a=$(shasum -a 256 $TMP/${name}.run1.json | cut -d' ' -f1)
    b=$(shasum -a 256 $TMP/${name}.run2.json | cut -d' ' -f1)
    if [[ $a == $b ]]; then
      print -- "  byte-identical  sha256 ${a[1,16]}  ($(wc -c < $TMP/${name}.run1.json | tr -d ' ') bytes)"
    else
      print -- "  DIFFERS  $a vs $b"
      print -- "  first differing keys:"
      $PY - $TMP/${name}.run1.json $TMP/${name}.run2.json <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
diffs = []
def walk(x, y, p=""):
    if len(diffs) >= 8: return
    if isinstance(x, dict):
        for k in x:
            if k in y: walk(x[k], y[k], f"{p}/{k}")
    elif isinstance(x, list):
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{p}[{i}]")
    elif x != y:
        diffs.append((p, x, y))
walk(a, b)
for p_, x, y in diffs:
    print(f"    {p_}: {x} -> {y}")
PYEOF
      fail=1
    fi
  fi
done

rm -rf $TMP
if (( fail )); then
  print -- "\nAT LEAST ONE SCRIPT IS NOT REPRODUCIBLE."
  exit 1
fi
print -- "\nAll checked scripts are byte-reproducible."
