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
#
# THE ONE STRUCTURAL RULE, and it is enforced rather than described. There are
# three lists here: `ARGS` (what a script is run with), `default_targets` (what
# a bare invocation runs) and `EXCLUDED` (what is too slow to default, named
# with its measured cost). Every ARGS key must be in exactly one of the last
# two, and the guard above the loop exits 2 if it is not. That rule is written
# in blood: p44 said "so it sits in the default set" for three rounds while it
# did not, and p32/p34/p37 sat in ARGS and in neither list for eleven -- p34
# being the script this file was written for. A comment could be wrong out
# loud; a list cannot.
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
  # --- the three that were in ARGS and in neither list below -----------------
  # Added here in the 08-19/08-20 rounds and never added to `targets`, so they
  # were named in this file, described nowhere as deliberately excluded, and run
  # twice by nobody. That is exactly p44's shape below, and it is what the
  # EXCLUDED array and the guard under it now make impossible: a key in ARGS is
  # in the default set or in EXCLUDED, and there is no third place to sit.
  # Timed 2026-09-02 with the drive mounted, each script on its own rather than
  # inside a pass of this script.
  #
  # p34 is the script this whole file exists for -- the seed built from
  # `hash(str)` -- and it was the one not in its own default set, with nothing
  # saying why. 105 s a run, so the pair is 3.5 min; both runs sha256
  # d5cb1d4bf7fff96d.
  p34_ksweep    "--months 202001,202012,202312,202402,202512,202606"
  # Reads results_p26/p27/p9.json and the survey micro-data, never the parquet.
  # 4.6 s a run (pair sha256 dac1d1ea8e03a908). One rng stream feeds the
  # 500-draw bootstrap, the 500 permutations and the 400-draw multinomial null
  # IN THAT ORDER, so it is reorder-fragile in precisely the way p44 and p61
  # are, which is what a pair of runs is for.
  p32_pmix      ""
  # 79 months x 3 panels x 3 levels off the cached arrival tables, plus the
  # holiday-free variant. 145 s a run, so the pair is ~4.9 min -- the same order
  # as p49, which is in the default set, and worth paying: `spectrum_dong`'s
  # gap_to_pm_max is the 0.00202 that SI 3 names as a lower bound, and the
  # abstract's "at most 0.0258" is p67's distance over these same matrices, so
  # p37 is the only producer of either. No rng anywhere in it; what a pair of runs catches is the pandas
  # pivot_table ordering underneath `matrices`. Both runs sha256 8b51858178530cfb.
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
  # --- the 2026-08-31 round --------------------------------------------------
  # p64 replaces the seven-point sign test with a regression over all 79 months
  # and recounts the rotation test's effective alignments. It reads results
  # files only (p42's series, p41/p51/p54/p58/p63 for anchors), touches no
  # parquet and holds no rng, so a pair of runs is cheap and says the HAC
  # arithmetic is deterministic. Seconds.
  p64_semreg    ""
  # --- the 2026-09-01 round --------------------------------------------------
  # p65 re-reads the survey CSV like p57 and then runs p27's own estimator four
  # times per arm. No rng, but it is NOT deterministic by inspection: the three
  # household arms build boolean masks with pandas .apply over free text and the
  # panel column is reassigned on a copy, so what a pair of runs catches here is
  # a row order or a mask that depends on something other than the file. About a
  # minute a run.
  p65_chaereply ""
  # --- the 2026-09-05 round --------------------------------------------------
  # p66 re-indexes p39/p59's bound from the design beta the grid was laid out on
  # onto the realised beta the survey is measured in. It reads two results files,
  # divides one stored median by another, and interpolates; no parquet, no rng,
  # no simulation. A pair of runs says the interpolation and the dict ordering
  # behind the conversion table are stable. Under a second.
  p66_betareal  ""
  # p67 recomputes the rank-one-to-null distance on p37's 79 stored matrices.
  # numpy's SVD is the only thing here that could drift, and that is exactly
  # what a pair of runs is for -- the phase exists because a difference of two
  # norms was published where a norm of a difference was meant, so its own
  # arithmetic had better be reproducible. Seconds.
  p67_rank1gap  ""
  # p69 re-draws every grid point p39 and p59 ran, from p39's own seeds and
  # stored knobs, to get the venue level in the SURVEY's pairing convention. Its
  # anchors are bit-exact against results_p39/p59, so a drift here is a drift in
  # the world, not in the reading -- but the multinomial that draws phi is the
  # one rng in it and it is drawn 8,800 times. About 9 minutes a run, so the pair
  # is a fifth of an hour and this one is named explicitly rather than sitting in
  # the default set:
  #   eda/determinism_check.sh p69_betacollapse
  p69_betacollapse ""
  # --- the 2026-09-01 round, part two ----------------------------------------
  # THE ABSTRACT'S OWN COUNTS. data_inventory supplies "79 consecutive months"
  # and "10.2 billion records", and it is the one producer in this project with
  # no assert of ANY kind -- it counts what it finds and says so. It is also the
  # only entry here that does not write results_pNN.json, which is what the OUT
  # map below exists for. About 39 s on the first run after a mount (the
  # directory walk is cold) and under a second warm, because DuckDB answers
  # count(*) out of the parquet footers rather than by reading rows. Cheap
  # enough for the default set; both runs sha256 ef2c27eacb277a88.
  # Its one plausible drift is not a seed but a dict: `months_partial` is built
  # from a filesystem glob and keeps that order, so it would move between runs
  # the moment any month stops holding exactly 24 files. It holds none today,
  # which is the point of running it twice rather than reasoning about it.
  data_inventory ""
  # p2 writes §2.1's masked-cell share and the 475,266 duplicate-key count that
  # CLAUDE.md calls load-bearing. 7 min 45 s cold and 5 min 43 s warm a run, so
  # a pair is 11-16 minutes and it is named explicitly rather than sitting in
  # the default set:   eda/determinism_check.sh p2_masking
  #
  # ⚠️ AND IT DOES NOT PASS, measured 2026-09-02: three runs (the committed
  # file, and two here) give three different sha256. The cause is row ORDER,
  # not arithmetic -- five of its twelve top-level keys are `GROUP BY` results
  # with no ORDER BY, and `connect()` runs DuckDB at PRAGMA threads=8, so the
  # groups come back in whatever order the hash aggregate finishes them in.
  # Canonicalising each list makes all twelve keys identical across all three
  # runs, so no VALUE moves: `dedup` (475,266) and `cell_vs_volume_total` (the
  # 26.1% share) are byte-identical run to run and only the heat-map row lists
  # permute. Recorded here rather than fixed, because the fix is an ORDER BY in
  # p2_masking.py and that file belongs to another workstream. Until it lands,
  # this entry is the record that the failure is known and bounded.
  p2_masking    ""
  # p8 is p2's shape at the same resolution and is here for the same reasons:
  # no rng, no asserts, and every number it writes is a DuckDB parallel float
  # sum. 7 min 43 s cold and 5 min 31 s warm a run, so a pair is 11-15 minutes
  # and it too is named rather than defaulted:
  #   eda/determinism_check.sh p8_panel
  # One thing is its own: it REWRITES derived/panel_core.parquet on every run,
  # which nothing else in this file does, so an interrupted pair leaves that
  # file from run 1 -- harmless, since p14/p15 re-read it and it is
  # regenerated, but worth knowing before it is blamed for something else.
  #
  # ⚠️ AND IT DOES NOT PASS EITHER, measured 2026-09-02: three runs, three
  # sha256 -- but for the OTHER reason, and the distinction is the whole value
  # of running both. p2 permutes rows and moves no value; p8 moves values. 50
  # leaves across five keys drift in the last one or two ulps, worst relative
  # 1.36e-14, and sorting the lists does not make them agree. That is p36's
  # finding arriving in a second script: `connect()` runs at PRAGMA threads=8,
  # DuckDB's parallel hash aggregate combines partial sums in thread-completion
  # order, and float addition is not associative. p36 fixed it by pinning
  # threads=1 at a cost of about four seconds per matrix. Nothing p8 concludes
  # moves at 1e-14; recorded here rather than fixed, because the fix belongs in
  # p8_panel.py / common.connect() and those are another workstream's files.
  p8_panel      ""
)

# The results file each script writes, for the ones where it is NOT
# results_pNN.json. `num=${name%%_*}` turns p33_coverage into p33 and works for
# every phase script; data_inventory has no phase number at all and writes
# results_inventory.json, so the derived name would be results_data.json and the
# pair would compare two files that have never existed -- silently, since the
# comparison below is skipped when neither run produced a file.
typeset -A OUT
OUT=(
  data_inventory results_inventory.json
)

# NOT `targets=(${@:-a b c})`: with an empty $@ that yields ONE element holding
# the whole default string, so the loop would look for a script named
# "p33_coverage p35_seir p36_recompute" and report it as a failure.
default_targets=(p33_coverage p35_seir p36_recompute p41_semester p42_monthscope
                 p49_dongwor p50_natpop p53_natbeta p54_semrecount
                 p44_beta p55_fig7 p56_fig5 p47_fig34 p48_fig1 p57_placecode
                 p58_semshift p60_agescale p61_survspec p62_multiplicity p63_fig2
                 p46_fig6 p64_semreg p65_chaereply p66_betareal p67_rank1gap
                 p32_pmix p34_ksweep p37_timeseries data_inventory)

# THE EXCLUSIONS ARE DATA, NOT PROSE. Every long script above already explains
# in a comment why it is named rather than defaulted, and p44's comment said "so
# it sits in the default set" for three rounds while it did not, because a
# comment cannot be wrong out loud. p32/p34/p37 then repeated it in the other
# direction: in ARGS, in no list, described nowhere. So the exclusions are a
# list, and the guard below refuses to start if any ARGS key is in neither
# `default_targets` nor `EXCLUDED`, or in both. A script added to ARGS and to
# neither is now a hard error instead of another eleven quiet rounds.
EXCLUDED=(p38_ksweep79 p39_recovery p45_r0 p51_natsym p52_r0nat p59_betastar
          p69_betacollapse p2_masking p8_panel)

for name in ${(k)ARGS}; do
  in_def=$(( ${default_targets[(Ie)$name]} > 0 ))
  in_exc=$(( ${EXCLUDED[(Ie)$name]} > 0 ))
  if (( in_def + in_exc != 1 )); then
    where=$(( in_def + in_exc ))
    print -u2 -- "$name is in ARGS and in $where of {default_targets, EXCLUDED}; it must be in exactly one."
    print -u2 -- "  the default set, or EXCLUDED with its measured cost as the reason."
    exit 2
  fi
done

if (( $# )); then
  targets=($@)
else
  targets=($default_targets)
fi
fail=0

for name in $targets; do
  num=${name%%_*}                       # p33_coverage -> p33
  out=$REPO/eda/${OUT[$name]:-results_${num}.json}
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
