#!/bin/zsh
# Fill the 생활이동 행정동 tree from 2021-01 to 2026-07, one month per process.
#
# dl_mobility.py can take every month in one --months list, but a 60-month run
# is ~20 hours and its --etl path raises RuntimeError (not BadFile) on a row
# mismatch, which would kill the whole loop. So: one invocation per month, the
# loop survives a bad month, and the failure is recorded and skipped past.
#
# Resumable and idempotent. A month whose parquet already holds 24 files is
# skipped without touching the network or re-reading 6 GB of CSV, so re-running
# after a kill costs seconds per finished month.
#
# SEVERAL PASSES, not one. The portal's failures are intermittent and correlated
# in time: on 2026-08-19 it served eleven months, then refused every attempt on
# the next two for twenty minutes, then served again. A single pass turns a bad
# twenty minutes into a permanently missing month. So the month list is walked
# repeatedly, each pass skipping what is already complete, until a whole pass
# adds nothing -- at which point the failures are real and the loop stops
# instead of hammering. PAUSE_BETWEEN_PASSES gives the server room to recover.
#
# THE CIRCUIT BREAKER, and why it matters more than the retry budget. The
# portal's failures are correlated in TIME, not in month: when it is corrupting
# the TLS stream it corrupts every month's, and when it recovers it serves every
# month. dl_mobility.py spends ~30 minutes exhausting its 15 attempts on one
# month, so a pass that walks 50 dead months costs 25 hours to learn one fact
# that three months would have told us. After MAX_CONSEC_FAIL months fail
# end-to-end, the pass is abandoned and the loop waits -- which is why MAX_PASSES
# is large: passes are cheap when the network is bad and only the good windows
# cost real time.
#
#   nohup caffeinate -is eda/dl_mobility_fill.sh >> "$LOG" 2>&1 &   (no setsid on macOS)
#
# Each month costs ~1.07 GB down, ~5.9 GB of CSV and ~0.58 GB of parquet on the
# drive; the zip is deleted once its CSVs verify (--delete-zip), sha256 kept in
# raw/mobility/manifest.json. 61 months ≈ 395 GB.
set -u

REPO=/Users/waynliu/Documents/Research/Explore/Korean
PY=$REPO/.venv/bin/python
DATA="/Volumes/My Passport/Korean_Data"
PARQUET="$DATA/parquet"
STATUS="$DATA/raw/mobility/fill_status.tsv"
MIN_FREE_GB=60          # stop before the drive is too full to expand a zip
MAX_PASSES=24           # give a flaky month many separated chances
PAUSE_BETWEEN_PASSES=900
MAX_CONSEC_FAIL=3       # see the circuit breaker below

MONTHS=()
for y in 2021 2022 2023 2024 2025 2026; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    ym="$y$m"
    [[ $ym -gt 202607 ]] && continue      # 202607 is the last month published
    MONTHS+=($ym)
  done
done

echo "=== fill started $(date -u +%FT%TZ) : ${#MONTHS[@]} months in range ==="
[[ -f $STATUS ]] || echo "ym\tstatus\tutc\tsecs" > $STATUS

# How many parquet files a month already has. The (N) qualifier is doing real
# work: zsh aborts a command whose glob matches nothing, which is every month not
# yet fetched. The obvious repairs are both wrong. `setopt NULL_GLOB` DELETES the
# unmatched pattern, so `ls $PARQUET/ym=202112/*.parquet` degrades to a bare `ls`
# that lists the working directory and reports its file count as the month's --
# 16 on this repo, and any directory holding 24 entries would mark an unfetched
# month complete and skip it silently. Piping ls into wc adds a subprocess and
# the same trap. (N) confines null-glob behaviour to this one pattern and yields
# an empty array, so a missing month counts 0.
done_count () { local f=($PARQUET/ym=$1/*.parquet(N)); print -r -- ${#f} }

for pass in {1..$MAX_PASSES}; do
  todo=()
  for ym in $MONTHS; do
    [[ $(done_count $ym) == 24 ]] || todo+=($ym)
  done
  if (( ${#todo} == 0 )); then
    echo "=== pass $pass: nothing left to fetch ==="
    break
  fi
  echo "=== pass $pass/$MAX_PASSES $(date -u +%FT%TZ) : ${#todo} months to go: $todo ==="
  gained=0
  consec=0
  for ym in $todo; do
    if (( consec >= MAX_CONSEC_FAIL )); then
      echo "=== $consec months failed end-to-end in a row: the portal is not"
      echo "    serving right now. Abandoning pass $pass rather than spending"
      echo "    30 minutes per month to learn the same thing ${#todo} times. ==="
      break
    fi
    free=$(df -g "$DATA" | tail -1 | awk '{print $4}')
    if [[ $free -lt $MIN_FREE_GB ]]; then
      echo "!!! only ${free} GB free on the drive, stopping before $ym"
      echo "$ym\tSTOPPED_DISK_FULL\t$(date -u +%FT%TZ)\t0" >> $STATUS
      exit 0
    fi
    n=$(done_count $ym)
    echo "=== $ym start $(date -u +%FT%TZ)  (pass $pass, ${free} GB free, $n/24 parquet) ==="
    t0=$SECONDS
    $PY $REPO/eda/dl_mobility.py --months $ym --etl --delete-zip
    rc=$?
    secs=$((SECONDS - t0))
    n=$(done_count $ym)
    if [[ $rc == 0 && $n == 24 ]]; then
      st=ok
      gained=$((gained + 1))
      consec=0
    else
      st="FAILED(rc=$rc,parquet=$n,pass=$pass)"
      consec=$((consec + 1))
    fi
    echo "=== $ym $st in ${secs}s ==="
    echo "$ym\t$st\t$(date -u +%FT%TZ)\t$secs" >> $STATUS
    sleep 5
  done
  # A pass that gained nothing is NOT a reason to stop any more -- with the
  # circuit breaker, a zero-gain pass is the expected outcome of a bad window and
  # costs about ninety minutes. Waiting is the whole strategy.
  echo "=== pass $pass gained $gained (${consec} consecutive failures at the end);"
  echo "    pausing ${PAUSE_BETWEEN_PASSES}s ==="
  sleep $PAUSE_BETWEEN_PASSES
done

echo "=== fill finished $(date -u +%FT%TZ) ==="
