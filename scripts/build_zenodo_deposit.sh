#!/usr/bin/env bash
# Build the Zenodo deposit zip for a version of the replication archive.
#
# WHAT GOES PUBLIC IS DECIDED IN EXACTLY ONE PLACE: the EXCLUDE list below.
# The deposit is built from the repo's TRACKED files minus EXCLUDE: whatever you
# publish came from the commit you built it from, and a file that is not tracked
# cannot leak into it.
#
# THE PUBLIC MIRROR IS BUILT BY THE SAME RULE, which is what `--mirror` is for.
# The Brazil archive worked the other way round -- an already-scrubbed public
# GitHub tree, with the deposit built from it -- and for five weeks this repo
# had no mirror at all, so the exclusion list had exactly one consumer. It now
# has two, and they must not drift: a mirror assembled by hand would be a second
# answer to "what is public", and the first time the two disagreed the private
# half would be whichever one nobody re-derived. So `--mirror DIR` runs the same
# copy, the same exclusions and the same front matter into DIR, and stops before
# the two things that would claim a release it is not making: it writes no
# release/zenodo_manifests/vN.txt and no zip.
#
#   bash scripts/build_zenodo_deposit.sh --mirror /tmp/mirror
#
# A mirror is a working tree, not a version. Zenodo versions are cut from it
# with the ordinary form of this command and are what the manuscript cites.
#
#   deposit = (git ls-files, minus EXCLUDE)
#           + the deposit-side front matter in release/deposit/
#           + generated EXCLUDED_FILES.txt and MANIFEST.sha256
#           [+ Tier A payload, only with --tier-a, only when the gate is lifted]
#
# THE TIER A GATE, and why --tier-a is not the default.
# release/data_release_and_packaging.md §5 item 1: the Seoul Open Data 문의하기
# sent 2026-08-24 "does not block the analysis, it blocks the upload". Every
# Tier A object in release/data_descriptor_plan.md §3 is marked §6-conditional.
#
# The first version of this gate had two states: the reply is pasted in, or
# refuse. That is not what the policy says. data_descriptor_plan.md §4.3 has
# THREE branches, and the middle one -- Q1 confirmed, Q2 unanswered -- is
# explicitly "可以走". A gate stricter than the rule it enforces blocks work the
# rule allows, which is what happened here for five days.
#
# So --tier-a now reads release/tier_a_basis.md and requires a declared basis:
# one of `reply`, `q1-only` or `three-grounds`. Whichever is declared, the file
# ships inside the deposit as TIER_A_BASIS.md, so the reason the aggregates were
# released is visible to whoever downloads them rather than living in a private
# planning document. The same file declares `a7:`, because A7's sources are
# 행정안전부 / 법무부 / SGIS and no answer from Seoul can speak for them.
#
# Usage:
#   bash scripts/build_zenodo_deposit.sh 1.0.0
#   bash scripts/build_zenodo_deposit.sh --mirror ../korean-mobility-mixing
#   bash scripts/build_zenodo_deposit.sh 1.1.0 --tier-a
#   bash scripts/build_zenodo_deposit.sh 1.0.1 --diff-against release/zenodo_manifests/v1.0.0.txt
#   bash scripts/build_zenodo_deposit.sh 1.0.0 --allow-dirty
#
# Then, on Zenodo: create a draft (or "New version" of an existing record),
# upload with scripts/zenodo_upload_deposit.sh, review in the browser, publish
# by hand. The concept DOI follows the new version automatically, so the
# manuscript's citation never needs to change.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MIRROR_DIR=""
if [ "${1:-}" = "--mirror" ]; then
  MIRROR_DIR="${2:-}"
  if [ -z "$MIRROR_DIR" ]; then
    echo "usage: bash scripts/build_zenodo_deposit.sh --mirror <dir> [--tier-a] [--allow-dirty]" >&2
    exit 2
  fi
  shift 2
  VERSION="mirror"
else
  VERSION="${1:-}"
  if [ -z "$VERSION" ] || [[ "$VERSION" == -* ]]; then
    echo "usage: bash scripts/build_zenodo_deposit.sh <version> [--tier-a] [--diff-against MANIFEST] [--allow-dirty]" >&2
    echo "       bash scripts/build_zenodo_deposit.sh --mirror <dir> [--tier-a] [--allow-dirty]" >&2
    echo "example: bash scripts/build_zenodo_deposit.sh 1.0.0" >&2
    exit 2
  fi
  shift
fi

MANIFEST_DIR="$ROOT/release/zenodo_manifests"
mkdir -p "$MANIFEST_DIR"
DIFF_AGAINST="$(ls -1 "$MANIFEST_DIR"/*.txt 2>/dev/null | sort -V | tail -1 || true)"
TIER_A=0
ALLOW_DIRTY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier-a)       TIER_A=1; shift ;;
    --diff-against) DIFF_AGAINST="$2"; shift 2 ;;
    --allow-dirty)  ALLOW_DIRTY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

BUILD="$ROOT/build/zenodo"
STAGE="$BUILD/korean-mobility-mixing-v${VERSION}"
ZIP="$BUILD/korean-mobility-mixing-v${VERSION}.zip"
if [ -n "$MIRROR_DIR" ]; then
  # A mirror stages into the caller's directory, and never into build/zenodo,
  # so it cannot overwrite the tree a published version was cut from. The .git
  # of an existing mirror checkout is not ours to delete, so only the tracked
  # payload is cleared: `git status` in the mirror then shows a deletion for
  # anything this build no longer produces, which is the point.
  mkdir -p "$MIRROR_DIR"
  STAGE="$(cd "$MIRROR_DIR" && pwd)"
  ZIP=""
fi

# --- the exclusion list ----------------------------------------------------
# Each entry is a prefix match against a repo-relative path. Every one of these
# is a decision, so each carries its reason; EXCLUDED_FILES.txt reproduces them
# in the deposit so a reader can see what was withheld without asking.
EXCLUDE_PREFIXES=(
  "Email_Discussion/"       # advisor correspondence and outgoing letters; private
  "release/"                # descriptor and packaging planning, incl. this gate's own record
  "archive/"                # superseded drafts, and manuscript_JBI.md is a DIFFERENT PAPER
  "paper/"                  # the manuscript itself; the deposit is code, not the article
  "cases/"                  # KDCA daily case counts: upstream licence not yet checked
  "CLAUDE.md"               # agent instructions, not documentation for a reader
  "OVERVIEW.md"             # internal status board, in Chinese, references private letters
  "meeting_note"            # meeting notes
  "drive_todo"              # run-sheet for the next time the drive is mounted; internal, cites the letters
  "JRSI_投稿規定"           # journal submission notes
  ".gstack/"                # local tooling
  "authors.md"              # symlink to the author-fields sheet shared with another submission
                            # (departments, e-mails, ORCIDs); cp follows the link, so it would ship
)
# Tier A payload: the derived aggregates. Held back unless --tier-a.
#
# This is matched on the BASENAME, not on a path prefix. The first version of
# this script used the prefix "eda/results_" and silently shipped 28 Tier A
# files, because the frozen snapshots live at eda/archive/<date>/results_*.json
# and a prefix rule does not see them. A rule that has to enumerate every
# directory a sensitive file might sit in is the wrong rule.
is_tier_a() {
  case "$(basename "$1")" in
    results_*.json) return 0 ;;
    *) return 1 ;;
  esac
}

is_excluded() {
  local f="$1" p
  for p in "${EXCLUDE_PREFIXES[@]}"; do [[ "$f" == "$p"* ]] && return 0; done
  if [ "$TIER_A" -eq 0 ] && is_tier_a "$f"; then return 0; fi
  return 1
}

# --- preflight -------------------------------------------------------------
if [ "$ALLOW_DIRTY" -eq 0 ] && [ -n "$(git -c core.quotePath=false status --porcelain)" ]; then
  echo "ERROR: working tree is dirty. A deposit must be reproducible from a commit." >&2
  echo "       Commit first, or pass --allow-dirty if you know why." >&2
  git -c core.quotePath=false status --short >&2
  exit 1
fi

BASIS_FILE="$ROOT/release/tier_a_basis.md"
TIER_A_BASIS=""
TIER_A_A7=""
if [ "$TIER_A" -eq 1 ]; then
  if [ ! -f "$BASIS_FILE" ]; then
    echo "ERROR: --tier-a refused: $BASIS_FILE does not exist." >&2
    echo "       It is the declaration of WHY the aggregates may be released," >&2
    echo "       and it ships inside the deposit. Write it first." >&2
    exit 1
  fi
  TIER_A_BASIS=$(grep -m1 -E '^basis:' "$BASIS_FILE" | sed 's/^basis:[[:space:]]*//' | tr -d '[:space:]')
  TIER_A_A7=$(grep -m1 -E '^a7:'    "$BASIS_FILE" | sed 's/^a7:[[:space:]]*//'    | tr -d '[:space:]')

  case "$TIER_A_BASIS" in
    reply)
      # The strongest branch, so it is the one with a second check: the reply
      # must actually be in the release file, not merely asserted here.
      if ! grep -q "문의하기 回覆全文" "$ROOT/release/data_release_and_packaging.md" 2>/dev/null; then
        echo "ERROR: basis is 'reply' but data_release_and_packaging.md has no" >&2
        echo "       section containing \"문의하기 回覆全文\". Paste the reply" >&2
        echo "       into §1 as the fifth source row, or declare a weaker basis." >&2
        exit 1
      fi ;;
    q1-only|three-grounds)
      : ;;   # both are documented branches of data_descriptor_plan.md §4.3
    *)
      cat >&2 <<EOF
ERROR: --tier-a refused: basis is "${TIER_A_BASIS:-<missing>}".

  release/tier_a_basis.md must declare one of three values, and each commits you
  to something different. The file itself explains them; in one line each:

    reply          the written reply is in data_release_and_packaging.md §1
    q1-only        Q1 confirmed, Q2 unanswered -- §4.3's middle branch, "可以走"
    three-grounds  releasing on §1's own three-layer judgement, reply not yet in

  This is deliberately not a yes/no flag. Whichever you declare is copied into
  the deposit as TIER_A_BASIS.md, so a reader can see on what authority the
  aggregates were released.
EOF
      exit 1 ;;
  esac

  if [ "$TIER_A_A7" != "cleared" ] && [ "$TIER_A_A7" != "blocked" ]; then
    echo "ERROR: --tier-a refused: a7 is \"${TIER_A_A7:-<missing>}\", expected" >&2
    echo "       'cleared' or 'blocked'. A7's sources are 행정안전부 / 법무부 /" >&2
    echo "       SGIS and no answer from Seoul speaks for them, so it is declared" >&2
    echo "       separately rather than folded into basis." >&2
    exit 1
  fi
fi

COMMIT="$(git rev-parse --short HEAD)"
echo "building v${VERSION} from commit ${COMMIT}$([ "$ALLOW_DIRTY" -eq 1 ] && echo ' (dirty)')"
if [ "$TIER_A" -eq 1 ]; then
  echo "Tier A payload: INCLUDED   basis=$TIER_A_BASIS  a7=$TIER_A_A7"
else
  echo "Tier A payload: held back (gate open)"
fi

if [ -n "$MIRROR_DIR" ]; then
  find "$STAGE" -mindepth 1 -maxdepth 1 ! -name .git ! -name .github -exec rm -rf {} +
else
  rm -rf "$STAGE" "$ZIP"
fi
mkdir -p "$STAGE"

# --- copy the tracked, non-excluded tree -----------------------------------
n_in=0; n_out=0
: > "$BUILD/.excluded.$$"
while IFS= read -r f; do
  if is_excluded "$f"; then
    printf '%s\n' "$f" >> "$BUILD/.excluded.$$"
    n_out=$((n_out+1))
    continue
  fi
  mkdir -p "$STAGE/$(dirname "$f")"
  cp "$f" "$STAGE/$f"
  n_in=$((n_in+1))
done < <(git -c core.quotePath=false ls-files)
echo "  ${n_in} files copied, ${n_out} withheld"

# --- deposit-side front matter --------------------------------------------
# These live in release/deposit/ in the repo and at the zip root in the deposit,
# because a reader of the archive should meet them first, not three levels down.
if [ -d "$ROOT/release/deposit" ]; then
  for f in "$ROOT/release/deposit/"*; do
    [ -f "$f" ] || continue
    cp "$f" "$STAGE/$(basename "$f")"
    echo "  front matter: $(basename "$f")"
  done
fi

# --- the Tier A declaration travels with the data it releases --------------
if [ "$TIER_A" -eq 1 ]; then
  cp "$BASIS_FILE" "$STAGE/TIER_A_BASIS.md"
  echo "  TIER_A_BASIS.md  (basis=$TIER_A_BASIS, a7=$TIER_A_A7)"
fi

# --- generated: what was withheld, and why ---------------------------------
{
  echo "# Files present in the source repository and deliberately not deposited"
  echo "#"
  echo "# Built from commit ${COMMIT} on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# Tier A payload: $([ "$TIER_A" -eq 1 ] && echo included || echo withheld)"
  echo "#"
  echo "# Reasons, by prefix:"
  echo "#   Email_Discussion/  advisor correspondence and outgoing letters; private"
  echo "#   release/           descriptor and packaging planning"
  echo "#   archive/           superseded drafts; archive/manuscript_JBI.md is a different paper"
  echo "#   paper/             the manuscript itself; this deposit is code, not the article"
  echo "#   cases/             KDCA daily case counts; upstream licence not yet checked"
  echo "#   CLAUDE.md          agent instructions"
  echo "#   OVERVIEW.md        internal status board"
  echo "#   results_*.json     Tier A derived aggregates, wherever they sit (including"
  echo "#                      the frozen snapshots under eda/archive/); see EXCLUSIONS.md"
  echo "#"
  echo "# Not listed here because git never tracked them: the parquet tree (42 GB),"
  echo "# derived/ (9.2 GB), raw/ (~77 GB of source zips) and .env. The download"
  echo "# helpers under eda/ retrieve the source files and verify them by SHA-256."
  echo
  sort "$BUILD/.excluded.$$"
} > "$STAGE/EXCLUDED_FILES.txt"
rm -f "$BUILD/.excluded.$$"

# --- generated: manifest ---------------------------------------------------
# -path prunes are for --mirror: the manifest describes the deposit, and a
# mirror checkout also carries a .git. Without this the first build after
# `git init` hashed 278 objects out of .git into the manifest, which then
# changed on every commit -- a file whose job is to be stable, made to churn by
# the act of recording it.
( cd "$STAGE" && find . \( -name .git -o -name .github \) -prune -o \
    -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 shasum -a 256 ) > "$STAGE/MANIFEST.sha256"
echo "  MANIFEST.sha256: $(wc -l < "$STAGE/MANIFEST.sha256" | tr -d ' ') entries"

if [ -n "$MIRROR_DIR" ]; then
  echo
  echo "mirrored the deposit tree into $STAGE"
  echo "  no zip and no release/zenodo_manifests entry were written: a mirror is"
  echo "  a working tree, not a release. Commit it in that repo; cut Zenodo"
  echo "  versions with the ordinary form of this command."
  exit 0
fi

# --- the file list, for the next build's diff ------------------------------
( cd "$STAGE" && find . -type f | sed 's|^\./||' | sort ) > "$MANIFEST_DIR/v${VERSION}.txt"

# --- say what changed ------------------------------------------------------
if [ -n "$DIFF_AGAINST" ] && [ -f "$DIFF_AGAINST" ] \
   && [ "$(basename "$DIFF_AGAINST")" != "v${VERSION}.txt" ]; then
  echo
  echo "file-set diff against $(basename "$DIFF_AGAINST"):"
  added=$(comm -13 "$DIFF_AGAINST" "$MANIFEST_DIR/v${VERSION}.txt" | wc -l | tr -d ' ')
  removed=$(comm -23 "$DIFF_AGAINST" "$MANIFEST_DIR/v${VERSION}.txt" | wc -l | tr -d ' ')
  echo "  +${added} added, -${removed} removed"
  comm -13 "$DIFF_AGAINST" "$MANIFEST_DIR/v${VERSION}.txt" | sed 's|^|    + |' | head -40
  comm -23 "$DIFF_AGAINST" "$MANIFEST_DIR/v${VERSION}.txt" | sed 's|^|    - |' | head -40
  if [ "$added" -eq 0 ] && [ "$removed" -eq 0 ]; then
    echo "    (no path added or removed -- this does NOT mean no file CHANGED;"
    echo "     compare MANIFEST.sha256 against the previous build to see that)"
  fi
fi

# --- zip -------------------------------------------------------------------
( cd "$BUILD" && zip -qr "$(basename "$ZIP")" "$(basename "$STAGE")" -x '*.DS_Store' )
echo
echo "wrote $ZIP"
echo "  $(du -h "$ZIP" | cut -f1), $(unzip -Z1 "$ZIP" | wc -l | tr -d ' ') entries"
echo
echo "Next:"
echo "  1. Create a draft on Zenodo (web UI), note its record id"
echo "  2. bash scripts/zenodo_upload_deposit.sh <recid> $ZIP"
echo "  3. Review the draft in the browser, then publish BY HAND"
echo "  4. bash scripts/verify_zenodo_record.sh $VERSION"
