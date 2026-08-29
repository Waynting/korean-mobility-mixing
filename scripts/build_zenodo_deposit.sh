#!/usr/bin/env bash
# Build the Zenodo deposit zip for a version of the replication archive.
#
# WHAT GOES PUBLIC IS DECIDED IN EXACTLY ONE PLACE: the EXCLUDE list below.
# This repo has no public GitHub mirror (`git remote -v` is empty), so unlike
# the Brazil archive there is no already-scrubbed tree to build from. The
# deposit is therefore built from the repo's TRACKED files minus EXCLUDE, which
# has the same property for the same reason: whatever you publish came from the
# commit you built it from, and a file that is not tracked cannot leak into it.
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
# A7 (regpop / foreign / dong_covariates) carries a second gate that the Seoul
# answer cannot lift: its sources are 행정안전부 / 법무부 / SGIS, each with its
# own redistribution terms, and those have not been checked one by one. So the
# derived aggregates are held back by default and --tier-a refuses to run until
# you record the answer in release/data_release_and_packaging.md §1.
#
# Usage:
#   bash scripts/build_zenodo_deposit.sh 1.0.0
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

VERSION="${1:-}"
if [ -z "$VERSION" ] || [[ "$VERSION" == -* ]]; then
  echo "usage: bash scripts/build_zenodo_deposit.sh <version> [--tier-a] [--diff-against MANIFEST] [--allow-dirty]" >&2
  echo "example: bash scripts/build_zenodo_deposit.sh 1.0.0" >&2
  exit 2
fi
shift

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
  "JRSI_投稿規定"           # journal submission notes
  ".gstack/"                # local tooling
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

if [ "$TIER_A" -eq 1 ]; then
  if ! grep -q "문의하기 回覆全文" "$ROOT/release/data_release_and_packaging.md" 2>/dev/null; then
    cat >&2 <<'EOF'
ERROR: --tier-a refused.

  release/data_release_and_packaging.md §5 item 1 records that the Seoul Open
  Data 문의하기 was sent on 2026-08-24 and that it BLOCKS THE UPLOAD until a
  reply arrives. This script looks for the string "문의하기 回覆全文" in that
  file as the marker that the reply has been pasted in. It is not there.

  Two gates, not one:
    1. the Seoul answer, for Tier A objects A1-A6, A8, A9
    2. 행정안전부 / 법무부 / SGIS terms, for A7 (regpop / foreign /
       dong_covariates) -- the Seoul answer cannot lift this one

  If the reply has arrived, paste it into §1 of that file as the fifth source
  row, under a heading containing "문의하기 回覆全文", and re-run.
EOF
    exit 1
  fi
fi

COMMIT="$(git rev-parse --short HEAD)"
echo "building v${VERSION} from commit ${COMMIT}$([ "$ALLOW_DIRTY" -eq 1 ] && echo ' (dirty)')"
echo "Tier A payload: $([ "$TIER_A" -eq 1 ] && echo INCLUDED || echo 'held back (gate open)')"

rm -rf "$STAGE" "$ZIP"
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
( cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 shasum -a 256 ) > "$STAGE/MANIFEST.sha256"
echo "  MANIFEST.sha256: $(wc -l < "$STAGE/MANIFEST.sha256" | tr -d ' ') entries"

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
