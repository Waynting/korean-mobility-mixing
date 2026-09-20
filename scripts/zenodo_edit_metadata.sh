#!/usr/bin/env bash
# Bring a PUBLISHED Zenodo record's description in line with
# release/zenodo_metadata.json, without cutting a new version.
#
# Files on a published record are immutable; metadata is not. This is the
# sequence written in the header of scripts/verify_zenodo_record.sh, made
# runnable: it uses the InvenioRDM API throughout and never the legacy
# /api/deposit/depositions endpoint, which holds ONE licence and would
# silently drop MIT from a record that carries both MIT and CC BY 4.0.
#
#   1. GET  /api/records/<id>           (InvenioRDM serialisation)  -> live.json
#   2. POST /api/records/<id>/draft     create (or reopen) the edit draft
#   3. GET  /api/records/<id>/draft     -> draft.json
#   4. PUT  /api/records/<id>/draft     draft.json with ONLY metadata.description replaced
#   5. GET  /api/records/<id>/draft     re-read, diff metadata against live.json,
#                                       refuse to go on if anything but description moved
#   6. POST /api/records/<id>/draft/actions/publish      only with --publish
#   7. GET  /api/records/<id>           confirm the live record now reads as intended
#
# Without --publish the draft is left open for inspection at
# https://zenodo.org/uploads/<id>; Zenodo shows it to the owner as a pending
# edit and nothing has changed for the public. Publishing is a separate,
# explicit step because it cannot be undone.
#
# Only the description is edited on purpose. The record's title is the
# deposit's own and is not the paper's; retitling is a decision, not a sync,
# and is not something this script does.
#
# Usage:
#   bash scripts/zenodo_edit_metadata.sh 22152090              # stage, diff, stop
#   bash scripts/zenodo_edit_metadata.sh 22152090 --publish    # stage, diff, publish, verify
#   bash scripts/zenodo_edit_metadata.sh 22152090 --check      # read-only: is the live text
#                                                              # what the repo says? (no token,
#                                                              # no draft; for after a web edit)
#
# Token, in order: $ZENODO_TOKEN, --token-file PATH, ~/.config/zenodo/token,
# then the two sibling archives' zenodo-upload/.env. Never echoed, never in a URL.
set -euo pipefail

RECID="${1:-}"
PUBLISH=0
CHECK=0
TOKEN_FILE=""
shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --publish)    PUBLISH=1; shift ;;
    --check)      CHECK=1; shift ;;
    --token-file) TOKEN_FILE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$RECID" ] || [[ "$RECID" == -* ]]; then
  echo "usage: bash scripts/zenodo_edit_metadata.sh <version-recid> [--publish | --check] [--token-file PATH]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTENDED="$ROOT/release/zenodo_metadata.json"
[ -f "$INTENDED" ] || { echo "ERROR: $INTENDED not found" >&2; exit 1; }

RDM="Accept: application/vnd.inveniordm.v1+json"
JSON="Content-Type: application/json"
API="https://zenodo.org/api/records/$RECID"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/zenodo-edit.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
die() { echo "ERROR: $*" >&2; exit 1; }

# A 504 from Zenodo comes back as an HTML page, and json.load on it dies with
# "Expecting value: line 1 column 1". Check the status before parsing anything.
fetch() {  # fetch <out> <curl args...>
  local out="$1"; shift
  local code
  code=$(curl -sS --max-time 90 -o "$out" -w '%{http_code}' "$@") || die "curl failed"
  echo "$code"
}

# --- --check: the web editor rewrites HTML (entities, attribute order), so a
# record edited by hand is compared as rendered text, not as markup ---------
if [ "$CHECK" -eq 1 ]; then
  code=$(fetch "$WORK/live.json" -H "$RDM" "$API")
  [ "$code" = "200" ] || die "live record: HTTP $code (Zenodo returns 504 when degraded; try later)"
  python3 - "$WORK/live.json" "$INTENDED" <<'PY'
import json, sys, re, difflib, html
# Words, case-folded: the web editor turns <h4> into capitals and <li> into
# "&nbsp; * ", and inline <em>/<strong> would otherwise split a sentence.
def words(s):
    t = html.unescape(re.sub(r'<[^>]+>', ' ', s)).replace('\xa0', ' ')
    return [w for w in t.lower().split() if w != '*']
live = words(json.load(open(sys.argv[1]))['metadata']['description'])
want = words(json.load(open(sys.argv[2]))['metadata']['description'])
if live == want:
    print(f"  live description reads as release/zenodo_metadata.json ({len(live)} words, markup ignored)"); sys.exit(0)
print("  live description differs from the repo (words, markup ignored):")
for l in difflib.unified_diff(live, want, 'live', 'repo', lineterm='', n=2): print("    " + l)
sys.exit(1)
PY
  exit $?
fi

# --- token, without ever printing it ---------------------------------------
TOK="${ZENODO_TOKEN:-}"
if [ -z "$TOK" ]; then
  for cand in "${TOKEN_FILE:-}" "$HOME/.config/zenodo/token" \
              "$HOME/Documents/Research/JBI/zenodo-upload/.env" \
              "$HOME/Documents/Research/Brazel/zenodo-upload/.env"; do
    [ -n "$cand" ] && [ -f "$cand" ] || continue
    TOK="$(grep -E '^ZENODO_TOKEN=' "$cand" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$TOK" ] || TOK="$(tr -d '[:space:]' < "$cand")"   # tolerate a bare token
    [ -n "$TOK" ] && { echo "using token from $cand"; break; }
  done
fi
[ -n "$TOK" ] || { echo "ERROR: no token (set ZENODO_TOKEN or pass --token-file)" >&2; exit 1; }

AUTH=(-H "Authorization: Bearer $TOK")

echo
echo "Editing Zenodo record $RECID"
echo

# --- 1. the live record, as the public sees it -----------------------------
code=$(fetch "$WORK/live.json" -H "$RDM" "$API")
[ "$code" = "200" ] || die "live record: HTTP $code (Zenodo returns 504 when degraded; try later)"

if python3 - "$WORK/live.json" "$INTENDED" <<'PY'
import json, sys
live = json.load(open(sys.argv[1]))['metadata']['description']
want = json.load(open(sys.argv[2]))['metadata']['description']
sys.exit(0 if live == want else 1)
PY
then
  echo "  live description already matches release/zenodo_metadata.json; nothing to do"
  exit 0
fi
echo "  live description differs from release/zenodo_metadata.json"

# --- 2. open the edit draft ------------------------------------------------
# POST creates the draft; if one is already open Zenodo answers with it, so a
# run interrupted after this step can simply be repeated.
code=$(fetch "$WORK/post.json" "${AUTH[@]}" -H "$JSON" -X POST "$API/draft")
case "$code" in
  200|201) echo "  edit draft open (HTTP $code)" ;;
  *) die "could not open an edit draft: HTTP $code (check the token's deposit:write scope)" ;;
esac

# --- 3. read it back in the InvenioRDM shape --------------------------------
code=$(fetch "$WORK/draft.json" "${AUTH[@]}" -H "$RDM" "$API/draft")
[ "$code" = "200" ] || die "cannot read the draft: HTTP $code"

# --- 4. replace ONLY the description, PUT the whole draft back ------------
# Putting the record back as read keeps rights in the [{"id": ...}] shape the
# InvenioRDM schema expects; the fields Zenodo treats as read-only are ignored.
python3 - "$WORK/draft.json" "$INTENDED" "$WORK/put.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d['metadata']['description'] = json.load(open(sys.argv[2]))['metadata']['description']
json.dump(d, open(sys.argv[3], 'w'), ensure_ascii=False)
PY
code=$(fetch "$WORK/put_resp.json" "${AUTH[@]}" -H "$JSON" -H "$RDM" -X PUT --data-binary "@$WORK/put.json" "$API/draft")
[ "$code" = "200" ] || { python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), indent=1)[:2000])" "$WORK/put_resp.json" >&2 || true; die "PUT rejected: HTTP $code"; }
echo "  description written to the draft"

# --- 5. re-read and diff against the live record ---------------------------
code=$(fetch "$WORK/draft2.json" "${AUTH[@]}" -H "$RDM" "$API/draft")
[ "$code" = "200" ] || die "cannot re-read the draft: HTTP $code"
python3 - "$WORK/live.json" "$WORK/draft2.json" "$INTENDED" <<'PY'
import json, sys, difflib, re
live = json.load(open(sys.argv[1])); draft = json.load(open(sys.argv[2]))
want = json.load(open(sys.argv[3]))['metadata']['description']
lm, dm = live['metadata'], draft['metadata']
moved = sorted(k for k in set(lm) | set(dm) if lm.get(k) != dm.get(k))
print(f"  metadata fields that differ between live and draft: {moved or 'none'}")
if moved != ['description']:
    print("  REFUSING: something other than the description moved. The draft is left open;")
    print(f"  inspect it at https://zenodo.org/uploads/{draft.get('id')} and discard it if wrong.")
    sys.exit(1)
if dm['description'] != want:
    print("  REFUSING: the draft's description is not what release/zenodo_metadata.json says")
    sys.exit(1)
# a licence dropped here is exactly the legacy-API failure this script exists to avoid
lr = sorted(r.get('id', '') for r in lm.get('rights', [])); dr = sorted(r.get('id', '') for r in dm.get('rights', []))
assert lr == dr, f"rights changed: {lr} -> {dr}"
strip = lambda s: re.sub(r'<[^>]+>', '', s).replace('&ndash;', '–')
for line in difflib.unified_diff(strip(lm['description']).splitlines(), strip(dm['description']).splitlines(),
                                 'live', 'draft', lineterm='', n=0):
    print("    " + line)
PY

if [ "$PUBLISH" -eq 0 ]; then
  cat <<EOF

Draft staged and verified; nothing is public yet.
  review:  https://zenodo.org/uploads/$RECID
  publish: bash scripts/zenodo_edit_metadata.sh $RECID --publish
EOF
  exit 0
fi

# --- 6. publish ------------------------------------------------------------
code=$(fetch "$WORK/pub.json" "${AUTH[@]}" -H "$JSON" -X POST "$API/draft/actions/publish")
case "$code" in
  200|202) echo "  published (HTTP $code)" ;;
  *) die "publish failed: HTTP $code; the draft is still open at https://zenodo.org/uploads/$RECID" ;;
esac

# --- 7. the public record again --------------------------------------------
code=$(fetch "$WORK/after.json" -H "$RDM" "$API")
[ "$code" = "200" ] || die "cannot re-read the published record: HTTP $code"
python3 - "$WORK/live.json" "$WORK/after.json" "$INTENDED" <<'PY'
import json, sys
before = json.load(open(sys.argv[1]))['metadata']; after = json.load(open(sys.argv[2]))['metadata']
want = json.load(open(sys.argv[3]))['metadata']['description']
ok = after['description'] == want
moved = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
print(f"  live description now matches the repo: {ok}")
print(f"  fields that changed on the public record: {moved}")
sys.exit(0 if ok and moved == ['description'] else 1)
PY
echo
echo "Done. Run scripts/verify_zenodo_record.sh to re-check the five invariants."
