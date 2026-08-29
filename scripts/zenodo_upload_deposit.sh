#!/usr/bin/env bash
# Upload a built deposit zip to an existing Zenodo draft, and verify it landed.
#
# Adapted from the Brazil archive's script, which was written after a 2026-07-29
# Zenodo storage outage. That failure mode is worth carrying over, because it
# looks like a client problem and is not: the bucket API returned HTTP 500 on
# GET for every record including published ones, and the content PUT returned
# HTTP 200 and then dropped the file, so the follow-up commit 404'd. Writes were
# accepted and silently discarded while reads stayed healthy.
#
# So: NEVER trust the upload's HTTP status. This script commits the file and
# then re-reads the draft's file listing to confirm size and status, and fails
# loudly if the file is not there. It is the same rule this repo already applies
# to the Korean government portals, which answer 200 with a not-found page.
#
# It uses the InvenioRDM three-step flow (init / content / commit) rather than
# the legacy bucket PUT. Re-running is safe: an existing incomplete entry for
# the same key is deleted first.
#
# Usage:
#   bash scripts/zenodo_upload_deposit.sh <draft-recid> <zipfile>
#   bash scripts/zenodo_upload_deposit.sh 12345678 build/zenodo/korean-mobility-mixing-v1.0.0.zip
#
# Token, in order: $ZENODO_TOKEN, then --token-file PATH, then $TOKEN_FILE,
# then ~/.config/zenodo/token, then the Brazil archive's ../../JBI/zenodo-upload/.env.
# The token is never echoed and never placed in a URL.
#
# This script deliberately does NOT publish. Publishing is irreversible and
# mints a permanent DOI, so it stays a human decision.
set -uo pipefail

RECID="${1:-}"
ZIPFILE="${2:-}"
shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
  case "$1" in
    --token-file) TOKEN_FILE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$RECID" ] || [ -z "$ZIPFILE" ]; then
  echo "usage: bash scripts/zenodo_upload_deposit.sh <draft-recid> <zipfile> [--token-file PATH]" >&2
  exit 2
fi
[ -f "$ZIPFILE" ] || { echo "ERROR: no such file: $ZIPFILE" >&2; exit 1; }

# --- token, without ever printing it ---------------------------------------
TOK="${ZENODO_TOKEN:-}"
if [ -z "$TOK" ]; then
  for cand in "${TOKEN_FILE:-}" "$HOME/.config/zenodo/token" \
              "$HOME/Documents/Research/JBI/zenodo-upload/.env"; do
    [ -n "$cand" ] && [ -f "$cand" ] || continue
    TOK="$(grep -E '^ZENODO_TOKEN=' "$cand" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$TOK" ] || TOK="$(tr -d '[:space:]' < "$cand")"   # tolerate a bare token
    [ -n "$TOK" ] && { echo "using token from $cand"; break; }
  done
fi
[ -n "$TOK" ] || { echo "ERROR: no token (set ZENODO_TOKEN or pass --token-file)" >&2; exit 1; }

AUTH=(-H "Authorization: Bearer $TOK")
API="https://zenodo.org/api/records/$RECID/draft"
KEY="$(basename "$ZIPFILE")"
SIZE=$(wc -c < "$ZIPFILE" | tr -d ' ')

say()  { printf '\n==> %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

say "Target"
echo "    draft:  $RECID"
echo "    file:   $KEY ($(du -h "$ZIPFILE" | cut -f1), $SIZE bytes)"

say "Preflight"
code=$(curl -s -o /tmp/zdraft.$$ -w '%{http_code}' "${AUTH[@]}" "$API")
[ "$code" = "200" ] || die "cannot read draft $RECID (HTTP $code); check the id and the token's scopes"
ver=$(python3 -c "import json;print(json.load(open('/tmp/zdraft.$$')).get('metadata',{}).get('version','?'))" 2>/dev/null)
echo "    draft version field: $ver"
rm -f /tmp/zdraft.$$

bucket=$(curl -s "${AUTH[@]}" "https://zenodo.org/api/deposit/depositions/$RECID" \
         | python3 -c "import json,sys;print(json.load(sys.stdin).get('links',{}).get('bucket',''))" 2>/dev/null)
if [ -n "$bucket" ]; then
  bcode=$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" "$bucket")
  if [ "$bcode" = "500" ]; then
    echo "    WARNING: bucket API returns HTTP 500. Zenodo's storage write layer is"
    echo "             likely degraded; uploads may be accepted and then dropped."
    echo "             Continuing -- the verify step below will catch it."
  else
    echo "    bucket reachable (HTTP $bcode)"
  fi
fi

say "Clearing any previous entry for $KEY"
curl -s -o /dev/null -X DELETE "${AUTH[@]}" "$API/files/$KEY" 2>/dev/null
echo "    done (404 here is normal and means there was nothing to clear)"

say "1/3 Initialising file entry"
code=$(curl -s -o /tmp/zinit.$$ -w '%{http_code}' -X POST "${AUTH[@]}" \
  -H "Content-Type: application/json" -d "[{\"key\":\"$KEY\"}]" "$API/files")
[ "$code" = "201" ] || { head -c 300 /tmp/zinit.$$; die "init failed (HTTP $code)"; }
rm -f /tmp/zinit.$$
echo "    ok"

say "2/3 Uploading content (this is the slow part)"
code=$(curl -s -o /tmp/zput.$$ -w '%{http_code}' -X PUT "${AUTH[@]}" \
  --retry 3 --retry-delay 10 --retry-all-errors \
  -H "Content-Type: application/octet-stream" \
  --upload-file "$ZIPFILE" "$API/files/$KEY/content")
echo "    HTTP $code"
[ "$code" = "200" ] || { head -c 300 /tmp/zput.$$; die "content upload failed (HTTP $code)"; }
rm -f /tmp/zput.$$

say "3/3 Committing"
code=$(curl -s -o /tmp/zcommit.$$ -w '%{http_code}' -X POST "${AUTH[@]}" "$API/files/$KEY/commit")
if [ "$code" != "200" ]; then
  head -c 300 /tmp/zcommit.$$; echo
  die "commit failed (HTTP $code).
  A 404 here means Zenodo accepted the bytes and then discarded them, which is
  the storage-outage signature. Nothing is wrong with the file or this script:
  wait for Zenodo to recover and re-run this command unchanged."
fi
rm -f /tmp/zcommit.$$
echo "    ok"

say "Verifying the file is really there"
curl -s "${AUTH[@]}" "$API/files" > /tmp/zfiles.$$
python3 - "$KEY" "$SIZE" /tmp/zfiles.$$ <<'PY'
import json, sys
key, want, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(path))
hit = next((e for e in d.get('entries', []) if e.get('key') == key), None)
if not hit:
    print(f"    FAIL  '{key}' is not in the draft's file listing"); sys.exit(1)
got, status = hit.get('size'), hit.get('status')
print(f"    key:      {hit.get('key')}")
print(f"    status:   {status}")
print(f"    size:     {got} bytes")
print(f"    checksum: {hit.get('checksum')}")
if status != 'completed':
    print(f"    FAIL  status is '{status}', expected 'completed'"); sys.exit(1)
if got != want:
    print(f"    FAIL  size mismatch: uploaded {want}, Zenodo has {got}"); sys.exit(1)
print("    ok    file is present, complete, and the size matches")
PY
rc=$?
rm -f /tmp/zfiles.$$
[ "$rc" -eq 0 ] || die "verification failed; do not publish this draft"

cat <<EOF

Upload verified.

Next, by hand (deliberately not automated, publishing is irreversible):
  1. Open https://zenodo.org/uploads/$RECID and check the file list and metadata
  2. Confirm the licence block shows BOTH MIT and CC BY 4.0
  3. Confirm EXCLUDED_FILES.txt in the zip says what you expect it to say
  4. Publish
  5. Record the concept DOI and the version DOI in release/zenodo_metadata.json
  6. bash scripts/verify_zenodo_record.sh $ver
EOF
