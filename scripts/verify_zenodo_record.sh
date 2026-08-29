#!/usr/bin/env bash
# Verify the published Zenodo record against the claims the manuscript makes.
#
# The manuscript's Data accessibility section asserts four things a reviewer can
# check without credentials, so this script checks them the same way:
#
#   1. the record is public          (an unauthenticated request must succeed)
#   2. concept and version DOIs are not reversed
#   3. the licences match the paper   (MIT for code, CC BY 4.0 for derived data)
#   4. the version the paper names is the version that is actually published
#   5. the record points at the public mirror the paper also names
#
# EDITING A PUBLISHED RECORD'S METADATA: use the InvenioRDM API, never the
# legacy deposit API. `GET /api/deposit/depositions/<id>` reports
# `license: "cc-by-4.0"` for a record that carries BOTH cc-by-4.0 and mit,
# because the legacy schema has room for one; PUT that back and the record
# silently loses MIT, and with it check 3 and the manuscript's sentence about
# the code licence. The working sequence, all with Accept and Content-Type
# application/json against /api/records/<id>/draft:
#   POST   /api/records/<id>/draft                     create the edit draft
#   GET    /api/records/<id>/draft   Accept: application/vnd.inveniordm.v1+json
#   PUT    /api/records/<id>/draft   {"metadata": {...}}  rights as [{"id":..}]
#   GET    the draft again, diff it against the live record, expect ONE field
#   POST   /api/records/<id>/draft/actions/publish
# Files are immutable on a published record; metadata is not.
#
# IMPORTANT, and the reason this is not a one-line curl: Zenodo's legacy API
# (`GET /api/records/<id>`, the default response) serialises only the FIRST
# entry of the record's `rights` list, so a record carrying both MIT and
# CC BY 4.0 reports `license: {"id": "mit-license"}` and `rights: null`.
# Reading that endpoint once led the Brazil archive to the wrong conclusion that
# a record could hold a single licence. The licence check below requests the
# InvenioRDM serialisation explicitly. Do not "simplify" it back.
#
# CONCEPT_RECID is empty until the first version is published. Fill it in from
# release/zenodo_metadata.json once a DOI exists.
#
# Usage:
#   bash scripts/verify_zenodo_record.sh              # check the concept record
#   bash scripts/verify_zenodo_record.sh 1.0.0        # also assert the latest version
set -uo pipefail

CONCEPT_RECID="22152089"          # concept DOI 10.5281/zenodo.22152089
MIRROR_URL="https://github.com/Waynting/korean-mobility-mixing"
EXPECT_VERSION="${1:-}"
API="https://zenodo.org/api/records"
RDM_ACCEPT="Accept: application/vnd.inveniordm.v1+json"

if [ -z "$CONCEPT_RECID" ]; then
  cat >&2 <<'EOF'
CONCEPT_RECID is empty, so there is nothing to verify yet.

Nothing has been published. When the first version goes out:
  1. put the concept record id in CONCEPT_RECID at the top of this script
  2. record both DOIs in release/zenodo_metadata.json
  3. re-run this script

The concept DOI is the one that always resolves to the latest version, and it
is the one the manuscript's Data accessibility should cite.
EOF
  exit 2
fi

pass=0; fail=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
note() { printf '        %s\n' "$1"; }

echo
echo "Verifying Zenodo concept record $CONCEPT_RECID"
echo

# --- 1. public -------------------------------------------------------------
# No credentials are sent, so a 200 here is what an outside reader would get.
code=$(curl -sL -o /dev/null -w '%{http_code}' "https://doi.org/10.5281/zenodo.$CONCEPT_RECID")
url=$(curl -sL -o /dev/null -w '%{url_effective}' "https://doi.org/10.5281/zenodo.$CONCEPT_RECID")
if [ "$code" = "200" ]; then ok "record is public (concept DOI -> HTTP 200, unauthenticated)"
else bad "concept DOI returned HTTP $code; the Data accessibility statement does not hold"; fi
note "resolves to $url"

# --- 2. concept vs version DOI --------------------------------------------
# -L is load-bearing: the concept id 302s to the latest version's record, and
# without it curl returns the redirect's HTML body and the parse below dies on
# "Expecting value: line 1 column 1".
curl -sL -H "$RDM_ACCEPT" "$API/$CONCEPT_RECID" > /tmp/zrec.$$
python3 - /tmp/zrec.$$ "$CONCEPT_RECID" "$EXPECT_VERSION" "$MIRROR_URL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); concept_id, want = sys.argv[2], sys.argv[3]
G, R = '\033[32mok\033[0m   ', '\033[31mFAIL\033[0m '

pids = d.get('parent', {}).get('pids', {}) or {}
concept_doi = (pids.get('doi') or {}).get('identifier', '')
version_doi = ((d.get('pids') or {}).get('doi') or {}).get('identifier', '')
print(f"  {G if concept_doi and version_doi and concept_doi != version_doi else R} "
      f"concept and version DOIs are distinct")
print(f"        concept: {concept_doi or '(none)'}")
print(f"        version: {version_doi or '(none)'}")

rights = [r.get('id') or (r.get('title') or {}).get('en', '') for r in (d.get('metadata', {}).get('rights') or [])]
have = {str(r).lower() for r in rights}
mit = any('mit' in r for r in have)
ccby = any('cc-by-4.0' in r or 'cc by 4.0' in r for r in have)
print(f"  {G if mit else R} MIT is listed (the code licence the paper names)")
print(f"  {G if ccby else R} CC BY 4.0 is listed (the derived-data licence the paper names)")
print(f"        rights: {rights or '(none)'}")
if not ccby:
    print("        note: if this version carries no derived data, CC BY 4.0 may be")
    print("              absent on purpose. Check that the paper says the same.")

got = d.get('metadata', {}).get('version', '')
if want:
    print(f"  {G if got == want else R} latest published version is {want!r}")
    print(f"        record says: {got!r}")
else:
    print(f"        latest published version: {got!r}")

# 5. the mirror. The paper's Data accessibility names a GitHub URL as well as
# the DOI, so the record has to point back at it or the two halves of that
# sentence are unconnected for anyone arriving from Zenodo. The relation is
# checked too: isSupplementTo plus a tree/<tag> URL is what Zenodo's own GitHub
# integration writes, and a bare repository URL would point at a moving target
# from a fixed version DOI.
mirror = sys.argv[4] if len(sys.argv) > 4 else ''
rels = d.get('metadata', {}).get('related_identifiers') or []
hit = [r for r in rels if mirror and mirror in str(r.get('identifier', ''))]
supp = [r for r in hit if (r.get('relation_type') or {}).get('id') == 'issupplementto']
print(f"  {G if supp else R} the public mirror is a related identifier "
      f"(isSupplementTo)")
for r in hit:
    print(f"        {(r.get('relation_type') or {}).get('id')}: {r.get('identifier')}")
if not hit:
    print(f"        no related identifier mentions {mirror or '(no URL given)'}")

files = (d.get('files') or {}).get('entries') or {}
print(f"        files on the record: {len(files)}")
for k, v in list(files.items())[:5]:
    size = v.get('size') if isinstance(v, dict) else ''
    print(f"          {k}  {size}")
PY
rm -f /tmp/zrec.$$

echo
echo "Reminder: this checks the record, not the deposit's contents. Whether the"
echo "zip holds what EXCLUDED_FILES.txt says it holds is checked by rebuilding"
echo "it with scripts/build_zenodo_deposit.sh and comparing MANIFEST.sha256."
