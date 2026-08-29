#!/usr/bin/env python
"""L5 — pull the Chae et al. national contact-survey microdata from figshare.

Phase 7 (the survey comparison, the step that decides Q2 vs Q1) needs the
respondent-level file, not the published tables. Sci Data 13, 603 (2026),
doi 10.1038/s41597-026-06896-y; data at doi 10.6084/m9.figshare.29312222, CC0.

figshare publishes an md5 per file, so verification here is exact rather than
heuristic: fetch, hash, compare, and refuse to keep a file that does not match.

    python eda/dl_chae.py            # fetch what is missing, verify all
    python eda/dl_chae.py --verify   # verify what is on disk, fetch nothing
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT  # noqa: E402

ARTICLE = 29312222
API = f"https://api.figshare.com/v2/articles/{ARTICLE}"
DEST = DATA_ROOT / "raw" / "chae2026"
MANIFEST = DEST / "manifest.json"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as r:
        meta = json.load(r)
    files = meta["files"]
    print(f"{meta['title']}\n  licence {meta.get('license', {}).get('name')}"
          f"  version {meta['doi']}  posted {meta['timeline']['posted']}")

    bad = []
    for f in files:
        p = DEST / f["name"]
        want = f["computed_md5"]
        if p.exists() and md5(p) == want:
            print(f"  {f['name']:<38} ok (cached)")
            continue
        if args.verify:
            bad.append((f["name"], "missing" if not p.exists() else "md5 mismatch"))
            continue
        print(f"  {f['name']:<38} downloading {f['size']:,} B ...", flush=True)
        urllib.request.urlretrieve(f["download_url"], p)
        got = md5(p)
        if got != want:
            p.unlink()
            bad.append((f["name"], f"md5 {got} != published {want}"))
            print("    MD5 MISMATCH — deleted")
        else:
            print(f"    ok, md5 {got}")

    with open(MANIFEST, "w") as fh:
        json.dump({"article": ARTICLE, "doi": meta["doi"],
                   "resource_doi": meta.get("resource_doi"),
                   "title": meta["title"],
                   "files": [{k: f[k] for k in ("name", "size", "computed_md5")}
                             for f in files]}, fh, indent=1, ensure_ascii=False)

    if bad:
        print(f"\n{len(bad)} FAILED:")
        for n, why in bad:
            print(f"  {n}: {why}")
        return 1
    print(f"\nall {len(files)} files verified against figshare's md5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
