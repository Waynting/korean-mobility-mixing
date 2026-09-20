#!/usr/bin/env python3
"""Re-check every reference in the JRSI manuscript against Crossref/DataCite.

WHY THIS EXISTS. The 09-16 letter asserted a per-entry re-check of the 40
references and left no artefact; the newest record on disk at the time
(archive/reference_verification_20260903.md) covered the 35-entry list. The
09-18 record (archive/reference_verification_20260918.md) was produced by a
script that lived in a session scratchpad, so "it can be re-run" was true only
on the machine that ran it. This is that script, in the tree.

WHAT IT CHECKS. Each numbered entry of the manuscript's reference list is
parsed for a DOI (or, failing that, a URL). A DOI is resolved at Crossref and,
if Crossref does not know it, at DataCite; three fields are compared against
the entry: the first author's family name, the year, and the title. A URL is
fetched and must return HTTP 200. Exit status is 1 if any entry fails.

THREE THINGS THAT LOOK LIKE MISMATCHES AND ARE NOT (all three were false
alarms on 2026-09-18, and each is handled below):

  - a DOI may contain parentheses (Nold 1980: 10.1016/0025-5564(80)90069-3),
    so the DOI is taken greedily to the next whitespace, not to the first ')';
  - Crossref's `issued` is the earliest date, which for an online-first paper
    is the year before the volume year the manuscript cites (Chang 2021), so
    the year matches if ANY of published-print / issued / published-online /
    published agrees;
  - a markdown autolink `<https://...>` must not carry its closing '>' into
    the request.

    python3 scripts/verify_refs.py                 # table to stdout, exit 1 on any failure
    python3 scripts/verify_refs.py --json out.json # also dump the raw fields

It makes ~40 HTTP requests and takes about a minute. It writes nothing unless
--json is given.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript_JRSI.md"
UA = "korean-mobility-mixing reference verification (scripts/verify_refs.py)"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.load(fh)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def parse_entries(text):
    """The numbered lines under '## References', in order."""
    block = text.split("## References", 1)[1]
    block = re.split(r"^## ", block, maxsplit=1, flags=re.M)[0]
    entries = [(int(m.group(1)), m.group(2))
               for m in re.finditer(r"^(\d+)\. (.+)$", block, flags=re.M)]
    nums = [n for n, _ in entries]
    if nums != list(range(1, len(nums) + 1)):
        sys.exit(f"reference numbering is not 1..{len(nums)}: {nums}")
    return entries


def fields_of(text):
    m = re.search(r"doi:(10\.\S+)", text)          # greedy: DOIs contain ')'
    doi = m.group(1).rstrip(".,") if m else None
    m = re.search(r"<?(https?://[^\s<>)]+)", text)  # never swallow the autolink's '>'
    url = m.group(1).rstrip(".,") if m else None
    first_author = text.split(".")[0].split(",")[0].strip()
    m = re.search(r"\b(19|20)\d{2}\b", text)
    year = m.group(0) if m else ""
    m = re.search(r"\*(.+?)\*", text)
    title = m.group(1) if m else ""
    return doi, url, first_author, year, title


def resolve_doi(doi):
    """(source, record) from Crossref, else DataCite, else (last error, None)."""
    status = ""
    for api, u in (("Crossref", f"https://api.crossref.org/works/{doi}"),
                   ("DataCite", f"https://api.datacite.org/dois/{doi}")):
        try:
            return api, get_json(u)
        except urllib.error.HTTPError as e:
            status = f"{api} HTTP {e.code}"
        except Exception as e:  # noqa: BLE001 - any transport failure is a status, not a crash
            status = f"{api} {type(e).__name__}"
        time.sleep(0.3)
    return status, None


def got_fields(src, rec):
    """(first author family name, years, title) as the registry has them."""
    if src == "Crossref":
        m = rec["message"]
        title = (m.get("title") or [""])[0]
        au = m.get("author") or []
        family = au[0].get("family", "") if au else ""
        years = set()
        for key in ("published-print", "issued", "published-online", "published"):
            dp = (m.get(key) or {}).get("date-parts") or []
            if dp and dp[0] and dp[0][0]:
                years.add(str(dp[0][0]))
        return family, sorted(years), title
    a = rec["data"]["attributes"]
    title = (a.get("titles") or [{}])[0].get("title", "")
    cr = a.get("creators") or []
    family = (cr[0].get("familyName") or cr[0].get("name", "")) if cr else ""
    return family, [str(a.get("publicationYear", ""))], title


def check_entry(num, text):
    doi, url, first_author, year, title = fields_of(text)
    row = dict(num=num, doi=doi, url=None if doi else url, first_author=first_author,
               year=year, title=title, src="", status="", got_a="", got_y="", got_t="")
    if doi:
        src, rec = resolve_doi(doi)
        if rec is None:
            row["src"], row["status"] = "unresolved", src
            return row
        got_a, got_ys, got_t = got_fields(src, rec)
        fam = first_author.split()[0].replace("'", "")
        ok_a = norm(fam) in norm(got_a) or norm(got_a) in norm(first_author)
        ok_y = year in got_ys
        tn, gn = norm(title), norm(got_t)
        ok_t = bool(tn) and (tn[:60] in gn or gn[:60] in tn)
        failed = [k for k, v in (("author", ok_a), ("year", ok_y), ("title", ok_t)) if not v]
        row.update(src=src, got_a=got_a, got_y=", ".join(got_ys), got_t=got_t,
                   status="match" if not failed else "MISMATCH " + " ".join(failed))
    elif url:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as fh:
                body = fh.read(400_000)
            row.update(src="live fetch", status=f"HTTP 200, {len(body)} bytes")
        except Exception as e:  # noqa: BLE001
            row.update(src="live fetch", status=f"FAILED {type(e).__name__}")
    else:
        row.update(src="none", status="no DOI and no URL")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--json", type=Path, default=None,
                    help="also write every parsed and fetched field here")
    args = ap.parse_args()

    entries = parse_entries(MS.read_text(encoding="utf-8"))
    rows = []
    for num, text in entries:
        row = check_entry(num, text)
        rows.append(row)
        print(f"  [{row['num']:2d}] {row['src']:10s} {row['status']:<32s} "
              f"{row['first_author']} {row['year']}", flush=True)

    ok = [r for r in rows if r["status"] == "match" or r["status"].startswith("HTTP 200")]
    bad = [r for r in rows if r not in ok]
    by_src = {}
    for r in rows:
        by_src[r["src"]] = by_src.get(r["src"], 0) + 1
    n_doi = sum(1 for r in rows if r["doi"])
    print(f"\n{len(rows)} entries: {n_doi} with a DOI "
          f"({', '.join(f'{k} {v}' for k, v in sorted(by_src.items()) if k in ('Crossref', 'DataCite'))}), "
          f"{len(rows) - n_doi} by URL; {len(ok)} pass, {len(bad)} fail")
    for r in bad:
        print(f"  [{r['num']}] {r['status']}  quoted={r['first_author']} {r['year']}  "
              f"got={r['got_a']} {r['got_y']}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
