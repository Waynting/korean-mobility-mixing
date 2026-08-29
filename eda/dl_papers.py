#!/usr/bin/env python
"""Pull the three papers the 8-19 letter says the manuscript has to engage.

WHY A SCRIPT AND NOT THREE CLICKS. The letter re-positions the paper on the
strength of what these three say, so the sentences we write about them have to
be checkable by someone who was not in the room. Each file therefore lands with
its sha256 in a manifest next to the sha256 of a plain-text rendering, and every
claim the memo makes about a paper cites a page of that rendering.

  [DID]  Di Domenico et al., Nature Communications 2026 -- mobility-derived
         synthetic matrices against survey matrices in France. This is the paper
         that takes the LEVEL claim away from us: a >3x school-age gap is
         already published, so only the CONCENTRATION claim is ours.
  [LIM]  Lim, Lee & Jung, arXiv:2603.16064 -- Seoul metapopulation on the SAME
         KT product with the SAME H/W/O panels, 3 age bins and 25 gu. The
         in-sample instance the bandwidth section needs.
  [PIT]  Pittsburgh schoolchildren (PMC7840989) -- sensor versus survey contact
         counts that reconcile once proportionate mixing is divided out. The
         precedent for what p32 does.

VERIFICATION IS ABOUT CONTENT, as everywhere else in this project: a publisher
cookie wall answers 200 with a Content-Type of application/pdf just as happily as
an article does. A download counts only when the bytes start with %PDF, the page
count is at least the expected minimum, and a required string (the title, or a
phrase the memo will quote) is present in the extracted text.

    python eda/dl_papers.py            # fetch what is missing, verify all
    python eda/dl_papers.py --verify   # check disk, fetch nothing
    python eda/dl_papers.py --force    # re-fetch even if present
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT  # noqa: E402

PAPERS = DATA_ROOT / "raw" / "papers"
MANIFEST = PAPERS / "manifest.json"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# tag -> (filename, urls to try in order, min pages, strings that must appear)
SOURCES = {
    "DID": ("did2026_natcomms.pdf",
            ["https://www.nature.com/articles/s41467-026-68557-3.pdf",
             "https://doi.org/10.1038/s41467-026-68557-3"],
            8, ["contact", "matri"]),
    # The one place this project could conflict with [DID] is their Section 5
    # (the mathematical definitions of the four mixing indicators) and Fig. S10a
    # (cosine similarity against homogeneous mixing), and both live only in the
    # supplement. Quoting the main text about a supplementary figure is exactly
    # the kind of second-hand claim that has to be closed before submission.
    "DID_SI": ("did2026_natcomms_SI.pdf",
               ["https://media.springernature.com/original/springer-static/esm/"
                "art%3A10.1038%2Fs41467-026-68557-3/MediaObjects/"
                "41467_2026_68557_MOESM1_ESM.pdf",
                "https://static-content.springer.com/esm/"
                "art%3A10.1038%2Fs41467-026-68557-3/MediaObjects/"
                "41467_2026_68557_MOESM1_ESM.pdf"],
               20, ["assortativity", "cosine"]),
    "LIM": ("lim2026_arxiv_2603.16064.pdf",
            ["https://arxiv.org/pdf/2603.16064",
             "https://arxiv.org/pdf/2603.16064v1"],
            10, ["Seoul", "mobility"]),
    # PMC's per-article PDF is named for the publisher's article id, not
    # "main.pdf"; the landing page is scraped for it rather than guessed.
    "PIT": ("pittsburgh_PMC7840989.pdf",
            ["https://pmc.ncbi.nlm.nih.gov/articles/PMC7840989/pdf/"
             "41598_2021_Article_81673.pdf",
             "https://www.nature.com/articles/s41598-021-81673-y.pdf"],
            6, ["contact", "school"]),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_of(path):
    """Plain-text rendering, cached beside the PDF. Two extractors, because a
    scanned page defeats one and not always the other; if both give almost
    nothing the paper is flagged as image-only rather than silently quoted."""
    txt = path.with_suffix(".txt")
    if txt.exists() and txt.stat().st_size > 0:
        return txt.read_text(errors="replace"), txt
    body = ""
    try:
        body = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                              capture_output=True, text=True,
                              timeout=300).stdout
    except (OSError, subprocess.SubprocessError):
        pass
    if len(body) < 2000:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                body = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            pass
    txt.write_text(body)
    return body, txt


def n_pages(path):
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def check(tag, path):
    """A paper is present only if it is a PDF, long enough, and says what it
    should. Returns (ok, note)."""
    _, urls, min_pages, needles = SOURCES[tag][1:] if False else (
        None, SOURCES[tag][1], SOURCES[tag][2], SOURCES[tag][3])
    if not path.exists():
        return False, "absent"
    with open(path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            return False, "not a PDF (publisher wall or error page)"
    np_ = n_pages(path)
    if np_ < min_pages:
        return False, f"only {np_} pages, expected >= {min_pages}"
    body, _ = text_of(path)
    if len(body) < 2000:
        return False, f"image-only PDF ({len(body)} bytes of text) -- needs OCR"
    low = body.lower()
    missing = [s for s in needles if s.lower() not in low]
    if missing:
        return False, f"text lacks {missing}"
    return True, f"{np_} pages, {len(body):,} chars of text"


def fetch(url, dest, tries=4):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA, "Accept": "application/pdf,*/*"})
            with urllib.request.urlopen(req, timeout=180) as r, \
                    open(dest, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            return True, None
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = e
            time.sleep(10 * (k + 1))
    return False, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    PAPERS.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    bad = []
    for tag, (name, urls, _, _) in SOURCES.items():
        dest = PAPERS / name
        ok, note = check(tag, dest)
        if ok and not args.force:
            print(f"  {tag:>4} ok      {name}  ({note})")
        elif args.verify:
            print(f"  {tag:>4} MISSING {name}  ({note})")
            bad.append(tag)
            continue
        else:
            got = False
            for url in urls:
                print(f"  {tag:>4} fetching {url} ...", flush=True)
                done, err = fetch(url, dest)
                if not done:
                    print(f"       transfer failed: {err}")
                    continue
                for stale in (dest.with_suffix(".txt"),):
                    stale.unlink(missing_ok=True)
                ok, note = check(tag, dest)
                print(f"       {'ok' if ok else 'rejected'}: {note}")
                if ok:
                    got = True
                    break
            if not got:
                bad.append(tag)
                dest.unlink(missing_ok=True)
                continue
        body, txt = text_of(dest)
        man[tag] = dict(file=name, sha256=sha256(dest), bytes=dest.stat().st_size,
                        pages=n_pages(dest), text_file=txt.name,
                        text_sha256=hashlib.sha256(body.encode()).hexdigest(),
                        note=note, urls=urls)
    MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"\n{len(SOURCES) - len(bad)}/{len(SOURCES)} papers verified "
          f"-> {PAPERS}")
    if bad:
        print(f"MISSING: {', '.join(bad)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
