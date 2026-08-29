"""Count the body words of a manuscript the way a journal's editorial office would.

Usage:
    .venv/bin/python wordcount.py paper/manuscript_JRSI.md
    .venv/bin/python wordcount.py paper/manuscript_JRSI.md --target 4000

This is the one place the word count is computed. Counting it by hand, or by
`wc -w`, gives a different number every time because the disagreement is never
about the words -- it is about what counts as body text. So the rule is written
down here once, the exclusions are printed alongside the total, and every number
this repo quotes about length comes from this script.

What is excluded, and why:

  headings            a journal counts them, but they are not what an author
                      cuts, and including them hides which section is over
  the abstract        counted separately, against its own limit
  references          counted separately everywhere
  figure captions     submitted as separate files, never in the body count
  tables              submitted separately and counted as display items
  images, code        not prose
  working sections    "Editorial constraints carried in this draft (not for
                      submission)" and anything like it is scaffolding the
                      author deletes before submitting, so counting it would
                      overstate the work still to do

What is NOT excluded:

  blockquotes that are not captions. A displayed pull-quote of a claim is body
  prose and an editor counts it. Only `> **Figure N.`/`> **Table N.` lines are
  dropped, because those are captions that happen to be indented.

The section split is on `## <digit>. <title>`, so it follows whatever numbering
the manuscript uses rather than assuming a fixed set of sections. A manuscript
with no numbered sections reports zero and says so, instead of silently
counting the whole file.
"""
import argparse
import pathlib
import re
import subprocess
import sys

SKIP_HEADS = (
    "## Abstract",
    "## References",
    "## Bibliography",
    "## Figure captions",
    "## Figure legends",
    "## Editorial constraints",
    "## Statement of Significance",
    "## Declarations",
    "## Acknowl",
    "## Supplementary",
)

# A caption paragraph, with or without a blockquote marker in front of it.
CAPTION_RE = re.compile(r"^>?\s*\*\*(Table|Figure|Fig\.)\s")
SECTION_RE = re.compile(r"^## (\d+)\.\s*(.*)")


def plain_words(markdown: str) -> int:
    """Render to plain text before counting, so `**bold**` is one word and not
    three tokens, and so a markdown link counts its text and not its URL.

    A token carrying no letter and no digit is not a word. Pandoc sets an em dash
    that had spaces around it as a standalone token, so a paragraph written with
    spaced dashes scored one word higher per dash -- which matters when the limit
    is a hard ceiling and the abstract is sitting on it.
    """
    if not markdown.strip():
        return 0
    p = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "plain"],
        input=markdown, capture_output=True, text=True,
    )
    if p.returncode != 0:
        sys.exit(f"pandoc failed: {p.stderr.strip()}")
    return sum(1 for tok in p.stdout.split() if any(c.isalnum() for c in tok))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--target", type=int, default=None,
                    help="body-word limit to compare against; prints the overage")
    args = ap.parse_args()

    src = args.path.read_text(encoding="utf-8")
    # Everything after a supplementary heading belongs to another document.
    src = re.split(r"^## Supplementary", src, maxsplit=1, flags=re.M)[0]

    sections, dropped = {}, {"heading": 0, "table": 0, "caption": 0,
                             "image": 0, "code": 0, "skipped section": 0}
    cur, in_code, in_skipped = None, False, False

    for ln in src.split("\n"):
        if ln.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            dropped["code"] += 1
            continue

        if ln.startswith("## ") or ln.startswith("# "):
            dropped["heading"] += 1
            if any(ln.startswith(h) for h in SKIP_HEADS):
                cur, in_skipped = None, True
                continue
            m = SECTION_RE.match(ln)
            cur, in_skipped = (f"§{m.group(1)} {m.group(2)}" if m else None), False
            continue

        if cur is None:
            if in_skipped and ln.strip():
                dropped["skipped section"] += 1
            continue
        if ln.startswith("### ") or ln.startswith("#### "):
            dropped["heading"] += 1
            continue
        if ln.startswith("|"):
            dropped["table"] += 1
            continue
        if ln.startswith("!["):
            dropped["image"] += 1
            continue
        if CAPTION_RE.match(ln):
            dropped["caption"] += 1
            continue
        if ln.startswith("---") or not ln.strip():
            continue
        sections.setdefault(cur, []).append(ln)

    if not sections:
        sys.exit(f"{args.path}: no `## <n>. <title>` sections found -- nothing counted.")

    total = 0
    for k in sorted(sections, key=lambda s: int(s.split()[0].lstrip("§"))):
        n = plain_words("\n\n".join(sections[k]))
        total += n
        print(f"  {k:38s} {n:>6d}")
    print(f"  {'BODY TOTAL':38s} {total:>6d}")

    if "## Abstract" in src:
        abstract = re.split(r"^## ", src.split("## Abstract", 1)[1], maxsplit=1, flags=re.M)[0]
        # Structured-abstract labels are formatting, not prose.
        abstract = re.sub(r"\*\*[A-Z][a-z]+:\*\*", "", abstract)
        print(f"  {'Abstract':38s} {plain_words(abstract):>6d}")

    print("\n  excluded from the body total:")
    for k, v in dropped.items():
        if v:
            print(f"    {k:36s} {v:>6d} lines")

    if args.target:
        over = total - args.target
        verdict = (f"{over:+d} over" if over > 0 else f"{-over:d} to spare")
        print(f"\n  target {args.target}: {verdict}")
        if over > 0:
            print(f"  the body is {total / args.target:.1f}x the target")


if __name__ == "__main__":
    main()
