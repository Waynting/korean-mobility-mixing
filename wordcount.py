"""Count the body words of a manuscript the way a journal's editorial office would.

Usage:
    .venv/bin/python wordcount.py paper/manuscript_JRSI.md
    .venv/bin/python wordcount.py paper/manuscript_JRSI.md --target 4000
    .venv/bin/python wordcount.py paper/manuscript_JRSI.md --submission 8000
    .venv/bin/python wordcount.py paper/manuscript_JRSI.md --captions

The body count below is the author's count: what is left to cut, section by
section. It is NOT the number an editorial office applies to a limit, because
the limit is stated over a different set of parts. `--submission` prints that
second count instead of estimating it: every part of the file is classified
once, nothing is counted twice, and the total is printed both with and without
the figure captions. Until 2026-09-12 that was because the captions' status was
unsettled; the editorial office has since confirmed that they count, so the
"captions included" line is the governed number and the other is kept so the
history in paper/editorial_constraints.md stays comparable. The same reply set
a ceiling for INITIAL submission above the stated limit, which is printed as a
second verdict. See the JRSI note under REPORTED_PARTS.

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
  figure captions     submitted as separate display items, never in the body
                      count. The WHOLE caption paragraph is dropped, not just
                      the line that opens it: the captions used to live in a
                      `## Figure captions` section that SKIP_HEADS threw away
                      whole, and when they moved next to the sections they
                      belong to, matching only the `**Figure N.` line left
                      2 000-odd words of caption prose in the body total
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

# The parts a journal's limit is stated over, in the order they appear. The
# author's body count folds several of these away because they are not what an
# author cuts; a word limit does not care about that distinction.
#
# JRSI, Research Article: "2,500-8,000 words, INCLUDING the cover page,
# references and acknowledgements" (JRSI_投稿規定整理.md §1). Figure captions
# are named in the Report row of the same table and NOT in the Research Article
# row, so until 2026-09-12 their status was unsettled and this script refused
# to decide it. The editorial office answered on 2026-09-12 (reply to
# Email_Discussion/jrsi_editorial_enquiry_20260901.md): "We do count figure
# captions in our word count, but we are also quite flexible with the word
# limit. If your manuscript is below 10k words, then we would be able to accept
# it for initial submission." So: captions count; the governed total is the
# one WITH captions; and the number that decides whether the initial
# submission is accepted is INITIAL_SUBMISSION_CEILING, not the stated 8,000.
# Both totals are still printed so the rows in editorial_constraints.md remain
# comparable across the change.
INITIAL_SUBMISSION_CEILING = 10_000   # editorial office, 2026-09-12, "below 10k"
FRONT_HEAD = "# "          # the H1 title, byline, affiliation, keywords
# Royal Society's own headings, which are not the ones an author would guess:
# "Authors' contributions" carries the apostrophe-s and "Conflict of interest
# declaration" is not "Competing interests". Both spellings are listed, because a
# heading that falls through this tuple is silently counted as body text and the
# submission total then disagrees with the per-section one for no visible reason.
DECLARATION_HEADS = (
    "## Data accessibility", "## Author contributions", "## Authors' contributions",
    "## Competing interests", "## Conflict of interest",
    "## Funding", "## Ethics", "## Use of generative AI", "## Acknowl",
    "## Declarations", "## Statement of Significance",
)
REFERENCE_HEADS = ("## References", "## Bibliography")
CAPTION_HEADS = ("## Figure captions", "## Figure legends")
ABSTRACT_HEADS = ("## Abstract",)
# Scaffolding the author deletes before submitting; counted by nobody.
WORKING_HEADS = ("## Editorial constraints", "## Supplementary")

# The FIRST LINE of a caption paragraph, with or without a blockquote marker.
# A caption runs from this line to the next blank line or rule; both counters
# below carry an `in_caption` flag rather than testing every line against this,
# because a caption is a paragraph and only its opening line looks like one.
CAPTION_RE = re.compile(r"^>?\s*\*\*(Table|Figure|Fig\.)\s")
# The label that a caption line and its image line share, so the two lines of
# one display item group together without either being matched twice:
# "![Figure 1: ...](...)" and "**Figure 1. ...**" both yield ("Figure", "1").
# The number carries an optional letter because the SI numbers its display items
# S1, S2; without it every SI caption fell into one unlabelled row together.
DISPLAY_LABEL_RE = re.compile(r"^(?:!\[|>?\s*\*\*)(Table|Figure|Fig\.)\s*([A-Z]?\d+)")
# A thin space (U+2009) between two digits is the thousands separator Royal
# Society style asks for, not a gap between two words.
THIN_SPACE_IN_NUMBER_RE = re.compile(r"(?<=\d) (?=\d)")
SECTION_RE = re.compile(r"^## (\d+)\.\s*(.*)")


def plain_words(markdown: str) -> int:
    """Render to plain text before counting, so `**bold**` is one word and not
    three tokens, and so a markdown link counts its text and not its URL.

    A token carrying no letter and no digit is not a word. Pandoc sets an em dash
    that had spaces around it as a standalone token, so a paragraph written with
    spaced dashes scored one word higher per dash -- which matters when the limit
    is a hard ceiling and the abstract is sitting on it.

    The same failure with the opposite sign: Royal Society house style sets the
    thousands separator as a thin space (U+2009), which pandoc preserves and
    `str.split()` treats as whitespace, so "10 186 891 962" scored four words
    instead of one. A thin space between two digits is a separator inside one
    number and not a word boundary, so it is removed before the split.
    """
    if not markdown.strip():
        return 0
    p = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "plain"],
        input=markdown, capture_output=True, text=True,
    )
    if p.returncode != 0:
        sys.exit(f"pandoc failed: {p.stderr.strip()}")
    text = THIN_SPACE_IN_NUMBER_RE.sub("", p.stdout)
    return sum(1 for tok in text.split() if any(c.isalnum() for c in tok))


def submission_count(path, limit):
    """Count every part of the manuscript once, and report the total both with
    and without the figure captions.

    The author's count above answers "what do I cut". This one answers "what
    does the limit apply to", and the two are different sets of parts. It is a
    separate function rather than a flag on the same walk because the two
    counts disagree on purpose: the body count drops the front matter, the
    declarations and the references because an author does not cut them, and a
    word limit counts them anyway.

    Nothing is estimated and nothing is counted twice. Each line lands in
    exactly one part, and the parts sum to the file minus the working sections.
    """
    src = path.read_text(encoding="utf-8")

    def part_of(head):
        if head.startswith(WORKING_HEADS):
            return None
        if head.startswith(ABSTRACT_HEADS):
            return "abstract"
        if head.startswith(REFERENCE_HEADS):
            return "references"
        if head.startswith(CAPTION_HEADS):
            return "figure captions"
        if head.startswith(DECLARATION_HEADS):
            return "declarations and acknowledgements"
        return "body (sections and headings)"

    parts, cur, in_code = {}, "front matter (title, byline, keywords)", False
    in_caption = False
    for ln in src.split("\n"):
        if ln.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if ln.startswith("## ") or ln.startswith("# "):
            in_caption = False
            if ln.startswith("## "):
                cur = part_of(ln)
            if cur is None:
                continue
            parts.setdefault(cur, []).append(ln.lstrip("# "))
            continue
        if not ln.strip() or ln.startswith("---"):
            in_caption = False
            continue
        if cur is None:
            continue
        if CAPTION_RE.match(ln):
            in_caption = True
        # A figure and its caption are one display item wherever they sit. They
        # were once a section of their own and CAPTION_HEADS classified them by
        # heading; they now sit inside the section they illustrate, so the line
        # decides, not the heading above it. Without this the two totals this
        # function exists to print apart become the same number.
        dest = "figure captions" if in_caption or ln.startswith("![") else cur
        parts.setdefault(dest, []).append(ln)

    order = ["front matter (title, byline, keywords)", "abstract",
             "body (sections and headings)",
             "declarations and acknowledgements", "references",
             "figure captions"]
    counts = {k: plain_words("\n\n".join(v)) for k, v in parts.items()}
    for k in order:
        if k in counts:
            print(f"  {k:38s} {counts[k]:>6d}")
    caps = counts.get("figure captions", 0)
    total = sum(counts.values())
    print(f"  {'TOTAL, captions excluded':38s} {total - caps:>6d}")
    print(f"  {'TOTAL, captions included':38s} {total:>6d}")

    print("\n  the limit is stated over the cover page, the references and the "
          "acknowledgements;\n  the editorial office confirmed on 2026-09-12 that "
          "figure captions count too,\n  so the governed total is the one with "
          "captions included. The other is kept for the history.")

    def _verdict(n, against):
        over = n - against
        return f"{over:+d} over" if over > 0 else f"{-over:d} to spare"

    if limit:
        for label, n in (("excluded", total - caps), ("included", total)):
            print(f"  against {limit} with captions {label}: {_verdict(n, limit)}")
        # The same reply: "If your manuscript is below 10k words, then we would
        # be able to accept it for initial submission." That is the number the
        # initial submission is actually judged against; the stated limit above
        # is what a revision may be asked to meet.
        print(f"  against {INITIAL_SUBMISSION_CEILING} (initial-submission ceiling, "
              f"captions included): {_verdict(total, INITIAL_SUBMISSION_CEILING)}")
    return 0


def caption_count(path):
    """Break the figure-caption bucket out into one row per display item.

    `--submission` reports the captions as a single number, because that is the
    size of the question put to the editorial office. This answers the next
    question -- which caption is the long one -- and it walks the file the same
    way, so the rows sum to that bucket instead of being a second opinion about
    it. The sum is checked against it here rather than left for the reader.

    An item is grouped by its LABEL, not by position: the image line
    "![Figure 1: ...](...)" and the caption paragraph "**Figure 1. ...**" both
    carry "Figure 1" and land in the same row wherever they sit. The alt text
    gets its own column because it repeats the caption's opening sentence and
    no reader of the typeset PDF ever sees it -- but it IS inside the bucket
    total, and folding it in silently would stop the rows from summing.
    """
    src = path.read_text(encoding="utf-8")

    items, all_caption_lines = {}, []
    in_caption, in_code, skipping, label = False, False, False, None
    for ln in src.split("\n"):
        if ln.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if ln.startswith("## ") or ln.startswith("# "):
            in_caption = False
            if ln.startswith("## "):
                skipping = ln.startswith(WORKING_HEADS)
            continue
        if skipping or not ln.strip() or ln.startswith("---"):
            if not skipping:
                in_caption = False
            continue

        m = DISPLAY_LABEL_RE.match(ln)
        if CAPTION_RE.match(ln):
            in_caption, bucket = True, "caption"
        elif ln.startswith("!["):
            bucket = "alt text"
        elif in_caption:
            bucket = "caption"
        else:
            continue
        if m:
            label = f"{m.group(1).rstrip('.')} {m.group(2)}"
        key = label or "(unlabelled display item)"
        items.setdefault(key, {"caption": [], "alt text": []})[bucket].append(ln)
        all_caption_lines.append(ln)

    if not items:
        sys.exit(f"{path}: no figure or table captions found -- nothing counted.")

    print(f"  {'display item':30s} {'caption':>8s} {'alt':>6s} {'item':>7s}")
    rows = 0
    for key, buckets in items.items():
        cap = plain_words("\n\n".join(buckets["caption"]))
        alt = plain_words("\n\n".join(buckets["alt text"]))
        rows += cap + alt
        print(f"  {key:30s} {cap:>8d} {alt:>6d} {cap + alt:>7d}")
    print(f"  {'TOTAL':30s} {rows:>23d}")

    bucket_total = plain_words("\n\n".join(all_caption_lines))
    if rows != bucket_total:
        # pandoc renders one block at a time, so a row is not guaranteed to
        # count the same as its share of the whole. Say so rather than print a
        # breakdown that does not add up to the number the limit is argued over.
        print(f"\n  NOTE: the rows sum to {rows}, the bucket `--submission` "
              f"reports is {bucket_total}\n  (difference {rows - bucket_total:+d}).")
    else:
        print(f"\n  the rows sum to the {bucket_total} that `--submission` "
              "reports as one bucket.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--target", type=int, default=None,
                    help="body-word limit to compare against; prints the overage")
    ap.add_argument("--captions", action="store_true",
                    help="print one row per figure or table caption, summing to "
                         "the single caption bucket --submission reports")
    ap.add_argument("--submission", type=int, nargs="?", const=0, default=None,
                    metavar="LIMIT",
                    help="print the journal-scope count (every part of the file, "
                         "classified once) instead of the author's body count, "
                         "and compare it against LIMIT if one is given")
    args = ap.parse_args()

    if args.captions:
        return caption_count(args.path)

    if args.submission is not None:
        return submission_count(args.path, args.submission or None)

    src = args.path.read_text(encoding="utf-8")
    # Everything after a supplementary heading belongs to another document.
    src = re.split(r"^## Supplementary", src, maxsplit=1, flags=re.M)[0]

    sections, dropped = {}, {"heading": 0, "table": 0, "caption": 0,
                             "image": 0, "code": 0, "skipped section": 0}
    cur, in_code, in_skipped, in_caption = None, False, False, False

    for ln in src.split("\n"):
        if ln.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            dropped["code"] += 1
            continue

        if ln.startswith("## ") or ln.startswith("# "):
            dropped["heading"] += 1
            in_caption = False
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
            in_caption = False
            continue
        if ln.startswith("|"):
            dropped["table"] += 1
            continue
        if ln.startswith("!["):
            dropped["image"] += 1
            continue
        if ln.startswith("---") or not ln.strip():
            in_caption = False
            continue
        if CAPTION_RE.match(ln):
            in_caption = True
        if in_caption:
            dropped["caption"] += 1
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
