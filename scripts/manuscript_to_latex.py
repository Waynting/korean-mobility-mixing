#!/usr/bin/env python3
"""Convert a manuscript.md file to a clean, self-contained LaTeX file for
Overleaf collaboration.

Plain `pandoc manuscript.md -o main.tex --pdf-engine=pdflatex` silently DROPS
many Unicode math symbols used in this document's prose (Greek letters,
arrows, relations, sub/superscripts) with no warning, corrupting formulas
like the reward function. This script wraps those symbols in proper LaTeX
math (via pandoc's raw_attribute passthrough, which survives markdown's own
backslash-escaping rules, unlike bare \\(...\\) or $...$) before handing off
to pandoc. Symbols inside code spans/blocks are transliterated to ASCII
instead, since verbatim/monospace contexts can't hold LaTeX macros.

It also rewrites markdown figures (`![Figure N: caption](path/to/img.png)`)
into pandoc image-attribute syntax so the LaTeX output gets a proper
`\\label{fig:N}` alongside pandoc's default `figure`/`\\includegraphics`/
`\\caption` environment (pandoc's implicit_figures extension already builds
the float; this script only adds the missing label and strips the
redundant "Figure N:" prefix from the caption text, since LaTeX's own
`\\caption` macro already numbers the figure), then replaces that float with
a single merged caption carrying the manuscript's real caption text -- the
`**Figure N.** ...` paragraph placed right after the image (see
`merge_captions`). Referenced image files are copied from next to the source markdown into the
output directory, preserving their relative path, so the resulting LaTeX
project is a self-contained tree suitable for an Overleaf upload (no
`\\graphicspath` pointing outside the project, no symlinks that an upload
tool might not follow).

Usage:
    python3 scripts/manuscript_to_latex.py --src SRC.md --out OUT.tex

Both are required and there are no defaults, deliberately. Until 2026-08-06 they
defaulted to `manuscript.md` -> `latex/main.tex`, which was the pre-split
consolidated draft rendered into a directory with no Overleaf project behind it.
A no-argument run therefore silently rebuilt a manuscript nobody submits, and
refreshed a `latex/figures/` copy of the figures that then sat alongside the
canonical ones under `paper_drafts/figures/` with nothing to tell them apart.
That build is deleted (see `paper_drafts/latex/README.md`); requiring the pair
is what keeps it deleted. `overleaf_sync_guide.md` lists the four real pairs.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTHOR = "Wei-Ting Liu, National Taiwan University"

# manuscript.md carries the author byline as a plain-text paragraph right
# under the H1 title (so it's visible to anyone reading the raw .md); the
# LaTeX title page gets it instead via the `-M author=` metadata below, so
# that leading paragraph is stripped here to avoid printing it twice.
BYLINE_RE = re.compile(r'^\*\*[^\n]+\*\*\n[^\n]+\n\n')

GREEK = {
    'β': r'\beta', 'α': r'\alpha', 'γ': r'\gamma', 'θ': r'\theta',
    'δ': r'\delta', 'σ': r'\sigma', 'Σ': r'\Sigma', 'Δ': r'\Delta',
    'η': r'\eta', 'λ': r'\lambda', 'ε': r'\varepsilon', 'μ': r'\mu',
    'ρ': r'\rho', 'τ': r'\tau', 'π': r'\pi', 'ω': r'\omega',
    'φ': r'\varphi', 'κ': r'\kappa', 'ν': r'\nu', 'χ': r'\chi',
    'ψ': r'\psi', 'Γ': r'\Gamma', 'Θ': r'\Theta', 'Λ': r'\Lambda',
    'Π': r'\Pi', 'Φ': r'\Phi', 'Ω': r'\Omega',
}
RELATIONS = {
    '→': r'\rightarrow', '↔': r'\leftrightarrow', '⇄': r'\rightleftarrows',
    '∈': r'\in', '≈': r'\approx', '≠': r'\neq', '≤': r'\leq', '≥': r'\geq',
    '≫': r'\gg', '≪': r'\ll', '−': '-', '𝟙': r'\mathbb{1}',
    # Binary operators and relations the prose uses in inline formulas.
    # 'r ⊗ r' is the proportionate-mixing null and appears in every Results
    # section; pdflatex hard-errors on U+2297 and produces no PDF at all.
    '⊗': r'\otimes', '⊕': r'\oplus', '≡': r'\equiv', '∝': r'\propto',
    '∞': r'\infty', '∪': r'\cup', '∩': r'\cap',
    # Primes. Box 1 / Figure 1's Step-2 criterion is written over an action pair
    # (a, a'), so a dropped prime turns it into a comparison of one action with
    # itself. pdflatex hard-errors on U+2032 and writes the page without it.
    '′': r"\prime", '″': r"\prime\prime",
}
# Full digit runs, not just the digits the prose happens to use today: a digit
# missing from these maps reaches LaTeX as a bare Unicode character and hard-errors
# the build ("! LaTeX Error: Unicode character \u2075"). The signs belong here for
# the same reason: a negative exponent such as 10\u207b\u00b3 is one run, and dropping
# only its sign would invert the quantity silently rather than fail loudly.
SUB = {'\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
       '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
       '\u208a': '+', '\u208b': '-'}
SUP = {'\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3', '\u2074': '4',
       '\u2075': '5', '\u2076': '6', '\u2077': '7', '\u2078': '8', '\u2079': '9',
       '\u207a': '+', '\u207b': '-'}

# Drafting status marks. These are emoji in the .md so the working document
# stays scannable, but pdflatex hard-errors on them ("! LaTeX Error: Unicode
# character") and pandoc passes them straight through, so a draft that carries
# any of them produces a .tex that will not build on Overleaf at all. They are
# mapped rather than dropped: a section silently losing its "[TODO]" marker in
# the PDF is how an unwritten section gets mistaken for a written one.
# U+FE0F is the variation selector that follows U+26A0 in "\u26a0\ufe0f"; it has
# no glyph and must go, but only after the base character has been replaced.
STATUS = {
    '\U0001f534': '[TODO]',      # red circle: section not written
    '\U0001f7e1': '[PARTIAL]',   # yellow circle: source material exists, needs translating
    '\u2705': '[DONE]',          # check mark: drafted
    '\u26a0': '[!]',             # warning sign: a rule that must not be violated
    '\ufe0f': '',                # variation selector, no glyph
}


def fix_status(text: str) -> str:
    for ch, rep in STATUS.items():
        text = text.replace(ch, rep)
    return text


CODE_ASCII = {
    'β': 'beta', 'α': 'alpha', 'γ': 'gamma', 'θ': 'theta', 'δ': 'delta',
    'σ': 'sigma', 'Σ': 'Sigma', 'Δ': 'Delta', 'η': 'eta', 'λ': 'lambda',
    'ε': 'epsilon', 'μ': 'mu', 'ρ': 'rho', 'τ': 'tau', 'π': 'pi', 'ω': 'omega',
    '→': '->', '↔': '<->', '⇄': '<=>', '∈': 'in', '≈': '~=', '≠': '!=',
    '≤': '<=', '≥': '>=', '≫': '>>', '≪': '<<', '−': '-', '𝟙': '1',
    '′': "'", '″': "''",
    '⊗': '(x)', '⊕': '(+)', '≡': '==', '∝': 'prop-to', '∞': 'inf',
    '∪': 'union', '∩': 'intersect',
    # A combining macron cannot survive into a verbatim context, and dropping
    # it turns the geometric mean w-bar into the profile w it is the mean of.
    '\u0304': 'bar',
}
CODE_ASCII.update({k: f'_{v}' for k, v in SUB.items()})
CODE_ASCII.update({k: f'^{v}' for k, v in SUP.items()})

BASE_CHARS = 'A-Za-zΔβαγθδσΣηλ0-9'
SUB_RUN = ''.join(SUB)
SUP_RUN = ''.join(SUP)
# The base is a *run* of digits, not a single character, so "10⁷" sets the script
# on 10 rather than leaving a bare "1" in text mode next to "$0^{7}$" (which is what
# a single-character base produced through 2026-08-03, in eight places across the two
# forks). The digit alternative is tried first; a letter or Greek base is still one
# character, so "R²" and "α₁" are unaffected.
combo_re = re.compile(r'([0-9]+|[' + BASE_CHARS + r'])([' + SUB_RUN + r']+|[' + SUP_RUN + r']+)')
# A sub/superscript run combo_re did not absorb has no base character directly
# in front of it, usually because markdown emphasis sits in between: "*R*₀"
# puts a `*` where combo_re expects the R. Set the run on its own so it still
# reaches the PDF as a script, instead of arriving at LaTeX as a bare Unicode
# character it has no glyph for ("! LaTeX Error: Unicode character ₀").
orphan_re = re.compile(r'([' + SUB_RUN + r']+)|([' + SUP_RUN + r']+)')
single_re = re.compile('([' + ''.join(GREEK) + ''.join(RELATIONS) + '])')
# A combining macron (U+0304) sits AFTER its base letter, so it is not a
# character the single-symbol map can catch: "w\u0304" is two code points and
# pdflatex hard-errors on the second. It is also not droppable -- w-bar is the
# geometric mean of the coverage profile w, and losing the bar silently turns
# a scalar into the vector it summarises.
macron_re = re.compile('([A-Za-z' + ''.join(GREEK) + '])\u0304')
SPLIT_RE = re.compile(r'(```.*?```|`[^`\n]+`)', re.DOTALL)


def _raw(latex_math: str) -> str:
    # raw_attribute passthrough: bare $...$ silently fails to parse when the
    # closing $ is immediately followed by a digit (common here, e.g.
    # "$\geq$0.5"), and \(...\) gets its backslashes stripped by markdown's
    # escape rules, leaving bare commands like \beta unprotected in text mode.
    return '`' + latex_math + '`{=latex}'


def _combo_repl(m):
    base, run = m.group(1), m.group(2)
    base_tex = GREEK.get(base, base)
    if run[0] in SUB:
        return _raw('$' + base_tex + '_{' + ''.join(SUB[c] for c in run) + '}$')
    return _raw('$' + base_tex + '^{' + ''.join(SUP[c] for c in run) + '}$')


def _orphan_repl(m):
    sub, sup = m.group(1), m.group(2)
    if sub:
        return _raw('$_{' + ''.join(SUB[c] for c in sub) + '}$')
    return _raw('$^{' + ''.join(SUP[c] for c in sup) + '}$')


def _single_repl(m):
    ch = m.group(1)
    return _raw('$' + (GREEK.get(ch) or RELATIONS.get(ch)) + '$')


def _macron_repl(m):
    base = m.group(1)
    return _raw('$\\bar{' + (GREEK.get(base) or base) + '}$')


def fix_prose(text: str) -> str:
    # The macron goes first: it consumes its base letter, so nothing later can
    # match that letter and leave the combining mark stranded on its own.
    text = macron_re.sub(_macron_repl, text)
    # combo_re next: it consumes the base character along with its script, so
    # orphan_re only ever sees the runs that genuinely have no base attached.
    text = combo_re.sub(_combo_repl, text)
    text = orphan_re.sub(_orphan_repl, text)
    return single_re.sub(_single_repl, text)


def fix_code(text: str) -> str:
    return ''.join(CODE_ASCII.get(ch, ch) for ch in text)


# Standalone markdown image paragraph: ![alt](path). Figures in this
# manuscript are woven in with alt text of the form "Figure N: caption...";
# capture the number so it can become a \label{fig:N} and be stripped from
# the caption text (pandoc's \caption already prints "Figure N:" via the
# LaTeX figure counter, so leaving the prefix in would print it twice).
IMAGE_RE = re.compile(r'^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)\)[ \t]*$', re.MULTILINE)
# `num` accepts an optional leading "S" so supplementary figures ("Figure S1:")
# get their prefix stripped from the caption too. Without this the alt text
# falls through unchanged and LaTeX prints "Figure 1: Figure S1: ...".
FIGNUM_RE = re.compile(r'^Figure\s+(?P<num>S?\d+)\s*:\s*(?P<caption>.*)$', re.DOTALL)


# A section whose heading says it is not for submission is a working aid -- a
# checklist the author keeps in the .md and deletes on the way out. Deleting it by
# hand before each build is exactly the step that gets forgotten once, and it is
# the one mistake that cannot be taken back after a manuscript is uploaded. So the
# build drops it instead, keyed on the marker rather than on one section's title,
# and says out loud what it dropped: a silent removal would be its own hazard.
NOT_FOR_SUBMISSION = re.compile(
    r'^##\s+[^\n]*\(not for submission\)[^\n]*\n.*?(?=^##\s|\Z)',
    re.M | re.S | re.I,
)


def drop_working_sections(md: str) -> str:
    out, n = NOT_FOR_SUBMISSION.subn('', md)
    for m in NOT_FOR_SUBMISSION.finditer(md):
        print(f'dropped, marked not for submission: {m.group(0).splitlines()[0][3:]}')
    if not n:
        print('no sections marked "(not for submission)"')
    return out


def process_images(md: str, src_dir: Path, out_dir: Path) -> str:
    """Rewrite markdown image syntax for proper LaTeX figures + labels, and
    copy each referenced image from next to the source markdown into the
    output directory (same relative path), so the LaTeX project is
    self-contained (Overleaf-uploadable, no external/absolute paths)."""

    def repl(m: re.Match) -> str:
        alt, path = m.group('alt'), m.group('path')

        fig_m = FIGNUM_RE.match(alt)
        if fig_m:
            num, caption = fig_m.group('num'), fig_m.group('caption')
            new_alt = f'![{caption}]({path}){{#fig:{num}}}'
        else:
            new_alt = m.group(0)

        src_file = (src_dir / path).resolve()
        if src_file.is_file():
            dst_file = out_dir / path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
        else:
            print(f'warning: image not found on disk, not copied: {src_file}')

        return new_alt

    return IMAGE_RE.sub(repl, md)


# Each figure reaches pandoc as an image whose alt text is only the short
# title, immediately followed by the manuscript's real caption as an ordinary
# "**Figure N.** ..." paragraph. Pandoc therefore emits two captions per
# figure: an auto-numbered \caption holding the short title, and a body
# paragraph holding the long one with its number hard-coded in the markdown.
# The two numbers agree only by luck -- LaTeX numbers floats by position, and
# the main text reaches the design-choice figure (§3.5) before the
# mechanism-forest figure (§4.2), which is the reverse of how they were once
# named. Merge the pair so there is one caption and one source of numbering.
#
# The merged captions run 1.4k-4k characters, and nothing that boxes a caption
# can hold text of that length. A float cannot: LaTeX measures a caption by
# setting it in one hbox, which at this size either overflows the page ("Float
# too large for page") or blows past TeX's maximum dimension outright
# ("! Dimension too large" -- fatal, no PDF at all). \captionof from the
# caption package survives the measurement but is still typeset as one
# unbreakable unit, so it will not start on a page it cannot finish on (half a
# page of white space under every figure) and, once a caption exceeds a full
# page, the remainder is silently dropped off the bottom.
#
# So the caption is not a caption object at all: the figure is emitted
# unfloated as a centred image followed by an ordinary paragraph that steps the
# figure counter itself. \refstepcounter + \label keeps \ref working and keeps
# the number coming from LaTeX rather than from the markdown, while the caption
# text stays a normal paragraph -- breakable across pages, no measurement, no
# dimension limit, and no caption package needed.
FIGURE_BLOCK_RE = re.compile(
    r'\\begin\{figure\}\n'
    r'\\centering\n'
    r'(?P<graphic>.*?)\n'
    r'\\caption\{(?P<short>.*?)\}(?P<label>\\label\{[^}]*\})?\n'
    r'\\end\{figure\}\n'
    r'\n'
    r'\\textbf\{Fig(?:ure)?\.?\s*(?P<num>S?\d+)\.\}[ ~]*(?P<long>.*?)(?=\n\n|\Z)',
    re.DOTALL,
)


def merge_captions(tex: str) -> str:
    """Fold each figure's "**Figure N.** ..." body paragraph into its caption,
    and unfloat the figure so a caption of that length can still be typeset."""

    seen_main = False
    switched = False

    def repl(m: re.Match) -> str:
        nonlocal seen_main, switched
        # A document that runs plain figures and then S-numbered ones (the JBI
        # fork keeps Figures 1-4 in the main text and demotes the rest to its
        # own supplementary section) has to restart the counter mid-document,
        # or LaTeX numbers the supplementary figures 5 and 6. Documents that
        # are S-numbered throughout get \thefigure from --supplement instead,
        # so nothing is injected for them.
        prefix = ''
        if m.group('num').startswith('S'):
            if seen_main and not switched:
                switched = True
                prefix = ('\\setcounter{figure}{0}\n'
                          '\\renewcommand{\\thefigure}{S\\arabic{figure}}\n\n')
        else:
            seen_main = True

        return (
            prefix
            + '\\begin{center}\n'
            f'{m.group("graphic")}\n'
            '\\end{center}\n'
            '\\begingroup\\small\n'
            f'\\refstepcounter{{figure}}{m.group("label") or ""}%\n'
            f'\\noindent\\textbf{{Figure \\thefigure:}} {m.group("long").strip()}\\par\n'
            '\\endgroup'
        )

    merged, n = FIGURE_BLOCK_RE.subn(repl, tex)

    # Anything left behind means a figure did not match the expected shape and
    # would silently keep its duplicate caption, so say so rather than pass.
    stranded = merged.count(r'\begin{figure}')
    orphans = re.findall(r'\\textbf\{Fig(?:ure)?\.?\s*S?\d+\.\}', merged)
    print(f'merged {n} figure caption(s)')
    if stranded:
        print(f'warning: {stranded} figure float(s) left unmerged')
    if orphans:
        print(f'warning: {len(orphans)} caption paragraph(s) left in the body: '
              f'{", ".join(orphans)}')
    return merged


# A markdown blockquote whose first line is bold "Box N. <title>" is an
# Elsevier-style Box, not a quotation. Pandoc has no notion of that, so it
# lands as \begin{quote}, which prints as a plain indented paragraph with no
# rule around it -- indistinguishable from body text once the bold line
# scrolls off. Promote it to a ruled, breakable tcolorbox here rather than
# writing raw LaTeX into the markdown, so the .md stays readable as markdown.
BOX_RE = re.compile(
    r'\\begin\{quote\}\n'
    r'\\textbf\{(?P<title>Box\s+\d+\.[^}]*?)\}\n'
    r'\n'
    r'(?P<body>.*?)\n'
    r'\\end\{quote\}',
    re.DOTALL,
)

# `breakable` needs the `most` library; without it a box longer than the
# remaining page is silently clipped.
BOX_PREAMBLE = r'''\usepackage[most]{tcolorbox}
\newtcolorbox{screenbox}[1]{%
  breakable, enhanced, sharp corners,
  colback=black!2, colframe=black!70, boxrule=0.8pt,
  left=8pt, right=8pt, top=8pt, bottom=8pt,
  fonttitle=\bfseries, coltitle=black, colbacktitle=black!8,
  title={#1}}
'''


def boxify(tex: str) -> str:
    def repl(m: re.Match) -> str:
        title = ' '.join(m.group('title').split())
        return (f'\\begin{{screenbox}}{{{title}}}\n'
                f'{m.group("body").strip()}\n'
                '\\end{screenbox}')

    boxed, n = BOX_RE.subn(repl, tex)
    print(f'promoted {n} blockquote(s) to a framed box')
    return boxed


# Characters pdflatex typesets without help under the default pandoc template
# (inputenc utf8 + T1 fontenc): punctuation and the Latin-1/Latin Extended-A
# letters. Everything else must have been mapped above.
SAFE_NON_ASCII = (set('\u2014\u2013\u00a7\u00d7\u00b7\u00b1\u2026\u00b0'
                      '\u2018\u2019\u201c\u201d\u2020\u2021\u2030\u20ac\u00a3')
                  | {chr(c) for c in range(0x00c0, 0x0180)})


def check_typesettable(tex: str, out_path: Path) -> None:
    """Abort if any unmapped non-ASCII character survived into the .tex.

    pdflatex turns an unmapped character into a fatal error and produces no
    PDF, and pandoc passes such characters straight through, so the failure
    surfaces two steps later as a wall of TeX log. Worse, the symbol maps
    above are the only thing standing between a formula and a silent change of
    meaning, and they are necessarily incomplete: the day a new symbol enters
    the prose is the day this script has to say so. Failing here names the
    character and the line rather than leaving it to be found in main.log.
    """
    bad = {}
    for lineno, line in enumerate(tex.split('\n'), 1):
        for ch in line:
            if ord(ch) > 127 and ch not in SAFE_NON_ASCII:
                bad.setdefault(ch, []).append(lineno)
    if not bad:
        return
    report = '\n'.join(
        f'  U+{ord(ch):04X} {ch!r}  x{len(lines)}  first at {out_path.name}:{lines[0]}'
        for ch, lines in sorted(bad.items(), key=lambda kv: -len(kv[1])))
    raise SystemExit(
        f'{out_path}: {len(bad)} character(s) that pdflatex cannot typeset '
        f'survived conversion:\n{report}\n'
        'Add each to GREEK/RELATIONS (with its CODE_ASCII twin) in '
        f'{Path(__file__).name}, or romanise it in the source markdown. '
        'Do not delete it: a dropped symbol changes what a formula says.')


def process(md: str) -> str:
    md = fix_status(md)
    parts = SPLIT_RE.split(md)
    out = []
    for part in parts:
        is_code = part.startswith('```') or (part.startswith('`') and part.endswith('`') and len(part) >= 2)
        out.append(fix_code(part) if is_code else fix_prose(part))
    return ''.join(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Required, with no defaults. See the module docstring: the old defaults
    # rebuilt a draft nobody submits into a directory with no project behind it.
    p.add_argument('--src', type=Path, required=True,
                    help='source manuscript markdown, e.g. paper_drafts/manuscript_JBI.md')
    p.add_argument('--out', type=Path, required=True,
                    help='output .tex path, e.g. paper_drafts/latex/JBI/main.tex')
    p.add_argument('--supplement', action='store_true',
                    help='number floats S1, S2, ... instead of 1, 2, ... '
                         '(use for the Supplementary Information document, whose '
                         'prose refers to "Figure S1" / "Table S1")')
    return p.parse_args()


def main():
    args = parse_args()
    src_path = args.src if args.src.is_absolute() else REPO_ROOT / args.src
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out

    src = src_path.read_text(encoding='utf-8')

    lines = src.split('\n', 2)
    assert lines[0].startswith('# '), f'expected {src_path.name} to start with an H1 title'
    title = lines[0][2:].strip()
    body = lines[2] if len(lines) > 2 else ''
    body = BYLINE_RE.sub('', body, count=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = drop_working_sections(body)
    body = process_images(body, src_dir=src_path.parent, out_dir=out_path.parent)
    fixed = process(body)

    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False, encoding='utf-8') as tmp:
        tmp.write(fixed)
        tmp_path = tmp.name

    # One header include carries everything injected into the preamble: the
    # box environment always, the supplementary float counters on request.
    with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False,
                                     encoding='utf-8') as hdr:
        hdr.write(BOX_PREAMBLE)
        if args.supplement:
            # Renew both counters so a supplementary float prints as "Figure S1"
            # rather than "Figure 1", matching how the SI prose refers to them.
            hdr.write('\\renewcommand{\\thefigure}{S\\arabic{figure}}\n')
            hdr.write('\\renewcommand{\\thetable}{S\\arabic{table}}\n')
        extra = ['-H', hdr.name]

    subprocess.run([
        'pandoc', tmp_path,
        '-f', 'markdown+smart+raw_attribute',
        '-o', str(out_path),
        '--standalone',
        '--pdf-engine=pdflatex',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt',
        '--toc=false',
        '-M', f'title={title}',
        '-M', f'author={AUTHOR}',
        '-M', 'date=',
    ] + extra, check=True)

    tex = boxify(merge_captions(out_path.read_text(encoding='utf-8')))
    check_typesettable(tex, out_path)
    out_path.write_text(tex, encoding='utf-8')

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
