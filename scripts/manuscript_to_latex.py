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

Displayed equations are written as `$$...$$` blocks of LaTeX in the markdown
master and pass through to pandoc's display math untouched. They used to be
four-space indented blocks of Unicode, which markdown reads as *code*: the
symbol pass treated them as prose and wrapped each symbol in a raw-attribute
span, and the verbatim environment then printed that span literally, so the
PDF showed `$-$`{=latex} in a monospace font where the equation should have
been.

It also rewrites markdown figures (`![Figure N: caption](path/to/img.png)`)
into pandoc image-attribute syntax so the LaTeX output gets a proper
`\\label{fig:N}` alongside pandoc's default `figure`/`\\includegraphics`/
`\\caption` environment (pandoc's implicit_figures extension already builds
the float; this script only adds the missing label and strips the
redundant "Figure N:" prefix from the caption text, since LaTeX's own
`\\caption` macro already numbers the figure), then replaces that float with
a single merged caption carrying the manuscript's real caption text -- the
`**Figure N. <title sentence>** ...` paragraph placed right after the image
(see `merge_captions`). Referenced image files are copied from next to the
source markdown into a `figures/` subdirectory of the output directory, and the
reference in the .tex is rewritten to point there, so the resulting LaTeX
project is a self-contained tree suitable for an Overleaf upload (no
`\\graphicspath` pointing outside the project, no symlinks that an upload
tool might not follow).

The manuscript's front matter -- byline, affiliation, corresponding author,
keywords -- is markdown prose sitting between the H1 and the first horizontal
rule. It is parsed out here and set as a title block, because pandoc otherwise
prints it as four ordinary paragraphs underneath a `\\maketitle` that has
already printed a byline of its own. `paper/latex/README.md` records what is
still owed on top of that, and why the output no longer carries the line
numbers and double spacing it did until 2026-08-29.

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
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

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
    # U+2009 THIN SPACE. Royal Society house style sets the thousands separator
    # as a thin space rather than a comma ("10 186 891 962"), so the markdown
    # master carries the real character and this maps it to LaTeX's own thin
    # space. Dropping it would run the digit groups together into a different
    # number; leaving it unmapped hard-errors pdflatex.
    ' ': r'\,',
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
    # The thin space of the RELATIONS map. A verbatim context cannot hold
    # `\,`, and a thousands separator that vanishes changes the number, so it
    # degrades to an ordinary space rather than to nothing.
    ' ': ' ',
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
# Four kinds of span are not prose and must not have the raw-attribute
# treatment applied to them: fenced code, display math, inline math, and
# inline code. `$$...$$` is tried before `$...$` so a display block is never
# split into two inline ones, and both come before the code-span alternative
# so a formula containing a backtick cannot be mistaken for a code span.
# Inline math is confined to a single line; display math may span several.
SPLIT_RE = re.compile(
    r'(```.*?```|\$\$.*?\$\$|\$[^$\n]+\$|`[^`\n]+`)', re.DOTALL)


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


def fix_math(text: str) -> str:
    """Map the Unicode symbols inside a `$$...$$` block to bare LaTeX macros.

    Math is already in math mode, so the raw-attribute wrapper `fix_prose`
    builds -- a backtick span carrying its own `$...$` -- would nest a math
    shift inside a math shift and break the build. The macros go in unwrapped
    instead. Both the displayed equations and the inline formulas are written
    in LaTeX in the markdown master, so today this function has nothing to do;
    it exists so that the day a Greek letter is typed straight into one, it
    lands as \\beta rather than as a nested `$\\beta$`{=latex}.
    """
    text = macron_re.sub(lambda m: '\\bar{' + (GREEK.get(m.group(1)) or m.group(1)) + '}', text)
    text = combo_re.sub(_combo_repl_math, text)
    text = orphan_re.sub(_orphan_repl_math, text)
    return single_re.sub(
        lambda m: (GREEK.get(m.group(1)) or RELATIONS.get(m.group(1))), text)


def _combo_repl_math(m):
    base, run = m.group(1), m.group(2)
    base_tex = GREEK.get(base, base)
    if run[0] in SUB:
        return base_tex + '_{' + ''.join(SUB[c] for c in run) + '}'
    return base_tex + '^{' + ''.join(SUP[c] for c in run) + '}'


def _orphan_repl_math(m):
    sub, sup = m.group(1), m.group(2)
    if sub:
        return '_{' + ''.join(SUB[c] for c in sub) + '}'
    return '^{' + ''.join(SUP[c] for c in sup) + '}'


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


# A float caption paragraph that numbers itself "S1", "S2", ... belongs to a
# document whose floats must carry the S prefix, because the main text refers
# to them by those names ("Table S1"). That prefix is not something pandoc
# knows about; it comes from the two \renewcommand lines --supplement writes
# into the preamble. Forgetting the flag therefore does not fail -- it
# produces a PDF whose tables are numbered 1, 2, ... while every reference to
# them in the other document still says S1, S2, and nothing in the build says
# so. That is the same class of silent breakage as a dropped symbol, so it is
# caught the same way: by refusing to write the file.
#
# Detection keys on captions the document *defines* ("**Table S1.** ...", at
# the start of a paragraph), not on the "Table S1" the main text uses to refer
# to one, so the two documents are told apart by what they contain rather than
# by their filenames. A document that defines both plain- and S-numbered
# floats is the mid-document restart `merge_captions` handles, and is left
# alone.
S_FLOAT_RE = re.compile(r'^\*\*(?:Table|Figure)\s+S\d+\.', re.M)
PLAIN_FLOAT_RE = re.compile(r'^\*\*(?:Table|Figure)\s+\d+\.', re.M)


def check_supplement_flag(md: str, supplement: bool, src_path: Path) -> None:
    s_floats = S_FLOAT_RE.findall(md)
    plain_floats = PLAIN_FLOAT_RE.findall(md)
    kind = ('mixed' if s_floats and plain_floats else
            'supplementary' if s_floats else
            'main' if plain_floats else 'no floats')
    print(f'float numbering: {kind} '
          f'({len(plain_floats)} plain, {len(s_floats)} S-numbered), '
          f'--supplement {"on" if supplement else "off"}')
    if kind == 'mixed':
        # merge_captions restarts the counter at the first S-numbered float.
        return
    if kind == 'supplementary' and not supplement:
        raise SystemExit(
            f'{src_path}: this document numbers its own floats '
            f'{", ".join(f[2:-1] for f in s_floats)}, but --supplement was not '
            'passed, so the .tex would number them 1, 2, ... and every '
            '"Table S1" pointing at them from the main text would point at '
            'nothing. Re-run with --supplement.')
    if kind == 'main' and supplement:
        raise SystemExit(
            f'{src_path}: this document numbers its own floats '
            f'{", ".join(f[2:-1] for f in plain_floats)}, so --supplement '
            'would print them as "Figure S1", "Table S1", which is what the '
            'supplementary document calls its own. Drop --supplement.')


# The front matter is everything between the H1 title and the first horizontal
# rule: the byline, the affiliation, the corresponding-author line and (in the
# main text) the keywords. Pandoc has no notion of any of it, so left alone it
# lands as four ordinary body paragraphs *underneath* a \maketitle that has
# already printed a second, differently-worded byline of its own. Parsing it
# here puts each piece where LaTeX expects it and leaves exactly one byline.
#
# Every paragraph must be recognised or the build stops. A front matter line
# that quietly falls through to the body is a corresponding-author address
# printed in the middle of the abstract, and the whole point of a title page
# is that a reader can find those four things without hunting.
FRONT_SEP_RE = re.compile(r'^---[ \t]*$', re.M)
FRONT_BYLINE_RE = re.compile(r'\*\*(?P<names>[^*\n]+)\*\*(?P<marks>[^\n]*)\Z', re.S)
FRONT_ADDRESS_RE = re.compile(r'\^[^\^\n]+\^[ \t]+.+\Z', re.S)
FRONT_KEYWORDS_RE = re.compile(r'\*\*Keywords:\*\*[ \t]*.+\Z', re.S)
# The separator is looked for near the top only. A manuscript with no rule
# under its front matter would otherwise swallow the whole first section.
FRONT_MAX_LINES = 24


def split_front_matter(md: str, src_path: Path):
    """Split the byline block off the body and classify each of its paragraphs.

    Returns (byline_md, address_mds, keywords_md_or_None, remaining_body).
    """
    sep = FRONT_SEP_RE.search(md)
    if sep is None or md.count('\n', 0, sep.start()) > FRONT_MAX_LINES:
        raise SystemExit(
            f'{src_path}: no "---" rule found in the first {FRONT_MAX_LINES} '
            'lines after the title, so the front matter (byline, affiliation, '
            'corresponding author, keywords) cannot be told apart from the '
            'body. Every manuscript here ends its front matter with one.')

    byline, addresses, keywords = None, [], None
    for para in [p.strip() for p in md[:sep.start()].split('\n\n')]:
        if not para:
            continue
        if byline is None and FRONT_BYLINE_RE.fullmatch(para):
            byline = para
        elif FRONT_KEYWORDS_RE.fullmatch(para):
            keywords = para
        elif FRONT_ADDRESS_RE.fullmatch(para):
            addresses.append(para)
        else:
            raise SystemExit(
                f'{src_path}: front matter paragraph not recognised as a '
                f'byline, an affiliation or a keyword list:\n  {para[:120]}\n'
                'Add a rule for it in split_front_matter(), or move it below '
                'the "---". Nothing here may reach the body unlabelled.')
    if byline is None:
        raise SystemExit(f'{src_path}: front matter has no **Author Name** byline.')

    print(f'front matter: byline, {len(addresses)} address line(s), '
          f'{"keywords" if keywords else "no keywords"}')
    return byline, addresses, keywords, md[sep.end():].lstrip('\n')


# One pandoc call renders every front matter fragment, so the byline and the
# addresses go through exactly the same escaping rules as the body -- a
# hand-rolled escaper is how "[TODO: postcode]" turns into a citation.
# The fragments come back concatenated, so they are separated by a marker that
# survives the round trip as itself.
FRAGMENT_SEP = r'\PandocFrontMatterSplit'


def render_fragments(fragments):
    joined = ('\n\n`' + FRAGMENT_SEP + '`{=latex}\n\n').join(fragments)
    out = subprocess.run(
        ['pandoc', '-f', 'markdown+smart+raw_attribute', '-t', 'latex'],
        input=joined, capture_output=True, text=True, check=True).stdout
    parts = [p.strip() for p in out.split(FRAGMENT_SEP)]
    assert len(parts) == len(fragments), (
        f'front matter round trip lost a fragment: sent {len(fragments)}, '
        f'got {len(parts)}')
    return parts


def title_block(address_texs, keywords_tex):
    """The centred block that sits under \\maketitle: affiliation,
    corresponding author, keywords. The byline itself goes into \\author.

    It is a `center` environment and not a `tabular`, because the affiliation
    line runs past the text width and a tabular cell does not wrap: the
    address would silently run off the right-hand edge of the page.
    """
    lines = list(address_texs)
    if keywords_tex:
        lines.append(keywords_tex)
    if not lines:
        return ''
    body = '\\\\[4pt]\n'.join(lines)
    return '\\begin{center}\\small\n' + body + '\n\\end{center}\n'


# Where a copied image lands inside the output directory. The image is copied
# under this one flat subdirectory and the .tex is rewritten to point there,
# rather than the reference being reproduced verbatim.
#
# WHY IT IS FLATTENED. The manuscripts live in `paper/` and the figures in
# `eda/fig/`, so a reference from the markdown master reads
# `../eda/fig/p48_figure1.pdf` -- and copying "to the same relative path" put
# that file at `paper/latex/JRSI/../eda/fig/...`, i.e. `paper/latex/eda/fig/...`,
# OUTSIDE the LaTeX project the copy exists to make self-contained. The .tex
# then carried a `..` that climbs out of the uploaded tree: the local build
# still worked (the escaped copy was sitting there), and the Overleaf upload
# built a manuscript with seven missing figures. Rewriting the reference to
# `figures/<basename>` keeps the copy and the reference inside `--out`'s own
# directory, which is the property `process_images` is here to provide.
#
# The flattening is by basename, so two figures with the same file name and
# different sources would silently overwrite each other. That is checked rather
# than assumed: the second one aborts the run.
FIGURE_SUBDIR = 'figures'


def process_images(md: str, src_dir: Path, out_dir: Path) -> str:
    """Rewrite markdown image syntax for proper LaTeX figures + labels, and
    copy each referenced image from next to the source markdown into
    `<out_dir>/figures/`, rewriting the reference to match, so the LaTeX
    project is self-contained (Overleaf-uploadable, no external/absolute
    paths and nothing reached through `..`)."""

    claimed: dict[str, Path] = {}

    def repl(m: re.Match) -> str:
        alt, path = m.group('alt'), m.group('path')

        src_file = (src_dir / path).resolve()
        if not src_file.is_file():
            raise SystemExit(
                f'image referenced by the manuscript does not exist: {src_file}\n'
                f'  reference: {m.group(0)}\n'
                'A missing figure is not a warning here: the .tex would still be '
                'written, pdflatex would stop on the \\includegraphics, and the '
                'figure would be missing from an upload that otherwise looks '
                'complete. Fix the path in the markdown, or draw the figure.')

        dst_rel = f'{FIGURE_SUBDIR}/{PurePosixPath(path).name}'
        previous = claimed.get(dst_rel)
        if previous is not None and previous != src_file:
            raise SystemExit(
                f'two different images would be copied to {dst_rel}:\n'
                f'  {previous}\n  {src_file}\n'
                'Images are flattened into one directory inside the LaTeX '
                'project, so their file names have to be unique. Rename one at '
                'the source.')
        claimed[dst_rel] = src_file

        dst_file = out_dir / dst_rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

        fig_m = FIGNUM_RE.match(alt)
        if fig_m:
            num, caption = fig_m.group('num'), fig_m.group('caption')
            return f'![{caption}]({dst_rel}){{#fig:{num}}}'
        return f'![{alt}]({dst_rel})'

    out = IMAGE_RE.sub(repl, md)
    print(f'copied {len(claimed)} image(s) into {FIGURE_SUBDIR}/')
    return out


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
#
# WHAT THE PATTERN STOPS AT, and why it stops there. Until 2026-08-29 it ran to
# `\textbf{Figure N.}` -- bold holding the number and nothing else -- and then
# read the caption prose as plain text after it. The manuscript's captions do
# not have that shape: house style bolds the whole title sentence
# ("**Figure 3. The passive matrix departs ... of the month.** (a) ..."), which
# pandoc renders as one `\textbf{...}` carrying the number AND the title, wrapped
# over several lines. None of the seven matched, and a caption that does not
# match keeps its float and prints twice.
#
# So the regex now stops at the number's full stop, INSIDE the still-open
# `\textbf{`, and the rest of the bold run is found by counting braces
# (`_match_brace`) rather than by a `[^}]*` that the first `$R_{0}$` in a title
# would end early. That keeps the bold title sentence bold in the PDF, keeps the
# markdown master in the format the captions are actually written in, and does
# not care how many lines pandoc wrapped the title over. A caption written the
# old way (`**Figure N.** ...`) still matches: its bold run is simply empty
# after the full stop.
# `(?:(?!\n\n).)*?` is `.*?` that may not cross a blank line. It matters: with a
# plain DOTALL `.*?`, a float whose caption paragraph is NOT directly underneath
# it (an intervening sentence, say) does not fail -- the match simply runs on to
# the NEXT figure's caption, swallowing the float, its own caption and the prose
# between them into the `graphic` group, and the run reports one fewer figure and
# no leftovers. Pandoc never puts a blank line inside a float, so refusing to
# cross one costs nothing and turns that into the abort below.
FIGURE_BLOCK_RE = re.compile(
    r'\\begin\{figure\}\n'
    r'\\centering\n'
    r'(?P<graphic>(?:(?!\n\n).)*?)\n'
    r'\\caption\{(?P<short>(?:(?!\n\n).)*?)\}(?P<label>\\label\{[^}]*\})?\n'
    r'\\end\{figure\}\n'
    r'\n'
    r'\\textbf(?P<open>\{)Fig(?:ure)?\.?\s*(?P<num>S?\d+)\.',
    re.DOTALL,
)
# The same head without the float in front of it: a caption paragraph still
# carrying its own hard-coded number after the merge is one that never found its
# image, and is reported by number.
CAPTION_HEAD_RE = re.compile(r'\\textbf\{Fig(?:ure)?\.?\s*(?P<num>S?\d+)\.')


def _match_brace(tex: str, open_idx: int, out_path: Path) -> int:
    """Index of the `}` closing the `{` at `open_idx`.

    A backslash escapes the character after it, so `\\{` and `\\}` inside the
    caption do not unbalance the count and a control sequence's name is skipped
    without being inspected.
    """
    depth = 0
    i = open_idx
    while i < len(tex):
        ch = tex[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise SystemExit(
        f'{out_path}: a figure caption\'s \\textbf{{...}} is never closed '
        f'(opened at character {open_idx}). The caption cannot be merged, so '
        'nothing is written.')


def merge_captions(tex: str, out_path: Path) -> str:
    """Fold each figure's "**Figure N. ...**" body paragraph into its caption,
    and unfloat the figure so a caption of that length can still be typeset."""

    seen_main = False
    switched = False
    out, pos, n = [], 0, 0

    for m in FIGURE_BLOCK_RE.finditer(tex):
        if m.start() < pos:      # already consumed by the previous caption
            continue
        # The bold run holds the number plus the title sentence; the rest of the
        # paragraph, up to the blank line, is the caption proper.
        close = _match_brace(tex, m.start('open'), out_path)
        title = tex[m.end():close].strip()
        para_end = tex.find('\n\n', close)
        if para_end == -1:
            para_end = len(tex)
        long = tex[close + 1:para_end].strip()

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

        out.append(tex[pos:m.start()])
        out.append(
            prefix
            + '\\begin{center}\n'
            f'{m.group("graphic")}\n'
            '\\end{center}\n'
            '\\begingroup\\small\n'
            f'\\refstepcounter{{figure}}{m.group("label") or ""}%\n'
            f'\\noindent\\textbf{{Figure \\thefigure.'
            f'{" " + title if title else ""}}} {long}\\par\n'
            '\\endgroup'
        )
        pos = para_end
        n += 1

    out.append(tex[pos:])
    merged = ''.join(out)

    # Anything left behind means a figure did not match the expected shape and
    # would keep its duplicate caption -- the image printed with pandoc's
    # auto-numbered short caption under it, then the real caption again as body
    # prose carrying a number LaTeX did not assign. That is the failure this
    # function exists to prevent and it is invisible in a build that otherwise
    # succeeds, so it stops the run rather than printing a warning nobody reads.
    stranded = merged.count(r'\begin{figure}')
    orphans = CAPTION_HEAD_RE.findall(merged)
    print(f'merged {n} figure caption(s)')
    if stranded or orphans:
        raise SystemExit(
            f'{out_path}: {stranded} figure float(s) and {len(orphans)} caption '
            f'paragraph(s) ({", ".join(orphans) or "none"}) did not pair up.\n'
            'A figure and its "**Figure N. ...**" paragraph must be adjacent in '
            'the markdown, with the paragraph immediately after the image and '
            'nothing between them. Nothing is written.')
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

# Nothing is added to the preamble for review formatting any more.
#
# This constant used to carry continuous line numbers (`lineno` +
# `\linenumbers`) and double spacing (`setspace` + `\doublespacing`), on the
# reasoning that a reviewer cites the manuscript by line and reads it
# marked-up. Both were removed on 2026-08-29 at the author's request: the
# journal's written requirements ask for neither, and the Royal Society's own
# guidelines make initial submission format-free, so the draft is read the way
# it is written instead. Restoring either is a matter of writing those lines
# into the header include below -- ahead of hyperref, which is where pandoc
# puts header-includes and the order both packages want.
REVIEW_PREAMBLE = ''

# `\urlstyle{same}` sets a URL in the surrounding font, and inside `\emph`
# that font is italic. url.sty makes every break character math-active, so an
# italic URL collects italic correction at each one and the reference block's
# note printed its DOI as "doi:10 .5 28 1/ zenodo.2 21 52 08 9" -- a DOI with
# spaces in it, which is not the DOI, and which copies and machine-reads as the
# wrong string. The fix is to set URLs in the upright body font wherever they
# appear; the break points, and so the wrapping of the 90-character MOE links,
# are untouched. It has to be \AtBeginDocument, because pandoc puts
# header-includes ahead of its own `\urlstyle{same}` and a style set here in
# the preamble would simply be overwritten.
URL_PREAMBLE = r"""\makeatletter
\def\url@uprightstyle{\def\UrlFont{\normalfont}}
\makeatother
\AtBeginDocument{\urlstyle{upright}}
"""

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


def check_urls_wrapped(tex: str, out_path: Path) -> None:
    """Abort if a bare URL reached the .tex outside \\url{}.

    A URL is one long token with no space in it, so TeX cannot break it and
    sets it into the margin instead -- and an over-long line is not an error,
    it is a warning buried in main.log among fourteen others. What reaches the
    page is a reference whose URL stops in the middle, which reads as a
    complete URL and is not one. That is the same failure mode as the emoji and
    the Hangul: pandoc passes it through, pdflatex does not complain loudly
    enough, and the damage is only visible in the PDF.

    The front matter is rendered by its own pandoc call, so a URL added to an
    affiliation line would arrive here unwrapped even though the main call
    carries the flag. This gate covers the whole emitted file for that reason.
    """
    bad = [(n, m.group(0))
           for n, line in enumerate(tex.split('\n'), 1)
           if '\\url{' not in line and '\\href{' not in line
           for m in [re.search(r'(?<![{\w])https?://\S+', line)] if m]
    if not bad:
        return
    report = '\n'.join(f'  {out_path.name}:{n}  {u[:72]}' for n, u in bad)
    raise SystemExit(
        f'{out_path}: {len(bad)} URL(s) reached the .tex outside \\url{{}}:\n'
        f'{report}\n'
        'pandoc needs +autolink_bare_uris on the call that rendered this text, '
        'or the URL has to be written as a markdown link in the source. '
        'Left bare, the line overflows the margin and the PDF prints a '
        'truncated URL that still looks like a whole one.')


def process(md: str) -> str:
    md = fix_status(md)
    parts = SPLIT_RE.split(md)
    out = []
    for part in parts:
        is_math = part.startswith('$') and part.endswith('$') and len(part) >= 2
        is_code = part.startswith('```') or (part.startswith('`') and part.endswith('`') and len(part) >= 2)
        if is_math:
            out.append(fix_math(part))
        else:
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

    check_supplement_flag(body, args.supplement, src_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = drop_working_sections(body)
    byline, addresses, keywords, body = split_front_matter(body, src_path)
    body = process_images(body, src_dir=src_path.parent, out_dir=out_path.parent)
    fixed = process(body)

    # The byline keeps its markers ("^a,\*^") but loses the bold: it is about
    # to be set as \author, which is already set apart from the body, and a
    # bold \author is the journal's decision and not this script's.
    byline_parts = FRONT_BYLINE_RE.fullmatch(byline)
    # The PDF's Author field gets the names without the affiliation markers.
    # It is read off the same byline rather than kept as a constant here: two
    # copies of an author list is how one of them ends up out of date.
    author_meta = byline_parts.group('names').strip()
    byline_md = byline_parts.group('names') + byline_parts.group('marks')
    fragments = [byline_md] + addresses + ([keywords] if keywords else [])
    rendered = render_fragments([process(f) for f in fragments])
    byline_tex, rest_tex = rendered[0], rendered[1:]
    keywords_tex = rest_tex.pop() if keywords else None

    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False, encoding='utf-8') as tmp:
        tmp.write(fixed)
        tmp_path = tmp.name

    # One header include carries everything injected into the preamble: the
    # box environment always, the supplementary float counters on request.
    # REVIEW_PREAMBLE is empty today; see the comment on it.
    with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False,
                                     encoding='utf-8') as hdr:
        hdr.write(BOX_PREAMBLE)
        hdr.write(URL_PREAMBLE)
        hdr.write(REVIEW_PREAMBLE)
        if args.supplement:
            # Renew both counters so a supplementary float prints as "Figure S1"
            # rather than "Figure 1", matching how the SI prose refers to them.
            hdr.write('\\renewcommand{\\thefigure}{S\\arabic{figure}}\n')
            hdr.write('\\renewcommand{\\thetable}{S\\arabic{table}}\n')

    # include-before-body lands immediately after \maketitle in pandoc's
    # template, which is where the address block belongs.
    with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False,
                                     encoding='utf-8') as front:
        front.write(title_block(rest_tex, keywords_tex))

    # pandoc writes to a scratch file, not to out_path: check_typesettable()
    # below aborts the run, and an abort that had already overwritten
    # out_path would leave a .tex on disk that has been through pandoc but
    # not through merge_captions/boxify -- a file that looks built, compiles,
    # and prints every figure caption twice. Nothing reaches out_path until
    # the gate has passed.
    with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False,
                                     encoding='utf-8') as raw:
        raw_path = raw.name

    subprocess.run([
        'pandoc', tmp_path,
        # autolink_bare_uris is what puts the reference list's URLs inside
        # \url{}. Without it pandoc emits them as ordinary prose, TeX has no
        # legal break point in a 90-character string with no spaces, and the
        # line runs off the right margin: the 2026-08-29 PDF printed
        # reference 7 as '...%EC%B4%88%C2%B7' and simply stopped, with five
        # MOE links clipped the same way. xurl is already in the preamble;
        # it only ever had bare text to work on. check_urls_wrapped() below
        # is the gate that says so if this flag is ever dropped.
        '-f', 'markdown+smart+raw_attribute+autolink_bare_uris',
        '-o', raw_path,
        '--standalone',
        '--pdf-engine=pdflatex',
        '-H', hdr.name,
        '-B', front.name,
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt',
        # -V, not -M: the byline is LaTeX (superscript affiliation markers)
        # and must reach \author unescaped. The plain-text twin goes to
        # metadata so the PDF's Author field is still a readable name.
        '-V', f'author={byline_tex}',
        '--toc=false',
        '-M', f'title={title}',
        '-M', f'author-meta={author_meta}',
        # British hyphenation: pandoc turns this into the `british` class
        # option that babel reads. House style is British English.
        '-M', 'lang=en-GB',
        '-M', 'date=',
    ], check=True)

    tex = boxify(merge_captions(Path(raw_path).read_text(encoding='utf-8'), out_path))
    check_typesettable(tex, out_path)
    check_urls_wrapped(tex, out_path)
    out_path.write_text(tex, encoding='utf-8')

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
