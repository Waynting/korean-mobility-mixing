"""The three conventions every submission figure shares, in one place.

WHY THIS FILE EXISTS. Seven draw scripts each carried their own copy of "7.00
in wide, 7.5 pt floor, save at 300 dpi", and each copy was a place the rule
could drift. Two of them already had: p55 saved with `bbox_inches="tight"` and
came out 6.907 in rather than 7.00, and all seven emitted RGBA. A convention
that is restated seven times is not a convention, it is seven literals.

WIDTH IS 6.50 IN, NOT 7.00, AND THAT IS THE POINT OF THIS ROUND.
`paper/latex/JRSI/main.log` reports \\textwidth = 469.75502 pt = 6.500 in.
`\\pandocbounded` scales an over-wide graphic down to \\linewidth and leaves a
narrower one alone, so a 7.00 in figure arrived on the page at 0.9286 and the
7.5 pt floor every script was written to printed as 6.96 pt -- under JRSI's
absolute 7.5 pt minimum, measured at 323 ppi in the submitted PDF rather than
inferred. Drawing at 6.50 in makes the scale factor exactly 1.000, so the size
written in the script is the size that prints. Raising the floor to 8.1 pt
instead (8.1 x 0.9286 = 7.52) would compensate for THIS main.tex's text width
and no other: JRSI's own measure is about 175 mm, and a figure the editor sets
single-column is scaled again. The type size is what the rule constrains; the
width is free, so the width is what moves.

The cost is real and is paid in area, not in legibility: every figure loses 7%
of its drawing width, and the layout constants in each script -- which are in
INCHES, deliberately, because what they reserve is a text height -- were
re-checked against the narrower page rather than scaled.

PANEL LETTERS. JRSI asks for "Alphabetical labels italicized in roman brackets,
eg '(a)'": an italic letter inside upright brackets, and never bold. Mathtext
sets a bare single letter in italic, so "($a$)" is the whole recipe -- the
brackets sit outside the maths and stay roman.

FLATTENED OUTPUT. JRSI asks for figures as a "single flattened layer".
matplotlib writes RGBA, so pdfTeX emitted a soft mask beside every one of the
seven images. Alpha is also the usual place a CMYK conversion goes wrong, with
transparent pixels filling black rather than white. `finish` composites onto
white and writes RGB, which is what the page shows anyway.

THE FLOOR IS NOW CHECKED, NOT ONLY DECLARED. `FLOOR_PT = 7.5` sat in this file
from 2026-08-24 with the comment "nothing in any script may go under it" and no
code anywhere that looked: `grep -rn FLOOR_PT eda/` returned the definition line
and nothing else. Six strings in `p63_fig2.py` were under it the whole time, at
7.2, 7.3 and 7.4 pt. A constant that only documents a rule is a comment with a
type. `finish` enforces it on every figure it writes, which is every submission
figure, because `finish` is the one door they all leave by.
"""
import os

from matplotlib.text import Text
from PIL import Image

# The JRSI text width, in inches, measured from main.log rather than assumed.
WIDTH = 6.50
# JRSI refuses figure text below this. Nothing in any script may go under it.
FLOOR_PT = 7.5
DPI = 300


def plabel(letter, text=""):
    """'(a)' as JRSI sets it: italic letter, roman brackets, never bold."""
    head = f"(${letter}$)"
    return f"{head}  {text}" if text else head


def _flatten(path):
    """Composite the PNG onto white and rewrite it as RGB at the same dpi.

    The dpi has to be restated because PIL does not carry the pHYs chunk across
    a convert(); 300 dpi is stored as 11811 pixels per metre either way, so the
    file records 299.9994 before and after and the figure is unmoved.
    """
    with Image.open(path) as im:
        if im.mode != "RGBA":
            return im.mode
        flat = Image.new("RGB", im.size, "white")
        flat.paste(im, mask=im.split()[3])
    flat.save(path, dpi=(DPI, DPI))
    return "RGB"


def _axname(i, ax):
    """Name an axes the way the person fixing it would look for it."""
    for s in (ax.get_title(loc="left"), ax.get_title(), ax.get_xlabel(),
              ax.get_ylabel()):
        if s and s.strip():
            return f"axes {i + 1}, {s.strip().splitlines()[0]!r}"
    return f"axes {i + 1}"


def _under_floor(fig):
    """Every visible string set below FLOOR_PT, as {(pt, what): count}.

    TWO PASSES, because neither is complete on its own. `findobj(Text)` reaches
    titles, axis labels, legend entries, annotations, offset text and every
    `ax.text`/`ax.annotate` -- everything whose string is known before the draw,
    which is where the sizes passed inline at the call site live. Tick labels
    are the exception: `tick_params(labelsize=...)` sets the size on them at
    once, but the STRING is filled in by the formatter at draw time, so before
    the draw they read as empty and the first pass steps over them. The second
    pass therefore walks the ticks themselves and names them by axis rather than
    by content -- which is how `p63`'s resolution strip hid an x-axis at 7.3 pt.

    Checked BEFORE the draw on purpose. A figure that breaks the floor should
    never reach the disk: a too-small PNG is not visibly wrong, and the next
    person to look at the directory would find a file that looks finished.

    A size in points is a size on the page only because `finish` has already
    asserted the figure is exactly WIDTH in wide, so `\\pandocbounded` leaves it
    at a scale factor of 1.000. The width assert and this one are the same rule
    read from two ends, which is why they live in one function.
    """
    bad, skip = {}, set()

    def record(pt, what):
        if pt < FLOOR_PT - 1e-9:
            key = (round(float(pt), 3), what)
            bad[key] = bad.get(key, 0) + 1

    # Ticks belonging to a switched-off axes (`ax.axis("off")`, the usual way a
    # text-only strip is made) are not on the page; their own get_visible()
    # does not know that, so they are collected here and stepped over below.
    for ax in fig.axes:
        if not ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                    skip.update((id(tk.label1), id(tk.label2)))

    for t in fig.findobj(Text):
        if id(t) in skip or not t.get_visible():
            continue
        s = (t.get_text() or "").strip()
        if s:
            record(t.get_fontsize(), repr(s.replace("\n", " ")))

    for i, ax in enumerate(fig.axes):
        if not ax.axison:
            continue
        for nm, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                for lb in (tk.label1, tk.label2):
                    if lb.get_visible() and not (lb.get_text() or "").strip():
                        record(lb.get_fontsize(),
                               f"<{nm} tick labels, {_axname(i, ax)}>")
    return bad


def finish(fig, png, pdf=False, **savekw):
    """Write the PNG (flattened) and, on request, the byte-stable PDF.

    CreationDate is omitted from the PDF on purpose: matplotlib stamps the wall
    clock into it, which makes two runs of an unchanged figure differ by bytes
    for a reason that has nothing to do with the figure.
    """
    w = fig.get_size_inches()[0]
    assert abs(w - WIDTH) < 1e-9, \
        f"{os.path.basename(png)} is {w:.3f} in wide, not the {WIDTH} in " \
        f"JRSI reproduces at; at any other width the type size on the page " \
        f"is not the type size in this script"
    small = _under_floor(fig)
    listing = "\n".join(
        f"    {pt:5.2f} pt   {what}" + (f"   (x{n})" if n > 1 else "")
        for (pt, what), n in sorted(small.items()))
    assert not small, \
        f"{os.path.basename(png)}: {sum(small.values())} string(s) are set " \
        f"below JRSI's absolute {FLOOR_PT} pt floor. The figure is exactly " \
        f"{WIDTH} in wide, so the scale factor is 1.000 and these ARE the " \
        f"printed sizes:\n{listing}\n" \
        f"  Raise each to {FLOOR_PT} pt or more. If that makes text collide, " \
        f"move the layout -- the floor is JRSI's, not this script's, and " \
        f"nothing was written at 7.2 pt because 7.2 pt was legible."
    fig.savefig(png, dpi=DPI, **savekw)
    mode = _flatten(png)
    if pdf:
        fig.savefig(png[:-4] + ".pdf", metadata={"CreationDate": None},
                    **savekw)
    return mode
