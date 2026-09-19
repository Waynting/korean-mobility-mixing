#!/usr/bin/env python3
"""Assemble one browsable folder holding every figure the manuscript uses,
each next to its own caption, the script that draws it, and the results file
its caption numbers come from.

WHY THIS IS GENERATED AND NOT WRITTEN BY HAND. A figure in this repo is spread
across four places: the drawing script (`eda/pNN_*.py`), the results file it
reads at draw time (`eda/results_pNN.json`), the image on disk (`eda/fig/`),
and the caption paragraph in the manuscript. Working on a figure means holding
all four at once, and there is nowhere that shows them together. A folder
assembled by hand would show them together exactly once -- until the next
caption edit, after which it would show a caption the paper no longer contains,
which is worse than no folder at all. So the sheet is a VIEW: every word of
every caption here is lifted verbatim from the manuscript at generation time,
and `--check` fails if the two have parted company.

WHERE TO EDIT WHAT. The sheet itself is never the place:

    the caption text  ->  paper/manuscript_JRSI.md   (then re-run this script)
    the figure        ->  eda/pNN_*.py               (then re-run that script)
    a number in a caption -> the script that writes eda/results_pNN.json

WHY THE IMAGES ARE SYMLINKS. `eda/fig/` is the canonical image tree and
`paper/latex/*/figures/` is already a copy of it, made for the Overleaf upload.
A third copy would be the copy nobody remembers to refresh, and this repo has
been bitten by precisely that once before (see `process_images` in
`manuscript_to_latex.py`). A symlink cannot go stale: re-running a drawing
script updates what the sheet shows, with no step in between.

THE MANUSCRIPT CITES THE SHEET, not `../eda/fig/` directly, so the folder you
edit in and the folder the paper reads are the same folder, and a figure cannot
be adjusted in one place while the paper quietly renders another. That makes the
sheet load-bearing, which is why `recover_image` exists below: deleting the
folder must not leave a manuscript whose images no longer resolve and
whose sheet can no longer be rebuilt to make them resolve again.

WHY IT IS NOT CALLED `paper/figures/` by default. That name is one letter of
context away from `paper/latex/JRSI/figures/`, the build output, and two
directories called `figures` with nothing to tell them apart is the incident
above. `figure_sheet` borrows the name `eda/README.md` already uses for the
`caption_numbers` blocks these pages render.

Usage:
    python3 scripts/figure_sheet.py --src paper/manuscript_JRSI.md
    python3 scripts/figure_sheet.py --src paper/manuscript_JRSI.md --check
    python3 scripts/figure_sheet.py --src paper/manuscript_JRSI_SI.md --out paper/figure_sheet_SI

`--out` defaults to the folder named after the manuscript itself:
`figure_sheet/` for `manuscript_JRSI.md`, `figure_sheet_SI/` for any `*_SI.md`.
That is what makes the destination DERIVED from `--src`, so forgetting the
argument cannot point a run at another manuscript's sheet. It was not always
true: until 2026-09-18 both defaulted to `figure_sheet/`, the docstring claimed
the guarantee anyway, and the SI's own generated README printed the run that
would have emptied the main sheet. `--out` can still name the wrong folder, so
`owner_of` reads the README already there and refuses a run whose `--src` is not
the one that wrote it.

THE SI HAS ITS OWN FOLDER. Both manuscripts
live in `paper/`, so both would default to the same `figure_sheet/`, and each
run deletes whatever the other wrote (see the stale-file sweep at the end of
`main`). Figure 7 moved into the SI as Figure S1 on 2026-09-12 and for three
days the main sheet still held `figure7.*` while the SI cited it; the first
main-manuscript rebuild would have deleted the SI's only image. The SI's sheet
is `paper/figure_sheet_SI/` and its image is `figureS1.png`, which
`recover_image` finds at `eda/fig/p55_figureS1.png` -- the drawing script names
its output by the figure number it carries, S included.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The manuscript writes a figure as an image paragraph whose alt text is the
# short title, followed by a blank line and the real caption paragraph, which
# repeats the number. Both halves are captured in one match, and the number is
# back-referenced rather than re-read: a caption that numbers itself differently
# from the image above it is the one error this file could otherwise carry into
# a sheet that looks right. `(?:(?!\n\n).)*` is `.*` that may not cross a blank
# line, so the caption stops at its own paragraph end instead of running on to
# the next figure.
FIGURE_RE = re.compile(
    r'^!\[Figure\s+(?P<num>S?\d+):\s*(?P<title>[^\]]*)\]\((?P<path>[^)\s]+)\)[ \t]*$\n'
    r'\n'
    r'(?P<caption>\*\*Figure\s+(?P=num)\.\s(?:(?!\n\n).)*)',
    re.MULTILINE | re.DOTALL,
)
# Every image paragraph that announces itself as a figure. Counted against the
# matches above so a figure whose caption is missing, misnumbered, or separated
# from its image by a stray line aborts the run instead of being dropped from a
# sheet that then silently has one page fewer than the manuscript cites.
FIGURE_IMAGE_RE = re.compile(r'^!\[Figure\s+(?P<num>S?\d+):', re.MULTILINE)
HEADING_RE = re.compile(r'^(#{2,4})\s+(?P<title>.+?)\s*$', re.MULTILINE)
# `p48_figure1.png` -> `p48`: the phase that drew it, which is also the name of
# its script and of its results file. Deriving the triple from the file name is
# what keeps this script free of a hand-written figure -> phase table that a new
# figure would silently fall out of.
PHASE_RE = re.compile(r'^(?P<phase>p\d+)_')


def generated_banner(src_rel: str) -> str:
    """The banner names the manuscript this sheet was lifted from. It used to
    hard-code `paper/manuscript_JRSI.md`, which was true until the SI got a
    sheet of its own (2026-09-15, `--out paper/figure_sheet_SI`): a page that
    tells you to edit the caption in the wrong file is the one kind of note
    this folder must not carry."""
    return ('<!-- 由 scripts/figure_sheet.py 產生，請勿手動編輯。 -->\n'
            f'<!-- 圖說要改：{src_rel}；圖要改：繪圖腳本；'
            '數字要改：寫出 results 檔的那支腳本。改完重跑本腳本。 -->\n')


def md_cell(value) -> str:
    """One results value as a markdown table cell."""
    if isinstance(value, bool):
        text = 'true' if value else 'false'
    elif isinstance(value, list):
        text = ', '.join(md_cell(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, float):
        # repr, not a rounded format: these are the numbers the caption quotes
        # and `p31_report_audit.py` gates, and a sheet that rounds them is a
        # sheet you cannot check a caption against.
        text = repr(value)
    else:
        text = str(value)
    return text.replace('|', r'\|').replace('\n', ' ')


def section_of(md: str, pos: int) -> str:
    """The nearest heading above `pos`, as it is written in the manuscript."""
    last = None
    for m in HEADING_RE.finditer(md, 0, pos):
        last = m
    return last.group('title') if last else '(no heading)'


def load_caption_numbers(results: Path, num: str):
    """The `(panel, what, value)` rows behind one figure's caption.

    Most figure phases store them under `caption_numbers`; where one script
    draws two figures (`p47` draws 3 and 4) each row also carries `figure`, and
    only the matching ones belong on this page. `p48` predates the convention
    and its whole file is the caption's number source, so it is rendered flat.
    """
    if not results.is_file():
        return None, 'results 檔不存在'
    data = json.loads(results.read_text(encoding='utf-8'))
    rows = data.get('caption_numbers')
    if rows is None:
        return [(None, k, v) for k, v in data.items()], 'flat'
    tagged = [r for r in rows if str(r.get('figure', num)) == num]
    return [(r.get('panel'), r.get('what'), r.get('value')) for r in tagged], 'caption_numbers'


def in_text_citations(md: str, num: str, skip_from: int, skip_to: int) -> list[int]:
    """Line numbers of the prose that refers to this figure by name, outside
    its own image line and caption. This is the reverse direction of the
    caption: a figure whose panels are re-cut has to answer to every sentence
    listed here, and those sentences are nowhere near the caption in the file."""
    # `(?!\d)` stops "Figure 1" from matching inside "Figure 12". It must not
    # also exclude "(", or the panel references the prose actually uses --
    # "Figure 3(c) shows the same absence" -- are the ones that go unlisted.
    pattern = re.compile(rf'\bFigure\s+{re.escape(num)}\b(?!\d)')
    lines = []
    for m in pattern.finditer(md):
        if skip_from <= m.start() < skip_to:
            continue
        lines.append(md.count('\n', 0, m.start()) + 1)
    return sorted(set(lines))


def recover_image(num: str, suffix: str) -> Path:
    """The canonical file behind Figure `num`, found without the sheet.

    Reached only when the manuscript cites a sheet link that is not there --
    the folder was deleted, or this is the first run after the references were
    pointed at it. Every drawing script names its output `p<phase>_figure<N>`,
    so the figure number in the manuscript is enough to find the file, and the
    glob is required to be unique rather than assumed to be: two candidates
    mean the naming convention has drifted, and picking one of them silently is
    how the sheet would come back rebuilt around the wrong image.
    """
    fig_dir = REPO_ROOT / 'eda' / 'fig'
    hits = sorted(fig_dir.glob(f'p*_figure{num}{suffix}'))
    if len(hits) != 1:
        raise SystemExit(
            f'Figure {num}: the sheet link is missing and the canonical file '
            f'cannot be recovered -- {len(hits)} file(s) match '
            f'eda/fig/p*_figure{num}{suffix}'
            + (f' ({", ".join(h.name for h in hits)})' if hits else '')
            + '\nDraw the figure, or point the manuscript back at the file '
              'under eda/fig/ by hand.')
    return hits[0].resolve()


def collect(src: Path, out_dir: Path):
    """Parse the manuscript into one record per figure."""
    md = src.read_text(encoding='utf-8')
    src_dir = src.parent
    matches = list(FIGURE_RE.finditer(md))
    announced = [m.group('num') for m in FIGURE_IMAGE_RE.finditer(md)]
    if len(matches) != len(announced):
        matched = {m.group('num') for m in matches}
        missing = [n for n in announced if n not in matched]
        raise SystemExit(
            f'{src}: {len(announced)} figure image(s), but only {len(matches)} '
            f'carry a caption paragraph directly underneath: {", ".join(missing)}\n'
            'A caption paragraph must follow its image after exactly one blank '
            'line and repeat the same number ("**Figure N. ..."). Left '
            'unmatched, the figure would simply be absent from the sheet, which '
            'is the one failure a generated view must not have.')

    figures = []
    for m in matches:
        num = m.group('num')
        rel = m.group('path')
        raw = src_dir / rel
        image = raw.resolve()
        if not image.is_file():
            if raw.parent.resolve() != out_dir:
                raise SystemExit(
                    f'{src}: Figure {num} points at a file that does not exist: '
                    f'{image}\n  reference: ' + m.group(0).splitlines()[0])
            image = recover_image(num, raw.suffix)
            print(f'圖 {num}:{rel} 不在,改用 {repo_rel(image)} 重建連結')
        phase_m = PHASE_RE.match(image.name)
        phase = phase_m.group('phase') if phase_m else None
        scripts = sorted((REPO_ROOT / 'eda').glob(f'{phase}_*.py')) if phase else []
        script = scripts[0] if len(scripts) == 1 else None
        results = REPO_ROOT / 'eda' / f'results_{phase}.json' if phase else None
        vector = image.with_suffix('.pdf')
        figures.append({
            'num': num,
            'title': m.group('title').strip(),
            'image': image,
            'vector': vector if vector.is_file() else None,
            'caption': m.group('caption').strip(),
            'caption_line': md.count('\n', 0, m.start('caption')) + 1,
            'image_line': md.count('\n', 0, m.start()) + 1,
            'section': section_of(md, m.start()),
            'phase': phase,
            'script': script,
            'script_takes_pdf': bool(
                script and '--pdf' in script.read_text(encoding='utf-8')),
            'results': results,
            'cited_on': in_text_citations(md, num, m.start(), m.end()),
        })
    return figures


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def redraw_command(fig) -> str:
    if fig['script'] is None:
        return f'(找不到 {fig["phase"]} 的唯一繪圖腳本)'
    cmd = f'.venv/bin/python {repo_rel(fig["script"])}'
    return cmd + (' --pdf' if fig['script_takes_pdf'] else '')


def render_page(fig, src: Path, link_name: str, vector_name) -> str:
    """One figure's page: what it is, where each part of it lives, the image,
    the caption verbatim, and the numbers the caption is allowed to quote."""
    src_rel = repo_rel(src)
    out = [f'# 圖 {fig["num"]} — {fig["title"]}', '', generated_banner(src_rel), '']
    rows = [
        ('所在小節', f'{fig["section"]}'),
        ('圖說出處', f'`{src_rel}:{fig["caption_line"]}`'),
        ('正文引用', ', '.join(f'`{src_rel}:{n}`' for n in fig['cited_on'])
                    or '（除圖說外，正文未提及）'),
        ('繪圖腳本', f'`{repo_rel(fig["script"])}`' if fig['script']
                    else f'（未定：eda/{fig["phase"]}_*.py 不是唯一一支）'),
        ('重繪指令', f'`{redraw_command(fig)}`'),
        ('數字來源', f'`{repo_rel(fig["results"])}`' if fig['results'] and fig['results'].is_file()
                    else '（無 results 檔）'),
        ('圖檔', f'`{link_name}` → `{repo_rel(fig["image"])}`（符號連結）'),
    ]
    if vector_name:
        rows.append(('向量版', f'`{vector_name}` → `{repo_rel(fig["vector"])}`（投稿不收 PDF，'
                              '此檔是同一次繪製的向量輸出，供檢視細節）'))
    out += ['| | |', '|---|---|']
    out += [f'| {k} | {v} |' for k, v in rows]
    out += ['', f'![Figure {fig["num"]}]({link_name})', '',
            '## 圖說（逐字取自手稿）', '', fig['caption'], '']

    if fig['results'] is not None:
        numbers, kind = load_caption_numbers(fig['results'], fig['num'])
        out += ['## 圖說裡的數字，以及它們的機器可讀來源', '']
        if numbers is None:
            out += [f'`{repo_rel(fig["results"])}` 不存在——先跑一次 '
                    f'`{redraw_command(fig)}`。', '']
        elif not numbers:
            out += [f'`{repo_rel(fig["results"])}` 裡沒有屬於圖 {fig["num"]} 的 '
                    'caption_numbers 列。', '']
        else:
            out += [f'共 {len(numbers)} 筆，取自 `{repo_rel(fig["results"])}`。'
                    '圖說改了數字而這裡沒改，就是圖說跟結果檔對不起來；'
                    '`eda/p31_report_audit.py --table` 是那道閘門。', '']
            has_panel = any(p is not None for p, _, _ in numbers)
            out += (['| 面板 | 是什麼 | 值 |', '|---|---|---|'] if has_panel
                    else ['| 是什麼 | 值 |', '|---|---|'])
            for panel, what, value in numbers:
                cells = ([md_cell(panel or '')] if has_panel else []) + [
                    md_cell(what), md_cell(value)]
                out += ['| ' + ' | '.join(cells) + ' |']
            out += ['']
    return '\n'.join(out).rstrip() + '\n'


def render_index(figures, src: Path, out_dir: Path) -> str:
    src_rel = repo_rel(src)
    # Print --out only when it is load-bearing: a reader who copies this line
    # must land back in this folder and not in the other manuscript's.
    out_arg = ('' if out_dir == default_out(src).resolve()
               else f' --out {repo_rel(out_dir)}')
    out = [f'# 圖與圖說總表 — {src_rel}', '', generated_banner(src_rel), '',
           f'`{src_rel}` 用到 {len(figures)} 張圖。每張圖一頁，頁面上有圖、'
           '逐字圖說、繪圖腳本、重繪指令，以及圖說裡每個數字的來源。', '',
           f'**手稿就是從這裡引用圖的**——`{src_rel}` 裡 {len(figures)} 個 `![...]()` 指的都是本'
           '資料夾的 `figureN.png`，而那些是指向 `eda/fig/` 的符號連結，所以重跑繪圖'
           '腳本，手稿看到的圖就跟著換，沒有中間那一步。', '',
           '**圖說不是**：它是產生當下從手稿逐字抄來的副本，改圖說要改手稿，'
           '不是改這裡。要改東西，改下表指的那個檔，然後重跑：', '',
           '```bash',
           f'python3 scripts/figure_sheet.py --src {src_rel}{out_arg}',
           f'python3 scripts/figure_sheet.py --src {src_rel}{out_arg} --check   '
           '# 只檢查：這裡的圖說是否還等於手稿裡的圖說',
           '```', '',
           '| 圖 | 頁面 | 小節 | 短標題 | 繪圖腳本 | 數字來源 |',
           '|---|---|---|---|---|---|']
    for fig in figures:
        page = f'figure{fig["num"]}.md'
        script = f'`{repo_rel(fig["script"])}`' if fig['script'] else '—'
        results = (f'`{repo_rel(fig["results"])}`'
                   if fig['results'] and fig['results'].is_file() else '—')
        out += [f'| {fig["num"]} | [{page}]({page}) | {fig["section"]} | '
                f'{md_cell(fig["title"])} | {script} | {results} |']
    # One command per SCRIPT, not per figure: `p47_fig34.py` draws figures 3
    # and 4 in a single run, and listing it twice reads as two runs to make.
    out += ['', '## 一次重畫全部', '', '```bash']
    drawn: dict[str, list[str]] = {}
    for fig in figures:
        drawn.setdefault(redraw_command(fig), []).append(fig['num'])
    for cmd, nums in drawn.items():
        out += [f'{cmd}   # 圖 {"、圖 ".join(nums)}']
    out += ['```', '',
            '重畫之後圖說裡的數字可能就跟著變了，所以接著要跑引用閘門：', '',
            '```bash',
            '.venv/bin/python eda/p31_report_audit.py --table',
            f'python3 scripts/figure_sheet.py --src {src_rel}',
            '```', '']
    return '\n'.join(out).rstrip() + '\n'


def build(src: Path, out_dir: Path):
    """The whole sheet as `{relative name: text}` plus the symlinks it needs,
    built in memory so `--check` and a real write share one code path."""
    figures = collect(src, out_dir)
    pages, links = {}, {}
    for fig in figures:
        link = f'figure{fig["num"]}{fig["image"].suffix}'
        links[link] = fig['image']
        vector = None
        if fig['vector'] is not None:
            vector = f'figure{fig["num"]}{fig["vector"].suffix}'
            links[vector] = fig['vector']
        pages[f'figure{fig["num"]}.md'] = render_page(fig, src, link, vector)
    pages['README.md'] = render_index(figures, src, out_dir)
    return figures, pages, links


def default_out(src: Path) -> Path:
    """The sheet folder that belongs to this manuscript.

    Derived from the manuscript's own name, not from a fixed string, so the
    SI and the main text cannot default to the same folder. `--out` overrides
    it; `owner_of` below is what stops an override from landing on the wrong
    manuscript's sheet.
    """
    suffix = '_SI' if src.stem.endswith('_SI') else ''
    return src.parent / f'figure_sheet{suffix}'


def owner_of(out_dir: Path):
    """Which manuscript last wrote this folder, read back from its README."""
    readme = out_dir / 'README.md'
    if not readme.is_file():
        return None
    m = re.search(r'--src (\S+)', readme.read_text(encoding='utf-8'))
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', required=True, type=Path,
                    help='the manuscript markdown to read figures from')
    ap.add_argument('--out', type=Path, default=None,
                    help='destination folder (default: figure_sheet/ beside --src)')
    ap.add_argument('--check', action='store_true',
                    help='do not write; exit non-zero if the sheet on disk no '
                         'longer matches the manuscript')
    args = ap.parse_args()

    src = args.src.resolve()
    if not src.is_file():
        raise SystemExit(f'no such manuscript: {src}')
    out_dir = (args.out or default_out(src)).resolve()

    # THE GUARD THAT THE DOCSTRING ABOVE PROMISES. The stale sweep at the end
    # of this function deletes every file in out_dir this run did not write, so
    # pointing a manuscript at another manuscript's sheet empties it. The
    # default cannot do that any more, but `--out` still can, and so can a
    # renamed manuscript. The folder says who owns it, so ask it.
    owner = owner_of(out_dir)
    if owner is not None and owner != repo_rel(src):
        raise SystemExit(
            f'{out_dir} is {owner}\'s sheet, not {repo_rel(src)}\'s.\n'
            f'  rebuilding it from {repo_rel(src)} would delete every page '
            f'{owner} cites.\n'
            f'  drop --out to use {repo_rel(default_out(src))}, or delete the '
            f'folder first if the rename is deliberate.')

    figures, pages, links = build(src, out_dir)

    if args.check:
        stale = []
        for name, text in pages.items():
            path = out_dir / name
            if not path.is_file():
                stale.append(f'{name}: 不存在')
            elif path.read_text(encoding='utf-8') != text:
                stale.append(f'{name}: 與手稿不一致')
        for name, target in links.items():
            path = out_dir / name
            if not path.is_symlink() or path.resolve() != target:
                stale.append(f'{name}: 連結不存在或指錯地方')
        extra = sorted(p.name for p in out_dir.iterdir()
                       if p.name not in pages and p.name not in links
                       ) if out_dir.is_dir() else []
        stale += [f'{name}: 多出來的檔案,手稿沒有用到' for name in extra]
        if stale:
            print(f'{out_dir} 與 {repo_rel(src)} 對不起來:')
            for line in stale:
                print(f'  {line}')
            print(f'重跑 python3 scripts/figure_sheet.py --src {repo_rel(src)}')
            return 1
        print(f'{out_dir}: {len(figures)} 張圖,圖說與手稿逐字相符')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        (out_dir / name).write_text(text, encoding='utf-8')
    for name, target in links.items():
        path = out_dir / name
        if path.is_symlink() or path.exists():
            path.unlink()
        path.symlink_to(os.path.relpath(target, out_dir))
    # Anything this run did not write is a figure the manuscript dropped or
    # renumbered. It is deleted, not left: a page for a figure that no longer
    # exists is the sheet telling you to go edit something the paper does not
    # contain, and it would sit there looking exactly as authoritative as the
    # real ones. This is the sweep `owner_of` guards: it is scoped to out_dir,
    # which is the whole danger when out_dir is the wrong manuscript's.
    keep = set(pages) | set(links)
    stale = sorted(p for p in out_dir.iterdir() if p.name not in keep)
    for path in stale:
        path.unlink()
    print(f'{out_dir}: {len(figures)} 張圖,{len(pages)} 個 md 檔,'
          f'{len(links)} 個符號連結'
          + (f';刪掉 {len(stale)} 個已不再引用的檔案 '
             f'({", ".join(p.name for p in stale)})' if stale else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
