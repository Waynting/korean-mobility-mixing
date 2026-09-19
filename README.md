# Age mixing in an aggregated mobility product: analysis code and reproduction harness

This archive holds the code behind a measurement of how much age-mixing structure survives
in Seoul's *Saenghwal Idong* (Living Migration) administrative-dong release: 79 consecutive
months (202001–202607), 424 administrative dong, 16 age bands, 10,186,891,962 rows.

The answer is that almost none survives, and the more useful part is that we now know how
little "almost none" is. Across all 79 months the best rank-one approximation of the measured
co-arrival matrix departs from the outer product of its own margins by at most
0.0258 in relative Frobenius norm (0.0209 at the median). Forcing those margins onto a contact survey's leaves a factor of 33 to 35. On
synthetic worlds whose answer is known the same pipeline recovers 92% to 101% of true age
assortativity when none of it lies below the spatial cell, and recovery falls as one minus the
hidden share.

**This archive is code, not data.** Neither the Seoul mobility product nor the Korean contact
survey is redistributed here. See `EXCLUSIONS.md`, which is an audit of what is held back and
why, not a disclaimer.

## Where this lives

The archive of record is Zenodo, concept DOI
[10.5281/zenodo.22152089](https://doi.org/10.5281/zenodo.22152089), which always resolves to
the latest published version; the manuscript cites that DOI and names the version it
corresponds to. This same tree is mirrored at <https://github.com/Waynting/korean-mobility-mixing>, where `main` is the working state and a
tag marks each published version.

Neither copy is assembled by hand. Both come out of `scripts/build_zenodo_deposit.sh`, which
takes the source repository's tracked files, subtracts one exclusion list, and adds the front
matter you are reading. `EXCLUDED_FILES.txt` lists every path that list withheld, so what is
missing can be seen without asking for it.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Point at wherever the source files live. The tree is ~9 GB of derived files
# on top of ~42 GB of parquet, so it is normally an external drive.
export KOREAN_DATA_ROOT=/Volumes/YOUR_DRIVE/Korean

.venv/bin/python eda/paths.py          # prints what resolves where, and what is missing
```

`eda/paths.py` is the only place that knows where data lives, and `require()` aborts when the
drive is not mounted. That abort is deliberate: an unmounted drive reading as an empty dataset
is the worst failure mode here, because the output still looks right.

## Getting the source data

Nothing in this archive contains the Seoul product. `eda/dl_mobility.py` retrieves it from the
Seoul Open Data Plaza and `raw/mobility/manifest_csv.json` carries a per-file SHA-256 for all
79 months (1,896 files, 420.9 GB expanded) so an independently retrieved copy can be checked
byte for byte against ours.

**The downloaders verify content, not HTTP status.** The Korean government portals return
session-timeout HTML with a `200` and a `Content-Disposition` header, so every `dl_*.py`
supports `--verify` and stores SHA-256 in a manifest. The same failure mode bites the
reference URLs in the paper: a Ministry of Education press-release link answers `200` with a
"page not found" body unless `&lev=0` is present.

One gap is permanent and is stated rather than hidden: the zip-level checksums cover
202101–202607 only. The 2020 zips were never kept, because the downloader hashed and deleted
them and that batch predates the downloader. The CSV-level manifest covers all 79 months.

## How to run it

Scripts run one at a time: `.venv/bin/python eda/pNN_name.py [args]`. **The run order, the
per-script cost, the arguments that are not optional, and the anchors each script asserts are
all in `eda/README.md`** — read the relevant entry before running a phase script. Several take
arguments that change *which question is being answered*; the published values are pinned in
`eda/determinism_check.sh`.

`eda/README.md`, `eda/memo/*.md` and the phase memos are written in Chinese; code, docstrings
and comments are in English.

## The unit of work is a phase

```
eda/pNN_name.py  →  eda/results_pNN.json  →  eda/memo/phaseNN-name.md   (+ eda/fig/pNN_*.png)
```

Phases are numbered roughly in the order they were run, not by topic. `dl_*.py` are
downloaders; `etl.py` turns CP949 CSV into the parquet tree, checking every file's row count
against `wc -l`.

The shared layer is three files. `eda/paths.py` is the only place that knows where data lives.
`eda/common.py` builds a DuckDB view over the parquet joined to the calendar; note that
`이동인구(합)` is a monthly *sum* over occurrences of that weekday, so nothing may be compared
across months or weekdays without dividing by the occurrence count. `eda/calendar_kr.py`
computes holidays and policy dose from the real calendar rather than hard-coding them.

**Anchors are the central mechanism.** Most phase scripts open by asserting they can reproduce
an upstream published number bit-for-bit, and abort if they cannot. These asserts have caught
real bugs. A firing anchor is a finding, not an obstacle to route around.

## How this archive checks itself

Three commands stand in for a test suite. Each answers a different question and none
substitutes for another.

```bash
.venv/bin/python eda/p31_report_audit.py --table   # do the documents quote the numbers the results files hold?
eda/determinism_check.sh                           # does a script give the same answer twice?
.venv/bin/python eda/p36_recompute.py              # is the answer right?
```

**Which of them run in this version of the archive.** `determinism_check.sh` and
`p36_recompute.py` run once you have fetched the source data: they rebuild everything they
need. `p31_report_audit.py` does **not** run here, and the reason is structural rather than a
missing file. It compares what a document quotes against what a results file holds, and this
archive deposits neither side: the manuscript and the correspondence are not part of a code
deposit, and the results files are held back by the licence gate in `EXCLUSIONS.md` §2. It is
shipped anyway, because it is the readable specification of every claim the paper makes and of
where each one comes from. When the gate lifts and the results files are added, it runs.

- **`p31_report_audit.py`** recomputes each quoted claim from the results file that owns it,
  checks the reverse direction as well (values verified but quoted nowhere), and is
  mutation-tested: values are deliberately perturbed and the gate must go red.
- **`determinism_check.sh`** runs a script twice and compares SHA-256. It exists because one
  phase was once irreproducible: its seed was built from `hash(str)`, which Python
  re-randomises per process.
- **`p36_recompute.py`** is an independent second implementation, not a gate. It rebuilds the
  headline quantities by different routes (DuckDB self-join, the `holidays` package,
  `csv` plus dict, power iteration) and agrees on 692 of 692 checks. It pins `threads=1`,
  because DuckDB's parallel hash aggregate merges partial sums in thread completion order and
  float addition is not associative.

## Invariants worth knowing before you edit anything

- **Results files are produced, never edited.** A wrong number inside a `results_*.json` is
  fixed in the script that wrote it, followed by a re-run.
- **Missing rows mean "no such observation", not zero.** 99.03% of the logical grid has no
  row; filling it out manufactures 99% zeros.
- **Dedup is per-path, not blanket.** 475,266 duplicate key groups exist in the raw CSV. Cell
  counts and matrices deduplicate; volume reconciliation against the official district file
  does not, and an unconditional `DISTINCT` degrades that reconciliation from +4.19e-06 to
  −1.52e-04 and flips its sign.
- **Never index months positionally.** Index by month label. One script once reported August's
  residual as September's after June was inserted.
- **Romanise Korean on figures.** Hangul renders as tofu in every font matplotlib ships, and a
  figure needing a locally installed CJK font breaks on the typesetter's machine.
- `eda/p5_baseline.py` is broken (it imports names that moved to `calendar_kr.py`), and
  `p4/p5/p6/p6b` use a superseded two-month definition. Their conclusions were replaced by
  `p8`/`p10`; they are kept for traceability.

## Licensing

Code is MIT (`LICENSE`). Derived data, when a version of this archive carries it, is
CC BY 4.0 (`LICENSE-DATA`) with attribution to the Seoul Metropolitan Government. Reuse of
derived aggregates must credit both this deposit and the upstream Seoul release.

## Citing

See `CITATION.cff`. Cite the **concept DOI**, which always resolves to the latest version.

The paper this archive accompanies is Liu W-T and Lee H-W, *Proportionate mixing is nearly exact in contact matrices derived from aggregated mobility data* (2026). It has
no DOI yet; when it does, the Zenodo record will carry it as a related identifier, and the
citation here will name it.
