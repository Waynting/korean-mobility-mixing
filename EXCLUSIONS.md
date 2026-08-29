# What is not in this archive, and why

This is an audit, not a disclaimer. Every entry below exists in the working repository and was
deliberately kept out. `EXCLUDED_FILES.txt`, generated at build time, lists the exact paths.

## 1. The source data, because it is not ours to redistribute

**The Seoul Living Migration (생활이동) administrative-dong product.** 79 monthly releases,
roughly 77 GB of source zips expanding to about 460 GB of CSV, plus the 42 GB parquet tree
built from them and 9.2 GB of derived intermediates. It is published by the Seoul Metropolitan
Government through the Seoul Open Data Plaza under its own terms. What ships instead is
`eda/dl_mobility.py`, `eda/dl_mobility_fill.sh` and a per-file SHA-256 manifest covering all
1,896 files across all 79 months, so an independently retrieved copy can be checked byte for
byte against the one these results were computed from.

One gap in that chain is permanent and is worth stating plainly rather than burying: the
**zip-level** checksums cover 202101–202607 only. The twelve 2020 zips were never kept,
because the downloader hashes and deletes the zip after expanding it and that batch predates
the downloader; re-fetching them is blocked by TLS/Range failures on the portal. The
**CSV-level** manifest covers all 79 months, and for the 67 months that do have zip records the
expanded byte counts agree month by month with zero mismatches. So the archive can prove that
your CSVs are byte-identical to ours; it cannot prove the 2020 chain of custody.

**The Korean nationwide contact survey.** Used from its own public, de-identified release
(doi:10.1038/s41597-026-06896-y), which carries its own ethical approval. Not one byte of it is
reproduced here.

## 2. The derived aggregates, because a licence question is still open

**Status as of the build date: withheld.** The per-phase results files (`eda/results_*.json`),
which carry the 79-month age-by-age matrices, the coverage profile and the measured masking
fill, are not in this version.

They are held back by two gates, and the second one is easy to miss:

1. **The Seoul terms.** A written enquiry was sent to the Seoul Open Data Plaza on 2026-08-24
   asking two questions: whether this product's 이용허락범위 is the same 공공누리 제1유형 as the
   catalogued release, and whether aggregates computed from it may be released under CC BY with
   attribution while the source files are not redistributed. The product page for the
   administrative-dong release carries no licence badge of its own, which is why the question
   had to be asked rather than assumed. No reply had arrived when this version was built.

2. **The other three agencies.** The resident-registration, registered-foreigner and covariate
   files come from 행정안전부, 법무부 and SGIS. They are not under the Seoul terms at all, so a
   favourable answer to (1) would not release them. Their conditions have not been checked one
   by one.

Neither gate affects the analysis or this archive's ability to reproduce it: every derived file
is rebuilt from the source data by the code that is here. What the gates affect is whether the
already-computed aggregates may be *shipped*, which would save a reader the compute rather than
enable anything new.

When both are settled, the aggregates are added as a new version under CC BY 4.0. The concept
DOI follows automatically, so nothing that cites this archive needs to change.

## 3. The manuscript and its correspondence

The article, its supplement, the figure sources, the planning documents and the advisor
correspondence are not deposited. This archive is the code and the harness; the article is
published separately and cites this archive by its concept DOI.

Two files are worth naming explicitly because their presence in the working repository would
otherwise be confusing:

- `archive/manuscript_JBI.md` is **a different paper entirely** — a reinforcement-learning
  screening protocol for Brazilian arboviral surveillance. It shares an author and nothing
  else.
- `archive/` generally holds superseded drafts, kept so a claim's history can be traced. None
  of it is authoritative and none of it is here.

## 4. KDCA daily case counts

`cases/` holds daily confirmed-case counts used in an exploratory phase. The upstream licence
has not been checked, so it is withheld under the same rule as everything else in this file:
not redistributed until the terms are known. No result in the paper depends on it.

## 5. Machine-local and generated files

`.env` (paths differ per machine; `.env.example` is included), the virtualenv, caches, LaTeX
build artefacts, and the scratch files the determinism harness writes. None carries
information.

## What this means for reproducing the results

Everything needed is here. Fetch the source data with the download helpers, verify it against
the manifest, and run the phases in the order `eda/README.md` gives. Most phase scripts open by
asserting they can reproduce an upstream published number bit-for-bit and abort if they cannot,
so a divergence announces itself at the phase where it starts rather than at the end.
