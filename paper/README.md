# StaleyNET paper

One text, two builds:

| File | Format | Purpose |
|---|---|---|
| `preprint.pdf` | single column, 11 pt | arXiv preprint and application material |
| `ieee.pdf` | IEEE conference, two columns | starting point for a conference version |

The content is in `body.tex`; `macros.tex` defines the pseudocode, definition
and theorem environments shared by both; `references.bib` holds the
bibliography. Build both with `./build.sh` (pdflatex and bibtex).

## Choosing a venue

Column format is set by the venue, not by the paper. Preprints and technical
reports are usually single-column. Conferences supply their own templates, and
their page limits change from year to year, so the current call for papers is
the authority. At the time of writing:

* ISMIR (music information retrieval) uses its own two-column template with a
  limit of about 6 pages plus references, under double-blind review.
* ICASSP uses the IEEE two-column format with a limit of about 4 pages plus 1
  page of references.
* NeurIPS workshops on audio use the single-column NeurIPS style.

A conference version is a shortened selection of this text, in the venue's
official template, with author names removed for double-blind review.
