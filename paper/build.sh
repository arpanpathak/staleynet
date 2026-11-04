#!/usr/bin/env bash
# Builds both versions of the paper: preprint.pdf (single column) and ieee.pdf (two columns).
set -euo pipefail
cd "$(dirname "$0")"
for name in preprint ieee; do
  pdflatex -interaction=nonstopmode -halt-on-error "$name" > "$name.build.log"
  bibtex "$name" >> "$name.build.log"
  pdflatex -interaction=nonstopmode -halt-on-error "$name" >> "$name.build.log"
  pdflatex -interaction=nonstopmode -halt-on-error "$name" >> "$name.build.log"
  echo "$name.pdf: $(grep -c 'Warning' "$name.log" || true) warnings"
done
