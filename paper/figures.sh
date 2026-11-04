#!/usr/bin/env bash
# Compiles each TikZ figure of the paper on its own and converts it to SVG for
# the book, so that the paper and the book draw every diagram from one source.
set -euo pipefail
cd "$(dirname "$0")"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for source in figures/*.tex; do
  name=$(basename "$source" .tex)
  cat > "$work/$name.tex" <<TEX
\documentclass[border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\input{$PWD/macros}
\begin{document}
\input{$PWD/figures/$name}
\end{document}
TEX
  (cd "$work" && pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" > "$name.log") || { echo "figure $name failed"; tail -20 "$work/$name.log"; exit 1; }
  pdftocairo -svg "$work/$name.pdf" "../book/src/figures/paper-$name.svg"
  echo "book/src/figures/paper-$name.svg"
done
