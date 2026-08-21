#!/usr/bin/env bash
# Direct compilation of the report (no latexmk / no Perl), all artifacts in build/.
# Sequence: pdflatex -> biber -> pdflatex -> pdflatex
# Usage: ./build.sh [clean]
set -e
export PATH="$HOME/.local/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
cd "$(dirname "$0")"

OUT=build

if [ "${1:-}" = "clean" ]; then
  rm -rf "$OUT"
  echo "cleaned."
  exit 0
fi

# -output-directory does not create subdirs; \include writes per-file .aux that
# mirror the source tree, so pre-create build/ and its sections/ subtree.
mkdir -p "$OUT"
find sections -type d -exec mkdir -p "$OUT/{}" \;

# TEXMFOUTPUT lets TeX read back generated files (.aux/.toc/.bbl) from build/.
export TEXMFOUTPUT="$OUT"

PDFLATEX="pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory=$OUT"
$PDFLATEX main
# biber's --output-directory is used for BOTH reading main.bcf and writing output,
# so the jobname must be bare `main` (passing "$OUT/main" doubles it to build/build/main).
biber --output-directory="$OUT" main
$PDFLATEX main
$PDFLATEX main
echo "built $OUT/main.pdf"
