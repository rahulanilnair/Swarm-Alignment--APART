#!/bin/sh
# Reproduce the paper from a clean clone: committed files only, a fresh uv environment, no model calls.
# Runs the tests, rebuilds every number, table, figure and the PDF, and checks that numbers.tex and the
# generated tables match the committed ones byte for byte.
#   scripts/repro_check.sh [repo-url-or-path] [branch]
set -eu
SRC=${1:-$(cd "$(dirname "$0")/.." && pwd)}
BRANCH=${2:-andrii/integration}
TMP=$(mktemp -d)
git clone -q --no-local --depth 1 -b "$BRANCH" "file://$SRC" "$TMP/repo" 2>/dev/null || git clone -q --depth 1 -b "$BRANCH" "$SRC" "$TMP/repo"
cd "$TMP/repo"
echo "clean clone at $(git log -1 --format='%h %s') in $TMP/repo"
make test
make paper > "$TMP/make_paper.log" 2>&1 || { tail -30 "$TMP/make_paper.log"; exit 1; }
if git diff --quiet -- manuscript/numbers.tex manuscript/tables; then
  echo "REPRODUCED: numbers.tex and every generated table match the committed ones"
else
  git diff --stat -- manuscript/numbers.tex manuscript/tables
  git diff -- manuscript/numbers.tex | head -40
  exit 1
fi
