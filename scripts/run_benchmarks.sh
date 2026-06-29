#!/usr/bin/env bash
# ============================================================================
#  Run every optimization step for a given compiler and collect the average
#  GFLOP/s into a CSV. Repeat for gcc / clang / icx to reproduce the study.
#
#  Usage:
#     ./scripts/run_benchmarks.sh [CC] [N] [THREADS]
#     ./scripts/run_benchmarks.sh gcc 16384 8
#
#  Output: benchmarks/results_<CC>.csv
# ============================================================================
set -euo pipefail

CC="${1:-gcc}"
N="${2:-16384}"
THREADS="${3:-$(nproc 2>/dev/null || echo 4)}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="benchmarks/results_${CC}.csv"
mkdir -p benchmarks

echo "Building all steps with CC=$CC ..."
make CC="$CC" >/dev/null

echo "step,gflops" > "$OUT"
echo "Running N=$N, THREADS=$THREADS, CC=$CC"
echo "--------------------------------------------"

for bin in bin/*_"$CC"; do
  [ -x "$bin" ] || continue
  step="$(basename "$bin" "_$CC")"
  # The average GFLOP/s is the number right before "+-" on the summary line.
  # Strip ANSI color codes first, then take the field preceding "+-".
  raw="$(OMP_NUM_THREADS="$THREADS" "$bin" "$N" 2>/dev/null || true)"
  gflops="$(printf '%s\n' "$raw" \
            | sed -r 's/\x1b\[[0-9;]*m//g' \
            | awk '/Average performance/ { for (i=1;i<=NF;i++) if ($i=="+-") { print $(i-1); exit } }')"
  gflops="${gflops:-NA}"
  printf "%-22s %s GFLOP/s\n" "$step" "$gflops"
  echo "${step},${gflops}" >> "$OUT"
done

echo "--------------------------------------------"
echo "Saved -> $OUT"
