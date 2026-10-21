#!/bin/bash

# Run one golden case on the WebGPU momentum backend and compare it with the
# OpenFOAM golden output against GATE 0's tolerances (WEBGPU_PLAN).
#
# Usage: scripts/check_momentum_gpu_golden.sh <WindNinja_cli> <compare_momentum> <case>
# Exit: 0 within tolerance, 1 outside it or on error, 77 (ctest skip) when
# this machine has no WebGPU adapter.

set -u

if [ $# -ne 3 ]; then
    echo "usage: $0 <WindNinja_cli> <compare_momentum> <case>" 1>&2
    exit 2
fi
CLI="$1"
COMPARE="$2"
CASE="$3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GOLDEN="$(dirname "$SCRIPT_DIR")/autotest/golden/momentum"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# GATE 0 (provisional): speed RMSE m/s, direction deg, k relative RMSE
MAX_SPEED_RMSE=0.25
MAX_DIRECTION=5
MAX_K_RMSE=0.25

if ! MOMENTUM_BACKEND=webgpu MOMENTUM_CASES="^$CASE\$" \
     "$SCRIPT_DIR/make_momentum_golden.sh" "$CLI" "$OUT" > "$OUT/log" 2>&1; then
    if grep -q "no adapter" "$OUT/log"; then
        echo "SKIP: no WebGPU adapter"
        exit 77
    fi
    cat "$OUT/log"
    exit 1
fi
grep "RANS" "$OUT/$CASE/metadata.txt"
"$COMPARE" "$OUT/$CASE" "$GOLDEN/$CASE" $MAX_SPEED_RMSE $MAX_DIRECTION $MAX_K_RMSE
