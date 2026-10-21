#!/bin/bash

# Check that the momentum solver still reproduces the golden output exactly.
#
# Regenerates every case with scripts/make_momentum_golden.sh into a temporary
# directory and requires each grid (u, v, k, vel, ang) to be byte-identical to
# autotest/golden/momentum. Use it to verify refactors that must not change
# behaviour. Golden output from a different OpenFOAM build or platform is not
# expected to match byte for byte.
#
# Usage:
#   source <openfoam>/etc/bashrc
#   scripts/check_momentum_golden.sh <path/to/WindNinja_cli>

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GOLDEN_DIR="$(dirname "$SCRIPT_DIR")/autotest/golden/momentum"

if [ $# -ne 1 ]; then
    echo "usage: $0 <path/to/WindNinja_cli>" 1>&2
    exit 2
fi

OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

"$SCRIPT_DIR/make_momentum_golden.sh" "$1" "$OUT_DIR"

status=0
for case_dir in "$GOLDEN_DIR"/*/; do
    name="$(basename "$case_dir")"
    for grid in u v k vel ang; do
        if ! cmp -s "$case_dir/$grid.asc" "$OUT_DIR/$name/$grid.asc"; then
            echo "MISMATCH: $name/$grid.asc"
            status=1
        fi
    done
done

if [ $status -eq 0 ]; then
    echo "All golden cases match."
fi
exit $status
