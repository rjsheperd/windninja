#!/bin/bash

# The mass-mesh outputs on the WebGPU momentum backend (WEBGPU_PLAN 5.2):
# Big Butte coarse with write_vtk_output, write_goog_output and
# WRITE_TURBULENCE must write the VTK volume, the colMax GeoTIFF and a KMZ
# with the colMax layer, as the OpenFOAM backend does.
#
# Usage: scripts/check_momentum_gpu_outputs.sh <WindNinja_cli>
# Exit: 0 pass, 1 fail, 77 (ctest skip) without a WebGPU adapter.

set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <WindNinja_cli>" 1>&2
    exit 2
fi
CLI="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WN_SRC_DIR="$(dirname "$SCRIPT_DIR")"
export WINDNINJA_DATA="$WN_SRC_DIR/data"
export NINJA_MOMENTUM_FALLBACK=NO   # a GPU test: no fallback to the CPU backend
export WRITE_TURBULENCE=TRUE
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

cp "$WINDNINJA_DATA/big_butte_small.tif" "$OUT/"
sed 's|^elevation_file.*|elevation_file = big_butte_small.tif|' \
    "$WN_SRC_DIR/autotest/config/momentum/momentum_webgpu_bigbutte.cfg" > "$OUT/outputs.cfg"
printf "write_vtk_output = true\nwrite_goog_output = true\n" >> "$OUT/outputs.cfg"

(cd "$OUT" && "$CLI" outputs.cfg > log.cli 2>&1)
status=$?
if grep -q "no adapter" "$OUT/log.cli"; then
    echo "SKIP: no WebGPU adapter"
    exit 77
fi
fail=0
[ $status -eq 0 ] || { echo "WindNinja_cli exited with $status"; tail -5 "$OUT/log.cli"; fail=1; }
check() { if ls $1 > /dev/null 2>&1; then echo "  ok: $2"; else echo "  missing: $2"; fail=1; fi; }
check "$OUT/*_80m.vtk" "VTK volume"
check "$OUT/*_surf.vtk" "VTK surface"
check "$OUT/*colMax*.tif" "colMax GeoTIFF"
check "$OUT/*.kmz" "KMZ"
if ! unzip -l "$OUT"/*.kmz 2>/dev/null | grep -q "colMax"; then
    echo "  missing: colMax layer in the KMZ"; fail=1
fi
[ $fail -eq 0 ] && echo "PASS" || echo "FAIL"
exit $fail
