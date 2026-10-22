#!/bin/bash

# Robustness on steep terrain (WEBGPU_PLAN 2.5): run Mackay on the fine mesh
# with the WebGPU momentum backend.  It must finish, with no more DEM
# smoothing retries than OpenFOAM needs there (3).
#
# Usage: scripts/check_momentum_gpu_steep.sh <WindNinja_cli>
# Exit: 0 pass, 1 fail, 77 (ctest skip) without a WebGPU adapter.

set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <WindNinja_cli>" 1>&2
    exit 2
fi
CLI="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WN_SRC_DIR="$(dirname "$SCRIPT_DIR")"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

export WINDNINJA_DATA="$WN_SRC_DIR/data"
export NINJA_MOMENTUM_FALLBACK=NO   # a GPU test: no fallback to the CPU backend
cp "$WINDNINJA_DATA/mackay_small.tif" "$OUT/"
cat > "$OUT/steep.cfg" << CFG
num_threads                = 2
elevation_file             = $OUT/mackay_small.tif
initialization_method      = domainAverageInitialization
momentum_flag              = true
mesh_choice                = fine
input_speed                = 10.0
input_speed_units          = mph
output_speed_units         = mph
input_direction            = 270
input_wind_height          = 20.0
units_input_wind_height    = ft
output_wind_height         = 20.0
units_output_wind_height   = ft
vegetation                 = grass
write_ascii_output         = true
momentum_backend           = webgpu
CFG

(cd "$OUT" && "$CLI" steep.cfg > log.cli 2>&1)
status=$?
grep "RANS" "$OUT/log.cli"
if grep -q "no adapter" "$OUT/log.cli"; then
    echo "SKIP: no WebGPU adapter"
    exit 77
fi
retries=$(grep -c "Smoothing the elevation" "$OUT/log.cli")
echo "exit $status, DEM smoothing retries: $retries (OpenFOAM: 3)"
if [ $status -ne 0 ] || [ "$retries" -gt 3 ] || ! ls "$OUT"/*_vel.asc > /dev/null 2>&1; then
    echo "FAIL"
    exit 1
fi
echo "PASS"
