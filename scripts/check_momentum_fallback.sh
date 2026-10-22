#!/bin/bash

# The WebGPU momentum backend falls back to the CPU backend (WEBGPU_PLAN 5.3)
# when it can't run, with a warning that says why:
#   no_adapter - NINJA_WEBGPU_DISABLE (as on a machine without WebGPU)
#   oom        - NINJA_WEBGPU_TEST_ALLOC_BYTES: a buffer allocation the device
#                rejects; the warning must name the requested size
# The run is a small synthetic hill at 60 m, so the CPU backend is quick.
#
# Usage: scripts/check_momentum_fallback.sh <WindNinja_cli> <no_adapter|oom>
# Exit: 0 pass, 1 fail, 77 (ctest skip) for oom without a WebGPU adapter.

set -u

if [ $# -ne 2 ]; then
    echo "usage: $0 <WindNinja_cli> <no_adapter|oom>" 1>&2
    exit 2
fi
CLI="$1"
MODE="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export WINDNINJA_DATA="$(dirname "$SCRIPT_DIR")/data"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# a 40 x 30 Gaussian hill at 30 m
awk 'BEGIN {
    print "ncols 40"; print "nrows 30"; print "xllcorner 500000"
    print "yllcorner 4800000"; print "cellsize 30"; print "NODATA_value -9999"
    for(r = 0; r < 30; r++) {
        line = ""
        for(c = 0; c < 40; c++)
            line = line sprintf("%.2f ", 1000 + 150 * exp(-((c - 20) ^ 2 + (r - 15) ^ 2) / 40))
        print line
    }
}' > "$OUT/hill.asc"
gdal_translate -q -a_srs EPSG:32612 -of GTiff "$OUT/hill.asc" "$OUT/hill.tif"

cat > "$OUT/hill.cfg" << CFG
num_threads                = 2
elevation_file             = $OUT/hill.tif
initialization_method      = domainAverageInitialization
momentum_flag              = true
momentum_backend           = webgpu
mesh_resolution            = 60
units_mesh_resolution      = m
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
CFG

case "$MODE" in
    no_adapter) export NINJA_WEBGPU_DISABLE=YES; expect="no adapter" ;;
    oom) export NINJA_WEBGPU_TEST_ALLOC_BYTES=1000000000000; expect="1000000000000 bytes" ;;
    *) echo "unknown mode $MODE" 1>&2; exit 2 ;;
esac

(cd "$OUT" && "$CLI" hill.cfg > log.cli 2>&1)
status=$?
grep -E "RANS|Falling back|momentum backend" "$OUT/log.cli" | sed 's/^/  /'
if [ "$MODE" = oom ] && grep -q "no adapter" "$OUT/log.cli"; then
    echo "SKIP: no WebGPU adapter, so no allocation to fail"
    exit 77
fi
fail=0
if [ $status -ne 0 ]; then echo "WindNinja_cli exited with $status"; fail=1; fi
if ! grep "Falling back to the CPU momentum backend" "$OUT/log.cli" | grep -q "$expect"; then
    echo "no fallback warning naming '$expect'"; fail=1
fi
if ! grep -q "CPU RANS stopped" "$OUT/log.cli"; then echo "the CPU backend didn't run"; fail=1; fi
if ! ls "$OUT"/*_vel.asc > /dev/null 2>&1; then echo "no output written"; fail=1; fi
[ $fail -eq 0 ] && echo "PASS" || echo "FAIL"
exit $fail
