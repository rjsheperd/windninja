#!/bin/bash

# Gridded initialization on the WebGPU momentum backend (WEBGPU_PLAN 5.2):
# uniform 10 mph / 270 deg speed and direction grids over Big Butte; the
# momentum solver takes their domain mean as its inflow.  The run must
# finish with speeds in a sane range.  (Gridded directions are taken
# grid-relative, domain-average ones geographic, so this isn't the same
# inflow as a domain-average run from 270 deg.)
#
# Usage: scripts/check_momentum_gpu_gridded.sh <WindNinja_cli>
# Exit: 0 pass, 1 fail, 77 (ctest skip) without a WebGPU adapter.

set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <WindNinja_cli>" 1>&2
    exit 2
fi
CLI="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export WINDNINJA_DATA="$(dirname "$SCRIPT_DIR")/data"
export NINJA_MOMENTUM_FALLBACK=NO   # a GPU test: no fallback to the CPU backend
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

cp "$WINDNINJA_DATA/big_butte_small.tif" "$OUT/"
# 500 m grids reaching 1 km past the DEM (they must cover it completely)
for grid in speed:10 dir:270; do
    name=${grid%%:*}
    gdal_create -q -of GTiff -outsize 20 21 -bands 1 -ot Float32 -burn ${grid##*:} \
                -a_srs EPSG:32612 -a_ullr 331006.522 4812267.578 341006.522 4801767.578 \
                "$OUT/$name.tif"
    gdal_translate -q -of AAIGrid "$OUT/$name.tif" "$OUT/$name.asc"
done
cat > "$OUT/gridded.cfg" << CFG
num_threads                = 2
elevation_file             = big_butte_small.tif
initialization_method      = griddedInitialization
input_speed_grid           = speed.asc
input_dir_grid             = dir.asc
input_speed_units          = mph
input_wind_height          = 20.0
units_input_wind_height    = ft
output_wind_height         = 20.0
units_output_wind_height   = ft
output_speed_units         = mph
vegetation                 = grass
mesh_choice                = coarse
momentum_flag              = true
momentum_backend           = webgpu
write_ascii_output         = true
CFG

(cd "$OUT" && "$CLI" gridded.cfg > log.cli 2>&1)
status=$?
grep "RANS" "$OUT/log.cli" | tail -2
if grep -q "no adapter" "$OUT/log.cli"; then
    echo "SKIP: no WebGPU adapter"
    exit 77
fi
if [ $status -ne 0 ]; then echo "WindNinja_cli exited with $status"; tail -5 "$OUT/log.cli"; echo FAIL; exit 1; fi
vel="$(ls "$OUT"/*_vel.asc 2>/dev/null | head -1)"
if [ -z "$vel" ]; then echo "no speed output"; echo FAIL; exit 1; fi
# speeds (mph) between 0 and 3x the input, no no-data
awk 'NR > 6 { for(i = 1; i <= NF; i++) if($i < 0 || $i > 30) bad++ } END { exit bad > 0 }' "$vel" ||
    { echo "speeds outside [0, 30] mph"; echo FAIL; exit 1; }
echo PASS
