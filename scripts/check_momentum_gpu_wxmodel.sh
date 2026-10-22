#!/bin/bash

# Weather-model initialization on the WebGPU momentum backend (WEBGPU_PLAN
# 5.2): the local NDFD forecast over Big Butte (data/20170603T1800.nc, as
# data/cli_wxModelInitialization_local_model.cfg).  Its 10 m wind sits 2 m
# above a tree canopy, so the flow aloft is fast; with the artificial-
# compressibility beta set from the input speed the solve never settled
# (mean 25.6 mph, max 82).  The run must settle before the step cap, with a
# mean speed within 10% of OpenFOAM's on the same case (27.18 mph).
#
# Usage: scripts/check_momentum_gpu_wxmodel.sh <WindNinja_cli>
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

cp "$WINDNINJA_DATA/big_butte_small.tif" "$WINDNINJA_DATA/20170603T1800.nc" "$OUT/"
sed -e 's|^elevation_file.*|elevation_file = big_butte_small.tif|' \
    -e 's|^forecast_filename.*|forecast_filename = 20170603T1800.nc|' \
    "$WINDNINJA_DATA/cli_wxModelInitialization_local_model.cfg" > "$OUT/wx.cfg"
printf "momentum_flag = true\nmomentum_backend = webgpu\n" >> "$OUT/wx.cfg"

(cd "$OUT" && "$CLI" wx.cfg > log.cli 2>&1)
status=$?
grep "RANS stopped" "$OUT/log.cli"
if grep -q "no adapter" "$OUT/log.cli"; then
    echo "SKIP: no WebGPU adapter"
    exit 77
fi
fail=0
[ $status -eq 0 ] || { echo "WindNinja_cli exited with $status"; fail=1; }
steps=$(sed -n 's/.*RANS stopped after \([0-9]*\) steps.*/\1/p' "$OUT/log.cli" | tail -1)
if [ -z "$steps" ] || [ "$steps" -ge 20000 ]; then echo "didn't settle before the step cap"; fail=1; fi
vel="$(ls "$OUT"/big_butte_small_*_vel.asc 2>/dev/null | head -1)"
if [ -z "$vel" ]; then
    echo "no speed output"; fail=1
else
    awk 'NR > 6 { for(i = 1; i <= NF; i++) { s += $i; n++ } }
         END { m = s / n; printf "mean speed %.2f mph (OpenFOAM 27.18)\n", m; exit (m < 24.46 || m > 29.90) }' "$vel" ||
        { echo "mean speed not within 10% of OpenFOAM's"; fail=1; }
fi
[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
