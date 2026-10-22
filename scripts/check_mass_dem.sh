#!/bin/bash

# The core-free DEM-to-wind path the browser uses (mass_dem, MASS_GPU_PLAN
# 3.1a) against WindNinja's CLI, on a 30 m crop of Big Butte (at the mesh
# resolution, so neither resamples):
#   cpu - autotest's mass_dem_solve on the CPU writes the same ASCII values as
#         the CLI over every DEM cell (the CLI pads its grids by a row and a
#         column to cover the DEM; those are skipped)
#   gpu - mass_dem_solve on the GPU lands as close to the CLI as the CPU with
#         the Jacobi preconditioner does (the solver's own tolerance, as in
#         plan 2.5): speed RMSE and mean direction difference each within
#         1.25x the Jacobi run's
# Three cases each: grass from 237 deg and trees from 90 deg at 6.096 m, and
# grass from 300 deg at 1 m (within the first layer: the profile, not linear).
#
# Usage: scripts/check_mass_dem.sh <WindNinja_cli> <mass_dem_solve> <cpu|gpu>
# Exit: 0 pass, 1 fail, 77 (ctest skip) for gpu without a WebGPU adapter.

set -u

if [ $# -ne 3 ]; then
    echo "usage: $0 <WindNinja_cli> <mass_dem_solve> <cpu|gpu>" 1>&2
    exit 2
fi
CLI="$1"
TOOL="$2"
MODE="$3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export WINDNINJA_DATA="$(dirname "$SCRIPT_DIR")/data"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

gdal_translate -q -srcwin 60 60 100 90 "$WINDNINJA_DATA/big_butte_small.tif" "$OUT/crop.tif" &&
    gdalwarp -q -tr 30 30 -r bilinear "$OUT/crop.tif" "$OUT/dem.tif" ||
    { echo "couldn't make the test DEM"; exit 1; }

# compare <cli.asc> <ours.asc>: over our cells (the CLI's south-west block),
# values differing, speed-style RMSE and max, circular mean difference
compare() {
    awk 'FNR == 1 { file++ } FNR <= 6 { if($1 == "nrows") rows[file] = $2; next }
         file == 1 { r = FNR - 6; for(i = 1; i <= NF; i++) a[r, i] = $i; next }
         { r = FNR - 6; off = rows[1] - rows[2]
           for(i = 1; i <= NF; i++) { x = a[r + off, i]; d = $i - x; n++
                                      if(d != 0) diff++
                                      s += d * d; if(d < 0) d = -d; if(d > m) m = d
                                      c = d; while(c > 180) c -= 360; if(c < 0) c = -c; t += c } }
         END { printf "%d %d %.4f %.3f %.3f\n", diff, n, sqrt(s / n), m, t / n }' "$1" "$2"
}

fail=0
for k in "grass 237 6.096" "trees 90 6.096" "grass 300 1"; do
    set -- $k
    veg=$1; dir=$2; height=$3
    case="${veg}_$dir"
    mkdir -p "$OUT/$case/cli"
    cat > "$OUT/$case/cli/run.cfg" << CFG
num_threads                = 1
elevation_file             = $OUT/dem.tif
initialization_method      = domainAverageInitialization
input_speed                = 10.0
input_speed_units          = mps
output_speed_units         = mps
input_direction            = $dir
input_wind_height          = 6.096
units_input_wind_height    = m
output_wind_height         = $height
units_output_wind_height   = m
vegetation                 = $veg
mesh_resolution            = 30
units_mesh_resolution      = m
write_ascii_output         = true
output_path                = $OUT/$case/cli
CFG
    "$CLI" "$OUT/$case/cli/run.cfg" > "$OUT/$case/cli/log.txt" 2>&1 || { echo "CLI failed ($case)"; exit 1; }
    extra=""
    [ "$MODE" = gpu ] && extra=gpu
    OMP_NUM_THREADS=1 "$TOOL" "$OUT/dem.tif" 10 "$dir" 6.096 "$height" "$veg" "$OUT/$case/ours" $extra \
        > "$OUT/$case/tool.log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        cat "$OUT/$case/tool.log"
        if [ "$MODE" = gpu ] && grep -qiE "no adapter|not a WebGPU build|Unavailable" "$OUT/$case/tool.log"; then
            echo "SKIP: no WebGPU"
            exit 77
        fi
        echo "mass_dem_solve failed ($case)"; exit 1
    fi
    read vdiff n vrmse vmax vdir <<< "$(compare "$OUT"/$case/cli/*_vel.asc "$OUT/$case/ours_vel.asc")"
    read adiff n2 armse amax adir <<< "$(compare "$OUT"/$case/cli/*_ang.asc "$OUT/$case/ours_ang.asc")"
    echo "  $veg from $dir at $height m, $MODE: speed $vdiff of $n cells differ (RMSE $vrmse, max $vmax m/s);" \
         "direction $adiff differ (mean $adir deg)"
    if [ "$MODE" = cpu ]; then
        [ "$vdiff" = 0 ] && [ "$adiff" = 0 ] || fail=1
    else   # the yardstick: the CPU with Jacobi
        OMP_NUM_THREADS=1 "$TOOL" "$OUT/dem.tif" 10 "$dir" 6.096 "$height" "$veg" "$OUT/$case/jacobi" jacobi \
            > "$OUT/$case/jacobi.log" 2>&1 || { echo "Jacobi run failed ($case)"; exit 1; }
        read jd jn jrmse jmax jdir <<< "$(compare "$OUT"/$case/cli/*_vel.asc "$OUT/$case/jacobi_vel.asc")"
        read jad jan jarmse jamax jadir <<< "$(compare "$OUT"/$case/cli/*_ang.asc "$OUT/$case/jacobi_ang.asc")"
        echo "  $veg from $dir at $height m, CPU Jacobi: speed RMSE $jrmse, max $jmax m/s; direction mean $jadir deg"
        awk -v r="$vrmse" -v d="$adir" -v jr="$jrmse" -v jd="$jadir" \
            'BEGIN { exit !(r <= 1.25 * jr && d <= 1.25 * jd) }' || fail=1
    fi
done
[ $fail -eq 0 ] && echo "PASS" || echo "FAIL"
exit $fail
