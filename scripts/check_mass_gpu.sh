#!/bin/bash

# The mass solver on the GPU (MASS_GPU_PLAN 2.2a, NINJA_MASS_SOLVER=WEBGPU):
#   match      - Big Butte (small, coarse mesh) solved on the GPU matches the
#                CPU solve: speed RMSE <= 0.05 mph, max <= 0.5 mph, mean
#                direction difference <= 0.5 deg (the GPU solves with Jacobi
#                in f32; CPU SSOR vs CPU Jacobi differ about as much)
#   no_adapter - with NINJA_WEBGPU_DISABLE the run warns and solves on the CPU
#
# Usage: scripts/check_mass_gpu.sh <WindNinja_cli> <match|no_adapter>
# Exit: 0 pass, 1 fail, 77 (ctest skip) for match without a WebGPU adapter.

set -u

if [ $# -ne 2 ]; then
    echo "usage: $0 <WindNinja_cli> <match|no_adapter>" 1>&2
    exit 2
fi
CLI="$1"
MODE="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export WINDNINJA_DATA="$(dirname "$SCRIPT_DIR")/data"
export CPL_DEBUG=NINJA_MASS_SOLVER
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# run <dir> <solver>: Big Butte small, coarse mesh, output in <dir>
run() {
    mkdir -p "$OUT/$1"
    cat > "$OUT/$1/run.cfg" << CFG
num_threads                = 2
elevation_file             = $WINDNINJA_DATA/big_butte_small.tif
initialization_method      = domainAverageInitialization
input_speed                = 10.0
input_speed_units          = mph
output_speed_units         = mph
input_direction            = 270
input_wind_height          = 20.0
units_input_wind_height    = ft
output_wind_height         = 20.0
units_output_wind_height   = ft
vegetation                 = grass
mesh_choice                = coarse
write_ascii_output         = true
output_path                = $OUT/$1
CFG
    NINJA_MASS_SOLVER="$2" "$CLI" "$OUT/$1/run.cfg" > "$OUT/$1/log.cli" 2>&1
}

fail=0
case "$MODE" in
    match)
        run cpu CPU || { echo "CPU run failed"; exit 1; }
        run gpu WEBGPU || { echo "GPU run failed"; cat "$OUT/gpu/log.cli"; exit 1; }
        grep -o "PCG.*" "$OUT/cpu/log.cli" "$OUT/gpu/log.cli" | sed "s#$OUT/##; s/^/  /"
        if grep -q "solving on the CPU instead" "$OUT/gpu/log.cli"; then
            echo "SKIP: no WebGPU adapter"
            exit 77
        fi
        grep -q "PCG (Jacobi, GPU)" "$OUT/gpu/log.cli" || { echo "the GPU didn't solve"; fail=1; }
        # speed RMSE and max (mph), mean circular direction difference (deg)
        awk 'FNR <= 6 { next }
             FILENAME ~ /cpu.*vel/ { for(i = 1; i <= NF; i++) va[FNR, i] = $i; next }
             FILENAME ~ /gpu.*vel/ { for(i = 1; i <= NF; i++) { d = $i - va[FNR, i]; s += d * d; n++
                                     if(d < 0) d = -d; if(d > m) m = d }; next }
             FILENAME ~ /cpu.*ang/ { for(i = 1; i <= NF; i++) aa[FNR, i] = $i; next }
             FILENAME ~ /gpu.*ang/ { for(i = 1; i <= NF; i++) { d = $i - aa[FNR, i]
                                     while(d > 180) d -= 360; while(d < -180) d += 360
                                     if(d < 0) d = -d; t += d; na++ } }
             END { rmse = sqrt(s / n); dir = t / na
                   printf "  speed RMSE %.4f mph, max %.4f mph, direction %.4f deg (%d cells)\n", rmse, m, dir, n
                   exit !(n > 0 && rmse <= 0.05 && m <= 0.5 && dir <= 0.5) }' \
            "$OUT"/cpu/*_vel.asc "$OUT"/gpu/*_vel.asc "$OUT"/cpu/*_ang.asc "$OUT"/gpu/*_ang.asc \
            || { echo "GPU and CPU outputs differ by more than the limits"; fail=1; }
        ;;
    no_adapter)
        NINJA_WEBGPU_DISABLE=YES run gpu WEBGPU
        status=$?
        grep -E "Mass solver|PCG" "$OUT/gpu/log.cli" | sed 's/^/  /'
        [ $status -eq 0 ] || { echo "WindNinja_cli exited with $status"; fail=1; }
        grep -q "no adapter (NINJA_WEBGPU_DISABLE); solving on the CPU instead" "$OUT/gpu/log.cli" \
            || { echo "no fallback warning"; fail=1; }
        grep -q "PCG (SSOR)" "$OUT/gpu/log.cli" || { echo "the CPU didn't solve"; fail=1; }
        ls "$OUT"/gpu/*_vel.asc > /dev/null 2>&1 || { echo "no output written"; fail=1; }
        ;;
    *) echo "unknown mode $MODE" 1>&2; exit 2 ;;
esac
[ $fail -eq 0 ] && echo "PASS" || echo "FAIL"
exit $fail
