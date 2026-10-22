#!/bin/bash

# The mass solver on the GPU against the CPU over many terrains
# (MASS_GPU_PLAN 2.5): every DEM given, at 4 directions, fine mesh, 10 mph,
# solved by CPU SSOR (WindNinja's default), CPU Jacobi and the GPU
# (NINJA_MASS_SOLVER=WEBGPU).  One CSV line per DEM and direction: nodes,
# iterations and whole-run seconds of each, and the 20 ft output's
# difference from CPU SSOR for the GPU and, as the yardstick for "solver
# tolerance", for CPU Jacobi: speed RMSE and max (mph), mean direction
# difference (deg).
#
# Usage: scripts/mass_gpu_sweep.sh <WindNinja_cli> <out.csv> <dem>...
#   e.g. scripts/mass_gpu_sweep.sh build/src/cli/WindNinja_cli sweep.csv data/*.tif

set -u

if [ $# -lt 3 ]; then
    echo "usage: $0 <WindNinja_cli> <out.csv> <dem>..." 1>&2
    exit 2
fi
CLI="$1"
CSV="$2"
shift 2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export WINDNINJA_DATA="$(dirname "$SCRIPT_DIR")/data"
export CPL_DEBUG=NINJA_MASS_SOLVER
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# diff <dirA> <dirB>: speed RMSE, max |diff| (mph), mean direction diff (deg)
diff_outputs() {
    awk 'FNR <= 6 { next }
         FILENAME ~ /\/a\/.*vel/ { for(i = 1; i <= NF; i++) va[FNR, i] = $i; next }
         FILENAME ~ /\/b\/.*vel/ { for(i = 1; i <= NF; i++) { if($i < 0 || va[FNR, i] < 0) continue
                                   d = $i - va[FNR, i]; s += d * d; n++
                                   if(d < 0) d = -d; if(d > m) m = d }; next }
         FILENAME ~ /\/a\/.*ang/ { for(i = 1; i <= NF; i++) aa[FNR, i] = $i; next }
         FILENAME ~ /\/b\/.*ang/ { for(i = 1; i <= NF; i++) { if($i < 0 || aa[FNR, i] < 0) continue
                                   d = $i - aa[FNR, i]
                                   while(d > 180) d -= 360; while(d < -180) d += 360
                                   if(d < 0) d = -d; t += d; na++ } }
         END { printf "%.4f,%.3f,%.3f", n ? sqrt(s / n) : -1, m, na ? t / na : -1 }' \
        "$1"/a/*_vel.asc "$2"/b/*_vel.asc "$1"/a/*_ang.asc "$2"/b/*_ang.asc
}

# run <dir> <dem> <direction> <solver> <precond>
run() {
    mkdir -p "$1"
    cp "$2" "$1/"
    cat > "$1/run.cfg" << CFG
num_threads                = 8
elevation_file             = $(basename "$2")
initialization_method      = domainAverageInitialization
mesh_choice                = fine
input_speed                = 10.0
input_speed_units          = mph
output_speed_units         = mph
input_direction            = $3
input_wind_height          = 20.0
units_input_wind_height    = ft
output_wind_height         = 20.0
units_output_wind_height   = ft
vegetation                 = grass
write_ascii_output         = true
CFG
    (cd "$1" && NINJA_MASS_SOLVER=$4 NINJA_MASS_PRECOND=$5 "$CLI" run.cfg > log.cli 2>&1)
}

field() {   # field <log> <sed expression>
    sed -n "$2" "$1" | tail -1
}

echo "dem,direction,nodes,ssor_iters,ssor_s,jacobi_iters,jacobi_s,gpu_iters,gpu_s,gpu_status,gpu_rmse,gpu_max,gpu_dir,jacobi_rmse,jacobi_max,jacobi_dir" > "$CSV"
for dem in "$@"; do
    name="$(basename "$dem" .tif)"
    for direction in 0 90 180 270; do
        r="$WORK/${name}_$direction"
        run "$r/ssor/a" "$dem" $direction CPU SSOR
        run "$r/jacobi/b" "$dem" $direction CPU JACOBI
        run "$r/gpu/b" "$dem" $direction WEBGPU SSOR
        status=ok
        grep -q "PCG (Jacobi, GPU)" "$r/gpu/b/log.cli" || status=failed
        it='s/.*PCG ([A-Za-z, ]*): \([0-9]*\) iterations.*/\1/p'
        tt='s/.*Total simulation time was \([0-9.]*\).*/\1/p'
        line="$name,$direction,$(field "$r/ssor/a/log.cli" 's/.*NUMNP \([0-9]*\).*/\1/p')"
        line="$line,$(field "$r/ssor/a/log.cli" "$it"),$(field "$r/ssor/a/log.cli" "$tt")"
        line="$line,$(field "$r/jacobi/b/log.cli" "$it"),$(field "$r/jacobi/b/log.cli" "$tt")"
        line="$line,$(field "$r/gpu/b/log.cli" "$it"),$(field "$r/gpu/b/log.cli" "$tt"),$status"
        line="$line,$(diff_outputs "$r/ssor" "$r/gpu"),$(diff_outputs "$r/ssor" "$r/jacobi")"
        echo "$line" | tee -a "$CSV"
    done
done
