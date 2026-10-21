#!/bin/bash

# Generate golden reference output for the momentum solver.
#
# Runs a fixed set of momentum solver cases with the OpenFOAM installation in
# the current environment and saves, per case, under OUT_DIR/<case>/:
#   u.asc, v.asc, k.asc  - raw OpenFOAM output on the output-height surface
#                          (bands of the case directory's foam.tif; m/s, m2/s2).
#                          If the first cell is taller than the output height
#                          ("output_height_resolved: no"), these are the
#                          ground-cell values, which WindNinja then log-profiles
#                          down to the output height for vel.asc.
#   vel.asc, ang.asc     - WindNinja_cli ascii speed (mph) and direction output
#   metadata.txt         - versions, settings, final residuals and retries
#
# Usage:
#   source <openfoam>/etc/bashrc
#   scripts/make_momentum_golden.sh <path/to/WindNinja_cli> [OUT_DIR]
#
# OUT_DIR defaults to autotest/golden/momentum in the source tree.
#
# MOMENTUM_BACKEND=cpu runs the same cases with the CPU RANS backend instead
# (no OpenFOAM needed). u, v and k then come from the backend's raw result
# (NINJA_MOMENTUM_RESULT_DIR) and the OpenFOAM mesh metadata is omitted.
# MOMENTUM_THREADS sets num_threads (default 2, as the golden set used).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WN_SRC_DIR="$(dirname "$SCRIPT_DIR")"

CLI="$1"
OUT_DIR="${2:-$WN_SRC_DIR/autotest/golden/momentum}"
BACKEND="${MOMENTUM_BACKEND:-openfoam}"
THREADS="${MOMENTUM_THREADS:-2}"

if [ ! -x "$CLI" ]; then
    echo "usage: $0 <path/to/WindNinja_cli> [OUT_DIR]" 1>&2
    exit 1
fi
if [ "$BACKEND" = openfoam ]; then
    if [ -z "$WM_PROJECT_VERSION" ]; then
        echo "No OpenFOAM environment loaded (source <openfoam>/etc/bashrc first)." 1>&2
        exit 1
    fi

    # macOS SIP strips DYLD_LIBRARY_PATH when starting /bin/bash, so OpenFOAM
    # could not dlopen libWindNinja from $FOAM_USER_LIBBIN.  Restore the
    # environment from the (unstripped) WM_PROJECT_DIR.  The OpenFOAM scripts
    # are not 'set -e' safe.
    set +e
    source "$WM_PROJECT_DIR/etc/bashrc" > /dev/null 2>&1
    set -e
fi

CLI="$(cd "$(dirname "$CLI")" && pwd)/$(basename "$CLI")"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

export WINDNINJA_DATA="$WN_SRC_DIR/data"
export GDAL_PAM_ENABLED=NO

# A flat 6 km x 6 km DEM at 30 m, for the analytic log-law check
gdal_create -q -of GTiff -outsize 200 200 -bands 1 -ot Float32 -burn 1000 \
            -a_srs EPSG:32612 -a_ullr 300000 4810000 306000 4804000 \
            "$WORK_DIR/flat.tif"

# Run one case and collect its output.
# Arguments: case name, DEM path, mesh choice, wind direction
run_case()
{
    local name="$1" dem="$2" mesh="$3" direction="$4"
    local run_dir="$WORK_DIR/$name"
    local out="$OUT_DIR/$name"

    echo "Running $name..."
    mkdir -p "$run_dir" "$out"
    cp "$dem" "$run_dir/"
    cat > "$run_dir/$name.cfg" << EOF
num_threads                = $THREADS
elevation_file             = $run_dir/$(basename "$dem")
initialization_method      = domainAverageInitialization
momentum_flag              = true
number_of_iterations       = 300
mesh_choice                = $mesh
input_speed                = 10.0
input_speed_units          = mph
output_speed_units         = mph
input_direction            = $direction
input_wind_height          = 20.0
units_input_wind_height    = ft
output_wind_height         = 20.0
units_output_wind_height   = ft
vegetation                 = grass
write_ascii_output         = true
momentum_backend           = $BACKEND
EOF

    if ! (cd "$run_dir" && NINJA_MOMENTUM_RESULT_DIR="$run_dir" "$CLI" "$name.cfg" > log.cli 2>&1); then
        echo "WindNinja_cli failed for $name:" 1>&2
        tail -20 "$run_dir/log.cli" 1>&2
        exit 1
    fi

    cp "$run_dir"/*_vel.asc "$out/vel.asc"
    cp "$run_dir"/*_ang.asc "$out/ang.asc"

    if [ "$BACKEND" != openfoam ]; then
        cp "$run_dir/u.asc" "$run_dir/v.asc" "$run_dir/k.asc" "$out/"
        rm -f "$out"/*.prj
        {
            echo "case: $name"
            echo "backend: $BACKEND"
            echo "dem: $(basename "$dem")"
            echo "mesh_choice: $mesh"
            echo "input: 10 mph from $direction deg at 20 ft, grass"
            echo "windninja_commit: $(git -C "$WN_SRC_DIR" rev-parse HEAD)"
            echo "platform: $(uname -s) $(uname -m)"
            grep 'CPU RANS' "$run_dir/log.cli" | sed 's/^Run 0: /  /'
        } > "$out/metadata.txt"
        return
    fi

    local case_dir
    case_dir="$(ls -d "$run_dir"/NINJAFOAM_* | head -1)"
    gdal_translate -q -b 1 -of AAIGrid -co SIGNIFICANT_DIGITS=8 \
                   "$case_dir/foam.tif" "$out/u.asc"
    gdal_translate -q -b 2 -of AAIGrid -co SIGNIFICANT_DIGITS=8 \
                   "$case_dir/foam.tif" "$out/v.asc"
    gdal_translate -q -b 3 -of AAIGrid -co SIGNIFICANT_DIGITS=8 \
                   "$case_dir/foam.tif" "$out/k.asc"
    rm -f "$out"/*.prj

    # First cell height as computed in ninjafoam.cpp: the blockMesh cell is
    # meshResolution*2*nRoundsRefinement tall and each refinement round halves it.
    local mesh_res rounds first_cell resolved
    mesh_res="$(sed -n 's/^meshResolution = //p' "$case_dir/log.ninja")"
    rounds="$(sed -n 's/^nRoundsRefinement = //p' "$case_dir/log.ninja")"
    first_cell="$(awk -v r="$mesh_res" -v n="$rounds" 'BEGIN { printf "%.2f", r * 2 * n / 2 ^ n }')"
    # output height is 20 ft = 6.096 m
    resolved="$(awk -v h="$first_cell" 'BEGIN { print (h < 6.096) ? "yes" : "no" }')"

    {
        echo "case: $name"
        echo "dem: $(basename "$dem")"
        echo "mesh_choice: $mesh"
        echo "input: 10 mph from $direction deg at 20 ft, grass"
        echo "number_of_iterations: 300"
        echo "mesh_resolution_m: $mesh_res"
        echo "refinement_rounds: $rounds"
        echo "first_cell_height_m: $first_cell"
        echo "output_height_resolved: $resolved"
        echo "openfoam_version: $WM_PROJECT_VERSION"
        echo "windninja_commit: $(git -C "$WN_SRC_DIR" rev-parse HEAD)"
        echo "platform: $(uname -s) $(uname -m)"
        echo "retries: $(grep -c 'simpleFoam tryIdx' "$run_dir/log.cli" || true)"
        echo "final_time: $(grep '^Time =' "$case_dir/log.simpleFoam" | tail -1 | cut -d' ' -f3)"
        echo "final_initial_residuals:"
        grep 'Initial residual' "$case_dir/log.simpleFoam" | tail -6 |
            sed -E 's/.*Solving for ([A-Za-z]+), Initial residual = ([^,]+),.*/  \1: \2/'
    } > "$out/metadata.txt"
}

for mesh in coarse medium; do
    for direction in 270 0 135; do
        run_case "big_butte_small_${mesh}_${direction}" \
                 "$WINDNINJA_DATA/big_butte_small.tif" "$mesh" "$direction"
    done
done
run_case "mackay_small_coarse_270" "$WINDNINJA_DATA/mackay_small.tif" coarse 270
run_case "flat_coarse_270" "$WORK_DIR/flat.tif" coarse 270

echo "Golden output written to $OUT_DIR"
