#!/usr/bin/env bash
# Time the mass-consistent solver and count its PCG iterations.
#
# Usage: scripts/mass_solver_timing.sh WindNinja_cli DEM [OUT_DIR]
#
# Runs a domain-average case (10 mph from 270) on the coarse, medium and fine
# meshes, with the SSOR and Jacobi preconditioners (NINJA_MASS_PRECOND), on
# 8 threads and 1 thread (a proxy for single-threaded WebAssembly). Prints a
# table: mesh, preconditioner, threads, nodes, iterations, solve and total
# seconds. Used for MASS_GPU_PLAN.org task 0.1.
set -euo pipefail

cli=$1
dem=$2
out=${3:-$(mktemp -d)}
src=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
cp "$dem" "$out/dem.tif"

printf 'mesh\tprecond\tthreads\tnodes\titers\tsolve_s\ttotal_s\n'
for mesh in coarse medium fine; do
  for precond in SSOR JACOBI; do
    for threads in 8 1; do
      cat > "$out/run.cfg" <<EOF
num_threads                = $threads
elevation_file             = $out/dem.tif
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
mesh_choice                = $mesh
write_ascii_output         = true
EOF
      log="$out/$mesh-$precond-$threads.log"
      (cd "$out" && WINDNINJA_DATA="$src/data" NINJA_MASS_PRECOND=$precond \
        CPL_DEBUG=NINJA_MASS_SOLVER "$cli" run.cfg > "$log" 2>&1)
      iters=$(sed -n 's/.*PCG ([A-Za-z]*): \([0-9]*\) iterations.*/\1/p' "$log" | head -1)
      nodes=$(sed -n 's/.*NUMNP \([0-9]*\).*/\1/p' "$log" | head -1)
      solve=$(sed -n 's/.*Solver time was \([0-9.]*\).*/\1/p' "$log")
      total=$(sed -n 's/.*Total simulation time was \([0-9.]*\).*/\1/p' "$log")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mesh" "$precond" "$threads" "$nodes" "$iters" "$solve" "$total"
    done
  done
done
