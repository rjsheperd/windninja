Momentum solver golden output
=============================

Reference output from the OpenFOAM momentum solver. New momentum solver
backends are compared against these files (see `WEBGPU_PLAN.org`).

Each case directory holds:

* `u.asc`, `v.asc`, `k.asc`: raw OpenFOAM output (m/s, m/s, m²/s²) on the
  output-height surface. These are the bands of the case directory's
  `foam.tif`. When the first cell is taller than the output height
  (`output_height_resolved: no` in `metadata.txt`, currently every case),
  these are ground-cell values. WindNinja then log-profiles them down to the
  output height.
* `vel.asc`, `ang.asc`: WindNinja_cli speed (mph) and direction (degrees)
  output at the output height (20 ft).
* `metadata.txt`: OpenFOAM version, WindNinja commit, platform, mesh
  resolution, refinement rounds, first cell height, retries and final
  residuals.

All cases are 10 mph at 20 ft over grass, domain average initialization, 300
iterations and 2 processes.

Regenerating
------------

    source <openfoam>/etc/bashrc
    scripts/make_momentum_golden.sh <path/to/WindNinja_cli>

The full set takes about 3 minutes on an Apple M-series laptop. Two runs on
the same machine give bitwise identical grids.
