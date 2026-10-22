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

Fine-mesh references
--------------------

`big_butte_small_fine_{0,135,270}` are OpenFOAM's fine mesh (25 m). They
are the reference for the terrain criterion (WEBGPU_PLAN GATE 2, decided
2026-10-21). On steep terrain, OpenFOAM's own coarse and medium results
differ by 0.38–0.47 m/s at 20 ft, so matching OpenFOAM at the same mesh
within GATE 0's 0.25 m/s would mean reproducing its mesh error. Instead, a
new backend's terrain result is compared with the fine mesh at the 20 ft
output and must be no worse than OpenFOAM at the same mesh, in speed RMSE
and mean direction difference:

    compare_momentum --relative CASE_DIR big_butte_small_coarse_270 big_butte_small_fine_270

Fine cases keep only `vel` and `ang`, as compressed GeoTIFFs (`vel.tif`,
`ang.tif`). There's no Mackay fine reference: OpenFOAM needs 3 retries
there, and each retry smooths the DEM, so it wouldn't be the same terrain.

Regenerating
------------

    source <openfoam>/etc/bashrc
    scripts/make_momentum_golden.sh <path/to/WindNinja_cli>

The full set takes about 3 minutes on an Apple M-series laptop. Two runs on
the same machine give bitwise identical grids.

Comparing
---------

`compare_momentum` (built in `autotest/`) compares a case directory against a
reference with the same layout:

    compare_momentum CASE_DIR REFERENCE_DIR [MAX_SPEED_RMSE MAX_DIRECTION_DIFF MAX_K_REL_RMSE]

It reports speed RMSE and max |diff| (m/s, from `vel.asc`), mean circular
direction difference (degrees, from `ang.asc`) and k relative RMSE
(RMSE / RMS of the reference, from `k.asc`). With limits given it prints
PASS or FAIL and exits 0 or 1. It exits 2 if the grids can't be read or differ
in size. The metrics live in `autotest/momentum_compare.{h,cpp}` and are tested
by the `momentum_compare` suite.
