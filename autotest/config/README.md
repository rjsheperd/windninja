Testing WindNinja with configuration files
==========================================

This folder contains tests that run WindNinja as a whole using various
configuration files.  Any file in the `pass` and `fail` folders that end with
`.cfg` will be run.  If the file is in the `pass` folder, WindNinja should run
the configuration file and return 0, suceeding.  Files in the `fail` folder
will be expected to fail, meaning returning 1.  Uncaught exceptions do not
count as passing.

The program `parse_input` is a helper program that extracts the dem file from
the config file.  If it can find the `elevation_file` tag it returns the file
name (no path) of the input DEM.  This should be located in the `data` folder
in the source tree.

Process
-------

Cmake will generate the helper executables.

Momentum solver tests
---------------------

Files in the `momentum` folder ending with `.cfg` are momentum solver runs.
They are only added when both `RUN_CFG_TESTS` and `NINJAFOAM` are on.  Each
is run through `momentum/run_momentum_cfg.py`, which runs `WindNinja_cli` and
then checks the `*_vel.asc` and `*_ang.asc` output: no NaN or no-data values,
speeds between 0 and 3 times `input_speed` (so set `output_speed_units` to
match `input_speed_units`), and angles between 0 and 360.

The tests need an OpenFOAM environment (`source <openfoam>/etc/bashrc` before
running `ctest`).  Without one (`WM_PROJECT_VERSION` unset) they are reported
as skipped, not failed.
