#!/usr/bin/env python3
"""Run a momentum-solver cfg with WindNinja_cli and sanity-check its output.

usage: run_momentum_cfg.py WindNinja_cli cfg_file

Exits 77 (ctest SKIP_RETURN_CODE) when no OpenFOAM environment is loaded,
so builds without OpenFOAM report the test as skipped, not failed.
"""

import glob
import math
import os
import subprocess
import sys

SKIP = 77
MAX_SPEED_FACTOR = 3.0


def cfg_value(cfg_file, key):
    with open(cfg_file) as f:
        for line in f:
            line = line.split("#", 1)[0]
            if "=" in line:
                k, v = (s.strip() for s in line.split("=", 1))
                if k == key:
                    return v
    return None


def read_ascii_grid(path):
    with open(path) as f:
        header = {}
        for _ in range(6):
            k, v = f.readline().split()
            header[k.lower()] = float(v)
        values = [float(v) for v in f.read().split()]
    n = int(header["ncols"]) * int(header["nrows"])
    if len(values) != n:
        raise ValueError("%s: expected %d values, found %d" % (path, n, len(values)))
    return header["nodata_value"], values


def check_grid(path, lo, hi):
    nodata, values = read_ascii_grid(path)
    for v in values:
        if math.isnan(v) or v == nodata or v < lo or v > hi:
            return "%s: value %r outside [%g, %g]" % (path, v, lo, hi)
    return None


def main(cli, cfg_file):
    if not os.environ.get("WM_PROJECT_VERSION"):
        print("OpenFOAM environment not loaded (WM_PROJECT_VERSION unset); skipping")
        return SKIP

    # runs in the test's own directory; drop outputs from a previous run so
    # they can't satisfy the checks below
    for stale in glob.glob("*_vel.asc") + glob.glob("*_ang.asc"):
        os.remove(stale)

    rc = subprocess.call([cli, cfg_file])
    if rc != 0:
        print("WindNinja_cli exited with %d" % rc)
        return 1

    speed = float(cfg_value(cfg_file, "input_speed"))
    vel = glob.glob("*_vel.asc")
    ang = glob.glob("*_ang.asc")
    if not vel or not ang:
        print("missing ascii output (vel: %s, ang: %s)" % (vel, ang))
        return 1

    errors = [check_grid(p, 0.0, MAX_SPEED_FACTOR * speed) for p in vel]
    errors += [check_grid(p, 0.0, 360.0) for p in ang]
    errors = [e for e in errors if e]
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
