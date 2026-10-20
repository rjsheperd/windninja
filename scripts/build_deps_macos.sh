#!/bin/bash
#
# Install WindNinja dependencies on macOS (Apple silicon) and build
# OpenFOAM 11 (openfoam.org) from source for the momentum solver.
#
# The OpenFOAM Foundation does not support macOS, so OpenFOAM 11 is built
# with the community patch from https://github.com/mrklein/openfoam-os-x
# (Intel only) plus openfoam11_macos_arm64.patch for Apple silicon.
#
# OpenFOAM needs a case-sensitive filesystem, so it is built inside a
# sparse disk image. Override the defaults with:
#   OPENFOAM_IMAGE  path of the disk image   (~/OpenFOAM-Foundation.sparseimage)
#   OPENFOAM_VOLUME mount point of the image (/Volumes/OpenFOAM-Foundation)
#
# Run this from a clean shell (not inside a nix or conda environment).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WN_SRC_DIR="$(dirname "$SCRIPT_DIR")"

OPENFOAM_IMAGE="${OPENFOAM_IMAGE:-$HOME/OpenFOAM-Foundation.sparseimage}"
OPENFOAM_VOLUME="${OPENFOAM_VOLUME:-/Volumes/OpenFOAM-Foundation}"
OPENFOAM_DIR="$OPENFOAM_VOLUME/OpenFOAM-11"
OPENFOAM_COMMIT=9cbf94f3339dd49286633b303c4155fd10ea8200

MRKLEIN_COMMIT=173bade65280e042b8360c25789689cb8c4ce179
MRKLEIN_PATCH=OpenFOAM-11-528c1e71cc.patch
MRKLEIN_SHA256=a0dab96d92db205f52d057ef047ab08da7539eb62a38589093125941447e64df

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "This script is for macOS on Apple silicon (arm64)." 1>&2
    exit 1
fi

if [ -n "$IN_NIX_SHELL" ]; then
    echo "Warning: running inside a nix shell; Homebrew libraries may not be found." 1>&2
fi

xcode-select -p > /dev/null || xcode-select --install

# Install WindNinja dependencies
brew install cmake \
             pkg-config \
             boost \
             shapelib \
             netcdf \
             gdal \
             libomp

# Install qt6
brew install qtbase qtwebengine qtwebchannel

# Install hwloc, MPI and decomposition libraries, for OpenFOAM
brew install hwloc open-mpi metis scotch

# Create and mount a case-sensitive volume for OpenFOAM
if [ ! -e "$OPENFOAM_IMAGE" ]; then
    hdiutil create -size 30g -type SPARSE -fs "Case-sensitive APFS" \
                   -volname OpenFOAM-Foundation "$OPENFOAM_IMAGE"
fi
if [ ! -d "$OPENFOAM_VOLUME" ]; then
    hdiutil attach -mountpoint "$OPENFOAM_VOLUME" "$OPENFOAM_IMAGE"
fi

# Get OpenFOAM 11 and apply the macOS patches
if [ ! -d "$OPENFOAM_DIR" ]; then
    git init -q "$OPENFOAM_DIR"
    git -C "$OPENFOAM_DIR" fetch -q --depth 1 \
        https://github.com/OpenFOAM/OpenFOAM-11.git "$OPENFOAM_COMMIT"
    git -C "$OPENFOAM_DIR" checkout -q FETCH_HEAD
fi

if [ ! -d "$OPENFOAM_DIR/wmake/rules/darwin64Clang" ]; then
    curl -fsSL -o "$OPENFOAM_DIR/$MRKLEIN_PATCH" \
        "https://raw.githubusercontent.com/mrklein/openfoam-os-x/$MRKLEIN_COMMIT/$MRKLEIN_PATCH"
    echo "$MRKLEIN_SHA256  $OPENFOAM_DIR/$MRKLEIN_PATCH" | shasum -a 256 -c -
    git -C "$OPENFOAM_DIR" apply "$MRKLEIN_PATCH"
    git -C "$OPENFOAM_DIR" apply "$SCRIPT_DIR/openfoam11_macos_arm64.patch"
fi

cat > "$OPENFOAM_DIR/etc/prefs.sh" << 'EOF'
export WM_COMPILER=Clang
export WM_COMPILE_OPTION=Opt
export WM_MPLIB=SYSTEMOPENMPI
export WM_NCOMPPROCS=$(sysctl -n hw.ncpu)
export WM_LABEL_SIZE=32
export WM_QUIET=1
EOF

# Build OpenFOAM (~45 minutes) and the WindNinja OpenFOAM libraries.
# The OpenFOAM environment scripts are not 'set -e' safe.
set +e
source "$OPENFOAM_DIR/etc/bashrc" > /dev/null 2>&1
set -e
ulimit -n 8192

(cd "$OPENFOAM_DIR" && ./Allwmake -k > log.Allwmake 2>&1)
if ! command -v foamRun > /dev/null; then
    echo "OpenFOAM build failed, see $OPENFOAM_DIR/log.Allwmake" 1>&2
    exit 1
fi

# Build a copy so wmake output stays out of the WindNinja source tree
rm -rf "$OPENFOAM_VOLUME/windninja-ninjafoam"
cp -R "$WN_SRC_DIR/src/ninjafoam/11" "$OPENFOAM_VOLUME/windninja-ninjafoam"
(cd "$OPENFOAM_VOLUME/windninja-ninjafoam" && wmake libso)
(cd "$OPENFOAM_VOLUME/windninja-ninjafoam/utility/applyInit" && wmake)

cat << EOF

Done. Configure WindNinja with:

    cmake -S $WN_SRC_DIR -B build -DCMAKE_PREFIX_PATH=$(brew --prefix) \\
          -DOpenMP_ROOT=$(brew --prefix libomp) \\
          -DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix)/lib -lshp -L$(brew --prefix libomp)/lib -lomp" \\
          -DCMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include"

Before running WindNinja with the momentum solver:

    hdiutil attach -mountpoint $OPENFOAM_VOLUME $OPENFOAM_IMAGE  # after a reboot
    source $OPENFOAM_DIR/etc/bashrc

EOF
