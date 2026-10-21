#!/bin/bash

# Download the pinned prebuilt Dawn (WebGPU) package for this platform.
#
# Dawn's nightly GitHub releases provide native Release builds (and the
# matching emdawnwebgpu package for Emscripten). They are best-effort builds,
# not signed or guaranteed by Google; only dawn.googlesource.com source is
# official. A shipped installer may want a source build of the same revision.
#
# Usage:
#   scripts/fetch_dawn.sh <DEST_DIR>
#   cmake -DWITH_WEBGPU=ON -DDawn_DIR=<printed path> ...
#
# The package is verified against a pinned SHA-256 before it is extracted.

set -e

TAG=v20261016.190333
REV=67d4d6e64b0755d638f76549794692badfdbc8ab

if [ $# -ne 1 ]; then
    echo "usage: $0 <DEST_DIR>" 1>&2
    exit 2
fi
DEST="$1"

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
        PLATFORM=macos-latest
        SHA256=736fccba78b8a2fde8bfd55d045e890933b0a01776aed5a29b9f024735404193 ;;
    Darwin-x86_64)
        PLATFORM=macos-15-intel
        SHA256=277dafe9b6d8c3374f10963d54cec5548c05ab478519f85c9cb335dfbd31dde1 ;;
    Linux-x86_64)
        PLATFORM=ubuntu-latest
        SHA256=20ff88448def9645ceeff30911e07295aeb433be23e26180c96019b4300bff5a ;;
    MINGW*|MSYS*|CYGWIN*)
        PLATFORM=windows-latest
        SHA256=57402c6aa887e30d3e5710707d6b0677d8292538ab941628c624635022bfa49f ;;
    *)
        echo "No prebuilt Dawn package for $(uname -s) $(uname -m); build Dawn from source." 1>&2
        exit 1 ;;
esac

NAME="Dawn-$REV-$PLATFORM-Release"
ARCHIVE="$DEST/$NAME.tar.gz"
mkdir -p "$DEST"

if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading $NAME..."
    curl -fsSL -o "$ARCHIVE" \
        "https://github.com/google/dawn/releases/download/$TAG/$NAME.tar.gz"
fi

if command -v shasum > /dev/null; then
    ACTUAL="$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
else
    ACTUAL="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
fi
if [ "$ACTUAL" != "$SHA256" ]; then
    echo "SHA-256 mismatch for $ARCHIVE" 1>&2
    echo "  expected $SHA256" 1>&2
    echo "  got      $ACTUAL" 1>&2
    exit 1
fi

if [ ! -d "$DEST/$NAME" ]; then
    tar -xzf "$ARCHIVE" -C "$DEST"
fi

echo "Dawn $TAG ($PLATFORM) is in $DEST/$NAME"
echo "Configure WindNinja with: -DWITH_WEBGPU=ON -DDawn_DIR=$DEST/$NAME/lib/cmake/Dawn"
