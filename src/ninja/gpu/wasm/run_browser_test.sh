#!/bin/bash

# Run the WASM module's index.html in headless Chromium (via spel) with
# WebGPU enabled, and report the module's RESULT line.
#
# Usage: src/ninja/gpu/wasm/run_browser_test.sh <WASM_BUILD_DIR>
# Exit: 0 PASS, 1 FAIL, 77 SKIP (no spel, or no WebGPU in the browser)

set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <WASM_BUILD_DIR>" 1>&2
    exit 2
fi
BUILD="$1"
if ! command -v spel > /dev/null; then
    echo "SKIP: spel is not installed"
    exit 77
fi

LOG="$(mktemp)"
SESSION="windninja-wasm-$$"
uv run --no-project python -u -m http.server 0 --bind 127.0.0.1 \
    --directory "$BUILD" > "$LOG" 2>&1 &
SERVER=$!
cleanup() {
    spel --session "$SESSION" close > /dev/null 2>&1
    kill "$SERVER" 2> /dev/null
    rm -f "$LOG"
}
trap cleanup EXIT

PORT=""
for _ in $(seq 1 50); do
    PORT="$(sed -n 's/.*port \([0-9]*\).*/\1/p' "$LOG" | head -1)"
    [ -n "$PORT" ] && break
    sleep 0.2
done
if [ -z "$PORT" ]; then
    echo "FAIL: the http server didn't start"
    cat "$LOG"
    exit 1
fi

# Without a backend hint headless Chromium falls back to SwiftShader (CPU)
ARGS="--enable-unsafe-webgpu,--enable-features=Vulkan"
if [ "$(uname -s)" = "Darwin" ]; then
    ARGS="$ARGS,--use-angle=metal"
fi
spel --session "$SESSION" --args "$ARGS" \
     open "http://127.0.0.1:$PORT/index.html" > /dev/null

RESULT=""
for _ in $(seq 1 60); do
    RESULT="$(spel --session "$SESSION" eval-sci \
        '(spel/evaluate "document.body.dataset.result || \"\"")' | tr -d '"')"
    [ -n "$RESULT" ] && break
    sleep 1
done
spel --session "$SESSION" eval-sci \
    '(spel/evaluate "document.getElementById(\"out\").textContent")' |
    sed 's/^"//; s/"$//; s/\\n/\n/g'

case "$RESULT" in
    PASS) exit 0 ;;
    SKIP) exit 77 ;;
    "") echo "FAIL: no RESULT within 60 s"; exit 1 ;;
    *) exit 1 ;;
esac
