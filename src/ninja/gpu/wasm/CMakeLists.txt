# Solver-only WebAssembly module (WEBGPU_PLAN 3.4).  A standalone project:
# it builds the ninja_gpu sources with Emscripten and Dawn's emdawnwebgpu
# port, with nothing from WindNinja core.
#
#   scripts/fetch_dawn.sh --wasm <DIR>
#   emcmake cmake -S src/ninja/gpu/wasm -B build-wasm -DEMDAWNWEBGPU_PORT=<port.py>
#   cmake --build build-wasm
#   serve build-wasm/ over http and open index.html in a WebGPU browser
cmake_minimum_required(VERSION 3.20)
project(windninja_gpu_wasm CXX)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "Configure with emcmake (Emscripten >= 4.0.10)")
endif()
set(EMDAWNWEBGPU_PORT "" CACHE FILEPATH
    "emdawnwebgpu.port.py from scripts/fetch_dawn.sh --wasm")
if(NOT EXISTS "${EMDAWNWEBGPU_PORT}")
    message(FATAL_ERROR "Set EMDAWNWEBGPU_PORT (scripts/fetch_dawn.sh --wasm)")
endif()
# How Context::wait() blocks in the browser: JSPI (smaller and faster; Chrome
# 137+) or ASYNCIFY (any WebGPU browser, larger code)
set(GPU_WASM_ASYNC "JSPI" CACHE STRING "JSPI or ASYNCIFY")
set_property(CACHE GPU_WASM_ASYNC PROPERTY STRINGS JSPI ASYNCIFY)

set(CMAKE_CXX_STANDARD 17)
set(GPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(PORT_FLAG "--use-port=${EMDAWNWEBGPU_PORT}")

# JSPI can't suspend through JS frames, and JS-based C++ exceptions
# (-fexceptions) route calls inside try blocks through JS invoke_ trampolines
# ("SuspendError: trying to suspend JS frames"), so JSPI uses native wasm
# exceptions.  Asyncify keeps -fexceptions.
if(GPU_WASM_ASYNC STREQUAL "JSPI")
    set(EH_FLAG -fwasm-exceptions)
else()
    set(EH_FLAG -fexceptions)
endif()

add_executable(saxpy_wasm saxpy_wasm.cpp ${GPU_DIR}/gpu_context.cpp)
target_include_directories(saxpy_wasm PRIVATE ${GPU_DIR})
target_compile_options(saxpy_wasm PRIVATE ${PORT_FLAG} ${EH_FLAG})
target_link_options(saxpy_wasm PRIVATE ${PORT_FLAG} ${EH_FLAG}
                    -s${GPU_WASM_ASYNC}=1 -sALLOW_MEMORY_GROWTH=1)
configure_file(index.html ${CMAKE_CURRENT_BINARY_DIR}/index.html COPYONLY)
