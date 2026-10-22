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

set(CMAKE_CXX_STANDARD 20)
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

# The momentum solver: the GPU layer plus the core-free RANS sources
# (SolveDemOnGpu, as the native WebGPU backend runs).
set(NINJA_DIR ${GPU_DIR}/..)
add_executable(windninja_wasm windninja_wasm.cpp
               ${GPU_DIR}/gpu_context.cpp ${GPU_DIR}/gpu_grid.cpp ${GPU_DIR}/gpu_reduce.cpp
               ${GPU_DIR}/gpu_rans.cpp ${GPU_DIR}/gpu_multigrid.cpp ${GPU_DIR}/gpu_mass.cpp
               ${NINJA_DIR}/rans_abl.cpp ${NINJA_DIR}/rans_grid.cpp
               ${NINJA_DIR}/rans_terrain.cpp ${NINJA_DIR}/rans_wind.cpp
               ${NINJA_DIR}/rans_solver.cpp ${NINJA_DIR}/rans_multigrid.cpp
               ${NINJA_DIR}/rans_gpu_bridge.cpp ${NINJA_DIR}/rans_gpu_solve.cpp
               ${NINJA_DIR}/rans_dem_solve.cpp
               ${NINJA_DIR}/mass_fem.cpp ${NINJA_DIR}/mass_pcg.cpp ${NINJA_DIR}/preconditioner.cpp
               ${NINJA_DIR}/mass_gpu_solve.cpp ${NINJA_DIR}/mass_terrain.cpp
               ${NINJA_DIR}/mass_wind.cpp ${NINJA_DIR}/mass_dem.cpp ${NINJA_DIR}/windProfile.cpp)
target_include_directories(windninja_wasm PRIVATE ${GPU_DIR} ${NINJA_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(windninja_wasm PRIVATE ${PORT_FLAG} ${EH_FLAG})
set(SOLVER_EXPORTS "-sEXPORTED_FUNCTIONS=_main,_windninja_solve,_windninja_mass_solve,_windninja_mass_dem_solve,_malloc,_free"
                   "-sEXPORTED_RUNTIME_METHODS=ccall,HEAPF64")
if(GPU_WASM_ASYNC STREQUAL "JSPI")
    # these wait on the GPU, so they must be suspending exports
    list(APPEND SOLVER_EXPORTS "-sJSPI_EXPORTS=windninja_solve,windninja_mass_solve,windninja_mass_dem_solve")
endif()
target_link_options(windninja_wasm PRIVATE ${PORT_FLAG} ${EH_FLAG}
                    -s${GPU_WASM_ASYNC}=1 -sALLOW_MEMORY_GROWTH=1 ${SOLVER_EXPORTS})
configure_file(solver.html ${CMAKE_CURRENT_BINARY_DIR}/solver.html COPYONLY)
