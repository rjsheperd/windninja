# WebGPU layer.  Depends only on Dawn (webgpu.h), not on WindNinja core, so
# the solver-only WebAssembly module (WEBGPU_PLAN 3.4) can reuse it.
add_library(ninja_gpu STATIC gpu_context.cpp gpu_grid.cpp gpu_reduce.cpp
                             gpu_mass.cpp
                             gpu_rans.cpp gpu_multigrid.cpp)
target_include_directories(ninja_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ninja_gpu PUBLIC dawn::webgpu_dawn)
