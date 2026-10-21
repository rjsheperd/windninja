/*
** WebGPU overhead micro-benchmark (WEBGPU_PLAN 3.3).
**
** For each size it measures the cost of one compute dispatch (queued
** back-to-back, and one dispatch waited on alone), an upload, a readback, and
** creating a readback staging buffer, then times the CPU RANS solver per
** iteration on a 100k-cell grid for comparison.
**
** Usage: gpu_overhead_bench [--quick]
**   --quick  small sizes and few repetitions (the ctest smoke run)
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu_context.h"
#include "rans_abl.h"
#include "rans_grid.h"
#include "rans_solver.h"
#include "rans_wind.h"

namespace {

// saxpy with 256 threads per workgroup, so 10M elements stay within the
// 65535-workgroup dispatch limit
const char *saxpy256 = R"wgsl(
struct Params {
    a : f32,
    n : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> x : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i < params.n) {
        y[i] = params.a * x[i] + y[i];
    }
}
)wgsl";

double now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Timings
{
    double queuedDispatch;   // per dispatch, many queued then one wait
    double waitedDispatch;   // one dispatch plus its wait
    double upload;           // writeBuffer plus wait
    double readback;         // copy, map and memcpy
    double stagingCreate;    // create and destroy a MapRead buffer
};

double stagingCreate(gpu::Context &c, std::size_t bytes, int reps)
{
    double t0 = now();
    for(int r = 0; r < reps; r++)
    {
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        desc.size = bytes;
        WGPUBuffer b = wgpuDeviceCreateBuffer(c.device(), &desc);
        wgpuBufferDestroy(b);
        wgpuBufferRelease(b);
    }
    c.finish();
    return (now() - t0) / reps;
}

Timings measure(gpu::Context &c, gpu::Kernel &kernel, uint32_t n, int reps)
{
    std::vector<float> x(n, 1.0f), y(n, 0.0f);
    struct { float a; uint32_t n; } params = {0.5f, n};
    gpu::Buffer p(c, sizeof params, gpu::Buffer::Uniform);
    gpu::Buffer bx(c, n * sizeof(float));
    gpu::Buffer by(c, n * sizeof(float));
    p.upload(&params, sizeof params);
    bx.upload(x);
    by.upload(y);
    const uint32_t groups = (n + 255) / 256;
    kernel.dispatch({&p, &bx, &by}, groups); // warm up
    c.finish();

    Timings t;
    double t0 = now();
    for(int r = 0; r < 10 * reps; r++)
        kernel.dispatch({&p, &bx, &by}, groups);
    c.finish();
    t.queuedDispatch = (now() - t0) / (10 * reps);

    t0 = now();
    for(int r = 0; r < reps; r++)
    {
        kernel.dispatch({&p, &bx, &by}, groups);
        c.finish();
    }
    t.waitedDispatch = (now() - t0) / reps;

    t0 = now();
    for(int r = 0; r < reps; r++)
    {
        bx.upload(x);
        c.finish();
    }
    t.upload = (now() - t0) / reps;

    t0 = now();
    for(int r = 0; r < reps; r++)
        by.download(y.data(), n * sizeof(float));
    t.readback = (now() - t0) / reps;

    t.stagingCreate = stagingCreate(c, by.size(), reps);
    return t;
}

// seconds per explicit iteration of the turbulent CPU solver on a flat
// nx x ny x nz grid
double cpuIteration(int nx, int ny, int nz, int threads, int iterations)
{
    std::vector<double> x, y, z;
    for(int k = 0; k <= nz; k++)
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i <= nx; i++)
            {
                x.push_back(i * 60.0);
                y.push_back(j * 60.0);
                z.push_back(1000.0 + 1500.0 * std::pow(k / double(nz), 2.0));
            }
    RansGrid g = RansGrid::FromNodes(nx, ny, nz, x, y, z);
    const double z0 = 0.01;
    double ustar = AblFrictionVelocity(4.4704, 6.1, z0);
    RansSolver s(g, 1.58946099535e-05, 25.0);
    s.threads = threads;
    ApplyWindBoundaries(s, ustar, z0, 270.0);
    InitializeLogProfile(s, ustar, z0, 270.0);
    s.iterate(2, 0.5); // warm up
    double t0 = now();
    s.iterate(iterations, 0.5);
    return (now() - t0) / iterations;
}

} // namespace

int main(int argc, char **argv)
{
    bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    std::vector<uint32_t> sizes = {25000, 100000, 1000000, 10000000};
    int reps = 20;
    if(quick)
    {
        sizes = {25000, 100000};
        reps = 2;
    }

    try
    {
        gpu::Context c;
        gpu::Kernel kernel(c, saxpy256);
        std::printf("adapter: %s\n", c.adapterName().c_str());
        std::printf("%10s %12s %12s %12s %12s %12s\n", "elements",
                    "queued_us", "waited_us", "upload_us", "readback_us",
                    "staging_us");
        for(uint32_t n : sizes)
        {
            Timings t = measure(c, kernel, n, reps);
            std::printf("%10u %12.1f %12.1f %12.1f %12.1f %12.1f\n", n,
                        1e6 * t.queuedDispatch, 1e6 * t.waitedDispatch,
                        1e6 * t.upload, 1e6 * t.readback,
                        1e6 * t.stagingCreate);
        }
    }
    catch(const gpu::Unavailable &e)
    {
        std::printf("SKIP: no WebGPU adapter (%s)\n", e.what());
        return 0;
    }

    // 100 x 50 x 20 = 100k cells
    int iterations = quick ? 2 : 20;
    double serial = cpuIteration(100, 50, 20, 1, iterations);
    double parallel = cpuIteration(100, 50, 20, 0, iterations);
    std::printf("cpu 100k cells: %.1f ms/iteration (1 thread), "
                "%.1f ms/iteration (all threads)\n",
                1e3 * serial, 1e3 * parallel);
    return 0;
}
