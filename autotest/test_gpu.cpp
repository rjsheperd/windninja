#ifdef WITH_WEBGPU

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <webgpu/webgpu.h>

#include <boost/test/unit_test.hpp>

#include "gpu_context.h"
#include "gpu_grid.h"
#include "gpu_kernels.h"
#include "gpu_reduce.h"
#include "rans_grid.h"

namespace {

// The shared context, or null when this machine has no WebGPU adapter; tests
// then print the marker that ctest's SKIP_REGULAR_EXPRESSION matches.
gpu::Context *context()
{
    static std::unique_ptr<gpu::Context> c;
    static bool tried = false;
    if(!tried)
    {
        tried = true;
        try
        {
            c.reset(new gpu::Context());
        }
        catch(const gpu::Unavailable &e)
        {
            std::cout << "SKIP: no WebGPU adapter (" << e.what() << ")\n";
        }
    }
    return c.get();
}

uint32_t bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// distance in units in the last place, for finite floats of the same sign
uint32_t ulps(float a, float b)
{
    uint32_t ua = bits(a), ub = bits(b);
    return ua > ub ? ua - ub : ub - ua;
}

} // namespace

BOOST_AUTO_TEST_SUITE(gpu)

BOOST_AUTO_TEST_CASE( dawn_instance )
{
    WGPUInstance instance = wgpuCreateInstance(NULL);
    BOOST_REQUIRE(instance != NULL);
    wgpuInstanceRelease(instance);
}

BOOST_AUTO_TEST_CASE( device_create )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    BOOST_CHECK(c->device() != nullptr);
    BOOST_CHECK(c->queue() != nullptr);
    BOOST_TEST_MESSAGE("adapter: " << c->adapterName());
}

BOOST_AUTO_TEST_CASE( buffer_roundtrip )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // 1M floats, including NaN payloads and denormals, must come back bit
    // for bit
    const size_t n = 1 << 20;
    std::vector<uint32_t> in(n);
    for(size_t i = 0; i < n; i++)
        in[i] = static_cast<uint32_t>(i * 2654435761u);
    gpu::Buffer buffer(*c, n * sizeof(float));
    buffer.upload(in);
    std::vector<uint32_t> out = buffer.download<uint32_t>(n);
    BOOST_CHECK(out == in);
}

BOOST_AUTO_TEST_CASE( kernel_saxpy )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // n isn't a multiple of the workgroup size, so the bounds check matters
    const uint32_t n = 100003;
    const float a = 1.7f;
    std::vector<float> x(n), y(n);
    for(uint32_t i = 0; i < n; i++)
    {
        x[i] = std::sin(0.001f * i) * 100.0f;
        y[i] = std::cos(0.002f * i) - 0.5f;
    }
    struct { float a; uint32_t n; } params = {a, n};

    gpu::Buffer p(*c, sizeof params, gpu::Buffer::Uniform);
    gpu::Buffer bx(*c, n * sizeof(float));
    gpu::Buffer by(*c, n * sizeof(float));
    p.upload(&params, sizeof params);
    bx.upload(x);
    by.upload(y);
    gpu::Kernel saxpy(*c, gpu::kernels::saxpy);
    saxpy.dispatch({&p, &bx, &by}, (n + 63) / 64);
    std::vector<float> out = by.download<float>(n);

    // the GPU may or may not fuse a * x + y, so accept either rounding
    // within 1 ulp
    uint32_t worst = 0;
    for(uint32_t i = 0; i < n; i++)
    {
        float separate = a * x[i] + y[i];
        float fused = std::fma(a, x[i], y[i]);
        uint32_t d = std::min(ulps(out[i], separate), ulps(out[i], fused));
        worst = std::max(worst, d);
    }
    BOOST_CHECK_LE(worst, 1u);
}

BOOST_AUTO_TEST_CASE( validation_error_throws )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    const char *bad = "@compute @workgroup_size(1)\n"
                      "fn main() {\n"
                      "    let x : f32 = undefined_name;\n"
                      "}\n";
    try
    {
        gpu::Kernel kernel(*c, bad);
        BOOST_FAIL("bad WGSL compiled");
    }
    catch(const gpu::Error &e)
    {
        std::string message = e.what();
        BOOST_TEST_MESSAGE(message);
        BOOST_CHECK(message.find("undefined_name") != std::string::npos);
        BOOST_CHECK(message.find("line 3") != std::string::npos);
    }
    // the context is still usable afterwards
    gpu::Kernel good(*c, gpu::kernels::saxpy);
}

BOOST_AUTO_TEST_CASE( reduction_sum )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // 10M values in [0, 1) at a float offset into a larger buffer; both the
    // sum and the sum of squares must match f64 within 1e-6 relative
    const size_t n = 10000000, offset = 1000;
    std::vector<float> x(n + offset + 7);
    uint32_t state = 12345;
    for(float &v : x)
    {
        state = state * 1664525u + 1013904223u;
        v = (state >> 8) / 16777216.0f;
    }
    gpu::Buffer buffer(*c, x.size() * sizeof(float));
    buffer.upload(x);
    gpu::Reducer reducer(*c);
    double sum = 0.0, squares = 0.0;
    for(size_t i = offset; i < offset + n; i++)
    {
        sum += x[i];
        squares += double(x[i]) * x[i];
    }
    double gpuSum = reducer.sum(buffer, offset, n);
    double gpuSquares = reducer.sum(buffer, offset, n, true);
    BOOST_TEST_MESSAGE("sum rel err " << std::fabs(gpuSum / sum - 1) <<
                       ", squares rel err " << std::fabs(gpuSquares / squares - 1));
    BOOST_CHECK_SMALL(gpuSum / sum - 1, 1e-6);
    BOOST_CHECK_SMALL(gpuSquares / squares - 1, 1e-6);
    // small and odd sizes: a single partial group, and fewer than 256
    BOOST_CHECK_CLOSE(reducer.sum(buffer, 3, 1),
                      double(x[3]), 1e-5);
    double s300 = 0.0;
    for(size_t i = 0; i < 300; i++) s300 += x[i];
    BOOST_CHECK_CLOSE(reducer.sum(buffer, 0, 300), s300, 1e-4);
}

BOOST_AUTO_TEST_CASE( grid_pack )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // a small terrain grid at a realistic elevation
    const int nx = 5, ny = 4, nz = 3;
    std::vector<double> x, y, z;
    for(int k = 0; k <= nz; k++)
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i <= nx; i++)
            {
                x.push_back(30.0 * i);
                y.push_back(30.0 * j);
                z.push_back(2000.0 + 5.0 * i + 50.0 * k * (1 + 0.1 * k));
            }
    RansGrid g = RansGrid::FromNodes(nx, ny, nz, x, y, z);
    gpu::GridView view;
    view.nx = g.nx; view.ny = g.ny; view.nz = g.nz;
    view.volume = g.volume.data();
    view.cx = g.cx.data(); view.cy = g.cy.data(); view.cz = g.cz.data();
    view.wallDistance = g.wallDistance.data();
    const std::vector<double> *s[3][3] = {{&g.iSx, &g.iSy, &g.iSz},
                                          {&g.jSx, &g.jSy, &g.jSz},
                                          {&g.kSx, &g.kSy, &g.kSz}};
    const std::vector<double> *f[3][3] = {{&g.iFx, &g.iFy, &g.iFz},
                                          {&g.jFx, &g.jFy, &g.jFz},
                                          {&g.kFx, &g.kFy, &g.kFz}};
    for(int d = 0; d < 3; d++)
        for(int m = 0; m < 3; m++)
        {
            view.s[d][m] = s[d][m]->data();
            view.f[d][m] = f[d][m]->data();
        }
    view.agl[0] = g.iAgl.data();
    view.agl[1] = g.jAgl.data();
    gpu::Grid grid(*c, view);

    BOOST_CHECK_EQUAL(grid.nCells, size_t(nx * ny * nz));
    BOOST_CHECK_EQUAL(grid.nFaces[2], size_t(nx * ny * (nz + 1)));
    double zMin = *std::min_element(g.cz.begin(), g.cz.end());
    BOOST_CHECK_EQUAL(grid.zOrigin, zMin);
    std::vector<float> cells = grid.cells.download<float>(gpu::cellSlots * grid.nCells);
    for(size_t i = 0; i < grid.nCells; i++)
    {
        BOOST_CHECK_CLOSE(cells[gpu::cellVolume * grid.nCells + i], g.volume[i], 1e-4);
        // z is local, so it keeps f32 precision (well under a millimetre)
        BOOST_CHECK_SMALL(cells[gpu::cellZ * grid.nCells + i] - (g.cz[i] - zMin), 1e-4);
    }
    std::vector<float> kf = grid.faces[2].download<float>(gpu::faceSlots * grid.nFaces[2]);
    for(size_t i = 0; i < grid.nFaces[2]; i++)
    {
        BOOST_CHECK_CLOSE(kf[gpu::faceSz * grid.nFaces[2] + i], g.kSz[i], 1e-4);
        BOOST_CHECK_EQUAL(kf[gpu::faceAgl * grid.nFaces[2] + i], 0.0f);
    }
    std::vector<float> jf = grid.faces[1].download<float>(gpu::faceSlots * grid.nFaces[1]);
    for(size_t i = 0; i < grid.nFaces[1]; i++)
        BOOST_CHECK_CLOSE(jf[gpu::faceAgl * grid.nFaces[1] + i], g.jAgl[i], 1e-4);
}

BOOST_AUTO_TEST_CASE( fields_slots )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    const size_t n = 1001;
    gpu::Fields fields(*c, n);
    std::vector<double> u(n), k(n);
    for(size_t i = 0; i < n; i++)
    {
        u[i] = 0.5 * i;
        k[i] = 1.0 / (i + 1);
    }
    fields.upload(gpu::fieldU, u);
    fields.upload(gpu::fieldK, k);
    std::vector<double> u2 = fields.download(gpu::fieldU);
    std::vector<double> k2 = fields.download(gpu::fieldK);
    for(size_t i = 0; i < n; i++)
    {
        BOOST_CHECK_EQUAL(u2[i], double(float(u[i])));
        BOOST_CHECK_EQUAL(k2[i], double(float(k[i])));
    }
    // slots don't overlap
    std::vector<double> v = fields.download(gpu::fieldV);
    BOOST_CHECK(std::all_of(v.begin(), v.end(), [](double x) { return x == 0.0; }));
}

BOOST_AUTO_TEST_SUITE_END()

#endif
