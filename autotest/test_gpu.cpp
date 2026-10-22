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
#include "gpu_multigrid.h"
#include "gpu_kernels.h"
#include "gpu_rans.h"
#include "gpu_reduce.h"
#include "rans_abl.h"
#include "rans_gpu_bridge.h"
#include "rans_grid.h"
#include "rans_multigrid.h"
#include "wasm/hill_selftest.h"
#include "rans_solver.h"
#include "rans_wind.h"

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

// A terrain-following grid over a Gaussian ridge (first layer h1, geometric
// growth), at a realistic elevation.
RansGrid RidgeGrid(int nx, int ny, int nz)
{
    std::vector<double> x, y, z;
    const double dx = 60.0, h1 = 20.0, growth = 1.15, top = 2600.0;
    double total = 0.0, h = h1;
    for(int m = 0; m < nz; m++) { total += h; h *= growth; }
    for(int k = 0; k <= nz; k++)
    {
        double zk = 0.0;
        h = h1;
        for(int m = 0; m < k; m++) { zk += h; h *= growth; }
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i <= nx; i++)
            {
                double px = i * dx, py = j * dx;
                double r = (px - nx * dx / 2) / 400.0;
                double gz = 1000.0 + 300.0 * std::exp(-r * r) + 0.02 * py;
                x.push_back(px);
                y.push_back(py);
                z.push_back(gz + (top - gz) * zk / total);
            }
    }
    return RansGrid::FromNodes(nx, ny, nz, x, y, z);
}

// max |a - b| over max |b|
double RelativeDiff(const std::vector<double> &a, const std::vector<double> &b)
{
    double diff = 0.0, scale = 0.0;
    for(size_t i = 0; i < a.size(); i++)
    {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
        scale = std::max(scale, std::fabs(b[i]));
    }
    return scale > 0.0 ? diff / scale : diff;
}

// The turbulent wind setup on the ridge grid with a smooth perturbation, so
// every term is active (w changes sign on the top outlet: backflow).
void PerturbedWindState(RansSolver &cpu)
{
    const RansGrid &g = cpu.grid;
    const double z0 = 0.01, ustar = AblFrictionVelocity(4.47, 6.1, z0);
    ApplyWindBoundaries(cpu, ustar, z0, 280.0);
    InitializeLogProfile(cpu, ustar, z0, 280.0);
    for(int i = 0; i < g.nCells(); i++)
    {
        double x = g.cx[i], y = g.cy[i], z = g.cz[i];
        cpu.p[i] = 3.0 * std::sin(x / 500.0) * std::cos(y / 300.0);
        cpu.u[i] += std::sin(y / 200.0);
        cpu.v[i] += -1.0 + 0.5 * std::cos(x / 350.0 + z / 400.0);
        cpu.w[i] = 0.3 * std::sin(x / 250.0);
        cpu.k[i] *= 1.0 + 0.5 * std::sin(x / 300.0) * std::cos(z / 500.0);
        cpu.epsilon[i] *= 1.0 + 0.3 * std::cos(y / 250.0);
        cpu.nut[i] = cpu.model.cmu * cpu.k[i] * cpu.k[i] / cpu.epsilon[i];
    }
}

// worst RelativeDiff over the 7 fields
double FieldsDiff(const RansSolver &cpu, gpu::RansSolver &gs)
{
    const std::vector<double> *f[7] = {&cpu.p, &cpu.u, &cpu.v, &cpu.w, &cpu.k,
                                       &cpu.epsilon, &cpu.nut};
    const char *names[7] = {"p", "u", "v", "w", "k", "epsilon", "nut"};
    double worst = 0.0;
    for(int slot = 0; slot < 7; slot++)
    {
        double r = RelativeDiff(gs.fields.download(gpu::FieldSlot(slot)), *f[slot]);
        BOOST_TEST_MESSAGE("  field " << names[slot] << ": " << r);
        worst = std::max(worst, r);
    }
    return worst;
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
    gpu::Grid grid(*c, GpuGridView(g));

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

BOOST_AUTO_TEST_CASE( kernel_gradients_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    RansGrid g = RidgeGrid(40, 12, 10);
    // ::RansSolver: this suite is namespace gpu, which has its own
    ::RansSolver cpu(g, 1.5e-5, 25.0);
    const double z0 = 0.01, ustar = AblFrictionVelocity(4.47, 6.1, z0);
    ApplyWindBoundaries(cpu, ustar, z0, 280.0);   // inlets W and N
    // a smooth, non-trivial state
    cpu.setInitial([](double x, double y, double z, double &p, double &u,
                      double &v, double &w)
    {
        p = 3.0 * std::sin(x / 500.0) * std::cos(y / 300.0);
        u = 5.0 + 0.004 * (z - 1000.0) + std::sin(y / 200.0);
        v = -1.0 + 0.5 * std::cos(x / 350.0 + z / 400.0);
        w = 0.3 * std::sin(x / 250.0) * std::exp(-(z - 1000.0) / 800.0);
    });
    cpu.computeGradients();

    gpu::RansSolver gs(*c, GpuGridView(g), cpu.nu, cpu.beta);
    CopyToGpu(cpu, gs);
    gs.computeGradients();

    double worst = 0.0;
    for(int field = 0; field < 4; field++)
        for(int d = 0; d < 3; d++)
        {
            double r = RelativeDiff(gs.gradient(field, d), cpu.gradient(field, d));
            worst = std::max(worst, r);
            BOOST_TEST_MESSAGE("gradient field " << field << " dir " << d << ": " << r);
        }
    BOOST_TEST_MESSAGE("worst relative difference " << worst);
    BOOST_CHECK_LE(worst, 1e-5);   // measured 9.6e-7 on Apple M3 Max
}

BOOST_AUTO_TEST_CASE( kernel_residuals_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    RansGrid g = RidgeGrid(40, 12, 10);
    ::RansSolver cpu(g, 1.58946e-5, 25.0);
    PerturbedWindState(cpu);
    cpu.computeGradients();
    cpu.computeResiduals();

    gpu::RansSolver gs(*c, GpuGridView(g), cpu.nu, cpu.beta);
    CopyToGpu(cpu, gs);
    gs.computeGradients();
    gs.computeResiduals();

    const char *names[9] = {"p", "u", "v", "w", "k", "epsilon", "spectral",
                            "wall epsilon", "wall G"};
    double worst = 0.0;
    for(int slot = 0; slot < 9; slot++)
    {
        double r = RelativeDiff(gs.residual(slot), cpu.residualArray(slot));
        worst = std::max(worst, r);
        BOOST_TEST_MESSAGE("residual " << names[slot] << ": " << r);
    }
    std::vector<double> walls = gs.residual(9);
    for(int i = 0; i < g.nCells(); i++)
        BOOST_CHECK_EQUAL(walls[i], double(cpu.wallFaces()[i]));
    BOOST_TEST_MESSAGE("worst relative difference " << worst);
    BOOST_CHECK_LE(worst, 1e-5);   // measured 3.0e-6 on Apple M3 Max
}

BOOST_AUTO_TEST_CASE( kernel_iterate_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    RansGrid g = RidgeGrid(40, 12, 10);
    ::RansSolver cpu(g, 1.58946e-5, 25.0);
    PerturbedWindState(cpu);
    gpu::RansSolver gs(*c, GpuGridView(g), cpu.nu, cpu.beta);
    CopyToGpu(cpu, gs);

    // one step: the update itself
    ::RansSolver::Residuals rc = cpu.iterate(1, 0.5);
    gpu::RansSolver::Residuals rg = gs.iterate(1, 0.5);
    BOOST_TEST_MESSAGE("after 1 step:");
    double one = FieldsDiff(cpu, gs);
    double norms[6][2] = {{rc.p, rg.p}, {rc.u, rg.u}, {rc.v, rg.v}, {rc.w, rg.w},
                          {rc.k, rg.k}, {rc.epsilon, rg.epsilon}};
    double worstNorm = 0.0;
    for(auto &nrm : norms)
        worstNorm = std::max(worstNorm, std::fabs(nrm[1] / nrm[0] - 1.0));
    BOOST_TEST_MESSAGE("  residual norms: worst relative " << worstNorm);
    BOOST_CHECK_LE(one, 1e-5);
    BOOST_CHECK_LE(worstNorm, 1e-5);

    // 200 more: f32 drift from the f64 CPU stays small
    cpu.iterate(200, 0.5);
    gs.iterate(200, 0.5);
    BOOST_TEST_MESSAGE("after 201 steps:");
    double many = FieldsDiff(cpu, gs);
    BOOST_CHECK_LE(many, 1e-4);   // measured 5.3e-6 on Apple M3 Max
}

BOOST_AUTO_TEST_CASE( iterate_deterministic )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // face passes and cell gathers, no atomics: two runs are bitwise equal
    RansGrid g = RidgeGrid(40, 12, 10);
    ::RansSolver cpu(g, 1.58946e-5, 25.0);
    PerturbedWindState(cpu);
    std::vector<std::vector<double>> runs[2];
    for(auto &run : runs)
    {
        gpu::RansSolver gs(*c, GpuGridView(g), cpu.nu, cpu.beta);
        CopyToGpu(cpu, gs);
        gs.iterate(300, 0.5);
        for(int slot = 0; slot < gpu::fieldSlots; slot++)
            run.push_back(gs.fields.download(gpu::FieldSlot(slot)));
    }
    for(int slot = 0; slot < gpu::fieldSlots; slot++)
        BOOST_CHECK(runs[0][slot] == runs[1][slot]);
}

BOOST_AUTO_TEST_CASE( multigrid_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // the CPU and GPU FAS V-cycles on the same hierarchy and state
    RansGrid g = RidgeGrid(40, 12, 10);
    ::RansSolver cpu(g, 1.58946e-5, 25.0);
    PerturbedWindState(cpu);
    gpu::RansSolver gs(*c, GpuGridView(g), cpu.nu, cpu.beta);
    CopyToGpu(cpu, gs);
    ::RansMultigrid mg(cpu, 4);
    gpu::RansMultigrid gm(gs, GpuMultigridLevels(mg));
    BOOST_REQUIRE_EQUAL(gm.levels(), mg.levels());
    BOOST_TEST_MESSAGE("levels: " << mg.levels());

    mg.cycle(0.5);
    gm.cycle(0.5);
    BOOST_TEST_MESSAGE("after 1 V-cycle:");
    double one = FieldsDiff(cpu, gs);
    BOOST_CHECK_LE(one, 1e-5);

    for(int i = 0; i < 30; i++)
    {
        mg.cycle(0.5);
        gm.cycle(0.5);
    }
    BOOST_TEST_MESSAGE("after 31 V-cycles:");
    double many = FieldsDiff(cpu, gs);
    BOOST_CHECK_LE(many, 1e-4);
}

BOOST_AUTO_TEST_CASE( dem_solve_hill )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // the browser module's self-test problem, natively (see hill_selftest.h)
    DemWind w = SolveDemOnGpu(*c, hill_selftest::Dem(), hill_selftest::Problem(),
                              hill_selftest::Options(),
                              [](const std::string &, bool) {}, []() {});
    hill_selftest::Stats s = hill_selftest::Summarize(w);
    std::printf("hill self-test: mean speed %.9g, max speed %.9g, mean k %.9g\n",
                s.meanSpeed, s.maxSpeed, s.meanK);
    BOOST_CHECK_EQUAL(w.u.size(), size_t(40 * 30));
    BOOST_CHECK(hill_selftest::Matches(s));
}

BOOST_AUTO_TEST_SUITE_END()

#endif
