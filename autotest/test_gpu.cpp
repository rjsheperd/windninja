#ifdef WITH_WEBGPU

#include <algorithm>
#include <chrono>
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
#include "gpu_mass.h"
#include "gpu_rans.h"
#include "gpu_reduce.h"
#include "mass_fem.h"
#include "mass_pcg.h"
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

// A mass-solver grid shaped like WindNinja's: 60 m cells over a Gaussian
// ridge at 1500 m, 20 m first layer growing 1.2x, flat top at 4000 m.
static MassGrid MassRidgeGrid(int ncols, int nrows, int nlayers)
{
    MassGrid g;
    g.ncols = ncols;
    g.nrows = nrows;
    g.nlayers = nlayers;
    const double dx = 60.0, top = 4000.0;
    std::vector<double> frac(nlayers, 0.0);
    double h = 20.0, total = 0.0;
    for(int k = 1; k < nlayers; k++) { total += h; frac[k] = total; h *= 1.2; }
    for(int k = 0; k < nlayers; k++) frac[k] /= total;
    for(int k = 0; k < nlayers; k++)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < ncols; j++)
            {
                double x = j * dx, y = i * dx;
                double cx = 0.5 * (ncols - 1) * dx, cy = 0.5 * (nrows - 1) * dx;
                double ground = 1500.0 + 400.0 * std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) /
                                                          (2.0 * 300.0 * 300.0));
                g.x.push_back(x);
                g.y.push_back(y);
                g.z.push_back(ground + (top - ground) * frac[k]);
            }
    return g;
}

// The system for a log-profile-like wind with some divergence.
static MassSystem MassRidgeSystem(const MassGrid &g)
{
    int n = g.numNodes();
    std::vector<double> u0(n), v0(n), w0(n, 0.0), alphaV(n, 1.0);
    for(int i = 0; i < n; i++)
    {
        u0[i] = 5.0 + 0.002 * g.x[i];
        v0[i] = 1.0 - 0.001 * g.y[i];
    }
    return AssembleMass(g, u0.data(), v0.data(), w0.data(), alphaV.data(), 1.0);
}

// y = A x from the upper-triangle CSR in double, and sum |A_ij x_j| per row
static void CsrMultiply(const MassSystem &s, const std::vector<double> &x,
                        std::vector<double> &y, std::vector<double> &rowAbs)
{
    int n = (int)s.RHS.size();
    y.assign(n, 0.0);
    rowAbs.assign(n, 0.0);
    for(int i = 0; i < n; i++)
        for(int l = s.row_ptr[i]; l < s.row_ptr[i + 1]; l++)
        {
            int j = s.col_ind[l];
            y[i] += s.SK[l] * x[j];
            rowAbs[i] += std::fabs(s.SK[l] * x[j]);
            if(j != i)
            {
                y[j] += s.SK[l] * x[i];
                rowAbs[j] += std::fabs(s.SK[l] * x[i]);
            }
        }
}

// Packing the CSR into the 14-slot stencil keeps every entry (rounded to
// float) and adds none.
BOOST_AUTO_TEST_CASE( mass_stencil_pack )
{
    MassGrid g = MassRidgeGrid(7, 6, 5);
    MassSystem s = MassRidgeSystem(g);
    gpu::MassStencil st = gpu::PackMassStencil(g.ncols, g.nrows, g.nlayers, s.SK.data(),
                                               s.col_ind.data(), s.row_ptr.data());
    const std::size_t n = st.nodes();
    BOOST_REQUIRE_EQUAL(st.coef.size(), 14 * n);
    // rebuild (row, col) -> value from the stencil
    std::size_t stored = 0, matched = 0;
    for(std::size_t row = 0; row < n; row++)
        for(int l = s.row_ptr[row]; l < s.row_ptr[row + 1]; l++)
        {
            int col = s.col_ind[l];
            int dc = col % g.ncols - (int)row % g.ncols;
            int dr = (col / g.ncols) % g.nrows - ((int)row / g.ncols) % g.nrows;
            int dl = col / (g.ncols * g.nrows) - (int)row / (g.ncols * g.nrows);
            int slot = -1;
            if(dc == 0 && dr == 0 && dl == 0)
                slot = 0;
            for(int k = 0; k < 13; k++)
                if(gpu::massOffsets[k][0] == dc && gpu::massOffsets[k][1] == dr &&
                   gpu::massOffsets[k][2] == dl)
                    slot = k + 1;
            BOOST_REQUIRE_GE(slot, 0);
            BOOST_CHECK_EQUAL(st.coef[slot * n + row], (float)s.SK[l]);
            matched++;
        }
    for(float c : st.coef)
        stored += c != 0.0f;
    std::size_t nonzero = 0;
    for(double v : s.SK)
        nonzero += (float)v != 0.0f;
    BOOST_CHECK_EQUAL(stored, nonzero);   // nothing extra
    BOOST_CHECK_EQUAL(matched, s.SK.size());

    // an entry outside the stencil is rejected
    std::vector<int> bad(s.col_ind);
    bad[s.row_ptr[0] + 1] = 2 * g.ncols * g.nrows;   // two layers up
    BOOST_CHECK_THROW(gpu::PackMassStencil(g.ncols, g.nrows, g.nlayers, s.SK.data(), bad.data(),
                                           s.row_ptr.data()),
                      gpu::Error);
}

// The GPU product matches the double CSR product to f32 rounding: within
// 1e-5 of sum |A_ij x_j| per row (seen: 2.7e-7).
BOOST_AUTO_TEST_CASE( kernel_mass_apply_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    MassGrid g = MassRidgeGrid(40, 33, 12);
    MassSystem s = MassRidgeSystem(g);
    int n = g.numNodes();
    std::vector<double> x(n);
    for(int i = 0; i < n; i++)
        x[i] = std::sin(0.37 * i) + 0.5 * std::cos(0.011 * i);   // fixed, varied
    std::vector<double> yRef, rowAbs;
    CsrMultiply(s, x, yRef, rowAbs);

    gpu::MassOperator op(*c, gpu::PackMassStencil(g.ncols, g.nrows, g.nlayers, s.SK.data(),
                                                  s.col_ind.data(), s.row_ptr.data()));
    std::vector<float> xf(x.begin(), x.end());
    gpu::Buffer xb(*c, n * sizeof(float)), yb(*c, n * sizeof(float));
    xb.upload(xf);
    op.apply(xb, yb);
    std::vector<float> y = yb.download<float>(n);
    double worst = 0.0;
    for(int i = 0; i < n; i++)
        worst = std::max(worst, std::fabs(y[i] - yRef[i]) / rowAbs[i]);
    BOOST_TEST_MESSAGE("mass apply: worst |error| / sum|A x| = " << worst);
    BOOST_CHECK_LT(worst, 1e-5);
}

// Not in ctest: times y = A x on a fine-mesh-sized grid (137 x 147 x 20).
// Run with --run_test=gpu/mass_apply_timing --log_level=message.
BOOST_AUTO_TEST_CASE( mass_apply_timing )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    MassGrid g = MassRidgeGrid(137, 147, 20);
    MassSystem s = MassRidgeSystem(g);
    int n = g.numNodes();
    gpu::MassOperator op(*c, gpu::PackMassStencil(g.ncols, g.nrows, g.nlayers, s.SK.data(),
                                                  s.col_ind.data(), s.row_ptr.data()));
    gpu::Buffer xb(*c, n * sizeof(float)), yb(*c, n * sizeof(float));
    xb.upload(std::vector<float>(n, 1.0f));
    op.apply(xb, yb);
    c->finish();
    const int reps = 200;
    auto t0 = std::chrono::steady_clock::now();
    for(int r = 0; r < reps; r++)
        op.apply(xb, yb);
    c->finish();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    BOOST_TEST_MESSAGE("mass apply: " << n << " nodes, " << ms / reps << " ms per apply ("
                                      << c->adapterName() << ")");
}

// an interior node (not on a Dirichlet boundary)
static int Node0Interior(const MassGrid &g)
{
    return 1 + g.ncols * (1 + g.nrows * 1);
}

static gpu::MassStencil Stencil(const MassGrid &g, const MassSystem &s)
{
    return gpu::PackMassStencil(g.ncols, g.nrows, g.nlayers, s.SK.data(), s.col_ind.data(),
                                s.row_ptr.data());
}

// ||A x - b|| / ||b|| in double
static double TrueResidual(const MassSystem &s, const std::vector<double> &x)
{
    std::vector<double> ax, rowAbs;
    CsrMultiply(s, x, ax, rowAbs);
    double r = 0.0, b = 0.0;
    for(std::size_t i = 0; i < ax.size(); i++)
    {
        r += (ax[i] - s.RHS[i]) * (ax[i] - s.RHS[i]);
        b += s.RHS[i] * s.RHS[i];
    }
    return std::sqrt(r / b);
}

// Max |difference| of the recovered horizontal wind, relative to the max
// initial wind speed.
static double WindDifference(const MassGrid &g, const std::vector<double> &phiA,
                             const std::vector<double> &phiB)
{
    int n = g.numNodes();
    std::vector<double> u0(n), v0(n), w0(n, 0.0), alphaV(n, 1.0);
    for(int i = 0; i < n; i++)
    {
        u0[i] = 5.0 + 0.002 * g.x[i];
        v0[i] = 1.0 - 0.001 * g.y[i];
    }
    std::vector<double> ua(n), va(n), wa(n), ub(n), vb(n), wb(n);
    RecoverVelocity(g, phiA.data(), u0.data(), v0.data(), w0.data(), alphaV.data(), 1.0,
                    ua.data(), va.data(), wa.data());
    RecoverVelocity(g, phiB.data(), u0.data(), v0.data(), w0.data(), alphaV.data(), 1.0,
                    ub.data(), vb.data(), wb.data());
    double worst = 0.0, speed = 0.0;
    for(int i = 0; i < n; i++)
    {
        worst = std::max(worst, std::hypot(ua[i] - ub[i], va[i] - vb[i]));
        speed = std::max(speed, std::hypot(u0[i], v0[i]));
    }
    return worst / speed;
}

// The GPU's f32 Jacobi-PCG against the CPU's double one: at a tight
// tolerance both reach the true residual, and at WindNinja's 0.1 the
// recovered wind differs by about as much as two CPU preconditioners do.
BOOST_AUTO_TEST_CASE( mass_pcg_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    MassGrid g = MassRidgeGrid(40, 33, 12);
    MassSystem s = MassRidgeSystem(g);
    int n = g.numNodes();
    gpu::MassPcg pcg(*c, Stencil(g, s));

    // tight: the f32 recurrence and the true residual agree
    std::vector<double> x(n, 0.0);
    gpu::MassPcg::Result r = pcg.solve(s.RHS, x, 1e-4, 2000, 10);
    BOOST_TEST_MESSAGE("gpu pcg 1e-4: " << r.iterations << " iterations, residual "
                       << r.residual << ", true " << TrueResidual(s, x));
    BOOST_CHECK(r.converged);
    BOOST_CHECK_LT(TrueResidual(s, x), 2e-4);

    // WindNinja's tolerance, against the CPU's Jacobi and SSOR
    MassSolveOptions options;
    options.jacobi = true;
    std::vector<double> cpuJacobi(n, 0.0), cpuSsor(n, 0.0), gpuPhi(n, 0.0);
    MassSolveResult cj = SolveMassPcg(s, cpuJacobi.data(), options);
    options.jacobi = false;
    SolveMassPcg(s, cpuSsor.data(), options);
    gpu::MassPcg::Result rg = pcg.solve(s.RHS, gpuPhi, 0.1, 2000, 10);
    double dGpu = WindDifference(g, gpuPhi, cpuJacobi);
    double dCpu = WindDifference(g, cpuSsor, cpuJacobi);
    BOOST_TEST_MESSAGE("tol 0.1: cpu jacobi " << cj.iterations << " iterations, gpu "
                       << rg.iterations << "; wind gpu-cpu " << dGpu << ", ssor-jacobi " << dCpu);
    BOOST_CHECK(rg.converged);
    BOOST_CHECK_LE(rg.iterations, cj.iterations + 10);   // checks every 10
    BOOST_CHECK_LT(TrueResidual(s, gpuPhi), 0.1);
    BOOST_CHECK_LT(dGpu, 2.0 * dCpu);

    // warm start from the CPU's answer: r = b - A x, not b
    std::vector<double> warm(cpuJacobi);
    gpu::MassPcg::Result rw = pcg.solve(s.RHS, warm, 1e-3, 2000, 1);
    BOOST_TEST_MESSAGE("warm start to 1e-3: " << rw.iterations << " iterations, true "
                       << TrueResidual(s, warm));
    BOOST_CHECK(rw.converged);
    BOOST_CHECK_LT(TrueResidual(s, warm), 2e-3);

    // same start, same answer: deterministic
    std::vector<double> again(n, 0.0);
    pcg.solve(s.RHS, again, 0.1, 2000, 10);
    BOOST_CHECK(again == gpuPhi);
}

// A NaN in the right-hand side stops the solve with an error, not a
// "converged" result (NaN > tol is false).
BOOST_AUTO_TEST_CASE( mass_pcg_nan_throws )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    MassGrid g = MassRidgeGrid(9, 8, 5);
    MassSystem s = MassRidgeSystem(g);
    gpu::MassPcg pcg(*c, Stencil(g, s));
    std::vector<double> b(s.RHS), x(g.numNodes(), 0.0);
    b[Node0Interior(g)] = std::nan("");
    BOOST_CHECK_THROW(pcg.solve(b, x, 0.1, 100, 1), gpu::Error);
}

// GPU assembly against the CPU's AssembleMass, on the ridge grid moved to
// UTM-sized, non-integer coordinates (without the local origin the error is
// 3e-4), with a varying alphaV and all three wind components.  Each
// coefficient within 1e-5 of its row's largest, each RHS within 1e-4 of the
// largest RHS (seen: 2.6e-6 and 4.9e-6).
BOOST_AUTO_TEST_CASE( kernel_mass_assemble_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    MassGrid g = MassRidgeGrid(40, 33, 12);
    for(int i = 0; i < g.numNodes(); i++)
    {
        // not integers: f32 holds integers this size exactly
        g.x[i] += 512345.37 + 0.11 * std::sin(0.7 * i);
        g.y[i] += 4812345.61 + 0.07 * std::cos(0.3 * i);
    }
    int n = g.numNodes();
    std::vector<double> u0(n), v0(n), w0(n), alphaV(n);
    for(int i = 0; i < n; i++)
    {
        u0[i] = 5.0 + 0.002 * (g.x[i] - g.x[0]) + 0.001 * (g.z[i] - 1500.0);
        v0[i] = 1.0 - 0.001 * (g.y[i] - g.y[0]);
        w0[i] = 0.2 * std::sin(0.01 * i);
        alphaV[i] = 0.8 + 0.4 * std::cos(0.003 * i);
    }
    const double alphaH = 1.3;
    MassSystem s = AssembleMass(g, u0.data(), v0.data(), w0.data(), alphaV.data(), alphaH);
    gpu::MassStencil ref = gpu::PackMassStencil(g.ncols, g.nrows, g.nlayers, s.SK.data(),
                                                s.col_ind.data(), s.row_ptr.data());

    gpu::MassFem fem(*c, gpu::MakeMassGeometry(g.ncols, g.nrows, g.nlayers,
                                                           g.x.data(), g.y.data(), g.z.data()));
    std::vector<float> wind(3 * n), av(alphaV.begin(), alphaV.end());
    for(int i = 0; i < n; i++)
    {
        wind[i] = (float)u0[i];
        wind[n + i] = (float)v0[i];
        wind[2 * n + i] = (float)w0[i];
    }
    gpu::Buffer windB(*c, 3 * n * sizeof(float)), alphaB(*c, n * sizeof(float)),
        coefB(*c, 14 * n * sizeof(float)), rhsB(*c, n * sizeof(float));
    windB.upload(wind);
    alphaB.upload(av);
    fem.assemble(windB, alphaB, (float)alphaH, coefB, rhsB);
    std::vector<float> coef = coefB.download<float>(14 * n), rhs = rhsB.download<float>(n);

    double worstCoef = 0.0, worstRhs = 0.0, rhsMax = 0.0;
    for(int i = 0; i < n; i++)
        rhsMax = std::max(rhsMax, std::fabs(s.RHS[i]));
    for(int i = 0; i < n; i++)
    {
        double rowMax = 0.0;
        for(int k = 0; k < 14; k++)
            rowMax = std::max(rowMax, (double)std::fabs(ref.coef[k * n + i]));
        for(int k = 0; k < 14; k++)
            worstCoef = std::max(worstCoef, std::fabs(coef[k * n + i] - ref.coef[k * n + i]) / rowMax);
        worstRhs = std::max(worstRhs, std::fabs(rhs[i] - s.RHS[i]) / rhsMax);
    }
    BOOST_TEST_MESSAGE("mass assemble: worst coefficient error / row max " << worstCoef
                       << ", worst rhs error / rhs max " << worstRhs);
    BOOST_CHECK_LT(worstCoef, 1e-5);
    BOOST_CHECK_LT(worstRhs, 1e-4);
}

// GPU wind recovery against the CPU's RecoverVelocity, from the CPU's Phi
// for a divergent wind on the same UTM-placed ridge grid: each component
// within 1e-5 of the largest initial speed (seen: 9e-7).
BOOST_AUTO_TEST_CASE( kernel_mass_recover_matches_cpu )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    MassGrid g = MassRidgeGrid(40, 33, 12);
    for(int i = 0; i < g.numNodes(); i++)
    {
        g.x[i] += 512345.37 + 0.11 * std::sin(0.7 * i);
        g.y[i] += 4812345.61 + 0.07 * std::cos(0.3 * i);
    }
    int n = g.numNodes();
    std::vector<double> u0(n), v0(n), w0(n), alphaV(n);
    for(int i = 0; i < n; i++)
    {
        u0[i] = 5.0 + 0.002 * (g.x[i] - g.x[0]);
        v0[i] = 1.0 - 0.001 * (g.y[i] - g.y[0]);
        w0[i] = 0.1 * std::sin(0.01 * i);
        alphaV[i] = 0.8 + 0.4 * std::cos(0.003 * i);
    }
    const double alphaH = 1.3;
    MassSystem s = AssembleMass(g, u0.data(), v0.data(), w0.data(), alphaV.data(), alphaH);
    std::vector<double> phi(n, 0.0);
    MassSolveOptions options;
    options.tol = 1e-6;
    SolveMassPcg(s, phi.data(), options);
    std::vector<double> u(n), v(n), w(n);
    RecoverVelocity(g, phi.data(), u0.data(), v0.data(), w0.data(), alphaV.data(), alphaH,
                    u.data(), v.data(), w.data());

    gpu::MassFem fem(*c, gpu::MakeMassGeometry(g.ncols, g.nrows, g.nlayers, g.x.data(),
                                               g.y.data(), g.z.data()));
    std::vector<float> wind0(3 * n), phiF(phi.begin(), phi.end()), av(alphaV.begin(), alphaV.end());
    for(int i = 0; i < n; i++)
    {
        wind0[i] = (float)u0[i];
        wind0[n + i] = (float)v0[i];
        wind0[2 * n + i] = (float)w0[i];
    }
    gpu::Buffer phiB(*c, n * sizeof(float)), wind0B(*c, 3 * n * sizeof(float)),
        alphaB(*c, n * sizeof(float)), windB(*c, 3 * n * sizeof(float));
    phiB.upload(phiF);
    wind0B.upload(wind0);
    alphaB.upload(av);
    fem.recover(phiB, wind0B, alphaB, (float)alphaH, windB);
    std::vector<float> wind = windB.download<float>(3 * n);

    double speed = 0.0, worst = 0.0, change = 0.0;
    for(int i = 0; i < n; i++)
    {
        speed = std::max(speed, std::sqrt(u0[i] * u0[i] + v0[i] * v0[i] + w0[i] * w0[i]));
        change = std::max(change, std::fabs(u[i] - u0[i]));
        worst = std::max({worst, std::fabs(wind[i] - u[i]), std::fabs(wind[n + i] - v[i]),
                          std::fabs(wind[2 * n + i] - w[i])});
    }
    BOOST_TEST_MESSAGE("mass recover: worst |error| / max speed " << worst / speed
                       << " (the solve changed u by up to " << change / speed << ")");
    BOOST_REQUIRE_GT(change / speed, 1e-2);   // Phi does something here
    BOOST_CHECK_LT(worst / speed, 1e-5);
}

// Not in ctest: the whole GPU solve at the fine-mesh size, against the CPU
// solver on the same system.  --run_test=gpu/mass_pcg_timing --log_level=message
BOOST_AUTO_TEST_CASE( mass_pcg_timing )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    MassGrid g = MassRidgeGrid(137, 147, 20);
    MassSystem s = MassRidgeSystem(g);
    int n = g.numNodes();
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = now();
    gpu::MassPcg pcg(*c, Stencil(g, s));
    auto t1 = now();
    std::vector<double> warm(n, 0.0);
    pcg.solve(s.RHS, warm, 0.1, 5000, 10);
    for(int every : {10, 25})
    {
        std::vector<double> x(n, 0.0);
        auto a = now();
        gpu::MassPcg::Result r = pcg.solve(s.RHS, x, 0.1, 5000, every);
        auto b = now();
        BOOST_TEST_MESSAGE("gpu jacobi-pcg, check every " << every << ": " << r.iterations
                           << " iterations, " << ms(a, b) << " ms (setup " << ms(t0, t1)
                           << " ms), true residual " << TrueResidual(s, x));
    }
    for(bool jacobi : {true, false})
    {
        MassSolveOptions options;
        options.jacobi = jacobi;
        std::vector<double> x(n, 0.0);
        auto a = now();
        MassSolveResult r = SolveMassPcg(s, x.data(), options);
        auto b = now();
        BOOST_TEST_MESSAGE("cpu " << (jacobi ? "jacobi" : "ssor") << "-pcg: " << r.iterations
                           << " iterations, " << ms(a, b) << " ms");
    }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
