/*
** WindNinja's momentum solver in the browser (WEBGPU_PLAN 3.4): the RANS
** solver on WebGPU, from a DEM given as an array.  It is SolveDemOnGpu, as
** the native WebGPU backend runs, with nothing from WindNinja core.
**
** Also the mass-consistent solver (MASS_GPU_PLAN.org 3.1), on the GPU
** (SolveMassWindOnGpu) or the CPU (AssembleMass, SolveMassPcg,
** RecoverVelocity), from a grid given as arrays.
**
** main() runs the self-tests (hill_selftest.h, mass_selftest.h), times the
** mass solver on the CPU and the GPU at a fine-mesh size, and prints
** "RESULT PASS", "RESULT FAIL ..." or "RESULT SKIP ...".  Pages call
** windninja_solve and windninja_mass_solve.
*/

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <emscripten.h>

#include "gpu_context.h"
#include "hill_selftest.h"
#include "mass_gpu_solve.h"
#include "mass_pcg.h"
#include "mass_selftest.h"
#include "rans_dem_solve.h"

namespace {

std::unique_ptr<gpu::Context> context;

void Log(const std::string &message, bool warning)
{
    std::printf("%s%s\n", warning ? "warning: " : "", message.c_str());
}

// The mass solve on the CPU, as WindNinja does by default (SSOR).
MassSolveResult MassOnCpu(const MassGrid &g, const double *u0, const double *v0, const double *w0,
                          const double *alphaV, double alphaH, double *u, double *v, double *w)
{
    MassSystem s = AssembleMass(g, u0, v0, w0, alphaV, alphaH);
    std::vector<double> phi(g.numNodes(), 0.0);
    MassSolveResult r = SolveMassPcg(s, phi.data(), MassSolveOptions());
    RecoverVelocity(g, phi.data(), u0, v0, w0, alphaV, alphaH, u, v, w);
    return r;
}

MassSolveResult MassOnGpu(const MassGrid &g, const double *u0, const double *v0, const double *w0,
                          const double *alphaV, double alphaH, double *u, double *v, double *w)
{
    return SolveMassWindOnGpu(g, u0, v0, w0, alphaV, alphaH, MassSolveOptions(), u, v, w);
}

} // namespace

extern "C" {

/*
** Solve the wind over a DEM.  elevation: nRows * nCols metres, row 0 north
** (image order); cellSize metres (also the mesh resolution); speed m/s at
** inputHeight m above the canopy, from direction (degrees clockwise from
** north, where the wind comes from), over roughness z0 m.  Writes the
** ground-cell wind (u east, v north, m/s; k m^2/s^2) per DEM cell in the
** same order, at half a cell above ground.  maxSteps caps the iterations.
** Returns 0 on success, 1 if the solution diverged, 2 without WebGPU, 3 on
** another error (printed).
*/
EMSCRIPTEN_KEEPALIVE int windninja_solve(int nCols, int nRows, double cellSize,
                                         const double *elevation, double speed,
                                         double inputHeight, double direction, double z0,
                                         int maxSteps, double *u, double *v, double *k)
{
    try
    {
        if(!context)
            context.reset(new gpu::Context());
        TerrainDem dem;
        dem.nCols = nCols;
        dem.nRows = nRows;
        dem.cellSize = cellSize;
        dem.z.resize(std::size_t(nCols) * nRows);
        for(int r = 0; r < nRows; r++)   // TerrainDem's row 0 is south
            for(int c = 0; c < nCols; c++)
                dem.z[std::size_t(r) * nCols + c] = elevation[std::size_t(nRows - 1 - r) * nCols + c];
        DemWindProblem p;
        p.speed = speed;
        p.inputHeight = inputHeight;
        p.direction = direction;
        p.z0 = z0;
        p.meshResolution = cellSize;
        GpuSolveOptions options;
        options.maxIterations = maxSteps;
        DemWind w = SolveDemOnGpu(*context, dem, p, options, Log, []() {});
        for(int r = 0; r < nRows; r++)
            for(int c = 0; c < nCols; c++)
            {
                std::size_t from = std::size_t(r) * nCols + c;
                std::size_t to = std::size_t(nRows - 1 - r) * nCols + c;
                u[to] = w.u[from];
                v[to] = w.v[from];
                k[to] = w.k[from];
            }
        return 0;
    }
    catch(const gpu::Unavailable &e)
    {
        std::printf("no WebGPU: %s\n", e.what());
        return 2;
    }
    catch(const RansDivergedError &e)
    {
        std::printf("%s\n", e.what());
        return 1;
    }
    catch(const std::exception &e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}

/*
** The mass-consistent solve on a ncols x nrows x nlayers terrain-following
** grid: x, y, z (m), u0, v0, w0 (m/s) and alphaV per node, node
** col + ncols * (row + nrows * layer), and alphaH.  Writes u, v, w per node.
** gpu: 1 on the GPU (Jacobi, f32), 0 on the CPU (SSOR, f64).  Returns the
** iteration count, -1 if it didn't converge, -2 without WebGPU, -3 on
** another error (printed).
*/
EMSCRIPTEN_KEEPALIVE int windninja_mass_solve(int ncols, int nrows, int nlayers, const double *x,
                                              const double *y, const double *z, const double *u0,
                                              const double *v0, const double *w0,
                                              const double *alphaV, double alphaH, int gpu,
                                              double *u, double *v, double *w)
{
    try
    {
        MassGrid g;
        g.ncols = ncols;
        g.nrows = nrows;
        g.nlayers = nlayers;
        g.x.assign(x, x + g.numNodes());
        g.y.assign(y, y + g.numNodes());
        g.z.assign(z, z + g.numNodes());
        MassSolveResult r = gpu ? MassOnGpu(g, u0, v0, w0, alphaV, alphaH, u, v, w)
                                : MassOnCpu(g, u0, v0, w0, alphaV, alphaH, u, v, w);
        return r.converged ? r.iterations : -1;
    }
    catch(const gpu::Unavailable &e)
    {
        std::printf("no WebGPU: %s\n", e.what());
        return -2;
    }
    catch(const std::exception &e)
    {
        std::printf("error: %s\n", e.what());
        return -3;
    }
}

} // extern "C"

namespace {

// The mass self-test (mass_selftest.h) on the GPU and the CPU; true if both
// match the native statistics.
bool MassSelfTest()
{
    MassGrid g = mass_selftest::Grid();
    mass_selftest::Wind in = mass_selftest::Initial(g);
    std::vector<double> u(g.numNodes()), v(g.numNodes()), w(g.numNodes());
    bool pass = true;
    for(int gpu = 1; gpu >= 0; gpu--)
    {
        if(gpu)
            MassOnGpu(g, in.u0.data(), in.v0.data(), in.w0.data(), in.alphaV.data(), in.alphaH,
                      u.data(), v.data(), w.data());
        else
            MassOnCpu(g, in.u0.data(), in.v0.data(), in.w0.data(), in.alphaV.data(), in.alphaH,
                      u.data(), v.data(), w.data());
        mass_selftest::Stats s = mass_selftest::Summarize(g, u, v, w);
        const mass_selftest::Stats &e = gpu ? mass_selftest::expectedGpu : mass_selftest::expectedCpu;
        std::printf("mass self-test %s: mean speed %.9g, max speed %.9g, mean w %.9g\n",
                    gpu ? "gpu" : "cpu", s.meanSpeed, s.maxSpeed, s.meanW);
        std::printf("native:             mean speed %.9g, max speed %.9g, mean w %.9g\n",
                    e.meanSpeed, e.maxSpeed, e.meanW);
        pass = pass && mass_selftest::Matches(s, e, gpu ? mass_selftest::gpuTolerance
                                                        : mass_selftest::cpuTolerance);
    }
    return pass;
}

// The mass solve at a fine-mesh size (137 x 147 x 20, as Big Butte fine):
// whole solve (system, Phi, wind), CPU (SSOR, one thread) and GPU.
void MassTiming()
{
    MassGrid g = mass_selftest::Grid(137, 147, 20);
    mass_selftest::Wind in = mass_selftest::Initial(g);
    std::vector<double> u(g.numNodes()), v(g.numNodes()), w(g.numNodes());
    for(int gpu = 1; gpu >= 0; gpu--)
    {
        if(gpu)   // once to warm up: context, pipelines
            MassOnGpu(g, in.u0.data(), in.v0.data(), in.w0.data(), in.alphaV.data(), in.alphaH,
                      u.data(), v.data(), w.data());
        double t0 = emscripten_get_now();
        MassSolveResult r = gpu ? MassOnGpu(g, in.u0.data(), in.v0.data(), in.w0.data(),
                                            in.alphaV.data(), in.alphaH, u.data(), v.data(), w.data())
                                : MassOnCpu(g, in.u0.data(), in.v0.data(), in.w0.data(),
                                            in.alphaV.data(), in.alphaH, u.data(), v.data(), w.data());
        double ms = emscripten_get_now() - t0;
        std::printf("mass timing %s: %d nodes, %d iterations, %.1f ms\n", gpu ? "gpu" : "cpu",
                    g.numNodes(), r.iterations, ms);
    }
}

} // namespace

int main()
{
    try
    {
        context.reset(new gpu::Context());
        std::printf("adapter: %s\n", context->adapterName().c_str());
        DemWind w = SolveDemOnGpu(*context, hill_selftest::Dem(), hill_selftest::Problem(),
                                  hill_selftest::Options(), Log, []() {});
        hill_selftest::Stats s = hill_selftest::Summarize(w);
        std::printf("hill self-test: mean speed %.9g, max speed %.9g, mean k %.9g\n",
                    s.meanSpeed, s.maxSpeed, s.meanK);
        std::printf("native:         mean speed %.9g, max speed %.9g, mean k %.9g\n",
                    hill_selftest::expected.meanSpeed, hill_selftest::expected.maxSpeed,
                    hill_selftest::expected.meanK);
        if(!hill_selftest::Matches(s))
        {
            std::printf("RESULT FAIL differs from the native solver\n");
            return 1;
        }
        if(!MassSelfTest())
        {
            std::printf("RESULT FAIL the mass solver differs from the native one\n");
            return 1;
        }
        MassTiming();
    }
    catch(const gpu::Unavailable &e)
    {
        std::printf("RESULT SKIP %s\n", e.what());
        return 0;
    }
    catch(const std::exception &e)
    {
        std::printf("RESULT FAIL %s\n", e.what());
        return 1;
    }
    std::printf("RESULT PASS\n");
    return 0;
}
