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
#include <stdexcept>
#include <string>
#include <vector>

#include <emscripten.h>

#include "gpu_context.h"
#include "hill_selftest.h"
#include "mass_dem.h"
#include "mass_gpu_solve.h"
#include "mass_pcg.h"
#include "mass_selftest.h"
#include "rans_dem_solve.h"

namespace {

std::unique_ptr<gpu::Context> context;
WGPUInstance sharedInstance = nullptr;      // the parent of an imported page device
std::unique_ptr<gpu::Buffer> keptWind;      // windninja_mass_solve_kept's result

// The module's context: the page's device if shared, else its own.
gpu::Context &Gpu()
{
    if(!context)
        context.reset(new gpu::Context());
    return *context;
}

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
    return SolveMassWindOnGpu(Gpu(), g, u0, v0, w0, alphaV, alphaH, MassSolveOptions(), u, v, w);
}

// SolveMassWindOnGpu as a MassWindSolver (for SolveMassDem)
MassSolveResult GpuWindSolver(const MassGrid &g, const double *u0, const double *v0,
                              const double *w0, const double *alphaV, double alphaH,
                              const MassSolveOptions &options, double *u, double *v, double *w)
{
    return SolveMassWindOnGpu(Gpu(), g, u0, v0, w0, alphaV, alphaH, options, u, v, w);
}

MassGrid GridFrom(int ncols, int nrows, int nlayers, const double *x, const double *y,
                  const double *z)
{
    MassGrid g;
    g.ncols = ncols;
    g.nrows = nrows;
    g.nlayers = nlayers;
    g.x.assign(x, x + g.numNodes());
    g.y.assign(y, y + g.numNodes());
    g.z.assign(z, z + g.numNodes());
    return g;
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
        MassGrid g = GridFrom(ncols, nrows, nlayers, x, y, z);
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

/*
** Sharing the page's GPUDevice (MASS_GPU_PLAN.org 3.2), so the page (a fire
** model, say) uses the wind where it lies on the GPU:
**   const dev = Module.WebGPU.importJsDevice(device, Module._windninja_gpu_instance());
**   Module._windninja_gpu_share_device(dev);
** windninja_gpu_instance returns the module's instance, the imported device's
** parent (its futures are waited on through it).  After sharing, every GPU
** solve here runs on the page's device.  Returns 0, or 3 on error.
*/
EMSCRIPTEN_KEEPALIVE WGPUInstance windninja_gpu_instance()
{
    if(sharedInstance == nullptr)
        sharedInstance = gpu::Context::CreateInstance();
    return sharedInstance;
}

EMSCRIPTEN_KEEPALIVE int windninja_gpu_share_device(WGPUDevice device)
{
    try
    {
        keptWind.reset();
        context.reset(new gpu::Context(windninja_gpu_instance(), device, "page device"));
        return 0;
    }
    catch(const std::exception &e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}

/*
** windninja_mass_solve on the GPU, leaving the wind there: returns the
** WGPUBuffer (Module.WebGPU.getJsObject(ptr) is the GPUBuffer) holding
** 3 * nodes f32, the u, v, w slots (slot s of node n at s * nodes + n), or 0
** on failure (printed).  It stays valid until the next call.
*/
EMSCRIPTEN_KEEPALIVE WGPUBuffer windninja_mass_solve_kept(int ncols, int nrows, int nlayers,
                                                          const double *x, const double *y,
                                                          const double *z, const double *u0,
                                                          const double *v0, const double *w0,
                                                          const double *alphaV, double alphaH)
{
    try
    {
        keptWind.reset();
        MassSolveResult r;
        keptWind = SolveMassWindKeptOnGpu(Gpu(), GridFrom(ncols, nrows, nlayers, x, y, z), u0, v0,
                                          w0, alphaV, alphaH, MassSolveOptions(), &r);
        if(!r.converged)
        {
            std::printf("error: the mass solve didn't converge\n");
            keptWind.reset();
            return nullptr;
        }
        return keptWind->handle();
    }
    catch(const std::exception &e)
    {
        std::printf("error: %s\n", e.what());
        return nullptr;
    }
}

/*
** WindNinja's domain-average mass solve from a DEM (SolveMassDem): what the
** CLI writes as its output-height ASCII grids.  elevation: nRows * nCols m,
** row 0 north (image order), at cellSize m, which is also the mesh
** resolution (resample first for coarse/medium/fine).  speed m/s at
** inputHeight m above the canopy, from direction degrees (geographic);
** vegetation 0 grass, 1 brush, 2 trees; latitude (degrees, or outside
** +-90 for none) and angleFromNorth (degrees, the DEM grid's, as WindNinja's
** GDALCalculateAngleFromNorth).  Writes speed (m/s) and angle (degrees the
** wind comes from, clockwise from the grid's north) per cell at
** outputHeight m above the canopy, row 0 north.  gpu: 1 GPU, 0 CPU.
** Returns the iteration count, -1 not converged, -2 no WebGPU, -3 error.
*/
EMSCRIPTEN_KEEPALIVE int windninja_mass_dem_solve(int nCols, int nRows, double cellSize,
                                                  const double *elevation, double speed,
                                                  double direction, double inputHeight,
                                                  double outputHeight, int vegetation,
                                                  double latitude, double angleFromNorth, int gpu,
                                                  double *speedOut, double *angleOut)
{
    try
    {
        TerrainDem dem;
        dem.nCols = nCols;
        dem.nRows = nRows;
        dem.cellSize = cellSize;
        dem.z.resize(std::size_t(nCols) * nRows);
        for(int r = 0; r < nRows; r++)   // TerrainDem's row 0 is south
            for(int c = 0; c < nCols; c++)
                dem.z[std::size_t(r) * nCols + c] = elevation[std::size_t(nRows - 1 - r) * nCols + c];
        MassDemProblem p;
        p.wind.speed = speed;
        p.wind.direction = direction;
        p.wind.inputHeight = inputHeight;
        p.wind.latitude = latitude;
        p.wind.angleFromNorth = angleFromNorth;
        const double z0[] = {0.01, 0.43, 1.0}, h[] = {0.0, 2.3, 15.4}, d[] = {0.0, 1.8, 12.0};
        if(vegetation < 0 || vegetation > 2)
            throw std::invalid_argument("vegetation must be 0 (grass), 1 (brush) or 2 (trees)");
        p.wind.roughness = z0[vegetation];
        p.wind.roughH = h[vegetation];
        p.wind.roughD = d[vegetation];
        p.outputHeight = outputHeight;
        MassDemWind w = gpu ? SolveMassDem(dem, p, MassSolveOptions(), GpuWindSolver)
                            : SolveMassDem(dem, p, MassSolveOptions());
        for(int r = 0; r < nRows; r++)
            for(int c = 0; c < nCols; c++)
            {
                std::size_t from = std::size_t(r) * nCols + c;
                std::size_t to = std::size_t(nRows - 1 - r) * nCols + c;
                speedOut[to] = w.speed[from];
                angleOut[to] = w.angle[from];
            }
        return w.solve.iterations;
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

// The DEM-in self-test (mass_selftest.h) through windninja_mass_dem_solve
// itself (row order, vegetation), on the GPU and the CPU; true if both match
// the native statistics (and so the CLI: scripts/check_mass_dem.sh).
bool MassDemSelfTest()
{
    TerrainDem dem = mass_selftest::Dem();
    MassDemProblem p = mass_selftest::DemProblem();
    const int nCols = dem.nCols, nRows = dem.nRows;
    std::vector<double> elevation(dem.z.size()), speed(dem.z.size()), angle(dem.z.size());
    for(int r = 0; r < nRows; r++)   // the export takes row 0 north
        for(int c = 0; c < nCols; c++)
            elevation[std::size_t(nRows - 1 - r) * nCols + c] = dem.z[std::size_t(r) * nCols + c];
    bool pass = true;
    for(int gpu = 1; gpu >= 0; gpu--)
    {
        int iterations = windninja_mass_dem_solve(nCols, nRows, dem.cellSize, elevation.data(),
                                                  p.wind.speed, p.wind.direction, p.wind.inputHeight,
                                                  p.outputHeight, 0, p.wind.latitude,
                                                  p.wind.angleFromNorth, gpu, speed.data(),
                                                  angle.data());
        if(iterations < 0)
            return false;
        MassDemWind w;   // back to row 0 south
        w.nCols = nCols;
        w.nRows = nRows;
        w.speed.resize(speed.size());
        w.angle.resize(angle.size());
        for(int r = 0; r < nRows; r++)
            for(int c = 0; c < nCols; c++)
            {
                w.speed[std::size_t(r) * nCols + c] = speed[std::size_t(nRows - 1 - r) * nCols + c];
                w.angle[std::size_t(r) * nCols + c] = angle[std::size_t(nRows - 1 - r) * nCols + c];
            }
        mass_selftest::DemStats s = mass_selftest::SummarizeDem(w);
        const mass_selftest::DemStats &e = gpu ? mass_selftest::expectedDemGpu
                                               : mass_selftest::expectedDemCpu;
        std::printf("mass dem self-test %s: mean speed %.9g, max speed %.9g, mean angle %.9g, "
                    "SW %.9g, NE %.9g\n", gpu ? "gpu" : "cpu", s.meanSpeed, s.maxSpeed, s.meanAngle,
                    s.speedSouthWest, s.speedNorthEast);
        std::printf("native:                 mean speed %.9g, max speed %.9g, mean angle %.9g, "
                    "SW %.9g, NE %.9g\n", e.meanSpeed, e.maxSpeed, e.meanAngle, e.speedSouthWest,
                    e.speedNorthEast);
        pass = pass && mass_selftest::MatchesDem(s, e, gpu ? mass_selftest::gpuTolerance
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
        if(!MassDemSelfTest())
        {
            std::printf("RESULT FAIL the mass solver from a DEM differs from the native one\n");
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
