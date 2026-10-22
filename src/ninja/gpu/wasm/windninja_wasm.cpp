/*
** WindNinja's momentum solver in the browser (WEBGPU_PLAN 3.4): the RANS
** solver on WebGPU, from a DEM given as an array.  It is SolveDemOnGpu, as
** the native WebGPU backend runs, with nothing from WindNinja core.
**
** main() runs a self-test (hill_selftest.h) and prints "RESULT PASS",
** "RESULT FAIL ..." or "RESULT SKIP ...".  Pages call windninja_solve.
*/

#include <cstdio>
#include <memory>
#include <string>

#include <emscripten.h>

#include "gpu_context.h"
#include "hill_selftest.h"
#include "rans_dem_solve.h"

namespace {

std::unique_ptr<gpu::Context> context;

void Log(const std::string &message, bool warning)
{
    std::printf("%s%s\n", warning ? "warning: " : "", message.c_str());
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

} // extern "C"

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
