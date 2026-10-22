/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Momentum backend running the RANS solver on WebGPU
 *
 ******************************************************************************
 *
 * THIS SOFTWARE WAS DEVELOPED AT THE ROCKY MOUNTAIN RESEARCH STATION (RMRS)
 * MISSOULA FIRE SCIENCES LABORATORY BY EMPLOYEES OF THE FEDERAL GOVERNMENT
 * IN THE COURSE OF THEIR OFFICIAL DUTIES. PURSUANT TO TITLE 17 SECTION 105
 * OF THE UNITED STATES CODE, THIS SOFTWARE IS NOT SUBJECT TO COPYRIGHT
 * PROTECTION AND IS IN THE PUBLIC DOMAIN. RMRS MISSOULA FIRE SCIENCES
 * LABORATORY ASSUMES NO RESPONSIBILITY WHATSOEVER FOR ITS USE BY OTHER
 * PARTIES,  AND MAKES NO GUARANTEES, EXPRESSED OR IMPLIED, ABOUT ITS QUALITY,
 * RELIABILITY, OR ANY OTHER CHARACTERISTIC.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include "webgpu_rans_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpl_conv.h"
#include "cpu_rans_backend.h"
#include "gpu_context.h"
#include "gpu_multigrid.h"
#include "gpu_rans.h"
#include "rans_gpu_bridge.h"
#include "rans_dem_solve.h"
#include "rans_gpu_solve.h"
#include "rans_grid.h"
#include "rans_multigrid.h"
#include "rans_solver.h"

bool WebGpuRansBackend::solve(const MomentumProblem &problem,
                              ninjaComClass *com,
                              const std::function<void()> &checkCancel,
                              const std::function<bool(const MomentumResult &)> &accept,
                              MomentumResult &result)
{
    std::unique_ptr<gpu::Context> context;
    try
    {
        // NINJA_WEBGPU_DISABLE: behave as if there were no adapter (tests)
        if(CPLTestBool(CPLGetConfigOption("NINJA_WEBGPU_DISABLE", "NO")))
            throw gpu::Unavailable("WebGPU: no adapter (NINJA_WEBGPU_DISABLE)");
        context.reset(new gpu::Context());
    }
    catch(const gpu::Unavailable &e)
    {
        throw MomentumBackendUnavailableError(std::string("momentum backend 'webgpu': ") + e.what());
    }
    com->ninjaCom(ninjaComClass::ninjaNone, "WebGPU RANS adapter: %s.",
                  context->adapterName().c_str());

    DemWindProblem p;
    p.speed = problem.speed;
    p.inputHeight = problem.inputHeight;
    p.direction = problem.direction;
    p.z0 = problem.z0;
    p.meshResolution = problem.meshResolution;
    p.layers = layers;
    p.growth = growth;
    GpuSolveOptions options;
    options.multigridLevels = multigridLevels;
    options.maxIterations = maxIterations;
    options.checkEvery = checkEvery;
    options.tolerance = tolerance;
    options.outputTolerance = outputTolerance;
    options.cfl = cfl;
    options.maxRetries = maxRetries;
    DemWind wind;
    try
    {
        // NINJA_WEBGPU_TEST_ALLOC_BYTES: first allocate a buffer of this size,
        // to exercise allocation failures (tests)
        if(const char *bytes = CPLGetConfigOption("NINJA_WEBGPU_TEST_ALLOC_BYTES", NULL))
            gpu::Buffer probe(*context, std::strtoull(bytes, NULL, 10));
        wind = SolveDemOnGpu(*context, CpuRansBackend::terrainDem(problem), p, options,
            [com](const std::string &message, bool warning)
            {
                com->ninjaCom(warning ? ninjaComClass::ninjaWarning : ninjaComClass::ninjaNone,
                              "%s", message.c_str());
            },
            checkCancel);
    }
    catch(const RansDivergedError &e)
    {
        // lets NinjaFoam retry on a smoothed DEM
        throw MomentumDivergedError(e.what());
    }
    catch(const gpu::Error &e)
    {
        // device lost, out of memory, ...: lets NinjaFoam fall back
        throw MomentumBackendUnavailableError(std::string("momentum backend 'webgpu': ") + e.what());
    }
    lastCfl = wind.report.cfl;

    // ground-cell values on the DEM grid, as OpenFOAM's 'cell' sampling
    result.u = *problem.dem;
    result.v = *problem.dem;
    result.k = *problem.dem;
    const int nCols = problem.dem->get_nCols();
    for(int row = 0; row < problem.dem->get_nRows(); row++)
        for(int col = 0; col < nCols; col++)
        {
            std::size_t i = std::size_t(row) * nCols + col;
            result.u(row, col) = wind.u[i];
            result.v(row, col) = wind.v[i];
            result.k(row, col) = wind.k[i];
        }
    result.sampleHeight = wind.sampleHeight;
    result.outputHeightResolved = 2.0 * wind.sampleHeight < problem.outputHeight;
    if(result.outputHeightResolved)
        throw std::runtime_error("RANS backend: output height inside the first layer is not supported yet");
    return accept(result);
}
