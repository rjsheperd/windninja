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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_rans_backend.h"
#include "gpu_context.h"
#include "gpu_multigrid.h"
#include "gpu_rans.h"
#include "rans_gpu_bridge.h"
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
        context.reset(new gpu::Context());
    }
    catch(const gpu::Unavailable &e)
    {
        throw std::runtime_error(std::string("momentum backend 'webgpu': ") + e.what());
    }
    com->ninjaCom(ninjaComClass::ninjaNone, "WebGPU RANS adapter: %s.",
                  context->adapterName().c_str());

    // the CPU backend's grid, boundaries and initial state
    CpuRansBackend cpu;
    cpu.layers = layers;
    cpu.growth = growth;
    const double firstLayer = problem.meshResolution;
    RansGrid g = cpu.buildGrid(problem, firstLayer);
    com->ninjaCom(ninjaComClass::ninjaNone, "WebGPU RANS grid: %d x %d x %d cells.",
                  g.nx, g.ny, g.nz);
    RansSolver setup = CpuRansBackend::setUp(problem, g);
    gpu::RansSolver s(*context, GpuGridView(g), setup.nu, setup.beta);
    GpuSolveOptions options;
    options.multigridLevels = multigridLevels;
    options.maxIterations = maxIterations;
    options.checkEvery = checkEvery;
    options.tolerance = tolerance;
    options.outputTolerance = outputTolerance;
    options.cfl = cfl;
    options.maxRetries = maxRetries;
    try
    {
        GpuSolveReport report = SolveOnGpu(s, setup, options,
            [com](const std::string &message, bool warning)
            {
                com->ninjaCom(warning ? ninjaComClass::ninjaWarning : ninjaComClass::ninjaNone,
                              "%s", message.c_str());
            },
            checkCancel);
        lastCfl = report.cfl;
    }
    catch(const RansDivergedError &e)
    {
        // lets NinjaFoam retry on a smoothed DEM
        throw MomentumDivergedError(e.what());
    }

    CpuRansBackend::sample(problem, g, s.fields.download(gpu::fieldU),
                           s.fields.download(gpu::fieldV), s.fields.download(gpu::fieldK),
                           firstLayer, result);
    return accept(result);
}
