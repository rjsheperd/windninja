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

#include "cpu_rans_backend.h"
#include "gpu_context.h"
#include "gpu_multigrid.h"
#include "gpu_rans.h"
#include "rans_gpu_bridge.h"
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
    CopyToGpu(setup, s);
    // the CPU multigrid only provides the hierarchy (grids and node maps)
    RansMultigrid hierarchy(setup, multigridLevels);
    std::unique_ptr<gpu::RansMultigrid> mg;
    if(hierarchy.levels() > 1)
        mg.reset(new gpu::RansMultigrid(s, GpuMultigridLevels(hierarchy)));
    com->ninjaCom(ninjaComClass::ninjaNone, "WebGPU RANS multigrid: %d levels.", hierarchy.levels());
    const char *unit = mg ? "cycle" : "step";

    // on divergence, restart from the initial state at half the CFL number
    // (as OpenFOAM's retries)
    double runCfl = cfl;
    int it = 0;
    double worst = 1.0;
    for(int attempt = 0; ; attempt++)
    {
        if(attempt > 0)
        {
            CopyToGpu(setup, s);
            if(mg)
                mg->reset();
        }
        gpu::RansSolver::Residuals peak, r;
        worst = 1.0;
        it = 0;
        bool diverged = false;
        while(it < maxIterations && worst > tolerance)
        {
            checkCancel();
            if(mg)
            {
                for(int i = 0; i < checkEvery; i++)
                    mg->cycle(runCfl);
                r = s.lastResiduals();
            }
            else
                r = s.iterate(checkEvery, runCfl);
            it += checkEvery;
            // NaN compares false, so without this a diverged run would stop
            // as if converged
            if(!std::isfinite(r.u) || !std::isfinite(r.v) || !std::isfinite(r.k))
            {
                diverged = true;
                break;
            }
            peak.u = std::max(peak.u, std::max(r.u, r.v));
            peak.k = std::max(peak.k, r.k);
            worst = std::max(std::max(r.u, r.v) / peak.u, r.k / peak.k);
            if(it % (mg ? 200 : 1000) == 0)
                com->ninjaCom(ninjaComClass::ninjaNone, "WebGPU RANS %s %d, residual %g", unit, it,
                              worst);
        }
        if(!diverged)
            break;
        if(attempt == maxRetries)
            throw std::runtime_error(std::string("WebGPU RANS diverged by ") + unit + " " +
                                     std::to_string(it) +
                                     " at CFL " + std::to_string(runCfl) + " after " +
                                     std::to_string(maxRetries) + " retries");
        com->ninjaCom(ninjaComClass::ninjaWarning,
                      "WebGPU RANS diverged by %s %d at CFL %g; retrying at CFL %g.",
                      unit, it, runCfl, runCfl / 2.0);
        runCfl /= 2.0;
    }
    lastCfl = runCfl;
    com->ninjaCom(ninjaComClass::ninjaNone, "WebGPU RANS stopped after %d %ss at CFL %g, residual %g.",
                  it, unit, runCfl, worst);

    CpuRansBackend::sample(problem, g, s.fields.download(gpu::fieldU),
                           s.fields.download(gpu::fieldV), s.fields.download(gpu::fieldK),
                           firstLayer, result);
    return accept(result);
}
