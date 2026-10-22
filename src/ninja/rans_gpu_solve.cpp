/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  The GPU RANS solve loop (core-free)
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

#include "rans_gpu_solve.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <vector>

#include "gpu_multigrid.h"
#include "rans_gpu_bridge.h"
#include "rans_multigrid.h"

static std::string Format(const char *fmt, ...)
{
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof buffer, fmt, args);
    va_end(args);
    return buffer;
}

GpuSolveReport SolveOnGpu(gpu::RansSolver &s, const RansSolver &setup,
                          const GpuSolveOptions &options, const GpuSolveLog &log,
                          const std::function<void()> &checkCancel)
{
    const RansGrid &g = setup.grid;
    CopyToGpu(setup, s);
    // the CPU multigrid only provides the hierarchy (grids and node maps)
    RansMultigrid hierarchy(const_cast<RansSolver &>(setup), options.multigridLevels);
    std::unique_ptr<gpu::RansMultigrid> mg;
    if(hierarchy.levels() > 1)
        mg.reset(new gpu::RansMultigrid(s, GpuMultigridLevels(hierarchy)));
    log(Format("WebGPU RANS multigrid: %d levels.", hierarchy.levels()), false);
    const char *unit = mg ? "cycle" : "step";

    // on divergence, restart from the initial state at half the CFL number
    // (as OpenFOAM's retries)
    double runCfl = options.cfl;
    int it = 0;
    double worst = 1.0, change = 1e30;
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
        change = 1e30;
        std::vector<float> previous;
        while(it < options.maxIterations && worst > options.tolerance &&
              change >= options.outputTolerance)
        {
            checkCancel();
            if(mg)
            {
                for(int i = 0; i < options.checkEvery; i++)
                    mg->cycle(runCfl);
                r = s.lastResiduals();
            }
            else
                r = s.iterate(options.checkEvery, runCfl);
            it += options.checkEvery;
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
            // RMS change of the ground-layer speed since the last check (the
            // ground layer is the first nx * ny cells of each field slot)
            const std::size_t n2 = std::size_t(g.nx) * g.ny, nc = s.grid.nCells;
            std::vector<float> ground(2 * n2);
            s.fields.buffer.download(ground.data(), n2 * sizeof(float),
                                     gpu::fieldU * nc * sizeof(float));
            s.fields.buffer.download(ground.data() + n2, n2 * sizeof(float),
                                     gpu::fieldV * nc * sizeof(float));
            if(!previous.empty())
            {
                double sum = 0.0;
                for(std::size_t i = 0; i < n2; i++)
                {
                    double d = std::hypot(ground[i], ground[n2 + i]) -
                               std::hypot(previous[i], previous[n2 + i]);
                    sum += d * d;
                }
                change = std::sqrt(sum / n2);
            }
            previous.swap(ground);
            if(it % (mg ? 200 : 1000) == 0)
                log(Format("WebGPU RANS %s %d, residual %g", unit, it, worst), false);
        }
        if(!diverged)
            break;
        if(attempt == options.maxRetries)
            throw RansDivergedError(Format("WebGPU RANS diverged by %s %d at CFL %g after %d retries",
                                           unit, it, runCfl, options.maxRetries));
        log(Format("WebGPU RANS diverged by %s %d at CFL %g; retrying at CFL %g.",
                   unit, it, runCfl, runCfl / 2.0), true);
        runCfl /= 2.0;
    }
    log(Format("WebGPU RANS stopped after %d %ss at CFL %g, residual %g, output change %g m/s.",
               it, unit, runCfl, worst, change), false);
    GpuSolveReport report;
    report.iterations = it;
    report.levels = hierarchy.levels();
    report.cfl = runCfl;
    report.residual = worst;
    report.outputChange = change;
    return report;
}
