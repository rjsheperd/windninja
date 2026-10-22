/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Momentum solver backend running the CPU reference RANS solver
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

#include "cpu_rans_backend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans_abl.h"
#include "rans_grid.h"
#include "rans_terrain.h"
#include "rans_multigrid.h"
#include "rans_solver.h"
#include "rans_wind.h"


RansGrid CpuRansBackend::buildGrid(const MomentumProblem &problem, double firstLayer) const
{
    return TerrainGrid(terrainDem(problem), problem.meshResolution, firstLayer, layers, growth);
}

TerrainDem CpuRansBackend::terrainDem(const MomentumProblem &problem)
{
    const Elevation &e = *problem.dem;
    TerrainDem dem;
    dem.nCols = e.get_nCols();
    dem.nRows = e.get_nRows();
    dem.cellSize = e.get_cellSize();
    dem.xll = e.get_xllCorner();
    dem.yll = e.get_yllCorner();
    dem.z.resize((size_t)dem.nCols * dem.nRows);
    for(int r = 0; r < dem.nRows; r++)
        for(int c = 0; c < dem.nCols; c++)
            dem.z[(size_t)r * dem.nCols + c] = e.get_cellValue(r, c);
    return dem;
}

bool CpuRansBackend::solve(const MomentumProblem &problem,
                           ninjaComClass *com,
                           const std::function<void()> &checkCancel,
                           const std::function<bool(const MomentumResult &)> &accept,
                           MomentumResult &result)
{
    // the first layer matches OpenFOAM's near-ground cell after refinement
    const double firstLayer = problem.meshResolution;
    RansGrid g = buildGrid(problem, firstLayer);
    com->ninjaCom(ninjaComClass::ninjaNone, "CPU RANS grid: %d x %d x %d cells.", g.nx, g.ny, g.nz);

    RansSolver s = setUp(problem, g);
    s.threads = problem.nThreads;

    RansSolver::Residuals peak, r;
    peak.p = peak.u = peak.v = peak.w = 0.0;
    double worst = 1.0;
    // FAS multigrid V-cycles (each ~4 fine smoothing steps plus coarse work)
    RansMultigrid mg(s, multigridLevels);
    com->ninjaCom(ninjaComClass::ninjaNone, "CPU RANS multigrid: %d levels.", mg.levels());
    int it = 0;
    while(it < maxIterations && worst > tolerance)
    {
        checkCancel();
        r = mg.cycle(0.5);
        it += 1;
        peak.u = std::max(peak.u, std::max(r.u, r.v));
        peak.k = std::max(peak.k, r.k);
        worst = std::max(std::max(r.u, r.v) / peak.u, r.k / peak.k);
        // NaN compares false, so without this a diverged run would stop as
        // if converged
        if(!std::isfinite(r.u) || !std::isfinite(r.v) || !std::isfinite(r.k))
            throw MomentumDivergedError("CPU RANS diverged by cycle " + std::to_string(it));
        if(it % 200 == 0)
            com->ninjaCom(ninjaComClass::ninjaNone, "CPU RANS cycle %d, residual %g", it, worst);
    }
    com->ninjaCom(ninjaComClass::ninjaNone, "CPU RANS stopped after %d cycles, residual %g.", it, worst);

    sample(problem, g, s.u, s.v, s.k, firstLayer, result);
    if(problem.wantVolume)
    {
        MomentumVolume &vol = result.volume;
        vol.nx = g.nx;
        vol.ny = g.ny;
        vol.nz = g.nz;
        vol.xll = problem.dem->get_xllCorner();
        vol.yll = problem.dem->get_yllCorner();
        vol.cellSize = problem.dem->get_cellSize();
        vol.z = g.cz;   // RansGrid's z is not shifted
        vol.u = s.u;
        vol.v = s.v;
        vol.w = s.w;
        vol.k = s.k;
    }
    return accept(result);
}

RansSolver CpuRansBackend::setUp(const MomentumProblem &problem, const RansGrid &g)
{
    return WindSetup(g, problem.speed, problem.inputHeight, problem.z0, problem.direction);
}

void CpuRansBackend::sample(const MomentumProblem &problem, const RansGrid &g,
                            const std::vector<double> &u, const std::vector<double> &v,
                            const std::vector<double> &k, double firstLayer,
                            MomentumResult &result)
{
    // ground-cell values on the DEM grid, as OpenFOAM's 'cell' sampling
    result.u = *problem.dem;
    result.v = *problem.dem;
    result.k = *problem.dem;
    for(int row = 0; row < g.ny; row++)
        for(int col = 0; col < g.nx; col++)
        {
            int c = g.cell(col, row, 0);
            result.u(row, col) = u[c];
            result.v(row, col) = v[c];
            result.k(row, col) = k[c];
        }
    result.sampleHeight = firstLayer / 2.0;
    result.outputHeightResolved = firstLayer < problem.outputHeight;
    if(result.outputHeightResolved)
    {
        // TODO (RJ Sheperd 2026-10-21) -- interpolate to the output height when
        // the first layer resolves it; WindNinja's coarse/medium/fine meshes
        // never do (first cell 59-286 m)
        throw std::runtime_error("RANS backend: output height inside the first layer is not supported yet");
    }
}
