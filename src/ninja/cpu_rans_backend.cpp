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
#include <vector>

#include "rans_abl.h"
#include "rans_grid.h"
#include "rans_solver.h"
#include "rans_wind.h"

static const double AIR_NU = 1.58946099535e-05;   // WindNinja's physicalProperties

RansGrid CpuRansBackend::buildGrid(const MomentumProblem &problem, double firstLayer) const
{
    const Elevation &dem = *problem.dem;
    const int nRows = dem.get_nRows(), nCols = dem.get_nCols();
    const double cs = dem.get_cellSize();
    const int nx = nCols, ny = nRows, nz = layers;

    // OpenFOAM's domain top (NinjaFoam::SetBlockMeshParametersFromDem),
    // including its minimum of 11 block layers; block cells are 2 x the mesh
    // resolution for one refinement round (coarse and medium meshes)
    double relief = dem.get_maxValue() - dem.get_minValue();
    double blockMeshDz = std::max(0.1 * std::max(dem.get_xDimension(), dem.get_yDimension()),
                                  relief + 0.1 * relief);
    blockMeshDz = std::max(blockMeshDz, 11 * 2.0 * problem.meshResolution);
    double top = dem.get_maxValue() + blockMeshDz;

    // fractions of the column above the first layer, growing geometrically
    std::vector<double> frac(nz, 0.0);
    double sum = 0.0;
    for(int m = 0; m < nz - 1; m++) sum += std::pow(growth, m);
    for(int k = 2; k < nz; k++) frac[k] = frac[k - 1] + std::pow(growth, k - 2) / sum;

    std::vector<double> x, y, z;
    size_t nNodes = (size_t)(nx + 1) * (ny + 1) * (nz + 1);
    x.reserve(nNodes); y.reserve(nNodes); z.reserve(nNodes);
    for(int k = 0; k <= nz; k++)
        for(int j = 0; j <= ny; j++)          // AsciiGrid row j (row 0 is south)
            for(int i = 0; i <= nx; i++)      // AsciiGrid column i
            {
                // ground at a cell corner: mean of the DEM cells touching it
                double g = 0.0;
                int n = 0;
                for(int r = j - 1; r <= j; r++)
                    for(int c = i - 1; c <= i; c++)
                        if(r >= 0 && r < nRows && c >= 0 && c < nCols)
                        {
                            g += dem.get_cellValue(r, c);
                            n++;
                        }
                g /= n;
                double zk;
                if(k == 0) zk = g;
                else if(k == nz) zk = top;
                else zk = g + firstLayer + (top - g - firstLayer) * frac[k];
                x.push_back(dem.get_xllCorner() + i * cs);
                y.push_back(dem.get_yllCorner() + j * cs);
                z.push_back(zk);
            }
    if(top - dem.get_maxValue() <= 2.0 * firstLayer)
        throw std::runtime_error("CPU RANS grid: no room above the first layer under the domain top");
    return RansGrid::FromNodes(nx, ny, nz, x, y, z);
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

    double ustar = AblFrictionVelocity(problem.speed, problem.inputHeight, problem.z0);
    // artificial compressibility: wave speed comparable to the flow speed
    double beta = std::max(1.0, 1.25 * problem.speed * problem.speed);
    RansSolver s(g, AIR_NU, beta);
    s.threads = problem.nThreads;
    ApplyWindBoundaries(s, ustar, problem.z0, problem.direction);
    InitializeLogProfile(s, ustar, problem.z0, problem.direction);

    RansSolver::Residuals peak, r;
    peak.p = peak.u = peak.v = peak.w = 0.0;
    double worst = 1.0;
    int it = 0;
    while(it < maxIterations && worst > tolerance)
    {
        checkCancel();
        r = s.iterate(200, 0.5);
        it += 200;
        peak.u = std::max(peak.u, std::max(r.u, r.v));
        peak.k = std::max(peak.k, r.k);
        worst = std::max(std::max(r.u, r.v) / peak.u, r.k / peak.k);
        if(it % 2000 == 0)
            com->ninjaCom(ninjaComClass::ninjaNone, "CPU RANS iteration %d, residual %g", it, worst);
    }
    com->ninjaCom(ninjaComClass::ninjaNone, "CPU RANS stopped after %d iterations, residual %g.", it, worst);

    // ground-cell values on the DEM grid, as OpenFOAM's 'cell' sampling
    result.u = *problem.dem;
    result.v = *problem.dem;
    result.k = *problem.dem;
    for(int row = 0; row < g.ny; row++)
        for(int col = 0; col < g.nx; col++)
        {
            int c = g.cell(col, row, 0);
            result.u(row, col) = s.u[c];
            result.v(row, col) = s.v[c];
            result.k(row, col) = s.k[c];
        }
    result.sampleHeight = firstLayer / 2.0;
    result.outputHeightResolved = firstLayer < problem.outputHeight;
    if(result.outputHeightResolved)
    {
        // TODO (RJ Sheperd 2026-10-21) -- interpolate to the output height when
        // the first layer resolves it; WindNinja's coarse/medium/fine meshes
        // never do (first cell 59-286 m)
        throw std::runtime_error("CPU RANS backend: output height inside the first layer is not supported yet");
    }
    return accept(result);
}
