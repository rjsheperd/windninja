/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Wind over a plain DEM on the GPU (core-free)
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

#include "rans_dem_solve.h"

#include <cstdio>
#include <string>

#include "gpu_rans.h"
#include "rans_gpu_bridge.h"
#include "rans_grid.h"
#include "rans_solver.h"
#include "rans_wind.h"

DemWind SolveDemOnGpu(gpu::Context &context, const TerrainDem &dem, const DemWindProblem &problem,
                      const GpuSolveOptions &options, const GpuSolveLog &log,
                      const std::function<void()> &checkCancel)
{
    // the first layer matches OpenFOAM's near-ground cell after refinement
    const double firstLayer = problem.meshResolution;
    RansGrid g = TerrainGrid(dem, problem.meshResolution, firstLayer, problem.layers,
                             problem.growth);
    char message[128];
    snprintf(message, sizeof message, "WebGPU RANS grid: %d x %d x %d cells.", g.nx, g.ny, g.nz);
    log(message, false);
    RansSolver setup = WindSetup(g, problem.speed, problem.inputHeight, problem.z0,
                                 problem.direction);
    gpu::RansSolver s(context, GpuGridView(g), setup.nu, setup.beta);

    DemWind wind;
    wind.report = SolveOnGpu(s, setup, options, log, checkCancel);
    // the ground layer is the first nx * ny cells, in the DEM's order
    const std::size_t n2 = std::size_t(g.nx) * g.ny;
    std::vector<double> u = s.fields.download(gpu::fieldU);
    std::vector<double> v = s.fields.download(gpu::fieldV);
    std::vector<double> k = s.fields.download(gpu::fieldK);
    wind.u.assign(u.begin(), u.begin() + n2);
    wind.v.assign(v.begin(), v.begin() + n2);
    wind.k.assign(k.begin(), k.begin() + n2);
    wind.sampleHeight = firstLayer / 2.0;
    return wind;
}
