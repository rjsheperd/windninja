/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Mass-consistent solver: DEM in, output wind out
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

#include "mass_dem.h"

#include <stdexcept>

MassSolveResult SolveMassWindOnCpu(const MassGrid &grid, const double *u0, const double *v0,
                                   const double *w0, const double *alphaV, double alphaH,
                                   const MassSolveOptions &options, double *u, double *v,
                                   double *w)
{
    MassSystem system = AssembleMass(grid, u0, v0, w0, alphaV, alphaH);
    std::vector<double> phi(grid.numNodes(), 0.0);
    MassSolveResult result = SolveMassPcg(system, phi.data(), options);
    RecoverVelocity(grid, phi.data(), u0, v0, w0, alphaV, alphaH, u, v, w);
    return result;
}

MassDemWind SolveMassDem(const TerrainDem &dem, const MassDemProblem &problem,
                         const MassSolveOptions &options, const MassWindSolver &solver)
{
    MassMeshOptions mesh;
    mesh.layers = problem.layers;
    mesh.inputHeight = problem.wind.inputHeight;
    mesh.outputHeight = problem.outputHeight;
    mesh.canopyHeight = problem.wind.roughH;
    MassGrid grid = MassTerrainGrid(dem, mesh);

    std::vector<double> u0, v0, w0;
    InitialWind(grid, problem.wind, u0, v0, w0);
    const int n = grid.numNodes();
    const double alphaH = 1.0;   // ninja's alphaH; neutral alphaV = alphaH
    std::vector<double> alphaV(n, alphaH), u(n), v(n), w(n);

    MassDemWind out;
    out.nCols = dem.nCols;
    out.nRows = dem.nRows;
    out.solve = solver(grid, u0.data(), v0.data(), w0.data(), alphaV.data(), alphaH, options,
                       u.data(), v.data(), w.data());
    if(!out.solve.converged)
        throw std::runtime_error("Solution did not converge.\nMAXITS reached.");
    OutputWind(grid, u, v, w, problem.wind, problem.outputHeight, out.speed, out.angle);
    return out;
}
