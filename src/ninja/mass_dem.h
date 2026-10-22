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

#ifndef MASS_DEM_H
#define MASS_DEM_H

#include <functional>
#include <vector>

#include "mass_pcg.h"
#include "mass_terrain.h"
#include "mass_wind.h"

/*
** WindNinja's domain-average mass solve from a plain DEM, free of WindNinja
** core: the mesh (MassTerrainGrid), the initial wind (InitialWind), the
** solve, and the output wind (OutputWind), as WindNinja's CLI computes them
** with no diurnal winds or stability, uniform vegetation, and the DEM
** already at the mesh resolution.
*/
struct MassDemProblem
{
    DomainAverageWind wind;     // speed, direction, input height, vegetation, latitude...
    double outputHeight = 6.096;   // m above the canopy
    int layers = 20;
};

struct MassDemWind
{
    int nCols = 0, nRows = 0;
    std::vector<double> speed, angle;   // per DEM cell, row 0 south (OutputWind)
    MassSolveResult solve;
};

// The solve itself: grid and initial wind in, u, v, w out (alphaV is alphaH
// everywhere: neutral).  SolveMassWindOnCpu, or SolveMassWindOnGpu.
using MassWindSolver = std::function<MassSolveResult(
    const MassGrid &, const double *u0, const double *v0, const double *w0, const double *alphaV,
    double alphaH, const MassSolveOptions &, double *u, double *v, double *w)>;

// AssembleMass, SolveMassPcg, RecoverVelocity: WindNinja's CPU path.
MassSolveResult SolveMassWindOnCpu(const MassGrid &grid, const double *u0, const double *v0,
                                   const double *w0, const double *alphaV, double alphaH,
                                   const MassSolveOptions &options, double *u, double *v,
                                   double *w);

MassDemWind SolveMassDem(const TerrainDem &dem, const MassDemProblem &problem,
                         const MassSolveOptions &options,
                         const MassWindSolver &solver = SolveMassWindOnCpu);

#endif /* MASS_DEM_H */
