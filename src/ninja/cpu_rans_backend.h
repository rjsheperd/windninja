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

#ifndef CPU_RANS_BACKEND_H
#define CPU_RANS_BACKEND_H

#include "momentum_backend.h"

#include <vector>

#include "rans_solver.h"

class RansGrid;

/*
** MomentumBackend running RansSolver (RNG k-epsilon, artificial
** compressibility) on a terrain-following grid built from the problem's DEM.
** Results follow the OpenFOAM backend's contract: u, v, k on the DEM grid,
** taken from the ground cell, whose centre is sampleHeight above ground.
*/
class CpuRansBackend : public MomentumBackend
{
public:
    int layers = 20;            // cells in the vertical
    double growth = 1.1;        // layer growth above the first layer
    int maxIterations = 5000;   // multigrid cycles
    int multigridLevels = 6;    // including the fine grid; 1 = single grid
    double tolerance = 1e-5;    // residual drop, relative to each equation's peak

    virtual bool solve(const MomentumProblem &problem,
                       ninjaComClass *com,
                       const std::function<void()> &checkCancel,
                       const std::function<bool(const MomentumResult &)> &accept,
                       MomentumResult &result);

    /*
    ** The grid for a problem: cell centres at the DEM's cell centres, a first
    ** layer firstLayer thick, then layers growing geometrically to OpenFOAM's
    ** flat domain top (max elevation + blockMeshDz).
    */
    RansGrid buildGrid(const MomentumProblem &problem, double firstLayer) const;

    /* Air viscosity, the artificial-compressibility beta for a problem, and
    ** the wind setup (log-profile inlets, outlets, rough ground) with the
    ** log-profile initial state.  Shared with the GPU backend. */
    static RansSolver setUp(const MomentumProblem &problem, const RansGrid &g);

    /* Fill result from the ground-cell u, v, k (per cell of g), as
    ** OpenFOAM's 'cell' sampling, with the first layer's sample height. */
    static void sample(const MomentumProblem &problem, const RansGrid &g,
                       const std::vector<double> &u, const std::vector<double> &v,
                       const std::vector<double> &k, double firstLayer,
                       MomentumResult &result);
};

#endif /* CPU_RANS_BACKEND_H */
