/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  FAS geometric multigrid for the CPU RANS solver
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

#ifndef RANS_MULTIGRID_H
#define RANS_MULTIGRID_H

#include <memory>
#include <vector>

#include "rans_grid.h"
#include "rans_solver.h"

/*
** Full Approximation Scheme (FAS) multigrid around RansSolver's explicit
** smoother.  Coarse grids subsample the fine grid's nodes (every second node,
** keeping the last one when a dimension is odd), so each is a full RansGrid
** and the fine boundary conditions apply unchanged.  Each V-cycle smooths,
** restricts the state (volume-weighted) and the residual (summed over child
** cells), solves the coarse problem with the FAS forcing
** F = sum(R_fine) - R_coarse(restricted state), and adds back the coarse
** correction (piecewise constant).  k, epsilon and nut are restricted and
** frozen on the coarse levels; only the fine solver transports them.
*/
class RansMultigrid
{
public:
    /* Build up to maxLevels levels (including the fine one), stopping before
    ** a level would have fewer than minCells cells. */
    RansMultigrid(RansSolver &fine, int maxLevels = 6, int minCells = 64);

    /* One V-cycle.  Returns the fine residuals (including k and epsilon)
    ** from the last fine smoothing step. */
    RansSolver::Residuals cycle(double cfl, int preSmooth = 2, int postSmooth = 2,
                                int coarsest = 8);

    // scales the prolonged coarse correction.  Undamped (1.0), the cavity
    // settles into a limit cycle; 0.5-0.6 converges fastest.
    double correctionDamping = 0.6;

    int levels() const { return (int)solvers.size(); }
    const RansGrid &grid(int level) const { return solvers[level]->grid; }

private:
    std::vector<std::unique_ptr<RansGrid>> grids;      // coarse grids (levels 1..)
    std::vector<std::unique_ptr<RansSolver>> owned;    // coarse solvers
    std::vector<RansSolver *> solvers;                 // all levels, 0 = fine
    std::vector<std::vector<int>> toCoarse;            // cell on level l -> cell on l+1
    // trilinear prolongation per direction: for each fine index, the coarse
    // neighbour (or -1) it leans towards and that neighbour's weight
    struct Lean { std::vector<int> nb; std::vector<double> w; };
    std::vector<Lean> leanX, leanY, leanZ;
    std::vector<std::vector<double>> q0;               // restricted state per coarse level

    RansSolver::Residuals vcycle(int level, double cfl, int preSmooth, int postSmooth, int coarsest);
    void restrictState(int level);
};

#endif /* RANS_MULTIGRID_H */
