/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  FAS geometric multigrid for the GPU RANS solver
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

#ifndef GPU_MULTIGRID_H
#define GPU_MULTIGRID_H

#include <memory>
#include <vector>

#include "gpu_rans.h"

namespace gpu {

/*
** FAS multigrid around gpu::RansSolver's explicit smoother, as RansMultigrid:
** coarse levels subsample the nodes, the state is restricted
** volume-weighted and the residual summed over children, the coarse
** problem runs with forcing F = sum(R_fine) - R_coarse(restricted state),
** and the damped correction is prolonged trilinearly.  Coarse levels freeze
** k, epsilon and nut.  A cycle reads nothing back, so it never waits on the
** GPU.
*/
class RansMultigrid
{
public:
    // A coarse level: its grid, and for each direction the index on the
    // level above of each of its nodes (RansMultigrid::coarseNodes).
    struct Level
    {
        GridView grid;
        std::vector<int> nodes[3];
    };

    // coarse levels 1.., each coarsening the one before (the fine's first);
    // they copy the fine solver's boundaries and turbulence model
    RansMultigrid(RansSolver &fine, const std::vector<Level> &coarse);
    ~RansMultigrid();

    // one V-cycle
    void cycle(double cfl, int preSmooth = 2, int postSmooth = 2, int coarsest = 8);
    // Zero the coarse forcing, before restarting from a new fine state.  The
    // in-place forcing update is exact for any finite previous forcing, but
    // not after a divergence left NaN in it.
    void reset();
    int levels() const { return static_cast<int>(solvers_.size()); }

    double correctionDamping = 0.6;   // as RansMultigrid

private:
    struct Transfer;

    void vcycle(int level, double cfl, int preSmooth, int postSmooth, int coarsest);
    void dispatch(Kernel &kernel, const std::vector<const Buffer *> &bindings,
                  std::size_t threads);

    Context &context_;
    std::vector<std::unique_ptr<RansSolver>> owned_;
    std::vector<RansSolver *> solvers_;              // all levels, 0 = fine
    std::vector<std::unique_ptr<Transfer>> transfers_;  // level l to l + 1
    Kernel restrict_, forcing_, prolong_;
    double damping_ = -1.0;                          // as uploaded
};

} // namespace gpu

#endif /* GPU_MULTIGRID_H */
