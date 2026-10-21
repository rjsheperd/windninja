/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  GPU port of the RANS solver (WEBGPU_PLAN Phase 4)
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

#ifndef GPU_RANS_H
#define GPU_RANS_H

#include <memory>
#include <vector>

#include "gpu_context.h"
#include "gpu_grid.h"

namespace gpu {

/*
** The RANS solver's numerics (RansSolver) as WebGPU compute stages.  Built
** up stage by stage in WEBGPU_PLAN 4.2; each stage is public so its test can
** compare it with the CPU solver's.
*/
class RansSolver
{
public:
    enum Side { xMin, xMax, yMin, yMax, zMin, zMax };

    // the boundary kinds the wind setup uses (RansSolver's dirichlet, a
    // test-only callback, has no GPU version)
    struct Boundary
    {
        enum Type { wall, slip, velocityInlet, pressureOutlet, logProfileInlet };
        Type type = wall;
        double p = 0.0, u = 0.0, v = 0.0, w = 0.0;
        double ustar = 0.0, z0 = 0.0;
    };

    RansSolver(Context &context, const GridView &view, double nu, double beta);

    void setBoundary(Side side, const Boundary &bc);

    // Green-Gauss gradients of p, u, v, w into gradients: slot
    // field * 3 + direction (field 0-3 = p, u, v, w) of nCells floats.
    void computeGradients();
    std::vector<double> gradient(int field, int direction);

    Context &context;
    Grid grid;
    Fields fields;
    Buffer gradients;

private:
    void dispatch(Kernel &kernel, const std::vector<const Buffer *> &bindings,
                  std::size_t threads);
    void uploadParams();

    double nu_, beta_;
    Boundary bc_[6];
    bool paramsDirty_ = true;
    Buffer params_[3];       // per face direction (Params.dir)
    Buffer faceValues_[3];   // per-face scratch, 12 slots
    Kernel gradientFaces_, gradientCells_;
};

} // namespace gpu

#endif /* GPU_RANS_H */
