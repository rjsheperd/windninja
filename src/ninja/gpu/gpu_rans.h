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
#include "gpu_reduce.h"

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
        double k = 0.0, epsilon = 0.0;   // velocity inlets, when turbulent
    };

    // RNG k-epsilon coefficients (as RansTurbulenceModel)
    struct TurbulenceModel
    {
        double cmu = 0.085, c1 = 0.92, c2 = 1.68;
        double sigmaK = 0.7179, sigmaEpsilon = 0.7179;
        double eta0 = 4.38, beta = 0.012;
    };

    // RMS over cells of the residual per unit volume (as RansSolver's)
    struct Residuals
    {
        double p = 0.0, u = 0.0, v = 0.0, w = 0.0, k = 0.0, epsilon = 0.0;
    };

    RansSolver(Context &context, const GridView &view, double nu, double beta);

    void setBoundary(Side side, const Boundary &bc);
    // off means laminar; the one-argument form uses the default model
    void setTurbulence(bool on, const TurbulenceModel &model);
    void setTurbulence(bool on);

    /* Take n pseudo-time steps; returns the residuals of the last step (one
    ** wait on the GPU, at the end). */
    Residuals iterate(int n, double cfl);

    // Green-Gauss gradients of p, u, v, w into gradients: slot
    // field * 3 + direction (field 0-3 = p, u, v, w) of nCells floats.
    void computeGradients();
    std::vector<double> gradient(int field, int direction);

    // Face fluxes summed per cell into residuals (after computeGradients):
    // slots 0-5 the p, u, v, w, k, epsilon flux residuals, 6 the spectral
    // radius, 7-8 the rough-wall epsilon and G sums, 9 the wall-face count
    // (as RansSolver::residualArray / wallFaces).
    void computeResiduals();
    std::vector<double> residual(int slot);

    Context &context;
    Grid grid;
    Fields fields;
    Buffer gradients;
    Buffer residuals;
    Buffer norms;          // per-cell residuals of the last step, 6 slots

private:
    void dispatch(Kernel &kernel, const std::vector<const Buffer *> &bindings,
                  std::size_t threads);
    void uploadParams();
    void step();

    double nu_, beta_, cfl_ = 0.0;
    bool turbulent_ = false;
    TurbulenceModel model_;
    Boundary bc_[6];
    bool paramsDirty_ = true;
    Buffer params_[3];       // per face direction (Params.dir)
    Buffer faceValues_[3];   // per-face scratch, 12 slots
    Kernel gradientFaces_, gradientCells_, fluxFaces_, fluxCells_, update_;
    Reducer reducer_;
};

} // namespace gpu

#endif /* GPU_RANS_H */
