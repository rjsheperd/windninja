/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Momentum backend running the RANS solver on WebGPU
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

#ifndef WEBGPU_RANS_BACKEND_H
#define WEBGPU_RANS_BACKEND_H

#include "momentum_backend.h"

/*
** MomentumBackend running the RANS solver's GPU port (gpu::RansSolver) on
** the CpuRansBackend's grid, with its boundary conditions and initial state,
** so the two differ only in where (and in what precision) they iterate.
** FAS multigrid V-cycles (gpu::RansMultigrid, on the CPU multigrid's
** hierarchy) or single grid.  Results follow the same contract: u, v, k of
** the ground cell on the DEM grid.
*/
class WebGpuRansBackend : public MomentumBackend
{
public:
    int layers = 20;               // as CpuRansBackend
    double growth = 1.1;
    // including the fine grid; 1 = single grid.  Single grid by default: on
    // the GPU, multigrid gives the same answers more slowly (WEBGPU_PLAN 4.3)
    int multigridLevels = 1;
    // limits in pseudo-time steps (single grid) or V-cycles (multigrid)
    int maxIterations = 20000;
    int checkEvery = 100;          // between residual readbacks
    double tolerance = 1e-5;       // residual drop, relative to each equation's peak
    // also stop when the output (ground-layer speed) changes by less than
    // this RMS, m/s, over checkEvery steps; 0 = residual test only.  On the
    // golden cases 1e-3 ends within 0.004 m/s RMS of the full run
    double outputTolerance = 1e-3;
    double cfl = 0.5;              // pseudo-time CFL number
    int maxRetries = 3;            // restarts at half the CFL after a divergence
    double lastCfl = 0.0;          // the CFL number the last solve finished at

    virtual bool solve(const MomentumProblem &problem,
                       ninjaComClass *com,
                       const std::function<void()> &checkCancel,
                       const std::function<bool(const MomentumResult &)> &accept,
                       MomentumResult &result);
};

#endif /* WEBGPU_RANS_BACKEND_H */
