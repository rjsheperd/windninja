/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  The GPU RANS solve loop (core-free)
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

#ifndef RANS_GPU_SOLVE_H
#define RANS_GPU_SOLVE_H

#include <functional>
#include <stdexcept>
#include <string>

#include "gpu_rans.h"
#include "rans_solver.h"

/*
** The WebGPU backend's solve loop, free of WindNinja core so the browser
** module (src/ninja/gpu/wasm) runs the same one: single grid or multigrid,
** stopping on the residual drop or on the output (ground-layer speed)
** settling, and restarting at half the CFL number after a divergence.
*/

struct GpuSolveOptions
{
    int multigridLevels = 1;     // including the fine grid; 1 = single grid
    int maxIterations = 20000;   // pseudo-time steps (single grid) or V-cycles
    int checkEvery = 100;        // between residual readbacks
    double tolerance = 1e-5;     // residual drop, relative to each equation's peak
    double outputTolerance = 1e-3;  // RMS ground-speed change per check, m/s
    double cfl = 0.5;
    int maxRetries = 3;          // restarts at half the CFL after a divergence
};

struct GpuSolveReport
{
    int iterations = 0, levels = 1;
    double cfl = 0.0, residual = 0.0, outputChange = 0.0;
};

// diverged after all CFL retries
class RansDivergedError : public std::runtime_error
{
public:
    explicit RansDivergedError(const std::string &what) : std::runtime_error(what) {}
};

typedef std::function<void(const std::string &message, bool warning)> GpuSolveLog;

/*
** Iterate s from setup's state (boundaries, initial fields) until it
** converges or runs out of iterations; s then holds the solution.  setup's
** grid gives the multigrid hierarchy.  checkCancel may throw to stop.
** Throws RansDivergedError.
*/
GpuSolveReport SolveOnGpu(gpu::RansSolver &s, const RansSolver &setup,
                          const GpuSolveOptions &options, const GpuSolveLog &log,
                          const std::function<void()> &checkCancel);

#endif /* RANS_GPU_SOLVE_H */
