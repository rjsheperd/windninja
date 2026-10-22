/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Mass-consistent solver: the solve on the GPU (WebGPU builds)
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

#ifndef MASS_GPU_SOLVE_H
#define MASS_GPU_SOLVE_H

#include <memory>
#include <string>

#include "gpu_context.h"
#include "mass_pcg.h"

/*
** The mass solver's whole job on the GPU, wind in and wind out: builds the
** system from the grid, the initial wind (u0, v0, w0) and alphaV
** (gpu::MassFem), solves for Phi from zero (gpu::MassPcg, Jacobi, f32) and
** recovers u, v, w (all per node, MassGrid's order).  As AssembleMass,
** SolveMassPcg and RecoverVelocity on the CPU; options.jacobi is ignored
** (always Jacobi); progress and checkCancel run at each residual check
** (every 10 iterations).  In f32 the residual is reliable down to tol ~
** 1e-2 (gpu::MassPcg); WindNinja uses 0.1.  A folded element throws
** std::runtime_error, as AssembleMass.  Throws gpu::Unavailable when
** there is no WebGPU adapter, gpu::Error on GPU failures.  adapter, if
** given, receives the adapter's name.  Free of WindNinja core.
*/
MassSolveResult SolveMassWindOnGpu(const MassGrid &grid, const double *u0, const double *v0,
                                   const double *w0, const double *alphaV, double alphaH,
                                   const MassSolveOptions &options, double *u, double *v,
                                   double *w, std::string *adapter = nullptr);

// The same on an existing context (e.g. one sharing a page's device).
MassSolveResult SolveMassWindOnGpu(gpu::Context &context, const MassGrid &grid, const double *u0,
                                   const double *v0, const double *w0, const double *alphaV,
                                   double alphaH, const MassSolveOptions &options, double *u,
                                   double *v, double *w);

/*
** The same, leaving the wind on the GPU for others on the context's device
** (MASS_GPU_PLAN.org 3.2): a buffer of 3 * nodes f32, u, v, w slots (slot s
** of node n at s * nodes + n, MassGrid's node order).  *result gets the
** solve's outcome.
*/
std::unique_ptr<gpu::Buffer> SolveMassWindKeptOnGpu(gpu::Context &context, const MassGrid &grid,
                                                    const double *u0, const double *v0,
                                                    const double *w0, const double *alphaV,
                                                    double alphaH, const MassSolveOptions &options,
                                                    MassSolveResult *result);

#endif /* MASS_GPU_SOLVE_H */
