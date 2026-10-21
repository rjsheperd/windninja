/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Bridge from the CPU RANS solver's types to the GPU port's
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

#ifndef RANS_GPU_BRIDGE_H
#define RANS_GPU_BRIDGE_H

#include "gpu_rans.h"
#include "rans_grid.h"
#include "rans_solver.h"

/*
** The GPU solver (src/ninja/gpu) is free of WindNinja types so it can build
** for the browser; these convert from the CPU solver's.  The CPU solver
** sets up a problem (ApplyWindBoundaries, InitializeLogProfile) and the GPU
** solver takes it over.
*/

// non-owning: g must outlive the view's use
gpu::GridView GpuGridView(const RansGrid &g);

gpu::RansSolver::TurbulenceModel GpuTurbulenceModel(const RansTurbulenceModel &m);

// Copies the boundaries, turbulence settings and state (p, u, v, w, and k,
// epsilon, nut when turbulent).  Throws std::invalid_argument for a dirichlet
// boundary, which has no GPU version.
void CopyToGpu(const RansSolver &cpu, gpu::RansSolver &gpu);

#endif /* RANS_GPU_BRIDGE_H */
