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

#include "mass_gpu_solve.h"

#include <algorithm>
#include <vector>

#include "gpu_context.h"
#include "gpu_mass.h"

MassSolveResult SolveMassOnGpu(int ncols, int nrows, int nlayers, const MassSystem &system,
                               double *phi, const MassSolveOptions &options,
                               std::string *adapter)
{
    gpu::Context context;
    if(adapter)
        *adapter = context.adapterName();
    gpu::MassPcg pcg(context, gpu::PackMassStencil(ncols, nrows, nlayers, system.SK.data(),
                                                   system.col_ind.data(),
                                                   system.row_ptr.data()));
    const std::size_t n = system.RHS.size();
    std::vector<double> x(phi, phi + n);
    double startResidual = -1.0;
    auto onCheck = [&](int iteration, double residual) {
        if(startResidual < 0.0)
            startResidual = residual;
        if(options.checkCancel)
            options.checkCancel();
        if(options.progress)
            options.progress(iteration, residual, startResidual);
    };
    const int checkEvery = 10;
    gpu::MassPcg::Result r = pcg.solve(system.RHS, x, options.tol, options.maxIterations,
                                       checkEvery, onCheck);
    std::copy(x.begin(), x.end(), phi);

    MassSolveResult result;
    result.iterations = r.iterations;
    result.residual = r.residual;
    result.converged = r.converged;
    result.jacobi = true;
    return result;
}
