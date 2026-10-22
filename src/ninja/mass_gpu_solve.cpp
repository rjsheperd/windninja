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

#include <stdexcept>
#include <vector>

#include "gpu_context.h"
#include "gpu_mass.h"

MassSolveResult SolveMassWindOnGpu(const MassGrid &grid, const double *u0, const double *v0,
                                   const double *w0, const double *alphaV, double alphaH,
                                   const MassSolveOptions &options, double *u, double *v,
                                   double *w, std::string *adapter)
{
    gpu::Context context;
    if(adapter)
        *adapter = context.adapterName();
    const std::size_t n = grid.numNodes();
    gpu::MassFem fem(context, gpu::MakeMassGeometry(grid.ncols, grid.nrows, grid.nlayers,
                                                    grid.x.data(), grid.y.data(), grid.z.data()));
    gpu::MassPcg pcg(context, grid.ncols, grid.nrows, grid.nlayers);

    std::vector<float> wind(3 * n), alpha(alphaV, alphaV + n);
    for(std::size_t i = 0; i < n; i++)
    {
        wind[i] = (float)u0[i];
        wind[n + i] = (float)v0[i];
        wind[2 * n + i] = (float)w0[i];
    }
    gpu::Buffer windIn(context, 3 * n * sizeof(float)), alphaIn(context, n * sizeof(float)),
        windOut(context, 3 * n * sizeof(float));
    windIn.upload(wind);
    alphaIn.upload(alpha);
    pcg.solution().upload(std::vector<float>(n, 0.0f));
    fem.assemble(windIn, alphaIn, (float)alphaH, pcg.coefficients(), pcg.rhs());
    if(fem.foldedElements())
        throw std::runtime_error("Volume Jacobian 1 is zero or negative.");   // as AssembleMass

    double startResidual = -1.0;
    auto onCheck = [&](int iteration, double residual) {
        if(startResidual < 0.0)
            startResidual = residual;
        if(options.checkCancel)
            options.checkCancel();
        if(options.progress)
            options.progress(iteration, residual, startResidual);
    };
    gpu::MassPcg::Result r = pcg.solveOnDevice(options.tol, options.maxIterations, 10, onCheck);

    fem.recover(pcg.solution(), windIn, alphaIn, (float)alphaH, windOut);
    windOut.download(wind.data(), 3 * n * sizeof(float));
    for(std::size_t i = 0; i < n; i++)
    {
        u[i] = wind[i];
        v[i] = wind[n + i];
        w[i] = wind[2 * n + i];
    }

    MassSolveResult result;
    result.iterations = r.iterations;
    result.residual = r.residual;
    result.converged = r.converged;
    result.jacobi = true;
    return result;
}
