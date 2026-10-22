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
#include <utility>
#include <vector>

#include "gpu_context.h"
#include "gpu_mass.h"

namespace {

gpu::MassGeometry Geometry(const MassGrid &g)
{
    return gpu::MakeMassGeometry(g.ncols, g.nrows, g.nlayers, g.x.data(), g.y.data(), g.z.data());
}

} // namespace

MassGpuSolver::MassGpuSolver(const MassGrid &grid)
    : MassGpuSolver(std::unique_ptr<gpu::Context>(new gpu::Context()), nullptr, grid)
{
}

MassGpuSolver::MassGpuSolver(gpu::Context &context, const MassGrid &grid)
    : MassGpuSolver(nullptr, &context, grid)
{
}

MassGpuSolver::MassGpuSolver(std::unique_ptr<gpu::Context> owned, gpu::Context *context,
                             const MassGrid &grid)
    : owned_(std::move(owned)), context_(owned_ ? owned_.get() : context), grid_(grid),
      fem_(*context_, Geometry(grid)), pcg_(*context_, grid.ncols, grid.nrows, grid.nlayers),
      windIn_(*context_, 3 * (std::size_t)grid.numNodes() * sizeof(float)),
      alphaIn_(*context_, (std::size_t)grid.numNodes() * sizeof(float)),
      windOut_(*context_, 3 * (std::size_t)grid.numNodes() * sizeof(float))
{
}

MassSolveResult MassGpuSolver::solveKept(const double *u0, const double *v0, const double *w0,
                                         const double *alphaV, double alphaH,
                                         const MassSolveOptions &options)
{
    const std::size_t n = grid_.numNodes();
    std::vector<float> wind(3 * n), alpha(alphaV, alphaV + n);
    for(std::size_t i = 0; i < n; i++)
    {
        wind[i] = (float)u0[i];
        wind[n + i] = (float)v0[i];
        wind[2 * n + i] = (float)w0[i];
    }
    windIn_.upload(wind);
    alphaIn_.upload(alpha);
    pcg_.solution().upload(std::vector<float>(n, 0.0f));
    fem_.assemble(windIn_, alphaIn_, (float)alphaH, pcg_.coefficients(), pcg_.rhs());
    if(fem_.foldedElements())
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
    gpu::MassPcg::Result r = pcg_.solveOnDevice(options.tol, options.maxIterations, 10, onCheck);
    fem_.recover(pcg_.solution(), windIn_, alphaIn_, (float)alphaH, windOut_);

    MassSolveResult result;
    result.iterations = r.iterations;
    result.residual = r.residual;
    result.converged = r.converged;
    result.jacobi = true;
    return result;
}

MassSolveResult MassGpuSolver::solve(const double *u0, const double *v0, const double *w0,
                                     const double *alphaV, double alphaH,
                                     const MassSolveOptions &options, double *u, double *v,
                                     double *w)
{
    MassSolveResult result = solveKept(u0, v0, w0, alphaV, alphaH, options);
    const std::size_t n = grid_.numNodes();
    std::vector<float> f = windOut_.download<float>(3 * n);
    for(std::size_t i = 0; i < n; i++)
    {
        u[i] = f[i];
        v[i] = f[n + i];
        w[i] = f[2 * n + i];
    }
    return result;
}

void MassGpuSolver::outputWind(const gpu::MassFem::OutputWindParams &params, gpu::Buffer &out)
{
    fem_.outputWind(windOut_, params, out);
}

std::unique_ptr<gpu::Buffer> SolveMassWindKeptOnGpu(gpu::Context &context, const MassGrid &grid,
                                                    const double *u0, const double *v0,
                                                    const double *w0, const double *alphaV,
                                                    double alphaH, const MassSolveOptions &options,
                                                    MassSolveResult *result)
{
    MassGpuSolver solver(context, grid);
    *result = solver.solveKept(u0, v0, w0, alphaV, alphaH, options);
    std::size_t bytes = 3 * (std::size_t)grid.numNodes() * sizeof(float);
    std::unique_ptr<gpu::Buffer> wind(new gpu::Buffer(context, bytes));
    wind->copyFrom(solver.wind(), bytes);
    return wind;
}

MassSolveResult SolveMassWindOnGpu(gpu::Context &context, const MassGrid &grid, const double *u0,
                                   const double *v0, const double *w0, const double *alphaV,
                                   double alphaH, const MassSolveOptions &options, double *u,
                                   double *v, double *w)
{
    return MassGpuSolver(context, grid).solve(u0, v0, w0, alphaV, alphaH, options, u, v, w);
}

MassSolveResult SolveMassWindOnGpu(const MassGrid &grid, const double *u0, const double *v0,
                                   const double *w0, const double *alphaV, double alphaH,
                                   const MassSolveOptions &options, double *u, double *v,
                                   double *w, std::string *adapter)
{
    MassGpuSolver solver(grid);
    if(adapter)
        *adapter = solver.context().adapterName();
    return solver.solve(u0, v0, w0, alphaV, alphaH, options, u, v, w);
}

std::unique_ptr<gpu::Buffer> SolveMassDemKeptOnGpu(gpu::Context &context, const TerrainDem &dem,
                                                   const MassDemProblem &problem,
                                                   const MassSolveOptions &options,
                                                   bool northFirst, MassSolveResult *result)
{
    MassMeshOptions mesh;
    mesh.layers = problem.layers;
    mesh.inputHeight = problem.wind.inputHeight;
    mesh.outputHeight = problem.outputHeight;
    mesh.canopyHeight = problem.wind.roughH;
    MassGrid grid = MassTerrainGrid(dem, mesh);
    std::vector<double> u0, v0, w0;
    InitialWind(grid, problem.wind, u0, v0, w0);
    std::vector<double> alphaV(grid.numNodes(), 1.0);   // neutral, alphaH = 1 (as SolveMassDem)

    MassGpuSolver solver(context, grid);
    *result = solver.solveKept(u0.data(), v0.data(), w0.data(), alphaV.data(), 1.0, options);
    if(!result->converged)
        throw std::runtime_error("Solution did not converge.\nMAXITS reached.");

    gpu::MassFem::OutputWindParams p;
    p.outputHeight = (float)problem.outputHeight;
    p.roughness = (float)problem.wind.roughness;
    p.roughH = (float)problem.wind.roughH;
    p.roughD = (float)problem.wind.roughD;
    p.blHeight = (float)BoundaryLayerHeight(problem.wind);
    p.northFirst = northFirst;
    std::unique_ptr<gpu::Buffer> out(
        new gpu::Buffer(context, 2 * (std::size_t)dem.nCols * dem.nRows * sizeof(float)));
    solver.outputWind(p, *out);
    return out;
}
