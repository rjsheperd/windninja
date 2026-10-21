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

#include "gpu_rans.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "gpu_rans_kernels.h"

namespace gpu {

namespace {

// layouts match the WGSL Params and Bc structs in kernels::ransPrelude
struct BcBlock
{
    uint32_t kind;
    float p, u, v, w, ustar, z0, k, epsilon, pad[3];
};

struct ParamsBlock
{
    uint32_t nx, ny, nz, dir;
    float nu, beta, sigmaK, sigmaEpsilon;
    uint32_t turbulent;
    float yPlusLam, cfl, pad0;
    float cmu, c1, c2, eta0, rngBeta, pad1, pad2, pad3;
    BcBlock bc[6];
};
static_assert(sizeof(BcBlock) == 48 && sizeof(ParamsBlock) == 80 + 6 * 48,
              "must match the WGSL Params layout");

// the wall functions' laminar / log-layer switch (as WallYPlusLam, with
// kappa 0.41 and E 9.8)
double WallYPlusLam()
{
    double ypl = 11.0;
    for(int i = 0; i < 10; i++)
        ypl = std::log(std::max(9.8 * ypl, 1.0)) / 0.41;
    return ypl;
}

std::string source(const char *stage)
{
    return std::string(kernels::ransPrelude) + stage;
}

const std::size_t maxGroupsX = 65535;

} // namespace

RansSolver::RansSolver(Context &context, const GridView &view, double nu, double beta)
    : context(context), grid(context, view), fields(context, grid.nCells),
      gradients(context, 12 * grid.nCells * sizeof(float)),
      residuals(context, 10 * grid.nCells * sizeof(float)),
      norms(context, 6 * grid.nCells * sizeof(float)), nu_(nu), beta_(beta),
      params_{Buffer(context, sizeof(ParamsBlock), Buffer::Uniform),
              Buffer(context, sizeof(ParamsBlock), Buffer::Uniform),
              Buffer(context, sizeof(ParamsBlock), Buffer::Uniform)},
      faceValues_{Buffer(context, 12 * grid.nFaces[0] * sizeof(float)),
                  Buffer(context, 12 * grid.nFaces[1] * sizeof(float)),
                  Buffer(context, 12 * grid.nFaces[2] * sizeof(float))},
      gradientFaces_(context, source(kernels::gradientFaces).c_str()),
      gradientCells_(context, source(kernels::gradientCells).c_str()),
      fluxFaces_(context, source(kernels::fluxFaces).c_str()),
      fluxCells_(context, source(kernels::fluxCells).c_str()),
      update_(context, source(kernels::update).c_str()),
      reducer_(context)
{
}

void RansSolver::setTurbulence(bool on)
{
    setTurbulence(on, TurbulenceModel());
}

void RansSolver::setTurbulence(bool on, const TurbulenceModel &model)
{
    turbulent_ = on;
    model_ = model;
    paramsDirty_ = true;
}

void RansSolver::setBoundary(Side side, const Boundary &bc)
{
    bc_[side] = bc;
    paramsDirty_ = true;
}

void RansSolver::uploadParams()
{
    if(!paramsDirty_)
        return;
    ParamsBlock b = {};
    b.nx = grid.nx;
    b.ny = grid.ny;
    b.nz = grid.nz;
    b.nu = static_cast<float>(nu_);
    b.beta = static_cast<float>(beta_);
    b.sigmaK = static_cast<float>(model_.sigmaK);
    b.sigmaEpsilon = static_cast<float>(model_.sigmaEpsilon);
    b.turbulent = turbulent_ ? 1u : 0u;
    b.yPlusLam = static_cast<float>(WallYPlusLam());
    b.cfl = static_cast<float>(cfl_);
    b.cmu = static_cast<float>(model_.cmu);
    b.c1 = static_cast<float>(model_.c1);
    b.c2 = static_cast<float>(model_.c2);
    b.eta0 = static_cast<float>(model_.eta0);
    b.rngBeta = static_cast<float>(model_.beta);
    for(int s = 0; s < 6; s++)
    {
        const Boundary &bc = bc_[s];
        b.bc[s] = {static_cast<uint32_t>(bc.type), float(bc.p), float(bc.u),
                   float(bc.v), float(bc.w), float(bc.ustar), float(bc.z0),
                   float(bc.k), float(bc.epsilon), {0.0f, 0.0f, 0.0f}};
    }
    for(uint32_t d = 0; d < 3; d++)
    {
        b.dir = d;
        params_[d].upload(&b, sizeof b);
    }
    paramsDirty_ = false;
}

// 256-thread workgroups, spread over y when x would exceed the limit
void RansSolver::dispatch(Kernel &kernel, const std::vector<const Buffer *> &bindings,
                          std::size_t threads)
{
    std::size_t groups = (threads + 255) / 256;
    std::size_t gx = groups < maxGroupsX ? groups : maxGroupsX;
    std::size_t gy = (groups + gx - 1) / gx;
    kernel.dispatch(bindings, static_cast<uint32_t>(gx), static_cast<uint32_t>(gy));
}

void RansSolver::computeGradients()
{
    uploadParams();
    for(int d = 0; d < 3; d++)
        dispatch(gradientFaces_,
                 {&params_[d], &grid.cells, &grid.faces[d], &fields.buffer, &faceValues_[d]},
                 grid.nFaces[d]);
    dispatch(gradientCells_,
             {&params_[0], &grid.cells, &faceValues_[0], &faceValues_[1], &faceValues_[2],
              &gradients},
             grid.nCells);
}

std::vector<double> RansSolver::gradient(int field, int direction)
{
    std::vector<float> f(grid.nCells);
    gradients.download(f.data(), f.size() * sizeof(float),
                       (field * 3 + direction) * grid.nCells * sizeof(float));
    return std::vector<double>(f.begin(), f.end());
}

void RansSolver::computeResiduals()
{
    uploadParams();
    for(int d = 0; d < 3; d++)
        dispatch(fluxFaces_,
                 {&params_[d], &grid.cells, &grid.faces[d], &fields.buffer, &gradients,
                  &faceValues_[d]},
                 grid.nFaces[d]);
    dispatch(fluxCells_,
             {&params_[0], &faceValues_[0], &faceValues_[1], &faceValues_[2], &residuals},
             grid.nCells);
}

void RansSolver::step()
{
    computeGradients();
    computeResiduals();
    dispatch(update_,
             {&params_[0], &grid.cells, &fields.buffer, &gradients, &residuals, &norms},
             grid.nCells);
}

RansSolver::Residuals RansSolver::iterate(int n, double cfl)
{
    if(cfl != cfl_)
    {
        cfl_ = cfl;
        paramsDirty_ = true;
    }
    for(int it = 0; it < n; it++)
        step();
    Residuals r;
    if(n == 0)
        return r;
    double *slots[6] = {&r.p, &r.u, &r.v, &r.w, &r.k, &r.epsilon};
    for(int s = 0; s < (turbulent_ ? 6 : 4); s++)
        *slots[s] = std::sqrt(reducer_.sum(norms, s * grid.nCells, grid.nCells, true) /
                              grid.nCells);
    return r;
}

std::vector<double> RansSolver::residual(int slot)
{
    std::vector<float> f(grid.nCells);
    residuals.download(f.data(), f.size() * sizeof(float),
                       slot * grid.nCells * sizeof(float));
    return std::vector<double>(f.begin(), f.end());
}

} // namespace gpu
