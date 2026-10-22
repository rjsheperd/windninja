/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  The mass solver's 27-point stencil operator on the GPU
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

#include "gpu_mass.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "gpu_mass_kernels.h"

namespace gpu {

const int massOffsets[13][3] = {
    {1, 0, 0},
    {-1, 1, 0}, {0, 1, 0}, {1, 1, 0},
    {-1, -1, 1}, {0, -1, 1}, {1, -1, 1},
    {-1, 0, 1}, {0, 0, 1}, {1, 0, 1},
    {-1, 1, 1}, {0, 1, 1}, {1, 1, 1},
};

namespace {

struct Params
{
    uint32_t ncols, nrows, nlayers, nodes;
};

const std::size_t maxGroupsX = 65535;

int Slot(int dcol, int drow, int dlayer)
{
    if(dcol == 0 && drow == 0 && dlayer == 0)
        return 0;
    for(int s = 0; s < 13; s++)
        if(massOffsets[s][0] == dcol && massOffsets[s][1] == drow && massOffsets[s][2] == dlayer)
            return s + 1;
    return -1;
}

} // namespace

MassStencil PackMassStencil(int ncols, int nrows, int nlayers, const double *SK,
                            const int *colInd, const int *rowPtr)
{
    MassStencil st;
    st.ncols = ncols;
    st.nrows = nrows;
    st.nlayers = nlayers;
    const std::size_t n = st.nodes();
    st.coef.assign(MassStencil::slots * n, 0.0f);
    const int layerSize = ncols * nrows;
    for(std::size_t row = 0; row < n; row++)
    {
        int c0 = (int)row % ncols, r0 = ((int)row / ncols) % nrows, l0 = (int)row / layerSize;
        for(int l = rowPtr[row]; l < rowPtr[row + 1]; l++)
        {
            int col = colInd[l];
            int c = col % ncols, r = (col / ncols) % nrows, k = col / layerSize;
            int s = Slot(c - c0, r - r0, k - l0);
            if(s < 0)
                throw Error("gpu::PackMassStencil: entry (" + std::to_string(row) + ", " +
                            std::to_string(col) + ") outside the 27-point stencil");
            st.coef[s * n + row] = (float)SK[l];
        }
    }
    return st;
}

MassOperator::MassOperator(Context &context, int ncols, int nrows, int nlayers)
    : context_(context), nodes_((std::size_t)ncols * nrows * nlayers),
      params_(context, sizeof(Params), Buffer::Uniform),
      coef_(context, MassStencil::slots * nodes_ * sizeof(float)),
      kernel_(context, kernels::massApply)
{
    if(nodes_ > UINT32_MAX)
        throw Error("gpu::MassOperator: too many nodes");
    Params p = {(uint32_t)ncols, (uint32_t)nrows, (uint32_t)nlayers, (uint32_t)nodes_};
    params_.upload(&p, sizeof p);
}

MassOperator::MassOperator(Context &context, const MassStencil &stencil)
    : MassOperator(context, stencil.ncols, stencil.nrows, stencil.nlayers)
{
    if(stencil.coef.size() != MassStencil::slots * nodes_)
        throw Error("gpu::MassOperator: stencil size doesn't match its grid");
    coef_.upload(stencil.coef);
}

void MassOperator::apply(const Buffer &x, Buffer &y)
{
    if(x.size() < nodes_ * sizeof(float) || y.size() < nodes_ * sizeof(float))
        throw Error("gpu::MassOperator::apply: vector shorter than the grid");
    std::size_t groups = (nodes_ + 255) / 256;
    std::size_t gx = groups < maxGroupsX ? groups : maxGroupsX;
    std::size_t gy = (groups + gx - 1) / gx;
    kernel_.dispatch({&params_, &coef_, &x, &y}, (uint32_t)gx, (uint32_t)gy);
}

namespace {

struct PcgParams
{
    uint32_t ncols, nrows, nlayers, nodes, groups, start, pad1, pad2;
};

enum Scalar { RHO = 0, PQ, ALPHA, BETA, RR, SCALARS = 8 };

std::string pcgSource(const char *stage)
{
    return std::string(kernels::massPcgPrelude) + stage;
}

} // namespace

MassPcg::MassPcg(Context &context, int ncols, int nrows, int nlayers)
    : context_(context), nodes_((std::size_t)ncols * nrows * nlayers),
      gx_(0), gy_(0), op_(context, ncols, nrows, nlayers),
      params_(context, sizeof(PcgParams), Buffer::Uniform),
      startParams_(context, sizeof(PcgParams), Buffer::Uniform),
      b_(context, nodes_ * sizeof(float)), x_(context, nodes_ * sizeof(float)),
      r_(context, nodes_ * sizeof(float)), z_(context, nodes_ * sizeof(float)),
      p_(context, nodes_ * sizeof(float)), q_(context, nodes_ * sizeof(float)),
      scalars_(context, SCALARS * sizeof(float)),
      partials_(context, 2 * ((nodes_ + 255) / 256 + maxGroupsX) * sizeof(float)),
      direction_(context, pcgSource(kernels::massPcgDirection).c_str()),
      apply_(context, pcgSource(kernels::massPcgApply).c_str()),
      alpha_(context, pcgSource(kernels::massPcgAlpha).c_str()),
      update_(context, pcgSource(kernels::massPcgUpdate).c_str()),
      beta_(context, pcgSource(kernels::massPcgBeta).c_str()),
      reducer_(context)
{
    std::size_t groups = (nodes_ + 255) / 256;
    gx_ = (uint32_t)(groups < maxGroupsX ? groups : maxGroupsX);
    gy_ = (uint32_t)((groups + gx_ - 1) / gx_);
    PcgParams pp = {(uint32_t)ncols, (uint32_t)nrows, (uint32_t)nlayers,
                    (uint32_t)nodes_, gx_ * gy_, 0, 0, 0};
    params_.upload(&pp, sizeof pp);
    pp.start = 1;
    startParams_.upload(&pp, sizeof pp);
}

MassPcg::MassPcg(Context &context, const MassStencil &stencil)
    : MassPcg(context, stencil.ncols, stencil.nrows, stencil.nlayers)
{
    if(stencil.coef.size() != MassStencil::slots * nodes_)
        throw Error("gpu::MassPcg: stencil size doesn't match its grid");
    op_.coefficients().upload(stencil.coef);
}

void MassPcg::dispatchNodes(Kernel &kernel, const std::vector<const Buffer *> &bindings)
{
    kernel.dispatch(bindings, gx_, gy_);
}

double MassPcg::residualSquared()
{
    float rr;
    scalars_.download(&rr, sizeof rr, RR * sizeof(float));
    if(!std::isfinite(rr))
        throw Error("gpu::MassPcg: the residual is no longer finite");
    return rr;
}

MassPcg::Result MassPcg::solve(const std::vector<double> &b, std::vector<double> &x, double tol,
                               int maxIterations, int checkEvery,
                               const std::function<void(int, double)> &onCheck)
{
    if(b.size() != nodes_ || x.size() != nodes_)
        throw Error("gpu::MassPcg::solve: vectors don't match the grid");
    std::vector<float> xf(x.begin(), x.end()), bf(b.begin(), b.end());
    x_.upload(xf);
    b_.upload(bf);
    Result result = solveOnDevice(tol, maxIterations, checkEvery, onCheck);
    x_.download(xf.data(), nodes_ * sizeof(float));
    x.assign(xf.begin(), xf.end());
    return result;
}

MassPcg::Result MassPcg::solveOnDevice(double tol, int maxIterations, int checkEvery,
                                       const std::function<void(int, double)> &onCheck)
{
    if(checkEvery < 1)
        checkEvery = 1;
    double normb = std::sqrt(reducer_.sum(b_, 0, nodes_, true));
    if(normb == 0.0)
        normb = 1.0;

    // start: q = A x, r = b - q, z = r / diag, rho = z.r; p holds b until
    // the first direction, and rho starts at infinity so that beta = 0.
    p_.copyFrom(b_, nodes_ * sizeof(float));
    float scalars[SCALARS] = {INFINITY, 0, 0, 0, 0, 0, 0, 0};
    scalars_.upload(scalars, sizeof scalars);
    op_.apply(x_, q_);
    const Buffer &coef = op_.coefficients();
    dispatchNodes(update_, {&startParams_, &scalars_, &x_, &p_, &q_, &r_, &coef, &z_, &partials_});
    beta_.dispatch({&params_, &partials_, &scalars_}, 1);

    Result result;
    result.residual = std::sqrt(residualSquared()) / normb;
    for(int i = 1; i <= maxIterations && result.residual > tol; i++)
    {
        dispatchNodes(direction_, {&params_, &scalars_, &z_, &p_});
        dispatchNodes(apply_, {&params_, &coef, &p_, &q_, &partials_});
        alpha_.dispatch({&params_, &partials_, &scalars_}, 1);
        dispatchNodes(update_, {&params_, &scalars_, &x_, &p_, &q_, &r_, &coef, &z_, &partials_});
        beta_.dispatch({&params_, &partials_, &scalars_}, 1);
        result.iterations = i;
        if(i % checkEvery == 0 || i == maxIterations)
        {
            result.residual = std::sqrt(residualSquared()) / normb;
            if(onCheck)
                onCheck(i, result.residual);
        }
    }
    result.converged = result.residual <= tol;
    return result;
}

MassGeometry MakeMassGeometry(int ncols, int nrows, int nlayers, const double *x,
                              const double *y, const double *z)
{
    MassGeometry g;
    g.ncols = ncols;
    g.nrows = nrows;
    g.nlayers = nlayers;
    const std::size_t n = g.nodes();
    double zMin = z[0];
    for(std::size_t i = 1; i < n; i++)
        zMin = std::min(zMin, z[i]);
    g.xyz.resize(3 * n);
    g.agl.resize(n);
    const std::size_t layer = (std::size_t)ncols * nrows;
    for(std::size_t i = 0; i < n; i++)
    {
        g.xyz[i] = (float)(x[i] - x[0]);
        g.xyz[n + i] = (float)(y[i] - y[0]);
        g.xyz[2 * n + i] = (float)(z[i] - zMin);
        g.agl[i] = (float)(z[i] - z[i % layer]);
    }
    return g;
}

namespace {

struct AssembleParams
{
    uint32_t ncols, nrows, nlayers, nodes;
    float alphaH;
    uint32_t pad[3];
};

struct OutputParams   // kernels::massOutputWind's Params
{
    uint32_t ncols, nrows, nlayers, northFirst;
    float outputHeight, roughness, roughH, roughD, blHeight;
    uint32_t pad[3];
};

} // namespace

MassFem::MassFem(Context &context, const MassGeometry &geometry)
    : context_(context), nodes_(geometry.nodes()), ncols_(geometry.ncols),
      nrows_(geometry.nrows), nlayers_(geometry.nlayers),
      params_(context, sizeof(AssembleParams), Buffer::Uniform),
      outputParams_(context, sizeof(OutputParams), Buffer::Uniform),
      xyz_(context, geometry.xyz.size() * sizeof(float)),
      agl_(context, geometry.agl.size() * sizeof(float)), folded_(context, sizeof(uint32_t)),
      assemble_(context, (std::string(kernels::massFemPrelude) + kernels::massAssemble).c_str()),
      recover_(context, (std::string(kernels::massFemPrelude) + kernels::massRecover).c_str()),
      output_(context, kernels::massOutputWind)
{
    if(geometry.xyz.size() != 3 * nodes_ || geometry.agl.size() != nodes_ || nodes_ > UINT32_MAX)
        throw Error("gpu::MassFem: geometry size doesn't match its grid");
    xyz_.upload(geometry.xyz);
    agl_.upload(geometry.agl);
}

void MassFem::dispatch(Kernel &kernel, const std::vector<const Buffer *> &bindings, float alphaH)
{
    AssembleParams p = {(uint32_t)ncols_, (uint32_t)nrows_, (uint32_t)nlayers_, (uint32_t)nodes_,
                        alphaH, {0, 0, 0}};
    params_.upload(&p, sizeof p);
    std::size_t groups = (nodes_ + 255) / 256;
    std::size_t gx = groups < maxGroupsX ? groups : maxGroupsX;
    std::size_t gy = (groups + gx - 1) / gx;
    kernel.dispatch(bindings, (uint32_t)gx, (uint32_t)gy);
}

void MassFem::assemble(const Buffer &wind0, const Buffer &alphaV, float alphaH, Buffer &coef,
                       Buffer &rhs)
{
    const std::size_t f = sizeof(float);
    if(wind0.size() < 3 * nodes_ * f || alphaV.size() < nodes_ * f ||
       coef.size() < MassStencil::slots * nodes_ * f || rhs.size() < nodes_ * f)
        throw Error("gpu::MassFem::assemble: buffer shorter than the grid");
    uint32_t zero = 0;
    folded_.upload(&zero, sizeof zero);
    dispatch(assemble_, {&params_, &xyz_, &wind0, &alphaV, &coef, &rhs, &folded_}, alphaH);
}

void MassFem::outputWind(const Buffer &wind, const OutputWindParams &p, Buffer &out)
{
    const std::size_t f = sizeof(float), cells = (std::size_t)ncols_ * nrows_;
    if(wind.size() < 3 * nodes_ * f || out.size() < 2 * cells * f)
        throw Error("gpu::MassFem::outputWind: buffer shorter than the grid");
    OutputParams q = {(uint32_t)ncols_, (uint32_t)nrows_, (uint32_t)nlayers_,
                      p.northFirst ? 1u : 0u, p.outputHeight, p.roughness, p.roughH, p.roughD,
                      p.blHeight, {0, 0, 0}};
    outputParams_.upload(&q, sizeof q);
    std::size_t groups = (cells + 255) / 256;
    std::size_t gx = groups < maxGroupsX ? groups : maxGroupsX;
    std::size_t gy = (groups + gx - 1) / gx;
    output_.dispatch({&outputParams_, &agl_, &wind, &out}, (uint32_t)gx, (uint32_t)gy);
}

bool MassFem::foldedElements()
{
    uint32_t flag = 0;
    folded_.download(&flag, sizeof flag);
    return flag != 0;
}

void MassFem::recover(const Buffer &phi, const Buffer &wind0, const Buffer &alphaV,
                      float alphaH, Buffer &wind)
{
    const std::size_t f = sizeof(float);
    if(phi.size() < nodes_ * f || wind0.size() < 3 * nodes_ * f || alphaV.size() < nodes_ * f ||
       wind.size() < 3 * nodes_ * f)
        throw Error("gpu::MassFem::recover: buffer shorter than the grid");
    dispatch(recover_, {&params_, &xyz_, &phi, &wind0, &alphaV, &wind}, alphaH);
}

} // namespace gpu
