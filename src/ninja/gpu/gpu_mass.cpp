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

MassOperator::MassOperator(Context &context, const MassStencil &stencil)
    : context_(context), nodes_(stencil.nodes()),
      params_(context, sizeof(Params), Buffer::Uniform),
      coef_(context, stencil.coef.size() * sizeof(float)),
      kernel_(context, kernels::massApply)
{
    if(stencil.coef.size() != MassStencil::slots * nodes_ || nodes_ > UINT32_MAX)
        throw Error("gpu::MassOperator: stencil size doesn't match its grid");
    Params p = {(uint32_t)stencil.ncols, (uint32_t)stencil.nrows, (uint32_t)stencil.nlayers,
                (uint32_t)nodes_};
    params_.upload(&p, sizeof p);
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

} // namespace gpu
