/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Solver grid metrics and fields on the GPU, packed for WebGPU
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

#include "gpu_grid.h"

#include <algorithm>
#include <limits>

namespace gpu {

namespace {

void put(std::vector<float> &packed, int slot, std::size_t count,
         const double *values, double shift = 0.0)
{
    float *out = packed.data() + slot * count;
    for(std::size_t i = 0; i < count; i++)
        out[i] = values == nullptr ? 0.0f : static_cast<float>(values[i] - shift);
}

double lowest(const double *values, std::size_t count)
{
    double m = std::numeric_limits<double>::max();
    for(std::size_t i = 0; i < count; i++)
        m = std::min(m, values[i]);
    return m;
}

} // namespace

Grid::Grid(Context &context, const GridView &view)
    : nx(view.nx), ny(view.ny), nz(view.nz),
      nCells(std::size_t(view.nx) * view.ny * view.nz),
      nFaces{std::size_t(view.nx + 1) * view.ny * view.nz,
             std::size_t(view.nx) * (view.ny + 1) * view.nz,
             std::size_t(view.nx) * view.ny * (view.nz + 1)},
      zOrigin(lowest(view.cz, nCells)),
      cells(context, cellSlots * nCells * sizeof(float)),
      faces{Buffer(context, faceSlots * nFaces[0] * sizeof(float)),
            Buffer(context, faceSlots * nFaces[1] * sizeof(float)),
            Buffer(context, faceSlots * nFaces[2] * sizeof(float))}
{
    std::vector<float> packed(cellSlots * nCells);
    put(packed, cellVolume, nCells, view.volume);
    put(packed, cellX, nCells, view.cx);
    put(packed, cellY, nCells, view.cy);
    put(packed, cellZ, nCells, view.cz, zOrigin);
    put(packed, cellWallDistance, nCells, view.wallDistance);
    cells.upload(packed);

    for(int d = 0; d < 3; d++)
    {
        std::size_t n = nFaces[d];
        packed.assign(faceSlots * n, 0.0f);
        for(int c = 0; c < 3; c++)
        {
            put(packed, faceSx + c, n, view.s[d][c]);
            put(packed, faceX + c, n, view.f[d][c], c == 2 ? zOrigin : 0.0);
        }
        put(packed, faceAgl, n, d < 2 ? view.agl[d] : nullptr);
        faces[d].upload(packed);
    }
}

Fields::Fields(Context &context, std::size_t nCells)
    : nCells(nCells), buffer(context, fieldSlots * nCells * sizeof(float))
{
}

void Fields::upload(FieldSlot slot, const std::vector<double> &values)
{
    if(values.size() != nCells)
        throw Error("gpu::Fields::upload: wrong number of values");
    std::vector<float> f(values.begin(), values.end());
    buffer.upload(f.data(), f.size() * sizeof(float), slot * nCells * sizeof(float));
}

std::vector<double> Fields::download(FieldSlot slot)
{
    std::vector<float> f(nCells);
    buffer.download(f.data(), nCells * sizeof(float), slot * nCells * sizeof(float));
    return std::vector<double>(f.begin(), f.end());
}

} // namespace gpu
