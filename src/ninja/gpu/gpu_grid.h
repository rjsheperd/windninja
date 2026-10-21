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

#ifndef GPU_GRID_H
#define GPU_GRID_H

#include <cstddef>
#include <vector>

#include "gpu_context.h"

/*
** The structured terrain-following grid of the RANS solver on the GPU.
**
** WebGPU guarantees only 8 storage buffers per shader stage, so the ~25
** metric arrays are packed: one buffer of cell metrics and one per face
** direction, each a structure of arrays whose slot s starts at s * count.
** Values are f32.  x and y are already local to the domain; z is shifted by
** zOrigin (the lowest cell centroid) so elevations keep their precision.
**
** Indexing matches RansGrid: cell(i, j, k) = i + nx * (j + ny * k), and the
** i, j and k faces as RansGrid::iFace / jFace / kFace.
*/
namespace gpu {

// non-owning view of a grid's metrics (RansGrid's arrays, or a browser's)
struct GridView
{
    int nx = 0, ny = 0, nz = 0;
    const double *volume = nullptr;
    const double *cx = nullptr, *cy = nullptr, *cz = nullptr;
    const double *wallDistance = nullptr;
    // per face direction (i, j, k): area vector, centroid, and for i and j
    // the centroid height above the ground below it (k faces have none)
    const double *s[3][3] = {};    // s[direction][component]
    const double *f[3][3] = {};    // f[direction][component]
    const double *agl[2] = {};     // i, j
};

enum CellSlot { cellVolume, cellX, cellY, cellZ, cellWallDistance, cellSlots };
enum FaceSlot { faceSx, faceSy, faceSz, faceX, faceY, faceZ, faceAgl, faceSlots };
enum FieldSlot { fieldP, fieldU, fieldV, fieldW, fieldK, fieldEpsilon, fieldNut,
                 fieldSlots };

class Grid
{
public:
    Grid(Context &context, const GridView &view);

    int nx, ny, nz;
    std::size_t nCells, nFaces[3];   // faces per direction (i, j, k)
    double zOrigin;
    Buffer cells;                    // cellSlots * nCells floats
    Buffer faces[3];                 // faceSlots * nFaces[d] floats
};

// The solver state: fieldSlots * nCells floats in one buffer.
class Fields
{
public:
    Fields(Context &context, std::size_t nCells);

    void upload(FieldSlot slot, const std::vector<double> &values);
    std::vector<double> download(FieldSlot slot);

    std::size_t nCells;
    Buffer buffer;
};

} // namespace gpu

#endif /* GPU_GRID_H */
