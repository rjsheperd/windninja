/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  RansGrid from WindNinja's mass-solver mesh (core-dependent part)
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


/* Kept apart from rans_grid.cpp, which stays free of WindNinja core so the
** solver can build for the browser (src/ninja/gpu/wasm). */

#include "rans_grid.h"

#include <vector>

#include "mesh.h"

RansGrid RansGrid::FromMesh(const Mesh &mesh)
{
    int nx = mesh.ncols - 1;
    int ny = mesh.nrows - 1;
    int nz = mesh.nlayers - 1;
    size_t nNodes = (size_t)mesh.ncols * mesh.nrows * mesh.nlayers;
    std::vector<double> x(nNodes), y(nNodes), z(nNodes);
    for(int k = 0; k <= nz; k++)
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i <= nx; i++)
            {
                size_t n = i + (size_t)(nx + 1) * (j + (size_t)(ny + 1) * k);
                // the mass-solver mesh is (row, col, layer) with x along col
                x[n] = mesh.XORD(j, i, k);
                y[n] = mesh.YORD(j, i, k);
                z[n] = mesh.ZORD(j, i, k);
            }
    return FromNodes(nx, ny, nz, x, y, z);
}
