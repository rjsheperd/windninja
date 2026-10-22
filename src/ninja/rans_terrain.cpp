/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Terrain-following RANS grid from a plain DEM
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

#include "rans_terrain.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

RansGrid TerrainGrid(const TerrainDem &dem, double meshResolution, double firstLayer,
                     int layers, double growth)
{
    const int nRows = dem.nRows, nCols = dem.nCols;
    const double cs = dem.cellSize;
    const int nx = nCols, ny = nRows, nz = layers;
    const double zMax = *std::max_element(dem.z.begin(), dem.z.end());
    const double zMin = *std::min_element(dem.z.begin(), dem.z.end());

    // OpenFOAM's domain top (NinjaFoam::SetBlockMeshParametersFromDem),
    // including its minimum of 11 block layers; block cells are 2 x the mesh
    // resolution for one refinement round (coarse and medium meshes)
    double relief = zMax - zMin;
    double blockMeshDz = std::max(0.1 * std::max((nCols * cs), (nRows * cs)),
                                  relief + 0.1 * relief);
    blockMeshDz = std::max(blockMeshDz, 11 * 2.0 * meshResolution);
    double top = zMax + blockMeshDz;

    // fractions of the column above the first layer, growing geometrically
    std::vector<double> frac(nz, 0.0);
    double sum = 0.0;
    for(int m = 0; m < nz - 1; m++) sum += std::pow(growth, m);
    for(int k = 2; k < nz; k++) frac[k] = frac[k - 1] + std::pow(growth, k - 2) / sum;

    std::vector<double> x, y, z;
    size_t nNodes = (size_t)(nx + 1) * (ny + 1) * (nz + 1);
    x.reserve(nNodes); y.reserve(nNodes); z.reserve(nNodes);
    for(int k = 0; k <= nz; k++)
        for(int j = 0; j <= ny; j++)          // AsciiGrid row j (row 0 is south)
            for(int i = 0; i <= nx; i++)      // AsciiGrid column i
            {
                // ground at a cell corner: mean of the DEM cells touching it
                double g = 0.0;
                int n = 0;
                for(int r = j - 1; r <= j; r++)
                    for(int c = i - 1; c <= i; c++)
                        if(r >= 0 && r < nRows && c >= 0 && c < nCols)
                        {
                            g += dem.at(r, c);
                            n++;
                        }
                g /= n;
                double zk;
                if(k == 0) zk = g;
                else if(k == nz) zk = top;
                else zk = g + firstLayer + (top - g - firstLayer) * frac[k];
                x.push_back(dem.xll + i * cs);
                y.push_back(dem.yll + j * cs);
                z.push_back(zk);
            }
    if(top - zMax <= 2.0 * firstLayer)
        throw std::runtime_error("RANS grid: no room above the first layer under the domain top");
    return RansGrid::FromNodes(nx, ny, nz, x, y, z);
}
