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

#ifndef RANS_TERRAIN_H
#define RANS_TERRAIN_H

#include <vector>

#include "rans_grid.h"

/* A DEM as plain arrays: elevations per cell, row 0 south (as AsciiGrid),
** with the lower-left corner and the cell size.  Free of WindNinja core, so
** the browser build can use it. */
struct TerrainDem
{
    int nCols = 0, nRows = 0;
    double cellSize = 0.0, xll = 0.0, yll = 0.0;
    std::vector<double> z;   // nRows * nCols, row-major

    double at(int r, int c) const { return z[(size_t)r * nCols + c]; }
};

/*
** The RANS backends' grid (CpuRansBackend::buildGrid): cell centres at the
** DEM's cell centres, a first layer firstLayer thick, then layers growing
** geometrically to OpenFOAM's flat domain top (max elevation +
** blockMeshDz, from the mesh resolution).
*/
RansGrid TerrainGrid(const TerrainDem &dem, double meshResolution, double firstLayer,
                     int layers, double growth);

#endif /* RANS_TERRAIN_H */
