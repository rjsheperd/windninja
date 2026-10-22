/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Mass-consistent solver: the terrain-following mesh from a plain DEM
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

#ifndef MASS_TERRAIN_H
#define MASS_TERRAIN_H

#include "mass_fem.h"
#include "rans_terrain.h"   // TerrainDem

/*
** WindNinja's mass-solver mesh (Mesh::buildStandardMesh) from a DEM already
** at the mesh resolution, free of WindNinja core: nodes at the DEM's cell
** centres, x and y from the DEM's lower-left corner (x = (col + 0.5) *
** resolution), the ground layer at the DEM, then layers growing by
** vertGrowth to a flat top at domainHeight.  The same arithmetic as Mesh.
*/
struct MassMeshOptions
{
    int layers = 20;               // the CLI's numVertLayers
    double vertGrowth = 1.3;       // Mesh's defaults
    double maxAspectRatio = 400.0;
    double inputHeight = 6.096;    // m above the canopy, for the domain height
    double outputHeight = 6.096;
    double canopyHeight = 0.0;     // the vegetation's Rough_h (0 for grass)
};

// Mesh::compute_cellsize: the resolution for a target number of horizontal
// cells (coarse 4000, medium 10000, fine 20000) over a DEM.
double MassMeshResolution(int nCols, int nRows, double cellSize, long targetCells);

// Mesh::compute_domain_height (m above sea level) for a resolution.
double MassDomainHeight(const TerrainDem &dem, double resolution, const MassMeshOptions &options);

// The mesh over dem (row 0 south) at its own cell size.
MassGrid MassTerrainGrid(const TerrainDem &dem, const MassMeshOptions &options = MassMeshOptions());

#endif /* MASS_TERRAIN_H */
