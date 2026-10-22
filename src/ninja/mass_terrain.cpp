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

#include "mass_terrain.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

double MassMeshResolution(int nCols, int nRows, double cellSize, long targetCells)
{
    double Xlength = nCols * cellSize;
    double Ylength = nRows * cellSize;
    double nXcells = 2 * std::sqrt((double)targetCells) * (Xlength / (Xlength + Ylength));
    double nYcells = 2 * std::sqrt((double)targetCells) * (Ylength / (Xlength + Ylength));
    double Xcellsize = Xlength / nXcells;
    double Ycellsize = Ylength / nYcells;
    return (Xcellsize + Ycellsize) / 2;
}

double MassDomainHeight(const TerrainDem &dem, double resolution, const MassMeshOptions &o)
{
    double first_cell_ht = resolution / o.maxAspectRatio;
    double domainHeight = (first_cell_ht * (std::pow(o.vertGrowth, double(o.layers)) - 1) /
                           (o.vertGrowth - 1));
    if(domainHeight < 3 * (o.outputHeight + o.canopyHeight))
        domainHeight = 3 * (o.outputHeight + o.canopyHeight);
    if(domainHeight < 3 * (o.inputHeight + o.canopyHeight))
        domainHeight = 3 * (o.inputHeight + o.canopyHeight);
    return domainHeight + *std::max_element(dem.z.begin(), dem.z.end());
}

MassGrid MassTerrainGrid(const TerrainDem &dem, const MassMeshOptions &o)
{
    if(dem.nCols < 2 || dem.nRows < 2 || o.layers < 2)
        throw std::invalid_argument("MassTerrainGrid: need at least 2 columns, rows and layers");
    if(o.vertGrowth == 1 || o.vertGrowth < 0.0)
        throw std::range_error("MassTerrainGrid: bad vertGrowth");
    const double resolution = dem.cellSize;
    const double domainHeight = MassDomainHeight(dem, resolution, o);
    const int numVertLayers = o.layers;
    const double vertGrowth = o.vertGrowth;

    MassGrid g;
    g.ncols = dem.nCols;
    g.nrows = dem.nRows;
    g.nlayers = o.layers;
    g.x.resize(g.numNodes());
    g.y.resize(g.numNodes());
    g.z.resize(g.numNodes());
    for(int k = 0; k < g.nlayers; k++)
        for(int i = 0; i < g.nrows; i++)
            for(int j = 0; j < g.ncols; j++)
            {
                int n = j + g.ncols * (i + g.nrows * k);
                double elev = dem.at(i, j);
                g.x[n] = (double)j * resolution + 0.5 * resolution;
                g.y[n] = (double)i * resolution + 0.5 * resolution;
                if(k == 0)
                    g.z[n] = elev;
                else   // Mesh::get_z
                    g.z[n] = (domainHeight - elev) *
                                 ((std::pow(vertGrowth, int(k - numVertLayers + 1)) -
                                   std::pow(vertGrowth, int(1 - numVertLayers))) /
                                  (1 - std::pow(vertGrowth, int(1 - numVertLayers)))) +
                             elev;
            }
    return g;
}
