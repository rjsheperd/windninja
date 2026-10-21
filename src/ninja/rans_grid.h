/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Structured finite-volume grid for the CPU RANS momentum solver
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

#ifndef RANS_GRID_H
#define RANS_GRID_H

#include <vector>

class Mesh;

/*
** A structured, terrain-following hexahedral grid with the finite-volume
** metrics the RANS solver needs.
**
** Cells are indexed (i, j, k) with i along x, j along y and k upward;
** there are nx * ny * nz cells and (nx+1) * (ny+1) * (nz+1) nodes.  Node
** coordinates are local: x and y have the domain origin (node 0,0,0)
** subtracted, so later single-precision code doesn't lose digits to UTM
** offsets.  z is unchanged.
**
** Faces normal to i, j and k are stored separately.  Each face's area vector
** points in the +i, +j or +k direction, and its magnitude is the face area.
*/
class RansGrid
{
public:
    /*
    ** Build from node coordinates laid out as node(i, j, k) =
    ** i + (nx+1) * (j + (ny+1) * k).
    */
    static RansGrid FromNodes(int nx, int ny, int nz,
                              const std::vector<double> &x,
                              const std::vector<double> &y,
                              const std::vector<double> &z);

    /* Build from the mass-solver mesh, whose nodes are (row, col, layer). */
    static RansGrid FromMesh(const Mesh &mesh);

    int nx, ny, nz;
    double xOrigin, yOrigin;    // subtracted from the input x and y

    std::vector<double> x, y, z;            // nodes, local coordinates

    std::vector<double> volume;             // per cell
    std::vector<double> cx, cy, cz;         // cell centroids
    std::vector<double> wallDistance;       // centroid height above ground

    // face area vectors (Sx, Sy, Sz) and face centroids, per face direction
    std::vector<double> iSx, iSy, iSz, iFx, iFy, iFz;   // (nx+1) * ny * nz
    std::vector<double> jSx, jSy, jSz, jFx, jFy, jFz;   // nx * (ny+1) * nz
    std::vector<double> kSx, kSy, kSz, kFx, kFy, kFz;   // nx * ny * (nz+1)

    int node(int i, int j, int k) const { return i + (nx + 1) * (j + (ny + 1) * k); }
    int cell(int i, int j, int k) const { return i + nx * (j + ny * k); }
    int iFace(int i, int j, int k) const { return i + (nx + 1) * (j + ny * k); }
    int jFace(int i, int j, int k) const { return i + nx * (j + (ny + 1) * k); }
    int kFace(int i, int j, int k) const { return i + nx * (j + ny * k); }

    int nCells() const { return nx * ny * nz; }

private:
    void computeMetrics();
};

#endif /* RANS_GRID_H */
