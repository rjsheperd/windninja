/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Mass-consistent solver: FEM assembly and velocity recovery
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

#ifndef MASS_FEM_H
#define MASS_FEM_H

#include <vector>

class Mesh;

/*
** The mass solver's terrain-following mesh as plain arrays: node
** n = col + ncols*(row + nrows*layer), the order of Mesh and wn_3dArray, so
** wn_3dScalarField data passes straight through.  Free of WindNinja core, so
** the browser build can use it.
*/
struct MassGrid
{
    int ncols = 0, nrows = 0, nlayers = 0;
    std::vector<double> x, y, z;   // node coordinates

    int numNodes() const { return ncols * nrows * nlayers; }
    int numElements() const { return (ncols - 1) * (nrows - 1) * (nlayers - 1); }

    static MassGrid FromMesh(const Mesh &mesh);   // mass_grid_mesh.cpp
};

/*
** Phi's linear system: the upper triangle of the symmetric stiffness matrix
** in compressed row storage, and the right-hand side.
*/
struct MassSystem
{
    std::vector<double> SK;        // values
    std::vector<int> col_ind;      // column of each value
    std::vector<int> row_ptr;      // numNodes + 1 offsets into SK
    std::vector<double> RHS;
};

/*
** Builds the system for
**   d/dx(Rx dPhi/dx) + d/dy(Ry dPhi/dy) + d/dz(Rz dPhi/dz) + div(u0) = 0,
**   Rx = Ry = 1/(2 alphaH^2), Rz = 1/(2 alphaV^2),
** with Phi = 0 on the sides and top and zero normal flux at the ground.
** One-point quadrature hexahedra, as ninja::discretize() always did.
** u0, v0, w0 and alphaV are per node.  The same arithmetic, in the same
** order, as the code it replaced; with OpenMP the element sums meet in
** thread order.
*/
MassSystem AssembleMass(const MassGrid &grid, const double *u0, const double *v0,
                        const double *w0, const double *alphaV, double alphaH);

/*
** u = u0 + dPhi/dx / (2 alphaH^2), and likewise v and w (with alphaV),
** with nodal gradients from the inverse-distance-weighted average of the
** surrounding elements' centre gradients ("stress smoothing", Thompson
** p. 228).  u, v, w hold numNodes values.
*/
void RecoverVelocity(const MassGrid &grid, const double *phi, const double *u0,
                     const double *v0, const double *w0, const double *alphaV,
                     double alphaH, double *u, double *v, double *w);

#endif /* MASS_FEM_H */
