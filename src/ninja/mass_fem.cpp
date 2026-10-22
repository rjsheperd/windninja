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

#include "mass_fem.h"

#include <cmath>
#include <stdexcept>

namespace {

const int NNPE = 8;      // nodes per element
const double WT = 8.0;   // one-point quadrature weight

/* Shape functions and their u, v, w derivatives at the element centre
** (u = v = w = 0), local nodes counter-clockwise from the lower-left,
** bottom layer first (element::SFNV, SFNVu, SFNVv, SFNVw). */
const double SF0[NNPE] = {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125};
const double SFU[NNPE] = {-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125};
const double SFV[NNPE] = {-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125};
const double SFW[NNPE] = {-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125};

/* Global node numbers of element e's local nodes (Mesh::get_global_node). */
void ElementNodes(const MassGrid &g, int e, int nodes[NNPE])
{
    int layerElements = (g.nrows - 1) * (g.ncols - 1);
    int layer = e / layerElements;
    int row = (e - layer * layerElements) / (g.ncols - 1);
    int col = e - layer * layerElements - row * (g.ncols - 1);
    int node0 = layer * (g.ncols * g.nrows) + row * g.ncols + col;
    int up = g.ncols * g.nrows;
    nodes[0] = node0;
    nodes[1] = node0 + 1;
    nodes[2] = node0 + g.ncols + 1;
    nodes[3] = node0 + g.ncols;
    nodes[4] = node0 + up;
    nodes[5] = node0 + up + 1;
    nodes[6] = node0 + up + g.ncols + 1;
    nodes[7] = node0 + up + g.ncols;
}

/* The Jacobian at the element centre and the shape functions' x, y, z
** derivatives there (element::computeJacobianQuadraturePoint), and the
** centre's coordinates.  compute() returns false for a folded element
** (DETJ <= 0): an exception can't leave an OpenMP loop. */
struct CentreElement
{
    double DETJ;
    double DNDX[NNPE], DNDY[NNPE], DNDZ[NNPE];
    double x, y, z;

    bool compute(const MassGrid &g, const int nodes[NNPE])
    {
        double J[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        x = 0.0;
        y = 0.0;
        z = 0.0;
        for(int k = 0; k < NNPE; k++)
        {
            int n = nodes[k];
            x = x + SF0[k] * g.x[n];
            y = y + SF0[k] * g.y[n];
            z = z + SF0[k] * g.z[n];
            J[0] = J[0] + SFU[k] * g.x[n];
            J[1] = J[1] + SFV[k] * g.x[n];
            J[2] = J[2] + SFW[k] * g.x[n];
            J[3] = J[3] + SFU[k] * g.y[n];
            J[4] = J[4] + SFV[k] * g.y[n];
            J[5] = J[5] + SFW[k] * g.y[n];
            J[6] = J[6] + SFU[k] * g.z[n];
            J[7] = J[7] + SFV[k] * g.z[n];
            J[8] = J[8] + SFW[k] * g.z[n];
        }
        DETJ = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) +
               J[2] * (J[3] * J[7] - J[4] * J[6]);
        if(DETJ <= 0)
            return false;
        double I[9];
        I[0] = (J[4] * J[8] - J[5] * J[7]) / DETJ;
        I[1] = (J[2] * J[7] - J[1] * J[8]) / DETJ;
        I[2] = (J[1] * J[5] - J[2] * J[4]) / DETJ;
        I[3] = (J[5] * J[6] - J[3] * J[8]) / DETJ;
        I[4] = (J[0] * J[8] - J[2] * J[6]) / DETJ;
        I[5] = (J[2] * J[3] - J[0] * J[5]) / DETJ;
        I[6] = (J[3] * J[7] - J[4] * J[6]) / DETJ;
        I[7] = (J[1] * J[6] - J[0] * J[7]) / DETJ;
        I[8] = (J[0] * J[4] - J[1] * J[3]) / DETJ;
        // the transpose of the inverse
        for(int k = 0; k < NNPE; k++)
        {
            DNDX[k] = I[0] * SFU[k] + I[3] * SFV[k] + I[6] * SFW[k];
            DNDY[k] = I[1] * SFU[k] + I[4] * SFV[k] + I[7] * SFW[k];
            DNDZ[k] = I[2] * SFU[k] + I[5] * SFV[k] + I[8] * SFW[k];
        }
        return true;
    }
};

void ThrowIfFolded(bool folded)
{
    if(folded)
        throw std::runtime_error("Volume Jacobian 1 is zero or negative.");
}

/* The upper-triangle CSR pattern: each node's 27-point neighbourhood
** (fewer at the boundary), columns >= row, in i, j, k order. */
void BuildPattern(const MassGrid &g, MassSystem &s)
{
    s.row_ptr.assign(g.numNodes() + 1, 0);
    s.col_ind.clear();
    for(int k = 0; k < g.nlayers; k++)
        for(int i = 0; i < g.nrows; i++)
            for(int j = 0; j < g.ncols; j++)
            {
                int row = k * g.ncols * g.nrows + i * g.ncols + j;
                s.row_ptr[row] = (int)s.col_ind.size();
                for(int kk = -1; kk < 2; kk++)
                    for(int ii = -1; ii < 2; ii++)
                        for(int jj = -1; jj < 2; jj++)
                        {
                            if(i + ii < 0 || i + ii > g.nrows - 1)
                                continue;
                            if(j + jj < 0 || j + jj > g.ncols - 1)
                                continue;
                            if(k + kk < 0 || k + kk > g.nlayers - 1)
                                continue;
                            int col = (k + kk) * g.ncols * g.nrows + (i + ii) * g.ncols + (j + jj);
                            if(col >= row)
                                s.col_ind.push_back(col);
                        }
            }
    s.row_ptr[g.numNodes()] = (int)s.col_ind.size();
    s.SK.assign(s.col_ind.size(), 0.0);
    s.RHS.assign(g.numNodes(), 0.0);
}

/* Phi = 0 on the sides and top: a boundary node's row becomes the identity
** and its column is zeroed (no right-hand-side terms move, since Phi = 0). */
void ApplyBoundaryConditions(const MassGrid &g, MassSystem &s)
{
    std::vector<char> boundary(g.numNodes());
    for(int k = 0; k < g.nlayers; k++)
        for(int i = 0; i < g.nrows; i++)
            for(int j = 0; j < g.ncols; j++)
                boundary[k * g.ncols * g.nrows + i * g.ncols + j] =
                    j == 0 || j == g.ncols - 1 || i == 0 || i == g.nrows - 1 || k == g.nlayers - 1;

    #pragma omp parallel for
    for(int row = 0; row < g.numNodes(); row++)
    {
        for(int l = s.row_ptr[row]; l < s.row_ptr[row + 1]; l++)
        {
            int col = s.col_ind[l];
            if(boundary[col])
                s.SK[l] = 0.;
            if(boundary[row])
                s.SK[l] = col == row ? 1. : 0.;
        }
        if(boundary[row])
            s.RHS[row] = 0.;
    }
}

} // namespace

MassSystem AssembleMass(const MassGrid &grid, const double *u0, const double *v0,
                        const double *w0, const double *alphaV, double alphaH)
{
    MassSystem s;
    BuildPattern(grid, s);
    double *SK = s.SK.data();
    double *RHS = s.RHS.data();
    bool folded = false;

    #pragma omp parallel for
    for(int e = 0; e < grid.numElements(); e++)
    {
        int nodes[NNPE];
        ElementNodes(grid, e, nodes);
        CentreElement elem;
        if(!elem.compute(grid, nodes))
        {
            #pragma omp atomic write
            folded = true;
            continue;
        }

        // H = du0/dx + dv0/dy + dw0/dz, and alphaV, at the centre
        double HVJ = 0.0;
        double alphaVJ = 0;
        for(int k = 0; k < NNPE; k++)
        {
            int n = nodes[k];
            HVJ = HVJ + ((elem.DNDX[k] * u0[n]) + (elem.DNDY[k] * v0[n]) + (elem.DNDZ[k] * w0[n]));
            alphaVJ = alphaVJ + SF0[k] * alphaV[n];
        }
        double RX = 1.0 / (2.0 * alphaH * alphaH);
        double RY = 1.0 / (2.0 * alphaH * alphaH);
        double RZ = 1.0 / (2.0 * alphaVJ * alphaVJ);
        double DV = elem.DETJ;

        double QE[NNPE], S[NNPE * NNPE];
        for(int k = 0; k < NNPE; k++)
        {
            QE[k] = 0.0 + WT * SF0[k] * HVJ * DV;
            for(int l = 0; l < NNPE; l++)
                S[k * NNPE + l] = 0.0 + WT * (elem.DNDX[k] * RX * elem.DNDX[l] +
                                              elem.DNDY[k] * RY * elem.DNDY[l] +
                                              elem.DNDZ[k] * RZ * elem.DNDZ[l]) * DV;
        }

        for(int j = 0; j < NNPE; j++)
        {
            int row = nodes[j];
            #pragma omp atomic
            RHS[row] += QE[j];
            for(int k = 0; k < NNPE; k++)
            {
                int col = nodes[k];
                if(col < row)
                    continue;
                int pos = s.row_ptr[row];
                while(s.col_ind[pos] != col)
                    pos++;
                #pragma omp atomic
                SK[pos] += S[j * NNPE + k];
            }
        }
    }

    ThrowIfFolded(folded);
    ApplyBoundaryConditions(grid, s);
    return s;
}

void RecoverVelocity(const MassGrid &grid, const double *phi, const double *u0,
                     const double *v0, const double *w0, const double *alphaV,
                     double alphaH, double *u, double *v, double *w)
{
    const int numNodes = grid.numNodes();
    std::vector<double> diag(numNodes, 0.0);
    bool folded = false;
    for(int i = 0; i < numNodes; i++)
    {
        u[i] = 0.;
        v[i] = 0.;
        w[i] = 0.;
    }

    #pragma omp parallel
    {
        // per-thread sums of weight * gradient and of the weights
        std::vector<double> uS(numNodes, 0.), vS(numNodes, 0.), wS(numNodes, 0.), dS(numNodes, 0.);

        #pragma omp for
        for(int e = 0; e < grid.numElements(); e++)
        {
            int nodes[NNPE];
            ElementNodes(grid, e, nodes);
            CentreElement elem;
            if(!elem.compute(grid, nodes))
            {
                #pragma omp atomic write
                folded = true;
                continue;
            }

            double DPHIDX = 0.0, DPHIDY = 0.0, DPHIDZ = 0.0;
            for(int k = 0; k < NNPE; k++)
            {
                DPHIDX = DPHIDX + elem.DNDX[k] * phi[nodes[k]];
                DPHIDY = DPHIDY + elem.DNDY[k] * phi[nodes[k]];
                DPHIDZ = DPHIDZ + elem.DNDZ[k] * phi[nodes[k]];
            }
            // inverse-distance weights from the centre to each node
            for(int k = 0; k < NNPE; k++)
            {
                int n = nodes[k];
                double wght = std::pow((grid.x[n] - elem.x), 2) + std::pow((grid.y[n] - elem.y), 2) +
                              std::pow((grid.z[n] - elem.z), 2);
                wght = 1.0 / (std::sqrt(wght));
                uS[n] = uS[n] + wght * DPHIDX;
                vS[n] = vS[n] + wght * DPHIDY;
                wS[n] = wS[n] + wght * DPHIDZ;
                dS[n] = dS[n] + wght;
            }
        }

        #pragma omp critical
        {
            for(int i = 0; i < numNodes; i++)
            {
                u[i] += uS[i];
                v[i] += vS[i];
                w[i] += wS[i];
                diag[i] += dS[i];
            }
        }

        #pragma omp barrier

        #pragma omp for
        for(int i = 0; i < numNodes; i++)
        {
            u[i] = u[i] / diag[i];   // the smoothed dPhi/dx, etc.
            v[i] = v[i] / diag[i];
            w[i] = w[i] / diag[i];
            double aV = alphaV[i];
            u[i] = u0[i] + 1.0 / (2.0 * alphaH * alphaH) * u[i];
            v[i] = v0[i] + 1.0 / (2.0 * alphaH * alphaH) * v[i];
            w[i] = w0[i] + 1.0 / (2.0 * aV * aV) * w[i];
        }
    }
    ThrowIfFolded(folded);
}
