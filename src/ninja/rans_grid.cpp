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

#include "rans_grid.h"

#include <stdexcept>


RansGrid RansGrid::FromNodes(int nx, int ny, int nz,
                             const std::vector<double> &x,
                             const std::vector<double> &y,
                             const std::vector<double> &z)
{
    if(nx < 1 || ny < 1 || nz < 1)
        throw std::invalid_argument("RansGrid needs at least one cell in each direction");
    size_t nNodes = (size_t)(nx + 1) * (ny + 1) * (nz + 1);
    if(x.size() != nNodes || y.size() != nNodes || z.size() != nNodes)
        throw std::invalid_argument("RansGrid node arrays don't match nx, ny, nz");

    RansGrid g;
    g.nx = nx;
    g.ny = ny;
    g.nz = nz;
    g.xOrigin = x[0];
    g.yOrigin = y[0];
    g.x.resize(nNodes);
    g.y.resize(nNodes);
    g.z = z;
    for(size_t n = 0; n < nNodes; n++)
    {
        g.x[n] = x[n] - g.xOrigin;
        g.y[n] = y[n] - g.yOrigin;
    }
    g.computeMetrics();
    return g;
}

/*
** Area vector of the quad a-b-c-d (in order around its edge) as half the
** cross product of its diagonals: the exact vector area of the bilinear
** surface, which makes every closed cell's outward vectors sum to zero.
*/
static void QuadAreaVector(const RansGrid &g, int a, int b, int c, int d,
                           double &sx, double &sy, double &sz,
                           double &fx, double &fy, double &fz)
{
    double d1x = g.x[c] - g.x[a], d1y = g.y[c] - g.y[a], d1z = g.z[c] - g.z[a];
    double d2x = g.x[d] - g.x[b], d2y = g.y[d] - g.y[b], d2z = g.z[d] - g.z[b];
    sx = 0.5 * (d1y * d2z - d1z * d2y);
    sy = 0.5 * (d1z * d2x - d1x * d2z);
    sz = 0.5 * (d1x * d2y - d1y * d2x);
    fx = 0.25 * (g.x[a] + g.x[b] + g.x[c] + g.x[d]);
    fy = 0.25 * (g.y[a] + g.y[b] + g.y[c] + g.y[d]);
    fz = 0.25 * (g.z[a] + g.z[b] + g.z[c] + g.z[d]);
}

void RansGrid::computeMetrics()
{
    size_t nI = (size_t)(nx + 1) * ny * nz;
    size_t nJ = (size_t)nx * (ny + 1) * nz;
    size_t nK = (size_t)nx * ny * (nz + 1);
    for(std::vector<double> *v : {&iSx, &iSy, &iSz, &iFx, &iFy, &iFz, &iAgl}) v->resize(nI);
    for(std::vector<double> *v : {&jSx, &jSy, &jSz, &jFx, &jFy, &jFz, &jAgl}) v->resize(nJ);
    for(std::vector<double> *v : {&kSx, &kSy, &kSz, &kFx, &kFy, &kFz}) v->resize(nK);

    // corners are ordered so the area vector points along +i, +j or +k
    for(int k = 0; k < nz; k++)
        for(int j = 0; j < ny; j++)
            for(int i = 0; i <= nx; i++)
            {
                int f = iFace(i, j, k);
                QuadAreaVector(*this, node(i, j, k), node(i, j + 1, k),
                               node(i, j + 1, k + 1), node(i, j, k + 1),
                               iSx[f], iSy[f], iSz[f], iFx[f], iFy[f], iFz[f]);
                iAgl[f] = iFz[f] - 0.5 * (z[node(i, j, 0)] + z[node(i, j + 1, 0)]);
            }
    for(int k = 0; k < nz; k++)
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i < nx; i++)
            {
                int f = jFace(i, j, k);
                QuadAreaVector(*this, node(i, j, k), node(i, j, k + 1),
                               node(i + 1, j, k + 1), node(i + 1, j, k),
                               jSx[f], jSy[f], jSz[f], jFx[f], jFy[f], jFz[f]);
                jAgl[f] = jFz[f] - 0.5 * (z[node(i, j, 0)] + z[node(i + 1, j, 0)]);
            }
    for(int k = 0; k <= nz; k++)
        for(int j = 0; j < ny; j++)
            for(int i = 0; i < nx; i++)
            {
                int f = kFace(i, j, k);
                QuadAreaVector(*this, node(i, j, k), node(i + 1, j, k),
                               node(i + 1, j + 1, k), node(i, j + 1, k),
                               kSx[f], kSy[f], kSz[f], kFx[f], kFy[f], kFz[f]);
            }

    size_t nC = (size_t)nCells();
    for(std::vector<double> *v : {&volume, &cx, &cy, &cz, &wallDistance}) v->resize(nC);
    for(int k = 0; k < nz; k++)
        for(int j = 0; j < ny; j++)
            for(int i = 0; i < nx; i++)
            {
                int c = cell(i, j, k);
                int n0 = node(i, j, k);

                // divergence theorem over the six outward faces, relative to a
                // corner to limit cancellation
                double v = 0.0;
                auto add = [&](double sign, double sx, double sy, double sz,
                               double fx, double fy, double fz)
                {
                    v += sign * ((fx - x[n0]) * sx + (fy - y[n0]) * sy + (fz - z[n0]) * sz);
                };
                int f;
                f = iFace(i, j, k);     add(-1, iSx[f], iSy[f], iSz[f], iFx[f], iFy[f], iFz[f]);
                f = iFace(i + 1, j, k); add(+1, iSx[f], iSy[f], iSz[f], iFx[f], iFy[f], iFz[f]);
                f = jFace(i, j, k);     add(-1, jSx[f], jSy[f], jSz[f], jFx[f], jFy[f], jFz[f]);
                f = jFace(i, j + 1, k); add(+1, jSx[f], jSy[f], jSz[f], jFx[f], jFy[f], jFz[f]);
                f = kFace(i, j, k);     add(-1, kSx[f], kSy[f], kSz[f], kFx[f], kFy[f], kFz[f]);
                f = kFace(i, j, k + 1); add(+1, kSx[f], kSy[f], kSz[f], kFx[f], kFy[f], kFz[f]);
                volume[c] = v / 3.0;

                double sx = 0.0, sy = 0.0, sz = 0.0;
                for(int dk = 0; dk <= 1; dk++)
                    for(int dj = 0; dj <= 1; dj++)
                        for(int di = 0; di <= 1; di++)
                        {
                            int n = node(i + di, j + dj, k + dk);
                            sx += x[n];
                            sy += y[n];
                            sz += z[n];
                        }
                cx[c] = sx / 8.0;
                cy[c] = sy / 8.0;
                cz[c] = sz / 8.0;

                double ground = 0.25 * (z[node(i, j, 0)] + z[node(i + 1, j, 0)] +
                                        z[node(i, j + 1, 0)] + z[node(i + 1, j + 1, 0)]);
                wallDistance[c] = cz[c] - ground;
            }
}
