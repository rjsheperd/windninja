/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Test the RANS solver grid metrics
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
#include "mesh.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <boost/test/unit_test.hpp>

/*
** Nodes of a terrain-following grid over [0, lx] x [0, ly] with ground
** height ground(x, y) and a flat top at height top.  Layers are spaced
** geometrically with ratio growth, like the mass-solver mesh.
*/
template <typename Ground>
static RansGrid TerrainGrid(int nx, int ny, int nz, double lx, double ly,
                            double top, double growth, Ground ground,
                            double x0 = 0.0, double y0 = 0.0)
{
    std::vector<double> x, y, z;
    double sum = 0.0;
    for(int k = 0; k < nz; k++) sum += std::pow(growth, k);
    for(int k = 0; k <= nz; k++)
    {
        double frac = 0.0;
        for(int m = 0; m < k; m++) frac += std::pow(growth, m) / sum;
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i <= nx; i++)
            {
                double px = lx * i / nx, py = ly * j / ny;
                double g = ground(px, py);
                x.push_back(x0 + px);
                y.push_back(y0 + py);
                z.push_back(g + (top - g) * frac);
            }
    }
    return RansGrid::FromNodes(nx, ny, nz, x, y, z);
}

static double Total(const std::vector<double> &v)
{
    double s = 0.0;
    for(double a : v) s += a;
    return s;
}

BOOST_AUTO_TEST_SUITE(rans_grid)

BOOST_AUTO_TEST_CASE( flat_box )
{
    // 4 x 3 x 5 cells of 2 x 3 m, 100 m tall with stretched layers, at UTM-sized
    // coordinates that should end up local
    RansGrid g = TerrainGrid(4, 3, 5, 8.0, 9.0, 1100.0, 1.3,
                             [](double, double) { return 1000.0; },
                             500000.0, 4800000.0);
    BOOST_CHECK_CLOSE(Total(g.volume), 8.0 * 9.0 * 100.0, 1e-10);
    BOOST_CHECK_EQUAL(g.x[0], 0.0);
    BOOST_CHECK_EQUAL(g.xOrigin, 500000.0);
    BOOST_CHECK_EQUAL(g.yOrigin, 4800000.0);
    for(size_t f = 0; f < g.kSz.size(); f++)
    {
        BOOST_CHECK_CLOSE(g.kSz[f], 2.0 * 3.0, 1e-10);
        BOOST_CHECK_SMALL(g.kSx[f], 1e-12);
        BOOST_CHECK_SMALL(g.kSy[f], 1e-12);
    }
    // i faces have area dy * dz of their layer and point along +x
    for(int k = 0; k < g.nz; k++)
    {
        double dz = g.z[g.node(0, 0, k + 1)] - g.z[g.node(0, 0, k)];
        int f = g.iFace(2, 1, k);
        BOOST_CHECK_CLOSE(g.iSx[f], 3.0 * dz, 1e-10);
        int fj = g.jFace(2, 1, k);
        BOOST_CHECK_CLOSE(g.jSy[fj], 2.0 * dz, 1e-10);
        BOOST_CHECK_CLOSE(g.volume[g.cell(1, 1, k)], 2.0 * 3.0 * dz, 1e-10);
    }
}

BOOST_AUTO_TEST_CASE( closed_surface )
{
    // geometric conservation: each cell's outward area vectors sum to zero,
    // even on twisted, non-planar faces
    RansGrid g = TerrainGrid(8, 7, 6, 800.0, 700.0, 2500.0, 1.3,
                             [](double x, double y)
                             { return 1200.0 + 300.0 * std::sin(x / 90.0) * std::cos(y / 70.0) + 0.4 * x; });
    double worst = 0.0, area = 0.0;
    for(int k = 0; k < g.nz; k++)
        for(int j = 0; j < g.ny; j++)
            for(int i = 0; i < g.nx; i++)
            {
                int a = g.iFace(i, j, k), b = g.iFace(i + 1, j, k);
                int c = g.jFace(i, j, k), d = g.jFace(i, j + 1, k);
                int e = g.kFace(i, j, k), h = g.kFace(i, j, k + 1);
                double sx = g.iSx[b] - g.iSx[a] + g.jSx[d] - g.jSx[c] + g.kSx[h] - g.kSx[e];
                double sy = g.iSy[b] - g.iSy[a] + g.jSy[d] - g.jSy[c] + g.kSy[h] - g.kSy[e];
                double sz = g.iSz[b] - g.iSz[a] + g.jSz[d] - g.jSz[c] + g.kSz[h] - g.kSz[e];
                worst = std::max(worst, std::sqrt(sx * sx + sy * sy + sz * sz));
                area = std::max(area, std::fabs(g.kSz[e]));
                BOOST_CHECK_GT(g.volume[g.cell(i, j, k)], 0.0);
            }
    BOOST_CHECK_SMALL(worst / area, 1e-12);
}

BOOST_AUTO_TEST_CASE( sine_hill )
{
    // hill A sin(pi x / L) sin(pi y / L) under a flat top at H: the exact
    // volume is L^2 H - A (2L/pi)^2.  The grid's piecewise-bilinear ground is
    // second order, so the error should drop about 4x per refinement.
    const double L = 1000.0, H = 800.0, A = 200.0, pi = 3.14159265358979323846;
    auto hill = [&](double x, double y) { return A * std::sin(pi * x / L) * std::sin(pi * y / L); };
    double exact = L * L * H - A * (2 * L / pi) * (2 * L / pi);
    double e1 = std::fabs(Total(TerrainGrid(16, 16, 4, L, L, H, 1.2, hill).volume) - exact) / exact;
    double e2 = std::fabs(Total(TerrainGrid(32, 32, 4, L, L, H, 1.2, hill).volume) - exact) / exact;
    BOOST_CHECK_LT(e2, 1e-3);
    BOOST_CHECK_GT(e1 / e2, 3.5);
    BOOST_CHECK_LT(e1 / e2, 4.5);
}

BOOST_AUTO_TEST_CASE( wall_distance )
{
    RansGrid g = TerrainGrid(2, 2, 4, 2.0, 2.0, 1040.0, 1.0,
                             [](double, double) { return 1000.0; });
    // uniform 10 m layers: centroids at 5, 15, 25, 35 m above ground
    for(int k = 0; k < g.nz; k++)
        BOOST_CHECK_CLOSE(g.wallDistance[g.cell(1, 0, k)], 5.0 + 10.0 * k, 1e-10);
}

BOOST_AUTO_TEST_CASE( from_mesh )
{
    // the mass-solver mesh is (row, col, layer) with x along col and y along row
    Mesh m;
    m.nrows = 3; m.ncols = 4; m.nlayers = 2;
    m.XORD.allocate(3, 4, 2); m.YORD.allocate(3, 4, 2); m.ZORD.allocate(3, 4, 2);
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 4; c++)
            for(int l = 0; l < 2; l++)
            {
                m.XORD(r, c, l) = 100.0 + 10.0 * c;
                m.YORD(r, c, l) = 200.0 + 20.0 * r;
                m.ZORD(r, c, l) = 50.0 + 5.0 * l;
            }
    RansGrid g = RansGrid::FromMesh(m);
    BOOST_CHECK_EQUAL(g.nx, 3);
    BOOST_CHECK_EQUAL(g.ny, 2);
    BOOST_CHECK_EQUAL(g.nz, 1);
    BOOST_CHECK_EQUAL(g.x[g.node(3, 0, 0)], 30.0);
    BOOST_CHECK_EQUAL(g.y[g.node(0, 2, 0)], 40.0);
    BOOST_CHECK_CLOSE(Total(g.volume), 30.0 * 40.0 * 5.0, 1e-10);
}

BOOST_AUTO_TEST_CASE( bad_input )
{
    std::vector<double> v(8, 0.0);
    BOOST_CHECK_THROW(RansGrid::FromNodes(0, 1, 1, v, v, v), std::invalid_argument);
    BOOST_CHECK_THROW(RansGrid::FromNodes(1, 1, 2, v, v, v), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
