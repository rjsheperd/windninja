/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Test the CPU RANS solver (laminar)
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
#include "rans_solver.h"

#include <cmath>
#include <vector>

#include <boost/test/unit_test.hpp>

/*
** Terrain-following nodes over [0, lx] x [0, ly]: ground(x, y) to a flat top,
** layers spaced geometrically with ratio growth.
*/
template <typename Ground>
static RansGrid TerrainGrid(int nx, int ny, int nz, double lx, double ly,
                            double top, double growth, Ground ground)
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
                double gz = ground(px, py);
                x.push_back(px);
                y.push_back(py);
                z.push_back(gz + (top - gz) * frac);
            }
    }
    return RansGrid::FromNodes(nx, ny, nz, x, y, z);
}

static RansSolver::Boundary Dirichlet(const RansSolver::StateFunction &f)
{
    RansSolver::Boundary b = {RansSolver::Boundary::dirichlet, 0, 0, 0, 0, f};
    return b;
}

BOOST_AUTO_TEST_SUITE(rans_solver)

BOOST_AUTO_TEST_CASE( freestream )
{
    // a uniform flow is an exact steady solution on any grid, so its residual
    // must vanish on a curved, stretched terrain grid (metric consistency)
    RansGrid g = TerrainGrid(6, 5, 4, 600.0, 500.0, 900.0, 1.3,
                             [](double x, double y)
                             { return 100.0 * std::sin(x / 120.0) * std::cos(y / 90.0) + 0.2 * x; });
    RansSolver s(g, 1.5e-5, 1.0);
    RansSolver::StateFunction uniform = [](double, double, double,
                                           double &p, double &u, double &v, double &w)
    { p = 0.3; u = 4.0; v = -1.5; w = 0.7; };
    for(int side = 0; side < 6; side++)
        s.setBoundary((RansSolver::Side)side, Dirichlet(uniform));
    s.setUniform(0.3, 4.0, -1.5, 0.7);
    RansSolver::Residuals r = s.residuals();
    BOOST_CHECK_SMALL(r.p, 1e-10);
    BOOST_CHECK_SMALL(r.u, 1e-10);
    BOOST_CHECK_SMALL(r.v, 1e-10);
    BOOST_CHECK_SMALL(r.w, 1e-10);
}

/*
** Manufactured solution in the x-z plane (independent of y, v = 0).  The
** source is the steady AC system applied to it, with 4th-order central
** differences standing in for analytic derivatives.
*/
static const double PI = 3.14159265358979323846;
static const double MMS_NU = 0.05;
static const double MMS_BETA = 1.0;

static void MmsExact(double x, double, double z, double &p, double &u, double &v, double &w)
{
    u = 1.0 + 0.5 * std::sin(PI * x) * std::cos(PI * z);
    v = 0.0;
    w = 0.5 * std::cos(PI * x) * std::sin(PI * z);
    p = 0.25 * std::cos(PI * x) * std::cos(PI * z);
}

template <typename F>
static double D1(F f, double x, double h)   // 4th-order first derivative
{
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h);
}
template <typename F>
static double D2(F f, double x, double h)   // 4th-order second derivative
{
    return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12 * h * h);
}

static void MmsSource(double x, double y, double z, double &sp, double &su, double &sv, double &sw)
{
    const double h = 1e-3;
    auto field = [&](int which, double xx, double zz)
    {
        double p, u, v, w;
        MmsExact(xx, y, zz, p, u, v, w);
        switch(which)
        {
            case 0: return p;
            case 1: return u;
            case 2: return w;
            case 3: return u * u;
            case 4: return u * w;
            default: return w * w;
        }
    };
    auto dx = [&](int q) { return D1([&](double t) { return field(q, t, z); }, x, h); };
    auto dz = [&](int q) { return D1([&](double t) { return field(q, x, t); }, z, h); };
    auto lap = [&](int q) { return D2([&](double t) { return field(q, t, z); }, x, h) +
                                   D2([&](double t) { return field(q, x, t); }, z, h); };
    sp = MMS_BETA * (dx(1) + dz(2));
    su = dx(3) + dz(4) + dx(0) - MMS_NU * lap(1);
    sv = 0.0;
    sw = dx(4) + dz(5) + dz(0) - MMS_NU * lap(2);
}

struct MmsError { double u, w, p; int iterations; };

static MmsError SolveMms(int n, double stretch = 2.0, double hill = 0.08)
{
    // curved ground and stretched layers so the metrics are exercised.  The
    // top layer is stretch times the bottom one at every resolution, so the
    // per-layer growth tends to 1 under refinement (a valid refinement study).
    double growth = std::pow(stretch, 1.0 / (n - 1));
    RansGrid g = TerrainGrid(n, 1, n, 1.0, 0.1, 1.0, growth,
                             [hill](double x, double) { return hill * std::sin(2 * PI * x); });
    RansSolver s(g, MMS_NU, MMS_BETA);
    for(int side = 0; side < 6; side++)
        s.setBoundary((RansSolver::Side)side, Dirichlet(MmsExact));
    s.setSource(MmsSource);
    s.setUniform(0.0, 1.0, 0.0, 0.0);

    RansSolver::Residuals r0 = s.iterate(1, 0.8);
    int it = 1;
    RansSolver::Residuals r = r0;
    while(it < 200000 && std::max(r.u / r0.u, r.w / r0.w) > 1e-9)
    {
        r = s.iterate(500, 0.8);
        it += 500;
    }
    MmsError e = {0.0, 0.0, 0.0, it};
    for(int c = 0; c < g.nCells(); c++)
    {
        double p, u, v, w;
        MmsExact(g.cx[c], g.cy[c], g.cz[c], p, u, v, w);
        e.u += (s.u[c] - u) * (s.u[c] - u);
        e.w += (s.w[c] - w) * (s.w[c] - w);
        e.p += (s.p[c] - p) * (s.p[c] - p);
    }
    e.u = std::sqrt(e.u / g.nCells());
    e.w = std::sqrt(e.w / g.nCells());
    e.p = std::sqrt(e.p / g.nCells());
    return e;
}

BOOST_AUTO_TEST_CASE( mms_laminar )
{
    MmsError e[3];
    int n[3] = {8, 16, 32};
    for(int m = 0; m < 3; m++)
    {
        e[m] = SolveMms(n[m]);
        BOOST_TEST_MESSAGE("n=" << n[m] << " iterations=" << e[m].iterations << " err u=" << e[m].u
                           << " w=" << e[m].w << " p=" << e[m].p);
    }
    for(int m = 1; m < 3; m++)
    {
        double ou = std::log2(e[m - 1].u / e[m].u);
        double ow = std::log2(e[m - 1].w / e[m].w);
        double op = std::log2(e[m - 1].p / e[m].p);
        BOOST_TEST_MESSAGE("order " << n[m - 1] << "->" << n[m] << ": u=" << ou << " w=" << ow << " p=" << op);
        if(m == 2)
        {
            BOOST_CHECK_GE(ou, 1.8);
            BOOST_CHECK_GE(ow, 1.8);
            BOOST_CHECK_GE(op, 1.5);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
