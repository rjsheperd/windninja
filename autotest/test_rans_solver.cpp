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
#include "rans_abl.h"
#include "rans_wind.h"
#include "rans_multigrid.h"

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

/*
** Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411, Re = 100: Table I
** (u on the vertical centreline) and Table II (v on the horizontal
** centreline).  Positions are the 129-point grid nodes (k/128).  Values match
** two independent transcriptions (gist.github.com/ivan-pi and
** github.com/sujaldave/2DLidDrivenCavityBenchmark); not checked against the
** printed paper.
*/
static const double GHIA_U[17][2] = {
    {1.0000000, 1.00000}, {0.9765625, 0.84123}, {0.9687500, 0.78871}, {0.9609375, 0.73722},
    {0.9531250, 0.68717}, {0.8515625, 0.23151}, {0.7343750, 0.00332}, {0.6171875, -0.13641},
    {0.5000000, -0.20581}, {0.4531250, -0.21090}, {0.2812500, -0.15662}, {0.1718750, -0.10150},
    {0.1015625, -0.06434}, {0.0703125, -0.04775}, {0.0625000, -0.04192}, {0.0546875, -0.03717},
    {0.0000000, 0.00000}};
static const double GHIA_V[17][2] = {
    {1.0000000, 0.00000}, {0.9687500, -0.05906}, {0.9609375, -0.07391}, {0.9531250, -0.08864},
    {0.9453125, -0.10313}, {0.9062500, -0.16914}, {0.8593750, -0.22445}, {0.8046875, -0.24533},
    {0.5000000, 0.05454}, {0.2343750, 0.17527}, {0.2265625, 0.17507}, {0.1562500, 0.16077},
    {0.0937500, 0.12317}, {0.0781250, 0.10890}, {0.0703125, 0.10091}, {0.0625000, 0.09233},
    {0.0000000, 0.00000}};

/*
** Lid-driven cavity in the x-z plane: lid at the top moving at +1, no-slip
** walls elsewhere, slip sides in y (one cell, so the flow is 2-D).
*/
static RansSolver Cavity(const RansGrid &g, double nu)
{
    RansSolver s(g, nu, 1.0);
    RansSolver::Boundary wall = {RansSolver::Boundary::wall, 0, 0, 0, 0, RansSolver::StateFunction()};
    RansSolver::Boundary lid = {RansSolver::Boundary::wall, 0, 1, 0, 0, RansSolver::StateFunction()};
    RansSolver::Boundary slip = {RansSolver::Boundary::slip, 0, 0, 0, 0, RansSolver::StateFunction()};
    s.setBoundary(RansSolver::xMin, wall);
    s.setBoundary(RansSolver::xMax, wall);
    s.setBoundary(RansSolver::zMin, wall);
    s.setBoundary(RansSolver::zMax, lid);
    s.setBoundary(RansSolver::yMin, slip);
    s.setBoundary(RansSolver::yMax, slip);
    s.setUniform(0, 0, 0, 0);
    return s;
}

/*
** Bilinear interpolation of a cell field at (x, z) on a uniform n x 1 x n
** cavity grid.  The centroids are extended with the wall values at their
** true positions (x, z = 0 and 1): top for the lid row, 0 on the other walls.
*/
static double CavitySample(const std::vector<double> &q, int n, double x, double z, double top)
{
    // extended coordinates: 0, (k + 0.5)/n for k < n, 1
    auto pos = [n](int k) { return k == 0 ? 0.0 : (k == n + 1 ? 1.0 : (k - 0.5) / n); };
    auto val = [&](int i, int k)
    {
        if(k == n + 1) return top;
        if(i == 0 || i == n + 1 || k == 0) return 0.0;
        return q[(i - 1) + n * (k - 1)];   // ny = 1
    };
    int i0 = 0, k0 = 0;
    while(i0 < n && pos(i0 + 1) < x) i0++;
    while(k0 < n && pos(k0 + 1) < z) k0++;
    double a = (x - pos(i0)) / (pos(i0 + 1) - pos(i0));
    double b = (z - pos(k0)) / (pos(k0 + 1) - pos(k0));
    return (1 - a) * (1 - b) * val(i0, k0) + a * (1 - b) * val(i0 + 1, k0) +
           (1 - a) * b * val(i0, k0 + 1) + a * b * val(i0 + 1, k0 + 1);
}

BOOST_AUTO_TEST_CASE( lid_cavity_re100 )
{
    // Re = U L / nu = 100 on a uniform 32 x 32 grid
    const int n = 32;
    RansGrid g = TerrainGrid(n, 1, n, 1.0, 1.0 / n, 1.0, 1.0, [](double, double) { return 0.0; });
    RansSolver s = Cavity(g, 0.01);

    // residuals relative to each equation's peak (w and p start at exactly 0)
    RansSolver::Residuals peak = {0, 0, 0, 0}, r;
    int it = 0, fiveOrders = -1;
    double worst = 1.0;
    while(it < 25000 && worst > 1e-6)
    {
        r = s.iterate(100, 0.8);
        it += 100;
        peak.u = std::max(peak.u, r.u); peak.w = std::max(peak.w, r.w); peak.p = std::max(peak.p, r.p);
        worst = std::max(std::max(r.u / peak.u, r.w / peak.w), r.p / peak.p);
        if(fiveOrders < 0 && worst <= 1e-5) fiveOrders = it;
    }
    BOOST_TEST_MESSAGE("five orders at " << fiveOrders << ", stopped at " << it << " (" << worst << ")");
    // measured 12100 on macOS arm64 (2026-10-21)
    BOOST_CHECK(fiveOrders > 0 && fiveOrders <= 15000);

    // centreline profiles within 2% of the lid speed (measured 0.80% and 0.33%)
    double eu = 0.0, ev = 0.0;
    for(const auto &row : GHIA_U)
        eu = std::max(eu, std::fabs(CavitySample(s.u, n, 0.5, row[0], 1.0) - row[1]));
    for(const auto &row : GHIA_V)
        ev = std::max(ev, std::fabs(CavitySample(s.w, n, row[0], 0.5, 0.0) - row[1]));
    BOOST_TEST_MESSAGE("max |u - ghia| = " << eu << ", max |v - ghia| = " << ev);
    BOOST_CHECK_LE(eu, 0.02);
    BOOST_CHECK_LE(ev, 0.02);
}

BOOST_AUTO_TEST_CASE( wind_direction_matches_ninjafoam )
{
    // NinjaFoam::ComputeDirection: wind from 270 blows toward +x, from 0
    // toward -y, from 225 toward (+x, +y) / sqrt(2)
    double dx, dy;
    WindDirectionVector(270.0, dx, dy);
    BOOST_CHECK_CLOSE(dx, 1.0, 1e-9); BOOST_CHECK_SMALL(dy, 1e-12);
    WindDirectionVector(0.0, dx, dy);
    BOOST_CHECK_SMALL(dx, 1e-12); BOOST_CHECK_CLOSE(dy, -1.0, 1e-9);
    WindDirectionVector(225.0, dx, dy);
    BOOST_CHECK_CLOSE(dx, std::sqrt(0.5), 1e-9); BOOST_CHECK_CLOSE(dy, std::sqrt(0.5), 1e-9);
}

BOOST_AUTO_TEST_CASE( wind_inlets_match_ninjafoam )
{
    // NinjaFoam::SetInlets (west = xMin, east = xMax, south = yMin, north = yMax)
    struct { double dir; bool west, east, south, north; } cases[] = {
        {270.0, true, false, false, false}, {90.0, false, true, false, false},
        {0.0, false, false, false, true}, {360.0, false, false, false, true},
        {180.0, false, false, true, false}, {45.0, false, true, false, true},
        {135.0, false, true, true, false}, {225.0, true, false, true, false},
        {315.0, true, false, false, true}};
    for(const auto &c : cases)
    {
        BOOST_CHECK_EQUAL(IsWindInlet(RansSolver::xMin, c.dir), c.west);
        BOOST_CHECK_EQUAL(IsWindInlet(RansSolver::xMax, c.dir), c.east);
        BOOST_CHECK_EQUAL(IsWindInlet(RansSolver::yMin, c.dir), c.south);
        BOOST_CHECK_EQUAL(IsWindInlet(RansSolver::yMax, c.dir), c.north);
        BOOST_CHECK(!IsWindInlet(RansSolver::zMin, c.dir));
        BOOST_CHECK(!IsWindInlet(RansSolver::zMax, c.dir));
    }
}

/*
** OpenFOAM-11 reference for flat_coarse_270 (scripts/make_momentum_golden.sh
** case: flat 6 km DEM, 10 mph at 20 ft, grass, wind 270 = 271.67 deg in grid
** coordinates): the mid-domain column's cell-centre heights above ground and
** |U|, from the converged case (time 351) via foamPostProcess
** -func writeCellCentres, on macOS arm64, 2026-10-21.
*/
static const double OF_FLAT_Z[11] = {30.853, 92.559, 185.298, 309.418, 434.416, 560.671,
                                     688.608, 818.702, 951.501, 1087.590, 1227.077};
static const double OF_FLAT_U[11] = {5.4502, 6.2204, 6.8101, 7.2369, 7.4701, 7.6390,
                                     7.7737, 7.8839, 7.9718, 8.0355, 8.0702};

BOOST_AUTO_TEST_CASE( flat_terrain_parity_openfoam )
{
    // layers whose centres match OpenFOAM's column; 59 m horizontal cells
    std::vector<double> zf(1, 0.0);
    for(int m = 0; m < 11; m++) zf.push_back(2 * OF_FLAT_Z[m] - zf.back());
    const int nx = 100, ny = 2, nz = 11;
    const double dxy = 59.0;
    std::vector<double> x, y, z;
    for(int kk = 0; kk <= nz; kk++)
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i <= nx; i++)
            { x.push_back(i * dxy); y.push_back(j * dxy); z.push_back(1000.0 + zf[kk]); }
    RansGrid g = RansGrid::FromNodes(nx, ny, nz, x, y, z);

    const double nu = 1.58946099535e-05, z0 = 0.01;
    double ustar = AblFrictionVelocity(4.4704, 6.095999804928006, z0);
    RansSolver s(g, nu, 25.0);
    ApplyWindBoundaries(s, ustar, z0, 271.67);
    InitializeLogProfile(s, ustar, z0, 271.67);
    RansSolver::Residuals peak = {0, 0, 0, 0}, r;
    int it = 0;
    double worst = 1.0;
    while(it < 20000 && worst > 1e-6)
    {
        r = s.iterate(500, 0.5);
        it += 500;
        peak.u = std::max(peak.u, r.u); peak.k = std::max(peak.k, r.k);
        worst = std::max(r.u / peak.u, r.k / peak.k);
    }
    BOOST_TEST_MESSAGE("converged to " << worst << " in " << it << " iterations");
    BOOST_CHECK_LE(worst, 1e-6);

    // provisional per-level bound until GATE 0 sets tolerances (measured:
    // +3.5% in the first cell, within 0.6% above 150 m)
    for(int kk = 0; kk < nz; kk++)
    {
        int c = g.cell(nx / 2, ny / 2, kk);
        double ratio = std::hypot(s.u[c], s.v[c]) / OF_FLAT_U[kk];
        BOOST_TEST_MESSAGE("z=" << OF_FLAT_Z[kk] << " U/U_openfoam=" << ratio);
        BOOST_CHECK_SMALL(ratio - 1.0, 0.05);
    }
}

BOOST_AUTO_TEST_CASE( multigrid_cavity )
{
    // FAS multigrid reaches the single-grid cavity solution in far fewer
    // fine-grid steps (measured: 314 cycles, ~0.99 s vs 14,500 iterations, 5.4 s)
    const int n = 32;
    RansGrid g = TerrainGrid(n, 1, n, 1.0, 1.0 / n, 1.0, 1.0, [](double, double) { return 0.0; });
    RansSolver s = Cavity(g, 0.01);
    RansMultigrid mg(s);
    BOOST_CHECK_EQUAL(mg.levels(), 3);
    RansSolver::Residuals peak = {0, 0, 0, 0}, r;
    int cycles = 0;
    double worst = 1.0;
    while(cycles < 2000 && worst > 1e-6)
    {
        r = mg.cycle(0.8);
        cycles++;
        peak.u = std::max(peak.u, r.u); peak.w = std::max(peak.w, r.w); peak.p = std::max(peak.p, r.p);
        worst = std::max(std::max(r.u / peak.u, r.w / peak.w), r.p / peak.p);
    }
    BOOST_TEST_MESSAGE("multigrid cavity: " << cycles << " cycles");
    BOOST_CHECK_LE(cycles, 600);
    double eu = 0.0, ev = 0.0;
    for(const auto &row : GHIA_U) eu = std::max(eu, std::fabs(CavitySample(s.u, n, 0.5, row[0], 1.0) - row[1]));
    for(const auto &row : GHIA_V) ev = std::max(ev, std::fabs(CavitySample(s.w, n, row[0], 0.5, 0.0) - row[1]));
    // the same discrete solution as single grid (0.0056039 / 0.0063357)
    BOOST_CHECK_CLOSE(eu, 0.0056039, 0.1);
    BOOST_CHECK_CLOSE(ev, 0.0063357, 0.1);
}

BOOST_AUTO_TEST_CASE( multigrid_matches_single_grid_turbulent )
{
    // turbulent flat ABL (OpenFOAM-parity grid): multigrid with frozen
    // coarse-level turbulence converges to the single-grid solution
    std::vector<double> zf(1, 0.0);
    for(int m = 0; m < 11; m++) zf.push_back(2 * OF_FLAT_Z[m] - zf.back());
    const int nx = 100, ny = 2, nz = 11;
    std::vector<double> x, y, z;
    for(int kk = 0; kk <= nz; kk++)
        for(int j = 0; j <= ny; j++)
            for(int i = 0; i <= nx; i++)
            { x.push_back(i * 59.0); y.push_back(j * 59.0); z.push_back(1000.0 + zf[kk]); }
    RansGrid g = RansGrid::FromNodes(nx, ny, nz, x, y, z);
    double ustar = AblFrictionVelocity(4.4704, 6.095999804928006, 0.01);
    std::vector<double> u[2];
    for(int mode = 0; mode < 2; mode++)
    {
        RansSolver s(g, 1.58946099535e-05, 25.0);
        ApplyWindBoundaries(s, ustar, 0.01, 271.67);
        InitializeLogProfile(s, ustar, 0.01, 271.67);
        RansMultigrid mg(s);
        RansSolver::Residuals peak = {0, 0, 0, 0}, r;
        double worst = 1.0;
        for(int step = 0; step < 5000 && worst > 1e-8; step++)
        {
            r = mode == 0 ? s.iterate(10, 0.5) : mg.cycle(0.5);
            peak.u = std::max(peak.u, r.u); peak.k = std::max(peak.k, r.k);
            worst = std::max(r.u / peak.u, r.k / peak.k);
        }
        BOOST_CHECK_LE(worst, 1e-8);
        u[mode] = s.u;
    }
    double diff = 0.0;
    for(size_t c = 0; c < u[0].size(); c++)
        diff = std::max(diff, std::fabs(u[1][c] - u[0][c]) / std::fabs(u[0][c]));
    BOOST_TEST_MESSAGE("max relative u difference, multigrid vs single grid: " << diff);
    BOOST_CHECK_SMALL(diff, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()
