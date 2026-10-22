/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Tests for the mass solver's FEM assembly and velocity recovery
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
#include "mass_terrain.h"
#include "mass_wind_init.h"
#include "domainAverageInitialization.h"
#include "wn_3dScalarField.h"
#include "WindNinjaInputs.h"
#include "mesh.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <boost/test/unit_test.hpp>

/*
** A ncols x nrows x nlayers grid with unit spacing, the ground at
** ground(x, y) and each layer one unit above the last.
*/
template <typename Ground>
static MassGrid UnitGrid(int ncols, int nrows, int nlayers, Ground ground)
{
    MassGrid g;
    g.ncols = ncols;
    g.nrows = nrows;
    g.nlayers = nlayers;
    for(int k = 0; k < nlayers; k++)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < ncols; j++)
            {
                g.x.push_back(j);
                g.y.push_back(i);
                g.z.push_back(ground(j, i) + k);
            }
    return g;
}

static MassGrid FlatGrid(int ncols, int nrows, int nlayers)
{
    return UnitGrid(ncols, nrows, nlayers, [](double, double) { return 0.0; });
}

static MassGrid SlopedGrid(int ncols, int nrows, int nlayers)
{
    return UnitGrid(ncols, nrows, nlayers,
                    [](double x, double y) { return 0.3 * x + 0.2 * std::sin(y); });
}

static int Node(const MassGrid &g, int col, int row, int layer)
{
    return col + g.ncols * (row + g.nrows * layer);
}

static double Entry(const MassSystem &s, int row, int col)
{
    for(int l = s.row_ptr[row]; l < s.row_ptr[row + 1]; l++)
        if(s.col_ind[l] == col)
            return s.SK[l];
    BOOST_FAIL("no entry in the pattern");
    return 0.0;
}

BOOST_AUTO_TEST_SUITE(mass_fem)

/*
** Unit cubes: per element DETJ = 1/8 and dN/dx = +-1/4, so an element adds
** (1/16)(Rx sx sx' + Ry sy sy' + Rz sz sz') to each pair of its nodes.
** alphaH = 1, alphaV = 2: Rx = Ry = 1/2, Rz = 1/8.
*/
BOOST_AUTO_TEST_CASE(unit_cube_stiffness)
{
    MassGrid g = FlatGrid(3, 3, 3);
    std::vector<double> zero(g.numNodes(), 0.0), alphaV(g.numNodes(), 2.0);
    MassSystem s = AssembleMass(g, zero.data(), zero.data(), zero.data(), alphaV.data(), 1.0);

    int ground = Node(g, 1, 1, 0), middle = Node(g, 1, 1, 1);
    // 4 and 8 elements of (1/16)(1/2 + 1/2 + 1/8)
    BOOST_CHECK_CLOSE(Entry(s, ground, ground), 0.28125, 1e-12);
    BOOST_CHECK_CLOSE(Entry(s, middle, middle), 0.5625, 1e-12);
    // vertical neighbours share 4 elements: (1/16)(1/2 + 1/2 - 1/8)
    BOOST_CHECK_CLOSE(Entry(s, ground, middle), 0.21875, 1e-12);
}

/*
** RHS = integral of N div(u0): u0 = x has div 1, and each element gives
** each of its nodes DETJ = 1/8.
*/
BOOST_AUTO_TEST_CASE(divergence_rhs)
{
    MassGrid g = FlatGrid(3, 3, 3);
    std::vector<double> u0(g.x), zero(g.numNodes(), 0.0), alphaV(g.numNodes(), 1.0);
    MassSystem s = AssembleMass(g, u0.data(), zero.data(), zero.data(), alphaV.data(), 1.0);
    BOOST_CHECK_CLOSE(s.RHS[Node(g, 1, 1, 0)], 0.5, 1e-12);
    BOOST_CHECK_CLOSE(s.RHS[Node(g, 1, 1, 1)], 1.0, 1e-12);
}

/* A uniform wind over sloped terrain is divergence free: no source. */
BOOST_AUTO_TEST_CASE(uniform_flow_no_source)
{
    MassGrid g = SlopedGrid(6, 5, 4);
    std::vector<double> u0(g.numNodes(), 4.0), v0(g.numNodes(), -3.0), w0(g.numNodes(), 0.5);
    std::vector<double> alphaV(g.numNodes(), 1.0);
    MassSystem s = AssembleMass(g, u0.data(), v0.data(), w0.data(), alphaV.data(), 1.0);
    for(double r : s.RHS)
        BOOST_CHECK_SMALL(r, 1e-12);
}

/*
** Upper-triangle CSR of the 27-point stencil: sorted columns >= row, 14
** entries for an interior node; boundary rows (sides, top) are the
** identity with zero RHS, and no row keeps a boundary column off the
** diagonal.
*/
BOOST_AUTO_TEST_CASE(pattern_and_boundaries)
{
    MassGrid g = SlopedGrid(5, 4, 4);
    std::vector<double> u0(g.x), v0(g.numNodes(), 1.0), w0(g.numNodes(), 0.0);
    std::vector<double> alphaV(g.numNodes(), 1.0);
    MassSystem s = AssembleMass(g, u0.data(), v0.data(), w0.data(), alphaV.data(), 1.0);

    BOOST_REQUIRE_EQUAL(s.row_ptr.size(), (size_t)g.numNodes() + 1);
    BOOST_CHECK_EQUAL(s.row_ptr[Node(g, 2, 1, 1) + 1] - s.row_ptr[Node(g, 2, 1, 1)], 14);
    for(int layer = 0; layer < g.nlayers; layer++)
        for(int row = 0; row < g.nrows; row++)
            for(int col = 0; col < g.ncols; col++)
            {
                int n = Node(g, col, row, layer);
                bool boundary = col == 0 || col == g.ncols - 1 || row == 0 ||
                                row == g.nrows - 1 || layer == g.nlayers - 1;
                BOOST_CHECK_EQUAL(s.col_ind[s.row_ptr[n]], n);
                for(int l = s.row_ptr[n] + 1; l < s.row_ptr[n + 1]; l++)
                    BOOST_CHECK_GT(s.col_ind[l], s.col_ind[l - 1]);
                if(boundary)
                {
                    BOOST_CHECK_EQUAL(s.SK[s.row_ptr[n]], 1.0);
                    for(int l = s.row_ptr[n] + 1; l < s.row_ptr[n + 1]; l++)
                        BOOST_CHECK_EQUAL(s.SK[l], 0.0);
                    BOOST_CHECK_EQUAL(s.RHS[n], 0.0);
                }
                else
                {
                    BOOST_CHECK_GT(s.SK[s.row_ptr[n]], 0.0);
                }
            }
    // interior rows: zero in every boundary column
    for(int n = 0; n < g.numNodes(); n++)
        for(int l = s.row_ptr[n] + 1; l < s.row_ptr[n + 1]; l++)
        {
            int c = s.col_ind[l];
            int col = c % g.ncols, row = (c / g.ncols) % g.nrows, layer = c / (g.ncols * g.nrows);
            if(col == 0 || col == g.ncols - 1 || row == 0 || row == g.nrows - 1 ||
               layer == g.nlayers - 1)
                BOOST_CHECK_EQUAL(s.SK[l], 0.0);
        }
}

/* Phi = 0 leaves the initial wind unchanged. */
BOOST_AUTO_TEST_CASE(zero_phi_keeps_initial_wind)
{
    MassGrid g = SlopedGrid(5, 4, 3);
    int n = g.numNodes();
    std::vector<double> phi(n, 0.0), u0(n), v0(n), w0(n), alphaV(n, 1.5), u(n), v(n), w(n);
    for(int i = 0; i < n; i++)
    {
        u0[i] = 1.0 + 0.1 * i;
        v0[i] = -2.0 + 0.05 * i;
        w0[i] = 0.01 * i;
    }
    RecoverVelocity(g, phi.data(), u0.data(), v0.data(), w0.data(), alphaV.data(), 2.0,
                    u.data(), v.data(), w.data());
    for(int i = 0; i < n; i++)
    {
        BOOST_CHECK_EQUAL(u[i], u0[i]);
        BOOST_CHECK_EQUAL(v[i], v0[i]);
        BOOST_CHECK_EQUAL(w[i], w0[i]);
    }
}

/*
** Phi linear in x, y, z: trilinear elements represent it exactly, so every
** node gets u = u0 + a / (2 alphaH^2), v = v0 + b / (2 alphaH^2),
** w = w0 + c / (2 alphaV^2), on sloped terrain too.
*/
BOOST_AUTO_TEST_CASE(linear_phi_gradient)
{
    MassGrid g = SlopedGrid(6, 5, 4);
    int n = g.numNodes();
    const double a = 0.7, b = -1.3, c = 2.1, alphaH = 2.0, alphaVal = 0.5;
    std::vector<double> phi(n), zero(n, 0.0), alphaV(n, alphaVal), u(n), v(n), w(n);
    for(int i = 0; i < n; i++)
        phi[i] = a * g.x[i] + b * g.y[i] + c * g.z[i];
    RecoverVelocity(g, phi.data(), zero.data(), zero.data(), zero.data(), alphaV.data(), alphaH,
                    u.data(), v.data(), w.data());
    for(int i = 0; i < n; i++)
    {
        BOOST_CHECK_CLOSE(u[i], a / (2 * alphaH * alphaH), 1e-9);
        BOOST_CHECK_CLOSE(v[i], b / (2 * alphaH * alphaH), 1e-9);
        BOOST_CHECK_CLOSE(w[i], c / (2 * alphaVal * alphaVal), 1e-9);
    }
}

/* A folded element (top below bottom) is rejected, as before. */
BOOST_AUTO_TEST_CASE(inverted_element_throws)
{
    MassGrid g = FlatGrid(2, 2, 2);
    for(int i = 4; i < 8; i++)
        g.z[i] = -1.0;
    std::vector<double> zero(g.numNodes(), 0.0), alphaV(g.numNodes(), 1.0);
    BOOST_CHECK_THROW(AssembleMass(g, zero.data(), zero.data(), zero.data(), alphaV.data(), 1.0),
                      std::runtime_error);
}

/* A hill DEM at 30 m, as an Elevation (native) and a TerrainDem (core-free). */
static void HillDems(Elevation &native, TerrainDem &plain)
{
    plain.nCols = 40;
    plain.nRows = 30;
    plain.cellSize = 30.0;
    plain.xll = 500000.0;
    plain.yll = 4800000.0;
    plain.z.clear();
    for(int r = 0; r < plain.nRows; r++)
        for(int c = 0; c < plain.nCols; c++)
        {
            double z = 1000.37 + 150.0 * std::exp(-((c - 20) * (c - 20) + (r - 12) * (r - 12)) / 40.0) +
                       0.013 * c * r;
            plain.z.push_back(z);
            native(r, c) = z;   // row 0 south in both
        }
}

/*
** The core-free mesh is WindNinja's: Mesh::buildStandardMesh on the same
** DEM (at the mesh resolution, so no resampling) gives the same nodes, bit
** for bit: grass, and trees whose canopy plus the output height, or plus
** the input height, sets the domain height.
*/
BOOST_AUTO_TEST_CASE(terrain_grid_matches_mesh)
{
    struct Case { double canopy, inputHeight, outputHeight; };
    for(Case k : {Case{0.0, 6.096, 10.0}, Case{15.4, 6.096, 10.0}, Case{15.4, 20.0, 6.096}})
    {
        const double canopy = k.canopy;
        WindNinjaInputs input;
        input.dem = Elevation(40, 30, 500000.0, 4800000.0, 30.0, -9999.0, Elevation::meters);
        TerrainDem plain;
        HillDems(input.dem, plain);
        input.surface.Rough_h.set_headerData(input.dem);
        for(int r = 0; r < 30; r++)
            for(int c = 0; c < 40; c++)
                input.surface.Rough_h(r, c) = canopy;
        input.inputWindHeight = k.inputHeight;
        input.outputWindHeight = k.outputHeight;

        Mesh mesh;
        mesh.set_numVertLayers(20);
        mesh.set_meshResolution(30.0, lengthUnits::meters);
        mesh.buildStandardMesh(input);
        MassGrid expected = MassGrid::FromMesh(mesh);

        MassMeshOptions options;
        options.inputHeight = k.inputHeight;
        options.outputHeight = k.outputHeight;
        options.canopyHeight = canopy;
        MassGrid ours = MassTerrainGrid(plain, options);

        BOOST_REQUIRE_EQUAL(ours.ncols, expected.ncols);
        BOOST_REQUIRE_EQUAL(ours.nrows, expected.nrows);
        BOOST_REQUIRE_EQUAL(ours.nlayers, expected.nlayers);
        int mismatches = 0;
        for(int n = 0; n < ours.numNodes(); n++)
            mismatches += ours.x[n] != expected.x[n] || ours.y[n] != expected.y[n] ||
                          ours.z[n] != expected.z[n];
        BOOST_CHECK_EQUAL(mismatches, 0);
        BOOST_CHECK_EQUAL(MassDomainHeight(plain, 30.0, options), mesh.domainHeight);
    }
}

/* Mesh::compute_cellsize for the coarse, medium and fine choices. */
BOOST_AUTO_TEST_CASE(mesh_resolution_matches_mesh)
{
    for(Mesh::eMeshChoice choice : {Mesh::coarse, Mesh::medium, Mesh::fine})
    {
        Elevation dem(173, 211, 500000.0, 4800000.0, 27.3, -9999.0, Elevation::meters);
        Mesh mesh;
        mesh.set_meshResChoice(choice);
        mesh.compute_cellsize(dem);
        BOOST_CHECK_EQUAL(MassMeshResolution(173, 211, 27.3, mesh.targetNumHorizCells),
                          mesh.meshResolution);
    }
}

/*
** The core-free initial wind is WindNinja's: domainAverageInitialization
** (no diurnal winds) on the same mesh gives the same u0, v0, w0, bit for
** bit, for grass and trees, a latitude and none, an oblique direction, and
** a DEM grid turned from north.
*/
BOOST_AUTO_TEST_CASE(initial_wind_matches_domain_average)
{
    struct Veg { double z0, h, d; };
    struct Case { Veg veg; double direction, latitude; };
    for(Case k : {Case{{0.01, 0.0, 0.0}, 270.0, 43.5}, Case{{1.0, 15.4, 12.0}, 37.5, -10000.0},
                  Case{{0.43, 2.3, 1.8}, 0.0, -33.0}})
    {
        WindNinjaInputs input;
        input.dem = Elevation(40, 30, 500000.0, 4800000.0, 30.0, -9999.0, Elevation::meters);
        TerrainDem plain;
        HillDems(input.dem, plain);
        input.dem.setAngleFromNorth(1.3);
        AsciiGrid<double> *grids[] = {&input.surface.Roughness, &input.surface.Rough_h,
                                      &input.surface.Rough_d};
        double values[] = {k.veg.z0, k.veg.h, k.veg.d};
        for(int g = 0; g < 3; g++)
        {
            grids[g]->set_headerData(input.dem);
            for(int r = 0; r < 30; r++)
                for(int c = 0; c < 40; c++)
                    (*grids[g])(r, c) = values[g];
        }
        input.inputSpeed = 4.4704;
        input.inputDirection_geog = k.direction;
        input.inputWindHeight = 6.096;
        input.outputWindHeight = 6.096;
        input.latitude = k.latitude;
        input.diurnalWinds = false;
        input.cloudCover = 0.0;

        Mesh mesh;
        mesh.set_numVertLayers(20);
        mesh.set_meshResolution(30.0, lengthUnits::meters);
        mesh.buildStandardMesh(input);
        wn_3dScalarField u0, v0, w0;
        u0.allocate(&mesh);   // as ninja does before initializing
        v0.allocate(&mesh);
        w0.allocate(&mesh);
        AsciiGrid<double> cloud;
        domainAverageInitialization init;
        init.initializeFields(input, mesh, u0, v0, w0, cloud);

        DomainAverageWind wind;
        wind.speed = 4.4704;
        wind.direction = k.direction;
        wind.inputHeight = 6.096;
        wind.roughness = k.veg.z0;
        wind.roughH = k.veg.h;
        wind.roughD = k.veg.d;
        wind.latitude = k.latitude;
        wind.angleFromNorth = 1.3;
        std::vector<double> u, v, w;
        InitialWind(MassGrid::FromMesh(mesh), wind, u, v, w);

        int mismatches = 0;
        double maxSpeed = 0.0;
        for(int n = 0; n < mesh.NUMNP; n++)
        {
            mismatches += u[n] != u0(n) || v[n] != v0(n) || w[n] != w0(n);
            maxSpeed = std::max(maxSpeed, std::hypot(u[n], v[n]));
        }
        BOOST_CHECK_EQUAL(mismatches, 0);
        BOOST_CHECK_GT(maxSpeed, 4.4704);   // the profile grows above the input height
    }
}

BOOST_AUTO_TEST_SUITE_END()
