/******************************************************************************
 *
 * $Id$ 
 *
 * Project:  WindNinja
 * Purpose:  Test ninja foam
 * Author:   Kyle Shannon <kyle at pobox.com>
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
#ifdef NINJAFOAM

#include "ninjafoam.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/test/unit_test.hpp>

struct NinjaFoamData
{
    NinjaFoamData()
    {
        GDALAllRegister();
        OGRRegisterAll();
    }
    ~NinjaFoamData()
    {
    }
    NinjaFoam f;
};

BOOST_FIXTURE_TEST_SUITE( ninjafoam, NinjaFoamData)

/*
** Characterisation tests for the NinjaFoam helpers that do not need an
** OpenFOAM installation.  They pin current behaviour ahead of moving the
** OpenFOAM code behind a solver backend interface.
*/

static std::vector<double> ComputeDirectionFor(NinjaFoam &f, double d)
{
    f.input.inputDirection_proj = d;
    f.direction.clear();
    f.ComputeDirection();
    return f.direction;
}

static std::vector<std::string> SetInletsFor(NinjaFoam &f, double d)
{
    f.input.inputDirection_proj = d;
    f.inlets.clear();
    f.SetInlets();
    return f.inlets;
}

BOOST_AUTO_TEST_CASE( compute_direction_axes )
{
    // wind *from* d, so the unit vector points the way the wind blows
    std::vector<double> v = ComputeDirectionFor(f, 270);
    BOOST_CHECK_CLOSE(v[0], 1.0, 1e-9);
    BOOST_CHECK_SMALL(v[1], 1e-12);
    v = ComputeDirectionFor(f, 0);
    BOOST_CHECK_SMALL(v[0], 1e-12);
    BOOST_CHECK_CLOSE(v[1], -1.0, 1e-9);
    v = ComputeDirectionFor(f, 90);
    BOOST_CHECK_CLOSE(v[0], -1.0, 1e-9);
    BOOST_CHECK_SMALL(v[1], 1e-12);
    v = ComputeDirectionFor(f, 180);
    BOOST_CHECK_SMALL(v[0], 1e-12);
    BOOST_CHECK_CLOSE(v[1], 1.0, 1e-9);
    BOOST_CHECK_EQUAL(v.size(), 3u);
    BOOST_CHECK_EQUAL(v[2], 0.0);
}

BOOST_AUTO_TEST_CASE( compute_direction_quadrants )
{
    const double r = std::sqrt(0.5);
    const double cases[4][3] = { { 225,  r,  r },
                                 { 315,  r, -r },
                                 {  45, -r, -r },
                                 { 135, -r,  r } };
    for(int i = 0; i < 4; i++)
    {
        std::vector<double> v = ComputeDirectionFor(f, cases[i][0]);
        BOOST_CHECK_CLOSE(v[0], cases[i][1], 1e-9);
        BOOST_CHECK_CLOSE(v[1], cases[i][2], 1e-9);
        BOOST_CHECK_CLOSE(v[0]*v[0] + v[1]*v[1], 1.0, 1e-9);
    }
}

BOOST_AUTO_TEST_CASE( set_inlets_axes )
{
    BOOST_CHECK(SetInletsFor(f, 0) == std::vector<std::string>{"north_face"});
    BOOST_CHECK(SetInletsFor(f, 360) == std::vector<std::string>{"north_face"});
    BOOST_CHECK(SetInletsFor(f, 90) == std::vector<std::string>{"east_face"});
    BOOST_CHECK(SetInletsFor(f, 180) == std::vector<std::string>{"south_face"});
    BOOST_CHECK(SetInletsFor(f, 270) == std::vector<std::string>{"west_face"});
}

BOOST_AUTO_TEST_CASE( set_inlets_quadrants )
{
    std::vector<std::string> ne = {"north_face", "east_face"};
    std::vector<std::string> es = {"east_face", "south_face"};
    std::vector<std::string> sw = {"south_face", "west_face"};
    std::vector<std::string> wn = {"west_face", "north_face"};
    BOOST_CHECK(SetInletsFor(f, 45) == ne);
    BOOST_CHECK(SetInletsFor(f, 135) == es);
    BOOST_CHECK(SetInletsFor(f, 225) == sw);
    BOOST_CHECK(SetInletsFor(f, 315) == wn);
}

BOOST_AUTO_TEST_CASE( replace_key )
{
    std::string s = "a $x$ b $x$";
    BOOST_CHECK_EQUAL(f.ReplaceKey(s, "$x$", "1"), TRUE);
    BOOST_CHECK_EQUAL(s, "a 1 b $x$");
    BOOST_CHECK_EQUAL(f.ReplaceKey(s, "$missing$", "1"), FALSE);
    BOOST_CHECK_EQUAL(s, "a 1 b $x$");
}

BOOST_AUTO_TEST_CASE( replace_keys )
{
    std::string s = "$x$ $x$ $x$";
    // replaces every occurrence, but returns the status of the final
    // (failed) search, so a successful replace-all reports FALSE
    BOOST_CHECK_EQUAL(f.ReplaceKeys(s, "$x$", "y"), FALSE);
    BOOST_CHECK_EQUAL(s, "y y y");

    s = "$x$ $x$ $x$";
    BOOST_CHECK_EQUAL(f.ReplaceKeys(s, "$x$", "y", 2), TRUE);
    BOOST_CHECK_EQUAL(s, "y y $x$");
}

BOOST_AUTO_TEST_CASE( is_momentum_solver )
{
    // ninjaArmy branches on this instead of comparing identify() strings
    ninja n;
    BOOST_CHECK(!n.isMomentumSolver());
    BOOST_CHECK(f.isMomentumSolver());
    ninja *base = &f;
    BOOST_CHECK(base->isMomentumSolver());
}

#ifndef WIN32
BOOST_AUTO_TEST_CASE( unknown_version )
{
    // an unsupported OpenFOAM (e.g. ESI v2606) must fail with a clear error,
    // not fall through the version branches and crash
    CPLSetConfigOption("WM_PROJECT_VERSION", "v9999");
    bool threw = false;
    try
    {
        f.simulate_wind();
    }
    catch(std::runtime_error &e)
    {
        threw = true;
        BOOST_CHECK(std::string(e.what()).find("'v9999'") != std::string::npos);
    }
    CPLSetConfigOption("WM_PROJECT_VERSION", NULL);
    BOOST_CHECK(threw);
}
#endif

BOOST_AUTO_TEST_SUITE_END()

#endif /* NINJAFOAM */

