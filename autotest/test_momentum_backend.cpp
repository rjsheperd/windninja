/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Test the momentum solver backend interface
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

#include "momentum_backend.h"
#include "cpu_rans_backend.h"
#ifdef WITH_WEBGPU
#include "webgpu_rans_backend.h"
#endif
#include "rans_grid.h"
#include "ninja.h"
#include "windninja.h"
#include "WindNinjaInputs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <stdexcept>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(momentum_backend)

BOOST_AUTO_TEST_CASE( problem_from_inputs )
{
    WindNinjaInputs input;
    // WindNinjaInputs stores SI units, so the problem should copy them as is
    input.inputSpeed = 4.4704;
    input.inputDirection_proj = 271.5;
    input.inputWindHeight = 6.096;
    input.outputWindHeight = 3.0;
    input.meshCount = 25000;
    input.nIterations = 300;
    input.numberCPUs = 2;

    // the backend gets domain means, as the OpenFOAM inlet BCs use today
    input.surface.Roughness = AsciiGrid<double>(2, 1, 0, 0, 30, -9999, 0.01);
    input.surface.Roughness(0, 1) = 0.03;
    input.surface.Rough_d = AsciiGrid<double>(2, 1, 0, 0, 30, -9999, 0.0);
    input.surface.Rough_d(0, 1) = 0.5;

    MomentumProblem p = MakeMomentumProblem(input);
    BOOST_CHECK(p.dem == &input.dem);
    BOOST_CHECK_CLOSE(p.speed, 4.4704, 1e-9);
    BOOST_CHECK_CLOSE(p.direction, 271.5, 1e-9);
    BOOST_CHECK_CLOSE(p.inputHeight, 6.096, 1e-9);
    BOOST_CHECK_CLOSE(p.outputHeight, 3.0, 1e-9);
    BOOST_CHECK_CLOSE(p.z0, 0.02, 1e-9);
    BOOST_CHECK_CLOSE(p.rd, 0.25, 1e-9);
    BOOST_CHECK_EQUAL(p.cellBudget, 25000);
    BOOST_CHECK_EQUAL(p.iterations, 300);
    BOOST_CHECK_EQUAL(p.nThreads, 2);
}

BOOST_AUTO_TEST_CASE( backend_default_openfoam )
{
    WindNinjaInputs input;
    BOOST_CHECK_EQUAL(input.momentumBackend, WindNinjaInputs::openFoamBackend);
    WindNinjaInputs copy(input);
    BOOST_CHECK_EQUAL(copy.momentumBackend, WindNinjaInputs::openFoamBackend);
}

BOOST_AUTO_TEST_CASE( default_iterations )
{
    // the core, CLI and GUI share one default
    WindNinjaInputs input;
    BOOST_CHECK_EQUAL(input.nIterations, WindNinjaInputs::defaultMomentumIterations);
    BOOST_CHECK_EQUAL(WindNinjaInputs::defaultMomentumIterations, 300);
}

BOOST_AUTO_TEST_CASE( backend_names )
{
    BOOST_CHECK_EQUAL(ninja::get_eMomentumBackend("openfoam"), WindNinjaInputs::openFoamBackend);
    BOOST_CHECK_EQUAL(ninja::get_eMomentumBackend("webgpu"), WindNinjaInputs::webGpuBackend);
    BOOST_CHECK_EQUAL(ninja::get_eMomentumBackend("cpu"), WindNinjaInputs::cpuBackend);
    BOOST_CHECK_EQUAL(ninja::get_momentumBackendName(WindNinjaInputs::cpuBackend), "cpu");
    BOOST_CHECK_THROW(ninja::get_eMomentumBackend("bogus"), std::invalid_argument);
    BOOST_CHECK_EQUAL(ninja::get_momentumBackendName(WindNinjaInputs::openFoamBackend), "openfoam");
    BOOST_CHECK_EQUAL(ninja::get_momentumBackendName(WindNinjaInputs::webGpuBackend), "webgpu");
}

BOOST_AUTO_TEST_CASE( backend_unavailable_throws )
{
    ninja n;
    BOOST_CHECK_NO_THROW(n.set_momentumBackend(WindNinjaInputs::openFoamBackend));
    BOOST_CHECK_NO_THROW(n.set_momentumBackend(WindNinjaInputs::cpuBackend));
    n.set_momentumBackend(WindNinjaInputs::openFoamBackend);
#ifdef WITH_WEBGPU
    BOOST_CHECK_NO_THROW(n.set_momentumBackend(WindNinjaInputs::webGpuBackend));
    BOOST_CHECK_EQUAL(n.input.momentumBackend, WindNinjaInputs::webGpuBackend);
#else
    // the WebGPU backend needs a WITH_WEBGPU build
    BOOST_CHECK_THROW(n.set_momentumBackend(WindNinjaInputs::webGpuBackend), std::invalid_argument);
    BOOST_CHECK_EQUAL(n.input.momentumBackend, WindNinjaInputs::openFoamBackend);
#endif
}

static std::string GetBackend(NinjaArmyH *army, int nIndex)
{
    const char *name = NinjaGetMomentumBackend(army, nIndex, NULL);
    std::string s = name ? name : "(null)";
    free((void *)name);
    return s;
}

BOOST_AUTO_TEST_CASE( c_api_set_get_backend )
{
    BOOST_REQUIRE_EQUAL(NinjaInit("C-API autotest", NULL), NINJA_SUCCESS);
    NinjaArmyH *army = NinjaInitializeArmy();
    BOOST_REQUIRE(army != NULL);
    const double speed[] = {5.0};
    const double direction[] = {270.0};
    BOOST_REQUIRE_EQUAL(NinjaMakeDomainAverageArmy(army, 1, true, speed, "mps", direction, NULL),
                        NINJA_SUCCESS);

    BOOST_CHECK_EQUAL(GetBackend(army, 0), "openfoam");
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, "openfoam", NULL), NINJA_SUCCESS);
#ifdef WITH_WEBGPU
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, "webgpu", NULL), NINJA_SUCCESS);
    BOOST_CHECK_EQUAL(GetBackend(army, 0), "webgpu");
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, "openfoam", NULL), NINJA_SUCCESS);
#else
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, "webgpu", NULL), NINJA_E_INVALID);
#endif
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, "bogus", NULL), NINJA_E_INVALID);
    BOOST_CHECK_EQUAL(GetBackend(army, 0), "openfoam");

    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, NULL, NULL), NINJA_E_NULL_PTR);
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(NULL, 0, "openfoam", NULL), NINJA_E_NULL_PTR);
    BOOST_CHECK(NinjaGetMomentumBackend(NULL, 0, NULL) == NULL);
    BOOST_CHECK_EQUAL(GetBackend(army, 5), "");

    BOOST_CHECK_EQUAL(NinjaDestroyArmy(army, NULL), NINJA_SUCCESS);
}

BOOST_AUTO_TEST_CASE( cpu_build_grid )
{
    // 4 x 3 cells of 100 m at a UTM-like origin, with a slope and one bump
    Elevation dem(4, 3, 500000.0, 4800000.0, 100.0, -9999.0, Elevation::meters);
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 4; c++)
            dem.set_cellValue(r, c, 1000.0 + 10.0 * c + 5.0 * r);
    dem.set_cellValue(1, 2, 1080.0);

    MomentumProblem p;
    p.dem = &dem;
    p.meshResolution = 100.0;
    CpuRansBackend backend;
    backend.layers = 6;
    RansGrid g = backend.buildGrid(p, 100.0);

    BOOST_REQUIRE_EQUAL(g.nx, 4);
    BOOST_REQUIRE_EQUAL(g.ny, 3);
    BOOST_REQUIRE_EQUAL(g.nz, 6);
    // cell centres on the DEM's cell centres (AsciiGrid row 0 is south)
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 4; c++)
        {
            double x, y;
            dem.get_cellPosition(r, c, &x, &y);
            int cell = g.cell(c, r, 0);
            BOOST_CHECK_CLOSE(g.cx[cell] + g.xOrigin, x, 1e-10);
            BOOST_CHECK_CLOSE(g.cy[cell] + g.yOrigin, y, 1e-10);
        }
    // corner ground is the mean of the DEM cells around it
    BOOST_CHECK_CLOSE(g.z[g.node(0, 0, 0)], dem.get_cellValue(0, 0), 1e-10);
    double inner = 0.25 * (dem.get_cellValue(0, 1) + dem.get_cellValue(0, 2) +
                           dem.get_cellValue(1, 1) + dem.get_cellValue(1, 2));
    BOOST_CHECK_CLOSE(g.z[g.node(2, 1, 0)], inner, 1e-10);
    // the first layer is the mesh resolution everywhere
    for(int j = 0; j <= g.ny; j++)
        for(int i = 0; i <= g.nx; i++)
            BOOST_CHECK_CLOSE(g.z[g.node(i, j, 1)] - g.z[g.node(i, j, 0)], 100.0, 1e-10);
    // flat top at OpenFOAM's height: max elevation plus the largest of 10% of
    // the extent, 1.1 x relief and 11 block cells of 2 x the mesh resolution
    double top = 1080.0 + std::max(std::max(0.1 * 400.0, 1.1 * 80.0), 11 * 200.0);
    for(int j = 0; j <= g.ny; j++)
        for(int i = 0; i <= g.nx; i++)
            BOOST_CHECK_CLOSE(g.z[g.node(i, j, g.nz)], top, 1e-10);
}

#ifdef WITH_WEBGPU
struct Cancelled {};

BOOST_AUTO_TEST_CASE( webgpu_cancel )
{
    // a 40 x 30 hill at 30 m; far more steps than can run before cancelling
    Elevation dem(40, 30, 500000.0, 4800000.0, 30.0, -9999.0, Elevation::meters);
    for(int r = 0; r < 30; r++)
        for(int c = 0; c < 40; c++)
            dem.set_cellValue(r, c, 1000.0 + 150.0 * std::exp(-((c - 20) * (c - 20) +
                                                                (r - 15) * (r - 15)) / 40.0));
    MomentumProblem p;
    p.dem = &dem;
    p.z0 = 0.01;
    p.rd = 0.0;
    p.speed = 4.47;
    p.direction = 270.0;
    p.inputHeight = 6.1;
    p.outputHeight = 6.1;
    p.meshResolution = 30.0;
    p.cellBudget = 0;
    p.iterations = 0;
    p.nThreads = 1;
    WebGpuRansBackend backend;
    backend.maxIterations = 100000000;

    // cancel 0.3 s in; the backend checks between 100-step chunks, so it must
    // return well within a second.  Ten runs in a row: each releases its GPU
    // objects (RAII), so later runs start clean.  (WebGPU doesn't expose
    // device memory use, so a leak can't be measured directly here.)
    typedef std::chrono::steady_clock Clock;
    for(int run = 0; run < 10; run++)
    {
        Clock::time_point start = Clock::now(), requested;
        bool cancelled = false;
        MomentumResult result;
        ninjaComClass com;
        try
        {
            backend.solve(p, &com,
                [&]()
                {
                    if(Clock::now() - start > std::chrono::milliseconds(300))
                    {
                        requested = Clock::now();
                        throw Cancelled();
                    }
                },
                [](const MomentumResult &) { return true; }, result);
        }
        catch(const Cancelled &)
        {
            cancelled = true;
        }
        catch(const std::runtime_error &e)
        {
            if(std::string(e.what()).find("no adapter") != std::string::npos)
            {
                std::cout << "SKIP: no WebGPU adapter\n";
                return;
            }
            throw;
        }
        BOOST_REQUIRE(cancelled);
        double latency = std::chrono::duration<double>(Clock::now() - requested).count();
        double total = std::chrono::duration<double>(Clock::now() - start).count();
        BOOST_TEST_MESSAGE("run " << run << ": cancelled after " << total << " s");
        BOOST_CHECK_LT(latency, 1.0);
        BOOST_CHECK_LT(total, 1.3);
    }
}
BOOST_AUTO_TEST_CASE( webgpu_cfl_backoff )
{
    // the 40 x 30 hill, started at a CFL far too large to be stable: the
    // backend must back off (8 -> 4 -> 2 -> 1 -> 0.5) and finish with finite
    // results instead of failing
    Elevation dem(40, 30, 500000.0, 4800000.0, 30.0, -9999.0, Elevation::meters);
    for(int r = 0; r < 30; r++)
        for(int c = 0; c < 40; c++)
            dem.set_cellValue(r, c, 1000.0 + 150.0 * std::exp(-((c - 20) * (c - 20) +
                                                                (r - 15) * (r - 15)) / 40.0));
    MomentumProblem p;
    p.dem = &dem;
    p.z0 = 0.01;
    p.rd = 0.0;
    p.speed = 4.47;
    p.direction = 270.0;
    p.inputHeight = 6.1;
    p.outputHeight = 6.1;
    p.meshResolution = 30.0;
    p.cellBudget = 0;
    p.iterations = 0;
    p.nThreads = 1;
    WebGpuRansBackend backend;
    backend.cfl = 8.0;
    backend.maxRetries = 5;
    backend.maxIterations = 3000;
    ninjaComClass com;
    MomentumResult result;
    try
    {
        BOOST_CHECK(backend.solve(p, &com, []() {},
                                  [](const MomentumResult &) { return true; }, result));
    }
    catch(const std::runtime_error &e)
    {
        if(std::string(e.what()).find("no adapter") != std::string::npos)
        {
            std::cout << "SKIP: no WebGPU adapter\n";
            return;
        }
        throw;
    }
    BOOST_TEST_MESSAGE("finished at CFL " << backend.lastCfl);
    BOOST_CHECK_LT(backend.lastCfl, 8.0);
    for(int r = 0; r < 30; r++)
        for(int c = 0; c < 40; c++)
            BOOST_CHECK(std::isfinite(result.u(r, c)) && std::isfinite(result.k(r, c)));

    // the same with multigrid, whose restart must also clear the coarse
    // forcing (NaN from the diverged attempt would poison it)
    backend.multigridLevels = 6;
    backend.maxIterations = 500;
    backend.checkEvery = 10;
    BOOST_CHECK(backend.solve(p, &com, []() {},
                              [](const MomentumResult &) { return true; }, result));
    BOOST_TEST_MESSAGE("multigrid finished at CFL " << backend.lastCfl);
    BOOST_CHECK_LT(backend.lastCfl, 8.0);
    for(int r = 0; r < 30; r++)
        for(int c = 0; c < 40; c++)
            BOOST_CHECK(std::isfinite(result.u(r, c)) && std::isfinite(result.k(r, c)));

    // and with no retries left it reports the divergence
    backend.maxRetries = 0;
    BOOST_CHECK_THROW(backend.solve(p, &com, []() {},
                                    [](const MomentumResult &) { return true; }, result),
                      MomentumDivergedError);
}
#endif /* WITH_WEBGPU */

BOOST_AUTO_TEST_SUITE_END()

#endif /* NINJAFOAM */
