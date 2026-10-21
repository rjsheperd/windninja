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
#include "ninja.h"
#include "windninja.h"
#include "WindNinjaInputs.h"

#include <cstdlib>
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
    BOOST_CHECK_THROW(ninja::get_eMomentumBackend("bogus"), std::invalid_argument);
    BOOST_CHECK_EQUAL(ninja::get_momentumBackendName(WindNinjaInputs::openFoamBackend), "openfoam");
    BOOST_CHECK_EQUAL(ninja::get_momentumBackendName(WindNinjaInputs::webGpuBackend), "webgpu");
}

BOOST_AUTO_TEST_CASE( backend_unavailable_throws )
{
    ninja n;
    BOOST_CHECK_NO_THROW(n.set_momentumBackend(WindNinjaInputs::openFoamBackend));
    // no WebGPU backend is compiled in yet
    BOOST_CHECK_THROW(n.set_momentumBackend(WindNinjaInputs::webGpuBackend), std::invalid_argument);
    BOOST_CHECK_EQUAL(n.input.momentumBackend, WindNinjaInputs::openFoamBackend);
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
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, "webgpu", NULL), NINJA_E_INVALID);
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, "bogus", NULL), NINJA_E_INVALID);
    BOOST_CHECK_EQUAL(GetBackend(army, 0), "openfoam");

    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(army, 0, NULL, NULL), NINJA_E_NULL_PTR);
    BOOST_CHECK_EQUAL(NinjaSetMomentumBackend(NULL, 0, "openfoam", NULL), NINJA_E_NULL_PTR);
    BOOST_CHECK(NinjaGetMomentumBackend(NULL, 0, NULL) == NULL);
    BOOST_CHECK_EQUAL(GetBackend(army, 5), "");

    BOOST_CHECK_EQUAL(NinjaDestroyArmy(army, NULL), NINJA_SUCCESS);
}

BOOST_AUTO_TEST_SUITE_END()

#endif /* NINJAFOAM */
