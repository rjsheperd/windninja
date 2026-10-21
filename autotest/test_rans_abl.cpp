/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Test the ABL inlet and wall-function formulas against OpenFOAM
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

#include "rans_abl.h"

#include <cmath>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(rans_abl)

/*
** Each formula restated literally from the OpenFOAM BC code it ports
** (src/ninjafoam/11 and OpenFOAM-11 epsilonWallFunction) and compared at
** several heights, as the plan's rans_bc_port_matches_openfoam test.
*/
BOOST_AUTO_TEST_CASE( bc_port_matches_openfoam )
{
    const double U = 4.4704, h = 6.096, z0 = 0.01;   // 10 mph at 20 ft over grass
    // logProfileVelocityInlet: ustar = U*0.41/log(h/z0); u = ustar/0.41*log(AGL/z0)
    double ustar = U * 0.41 / std::log(h / z0);
    BOOST_CHECK_CLOSE(AblFrictionVelocity(U, h, z0), ustar, 1e-10);
    for(double agl : {0.5, 3.0, 6.096, 40.0, 250.0})
    {
        BOOST_CHECK_CLOSE(AblInletVelocity(ustar, agl, z0), ustar / 0.41 * std::log(agl / z0), 1e-10);
        // logProfileDissipationRateInlet: kappa = 0.4187
        BOOST_CHECK_CLOSE(AblInletEpsilon(ustar, agl, z0), std::pow(ustar, 3) / (0.4187 * (agl + z0)), 1e-10);
    }
    // logProfileTurbulentKineticEnergyInlet: Cmu = 0.085
    BOOST_CHECK_CLOSE(AblInletK(ustar), ustar * ustar / std::sqrt(0.085), 1e-10);

    // nutkAtmRoughWallFunction with WindNinja's Cmu 0.085, kappa 0.41
    const double nu = 1.58946099535e-05, k = 0.4, y = 40.0;
    double yPlus = std::pow(0.085, 0.25) * std::sqrt(k) * y / nu;
    BOOST_CHECK_CLOSE(RoughWallNut(nu, k, y, z0), nu * (yPlus * 0.41 / std::log((y + z0) / z0) - 1), 1e-10);
}

BOOST_AUTO_TEST_CASE( inlet_profile_hits_input_speed )
{
    const double U = 4.4704, h = 6.096, z0 = 0.01;
    double ustar = AblFrictionVelocity(U, h, z0);
    BOOST_CHECK_CLOSE(AblInletVelocity(ustar, h, z0), U, 1e-10);
    BOOST_CHECK_CLOSE(ustar, 0.28581, 0.01);   // hand-computed
}

BOOST_AUTO_TEST_CASE( wall_functions )
{
    // OpenFOAM's yPlusLam for kappa 0.41, E 9.8
    BOOST_CHECK_CLOSE(WallYPlusLam(), 11.53, 0.1);

    // log branch: epsilon = Cmu^0.75 k^1.5 / (kappa y), G uses (nut + nu)
    const double nu = 1.58946099535e-05, k = 0.4, y = 40.0, nutw = 3.0, dUdn = 0.05, z0 = 0.01;
    double eps, G;
    EpsilonWallFunction(nu, nutw, k, y, dUdn, eps, G);
    BOOST_CHECK_CLOSE(eps, std::pow(0.085, 0.75) * std::pow(k, 1.5) / (0.41 * y), 1e-10);
    BOOST_CHECK_CLOSE(G, (nutw + nu) * dUdn * std::pow(0.085, 0.25) * std::sqrt(k) / (0.41 * y), 1e-10);

    // laminar branch (y+ below yPlusLam): epsilon = 2 k nu / y^2, no G
    EpsilonWallFunction(nu, 0.0, 1e-8, 1e-3, dUdn, eps, G);
    BOOST_CHECK_CLOSE(eps, 2.0 * 1e-8 * nu / 1e-6, 1e-10);
    BOOST_CHECK_EQUAL(G, 0.0);

    // the Edash clamp keeps nut finite when y is far below z0
    BOOST_CHECK(std::isfinite(RoughWallNut(nu, k, 1e-9, z0)));
}

BOOST_AUTO_TEST_SUITE_END()
