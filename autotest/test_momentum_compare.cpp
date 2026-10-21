/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Test the momentum solver golden comparison metrics
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

#include "momentum_compare.h"

#include <stdexcept>
#include <string>

#include <boost/test/unit_test.hpp>

static CompareGrid MakeGrid(std::vector<double> values, double noData = -9999.0)
{
    CompareGrid g;
    g.nCols = (int)values.size();
    g.nRows = 1;
    g.noData = noData;
    g.values = values;
    return g;
}

static std::string GoldenCase(const std::string &name)
{
    return std::string(WINDNINJA_SOURCE_DIR) + "/autotest/golden/momentum/" + name;
}

BOOST_AUTO_TEST_SUITE(momentum_compare)

BOOST_AUTO_TEST_CASE( identity )
{
    CompareGrid a = MakeGrid({1.0, 2.0, 3.0, 4.0});
    BOOST_CHECK_EQUAL(Rmse(a, a), 0.0);
    BOOST_CHECK_EQUAL(MaxAbsDifference(a, a), 0.0);
    BOOST_CHECK_EQUAL(MeanCircularDifference(a, a), 0.0);
    BOOST_CHECK_EQUAL(RelativeRmse(a, a), 0.0);
}

BOOST_AUTO_TEST_CASE( identity_golden_case )
{
    std::string c = GoldenCase("big_butte_small_coarse_270");
    MomentumMetrics m = CompareMomentumCases(c, c);
    BOOST_CHECK_EQUAL(m.speedRmse, 0.0);
    BOOST_CHECK_EQUAL(m.speedMaxAbs, 0.0);
    BOOST_CHECK_EQUAL(m.directionMeanAbs, 0.0);
    BOOST_CHECK_EQUAL(m.kRelativeRmse, 0.0);
    BOOST_CHECK_EQUAL(m.nCells, 95 * 105);
}

BOOST_AUTO_TEST_CASE( known_offset )
{
    CompareGrid ref = MakeGrid({1.0, 2.0, 3.0, 4.0});
    CompareGrid shifted = MakeGrid({2.0, 3.0, 4.0, 5.0});
    BOOST_CHECK_CLOSE(Rmse(shifted, ref), 1.0, 1e-9);
    BOOST_CHECK_CLOSE(MaxAbsDifference(shifted, ref), 1.0, 1e-9);
}

BOOST_AUTO_TEST_CASE( known_rotation )
{
    CompareGrid ref = MakeGrid({0.0, 90.0, 270.0, 355.0});
    CompareGrid rotated = MakeGrid({10.0, 100.0, 280.0, 5.0});
    BOOST_CHECK_CLOSE(MeanCircularDifference(rotated, ref), 10.0, 1e-9);
}

BOOST_AUTO_TEST_CASE( circular_difference_wraps )
{
    BOOST_CHECK_CLOSE(CircularDifference(355.0, 5.0), 10.0, 1e-9);
    BOOST_CHECK_CLOSE(CircularDifference(5.0, 355.0), 10.0, 1e-9);
    BOOST_CHECK_CLOSE(CircularDifference(0.0, 180.0), 180.0, 1e-9);
    BOOST_CHECK_CLOSE(CircularDifference(720.0, 10.0), 10.0, 1e-9);
    BOOST_CHECK_SMALL(CircularDifference(360.0, 0.0), 1e-12);
}

BOOST_AUTO_TEST_CASE( relative_rmse )
{
    // RMSE(a - b) = 1, RMS(b) = 2, so 0.5
    CompareGrid ref = MakeGrid({2.0, 2.0, 2.0, 2.0});
    CompareGrid a = MakeGrid({3.0, 1.0, 3.0, 1.0});
    BOOST_CHECK_CLOSE(RelativeRmse(a, ref), 0.5, 1e-9);
}

BOOST_AUTO_TEST_CASE( no_data_skipped )
{
    CompareGrid ref = MakeGrid({1.0, -9999.0, 3.0});
    CompareGrid a = MakeGrid({1.0, 500.0, 3.0});
    BOOST_CHECK_EQUAL(MaxAbsDifference(a, ref), 0.0);
}

BOOST_AUTO_TEST_CASE( mismatched_size_throws )
{
    CompareGrid a = MakeGrid({1.0, 2.0});
    CompareGrid b = MakeGrid({1.0, 2.0, 3.0});
    BOOST_CHECK_THROW(Rmse(a, b), std::runtime_error);
}

BOOST_AUTO_TEST_CASE( no_valid_cells_throws )
{
    CompareGrid a = MakeGrid({-9999.0, -9999.0});
    BOOST_CHECK_THROW(Rmse(a, a), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
