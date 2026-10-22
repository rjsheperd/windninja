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

#include <cmath>
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

BOOST_AUTO_TEST_CASE( resample_linear_exact )
{
    // f = 2x + 3y at cell centres of a 10 x 8 grid of 10 m cells
    CompareGrid src;
    src.nCols = 10; src.nRows = 8; src.noData = -9999.0;
    src.xll = 1000.0; src.yll = 2000.0; src.cellSize = 10.0;
    for(int r = 0; r < src.nRows; r++)
        for(int c = 0; c < src.nCols; c++)
        {
            double x = src.xll + (c + 0.5) * 10.0, y = src.yll + (src.nRows - r - 0.5) * 10.0;
            src.values.push_back(2.0 * x + 3.0 * y);
        }
    // a coarser, offset target partly outside src's cell centres
    CompareGrid target;
    target.nCols = 9; target.nRows = 7; target.noData = -9999.0;
    target.xll = 1001.0; target.yll = 2002.0; target.cellSize = 13.0;
    target.values.assign(9 * 7, 0.0);
    CompareGrid out = ResampleTo(src, target);
    int inside = 0;
    for(int r = 0; r < target.nRows; r++)
        for(int c = 0; c < target.nCols; c++)
        {
            double x = target.xll + (c + 0.5) * 13.0, y = target.yll + (target.nRows - r - 0.5) * 13.0;
            double v = out.values[(size_t)r * target.nCols + c];
            bool in = x >= 1005.0 && x <= 1095.0 && y >= 2005.0 && y <= 2075.0;
            if(in)
            {
                BOOST_CHECK_SMALL(v - (2.0 * x + 3.0 * y), 1e-9);
                inside++;
            }
            else
                BOOST_CHECK(std::isnan(v));
        }
    BOOST_CHECK_GT(inside, 20);
}

BOOST_AUTO_TEST_CASE( resampled_identity_golden_case )
{
    std::string c = GoldenCase("big_butte_small_coarse_270");
    MomentumMetrics m = CompareMomentumCasesResampled(c, c);
    BOOST_CHECK_SMALL(m.speedRmse, 1e-9);
    BOOST_CHECK_SMALL(m.directionMeanAbs, 1e-6);
    BOOST_CHECK(std::isnan(m.kRelativeRmse));
}

BOOST_AUTO_TEST_CASE( resampled_matches_independent_check )
{
    // OpenFOAM coarse vs medium at 20 ft, cross-checked with an independent
    // numpy implementation (WEBGPU_PLAN 2.4: 0.382 m/s, 4.04 deg)
    MomentumMetrics m = CompareMomentumCasesResampled(GoldenCase("big_butte_small_coarse_270"),
                                                      GoldenCase("big_butte_small_medium_270"));
    BOOST_TEST_MESSAGE("speed RMSE " << m.speedRmse << ", direction " << m.directionMeanAbs);
    BOOST_CHECK_CLOSE(m.speedRmse, 0.382, 1.0);
    BOOST_CHECK_CLOSE(m.directionMeanAbs, 4.04, 1.0);
}

BOOST_AUTO_TEST_CASE( no_valid_cells_throws )
{
    CompareGrid a = MakeGrid({-9999.0, -9999.0});
    BOOST_CHECK_THROW(Rmse(a, a), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
