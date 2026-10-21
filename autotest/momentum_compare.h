/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Compare momentum solver output against golden reference grids
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

#ifndef MOMENTUM_COMPARE_H
#define MOMENTUM_COMPARE_H

#include <string>
#include <vector>

/*
** A grid flattened to values, with its no-data marker.  Cells equal to
** noData in either grid are left out of every metric.
*/
struct CompareGrid
{
    int nCols;
    int nRows;
    double noData;
    std::vector<double> values;
};

struct MomentumMetrics
{
    double speedRmse;       // m/s
    double speedMaxAbs;     // m/s
    double directionMeanAbs;  // degrees, circular
    double kRelativeRmse;   // RMSE(a - b) / RMS(b)
    int nCells;             // cells valid in both speed grids
};

/* Smallest absolute difference between two directions, in [0, 180]. */
double CircularDifference(double a, double b);

double Rmse(const CompareGrid &a, const CompareGrid &b);
double MaxAbsDifference(const CompareGrid &a, const CompareGrid &b);
double MeanCircularDifference(const CompareGrid &a, const CompareGrid &b);
/* Relative to b, which is taken as the reference. */
double RelativeRmse(const CompareGrid &a, const CompareGrid &b);

/* Read band 1 of a GDAL-readable grid.  Throws std::runtime_error. */
CompareGrid ReadCompareGrid(const std::string &path);

/*
** Compare the vel.asc (mph), ang.asc and k.asc grids of two golden case
** directories (see autotest/golden/momentum/README.md).  b is the reference.
*/
MomentumMetrics CompareMomentumCases(const std::string &a, const std::string &b);

#endif /* MOMENTUM_COMPARE_H */
