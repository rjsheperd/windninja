/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Self-test problem shared by the native and browser solver tests
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

#ifndef HILL_SELFTEST_H
#define HILL_SELFTEST_H

#include <cmath>
#include <vector>

#include "rans_dem_solve.h"

/*
** A Gaussian hill solved with SolveDemOnGpu, natively (autotest
** gpu/dem_solve_hill) and in the browser (windninja_wasm's self-test), and
** checked against the same expected statistics, so the browser build is
** held to the native result.
*/
namespace hill_selftest {

inline TerrainDem Dem()
{
    TerrainDem dem;
    dem.nCols = 40;
    dem.nRows = 30;
    dem.cellSize = 30.0;
    dem.xll = 500000.0;
    dem.yll = 4800000.0;
    for(int r = 0; r < dem.nRows; r++)
        for(int c = 0; c < dem.nCols; c++)
            dem.z.push_back(1000.0 + 150.0 * std::exp(-((c - 20) * (c - 20) +
                                                        (r - 15) * (r - 15)) / 40.0));
    return dem;
}

inline DemWindProblem Problem()
{
    DemWindProblem p;
    p.speed = 4.4704;          // 10 mph
    p.inputHeight = 6.096;     // 20 ft
    p.direction = 270.0;
    p.z0 = 0.01;
    p.meshResolution = 30.0;
    return p;
}

inline GpuSolveOptions Options()
{
    GpuSolveOptions o;
    o.maxIterations = 3000;
    o.outputTolerance = 0.0;   // a fixed number of steps
    o.tolerance = 0.0;
    return o;
}

struct Stats
{
    double meanSpeed, maxSpeed, meanK;
};

inline Stats Summarize(const DemWind &w)
{
    Stats s = {0.0, 0.0, 0.0};
    for(std::size_t i = 0; i < w.u.size(); i++)
    {
        double speed = std::hypot(w.u[i], w.v[i]);
        s.meanSpeed += speed;
        s.maxSpeed = std::max(s.maxSpeed, speed);
        s.meanK += w.k[i];
    }
    s.meanSpeed /= w.u.size();
    s.meanK /= w.u.size();
    return s;
}

// measured natively on Apple M3 Max (Dawn, Metal); f32 on any GPU should
// agree well within the tolerance
const Stats expected = {4.54135083, 7.49628572, 0.417968784};
const double tolerance = 1e-3;   // relative

inline bool Matches(const Stats &s)
{
    auto close = [](double a, double b) { return std::fabs(a - b) <= tolerance * std::fabs(b); };
    return close(s.meanSpeed, expected.meanSpeed) && close(s.maxSpeed, expected.maxSpeed) &&
           close(s.meanK, expected.meanK);
}

} // namespace hill_selftest

#endif /* HILL_SELFTEST_H */
