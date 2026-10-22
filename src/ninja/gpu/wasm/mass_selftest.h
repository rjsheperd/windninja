/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  A shared native and browser self-test for the mass solver
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

#ifndef MASS_SELFTEST_H
#define MASS_SELFTEST_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "mass_dem.h"
#include "mass_fem.h"

/*
** A mass-solver problem solved natively (autotest gpu/mass_selftest) and in
** the browser (windninja_wasm's self-test), on the GPU (SolveMassWindOnGpu)
** and on the CPU (AssembleMass, SolveMassPcg with SSOR, RecoverVelocity),
** each checked against statistics measured natively, so the browser build is
** held to the native result.  Grid() also sizes the browser's timing run.
*/
namespace mass_selftest {

/*
** A ncols x nrows x nlayers terrain-following grid shaped like WindNinja's:
** 60 m cells at UTM-sized, non-integer coordinates over a Gaussian ridge at
** 1500 m, a 20 m first layer growing 1.2x, flat top at 4000 m.
*/
inline MassGrid Grid(int ncols = 40, int nrows = 33, int nlayers = 12)
{
    MassGrid g;
    g.ncols = ncols;
    g.nrows = nrows;
    g.nlayers = nlayers;
    const double dx = 60.0, top = 4000.0, x0 = 512345.37, y0 = 4812345.61;
    std::vector<double> frac(nlayers, 0.0);
    double h = 20.0, total = 0.0;
    for(int k = 1; k < nlayers; k++) { total += h; frac[k] = total; h *= 1.2; }
    for(int k = 0; k < nlayers; k++) frac[k] /= total;
    const double cx = 0.5 * (ncols - 1) * dx, cy = 0.5 * (nrows - 1) * dx;
    for(int k = 0; k < nlayers; k++)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < ncols; j++)
            {
                double x = j * dx, y = i * dx;
                double ground = 1500.0 + 400.0 * std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) /
                                                          (2.0 * 300.0 * 300.0));
                g.x.push_back(x0 + x);
                g.y.push_back(y0 + y);
                g.z.push_back(ground + (top - ground) * frac[k]);
            }
    return g;
}

// A log-like profile from the west-south-west, with a little divergence.
struct Wind
{
    std::vector<double> u0, v0, w0, alphaV;
    double alphaH = 1.0;
};

inline Wind Initial(const MassGrid &g)
{
    Wind w;
    const int n = g.numNodes();
    const int layer = g.ncols * g.nrows;
    w.u0.resize(n);
    w.v0.resize(n);
    w.w0.assign(n, 0.0);
    w.alphaV.assign(n, 1.0);
    for(int i = 0; i < n; i++)
    {
        double agl = g.z[i] - g.z[i % layer];   // height above the ground node below
        double profile = std::log((agl + 0.1) / 0.01) / std::log(6.096 / 0.01);
        w.u0[i] = 4.4704 * profile * 0.94 + 0.0005 * (g.x[i] - g.x[0]);
        w.v0[i] = 4.4704 * profile * 0.34;
    }
    return w;
}

struct Stats
{
    double meanSpeed, maxSpeed, meanW;
};

// over the ground layer and the one above, where the terrain acts most
inline Stats Summarize(const MassGrid &g, const std::vector<double> &u,
                       const std::vector<double> &v, const std::vector<double> &w)
{
    Stats s = {0.0, 0.0, 0.0};
    const int count = 2 * g.ncols * g.nrows;
    for(int i = 0; i < count; i++)
    {
        double speed = std::hypot(u[i], v[i]);
        s.meanSpeed += speed;
        s.maxSpeed = std::max(s.maxSpeed, speed);
        s.meanW += w[i];
    }
    s.meanSpeed /= count;
    s.meanW /= count;
    return s;
}

// measured natively on Apple M3 Max: the GPU (Dawn, Metal, f32 Jacobi) and
// the CPU (f64 SSOR).  The browser matched both to all 9 digits (Chromium,
// Metal).  The tolerances are tight enough to tell the two solvers apart
// (their max speeds differ by 4.4e-4 relative): f32 leaves room for other
// GPUs' rounding, f64 for summation order.
const Stats expectedGpu = {4.42852295, 8.8866409, 0.0157581090};
const Stats expectedCpu = {4.42845500, 8.88270784, 0.0151645325};
const double gpuTolerance = 1e-5;   // relative (meanW: of meanSpeed)
const double cpuTolerance = 1e-8;

inline bool Matches(const Stats &s, const Stats &e, double tolerance)
{
    auto close = [tolerance](double a, double b, double scale) {
        return std::fabs(a - b) <= tolerance * scale;
    };
    return close(s.meanSpeed, e.meanSpeed, e.meanSpeed) && close(s.maxSpeed, e.maxSpeed, e.maxSpeed) &&
           close(s.meanW, e.meanW, e.meanSpeed);
}

/*
** DEM in, wind out (SolveMassDem, the browser's windninja_mass_dem_solve): a
** hill at 30 m, grass, 10 mph from 237 degrees at 20 ft, out at 20 ft.  The
** CPU path writes exactly the CLI's output (scripts/check_mass_dem.sh), so
** matching these statistics holds the browser to the CLI.
*/
inline TerrainDem Dem()
{
    TerrainDem dem;
    dem.nCols = 48;
    dem.nRows = 36;
    dem.cellSize = 30.0;
    dem.xll = 333861.94;
    dem.yll = 4806622.16;
    for(int r = 0; r < dem.nRows; r++)
        for(int c = 0; c < dem.nCols; c++)
            dem.z.push_back(1500.0 + 220.0 * std::exp(-((c - 22) * (c - 22) + (r - 17) * (r - 17)) / 60.0) +
                            0.8 * c - 0.5 * r);
    return dem;
}

inline MassDemProblem DemProblem()
{
    MassDemProblem p;
    p.wind.speed = 4.4704;   // 10 mph
    p.wind.direction = 237.0;
    p.wind.inputHeight = 6.096;
    p.wind.latitude = 43.4;
    p.wind.angleFromNorth = -1.4;
    p.outputHeight = 6.096;
    return p;   // grass: z0 0.01, no canopy
}

// The probes catch a flipped or shifted grid, which the means can't.
struct DemStats
{
    double meanSpeed, maxSpeed, meanAngle, speedSouthWest, speedNorthEast;
};

// speed and angle per cell, row 0 south, nCols wide
inline DemStats SummarizeDem(const MassDemWind &w)
{
    DemStats s = {0.0, 0.0, 0.0, 0.0, 0.0};
    s.speedSouthWest = w.speed[3 * w.nCols + 4];                    // row 3, col 4
    s.speedNorthEast = w.speed[(w.nRows - 4) * w.nCols + w.nCols - 5];
    for(std::size_t i = 0; i < w.speed.size(); i++)
    {
        s.meanSpeed += w.speed[i];
        s.maxSpeed = std::max(s.maxSpeed, w.speed[i]);
        s.meanAngle += w.angle[i];   // all near 238: no wrap
    }
    s.meanSpeed /= w.speed.size();
    s.meanAngle /= w.speed.size();
    return s;
}

// native, as expectedGpu / expectedCpu (gpuTolerance, cpuTolerance)
const DemStats expectedDemGpu = {4.34612390, 6.67289132, 238.508957, 4.20736024, 4.37419166};
const DemStats expectedDemCpu = {4.34580825, 6.67514947, 238.485491, 4.19083457, 4.36724479};

inline bool MatchesDem(const DemStats &s, const DemStats &e, double tolerance)
{
    auto close = [tolerance](double a, double b) { return std::fabs(a - b) <= tolerance * std::fabs(b); };
    return close(s.meanSpeed, e.meanSpeed) && close(s.maxSpeed, e.maxSpeed) &&
           close(s.meanAngle, e.meanAngle) && close(s.speedSouthWest, e.speedSouthWest) &&
           close(s.speedNorthEast, e.speedNorthEast);
}

} // namespace mass_selftest

#endif /* MASS_SELFTEST_H */
