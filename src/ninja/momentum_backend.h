/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Interface between the momentum solver and its CFD backends
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

#ifndef MOMENTUM_BACKEND_H
#define MOMENTUM_BACKEND_H

#include <functional>

#include "ascii_grid.h"
#include "Elevation.h"
#include "ninjaCom.h"

class WindNinjaInputs;

/*
** Everything a momentum solver backend needs, in SI units.  The fields mirror
** what the OpenFOAM backend consumes today: one inflow speed and direction,
** and the domain mean roughness and displacement height.
*/
struct MomentumProblem
{
    const Elevation *dem;    // resampled to the mesh resolution
    double z0;               // m, domain mean of surface.Roughness
    double rd;               // m, domain mean of surface.Rough_d
    double speed;            // m/s, at inputHeight
    double direction;        // degrees clockwise from the dem's y axis,
                             // direction the wind comes from
    double inputHeight;      // m above the canopy
    double outputHeight;     // m above the canopy
    int cellBudget;          // target number of cells (WindNinjaInputs::meshCount)
    int iterations;
    int nThreads;
};

/*
** Grids on the output-height surface: u and v in m/s, k in m^2/s^2.
**
** If the mesh does not resolve the output height (first cell taller than
** it), the grids hold ground-cell values representative of sampleHeight
** rather than of the output height.  The caller then log-profiles them down
** to the output height, so every backend gets the same post-processing.
*/
struct MomentumResult
{
    AsciiGrid<double> u;
    AsciiGrid<double> v;
    AsciiGrid<double> k;
    bool outputHeightResolved;
    double sampleHeight;     // m above the ground that u, v, k represent when
                             // !outputHeightResolved (the caller converts to
                             // height above the canopy per cell)
};

class MomentumBackend
{
public:
    virtual ~MomentumBackend() {}

    /*
    ** Solve the problem.  Progress goes to com; checkCancel is called
    ** periodically and throws cancelledByUser to stop the run.
    */
    virtual MomentumResult solve(const MomentumProblem &problem,
                                 ninjaComClass *com,
                                 const std::function<void()> &checkCancel) = 0;
};

/* Build a MomentumProblem from the inputs of a momentum solver run. */
MomentumProblem MakeMomentumProblem(const WindNinjaInputs &input);

#endif /* MOMENTUM_BACKEND_H */
