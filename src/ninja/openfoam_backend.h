/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Momentum solver backend that runs OpenFOAM
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

#ifndef OPENFOAM_BACKEND_H
#define OPENFOAM_BACKEND_H

#include "momentum_backend.h"

class NinjaFoam;

/*
** MomentumBackend that runs OpenFOAM.  It forwards to the OpenFOAM pipeline
** in NinjaFoam (case files, meshing, solve, sampling, case reuse and the
** DEM-smoothing retry), which reads its settings from the NinjaFoam inputs
** rather than from the MomentumProblem.
*/
class OpenFoamBackend : public MomentumBackend
{
public:
    explicit OpenFoamBackend(NinjaFoam &foam) : foam(foam) {}

    virtual bool solve(const MomentumProblem &problem,
                       ninjaComClass *com,
                       const std::function<void()> &checkCancel,
                       const std::function<bool(const MomentumResult &)> &accept,
                       MomentumResult &result);

private:
    NinjaFoam &foam;
};

#endif /* OPENFOAM_BACKEND_H */
