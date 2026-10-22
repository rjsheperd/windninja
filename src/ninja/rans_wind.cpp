/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Atmospheric wind setup for the CPU RANS solver
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

#include "rans_wind.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "rans_abl.h"
#include "rans_solver.h"

void WindDirectionVector(double directionFrom, double &dx, double &dy)
{
    // blowing toward directionFrom - 180; clockwise from +y means
    // x = sin, y = cos
    double toward = (directionFrom - 180.0) * 3.14159265358979323846 / 180.0;
    dx = std::sin(toward);
    dy = std::cos(toward);
    // keep exact zeros on the axes
    if(std::fabs(dx) < 1e-15) dx = 0.0;
    if(std::fabs(dy) < 1e-15) dy = 0.0;
}

bool IsWindInlet(RansSolver::Side side, double d)
{
    d = std::fmod(std::fmod(d, 360.0) + 360.0, 360.0);
    bool north = (d == 0.0) || (d > 0.0 && d < 90.0) || (d > 270.0 && d < 360.0);
    bool east = (d == 90.0) || (d > 0.0 && d < 90.0) || (d > 90.0 && d < 180.0);
    bool south = (d == 180.0) || (d > 90.0 && d < 180.0) || (d > 180.0 && d < 270.0);
    bool west = (d == 270.0) || (d > 180.0 && d < 270.0) || (d > 270.0 && d < 360.0);
    switch(side)
    {
        case RansSolver::yMax: return north;
        case RansSolver::xMax: return east;
        case RansSolver::yMin: return south;
        case RansSolver::xMin: return west;
        default: return false;
    }
}

void ApplyWindBoundaries(RansSolver &solver, double ustar, double z0, double directionFrom)
{
    double dx, dy;
    WindDirectionVector(directionFrom, dx, dy);

    RansSolver::Boundary inlet = {RansSolver::Boundary::logProfileInlet, 0.0, dx, dy, 0.0,
                                  RansSolver::StateFunction()};
    inlet.ustar = ustar;
    inlet.z0 = z0;
    RansSolver::Boundary outlet = {RansSolver::Boundary::pressureOutlet, 0.0, 0.0, 0.0, 0.0,
                                   RansSolver::StateFunction()};
    RansSolver::Boundary ground = {RansSolver::Boundary::wall, 0.0, 0.0, 0.0, 0.0,
                                   RansSolver::StateFunction()};
    ground.z0 = z0;

    for(RansSolver::Side side : {RansSolver::xMin, RansSolver::xMax, RansSolver::yMin, RansSolver::yMax})
        solver.setBoundary(side, IsWindInlet(side, directionFrom) ? inlet : outlet);
    solver.setBoundary(RansSolver::zMax, outlet);
    solver.setBoundary(RansSolver::zMin, ground);
}

void InitializeLogProfile(RansSolver &solver, double ustar, double z0, double directionFrom)
{
    double dx, dy;
    WindDirectionVector(directionFrom, dx, dy);
    const RansGrid &g = solver.grid;
    solver.enableTurbulence(AblInletK(ustar), AblInletEpsilon(ustar, 1.0, z0));
    for(int c = 0; c < g.nCells(); c++)
    {
        double y = g.wallDistance[c];
        double speed = AblInletVelocity(ustar, y, z0);
        solver.p[c] = 0.0;
        solver.u[c] = speed * dx;
        solver.v[c] = speed * dy;
        solver.w[c] = 0.0;
        solver.k[c] = AblInletK(ustar);
        solver.epsilon[c] = AblInletEpsilon(ustar, y, z0);
        solver.nut[c] = solver.model.cmu * solver.k[c] * solver.k[c] / solver.epsilon[c];
    }
}

RansSolver WindSetup(const RansGrid &g, double speed, double inputHeight, double z0,
                     double directionFrom)
{
    double ustar = AblFrictionVelocity(speed, inputHeight, z0);
    // artificial compressibility: wave speed comparable to the flow speed
    double beta = std::max(1.0, 1.25 * speed * speed);
    RansSolver s(g, RANS_AIR_NU, beta);
    ApplyWindBoundaries(s, ustar, z0, directionFrom);
    InitializeLogProfile(s, ustar, z0, directionFrom);
    return s;
}
