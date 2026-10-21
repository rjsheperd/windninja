/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  CPU reference incompressible RANS solver (artificial compressibility)
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

#ifndef RANS_SOLVER_H
#define RANS_SOLVER_H

#include <functional>
#include <vector>

#include "rans_grid.h"

/*
** Steady incompressible flow on a RansGrid by artificial compressibility:
**
**   dp/dtau + beta div(u)               = 0
**   du/dtau + div(u u) + grad(p)        = nu lap(u) + source
**
** with kinematic pressure p.  Cell-centred finite volume: Rusanov flux on
** linearly reconstructed face states (second order), Green-Gauss gradients,
** corrected face gradients for the viscous terms, and explicit pseudo-time
** steps with a local time step.
*/
class RansSolver
{
public:
    enum Side { xMin, xMax, yMin, yMax, zMin, zMax };

    /* State function: fills p, u, v, w at a point. */
    typedef std::function<void(double x, double y, double z,
                               double &p, double &u, double &v, double &w)> StateFunction;

    struct Boundary
    {
        enum Type
        {
            wall,            // no slip, velocity = (u, v, w) (moving wall)
            slip,            // no normal velocity, no shear
            velocityInlet,   // velocity = (u, v, w), pressure extrapolated
            pressureOutlet,  // pressure = p, velocity extrapolated
            dirichlet        // everything from state (e.g. an exact solution)
        };
        Type type;
        double p, u, v, w;
        StateFunction state;   // for dirichlet; local coordinates
    };

    struct Residuals
    {
        double p, u, v, w;     // L2 norm over cells of the residual / volume
    };

    RansSolver(const RansGrid &grid, double nu, double beta);

    void setBoundary(Side side, const Boundary &bc);
    /* Volumetric source added to each equation (per unit volume). */
    void setSource(const StateFunction &source);
    void setInitial(const StateFunction &state);
    void setUniform(double p, double u, double v, double w);

    /* Take n pseudo-time steps; returns the residuals of the last step. */
    Residuals iterate(int n, double cfl);
    Residuals residuals();

    const RansGrid &grid;
    double nu, beta;
    std::vector<double> p, u, v, w;      // per cell

private:
    Boundary bc[6];
    StateFunction source;

    // per-cell gradients of p, u, v, w (x, y, z components)
    std::vector<double> gp[3], gu[3], gv[3], gw[3];
    // per-cell residual (flux out minus source) and inverse local time step
    std::vector<double> rp, ru, rv, rw, spectral;

    void computeGradients();
    void computeResiduals();
    Residuals norms() const;
};

#endif /* RANS_SOLVER_H */
