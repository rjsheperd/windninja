/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Atmospheric boundary layer inlet and wall-function formulas
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

#ifndef RANS_ABL_H
#define RANS_ABL_H

/*
** The atmospheric boundary layer conditions WindNinja's OpenFOAM cases use,
** ported for parity (src/ninjafoam/11/{inletBC,wallBC}, data/ninjafoam/11/0,
** OpenFOAM-11 epsilonWallFunction).  The constants differ between formulas
** on purpose: the velocity inlet uses kappa 0.41 but the dissipation inlet
** 0.4187, and the inlet k uses the RNG Cmu.
*/

const double ABL_KAPPA = 0.41;              // velocity inlet, wall functions
const double ABL_KAPPA_EPSILON = 0.4187;    // dissipation-rate inlet
const double ABL_CMU = 0.085;               // RNG Cmu; also set on the wall functions
const double ABL_E = 9.8;                   // wall-function roughness constant

/* Friction velocity of a log profile with speed at inputHeight (m above the
** canopy) over roughness z0. */
double AblFrictionVelocity(double speed, double inputHeight, double z0);

/* Log-profile inlet values at height agl above the ground. */
double AblInletVelocity(double ustar, double agl, double z0);
double AblInletK(double ustar);
double AblInletEpsilon(double ustar, double agl, double z0);

/* Rough-wall turbulent viscosity at a wall face (nutkAtmRoughWallFunction):
** y is the wall-adjacent cell centre's distance from the wall. */
double RoughWallNut(double nu, double k, double y, double z0);

/* The y+ where the laminar and log laws meet, as OpenFOAM computes it. */
double WallYPlusLam();

/* epsilonWallFunction for a wall-adjacent cell: the fixed epsilon and the
** production G that replace the cell's own values. */
void EpsilonWallFunction(double nu, double nutw, double k, double y,
                         double magGradUw, double &epsilon, double &G);

#endif /* RANS_ABL_H */
