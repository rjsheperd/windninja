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

#ifndef RANS_WIND_H
#define RANS_WIND_H

/*
** Unit vector (x, y) the wind blows toward, for a wind coming from
** directionFrom degrees clockwise from the grid's +y axis.
*/
void WindDirectionVector(double directionFrom, double &dx, double &dy);

#include "rans_solver.h"

/*
** Whether the wind comes in through a side, with NinjaFoam::SetInlets' rules:
** one inlet side on the axes, two in between.  Never true for the top or
** ground.
*/
bool IsWindInlet(RansSolver::Side side, double directionFrom);

/*
** Set the boundaries of a terrain-following wind run the way WindNinja's
** OpenFOAM cases do: sides the wind comes from are log-profile inlets (as
** NinjaFoam::SetInlets), the other sides and the top are p = 0 outlets, and
** the ground is a rough wall with roughness z0.
*/
void ApplyWindBoundaries(RansSolver &solver, double ustar, double z0, double directionFrom);

/*
** Start from the log law by wall distance along the wind direction (as
** applyInit), with k and epsilon from the inlet formulas.  Enables
** turbulence on the solver.
*/
void InitializeLogProfile(RansSolver &solver, double ustar, double z0, double directionFrom);

#endif /* RANS_WIND_H */
