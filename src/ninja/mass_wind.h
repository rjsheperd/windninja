/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Mass-consistent solver: domain-average wind, initial and output
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

#ifndef MASS_WIND_H
#define MASS_WIND_H

#include <vector>

#include "mass_fem.h"

/*
** WindNinja's domain-average initial wind (domainAverageInitialization,
** without diurnal winds: a neutral atmosphere), free of WindNinja core: in
** each DEM column, the Monin-Obukhov profile (windProfile, L = 9999) from
** the input speed at inputHeight above the canopy, up to the neutral
** boundary-layer height 0.2 u* / f, constant above it.  Uniform vegetation.
*/
struct DomainAverageWind
{
    double speed = 0.0;            // m/s at inputHeight
    double direction = 0.0;        // degrees from north the wind comes from (geographic)
    double inputHeight = 6.096;    // m above the canopy
    double roughness = 0.01;       // z0, m (grass 0.01, brush 0.43, trees 1.0)
    double roughH = 0.0;           // canopy height, m (grass 0, brush 2.3, trees 15.4)
    double roughD = 0.0;           // displacement height, m (grass 0, brush 1.8, trees 12.0)
    double latitude = -10000.0;    // degrees; outside [-90, 90]: f = 1e-4 (mid-latitude)
    double angleFromNorth = 0.0;   // degrees, the DEM grid's north offset (Elevation)
};

// The neutral boundary-layer height, m (initializeBoundaryLayer).
double BoundaryLayerHeight(const DomainAverageWind &wind);

// u0, v0, w0 per node of grid (MassGrid's order; the ground layer is the DEM).
void InitialWind(const MassGrid &grid, const DomainAverageWind &wind, std::vector<double> &u0,
                 std::vector<double> &v0, std::vector<double> &w0);

/*
** WindNinja's output wind (ninja::interp_uvw): per DEM cell, the wind at
** outputHeight above the canopy from the solved u, v, w; the profile
** (windProfile, as InitialWind) within the first layer, linear between
** layers above it.  speed is horizontal (m/s); angle is the direction the
** wind comes from, degrees clockwise from the grid's north (WindNinja's
** AngleGrid, as written to ASCII output).  Row 0 south, as the grid.
*/
void OutputWind(const MassGrid &grid, const std::vector<double> &u, const std::vector<double> &v,
                const std::vector<double> &w, const DomainAverageWind &wind, double outputHeight,
                std::vector<double> &speed, std::vector<double> &angle);

#endif /* MASS_WIND_H */
