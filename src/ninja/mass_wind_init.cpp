/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Mass-consistent solver: domain-average initial wind
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

#include "mass_wind_init.h"

#include <cmath>

#include "ninjaMathUtility.h"
#include "windProfile.h"

void InitialWind(const MassGrid &grid, const DomainAverageWind &wind, std::vector<double> &u0,
                 std::vector<double> &v0, std::vector<double> &w0)
{
    const int n = grid.numNodes();
    const int layer = grid.ncols * grid.nrows;
    u0.assign(n, 0.0);
    v0.assign(n, 0.0);
    w0.assign(n, 0.0);

    // setInitializationGrids: projected direction, u and v of the input wind
    double direction = wrap0to360(wind.direction - wind.angleFromNorth);
    double inwindu = 0.0, inwindv = 0.0;
    wind_sd_to_uv(wind.speed, direction, &inwindu, &inwindv);

    // initializeBoundaryLayer, neutral: u*, the boundary-layer height
    double velocity = std::pow(inwindu * inwindu + inwindv * inwindv, 0.5);
    double blHeight = 1000.0;   // a null run's
    if(velocity != 0.0)
    {
        double f;
        if(wind.latitude <= 90.0 && wind.latitude >= -90.0)
        {
            f = (1.4544e-4) * sin(pi / 180 * wind.latitude);   // 2 omega sin(latitude)
            if(f < 0)
                f = -f;
        }
        else
            f = 1e-4;
        if(f == 0.0)
            f = 1e-8;
        double uStar = velocity * 0.4 /
                       (log((wind.inputHeight + wind.roughH - wind.roughD) / wind.roughness));
        blHeight = 0.2 * uStar / f;
    }

    // initializeWindFromProfile
    windProfile profile;
    profile.profile_switch = windProfile::monin_obukov_similarity;
    profile.ObukovLength = 9999.;   // neutral: L -> infinity
    profile.ABL_height = blHeight;
    profile.Roughness = wind.roughness;
    profile.Rough_h = wind.roughH;
    profile.Rough_d = wind.roughD;
    profile.inputWindHeight = wind.inputHeight;
    for(int column = 0; column < layer; column++)
    {
        double ground = grid.z[column];
        for(int k = 0; k < grid.nlayers; k++)
        {
            int node = column + layer * k;
            profile.AGL = grid.z[node] - ground;   // height above the ground
            profile.inputWindSpeed = inwindu;
            u0[node] += profile.getWindSpeed();
            profile.inputWindSpeed = inwindv;
            v0[node] += profile.getWindSpeed();
            profile.inputWindSpeed = 0.0;
            w0[node] += profile.getWindSpeed();
        }
    }
}
