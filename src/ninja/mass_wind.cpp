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

#include "mass_wind.h"

#include <cmath>
#include <stdexcept>

#include "ninjaMathUtility.h"
#include "windProfile.h"

double BoundaryLayerHeight(const DomainAverageWind &wind)
{
    double direction = wrap0to360(wind.direction - wind.angleFromNorth);
    double inwindu = 0.0, inwindv = 0.0;
    wind_sd_to_uv(wind.speed, direction, &inwindu, &inwindv);
    double velocity = std::pow(inwindu * inwindu + inwindv * inwindv, 0.5);
    if(velocity == 0.0)
        return 1000.0;   // a null run's
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
    return 0.2 * uStar / f;
}

namespace {

// windProfile set up as domainAverageInitialization's, for neutral air
windProfile NeutralProfile(const DomainAverageWind &wind)
{
    windProfile profile;
    profile.profile_switch = windProfile::monin_obukov_similarity;
    profile.ObukovLength = 9999.;   // neutral: L -> infinity
    profile.ABL_height = BoundaryLayerHeight(wind);
    profile.Roughness = wind.roughness;
    profile.Rough_h = wind.roughH;
    profile.Rough_d = wind.roughD;
    profile.inputWindHeight = wind.inputHeight;
    return profile;
}

} // namespace

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

    // initializeWindFromProfile
    windProfile profile = NeutralProfile(wind);
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

void OutputWind(const MassGrid &grid, const std::vector<double> &u, const std::vector<double> &v,
                const std::vector<double> &w, const DomainAverageWind &wind, double outputHeight,
                std::vector<double> &speed, std::vector<double> &angle)
{
    const int layer = grid.ncols * grid.nrows;
    speed.assign(layer, 0.0);
    angle.assign(layer, 0.0);
    windProfile profile = NeutralProfile(wind);
    const double target = outputHeight + wind.roughH;
    for(int column = 0; column < layer; column++)
    {
        auto at = [&](int k) { return column + layer * k; };
        int k = 1;
        double h2 = 0, h1 = 0.0;
        while(h2 < target)
        {
            if(k >= grid.nlayers)
                throw std::runtime_error("OutputWind: the output height is above the mesh");
            h2 = grid.z[at(k)] - grid.z[at(0)];
            h1 = grid.z[at(k - 1)] - grid.z[at(0)];
            k++;
        }
        k--;
        double uu, vv;
        if(k <= 1)   // within the first layer: the profile
        {
            profile.inputWindHeight = h2 - wind.roughH;
            profile.AGL = target;
            profile.inputWindSpeed = u[at(k)];
            uu = profile.getWindSpeed();
            profile.inputWindSpeed = v[at(k)];
            vv = profile.getWindSpeed();
        }
        else   // linear between layers k - 1 and k
        {
            double slopeu = (u[at(k)] - u[at(k - 1)]) / (h2 - h1);
            double slopev = (v[at(k)] - v[at(k - 1)]) / (h2 - h1);
            uu = slopeu * target + u[at(k - 1)] - slopeu * h1;
            vv = slopev * target + v[at(k - 1)] - slopev * h1;
        }
        speed[column] = std::pow((uu * uu + vv * vv), 0.5);
        double a = (uu == 0.0 && vv == 0.0) ? 0.0 : -atan2(uu, -vv);
        if(a < 0)
            a += 2.0 * pi;
        angle[column] = 180.0 / pi * a;
    }
}
