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

#include "rans_abl.h"

#include <algorithm>
#include <cmath>

double AblFrictionVelocity(double speed, double inputHeight, double z0)
{
    return speed * ABL_KAPPA / std::log(inputHeight / z0);
}

double AblInletVelocity(double ustar, double agl, double z0)
{
    return ustar / ABL_KAPPA * std::log(agl / z0);
}

double AblInletK(double ustar)
{
    return ustar * ustar / std::sqrt(ABL_CMU);
}

double AblInletEpsilon(double ustar, double agl, double z0)
{
    return ustar * ustar * ustar / (ABL_KAPPA_EPSILON * (agl + z0));
}

double RoughWallNut(double nu, double k, double y, double z0)
{
    double yPlus = std::pow(ABL_CMU, 0.25) * std::sqrt(k) * y / nu;
    double eDash = (y + z0) / z0;
    return nu * (yPlus * ABL_KAPPA / std::log(std::max(eDash, 1.0 + 1e-4)) - 1.0);
}

double WallYPlusLam()
{
    double ypl = 11.0;
    for(int i = 0; i < 10; i++)
        ypl = std::log(std::max(ABL_E * ypl, 1.0)) / ABL_KAPPA;
    return ypl;
}

void EpsilonWallFunction(double nu, double nutw, double k, double y,
                         double magGradUw, double &epsilon, double &G)
{
    double cmu25 = std::pow(ABL_CMU, 0.25);
    double yPlus = cmu25 * y * std::sqrt(k) / nu;
    if(yPlus > WallYPlusLam())
    {
        epsilon = std::pow(ABL_CMU, 0.75) * std::pow(k, 1.5) / (ABL_KAPPA * y);
        G = (nutw + nu) * magGradUw * cmu25 * std::sqrt(k) / (ABL_KAPPA * y);
    }
    else
    {
        epsilon = 2.0 * k * nu / (y * y);
        G = 0.0;
    }
}
