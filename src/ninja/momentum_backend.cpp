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

#include "momentum_backend.h"

#include <algorithm>

#include "WindNinjaInputs.h"

MomentumProblem MakeMomentumProblem(const WindNinjaInputs &input)
{
    // WindNinjaInputs already stores speeds in m/s and heights in m
    MomentumProblem p;
    p.dem = &input.dem;
    p.z0 = input.surface.Roughness.get_meanValue();
    p.rd = input.surface.Rough_d.get_meanValue();
    p.speed = input.inputSpeed;
    p.direction = input.inputDirection_proj;
    p.inputHeight = input.inputWindHeight;
    p.outputHeight = input.outputWindHeight;
    p.meshResolution = input.dem.get_cellSize();
    p.cellBudget = input.meshCount;
    p.iterations = input.nIterations;
    p.nThreads = input.numberCPUs;
    return p;
}

bool MomentumVolume::sample(double x, double y, double zp, double &uOut, double &vOut,
                            double &wOut, double &kOut) const
{
    // fractional column indices of the cell centres, clamped to the domain
    double fi = std::min(std::max((x - xll) / cellSize - 0.5, 0.0), double(nx - 1));
    double fj = std::min(std::max((y - yll) / cellSize - 0.5, 0.0), double(ny - 1));
    int i0 = std::min(int(fi), nx - 2 < 0 ? 0 : nx - 2), j0 = std::min(int(fj), ny - 2 < 0 ? 0 : ny - 2);
    int i1 = std::min(i0 + 1, nx - 1), j1 = std::min(j0 + 1, ny - 1);
    double ti = fi - i0, tj = fj - j0;
    const int ci[4] = {i0, i1, i0, i1}, cj[4] = {j0, j0, j1, j1};
    const double wt[4] = {(1 - ti) * (1 - tj), ti * (1 - tj), (1 - ti) * tj, ti * tj};
    uOut = vOut = wOut = kOut = 0.0;
    for(int n = 0; n < 4; n++)
    {
        auto cell = [&](int layer) { return ci[n] + nx * (cj[n] + ny * layer); };
        if(zp < z[cell(0)])
            return false;
        int m = 0;   // the layer at or below zp
        while(m + 1 < nz && z[cell(m + 1)] < zp)
            m++;
        double a = 0.0;   // weight of layer m + 1
        if(m + 1 < nz)
            a = (zp - z[cell(m)]) / (z[cell(m + 1)] - z[cell(m)]);
        int lo = cell(m), hi = cell(std::min(m + 1, nz - 1));
        uOut += wt[n] * ((1 - a) * u[lo] + a * u[hi]);
        vOut += wt[n] * ((1 - a) * v[lo] + a * v[hi]);
        wOut += wt[n] * ((1 - a) * w[lo] + a * w[hi]);
        kOut += wt[n] * ((1 - a) * k[lo] + a * k[hi]);
    }
    return true;
}
