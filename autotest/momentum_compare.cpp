/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Compare momentum solver output against golden reference grids
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

#include "momentum_compare.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "gdal_priv.h"

static const double MPH_TO_MPS = 0.44704;

/*
** Call f(a, b) for every cell valid in both grids and return how many there
** were.  Grids must be the same size.
*/
template <typename F>
static int ForEachValidPair(const CompareGrid &a, const CompareGrid &b, F f)
{
    if(a.nCols != b.nCols || a.nRows != b.nRows ||
       a.values.size() != b.values.size())
    {
        throw std::runtime_error("compared grids differ in size");
    }
    int n = 0;
    for(size_t i = 0; i < a.values.size(); i++)
    {
        double x = a.values[i];
        double y = b.values[i];
        if(x == a.noData || y == b.noData || std::isnan(x) || std::isnan(y))
            continue;
        f(x, y);
        n++;
    }
    if(n == 0)
        throw std::runtime_error("compared grids have no valid cells in common");
    return n;
}

double CircularDifference(double a, double b)
{
    return std::fabs(std::fmod(std::fmod(a - b, 360.0) + 540.0, 360.0) - 180.0);
}

double Rmse(const CompareGrid &a, const CompareGrid &b)
{
    double sum = 0.0;
    int n = ForEachValidPair(a, b, [&](double x, double y)
            { sum += (x - y) * (x - y); });
    return std::sqrt(sum / n);
}

double MaxAbsDifference(const CompareGrid &a, const CompareGrid &b)
{
    double worst = 0.0;
    ForEachValidPair(a, b, [&](double x, double y)
            { worst = std::max(worst, std::fabs(x - y)); });
    return worst;
}

double MeanCircularDifference(const CompareGrid &a, const CompareGrid &b)
{
    double sum = 0.0;
    int n = ForEachValidPair(a, b, [&](double x, double y)
            { sum += CircularDifference(x, y); });
    return sum / n;
}

double RelativeRmse(const CompareGrid &a, const CompareGrid &b)
{
    double sumDiff = 0.0;
    double sumRef = 0.0;
    ForEachValidPair(a, b, [&](double x, double y)
            { sumDiff += (x - y) * (x - y); sumRef += y * y; });
    if(sumRef == 0.0)
        throw std::runtime_error("reference grid is all zero; relative RMSE undefined");
    return std::sqrt(sumDiff / sumRef);
}

CompareGrid ReadCompareGrid(const std::string &path)
{
    GDALAllRegister();
    GDALDataset *ds = (GDALDataset *)GDALOpen(path.c_str(), GA_ReadOnly);
    if(ds == NULL)
        throw std::runtime_error("cannot open grid: " + path);
    GDALRasterBand *band = ds->GetRasterBand(1);
    CompareGrid g;
    g.nCols = ds->GetRasterXSize();
    g.nRows = ds->GetRasterYSize();
    double gt[6];
    if(ds->GetGeoTransform(gt) == CE_None)
    {
        g.cellSize = gt[1];
        g.xll = gt[0];
        g.yll = gt[3] + g.nRows * gt[5];
    }
    int hasNoData = FALSE;
    g.noData = band->GetNoDataValue(&hasNoData);
    if(!hasNoData)
        g.noData = NAN;
    g.values.resize((size_t)g.nCols * g.nRows);
    CPLErr err = band->RasterIO(GF_Read, 0, 0, g.nCols, g.nRows, &g.values[0],
                                g.nCols, g.nRows, GDT_Float64, 0, 0);
    GDALClose(ds);
    if(err != CE_None)
        throw std::runtime_error("cannot read grid: " + path);
    return g;
}

MomentumMetrics CompareMomentumCases(const std::string &a, const std::string &b)
{
    CompareGrid velA = ReadCompareGrid(a + "/vel.asc");
    CompareGrid velB = ReadCompareGrid(b + "/vel.asc");
    for(size_t i = 0; i < velA.values.size(); i++)
        if(velA.values[i] != velA.noData)
            velA.values[i] *= MPH_TO_MPS;
    for(size_t i = 0; i < velB.values.size(); i++)
        if(velB.values[i] != velB.noData)
            velB.values[i] *= MPH_TO_MPS;

    MomentumMetrics m;
    m.speedRmse = Rmse(velA, velB);
    m.speedMaxAbs = MaxAbsDifference(velA, velB);
    m.nCells = ForEachValidPair(velA, velB, [](double, double) {});
    m.directionMeanAbs = MeanCircularDifference(ReadCompareGrid(a + "/ang.asc"),
                                                ReadCompareGrid(b + "/ang.asc"));
    m.kRelativeRmse = RelativeRmse(ReadCompareGrid(a + "/k.asc"),
                                   ReadCompareGrid(b + "/k.asc"));
    return m;
}

CompareGrid ResampleTo(const CompareGrid &src, const CompareGrid &target)
{
    if(src.cellSize <= 0.0 || target.cellSize <= 0.0)
        throw std::runtime_error("resampling needs georeferenced grids");
    CompareGrid out = target;
    out.noData = NAN;
    auto valid = [&](double v) { return v != src.noData && !std::isnan(v); };
    for(int r = 0; r < target.nRows; r++)
        for(int c = 0; c < target.nCols; c++)
        {
            double x = target.xll + (c + 0.5) * target.cellSize;
            double y = target.yll + (target.nRows - r - 0.5) * target.cellSize;
            // fractional column / row of src cell centres (row 0 north)
            double fc = (x - src.xll) / src.cellSize - 0.5;
            double fr = (src.yll + src.nRows * src.cellSize - y) / src.cellSize - 0.5;
            double v = NAN;
            if(fc >= 0.0 && fr >= 0.0 && fc <= src.nCols - 1 && fr <= src.nRows - 1)
            {
                int c0 = std::min((int)fc, src.nCols - 2), r0 = std::min((int)fr, src.nRows - 2);
                double tc = fc - c0, tr = fr - r0;
                double q[4] = {src.values[(size_t)r0 * src.nCols + c0],
                               src.values[(size_t)r0 * src.nCols + c0 + 1],
                               src.values[(size_t)(r0 + 1) * src.nCols + c0],
                               src.values[(size_t)(r0 + 1) * src.nCols + c0 + 1]};
                if(valid(q[0]) && valid(q[1]) && valid(q[2]) && valid(q[3]))
                    v = (1 - tc) * (1 - tr) * q[0] + tc * (1 - tr) * q[1] +
                        (1 - tc) * tr * q[2] + tc * tr * q[3];
            }
            out.values[(size_t)r * target.nCols + c] = v;
        }
    return out;
}

// name.asc, or name.tif (the compressed fine-mesh references)
static CompareGrid ReadCaseGrid(const std::string &dir, const std::string &name)
{
    std::string asc = dir + "/" + name + ".asc";
    VSIStatBufL st;
    if(VSIStatL(asc.c_str(), &st) == 0)
        return ReadCompareGrid(asc);
    return ReadCompareGrid(dir + "/" + name + ".tif");
}

MomentumMetrics CompareMomentumCasesResampled(const std::string &a, const std::string &b, int rim)
{
    CompareGrid velA = ReadCaseGrid(a, "vel"), angA = ReadCaseGrid(a, "ang");
    CompareGrid velB = ReadCaseGrid(b, "vel"), angB = ReadCaseGrid(b, "ang");
    // the reference's wind as components (direction the wind comes from)
    CompareGrid uB = velB, vB = velB;
    for(size_t i = 0; i < velB.values.size(); i++)
    {
        double s = velB.values[i], t = angB.values[i] * M_PI / 180.0;
        bool ok = s != velB.noData && angB.values[i] != angB.noData;
        uB.values[i] = ok ? -s * std::sin(t) : velB.noData;
        vB.values[i] = ok ? -s * std::cos(t) : velB.noData;
    }
    CompareGrid u = ResampleTo(uB, velA), v = ResampleTo(vB, velA);
    CompareGrid speedB = u, dirB = u;
    for(size_t i = 0; i < u.values.size(); i++)
    {
        int r = (int)(i / velA.nCols), c = (int)(i % velA.nCols);
        bool inRim = r < rim || c < rim || r >= velA.nRows - rim || c >= velA.nCols - rim;
        if(inRim || std::isnan(u.values[i]))
        {
            speedB.values[i] = dirB.values[i] = NAN;
            continue;
        }
        speedB.values[i] = std::hypot(u.values[i], v.values[i]);
        double d = std::atan2(-u.values[i], -v.values[i]) * 180.0 / M_PI;
        dirB.values[i] = d < 0.0 ? d + 360.0 : d;
    }
    for(size_t i = 0; i < velA.values.size(); i++)
        if(velA.values[i] != velA.noData)
            velA.values[i] *= MPH_TO_MPS;
    for(size_t i = 0; i < speedB.values.size(); i++)
        speedB.values[i] *= MPH_TO_MPS;

    MomentumMetrics m;
    m.speedRmse = Rmse(velA, speedB);
    m.speedMaxAbs = MaxAbsDifference(velA, speedB);
    m.nCells = ForEachValidPair(velA, speedB, [](double, double) {});
    m.directionMeanAbs = MeanCircularDifference(angA, dirB);
    m.kRelativeRmse = NAN;
    return m;
}
