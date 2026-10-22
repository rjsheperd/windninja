/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  The mass solver from a DEM, core-free (mass_dem), for comparing with the CLI
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

/*
** usage: mass_dem_solve DEM SPEED DIRECTION INPUT_HEIGHT OUTPUT_HEIGHT VEGETATION OUT_PREFIX
**                       [gpu|jacobi]
**
** Solves the domain-average mass problem over DEM with mass_dem (the
** core-free path the browser uses): SPEED m/s from DIRECTION degrees at
** INPUT_HEIGHT m above the canopy, VEGETATION grass, brush or trees, the
** wind at OUTPUT_HEIGHT m, on the CPU (SSOR, as WindNinja; "jacobi": the
** CPU with the Jacobi preconditioner) or (with "gpu", in a WebGPU build) the
** GPU.  Takes the DEM's angle from north and latitude as WindNinja does, and
** writes OUT_PREFIX_vel.asc (m/s, 2 decimals) and OUT_PREFIX_ang.asc
** (degrees, 0 decimals) as WindNinja's ASCII output does, so the two can be
** compared value by value (scripts/check_mass_dem.sh).  The DEM must be at
** the mesh resolution.
*/

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "gdal_priv.h"

#include "ascii_grid.h"
#include "gdal_util.h"
#include "mass_dem.h"
#ifdef WITH_WEBGPU
#include "mass_gpu_solve.h"
#endif

int main(int argc, char *argv[])
{
    if(argc != 8 && argc != 9)
    {
        std::fprintf(stderr, "usage: %s DEM SPEED DIRECTION INPUT_HEIGHT OUTPUT_HEIGHT "
                             "grass|brush|trees OUT_PREFIX [gpu|jacobi]\n", argv[0]);
        return 2;
    }
    try
    {
        GDALAllRegister();
        GDALDataset *ds = (GDALDataset *)GDALOpen(argv[1], GA_ReadOnly);
        if(ds == nullptr)
            throw std::runtime_error(std::string("cannot open ") + argv[1]);
        double gt[6];
        ds->GetGeoTransform(gt);
        TerrainDem dem;
        dem.nCols = ds->GetRasterXSize();
        dem.nRows = ds->GetRasterYSize();
        dem.cellSize = gt[1];
        dem.xll = gt[0];
        dem.yll = gt[3] + gt[5] * dem.nRows;
        std::vector<double> rows((size_t)dem.nCols * dem.nRows);
        if(ds->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, dem.nCols, dem.nRows, rows.data(),
                                          dem.nCols, dem.nRows, GDT_Float64, 0, 0) != CE_None)
            throw std::runtime_error("cannot read the DEM");
        dem.z.resize(rows.size());
        for(int r = 0; r < dem.nRows; r++)   // GDAL's row 0 is north; TerrainDem's south
            for(int c = 0; c < dem.nCols; c++)
                dem.z[(size_t)r * dem.nCols + c] = rows[(size_t)(dem.nRows - 1 - r) * dem.nCols + c];

        MassDemProblem problem;
        problem.wind.speed = std::atof(argv[2]);
        problem.wind.direction = std::atof(argv[3]);
        problem.wind.inputHeight = std::atof(argv[4]);
        problem.outputHeight = std::atof(argv[5]);
        std::string veg = argv[6];
        if(veg == "grass") { problem.wind.roughness = 0.01; problem.wind.roughH = 0.0; problem.wind.roughD = 0.0; }
        else if(veg == "brush") { problem.wind.roughness = 0.43; problem.wind.roughH = 2.3; problem.wind.roughD = 1.8; }
        else if(veg == "trees") { problem.wind.roughness = 1.0; problem.wind.roughH = 15.4; problem.wind.roughD = 12.0; }
        else throw std::runtime_error("vegetation must be grass, brush or trees");
        double angle = 0.0, longitude = 0.0, latitude = 0.0;
        GDALCalculateAngleFromNorth(ds, angle);   // as readInputFile
        if(GDALGetCenter(ds, &longitude, &latitude))   // as ninja::set_position
            problem.wind.latitude = latitude;
        problem.wind.angleFromNorth = angle;
        std::string prj = ds->GetProjectionRef();
        GDALClose((GDALDatasetH)ds);

        MassWindSolver solver = SolveMassWindOnCpu;
        if(argc == 9 && std::string(argv[8]) == "gpu")
        {
#ifdef WITH_WEBGPU
            solver = [](const MassGrid &g, const double *u0, const double *v0, const double *w0,
                        const double *alphaV, double alphaH, const MassSolveOptions &o, double *u,
                        double *v, double *w) {
                return SolveMassWindOnGpu(g, u0, v0, w0, alphaV, alphaH, o, u, v, w);
            };
#else
            throw std::runtime_error("not a WebGPU build");
#endif
        }
        MassSolveOptions options;
        options.jacobi = argc == 9 && std::string(argv[8]) == "jacobi";
        MassDemWind out = SolveMassDem(dem, problem, options, solver);
        std::printf("mass_dem_solve: %d iterations, residual %g, angle from north %.6f, latitude %.6f\n",
                    out.solve.iterations, out.solve.residual, angle, problem.wind.latitude);

        AsciiGrid<double> vel, ang;
        vel.set_headerData(dem.nCols, dem.nRows, dem.xll, dem.yll, dem.cellSize, -9999.0, 0.0, prj);
        ang.set_headerData(dem.nCols, dem.nRows, dem.xll, dem.yll, dem.cellSize, -9999.0, 0.0, prj);
        for(int r = 0; r < dem.nRows; r++)
            for(int c = 0; c < dem.nCols; c++)
            {
                vel(r, c) = out.speed[(size_t)r * dem.nCols + c];
                ang(r, c) = out.angle[(size_t)r * dem.nCols + c];
            }
        vel.write_Grid(std::string(argv[7]) + "_vel.asc", 2);
        ang.write_Grid(std::string(argv[7]) + "_ang.asc", 0);
    }
    catch(const std::exception &e)
    {
        std::fprintf(stderr, "mass_dem_solve: %s\n", e.what());
        return 1;
    }
    return 0;
}
