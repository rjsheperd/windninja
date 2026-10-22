/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Wind over a plain DEM on the GPU (core-free)
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

#ifndef RANS_DEM_SOLVE_H
#define RANS_DEM_SOLVE_H

#include <functional>
#include <vector>

#include "gpu_context.h"
#include "rans_gpu_solve.h"
#include "rans_terrain.h"

/*
** DEM in, ground-layer wind out, on the GPU: the WebGPU momentum backend
** and the browser module (src/ninja/gpu/wasm) both run this, free of
** WindNinja core.
*/

struct DemWindProblem
{
    double speed = 0.0;           // m/s at inputHeight
    double inputHeight = 0.0;     // m above the canopy
    double direction = 0.0;       // degrees clockwise from the DEM's y axis,
                                  // direction the wind comes from
    double z0 = 0.01;             // m, roughness
    double meshResolution = 0.0;  // m; also the first layer's thickness
    int layers = 20;              // cells in the vertical
    double growth = 1.1;          // layer growth above the first layer
    bool volume = false;          // also return the 3-D fields
};

struct DemWind
{
    // ground-cell wind per DEM cell, in TerrainDem order (row 0 south)
    std::vector<double> u, v, k;
    double sampleHeight = 0.0;    // m above ground of the ground cells' centres
    GpuSolveReport report;
    // with DemWindProblem::volume: every cell's centre elevation and fields,
    // cell index col + nCols * (row + nRows * layer), row 0 south
    int layers = 0;
    std::vector<double> cellZ, volumeU, volumeV, volumeW, volumeK;
};

/* Throws RansDivergedError, gpu::Error, or what checkCancel throws. */
DemWind SolveDemOnGpu(gpu::Context &context, const TerrainDem &dem, const DemWindProblem &problem,
                      const GpuSolveOptions &options, const GpuSolveLog &log,
                      const std::function<void()> &checkCancel);

#endif /* RANS_DEM_SOLVE_H */
