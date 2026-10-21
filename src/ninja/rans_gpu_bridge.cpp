/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Bridge from the CPU RANS solver's types to the GPU port's
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

#include "rans_gpu_bridge.h"

#include <stdexcept>

gpu::GridView GpuGridView(const RansGrid &g)
{
    gpu::GridView view;
    view.nx = g.nx;
    view.ny = g.ny;
    view.nz = g.nz;
    view.volume = g.volume.data();
    view.cx = g.cx.data();
    view.cy = g.cy.data();
    view.cz = g.cz.data();
    view.wallDistance = g.wallDistance.data();
    const std::vector<double> *s[3][3] = {{&g.iSx, &g.iSy, &g.iSz},
                                          {&g.jSx, &g.jSy, &g.jSz},
                                          {&g.kSx, &g.kSy, &g.kSz}};
    const std::vector<double> *f[3][3] = {{&g.iFx, &g.iFy, &g.iFz},
                                          {&g.jFx, &g.jFy, &g.jFz},
                                          {&g.kFx, &g.kFy, &g.kFz}};
    for(int d = 0; d < 3; d++)
        for(int m = 0; m < 3; m++)
        {
            view.s[d][m] = s[d][m]->data();
            view.f[d][m] = f[d][m]->data();
        }
    view.agl[0] = g.iAgl.data();
    view.agl[1] = g.jAgl.data();
    return view;
}

gpu::RansSolver::TurbulenceModel GpuTurbulenceModel(const RansTurbulenceModel &m)
{
    gpu::RansSolver::TurbulenceModel t;
    t.cmu = m.cmu;
    t.c1 = m.c1;
    t.c2 = m.c2;
    t.sigmaK = m.sigmaK;
    t.sigmaEpsilon = m.sigmaEpsilon;
    t.eta0 = m.eta0;
    t.beta = m.beta;
    return t;
}

static gpu::RansSolver::Boundary GpuBoundary(const RansSolver::Boundary &b)
{
    typedef gpu::RansSolver::Boundary G;
    G g;
    switch(b.type)
    {
        case RansSolver::Boundary::wall: g.type = G::wall; break;
        case RansSolver::Boundary::slip: g.type = G::slip; break;
        case RansSolver::Boundary::velocityInlet: g.type = G::velocityInlet; break;
        case RansSolver::Boundary::pressureOutlet: g.type = G::pressureOutlet; break;
        case RansSolver::Boundary::logProfileInlet: g.type = G::logProfileInlet; break;
        default:
            throw std::invalid_argument("RANS boundary kind has no GPU version");
    }
    g.p = b.p;
    g.u = b.u;
    g.v = b.v;
    g.w = b.w;
    g.ustar = b.ustar;
    g.z0 = b.z0;
    g.k = b.k;
    g.epsilon = b.epsilon;
    return g;
}

void CopyToGpu(const RansSolver &cpu, gpu::RansSolver &gpu)
{
    for(int side = 0; side < 6; side++)
        gpu.setBoundary(gpu::RansSolver::Side(side),
                        GpuBoundary(cpu.boundary(RansSolver::Side(side))));
    gpu.setTurbulence(cpu.turbulent, GpuTurbulenceModel(cpu.model));
    gpu.fields.upload(gpu::fieldP, cpu.p);
    gpu.fields.upload(gpu::fieldU, cpu.u);
    gpu.fields.upload(gpu::fieldV, cpu.v);
    gpu.fields.upload(gpu::fieldW, cpu.w);
    if(cpu.turbulent)
    {
        gpu.fields.upload(gpu::fieldK, cpu.k);
        gpu.fields.upload(gpu::fieldEpsilon, cpu.epsilon);
        gpu.fields.upload(gpu::fieldNut, cpu.nut);
    }
}

std::vector<gpu::RansMultigrid::Level> GpuMultigridLevels(const RansMultigrid &mg)
{
    std::vector<gpu::RansMultigrid::Level> levels;
    for(int l = 1; l < mg.levels(); l++)
    {
        gpu::RansMultigrid::Level level;
        level.grid = GpuGridView(mg.grid(l));
        for(int d = 0; d < 3; d++)
            level.nodes[d] = mg.coarseNodes(l - 1, d);
        levels.push_back(level);
    }
    return levels;
}
