/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  FAS geometric multigrid for the GPU RANS solver
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

#include "gpu_multigrid.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "gpu_multigrid_kernels.h"

namespace gpu {

namespace {

// layout of the WGSL MgParams
struct MgParamsBlock
{
    uint32_t fine[4], coarse[4], home[4], lean[4], start[4], w[4];
    float damping, pad[3];
};
static_assert(sizeof(MgParamsBlock) == 112, "must match the WGSL MgParams layout");

std::string source(const char *stage)
{
    return std::string(kernels::multigridPrelude) + stage;
}

// as RansMultigrid's CellMap and Leaning, from the kept node indices
void Tables(const std::vector<int> &idx, std::vector<int> &home, std::vector<int> &lean,
            std::vector<float> &w)
{
    int nFine = idx.back(), nCoarse = static_cast<int>(idx.size()) - 1;
    home.assign(nFine, 0);
    lean.assign(nFine, -1);
    w.assign(nFine, 0.0f);
    for(int c = 0; c < nCoarse; c++)
    {
        for(int i = idx[c]; i < idx[c + 1]; i++)
            home[i] = c;
        if(idx[c + 1] - idx[c] != 2)
            continue;
        if(c > 0) { lean[idx[c]] = c - 1; w[idx[c]] = 0.25f; }
        if(c + 1 < nCoarse) { lean[idx[c] + 1] = c + 1; w[idx[c] + 1] = 0.25f; }
    }
}

// the transfer tables of one level pair, packed for the kernels
struct TransferTables
{
    MgParamsBlock block = {};
    std::vector<int> maps;
    std::vector<float> weights;
};

TransferTables MakeTables(const Grid &fg, const Grid &cg, const RansMultigrid::Level &level)
{
    TransferTables t;
    uint32_t fn[3] = {uint32_t(fg.nx), uint32_t(fg.ny), uint32_t(fg.nz)};
    uint32_t cn[3] = {uint32_t(cg.nx), uint32_t(cg.ny), uint32_t(cg.nz)};
    std::vector<int> home[3], lean[3];
    std::vector<float> w[3];
    for(int d = 0; d < 3; d++)
    {
        const std::vector<int> &idx = level.nodes[d];
        if(idx.empty() || uint32_t(idx.back()) != fn[d] || idx.size() != cn[d] + 1)
            throw Error("gpu::RansMultigrid: coarse nodes don't match the grids");
        Tables(idx, home[d], lean[d], w[d]);
        t.block.fine[d] = fn[d];
        t.block.coarse[d] = cn[d];
    }
    t.block.fine[3] = uint32_t(fg.nCells);
    t.block.coarse[3] = uint32_t(cg.nCells);
    auto append = [&](const std::vector<int> &a, uint32_t &offset)
    {
        offset = uint32_t(t.maps.size());
        t.maps.insert(t.maps.end(), a.begin(), a.end());
    };
    for(int d = 0; d < 3; d++) append(home[d], t.block.home[d]);
    for(int d = 0; d < 3; d++) append(lean[d], t.block.lean[d]);
    for(int d = 0; d < 3; d++) append(level.nodes[d], t.block.start[d]);
    for(int d = 0; d < 3; d++)
    {
        t.block.w[d] = uint32_t(t.weights.size());
        t.weights.insert(t.weights.end(), w[d].begin(), w[d].end());
    }
    return t;
}

} // namespace

struct RansMultigrid::Transfer
{
    Transfer(Context &context, const TransferTables &t, std::size_t coarseCells)
        : block(t.block), params(context, sizeof(MgParamsBlock), Buffer::Uniform),
          maps(context, t.maps.size() * sizeof(int)),
          weights(context, t.weights.size() * sizeof(float)),
          start(context, 4 * coarseCells * sizeof(float))
    {
        maps.upload(t.maps);
        weights.upload(t.weights);
    }

    MgParamsBlock block;
    Buffer params, maps, weights, start;
};

RansMultigrid::RansMultigrid(RansSolver &fine, const std::vector<Level> &coarse)
    : context_(fine.context),
      restrict_(fine.context, source(kernels::restrictState).c_str()),
      forcing_(fine.context, source(kernels::forcing).c_str()),
      prolong_(fine.context, source(kernels::prolong).c_str())
{
    solvers_.push_back(&fine);
    for(const Level &level : coarse)
    {
        owned_.emplace_back(new RansSolver(context_, level.grid, fine.nu(), fine.beta()));
        RansSolver &c = *owned_.back();
        for(int side = 0; side < 6; side++)
            c.setBoundary(RansSolver::Side(side), fine.boundary(RansSolver::Side(side)));
        c.setTurbulence(fine.turbulent(), fine.model());
        c.setFrozenTurbulence(fine.turbulent());
        transfers_.emplace_back(new Transfer(context_, MakeTables(solvers_.back()->grid, c.grid, level),
                                             c.grid.nCells));
        solvers_.push_back(&c);
    }
}

RansMultigrid::~RansMultigrid() = default;

void RansMultigrid::reset()
{
    for(std::size_t l = 1; l < solvers_.size(); l++)
    {
        std::vector<float> zero(4 * solvers_[l]->grid.nCells, 0.0f);
        solvers_[l]->forcing.upload(zero);
    }
}

void RansMultigrid::dispatch(Kernel &kernel, const std::vector<const Buffer *> &bindings,
                             std::size_t threads)
{
    const std::size_t maxGroupsX = 65535;
    std::size_t groups = (threads + 255) / 256;
    std::size_t gx = std::min(groups, maxGroupsX);
    std::size_t gy = (groups + gx - 1) / gx;
    kernel.dispatch(bindings, static_cast<uint32_t>(gx), static_cast<uint32_t>(gy));
}

void RansMultigrid::cycle(double cfl, int preSmooth, int postSmooth, int coarsest)
{
    if(correctionDamping != damping_)
    {
        damping_ = correctionDamping;
        for(auto &t : transfers_)
        {
            t->block.damping = static_cast<float>(damping_);
            t->params.upload(&t->block, sizeof t->block);
        }
    }
    vcycle(0, cfl, preSmooth, postSmooth, coarsest);
}

void RansMultigrid::vcycle(int level, double cfl, int preSmooth, int postSmooth, int coarsest)
{
    RansSolver &f = *solvers_[level];
    if(level + 1 == levels())
    {
        f.steps(coarsest, cfl);
        return;
    }
    f.steps(preSmooth, cfl);
    // fine residual (with this level's own forcing) at the smoothed state
    f.computeGradients();
    f.computeResiduals();

    RansSolver &c = *solvers_[level + 1];
    Transfer &t = *transfers_[level];
    dispatch(restrict_, {&t.params, &f.grid.cells, &f.fields.buffer, &c.fields.buffer, &t.start,
                         &t.maps},
             c.grid.nCells);
    // the coarse residual of the restricted state (with the previous
    // forcing, which the forcing kernel takes back out)
    c.computeGradients();
    c.computeResiduals();
    dispatch(forcing_, {&t.params, &f.residuals, &c.residuals, &c.forcing, &t.maps},
             c.grid.nCells);

    vcycle(level + 1, cfl, preSmooth, postSmooth, coarsest);

    dispatch(prolong_, {&t.params, &c.fields.buffer, &t.start, &f.fields.buffer, &t.maps,
                        &t.weights},
             f.grid.nCells);
    f.steps(std::max(postSmooth, 1), cfl);
}

} // namespace gpu
