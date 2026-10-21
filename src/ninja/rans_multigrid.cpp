/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  FAS geometric multigrid for the CPU RANS solver
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

#include "rans_multigrid.h"

#include <algorithm>

/* Fine node indices kept on the coarse grid: every second, plus the last. */
static std::vector<int> CoarseNodes(int n)
{
    std::vector<int> idx;
    for(int i = 0; i <= n; i += 2)
        idx.push_back(i);
    if(idx.back() != n)
        idx.push_back(n);
    return idx;
}

/* For each fine cell index along a direction, the coarse cell containing it. */
static std::vector<int> CellMap(const std::vector<int> &idx)
{
    std::vector<int> map(idx.back());
    for(size_t c = 0; c + 1 < idx.size(); c++)
        for(int i = idx[c]; i < idx[c + 1]; i++)
            map[i] = (int)c;
    return map;
}

/*
** Cell-centred linear prolongation along one direction: a fine cell in a
** coarse cell split in two takes 3/4 of its parent and 1/4 of the coarse
** neighbour on its side; at a domain edge, or in an unsplit coarse cell, it
** takes the parent only.
*/
static void Leaning(const std::vector<int> &idx, std::vector<int> &nb, std::vector<double> &w)
{
    int nFine = idx.back(), nCoarse = (int)idx.size() - 1;
    nb.assign(nFine, -1);
    w.assign(nFine, 0.0);
    for(int c = 0; c < nCoarse; c++)
    {
        if(idx[c + 1] - idx[c] != 2)
            continue;
        if(c > 0) { nb[idx[c]] = c - 1; w[idx[c]] = 0.25; }
        if(c + 1 < nCoarse) { nb[idx[c] + 1] = c + 1; w[idx[c] + 1] = 0.25; }
    }
}

RansMultigrid::RansMultigrid(RansSolver &fine, int maxLevels, int minCells)
{
    solvers.push_back(&fine);
    while((int)solvers.size() < maxLevels)
    {
        const RansGrid &g = solvers.back()->grid;
        std::vector<int> ix = CoarseNodes(g.nx), iy = CoarseNodes(g.ny), iz = CoarseNodes(g.nz);
        int nx = (int)ix.size() - 1, ny = (int)iy.size() - 1, nz = (int)iz.size() - 1;
        if(nx * ny * nz < minCells || (nx == g.nx && ny == g.ny && nz == g.nz))
            break;

        std::vector<double> x, y, z;
        for(int k : iz)
            for(int j : iy)
                for(int i : ix)
                {
                    int n = g.node(i, j, k);
                    x.push_back(g.x[n]);
                    y.push_back(g.y[n]);
                    z.push_back(g.z[n]);
                }
        grids.emplace_back(new RansGrid(RansGrid::FromNodes(nx, ny, nz, x, y, z)));
        nodeIdx.push_back({ix, iy, iz});
        const RansGrid &cg = *grids.back();

        std::vector<int> mx = CellMap(ix), my = CellMap(iy), mz = CellMap(iz);
        std::vector<int> map(g.nCells());
        for(int k = 0; k < g.nz; k++)
            for(int j = 0; j < g.ny; j++)
                for(int i = 0; i < g.nx; i++)
                    map[g.cell(i, j, k)] = cg.cell(mx[i], my[j], mz[k]);
        toCoarse.push_back(map);
        Lean lx, ly, lz;
        Leaning(ix, lx.nb, lx.w);
        Leaning(iy, ly.nb, ly.w);
        Leaning(iz, lz.nb, lz.w);
        leanX.push_back(lx);
        leanY.push_back(ly);
        leanZ.push_back(lz);

        const RansSolver &f = *solvers.front();
        owned.emplace_back(new RansSolver(cg, f.nu, f.beta));
        RansSolver &c = *owned.back();
        c.threads = f.threads;
        for(int side = 0; side < 6; side++)
            c.bc[side] = f.bc[side];
        c.source = f.source;
        if(f.turbulent)
        {
            c.enableTurbulence(1.0, 1.0, f.model);
            c.frozenTurbulence = true;
        }
        size_t n = cg.nCells();
        for(std::vector<double> *a : {&c.forceP, &c.forceU, &c.forceV, &c.forceW})
            a->assign(n, 0.0);
        q0.push_back(std::vector<double>(4 * n, 0.0));
        solvers.push_back(&c);
    }
}

/* Volume-weighted restriction of the state from level to level + 1. */
void RansMultigrid::restrictState(int level)
{
    RansSolver &f = *solvers[level];
    RansSolver &c = *solvers[level + 1];
    const std::vector<int> &map = toCoarse[level];
    size_t n = c.grid.nCells();
    std::vector<double> vol(n, 0.0);
    std::fill(c.p.begin(), c.p.end(), 0.0);
    std::fill(c.u.begin(), c.u.end(), 0.0);
    std::fill(c.v.begin(), c.v.end(), 0.0);
    std::fill(c.w.begin(), c.w.end(), 0.0);
    if(c.turbulent)
    {
        std::fill(c.k.begin(), c.k.end(), 0.0);
        std::fill(c.epsilon.begin(), c.epsilon.end(), 0.0);
        std::fill(c.nut.begin(), c.nut.end(), 0.0);
    }
    for(int fc = 0; fc < f.grid.nCells(); fc++)
    {
        int cc = map[fc];
        double V = f.grid.volume[fc];
        vol[cc] += V;
        c.p[cc] += V * f.p[fc];
        c.u[cc] += V * f.u[fc];
        c.v[cc] += V * f.v[fc];
        c.w[cc] += V * f.w[fc];
        if(c.turbulent)
        {
            c.k[cc] += V * f.k[fc];
            c.epsilon[cc] += V * f.epsilon[fc];
            c.nut[cc] += V * f.nut[fc];
        }
    }
    for(size_t cc = 0; cc < n; cc++)
    {
        c.p[cc] /= vol[cc]; c.u[cc] /= vol[cc]; c.v[cc] /= vol[cc]; c.w[cc] /= vol[cc];
        if(c.turbulent)
        {
            c.k[cc] /= vol[cc]; c.epsilon[cc] /= vol[cc]; c.nut[cc] /= vol[cc];
        }
    }
}

RansSolver::Residuals RansMultigrid::vcycle(int level, double cfl, int preSmooth, int postSmooth,
                                            int coarsest)
{
    RansSolver &f = *solvers[level];
    if(level + 1 == levels())
        return f.iterate(coarsest, cfl);
    f.iterate(preSmooth, cfl);

    // fine residual (with this level's own forcing) at the smoothed state
    f.computeGradients();
    f.computeResiduals();

    RansSolver &c = *solvers[level + 1];
    restrictState(level);
    std::vector<double> &start = q0[level];
    size_t n = c.grid.nCells();
    for(size_t cc = 0; cc < n; cc++)
    {
        start[4 * cc] = c.p[cc]; start[4 * cc + 1] = c.u[cc];
        start[4 * cc + 2] = c.v[cc]; start[4 * cc + 3] = c.w[cc];
    }

    // FAS forcing: F = sum of fine residuals - coarse residual of the
    // restricted state, so the coarse residual starts equal to the fine one
    std::fill(c.forceP.begin(), c.forceP.end(), 0.0);
    std::fill(c.forceU.begin(), c.forceU.end(), 0.0);
    std::fill(c.forceV.begin(), c.forceV.end(), 0.0);
    std::fill(c.forceW.begin(), c.forceW.end(), 0.0);
    c.computeGradients();
    c.computeResiduals();
    std::vector<double> fp(n, 0.0), fu(n, 0.0), fv(n, 0.0), fw(n, 0.0);
    const std::vector<int> &map = toCoarse[level];
    for(int fc = 0; fc < f.grid.nCells(); fc++)
    {
        int cc = map[fc];
        fp[cc] += f.rp[fc]; fu[cc] += f.ru[fc]; fv[cc] += f.rv[fc]; fw[cc] += f.rw[fc];
    }
    for(size_t cc = 0; cc < n; cc++)
    {
        c.forceP[cc] = fp[cc] - c.rp[cc];
        c.forceU[cc] = fu[cc] - c.ru[cc];
        c.forceV[cc] = fv[cc] - c.rv[cc];
        c.forceW[cc] = fw[cc] - c.rw[cc];
    }

    vcycle(level + 1, cfl, preSmooth, postSmooth, coarsest);

    // prolong the coarse correction (trilinear in index space)
    const RansGrid &fg = f.grid, &cg = c.grid;
    const Lean &lx = leanX[level], &ly = leanY[level], &lz = leanZ[level];
    #pragma omp parallel for collapse(2) schedule(static) num_threads(f.nThreads())
    for(int k = 0; k < fg.nz; k++)
        for(int j = 0; j < fg.ny; j++)
            for(int i = 0; i < fg.nx; i++)
            {
                int fc = fg.cell(i, j, k);
                int home = map[fc];
                int I = home % cg.nx, J = (home / cg.nx) % cg.ny, K = home / (cg.nx * cg.ny);
                int ci[2] = {I, lx.nb[i]}, cj[2] = {J, ly.nb[j]}, ck[2] = {K, lz.nb[k]};
                double wi[2] = {1.0 - lx.w[i], lx.w[i]};
                double wj[2] = {1.0 - ly.w[j], ly.w[j]};
                double wk[2] = {1.0 - lz.w[k], lz.w[k]};
                double d[4] = {0.0, 0.0, 0.0, 0.0};
                for(int a = 0; a < 2; a++)
                    for(int b = 0; b < 2; b++)
                        for(int e = 0; e < 2; e++)
                        {
                            double wt = wi[a] * wj[b] * wk[e];
                            if(wt == 0.0) continue;
                            int cc = cg.cell(ci[a], cj[b], ck[e]);
                            d[0] += wt * (c.p[cc] - start[4 * cc]);
                            d[1] += wt * (c.u[cc] - start[4 * cc + 1]);
                            d[2] += wt * (c.v[cc] - start[4 * cc + 2]);
                            d[3] += wt * (c.w[cc] - start[4 * cc + 3]);
                        }
                f.p[fc] += correctionDamping * d[0];
                f.u[fc] += correctionDamping * d[1];
                f.v[fc] += correctionDamping * d[2];
                f.w[fc] += correctionDamping * d[3];
            }
    return f.iterate(std::max(postSmooth, 1), cfl);
}

RansSolver::Residuals RansMultigrid::cycle(double cfl, int preSmooth, int postSmooth, int coarsest)
{
    return vcycle(0, cfl, preSmooth, postSmooth, coarsest);
}
