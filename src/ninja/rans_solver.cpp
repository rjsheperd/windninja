/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  CPU reference incompressible RANS solver (artificial compressibility)
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

#include "rans_solver.h"

#include "rans_abl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

/*
** Visit every face once.  f(owner, neighbour, side, sx, sy, sz, fx, fy, fz)
** gets the area vector pointing out of owner; neighbour is -1 and side is the
** block side for boundary faces, and side is -1 for interior faces.  agl is
** the face centroid's height above the ground (side faces only; 0 on k faces).
*/
template <typename F>
static void ForEachFace(const RansGrid &g, int threads, F f)
{
    // Three passes, one per face direction.  Within a pass, faces on
    // different grid lines (i faces on a (j, k) line, j faces on an (i, k)
    // line, k faces on an (i, j) column) touch disjoint cells, so the lines
    // run in parallel without races.  Each cell still receives its face
    // contributions in the serial order, so results don't depend on the
    // thread count.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for(int k = 0; k < g.nz; k++)
        for(int j = 0; j < g.ny; j++)
            for(int i = 0; i <= g.nx; i++)
            {
                int n = g.iFace(i, j, k);
                double sx = g.iSx[n], sy = g.iSy[n], sz = g.iSz[n];
                if(i == 0)
                    f(g.cell(0, j, k), -1, RansSolver::xMin, -sx, -sy, -sz, g.iFx[n], g.iFy[n], g.iFz[n], g.iAgl[n]);
                else if(i == g.nx)
                    f(g.cell(i - 1, j, k), -1, RansSolver::xMax, sx, sy, sz, g.iFx[n], g.iFy[n], g.iFz[n], g.iAgl[n]);
                else
                    f(g.cell(i - 1, j, k), g.cell(i, j, k), -1, sx, sy, sz, g.iFx[n], g.iFy[n], g.iFz[n], g.iAgl[n]);
            }
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for(int k = 0; k < g.nz; k++)
        for(int i = 0; i < g.nx; i++)
            for(int j = 0; j <= g.ny; j++)
            {
                int n = g.jFace(i, j, k);
                double sx = g.jSx[n], sy = g.jSy[n], sz = g.jSz[n];
                if(j == 0)
                    f(g.cell(i, 0, k), -1, RansSolver::yMin, -sx, -sy, -sz, g.jFx[n], g.jFy[n], g.jFz[n], g.jAgl[n]);
                else if(j == g.ny)
                    f(g.cell(i, j - 1, k), -1, RansSolver::yMax, sx, sy, sz, g.jFx[n], g.jFy[n], g.jFz[n], g.jAgl[n]);
                else
                    f(g.cell(i, j - 1, k), g.cell(i, j, k), -1, sx, sy, sz, g.jFx[n], g.jFy[n], g.jFz[n], g.jAgl[n]);
            }
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for(int j = 0; j < g.ny; j++)
        for(int i = 0; i < g.nx; i++)
            for(int k = 0; k <= g.nz; k++)
            {
                int n = g.kFace(i, j, k);
                double sx = g.kSx[n], sy = g.kSy[n], sz = g.kSz[n];
                if(k == 0)
                    f(g.cell(i, j, 0), -1, RansSolver::zMin, -sx, -sy, -sz, g.kFx[n], g.kFy[n], g.kFz[n], 0.0);
                else if(k == g.nz)
                    f(g.cell(i, j, k - 1), -1, RansSolver::zMax, sx, sy, sz, g.kFx[n], g.kFy[n], g.kFz[n], 0.0);
                else
                    f(g.cell(i, j, k - 1), g.cell(i, j, k), -1, sx, sy, sz, g.kFx[n], g.kFy[n], g.kFz[n], 0.0);
            }
}

#ifdef _OPENMP
#include <omp.h>
#endif

int RansSolver::nThreads() const
{
#ifdef _OPENMP
    // small grids: thread start-up and synchronisation outweigh the work
    if(threads <= 0 && grid.nCells() < 20000)
        return 1;
    return threads > 0 ? threads : omp_get_max_threads();
#else
    return 1;
#endif
}

RansSolver::RansSolver(const RansGrid &grid, double nu, double beta)
    : grid(grid), nu(nu), beta(beta)
{
    if(nu < 0.0 || beta <= 0.0)
        throw std::invalid_argument("RansSolver needs nu >= 0 and beta > 0");
    size_t n = grid.nCells();
    for(std::vector<double> *a : {&p, &u, &v, &w, &rp, &ru, &rv, &rw, &spectral})
        a->assign(n, 0.0);
    for(int d = 0; d < 3; d++)
        for(std::vector<double> *a : {&gp[d], &gu[d], &gv[d], &gw[d]})
            a->assign(n, 0.0);
    Boundary slipWall = {Boundary::slip, 0.0, 0.0, 0.0, 0.0, StateFunction()};
    for(int s = 0; s < 6; s++)
        bc[s] = slipWall;
}

void RansSolver::setBoundary(Side side, const Boundary &b)
{
    if(b.type == Boundary::dirichlet && !b.state)
        throw std::invalid_argument("a dirichlet boundary needs a state function");
    bc[side] = b;
}

void RansSolver::setSource(const StateFunction &s)
{
    source = s;
}

void RansSolver::setInitial(const StateFunction &state)
{
    for(int c = 0; c < grid.nCells(); c++)
        state(grid.cx[c], grid.cy[c], grid.cz[c], p[c], u[c], v[c], w[c]);
}

void RansSolver::enableTurbulence(double k0, double epsilon0, const TurbulenceModel &m)
{
    if(k0 <= 0.0 || epsilon0 <= 0.0)
        throw std::invalid_argument("enableTurbulence needs k0 > 0 and epsilon0 > 0");
    turbulent = true;
    model = m;
    size_t n = grid.nCells();
    k.assign(n, k0);
    epsilon.assign(n, epsilon0);
    nut.assign(n, model.cmu * k0 * k0 / epsilon0);
    for(std::vector<double> *a : {&rk, &rEps, &epsWall, &gWall, &resKCell, &resECell})
        a->assign(n, 0.0);
    nWall.assign(n, 0);
}

/*
** Turbulence production G = nut S2 in cell c, with
** S2 = grad(u) : dev(2 symm(grad(u))) as in OpenFOAM's RNGkEpsilon.
*/
double RansSolver::production(int c, double &s2) const
{
    const std::vector<double> *gs[3] = {gu, gv, gw};
    double sum = 0.0, trace = 0.0;
    for(int i = 0; i < 3; i++)
    {
        trace += gs[i][i][c];
        for(int j = 0; j < 3; j++)
            sum += gs[i][j][c] * (gs[i][j][c] + gs[j][i][c]);
    }
    s2 = sum - 2.0 / 3.0 * trace * trace;
    return nut[c] * s2;
}

void RansSolver::setUniform(double p0, double u0, double v0, double w0)
{
    std::fill(p.begin(), p.end(), p0);
    std::fill(u.begin(), u.end(), u0);
    std::fill(v.begin(), v.end(), v0);
    std::fill(w.begin(), w.end(), w0);
}

/*
** State on a boundary face from the owner cell and the side's condition.
** fixedVelocity says whether the face velocity is prescribed (so viscous
** stress acts on it); otherwise the boundary carries no viscous flux.
*/
static void BoundaryState(const RansSolver::Boundary &b, double nx, double ny, double nz,
                          double fx, double fy, double fz, double agl,
                          double pc, double uc, double vc, double wc,
                          double &pb, double &ub, double &vb, double &wb, bool &fixedVelocity)
{
    switch(b.type)
    {
        case RansSolver::Boundary::wall:
        case RansSolver::Boundary::velocityInlet:
            pb = pc; ub = b.u; vb = b.v; wb = b.w;
            fixedVelocity = true;
            break;
        case RansSolver::Boundary::slip:
        {
            double un = uc * nx + vc * ny + wc * nz;
            pb = pc; ub = uc - un * nx; vb = vc - un * ny; wb = wc - un * nz;
            fixedVelocity = false;
            break;
        }
        case RansSolver::Boundary::pressureOutlet:
            pb = b.p; ub = uc; vb = vc; wb = wc;
            fixedVelocity = false;
            break;
        case RansSolver::Boundary::dirichlet:
            b.state(fx, fy, fz, pb, ub, vb, wb);
            fixedVelocity = true;
            break;
        case RansSolver::Boundary::logProfileInlet:
        {
            double speed = AblInletVelocity(b.ustar, agl, b.z0);
            pb = pc; ub = speed * b.u; vb = speed * b.v; wb = speed * b.w;
            fixedVelocity = true;
            break;
        }
    }
}

/*
** k and epsilon on a boundary face.  fixed says whether the value is
** prescribed (diffusive flux into the cell) or zero-gradient.  Walls are
** zero-gradient here: the wall functions act on the wall cells instead.
*/
static void BoundaryTurbulence(const RansSolver::Boundary &b, double agl,
                               double kc, double ec, double &kb, double &eb, bool &fixed)
{
    switch(b.type)
    {
        case RansSolver::Boundary::velocityInlet:
            kb = b.k; eb = b.epsilon; fixed = true;
            break;
        case RansSolver::Boundary::logProfileInlet:
            kb = AblInletK(b.ustar);
            eb = AblInletEpsilon(b.ustar, agl, b.z0);
            fixed = true;
            break;
        default:
            kb = kc; eb = ec; fixed = false;
            break;
    }
}

void RansSolver::computeGradients()
{
    for(int d = 0; d < 3; d++)
        for(std::vector<double> *a : {&gp[d], &gu[d], &gv[d], &gw[d]})
            std::fill(a->begin(), a->end(), 0.0);

    const RansGrid &g = grid;
    ForEachFace(g, nThreads(), [&](int o, int nb, int side, double sx, double sy, double sz,
                       double fx, double fy, double fz, double agl)
    {
        double S[3] = {sx, sy, sz};
        double fp, fu, fv, fw;
        if(nb >= 0)
        {
            // inverse-distance weights from the two centroids
            double dO = std::hypot(fx - g.cx[o], fy - g.cy[o], fz - g.cz[o]);
            double dN = std::hypot(fx - g.cx[nb], fy - g.cy[nb], fz - g.cz[nb]);
            double a = dN / (dO + dN);
            fp = a * p[o] + (1 - a) * p[nb];
            fu = a * u[o] + (1 - a) * u[nb];
            fv = a * v[o] + (1 - a) * v[nb];
            fw = a * w[o] + (1 - a) * w[nb];
        }
        else
        {
            double area = std::sqrt(sx * sx + sy * sy + sz * sz);
            bool fixed;
            BoundaryState(bc[side], sx / area, sy / area, sz / area, fx, fy, fz, agl,
                          p[o], u[o], v[o], w[o], fp, fu, fv, fw, fixed);
        }
        for(int d = 0; d < 3; d++)
        {
            gp[d][o] += fp * S[d]; gu[d][o] += fu * S[d];
            gv[d][o] += fv * S[d]; gw[d][o] += fw * S[d];
            if(nb >= 0)
            {
                gp[d][nb] -= fp * S[d]; gu[d][nb] -= fu * S[d];
                gv[d][nb] -= fv * S[d]; gw[d][nb] -= fw * S[d];
            }
        }
    });
    #pragma omp parallel for schedule(static) num_threads(nThreads())
    for(int c = 0; c < g.nCells(); c++)
        for(int d = 0; d < 3; d++)
        {
            gp[d][c] /= g.volume[c]; gu[d][c] /= g.volume[c];
            gv[d][c] /= g.volume[c]; gw[d][c] /= g.volume[c];
        }
}

/* Inviscid flux of the AC system through area vector S for state q. */
static void InviscidFlux(double beta, double sx, double sy, double sz,
                         double p, double u, double v, double w, double f[4])
{
    double U = u * sx + v * sy + w * sz;
    f[0] = beta * U;
    f[1] = u * U + p * sx;
    f[2] = v * U + p * sy;
    f[3] = w * U + p * sz;
}

void RansSolver::computeResiduals()
{
    const RansGrid &g = grid;
    std::fill(rp.begin(), rp.end(), 0.0);
    std::fill(ru.begin(), ru.end(), 0.0);
    std::fill(rv.begin(), rv.end(), 0.0);
    std::fill(rw.begin(), rw.end(), 0.0);
    std::fill(spectral.begin(), spectral.end(), 0.0);
    if(turbulent)
    {
        std::fill(rk.begin(), rk.end(), 0.0);
        std::fill(rEps.begin(), rEps.end(), 0.0);
        std::fill(epsWall.begin(), epsWall.end(), 0.0);
        std::fill(gWall.begin(), gWall.end(), 0.0);
        std::fill(nWall.begin(), nWall.end(), 0);
    }
    const double sigmaMin = std::min(model.sigmaK, model.sigmaEpsilon);

    ForEachFace(g, nThreads(), [&](int o, int nb, int side, double sx, double sy, double sz,
                       double fx, double fy, double fz, double agl)
    {
        double area2 = sx * sx + sy * sy + sz * sz;
        double area = std::sqrt(area2);
        double f[4];
        double gradU[3][3];   // face velocity gradient: [component][direction]
        bool viscous = true;
        // turbulence at the face: eddy viscosity, centroid distance and S.e
        // for orthogonal diffusion, and boundary k / epsilon
        double nutF = 0.0, dist = 0.0, sDotE = 0.0, kb = 0.0, eb = 0.0;
        bool fixedTurb = false;

        if(nb >= 0)
        {
            // second-order reconstruction to the face from both sides
            double dxo = fx - g.cx[o], dyo = fy - g.cy[o], dzo = fz - g.cz[o];
            double dxn = fx - g.cx[nb], dyn = fy - g.cy[nb], dzn = fz - g.cz[nb];
            auto rec = [](const std::vector<double> &q, const std::vector<double> *gq,
                          int c, double dx, double dy, double dz)
            { return q[c] + gq[0][c] * dx + gq[1][c] * dy + gq[2][c] * dz; };
            double pL = rec(p, gp, o, dxo, dyo, dzo), pR = rec(p, gp, nb, dxn, dyn, dzn);
            double uL = rec(u, gu, o, dxo, dyo, dzo), uR = rec(u, gu, nb, dxn, dyn, dzn);
            double vL = rec(v, gv, o, dxo, dyo, dzo), vR = rec(v, gv, nb, dxn, dyn, dzn);
            double wL = rec(w, gw, o, dxo, dyo, dzo), wR = rec(w, gw, nb, dxn, dyn, dzn);

            double fL[4], fR[4];
            InviscidFlux(beta, sx, sy, sz, pL, uL, vL, wL, fL);
            InviscidFlux(beta, sx, sy, sz, pR, uR, vR, wR, fR);
            double UL = uL * sx + vL * sy + wL * sz, UR = uR * sx + vR * sy + wR * sz;
            double lambda = std::max(std::fabs(UL) + std::sqrt(UL * UL + beta * area2),
                                     std::fabs(UR) + std::sqrt(UR * UR + beta * area2));
            f[0] = 0.5 * (fL[0] + fR[0]) - 0.5 * lambda * (pR - pL);
            // momentum is dissipated at the convective speed only (upwind,
            // like OpenFOAM's linearUpwind U).  The acoustic speed above
            // stays on the pressure, where it plays the Rhie-Chow role;
            // applying it to velocity acts as a large artificial viscosity
            // on coarse near-wall cells.
            double lambdaU = std::max(std::fabs(UL), std::fabs(UR));
            f[1] = 0.5 * (fL[1] + fR[1]) - 0.5 * lambdaU * (uR - uL);
            f[2] = 0.5 * (fL[2] + fR[2]) - 0.5 * lambdaU * (vR - vL);
            f[3] = 0.5 * (fL[3] + fR[3]) - 0.5 * lambdaU * (wR - wL);

            // averaged gradient corrected along the centroid line
            double ex = g.cx[nb] - g.cx[o], ey = g.cy[nb] - g.cy[o], ez = g.cz[nb] - g.cz[o];
            double d = std::sqrt(ex * ex + ey * ey + ez * ez);
            ex /= d; ey /= d; ez /= d;
            const std::vector<double> *qs[3] = {&u, &v, &w};
            const std::vector<double> *gs[3] = {gu, gv, gw};
            for(int m = 0; m < 3; m++)
            {
                double a[3];
                for(int dd = 0; dd < 3; dd++)
                    a[dd] = 0.5 * (gs[m][dd][o] + gs[m][dd][nb]);
                double corr = ((*qs[m])[nb] - (*qs[m])[o]) / d - (a[0] * ex + a[1] * ey + a[2] * ez);
                gradU[m][0] = a[0] + corr * ex;
                gradU[m][1] = a[1] + corr * ey;
                gradU[m][2] = a[2] + corr * ez;
            }
            spectral[o] += 0.5 * lambda + nu * area2 / (d * area);
            spectral[nb] += 0.5 * lambda + nu * area2 / (d * area);
            if(turbulent)
            {
                dist = d;
                sDotE = sx * ex + sy * ey + sz * ez;
                nutF = 0.5 * (nut[o] + nut[nb]);
                double extra = nutF / sigmaMin * area2 / (d * area);
                spectral[o] += extra;
                spectral[nb] += extra;
            }
        }
        else
        {
            double pb, ub, vb, wb;
            bool fixedVelocity;
            BoundaryState(bc[side], sx / area, sy / area, sz / area, fx, fy, fz, agl,
                          p[o], u[o], v[o], w[o], pb, ub, vb, wb, fixedVelocity);
            double Ub = ub * sx + vb * sy + wb * sz;
            double lambda = std::fabs(Ub) + std::sqrt(Ub * Ub + beta * area2);
            if(bc[side].type == Boundary::dirichlet)
            {
                // Rusanov flux between the reconstructed interior state and
                // the prescribed state, so the pressure wave can carry a mass
                // imbalance out.  Imposing F(state) instead fixes the total
                // boundary mass flux, and its O(h^2) mismatch with the source
                // then makes the pressure drift without converging.
                double dx = fx - g.cx[o], dy = fy - g.cy[o], dz = fz - g.cz[o];
                double pI = p[o] + gp[0][o] * dx + gp[1][o] * dy + gp[2][o] * dz;
                double uI = u[o] + gu[0][o] * dx + gu[1][o] * dy + gu[2][o] * dz;
                double vI = v[o] + gv[0][o] * dx + gv[1][o] * dy + gv[2][o] * dz;
                double wI = w[o] + gw[0][o] * dx + gw[1][o] * dy + gw[2][o] * dz;
                double fI[4], fB[4];
                InviscidFlux(beta, sx, sy, sz, pI, uI, vI, wI, fI);
                InviscidFlux(beta, sx, sy, sz, pb, ub, vb, wb, fB);
                double UI = uI * sx + vI * sy + wI * sz;
                lambda = std::max(lambda, std::fabs(UI) + std::sqrt(UI * UI + beta * area2));
                f[0] = 0.5 * (fI[0] + fB[0]) - 0.5 * lambda * (pb - pI);
                double lambdaU = std::max(std::fabs(Ub), std::fabs(UI));
                f[1] = 0.5 * (fI[1] + fB[1]) - 0.5 * lambdaU * (ub - uI);
                f[2] = 0.5 * (fI[2] + fB[2]) - 0.5 * lambdaU * (vb - vI);
                f[3] = 0.5 * (fI[3] + fB[3]) - 0.5 * lambdaU * (wb - wI);
            }
            else
            {
                InviscidFlux(beta, sx, sy, sz, pb, ub, vb, wb, f);
            }

            double ex = fx - g.cx[o], ey = fy - g.cy[o], ez = fz - g.cz[o];
            double d = std::sqrt(ex * ex + ey * ey + ez * ez);
            ex /= d; ey /= d; ez /= d;
            viscous = fixedVelocity;
            const double qb[3] = {ub, vb, wb};
            const std::vector<double> *qs[3] = {&u, &v, &w};
            const std::vector<double> *gs[3] = {gu, gv, gw};
            for(int m = 0; m < 3; m++)
            {
                double a[3] = {gs[m][0][o], gs[m][1][o], gs[m][2][o]};
                double corr = (qb[m] - (*qs[m])[o]) / d - (a[0] * ex + a[1] * ey + a[2] * ez);
                gradU[m][0] = a[0] + corr * ex;
                gradU[m][1] = a[1] + corr * ey;
                gradU[m][2] = a[2] + corr * ez;
            }
            spectral[o] += 0.5 * lambda + (viscous ? nu * area2 / (d * area) : 0.0);
            if(turbulent)
            {
                dist = d;
                sDotE = sx * ex + sy * ey + sz * ez;
                nutF = nut[o];
                if(bc[side].type == Boundary::wall && bc[side].z0 > 0.0)
                {
                    // rough-wall functions on the wall cell; y is the cell
                    // centre's normal distance from the wall
                    double y = std::fabs((g.cx[o] - fx) * sx + (g.cy[o] - fy) * sy +
                                         (g.cz[o] - fz) * sz) / area;
                    nutF = std::max(0.0, RoughWallNut(nu, k[o], y, bc[side].z0));
                    double du = std::sqrt((u[o] - ub) * (u[o] - ub) + (v[o] - vb) * (v[o] - vb) +
                                          (w[o] - wb) * (w[o] - wb));
                    double eW, gW;
                    EpsilonWallFunction(nu, nutF, k[o], y, du / y, eW, gW);
                    epsWall[o] += eW;
                    gWall[o] += gW;
                    nWall[o]++;
                }
                BoundaryTurbulence(bc[side], agl, k[o], epsilon[o], kb, eb, fixedTurb);
                if(viscous || fixedTurb)
                    spectral[o] += nutF / sigmaMin * area2 / (d * area);
            }
        }

        if(viscous && !turbulent)
        {
            f[1] -= nu * (gradU[0][0] * sx + gradU[0][1] * sy + gradU[0][2] * sz);
            f[2] -= nu * (gradU[1][0] * sx + gradU[1][1] * sy + gradU[1][2] * sz);
            f[3] -= nu * (gradU[2][0] * sx + gradU[2][1] * sy + gradU[2][2] * sz);
        }
        else if(viscous)
        {
            // nu_eff grad(u).S plus nu_t grad(u)^T.S (OpenFOAM's divDevSigma;
            // the trace part vanishes for incompressible flow)
            double nuEff = nu + nutF;
            for(int m = 0; m < 3; m++)
                f[1 + m] -= nuEff * (gradU[m][0] * sx + gradU[m][1] * sy + gradU[m][2] * sz) +
                            nutF * (gradU[0][m] * sx + gradU[1][m] * sy + gradU[2][m] * sz);
        }

        if(turbulent)
        {
            // k and epsilon: first-order upwind convection on the face mass
            // flux (as WindNinja's fvSchemes), orthogonal diffusion
            double F = f[0] / beta;
            double Dk = nu + nutF / model.sigmaK, De = nu + nutF / model.sigmaEpsilon;
            if(nb >= 0)
            {
                double fk = F * (F > 0 ? k[o] : k[nb]) - Dk * (k[nb] - k[o]) / dist * sDotE;
                double fe = F * (F > 0 ? epsilon[o] : epsilon[nb]) -
                            De * (epsilon[nb] - epsilon[o]) / dist * sDotE;
                rk[o] += fk; rk[nb] -= fk;
                rEps[o] += fe; rEps[nb] -= fe;
            }
            else
            {
                double fk = F * (F > 0 ? k[o] : kb);
                double fe = F * (F > 0 ? epsilon[o] : eb);
                if(fixedTurb)
                {
                    fk -= Dk * (kb - k[o]) / dist * sDotE;
                    fe -= De * (eb - epsilon[o]) / dist * sDotE;
                }
                rk[o] += fk;
                rEps[o] += fe;
            }
        }

        rp[o] += f[0]; ru[o] += f[1]; rv[o] += f[2]; rw[o] += f[3];
        if(nb >= 0)
        {
            rp[nb] -= f[0]; ru[nb] -= f[1]; rv[nb] -= f[2]; rw[nb] -= f[3];
        }
    });

    if(source)
    {
        #pragma omp parallel for schedule(static) num_threads(nThreads())
        for(int c = 0; c < g.nCells(); c++)
        {
            double s[4];
            source(g.cx[c], g.cy[c], g.cz[c], s[0], s[1], s[2], s[3]);
            rp[c] -= s[0] * g.volume[c];
            ru[c] -= s[1] * g.volume[c];
            rv[c] -= s[2] * g.volume[c];
            rw[c] -= s[3] * g.volume[c];
        }
    }
}

RansSolver::Residuals RansSolver::norms() const
{
    Residuals r = {0.0, 0.0, 0.0, 0.0};
    int n = grid.nCells();
    for(int c = 0; c < n; c++)
    {
        double V = grid.volume[c];
        r.p += (rp[c] / V) * (rp[c] / V);
        r.u += (ru[c] / V) * (ru[c] / V);
        r.v += (rv[c] / V) * (rv[c] / V);
        r.w += (rw[c] / V) * (rw[c] / V);
    }
    r.p = std::sqrt(r.p / n); r.u = std::sqrt(r.u / n);
    r.v = std::sqrt(r.v / n); r.w = std::sqrt(r.w / n);
    return r;
}

RansSolver::Residuals RansSolver::residuals()
{
    computeGradients();
    computeResiduals();
    return norms();
}

RansSolver::Residuals RansSolver::iterate(int n, double cfl)
{
    Residuals r = {0.0, 0.0, 0.0, 0.0};
    for(int it = 0; it < n; it++)
    {
        computeGradients();
        computeResiduals();
        r = norms();
        #pragma omp parallel for schedule(static) num_threads(nThreads())
        for(int c = 0; c < grid.nCells(); c++)
        {
            // local pseudo time step: dtau / V = cfl / spectral radius
            double dt = cfl / spectral[c];
            if(turbulent)
            {
                // point-implicit sinks (as fvm::Sp), so k and epsilon stay
                // positive; wall cells take G and epsilon from the wall
                // functions
                double s2, G = production(c, s2);
                if(nWall[c] > 0) G = gWall[c] / nWall[c];
                double V = grid.volume[c], kc = k[c], ec = epsilon[c];
                double eta = std::sqrt(std::fabs(s2)) * kc / ec;
                double R = eta * (1.0 - eta / model.eta0) / (model.beta * eta * eta * eta + 1.0);
                double kSrc = G - ec;
                double eSrc = (model.c1 - R) * G * ec / kc - model.c2 * ec * ec / kc;
                double kn = (kc - dt * rk[c] + dt * V * G) / (1.0 + dt * V * ec / kc);
                double en = (ec - dt * rEps[c] + dt * V * (model.c1 - R) * G * ec / kc) /
                            (1.0 + dt * V * model.c2 * ec / kc);
                double resK = rk[c] / V - kSrc, resE = rEps[c] / V - eSrc;
                if(nWall[c] > 0)
                {
                    en = epsWall[c] / nWall[c];
                    resE = 0.0;
                }
                resKCell[c] = resK;
                resECell[c] = resE;
                k[c] = std::max(kn, 1e-12);
                epsilon[c] = std::max(en, 1e-12);
            }
            p[c] -= dt * rp[c];
            u[c] -= dt * ru[c];
            v[c] -= dt * rv[c];
            w[c] -= dt * rw[c];
        }
        if(turbulent)
        {
            #pragma omp parallel for schedule(static) num_threads(nThreads())
            for(int c = 0; c < grid.nCells(); c++)
                nut[c] = model.cmu * k[c] * k[c] / epsilon[c];
            // summed serially so the residual doesn't depend on the threads
            double sumK = 0.0, sumE = 0.0;
            for(int c = 0; c < grid.nCells(); c++)
            {
                sumK += resKCell[c] * resKCell[c];
                sumE += resECell[c] * resECell[c];
            }
            r.k = std::sqrt(sumK / grid.nCells());
            r.epsilon = std::sqrt(sumE / grid.nCells());
        }
    }
    return r;
}
