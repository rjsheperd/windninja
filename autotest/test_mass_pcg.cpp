/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Tests for the mass solver's preconditioned conjugate gradient
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

#include "mass_pcg.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <boost/test/unit_test.hpp>

/* A sloped terrain grid with unit spacing, as in test_mass_fem.cpp. */
static MassGrid PcgGrid(int ncols, int nrows, int nlayers)
{
    MassGrid g;
    g.ncols = ncols;
    g.nrows = nrows;
    g.nlayers = nlayers;
    for(int k = 0; k < nlayers; k++)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < ncols; j++)
            {
                g.x.push_back(j);
                g.y.push_back(i);
                g.z.push_back(0.3 * j + 0.2 * std::sin((double)i) + k);
            }
    return g;
}

/* A divergent initial wind (u0 = x, v0 = -0.5 y) gives a non-trivial RHS. */
static MassSystem PcgSystem(const MassGrid &g)
{
    std::vector<double> u0(g.x), v0(g.numNodes()), w0(g.numNodes(), 0.0),
        alphaV(g.numNodes(), 1.0);
    for(int i = 0; i < g.numNodes(); i++)
        v0[i] = -0.5 * g.y[i];
    return AssembleMass(g, u0.data(), v0.data(), w0.data(), alphaV.data(), 1.0);
}

/* ||A x - b|| / ||b||, with A from its upper triangle, computed directly. */
static double RelativeResidual(const MassSystem &s, const std::vector<double> &x)
{
    int n = (int)s.RHS.size();
    std::vector<double> ax(n, 0.0);
    for(int i = 0; i < n; i++)
        for(int l = s.row_ptr[i]; l < s.row_ptr[i + 1]; l++)
        {
            int j = s.col_ind[l];
            ax[i] += s.SK[l] * x[j];
            if(j != i)
                ax[j] += s.SK[l] * x[i];
        }
    double r = 0.0, b = 0.0;
    for(int i = 0; i < n; i++)
    {
        r += (ax[i] - s.RHS[i]) * (ax[i] - s.RHS[i]);
        b += s.RHS[i] * s.RHS[i];
    }
    return std::sqrt(r / b);
}

BOOST_AUTO_TEST_SUITE(mass_pcg)

/* Both preconditioners reach a tight tolerance and agree on Phi. */
BOOST_AUTO_TEST_CASE(solves_to_tolerance)
{
    MassGrid g = PcgGrid(9, 8, 6);
    MassSystem s = PcgSystem(g);
    std::vector<double> ssor(g.numNodes(), 0.0), jacobi(g.numNodes(), 0.0);
    MassSolveOptions options;
    options.tol = 1e-10;

    MassSolveResult a = SolveMassPcg(s, ssor.data(), options);
    BOOST_CHECK(a.converged);
    BOOST_CHECK(!a.jacobi);
    BOOST_CHECK_LE(a.residual, 1e-10);
    BOOST_CHECK_LE(RelativeResidual(s, ssor), 1e-9);

    options.jacobi = true;
    MassSolveResult b = SolveMassPcg(s, jacobi.data(), options);
    BOOST_CHECK(b.converged);
    BOOST_CHECK(b.jacobi);
    BOOST_CHECK_LE(RelativeResidual(s, jacobi), 1e-9);
    BOOST_CHECK_GT(b.iterations, a.iterations);   // SSOR is the stronger preconditioner

    double scale = 0.0;
    for(double v : ssor)
        scale = std::max(scale, std::fabs(v));
    BOOST_REQUIRE_GT(scale, 0.0);
    for(int i = 0; i < g.numNodes(); i++)
        BOOST_CHECK_SMALL(ssor[i] - jacobi[i], 1e-7 * scale);
}

/* A converged start returns at once, with Phi untouched. */
BOOST_AUTO_TEST_CASE(zero_rhs_returns_immediately)
{
    MassGrid g = PcgGrid(5, 5, 4);
    std::vector<double> zero(g.numNodes(), 0.0), alphaV(g.numNodes(), 1.0);
    MassSystem s = AssembleMass(g, zero.data(), zero.data(), zero.data(), alphaV.data(), 1.0);
    std::vector<double> phi(g.numNodes(), 0.0);
    MassSolveResult r = SolveMassPcg(s, phi.data(), MassSolveOptions());
    BOOST_CHECK(r.converged);
    BOOST_CHECK_EQUAL(r.iterations, 0);
    for(double v : phi)
        BOOST_CHECK_EQUAL(v, 0.0);
}

/* Out of iterations: not converged, the residual reported honestly. */
BOOST_AUTO_TEST_CASE(iteration_cap)
{
    MassGrid g = PcgGrid(9, 8, 6);
    MassSystem s = PcgSystem(g);
    std::vector<double> phi(g.numNodes(), 0.0);
    MassSolveOptions options;
    options.tol = 1e-12;
    options.maxIterations = 2;
    MassSolveResult r = SolveMassPcg(s, phi.data(), options);
    BOOST_CHECK(!r.converged);
    BOOST_CHECK_EQUAL(r.iterations, 2);
    BOOST_CHECK_CLOSE(r.residual, RelativeResidual(s, phi), 1e-6);
    BOOST_CHECK_CLOSE(MassRelativeResidual(s, phi.data()), RelativeResidual(s, phi), 1e-9);
}

/* Progress every N iterations; a throwing cancel check stops the solve. */
BOOST_AUTO_TEST_CASE(progress_and_cancel)
{
    MassGrid g = PcgGrid(9, 8, 6);
    MassSystem s = PcgSystem(g);
    std::vector<double> phi(g.numNodes(), 0.0);
    std::vector<int> seen;
    MassSolveOptions options;
    options.tol = 1e-10;
    options.jacobi = true;
    options.progressEvery = 3;
    options.progress = [&](int i, double resid, double start) {
        seen.push_back(i);
        BOOST_CHECK_GT(start, 0.0);
        BOOST_CHECK_GT(resid, 0.0);
    };
    MassSolveResult r = SolveMassPcg(s, phi.data(), options);
    BOOST_REQUIRE(!seen.empty());
    for(size_t k = 0; k < seen.size(); k++)
        BOOST_CHECK_EQUAL(seen[k], 3 * (int)(k + 1));
    BOOST_CHECK_LE(seen.back(), r.iterations);

    int calls = 0;
    options.checkCancel = [&]() {
        if(++calls == 4)
            throw std::runtime_error("cancelled");
    };
    std::fill(phi.begin(), phi.end(), 0.0);
    BOOST_CHECK_THROW(SolveMassPcg(s, phi.data(), options), std::runtime_error);
    BOOST_CHECK_EQUAL(calls, 4);
}

BOOST_AUTO_TEST_SUITE_END()
