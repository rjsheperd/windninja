/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Mass-consistent solver: preconditioned conjugate gradient
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

#include <cmath>
#include <stdexcept>
#include <vector>

#include "preconditioner.h"

namespace {

/* y = A x for the upper-triangle CSR of a symmetric A (ninja::mkl_dcsrmv):
** the stored rows in parallel, then their transposes serially. */
void SymmetricMultiply(int n, const double *val, const int *col, const int *rowPtr,
                       const double *x, double *y)
{
    int i, j;
    #pragma omp parallel private(i, j)
    {
        #pragma omp for
        for(i = 0; i < n; i++)
            y[i] = 0.0;
        #pragma omp for
        for(i = 0; i < n; i++)
        {
            y[i] += val[rowPtr[i]] * x[i];   // diagonal
            for(j = rowPtr[i] + 1; j < rowPtr[i + 1]; j++)
                y[i] += val[j] * x[col[j]];
        }
    }
    for(i = 0; i < n; i++)
        for(j = rowPtr[i] + 1; j < rowPtr[i + 1]; j++)
            y[col[j]] += val[j] * x[i];
}

double Dot(int n, const double *x, const double *y)
{
    double val = 0.0;
    #pragma omp parallel for reduction(+:val)
    for(int i = 0; i < n; i++)
        val += x[i] * y[i];
    return val;
}

double Norm2(int n, const double *x)
{
    double val = 0.0;
    for(int i = 0; i < n; i++)
        val += x[i] * x[i];
    return std::sqrt(val);
}

void Axpy(int n, double alpha, const double *x, double *y)
{
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        y[i] += alpha * x[i];
}

} // namespace

MassSolveResult SolveMassPcg(const MassSystem &system, double *phi,
                             const MassSolveOptions &options)
{
    const int n = (int)system.RHS.size();
    // The CSR is only read; Preconditioner's API takes non-const pointers.
    double *A = const_cast<double *>(system.SK.data());
    int *rowPtr = const_cast<int *>(system.row_ptr.data());
    int *col = const_cast<int *>(system.col_ind.data());
    const double *b = system.RHS.data();

    MassSolveResult result;
    char matdescra[6] = {'s', 'u', 'n', 'c', 0, 0};   // symmetric, upper, C indices
    Preconditioner M;
    result.jacobi = options.jacobi;
    if(!options.jacobi && !M.initialize(n, A, rowPtr, col, M.SSOR, matdescra))
    {
        if(options.warn)
            options.warn("Initialization of SSOR preconditioner failed, trying Jacobi preconditioner...");
        result.jacobi = true;
        if(!M.initialize(n, A, rowPtr, col, M.Jacobi, matdescra))
            throw std::runtime_error("Initialization of Jacobi preconditioner failed.");
    }
    else if(options.jacobi && !M.initialize(n, A, rowPtr, col, M.Jacobi, matdescra))
        throw std::runtime_error("Initialization of Jacobi preconditioner failed.");

    std::vector<double> p(n), z(n), q(n), r(n);
    SymmetricMultiply(n, A, col, rowPtr, phi, r.data());
    for(int i = 0; i < n; i++)
        r[i] = b[i] - r[i];   // initial residual

    double normb = Norm2(n, b);
    if(normb == 0.0)
        normb = 1.;
    double resid = Norm2(n, r.data()) / normb;
    result.residual = resid;
    if(resid <= options.tol)
    {
        result.converged = true;
        return result;
    }

    double rho = 0.0, rho_1 = 0.0, startResid = 0.0;
    int i;
    for(i = 1; i <= options.maxIterations; i++)
    {
        if(options.checkCancel)
            options.checkCancel();
        M.solve(r.data(), z.data(), rowPtr, col);   // apply the preconditioner
        rho = Dot(n, z.data(), r.data());
        if(i == 1)
        {
            p = z;
        }
        else
        {
            double beta = rho / rho_1;
            #pragma omp parallel for
            for(int j = 0; j < n; j++)
                p[j] = z[j] + beta * p[j];
        }
        SymmetricMultiply(n, A, col, rowPtr, p.data(), q.data());
        double alpha = rho / Dot(n, p.data(), q.data());
        Axpy(n, alpha, p.data(), phi);
        Axpy(n, -alpha, q.data(), r.data());
        resid = Norm2(n, r.data()) / normb;
        if(i == 1)
            startResid = resid;
        if(i % options.progressEvery == 0 && options.progress)
            options.progress(i, resid, startResid);
        if(resid <= options.tol)
            break;
        rho_1 = rho;
    }

    result.iterations = i < options.maxIterations ? i : options.maxIterations;
    result.residual = resid;
    result.converged = resid <= options.tol;
    return result;
}

double MassRelativeResidual(const MassSystem &system, const double *x)
{
    const int n = (int)system.RHS.size();
    std::vector<double> ax(n);
    SymmetricMultiply(n, system.SK.data(), system.col_ind.data(), system.row_ptr.data(), x,
                      ax.data());
    for(int i = 0; i < n; i++)
        ax[i] = system.RHS[i] - ax[i];
    double normb = Norm2(n, system.RHS.data());
    return Norm2(n, ax.data()) / (normb == 0.0 ? 1.0 : normb);
}
