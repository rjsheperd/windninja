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

#ifndef MASS_PCG_H
#define MASS_PCG_H

#include <functional>
#include <string>

#include "mass_fem.h"

struct MassSolveOptions
{
    double tol = 1e-1;          // stop when ||r||_2 / ||b||_2 <= tol
    int maxIterations = 100000;
    bool jacobi = false;        // Jacobi preconditioner instead of SSOR
    int progressEvery = 10;     // iterations between progress() calls
    // (iteration, relative residual, relative residual after iteration 1)
    std::function<void(int, double, double)> progress;
    std::function<void()> checkCancel;              // may throw to stop
    std::function<void(const std::string &)> warn;  // e.g. SSOR unavailable
};

struct MassSolveResult
{
    int iterations = 0;
    double residual = 0.0;      // final ||r||_2 / ||b||_2
    bool converged = false;
    bool jacobi = false;        // the preconditioner actually used
};

/*
** Solves the system for Phi by conjugate gradient preconditioned with SSOR
** (Jacobi if asked, or if SSOR can't be set up), starting from phi.  The
** same arithmetic, in the same order, as ninja::solve(), which it replaced;
** the system isn't modified.
*/
MassSolveResult SolveMassPcg(const MassSystem &system, double *phi,
                             const MassSolveOptions &options);

#endif /* MASS_PCG_H */
