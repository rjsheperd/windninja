/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  The mass solver's 27-point stencil operator on the GPU
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

#ifndef GPU_MASS_H
#define GPU_MASS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "gpu_context.h"

namespace gpu {

/*
** The mass solver's matrix as a stencil: 14 coefficients per node, slot 0
** the diagonal and slot s the coupling to the node at +massOffsets[s - 1]
** (the 13 neighbours after it in node order).  A node's coupling to the 13
** before it is the neighbour's slot, since the matrix is symmetric.  Slot s
** of node n is at s * nodes + n.  Nodes are n = col + ncols * (row + nrows
** * layer), as in MassGrid.
*/
struct MassStencil
{
    int ncols = 0, nrows = 0, nlayers = 0;
    std::vector<float> coef;   // 14 * nodes

    std::size_t nodes() const { return (std::size_t)ncols * nrows * nlayers; }
    static const int slots = 14;
};

// (dcol, drow, dlayer) of the 13 upper neighbours
extern const int massOffsets[13][3];

/*
** Packs the upper-triangle CSR of the mass system (MassSystem's SK,
** col_ind, row_ptr) into a stencil.  Throws Error for an entry outside the
** 27-point neighbourhood.
*/
MassStencil PackMassStencil(int ncols, int nrows, int nlayers, const double *SK,
                            const int *colInd, const int *rowPtr);

/* y = A x on the GPU, one thread per node gathering its 27 couplings. */
class MassOperator
{
public:
    MassOperator(Context &context, const MassStencil &stencil);

    std::size_t nodes() const { return nodes_; }
    const Buffer &coefficients() const { return coef_; }

    // x and y hold at least nodes() floats; queued, not waited on
    void apply(const Buffer &x, Buffer &y);

private:
    Context &context_;
    std::size_t nodes_;
    Buffer params_, coef_;
    Kernel kernel_;
};

/*
** Jacobi-preconditioned conjugate gradient for the mass system on the GPU,
** in f32.  alpha and beta stay on the GPU; the host reads back r.r every
** checkEvery iterations to test ||r|| / ||b|| <= tol, so a converged solve
** can run up to checkEvery - 1 iterations past the tolerance.
**
** f32 limit (MASS_GPU_PLAN.org 2.3): the true residual follows the
** reported one down to about 1e-2 and stalls near 1e-3 (Big Butte fine),
** while the reported one keeps falling.  WindNinja stops at 0.1.  For
** tighter tolerances, refine: residual in double on the CPU, correction
** solved here.
*/
class MassPcg
{
public:
    MassPcg(Context &context, const MassStencil &stencil);

    struct Result
    {
        int iterations = 0;
        double residual = 0.0;   // ||r|| / ||b|| from the GPU's r.r
        bool converged = false;
    };

    // Solves A x = b starting from x; throws Error if r.r stops being finite.
    // onCheck(iteration, residual) runs at each residual check and may
    // throw to stop.
    Result solve(const std::vector<double> &b, std::vector<double> &x, double tol,
                 int maxIterations, int checkEvery = 10,
                 const std::function<void(int, double)> &onCheck = nullptr);

private:
    void dispatchNodes(Kernel &kernel, const std::vector<const Buffer *> &bindings);
    double residualSquared();

    Context &context_;
    std::size_t nodes_;
    uint32_t gx_, gy_;
    MassOperator op_;
    Buffer params_, startParams_, invDiag_, x_, r_, z_, p_, q_, scalars_, partials_;
    Kernel direction_, apply_, alpha_, update_, beta_;
};

/*
** The mass mesh's node coordinates for the GPU: x, y, z slots (slot s of
** node n at s * nodes + n), in f32 relative to a local origin (the first
** node's x and y, the lowest z) so that elevations keep sub-millimetre
** precision.
*/
struct MassGeometry
{
    int ncols = 0, nrows = 0, nlayers = 0;
    std::vector<float> xyz;   // 3 * nodes

    std::size_t nodes() const { return (std::size_t)ncols * nrows * nlayers; }
};

MassGeometry MakeMassGeometry(int ncols, int nrows, int nlayers, const double *x,
                              const double *y, const double *z);

/*
** Builds the mass system on the GPU (kernels::massAssemble), as AssembleMass
** does on the CPU, straight into MassOperator / MassPcg's stencil layout.
*/
class MassAssembler
{
public:
    MassAssembler(Context &context, const MassGeometry &geometry);

    std::size_t nodes() const { return nodes_; }
    const Buffer &geometry() const { return xyz_; }

    // wind0: u0, v0, w0 slots (3 * nodes floats); alphaV: nodes floats;
    // coef: MassStencil::slots * nodes floats; rhs: nodes floats.  Queued.
    void assemble(const Buffer &wind0, const Buffer &alphaV, float alphaH, Buffer &coef,
                  Buffer &rhs);

private:
    Context &context_;
    std::size_t nodes_;
    int ncols_, nrows_, nlayers_;
    Buffer params_, xyz_;
    Kernel kernel_;
};

} // namespace gpu

#endif /* GPU_MASS_H */
