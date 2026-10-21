/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  Sums and residual norms over GPU buffers
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

#ifndef GPU_REDUCE_H
#define GPU_REDUCE_H

#include <cstddef>

#include "gpu_context.h"

namespace gpu {

/*
** Two-pass sum of a range of floats on the GPU (see kernels::reduceSum).
** Each call reads one float back, so it waits on the GPU: use it for
** residual norms every N iterations, not every step.
*/
class Reducer
{
public:
    explicit Reducer(Context &context);

    // Sum of count floats (or of their squares) of data from float index
    // offset.
    double sum(const Buffer &data, std::size_t offset, std::size_t count,
               bool squares = false);

    static const unsigned maxGroups = 4096;

private:
    Context &context_;
    Kernel kernel_;
    Buffer params_[2], partials_, result_;
};

} // namespace gpu

#endif /* GPU_REDUCE_H */
