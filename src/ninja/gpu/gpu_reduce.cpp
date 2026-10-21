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

#include "gpu_reduce.h"

#include <algorithm>
#include <cstdint>

#include "gpu_kernels.h"

namespace gpu {

namespace {

struct Params
{
    uint32_t offset, count, square, pad;
};

} // namespace

Reducer::Reducer(Context &context)
    : context_(context), kernel_(context, kernels::reduceSum),
      params_{Buffer(context, sizeof(Params), Buffer::Uniform),
              Buffer(context, sizeof(Params), Buffer::Uniform)},
      partials_(context, maxGroups * sizeof(float)),
      result_(context, sizeof(float))
{
}

double Reducer::sum(const Buffer &data, std::size_t offset, std::size_t count,
                    bool squares)
{
    if(count == 0)
        return 0.0;
    if((offset + count) * sizeof(float) > data.size() || offset + count > UINT32_MAX)
        throw Error("gpu::Reducer::sum: range outside the buffer");
    uint32_t groups = static_cast<uint32_t>(
        std::min<std::size_t>(maxGroups, (count + 255) / 256));
    Params first = {static_cast<uint32_t>(offset), static_cast<uint32_t>(count),
                    squares ? 1u : 0u, 0u};
    Params second = {0u, groups, 0u, 0u};
    params_[0].upload(&first, sizeof first);
    params_[1].upload(&second, sizeof second);
    kernel_.dispatch({&params_[0], &data, &partials_}, groups);
    kernel_.dispatch({&params_[1], &partials_, &result_}, 1);
    float total;
    result_.download(&total, sizeof total);
    return total;
}

} // namespace gpu
