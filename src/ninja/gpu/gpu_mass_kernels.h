/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  WGSL for the mass solver on the GPU
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

#ifndef GPU_MASS_KERNELS_H
#define GPU_MASS_KERNELS_H

namespace gpu {
namespace kernels {

// y = A x for the mass solver's stencil (gpu::MassStencil): each node adds
// its own 14 coefficients (diagonal and the 13 neighbours after it) and, by
// symmetry, its 13 earlier neighbours' coefficients toward it.  A gather:
// no atomics, so results are deterministic.  256-thread workgroups over a
// 2-D dispatch.
constexpr const char *massApply = R"wgsl(
struct Params {
    ncols : u32,
    nrows : u32,
    nlayers : u32,
    nodes : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> coef : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read_write> y : array<f32>;

// (dcol, drow, dlayer) of slots 1..13 (gpu::massOffsets)
const offsets = array<vec3<i32>, 13>(
    vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 1, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(-1, -1, 1), vec3<i32>(0, -1, 1), vec3<i32>(1, -1, 1),
    vec3<i32>(-1, 0, 1), vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1),
    vec3<i32>(-1, 1, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1));

fn inside(p : vec3<i32>) -> bool {
    return all(p >= vec3<i32>(0)) &&
           all(p < vec3<i32>(i32(params.ncols), i32(params.nrows), i32(params.nlayers)));
}

fn node(p : vec3<i32>) -> u32 {
    return u32(p.x) + params.ncols * (u32(p.y) + params.nrows * u32(p.z));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = gid.x + gid.y * groups.x * 256u;
    if (n >= params.nodes) {
        return;
    }
    let layerSize = params.ncols * params.nrows;
    let p = vec3<i32>(i32(n % params.ncols), i32((n / params.ncols) % params.nrows),
                      i32(n / layerSize));
    var acc = coef[n] * x[n];
    for (var s = 0u; s < 13u; s++) {
        let up = p + offsets[s];
        if (inside(up)) {
            acc += coef[(s + 1u) * params.nodes + n] * x[node(up)];
        }
        let down = p - offsets[s];
        if (inside(down)) {
            let m = node(down);
            acc += coef[(s + 1u) * params.nodes + m] * x[m];
        }
    }
    y[n] = acc;
}
)wgsl";

} // namespace kernels
} // namespace gpu

#endif /* GPU_MASS_KERNELS_H */
