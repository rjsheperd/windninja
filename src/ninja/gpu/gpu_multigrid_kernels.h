/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  WGSL kernels of the GPU FAS multigrid
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

#ifndef GPU_MULTIGRID_KERNELS_H
#define GPU_MULTIGRID_KERNELS_H

/*
** Transfer kernels between a level (fine) and the next (coarse) of the GPU
** multigrid (as RansMultigrid).  The coarse grid keeps every second node of
** the fine one (plus the last), so along each direction a coarse cell has
** one or two fine children.  maps (i32) and weights (f32) hold, per
** direction, the index tables at the offsets in MgParams:
**   home  - for each fine index, the coarse cell containing it
**   lean  - for each fine index, the coarse neighbour its prolongation leans
**           towards, or -1
**   start - for each coarse index (and one past the end), its first fine child
**   w     - for each fine index, the weight on lean (0 or 1/4)
*/
namespace gpu {
namespace kernels {

constexpr const char *multigridPrelude = R"wgsl(
struct MgParams {
    fine : vec4<u32>,       // nx, ny, nz, nCells
    coarse : vec4<u32>,     // nx, ny, nz, nCells
    home : vec4<u32>,       // offsets into maps per direction (x, y, z)
    lean : vec4<u32>,
    start : vec4<u32>,
    w : vec4<u32>,          // offsets into weights
    damping : f32,
    pad0 : f32,
    pad1 : f32,
    pad2 : f32,
}

@group(0) @binding(0) var<uniform> M : MgParams;

fn threadIndex(gid : vec3<u32>, groups : vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 256u;
}

fn fineCell(i : u32, j : u32, k : u32) -> u32 {
    return i + M.fine.x * (j + M.fine.y * k);
}

fn coarseCell(i : u32, j : u32, k : u32) -> u32 {
    return i + M.coarse.x * (j + M.coarse.y * k);
}
)wgsl";

// Volume-weighted restriction of the 7 fields (p, u, v, w, k, epsilon, nut)
// to the coarse level, one thread per coarse cell; also keeps the restricted
// p, u, v, w as the correction's starting point (4 slots).
constexpr const char *restrictState = R"wgsl(
@group(0) @binding(1) var<storage, read> fineCells : array<f32>;
@group(0) @binding(2) var<storage, read> fineFields : array<f32>;
@group(0) @binding(3) var<storage, read_write> coarseFields : array<f32>;
@group(0) @binding(4) var<storage, read_write> start : array<f32>;
@group(0) @binding(5) var<storage, read> maps : array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let nc = M.coarse.w;
    let c = threadIndex(gid, groups);
    if (c >= nc) {
        return;
    }
    let I = c % M.coarse.x;
    let J = (c / M.coarse.x) % M.coarse.y;
    let K = c / (M.coarse.x * M.coarse.y);
    let nf = M.fine.w;
    var sum : array<f32, 7>;
    var vol = 0.0;
    for (var k = u32(maps[M.start.z + K]); k < u32(maps[M.start.z + K + 1u]); k++) {
        for (var j = u32(maps[M.start.y + J]); j < u32(maps[M.start.y + J + 1u]); j++) {
            for (var i = u32(maps[M.start.x + I]); i < u32(maps[M.start.x + I + 1u]); i++) {
                let f = fineCell(i, j, k);
                let V = fineCells[f];   // cell slot 0: volume
                vol += V;
                for (var s = 0u; s < 7u; s++) {
                    sum[s] += V * fineFields[s * nf + f];
                }
            }
        }
    }
    for (var s = 0u; s < 7u; s++) {
        let q = sum[s] / vol;
        coarseFields[s * nc + c] = q;
        if (s < 4u) {
            start[s * nc + c] = q;
        }
    }
}
)wgsl";

// FAS forcing on the coarse level, in place: the coarse residual was
// computed with the previous forcing F_old included, so
// F = sum of fine residuals - (residual - F_old).  4 slots (p, u, v, w).
constexpr const char *forcing = R"wgsl(
@group(0) @binding(1) var<storage, read> fineResiduals : array<f32>;
@group(0) @binding(2) var<storage, read> coarseResiduals : array<f32>;
@group(0) @binding(3) var<storage, read_write> force : array<f32>;
@group(0) @binding(4) var<storage, read> maps : array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let nc = M.coarse.w;
    let c = threadIndex(gid, groups);
    if (c >= nc) {
        return;
    }
    let I = c % M.coarse.x;
    let J = (c / M.coarse.x) % M.coarse.y;
    let K = c / (M.coarse.x * M.coarse.y);
    let nf = M.fine.w;
    var sum = vec4<f32>(0.0);
    for (var k = u32(maps[M.start.z + K]); k < u32(maps[M.start.z + K + 1u]); k++) {
        for (var j = u32(maps[M.start.y + J]); j < u32(maps[M.start.y + J + 1u]); j++) {
            for (var i = u32(maps[M.start.x + I]); i < u32(maps[M.start.x + I + 1u]); i++) {
                let f = fineCell(i, j, k);
                sum += vec4<f32>(fineResiduals[f], fineResiduals[nf + f],
                                 fineResiduals[2u * nf + f], fineResiduals[3u * nf + f]);
            }
        }
    }
    for (var s = 0u; s < 4u; s++) {
        force[s * nc + c] = sum[s] - (coarseResiduals[s * nc + c] - force[s * nc + c]);
    }
}
)wgsl";

// Trilinear (index-space) prolongation of the damped coarse correction
// (coarse state - start) onto the fine p, u, v, w, one thread per fine cell.
constexpr const char *prolong = R"wgsl(
@group(0) @binding(1) var<storage, read> coarseFields : array<f32>;
@group(0) @binding(2) var<storage, read> start : array<f32>;
@group(0) @binding(3) var<storage, read_write> fineFields : array<f32>;
@group(0) @binding(4) var<storage, read> maps : array<i32>;
@group(0) @binding(5) var<storage, read> weights : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let nf = M.fine.w;
    let f = threadIndex(gid, groups);
    if (f >= nf) {
        return;
    }
    let i = f % M.fine.x;
    let j = (f / M.fine.x) % M.fine.y;
    let k = f / (M.fine.x * M.fine.y);
    let ci = array<i32, 2>(maps[M.home.x + i], maps[M.lean.x + i]);
    let cj = array<i32, 2>(maps[M.home.y + j], maps[M.lean.y + j]);
    let ck = array<i32, 2>(maps[M.home.z + k], maps[M.lean.z + k]);
    let wi = array<f32, 2>(1.0 - weights[M.w.x + i], weights[M.w.x + i]);
    let wj = array<f32, 2>(1.0 - weights[M.w.y + j], weights[M.w.y + j]);
    let wk = array<f32, 2>(1.0 - weights[M.w.z + k], weights[M.w.z + k]);
    let nc = M.coarse.w;
    var d = vec4<f32>(0.0);
    for (var a = 0u; a < 2u; a++) {
        for (var b = 0u; b < 2u; b++) {
            for (var e = 0u; e < 2u; e++) {
                let wt = wi[a] * wj[b] * wk[e];
                if (wt == 0.0) {
                    continue;
                }
                let cc = coarseCell(u32(ci[a]), u32(cj[b]), u32(ck[e]));
                for (var s = 0u; s < 4u; s++) {
                    d[s] += wt * (coarseFields[s * nc + cc] - start[s * nc + cc]);
                }
            }
        }
    }
    for (var s = 0u; s < 4u; s++) {
        fineFields[s * nf + f] += M.damping * d[s];
    }
}
)wgsl";

} // namespace kernels
} // namespace gpu

#endif /* GPU_MULTIGRID_KERNELS_H */
