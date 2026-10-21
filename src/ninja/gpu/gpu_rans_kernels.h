/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  WGSL kernels of the GPU RANS solver
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

#ifndef GPU_RANS_KERNELS_H
#define GPU_RANS_KERNELS_H

/*
** WGSL for the GPU port of RansSolver (WEBGPU_PLAN 4.2).  Each kernel is the
** common prelude followed by its own source.  Stages are a face pass (one
** thread per face of one direction, writing per-face values) and a cell
** gather (one thread per cell, upper face minus lower face in each
** direction), so there are no write races and results are deterministic.
** Layouts are gpu_grid.h's slots.
*/
namespace gpu {
namespace kernels {

constexpr const char *ransPrelude = R"wgsl(
// boundary kinds, as gpu::RansSolver::Boundary::Type
const WALL = 0u;
const SLIP = 1u;
const VELOCITY_INLET = 2u;
const PRESSURE_OUTLET = 3u;
const LOG_PROFILE_INLET = 4u;

const KAPPA = 0.41;

struct Bc {
    kind : u32,
    p : f32,
    u : f32,
    v : f32,
    w : f32,
    ustar : f32,
    z0 : f32,
    pad : f32,
}

struct Params {
    nx : u32,
    ny : u32,
    nz : u32,
    dir : u32,       // face direction of a face pass
    nu : f32,
    beta : f32,
    pad0 : f32,
    pad1 : f32,
    bc : array<Bc, 6>,
}

@group(0) @binding(0) var<uniform> P : Params;

// cell and face slots (gpu_grid.h)
const VOLUME = 0u;
const CX = 1u;
const CY = 2u;
const CZ = 3u;
const SX = 0u;
const FX = 3u;
const AGL = 6u;
// field slots
const FP = 0u;
const FU = 1u;
const FV = 2u;
const FW = 3u;

fn nCells() -> u32 { return P.nx * P.ny * P.nz; }

fn cellIndex(i : u32, j : u32, k : u32) -> u32 {
    return i + P.nx * (j + P.ny * k);
}

// faces of direction d span (nx, ny, nz) plus one along d
fn faceExtent(d : u32) -> vec3<u32> {
    var e = vec3<u32>(P.nx, P.ny, P.nz);
    e[d] += 1u;
    return e;
}

fn nFaces(d : u32) -> u32 {
    let e = faceExtent(d);
    return e.x * e.y * e.z;
}

fn faceIndex(d : u32, ijk : vec3<u32>) -> u32 {
    let e = faceExtent(d);
    return ijk.x + e.x * (ijk.y + e.y * ijk.z);
}

// thread index over a 2-D dispatch of 256-thread workgroups
fn threadIndex(gid : vec3<u32>, groups : vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 256u;
}

// Face state (p, u, v, w) on a boundary face from the owner cell's state;
// n is the outward unit normal, agl the face's height above ground.  For
// gradients, so outlets extrapolate the velocity without the flux's
// backflow treatment (as BoundaryState(..., backflow = false)).
fn boundaryState(b : Bc, n : vec3<f32>, agl : f32, q : vec4<f32>) -> vec4<f32> {
    switch b.kind {
        case WALL, VELOCITY_INLET: {
            return vec4<f32>(q.x, b.u, b.v, b.w);
        }
        case SLIP: {
            let un = dot(q.yzw, n);
            return vec4<f32>(q.x, q.yzw - un * n);
        }
        case PRESSURE_OUTLET: {
            return vec4<f32>(b.p, q.yzw);
        }
        default: {   // LOG_PROFILE_INLET
            let speed = b.ustar / KAPPA * log(agl / b.z0);
            return vec4<f32>(q.x, speed * vec3<f32>(b.u, b.v, b.w));
        }
    }
}
)wgsl";

// Face pass of the Green-Gauss gradients: for each face of direction P.dir,
// the face value of (p, u, v, w) times the face's area vector S (as stored,
// pointing from lower to upper index), as 12 slots: field * 3 + component.
constexpr const char *gradientFaces = R"wgsl(
@group(0) @binding(1) var<storage, read> cells : array<f32>;
@group(0) @binding(2) var<storage, read> faces : array<f32>;
@group(0) @binding(3) var<storage, read> fields : array<f32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;

fn state(c : u32) -> vec4<f32> {
    let n = nCells();
    return vec4<f32>(fields[FP * n + c], fields[FU * n + c],
                     fields[FV * n + c], fields[FW * n + c]);
}

fn centroid(c : u32) -> vec3<f32> {
    let n = nCells();
    return vec3<f32>(cells[CX * n + c], cells[CY * n + c], cells[CZ * n + c]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let d = P.dir;
    let nf = nFaces(d);
    let f = threadIndex(gid, groups);
    if (f >= nf) {
        return;
    }
    let e = faceExtent(d);
    var ijk = vec3<u32>(f % e.x, (f / e.x) % e.y, f / (e.x * e.y));
    let S = vec3<f32>(faces[(SX + 0u) * nf + f], faces[(SX + 1u) * nf + f],
                      faces[(SX + 2u) * nf + f]);
    let F = vec3<f32>(faces[(FX + 0u) * nf + f], faces[(FX + 1u) * nf + f],
                      faces[(FX + 2u) * nf + f]);
    let m = ijk[d];
    let last = vec3<u32>(P.nx, P.ny, P.nz)[d];
    var value : vec4<f32>;
    if (m == 0u || m == last) {
        // boundary: the owner is the adjacent cell; the outward normal is -S
        // on the min side
        if (m == last) {
            ijk[d] = m - 1u;
        }
        let o = cellIndex(ijk.x, ijk.y, ijk.z);
        let side = 2u * d + select(0u, 1u, m == last);
        let n = select(-S, S, m == last) / length(S);
        value = boundaryState(P.bc[side], n, faces[AGL * nf + f], state(o));
    } else {
        // inverse-distance weights from the two centroids
        var lo = ijk;
        lo[d] = m - 1u;
        let o = cellIndex(lo.x, lo.y, lo.z);
        let nb = cellIndex(ijk.x, ijk.y, ijk.z);
        let dO = distance(F, centroid(o));
        let dN = distance(F, centroid(nb));
        let a = dN / (dO + dN);
        value = a * state(o) + (1.0 - a) * state(nb);
    }
    for (var q = 0u; q < 4u; q++) {
        for (var c = 0u; c < 3u; c++) {
            out[(q * 3u + c) * nf + f] = value[q] * S[c];
        }
    }
}
)wgsl";

// Cell gather of the gradients: (upper face - lower face) in each direction,
// over the cell volume; 12 slots (field * 3 + component) of nCells.
constexpr const char *gradientCells = R"wgsl(
@group(0) @binding(1) var<storage, read> cells : array<f32>;
@group(0) @binding(2) var<storage, read> face0 : array<f32>;
@group(0) @binding(3) var<storage, read> face1 : array<f32>;
@group(0) @binding(4) var<storage, read> face2 : array<f32>;
@group(0) @binding(5) var<storage, read_write> gradients : array<f32>;

fn faceValue(d : u32, slot : u32, f : u32) -> f32 {
    let i = slot * nFaces(d) + f;
    switch d {
        case 0u: { return face0[i]; }
        case 1u: { return face1[i]; }
        default: { return face2[i]; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = nCells();
    let c = threadIndex(gid, groups);
    if (c >= n) {
        return;
    }
    let ijk = vec3<u32>(c % P.nx, (c / P.nx) % P.ny, c / (P.nx * P.ny));
    let v = cells[VOLUME * n + c];
    for (var slot = 0u; slot < 12u; slot++) {
        var sum = 0.0;
        for (var d = 0u; d < 3u; d++) {
            var up = ijk;
            up[d] += 1u;
            sum += faceValue(d, slot, faceIndex(d, up)) - faceValue(d, slot, faceIndex(d, ijk));
        }
        gradients[slot * n + c] = sum / v;
    }
}
)wgsl";

} // namespace kernels
} // namespace gpu

#endif /* GPU_RANS_KERNELS_H */
