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

// Jacobi-preconditioned conjugate gradient (gpu::MassPcg).  Each stage is
// massPcgPrelude + the stage.  scalars holds rho = z.r, p.q, alpha, beta and
// r.r (MassPcg's slots); the GPU computes alpha and beta itself, so the host
// reads back only r.r, every few iterations.  Dot products are per-workgroup
// tree sums into partials (one per workgroup, slot 1 at params.groups),
// finished by one workgroup: fixed order, so results are deterministic.
constexpr const char *massPcgPrelude = R"wgsl(
struct Params {
    ncols : u32,
    nrows : u32,
    nlayers : u32,
    nodes : u32,
    groups : u32,   // workgroups in the node dispatch = partials per slot
    pad0 : u32,
    pad1 : u32,
    pad2 : u32,
}

@group(0) @binding(0) var<uniform> params : Params;

const RHO = 0u;
const PQ = 1u;
const ALPHA = 2u;
const BETA = 3u;
const RR = 4u;

var<workgroup> scratchA : array<f32, 256>;
var<workgroup> scratchB : array<f32, 256>;

fn threadIndex(gid : vec3<u32>, groups : vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 256u;
}

fn groupIndex(wid : vec3<u32>, groups : vec3<u32>) -> u32 {
    return wid.x + wid.y * groups.x;
}

// Tree sums of scratchA and scratchB over the workgroup; call from uniform
// control flow after every thread has stored its values.
fn treeSum(lid : u32) {
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratchA[lid] += scratchA[lid + s];
            scratchB[lid] += scratchB[lid + s];
        }
        workgroupBarrier();
    }
}
)wgsl";

// p = z + beta p
constexpr const char *massPcgDirection = R"wgsl(
@group(0) @binding(1) var<storage, read> scalars : array<f32>;
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read_write> p : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = threadIndex(gid, groups);
    if (n < params.nodes) {
        p[n] = z[n] + scalars[BETA] * p[n];
    }
}
)wgsl";

// q = A p (as massApply) and partials of p.q
constexpr const char *massPcgApply = R"wgsl(
@group(0) @binding(1) var<storage, read> coef : array<f32>;
@group(0) @binding(2) var<storage, read> p : array<f32>;
@group(0) @binding(3) var<storage, read_write> q : array<f32>;
@group(0) @binding(4) var<storage, read_write> partials : array<f32>;

const offsets = array<vec3<i32>, 13>(
    vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 1, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(-1, -1, 1), vec3<i32>(0, -1, 1), vec3<i32>(1, -1, 1),
    vec3<i32>(-1, 0, 1), vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1),
    vec3<i32>(-1, 1, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1));

fn inside(c : vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) &&
           all(c < vec3<i32>(i32(params.ncols), i32(params.nrows), i32(params.nlayers)));
}

fn node(c : vec3<i32>) -> u32 {
    return u32(c.x) + params.ncols * (u32(c.y) + params.nrows * u32(c.z));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32,
        @builtin(workgroup_id) wid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = threadIndex(gid, groups);
    var pq = 0.0;
    if (n < params.nodes) {
        let layerSize = params.ncols * params.nrows;
        let c = vec3<i32>(i32(n % params.ncols), i32((n / params.ncols) % params.nrows),
                          i32(n / layerSize));
        var acc = coef[n] * p[n];
        for (var s = 0u; s < 13u; s++) {
            let up = c + offsets[s];
            if (inside(up)) {
                acc += coef[(s + 1u) * params.nodes + n] * p[node(up)];
            }
            let down = c - offsets[s];
            if (inside(down)) {
                let m = node(down);
                acc += coef[(s + 1u) * params.nodes + m] * p[m];
            }
        }
        q[n] = acc;
        pq = p[n] * acc;
    }
    scratchA[lid] = pq;
    scratchB[lid] = 0.0;
    treeSum(lid);
    if (lid == 0u) {
        partials[groupIndex(wid, groups)] = scratchA[0];
    }
}
)wgsl";

// p.q from the partials, alpha = rho / p.q (one workgroup)
constexpr const char *massPcgAlpha = R"wgsl(
@group(0) @binding(1) var<storage, read> partials : array<f32>;
@group(0) @binding(2) var<storage, read_write> scalars : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid : u32) {
    var acc = 0.0;
    for (var i = lid; i < params.groups; i += 256u) {
        acc += partials[i];
    }
    scratchA[lid] = acc;
    scratchB[lid] = 0.0;
    treeSum(lid);
    if (lid == 0u) {
        scalars[PQ] = scratchA[0];
        scalars[ALPHA] = scalars[RHO] / scratchA[0];
    }
}
)wgsl";

// x += alpha p, r -= alpha q, z = r / diag; partials of z.r and r.r.  With
// params.pad0 = 1 it is the start instead: r = b - q (q = A x), z = r / diag.
constexpr const char *massPcgUpdate = R"wgsl(
@group(0) @binding(1) var<storage, read> scalars : array<f32>;
@group(0) @binding(2) var<storage, read_write> x : array<f32>;
@group(0) @binding(3) var<storage, read> p : array<f32>;
@group(0) @binding(4) var<storage, read> q : array<f32>;
@group(0) @binding(5) var<storage, read_write> r : array<f32>;
@group(0) @binding(6) var<storage, read> invDiag : array<f32>;
@group(0) @binding(7) var<storage, read_write> z : array<f32>;
@group(0) @binding(8) var<storage, read_write> partials : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32,
        @builtin(workgroup_id) wid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = threadIndex(gid, groups);
    var zr = 0.0;
    var rr = 0.0;
    if (n < params.nodes) {
        var rn : f32;
        if (params.pad0 == 1u) {
            rn = p[n] - q[n];   // start: p holds b
        } else {
            let alpha = scalars[ALPHA];
            x[n] = x[n] + alpha * p[n];
            rn = r[n] - alpha * q[n];
        }
        r[n] = rn;
        let zn = invDiag[n] * rn;
        z[n] = zn;
        zr = zn * rn;
        rr = rn * rn;
    }
    scratchA[lid] = zr;
    scratchB[lid] = rr;
    treeSum(lid);
    if (lid == 0u) {
        let g = groupIndex(wid, groups);
        partials[g] = scratchA[0];
        partials[params.groups + g] = scratchB[0];
    }
}
)wgsl";

// z.r and r.r from the partials; beta = new rho / old rho (one workgroup)
constexpr const char *massPcgBeta = R"wgsl(
@group(0) @binding(1) var<storage, read> partials : array<f32>;
@group(0) @binding(2) var<storage, read_write> scalars : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid : u32) {
    var a = 0.0;
    var b = 0.0;
    for (var i = lid; i < params.groups; i += 256u) {
        a += partials[i];
        b += partials[params.groups + i];
    }
    scratchA[lid] = a;
    scratchB[lid] = b;
    treeSum(lid);
    if (lid == 0u) {
        scalars[BETA] = scratchA[0] / scalars[RHO];
        scalars[RHO] = scratchA[0];
        scalars[RR] = scratchB[0];
    }
}
)wgsl";

// Builds the mass system on the GPU (gpu::MassAssembler), as AssembleMass
// does on the CPU: one-point-quadrature hexahedra, the stencil's 14 slots
// and the right-hand side, then Phi = 0 rows and columns on the sides and
// top.  A gather: each node visits its (up to 8) elements, recomputes each
// element's centre Jacobian and adds the element's terms for its own row,
// keeping only couplings to itself and later nodes (the slots it owns).
// Coordinates are relative to a local origin (f32).
constexpr const char *massAssemble = R"wgsl(
struct Params {
    ncols : u32,
    nrows : u32,
    nlayers : u32,
    nodes : u32,
    alphaH : f32,
    pad0 : u32,
    pad1 : u32,
    pad2 : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> xyz : array<f32>;     // x, y, z slots
@group(0) @binding(2) var<storage, read> wind0 : array<f32>;   // u0, v0, w0 slots
@group(0) @binding(3) var<storage, read> alphaV : array<f32>;
@group(0) @binding(4) var<storage, read_write> coef : array<f32>;
@group(0) @binding(5) var<storage, read_write> rhs : array<f32>;

// local corner k of an element: (col, row, layer) offset from its origin
const corner = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(1, 1, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(1, 1, 1), vec3<i32>(0, 1, 1));

// slot of the coupling to the node at offset d (index (d.x+1) + 3 (d.y+1)
// + 9 (d.z+1)): 0 the diagonal, 1..13 gpu::massOffsets, -1 an earlier node
const slotOf = array<i32, 27>(
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, 0, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13);

const offsets = array<vec3<i32>, 13>(
    vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 1, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(-1, -1, 1), vec3<i32>(0, -1, 1), vec3<i32>(1, -1, 1),
    vec3<i32>(-1, 0, 1), vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1),
    vec3<i32>(-1, 1, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1));

fn node(c : vec3<i32>) -> u32 {
    return u32(c.x) + params.ncols * (u32(c.y) + params.nrows * u32(c.z));
}

fn boundary(c : vec3<i32>) -> bool {
    return c.x == 0 || c.x == i32(params.ncols) - 1 || c.y == 0 ||
           c.y == i32(params.nrows) - 1 || c.z == i32(params.nlayers) - 1;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = gid.x + gid.y * groups.x * 256u;
    if (n >= params.nodes) {
        return;
    }
    let N = params.nodes;
    let layerSize = params.ncols * params.nrows;
    let p = vec3<i32>(i32(n % params.ncols), i32((n / params.ncols) % params.nrows),
                      i32(n / layerSize));
    var row : array<f32, 14>;
    for (var s = 0u; s < 14u; s++) {
        row[s] = 0.0;
    }
    var q = 0.0;
    let rx = 1.0 / (2.0 * params.alphaH * params.alphaH);

    if (!boundary(p)) {
        let cells = vec3<i32>(i32(params.ncols) - 1, i32(params.nrows) - 1, i32(params.nlayers) - 1);
        for (var e = 0u; e < 8u; e++) {
            let origin = p - corner[e];   // the element in which this node is corner e
            if (any(origin < vec3<i32>(0)) || any(origin >= cells)) {
                continue;
            }
            var J = mat3x3<f32>();   // columns: d/du, d/dv, d/dw of (x, y, z)
            var hv = 0.0;
            var av = 0.0;
            var ids : array<u32, 8>;
            for (var k = 0u; k < 8u; k++) {
                ids[k] = node(origin + corner[k]);
                let s = (vec3<f32>(corner[k]) * 2.0 - 1.0) * 0.125;   // dN/du, dN/dv, dN/dw
                let c = vec3<f32>(xyz[ids[k]], xyz[N + ids[k]], xyz[2u * N + ids[k]]);
                J[0] += s.x * c;
                J[1] += s.y * c;
                J[2] += s.z * c;
            }
            let detJ = determinant(J);
            let inv = transpose(mat3x3<f32>(cross(J[1], J[2]), cross(J[2], J[0]),
                                            cross(J[0], J[1]))) * (1.0 / detJ);   // inverse of J
            // dN/dx, dN/dy, dN/dz of corner k = s^T inv
            var g : array<vec3<f32>, 8>;
            for (var k = 0u; k < 8u; k++) {
                let s = (vec3<f32>(corner[k]) * 2.0 - 1.0) * 0.125;
                g[k] = s * inv;
                let m = ids[k];
                hv += g[k].x * wind0[m] + g[k].y * wind0[N + m] + g[k].z * wind0[2u * N + m];
                av += 0.125 * alphaV[m];
            }
            let R = vec3<f32>(rx, rx, 1.0 / (2.0 * av * av));
            q += 8.0 * 0.125 * hv * detJ;
            for (var k = 0u; k < 8u; k++) {
                let d = corner[k] - corner[e];
                let s = slotOf[(d.x + 1) + 3 * (d.y + 1) + 9 * (d.z + 1)];
                if (s >= 0) {
                    row[s] += 8.0 * dot(g[e] * R, g[k]) * detJ;
                }
            }
        }
        // Phi = 0 on boundary nodes: their columns go
        for (var s = 0u; s < 13u; s++) {
            if (boundary(p + offsets[s])) {
                row[s + 1u] = 0.0;
            }
        }
    } else {
        row[0] = 1.0;   // Phi = 0 rows: identity, zero right-hand side
    }
    for (var s = 0u; s < 14u; s++) {
        coef[s * N + n] = row[s];
    }
    rhs[n] = q;
}
)wgsl";

} // namespace kernels
} // namespace gpu

#endif /* GPU_MASS_KERNELS_H */
