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

// Jacobi-preconditioned conjugate gradient (gpu::MassPcg), three dispatches
// an iteration: direction (p = z + beta p), apply (q = A p) and update (x,
// r, z).  Each stage is massPcgPrelude + the stage.  Dot products are
// per-workgroup tree sums into partials, one per workgroup per slot: p.q
// (slot 0), r.r (slot 1) and z.r (slots 2 and 3, alternating by iteration,
// so the previous rho survives until the next direction).  The stages that
// need a scalar (beta, alpha) finish those sums themselves, every workgroup
// in the same fixed order, so all agree and results are deterministic; the
// host reads only r.r, every few iterations.
constexpr const char *massPcgPrelude = R"wgsl(
struct Params {
    ncols : u32,
    nrows : u32,
    nlayers : u32,
    nodes : u32,
    groups : u32,   // workgroups in the node dispatch = partials per slot
    start : u32,    // update: the start (r = b - A x) instead of an iteration
    current : u32,  // this iteration's z.r slot is 2 + current; the last one's 2 + (1 - current)
    first : u32,    // direction: the first iteration (beta = 0)
}

@group(0) @binding(0) var<uniform> params : Params;

const PQ = 0u;
const RR = 1u;
const ZR = 2u;

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

// The totals of partials slots a and b (each stage declares partials), in
// the same order in every workgroup; call from uniform control flow.
fn sumSlots(a : u32, b : u32, lid : u32) -> vec2<f32> {
    var x = 0.0;
    var y = 0.0;
    for (var i = lid; i < params.groups; i += 256u) {
        x += partials[a * params.groups + i];
        y += partials[b * params.groups + i];
    }
    scratchA[lid] = x;
    scratchB[lid] = y;
    treeSum(lid);
    let total = vec2<f32>(scratchA[0], scratchB[0]);
    workgroupBarrier();   // before scratch is reused
    return total;
}
)wgsl";

// p = z + beta p, beta = (this rho) / (the last one), 0 the first time
constexpr const char *massPcgDirection = R"wgsl(
@group(0) @binding(1) var<storage, read> partials : array<f32>;
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read_write> p : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32,
        @builtin(num_workgroups) groups : vec3<u32>) {
    var beta = 0.0;
    if (params.first == 0u) {
        // rho from the last update (slot 1 - current), the one before (current)
        let rho = sumSlots(ZR + 1u - params.current, ZR + params.current, lid);
        beta = rho.x / rho.y;
    }
    let n = threadIndex(gid, groups);
    if (n < params.nodes) {
        p[n] = z[n] + beta * p[n];
    }
}
)wgsl";

// q = A p (as massApply) and partials of p.q (slot PQ)
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
        partials[PQ * params.groups + groupIndex(wid, groups)] = scratchA[0];
    }
}
)wgsl";

// alpha = rho / p.q; x += alpha p, r -= alpha q, z = r / diag (the stencil's
// slot 0); partials of z.r (slot 2 + current) and r.r.  With params.start it
// is the start instead: r = b - q (q = A x, p holds b), z = r / diag.
constexpr const char *massPcgUpdate = R"wgsl(
@group(0) @binding(1) var<storage, read_write> x : array<f32>;
@group(0) @binding(2) var<storage, read> p : array<f32>;
@group(0) @binding(3) var<storage, read> q : array<f32>;
@group(0) @binding(4) var<storage, read_write> r : array<f32>;
@group(0) @binding(5) var<storage, read> coef : array<f32>;   // slot 0: the diagonal
@group(0) @binding(6) var<storage, read_write> z : array<f32>;
@group(0) @binding(7) var<storage, read_write> partials : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32,
        @builtin(workgroup_id) wid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    var alpha = 0.0;
    if (params.start == 0u) {
        // rho from the last update (slot 1 - current) and this p.q
        let s = sumSlots(ZR + 1u - params.current, PQ, lid);
        alpha = s.x / s.y;
    }
    let n = threadIndex(gid, groups);
    var zr = 0.0;
    var rr = 0.0;
    if (n < params.nodes) {
        var rn : f32;
        if (params.start == 1u) {
            rn = p[n] - q[n];
        } else {
            x[n] = x[n] + alpha * p[n];
            rn = r[n] - alpha * q[n];
        }
        r[n] = rn;
        let zn = rn / coef[n];
        z[n] = zn;
        zr = zn * rn;
        rr = rn * rn;
    }
    scratchA[lid] = zr;
    scratchB[lid] = rr;
    treeSum(lid);
    if (lid == 0u) {
        let g = groupIndex(wid, groups);
        partials[(ZR + params.current) * params.groups + g] = scratchA[0];
        partials[RR * params.groups + g] = scratchB[0];
    }
}
)wgsl";

// Shared by massAssemble and massRecover (gpu::MassFem): the grid, the
// node coordinates (x, y, z slots, f32 from a local origin) and one
// element's geometry at its centre (one-point quadrature, as element.cpp).
constexpr const char *massFemPrelude = R"wgsl(
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
@group(0) @binding(1) var<storage, read> xyz : array<f32>;

// local corner k of an element: (col, row, layer) offset from its origin
const corner = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(1, 1, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(1, 1, 1), vec3<i32>(0, 1, 1));

fn node(c : vec3<i32>) -> u32 {
    return u32(c.x) + params.ncols * (u32(c.y) + params.nrows * u32(c.z));
}

fn position(n : u32) -> vec3<i32> {
    let layerSize = params.ncols * params.nrows;
    return vec3<i32>(i32(n % params.ncols), i32((n / params.ncols) % params.nrows),
                     i32(n / layerSize));
}

fn coords(n : u32) -> vec3<f32> {
    return vec3<f32>(xyz[n], xyz[params.nodes + n], xyz[2u * params.nodes + n]);
}

// the element with this origin exists
fn isElement(origin : vec3<i32>) -> bool {
    let cells = vec3<i32>(i32(params.ncols) - 1, i32(params.nrows) - 1, i32(params.nlayers) - 1);
    return all(origin >= vec3<i32>(0)) && all(origin < cells);
}

struct Element {
    ids : array<u32, 8>,             // global nodes of the corners
    g : array<vec3<f32>, 8>,         // dN/dx, dN/dy, dN/dz of each corner
    detJ : f32,
    centre : vec3<f32>,
}

fn element(origin : vec3<i32>) -> Element {
    var el : Element;
    var J = mat3x3<f32>();   // columns: d/du, d/dv, d/dw of (x, y, z)
    el.centre = vec3<f32>(0.0);
    for (var k = 0u; k < 8u; k++) {
        el.ids[k] = node(origin + corner[k]);
        let s = (vec3<f32>(corner[k]) * 2.0 - 1.0) * 0.125;   // dN/du, dN/dv, dN/dw
        let c = coords(el.ids[k]);
        J[0] += s.x * c;
        J[1] += s.y * c;
        J[2] += s.z * c;
        el.centre += 0.125 * c;
    }
    el.detJ = determinant(J);
    let inv = transpose(mat3x3<f32>(cross(J[1], J[2]), cross(J[2], J[0]),
                                    cross(J[0], J[1]))) * (1.0 / el.detJ);   // inverse of J
    for (var k = 0u; k < 8u; k++) {
        el.g[k] = ((vec3<f32>(corner[k]) * 2.0 - 1.0) * 0.125) * inv;   // s^T J^-1
    }
    return el;
}
)wgsl";

// Builds the mass system on the GPU (gpu::MassFem::assemble), as
// AssembleMass does on the CPU: the stencil's 14 slots and the right-hand
// side, then Phi = 0 rows and columns on the sides and top.  A gather: each
// node visits its (up to 8) elements, recomputes each element's centre
// geometry and adds the element's terms for its own row, keeping only
// couplings to itself and later nodes (the slots it owns).  An element with
// DETJ <= 0 (folded) sets the folded flag; AssembleMass throws for it.
constexpr const char *massAssemble = R"wgsl(
@group(0) @binding(2) var<storage, read> wind0 : array<f32>;   // u0, v0, w0 slots
@group(0) @binding(3) var<storage, read> alphaV : array<f32>;
@group(0) @binding(4) var<storage, read_write> coef : array<f32>;
@group(0) @binding(5) var<storage, read_write> rhs : array<f32>;
@group(0) @binding(6) var<storage, read_write> folded : atomic<u32>;   // set: some DETJ <= 0

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
    let p = position(n);
    var row : array<f32, 14>;
    for (var s = 0u; s < 14u; s++) {
        row[s] = 0.0;
    }
    var q = 0.0;
    let rx = 1.0 / (2.0 * params.alphaH * params.alphaH);

    if (!boundary(p)) {
        for (var e = 0u; e < 8u; e++) {
            let origin = p - corner[e];   // the element in which this node is corner e
            if (!isElement(origin)) {
                continue;
            }
            let el = element(origin);
            if (!(el.detJ > 0.0)) {
                atomicStore(&folded, 1u);
            }
            var hv = 0.0;
            var av = 0.0;
            for (var k = 0u; k < 8u; k++) {
                let m = el.ids[k];
                hv += el.g[k].x * wind0[m] + el.g[k].y * wind0[N + m] + el.g[k].z * wind0[2u * N + m];
                av += 0.125 * alphaV[m];
            }
            let R = vec3<f32>(rx, rx, 1.0 / (2.0 * av * av));
            q += 8.0 * 0.125 * hv * el.detJ;
            for (var k = 0u; k < 8u; k++) {
                let d = corner[k] - corner[e];
                let s = slotOf[(d.x + 1) + 3 * (d.y + 1) + 9 * (d.z + 1)];
                if (s >= 0) {
                    row[s] += 8.0 * dot(el.g[e] * R, el.g[k]) * el.detJ;
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

// The wind from Phi (gpu::MassFem::recover), as RecoverVelocity: each node
// averages its elements' centre gradients of Phi, weighted by the inverse
// distance from the node to each centre, then
//   u = u0 + dPhi/dx / (2 alphaH^2), v likewise, w = w0 + dPhi/dz / (2 alphaV^2).
constexpr const char *massRecover = R"wgsl(
@group(0) @binding(2) var<storage, read> phi : array<f32>;
@group(0) @binding(3) var<storage, read> wind0 : array<f32>;   // u0, v0, w0 slots
@group(0) @binding(4) var<storage, read> alphaV : array<f32>;
@group(0) @binding(5) var<storage, read_write> wind : array<f32>;   // u, v, w slots

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = gid.x + gid.y * groups.x * 256u;
    if (n >= params.nodes) {
        return;
    }
    let N = params.nodes;
    let p = position(n);
    let here = coords(n);
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var e = 0u; e < 8u; e++) {
        let origin = p - corner[e];
        if (!isElement(origin)) {
            continue;
        }
        let el = element(origin);
        var grad = vec3<f32>(0.0);
        for (var k = 0u; k < 8u; k++) {
            grad += el.g[k] * phi[el.ids[k]];
        }
        let w = 1.0 / distance(here, el.centre);
        sum += w * grad;
        weights += w;
    }
    let d = sum / weights;
    let av = alphaV[n];
    let rx = 1.0 / (2.0 * params.alphaH * params.alphaH);
    wind[n] = wind0[n] + rx * d.x;
    wind[N + n] = wind0[N + n] + rx * d.y;
    wind[2u * N + n] = wind0[2u * N + n] + 1.0 / (2.0 * av * av) * d.z;
}
)wgsl";

// WindNinja's output wind on the GPU (gpu::MassFem::outputWind), as
// OutputWind / ninja::interp_uvw: one thread per DEM cell finds the layer
// holding outputHeight + the canopy, then the neutral Monin-Obukhov profile
// (windProfile::getWindSpeed, L = 9999) within the first layer, linear
// between layers above.  Writes speed (slot 0) and the direction the wind
// comes from, degrees from the grid's north (slot 1), per cell, row 0 south
// or (northFirst) north.
constexpr const char *massOutputWind = R"wgsl(
struct Params {
    ncols : u32,
    nrows : u32,
    nlayers : u32,
    northFirst : u32,
    outputHeight : f32,
    roughness : f32,   // z0
    roughH : f32,      // canopy height
    roughD : f32,      // displacement height
    blHeight : f32,    // boundary-layer height
    pad0 : u32,
    pad1 : u32,
    pad2 : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> agl : array<f32>;    // height above the ground node
@group(0) @binding(2) var<storage, read> wind : array<f32>;   // u, v, w slots
@group(0) @binding(3) var<storage, read_write> out : array<f32>;   // speed, angle slots

const PI = 3.141592653589793;
const L = 9999.0;   // neutral

// windProfile::stability_function, stable branch (L > 0)
fn psi(zOverL : f32) -> f32 {
    return -17.0 * (1.0 - exp(-0.29 * zOverL));
}

// windProfile::monin_obukov
fn moninObukhov(zIn : f32, u1 : f32, z1 : f32, z0 : f32) -> f32 {
    var z = zIn;
    if (z / z0 < 1.0) {
        z = z0;
    }
    return u1 * (log(z / z0) - psi(z / L)) / (log(z1 / z0) - psi(z1 / L));
}

// windProfile::getWindSpeed (monin_obukov_similarity) at agl, for the speed
// u at inputHeight above the canopy
fn profile(agl : f32, u : f32, inputHeight : f32) -> f32 {
    if (agl == 0.0) {
        return 0.0;
    }
    let z0 = params.roughness;
    let d = params.roughD;
    let inwindheight = (inputHeight + params.roughH) - d;
    if (inwindheight < 3.0 * z0) {
        return u * (agl / (inwindheight + d));
    }
    if (agl < d + 7.0 * z0) {
        return moninObukhov(7.0 * z0, u, inwindheight, z0) * (agl / (7.0 * z0 + d));
    }
    if (agl < d + params.blHeight) {
        return moninObukhov(agl - d, u, inwindheight, z0);
    }
    if (params.blHeight <= 0.0) {
        return 0.0;
    }
    return moninObukhov(params.blHeight, u, inwindheight, z0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let column = gid.x + gid.y * groups.x * 256u;
    let layer = params.ncols * params.nrows;
    if (column >= layer) {
        return;
    }
    let N = layer * params.nlayers;
    let height = params.outputHeight + params.roughH;
    var k = 1u;
    var h2 = 0.0;
    var h1 = 0.0;
    loop {
        if (h2 >= height || k >= params.nlayers) {
            break;
        }
        h2 = agl[column + layer * k];
        h1 = agl[column + layer * (k - 1u)];
        k++;
    }
    k--;
    let at = column + layer * k;
    let below = column + layer * (k - 1u);
    var uu : f32;
    var vv : f32;
    if (k <= 1u) {   // within the first layer: the profile
        let inputHeight = h2 - params.roughH;
        uu = profile(height, wind[at], inputHeight);
        vv = profile(height, wind[N + at], inputHeight);
    } else {   // linear between layers k - 1 and k
        let su = (wind[at] - wind[below]) / (h2 - h1);
        let sv = (wind[N + at] - wind[N + below]) / (h2 - h1);
        uu = su * height + wind[below] - su * h1;
        vv = sv * height + wind[N + below] - sv * h1;
    }
    var a = 0.0;
    if (!(uu == 0.0 && vv == 0.0)) {
        a = -atan2(uu, -vv);
    }
    if (a < 0.0) {
        a += 2.0 * PI;
    }
    var cell = column;
    if (params.northFirst != 0u) {
        let row = column / params.ncols;
        cell = (params.nrows - 1u - row) * params.ncols + column % params.ncols;
    }
    out[cell] = sqrt(uu * uu + vv * vv);
    out[layer + cell] = 180.0 / PI * a;
}
)wgsl";

} // namespace kernels
} // namespace gpu

#endif /* GPU_MASS_KERNELS_H */
