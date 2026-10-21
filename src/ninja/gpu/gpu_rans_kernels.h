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

// ABL and wall-function constants (rans_abl.h)
const KAPPA = 0.41;
const KAPPA_EPSILON = 0.4187;
const WALL_CMU = 0.085;

struct Bc {
    kind : u32,
    p : f32,
    u : f32,
    v : f32,
    w : f32,
    ustar : f32,
    z0 : f32,
    k : f32,
    epsilon : f32,
    pad0 : f32,
    pad1 : f32,
    pad2 : f32,
}

struct Params {
    nx : u32,
    ny : u32,
    nz : u32,
    dir : u32,          // face direction of a face pass
    nu : f32,
    beta : f32,
    sigmaK : f32,
    sigmaEpsilon : f32,
    turbulent : u32,    // 0 laminar, 1 k-epsilon
    yPlusLam : f32,     // wall-function laminar / log switch
    cfl : f32,
    pad0 : f32,
    // RNG k-epsilon
    cmu : f32,
    c1 : f32,
    c2 : f32,
    eta0 : f32,
    rngBeta : f32,
    pad1 : f32,
    pad2 : f32,
    pad3 : f32,
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
const FK = 4u;
const FEPSILON = 5u;
const FNUT = 6u;

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

// Face state (p, u, v, w) on a boundary face from the owner cell's state
// q, as RansSolver's BoundaryState; n is the outward unit normal, agl the
// face's height above ground.  backflow applies the outlets' backflow
// treatment (fluxes only).
fn boundaryState(b : Bc, n : vec3<f32>, agl : f32, q : vec4<f32>,
                 backflow : bool) -> vec4<f32> {
    switch b.kind {
        case WALL, VELOCITY_INLET: {
            return vec4<f32>(q.x, b.u, b.v, b.w);
        }
        case SLIP: {
            let un = dot(q.yzw, n);
            return vec4<f32>(q.x, q.yzw - un * n);
        }
        case PRESSURE_OUTLET: {
            let un = dot(q.yzw, n);
            if (backflow && un < 0.0) {
                return vec4<f32>(b.p - 0.5 * un * un, un * n);
            }
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
        value = boundaryState(P.bc[side], n, faces[AGL * nf + f], state(o), false);
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

// Face pass of the residuals (RansSolver::computeResiduals): per face, 10
// slots of nFaces.  0-5: the p, u, v, w, k, epsilon fluxes in the stored S
// orientation (so the gather takes upper minus lower); 6: the spectral
// radius contribution to each adjacent cell; 7-9: the rough-wall epsilon, G
// and a 1 on wall faces (boundary faces only).
constexpr const char *fluxFaces = R"wgsl(
@group(0) @binding(1) var<storage, read> cells : array<f32>;
@group(0) @binding(2) var<storage, read> faces : array<f32>;
@group(0) @binding(3) var<storage, read> fields : array<f32>;
@group(0) @binding(4) var<storage, read> gradients : array<f32>;
@group(0) @binding(5) var<storage, read_write> out : array<f32>;

fn field(slot : u32, c : u32) -> f32 { return fields[slot * nCells() + c]; }

fn state(c : u32) -> vec4<f32> {
    return vec4<f32>(field(FP, c), field(FU, c), field(FV, c), field(FW, c));
}

fn centroid(c : u32) -> vec3<f32> {
    let n = nCells();
    return vec3<f32>(cells[CX * n + c], cells[CY * n + c], cells[CZ * n + c]);
}

// gradient of field q (0-3 = p, u, v, w) in cell c
fn grad(q : u32, c : u32) -> vec3<f32> {
    let n = nCells();
    return vec3<f32>(gradients[(q * 3u) * n + c], gradients[(q * 3u + 1u) * n + c],
                     gradients[(q * 3u + 2u) * n + c]);
}

// linear reconstruction of (p, u, v, w) from cell c by offset r
fn reconstruct(c : u32, r : vec3<f32>) -> vec4<f32> {
    let q = state(c);
    return q + vec4<f32>(dot(grad(0u, c), r), dot(grad(1u, c), r),
                         dot(grad(2u, c), r), dot(grad(3u, c), r));
}

// inviscid flux of the artificial-compressibility system through S
fn inviscid(S : vec3<f32>, q : vec4<f32>) -> vec4<f32> {
    let U = dot(q.yzw, S);
    return vec4<f32>(P.beta * U, q.yzw * U + q.x * S);
}

fn roughWallNut(k : f32, y : f32, z0 : f32) -> f32 {
    let yPlus = pow(WALL_CMU, 0.25) * sqrt(k) * y / P.nu;
    let eDash = (y + z0) / z0;
    return P.nu * (yPlus * KAPPA / log(max(eDash, 1.0 + 1e-4)) - 1.0);
}

// (epsilon, G) in a wall cell, as EpsilonWallFunction
fn epsilonWall(nutw : f32, k : f32, y : f32, magGradUw : f32) -> vec2<f32> {
    let cmu25 = pow(WALL_CMU, 0.25);
    let yPlus = cmu25 * y * sqrt(k) / P.nu;
    if (yPlus > P.yPlusLam) {
        return vec2<f32>(pow(WALL_CMU, 0.75) * pow(k, 1.5) / (KAPPA * y),
                         (nutw + P.nu) * magGradUw * cmu25 * sqrt(k) / (KAPPA * y));
    }
    return vec2<f32>(2.0 * k * P.nu / (y * y), 0.0);
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
    let e3 = faceExtent(d);
    var ijk = vec3<u32>(f % e3.x, (f / e3.x) % e3.y, f / (e3.x * e3.y));
    let S = vec3<f32>(faces[(SX + 0u) * nf + f], faces[(SX + 1u) * nf + f],
                      faces[(SX + 2u) * nf + f]);
    let F = vec3<f32>(faces[(FX + 0u) * nf + f], faces[(FX + 1u) * nf + f],
                      faces[(FX + 2u) * nf + f]);
    let agl = faces[AGL * nf + f];
    let m = ijk[d];
    let last = vec3<u32>(P.nx, P.ny, P.nz)[d];
    let boundary = m == 0u || m == last;
    let turbulent = P.turbulent != 0u;
    let sigmaMin = min(P.sigmaK, P.sigmaEpsilon);
    let area2 = dot(S, S);
    let area = sqrt(area2);

    var flux : vec4<f32>;
    var gU : array<vec3<f32>, 3>;   // face velocity gradient per component
    var Sv = S;                     // area vector the cell sees (outward)
    var viscous = true;
    var spec = 0.0;
    var nutF = 0.0;
    var dist = 1.0;
    var sDotE = 0.0;
    var kb = 0.0;
    var eb = 0.0;
    var fixedTurb = false;
    var wall = vec3<f32>(0.0);      // epsilon, G, is-wall
    var o : u32;
    var nb = 0u;

    if (!boundary) {
        var lo = ijk;
        lo[d] = m - 1u;
        o = cellIndex(lo.x, lo.y, lo.z);
        nb = cellIndex(ijk.x, ijk.y, ijk.z);
        // second-order reconstruction to the face from both sides
        let qL = reconstruct(o, F - centroid(o));
        let qR = reconstruct(nb, F - centroid(nb));
        let fL = inviscid(S, qL);
        let fR = inviscid(S, qR);
        let UL = dot(qL.yzw, S);
        let UR = dot(qR.yzw, S);
        let lambda = max(abs(UL) + sqrt(UL * UL + P.beta * area2),
                         abs(UR) + sqrt(UR * UR + P.beta * area2));
        // acoustic dissipation on pressure only; momentum is dissipated at
        // the convective speed (see RansSolver::computeResiduals)
        let lambdaU = max(abs(UL), abs(UR));
        flux = vec4<f32>(0.5 * (fL.x + fR.x) - 0.5 * lambda * (qR.x - qL.x),
                         0.5 * (fL.yzw + fR.yzw) - 0.5 * lambdaU * (qR.yzw - qL.yzw));
        // averaged gradient corrected along the centroid line
        var e = centroid(nb) - centroid(o);
        let dd = length(e);
        e = e / dd;
        let qo = state(o);
        let qn = state(nb);
        for (var c = 0u; c < 3u; c++) {
            let a = 0.5 * (grad(1u + c, o) + grad(1u + c, nb));
            let corr = (qn[1u + c] - qo[1u + c]) / dd - dot(a, e);
            gU[c] = a + corr * e;
        }
        spec = 0.5 * lambda + P.nu * area2 / (dd * area);
        if (turbulent) {
            dist = dd;
            sDotE = dot(S, e);
            nutF = 0.5 * (field(FNUT, o) + field(FNUT, nb));
            spec += nutF / sigmaMin * area2 / (dd * area);
        }
    } else {
        if (m == last) {
            ijk[d] = m - 1u;
        }
        o = cellIndex(ijk.x, ijk.y, ijk.z);
        let side = 2u * d + select(0u, 1u, m == last);
        let b = P.bc[side];
        Sv = select(-S, S, m == last);
        let n = Sv / area;
        let qo = state(o);
        let qb = boundaryState(b, n, agl, qo, true);
        viscous = b.kind == WALL || b.kind == VELOCITY_INLET || b.kind == LOG_PROFILE_INLET;
        flux = inviscid(Sv, qb);
        let Ub = dot(qb.yzw, Sv);
        let lambda = abs(Ub) + sqrt(Ub * Ub + P.beta * area2);
        var e = F - centroid(o);
        let dd = length(e);
        e = e / dd;
        for (var c = 0u; c < 3u; c++) {
            let a = grad(1u + c, o);
            let corr = (qb[1u + c] - qo[1u + c]) / dd - dot(a, e);
            gU[c] = a + corr * e;
        }
        spec = 0.5 * lambda + select(0.0, P.nu * area2 / (dd * area), viscous);
        if (turbulent) {
            dist = dd;
            sDotE = dot(Sv, e);
            nutF = field(FNUT, o);
            let ko = field(FK, o);
            if (b.kind == WALL && b.z0 > 0.0) {
                // rough-wall functions; y is the centroid's normal distance
                let y = abs(dot(centroid(o) - F, Sv)) / area;
                nutF = max(0.0, roughWallNut(ko, y, b.z0));
                let du = length(qo.yzw - qb.yzw);
                wall = vec3<f32>(epsilonWall(nutF, ko, y, du / y), 1.0);
            }
            if (b.kind == VELOCITY_INLET) {
                kb = b.k;
                eb = b.epsilon;
                fixedTurb = true;
            } else if (b.kind == LOG_PROFILE_INLET) {
                kb = b.ustar * b.ustar / sqrt(WALL_CMU);
                eb = b.ustar * b.ustar * b.ustar / (KAPPA_EPSILON * (agl + b.z0));
                fixedTurb = true;
            } else {
                kb = ko;
                eb = field(FEPSILON, o);
            }
            if (viscous || fixedTurb) {
                spec += nutF / sigmaMin * area2 / (dd * area);
            }
        }
    }

    if (viscous && !turbulent) {
        for (var c = 0u; c < 3u; c++) {
            flux[1u + c] -= P.nu * dot(gU[c], Sv);
        }
    } else if (viscous) {
        // nu_eff grad(u).S plus nu_t grad(u)^T.S
        let nuEff = P.nu + nutF;
        for (var c = 0u; c < 3u; c++) {
            let gT = vec3<f32>(gU[0][c], gU[1][c], gU[2][c]);
            flux[1u + c] -= nuEff * dot(gU[c], Sv) + nutF * dot(gT, Sv);
        }
    }

    var fk = 0.0;
    var fe = 0.0;
    if (turbulent) {
        // first-order upwind convection on the face mass flux, orthogonal
        // diffusion
        let mass = flux.x / P.beta;
        let Dk = P.nu + nutF / P.sigmaK;
        let De = P.nu + nutF / P.sigmaEpsilon;
        let ko = field(FK, o);
        let eo = field(FEPSILON, o);
        if (!boundary) {
            let kn = field(FK, nb);
            let en = field(FEPSILON, nb);
            fk = mass * select(kn, ko, mass > 0.0) - Dk * (kn - ko) / dist * sDotE;
            fe = mass * select(en, eo, mass > 0.0) - De * (en - eo) / dist * sDotE;
        } else {
            fk = mass * select(kb, ko, mass > 0.0);
            fe = mass * select(eb, eo, mass > 0.0);
            if (fixedTurb) {
                fk -= Dk * (kb - ko) / dist * sDotE;
                fe -= De * (eb - eo) / dist * sDotE;
            }
        }
    }

    // a min-side boundary computed its flux outward (-S): store it along S
    let sign = select(1.0, -1.0, boundary && m == 0u);
    out[0u * nf + f] = sign * flux.x;
    out[1u * nf + f] = sign * flux.y;
    out[2u * nf + f] = sign * flux.z;
    out[3u * nf + f] = sign * flux.w;
    out[4u * nf + f] = sign * fk;
    out[5u * nf + f] = sign * fe;
    out[6u * nf + f] = spec;
    out[7u * nf + f] = wall.x;
    out[8u * nf + f] = wall.y;
    out[9u * nf + f] = wall.z;
}
)wgsl";

// Cell gather of the residuals: 10 slots of nCells.  0-5 (fluxes): upper
// minus lower face in each direction; 6-9 (spectral radius, wall epsilon,
// wall G, wall faces): the sum over the six faces.
constexpr const char *fluxCells = R"wgsl(
@group(0) @binding(1) var<storage, read> face0 : array<f32>;
@group(0) @binding(2) var<storage, read> face1 : array<f32>;
@group(0) @binding(3) var<storage, read> face2 : array<f32>;
@group(0) @binding(4) var<storage, read_write> residuals : array<f32>;

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
    for (var slot = 0u; slot < 10u; slot++) {
        let sign = select(1.0, -1.0, slot < 6u);
        var sum = 0.0;
        for (var d = 0u; d < 3u; d++) {
            var up = ijk;
            up[d] += 1u;
            sum += faceValue(d, slot, faceIndex(d, up)) + sign * faceValue(d, slot, faceIndex(d, ijk));
        }
        residuals[slot * n + c] = sum;
    }
}
)wgsl";

// Local pseudo-time update of each cell (the body of RansSolver::iterate),
// in place: p, u, v, w by the flux residuals; k and epsilon point-implicitly
// with the RNG sources (wall cells take G and epsilon from the wall
// functions); then nut.  Also writes the per-cell residuals the norms use,
// 6 slots of nCells: rp/V, ru/V, rv/V, rw/V and the k and epsilon residuals.
constexpr const char *update = R"wgsl(
@group(0) @binding(1) var<storage, read> cells : array<f32>;
@group(0) @binding(2) var<storage, read_write> fields : array<f32>;
@group(0) @binding(3) var<storage, read> gradients : array<f32>;
@group(0) @binding(4) var<storage, read> residuals : array<f32>;
@group(0) @binding(5) var<storage, read_write> norms : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let n = nCells();
    let c = threadIndex(gid, groups);
    if (c >= n) {
        return;
    }
    let V = cells[VOLUME * n + c];
    // local pseudo time step: dtau / V = cfl / spectral radius
    let dt = P.cfl / residuals[6u * n + c];
    let rp = residuals[c];
    let ru = residuals[n + c];
    let rv = residuals[2u * n + c];
    let rw = residuals[3u * n + c];
    var resK = 0.0;
    var resE = 0.0;
    if (P.turbulent == 1u) {
        // production nut * (dev(twoSymm(grad U)) && grad U)
        var sum = 0.0;
        var trace = 0.0;
        for (var i = 0u; i < 3u; i++) {
            trace += gradients[((1u + i) * 3u + i) * n + c];
            for (var j = 0u; j < 3u; j++) {
                let gij = gradients[((1u + i) * 3u + j) * n + c];
                let gji = gradients[((1u + j) * 3u + i) * n + c];
                sum += gij * (gij + gji);
            }
        }
        let s2 = sum - 2.0 / 3.0 * trace * trace;
        let kc = fields[FK * n + c];
        let ec = fields[FEPSILON * n + c];
        let walls = residuals[9u * n + c];
        var G = fields[FNUT * n + c] * s2;
        if (walls > 0.0) {
            G = residuals[8u * n + c] / walls;
        }
        let eta = sqrt(abs(s2)) * kc / ec;
        let R = eta * (1.0 - eta / P.eta0) / (P.rngBeta * eta * eta * eta + 1.0);
        let rk = residuals[4u * n + c];
        let re = residuals[5u * n + c];
        let kSrc = G - ec;
        let eSrc = (P.c1 - R) * G * ec / kc - P.c2 * ec * ec / kc;
        // point-implicit sinks (as fvm::Sp) keep k and epsilon positive
        let kn = (kc - dt * rk + dt * V * G) / (1.0 + dt * V * ec / kc);
        var en = (ec - dt * re + dt * V * (P.c1 - R) * G * ec / kc) /
                 (1.0 + dt * V * P.c2 * ec / kc);
        resK = rk / V - kSrc;
        resE = re / V - eSrc;
        if (walls > 0.0) {
            en = residuals[7u * n + c] / walls;
            resE = 0.0;
        }
        let k = max(kn, 1e-12);
        let e = max(en, 1e-12);
        fields[FK * n + c] = k;
        fields[FEPSILON * n + c] = e;
        fields[FNUT * n + c] = P.cmu * k * k / e;
    }
    fields[FP * n + c] -= dt * rp;
    fields[FU * n + c] -= dt * ru;
    fields[FV * n + c] -= dt * rv;
    fields[FW * n + c] -= dt * rw;
    norms[c] = rp / V;
    norms[n + c] = ru / V;
    norms[2u * n + c] = rv / V;
    norms[3u * n + c] = rw / V;
    norms[4u * n + c] = resK;
    norms[5u * n + c] = resE;
}
)wgsl";

} // namespace kernels
} // namespace gpu

#endif /* GPU_RANS_KERNELS_H */
