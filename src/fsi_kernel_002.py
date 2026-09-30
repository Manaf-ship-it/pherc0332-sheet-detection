"""Fluid-structure kernels: LBM with moving matter walls + bonded material points.

Fluid: identical to the frozen 001 kernel (lbm_kernel_001.py: D2Q9 BGK, Guo forcing, populations
stored as d = f - w) except that a link pulled from a matter node uses Ladd's moving bounce-back
    f_i(x) = f*_opp(i)(x) + 6 w_i rho_w (c_i . U_wall)
with U_wall the velocity of the matter node x - c_i (rho_w = 1). Frame edges stay static walls.

Force on a matter node (momentum exchange, gauge rho_ref = 1):
    F += c_i * (2 d*_i(x) - 6 w_i (c_i . U_wall))      for each fluid x, link i with x + c_i matter.

Solid: points p (one per initial matter node), mass M, bonds (i, j) with rest length L0, stiffness
k, reference normal component |e0 . n| (for the rupture law), damping c.
Contact (PROTOCOL amendment_1, k_c given): repulsion between non-reference-neighbour points closer than 1 node.
Build PERIODIC_X variants for the validation tests only.
"""
import cupy as cp
import numpy as np

SRC = r"""
__constant__ int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant__ float w[9] = {4.f/9, 1.f/9, 1.f/9, 1.f/9, 1.f/9, 1.f/36, 1.f/36, 1.f/36, 1.f/36};
__constant__ int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

__device__ __forceinline__ int wrapx(int x, int NX) {
#ifdef PERIODIC_X
    return (x + NX) % NX;
#else
    return x;
#endif
}

extern "C" __global__ void lbm_step_mw(const float* fin, float* fout, const unsigned char* solid,
                                       const float* UWx, const float* UWy,
                                       const float* Fx, const float* Fy, float omega, int NX, int NY) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)NX * NY;
    if (idx >= N || solid[idx]) return;
    int x = idx % NX, y = idx / NX;
    float f[9];
    for (int i = 0; i < 9; i++) {
        int xs = wrapx(x - cx[i], NX), ys = y - cy[i];
        if (xs < 0 || ys < 0 || xs >= NX || ys >= NY) {
            f[i] = fin[opp[i] * N + idx];                                  // static frame wall
        } else {
            long long j = (long long)ys * NX + xs;
            if (solid[j])
                f[i] = fin[opp[i] * N + idx] + 6.f * w[i] * (cx[i] * UWx[j] + cy[i] * UWy[j]);
            else
                f[i] = fin[i * N + j];
        }
    }
    float drho = 0.f, jx = 0.f, jy = 0.f;
    for (int i = 0; i < 9; i++) { drho += f[i]; jx += f[i] * cx[i]; jy += f[i] * cy[i]; }
    float rho = 1.f + drho;
    float fx = Fx[idx], fy = Fy[idx];
    float ux = (jx + 0.5f * fx) / rho, uy = (jy + 0.5f * fy) / rho;
    float usq = ux * ux + uy * uy;
    for (int i = 0; i < 9; i++) {
        float cu = cx[i] * ux + cy[i] * uy;
        float deq = w[i] * (drho + rho * (3.f * cu + 4.5f * cu * cu - 1.5f * usq));
        float si = (1.f - 0.5f * omega) * w[i] *
                   (3.f * ((cx[i] - ux) * fx + (cy[i] - uy) * fy) + 9.f * cu * (cx[i] * fx + cy[i] * fy));
        fout[i * N + idx] = f[i] - omega * (f[i] - deq) + si;
    }
}

extern "C" __global__ void lbm_macro_mw(const float* fin, const unsigned char* solid,
                                        const float* UWx, const float* UWy, const float* Fx,
                                        const float* Fy, int NX, int NY, float* rho_o, float* ux_o,
                                        float* uy_o) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)NX * NY;
    if (idx >= N) return;
    if (solid[idx]) { rho_o[idx] = 0.f; ux_o[idx] = UWx[idx]; uy_o[idx] = UWy[idx]; return; }
    int x = idx % NX, y = idx / NX;
    float drho = 0.f, jx = 0.f, jy = 0.f;
    for (int i = 0; i < 9; i++) {
        int xs = wrapx(x - cx[i], NX), ys = y - cy[i];
        float fi;
        if (xs < 0 || ys < 0 || xs >= NX || ys >= NY) fi = fin[opp[i] * N + idx];
        else {
            long long j = (long long)ys * NX + xs;
            fi = solid[j] ? fin[opp[i] * N + idx] + 6.f * w[i] * (cx[i] * UWx[j] + cy[i] * UWy[j])
                          : fin[i * N + j];
        }
        drho += fi; jx += fi * cx[i]; jy += fi * cy[i];
    }
    float rho = 1.f + drho;
    rho_o[idx] = rho;
    ux_o[idx] = (jx + 0.5f * Fx[idx]) / rho;
    uy_o[idx] = (jy + 0.5f * Fy[idx]) / rho;
}

extern "C" __global__ void wall_force_mw(const float* fpost, const unsigned char* solid,
                                         const float* UWx, const float* UWy, int NX, int NY,
                                         float* Wx, float* Wy) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long N = (long long)NX * NY;
    if (idx >= N || solid[idx]) return;
    int x = idx % NX, y = idx / NX;
    for (int i = 1; i < 9; i++) {
        int xn = wrapx(x + cx[i], NX), yn = y + cy[i];
        if (xn < 0 || yn < 0 || xn >= NX || yn >= NY) continue;
        long long j = (long long)yn * NX + xn;
        if (!solid[j]) continue;
        float m = 2.f * fpost[i * N + idx] - 6.f * w[i] * (cx[i] * UWx[j] + cy[i] * UWy[j]);
        atomicAdd(&Wx[j], m * cx[i]);
        atomicAdd(&Wy[j], m * cy[i]);
    }
}

// nodes freed by matter since the previous step get the equilibrium of the last node velocity
extern "C" __global__ void refill(float* f, const unsigned char* solid_now, const unsigned char* solid_prev,
                                  const float* UWx_prev, const float* UWy_prev, long long N, int* count) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || solid_now[idx] || !solid_prev[idx]) return;
    float ux = UWx_prev[idx], uy = UWy_prev[idx], usq = ux * ux + uy * uy;
    for (int i = 0; i < 9; i++) {
        float cu = cx[i] * ux + cy[i] * uy;
        f[i * N + idx] = w[i] * (3.f * cu + 4.5f * cu * cu - 1.5f * usq);
    }
    atomicAdd(count, 1);
}

// points -> nodes: occupancy, summed velocity
extern "C" __global__ void scatter_points(const float* px, const float* py, const float* vx, const float* vy,
                                          int P, int NX, int NY, int* cnt, float* sx, float* sy, int* node_of) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int xi = wrapx(__float2int_rn(px[p]), NX), yi = __float2int_rn(py[p]);
    if (xi < 0 || yi < 0 || xi >= NX || yi >= NY) { node_of[p] = -1; return; }
    long long j = (long long)yi * NX + xi;
    node_of[p] = (int)j;
    atomicAdd(&cnt[j], 1);
    atomicAdd(&sx[j], vx[p]);
    atomicAdd(&sy[j], vy[p]);
}

extern "C" __global__ void nodes_finalize(const int* cnt, float* sx, float* sy, unsigned char* solid, long long N) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    int c = cnt[j];
    solid[j] = c > 0;
    if (c > 0) { sx[j] /= c; sy[j] /= c; } else { sx[j] = 0.f; sy[j] = 0.f; }
}

// bond forces (elastic + damping) and the rupture law
extern "C" __global__ void bond_forces(const int* bi, const int* bj, const float* L0, const float* k,
                                       const float* en, const float* cdamp, unsigned char* broken,
                                       const float* px, const float* py, const float* vx, const float* vy,
                                       int B, int NX, float Fcrit, float* ax, float* ay, int* nbreak) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || broken[b]) return;
    int i = bi[b], j = bj[b];
    float dx = px[j] - px[i], dy = py[j] - py[i];
#ifdef PERIODIC_X
    dx -= NX * rintf(dx / NX);
#endif
    float L = sqrtf(dx * dx + dy * dy);
    float ex = dx / L, ey = dy / L;
    float eps = (L - L0[b]) / L0[b];
    float fel = k[b] * eps;
    if (eps > 0.f && fel * en[b] > Fcrit) { broken[b] = 1; atomicAdd(nbreak, 1); return; }
    float dv = (vx[j] - vx[i]) * ex + (vy[j] - vy[i]) * ey;
    float f = fel + cdamp[b] * dv;
    atomicAdd(&ax[i], f * ex); atomicAdd(&ay[i], f * ey);
    atomicAdd(&ax[j], -f * ex); atomicAdd(&ay[j], -f * ey);
}

// fluid force of each matter node shared equally by the points sitting on it; then integrate
extern "C" __global__ void integrate(float* px, float* py, float* vx, float* vy, const float* ax, const float* ay,
                                     const int* node_of, const int* cnt, const float* Wx, const float* Wy,
                                     const unsigned char* fixed, float M, int P, int NX,
                                     float* fluid_fx, float* fluid_fy) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float ffx = 0.f, ffy = 0.f;
    int j = node_of[p];
    if (j >= 0) { ffx = Wx[j] / cnt[j]; ffy = Wy[j] / cnt[j]; }
    fluid_fx[p] = ffx; fluid_fy[p] = ffy;
    if (fixed[p]) { vx[p] = 0.f; vy[p] = 0.f; return; }
    vx[p] += (ax[p] + ffx) / M;
    vy[p] += (ay[p] + ffy) / M;
    px[p] += vx[p];
    py[p] += vy[p];
#ifdef PERIODIC_X
    px[p] = fmodf(px[p] + NX, (float)NX);
#endif
}
// contact: linked lists of points per node, then repulsion between points that were not
// reference-lattice neighbours and are closer than 1 node (no friction)
extern "C" __global__ void build_lists(const int* node_of, int P, int* head, int* nxt) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int j = node_of[p];
    nxt[p] = (j >= 0) ? atomicExch(&head[j], p) : -1;
}

extern "C" __global__ void contact_forces(const float* px, const float* py, const float* px0, const float* py0,
                                          const int* node_of, const int* head, const int* nxt, int P, int NX, int NY,
                                          float kc, float* ax, float* ay, int* ncontact) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int j = node_of[p];
    if (j < 0) return;
    int x = j % NX, y = j / NX;
    float fx = 0.f, fy = 0.f;
    int nc = 0;
    for (int oy = -1; oy <= 1; oy++) for (int ox = -1; ox <= 1; ox++) {
        int xn = wrapx(x + ox, NX), yn = y + oy;
        if (xn < 0 || yn < 0 || xn >= NX || yn >= NY) continue;
        for (int q = head[(long long)yn * NX + xn]; q >= 0; q = nxt[q]) {
            if (q == p) continue;
            if (fabsf(px0[q] - px0[p]) <= 1.01f && fabsf(py0[q] - py0[p]) <= 1.01f) continue;
            float dx = px[p] - px[q], dy = py[p] - py[q];
#ifdef PERIODIC_X
            dx -= NX * rintf(dx / NX);
#endif
            float r = sqrtf(dx * dx + dy * dy);
            if (r < 1.f && r > 1e-6f) { float f = kc * (1.f - r) / r; fx += f * dx; fy += f * dy; nc++; }
        }
    }
    ax[p] += fx; ay[p] += fy;
    if (nc) atomicAdd(ncontact, nc);
}
"""

_mods = {}


def module(periodic_x=False):
    if periodic_x not in _mods:
        _mods[periodic_x] = cp.RawModule(code=SRC, options=("-DPERIODIC_X",) if periodic_x else ())
    return _mods[periodic_x]


def build_bonds(matter, tx, ty, k_t, k_n, mass, zeta, periodic_x=False):
    """Points = matter nodes (row-major order). Bonds to the 4 forward neighbours (8-neighbourhood).

    Returns point coords, bond arrays (i, j, L0, k, |e0.n|, damping)."""
    H, W = matter.shape
    ys, xs = np.nonzero(matter)
    pid = -np.ones(matter.shape, np.int64)
    pid[ys, xs] = np.arange(len(ys))
    bi, bj, L0, kk, en, cd = [], [], [], [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        x2 = xs + dx
        y2 = ys + dy
        if periodic_x:
            x2 = x2 % W
            ok = (y2 >= 0) & (y2 < H)
        else:
            ok = (x2 >= 0) & (x2 < W) & (y2 >= 0) & (y2 < H)
        a = np.nonzero(ok)[0]
        a = a[matter[y2[a], x2[a]]]
        b = pid[y2[a], x2[a]]
        L = np.hypot(dx, dy)
        ex, ey = dx / L, dy / L
        # mean material tangent of the two points (doubled-angle average, sign-free)
        c2 = (tx[ys[a], xs[a]] ** 2 - ty[ys[a], xs[a]] ** 2) + (tx[ys[b], xs[b]] ** 2 - ty[ys[b], xs[b]] ** 2)
        s2 = 2 * tx[ys[a], xs[a]] * ty[ys[a], xs[a]] + 2 * tx[ys[b], xs[b]] * ty[ys[b], xs[b]]
        th = 0.5 * np.arctan2(s2, c2)
        cos_phi = np.abs(ex * np.cos(th) + ey * np.sin(th))
        cos2 = cos_phi ** 2
        k = k_t * cos2 + k_n * (1 - cos2)
        bi.append(a); bj.append(b); L0.append(np.full(len(a), L)); kk.append(k)
        en.append(np.sqrt(np.clip(1 - cos2, 0, 1)))
        cd.append(zeta * 2 * np.sqrt(k * mass / 2))
    cat = lambda v, t: np.concatenate(v).astype(t)
    return (xs.astype(np.float32), ys.astype(np.float32), cat(bi, np.int32), cat(bj, np.int32),
            cat(L0, np.float32), cat(kk, np.float32), cat(en, np.float32), cat(cd, np.float32))


class FSI:
    """State + one coupled step. Arrays on GPU."""

    def __init__(self, matter, tx, ty, Fx, Fy, rho, ux, uy, k_t, k_n, mass, zeta, F_crit, fixed_mask=None,
                 periodic_x=False, tau=1.0, k_c=None):
        self.m = module(periodic_x)
        self.k = {n: self.m.get_function(n) for n in
                  ("lbm_step_mw", "lbm_macro_mw", "wall_force_mw", "refill", "scatter_points",
                   "nodes_finalize", "bond_forces", "integrate", "build_lists", "contact_forces")}
        H, W = matter.shape
        self.NX, self.NY, self.N = W, H, W * H
        px, py, bi, bj, L0, kk, en, cd = build_bonds(matter, tx, ty, k_t, k_n, mass, zeta, periodic_x)
        self.P, self.B = len(px), len(bi)
        g = cp.asarray
        self.px, self.py = g(px), g(py)
        self.px0, self.py0 = g(px.copy()), g(py.copy())
        self.vx = cp.zeros(self.P, cp.float32); self.vy = cp.zeros(self.P, cp.float32)
        self.bi, self.bj, self.L0, self.kb, self.en, self.cd = g(bi), g(bj), g(L0), g(kk), g(en), g(cd)
        self.broken = cp.zeros(self.B, cp.uint8)
        fixed = np.zeros(self.P, np.uint8) if fixed_mask is None else fixed_mask[py.astype(int), px.astype(int)].astype(np.uint8)
        self.fixed = g(fixed)
        self.M, self.F_crit, self.omega = np.float32(mass), np.float32(F_crit), np.float32(1.0 / tau)
        self.Fx, self.Fy = g(Fx.astype(np.float32)), g(Fy.astype(np.float32))
        # fluid populations (deviations) at equilibrium of the given macroscopic state
        r, u, v = g(rho.astype(np.float32)), g(ux.astype(np.float32)), g(uy.astype(np.float32))
        CXs = (0, 1, 0, -1, 0, 1, -1, -1, 1); CYs = (0, 0, 1, 0, -1, 1, 1, -1, -1)
        Ws = (4 / 9,) + (1 / 9,) * 4 + (1 / 36,) * 4
        self.f = cp.empty((9, H, W), cp.float32)
        usq = u * u + v * v
        for i in range(9):
            cu = CXs[i] * u + CYs[i] * v
            self.f[i] = Ws[i] * ((r - 1) + r * (3 * cu + 4.5 * cu * cu - 1.5 * usq))
        self.g = self.f.copy()
        z = lambda t=cp.float32: cp.zeros((H, W), t)
        self.cnt = z(cp.int32); self.UWx = z(); self.UWy = z()
        self.solid = z(cp.uint8); self.solid_prev = z(cp.uint8)
        self.UWx_prev = z(); self.UWy_prev = z()
        self.Wx = z(); self.Wy = z()
        self.node_of = cp.empty(self.P, cp.int32)
        self.ax = cp.zeros(self.P, cp.float32); self.ay = cp.zeros(self.P, cp.float32)
        self.ffx = cp.zeros(self.P, cp.float32); self.ffy = cp.zeros(self.P, cp.float32)
        self.nbreak = cp.zeros(1, cp.int32); self.nrefill = cp.zeros(1, cp.int32)
        self.head = cp.empty((H, W), cp.int32); self.nxt = cp.empty(self.P, cp.int32)
        self.ncontact = cp.zeros(1, cp.int32)
        self.k_contact = np.float32(k_c) if k_c is not None else None
        self.blk = 256
        self._update_nodes()
        self.solid_prev[...] = self.solid
        self.steps = 0

    def _grid(self, n):
        return ((int(n) + self.blk - 1) // self.blk,)

    def _update_nodes(self):
        self.cnt.fill(0); self.UWx.fill(0); self.UWy.fill(0)
        self.k["scatter_points"](self._grid(self.P), (self.blk,),
                                 (self.px, self.py, self.vx, self.vy, np.int32(self.P), np.int32(self.NX),
                                  np.int32(self.NY), self.cnt, self.UWx, self.UWy, self.node_of))
        self.k["nodes_finalize"](self._grid(self.N), (self.blk,),
                                 (self.cnt, self.UWx, self.UWy, self.solid, np.int64(self.N)))

    def step(self, move_solid=True):
        k, NX, NY = self.k, np.int32(self.NX), np.int32(self.NY)
        # 1. matter -> nodes, refill freed nodes
        self.solid_prev[...] = self.solid
        self.UWx_prev[...] = self.UWx; self.UWy_prev[...] = self.UWy
        self._update_nodes()
        k["refill"](self._grid(self.N), (self.blk,), (self.f, self.solid, self.solid_prev, self.UWx_prev,
                                                      self.UWy_prev, np.int64(self.N), self.nrefill))
        # 2. fluid step
        k["lbm_step_mw"](self._grid(self.N), (self.blk,), (self.f, self.g, self.solid, self.UWx, self.UWy,
                                                           self.Fx, self.Fy, self.omega, NX, NY))
        # 3. fluid -> matter nodes
        self.Wx.fill(0); self.Wy.fill(0)
        k["wall_force_mw"](self._grid(self.N), (self.blk,), (self.g, self.solid, self.UWx, self.UWy, NX, NY,
                                                             self.Wx, self.Wy))
        # 4. solid
        self.ax.fill(0); self.ay.fill(0)
        k["bond_forces"](self._grid(self.B), (self.blk,),
                         (self.bi, self.bj, self.L0, self.kb, self.en, self.cd, self.broken, self.px, self.py,
                          self.vx, self.vy, np.int32(self.B), NX, self.F_crit, self.ax, self.ay, self.nbreak))
        if self.k_contact is not None:   # PROTOCOL amendment_1
            self.head.fill(-1)
            k["build_lists"](self._grid(self.P), (self.blk,), (self.node_of, np.int32(self.P), self.head, self.nxt))
            k["contact_forces"](self._grid(self.P), (self.blk,),
                                (self.px, self.py, self.px0, self.py0, self.node_of, self.head, self.nxt,
                                 np.int32(self.P), NX, NY, self.k_contact, self.ax, self.ay, self.ncontact))
        if move_solid:
            k["integrate"](self._grid(self.P), (self.blk,),
                           (self.px, self.py, self.vx, self.vy, self.ax, self.ay, self.node_of, self.cnt,
                            self.Wx, self.Wy, self.fixed, self.M, np.int32(self.P), NX, self.ffx, self.ffy))
        self.f, self.g = self.g, self.f
        self.steps += 1

    def macro(self):
        rho = cp.empty((self.NY, self.NX), cp.float32); ux = cp.empty_like(rho); uy = cp.empty_like(rho)
        self.k["lbm_macro_mw"](self._grid(self.N), (self.blk,), (self.f, self.solid, self.UWx, self.UWy, self.Fx,
                                                                 self.Fy, np.int32(self.NX), np.int32(self.NY),
                                                                 rho, ux, uy))
        return rho, ux, uy
