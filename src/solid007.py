"""007 solid kernels in DISPLACEMENT form (float32 precision fix).

Bug found in 007's first smoke test: the 002-006 kernels store absolute positions in float32. At |p| ~ 2000-4000 px
the float32 spacing is 1.2e-4 - 2.4e-4 px, so an increment v * dt below half of it is rounded away (points with
speed 5e-5 and dt 0.25 never moved in 4000 steps). Here every point stores its displacement u = p - p0 (small,
precise); bond vectors are (p0_j - p0_i) + (u_j - u_i) (reference coordinates are integers, exact in float32).
Physics is identical to the 002 bond law, 002 contact law and 004 integrate_dt (brake off here).
"""
import cupy as cp

SRC = r"""
extern "C" __global__ void bond_forces_u(const int* bi, const int* bj, const float* L0, const float* k, const float* en,
                                         const float* cdamp, unsigned char* broken, const float* px0, const float* py0,
                                         const float* ux, const float* uy, const float* vx, const float* vy, int B,
                                         float Fcrit, float* ax, float* ay, int* nbreak) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || broken[b]) return;
    int i = bi[b], j = bj[b];
    float dx = (px0[j] - px0[i]) + (ux[j] - ux[i]);
    float dy = (py0[j] - py0[i]) + (uy[j] - uy[i]);
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

extern "C" __global__ void node_of_u(const float* px0, const float* py0, const float* ux, const float* uy, int P,
                                     int NX, int NY, int* node_of) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int xi = __float2int_rn(px0[p] + ux[p]), yi = __float2int_rn(py0[p] + uy[p]);
    node_of[p] = (xi < 0 || yi < 0 || xi >= NX || yi >= NY) ? -1 : yi * NX + xi;
}

extern "C" __global__ void build_lists(const int* node_of, int P, int* head, int* nxt) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int j = node_of[p];
    nxt[p] = (j >= 0) ? atomicExch(&head[j], p) : -1;
}

extern "C" __global__ void contact_u(const float* px0, const float* py0, const float* ux, const float* uy,
                                     const int* node_of, const int* head, const int* nxt, int P, int NX, int NY,
                                     float kc, float* ax, float* ay, int* ncontact) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int j = node_of[p];
    if (j < 0) return;
    int x = j % NX, y = j / NX;
    float fx = 0.f, fy = 0.f; int nc = 0;
    for (int oy = -1; oy <= 1; oy++) for (int ox = -1; ox <= 1; ox++) {
        int xn = x + ox, yn = y + oy;
        if (xn < 0 || yn < 0 || xn >= NX || yn >= NY) continue;
        for (int q = head[yn * NX + xn]; q >= 0; q = nxt[q]) {
            if (q == p) continue;
            if (fabsf(px0[q] - px0[p]) <= 1.01f && fabsf(py0[q] - py0[p]) <= 1.01f) continue;
            float dx = (px0[p] - px0[q]) + (ux[p] - ux[q]), dy = (py0[p] - py0[q]) + (uy[p] - uy[q]);
            float r = sqrtf(dx * dx + dy * dy);
            if (r < 1.f && r > 1e-6f) { float f = kc * (1.f - r) / r; fx += f * dx; fy += f * dy; nc++; }
        }
    }
    ax[p] += fx; ay[p] += fy;
    if (nc) atomicAdd(ncontact, nc);
}

// centrifugal force F = m * om2 * (p - c) added here, then symplectic Euler on (v, u)
extern "C" __global__ void integrate_u(const float* px0, const float* py0, float* ux, float* uy, float* vx, float* vy,
                                       const float* ax, const float* ay, const unsigned char* fixed, float M, float om2,
                                       float xc, float yc, int P, float dt, float xmax, float ymax, int* nclamp) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    if (fixed[p]) { vx[p] = 0.f; vy[p] = 0.f; return; }
    float cx = (px0[p] - xc) + ux[p], cy = (py0[p] - yc) + uy[p];
    vx[p] += (ax[p] + M * om2 * cx) / M * dt;
    vy[p] += (ay[p] + M * om2 * cy) / M * dt;
    ux[p] += vx[p] * dt; uy[p] += vy[p] * dt;
    // frame: stop at the edge
    float X = px0[p] + ux[p], Y = py0[p] + uy[p];
    if (X < 0.f || X > xmax || Y < 0.f || Y > ymax) {
        ux[p] = fminf(fmaxf(X, 0.f), xmax) - px0[p]; uy[p] = fminf(fmaxf(Y, 0.f), ymax) - py0[p];
        vx[p] = 0.f; vy[p] = 0.f; atomicAdd(nclamp, 1);
    }
}
"""
_m = None


def module():
    global _m
    if _m is None:
        _m = cp.RawModule(code=SRC)
    return _m
