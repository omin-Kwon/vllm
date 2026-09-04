"""[nested_ssm 2026-09-04] GDN LS6 flush epilogue (CUDA): per flushed (row, hv)

    U      = S_W[:, :m]                                 -> ls6_ubar (G,V) copy
    Phi_r  = S_W[:, :r]^T S_W[:, :m] + eta I[:r,:m]     eta = ridge * ||S_W||_F^2 / K
    M      = Phi_r[:m]                                  (metric ridge, fn:ridge)
    Phibar = Phi_r M^{-1}                               -> ls6_phi (G,r) transposed
    s_q    = M^{-1} (U^T (S_W qbar) + eta qbar[:m])      s_k likewise
    a_q    = s_q - Phibar^T qbar[:r]                     -> ls6_aq (tail anchor), ls6_ak

Everything is IEEE fp32 (no fast-math): Gram, Cholesky and the solves are the
numerically sensitive part of the paper's coefficient path.  The state is in R'
coordinates whose first G axes are the latch basis, so Omega = I[:, :G] and no
separate basis matrix exists.  Spec: nested_ssm/scale/test_gdn_ls6_phi.py.
Two launches: gram (parallel: S read, Gram, S qbar, U copy -> scratch) and solve (64-thread
blocks: Cholesky, tail/anchor solves, outputs).  Phibar[:m] = M M^{-1} = I, so only the R-m tail
rows and the two anchors are solved.  The streaming tensor-core flush absorbs the gram part.
"""
import os

import torch

_COMMON_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define FULL 0xffffffffu
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}

// Transposed warp reduction: lane l ends up with sum over lanes of v[l] (N == 32).
__device__ __forceinline__ float xpose32(float (&v)[32], int lane) {
    #pragma unroll
    for (int h = 16; h >= 1; h >>= 1) {
        #pragma unroll
        for (int i = 0; i < h; ++i) {
            const bool up = (lane & h) != 0;
            const float a = up ? v[i] : v[i + h];
            const float b = __shfl_xor_sync(FULL, a, h);
            v[i] = (up ? v[i + h] : v[i]) + b;
        }
    }
    return v[0];
}

"""

_GRAM_SRC = r"""
// ───────────── kernel A: block = 256 threads = one (row, hv); parallel part ─────────────
// dynamic smem: sSr[V][DS] | sPhi[D][G] | hq[V] | hk[V] | qb[K] | kb[K], D=max(R,G).
// The first m rows of sPhi are the full M even when m>R; rows [:R] are Phi_r.
// writes scratch: Phi_D[D][G] (+eta on the diagonal, columns >= m zero) and raw anchors [2][G]; U copy.
#define NT 256
template <int K, int V>
__global__ void __launch_bounds__(NT, 4)
gdn_ls6_gram_kernel(
    const float* __restrict__ h0, const int* __restrict__ rows, const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_map, const float* __restrict__ fz_qbar, const float* __restrict__ fz_kbar,
    const int* __restrict__ ls6_mh, float* __restrict__ ls6_ubar, float* __restrict__ scratch,
    long s_h0_slot, long s_h0_h, long s_fzb_slot, long s_u_slot,
    int H, int HV, int G, int R, float ridge, int stage)
{
    const int r_i = blockIdx.x, hv = blockIdx.y;
    if (r_i >= *n_ptr) return;
    const int m = ls6_mh[hv];
    if (m <= 0) return;
    const long sidx = rows[r_i];
    if (sidx <= 0) return;
    const long cidx = ls6_map ? ls6_map[sidx] : sidx;
    const int h = hv / (HV / H);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int D = max(R, G), RS = D + 4;
    extern __shared__ __align__(16) float dsm[];
    float* sSr = dsm;                          // [V][RS]
    float* sPhi = sSr + V * RS;                // [D][G]
    float* hq = sPhi + D * G;                  // [V]
    float* hk = hq + V;                        // [V]
    float* qb = hk + V;                        // [K]
    float* kb = qb + K;                        // [K]
    __shared__ float s_eta;
    __shared__ float s_sq[NT / 32];

    const float* pS = h0 + sidx * s_h0_slot + hv * s_h0_h;
    const float* pq = fz_qbar + cidx * s_fzb_slot + h * K;
    const float* pk = fz_kbar + cidx * s_fzb_slot + h * K;
    for (int i = t; i < K; i += NT) { qb[i] = pq[i]; kb[i] = pk[i]; }
    __syncthreads();
    // ── pass 1: rows v = warp + 8i (16 per warp). sumsq, S qbar, S kbar, stage columns [:R] ──
    constexpr int NV = V / (NT / 32);
    float sumsq = 0.f;
    float part[32];                            // [0..15] = dq per row, [16..31] = dk per row
    {
        const int c = lane * 4;
        const float q0 = qb[c], q1 = qb[c + 1], q2 = qb[c + 2], q3 = qb[c + 3];
        const float k0 = kb[c], k1 = kb[c + 1], k2 = kb[c + 2], k3 = kb[c + 3];
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            float4 xs_[NV / 2];
            #pragma unroll
            for (int i = 0; i < NV / 2; ++i)
                xs_[i] = *((const float4*)(pS + (long)(warp + (NT / 32) * (half * NV / 2 + i)) * K + c));
            #pragma unroll
            for (int i = 0; i < NV / 2; ++i) {
                const int ii = half * NV / 2 + i, v = warp + (NT / 32) * ii;
                const float4 x = xs_[i];
                sumsq += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
                part[ii] = x.x * q0 + x.y * q1 + x.z * q2 + x.w * q3;
                part[NV + ii] = x.x * k0 + x.y * k1 + x.z * k2 + x.w * k3;
                if (c < D) *((float4*)(sSr + v * RS + c)) = x;
            }
        }
    }
    {
        const float red = xpose32(part, lane);           // lane l: sum of part[l] over lanes
        const int ii = lane & (NV - 1), v = warp + (NT / 32) * ii;
        if (lane < NV) hq[v] = red; else hk[v] = red;
    }
    sumsq = warp_sum(sumsq);
    if (lane == 0) s_sq[warp] = sumsq;
    __syncthreads();
    if (t == 0) { float a = 0.f; for (int w = 0; w < NT / 32; ++w) a += s_sq[w]; s_eta = ridge * a / (float)K; }
    __syncthreads();
    const float eta = s_eta;
    if (stage == 1) return;
    // ── Phi_r[tt][g] = sum_v S[v][tt] S[v][g] + eta [tt==g]: 4x4 register tiles, symmetric block mirrored ──
    const int ntile = (D / 4) * (G / 4);
    for (int o = t; o < ntile; o += NT) {
        const int tt0 = (o % (D / 4)) * 4, g0 = (o / (D / 4)) * 4;
        const bool mirror = tt0 < G;                     // tile lies in the symmetric M block
        if (g0 >= m) {
            #pragma unroll
            for (int a = 0; a < 4; ++a)
                #pragma unroll
                for (int b = 0; b < 4; ++b) sPhi[(tt0 + a) * G + g0 + b] = 0.f;
            continue;
        }
        if (mirror && tt0 > g0 && tt0 < m) continue;     // provided by the transposed tile (g0' = tt0 < m)
        float acc[4][4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) { acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.f; }
        for (int v = 0; v < V; ++v) {
            const float4 xa = *((const float4*)(sSr + v * RS + tt0));
            const float4 xb = *((const float4*)(sSr + v * RS + g0));
            const float ta[4] = {xa.x, xa.y, xa.z, xa.w}, gb[4] = {xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int a = 0; a < 4; ++a)
                #pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(ta[a], gb[b], acc[a][b]);
        }
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int tt = tt0 + a, g = g0 + b;
                const float val = acc[a][b] + ((tt == g) ? eta : 0.f);
                sPhi[tt * G + g] = (g < m) ? val : 0.f;
                if (mirror && tt < m) sPhi[g * G + tt] = val;     // Phi_r[g][tt], g < m always here
            }
    }
    __syncthreads();                                     // sPhi complete
    if (stage == 2) return;
    // ── raw anchors: U^T (S qbar) + eta qbar[:m] ──
    float* sc = scratch + ((long)r_i * HV + hv) * (D * G + 2 * G);
    if (t < G) {
        float aq0 = 0.f, ak0 = 0.f;
        if (t < m) {
            for (int v = 0; v < V; ++v) { const float sv = sSr[v * RS + t]; aq0 = fmaf(sv, hq[v], aq0); ak0 = fmaf(sv, hk[v], ak0); }
            aq0 += eta * qb[t]; ak0 += eta * kb[t];
        }
        sc[D * G + t] = aq0; sc[D * G + G + t] = ak0;
    }
    // ── U copy out: lane -> v (coalesced stores), float4 over g (conflict-free with RS = R+4) ──
    float* pu = ls6_ubar + cidx * s_u_slot + (long)hv * G * V;
    for (int o = t; o < (G / 4) * V; o += NT) {
        const int v = o % V, g0 = (o / V) * 4;
        const float4 x = *((const float4*)(sSr + v * RS + g0));
        pu[(long)(g0 + 0) * V + v] = (g0 + 0 < m) ? x.x : 0.f;
        pu[(long)(g0 + 1) * V + v] = (g0 + 1 < m) ? x.y : 0.f;
        pu[(long)(g0 + 2) * V + v] = (g0 + 2 < m) ? x.z : 0.f;
        pu[(long)(g0 + 3) * V + v] = (g0 + 3 < m) ? x.w : 0.f;
    }
    for (int o = t; o < D * G / 4; o += NT) ((float4*)sc)[o] = ((const float4*)sPhi)[o];
}

"""

# Shared with gdn_flush_stream_cuda (which produces the same scratch layout).
_SOLVE_SRC = r"""
#define NTB 128
// Forward/back substitution for CPW columns per warp with the unknowns held in registers
// (lane r0 owns rows r0 + q*RL); L is broadcast per row with one shuffle per step.
template <int CPW, int NQ>
__device__ __forceinline__ void ls6_solve_cols(
    const float* __restrict__ sM, float* __restrict__ sX, const float* __restrict__ s_rd,
    int m, int ncol, int GS, int NCS, int lane, int warp)
{
    constexpr int RL = 32 / CPW, LRL = (RL == 32) ? 5 : ((RL == 16) ? 4 : 3);
    const int cl = lane / RL, r0 = lane % RL;
    int off[NQ];
    #pragma unroll
    for (int q = 0; q < NQ; ++q) off[q] = min(r0 + q * RL, m - 1) * GS;
    for (int cb = warp * CPW; cb < ncol; cb += (NTB / 32) * CPW) {
        const int ci = min(cb + cl, ncol - 1);
        float x[NQ];
        #pragma unroll
        for (int q = 0; q < NQ; ++q) { const int i = r0 + q * RL; x[q] = (i < m) ? sX[i * NCS + ci] : 0.f; }
        for (int k = 0; k < m; ++k) {                    // L y = b
            const int kq = k >> LRL, kr = k & (RL - 1);
            float v = x[0];
            #pragma unroll
            for (int q = 1; q < NQ; ++q) if (kq == q) v = x[q];
            const float yk = __shfl_sync(0xffffffffu, v * s_rd[k], cl * RL + kr);
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float upd = fmaf(-sM[off[q] + k], yk, x[q]);
                if (q > kq || (q == kq && r0 > kr)) x[q] = upd;
                else if (q == kq && r0 == kr) x[q] = yk;
            }
        }
        for (int k = m - 1; k >= 0; --k) {               // L^T x = y
            const int kq = k >> LRL, kr = k & (RL - 1);
            float v = x[0];
            #pragma unroll
            for (int q = 1; q < NQ; ++q) if (kq == q) v = x[q];
            const float xk = __shfl_sync(0xffffffffu, v * s_rd[k], cl * RL + kr);
            const float* lk = sM + k * GS;
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float upd = fmaf(-lk[min(r0 + q * RL, m - 1)], xk, x[q]);
                if (q < kq || (q == kq && r0 < kr)) x[q] = upd;
                else if (q == kq && r0 == kr) x[q] = xk;
            }
        }
        if (cb + cl < ncol) {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) { const int i = r0 + q * RL; if (i < m) sX[i * NCS + ci] = x[q]; }
        }
    }
}

// ───────────── kernel B: block = 128 threads = one (row, hv); Cholesky, solves, outputs ─────────────
// dynamic smem: sM[G][GS] (GS = G+1) | sX[m][NCS] (NCS = ncol|1; tail columns then the 2 anchors)
__global__ void __launch_bounds__(NTB)
gdn_ls6_solve_kernel(
    const int* __restrict__ rows, const int* __restrict__ n_ptr, const int* __restrict__ ls6_map,
    const float* __restrict__ fz_qbar, const float* __restrict__ fz_kbar, const int* __restrict__ ls6_mh,
    const float* __restrict__ scratch, float* __restrict__ ls6_phi, float* __restrict__ ls6_aq, float* __restrict__ ls6_ak,
    long s_fzb_slot, long s_phi_slot, long s_a_slot, int H, int HV, int K, int G, int R, int stage)
{
    const int r_i = blockIdx.x, hv = blockIdx.y;
    if (r_i >= *n_ptr) return;
    const int m = ls6_mh[hv];
    if (m <= 0) return;
    const long sidx = rows[r_i];
    if (sidx <= 0) return;
    const long cidx = ls6_map ? ls6_map[sidx] : sidx;
    const int h = hv / (HV / H);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int D = max(R, G), tail = max(R - m, 0);
    const int GS = G + 1, ncol = tail + 2, NCS = ncol | 1;
    extern __shared__ __align__(16) float dsm[];
    float* sM = dsm;                           // [G][GS]
    float* sX = sM + G * GS;                   // [m][NCS]
    __shared__ float s_rd[128];                // 1 / Cholesky diagonal
    const float* sc = scratch + ((long)r_i * HV + hv) * (D * G + 2 * G);
    const int m4 = (m + 3) & ~3;                     // padded with identity rows/cols: static 4-wide panels
    for (int o = t; o < m4 * m4; o += NTB) {
        const int i = o / m4, j = o % m4;
        sM[i * GS + j] = (i < m && j < m) ? sc[i * G + j] : ((i == j) ? 1.f : 0.f);
    }
    for (int o = t; o < m * ncol; o += NTB) {
        const int i = o / ncol, c = o % ncol;
        sX[i * NCS + c] = (c < tail) ? sc[(m + c) * G + i] : sc[D * G + (c - tail) * G + i];
    }
    __syncthreads();
    if (stage == 4) { if (t == 0) ls6_aq[cidx * s_a_slot + hv * G] = sX[0]; return; }
    // ── Cholesky, right-looking, width-4 panels; thread = row; 2 syncs per panel ──
    for (int jb = 0; jb < m4; jb += 4) {
        float L[4][4], rd[4];                            // the 4x4 diagonal block, factored redundantly
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            float sd = sM[(jb + d) * GS + jb + d];
            #pragma unroll
            for (int e = 0; e < d; ++e) sd -= L[d][e] * L[d][e];
            rd[d] = rsqrtf(fmaxf(sd, 1e-30f));
            #pragma unroll
            for (int c = d + 1; c < 4; ++c) {
                float v = sM[(jb + c) * GS + jb + d];
                #pragma unroll
                for (int e = 0; e < d; ++e) v -= L[c][e] * L[d][e];
                L[c][d] = v * rd[d];
            }
        }
        float l[4] = {0.f, 0.f, 0.f, 0.f};               // this row's panel entries
        const int i = t;
        const bool below = i >= jb + 4 && i < m4;
        if (below) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float v = sM[i * GS + jb + c];
                #pragma unroll
                for (int e = 0; e < c; ++e) v -= l[e] * L[c][e];
                l[c] = v * rd[c];
            }
        }
        __syncthreads();                                 // everyone read the old panel
        if (below) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) sM[i * GS + jb + c] = l[c];
        } else {
            #pragma unroll
            for (int d = 0; d < 4; ++d) if (i == jb + d) {
                s_rd[i] = rd[d];
                #pragma unroll
                for (int e = 0; e < d; ++e) sM[i * GS + jb + e] = L[d][e];
            }
        }
        __syncthreads();                                 // panel visible
        if (below) {
            #pragma unroll 4
            for (int k = jb + 4; k <= i; ++k) {
                float a = sM[i * GS + k];
                const float* lk = sM + k * GS + jb;
                #pragma unroll
                for (int c = 0; c < 4; ++c) a = fmaf(-l[c], lk[c], a);
                sM[i * GS + k] = a;
            }
        }
        __syncthreads();
    }
    if (stage == 5) { if (t == 0) ls6_aq[cidx * s_a_slot + hv * G] = s_rd[0]; return; }
    // ── solves in registers: warp handles CPW columns (RL = 32/CPW row lanes, NQ rows per lane) ──
    if (m > 32 || ncol <= 8) ls6_solve_cols<1, 4>(sM, sX, s_rd, m, ncol, GS, NCS, lane, warp);
    else if (m > 16 || ncol <= 16) ls6_solve_cols<2, 2>(sM, sX, s_rd, m, ncol, GS, NCS, lane, warp);
    else ls6_solve_cols<4, 2>(sM, sX, s_rd, m, ncol, GS, NCS, lane, warp);
    __syncthreads();
    // ── outputs: Phibar^T [G][R] (identity block for tt < m, tail from X), tail anchors ──
    float* pphi = ls6_phi + cidx * s_phi_slot + (long)hv * G * R;
    for (int g = warp; g < G; g += NTB / 32)
        for (int tt = lane; tt < R; tt += 32) {
            float v = 0.f;
            if (g < m) v = (tt < m) ? ((tt == g) ? 1.f : 0.f) : sX[g * NCS + (tt - m)];
            pphi[g * R + tt] = v;
        }
    if (t < G) {
        float aq = 0.f, ak = 0.f;
        if (t < m) {
            const float* pq = fz_qbar + cidx * s_fzb_slot + h * K;
            const float* pk = fz_kbar + cidx * s_fzb_slot + h * K;
            aq = sX[t * NCS + tail] - ((t < R) ? pq[t] : 0.f);
            ak = sX[t * NCS + tail + 1] - ((t < R) ? pk[t] : 0.f);  // truncated identity block
            #pragma unroll 4
            for (int tt = m; tt < R; ++tt) { const float pv = sX[t * NCS + (tt - m)]; aq = fmaf(-pv, pq[tt], aq); ak = fmaf(-pv, pk[tt], ak); }
        }
        ls6_aq[cidx * s_a_slot + hv * G + t] = aq;
        ls6_ak[cidx * s_a_slot + hv * G + t] = ak;
    }
}

"""

_SRC = _COMMON_SRC + _GRAM_SRC + _SOLVE_SRC + r"""
void gdn_ls6_epilogue(torch::Tensor h0, torch::Tensor rows, torch::Tensor n_ptr, c10::optional<torch::Tensor> ls6_map,
    torch::Tensor fz_qbar, torch::Tensor fz_kbar, torch::Tensor ls6_mh, torch::Tensor ls6_ubar, torch::Tensor ls6_phi,
    torch::Tensor ls6_aq, torch::Tensor ls6_ak, torch::Tensor scratch, int max_rows, int H, int HV, int K, int V, int G, int R, double ridge, int stage)
{
    TORCH_CHECK(K == 128 && V == 128, "epilogue: K=V=128 only");
    TORCH_CHECK(G >= 4 && G <= 128 && R >= 4 && R <= K && R % 4 == 0, "epilogue: G/R");
    const int D = std::max(R, G);
    TORCH_CHECK(scratch.numel() >= (long)max_rows * HV * (D * G + 2 * G), "epilogue: scratch too small");
    const size_t smemA = (size_t)(V * (D + 4) + D * G + 2 * V + 2 * K) * 4;
    int maxX = 0;
    for (int m = 1; m <= G; ++m) maxX = std::max(maxX, m * ((std::max(R - m, 0) + 2) | 1));
    const size_t smemB = (size_t)(G * (G + 1) + maxX) * 4;
    auto st = at::cuda::getCurrentCUDAStream().stream();
    static size_t attrA = 0, attrB = 0;
    if (attrA < smemA) { cudaFuncSetAttribute(gdn_ls6_gram_kernel<128, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA); attrA = smemA; }
    if (attrB < smemB) { cudaFuncSetAttribute(gdn_ls6_solve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB); attrB = smemB; }
    if (max_rows <= 0) return;
    dim3 grid(max_rows, HV);
    gdn_ls6_gram_kernel<128, 128><<<grid, NT, smemA, st>>>(
        h0.data_ptr<float>(), rows.data_ptr<int>(), n_ptr.data_ptr<int>(),
        ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
        fz_qbar.data_ptr<float>(), fz_kbar.data_ptr<float>(), ls6_mh.data_ptr<int>(),
        ls6_ubar.data_ptr<float>(), scratch.data_ptr<float>(),
        h0.stride(0), h0.stride(1), fz_qbar.stride(0), ls6_ubar.stride(0),
        H, HV, G, R, (float)ridge, stage);
    if (stage > 0 && stage < 4) return;
    gdn_ls6_solve_kernel<<<grid, NTB, smemB, st>>>(
        rows.data_ptr<int>(), n_ptr.data_ptr<int>(), ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
        fz_qbar.data_ptr<float>(), fz_kbar.data_ptr<float>(), ls6_mh.data_ptr<int>(), scratch.data_ptr<float>(),
        ls6_phi.data_ptr<float>(), ls6_aq.data_ptr<float>(), ls6_ak.data_ptr<float>(),
        fz_qbar.stride(0), ls6_phi.stride(0), ls6_aq.stride(0), H, HV, K, G, R, stage);
}
"""

_CPP = r"""
#include <torch/extension.h>
void gdn_ls6_epilogue(torch::Tensor h0, torch::Tensor rows, torch::Tensor n_ptr, c10::optional<torch::Tensor> ls6_map,
    torch::Tensor fz_qbar, torch::Tensor fz_kbar, torch::Tensor ls6_mh, torch::Tensor ls6_ubar, torch::Tensor ls6_phi,
    torch::Tensor ls6_aq, torch::Tensor ls6_ak, torch::Tensor scratch, int max_rows, int H, int HV, int K, int V, int G, int R, double ridge, int stage);
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        bd = os.environ.get("NS_GDN_CUDA_BUILD_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gdn_step_build"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load_inline(
            name="ns_gdn_ls6_epilogue_v13", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["gdn_ls6_epilogue"], build_directory=bd,
            extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


def ls6_ridge() -> float:
    return float(os.environ.get("NS_GDN_LS6_RIDGE", "0.1"))


def gdn_ls6_epilogue(h0, rows, n_ptr, ls6_map, fz_qbar, fz_kbar, ls6_mh, ls6_ubar, ls6_phi,
                     ls6_aq, ls6_ak, max_rows, H, HV, K, V, G, R, ridge=None):
    """Rebuild U / Phibar[:r] / tail anchors for rows[0:n_ptr[0]] (state row indices).

    Args:
        h0: (NX,HV,V,K) fp32 state after the flush fold, R' coordinates.
        rows: int32 state row indices; n_ptr: int32 scalar tensor holding the count.
        ls6_map: optional int32 (NX,) state row -> compact slot.
        fz_qbar/fz_kbar: (NS,H,K) window means (rotated); ls6_mh: (HV,) latch widths.
        ls6_ubar (NS,HV,G,V), ls6_phi (NS,HV,G,R), ls6_aq/ak (NS,HV,G): outputs.
    """
    # vLLM packs every layer state into a larger cache page.  The (HV,V,K)
    # payload is dense, but consecutive sequence rows have the page stride,
    # so the layer view is intentionally not globally contiguous.  The CUDA
    # kernel accepts that layout through s_h0_slot/s_h0_h.
    h0_dense_tail = (
        h0.dtype == torch.float32
        and h0.dim() == 4
        and tuple(h0.shape[1:]) == (HV, V, K)
        and h0.stride(3) == 1
        and h0.stride(2) == K
        and h0.stride(1) == V * K
        and h0.stride(0) % 4 == 0
    )
    if not h0_dense_tail:
        raise TypeError(
            "gdn_ls6_epilogue: h0 must be fp32 with a dense (HV,V,K) "
            f"tail; shape={tuple(h0.shape)} stride={h0.stride()} dtype={h0.dtype}"
        )
    for nm, tt in (("fz_qbar", fz_qbar), ("fz_kbar", fz_kbar), ("ls6_ubar", ls6_ubar),
                   ("ls6_phi", ls6_phi), ("ls6_aq", ls6_aq), ("ls6_ak", ls6_ak)):
        if tt.dtype != torch.float32 or not tt.is_contiguous():
            raise TypeError(f"gdn_ls6_epilogue: {nm} must be contiguous fp32 ({tt.dtype})")
    if rows.dtype != torch.int32 or n_ptr.dtype != torch.int32 or ls6_mh.dtype != torch.int32:
        raise TypeError("gdn_ls6_epilogue: rows/n_ptr/ls6_mh must be int32")
    if tuple(ls6_phi.shape[1:]) != (HV, G, R):
        raise ValueError(f"ls6_phi shape {tuple(ls6_phi.shape)} != (NS,{HV},{G},{R})")
    D = max(R, G)
    scratch = torch.empty(int(max_rows) * HV * (D * G + 2 * G), dtype=torch.float32, device=h0.device)
    _ext().gdn_ls6_epilogue(h0, rows, n_ptr, ls6_map, fz_qbar, fz_kbar, ls6_mh, ls6_ubar, ls6_phi,
                            ls6_aq, ls6_ak, scratch, int(max_rows), H, HV, K, V, G, R,
                            ls6_ridge() if ridge is None else float(ridge),
                            int(os.environ.get("NS_GDN_EPI_STAGE", "99")))
