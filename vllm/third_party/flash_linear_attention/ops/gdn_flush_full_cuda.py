# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-coordinate GDN exact flush with no input-rank or anchor path.

Keeps the established FP32 hi/lo tensor-core arithmetic and raw-WY contract.
Metadata is U=H[:,:m] and P=(U^T U+eta I)^-1 (H^T H+eta I)[:m,:].
Full-width P is implicit identity. No changes to the reference implementation.
"""

import functools
import math
import os
from pathlib import Path

import torch

_SRC = r"""

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <unordered_map>
#include <ATen/cuda/CUDAContext.h>

#define FULL 0xffffffffu
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}


#define NTB 128
// One thread owns a tail RHS. Static row indices keep the vector in registers;
// all RHS advance together and broadcast the same Cholesky entry from shared memory.
template <int M>
__device__ __forceinline__ void solve_rhs_registers(
    const float* sM, float* sX, const float* s_rd,
    int m, int ncol, int GS, int NCS)
{
    const int ci = threadIdx.x;
    if (ci >= ncol) return;
    float x[M];
    #pragma unroll
    for (int i = 0; i < M; ++i) x[i] = i < m ? sX[i * NCS + ci] : 0.f;
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        if (i < m) {
            #pragma unroll
            for (int k = 0; k < i; ++k) x[i] = fmaf(-sM[i * GS + k], x[k], x[i]);
            x[i] *= s_rd[i];
        }
    }
    #pragma unroll
    for (int i = M - 1; i >= 0; --i) {
        if (i < m) {
            #pragma unroll
            for (int k = M - 1; k > i; --k)
                if (k < m) x[i] = fmaf(-sM[k * GS + i], x[k], x[i]);
            x[i] *= s_rd[i];
        }
    }
    #pragma unroll
    for (int i = 0; i < M; ++i) if (i < m) sX[i * NCS + ci] = x[i];
}

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

// One (row, hv) per block, filtered by width bucket. Shared memory contains
// the bucket-sized Gram (allocation-sized above 64) and only the tail RHS.
template <int M>
__global__ void __launch_bounds__(NTB)
gdn_ls6_solve_kernel(
    const int* __restrict__ rows, const int* __restrict__ n_ptr, const int* __restrict__ ls6_map,
    const int* __restrict__ ls6_mh,
    const float* __restrict__ scratch, float* __restrict__ ls6_phi,
    long s_phi_slot, int HV, int G)
{
    const int r_i = blockIdx.x, hv = blockIdx.y;
    if (r_i >= *n_ptr) return;
    const int m = ls6_mh[hv];
    constexpr int LOWER = M == 8 ? 0 : M == 48 ? 32 : M == 64 ? 48 : M / 2;
    if (m <= LOWER || m > M || m == 128) return;
    const long sidx = rows[r_i];
    if (sidx <= 0) return;
    const long cidx = ls6_map ? ls6_map[sidx] : sidx;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int K = 128, D = K;
    const int tail = K - m;
    const int stride_m = M == 128 ? G : M;
    const int GS = stride_m + 1, ncol = tail, NCS = ncol | 1;
    extern __shared__ __align__(16) float dsm[];
    float* sM = dsm;                           // [stride_m][GS]
    float* sX = sM + stride_m * GS;             // [m][NCS]
    __shared__ float s_rd[128];                // 1 / Cholesky diagonal
    const float* sc = scratch + ((long)r_i * HV + hv) * (D * G);
    const int m4 = (m + 3) & ~3;                     // padded with identity rows/cols: static 4-wide panels
    for (int o = t; o < m4 * m4; o += NTB) {
        const int i = o / m4, j = o % m4;
        sM[i * GS + j] = (i < m && j < m) ? sc[i * G + j] : ((i == j) ? 1.f : 0.f);
    }
    for (int o = t; o < m * ncol; o += NTB) {
        const int i = o / ncol, c = o % ncol;
        sX[i * NCS + c] = sc[(m + c) * G + i];
    }
    __syncthreads();
    // Cholesky, right-looking, width-four panels; thread = row, three barriers per panel.
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
    // Small widths solve all RHS together; large widths distribute rows across lanes.
    if constexpr (M <= 64) solve_rhs_registers<M>(sM, sX, s_rd, m, ncol, GS, NCS);
    else ls6_solve_cols<1, 4>(sM, sX, s_rd, m, ncol, GS, NCS, lane, warp);
    __syncthreads();
    // ── outputs: Phibar^T [G][K] (identity block for tt < m, tail from X) ──
    float* pphi = ls6_phi + cidx * s_phi_slot + (long)hv * G * K;
    for (int g = warp; g < G; g += NTB / 32)
        for (int tt = lane; tt < K; tt += 32) {
            float v = 0.f;
            if (g < m) v = (tt < m) ? ((tt == g) ? 1.f : 0.f) : sX[g * NCS + (tt - m)];
            pphi[g * K + tt] = v;
        }

}

#define NT 256
template <int K, int V>
__global__ void __launch_bounds__(NT, 4)
gdn_ls6_gram_kernel(
    const float* __restrict__ h0, const int* __restrict__ rows, const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_map,
    const int* __restrict__ ls6_mh, float* __restrict__ ls6_ubar, float* __restrict__ scratch,
    long s_h0_slot, long s_h0_h, long s_u_slot,
    int HV, int G, float ridge)
{
    const int r_i = blockIdx.x, hv = blockIdx.y;
    if (r_i >= *n_ptr) return;
    const int m = ls6_mh[hv];
    if (m <= 0) return;
    const long sidx = rows[r_i];
    if (sidx <= 0) return;
    const long cidx = ls6_map ? ls6_map[sidx] : sidx;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int D = K, RS = K + 4;
    extern __shared__ __align__(16) float dsm[];
    float* sSr = dsm;                          // [V][RS]
    float* sPhi = sSr + V * RS;                // [D][G]
    __shared__ float s_eta;
    __shared__ float s_sq[NT / 32];

    const float* pS = h0 + sidx * s_h0_slot + hv * s_h0_h;
    if (m == K) {
        float* pu = ls6_ubar + cidx * s_u_slot + (long)hv * G * V;
        for (int o = t; o < K * V; o += NT) pu[o] = pS[(o % V) * K + o / V];
        return;
    }
    constexpr int NV = V / (NT / 32);
    float sumsq = 0.f;
    {
        const int c = lane * 4;
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
                if (c < D) *((float4*)(sSr + v * RS + c)) = x;
            }
        }
    }
    sumsq = warp_sum(sumsq);
    if (lane == 0) s_sq[warp] = sumsq;
    __syncthreads();
    if (t == 0) { float a = 0.f; for (int w = 0; w < NT / 32; ++w) a += s_sq[w]; s_eta = ridge * a / (float)K; }
    __syncthreads();
    const float eta = s_eta;
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
    float* sc = scratch + ((long)r_i * HV + hv) * (D * G);
    // ── U copy out: lane -> v (coalesced stores), float4 over g (conflict-free with RS = K+4) ──
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


#include <cuda.h>
#include <cuda_fp16.h>

#define SK 128
#define SV 128
#define SW 16
#define NTS 256
#define LDD 132     // d ring [W][LDD] f32 (x2 buffers); the consumed one holds Phi partial sums
#define LDK 128
#define KSS (SW * SK) // K_r split [W][SK] f16 (hi | lo), 16 B chunks XOR-swizzled by (row&7)
#define LDT2 16     // T'' split transposed [16 c][LDT2] f16
#define SEXP 10
#define SPHI_MAX 2048         // Phi partial-sum chunk (floats): sPhiA, then the store staging
#define TBL 64      // item table entries (a CTA's work list, index chains resolved up front)
#define RING_H (2 * KSS + 2 * 16 * LDT2)
#define TE 8        // entry: sidx, cidx, hv, i_h, rh, rt, m, kK | kT << 16
#define RING_F (SW * LDD + SW)                             // d rows | gates (f32)
#define SMEM_STREAM ((SV * SK + 2 * RING_F + SPHI_MAX + TBL * TE) * 4 + 2 * RING_H * 2)
// Key hi/lo records per request/key head; T records per request/value head.
#define PREP_K_BYTES (2 * SW * SK * 2)
#define PREP_T_BYTES (2 * 16 * 16 * 2)

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x = fmaxf(x, __shfl_xor_sync(FULL, x, o));
    return x;
}
__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ int lds_i32(unsigned a) {
    int v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
    return v;
}
// bulk copies (TMA, 1D) driven by mbarriers: the state rows and the rings bypass the LSU pipe
__device__ __forceinline__ void mbar_init(unsigned a, unsigned cnt) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(unsigned a, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned a, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
                 " @!p bra WAIT_%=;\n}" :: "r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(unsigned dst, const void* src, unsigned bytes, unsigned mbar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
// 2D tensor copy (box 32 cols x 16 rows of h0 viewed as [rows][128] f32, 128 B swizzle)
__device__ __forceinline__ void tma_load_2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned mbar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"((unsigned long long)map), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}

__device__ __forceinline__ int scale_exp(float mx) { return mx > 0.f ? min(40, SEXP - ilogbf(mx)) : 0; }
__device__ __forceinline__ float amax4(float4 x) {
    return fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
}
__device__ __forceinline__ float block_max(float v, float* s_red, int t) {
    v = warp_max(v);
    if ((t & 31) == 0) s_red[t >> 5] = v;
    __syncthreads();
    float m = s_red[0];
    #pragma unroll
    for (int i = 1; i < NTS / 32; ++i) m = fmaxf(m, s_red[i]);
    return m;
}
// (a, b) * sc as packed fp16 hi/lo pairs: hi + lo carries 22 significant bits.
__device__ __forceinline__ void split_u(float a, float b, float sc, unsigned& h, unsigned& l) {
    const float as = a * sc, bs = b * sc;
    const __half2 hh = __floats2half2_rn(as, bs);
    const float2 hf = __half22float2(hh);
    const __half2 ll = __floats2half2_rn(as - hf.x, bs - hf.y);
    h = *reinterpret_cast<const unsigned*>(&hh);
    l = *reinterpret_cast<const unsigned*>(&ll);
}
__device__ __forceinline__ __half2 as_h2(unsigned u) { return *reinterpret_cast<const __half2*>(&u); }
// C fragments of n-tiles (2q, 2q+1) -> A fragment for k-block q (logical order).
__device__ __forceinline__ void c2a(const float* c0, const float* c1, float sc, unsigned* ah, unsigned* al) {
    split_u(c0[0], c0[1], sc, ah[0], al[0]);
    split_u(c0[2], c0[3], sc, ah[1], al[1]);
    split_u(c1[0], c1[1], sc, ah[2], al[2]);
    split_u(c1[2], c1[3], sc, ah[3], al[3]);
}
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma3(float* c, const unsigned* ah, const unsigned* al,
                                     unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    mma16816(c, ah, bl0, bl1);
    mma16816(c, al, bh0, bh1);
    mma16816(c, ah, bh0, bh1);
}
__device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm2t(unsigned* r, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
}

__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
// 8x8 b16 transpose within the warp (fragment layout in == out).
__device__ __forceinline__ unsigned movtrans(unsigned a) {
    unsigned d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
    return d;
}

#define ROW_OF(hh) (16 * warp + g + 8 * (hh))

// acc[nt][e]: rows ROW_OF(e>>1), cols 8nt + 2c + (e&1).  The warp's 16 state rows land in its
// own 8 KB block of sS by four tensor copies on a per-warp mbarrier (issued an item ahead, right
// after the previous fragments were read); S_W goes back to h0 straight from the fragments.
// Block layout [4 col-chunks][16 rows][32] f32 with the TMA 128 B swizzle on a 1 KB-aligned base:
// element (row, col) sits at (col>>5)*512 + row*32 + ((((col&31)>>2) ^ (row&7))<<2) + (col&3).
// For the fragment (row = g + 8hh, col = 8nt + 2c) the XOR term is (2(nt&3)) ^ ((c>>1) ^ g), so
// every address is a lane constant plus an immediate.
__device__ __forceinline__ void load_state(float (&acc)[16][4], const float* sSw, int g, int c) {
    const int x = (c >> 1) ^ g;
    const float* base = sSw + (g << 5) + ((c & 1) << 1);
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int xo = ((2 * (nt & 3)) ^ x) << 2;
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const float2 v = *(const float2*)(base + (nt >> 2) * 512 + hh * 256 + xo);
            acc[nt][2 * hh] = v.x; acc[nt][2 * hh + 1] = v.y;
        }
    }
}
__device__ __forceinline__ void store_state(const float (&acc)[16][4], float* ph, int warp, int g, int c) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        float* pr = ph + (long)ROW_OF(hh) * SK + 2 * c;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) *(float2*)(pr + 8 * nt) = make_float2(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
    }
}
__device__ __forceinline__ void issue_state(unsigned sSw_u, const CUtensorMap* map, int row0, unsigned mb, int lane) {
    if (lane == 0) {
        mbar_expect_tx(mb, 16 * SK * 4);
        #pragma unroll
        for (int k = 0; k < 4; ++k) tma_load_2d(sSw_u + 2048u * k, map, 32 * k, row0, mb);
    }
    __syncwarp();
}
// K split element (row, 8-column chunk) -> swizzled half offset (ldmatrix rows hit distinct banks);
// the prep records are written in this smem image order
__device__ __forceinline__ int ks_off(int row, int chunk) { return row * SK + 8 * (chunk ^ (row & 7)); }
__device__ __forceinline__ void issue_rings(float* sD, __half* sK, __half* sT2, float* sG,
                                            const float* d_src, const float* g_src,
                                            const unsigned char* k_src, const unsigned char* t_src,
                                            bool tq, unsigned mb, int lane) {
    if (lane == 0)
        mbar_expect_tx(mb, SW * SV * 4 + SW * 4 + PREP_K_BYTES + (tq ? PREP_T_BYTES : 0u));
    __syncwarp();
    if (lane < SW) bulk_g2s(smem_u32(sD + lane * LDD), d_src + lane * SV, SV * 4, mb);
    else if (lane == 16) bulk_g2s(smem_u32(sK), k_src, PREP_K_BYTES, mb);
    else if (lane == 17) bulk_g2s(smem_u32(sG), g_src, SW * 4, mb);
    else if (tq && lane == 18) bulk_g2s(smem_u32(sT2), t_src, PREP_T_BYTES, mb);
}
// rings of the item in table entry eu (item table: see fill_tbl)
#define RINGS_OF(b, eu) sD0 + (b) * RING_F, sK0 + (b) * RING_H, sK0 + (b) * RING_H + 2 * KSS, \
    sD0 + (b) * RING_F + SW * LDD, \
    d_cache + (long)lds_i32(eu) * s_d_slot + (long)lds_i32(eu + 8) * SW * SV, \
    g_cache + (long)lds_i32(eu) * s_g_slot + (long)lds_i32(eu + 8) * SW, \
    prep_k + (long)lds_i32(eu + 16) * PREP_K_BYTES, prep_t + (long)lds_i32(eu + 20) * PREP_T_BYTES, \
    lds_i32(eu + 24) > 0

// Prep kernel, one 128-thread block per (row, h): K_r split (hi/lo f16, block exponent kK),
// written as records the stream kernel pulls in with bulk copies.
__global__ void __launch_bounds__(128)
gdn_flush_prep_kernel(
    const float* __restrict__ k_cache, const float* __restrict__ g_cache, const int* __restrict__ flush_list,
    const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_map, const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    unsigned char* __restrict__ prep_k, unsigned char* __restrict__ prep_t,
    int* __restrict__ prep_i, int* __restrict__ prep_it,
    long s_k_slot, long s_g_slot, long s_beta_slot, int H, int HV)
{
    __shared__ __align__(128) __half sKh[SW * LDK], sKl[SW * LDK];
    __shared__ float sG[256], sGam[4][256], s_red[4];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, c = lane & 3;
    const int l07 = lane & 7, l3 = (lane >> 3) & 1, l4 = lane >> 4;
    const int rh = blockIdx.x;
    if (rh >= n_ptr[0] * H) return;
    const int hpg = HV / H, r = rh / H, h = rh - r * H, hv = h * hpg + warp;
    const long sidx = flush_list[r];
    const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
    const int m_w = (warp < hpg) ? ls6_mh[hv] : 0;
    const bool any_exact = __syncthreads_or(m_w > 0);
    // ── K_r exponent + split ──
    const float* pk = k_cache + sidx * s_k_slot + (long)h * SW * SK;
    float4 kv[4];
    float mk = 0.f;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int i = t + q * 128, s = i >> 5, c4 = (i & 31) * 4;
        kv[q] = *(const float4*)(pk + s * SK + c4);
        mk = fmaxf(mk, amax4(kv[q]));
    }
    mk = warp_max(mk);
    if (lane == 0) s_red[warp] = mk;
    __syncthreads();
    const int kK = scale_exp(fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3])));
    const float scK = ldexpf(1.f, kK);
    uint2* gk = (uint2*)(prep_k + (long)rh * PREP_K_BYTES);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int i = t + q * 128, s = i >> 5, c4 = (i & 31) * 4;
        unsigned h0_, l0_, h1_, l1_;
        split_u(kv[q].x, kv[q].y, scK, h0_, l0_);
        split_u(kv[q].z, kv[q].w, scK, h1_, l1_);
        *(uint2*)(sKh + s * LDK + c4) = make_uint2(h0_, h1_);
        *(uint2*)(sKl + s * LDK + c4) = make_uint2(l0_, l1_);
        gk[(ks_off(s, c4 >> 3) + (c4 & 7)) >> 2] = make_uint2(h0_, h1_);
        gk[(SW * SK + ks_off(s, c4 >> 3) + (c4 & 7)) >> 2] = make_uint2(l0_, l1_);
    }
    if (t == 0) prep_i[rh] = kK;
    if (!any_exact) return;
    __syncthreads();
    if (warp == 0) {
        // ── Gram <k_j, k_s> 2^-2kK (both s-tiles) ──
        float G0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            unsigned ah[4], al[4];
            ldsm4(ah, sKh + (l07 + 8 * l3) * LDK + 16 * kt + 8 * l4);
            ldsm4(al, sKl + (l07 + 8 * l3) * LDK + 16 * kt + 8 * l4);
            mma3(G0[0], ah, al, ah[0], ah[2], al[0], al[2]);
            mma3(G0[1], ah, al, ah[1], ah[3], al[1], al[3]);
        }
        const float f = ldexpf(1.f, -2 * kK);
        #pragma unroll
        for (int st = 0; st < 2; ++st)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = g + 8 * (e >> 1), s = 8 * st + 2 * c + (e & 1);
                sG[j * 16 + s] = G0[st][e] * f;
            }
    }
    __syncthreads();
    if (m_w <= 0) return;
    // ── per hv (warp): Gamma~[j][s] = beta_s <k_j,k_s> exp(pre_s - pre_j), j < s; T'' = D T diag(rep) ──
    float pre = (lane < SW) ? g_cache[sidx * s_g_slot + (long)hv * SW + lane] : 0.f;
    const float beta = (lane < SW) ? ls6_beta[cidx * s_beta_slot + (long)hv * SW + lane] : 0.f;
    #pragma unroll
    for (int o = 1; o < SW; o <<= 1) {
        const float y = __shfl_up_sync(FULL, pre, o);
        if (lane >= o) pre += y;
    }
    const float gt = __shfl_sync(FULL, pre, SW - 1);
    const float rep = expf(gt - pre);
    float* gam = sGam[warp];
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        const int idx = lane + 32 * q, j = idx >> 4, s = idx & 15;
        const float bs = __shfl_sync(FULL, beta, s), ps = __shfl_sync(FULL, pre, s), pj = __shfl_sync(FULL, pre, j);
        gam[idx] = (j < s) ? sG[idx] * bs * expf(ps - pj) : 0.f;
    }
    __syncwarp();
    float x[SW];
    float mt = 0.f;
    if (lane < SW) {
        #pragma unroll
        for (int s = SW - 1; s >= 0; --s) {
            float v = (s == lane) ? 1.f : 0.f;
            #pragma unroll
            for (int j = s + 1; j < SW; ++j) v = fmaf(-gam[s * 16 + j], x[j], v);
            x[s] = v;
        }
    }
    #pragma unroll
    for (int s = 0; s < SW; ++s) {
        const float bs = __shfl_sync(FULL, beta, s), ps = __shfl_sync(FULL, pre, s);
        if (lane < SW) {
            x[s] *= bs * expf(ps) * rep;
            mt = fmaxf(mt, fabsf(x[s]));
        }
    }
    const int kT = scale_exp(warp_max(mt));
    if (lane < SW) {
        const float scT = ldexpf(1.f, kT);
        unsigned* gt2 = (unsigned*)(prep_t + ((long)r * HV + hv) * PREP_T_BYTES);
        #pragma unroll
        for (int s = 0; s < SW; s += 2) {
            unsigned h_, l_;
            split_u(x[s], x[s + 1], scT, h_, l_);
            gt2[(lane * 16 + s) >> 1] = h_;
            gt2[(256 + lane * 16 + s) >> 1] = l_;
        }
    }
    if (lane == 0) prep_it[r * HV + hv] = kT;
}

template <bool SINGLE>
__device__ __forceinline__ void gram_private_tiles(
    const float (&acc)[16][4], float* sPhiA, float* sPhiB, const float* s_red,
    float* sc, int m, int G, float scW, int kW, float ridge)
{
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int ntm = (m + 7) >> 3;
    const int row_tiles = SINGLE ? 2 : 1;
    const float fW = ldexpf(1.f, -2 * kW);
    unsigned tH[8][2], tL[8][2];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            unsigned h_, l_;
            split_u(acc[nt][2 * hh], acc[nt][2 * hh + 1], scW, h_, l_);
            tH[nt][hh] = movtrans(h_); tL[nt][hh] = movtrans(l_);
        }
    int cbuf = 0;
    #pragma unroll
    for (int rb = 0; rb < 8; rb += row_tiles) {
        #pragma unroll
        for (int cb = 0; cb < (SINGLE ? 1 : 16); cb += 2) {
            if (cb >= ntm) break;
            float* partial = cbuf ? sPhiB : sPhiA;
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int rbi = rb + (SINGLE ? pair : 0);
                const int ntg = SINGLE ? 0 : cb + pair;
                float pacc[4] = {0.f, 0.f, 0.f, 0.f};
                if (ntg < ntm) {
                    unsigned ah[4], al[4];
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh)
                        #pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            const int ai = 2 * rbi + q;
                            if (ai < 8) {
                                ah[q + 2 * hh] = tH[ai][hh];
                                al[q + 2 * hh] = tL[ai][hh];
                            } else {
                                unsigned h_, l_;
                                split_u(acc[ai][2 * hh], acc[ai][2 * hh + 1], scW, h_, l_);
                                ah[q + 2 * hh] = movtrans(h_);
                                al[q + 2 * hh] = movtrans(l_);
                            }
                        }
                    unsigned bh[2], bl[2];
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        if (ntg < 8) {
                            bh[hh] = tH[ntg][hh]; bl[hh] = tL[ntg][hh];
                        } else {
                            unsigned h_, l_;
                            split_u(acc[ntg][2 * hh], acc[ntg][2 * hh + 1], scW, h_, l_);
                            bh[hh] = movtrans(h_); bl[hh] = movtrans(l_);
                        }
                    }
                    mma3(pacc, ah, al, bh[0], bh[1], bl[0], bl[1]);
                }
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    partial[warp * 256 + pair * 128 + e * 32 + lane] = pacc[e] * fW;
            }
            __syncthreads();
            float ssb = s_red[0];
            #pragma unroll
            for (int i = 1; i < NTS / 32; ++i) ssb += s_red[i];
            const float eta = ridge * ssb / (float)SK;
            const int pair = t >> 7, e = (t >> 5) & 3;
            const int tt = 16 * (rb + (SINGLE ? pair : 0)) + (lane >> 2) + 8 * (e >> 1);
            const int gc = (SINGLE ? 0 : 8 * (cb + pair)) + 2 * (lane & 3) + (e & 1);
            if (gc < m) {
                float v = partial[t];
                #pragma unroll
                for (int b = 1; b < 8; ++b) v += partial[b * 256 + t];
                sc[tt * G + gc] = v + ((tt == gc) ? eta : 0.f);
            }
            cbuf ^= 1;
        }
    }
}

__global__ void __launch_bounds__(NTS, 2)
gdn_flush_stream_kernel(
    const __grid_constant__ CUtensorMap h0_map,
    float* __restrict__ h0, const float* __restrict__ d_cache, const float* __restrict__ k_cache,
    const float* __restrict__ g_cache, const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_map, const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    float* __restrict__ ls6_ubar, float* __restrict__ scratch,
    const unsigned char* __restrict__ prep_k,
    const unsigned char* __restrict__ prep_t, const int* __restrict__ prep_i, const int* __restrict__ prep_it,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot, long s_g_slot,
    long s_u_slot, int H, int HV, int G, float ridge)
{
    extern __shared__ __align__(128) float dsm_raw[];
    // swizzle phase zero; the state block sits last because its position measurably affects
    // the TMA/LDS overlap)
    const unsigned raw_u = smem_u32(dsm_raw);
    float* sD0 = dsm_raw;
    float* sPhiA = sD0 + 2 * RING_F;
    int* sTbl = (int*)(sPhiA + SPHI_MAX);
    __half* sK0 = (__half*)(sTbl + TBL * TE);
    const unsigned end_u = raw_u + (unsigned)(SMEM_STREAM - SV * SK * 4);
    float* sS = dsm_raw + ((((end_u + 1023u) & ~1023u) - raw_u) >> 2);
    __shared__ float s_red[NTS / 32];
    __shared__ int s_cnt[2];
    __shared__ __align__(8) unsigned long long mbS[NTS / 32], mbR[2];

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, c = lane & 3;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;
    if ((int)blockIdx.x >= n_work) return;
    const int n_it = (n_work - 1 - (int)blockIdx.x) / (int)gridDim.x + 1;
    // item table: this CTA's items blockIdx.x + it * gridDim.x; every index chain (flush_list ->
    // ls6_map, prep exponents) is resolved here, one thread per item, so the loop reads
    // scalars from smem at the point of use (no long-lived registers, no dependent loads)
    auto fill_tbl = [&](int it0, int cnt) {
        for (int i = t; i < cnt; i += NTS) {
            const int it = it0 + i;
            const int w = (int)blockIdx.x + it * (int)gridDim.x;
            if (it >= n_it) break;
            int* e = sTbl + TE * (it & (TBL - 1));
            const int r = w / HV, hv = w - r * HV, i_h = hv / hpg;
            const int sidx = flush_list[r];
            const int m = ls6_mh[hv];
            const int rh = r * H + i_h, rt = r * HV + hv;
            const int kK = prep_i[rh];
            const int kT = (m > 0) ? prep_it[rt] : 0;
            e[0] = sidx; e[1] = ls6_map ? ls6_map[sidx] : sidx; e[2] = hv; e[3] = i_h; e[4] = rh; e[5] = rt;
            e[6] = m; e[7] = (kK & 255) | ((kT & 255) << 16);
        }
    };
    float acc[16][4];
    for (int o = t; o < SPHI_MAX; o += NTS) sPhiA[o] = 0.f;
    if (t < NTS / 32) mbar_init(smem_u32(&mbS[t]), 1);
    if (t < 2) mbar_init(smem_u32(&mbR[t]), 1);
    if (t < 2) s_cnt[t] = 0;
    fill_tbl(0, TBL);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();
    const unsigned mbS_w = smem_u32(&mbS[warp]), tbl_u = smem_u32(sTbl);
    const unsigned mbR_u0 = smem_u32(&mbR[0]);
    unsigned parS = 0;
    float* sSw = sS + warp * 16 * SK;
    const unsigned sSw_u = smem_u32(sSw);
    const long rows_slot = s_h0_slot / SK, rows_h = s_h0_h / SK;
    auto row0_of = [&](unsigned e) { return (int)((long)lds_i32(e) * rows_slot + (long)lds_i32(e + 8) * rows_h) + 16 * warp; };
    if (warp == NTS / 32 - 1) issue_rings(RINGS_OF(0, tbl_u), mbR_u0, lane);
    if (warp == NTS / 32 - 2 && n_it > 1) issue_rings(RINGS_OF(1, tbl_u + 4u * TE), mbR_u0 + 8u, lane);
    issue_state(sSw_u, &h0_map, row0_of(tbl_u), mbS_w, lane);
    // ldmatrix lane addressing: rows (lane&7) + 8*hi(lane), columns 8*lo(lane)
    const int l07 = lane & 7, l3 = (lane >> 3) & 1, l4 = lane >> 4;

    for (int it = 0; it < n_it; ++it) {
        if ((it & 31) == 1 && it > 32) {                             // > TBL items: refill the
            __syncthreads();                                        // half every warp has left
            fill_tbl(it + 31, 32);
            __syncthreads();
        }
        const unsigned eu = tbl_u + 4u * TE * (it & (TBL - 1));
        const unsigned nu = tbl_u + 4u * TE * ((it + 1) & (TBL - 1));
        const bool has_next = it + 1 < n_it;
        const int rb = it & 1;                                      // ring buffer of this item
        float* sDc = sD0 + rb * RING_F;
        const __half* sKh_c = sK0 + rb * RING_H;
        const __half* sKl_c = sKh_c + KSS;
        const __half* sT2h_c = sKl_c + KSS;
        const __half* sT2l_c = sT2h_c + 16 * LDT2;
        const int m = lds_i32(eu + 24), kx = lds_i32(eu + 28);
        const int kK = (kx << 24) >> 24, kT = (kx << 8) >> 24;
        const bool exact = m > 0;
        const bool do_phi = exact && m < SK;
        constexpr int D = SK;
        mbar_wait(mbS_w, parS); parS ^= 1u;                         // own state rows landed
        load_state(acc, sSw, g, c);
        if (has_next) issue_state(sSw_u, &h0_map, row0_of(nu), mbS_w, lane);
        mbar_wait(mbR_u0 + 8u * rb, (it >> 1) & 1);                 // rings landed
        // gate scan (per warp): rep_s = exp(gt - pre_s) in lane s, tot = exp(gt)
        float pre = (lane < SW) ? sDc[SW * LDD + lane] : 0.f;
        #pragma unroll
        for (int o = 1; o < SW; o <<= 1) {
            const float y = __shfl_up_sync(FULL, pre, o);
            if (lane >= o) pre += y;
        }
        const float gt = __shfl_sync(FULL, pre, SW - 1);
        const float rep_l = expf(gt - pre), tot = expf(gt);
        float rep[2][2];
        #pragma unroll
        for (int nt2 = 0; nt2 < 2; ++nt2)
            #pragma unroll
            for (int e1 = 0; e1 < 2; ++e1) rep[nt2][e1] = __shfl_sync(FULL, rep_l, 8 * nt2 + 2 * c + e1);
        // ── per warp: P, Y, X (exact) or X = diag(rep) d (dense); fold S_W = tot S_0 + X K_r ──
        float X[2][4];
        if (exact) {
            float ms = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) ms = fmaxf(ms, fabsf(acc[nt][e]));
            const int kS = scale_exp(warp_max(ms));
            const float scS = ldexpf(1.f, kS), fP = ldexpf(1.f, -(kS + kK));
            float P[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                unsigned ah[4], al[4], bh[4], bl[4];
                c2a(acc[2 * kt], acc[2 * kt + 1], scS, ah, al);
                ldsm4(bh, sKh_c + ks_off(l07 + 8 * l4, 2 * kt + l3));
                ldsm4(bl, sKl_c + ks_off(l07 + 8 * l4, 2 * kt + l3));
                mma3(P[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
                mma3(P[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            }
            float mp = 0.f;
            #pragma unroll
            for (int st = 0; st < 2; ++st)
                #pragma unroll
                for (int e = 0; e < 4; ++e) { P[st][e] *= fP; mp = fmaxf(mp, fabsf(P[st][e])); }
            const int kP = scale_exp(warp_max(mp));
            unsigned ph_[4], pl_[4];
            c2a(P[0], P[1], ldexpf(1.f, kP), ph_, pl_);
            const float fY = ldexpf(1.f, -(kP + kT));
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                float Y[4] = {0.f, 0.f, 0.f, 0.f};
                const __half* th = sT2h_c + (8 * nt2 + g) * LDT2 + 2 * c;
                const __half* tl = sT2l_c + (8 * nt2 + g) * LDT2 + 2 * c;
                mma3(Y, ph_, pl_, *(const unsigned*)th, *(const unsigned*)(th + 8),
                     *(const unsigned*)tl, *(const unsigned*)(tl + 8));
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int s = 8 * nt2 + 2 * c + (e & 1);
                    X[nt2][e] = sDc[s * LDD + ROW_OF(e >> 1)] * rep[nt2][e & 1] - Y[e] * fY;
                }
            }
        } else {
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int s = 8 * nt2 + 2 * c + (e & 1);
                    X[nt2][e] = sDc[s * LDD + ROW_OF(e >> 1)] * rep[nt2][e & 1];
                }
        }
        {
            float mx = 0.f;
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e = 0; e < 4; ++e) mx = fmaxf(mx, fabsf(X[nt2][e]));
            const int kX = scale_exp(warp_max(mx));
            const float fO = ldexpf(1.f, -(kX + kK));
            unsigned aXh[4], aXl[4];
            c2a(X[0], X[1], ldexpf(1.f, kX), aXh, aXl);
            #pragma unroll
            for (int nt = 0; nt < 16; nt += 2) {
                unsigned bh[4], bl[4];
                ldsm4t(bh, sKh_c + ks_off(l07 + 8 * l3, nt + l4));
                ldsm4t(bl, sKl_c + ks_off(l07 + 8 * l3, nt + l4));
                float tmp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                mma3(tmp[0], aXh, aXl, bh[0], bh[1], bl[0], bl[1]);
                mma3(tmp[1], aXh, aXl, bh[2], bh[3], bl[2], bl[3]);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[nt][e] = fmaf(tot, acc[nt][e], tmp[0][e] * fO);
                    acc[nt + 1][e] = fmaf(tot, acc[nt + 1][e], tmp[1][e] * fO);
                }
            }
        }
        // the last warp done with this item's rings pulls in item it+2's into the same buffer
        // (one counter per buffer, only growing: a warp can run an item ahead of the others but
        // not two, since that needs the rings this issue provides)
        auto rings_done = [&]() {
            int old = 0;
            if (lane == 0) old = atomicAdd(&s_cnt[rb], 1);
            old = __shfl_sync(FULL, old, 0);
            if (it + 2 < n_it && ((old + 1) & (NTS / 32 - 1)) == 0)
                issue_rings(RINGS_OF(rb, tbl_u + 4u * TE * ((it + 2) & (TBL - 1))), mbR_u0 + 8u * rb, lane);
        };
        auto store = [&]() {
            store_state(acc, h0 + (long)lds_i32(eu) * s_h0_slot + (long)lds_i32(eu + 8) * s_h0_h, warp, g, c);
        };
        if (exact) {
            float* pu = ls6_ubar + (long)lds_i32(eu + 4) * s_u_slot + (long)lds_i32(eu + 8) * G * SV;
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                if (8 * nt >= m) break;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int L = 8 * nt + 2 * c + (e & 1);
                    if (L < m) pu[(long)L * SV + ROW_OF(e >> 1)] = acc[nt][e];
                }
            }
            for (int o = m * SV + t; o < G * SV; o += NTS) pu[o] = 0.f;

        }
        if (do_phi) {
            // ── S_W exponent + ||S_W||^2 (block) ──
            float mw = 0.f, ss = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float a = acc[nt][e];
                    mw = fmaxf(mw, fabsf(a)); ss = fmaf(a, a, ss);
                }
            mw = warp_max(mw); ss = warp_sum(ss);
            if (lane == 0) s_red[warp] = ss;                        // read after S8
            const int kW = scale_exp(mw); // Each warp rescales its Gram partials before reduction.
            const float scW = ldexpf(1.f, kW);
            store();
            float* sc = scratch + (long)lds_i32(eu + 20) * (D * G);
            // Two 16x8 MMA tiles per warp fit in each 2048-float chunk buffer.
            // Store in fragment order: lane-contiguous, disjoint warp slices.
            // Alternating buffers let the next tile overlap the previous drain.
            float* sPhiB = sDc;
            if (m < G)
                for (int o = t; o < D * G; o += NTS)
                    if ((o % G) >= m) sc[o] = 0.f;
            if (m <= 8) gram_private_tiles<true>(acc, sPhiA, sPhiB, s_red, sc, m, G, scW, kW, ridge);
            else gram_private_tiles<false>(acc, sPhiA, sPhiB, s_red, sc, m, G, scW, kW, ridge);
        } else {
            store();
        }
        rings_done();
    }
}

// h0 as a [rows][128] f32 tensor map (box 32 x 16, 128 B swizzle), cached per (base, extent)
typedef CUresult (*encode_fn_t)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static const CUtensorMap* h0_tensor_map(const torch::Tensor& h0) {
    static CUtensorMap map;
    static void* cached_ptr = nullptr;
    static long cached_rows = 0;
    static encode_fn_t encode = nullptr;
    void* ptr = h0.data_ptr();
    const long rows = ((long)(h0.size(0) - 1) * h0.stride(0) + (long)(h0.size(1) - 1) * h0.stride(1)) / SK + SV;
    if (ptr == cached_ptr && rows == cached_rows) return &map;
    if (encode == nullptr) {
        cudaDriverEntryPointQueryResult q;
        TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&encode, cudaEnableDefault, &q)
                    == cudaSuccess && encode != nullptr, "flush_stream: cuTensorMapEncodeTiled unavailable");
    }
    const cuuint64_t gdim[2] = {(cuuint64_t)SK, (cuuint64_t)rows};
    const cuuint64_t gstr[1] = {(cuuint64_t)SK * 4};
    const cuuint32_t box[2] = {32, 16}, estr[2] = {1, 1};
    const CUresult rc = encode(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, ptr, gdim, gstr, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rc == CUDA_SUCCESS, "flush_stream: cuTensorMapEncodeTiled failed ", (int)rc);
    cached_ptr = ptr; cached_rows = rows;
    return &map;
}


void run(int phase, int grid, torch::Tensor h0, torch::Tensor writes, torch::Tensor keys,
    torch::Tensor gates, torch::Tensor rows, int n_off, torch::Tensor mapping,
    torch::Tensor widths, torch::Tensor beta, torch::Tensor u, torch::Tensor phi,
    torch::Tensor scratch, torch::Tensor prep, torch::Tensor prep_i, double ridge)
{
    const int H = keys.size(1), HV = h0.size(1), G = u.size(2), max_rows = n_off;
    const size_t smemS = SMEM_STREAM + 1024;
    const size_t smemB128 = (G * (G + 1) + 64 * 65) * 4;
    const size_t smemA = (size_t)(SV * (SK + 4) + SK * G) * 4;
    TORCH_CHECK(HV % H == 0 && HV / H <= 4, "full flush: HV/H must be <=4");
    TORCH_CHECK(G >= 4 && G <= SK && G % 4 == 0, "full flush: G must be a multiple of 4 in [4,128]");
    TORCH_CHECK(h0.stride(0) % SK == 0, "full flush: state page stride must be divisible by 128");
    TORCH_CHECK(scratch.numel() >= (long)max_rows * HV * SK * G, "full flush: scratch size");
    TORCH_CHECK(prep.numel() >= (long)max_rows * (H * PREP_K_BYTES + HV * PREP_T_BYTES), "full flush: prep size");
    TORCH_CHECK(prep_i.numel() >= (long)max_rows * (H + HV), "full flush: exponent size");
    if (max_rows == 0) return;
    auto st = at::cuda::getCurrentCUDAStream().stream();
    static std::unordered_map<int, std::array<size_t, 3>> attributes;
    auto& attr = attributes[h0.get_device()];
    if (attr[0] < smemS) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_flush_stream_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemS));
        attr[0] = smemS;
    }
    if (G > 64 && attr[1] < smemB128) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_ls6_solve_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemB128));
        attr[1] = smemB128;
    }
    const int* fl = rows.data_ptr<int>();
    const int* lm = mapping.data_ptr<int>();
    unsigned char* pk = prep.data_ptr<unsigned char>();
    unsigned char* pt = pk + (long)max_rows * H * PREP_K_BYTES;
    int* pi = prep_i.data_ptr<int>();
    int* pit = pi + (long)max_rows * H;
    if (phase == 0 || phase == 1) {
        gdn_flush_prep_kernel<<<max_rows * H, 128, 0, st>>>(
            keys.data_ptr<float>(), gates.data_ptr<float>(), fl, fl + n_off, lm,
            widths.data_ptr<int>(), beta.data_ptr<float>(), pk, pt, pi, pit,
            keys.stride(0), gates.stride(0), beta.stride(0), H, HV);
    }
    if (phase == 0 || phase == 2) {
        const CUtensorMap* map = h0_tensor_map(h0);
        gdn_flush_stream_kernel<<<grid, NTS, smemS, st>>>(
            *map, h0.data_ptr<float>(), writes.data_ptr<float>(), keys.data_ptr<float>(), gates.data_ptr<float>(),
            fl, fl + n_off, lm, widths.data_ptr<int>(), beta.data_ptr<float>(),
            u.data_ptr<float>(), scratch.data_ptr<float>(), pk, pt, pi, pit,
            h0.stride(0), h0.stride(1), writes.stride(0), keys.stride(0), gates.stride(0),
            u.stride(0), H, HV, G, (float)ridge);
    }
    if (phase == 4) {
        if (attr[2] < smemA) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_ls6_gram_kernel<128,128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemA));
            attr[2] = smemA;
        }
        gdn_ls6_gram_kernel<128,128><<<dim3(max_rows, HV), NT, smemA, st>>>(
            h0.data_ptr<float>(), fl, fl + n_off, lm, widths.data_ptr<int>(),
            u.data_ptr<float>(), scratch.data_ptr<float>(), h0.stride(0), h0.stride(1), u.stride(0), HV, G, (float)ridge);
    }
    if (phase == 0 || phase == 3 || phase == 4) {
        // Width predicates are device-side; graph replay never reads widths on the CPU.
        #define SOLVE(M) gdn_ls6_solve_kernel<M><<<dim3(max_rows, HV), NTB, M == 128 ? smemB128 : (M * (M + 1) + M * (129 - M)) * 4, st>>>( \
            fl, fl + n_off, lm, widths.data_ptr<int>(), scratch.data_ptr<float>(), phi.data_ptr<float>(), phi.stride(0), HV, G)
        SOLVE(8);
        if (G > 8) { SOLVE(16); }
        if (G > 16) { SOLVE(32); }
        if (G > 32) { SOLVE(48); }
        if (G > 48) { SOLVE(64); }
        if (G > 64) { SOLVE(128); }
        #undef SOLVE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

"""

_CPP = r"""
#include <torch/extension.h>
void run(int phase, int grid, torch::Tensor h0, torch::Tensor writes, torch::Tensor keys,
    torch::Tensor gates, torch::Tensor rows, int n_off, torch::Tensor mapping,
    torch::Tensor widths, torch::Tensor beta, torch::Tensor u, torch::Tensor phi,
    torch::Tensor scratch, torch::Tensor prep, torch::Tensor prep_i, double ridge);
"""


@functools.lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load_inline

    build = Path(
        os.environ.get(
            "NS_GDN_FULL_FLUSH_BUILD_DIR", "/disk2/omin/.cache/gdn_full_flush"
        )
    )
    build.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="gdn_full_flush_v1",
        cpp_sources=_CPP,
        cuda_sources=_SRC,
        functions=["run"],
        build_directory=str(build),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_100f,code=sm_100f",
        ],
        verbose=False,
    )


class FlushWorkspace:
    """Reusable scratch for full-coordinate GDN flush and initial refresh.

    Allocate before graph capture. K=V=128, W=16; G is a multiple of four.
    Widths are fixed per value head in [0,G]. Zero denotes the exact dense
    delta-ring convention; positive widths use raw-WY writes and beta history.
    Active rows must be unique, positive state slots with valid compact mappings.
    The device count rows[n_off] must be in [0,n_off]. No host scalar reads occur.
    Full-width Phi is unused and deliberately left untouched, as in the new step.
    The caller advances positions; neither flush nor refresh resets rings.
    """

    def __init__(self, max_rows, h, hv, g, device, *, ridge=None, grid=None):
        if ridge is None:
            ridge = float(os.environ.get("NS_GDN_LS6_RIDGE", "0.1"))
        if max_rows < 0 or h < 1 or hv % h or not 1 <= hv // h <= 4:
            raise ValueError("invalid row/head dimensions")
        if not 4 <= g <= 128 or g % 4 or not math.isfinite(ridge) or ridge < 0:
            raise ValueError("G must be a multiple of four in [4,128]; ridge >=0")
        self.max_rows, self.h, self.hv, self.g = max_rows, h, hv, g
        self.ridge = float(ridge)
        # The initialization phase needs only the key-head dimension, not rings.
        self._refresh_keys = torch.empty((0, h, 16, 128), device=device)
        self.scratch = torch.empty(
            max_rows * hv * 128 * g, dtype=torch.float32, device=device
        )
        self.prep = torch.empty(
            max_rows * (h * 8192 + hv * 1024), dtype=torch.uint8, device=device
        )
        self.prep_i = torch.empty(max_rows * (h + hv), dtype=torch.int32, device=device)
        if self.scratch.device.type != "cuda":
            raise ValueError("flush workspace requires CUDA")
        sm = torch.cuda.get_device_properties(self.scratch.device).multi_processor_count
        self.grid = min(2 * sm, max(1, max_rows * hv)) if grid is None else int(grid)
        if self.grid <= 0:
            raise ValueError("grid must be positive")
        self._ext = _extension()

    def _run(
        self, phase, state, writes, keys, gates, rows, mapping, widths, beta, u, phi
    ):
        nx, hv, v, k = state.shape
        if phase == 4:
            writes = gates = beta = state
            keys = self._refresh_keys
        ns = u.shape[0]
        if (
            (hv, v, k) != (self.hv, 128, 128)
            or state.stride()[1:] != (16384, 128, 1)
            or state.stride(0) % 128
        ):
            raise ValueError(
                "state requires dense (HV,128,128) pages with stride divisible by 128"
            )
        tensors = [
            ("state", state, (nx, hv, 128, 128), torch.float32),
            ("writes", writes, (nx, hv, 16, 128), torch.float32),
            ("keys", keys, (nx, self.h, 16, 128), torch.float32),
            ("gates", gates, (nx, hv, 16), torch.float32),
            ("rows", rows, (self.max_rows + 1,), torch.int32),
            ("mapping", mapping, (nx,), torch.int32),
            ("widths", widths, (hv,), torch.int32),
            ("beta", beta, (ns, hv, 16), torch.float32),
            ("u", u, (ns, hv, self.g, 128), torch.float32),
            ("phi", phi, (ns, hv, self.g, 128), torch.float32),
        ]
        for name, tensor, shape, dtype in tensors:
            if phase == 4 and name in ("writes", "keys", "gates", "beta"):
                continue
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or tensor.device != self.scratch.device
            ):
                raise ValueError(
                    f"invalid {name}: expected {shape}, {dtype}, {self.scratch.device}"
                )
            paged = name in ("state", "writes", "keys", "gates")
            if (
                (not paged and not tensor.is_contiguous())
                or (paged and not tensor[0].is_contiguous())
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"{name} must have contiguous tails and aligned storage"
                )
        with torch.cuda.device(state.device):
            self._ext.run(
                phase,
                self.grid,
                state,
                writes,
                keys,
                gates,
                rows,
                self.max_rows,
                mapping,
                widths,
                beta,
                u,
                phi,
                self.scratch,
                self.prep,
                self.prep_i,
                self.ridge,
            )

    def flush(self, state, writes, keys, gates, rows, mapping, widths, beta, u, phi):
        """Fold the exact window into state, then refresh U and partial-width Phi."""
        self._run(0, state, writes, keys, gates, rows, mapping, widths, beta, u, phi)

    def refresh(self, state, writes, keys, gates, rows, mapping, widths, beta, u, phi):
        """Initialize metadata from an exact prefill/boundary state without folding."""
        self._run(4, state, writes, keys, gates, rows, mapping, widths, beta, u, phi)

    def refresh_state(self, state, rows, mapping, widths, u, phi):
        """Build boundary metadata before any decode ring has been written."""
        self._run(4, state, None, None, None, rows, mapping, widths, None, u, phi)
