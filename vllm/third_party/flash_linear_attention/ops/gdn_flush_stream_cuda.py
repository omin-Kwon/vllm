"""[nested_ssm 2026-09-04] GDN register-resident tensor-core flush (CUDA, phi mode).

Per flushed (row, hv) one 128-thread CTA holds the whole 128x128 fp32 state in
mma.sync m16n8k16 accumulator fragments (32 rows per warp) and runs the window fold
and the LS6 epilogue's parallel part without touching shared memory for the state:

    P      = S_0 K_r^T                             (A from the state fragments)
    X      = diag(rep) d - P T''                    T'' = D (I + Gamma~)^{-1} diag(rep)
    S_W    = tot S_0 + X K_r                        (accumulated in place)
    Phi   += S_W^T [S_W[:, :m] | S_W qbar | S_W kbar]   (S_W split into smem, ldmatrix)

The accumulator layout of m16n8k16 (row g = lane/4, cols 2(lane%4)+{0,1}) is also its
A-operand layout, so fp32 results feed the next product after an in-register fp16
hi/lo split.  State columns are permuted on the register side so that each thread's
global loads/stores stay float4: n-tile pair j holds physical columns
16j + 4(lane%4) + {0,1} (tile 2j) and {2,3} (tile 2j+1).  Gamma~ (K_r K_r^T, one warp)
uses the same fp16 x3 products; the 16x16 triangular inverse, Y and the Cholesky
solves stay IEEE fp32.  Every product uses a per-operand power-of-two block exponent
(max * 2^k in [2^10, 2^11)) and hi + lo carries 22 significant bits.  Dense heads
(m_h == 0) do only the fold.  The next item's state loads are issued before Phi, so
Phi's tensor work overlaps them.  Writes h0 (S_W), ls6_ubar and the epilogue scratch
consumed by gdn_ls6_solve_kernel (gdn_ls6_epilogue_cuda) -> ls6_phi / ls6_aq / ls6_ak.
"""

import os

import torch

from .gdn_ls6_epilogue_cuda import _COMMON_SRC, _SOLVE_SRC, ls6_ridge

_SRC = _COMMON_SRC + _SOLVE_SRC + r"""
#include <cuda_fp16.h>

#define SK 128
#define SV 128
#define SW 16
#define NTS 256
#define LDT 132     // staged state [V][LDT] f32
#define LDKF 132    // raw K_r ring [W][LDKF] f32; dead after the split
#define LDD 132     // d ring [W][LDD] f32 (x2); the dead one holds the Phi partial sums
#define LDK 136     // K_r split [W][LDK] f16
#define LDT2 24     // T'' split transposed [16 c][LDT2] f16
#define SEXP 10
#define NTMAX8 8    // S_W n-tiles of Phi (G <= 64)
#define SPHI_MAX (SW * LDD)
#define SMEM_STREAM ((SV * LDT + SW * LDKF + 2 * SW * LDD + 256) * 4 \
                     + (2 * SW * LDK + 2 * 16 * LDT2 + 2 * 2 * LDK) * 2)

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x = fmaxf(x, __shfl_xor_sync(FULL, x, o));
    return x;
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

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

// acc[nt][e]: rows ROW_OF(e>>1), cols 8nt + 2c + (e&1).
__device__ __forceinline__ void load_state(float (&acc)[16][4], const float* sS, int warp, int g, int c) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh)
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const float2 x = *(const float2*)(sS + ROW_OF(hh) * LDT + 8 * nt + 2 * c);
            acc[nt][2 * hh] = x.x; acc[nt][2 * hh + 1] = x.y;
        }
}
__device__ __forceinline__ void store_state(const float (&acc)[16][4], float* ph, int warp, int g, int c) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh)
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt)
            *(float2*)(ph + (long)ROW_OF(hh) * SK + 8 * nt + 2 * c) = make_float2(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
}
__device__ __forceinline__ void issue_state(float* sS, const float* ph, int t) {
    #pragma unroll
    for (int q = 0; q < SV * SK / 4 / NTS; ++q) {
        const int i = t + q * NTS, row = i >> 5, c4 = (i & 31) * 4;
        cp16(sS + row * LDT + c4, ph + (long)row * SK + c4);
    }
}
__device__ __forceinline__ void issue_ring(float* dst, int ld, const float* src, int t) {
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int i = t + q * NTS, s = i >> 5, c4 = (i & 31) * 4;
        cp16(dst + s * ld + c4, src + s * SK + c4);
    }
}

// dynamic smem: sS [V][LDT] f32 (next state, cp.async) | sKf [W][LDKF] f32 | sD[2] [W][LDD] f32
//   | sGam [16][16] f32 (anchor partials alias it after the solve) | sKh/sKl [W][LDK] f16
//   | sT2h/sT2l [16][LDT2] f16 | sQh/sQl [2][LDK] f16
// The next item's d ring and state are issued as soon as the accumulators are loaded, its K ring
// once the split is done, so the transfers overlap the whole item.  Phi is formed from the S_W
// fragments in registers (movmatrix transposes) and reduced over the warps by smem atomics.
__global__ void __launch_bounds__(NTS, 2)
gdn_flush_stream_kernel(
    float* __restrict__ h0, const float* __restrict__ d_cache, const float* __restrict__ k_cache,
    const float* __restrict__ g_cache, const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const float* __restrict__ fz_qbar, const float* __restrict__ fz_kbar,
    const int* __restrict__ ls6_map, const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    float* __restrict__ ls6_ubar, float* __restrict__ scratch,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot, long s_g_slot, long s_fzb_slot,
    long s_u_slot, long s_beta_slot, int H, int HV, int G, int R, float ridge, int dbg)
{
    extern __shared__ __align__(128) float dsm[];
    float* sS = dsm;
    float* sKf = sS + SV * LDT;
    float* sD0 = sKf + SW * LDKF;
    float* sGam = sD0 + 2 * SW * LDD;
    __half* sKh = (__half*)(sGam + 256);
    __half* sKl = sKh + SW * LDK;
    __half* sT2h = sKl + SW * LDK;
    __half* sT2l = sT2h + 16 * LDT2;
    __half* sQh = sT2l + 16 * LDT2;
    __half* sQl = sQh + 2 * LDK;
    __shared__ float sPre[SW], sRep[SW], sBeta[SW], s_red[2 * NTS / 32], s_tot;
    __shared__ int s_kT, s_kQ;

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, c = lane & 3;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;
    const bool dense_only = (dbg & 2) != 0;
    int w = blockIdx.x;
    if (w >= n_work) return;
    float acc[16][4];
    long sidx = flush_list[w / HV];
    int i_hv = w % HV;
    int buf = 0;
    issue_ring(sKf, LDKF, k_cache + sidx * s_k_slot + (long)(i_hv / hpg) * SW * SK, t);
    issue_ring(sD0, LDD, d_cache + sidx * s_d_slot + (long)i_hv * SW * SV, t);
    cp_commit();
    issue_state(sS, h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h, t);
    cp_commit();
    float g_reg = (t < SW) ? g_cache[sidx * s_g_slot + (long)i_hv * SW + t] : 0.f;
    int m_reg = (ls6_beta != nullptr && !dense_only) ? ls6_mh[i_hv] : 0;
    long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
    float beta_reg = (m_reg > 0 && t < SW) ? ls6_beta[cidx * s_beta_slot + (long)i_hv * SW + t] : 0.f;
    // ldmatrix lane addressing: rows (lane&7) + 8*hi(lane), columns 8*lo(lane)
    const int l07 = lane & 7, l3 = (lane >> 3) & 1, l4 = lane >> 4;

    for (; w < n_work; w += gridDim.x) {
        const int r = w / HV;
        i_hv = w - r * HV;
        const int i_h = i_hv / hpg;
        const int m = m_reg;
        const bool exact = m > 0;
        const bool do_phi = exact && !(dbg & 1);
        float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
        float* sD = sD0 + buf * SW * LDD;
        const int wn = w + gridDim.x;
        const bool has_next = wn < n_work;
        long sidx_n = 0; int hv_n = 0; long cidx_n = 0; int m_n = 0;
        if (has_next) {
            const int rn = wn / HV; hv_n = wn - rn * HV;
            sidx_n = flush_list[rn];
            cidx_n = ls6_map ? (long)ls6_map[sidx_n] : sidx_n;
            m_n = (ls6_beta != nullptr && !dense_only) ? ls6_mh[hv_n] : 0;
        }
        if (t < 32) {
            float pre = g_reg;
            #pragma unroll
            for (int o = 1; o < SW; o <<= 1) {
                const float y = __shfl_up_sync(FULL, pre, o);
                if (t >= o) pre += y;
            }
            const float gt = __shfl_sync(FULL, pre, SW - 1);
            if (t < SW) { sPre[t] = pre; sRep[t] = expf(gt - pre); sBeta[t] = beta_reg; }
            if (t == 0) s_tot = expf(gt);
        }
        cp_wait<0>();
        __syncthreads();                                            // S1: state + rings landed
        const float tot = s_tot;
        // ── K_r exponent + split; qbar/kbar split (warp 0) ──
        float mk = 0.f;
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int i = t + q * NTS, s = i >> 5, c4 = (i & 31) * 4;
            mk = fmaxf(mk, amax4(*(const float4*)(sKf + s * LDKF + c4)));
        }
        const int kK = scale_exp(block_max(mk, s_red, t));         // S2
        const float scK = ldexpf(1.f, kK);
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int i = t + q * NTS, s = i >> 5, c4 = (i & 31) * 4;
            const float4 x = *(const float4*)(sKf + s * LDKF + c4);
            unsigned h0_, l0_, h1_, l1_;
            split_u(x.x, x.y, scK, h0_, l0_);
            split_u(x.z, x.w, scK, h1_, l1_);
            *(uint2*)(sKh + s * LDK + c4) = make_uint2(h0_, h1_);
            *(uint2*)(sKl + s * LDK + c4) = make_uint2(l0_, l1_);
        }
        if (exact && warp == 0) {
            const float4 qv = *(const float4*)(fz_qbar + cidx * s_fzb_slot + (long)i_h * SK + 4 * lane);
            const float4 kv = *(const float4*)(fz_kbar + cidx * s_fzb_slot + (long)i_h * SK + 4 * lane);
            const int kQ = scale_exp(warp_max(fmaxf(amax4(qv), amax4(kv))));
            const float scQ = ldexpf(1.f, kQ);
            unsigned h0_, l0_, h1_, l1_;
            split_u(qv.x, qv.y, scQ, h0_, l0_); split_u(qv.z, qv.w, scQ, h1_, l1_);
            *(uint2*)(sQh + 4 * lane) = make_uint2(h0_, h1_); *(uint2*)(sQl + 4 * lane) = make_uint2(l0_, l1_);
            split_u(kv.x, kv.y, scQ, h0_, l0_); split_u(kv.z, kv.w, scQ, h1_, l1_);
            *(uint2*)(sQh + LDK + 4 * lane) = make_uint2(h0_, h1_); *(uint2*)(sQl + LDK + 4 * lane) = make_uint2(l0_, l1_);
            if (lane == 0) s_kQ = kQ;
        }
        __syncthreads();                                            // S3: splits; sKf free
        if (exact) {
            // ── Gamma~[j][s] = beta_s <k_j,k_s> exp(pre_s - pre_j), j < s (warps 0/1: s-tile) ──
            if (warp < 2) {
                const int st = warp;
                float G0[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kt = 0; kt < 8; ++kt) {
                    unsigned ah[4], al[4];
                    ldsm4(ah, sKh + (l07 + 8 * l3) * LDK + 16 * kt + 8 * l4);
                    ldsm4(al, sKl + (l07 + 8 * l3) * LDK + 16 * kt + 8 * l4);
                    if (st == 0) mma3(G0, ah, al, ah[0], ah[2], al[0], al[2]);
                    else mma3(G0, ah, al, ah[1], ah[3], al[1], al[3]);
                }
                const float f = ldexpf(1.f, -2 * kK);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j = g + 8 * (e >> 1), s = 8 * st + 2 * c + (e & 1);
                    sGam[j * 16 + s] = (j < s) ? G0[e] * f * sBeta[s] * expf(sPre[s] - sPre[j]) : 0.f;
                }
            }
            __syncthreads();                                        // S4: Gamma~
            if (warp == 0) {
                // T = (I + Gamma~)^{-1} by back substitution (column t in registers); T'' = D T diag(rep)
                float x[SW];
                float mt = 0.f;
                if (t < SW) {
                    #pragma unroll
                    for (int s = SW - 1; s >= 0; --s) {
                        float v = (s == t) ? 1.f : 0.f;
                        #pragma unroll
                        for (int j = s + 1; j < SW; ++j) v = fmaf(-sGam[s * 16 + j], x[j], v);
                        x[s] = v;
                    }
                    const float rc = sRep[t];
                    #pragma unroll
                    for (int s = 0; s < SW; ++s) {
                        x[s] *= sBeta[s] * expf(sPre[s]) * rc;
                        mt = fmaxf(mt, fabsf(x[s]));
                    }
                }
                const int kT = scale_exp(warp_max(mt));
                if (t < SW) {
                    const float scT = ldexpf(1.f, kT);
                    #pragma unroll
                    for (int s = 0; s < SW; s += 2) {
                        unsigned h_, l_;
                        split_u(x[s], x[s + 1], scT, h_, l_);
                        *(unsigned*)(sT2h + t * LDT2 + s) = h_;
                        *(unsigned*)(sT2l + t * LDT2 + s) = l_;
                    }
                }
                if (t == 0) s_kT = kT;
            }
            __syncthreads();                                        // S5: T''
        }
        load_state(acc, sS, warp, g, c);
        __syncthreads();                                            // S6a: sS, sD[buf^1] free
        if (has_next) {
            issue_ring(sKf, LDKF, k_cache + sidx_n * s_k_slot + (long)(hv_n / hpg) * SW * SK, t);
            issue_ring(sD0 + (buf ^ 1) * SW * LDD, LDD, d_cache + sidx_n * s_d_slot + (long)hv_n * SW * SV, t);
            cp_commit();
            issue_state(sS, h0 + sidx_n * s_h0_slot + (long)hv_n * s_h0_h, t);
            cp_commit();
            g_reg = (t < SW) ? g_cache[sidx_n * s_g_slot + (long)hv_n * SW + t] : 0.f;
            beta_reg = (m_n > 0 && t < SW) ? ls6_beta[cidx_n * s_beta_slot + (long)hv_n * SW + t] : 0.f;
        }
        // ── per warp: P, Y, X (exact) or X = diag(rep) d (dense); fold S_W = tot S_0 + X K_r ──
        float X[2][4];
        if (exact) {
            const int kT = s_kT;
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
                ldsm4(bh, sKh + (l07 + 8 * l4) * LDK + 16 * kt + 8 * l3);
                ldsm4(bl, sKl + (l07 + 8 * l4) * LDK + 16 * kt + 8 * l3);
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
                const __half* th = sT2h + (8 * nt2 + g) * LDT2 + 2 * c;
                const __half* tl = sT2l + (8 * nt2 + g) * LDT2 + 2 * c;
                mma3(Y, ph_, pl_, *(const unsigned*)th, *(const unsigned*)(th + 8),
                     *(const unsigned*)tl, *(const unsigned*)(tl + 8));
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int s = 8 * nt2 + 2 * c + (e & 1);
                    X[nt2][e] = sD[s * LDD + ROW_OF(e >> 1)] * sRep[s] - Y[e] * fY;
                }
            }
        } else {
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int s = 8 * nt2 + 2 * c + (e & 1);
                    X[nt2][e] = sD[s * LDD + ROW_OF(e >> 1)] * sRep[s];
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
                ldsm4t(bh, sKh + (l07 + 8 * l3) * LDK + 8 * (nt + l4));
                ldsm4t(bl, sKl + (l07 + 8 * l3) * LDK + 8 * (nt + l4));
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
        if (exact) {
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
            if (lane == 0) { s_red[warp] = mw; s_red[NTS / 32 + warp] = ss; }
            __syncthreads();                                        // S6: fold done; sD, sGam free
            float mwb = s_red[0];
            #pragma unroll
            for (int i = 1; i < NTS / 32; ++i) mwb = fmaxf(mwb, s_red[i]);
            const int kW = scale_exp(mwb);
            const float scW = ldexpf(1.f, kW), fH = ldexpf(1.f, -(kW + s_kQ)), scA = ldexpf(1.f, kW - 4);
            // ── hq/hk = S_W [qbar kbar] → transposed anchor B fragment; Ubar ──
            unsigned bAh[2], bAl[2];
            {
                float HQ[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kt = 0; kt < 8; ++kt) {
                    unsigned ah[4], al[4];
                    c2a(acc[2 * kt], acc[2 * kt + 1], scW, ah, al);
                    unsigned bh0 = 0u, bh1 = 0u, bl0 = 0u, bl1 = 0u;
                    if (g < 2) {
                        bh0 = *(const unsigned*)(sQh + g * LDK + 16 * kt + 2 * c);
                        bh1 = *(const unsigned*)(sQh + g * LDK + 16 * kt + 8 + 2 * c);
                        bl0 = *(const unsigned*)(sQl + g * LDK + 16 * kt + 2 * c);
                        bl1 = *(const unsigned*)(sQl + g * LDK + 16 * kt + 8 + 2 * c);
                    }
                    mma3(HQ, ah, al, bh0, bh1, bl0, bl1);
                }
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    unsigned h_, l_;
                    split_u(HQ[2 * hh] * fH, HQ[2 * hh + 1] * fH, scA, h_, l_);
                    if (c != 0) { h_ = 0u; l_ = 0u; }
                    bAh[hh] = movtrans(h_); bAl[hh] = movtrans(l_);
                }
                float* pu = ls6_ubar + cidx * s_u_slot + (long)i_hv * G * SV;
                #pragma unroll
                for (int nt = 0; nt < NTMAX8; ++nt)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int L = 8 * nt + 2 * c + (e & 1);
                        if (L < m) pu[(long)L * SV + ROW_OF(e >> 1)] = acc[nt][e];
                    }
                for (int o = m * SV + t; o < G * SV; o += NTS) pu[o] = 0.f;
            }
            store_state(acc, ph, warp, g, c);
            float ssb = s_red[NTS / 32];
            #pragma unroll
            for (int i = 1; i < NTS / 32; ++i) ssb += s_red[NTS / 32 + i];
            const float eta = ridge * ssb / (float)SK;
            float* sc = scratch + ((long)r * HV + i_hv) * (R * G + 2 * G);
            const bool phi_smem = R * m <= SPHI_MAX;
            float* sPhi = phi_smem ? sD : sc;                      // [R][m] partial sums
            float* sA = phi_smem ? sGam : sc + R * G;               // [2][R] anchor partials
            if (do_phi) {
                if (phi_smem) {
                    for (int o = t; o < R * m; o += NTS) sPhi[o] = 0.f;
                    if (t < 2 * R) sA[t] = 0.f;
                }
                for (int o = t; o < R * G; o += NTS)
                    if (!phi_smem || (o % G) >= m) sc[o] = 0.f;
                if (!phi_smem && t < 2 * G) sc[R * G + t] = 0.f;
            } else {
                for (int o = t; o < R * G + 2 * G; o += NTS) sc[o] = 0.f;
            }
            if (do_phi) {
                __syncthreads();                                    // S7: partial buffers zeroed
                // Phi_r += S_W^T [S_W[:, :m] | hq hk] over this warp's 16 rows: transposed
                // fragments of S_W n-tiles serve as the A (row tile rb) and B (col tile nt) operands.
                const int NTm8 = (m + 7) >> 3;
                const float fW = ldexpf(1.f, -2 * kW), fA = ldexpf(1.f, -(2 * kW - 4));
                const int ldp = phi_smem ? m : G, lda = phi_smem ? R : G;
                // transposed hi/lo fragments of n-tiles 0..7: B operand and A rows < 64; rows >= 64 (R = 128)
                // transpose on demand from acc
                unsigned tH[8][2], tL[8][2];
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        unsigned h_, l_;
                        split_u(acc[nt][2 * hh], acc[nt][2 * hh + 1], scW, h_, l_);
                        tH[nt][hh] = movtrans(h_); tL[nt][hh] = movtrans(l_);
                    }
                #pragma unroll
                for (int rb = 0; rb < 8; ++rb) {
                    if (rb >= R / 16) break;
                    const int q0 = 2 * (rb & 3);
                    unsigned ah[4], al[4];
                    if (rb < 4) {
                        ah[0] = tH[q0][0]; ah[1] = tH[q0 + 1][0]; ah[2] = tH[q0][1]; ah[3] = tH[q0 + 1][1];
                        al[0] = tL[q0][0]; al[1] = tL[q0 + 1][0]; al[2] = tL[q0][1]; al[3] = tL[q0 + 1][1];
                    } else {
                        #pragma unroll
                        for (int hh = 0; hh < 2; ++hh)
                            #pragma unroll
                            for (int q = 0; q < 2; ++q) {
                                unsigned h_, l_;
                                split_u(acc[8 + q0 + q][2 * hh], acc[8 + q0 + q][2 * hh + 1], scW, h_, l_);
                                ah[q + 2 * hh] = movtrans(h_); al[q + 2 * hh] = movtrans(l_);
                            }
                    }
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        if (4 * half >= NTm8) break;
                        float pacc[5][4];
                        #pragma unroll
                        for (int nt = 0; nt < 5; ++nt)
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) pacc[nt][e] = 0.f;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int ntg = 4 * half + nt;
                            if (ntg < NTm8) mma3(pacc[nt], ah, al, tH[ntg][0], tH[ntg][1], tL[ntg][0], tL[ntg][1]);
                        }
                        if (half == 0) mma3(pacc[4], ah, al, bAh[0], bAh[1], bAl[0], bAl[1]);
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt)
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const int tt = 16 * rb + g + 8 * (e >> 1), gc = 8 * (4 * half + nt) + 2 * c + (e & 1);
                                if (gc < m) atomicAdd(sPhi + tt * ldp + gc, pacc[nt][e] * fW);
                            }
                        if (half == 0 && c == 0) {
                            #pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
                                const int tt = 16 * rb + g + 8 * hh;
                                if (tt < m) {
                                    atomicAdd(sA + tt, pacc[4][2 * hh] * fA);
                                    atomicAdd(sA + lda + tt, pacc[4][2 * hh + 1] * fA);
                                }
                            }
                        }
                    }
                }
                __syncthreads();                                    // S8: partial sums complete
                const float* pq = fz_qbar + cidx * s_fzb_slot + (long)i_h * SK;
                const float* pk = fz_kbar + cidx * s_fzb_slot + (long)i_h * SK;
                if (phi_smem) {
                    for (int o = t; o < R * m; o += NTS) {
                        const int tt = o / m, gc = o - tt * m;
                        sc[tt * G + gc] = sPhi[o] + ((tt == gc) ? eta : 0.f);
                    }
                    if (t < G) {
                        sc[R * G + t] = (t < m) ? sA[t] + eta * pq[t] : 0.f;
                        sc[R * G + G + t] = (t < m) ? sA[R + t] + eta * pk[t] : 0.f;
                    }
                } else if (t < G) {
                    if (t < m) {
                        atomicAdd(sc + t * G + t, eta);
                        atomicAdd(sc + R * G + t, eta * pq[t]);
                        atomicAdd(sc + R * G + G + t, eta * pk[t]);
                    }
                }
            }
        } else {
            store_state(acc, ph, warp, g, c);
        }
        sidx = sidx_n; cidx = cidx_n; m_reg = m_n; buf ^= 1;
    }
    cp_wait<0>();
}

void gdn_flush_stream(int grid, torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor flush_list, int n_off, torch::Tensor fz_qbar, torch::Tensor fz_kbar,
    c10::optional<torch::Tensor> ls6_map, torch::Tensor ls6_mh, c10::optional<torch::Tensor> ls6_beta,
    torch::Tensor ls6_ubar, torch::Tensor ls6_phi, torch::Tensor ls6_aq, torch::Tensor ls6_ak, torch::Tensor scratch,
    int max_rows, int H, int HV, int K, int V, int W, int G, int R, double ridge, int dbg)
{
    TORCH_CHECK(K == SK && V == SV && W == SW, "flush_stream: K=V=128, W=16 only");
    TORCH_CHECK(G >= 4 && G <= 8 * NTMAX8 && R >= G && R <= K && R % 16 == 0, "flush_stream: G/R");
    TORCH_CHECK(scratch.numel() >= (long)max_rows * HV * (R * G + 2 * G), "flush_stream: scratch too small");
    const size_t smemS = SMEM_STREAM;
    int maxX = 0;
    for (int m = 1; m <= G; ++m) maxX = std::max(maxX, m * (((R - m) + 2) | 1));
    const size_t smemB = (size_t)(G * (G + 1) + maxX) * 4;
    auto st = at::cuda::getCurrentCUDAStream().stream();
    static size_t attrS = 0, attrB = 0;
    if (attrS < smemS) { cudaFuncSetAttribute(gdn_flush_stream_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemS); attrS = smemS; }
    if (attrB < smemB) { cudaFuncSetAttribute(gdn_ls6_solve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB); attrB = smemB; }
    if (max_rows <= 0 || grid <= 0) return;
    const int* fl = flush_list.data_ptr<int>();
    const int* lm = ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr;
    gdn_flush_stream_kernel<<<grid, NTS, smemS, st>>>(
        h0.data_ptr<float>(), d_cache.data_ptr<float>(), k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
        fl, fl + n_off, fz_qbar.data_ptr<float>(), fz_kbar.data_ptr<float>(), lm, ls6_mh.data_ptr<int>(),
        ls6_beta.has_value() ? ls6_beta->data_ptr<float>() : nullptr, ls6_ubar.data_ptr<float>(), scratch.data_ptr<float>(),
        h0.stride(0), h0.stride(1), d_cache.stride(0), k_cache.stride(0), g_cache.stride(0), fz_qbar.stride(0),
        ls6_ubar.stride(0), ls6_beta.has_value() ? ls6_beta->stride(0) : 0, H, HV, G, R, (float)ridge, dbg);
    if (dbg & 4) return;
    gdn_ls6_solve_kernel<<<dim3(max_rows, HV), NTB, smemB, st>>>(
        fl, fl + n_off, lm, fz_qbar.data_ptr<float>(), fz_kbar.data_ptr<float>(), ls6_mh.data_ptr<int>(),
        scratch.data_ptr<float>(), ls6_phi.data_ptr<float>(), ls6_aq.data_ptr<float>(), ls6_ak.data_ptr<float>(),
        fz_qbar.stride(0), ls6_phi.stride(0), ls6_aq.stride(0), H, HV, K, G, R, 99);
}
"""

_CPP = r"""
#include <torch/extension.h>
void gdn_flush_stream(int grid, torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor flush_list, int n_off, torch::Tensor fz_qbar, torch::Tensor fz_kbar,
    c10::optional<torch::Tensor> ls6_map, torch::Tensor ls6_mh, c10::optional<torch::Tensor> ls6_beta,
    torch::Tensor ls6_ubar, torch::Tensor ls6_phi, torch::Tensor ls6_aq, torch::Tensor ls6_ak, torch::Tensor scratch,
    int max_rows, int H, int HV, int K, int V, int W, int G, int R, double ridge, int dbg);
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
            name="ns_gdn_flush_stream_v5c", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["gdn_flush_stream"], build_directory=bd,
            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_100f,code=sm_100f"], verbose=False)
    return _EXT


def gdn_flush_stream_supported(K, V, W, G, R):
    return K == 128 and V == 128 and W == 16 and 4 <= G <= 64 and G <= R <= K and R % 16 == 0


def _sm_count():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def gdn_flush_stream(h0, d_cache, k_cache, g_cache, flush_list, n_off, fz_qbar, fz_kbar, ls6_map, ls6_mh,
                     ls6_beta, ls6_ubar, ls6_phi, ls6_aq, ls6_ak, H, HV, K, V, W, G, R, ridge=None, grid=None):
    """Register-resident flush + LS6 epilogue for state rows flush_list[:flush_list[n_off]].

    Args:
        h0: (NX,HV,V,K) fp32 state (S_0 in, S_W out).
        d_cache/k_cache/g_cache: window rings (NX,HV,W,V), (NX,H,W,K), (NX,HV,W).
        flush_list: int32 (>= n_off+1,) state indices, count at [n_off].
        fz_qbar/fz_kbar: (NS,H,K) window means; ls6_map: optional state row -> compact slot.
        ls6_mh: (HV,) latch widths; ls6_beta: optional (NS,HV,W) betas (None = all dense).
        ls6_ubar (NS,HV,G,V), ls6_phi (NS,HV,G,R), ls6_aq/ak (NS,HV,G): outputs.
    """
    for nm, tt in (("h0", h0), ("d_cache", d_cache), ("k_cache", k_cache), ("g_cache", g_cache),
                   ("fz_qbar", fz_qbar), ("fz_kbar", fz_kbar), ("ls6_ubar", ls6_ubar),
                   ("ls6_phi", ls6_phi), ("ls6_aq", ls6_aq), ("ls6_ak", ls6_ak)):
        if tt.dtype != torch.float32 or not tt.is_contiguous():
            raise TypeError(f"gdn_flush_stream: {nm} must be contiguous fp32 ({tt.dtype})")
    if flush_list.dtype != torch.int32 or ls6_mh.dtype != torch.int32:
        raise TypeError("gdn_flush_stream: flush_list/ls6_mh must be int32")
    max_rows = int(n_off)
    scratch = torch.empty(max_rows * HV * (R * G + 2 * G), dtype=torch.float32, device=h0.device)
    if grid is None:
        grid = int(os.environ.get("NS_GDN_FLUSH_GRID", 0)) or min(2 * _sm_count(), max(1, max_rows * HV))
    _ext().gdn_flush_stream(int(grid), h0, d_cache, k_cache, g_cache, flush_list, int(n_off), fz_qbar, fz_kbar,
                            ls6_map, ls6_mh, ls6_beta, ls6_ubar, ls6_phi, ls6_aq, ls6_ak, scratch,
                            max_rows, H, HV, K, V, W, G, R, ls6_ridge() if ridge is None else float(ridge),
                            int(os.environ.get("NS_GDN_FLUSH_STREAM_DBG", "0")))
