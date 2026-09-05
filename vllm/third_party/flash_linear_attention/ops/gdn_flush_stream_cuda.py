"""[nested_ssm 2026-09-04] GDN register-resident tensor-core flush (CUDA, phi mode).

Per flushed (row, hv) one 256-thread CTA holds the whole 128x128 fp32 state in
mma.sync m16n8k16 accumulator fragments (16 rows per warp, logical column order) and
runs the window fold and the LS6 epilogue's parallel part without staging the state
through shared memory more than once (cp.async prefetch of the next item's state):

    P      = S_0 K_r^T                             (A from the state fragments)
    X      = diag(rep) d - P T''                    T'' = D (I + Gamma~)^{-1} diag(rep)
    S_W    = tot S_0 + X K_r                        (accumulated in place)
    Phi   += S_W^T [S_W[:, :m] | S_W qbar | S_W kbar]   (movmatrix transposes of the
                                                         S_W fragments, per-warp smem slices)

The accumulator layout of m16n8k16 (row g = lane/4, cols 2(lane%4)+{0,1}) is also its
A-operand layout, so fp32 results feed the next product after an in-register fp16
hi/lo split; K_r^T fragments come from ldmatrix.trans on the row-major split.
Gamma~ (K_r K_r^T, one warp) uses the same fp16 x3 products; the 16x16 triangular
inverse, Y and the Cholesky solves stay IEEE fp32.  Every product uses a per-operand
power-of-two block exponent (max * 2^k in [2^10, 2^11)) and hi + lo carries 22
significant bits.  Block exponents are per warp (kK excepted: every warp scans the
whole K ring), so the exact path has no block-wide reductions; Gamma~ and the
triangular solve run on warp 0 while the other warps load their state fragments.
Dense heads (m_h == 0) do only the fold.  Phi_r has D = max(R, G) rows (R'
coordinates, then the extra latch coordinates when G > R); each warp writes its
16*RBC-row partial into its own smem slice when all eight fit (m <= 16), else fewer
slices are shared with red.shared (shared f32 atomics are CAS loops on sm_100), and
each chunk is drained to the epilogue scratch after one barrier, alternating between
a dedicated buffer and the dead K_r split.  Writes h0 (S_W), ls6_ubar and the
epilogue scratch consumed by gdn_ls6_solve_kernel (gdn_ls6_epilogue_cuda) ->
ls6_phi / ls6_aq / ls6_ak.

NS_GDN_FLUSH_STREAM_DBG bits: 1 skip Phi, 2 all heads dense, 4 skip the solve
kernel, 8 prefer larger row chunks over private slices (atomic Phi reduction).
"""

import os

import torch

from .gdn_ls6_epilogue_cuda import _COMMON_SRC, _SOLVE_SRC, ls6_ridge

_SRC = _COMMON_SRC + _SOLVE_SRC + r"""
#include <cuda.h>
#include <cuda_fp16.h>

#define SK 128
#define SV 128
#define SW 16
#define NTS 256
#define LDD 132     // d ring [W][LDD] f32 (x2 buffers); the consumed one holds Phi partial sums
#define LDK 128     // qbar/kbar split [2][LDK] f16
#define KSS (SW * SK) // K_r split [W][SK] f16 (hi | lo), 16 B chunks XOR-swizzled by (row&7)
#define LDT2 16     // T'' split transposed [16 c][LDT2] f16
#define SEXP 10
#define SPHI_MAX 2048         // Phi partial-sum chunk (floats): sPhiA, then the store staging
#define TBL 64      // item table entries (a CTA's work list, index chains resolved up front)
#define TE 8        // entry: sidx, cidx, hv, i_h, rh, rt, m, kK | kQ << 8 | kT << 16
#define RING_H (2 * KSS + 2 * 16 * LDT2 + 2 * 2 * LDK)   // K split | T'' split | qbar/kbar split (f16)
#define RING_F (SW * LDD + SW)                             // d rows | gates (f32)
#define SMEM_STREAM ((SV * SK + 2 * RING_F + SPHI_MAX + 256 + TBL * TE) * 4 + 2 * RING_H * 2)
// prep records (bytes): K split [2][W][SK] f16 per (row, h); qbar/kbar split [2][2][SK] f16 per
// (row, h); T'' split [2][16 c][16 s] f16 per (row, hv); ints {kK, kQ} per (row, h), kT per (row, hv)
#define PREP_K_BYTES (2 * SW * SK * 2)
#define PREP_Q_BYTES (2 * 2 * SK * 2)
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
__device__ __forceinline__ void sts_f32(unsigned a, float v) {
    asm volatile("st.shared.f32 [%0], %1;" :: "r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void red_shared_f32(unsigned a, float v) {
    asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(a), "f"(v) : "memory");
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
// one warp: d ring rows, K split, T'' and qbar/kbar records of an item -> ring buffers
__device__ __forceinline__ void issue_rings(float* sD, __half* sK, __half* sT2, __half* sQ, float* sG,
                                            const float* d_src, const float* g_src,
                                            const unsigned char* k_src, const unsigned char* t_src,
                                            const unsigned char* q_src, bool tq, unsigned mb, int lane) {
    if (lane == 0)
        mbar_expect_tx(mb, SW * SV * 4 + SW * 4 + PREP_K_BYTES + (tq ? PREP_T_BYTES + PREP_Q_BYTES : 0u));
    __syncwarp();
    if (lane < SW) bulk_g2s(smem_u32(sD + lane * LDD), d_src + lane * SV, SV * 4, mb);
    else if (lane == 16) bulk_g2s(smem_u32(sK), k_src, PREP_K_BYTES, mb);
    else if (lane == 17) bulk_g2s(smem_u32(sG), g_src, SW * 4, mb);
    else if (tq && lane == 18) bulk_g2s(smem_u32(sT2), t_src, PREP_T_BYTES, mb);
    else if (tq && lane == 19) bulk_g2s(smem_u32(sQ), q_src, PREP_Q_BYTES, mb);
}
// rings of the item in table entry eu (item table: see fill_tbl)
#define RINGS_OF(b, eu) sD0 + (b) * RING_F, sK0 + (b) * RING_H, sK0 + (b) * RING_H + 2 * KSS, \
    sK0 + (b) * RING_H + 2 * KSS + 2 * 16 * LDT2, sD0 + (b) * RING_F + SW * LDD, \
    d_cache + (long)lds_i32(eu) * s_d_slot + (long)lds_i32(eu + 8) * SW * SV, \
    g_cache + (long)lds_i32(eu) * s_g_slot + (long)lds_i32(eu + 8) * SW, \
    prep_k + (long)lds_i32(eu + 16) * PREP_K_BYTES, prep_t + (long)lds_i32(eu + 20) * PREP_T_BYTES, \
    prep_q + (long)lds_i32(eu + 16) * PREP_Q_BYTES, lds_i32(eu + 24) > 0

// Prep kernel, one 128-thread block per (row, h): K_r split (hi/lo f16, block exponent kK),
// qbar/kbar split (kQ) and, per hv of the head, Gamma~ -> T = (I + Gamma~)^{-1} -> T'' split (kT),
// written as records the stream kernel pulls in with bulk copies.
__global__ void __launch_bounds__(128)
gdn_flush_prep_kernel(
    const float* __restrict__ k_cache, const float* __restrict__ g_cache, const int* __restrict__ flush_list,
    const int* __restrict__ n_ptr, const float* __restrict__ fz_qbar, const float* __restrict__ fz_kbar,
    const int* __restrict__ ls6_map, const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    unsigned char* __restrict__ prep_k, unsigned char* __restrict__ prep_q, unsigned char* __restrict__ prep_t,
    int* __restrict__ prep_i, int* __restrict__ prep_it,
    long s_k_slot, long s_g_slot, long s_fzb_slot, long s_beta_slot, int H, int HV, int dbg)
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
    const int m_w = (warp < hpg && ls6_beta != nullptr && !(dbg & 2)) ? ls6_mh[hv] : 0;
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
    if (t == 0) prep_i[2 * rh] = kK;
    if (!any_exact) return;
    if (warp == 0) {
        const float4 qv = *(const float4*)(fz_qbar + cidx * s_fzb_slot + (long)h * SK + 4 * lane);
        const float4 kv2 = *(const float4*)(fz_kbar + cidx * s_fzb_slot + (long)h * SK + 4 * lane);
        const int kQ = scale_exp(warp_max(fmaxf(amax4(qv), amax4(kv2))));
        const float scQ = ldexpf(1.f, kQ);
        uint2* gq = (uint2*)(prep_q + (long)rh * PREP_Q_BYTES);
        unsigned h0_, l0_, h1_, l1_;
        split_u(qv.x, qv.y, scQ, h0_, l0_); split_u(qv.z, qv.w, scQ, h1_, l1_);
        gq[lane] = make_uint2(h0_, h1_); gq[64 + lane] = make_uint2(l0_, l1_);
        split_u(kv2.x, kv2.y, scQ, h0_, l0_); split_u(kv2.z, kv2.w, scQ, h1_, l1_);
        gq[32 + lane] = make_uint2(h0_, h1_); gq[96 + lane] = make_uint2(l0_, l1_);
        if (lane == 0) prep_i[2 * rh + 1] = kQ;
    }
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

template <bool BIG>
__global__ void __launch_bounds__(NTS, 2)
gdn_flush_stream_kernel(
    const __grid_constant__ CUtensorMap h0_map,
    float* __restrict__ h0, const float* __restrict__ d_cache, const float* __restrict__ k_cache,
    const float* __restrict__ g_cache, const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const float* __restrict__ fz_qbar, const float* __restrict__ fz_kbar,
    const int* __restrict__ ls6_map, const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    float* __restrict__ ls6_ubar, float* __restrict__ scratch,
    const unsigned char* __restrict__ prep_k, const unsigned char* __restrict__ prep_q,
    const unsigned char* __restrict__ prep_t, const int* __restrict__ prep_i, const int* __restrict__ prep_it,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot, long s_g_slot, long s_fzb_slot,
    long s_u_slot, long s_beta_slot, int H, int HV, int G, int R, float ridge, int dbg)
{
    extern __shared__ __align__(128) float dsm_raw[];
    // dynamic smem: 2 x {sD [W][LDD], sG [W]} f32 | sPhiA | sAnc | table |
    // 2 x {K split, T'' split, qbar/kbar split} f16 | sS [8 warps][4][16][32] f32 (1 KB aligned:
    // swizzle phase zero; the state block sits last because its position measurably affects
    // the TMA/LDS overlap)
    const unsigned raw_u = smem_u32(dsm_raw);
    float* sD0 = dsm_raw;
    float* sPhiA = sD0 + 2 * RING_F;
    float* sAnc = sPhiA + SPHI_MAX;
    int* sTbl = (int*)(sAnc + 256);
    __half* sK0 = (__half*)(sTbl + TBL * TE);
    const unsigned end_u = raw_u + (unsigned)(SMEM_STREAM - SV * SK * 4);
    float* sS = dsm_raw + ((((end_u + 1023u) & ~1023u) - raw_u) >> 2);
    __shared__ float s_red[NTS / 32];
    __shared__ int s_cnt[2];
    __shared__ __align__(8) unsigned long long mbS[NTS / 32], mbR[2];

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, c = lane & 3;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;
    const bool dense_only = (dbg & 2) != 0;
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
            const int m = (ls6_beta != nullptr && !dense_only) ? ls6_mh[hv] : 0;
            const int rh = r * H + i_h, rt = r * HV + hv;
            const int kK = prep_i[2 * rh], kQ = (m > 0) ? prep_i[2 * rh + 1] : 0;
            const int kT = (m > 0) ? prep_it[rt] : 0;
            e[0] = sidx; e[1] = ls6_map ? ls6_map[sidx] : sidx; e[2] = hv; e[3] = i_h; e[4] = rh; e[5] = rt;
            e[6] = m; e[7] = (kK & 255) | ((kQ & 255) << 8) | ((kT & 255) << 16);
        }
    };
    float acc[16][4];
    for (int o = t; o < SPHI_MAX + 256; o += NTS) sPhiA[o] = 0.f;
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
        const __half* sQh_c = sT2l_c + 16 * LDT2;
        const __half* sQl_c = sQh_c + 2 * LDK;
        const int m = lds_i32(eu + 24), kx = lds_i32(eu + 28);
        const int kK = (kx << 24) >> 24, kQ = (kx << 16) >> 24, kT = (kx << 8) >> 24;
        const bool exact = m > 0;
        const bool do_phi = exact && !(dbg & 1);
        const int D = max(R, G), DT = (D + 15) >> 4;
        const int mc = (m + 7) & ~7;                                // Phi slice row stride
        int RBC = 1, NB = 8;
        if (exact) {
            RBC = (dbg & 8) ? min(DT, SPHI_MAX / (16 * mc)) : max(1, min(DT, SPHI_MAX / (128 * mc)));
            NB = min(8, SPHI_MAX / (16 * RBC * mc));
        }
        const int cw = 16 * RBC * mc;
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
            const int kW = scale_exp(mw);                           // per warp: Phi partials are
                                                                    // rescaled before the atomics
            const float scW = ldexpf(1.f, kW), fH = ldexpf(1.f, -(kW + kQ)), scA = ldexpf(1.f, kW - 4);
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
                        bh0 = *(const unsigned*)(sQh_c + g * LDK + 16 * kt + 2 * c);
                        bh1 = *(const unsigned*)(sQh_c + g * LDK + 16 * kt + 8 + 2 * c);
                        bl0 = *(const unsigned*)(sQl_c + g * LDK + 16 * kt + 2 * c);
                        bl1 = *(const unsigned*)(sQl_c + g * LDK + 16 * kt + 8 + 2 * c);
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
            store();
            float* sc = scratch + (long)lds_i32(eu + 20) * (D * G + 2 * G);
            // Phi partial sums: chunks of 16*RBC rows x mc columns (m rounded up to the n-tile, so
            // the fragment stores need no predicate), NB slices (warp % NB).  NB == 8: private
            // slices, plain stores, nothing zeroed.  NB < 8: shared slices via red.shared (f32
            // atomicAdd is a CAS loop), zeroed here (the buffers may hold a previous item's
            // partials) behind one barrier, and re-zeroed by the drain.  Chunk buffers alternate
            // between sPhiA and this item's dead d ring.  Anchors [2][G] in sAnc (zero on entry,
            // re-zeroed by the last drain).
            float* sA = sAnc;
            float* sPhiB = sDc;
            if (NB < 8) {
                #pragma unroll
                for (int i = 0; i < SPHI_MAX / NTS; i += 4)
                    *(float4*)(sPhiA + 4 * t + NTS * i) = make_float4(0.f, 0.f, 0.f, 0.f);
                *(float4*)(sDc + (lane >> 1) * LDD + 16 * warp + 8 * (lane & 1)) = make_float4(0.f, 0.f, 0.f, 0.f);
                *(float4*)(sDc + (lane >> 1) * LDD + 16 * warp + 8 * (lane & 1) + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
                if (warp == NTS / 32 - 1 && lane < SW) *(float4*)(sDc + lane * LDD + SV) = make_float4(0.f, 0.f, 0.f, 0.f);
                __syncthreads();
            }
            if (do_phi) {
                if (m < G)
                    for (int o = t; o < D * G; o += NTS)
                        if ((o % G) >= m) sc[o] = 0.f;
            } else {
                for (int o = t; o < D * G + 2 * G; o += NTS) sc[o] = 0.f;
            }
            if (do_phi) {
                // Phi_r += S_W^T [S_W[:, :m] | hq hk] over this warp's 16 rows: transposed
                // fragments of S_W n-tiles serve as the A (row tile rb) and B (col tile nt) operands.
                // Row tiles are accumulated RBC at a time in smem, then drained to scratch.
                const int NTm8 = mc >> 3, m8 = 8 * mc;
                const unsigned sAu = smem_u32(sA);
                const float fW = ldexpf(1.f, -2 * kW), fA = ldexpf(1.f, -(2 * kW - 4));
                const float* pq = fz_qbar + (long)lds_i32(eu + 4) * s_fzb_slot + (long)lds_i32(eu + 12) * SK;
                const float* pk = fz_kbar + (long)lds_i32(eu + 4) * s_fzb_slot + (long)lds_i32(eu + 12) * SK;
                // transposed hi/lo fragments of n-tiles 0..7: B operand and A rows < 64; tiles >= 8
                // (m > 64 / rows >= 64) transpose on demand from acc
                unsigned tH[8][2], tL[8][2];
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        unsigned h_, l_;
                        split_u(acc[nt][2 * hh], acc[nt][2 * hh + 1], scW, h_, l_);
                        tH[nt][hh] = movtrans(h_); tL[nt][hh] = movtrans(l_);
                    }
                // BIG = rows >= 64 or m > 64: acc[8..15] stay live for the on-demand transposes;
                // otherwise acc is dead here and the loop runs within the register budget.
                constexpr int RBN = BIG ? 8 : 4, HFN = BIG ? 4 : 2;
                int crow = 0, cbuf = 0;
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    if (rb >= DT) break;
                    const unsigned prow = smem_u32(cbuf ? sPhiB : sPhiA)
                                          + 4u * ((warp % NB) * cw + (crow + g) * mc);
                    const int q0 = 2 * (rb & 3);
                    unsigned ah[4], al[4];
                    if (!BIG || rb < 4) {
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
                    for (int half = 0; half < HFN; ++half) {
                        if (4 * half >= NTm8) break;
                        float pacc[5][4];
                        #pragma unroll
                        for (int nt = 0; nt < 5; ++nt)
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) pacc[nt][e] = 0.f;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int ntg = 4 * half + nt;
                            if (ntg >= NTm8) continue;
                            unsigned bh0, bh1, bl0, bl1;
                            if (!BIG || ntg < 8) {
                                bh0 = tH[ntg][0]; bh1 = tH[ntg][1]; bl0 = tL[ntg][0]; bl1 = tL[ntg][1];
                            } else {
                                unsigned h_, l_;
                                split_u(acc[ntg][0], acc[ntg][1], scW, h_, l_);
                                bh0 = movtrans(h_); bl0 = movtrans(l_);
                                split_u(acc[ntg][2], acc[ntg][3], scW, h_, l_);
                                bh1 = movtrans(h_); bl1 = movtrans(l_);
                            }
                            mma3(pacc[nt], ah, al, bh0, bh1, bl0, bl1);
                        }
                        if (half == 0) mma3(pacc[4], ah, al, bAh[0], bAh[1], bAl[0], bAl[1]);
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            if (4 * half + nt >= NTm8) break;
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const int gc = 8 * (4 * half + nt) + 2 * c + (e & 1);
                                const unsigned p = prow + 4u * ((e >> 1) * m8 + gc);
                                if (NB == 8) sts_f32(p, pacc[nt][e] * fW);
                                else red_shared_f32(p, pacc[nt][e] * fW);
                            }
                        }
                        if (half == 0 && c == 0) {
                            #pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
                                const int tt = 16 * rb + g + 8 * hh;
                                if (tt < m) {
                                    red_shared_f32(sAu + 4u * tt, pacc[4][2 * hh] * fA);
                                    red_shared_f32(sAu + 4u * (G + tt), pacc[4][2 * hh + 1] * fA);
                                }
                            }
                        }
                    }
                    crow += 16;
                    if (crow == 16 * RBC || rb + 1 == DT) {
                        __syncthreads();                            // S8: chunk partial sums complete
                        const int rbase = 16 * (rb + 1) - crow;
                        float ssb = s_red[0];                       // every warp's ||S_W||^2 is in
                        #pragma unroll
                        for (int i = 1; i < NTS / 32; ++i) ssb += s_red[i];
                        const float eta = ridge * ssb / (float)SK;
                        float* sPhi0 = cbuf ? sPhiB : sPhiA;
                        for (int o = t; o < crow * mc; o += NTS) {
                            const int lr = o / mc, gc = o - lr * mc, tt = rbase + lr;
                            if (gc >= m) continue;
                            float v = sPhi0[o];
                            if (NB == 8) {
                                #pragma unroll
                                for (int b = 1; b < 8; ++b) v += sPhi0[b * cw + o];
                            } else {
                                sPhi0[o] = 0.f;
                                for (int b = 1; b < NB; ++b) { v += sPhi0[b * cw + o]; sPhi0[b * cw + o] = 0.f; }
                            }
                            if (tt < D) sc[tt * G + gc] = v + ((tt == gc) ? eta : 0.f);
                        }
                        if (rb + 1 == DT && t < G) {
                            sc[D * G + t] = (t < m) ? sA[t] + eta * pq[t] : 0.f;
                            sc[D * G + G + t] = (t < m) ? sA[G + t] + eta * pk[t] : 0.f;
                            sA[t] = 0.f; sA[G + t] = 0.f;
                        }
                        crow = 0; cbuf ^= 1;
                    }
                }
            }
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

void gdn_flush_stream(int grid, torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor flush_list, int n_off, torch::Tensor fz_qbar, torch::Tensor fz_kbar,
    c10::optional<torch::Tensor> ls6_map, torch::Tensor ls6_mh, c10::optional<torch::Tensor> ls6_beta,
    torch::Tensor ls6_ubar, torch::Tensor ls6_phi, torch::Tensor ls6_aq, torch::Tensor ls6_ak, torch::Tensor scratch,
    torch::Tensor prep, torch::Tensor prep_i, int max_rows, int H, int HV, int K, int V, int W, int G, int R,
    double ridge, int dbg)
{
    TORCH_CHECK(K == SK && V == SV && W == SW, "flush_stream: K=V=128, W=16 only");
    TORCH_CHECK(G >= 4 && G <= 128 && R >= 16 && R <= K && R % 16 == 0, "flush_stream: G/R");
    const int D = std::max(R, G);
    TORCH_CHECK(scratch.numel() >= (long)max_rows * HV * (D * G + 2 * G), "flush_stream: scratch too small");
    TORCH_CHECK(HV % H == 0 && HV / H <= 4, "flush_stream: HV/H <= 4");
    TORCH_CHECK(prep.numel() >= (long)max_rows * (H * (PREP_K_BYTES + PREP_Q_BYTES) + HV * PREP_T_BYTES)
                && prep_i.numel() >= (long)max_rows * (2 * H + HV), "flush_stream: prep too small");
    const size_t smemS = SMEM_STREAM + 1024;
    int maxX = 0;
    for (int m = 1; m <= G; ++m) maxX = std::max(maxX, m * ((std::max(R - m, 0) + 2) | 1));
    const size_t smemB = (size_t)(G * (G + 1) + maxX) * 4;
    auto st = at::cuda::getCurrentCUDAStream().stream();
    const bool big = D > 64 || G > 64;
    auto kern = big ? gdn_flush_stream_kernel<true> : gdn_flush_stream_kernel<false>;
    static size_t attrS[2] = {0, 0}, attrB = 0;
    if (attrS[big] < smemS) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemS); attrS[big] = smemS; }
    if (attrB < smemB) { cudaFuncSetAttribute(gdn_ls6_solve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB); attrB = smemB; }
    if (max_rows <= 0 || grid <= 0) return;
    TORCH_CHECK(h0.stride(2) == SK && h0.stride(3) == 1 && h0.stride(1) % SK == 0 && h0.stride(0) % SK == 0,
                "flush_stream: h0 rows must be dense");
    const CUtensorMap* map = h0_tensor_map(h0);
    const int* fl = flush_list.data_ptr<int>();
    const int* lm = ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr;
    const float* beta = ls6_beta.has_value() ? ls6_beta->data_ptr<float>() : nullptr;
    const long s_beta = ls6_beta.has_value() ? ls6_beta->stride(0) : 0;
    unsigned char* pk = prep.data_ptr<unsigned char>();
    unsigned char* pq = pk + (long)max_rows * H * PREP_K_BYTES;
    unsigned char* pt = pq + (long)max_rows * H * PREP_Q_BYTES;
    int* pi = prep_i.data_ptr<int>();
    int* pit = pi + 2L * max_rows * H;
    gdn_flush_prep_kernel<<<max_rows * H, 128, 0, st>>>(
        k_cache.data_ptr<float>(), g_cache.data_ptr<float>(), fl, fl + n_off, fz_qbar.data_ptr<float>(),
        fz_kbar.data_ptr<float>(), lm, ls6_mh.data_ptr<int>(), beta, pk, pq, pt, pi, pit,
        k_cache.stride(0), g_cache.stride(0), fz_qbar.stride(0), s_beta, H, HV, dbg);
    kern<<<grid, NTS, smemS, st>>>(
        *map, h0.data_ptr<float>(), d_cache.data_ptr<float>(), k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
        fl, fl + n_off, fz_qbar.data_ptr<float>(), fz_kbar.data_ptr<float>(), lm, ls6_mh.data_ptr<int>(),
        beta, ls6_ubar.data_ptr<float>(), scratch.data_ptr<float>(), pk, pq, pt, pi, pit,
        h0.stride(0), h0.stride(1), d_cache.stride(0), k_cache.stride(0), g_cache.stride(0), fz_qbar.stride(0),
        ls6_ubar.stride(0), s_beta, H, HV, G, R, (float)ridge, dbg);
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
    torch::Tensor prep, torch::Tensor prep_i, int max_rows, int H, int HV, int K, int V, int W, int G, int R,
    double ridge, int dbg);
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
            name="ns_gdn_flush_stream_v12", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["gdn_flush_stream"], build_directory=bd,
            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_100f,code=sm_100f"], verbose=False)
    return _EXT


def gdn_flush_stream_supported(K, V, W, G, R):
    return K == 128 and V == 128 and W == 16 and 4 <= G <= 128 and 16 <= R <= K and R % 16 == 0


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
    # h0 may be a layer view of vLLM's packed cache page: dense (HV,V,K) tail,
    # page stride on dim 0 (16-byte aligned for the bulk copies).
    h0_ok = (
        h0.dtype == torch.float32
        and h0.dim() == 4
        and tuple(h0.shape[1:]) == (HV, V, K)
        and h0.stride(3) == 1
        and h0.stride(2) == K
        and h0.stride(1) == V * K
        and h0.stride(0) % 4 == 0
    )
    if not h0_ok:
        raise TypeError(
            "gdn_flush_stream: h0 must be fp32 with a dense (HV,V,K) tail; "
            f"shape={tuple(h0.shape)} stride={h0.stride()} dtype={h0.dtype}"
        )
    for nm, tt in (("d_cache", d_cache), ("k_cache", k_cache), ("g_cache", g_cache),
                   ("fz_qbar", fz_qbar), ("fz_kbar", fz_kbar), ("ls6_ubar", ls6_ubar),
                   ("ls6_phi", ls6_phi), ("ls6_aq", ls6_aq), ("ls6_ak", ls6_ak)):
        if tt.dtype != torch.float32 or not tt.is_contiguous():
            raise TypeError(f"gdn_flush_stream: {nm} must be contiguous fp32 ({tt.dtype})")
    if flush_list.dtype != torch.int32 or ls6_mh.dtype != torch.int32:
        raise TypeError("gdn_flush_stream: flush_list/ls6_mh must be int32")
    max_rows = int(n_off)
    D = max(R, G)
    scratch = torch.empty(max_rows * HV * (D * G + 2 * G), dtype=torch.float32, device=h0.device)
    prep = torch.empty(max_rows * (H * (8192 + 1024) + HV * 1024), dtype=torch.uint8, device=h0.device)
    prep_i = torch.empty(max_rows * (2 * H + HV), dtype=torch.int32, device=h0.device)
    if grid is None:
        grid = int(os.environ.get("NS_GDN_FLUSH_GRID", 0)) or min(2 * _sm_count(), max(1, max_rows * HV))
    _ext().gdn_flush_stream(int(grid), h0, d_cache, k_cache, g_cache, flush_list, int(n_off), fz_qbar, fz_kbar,
                            ls6_map, ls6_mh, ls6_beta, ls6_ubar, ls6_phi, ls6_aq, ls6_ak, scratch, prep, prep_i,
                            max_rows, H, HV, K, V, W, G, R, ls6_ridge() if ridge is None else float(ridge),
                            int(os.environ.get("NS_GDN_FLUSH_STREAM_DBG", "0")))
