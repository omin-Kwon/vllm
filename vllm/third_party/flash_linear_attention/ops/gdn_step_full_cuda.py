# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-K CUDA specialization of the validated GDN Phi step.

Keeps warp-per-value-head scheduling and asynchronous double-buffered streaming.
No truncation, anchors or freeze mode. Initially consumes full Phi storage so
its arithmetic can be compared directly with the unchanged CUDA reference.
Does not perform a flush or advance ring positions. K=V=128, W=16, HV/H=3.
"""

import os
from functools import lru_cache
from pathlib import Path

import torch

_SRC = r"""

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>

#define WMAX 16
#define FULL 0xffffffffu

template <typename T> __device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }

__device__ __forceinline__ float ldx(const void* p, int code, long i) {
    if (code == 0) return ((const float*)p)[i];
    if (code == 1) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    return __half2float(((const __half*)p)[i]);
}
__device__ __forceinline__ float round_to(float x, int code) {
    if (code == 1) return __bfloat162float(__float2bfloat16(x));
    if (code == 2) return __half2float(__float2half(x));
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}

template <int CK> __device__ __forceinline__ void ld_chunk(const float* p, float* o, bool evict) {
    if (CK == 4) {
        float4 x = evict ? __ldcs((const float4*)p) : *((const float4*)p);
        o[0] = x.x; o[1] = x.y; o[2] = x.z; o[3] = x.w;
    } else if (CK == 2) {
        float2 x = evict ? __ldcs((const float2*)p) : *((const float2*)p);
        o[0] = x.x; o[1] = x.y;
    } else {
        o[0] = evict ? __ldcs(p) : *p;
    }
}

template <typename T> __device__ __forceinline__ float4 ld4(const T* p);
template <> __device__ __forceinline__ float4 ld4<float>(const float* p) { return *((const float4*)p); }
template <> __device__ __forceinline__ float4 ld4<__nv_bfloat16>(const __nv_bfloat16* p) {
    const uint2 u = *((const uint2*)p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}
template <typename T> __device__ __forceinline__ void st4(T* p, float4 v);
template <> __device__ __forceinline__ void st4<float>(float* p, float4 v) { *((float4*)p) = v; }
template <> __device__ __forceinline__ void st4<__nv_bfloat16>(__nv_bfloat16* p, float4 v) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    uint2 u; u.x = *reinterpret_cast<unsigned*>(&a); u.y = *reinterpret_cast<unsigned*>(&b);
    *((uint2*)p) = u;
}

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, unsigned long long* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned phase) {
    unsigned done;
    do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem_u32(bar)), "r"(phase) : "memory");
    } while (!done);
}

template <int N> __device__ __forceinline__ float xposeN(float (&v)[N], int lane) {
    #pragma unroll
    for (int off = N / 2; off >= 1; off >>= 1) {
        const bool up = (lane & off) != 0;
        #pragma unroll
        for (int i = 0; i < off; ++i) {
            const float send = up ? v[i] : v[i + off];
            const float keep = up ? v[i + off] : v[i];
            v[i] = keep + __shfl_xor_sync(FULL, send, off);
        }
    }
    float r = v[0];
    #pragma unroll
    for (int o = N; o < 32; o <<= 1) r += __shfl_xor_sync(FULL, r, o);
    return r;
}
template <int N> __device__ __forceinline__ float xposeNL(float (&v)[N], int lane, int L) {
    #pragma unroll
    for (int off = N / 2; off >= 1; off >>= 1) {
        const bool up = (lane & off) != 0;
        #pragma unroll
        for (int i = 0; i < off; ++i) {
            const float send = up ? v[i] : v[i + off];
            const float keep = up ? v[i + off] : v[i];
            v[i] = keep + __shfl_xor_sync(FULL, send, off);
        }
    }
    float r = v[0];
    for (int o = N; o < L; o <<= 1) r += __shfl_xor_sync(FULL, r, o);
    return r;
}

template <int N> struct Pow2 { static constexpr int v = (N <= 1) ? 1 : (N <= 2) ? 2 : (N <= 4) ? 4 : (N <= 8) ? 8 : (N <= 16) ? 16 : 32; };

template <int K, int V, int HPG> struct Smem {
    static constexpr int NR = 8;
    static constexpr int BYTES = HPG * 2 * NR * V * 4;
};

template <int K, int V, int GT, int HPG, typename TIO>
__global__ void __maxnreg__(80)
gdn_step_kernel(
    const TIO* __restrict__ mixed_qkv, const void* __restrict__ a, const void* __restrict__ b, int ab_code,
    const void* __restrict__ A_log, const void* __restrict__ dt_bias, int p_code,
    TIO* __restrict__ out, const float* __restrict__ h0,
    float* __restrict__ d_cache, float* __restrict__ k_cache, float* __restrict__ g_cache,
    const int* __restrict__ ssm_state_indices, const int* __restrict__ write_pos, float scale,
    const float* __restrict__ ls6_ubar, const float* __restrict__ ls6_phi,
    const int* __restrict__ ls6_mh,
    float* __restrict__ ls6_fs, const int* __restrict__ ls6_map, float* __restrict__ beta_ring,
    long s_mix, long s_a, long s_b, long s_h0_slot, long s_h0_h, long s_ind,
    long s_d_slot, long s_k_slot, long s_g_slot,
    long s_u_slot, long s_phi_slot, long s_fs_slot,
    int H, int HV, int W, int G)
{
    using SM = Smem<K, V, HPG>;
    constexpr int NW = HPG;
    constexpr int NR = SM::NR;
    constexpr int CK = K / 32;
    constexpr int VL = V / 32;
    constexpr int NSW = (WMAX - 1 + NW - 1) / NW;
    constexpr int NKV = Pow2<2 * NSW>::v;
    constexpr int NG = (GT + 31) / 32;
    constexpr int CGW = K / 4;
    constexpr int NQ = CGW / 4;
    static_assert(K % 32 == 0 && K <= 128 && V == 128, "K in {32,64,128}, V == 128");
    static_assert(GT % 4 == 0 && HPG >= 2 && HPG <= 4, "GT/HPG");
    static_assert(VL == 4 && (NQ & (NQ - 1)) == 0 && V % NR == 0 && 32 % NR == 0, "layout");
    constexpr bool LS6 = true, L2N = true;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int i_n = blockIdx.x, i_h = blockIdx.y, hv0 = i_h * HPG, i_hv = hv0 + warp;
    extern __shared__ __align__(128) unsigned char dsm[];
    float* sS = (float*)dsm;
    __shared__ __align__(16) float sQK[2][K];
    __shared__ __align__(16) float sKq[WMAX], sKk[WMAX];
    __shared__ __align__(16) float2 sC[NW][WMAX];
    __shared__ __align__(16) float sA[NW][2 * GT];
    __shared__ __align__(8) unsigned long long mbar_st[NW][2];

    const long sidx = ssm_state_indices[(long)i_n * s_ind];
    const long cidx = (ls6_map && sidx > 0) ? (long)ls6_map[sidx] : sidx;
    const int wp = write_pos[i_n];
    const int mh_raw = ls6_mh[i_hv];
    float qc[CK], kc[CK];
    {
        const TIO* pq = mixed_qkv + (long)i_n * s_mix + (i_h * K + lane * CK);
        const TIO* pk = pq + H * K;
        if (CK == 4) {
            const float4 q4 = ld4<TIO>(pq), k4 = ld4<TIO>(pk);
            qc[0] = q4.x; qc[1] = q4.y; qc[2] = q4.z; qc[3] = q4.w;
            kc[0] = k4.x; kc[1] = k4.y; kc[2] = k4.z; kc[3] = k4.w;
        } else {
            #pragma unroll
            for (int c = 0; c < CK; ++c) { qc[c] = to_f<TIO>(pq[c]); kc[c] = to_f<TIO>(pk[c]); }
        }
    }
    const float a_val = ldx(a, ab_code, (long)i_n * s_a + i_hv);
    const float b_val = ldx(b, ab_code, (long)i_n * s_b + i_hv);
    const float Al = ldx(A_log, p_code, i_hv);
    const float dtb = ldx(dt_bias, p_code, i_hv);
    TIO* p_o = out + (long)(i_n * HV + i_hv) * V + lane * VL;
    if (sidx <= 0) { st4<TIO>(p_o, make_float4(0.f, 0.f, 0.f, 0.f)); return; }
    const int mh = mh_raw;
    const bool latch = mh > 0;
    const int nf = latch ? 0 : K;

    const int s0k = warp * NSW;
    const int nrk = (wp - s0k < NSW) ? (wp - s0k) : NSW;
    const int n_s = (nf > 0) ? V / NR : 0;
    const int n_d = (wp + NR - 1) / NR;
    const int RPC = (NR * V) / K;
    const int FPC = (NR * V) / G;
    const int n_p = latch && mh < K ? (mh + RPC - 1) / RPC : 0;
    const int n_f = latch ? (wp + FPC - 1) / FPC : 0;
    const int n_u = (mh + NR - 1) / NR;
    const int c_s = 0, c_d = n_s, c_p = c_d + n_d, c_f = c_p + n_p, c_u = c_f + n_f;
    const int NC = c_u + n_u;
    const float* pst = h0 + sidx * s_h0_slot + i_hv * (int)s_h0_h;
    const float* bd = d_cache + sidx * s_d_slot;
    const float* pdr = bd + i_hv * (W * V);
    float* bk = k_cache + sidx * s_k_slot;
    float* bg = g_cache + sidx * s_g_slot;
    const float* bu = LS6 ? ls6_ubar + cidx * s_u_slot : nullptr;
    const float* bp = LS6 ? ls6_phi + cidx * s_phi_slot + i_hv * (G * K) : nullptr;
    float* pf = LS6 ? ls6_fs + cidx * s_fs_slot + i_hv * (W * G) : nullptr;
    auto issue = [&](int c, int buf) {
        float* dst = sS + ((warp * 2 + buf) * NR) * V;
        unsigned long long* bar = &mbar_st[warp][buf];
        if (c < c_d) {
            if (lane == 0) { mbar_expect_tx(bar, NR * K * 4); bulk_g2s(dst, pst + c * (NR * K), NR * K * 4, bar); }
        } else if (c < c_p) {
            const int r0 = (c - c_d) * NR;
            const int nr = (wp - r0 < NR) ? (wp - r0) : NR;
            if (lane == 0) { mbar_expect_tx(bar, nr * V * 4); bulk_g2s(dst, pdr + r0 * V, nr * V * 4, bar); }
        } else if (c < c_f) {
            const int r0 = (c - c_p) * RPC;
            const int nr = (mh - r0 < RPC) ? (mh - r0) : RPC;
            if (lane == 0) { mbar_expect_tx(bar, nr * K * 4); bulk_g2s(dst, bp + r0 * K, nr * K * 4, bar); }
        } else if (c < c_u) {
            const int s0 = (c - c_f) * FPC;
            const int ns = (wp - s0 < FPC) ? (wp - s0) : FPC;
            if (lane == 0) { mbar_expect_tx(bar, ns * G * 4); bulk_g2s(dst, pf + s0 * G, ns * G * 4, bar); }
        } else {
            const int r0 = (c - c_u) * NR;
            const int nr = (mh - r0 < NR) ? (mh - r0) : NR;
            if (lane == 0) { mbar_expect_tx(bar, nr * V * 4); bulk_g2s(dst, bu + (i_hv * G + r0) * V, nr * V * 4, bar); }
        }
    };
    if (lane == 0) { mbar_init(&mbar_st[warp][0], 1); mbar_init(&mbar_st[warp][1], 1); }
    __syncwarp();
    if (NC > 0) issue(0, 0);
    if (NC > 1) issue(1, 1);

    const float gs = (lane < wp) ? bg[i_hv * W + lane] : 0.f;
    float kr[NSW][CK];
    #pragma unroll
    for (int i = 0; i < NSW; ++i) {
        if (i < nrk) ld_chunk<CK>(bk + (i_h * W + s0k + i) * K + lane * CK, kr[i], false);
        else {
            #pragma unroll
            for (int c = 0; c < CK; ++c) kr[i][c] = 0.f;
        }
    }
    const float xg = a_val + dtb;
    const float sp = (xg <= 20.f) ? logf(1.f + expf(xg)) : xg;
    const float g_val = -expf(Al) * sp;
    const float alpha = expf(g_val);
    const float beta = round_to(1.f / (1.f + expf(-b_val)), ab_code);
    if (beta_ring && lane == 0) beta_ring[(cidx * HV + i_hv) * W + wp] = beta;
    float nrm[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int c = 0; c < CK; ++c) { nrm[0] += qc[c] * qc[c]; nrm[1] += kc[c] * kc[c]; nrm[2] += qc[c] * kc[c]; }
    const float nv = xposeN<4>(nrm, lane);
    const float sq = __shfl_sync(FULL, nv, 0), sk = __shfl_sync(FULL, nv, 1), qk = __shfl_sync(FULL, nv, 2);
    const float q_sc = (L2N ? (1.f / sqrtf(sq + 1e-6f)) : 1.f) * scale;
    const float k_rn = L2N ? (1.f / sqrtf(sk + 1e-6f)) : 1.f;
    #pragma unroll
    for (int c = 0; c < CK; ++c) { qc[c] *= q_sc; kc[c] *= k_rn; }
    const float cur_kq = qk * q_sc * k_rn;
    #pragma unroll
    for (int c = 0; c < CK; ++c) { sQK[0][lane * CK + c] = qc[c]; sQK[1][lane * CK + c] = kc[c]; }
    if (warp == 0) {
        float* pkw = bk + (i_h * W + wp) * K + lane * CK;
        #pragma unroll
        for (int c = 0; c < CK; ++c) pkw[c] = kc[c];
    }
    __syncwarp();

    float pre = gs;
    #pragma unroll
    for (int o = 1; o < WMAX; o <<= 1) { const float n = __shfl_up_sync(FULL, pre, o); if (lane >= o) pre += n; }
    const float gtot = __shfl_sync(FULL, pre, WMAX - 1);
    const float rep = (lane < wp) ? expf(gtot - pre) : 0.f;
    const float tot = expf(gtot);

    {
        float kv[NKV];
        #pragma unroll
        for (int i = 0; i < NKV; ++i) kv[i] = 0.f;
        #pragma unroll
        for (int i = 0; i < NSW; ++i) {
            float pq_ = 0.f, pk_ = 0.f;
            #pragma unroll
            for (int c = 0; c < CK; ++c) { pq_ += kr[i][c] * qc[c]; pk_ += kr[i][c] * kc[c]; }
            kv[i] = pq_; kv[NSW + i] = pk_;
        }
        const float kval = xposeN<NKV>(kv, lane);
        const int e = lane & (NKV - 1);
        const int i = (e < NSW) ? e : e - NSW;
        if (lane < NKV && i < nrk) {
            if (e < NSW) sKq[s0k + i] = kval; else if (e < 2 * NSW) sKk[s0k + i] = kval;
        }
    }

    const float4 vv = ld4<TIO>(mixed_qkv + (long)i_n * s_mix + (2 * H * K + i_hv * V + lane * VL));

    __syncthreads();
    if (lane < WMAX) sC[warp][lane] = make_float2(sKq[lane] * rep, sKk[lane] * rep);
    __syncwarp();

    float4 hq = make_float4(0.f, 0.f, 0.f, 0.f), hk = hq;
    if (n_s > 0) {
        const int r = lane & (NR - 1), cg = lane / NR;
        float oq[4] = {0.f, 0.f, 0.f, 0.f}, ok_[4] = {0.f, 0.f, 0.f, 0.f};
        auto consume = [&](int buf, float& hq_r, float& hk_r) {
            const float* src = sS + ((warp * 2 + buf) * NR) * V + r * K + cg * CGW;
            const float* q0 = sQK[0] + cg * CGW;
            const float* k0 = sQK[1] + cg * CGW;
            float aq[4] = {0.f, 0.f, 0.f, 0.f}, ak[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int i = 0; i < NQ; ++i) {
                const int cc = ((i + r) & (NQ - 1)) * 4;
                float4 x = *((const float4*)(src + cc));
                const float4 q4 = *((const float4*)(q0 + cc)), k4 = *((const float4*)(k0 + cc));
                aq[i & 3] += x.x * q4.x + x.y * q4.y + x.z * q4.z + x.w * q4.w;
                ak[i & 3] += x.x * k4.x + x.y * k4.y + x.z * k4.z + x.w * k4.w;
            }
            hq_r = (aq[0] + aq[1]) + (aq[2] + aq[3]);
            hk_r = (ak[0] + ak[1]) + (ak[2] + ak[3]);
            hq_r += __shfl_xor_sync(FULL, hq_r, 8);  hk_r += __shfl_xor_sync(FULL, hk_r, 8);
            hq_r += __shfl_xor_sync(FULL, hq_r, 16); hk_r += __shfl_xor_sync(FULL, hk_r, 16);
        };
        #pragma unroll 1
        for (int c = c_s; c < c_d; ++c) {
            const int buf = c & 1;
            mbar_wait(&mbar_st[warp][buf], (c >> 1) & 1);
            float hq_r, hk_r;
            consume(buf, hq_r, hk_r);
            __syncwarp();
            if (c + 2 < NC) issue(c + 2, buf);
            const int cs = c - c_s;
            const bool mine = (lane >> 1) == cs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int src_l = 4 * (lane & 1) + j;
                const float vq = __shfl_sync(FULL, hq_r, src_l), vk = __shfl_sync(FULL, hk_r, src_l);
                if (mine) { oq[j] = vq; ok_[j] = vk; }
            }
        }
        hq = make_float4(oq[0], oq[1], oq[2], oq[3]);
        hk = make_float4(ok_[0], ok_[1], ok_[2], ok_[3]);
    }
    float4 s_q = make_float4(0.f, 0.f, 0.f, 0.f), s_k = s_q;
    #pragma unroll 1
    for (int c = c_d; c < c_p; ++c) {
        const int buf = c & 1;
        mbar_wait(&mbar_st[warp][buf], (c >> 1) & 1);
        const float* pb = sS + ((warp * 2 + buf) * NR) * V + lane * VL;
        const int r0 = (c - c_d) * NR;
        const float2* cp = sC[warp] + r0;
        #pragma unroll
        for (int s = 0; s < NR; ++s) {
            if (r0 + s >= wp) break;
            const float2 cc = cp[s];
            const float4 d = *((const float4*)(pb + s * V));
            s_q.x += d.x * cc.x; s_q.y += d.y * cc.x; s_q.z += d.z * cc.x; s_q.w += d.w * cc.x;
            s_k.x += d.x * cc.y; s_k.y += d.y * cc.y; s_k.z += d.z * cc.y; s_k.w += d.w * cc.y;
        }
        __syncwarp();
        if (c + 2 < NC) issue(c + 2, buf);
    }
    if (LS6) {
        // Full embedded basis: Phi is identity; no coefficient matrix read.
        if (mh == K) {
            #pragma unroll
            for (int i=0; i<NG; ++i) {
                const int g=lane+32*i;
                if (g<K) { sA[warp][g]=sQK[0][g]; sA[warp][GT+g]=sQK[1][g]; }
            }
            __syncwarp();
        }
        const int L = K >> 2, lgL = __ffs(L) - 1, RP = 32 >> lgL;
        const int col = (lane & (L - 1)) * 4;
        const float4 q4 = *((const float4*)(sQK[0] + col)), k4 = *((const float4*)(sQK[1] + col));
        const int myrow = (lane & 7) * RP + (lane >> lgL);
        const bool primary = (lane & (L - 1)) < 8;
        #pragma unroll 1
        for (int c = c_p; c < c_f; ++c) {
            const int buf = c & 1;
            mbar_wait(&mbar_st[warp][buf], (c >> 1) & 1);
            const float* pb = sS + ((warp * 2 + buf) * NR) * V + lane * VL;
            float pq_[8], pk_[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 p = *((const float4*)(pb + j * 128));
                pq_[j] = (p.x * q4.x + p.y * q4.y) + (p.z * q4.z + p.w * q4.w);
                pk_[j] = (p.x * k4.x + p.y * k4.y) + (p.z * k4.z + p.w * k4.w);
            }
            __syncwarp();
            if (c + 2 < NC) issue(c + 2, buf);
            const float xq = xposeNL<8>(pq_, lane, L), xk = xposeNL<8>(pk_, lane, L);
            const int row = (c - c_p) * RPC + myrow;
            if (primary && row < mh) { sA[warp][row] = xq; sA[warp][GT + row] = xk; }
        }
        float erq[NG], erk[NG];
        #pragma unroll
        for (int i = 0; i < NG; ++i) { erq[i] = 0.f; erk[i] = 0.f; }
        #pragma unroll 1
        for (int c = c_f; c < c_u; ++c) {
            const int buf = c & 1;
            mbar_wait(&mbar_st[warp][buf], (c >> 1) & 1);
            const float* pb = sS + ((warp * 2 + buf) * NR) * V;
            const int s0 = (c - c_f) * FPC;
            const int ns = (wp - s0 < FPC) ? (wp - s0) : FPC;
            for (int s = 0; s < ns; ++s) {
                const float kq = sKq[s0 + s], kk = sKk[s0 + s];
                #pragma unroll
                for (int i = 0; i < NG; ++i) {
                    const int g = lane + 32 * i;
                    const float fv = (g < mh) ? pb[s * G + g] : 0.f;
                    erq[i] = fmaf(fv, kq, erq[i]);
                    erk[i] = fmaf(fv, kk, erk[i]);
                }
            }
            __syncwarp();
            if (c + 2 < NC) issue(c + 2, buf);
        }
        __syncwarp();
        if (latch) {
            #pragma unroll
            for (int i = 0; i < NG; ++i) {
                const int g = lane + 32 * i;
                if (g < mh) {
                    const float baseq = sA[warp][g];
                    const float basek = sA[warp][GT + g];
                    const float fcur = beta * (basek - erk[i]);
                    sA[warp][g] = baseq - erq[i] - fcur * cur_kq;
                    pf[wp * G + g] = fcur;
                }
            }
        }
        __syncwarp();
        #pragma unroll 1
        for (int c = c_u; c < NC; ++c) {
            const int buf = c & 1;
            mbar_wait(&mbar_st[warp][buf], (c >> 1) & 1);
            const float* pb = sS + ((warp * 2 + buf) * NR) * V + lane * VL;
            const int r0 = (c - c_u) * NR;
            const float* px = sA[warp] + r0;
            #pragma unroll
            for (int g = 0; g < NR; ++g) {
                if (r0 + g >= mh) break;
                const float xq = px[g];
                const float4 u = *((const float4*)(pb + g * V));
                hq.x += u.x * xq; hq.y += u.y * xq; hq.z += u.z * xq; hq.w += u.w * xq;
            }
            __syncwarp();
            if (c + 2 < NC) issue(c + 2, buf);
        }
    }
    {
        float o[4], dc[4];
        const float hqv[4] = {hq.x, hq.y, hq.z, hq.w}, hkv[4] = {hk.x, hk.y, hk.z, hk.w};
        const float sqv[4] = {s_q.x, s_q.y, s_q.z, s_q.w}, skv[4] = {s_k.x, s_k.y, s_k.z, s_k.w};
        const float vvv[4] = {vv.x, vv.y, vv.z, vv.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float stq = alpha * (hqv[c] * tot + sqv[c]);
            const float stk = alpha * (hkv[c] * tot + skv[c]);
            dc[c] = latch ? beta * (vvv[c] - alpha * skv[c])
                          : beta * (vvv[c] - stk);
            o[c] = stq + dc[c] * cur_kq;
        }
        st4<TIO>(p_o, make_float4(o[0], o[1], o[2], o[3]));
        *((float4*)(bd + (i_hv * W + wp) * V + lane * VL)) = make_float4(dc[0], dc[1], dc[2], dc[3]);
        if (lane == 0) bg[i_hv * W + wp] = g_val;
    }
}

static int dt_code(const torch::Tensor& x) {
    if (x.scalar_type() == torch::kFloat32) return 0;
    if (x.scalar_type() == torch::kBFloat16) return 1;
    if (x.scalar_type() == torch::kFloat16) return 2;
    TORCH_CHECK(false, "gdn_step: unsupported dtype ", x.scalar_type());
    return -1;
}


template<int GT, typename TIO>
void launch(torch::Tensor mixed, torch::Tensor a, torch::Tensor b, torch::Tensor alog, torch::Tensor bias,
    torch::Tensor out, torch::Tensor state, torch::Tensor writes, torch::Tensor keys, torch::Tensor gates,
    torch::Tensor index, torch::Tensor pos, torch::Tensor u, torch::Tensor phi,
    torch::Tensor widths, torch::Tensor factors, torch::Tensor mapping, c10::optional<torch::Tensor> beta_ring, double scale) {
    constexpr int K=128, V=128, HPG=3;
    const int B=mixed.size(0), H=keys.size(1), HV=state.size(1), W=16, G=u.size(2);
    constexpr int SMEM=Smem<K,V,HPG>::BYTES;
    static bool initialized=false;
    if (!initialized) {
        cudaFuncSetAttribute(gdn_step_kernel<K,V,GT,HPG,TIO>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        initialized=true;
    }
    gdn_step_kernel<K,V,GT,HPG,TIO><<<dim3(B,H),96,SMEM,at::cuda::getCurrentCUDAStream()>>>(
        (const TIO*)mixed.data_ptr(), a.data_ptr(), b.data_ptr(), dt_code(a),
        alog.data_ptr(), bias.data_ptr(), dt_code(alog), (TIO*)out.data_ptr(), state.data_ptr<float>(),
        writes.data_ptr<float>(), keys.data_ptr<float>(), gates.data_ptr<float>(),
        index.data_ptr<int>(), pos.data_ptr<int>(), (float)scale,
        u.data_ptr<float>(),phi.data_ptr<float>(),widths.data_ptr<int>(),factors.data_ptr<float>(),mapping.data_ptr<int>(), beta_ring.has_value() ? beta_ring->data_ptr<float>() : nullptr,
        mixed.stride(0),a.stride(0),b.stride(0),state.stride(0),state.stride(1),index.stride(0),
        writes.stride(0),keys.stride(0),gates.stride(0),u.stride(0),phi.stride(0),factors.stride(0),H,HV,W,G);
}

void step(torch::Tensor mixed, torch::Tensor a, torch::Tensor b, torch::Tensor alog, torch::Tensor bias,
    torch::Tensor out, torch::Tensor state, torch::Tensor writes, torch::Tensor keys, torch::Tensor gates,
    torch::Tensor index, torch::Tensor pos, torch::Tensor u, torch::Tensor phi,
    torch::Tensor widths, torch::Tensor factors, torch::Tensor mapping, c10::optional<torch::Tensor> beta_ring, double scale) {
    const int G=u.size(2);
#define CASE(GT) if (G<=GT) { if (mixed.scalar_type()==torch::kFloat32) launch<GT,float>(mixed,a,b,alog,bias,out,state,writes,keys,gates,index,pos,u,phi,widths,factors,mapping,beta_ring,scale); else launch<GT,__nv_bfloat16>(mixed,a,b,alog,bias,out,state,writes,keys,gates,index,pos,u,phi,widths,factors,mapping,beta_ring,scale); return; }
    CASE(8) CASE(16) CASE(32) CASE(48) CASE(64) CASE(80) CASE(128)
#undef CASE
    TORCH_CHECK(false,"Unsupported G");
}
"""

_CPP = r"""
#include <torch/extension.h>
void step(torch::Tensor mixed, torch::Tensor a, torch::Tensor b, torch::Tensor alog, torch::Tensor bias,
    torch::Tensor out, torch::Tensor state, torch::Tensor writes, torch::Tensor keys, torch::Tensor gates,
    torch::Tensor index, torch::Tensor pos, torch::Tensor u, torch::Tensor phi,
    torch::Tensor widths, torch::Tensor factors, torch::Tensor mapping, c10::optional<torch::Tensor> beta_ring, double scale);
"""


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load_inline

    build = Path(
        os.environ.get("NS_GDN_FULL_BUILD_DIR", "/disk2/omin/.cache/gdn_full_cuda")
    )
    build.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="gdn_full_cuda_v1",
        cpp_sources=_CPP,
        cuda_sources=_SRC,
        functions=["step"],
        build_directory=str(build),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "--ptxas-options=-v"],
        verbose=False,
    )


def step(
    mixed,
    a,
    b,
    a_log,
    bias,
    out,
    state,
    writes,
    keys,
    gates,
    indices,
    positions,
    u,
    phi,
    widths,
    factors,
    mapping,
    scale,
    *,
    beta_ring=None,
):
    """Run the full-dimension CUDA step with the reference ring contract.

    Widths must be in [0,G]; zero selects an exact dense head. Phi must be the
    embedded full-coordinate reconstruction map (identity prefix). Active
    physical slots must be unique and mapped to initialized compact metadata,
    positions in [0,15], and nonpositive indices denote padding. This operation
    only appends a step; the caller owns exact boundary refresh and positions.
    Optional beta_ring records the already rounded beta in compact-slot order,
    avoiding a separate sigmoid/scatter kernel for the exact flush.
    """
    batch = mixed.shape[0]
    ns, hv, v, k = state.shape
    h, w = keys.shape[1:3]
    compact, _, g, _ = u.shape
    if (k, v, w, hv // h) != (128, 128, 16, 3) or hv % h:
        raise ValueError("Expected K=V=128, W=16, HV/H=3")
    if not 4 <= g <= 128 or g % 4:
        raise ValueError("G must be a multiple of four in [4,128]")
    expected = (
        (mixed, (batch, 2 * h * k + hv * v)),
        (a, (batch, hv)),
        (b, (batch, hv)),
        (a_log, (hv,)),
        (bias, (hv,)),
        (out, (batch, hv, v)),
        (writes, (ns, hv, w, v)),
        (keys, (ns, h, w, k)),
        (gates, (ns, hv, w)),
        (u, (compact, hv, g, v)),
        (phi, (compact, hv, g, k)),
        (factors, (compact, hv, w, g)),
        (widths, (hv,)),
        (indices, (batch,)),
        (positions, (batch,)),
        (mapping, (ns,)),
    )
    for index, (tensor, shape) in enumerate(expected):
        if tuple(tensor.shape) != shape or tensor.device != mixed.device:
            raise ValueError(
                f"Invalid input {index}: shape={tuple(tensor.shape)}, "
                f"device={tensor.device}; expected {shape} on {mixed.device}"
            )
    if not mixed.is_cuda or state.device != mixed.device:
        raise ValueError("CUDA tensors on one device are required")
    if mixed.dtype not in (torch.float32, torch.bfloat16) or out.dtype != mixed.dtype:
        raise ValueError("Matching FP32/BF16 I/O required")
    if a.dtype != b.dtype or a_log.dtype != bias.dtype:
        raise ValueError("Gate input dtype pairs must match")
    for tensor in (a, b, a_log, bias):
        if (
            tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or tensor.stride(-1) != 1
        ):
            raise ValueError("Invalid gate layout/dtype")
    for tensor in (state, writes, keys, gates, u, phi, factors):
        if (
            tensor.dtype != torch.float32
            or not tensor[0].is_contiguous()
            or tensor.data_ptr() % 16
            or tensor.stride(0) % 4
        ):
            raise ValueError("Aligned FP32 contiguous cache tails required")
    for tensor in (indices, positions, widths, mapping):
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError("Contiguous int32 indices required")
    if (
        mixed.stride(-1) != 1
        or mixed.stride(0) % 4
        or not out.is_contiguous()
        or mixed.data_ptr() % 16
        or out.data_ptr() % 16
    ):
        raise ValueError("Invalid I/O layout")
    if beta_ring is not None:
        if (
            beta_ring.shape != (compact, hv, w)
            or beta_ring.dtype != torch.float32
            or beta_ring.device != mixed.device
            or not beta_ring.is_contiguous()
        ):
            raise ValueError("beta_ring must be contiguous FP32 (compact,HV,W)")
    with torch.accelerator.device_index(mixed.device.index):
        _extension().step(
            mixed,
            a,
            b,
            a_log,
            bias,
            out,
            state,
            writes,
            keys,
            gates,
            indices,
            positions,
            u,
            phi,
            widths,
            factors,
            mapping,
            beta_ring,
            float(scale),
        )
