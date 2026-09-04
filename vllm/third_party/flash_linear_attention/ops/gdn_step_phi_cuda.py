"""[nested_ssm 2026-09-04] GDN ReplaySSM v2 **스텝 커널(CUDA, Φ̄ 경로, v1)**.

v8e(gdn_step_cuda.py)의 행 스트림·κ·d 접기·출력은 그대로 두고, 래치 계수만 논문의 LS 로 바꾼다:
  x_q = Φ̄_rᵀ q̂[:R] + a_q,  x_k = Φ̄_rᵀ k̂[:R] + a_k     (Φ̄ = Φ_η M_η⁻¹, hv 마다; a_* 는 꼬리 앵커)
  F_t = β [x_k − Σ_{s<t} F_s κk_s],  x = x_q − Σ_{s≤t} F_s κq_s,  hq += Ū x.
Z/Z̄/zk/q̄k̄ 가 사라져 **워프(=hv)가 자기 완결**이다: sX·sync B 가 없고, 블록 동기화는 sync A(κ) 하나.
  * 행 스트림(워프마다): hot state → d 링 → Φ̄_r 청크(행 = R 폭, 청크당 1024/R 행) → f_s 청크(행 = G 폭)
    → Ū 청크. 전부 cp.async.bulk 더블버퍼. f_s 를 스트림에 태워 `pf[s*G]` 의존 전역 로드 사슬을 없앤다.
  * Φ̄ 청크 소비: 레인 l 이 float4 열 4(l mod L) (L = R/4 레인이 행 하나) → 8 패스 부분합 → 전치 리덕션
    (그룹 폭 L) → sA[warp]. R ∈ {32, 64, 128}.
  * 앵커 a_q/a_k 는 레지스터, 합산은 f_s 단계에서.
버퍼: ls6_phi (NS,HV,G,R) = Φ̄[:R]ᵀ, ls6_ubar (NS,HV,G,V) = S[:, :G]ᵀ 사본, ls6_aq/ak (NS,HV,G), ls6_fs (NS,HV,W,G).
에필로그(플러시 뒤 Φ̄·앵커·Ū 갱신)는 gdn_ls6_epilogue_cuda.py.
설계: nested_ssm/scale/docs/latch/GDN_LS6_PHI_DESIGN_20260904.md
"""
import os

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

// dtype 코드: 0 f32, 1 bf16, 2 f16 (a/b/A_log/dt_bias 스칼라용)
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

// ── TMA 1D bulk copy + mbarrier (sm_90+) ──────────────────────────────────────────────
//   레지스터를 안 쓰고 바이트를 띄운다 — 스텝은 "떠 있는 바이트 수"가 전부라 이게 핵심이다.
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

// 전치 리덕션: 레인마다 N 개 부분합 v[i] → 반환값 = 항목 (lane & (N-1)) 의 32 레인 합 (32/N 개 레인 그룹이
// 모두 같은 값을 든다). 셔플 N-1 + log2(32/N) 번 — warp_sum N 번(5N)보다 훨씬 싸다.
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
// 그룹 폭 L(런타임, 2 의 거듭제곱 ≥ N) 안에서만 합치는 전치 리덕션: 레인 l 은 항목 (l & (N-1)) 의
//   [l & ~(L-1), +L) 레인 합을 든다.
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

// 동적 smem 배치(바이트): 워프 사유 **행 스트림** 더블 버퍼 [HPG][2][NR][V].
//   hot state 행(nf>0) → d 링 슬롯 → Ū 행이 한 줄로 이 버퍼를 지나간다(전부 TMA 1D, 레지스터 0).
template <int K, int V, int HPG> struct Smem {
    static constexpr int NR = 8;                                  // 청크 행 수
    static constexpr int BYTES = HPG * 2 * NR * V * 4;            // 24KB (V=128, HPG=3)
};

// 블록 = (행 i_n, h), 워프 j = hv (hv0 + j). 블록 = 32·HPG 스레드. 레인 l 은 v ∈ [4l, 4l+4) 를 float4 로.
//   흐름: sidx/wp → 워프마다 행 스트림 청크 0·1 발행 → 레지스터 로드(k 링/g 링/앵커) → 게이트·q̂k̂ →
//         κ(워프가 슬롯을 나눔) → sync A(κ 공개) → hot state 청크 → d 링 청크 → Φ̄ 청크(x_q/x_k → sA)
//         → f_s 청크(Σ F_s κ) → F_t 쓰기 → Ū 청크 → 출력.
//   레지스터 ≤ 80: SMSP 당 16K 라 워프당 80·32 이면 6 워프/SMSP → 블록(3 워프) 8 개/SM (smem 24KB+static).
template <int K, int V, int GT, int HPG, typename TIO>
__global__ void __maxnreg__(80)
gdn_step_kernel(
    const TIO* __restrict__ mixed_qkv, const void* __restrict__ a, const void* __restrict__ b, int ab_code,
    const void* __restrict__ A_log, const void* __restrict__ dt_bias, int p_code,
    TIO* __restrict__ out, const float* __restrict__ h0,
    float* __restrict__ d_cache, float* __restrict__ k_cache, float* __restrict__ g_cache,
    const int* __restrict__ ssm_state_indices, const int* __restrict__ write_pos, float scale,
    const int* __restrict__ fz_nf, const float* __restrict__ fz_u, const float* __restrict__ fz_z,
    const float* __restrict__ ls6_ubar, const float* __restrict__ ls6_phi,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_aq, const float* __restrict__ ls6_ak,
    float* __restrict__ ls6_fs, const int* __restrict__ ls6_map,
    long s_mix, long s_a, long s_b, long s_h0_slot, long s_h0_h, long s_ind,
    long s_d_slot, long s_k_slot, long s_g_slot, long s_fz_slot,
    long s_u_slot, long s_phi_slot, long s_a6_slot, long s_fs_slot,
    int H, int HV, int W, int G, int R, int flags)
{
    using SM = Smem<K, V, HPG>;
    constexpr int NW = HPG;                    // 워프 수
    constexpr int NR = SM::NR;
    constexpr int CK = K / 32;                 // 레인당 k 원소
    constexpr int VL = V / 32;                 // 레인당 v 원소 (=4 → float4)
    constexpr int NSW = (WMAX - 1 + NW - 1) / NW;  // 워프당 κ 슬롯 (슬롯 ≤ W-1 = 15)
    constexpr int NKV = Pow2<2 * NSW>::v;      // κ 전치 폭
    constexpr int NG = (GT + 31) / 32;         // 레인당 g 항목 수 (앵커·F_s)
    constexpr int CGW = K / 4;                 // hot state: 열 그룹 폭 (4 그룹)
    constexpr int NQ = CGW / 4;                // 열 그룹의 float4 수 (8/4/2)
    static_assert(K % 32 == 0 && K <= 128 && V == 128, "K in {32,64,128}, V == 128");
    static_assert(GT % 4 == 0 && HPG >= 2 && HPG <= 4, "GT/HPG");
    static_assert(VL == 4 && (NQ & (NQ - 1)) == 0 && V % NR == 0 && 32 % NR == 0, "layout");
    const bool FZ = flags & 1, LS6 = flags & 4, L2N = flags & 8;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int i_n = blockIdx.x, i_h = blockIdx.y, hv0 = i_h * HPG, i_hv = hv0 + warp;
    extern __shared__ __align__(128) unsigned char dsm[];
    float* sS = (float*)dsm;                                     // [NW][2][NR][V]
    __shared__ __align__(16) float sQK[2][K];                    // q̂, k̂ (블록 공유; 워프마다 같은 값을 쓴다)
    __shared__ __align__(16) float sKq[WMAX], sKk[WMAX];        // κ (감쇠 전)
    __shared__ __align__(16) float2 sC[NW][WMAX];               // (κq·r, κk·r)  워프 사유
    __shared__ __align__(16) float sA[NW][2 * GT];              // x_q, x_k → x  워프 사유
    __shared__ __align__(8) unsigned long long mbar_st[NW][2];

    // ── sidx 와 무관한 로드부터 띄운다(분기 앞) ─────────────────────────────────────────
    const long sidx = ssm_state_indices[(long)i_n * s_ind];
    // 슬롯 간접: fz_*/ls6_* 버퍼는 NX(mamba 블록 수, Flash-Next ≈1.4k)가 아니라 compact 슬롯
    //   NS(≈max_num_seqs) 로 잡는다. ls6_map[NX] → compact 행 (호스트 LRU 가 유지). null 이면 항등.
    const long cidx = (ls6_map && sidx > 0) ? (long)ls6_map[sidx] : sidx;
    const int wp = write_pos[i_n];
    const int mh_raw = (LS6 && FZ) ? ls6_mh[i_hv] : 0;
    const int nf_raw = FZ ? fz_nf[i_hv] : K;
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
    const bool is_flush = (wp == W - 1);
    const int mh = mh_raw;                     // 래치 폭 (0 = 안 탄다)
    const bool latch = mh > 0;
    // Latch heads use the same raw-write/f_s path on the flush token.  Only a
    // dense fallback head may turn the flush step into a full checkpoint read.
    const int nf = (is_flush && !latch) ? K : nf_raw;
    const int nfc = (nf + 3) & ~3;             // 복사 열 수(16B 배수; 열 마스크는 nf)

    // ── 행 스트림: hot state → d 링 → Φ̄ → f_s → Ū 청크 ─────────────────────────────────────────
    const int s0k = warp * NSW;
    const int nrk = (wp - s0k < NSW) ? (wp - s0k) : NSW;   // 이 워프의 κ 슬롯 수 (≤ 0 이면 없음)
    const int n_s = (nf > 0) ? V / NR : 0;
    const int n_d = (wp + NR - 1) / NR;
    const int RPC = (NR * V) / R;              // Φ̄ 청크당 행 수 (R=64 → 16)
    const int FPC = (NR * V) / G;              // f_s 청크당 슬롯 수 (G ≤ 64 → ≥ 16 = W 전부)
    const int n_p = latch ? (mh + RPC - 1) / RPC : 0;
    const int n_f = latch ? (wp + FPC - 1) / FPC : 0;
    const int n_u = (mh + NR - 1) / NR;        // Ū 행 mh 개 (GT > NR 이면 여러 청크)
    const int c_s = 0, c_d = n_s, c_p = c_d + n_d, c_f = c_p + n_p, c_u = c_f + n_f;
    const int NC = c_u + n_u;
    // 슬롯 베이스만 64-bit 로 한 번 잡고, 그 안의 오프셋은 int 로(64-bit IMAD 사슬을 줄인다; 슬롯 안 원소 수 < 2^31)
    const float* pst = h0 + sidx * s_h0_slot + i_hv * (int)s_h0_h;
    const float* bd = d_cache + sidx * s_d_slot;
    const float* pdr = bd + i_hv * (W * V);
    float* bk = k_cache + sidx * s_k_slot;
    float* bg = g_cache + sidx * s_g_slot;
    const float* bu = LS6 ? ls6_ubar + cidx * s_u_slot : nullptr;
    const float* bp = LS6 ? ls6_phi + cidx * s_phi_slot + i_hv * (G * R) : nullptr;
    float* pf = LS6 ? ls6_fs + cidx * s_fs_slot + i_hv * (W * G) : nullptr;
    auto issue = [&](int c, int buf) {
        float* dst = sS + ((warp * 2 + buf) * NR) * V;
        unsigned long long* bar = &mbar_st[warp][buf];
        if (c < c_d) {
            if (nfc == K) {
                if (lane == 0) { mbar_expect_tx(bar, NR * K * 4); bulk_g2s(dst, pst + c * (NR * K), NR * K * 4, bar); }
            } else {
                if (lane == 0) mbar_expect_tx(bar, (unsigned)(NR * nfc * 4));
                if (lane < NR) bulk_g2s(dst + lane * K, pst + (c * NR + lane) * K, nfc * 4, bar);
            }
        } else if (c < c_p) {
            const int r0 = (c - c_d) * NR;
            const int nr = (wp - r0 < NR) ? (wp - r0) : NR;
            if (lane == 0) { mbar_expect_tx(bar, nr * V * 4); bulk_g2s(dst, pdr + r0 * V, nr * V * 4, bar); }
        } else if (c < c_f) {
            const int r0 = (c - c_p) * RPC;
            const int nr = (mh - r0 < RPC) ? (mh - r0) : RPC;
            if (lane == 0) { mbar_expect_tx(bar, nr * R * 4); bulk_g2s(dst, bp + r0 * R, nr * R * 4, bar); }
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

    // ── 복사가 날아오는 동안: 레지스터 로드 (k 링 슬롯 s0k.., g 링, 앵커) ──
    const float gs = (lane < wp) ? bg[i_hv * W + lane] : 0.f;
    float kr[NSW][CK];                          // k 링 슬롯 s = s0k + i
    #pragma unroll
    for (int i = 0; i < NSW; ++i) {
        if (i < nrk) ld_chunk<CK>(bk + (i_h * W + s0k + i) * K + lane * CK, kr[i], false);
        else {
            #pragma unroll
            for (int c = 0; c < CK; ++c) kr[i][c] = 0.f;
        }
    }
    // 꼬리 앵커 a_q/a_k[g], g = lane + 32·i
    float aqr[NG], akr[NG];
    #pragma unroll
    for (int i = 0; i < NG; ++i) {
        const int g = lane + 32 * i;
        aqr[i] = 0.f; akr[i] = 0.f;
        if (latch && g < mh) {
            const long o = cidx * s_a6_slot + (i_hv * G + g);
            aqr[i] = ls6_aq[o]; akr[i] = ls6_ak[o];
        }
    }
    // ── 게이트 ───────────────────────────────────────────────────────────────────────
    const float xg = a_val + dtb;
    const float sp = (xg <= 20.f) ? logf(1.f + expf(xg)) : xg;
    const float g_val = -expf(Al) * sp;
    const float alpha = expf(g_val);
    const float beta = round_to(1.f / (1.f + expf(-b_val)), ab_code);
    // ── q̂, k̂ (워프마다 중복 계산; smem 엔 같은 값을 쓴다) ────────────────────────────
    float nrm[4] = {0.f, 0.f, 0.f, 0.f};       // Σq², Σk², Σqk — 전치 리덕션 한 번
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
    __syncwarp();                                   // sQK 공개(워프 안)

    // ── g 링 → replay(레인 s), tot (W ≤ WMAX 라 WMAX 레인 스캔이면 된다) ────────────────
    float pre = gs;
    #pragma unroll
    for (int o = 1; o < WMAX; o <<= 1) { const float n = __shfl_up_sync(FULL, pre, o); if (lane >= o) pre += n; }
    const float gtot = __shfl_sync(FULL, pre, WMAX - 1);
    const float rep = (lane < wp) ? expf(gtot - pre) : 0.f;
    const float tot = expf(gtot);

    // ── κ_s = k_s·q̂, k_s·k̂ (슬롯 s = s0k + i) → 전치 리덕션 → sKq/sKk ────────────────
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
        const float kval = xposeN<NKV>(kv, lane);          // 항목 lane&(NKV-1)
        const int e = lane & (NKV - 1);
        const int i = (e < NSW) ? e : e - NSW;
        if (lane < NKV && i < nrk) {
            if (e < NSW) sKq[s0k + i] = kval; else if (e < 2 * NSW) sKk[s0k + i] = kval;
        }
    }

    // v 는 출력 직전에만 쓰니 kr 이 죽은 여기서 띄운다(sync A·d 접기 동안 날아온다)
    const float4 vv = ld4<TIO>(mixed_qkv + (long)i_n * s_mix + (2 * H * K + i_hv * V + lane * VL));

    __syncthreads();   // sync A: κ(sKq/sKk) 공개 — d 접기가 여기서 풀린다
    if (lane < WMAX) sC[warp][lane] = make_float2(sKq[lane] * rep, sKk[lane] * rep);
    __syncwarp();

    // ── hot state 청크 (자기 hv 의 V 행 전부): 레인 = (행 r = lane&7, 열 그룹 cg = lane>>3) ─────
    //   열 그룹 안의 float4 를 (i + r) 로 회전해 읽어 smem 뱅크 충돌을 없앤다. q̂k̂ 도 같은 열을 smem 에서.
    //   행 r 의 합은 cg 4 레인의 셔플 2 단으로, 그 뒤 float4 소유 레인으로 옮긴다.
    float4 hq = make_float4(0.f, 0.f, 0.f, 0.f), hk = hq;
    if (n_s > 0) {
        const int r = lane & (NR - 1), cg = lane / NR;
        float oq[4] = {0.f, 0.f, 0.f, 0.f}, ok_[4] = {0.f, 0.f, 0.f, 0.f};
        // 부분 nf(< K)면 복사 안 된 열의 smem 이 쓰레기(NaN 일 수 있다) → 곱하기 전에 0 으로 가린다.
        const bool masked = (nf < K);
        auto consume = [&](int buf, float& hq_r, float& hk_r, auto MASK) {
            const float* src = sS + ((warp * 2 + buf) * NR) * V + r * K + cg * CGW;   // state 행은 K 폭
            const float* q0 = sQK[0] + cg * CGW;
            const float* k0 = sQK[1] + cg * CGW;
            float aq[4] = {0.f, 0.f, 0.f, 0.f}, ak[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int i = 0; i < NQ; ++i) {
                const int cc = ((i + r) & (NQ - 1)) * 4;
                float4 x = *((const float4*)(src + cc));
                const float4 q4 = *((const float4*)(q0 + cc)), k4 = *((const float4*)(k0 + cc));
                if (MASK) {
                    const int col = cg * CGW + cc;
                    x.x = (col < nf) ? x.x : 0.f; x.y = (col + 1 < nf) ? x.y : 0.f;
                    x.z = (col + 2 < nf) ? x.z : 0.f; x.w = (col + 3 < nf) ? x.w : 0.f;
                }
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
            if (masked) consume(buf, hq_r, hk_r, std::true_type{});
            else consume(buf, hq_r, hk_r, std::false_type{});
            __syncwarp();
            if (c + 2 < NC) issue(c + 2, buf);  // 버퍼를 다 읽었으니 재사용
            // 행 8cs + r 이 레인 r 에 있다 → 레인 l' = 2cs + (r>>2) 의 성분 r&3 로
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
    // ── d 링 청크 소비: s_q[v] = Σ_s d_s[v] κ_s r_s ───────────────────────────────────
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
    // FZ(비-LS6) 의 frozen 읽기 — 접기 동안 날아오게 여기서 띄운다(레지스터 수명을 짧게)
    float4 fu = make_float4(0.f, 0.f, 0.f, 0.f), fzv = fu;
    if (FZ && !LS6 && !is_flush) {
        const long o = cidx * s_fz_slot + (i_hv * V + lane * VL);
        fu = *((const float4*)(fz_u + o)); fzv = *((const float4*)(fz_z + o));
    }
    if (LS6) {
        // ── Φ̄ 청크 소비: x_q[g] = Φ̄[g]·q̂[:R], x_k[g] = Φ̄[g]·k̂[:R] → sA[warp][g], sA[warp][GT+g].
        //   레인 l 은 열 4(l & (L-1)) 의 float4, 패스 j 는 청크 안 float4 j·32+l → 행 j·RP + (l >> lgL).
        //   전치 리덕션(그룹 L) 뒤 레인 l 이 항목 j = l&7 → 행 (l&7)·RP + (l>>lgL) 를 든다.
        const int L = R >> 2, lgL = __ffs(L) - 1, RP = 32 >> lgL;
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
        // ── f_s 청크 소비: Σ_{s<t} F_s κq_s, Σ F_s κk_s (레인 g) ─────────────────────────
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
        __syncwarp();   // sA(x_q/x_k) 공개(워프 안)
        // F_t = β [x_k − Σ_{s<t} F_s κk_s],  x = x_q − Σ_{s<t} F_s κq_s − F_t κ_t  (κ_t = k̂·q̂)
        if (latch) {
            #pragma unroll
            for (int i = 0; i < NG; ++i) {
                const int g = lane + 32 * i;
                if (g < mh) {
                    const float baseq = sA[warp][g] + aqr[i];
                    const float basek = sA[warp][GT + g] + akr[i];
                    const float fcur = beta * (basek - erk[i]);
                    sA[warp][g] = baseq - erq[i] - fcur * cur_kq;
                    pf[wp * G + g] = fcur;
                }
            }
        }
        __syncwarp();
        // ── Ū 청크 소비: hq += Ū x ─────────────────────────────────────────────────
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
    if (FZ && !LS6 && !is_flush) {
        hq.x += fu.x; hq.y += fu.y; hq.z += fu.z; hq.w += fu.w;
        hk.x += fzv.x; hk.y += fzv.y; hk.z += fzv.z; hk.w += fzv.w;
    }

    // ── 출력, d_cur, g ─────────────────────────────────────────────────────────────────
    {
        float o[4], dc[4];
        const float hqv[4] = {hq.x, hq.y, hq.z, hq.w}, hkv[4] = {hk.x, hk.y, hk.z, hk.w};
        const float sqv[4] = {s_q.x, s_q.y, s_q.z, s_q.w}, skv[4] = {s_k.x, s_k.y, s_k.z, s_k.w};
        const float vvv[4] = {vv.x, vv.y, vv.z, vv.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float stq = alpha * (hqv[c] * tot + sqv[c]);
            const float stk = alpha * (hkv[c] * tot + skv[c]);
            // A latch head stores the state-independent transformed raw write
            // u_t.  A dense fallback head retains the old exact delta ring.
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

template <int K, int V, int GT, int HPG, typename TIO>
static void launch(cudaStream_t st, int B,
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor out, torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor ssm_state_indices, torch::Tensor write_pos, double scale,
    c10::optional<torch::Tensor> fz_nf, c10::optional<torch::Tensor> fz_u, c10::optional<torch::Tensor> fz_z,
    c10::optional<torch::Tensor> ls6_ubar, c10::optional<torch::Tensor> ls6_phi,
    c10::optional<torch::Tensor> ls6_mh, c10::optional<torch::Tensor> ls6_aq, c10::optional<torch::Tensor> ls6_ak,
    c10::optional<torch::Tensor> ls6_fs, c10::optional<torch::Tensor> ls6_map,
    int H, int HV, int W, int G, int R, int flags)
{
    const bool fz = fz_u.has_value(), ls6 = ls6_ubar.has_value();
    dim3 grid(B, H);
    constexpr int SMEM = Smem<K, V, HPG>::BYTES;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(gdn_step_kernel<K, V, GT, HPG, TIO>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        attr_set = true;
    }
    gdn_step_kernel<K, V, GT, HPG, TIO><<<grid, 32 * HPG, SMEM, st>>>(
        (const TIO*)mixed_qkv.data_ptr(), a.data_ptr(), b.data_ptr(), dt_code(a),
        A_log.data_ptr(), dt_bias.data_ptr(), dt_code(A_log),
        (TIO*)out.data_ptr(), h0.data_ptr<float>(),
        d_cache.data_ptr<float>(), k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
        ssm_state_indices.data_ptr<int>(), write_pos.data_ptr<int>(), (float)scale,
        fz ? fz_nf->data_ptr<int>() : nullptr, fz ? fz_u->data_ptr<float>() : nullptr, fz ? fz_z->data_ptr<float>() : nullptr,
        ls6 ? ls6_ubar->data_ptr<float>() : nullptr, ls6 ? ls6_phi->data_ptr<float>() : nullptr,
        ls6 ? ls6_mh->data_ptr<int>() : nullptr,
        ls6 ? ls6_aq->data_ptr<float>() : nullptr, ls6 ? ls6_ak->data_ptr<float>() : nullptr,
        ls6 ? ls6_fs->data_ptr<float>() : nullptr,
        ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
        mixed_qkv.stride(0), a.stride(0), b.stride(0), h0.stride(0), h0.stride(1), ssm_state_indices.stride(0),
        d_cache.stride(0), k_cache.stride(0), g_cache.stride(0),
        fz ? fz_u->stride(0) : 0,
        ls6 ? ls6_ubar->stride(0) : 0, ls6 ? ls6_phi->stride(0) : 0,
        ls6 ? ls6_aq->stride(0) : 0, ls6 ? ls6_fs->stride(0) : 0,
        H, HV, W, G, R, flags);
}

void gdn_step(int B,
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor out, torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor ssm_state_indices, torch::Tensor write_pos, double scale,
    c10::optional<torch::Tensor> fz_nf, c10::optional<torch::Tensor> fz_u, c10::optional<torch::Tensor> fz_z,
    c10::optional<torch::Tensor> ls6_ubar, c10::optional<torch::Tensor> ls6_phi,
    c10::optional<torch::Tensor> ls6_mh, c10::optional<torch::Tensor> ls6_aq, c10::optional<torch::Tensor> ls6_ak,
    c10::optional<torch::Tensor> ls6_fs, c10::optional<torch::Tensor> ls6_map,
    int H, int HV, int K, int V, int W, int G, int R, int flags)
{
    TORCH_CHECK(W <= WMAX, "W<=16");
    TORCH_CHECK(!ls6_phi.has_value() || R == 32 || R == 64 || R == 128, "phi 경로: R in {32,64,128}");
    auto st = at::cuda::getCurrentCUDAStream().stream();
    const int gt = G <= 8 ? 8 : (G <= 16 ? 16 : (G <= 32 ? 32 :
                   (G <= 48 ? 48 : (G <= 64 ? 64 : (G <= 80 ? 80 : 128)))));
    const int io = dt_code(mixed_qkv);
    const int hpg = HV / H;
#define ARGS st, B, mixed_qkv, a, b, A_log, dt_bias, out, h0, d_cache, k_cache, g_cache, ssm_state_indices, write_pos, scale, fz_nf, fz_u, fz_z, ls6_ubar, ls6_phi, ls6_mh, ls6_aq, ls6_ak, ls6_fs, ls6_map, H, HV, W, G, R, flags
#define CASE_T(KK, VV, GG, HH) \
    if (K == KK && V == VV && gt == GG && hpg == HH) { \
        if (io == 0) { launch<KK, VV, GG, HH, float>(ARGS); return; } \
        if (io == 1) { launch<KK, VV, GG, HH, __nv_bfloat16>(ARGS); return; } \
        TORCH_CHECK(false, "gdn_step: fp16 io 미지원"); }
#define CASE_G(KK, VV, HH) CASE_T(KK, VV, 8, HH) CASE_T(KK, VV, 16, HH) CASE_T(KK, VV, 64, HH)
#define CASE(KK, VV) CASE_G(KK, VV, 2) CASE_G(KK, VV, 3) CASE_G(KK, VV, 4)
    CASE(128, 128) CASE(64, 128) CASE(32, 128)
    // Qwen3.8 Flash-Next: K=V=128, three value heads per key head.
    // Compile its G=32 and G=128 geometries without multiplying the complete
    // specialization matrix; other non-Qwen geometries still use old buckets.
    CASE_T(128, 128, 32, 3)
    CASE_T(128, 128, 48, 3)
    CASE_T(128, 128, 80, 3)
    CASE_T(128, 128, 128, 3)
#undef CASE
#undef CASE_G
#undef CASE_T
#undef ARGS
    TORCH_CHECK(false, "gdn_step: unsupported (K,V,G)=(", K, ",", V, ",", G, ")");
}
"""

_CPP = r"""
#include <torch/extension.h>
void gdn_step(int B,
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor out, torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor ssm_state_indices, torch::Tensor write_pos, double scale,
    c10::optional<torch::Tensor> fz_nf, c10::optional<torch::Tensor> fz_u, c10::optional<torch::Tensor> fz_z,
    c10::optional<torch::Tensor> ls6_ubar, c10::optional<torch::Tensor> ls6_phi,
    c10::optional<torch::Tensor> ls6_mh, c10::optional<torch::Tensor> ls6_aq, c10::optional<torch::Tensor> ls6_ak,
    c10::optional<torch::Tensor> ls6_fs, c10::optional<torch::Tensor> ls6_map,
    int H, int HV, int K, int V, int W, int G, int R, int flags);
"""

_EXT = None
_SUPPORTED_KV = {(128, 128), (64, 128), (32, 128)}


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        bd = os.environ.get("NS_GDN_CUDA_BUILD_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gdn_step_phi_build"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load_inline(
            name="ns_gdn_step_phi_v1", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["gdn_step"], build_directory=bd,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)
    return _EXT


def gdn_step_supported(mixed_qkv, out, h0, d_cache, k_cache, g_cache, ssm_state_indices, write_pos,
                       H, HV, K, V, W, G, ls6_on, R=0):
    """CUDA 스텝이 받는 형상/dtype 인지. 아니면 래퍼가 Triton 스텝으로 되돌린다(조용히 틀리지 않게
    이유를 문자열로 돌려준다)."""
    if (K, V) not in _SUPPORTED_KV:
        return f"(K,V)=({K},{V}) 미지원"
    if W > 16:
        return f"W={W}>16"
    if HV % H or HV // H > 4 or HV // H < 2:
        return f"HV/H={HV}/{H} 는 2..4 만"
    if mixed_qkv.data_ptr() % 16 or out.data_ptr() % 16 or mixed_qkv.stride(0) % 4:
        return "mixed_qkv/out 16B 정렬 아님"
    if W % 4:
        return f"W={W} 는 4 의 배수만 (g 링 bulk copy)"
    for nm, x in (("d_cache", d_cache), ("k_cache", k_cache), ("g_cache", g_cache), ("state", h0)):
        if x.data_ptr() % 16 or x.stride(0) % 4:
            return f"{nm} 16B 정렬 아님 (bulk copy)"
    if ls6_on and (G < 4 or G > 128 or G % 4):
        return f"G={G} 는 4..128 (4 의 배수) 만"
    if ls6_on and G > 64 and (K, V, HV // H) != (128, 128, 3):
        return f"G={G}>64 특수화는 (K,V,HV/H)=(128,128,3) 만"
    if ls6_on and R not in (32, 64, 128):
        return f"R={R} 는 32/64/128 만 (Φ̄ 청크 레인 그룹)"
    for nm, tt in (("h0", h0), ("d_cache", d_cache), ("k_cache", k_cache), ("g_cache", g_cache)):
        if tt.dtype != torch.float32:
            return f"{nm} dtype {tt.dtype} != fp32"
    if h0.stride(3) != 1 or h0.stride(2) != K:
        return f"state 가 K-연속이 아님 {h0.stride()}"
    if d_cache.stride(3) != 1 or d_cache.stride(2) != V or k_cache.stride(3) != 1 or k_cache.stride(2) != K \
            or g_cache.stride(2) != 1:
        return "링 캐시가 연속이 아님"
    if mixed_qkv.dtype not in (torch.float32, torch.bfloat16) or out.dtype != mixed_qkv.dtype:
        return f"mixed_qkv/out dtype {mixed_qkv.dtype}/{out.dtype}"
    if mixed_qkv.stride(-1) != 1 or out.stride(-1) != 1 or not out.is_contiguous():
        return "mixed_qkv/out 비연속"
    if ssm_state_indices.dtype != torch.int32 or write_pos.dtype != torch.int32:
        return "indices/write_pos 가 int32 가 아님"
    return None


def gdn_step_cuda(B, mixed_qkv, a, b, A_log, dt_bias, out, h0, d_cache, k_cache, g_cache,
                  ssm_state_indices, write_pos, scale,
                  fz_nf, fz_u, fz_z,
                  ls6_ubar, ls6_phi, ls6_mh, ls6_aq, ls6_ak, ls6_fs,
                  H, HV, K, V, W, G, R, use_qk_l2norm, evict, ls6_map=None):
    for nm, tt in (("fz_u", fz_u), ("fz_z", fz_z), ("ls6_ubar", ls6_ubar), ("ls6_phi", ls6_phi),
                   ("ls6_aq", ls6_aq), ("ls6_ak", ls6_ak), ("ls6_fs", ls6_fs)):
        if tt is not None and tt.dtype != torch.float32:
            raise TypeError(f"gdn_step_cuda: {nm} 은 fp32 여야 한다 (받음 {tt.dtype})")
    if ls6_ubar is not None:
        for nm, tt, tail in (("ls6_ubar", ls6_ubar, (HV, G, V)), ("ls6_phi", ls6_phi, (HV, G, R)),
                             ("ls6_fs", ls6_fs, (HV, W, G)), ("ls6_aq", ls6_aq, (HV, G)),
                             ("ls6_ak", ls6_ak, (HV, G))):
            if tuple(tt.shape[1:]) != tail or tt.stride(-1) != 1 or \
                    tt[0].is_contiguous() is False:
                raise TypeError(f"gdn_step_cuda: {nm} 은 (NS,{tail}) 슬롯 연속이어야 한다 "
                                f"(받음 {tuple(tt.shape)} {tt.stride()})")
    if a.dtype != b.dtype or A_log.dtype != dt_bias.dtype:
        # 스칼라 dtype 코드를 (a,b)/(A_log,dt_bias) 쌍으로 하나씩만 넘긴다
        b = b.to(a.dtype)
        dt_bias = dt_bias.to(A_log.dtype)
    if fz_nf is not None and fz_nf.dtype != torch.int32:
        fz_nf = fz_nf.to(torch.int32)
    if ls6_mh is not None and ls6_mh.dtype != torch.int32:
        ls6_mh = ls6_mh.to(torch.int32)
    if ls6_map is not None and (ls6_map.dtype != torch.int32 or not ls6_map.is_contiguous() or ls6_map.dim() != 1):
        raise TypeError(f"gdn_step_cuda: ls6_map 은 연속 int32 (NX,) 여야 한다 (받음 {ls6_map.dtype} {tuple(ls6_map.shape)})")
    flags = (1 if fz_u is not None else 0) | (4 if ls6_ubar is not None else 0) | \
            (8 if use_qk_l2norm else 0) | (16 if evict else 0)
    _ext().gdn_step(int(B), mixed_qkv, a, b, A_log, dt_bias, out, h0, d_cache, k_cache, g_cache,
                    ssm_state_indices, write_pos, float(scale),
                    fz_nf, fz_u, fz_z, ls6_ubar, ls6_phi, ls6_mh, ls6_aq, ls6_ak, ls6_fs, ls6_map,
                    H, HV, K, V, W, G, R, flags)
