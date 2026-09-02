"""[nested_ssm 2026-09-02] GDN ReplaySSM v2 **flush 커널(CUDA)**.

왜 CUDA 인가 — Triton 으로는 flush 가 발행 바운드에서 안 벗어났다.
  S_new[V,K] = tot'·S₀ + Σ_{s<W} r'_s d_s k_sᵀ 는 head 당 K·V·W = 262K FMA 인데
  * 스칼라 외적 루프: d_s 브로드캐스트가 smem 왕복이라 FMA 당 2.7 명령 (all-flush 1024us)
  * tl.dot(tf32x3): B300 은 tcgen05 라 피연산자 smem·누산기 TMEM 왕복 + 255 reg 스필 (586us)
  이론은 100M FMA 명령 / 592 스케줄러 ≈ 90us, HBM 1.6GB ≈ 200us 다. 스레드가 acc 를
  레지스터에 두고 smem 에서 float4 로 d/k 를 읽으면 FMA 당 ~1.05 명령이 나온다 — 그건
  Triton 이 레이아웃을 안 맡기니 손으로 쓴다.

일감 = (flush 행, hv) = state 한 장. 블록 128 스레드 = (TV=8)×(TK=16) 격자, 스레드가
V/8 행 × K/16 열의 acc 를 레지스터에 든다(128×128 이면 16×8 = 128 reg).
  1. g 링 → tot', r'_s.  D̃[s][v] = r'_s d_s[v], K[s][k] 를 smem 에 올린다 (각 W·V, W·K).
  2. acc = tot'·S₀ (float4).  s 마다 smem 에서 d̃ (VPT) 와 k (KPT) 를 float4 로 읽어 외적 누산.
  3. S_new 저장.
  4. FZ_V2: u[v] = Σ_{k≥nf} S_new[v,k] q̄[k], z 도 — 스레드 부분합 후 TK 레인 셔플 리덕션.
  5. LS6: Ū[g,v] = tot'·Ū[g,v] + Σ_s zk[s,g] D̃[s][v]  (zk = Z̄ᵀk̂_s 는 스텝 커널이 링에 둔 것).
     ⚠ Ū_old = S₀Z̄ 가 전제(호스트 시딩) — Triton 판과 같은 불변식.

fp32 전용이다(state·링·Ū 전부). 다른 dtype 이면 래퍼가 죽는다 — 조용히 캐스팅해 느려지느니
막는다(`--cache_dt float32` 가 규약).
"""
import os

import torch

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>

#define TK 16
#define WMAX 16
#define GMAX 128

// Correct the approximate delta ring before the ordinary fold.  One block owns
// (flush row, value head); each 16-lane subgroup owns V/TV rows.  The checkpoint
// row and Ubar coefficients stay in registers while the subgroup forms all W
// key reads.  Only lane tk=0 solves the W-step triangular recurrence and writes
// the corrected d values back to the ring.
template <int K, int V, int TV>
__global__ void __launch_bounds__(TV * TK, 2)
gdn_exact_delta_kernel(
    const float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ fz_nf, const float* __restrict__ ls6_ubar,
    const float* __restrict__ ls6_xk, const int* __restrict__ ls6_mh,
    const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_u_slot, long s_xk_slot, long s_beta_slot,
    int H, int HV, int W, int G)
{
    constexpr int NT = TV * TK;
    constexpr int KPT = K / TK;
    constexpr int VPT = V / TV;
    constexpr int GPT = GMAX / TK;
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ float sX[WMAX * GMAX];
    __shared__ float sRho[WMAX * WMAX];
    __shared__ float sPrefix[WMAX];
    __shared__ float sBeta[WMAX];
    const int t = threadIdx.x;
    const int tk = t % TK, tv = t / TK;
    const int k0 = tk * KPT, v0 = tv * VPT;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;
    for (int w = blockIdx.x; w < n_work; w += gridDim.x) {
        const int rr = w / HV, i_hv = w % HV, i_h = i_hv / hpg;
        const int mh = ls6_mh[i_hv];
        if (mh <= 0) continue;
        const int nf = fz_nf[i_hv];
        const long sidx = (long)flush_list[rr];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        const float* pg = g_cache + sidx * s_g_slot + (long)i_hv * W;
        const float* px = ls6_xk + cidx * s_xk_slot + (long)i_h * W * G;
        const float* pb = ls6_beta + cidx * s_beta_slot + (long)i_hv * W;

        for (int i = t; i < W * K; i += NT) sK[i] = pk[i];
        for (int i = t; i < W * G; i += NT)
            sX[i] = (i % G < mh) ? px[i] : 0.f;
        if (t < W) sBeta[t] = pb[t];
        const float gv = (t < W) ? pg[t] : 0.f;
        if (t < 32) {
            float pre = gv;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, pre, o);
                if (t >= o) pre += y;
            }
            // Keep log-prefixes.  exp(prefix_s)/exp(prefix_j) becomes 0/0
            // under strong real-model decay, while exp(prefix_s-prefix_j)
            // is equivalent and its argument is always non-positive.
            if (t < W) sPrefix[t] = pre;
        }
        __syncthreads();

        // rho_{j,s} = <k_j,k_s> exp(prefix_s-prefix_j).
        for (int ix = t; ix < W * W; ix += NT) {
            const int j = ix / W, s = ix - j * W;
            float kap = 0.f;
            if (j < s) {
                for (int kk = 0; kk < K; ++kk)
                    kap = fmaf(sK[j * K + kk], sK[s * K + kk], kap);
                kap *= __expf(sPrefix[s] - sPrefix[j]);
            }
            sRho[ix] = kap;
        }
        __syncthreads();

        const float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
        const float* pu = ls6_ubar + cidx * s_u_slot + (long)i_hv * G * V;
        float* pd = d_cache + sidx * s_d_slot + (long)i_hv * W * V;
        #pragma unroll
        for (int vi = 0; vi < VPT; ++vi) {
            const int v = v0 + vi;
            float hv[KPT];
            #pragma unroll
            for (int j = 0; j < KPT; ++j)
                hv[j] = ((k0 + j) >= nf) ? ph[(long)v * K + k0 + j] : 0.f;
            float uv[GPT];
            #pragma unroll
            for (int q = 0; q < GPT; ++q) {
                const int gg = tk + q * TK;
                uv[q] = (gg < mh) ? pu[(long)gg * V + v] : 0.f;
            }
            float eps[WMAX];
            #pragma unroll
            for (int s = 0; s < WMAX; ++s) {
                float e = 0.f;
                if (s < W - 1) {
                    #pragma unroll
                    for (int j = 0; j < KPT; ++j)
                        e = fmaf(-hv[j], sK[s * K + k0 + j], e);
                    #pragma unroll
                    for (int q = 0; q < GPT; ++q) {
                        const int gg = tk + q * TK;
                        if (gg < mh) e = fmaf(uv[q], sX[s * G + gg], e);
                    }
                    #pragma unroll
                    for (int o = TK / 2; o > 0; o >>= 1)
                        e += __shfl_xor_sync(0xffffffffu, e, o);
                    e *= __expf(sPrefix[s]);
                }
                eps[s] = e;
            }
            if (tk == 0) {
                float delta[WMAX];
                #pragma unroll
                for (int s = 0; s < WMAX; ++s) {
                    float prev = 0.f;
                    if (s < W) {
                        #pragma unroll
                        for (int j = 0; j < WMAX; ++j)
                            if (j < s) prev = fmaf(sRho[j * W + s], delta[j], prev);
                        delta[s] = -sBeta[s] * (eps[s] + prev);
                        pd[(long)s * V + v] -= delta[s];
                    } else {
                        delta[s] = 0.f;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Flash-Next production geometry.  The step kernel stores the coefficient
// x_s = a_k + Z_r^T(k_s-kbar) once per key head.  Two 64-row CTAs per value
// head evaluate Ubar^T x - S0 k_cold with TF32 tensor cores; the triangular
// solve remains fp32.  This avoids rebuilding x in every value-head CTA and
// gives the small 128x(128+G)x16 product enough independent output tiles.
__global__ void __launch_bounds__(128, 4)
gdn_exact_delta_tc_128_kernel(
    const float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ fz_nf, const float* __restrict__ ls6_ubar,
    const float* __restrict__ ls6_xk,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_u_slot, long s_xk_slot, long s_beta_slot,
    int H, int HV, int W, int G)
{
    constexpr int K = 128, V = 128, VT = 64, NT = 128;
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ __align__(16) float sX[WMAX * GMAX];
    __shared__ __align__(16) float sEps[VT * WMAX];
    __shared__ float sRho[WMAX * WMAX];
    __shared__ float sPrefix[WMAX];
    __shared__ float sBeta[WMAX];
    const int t = threadIdx.x;
    const int n_work = n_ptr[0] * HV * 2;
    const int hpg = HV / H;
    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int tile = work & 1;
        const int wh = work >> 1;
        const int rr = wh / HV, i_hv = wh % HV, i_h = i_hv / hpg;
        const int vbase = tile * VT;
        const int mh = ls6_mh[i_hv];
        if (mh <= 0) continue;
        const int nf = fz_nf[i_hv];
        const long sidx = (long)flush_list[rr];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        const float* pg = g_cache + sidx * s_g_slot + (long)i_hv * W;
        const float* px = ls6_xk + cidx * s_xk_slot + (long)i_h * W * G;
        const float* pb = ls6_beta + cidx * s_beta_slot + (long)i_hv * W;

        for (int i = t; i < W * K; i += NT) sK[i] = pk[i];
        for (int i = t; i < W * G; i += NT)
            sX[i] = (i % G < mh) ? px[i] : 0.f;
        if (t < W) sBeta[t] = pb[t];
        const float gv = (t < W) ? pg[t] : 0.f;
        if (t < 32) {
            float pre = gv;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, pre, o);
                if (t >= o) pre += y;
            }
            // Store log-prefixes to avoid exp-underflow ratios (0/0).
            if (t < W) sPrefix[t] = pre;
        }
        __syncthreads();

        for (int ix = t; ix < W * W; ix += NT) {
            const int j = ix / W, s = ix - j * W;
            float kap = 0.f;
            if (j < s) {
                for (int kk = 0; kk < K; ++kk)
                    kap = fmaf(sK[j * K + kk], sK[s * K + kk], kap);
                kap *= __expf(sPrefix[s] - sPrefix[j]);
            }
            sRho[ix] = kap;
        }
        __syncthreads();
        // The W-by-K ring is the column-major K-by-W operand.  Zero its hot
        // prefix and negate the cold suffix so one accumulator can add both
        // Ubar^T x and -S0 k_cold.
        for (int i = t; i < W * K; i += NT)
            sK[i] = (i % K >= nf) ? -sK[i] : 0.f;
        __syncthreads();

        using namespace nvcuda;
        const int warp = t >> 5;
        const int vl = warp * 16;
        const int vg = vbase + vl;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
        wmma::fill_fragment(acc, 0.f);
        const float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
        #pragma unroll
        for (int kk = 0; kk < K; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> bf;
            wmma::load_matrix_sync(af, ph + (long)vg * K + kk, K);
            wmma::load_matrix_sync(bf, sK + kk, K);
            wmma::mma_sync(acc, af, bf, acc);
        }
        const float* pu = ls6_ubar + cidx * s_u_slot + (long)i_hv * G * V;
        #pragma unroll
        for (int gg = 0; gg < GMAX; gg += 8) {
            if (gg >= G) break;
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> bf;
            wmma::load_matrix_sync(af, pu + (long)gg * V + vg, V);
            wmma::load_matrix_sync(bf, sX + gg, G);
            wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(sEps + vl * WMAX, acc, WMAX,
                                wmma::mem_row_major);
        __syncthreads();

        float* pd = d_cache + sidx * s_d_slot + (long)i_hv * W * V;
        if (t < VT) {
            const int v = vbase + t;
            float delta[WMAX];
            #pragma unroll
            for (int s = 0; s < WMAX; ++s) {
                float prev = 0.f;
                if (s < W) {
                    #pragma unroll
                    for (int j = 0; j < WMAX; ++j)
                        if (j < s) prev = fmaf(sRho[j * W + s], delta[j], prev);
                    // The final slot was evaluated exactly by the step kernel.
                    const float eps = (s < W - 1) ? sEps[t * WMAX + s] * __expf(sPrefix[s]) : 0.f;
                    delta[s] = -sBeta[s] * (eps + prev);
                    pd[(long)s * V + v] -= delta[s];
                } else {
                    delta[s] = 0.f;
                }
            }
        }
        __syncthreads();
    }
}

template <int K, int V, int TV>
__global__ void __launch_bounds__(TV * TK, (TV >= 32 ? 2 : 2))
gdn_flush_kernel(
    float* __restrict__ h0, const float* __restrict__ d_cache, const float* __restrict__ k_cache,
    const float* __restrict__ g_cache, const int* __restrict__ ssm_state_indices,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ fz_nf, float* __restrict__ fz_u, float* __restrict__ fz_z,
    const float* __restrict__ fz_qbar, const float* __restrict__ fz_kbar,
    float* __restrict__ ls6_ubar, const float* __restrict__ ls6_zk, const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_ind, long s_d_slot, long s_k_slot, long s_g_slot,
    long s_fz_slot, long s_fzb_slot, long s_u_slot, long s_zk_slot,
    int H, int HV, int W, int G, int dbg)
{
    constexpr int NT = TV * TK;
    constexpr int KPT = K / TK;   // 스레드당 k 열
    constexpr int VPT = V / TV;   // 스레드당 v 행
    constexpr int ND4 = (WMAX * V / 4 + NT - 1) / NT;   // 스테이징 float4 / 스레드 (정적 → 로드가 한꺼번에 뜬다)
    constexpr int NK4 = (WMAX * K / 4 + NT - 1) / NT;
    static_assert(KPT % 4 == 0 || KPT == 2 || KPT == 1, "KPT");
    __shared__ __align__(16) float sD[WMAX * V];     // r'_s d_s[v]
    __shared__ __align__(16) float sK[WMAX * K];     // k_s[k]
    __shared__ float sRep[WMAX + 1];                 // [W] = tot'
    __shared__ float sZk[WMAX * GMAX];
    const int t = threadIdx.x;
    const int tk = t % TK, tv = t / TK;
    const int k0 = tk * KPT, v0 = tv * VPT;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;
    const bool ls6 = ls6_ubar != nullptr;
    const int nd4 = W * V / 4, nk4 = W * K / 4;
    for (int w = blockIdx.x; w < n_work; w += gridDim.x) {
        const int r = w / HV, i_hv = w % HV, i_h = i_hv / hpg;
        // compact 커널이 flush_list 에 **state_idx 를 직접** 넣는다(행 번호가 아니라) — 종속 로드 1단 절약.
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;   // fz_*/ls6_* 는 compact 슬롯 (스텝 커널과 같은 규약)
        const float* pg = g_cache + sidx * s_g_slot + (long)i_hv * W;
        // ── 종속 사슬을 짧게: S₀·링·g·zk 로드를 **전부 먼저** 띄운다(레지스터로) ──
        float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
        float acc[VPT][KPT];
        #pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const float* row = ph + (long)(v0 + i) * K + k0;
            if (dbg & 1) { for (int j = 0; j < KPT; ++j) acc[i][j] = 0.f; continue; }
            if constexpr (KPT % 4 == 0) {
                #pragma unroll
                for (int j = 0; j < KPT; j += 4) {
                    float4 x = *reinterpret_cast<const float4*>(row + j);
                    acc[i][j] = x.x; acc[i][j+1] = x.y; acc[i][j+2] = x.z; acc[i][j+3] = x.w;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < KPT; ++j) acc[i][j] = row[j];
            }
        }
        const float4* pd4 = reinterpret_cast<const float4*>(d_cache + sidx * s_d_slot + (long)i_hv * W * V);
        const float4* pk4 = reinterpret_cast<const float4*>(k_cache + sidx * s_k_slot + (long)i_h * W * K);
        float4 dreg[ND4], kreg[NK4];
        #pragma unroll
        for (int q = 0; q < ND4; ++q) { const int i = t + q * NT; dreg[q] = (i < nd4 && !(dbg & 8)) ? pd4[i] : make_float4(0.f, 0.f, 0.f, 0.f); }
        #pragma unroll
        for (int q = 0; q < NK4; ++q) { const int i = t + q * NT; kreg[q] = (i < nk4) ? pk4[i] : make_float4(0.f, 0.f, 0.f, 0.f); }
        const float gv = (t < W) ? pg[t] : 0.f;
        if (ls6) {
            const float* pz = ls6_zk + cidx * s_zk_slot + (long)i_h * W * G;
            for (int i = t; i < W * G; i += NT) sZk[i] = pz[i];
        }
        // ── tot', r'_s: 워프 0 셔플 prefix (g 를 s 마다 읽으면 종속 로드 W 단이라 커널이 그 지연에 묶였다) ──
        if (t < 32) {
            float pre = gv;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, pre, o);
                if (t >= o) pre += y;
            }
            const float gt = __shfl_sync(0xffffffffu, pre, 31);
            if (t < W) sRep[t] = __expf(gt - pre);
            if (t == 0) sRep[WMAX] = __expf(gt);
        }
        #pragma unroll
        for (int q = 0; q < NK4; ++q) { const int i = t + q * NT; if (i < nk4) reinterpret_cast<float4*>(sK)[i] = kreg[q]; }
        __syncthreads();
        const float tot = sRep[WMAX];
        #pragma unroll
        for (int q = 0; q < ND4; ++q) {
            const int i = t + q * NT;
            if (i < nd4) {
                float4 x = dreg[q];
                const float rr = sRep[(i * 4) / V];
                x.x *= rr; x.y *= rr; x.z *= rr; x.w *= rr;
                reinterpret_cast<float4*>(sD)[i] = x;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < VPT; ++i)
            #pragma unroll
            for (int j = 0; j < KPT; ++j) acc[i][j] *= tot;
        // ── Σ_s d̃_s k_sᵀ ──
        for (int s = 0; s < ((dbg & 2) ? 0 : W); ++s) {
            float dv[VPT], kk[KPT];
            const float* ds = sD + s * V + v0;
            const float* ks = sK + s * K + k0;
            #pragma unroll
            for (int i = 0; i < VPT; i += 4) {
                if constexpr (VPT % 4 == 0) {
                    float4 x = *reinterpret_cast<const float4*>(ds + i);
                    dv[i] = x.x; dv[i+1] = x.y; dv[i+2] = x.z; dv[i+3] = x.w;
                } else {
                    #pragma unroll
                    for (int q = 0; q < 4 && i + q < VPT; ++q) dv[i+q] = ds[i+q];
                }
            }
            #pragma unroll
            for (int j = 0; j < KPT; j += 4) {
                if constexpr (KPT % 4 == 0) {
                    float4 x = *reinterpret_cast<const float4*>(ks + j);
                    kk[j] = x.x; kk[j+1] = x.y; kk[j+2] = x.z; kk[j+3] = x.w;
                } else {
                    #pragma unroll
                    for (int q = 0; q < 4 && j + q < KPT; ++q) kk[j+q] = ks[j+q];
                }
            }
            #pragma unroll
            for (int i = 0; i < VPT; ++i)
                #pragma unroll
                for (int j = 0; j < KPT; ++j)
                    acc[i][j] = fmaf(dv[i], kk[j], acc[i][j]);
        }
        // ── 저장 ──
        #pragma unroll
        for (int i = 0; i < VPT; ++i) {
            if (dbg & 4) break;
            float* row = ph + (long)(v0 + i) * K + k0;
            if constexpr (KPT % 4 == 0) {
                #pragma unroll
                for (int j = 0; j < KPT; j += 4)
                    *reinterpret_cast<float4*>(row + j) = make_float4(acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]);
            } else {
                #pragma unroll
                for (int j = 0; j < KPT; ++j) row[j] = acc[i][j];
            }
        }
        // ── FZ_V2: u = S_new q̄_cold, z = S_new k̄_cold ──
        if (fz_u != nullptr) {
            const int nf = fz_nf[i_hv];
            const float* qb = fz_qbar + cidx * s_fzb_slot + (long)i_h * K + k0;
            const float* kb = fz_kbar + cidx * s_fzb_slot + (long)i_h * K + k0;
            float qv[KPT], kv[KPT];
            #pragma unroll
            for (int j = 0; j < KPT; ++j) {
                const bool cold = (k0 + j) >= nf;
                qv[j] = cold ? qb[j] : 0.f;
                kv[j] = cold ? kb[j] : 0.f;
            }
            #pragma unroll
            for (int i = 0; i < VPT; ++i) {
                float pu = 0.f, pz = 0.f;
                #pragma unroll
                for (int j = 0; j < KPT; ++j) { pu = fmaf(acc[i][j], qv[j], pu); pz = fmaf(acc[i][j], kv[j], pz); }
                #pragma unroll
                for (int o = TK / 2; o > 0; o >>= 1) {
                    pu += __shfl_xor_sync(0xffffffffu, pu, o);
                    pz += __shfl_xor_sync(0xffffffffu, pz, o);
                }
                if (tk == 0) {
                    fz_u[cidx * s_fz_slot + (long)i_hv * V + v0 + i] = pu;
                    fz_z[cidx * s_fz_slot + (long)i_hv * V + v0 + i] = pz;
                }
            }
        }
        // ── LS6: Ū[g,v] = tot'·Ū[g,v] + Σ_s zk[s,g] d̃_s[v]   (idx=(g,v) 를 스레드에 펼치고 u_old 는 묶어 읽는다) ──
        if (ls6) {
            float* pu = ls6_ubar + cidx * s_u_slot + (long)i_hv * G * V;
            const int n_gv = G * V;
            for (int base = 0; base < n_gv; base += NT * 4) {
                float uo[4]; int gi[4], vi[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int idx = base + t + q * NT;
                    gi[q] = idx / V; vi[q] = idx - gi[q] * V;
                    uo[q] = (idx < n_gv) ? pu[idx] : 0.f;
                }
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int idx = base + t + q * NT;
                    if (idx < n_gv) {
                        float u = tot * uo[q];
                        for (int s = 0; s < W; ++s) u = fmaf(sZk[s * G + gi[q]], sD[s * V + vi[q]], u);
                        pu[idx] = u;
                    }
                }
            }
        }
        __syncthreads();   // 다음 일감이 smem 을 덮기 전에
    }
}

template <int K, int V, int TV>
static void launch(int grid, cudaStream_t st,
    torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor ssm_state_indices, torch::Tensor flush_list, int n_off,
    c10::optional<torch::Tensor> fz_nf, c10::optional<torch::Tensor> fz_u, c10::optional<torch::Tensor> fz_z,
    c10::optional<torch::Tensor> fz_qbar, c10::optional<torch::Tensor> fz_kbar,
    c10::optional<torch::Tensor> ls6_ubar, c10::optional<torch::Tensor> ls6_zk,
    c10::optional<torch::Tensor> ls6_xk, c10::optional<torch::Tensor> ls6_map,
    c10::optional<torch::Tensor> ls6_mh, c10::optional<torch::Tensor> ls6_beta,
    int H, int HV, int W, int G, int dbg)
{
    const bool fz = fz_u.has_value();
    const bool ls6 = ls6_ubar.has_value();
    const bool exact = ls6_beta.has_value();
    if (exact) {
        if constexpr (K == 128 && V == 128) {
            if (G >= 8 && G % 8 == 0) {
                gdn_exact_delta_tc_128_kernel<<<grid, 128, 0, st>>>(
                h0.data_ptr<float>(), d_cache.data_ptr<float>(), k_cache.data_ptr<float>(),
                g_cache.data_ptr<float>(), flush_list.data_ptr<int>(),
                flush_list.data_ptr<int>() + n_off, fz_nf->data_ptr<int>(),
                ls6_ubar->data_ptr<float>(), ls6_xk->data_ptr<float>(),
                ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                h0.stride(0), h0.stride(1), d_cache.stride(0), k_cache.stride(0),
                g_cache.stride(0), ls6_ubar->stride(0), ls6_xk->stride(0),
                ls6_beta->stride(0), H, HV, W, G);
            } else {
                gdn_exact_delta_kernel<K, V, TV><<<grid, TV * TK, 0, st>>>(
                    h0.data_ptr<float>(), d_cache.data_ptr<float>(), k_cache.data_ptr<float>(),
                    g_cache.data_ptr<float>(), flush_list.data_ptr<int>(),
                    flush_list.data_ptr<int>() + n_off, fz_nf->data_ptr<int>(),
                    ls6_ubar->data_ptr<float>(), ls6_xk->data_ptr<float>(),
                    ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                    ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                    h0.stride(0), h0.stride(1), d_cache.stride(0), k_cache.stride(0),
                    g_cache.stride(0), ls6_ubar->stride(0), ls6_xk->stride(0),
                    ls6_beta->stride(0), H, HV, W, G);
            }
        } else {
            gdn_exact_delta_kernel<K, V, TV><<<grid, TV * TK, 0, st>>>(
                h0.data_ptr<float>(), d_cache.data_ptr<float>(), k_cache.data_ptr<float>(),
                g_cache.data_ptr<float>(), flush_list.data_ptr<int>(),
                flush_list.data_ptr<int>() + n_off, fz_nf->data_ptr<int>(),
                ls6_ubar->data_ptr<float>(), ls6_xk->data_ptr<float>(),
                ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                h0.stride(0), h0.stride(1), d_cache.stride(0), k_cache.stride(0),
                g_cache.stride(0), ls6_ubar->stride(0), ls6_xk->stride(0),
                ls6_beta->stride(0), H, HV, W, G);
        }
    }
    gdn_flush_kernel<K, V, TV><<<grid, TV * TK, 0, st>>>(
        h0.data_ptr<float>(), d_cache.data_ptr<float>(), k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
        ssm_state_indices.data_ptr<int>(), flush_list.data_ptr<int>(), flush_list.data_ptr<int>() + n_off,
        fz ? fz_nf->data_ptr<int>() : nullptr, fz ? fz_u->data_ptr<float>() : nullptr, fz ? fz_z->data_ptr<float>() : nullptr,
        fz ? fz_qbar->data_ptr<float>() : nullptr, fz ? fz_kbar->data_ptr<float>() : nullptr,
        ls6 ? ls6_ubar->data_ptr<float>() : nullptr, ls6 ? ls6_zk->data_ptr<float>() : nullptr,
        ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
        h0.stride(0), h0.stride(1), ssm_state_indices.stride(0), d_cache.stride(0), k_cache.stride(0), g_cache.stride(0),
        fz ? fz_u->stride(0) : 0, fz ? fz_qbar->stride(0) : 0, ls6 ? ls6_ubar->stride(0) : 0, ls6 ? ls6_zk->stride(0) : 0,
        H, HV, W, G, dbg);
}

void gdn_flush(int grid,
    torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor ssm_state_indices, torch::Tensor flush_list, int n_off,
    c10::optional<torch::Tensor> fz_nf, c10::optional<torch::Tensor> fz_u, c10::optional<torch::Tensor> fz_z,
    c10::optional<torch::Tensor> fz_qbar, c10::optional<torch::Tensor> fz_kbar,
    c10::optional<torch::Tensor> ls6_ubar, c10::optional<torch::Tensor> ls6_zk,
    c10::optional<torch::Tensor> ls6_xk, c10::optional<torch::Tensor> ls6_map,
    c10::optional<torch::Tensor> ls6_mh, c10::optional<torch::Tensor> ls6_beta,
    int H, int HV, int K, int V, int W, int G, int tv, int dbg)
{
    TORCH_CHECK(W <= WMAX && G <= GMAX, "W<=16, G<=128");
    auto st = at::cuda::getCurrentCUDAStream().stream();
#define CASE(KK, VV, TVV) if (K == KK && V == VV && tv == TVV) { launch<KK, VV, TVV>(grid, st, h0, d_cache, k_cache, g_cache, ssm_state_indices, flush_list, n_off, fz_nf, fz_u, fz_z, fz_qbar, fz_kbar, ls6_ubar, ls6_zk, ls6_xk, ls6_map, ls6_mh, ls6_beta, H, HV, W, G, dbg); return; }
    CASE(128, 128, 8) CASE(128, 128, 16) CASE(128, 128, 32)
    CASE(64, 128, 8) CASE(64, 128, 16) CASE(64, 128, 32)
    CASE(64, 64, 8) CASE(64, 64, 16) CASE(32, 32, 8) CASE(128, 64, 8) CASE(128, 64, 16)
#undef CASE
    TORCH_CHECK(false, "gdn_flush: unsupported (K,V,tv)=(", K, ",", V, ",", tv, ")");
}
"""

_CPP = r"""
#include <torch/extension.h>
void gdn_flush(int grid,
    torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor ssm_state_indices, torch::Tensor flush_list, int n_off,
    c10::optional<torch::Tensor> fz_nf, c10::optional<torch::Tensor> fz_u, c10::optional<torch::Tensor> fz_z,
    c10::optional<torch::Tensor> fz_qbar, c10::optional<torch::Tensor> fz_kbar,
    c10::optional<torch::Tensor> ls6_ubar, c10::optional<torch::Tensor> ls6_zk,
    c10::optional<torch::Tensor> ls6_xk, c10::optional<torch::Tensor> ls6_map,
    c10::optional<torch::Tensor> ls6_mh, c10::optional<torch::Tensor> ls6_beta,
    int H, int HV, int K, int V, int W, int G, int tv, int dbg);
"""

_EXT = None

_SUPPORTED_KV = {(128, 128), (64, 128), (64, 64), (32, 32), (128, 64)}


def gdn_flush_supported(h0, d_cache, k_cache, g_cache, K, V, W, G):
    """Return None when the hand-written CUDA fold can consume this layout."""
    if (K, V) not in _SUPPORTED_KV:
        return f"(K,V)=({K},{V}) unsupported"
    if W > 16:
        return f"W={W}>16"
    if G > 128:
        return f"G={G}>128"
    if K % 16 or V % 8:
        return f"K%16={K % 16}, V%8={V % 8}"
    if h0.dtype != torch.float32 or d_cache.dtype != torch.float32 \
            or k_cache.dtype != torch.float32 or g_cache.dtype != torch.float32:
        return "state/ring dtype is not fp32"
    if h0.stride(3) != 1 or h0.stride(2) != K:
        return f"state is not K-contiguous: {h0.stride()}"
    return None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        bd = os.environ.get("NS_GDN_CUDA_BUILD_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gdn_flush_build"))
        os.makedirs(bd, exist_ok=True)
        _EXT = load_inline(
            name="ns_gdn_flush_v6k", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["gdn_flush"], build_directory=bd,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)
    return _EXT


def gdn_flush_cuda(grid, h0, d_cache, k_cache, g_cache, ssm_state_indices, flush_list, n_off,
                   fz_nf, fz_u, fz_z, fz_qbar, fz_kbar, ls6_ubar, ls6_zk, ls6_xk,
                   H, HV, K, V, W, G,
                   ls6_map=None, ls6_mh=None, ls6_beta=None):
    for nm, tt in (("h0", h0), ("d_cache", d_cache), ("k_cache", k_cache), ("g_cache", g_cache),
                   ("fz_u", fz_u), ("fz_z", fz_z), ("fz_qbar", fz_qbar), ("fz_kbar", fz_kbar),
                   ("ls6_ubar", ls6_ubar), ("ls6_zk", ls6_zk), ("ls6_xk", ls6_xk),
                   ("ls6_beta", ls6_beta)):
        if tt is not None and tt.dtype != torch.float32:
            raise TypeError(f"gdn_flush_cuda: {nm} 은 fp32 여야 한다 (받음 {tt.dtype}) — state 는 fp32 규약")
    if h0.stride(3) != 1 or h0.stride(2) != K:
        raise ValueError(f"gdn_flush_cuda: state 는 (…,V,K) K-연속이어야 한다: strides {h0.stride()}")
    if K % 16 or V % 8:
        raise ValueError(f"gdn_flush_cuda: K%16==0, V%8==0 필요 (K={K}, V={V})")
    # 블록 = tv×16 스레드, 스레드당 V/tv 행. tv=32(512 스레드, acc 32 reg)가 기본 — 블록당
    # 레지스터가 작아야 SM 에 워프가 많이 올라 종속 로드 사슬이 겹친다.
    tv = min(int(os.environ.get("NS_GDN_FLUSH_TV", "32")), max(8, V // 4))
    if ls6_map is not None and (ls6_map.dtype != torch.int32 or not ls6_map.is_contiguous() or ls6_map.dim() != 1):
        raise TypeError(f"gdn_flush_cuda: ls6_map 은 연속 int32 (NX,) 여야 한다 (받음 {ls6_map.dtype} {tuple(ls6_map.shape)})")
    corr = (ls6_mh, ls6_beta)
    n_corr = sum(t is not None for t in corr)
    if n_corr not in (0, len(corr)):
        raise ValueError(
            f"gdn_flush_cuda: exact 인자는 모두 주거나 빼야 한다 ({n_corr}/2)"
        )
    if n_corr and ls6_xk is None:
        raise ValueError("gdn_flush_cuda: exact-flush에는 ls6_xk 계수 링이 필요하다")
    if ls6_mh is not None and ls6_mh.dtype != torch.int32:
        raise TypeError(
            f"gdn_flush_cuda: ls6_mh 는 int32여야 한다 ({ls6_mh.dtype})"
        )
    _ext().gdn_flush(int(grid), h0, d_cache, k_cache, g_cache, ssm_state_indices, flush_list, int(n_off),
                     fz_nf, fz_u, fz_z, fz_qbar, fz_kbar, ls6_ubar, ls6_zk, ls6_xk, ls6_map,
                     ls6_mh, ls6_beta, H, HV, K, V, W, G, tv,
                     int(os.environ.get('NS_GDN_FLUSH_DBG', '0')))
