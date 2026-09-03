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

template <int N>
__device__ __forceinline__ float transpose_reduce(float (&v)[N], int lane) {
    static_assert((N & (N - 1)) == 0 && N <= 32, "power-of-two N");
    #pragma unroll
    for (int off = N / 2; off >= 1; off >>= 1) {
        const bool upper = (lane & off) != 0;
        #pragma unroll
        for (int i = 0; i < off; ++i) {
            const float send = upper ? v[i] : v[i + off];
            const float keep = upper ? v[i + off] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
    }
    float result = v[0];
    #pragma unroll
    for (int off = N; off < 32; off <<= 1)
        result += __shfl_xor_sync(0xffffffffu, result, off);
    return result;
}

template <int ROWS>
__device__ __forceinline__ int tc_swizzle(int row, int col) {
    const int x = row * 32 + (col & 31) + (col >> 5) * ROWS * 32;
    // CuTe lowers Swizzle<3,4,3> on the fp32 shared byte address to
    // b' = b ^ ((b >> 3) & 0x70).  For a 1024-B-aligned tile base this is
    // the following permutation of the element offset.
    return x ^ ((x & 0xe0) >> 3);
}

__device__ __forceinline__ unsigned tc_smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ unsigned long long tc_sdesc(
    const void* p, unsigned leading_byte_offset) {
    constexpr unsigned stride_byte_offset = 8 * 128;
    const unsigned addr = tc_smem_u32(p);
    // A 128-B-swizzled matrix whose base is not on the 1024-B pattern
    // boundary must carry the starting phase in descriptor bits 49..51.
    // CuTe's standalone kernel allocates both operands on 64-KiB/8-KiB
    // boundaries, whereas this fused kernel's dynamic tile follows its
    // static shared segment and therefore needs the phase explicitly.
    return ((unsigned long long)addr >> 4)
        | ((unsigned long long)(leading_byte_offset >> 4) << 16)
        | ((unsigned long long)(stride_byte_offset >> 4) << 32)
        | ((unsigned long long)((addr >> 7) & 7u) << 49)
        | (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ void tc_alloc(unsigned* dst) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(tc_smem_u32(dst)), "r"(32) : "memory");
    // Allocation ownership and the allocator permit are distinct.  Keeping
    // the permit until dealloc serializes otherwise-resident CTAs on an SM.
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
        ::: "memory");
}
__device__ __forceinline__ void tc_alloc_n(unsigned* dst, unsigned ncols) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(tc_smem_u32(dst)), "r"(ncols) : "memory");
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
        ::: "memory");
}
__device__ __forceinline__ void tc_dealloc(unsigned tmem_base) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_base), "r"(32) : "memory");
}
__device__ __forceinline__ void tc_dealloc_n(
    unsigned tmem_base, unsigned ncols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_base), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_mma_tf32(
    unsigned d, unsigned long long a, unsigned long long b, unsigned idesc,
    bool accumulate) {
    unsigned mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, "
        "{%5, %6, %7, %8}, p; }"
        :: "r"(d), "l"(a), "l"(b), "r"(idesc),
           "r"((unsigned)accumulate), "r"(mask[0]), "r"(mask[1]),
           "r"(mask[2]), "r"(mask[3]) : "memory");
}
__device__ __forceinline__ void tc_commit(unsigned long long* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];" :: "r"(tc_smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tc_mbar_wait(
    unsigned long long* bar, unsigned parity = 0) {
    unsigned done;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
            "selp.u32 %0, 1, 0, p; }"
            : "=r"(done) : "r"(tc_smem_u32(bar)), "r"(parity) : "memory");
    } while (!done);
}
__device__ __forceinline__ float tc_load1(
    unsigned tmem_base, int row, int col) {
    unsigned x;
    const unsigned addr = ((unsigned)row << 16) | (tmem_base + (unsigned)col);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
        : "=r"(x) : "r"(addr));
    return __uint_as_float(x);
}
__device__ __forceinline__ void tc_load16(
    unsigned tmem_base, int row, float (&out)[16]) {
    unsigned x[16];
    const unsigned addr = ((unsigned)row << 16) | tmem_base;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]),
          "=r"(x[4]), "=r"(x[5]), "=r"(x[6]), "=r"(x[7]),
          "=r"(x[8]), "=r"(x[9]), "=r"(x[10]), "=r"(x[11]),
          "=r"(x[12]), "=r"(x[13]), "=r"(x[14]), "=r"(x[15])
        : "r"(addr) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = __uint_as_float(x[i]);
}
__device__ __forceinline__ void tc_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// Exact checkpoint refresh in the algebra used by the raw-write/f_s online
// kernel.  Write H=S^T and split it into
//
//   H_s = C_s + R_s,
//   C_s = alpha_s C_{s-1}(I-beta_s k_s k_s^T),
//   R_s = alpha_s R_{s-1} + u_s k_s^T.
//
// The online kernel stores the state-independent u_s in d_cache.  Therefore
// one warp can carry one row of C and R through the sixteen causal steps using
// only eight data registers per lane.  This is exactly the raw transition in
// Eq. (exact-refresh), but avoids materialising S0 K^T, K K^T, or a triangular
// solve.  Every checkpoint element is read and written once.
template <int K, int V, int NW, int RV>
__global__ void __launch_bounds__(NW * 32, 2)
gdn_exact_causal_kernel(
    float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_beta_slot, int H, int HV, int W)
{
    static_assert(K % 32 == 0, "K must be warp divisible");
    constexpr int CK = K / 32;
    constexpr int NT = NW * 32;
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ float sAlpha[WMAX];
    __shared__ float sBeta[WMAX];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int rr = work / HV, i_hv = work - rr * HV;
        const int i_h = i_hv / hpg;
        const long sidx = (long)flush_list[rr];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        const float* pg = g_cache + sidx * s_g_slot + (long)i_hv * W;
        const float* pb = ls6_beta + cidx * s_beta_slot + (long)i_hv * W;

        for (int x = t; x < W * K; x += NT) sK[x] = pk[x];
        if (t < W) {
            sAlpha[t] = __expf(pg[t]);
            sBeta[t] = pb[t];
        }
        __syncthreads();

        // Dense-fallback heads already stored exact deltas online and remain
        // on the ordinary Replay16 fold.  This prototype path is latch-only.
        if (ls6_mh[i_hv] > 0) {
            float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
            float* pd = d_cache + sidx * s_d_slot + (long)i_hv * W * V;
            // Interleave RV independent rows in each warp.  A single row has
            // a shuffle-dependent dot-product chain; four rows supply enough
            // independent instructions to cover it without increasing state
            // traffic.
            #pragma unroll 1
            for (int vb = 0; vb < V; vb += NW * RV) {
                float c[RV][CK], r[RV][CK];
                #pragma unroll
                for (int q = 0; q < RV; ++q) {
                    const int v = vb + warp * RV + q;
                    #pragma unroll
                    for (int j = 0; j < CK; ++j) {
                        c[q][j] = ph[(long)v * K + lane * CK + j];
                        r[q][j] = 0.f;
                    }
                }
                #pragma unroll
                for (int s = 0; s < WMAX; ++s) {
                    if (s >= W) break;
                    const float* ks = sK + s * K + lane * CK;
                    float ck[RV];
                    #pragma unroll
                    for (int q = 0; q < RV; ++q) {
                        ck[q] = 0.f;
                        #pragma unroll
                        for (int j = 0; j < CK; ++j)
                            ck[q] = fmaf(c[q][j], ks[j], ck[q]);
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        #pragma unroll
                        for (int q = 0; q < RV; ++q)
                            ck[q] += __shfl_down_sync(
                                0xffffffffu, ck[q], off);
                    }
                    const float alpha = sAlpha[s], beta = sBeta[s];
                    #pragma unroll
                    for (int q = 0; q < RV; ++q) {
                        const int v = vb + warp * RV + q;
                        ck[q] = __shfl_sync(0xffffffffu, ck[q], 0);
                        float u = lane == 0 ? pd[(long)s * V + v] : 0.f;
                        u = __shfl_sync(0xffffffffu, u, 0);
                        const float corr = u - alpha * beta * ck[q];
                        if (lane == 0) pd[(long)s * V + v] = corr;
                        #pragma unroll
                        for (int j = 0; j < CK; ++j) {
                            const float kv = ks[j];
                            c[q][j] = alpha * fmaf(
                                -beta * ck[q], kv, c[q][j]);
                            r[q][j] = fmaf(u, kv, alpha * r[q][j]);
                        }
                    }
                }
                #pragma unroll
                for (int q = 0; q < RV; ++q) {
                    const int v = vb + warp * RV + q;
                    #pragma unroll
                    for (int j = 0; j < CK; ++j)
                        ph[(long)v * K + lane * CK + j] =
                            c[q][j] + r[q][j];
                }
            }
        }
        __syncthreads();
    }
}

// Two-pass exact refresh: tensorize only the extra checkpoint projection
// P=S0 K^T, solve the tiny causal system in FP32, and leave the well-tuned
// Replay16 outer-product fold unchanged.  Eight warps cover the complete
// 128x16 output of one value head, so no tensor-memory or cross-CTA staging is
// needed.  This path is also the performance reference for deciding whether a
// fused 2KV version hides enough work to justify its extra complexity.
template <int K, int V>
__global__ void __launch_bounds__(256, 3)
gdn_exact_wmma_kernel(
    const float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_beta_slot, int H, int HV, int W, int dbg)
{
    static_assert(K == 128 && V == 128, "WMMA path is Qwen-specialized");
    constexpr int NT = 256;
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ __align__(16) float sP[V * WMAX];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sPrefix[WMAX];
    __shared__ float sBeta[WMAX];
    const int t = threadIdx.x, warp = t >> 5;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int rr = work / HV, i_hv = work - rr * HV;
        const int i_h = i_hv / hpg;
        const long sidx = (long)flush_list[rr];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        const float* pg = g_cache + sidx * s_g_slot + (long)i_hv * W;
        const float* pb = ls6_beta + cidx * s_beta_slot + (long)i_hv * W;
        for (int x = t; x < W * K; x += NT) sK[x] = pk[x];
        const float gv = t < W ? pg[t] : 0.f;
        if (t < 32) {
            float prefix = gv;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float prev = __shfl_up_sync(0xffffffffu, prefix, off);
                if (t >= off) prefix += prev;
            }
            if (t < W) {
                sPrefix[t] = prefix;
                sBeta[t] = pb[t];
            }
        }
        __syncthreads();

        if (ls6_mh[i_hv] > 0) {
            using namespace nvcuda;
            const int vbase = warp * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
            wmma::fill_fragment(acc, 0.f);
            const float* ph = h0 + sidx * s_h0_slot
                              + (long)i_hv * s_h0_h;
            #pragma unroll
            for (int kk = 0; kk < K; kk += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::col_major> bf;
                wmma::load_matrix_sync(af, ph + (long)vbase * K + kk, K);
                wmma::load_matrix_sync(bf, sK + kk, K);
                wmma::mma_sync(acc, af, bf, acc);
            }
            wmma::store_matrix_sync(
                sP + vbase * WMAX, acc, WMAX, wmma::mem_row_major);

            // One thread owns one (j,s) pair.  Gram is shared by all V rows.
            const int j = t / WMAX, s = t - j * WMAX;
            float gram = 0.f;
            if (!(dbg & 64) && j < s && s < W) {
                #pragma unroll
                for (int kk = 0; kk < K; ++kk)
                    gram = fmaf(sK[j * K + kk], sK[s * K + kk], gram);
                gram *= __expf(sPrefix[s] - sPrefix[j]);
            }
            sGram[t] = gram;
            __syncthreads();

            if (!(dbg & 128) && t < V) {
                float hist[WMAX];
                float* pd = d_cache + sidx * s_d_slot
                            + (long)i_hv * W * V;
                #pragma unroll
                for (int s2 = 0; s2 < WMAX; ++s2) {
                    float prev = 0.f;
                    #pragma unroll
                    for (int j2 = 0; j2 < WMAX; ++j2)
                        if (j2 < s2)
                            prev = fmaf(
                                hist[j2], sBeta[s2] * sGram[j2 * WMAX + s2],
                                prev);
                    const float h = sBeta[s2] * __expf(sPrefix[s2])
                                    * sP[t * WMAX + s2] - prev;
                    hist[s2] = h;
                    pd[(long)s2 * V + t] -= h;
                }
            }
        }
        __syncthreads();
    }
}

// Fused exact refresh in normalized full-WY form.  For scalar decay define
// F_s = pi_s / gamma_s.  The state-independent recurrence is
//
//   F_s = beta_s (k_s - sum_{j<s} F_j <k_j,k_s>),
//   (gamma_W/gamma_s) d_s
//       = (gamma_W/gamma_s) u_s - gamma_W S0^T F_s.
//
// This moves the triangular dependency entirely to the KxW key factor.  A
// 16-row checkpoint tile is loaded once into shared memory, used by one warp
// for S0^T F, then reused by eight warps as the accumulator for the final
// D^T K fold.  Every checkpoint element therefore makes exactly one global
// read and one global write; both WKV products execute on tensor cores and no
// per-value triangular solve or TMEM transfer remains.
template <int K, int V>
__global__ void __launch_bounds__(256, 2)
gdn_exact_wy_fused_kernel(
    float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_beta_slot, int H, int HV, int W)
{
    static_assert(K == 128 && V == 128, "full-WY path is Qwen-specialized");
    constexpr int NT = 256;
    constexpr int VT = 16;
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ __align__(16) float sF[WMAX * K];
    // The 64-KiB checkpoint tile is dynamic so this kernel can opt in above
    // CUDA's default static-shared limit while the small operands stay static.
    extern __shared__ __align__(16) float sH[];
    __shared__ __align__(16) float sD[V * WMAX];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sPrefix[WMAX];
    __shared__ float sRep[WMAX + 1];
    __shared__ float sBeta[WMAX];
    const int t = threadIdx.x, warp = t >> 5;
    const int n_work = n_ptr[0] * HV;
    const int hpg = HV / H;

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int rr = work / HV, i_hv = work - rr * HV;
        const int i_h = i_hv / hpg;
        const long sidx = (long)flush_list[rr];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        const float* pg = g_cache + sidx * s_g_slot + (long)i_hv * W;
        const float* pb = ls6_beta + cidx * s_beta_slot + (long)i_hv * W;
        float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
        float* pd = d_cache + sidx * s_d_slot + (long)i_hv * W * V;

        for (int x = t; x < W * K; x += NT) sK[x] = pk[x];
        const float gv = t < W ? pg[t] : 0.f;
        if (t < 32) {
            float prefix = gv;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float prev = __shfl_up_sync(0xffffffffu, prefix, off);
                if (t >= off) prefix += prev;
            }
            const float total = __shfl_sync(0xffffffffu, prefix, 31);
            if (t < W) {
                sPrefix[t] = prefix;
                sRep[t] = __expf(total - prefix);
                sBeta[t] = pb[t];
            }
            if (t == 0) sRep[WMAX] = __expf(total);
        }
        __syncthreads();

        const bool latch = ls6_mh[i_hv] > 0;
        if (latch) {
            // The key Gram is shared by every value row and full-WY channel.
            const int j = t / WMAX, s = t - j * WMAX;
            float gram = 0.f;
            if (j < s && s < W) {
                #pragma unroll
                for (int kk = 0; kk < K; ++kk)
                    gram = fmaf(sK[j * K + kk], sK[s * K + kk], gram);
            }
            sGram[t] = gram;
            __syncthreads();

            // One thread owns one key channel.  Keeping its W factors in
            // registers removes all sixteen causal CTA barriers.
            if (t < K) {
                float hist[WMAX];
                #pragma unroll
                for (int s2 = 0; s2 < WMAX; ++s2) {
                    float f = sK[s2 * K + t];
                    #pragma unroll
                    for (int j2 = 0; j2 < WMAX; ++j2)
                        if (j2 < s2)
                            f = fmaf(-hist[j2], sGram[j2 * WMAX + s2], f);
                    f *= sBeta[s2];
                    hist[s2] = f;
                    sF[s2 * K + t] = f;
                }
            }
            __syncthreads();
        }

        // Stage the checkpoint once.  All eight warps then remain useful in
        // both products: warp w owns value rows [16w,16w+16).
        for (int x = t; x < V * K; x += NT) sH[x] = ph[x];
        __syncthreads();

        using namespace nvcuda;
        const int vb = warp * VT;
        if (latch) {
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> pacc;
            wmma::fill_fragment(pacc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < K; kk += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::col_major> bf;
                wmma::load_matrix_sync(af, sH + vb * K + kk, K);
                wmma::load_matrix_sync(bf, sF + kk, K);
                wmma::mma_sync(pacc, af, bf, pacc);
            }
            wmma::store_matrix_sync(
                sD + vb * WMAX, pacc, WMAX, wmma::mem_row_major);
        }
        __syncthreads();

        // Convert the raw writes to exact deltas and immediately keep the
        // replay-scaled form consumed by the second tensor product.
        for (int x = t; x < V * WMAX; x += NT) {
            const int vr = x / WMAX, s = x - vr * WMAX;
            const float u = pd[(long)s * V + vr];
            float d = u;
            if (latch) d -= __expf(sPrefix[s]) * sD[x];
            pd[(long)s * V + vr] = d;
            sD[x] = d * sRep[s];
        }
        __syncthreads();

        // Warp w keeps its value rows and walks the eight key-column tiles.
        // Every load of the initial accumulator comes from the same resident
        // shared checkpoint, never from global memory.
        const float total = sRep[WMAX];
        #pragma unroll 1
        for (int kb = 0; kb < K; kb += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> oacc;
            wmma::load_matrix_sync(
                oacc, sH + vb * K + kb, K, wmma::mem_row_major);
            #pragma unroll
            for (int x = 0; x < oacc.num_elements; ++x) oacc.x[x] *= total;
            #pragma unroll
            for (int ws = 0; ws < WMAX; ws += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::row_major> bf;
                wmma::load_matrix_sync(af, sD + vb * WMAX + ws, WMAX);
                wmma::load_matrix_sync(bf, sK + ws * K + kb, K);
                wmma::mma_sync(oacc, af, bf, oacc);
            }
            wmma::store_matrix_sync(
                ph + (long)vb * K + kb, oacc, K,
                wmma::mem_row_major);
        }
        __syncthreads();
    }
}

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

// Exact projected-f_s refresh, tiled over the value rows.  Four CTAs together
// own one [V=128,K=128] checkpoint, so every state element is still fetched
// and stored exactly once.  Compared with the full-state scalar kernel this
// reduces the live accumulator from 128 to 32 floats per thread and removes
// the low-occupancy, barrier-heavy producer/consumer split of the tcgen05
// prototype.  The small key/Gram data is duplicated across the four row tiles
// and normally hits L2; checkpoint HBM traffic remains exactly 2*K*V.
template <int K, int V, int VT=32, int TV=8>
__global__ void __launch_bounds__(TV * TK, 2)
gdn_exact_tiled_kernel(
    float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_beta_slot, int H, int HV, int W,
    int row_offset, int row_limit)
{
    static_assert(V % VT == 0 && VT % TV == 0, "value tiling");
    constexpr int NT = TV * TK;
    constexpr int KPT = K / TK;
    constexpr int VPT = VT / TV;
    constexpr int VTS = V / VT;
    __shared__ __align__(16) float sD[WMAX * VT];
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sRep[WMAX + 1];
    __shared__ float sPrefix[WMAX];
    __shared__ float sBeta[WMAX];
    const int t = threadIdx.x;
    const int tk = t % TK, tv = t / TK;
    const int k0 = tk * KPT;
    const int n_all = n_ptr[0];
    const int n_rows = n_all > row_offset
        ? min(row_limit, n_all - row_offset) : 0;
    const int n_work = n_rows * HV * VTS;
    const int hpg = HV / H;

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int row_work = work / VTS;
        const int vt = work - row_work * VTS;
        const int r_local = row_work / HV;
        const int i_hv = row_work - r_local * HV;
        const int i_h = i_hv / hpg;
        const int r = r_local + row_offset;
        const int vbase = vt * VT;
        const int v0 = vbase + tv * VPT;
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const bool exact_head = ls6_mh[i_hv] > 0;
        float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;

        float acc[VPT][KPT];
        #pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const float* row = ph + (long)(v0 + i) * K + k0;
            #pragma unroll
            for (int j = 0; j < KPT; j += 4) {
                const float4 x = *reinterpret_cast<const float4*>(row + j);
                acc[i][j] = x.x; acc[i][j + 1] = x.y;
                acc[i][j + 2] = x.z; acc[i][j + 3] = x.w;
            }
        }

        const float* pd = d_cache + sidx * s_d_slot
            + (long)i_hv * W * V;
        #pragma unroll
        for (int q = 0; q < WMAX * VT / NT; ++q) {
            const int x = t + q * NT;
            const int s = x / VT, vl = x - s * VT;
            if (s < W) sD[x] = pd[(long)s * V + vbase + vl];
        }
        const float* pk = k_cache + sidx * s_k_slot
            + (long)i_h * W * K;
        #pragma unroll
        for (int q = 0; q < WMAX * K / NT; ++q) {
            const int x = t + q * NT;
            if (x < W * K) sK[x] = pk[x];
        }

        const float gv = t < W
            ? g_cache[sidx * s_g_slot + (long)i_hv * W + t] : 0.f;
        if (t < 32) {
            float pre = gv;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, pre, o);
                if (t >= o) pre += y;
            }
            const float total = __shfl_sync(0xffffffffu, pre, 31);
            if (t < W) {
                sRep[t] = __expf(total - pre);
                sPrefix[t] = pre;
                if (exact_head)
                    sBeta[t] = ls6_beta[
                        cidx * s_beta_slot + (long)i_hv * W + t];
            }
            if (t == 0) sRep[WMAX] = __expf(total);
        }
        __syncthreads();

        if (exact_head) {
            #pragma unroll
            for (int q = 0; q < WMAX * WMAX / NT; ++q) {
                const int x = t + q * NT;
                const int j = x / WMAX, s = x - j * WMAX;
                float dot = 0.f;
                if (j < W && s < W && j < s) {
                    #pragma unroll
                    for (int kk = 0; kk < K; ++kk)
                        dot = fmaf(sK[j * K + kk], sK[s * K + kk], dot);
                    dot *= __expf(sPrefix[s] - sPrefix[j]);
                }
                sGram[x] = dot;
            }
            __syncthreads();

            // Each 16-lane subgroup owns four rows.  Its resident S0 tile is
            // reused first for P=S0*K^T and then for the final Replay16 fold.
            float hist[WMAX];
            #pragma unroll
            for (int s = 0; s < WMAX; ++s) {
                float mine = 0.f;
                if (s < W) {
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i) {
                        float dot = 0.f;
                        #pragma unroll
                        for (int j = 0; j < KPT; ++j)
                            dot = fmaf(acc[i][j], sK[s * K + k0 + j], dot);
                        #pragma unroll
                        for (int o = TK / 2; o > 0; o >>= 1)
                            dot += __shfl_xor_sync(0xffffffffu, dot, o);
                        if (tk == i) mine = dot;
                    }
                    if (tk < VPT) {
                        float prev = 0.f;
                        #pragma unroll
                        for (int j = 0; j < WMAX; ++j)
                            if (j < s) prev = fmaf(
                                hist[j], sBeta[s] * sGram[j * WMAX + s],
                                prev);
                        hist[s] = sBeta[s] * __expf(sPrefix[s]) * mine - prev;
                        const int vl = tv * VPT + tk;
                        const float corr = sD[s * VT + vl] - hist[s];
                        sD[s * VT + vl] = corr * sRep[s];
                        d_cache[sidx * s_d_slot + (long)i_hv * W * V
                                + (long)s * V + vbase + vl] = corr;
                    }
                } else if (tk < VPT) {
                    hist[s] = 0.f;
                }
            }
        } else {
            #pragma unroll
            for (int q = 0; q < WMAX * VT / NT; ++q) {
                const int x = t + q * NT;
                const int s = x / VT;
                if (s < W) sD[x] *= sRep[s];
            }
        }
        __syncthreads();

        const float total = sRep[WMAX];
        #pragma unroll
        for (int i = 0; i < VPT; ++i)
            #pragma unroll
            for (int j = 0; j < KPT; ++j) acc[i][j] *= total;
        #pragma unroll 1
        for (int s = 0; s < W; ++s) {
            float dv[VPT], kk[KPT];
            #pragma unroll
            for (int i = 0; i < VPT; ++i)
                dv[i] = sD[s * VT + tv * VPT + i];
            #pragma unroll
            for (int j = 0; j < KPT; ++j)
                kk[j] = sK[s * K + k0 + j];
            #pragma unroll
            for (int i = 0; i < VPT; ++i)
                #pragma unroll
                for (int j = 0; j < KPT; ++j)
                    acc[i][j] = fmaf(dv[i], kk[j], acc[i][j]);
        }

        #pragma unroll
        for (int i = 0; i < VPT; ++i) {
            float* row = ph + (long)(v0 + i) * K + k0;
            #pragma unroll
            for (int j = 0; j < KPT; j += 4)
                *reinterpret_cast<float4*>(row + j) = make_float4(
                    acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
        }
        __syncthreads();
    }
}

// Exact GDN refresh in coordinates of the window key span.  If Q has the
// W raw keys as columns, the normalized full-WY factor is exactly F=Q C,
// where the small upper-triangular C obeys
//
//   C[i,s] = beta_s (1[i=s] - sum_{j<s} C[i,j] <k_j,k_s>).
//
// A warp can therefore finish one value row independently:
// A=S0^T Q, P=A C, d_s=u_s-gamma_s P_s, then
// Snew^T=gamma_W S0^T + (diag(r)d)^T Q^T.  The state row is read and
// written once, but unlike the full-state resident kernel it occupies only
// four registers per lane and never spans a CTA barrier.  This development
// kernel forms C at the refresh; the production variant will consume the
// same C columns formed incrementally by the online step.
template <int K, int V, int NW=8>
__global__ void __launch_bounds__(NW * 32, 2)
gdn_exact_coord_kernel(
    float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    float* __restrict__ ls6_ubar, const float* __restrict__ ls6_zk,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_u_slot, long s_zk_slot, long s_beta_slot,
    int H, int HV, int W, int G, int row_offset, int row_limit)
{
    static_assert(K == 128 && V == 128 && NW == 8,
                  "coordinate path is specialized for Qwen GDN");
    constexpr int CK = K / 32;
    constexpr int NT = NW * 32;
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sC[WMAX * WMAX];
    __shared__ float sPrefix[WMAX];
    __shared__ float sRep[WMAX + 1];
    __shared__ float sBeta[WMAX];
    __shared__ __align__(16) float sZk[WMAX * GMAX];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int n_all = n_ptr[0];
    const int n_rows = n_all > row_offset
        ? min(row_limit, n_all - row_offset) : 0;
    const int n_work = n_rows * HV;
    const int hpg = HV / H;

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int r_local = work / HV;
        const int i_hv = work - r_local * HV;
        const int i_h = i_hv / hpg;
        const int r = r_local + row_offset;
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const bool latch = ls6_mh[i_hv] > 0;
        const int mh = latch ? ls6_mh[i_hv] : 0;
        const float* pk = k_cache + sidx * s_k_slot
                          + (long)i_h * W * K;
        const float* pg = g_cache + sidx * s_g_slot
                          + (long)i_hv * W;
        const float* pb = ls6_beta + cidx * s_beta_slot
                          + (long)i_hv * W;
        float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
        float* pd = d_cache + sidx * s_d_slot + (long)i_hv * W * V;

        for (int x = t; x < W * K; x += NT) sK[x] = pk[x];
        if (latch && ls6_zk != nullptr) {
            const float* pz = ls6_zk + cidx * s_zk_slot
                              + (long)i_h * W * G;
            for (int x = t; x < W * G; x += NT) sZk[x] = pz[x];
        }
        const float gv = t < W ? pg[t] : 0.f;
        if (t < 32) {
            float prefix = gv;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float prev = __shfl_up_sync(
                    0xffffffffu, prefix, off);
                if (t >= off) prefix += prev;
            }
            const float total = __shfl_sync(0xffffffffu, prefix, WMAX - 1);
            if (t < W) {
                sPrefix[t] = prefix;
                sRep[t] = __expf(total - prefix);
                sBeta[t] = pb[t];
            }
            if (t == 0) sRep[WMAX] = __expf(total);
        }
        __syncthreads();

        if (latch) {
            // One tensor-core warp forms the key Gram.  C itself is a tiny
            // scalar recurrence and has no K-dimensional dependency.
            if (t < 32) {
                using namespace nvcuda;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
                wmma::fill_fragment(acc, 0.f);
                #pragma unroll
                for (int kk = 0; kk < K; kk += 8) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 8,
                                   wmma::precision::tf32,
                                   wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                   wmma::precision::tf32,
                                   wmma::col_major> bf;
                    wmma::load_matrix_sync(af, sK + kk, K);
                    wmma::load_matrix_sync(bf, sK + kk, K);
                    wmma::mma_sync(acc, af, bf, acc);
                }
                wmma::store_matrix_sync(
                    sGram, acc, WMAX, wmma::mem_row_major);
            }
            __syncthreads();
            if (t < WMAX) {
                const int i = t;
                float hist[WMAX];
                #pragma unroll
                for (int s = 0; s < WMAX; ++s) {
                    float c = i == s ? 1.f : 0.f;
                    if (i <= s) {
                        #pragma unroll
                        for (int j = 0; j < WMAX; ++j)
                            if (j < s)
                                c = fmaf(-hist[j], sGram[j * WMAX + s], c);
                        c *= sBeta[s];
                    } else {
                        c = 0.f;
                    }
                    hist[s] = c;
                    sC[i * WMAX + s] = c;
                }
            }
        }
        __syncthreads();

        const float total = sRep[WMAX];
        #pragma unroll 1
        for (int v = warp; v < V; v += NW) {
            float h[CK];
            #pragma unroll
            for (int q = 0; q < CK; ++q)
                h[q] = ph[(long)v * K + lane * CK + q];

            float p_lane = 0.f;
            if (latch) {
                float a[WMAX];
                #pragma unroll
                for (int s = 0; s < WMAX; ++s) {
                    float x = 0.f;
                    #pragma unroll
                    for (int q = 0; q < CK; ++q)
                        x = fmaf(h[q], sK[s * K + lane * CK + q], x);
                    a[s] = x;
                }
                const float a_lane = transpose_reduce<WMAX>(a, lane);
                float pc[WMAX];
                #pragma unroll
                for (int s = 0; s < WMAX; ++s)
                    pc[s] = lane < WMAX
                        ? a_lane * sC[lane * WMAX + s] : 0.f;
                p_lane = transpose_reduce<WMAX>(pc, lane);
            }

            float d_lane = 0.f;
            if (lane < WMAX) {
                const int s = lane;
                const float raw = pd[(long)s * V + v];
                const float corr = latch
                    ? raw - __expf(sPrefix[s]) * p_lane : raw;
                if (latch) pd[(long)s * V + v] = corr;
                d_lane = corr * sRep[s];
            }

            #pragma unroll
            for (int q = 0; q < CK; ++q) {
                float out = total * h[q];
                #pragma unroll
                for (int s = 0; s < WMAX; ++s) {
                    const float ds = __shfl_sync(0xffffffffu, d_lane, s);
                    out = fmaf(ds, sK[s * K + lane * CK + q], out);
                }
                ph[(long)v * K + lane * CK + q] = out;
            }

            // The exact corrected deltas also update the next window's
            // latches.  All lanes execute the shuffle before inactive G
            // columns are predicated away.
            if (latch && ls6_ubar != nullptr) {
                float* pu = ls6_ubar + cidx * s_u_slot
                            + (long)i_hv * G * V;
                for (int gb = 0; gb < GMAX; gb += 32) {
                    if (gb >= G) break;
                    const int g = gb + lane;
                    const bool active = g < mh;
                    float u = active ? total * pu[(long)g * V + v] : 0.f;
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s) {
                        const float ds = __shfl_sync(
                            0xffffffffu, d_lane, s);
                        if (active)
                            u = fmaf(sZk[s * G + g], ds, u);
                    }
                    if (active) pu[(long)g * V + v] = u;
                }
            }
        }
        __syncthreads();
    }
}

// Exact block-WY refresh with the dependency ordered in value space.
// For H=S^T and Q=[k_0,...,k_{W-1}], form P=H0 Q once, then
//
//   Y_s = beta_s (P_s - sum_{j<s} Y_j <k_j,k_s>),
//   X_s = (gamma/gamma_s) u_s - gamma Y_s,
//   H_W = gamma H0 + X Q^T.
//
// This is the same normalized full-WY identity as the direct kernel, but it
// never materializes the KxW factor F, never writes corrected deltas back to
// global memory, and leaves the only W-step dependency in the tiny VxW tile.
// One CTA owns a complete GQA key-head group.  It shares the raw key tile and
// Gram across its three value heads, stages each checkpoint exactly once, and
// delays the register-resident Replay16 fold until the triangular temporaries
// are dead.  With 256 threads this leaves only 64 state accumulators/thread.
template <int K, int V>
__global__ void __launch_bounds__(256, 2)
gdn_exact_block_wy_kernel(
    float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    float* __restrict__ ls6_ubar, const float* __restrict__ ls6_zk,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_u_slot, long s_zk_slot, long s_beta_slot,
    int H, int HV, int W, int G, int row_offset, int row_limit, int dbg)
{
    static_assert(K == 128 && V == 128, "block-WY path is Qwen-specialized");
    constexpr int NT = 256, TV = 16, KPT = K / 16, VPT = V / TV;
    __shared__ __align__(1024) float sK[WMAX * K];       // tcgen swizzle
    __shared__ __align__(16) float sRawK[WMAX * K];     // row-major fold
    __shared__ __align__(16) float sD[WMAX * V];        // X, s-major
    __shared__ __align__(16) float sZk[WMAX * GMAX];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sRep[WMAX + 1], sPrefix[WMAX], sBeta[WMAX];
    __shared__ __align__(8) unsigned long long sMmaBar;
    __shared__ unsigned sTaddr[4];
    extern __shared__ __align__(1024) float sH[];        // swizzled VxK

    const int t = threadIdx.x, tk = t & 15, tv = t >> 4;
    const int k0 = tk * KPT, v0 = tv * VPT;
    const int n_all = n_ptr[0];
    const int n_rows = n_all > row_offset
        ? min(row_limit, n_all - row_offset) : 0;
    const int hpg = HV / H;
    const int n_work = n_rows * H;
    if (t < 32) tc_alloc(sTaddr);
    __syncthreads();

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int r_local = work / H;
        const int i_h = work - r_local * H;
        const int r = r_local + row_offset;
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;

        // Key traffic and its state-independent Gram are shared by all three
        // GQA value heads.  Keep both tcgen and ordinary row-major layouts;
        // they are only 8 KiB apiece and remove a later repack/barrier.
        for (int x = t; x < W * K; x += NT) {
            const int s = x / K, k = x - s * K;
            const float kv = pk[x];
            sRawK[x] = kv;
            sK[tc_swizzle<WMAX>(s, k)] = kv;
        }
        bool any_latch = false;
        #pragma unroll
        for (int sub = 0; sub < 3; ++sub)
            any_latch |= sub < hpg && ls6_mh[i_h * hpg + sub] > 0;
        __syncthreads();

        if (any_latch && t < 32) {
            using namespace nvcuda;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> gacc;
            wmma::fill_fragment(gacc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < K; kk += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::row_major> ga;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::col_major> gb;
                wmma::load_matrix_sync(ga, sRawK + kk, K);
                wmma::load_matrix_sync(gb, sRawK + kk, K);
                if (!(dbg & 1024)) wmma::mma_sync(gacc, ga, gb, gacc);
            }
            wmma::store_matrix_sync(
                sGram, gacc, WMAX, wmma::mem_row_major);
        }
        if (any_latch && ls6_ubar != nullptr && ls6_zk != nullptr) {
            const float* pz = ls6_zk + cidx * s_zk_slot
                              + (long)i_h * W * G;
            for (int x = t; x < W * G; x += NT) sZk[x] = pz[x];
        }
        __syncthreads();

        #pragma unroll 1
        for (int sub = 0; sub < 3; ++sub) {
            if (sub >= hpg) break;
            const int i_hv = i_h * hpg + sub;
            const bool latch = ls6_mh[i_hv] > 0;
            const bool do_exact = latch && !(dbg & 2048);
            float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
            const float* pd = d_cache + sidx * s_d_slot
                              + (long)i_hv * W * V;
            const float* pg = g_cache + sidx * s_g_slot
                              + (long)i_hv * W;
            const float* pb = ls6_beta + cidx * s_beta_slot
                              + (long)i_hv * W;

            const bool wmma_projection = (dbg & 1048576) != 0;
            for (int x = t; x < V * K; x += NT) {
                const int v = x / K, k = x - v * K;
                if (wmma_projection) sH[x] = ph[x];
                else sH[tc_swizzle<V>(v, k)] = ph[x];
            }
            for (int x = t; x < W * V; x += NT) sD[x] = pd[x];
            const float gv = t < W ? pg[t] : 0.f;
            if (t < 32) {
                float prefix = gv;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const float prev = __shfl_up_sync(
                        0xffffffffu, prefix, off);
                    if (t >= off) prefix += prev;
                }
                const float total = __shfl_sync(
                    0xffffffffu, prefix, WMAX - 1);
                if (t < W) {
                    sPrefix[t] = prefix;
                    sRep[t] = __expf(total - prefix);
                    sBeta[t] = pb[t];
                }
                if (t == 0) {
                    sRep[WMAX] = __expf(total);
                    if (latch) {
                        asm volatile(
                            "mbarrier.init.shared::cta.b64 [%0], %1;"
                            :: "r"(tc_smem_u32(&sMmaBar)), "r"(1));
                        asm volatile(
                            "fence.mbarrier_init.release.cluster;"
                            ::: "memory");
                    }
                }
            }
            __syncthreads();

            const float total = sRep[WMAX];
            if (latch) {
                // Native tensor product P=H0 Q.  H0 is already resident in
                // shared memory and will later seed the ordinary scalar fold.
                if (wmma_projection) {
                    using namespace nvcuda;
                    const int warp = t >> 5;
                    const int vb = warp * 16;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> pacc;
                    wmma::fill_fragment(pacc, 0.f);
                    #pragma unroll
                    for (int kk = 0; kk < K; kk += 8) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 8,
                                       wmma::precision::tf32,
                                       wmma::row_major> pa;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                       wmma::precision::tf32,
                                       wmma::col_major> pbm;
                        wmma::load_matrix_sync(pa, sH + vb * K + kk, K);
                        wmma::load_matrix_sync(pbm, sRawK + kk, K);
                        wmma::mma_sync(pacc, pa, pbm, pacc);
                    }
                    wmma::store_matrix_sync(
                        sD + vb * WMAX, pacc, WMAX,
                        wmma::mem_row_major);
                } else if (t == 0) {
                    constexpr unsigned idesc =
                        (1u << 4) | (2u << 7) | (2u << 10)
                        | ((WMAX >> 3) << 17) | ((V >> 4) << 24);
                    const unsigned long long ah = tc_sdesc(sH, V * 128);
                    const unsigned long long bk = tc_sdesc(sK, V * 128);
                    #pragma unroll
                    for (int kb = 0; kb < K / 32; ++kb) {
                        #pragma unroll
                        for (int ki = 0; ki < 32 / 8; ++ki) {
                            const unsigned long long ad = ah
                                + (unsigned long long)(
                                    (kb * V * 128 + ki * 32) >> 4);
                            const unsigned long long bd = bk
                                + (unsigned long long)(
                                    (kb * WMAX * 128 + ki * 32) >> 4);
                            tc_mma_tf32(
                                sTaddr[0], ad, bd, idesc,
                                kb != 0 || ki != 0);
                        }
                    }
                    tc_commit(&sMmaBar);
                    tc_mbar_wait(&sMmaBar, 0);
                }
                __syncthreads();
                if (!wmma_projection) tc_fence_after_sync();

                // One thread owns one value row.  This applies C implicitly
                // after H0Q, so no K-channel WY factor is ever live.
                float xrow[WMAX];
                if (t < V) {
                    float p[WMAX], yhist[WMAX];
                    if (wmma_projection) {
                        #pragma unroll
                        for (int s = 0; s < WMAX; ++s)
                            p[s] = sD[t * WMAX + s];
                    } else {
                        tc_load16(sTaddr[0], (t / 32) * 32, p);
                    }
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s) {
                        if (dbg & 512)
                            d_cache[sidx * s_d_slot
                                    + (long)i_hv * W * V
                                    + (long)s * V + t] = p[s];
                        float y = p[s];
                        #pragma unroll
                        for (int j = 0; j < WMAX; ++j)
                            if (j < s)
                                y = fmaf(-yhist[j],
                                         sGram[j * WMAX + s], y);
                        y *= sBeta[s];
                        yhist[s] = y;
                        xrow[s] = sRep[s] * pd[(long)s * V + t]
                                  - total * y;
                    }
                }
                __syncthreads();
                if (t < V) {
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s)
                        sD[s * V + t] = xrow[s];
                }
            } else {
                // Dense fallback stored exact deltas online; it needs only
                // the standard Replay16 decay weighting.
                for (int x = t; x < W * V; x += NT) {
                    const int s = x / V;
                    sD[x] *= sRep[s];
                }
            }
            __syncthreads();

            // Mature scalar outer-product fold.  Exact preparation is over,
            // so only 64 long-lived state values remain in each thread.
            {
                float acc[VPT][KPT];
                #pragma unroll
                for (int i = 0; i < VPT; ++i)
                    #pragma unroll
                    for (int j = 0; j < KPT; ++j)
                        acc[i][j] = total * (wmma_projection
                            ? sH[(v0 + i) * K + k0 + j]
                            : sH[tc_swizzle<V>(v0 + i, k0 + j)]);
                #pragma unroll 1
                for (int s = 0; s < WMAX; ++s) {
                    float dv[VPT], kv[KPT];
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i)
                        dv[i] = sD[s * V + v0 + i];
                    #pragma unroll
                    for (int j = 0; j < KPT; ++j)
                        kv[j] = sRawK[s * K + k0 + j];
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i)
                        #pragma unroll
                        for (int j = 0; j < KPT; ++j)
                            acc[i][j] = fmaf(dv[i], kv[j], acc[i][j]);
                }
                #pragma unroll
                for (int i = 0; i < VPT; ++i) {
                    float* row = ph + (long)(v0 + i) * K + k0;
                    #pragma unroll
                    for (int j = 0; j < KPT; j += 4)
                        *reinterpret_cast<float4*>(row + j) = make_float4(
                            acc[i][j], acc[i][j + 1],
                            acc[i][j + 2], acc[i][j + 3]);
                }
            }
            __syncthreads();

            // Ubar_W = gamma Ubar_0 + X (Q^T Z).  X is still resident in sD;
            // no corrected-delta round trip is needed.
            if (latch && ls6_ubar != nullptr && ls6_zk != nullptr) {
                float* pu = ls6_ubar + cidx * s_u_slot
                            + (long)i_hv * G * V;
                const int mh = ls6_mh[i_hv];
                for (int x = t; x < mh * V; x += NT) {
                    const int g = x / V, v = x - g * V;
                    float u = total * pu[x];
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s)
                        u = fmaf(sZk[s * G + g], sD[s * V + v], u);
                    pu[x] = u;
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (t < 32) tc_dealloc(sTaddr[0]);
}

// Warp-tiled realization of the same block-WY identity.  Each of eight warps
// owns a 16x128 value tile.  Its sole global checkpoint load simultaneously
// fills (1) the 64 scalar Replay16 accumulators per thread and (2) the WMMA
// operand in shared memory.  Thus H0 never makes the costly shared->register
// round trip of gdn_exact_block_wy_kernel, while the exact temporary is only a
// 16-element row recurrence alongside the half-sized state accumulator.
template <int K, int V>
__global__ void __launch_bounds__(256, 2)
gdn_exact_tile_wy_kernel(
    float* __restrict__ h0, const float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    float* __restrict__ ls6_ubar, const float* __restrict__ ls6_zk,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_u_slot, long s_zk_slot, long s_beta_slot,
    int H, int HV, int W, int G, int row_offset, int row_limit, int dbg)
{
    static_assert(K == 128 && V == 128, "tile-WY path is Qwen-specialized");
    constexpr int NT = 256, TV = 16, TK2 = 16;
    constexpr int VPT = V / TV, KPT = K / TK2;
    __shared__ __align__(16) float sK[WMAX * K];
    __shared__ __align__(16) float sD[WMAX * V];
    __shared__ __align__(16) float sZk[WMAX * GMAX];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sRep[WMAX + 1], sBeta[WMAX];
    extern __shared__ __align__(16) float sH[];

    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int tk = t & 15, tv = t >> 4;
    const int v0 = tv * VPT, k0 = tk * KPT;
    const int n_all = n_ptr[0];
    const int n_rows = n_all > row_offset
        ? min(row_limit, n_all - row_offset) : 0;
    const int hpg = HV / H;
    const int n_work = n_rows * H;

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int r_local = work / H;
        const int i_h = work - r_local * H;
        const int r = r_local + row_offset;
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        for (int x = t; x < W * K; x += NT) sK[x] = pk[x];
        bool any_latch = false;
        #pragma unroll
        for (int sub = 0; sub < 3; ++sub)
            any_latch |= sub < hpg && ls6_mh[i_h * hpg + sub] > 0;
        __syncthreads();

        if (any_latch && t < 32) {
            using namespace nvcuda;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> gacc;
            wmma::fill_fragment(gacc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < K; kk += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::row_major> ga;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::col_major> gb;
                wmma::load_matrix_sync(ga, sK + kk, K);
                wmma::load_matrix_sync(gb, sK + kk, K);
                wmma::mma_sync(gacc, ga, gb, gacc);
            }
            wmma::store_matrix_sync(
                sGram, gacc, WMAX, wmma::mem_row_major);
        }
        if (any_latch && ls6_ubar != nullptr && ls6_zk != nullptr) {
            const float* pz = ls6_zk + cidx * s_zk_slot
                              + (long)i_h * W * G;
            for (int x = t; x < W * G; x += NT) sZk[x] = pz[x];
        }
        __syncthreads();

        #pragma unroll 1
        for (int sub = 0; sub < 3; ++sub) {
            if (sub >= hpg) break;
            const int i_hv = i_h * hpg + sub;
            const bool latch = ls6_mh[i_hv] > 0;
            const bool do_exact = latch && !(dbg & 2048);
            float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
            const float* pd = d_cache + sidx * s_d_slot
                              + (long)i_hv * W * V;
            const float* pg = g_cache + sidx * s_g_slot
                              + (long)i_hv * W;
            const float* pb = ls6_beta + cidx * s_beta_slot
                              + (long)i_hv * W;

            {
                float acc[VPT][KPT];
                #pragma unroll
                for (int i = 0; i < VPT; ++i) {
                    const float* row = ph + (long)(v0 + i) * K + k0;
                    #pragma unroll
                    for (int j = 0; j < KPT; j += 4) {
                        const float4 x = *reinterpret_cast<const float4*>(
                            row + j);
                        acc[i][j] = x.x; acc[i][j + 1] = x.y;
                        acc[i][j + 2] = x.z; acc[i][j + 3] = x.w;
                        if (do_exact) {
                            *reinterpret_cast<float4*>(
                                sH + (v0 + i) * K + k0 + j) = x;
                        }
                    }
                }
                const float gv = t < W ? pg[t] : 0.f;
                if (t < 32) {
                    float prefix = gv;
                    #pragma unroll
                    for (int off = 1; off < 32; off <<= 1) {
                        const float prev = __shfl_up_sync(
                            0xffffffffu, prefix, off);
                        if (t >= off) prefix += prev;
                    }
                    const float total_log = __shfl_sync(
                        0xffffffffu, prefix, WMAX - 1);
                    if (t < W) {
                        sRep[t] = __expf(total_log - prefix);
                        sBeta[t] = pb[t];
                    }
                    if (t == 0) sRep[WMAX] = __expf(total_log);
                }
                __syncthreads();

                const float total = sRep[WMAX];
                if (do_exact) {
                    using namespace nvcuda;
                    const int vb = warp * 16;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> pacc;
                    wmma::fill_fragment(pacc, 0.f);
                    #pragma unroll
                    for (int kk = 0; kk < K; kk += 8) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 8,
                                       wmma::precision::tf32,
                                       wmma::row_major> pa;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                       wmma::precision::tf32,
                                       wmma::col_major> pbm;
                        wmma::load_matrix_sync(pa, sH + vb * K + kk, K);
                        wmma::load_matrix_sync(pbm, sK + kk, K);
                        if (!(dbg & 4096))
                            wmma::mma_sync(pacc, pa, pbm, pacc);
                    }
                    wmma::store_matrix_sync(
                        sD + vb * WMAX, pacc, WMAX,
                        wmma::mem_row_major);
                    __syncthreads();

                    if (lane < 16) {
                        const int v = vb + lane;
                        #pragma unroll
                        for (int s = 0; s < WMAX; ++s) {
                            float y = 0.f;
                            if (!(dbg & 4194304)) {
                                y = sD[v * WMAX + s];
                                #pragma unroll
                                for (int j = 0; j < WMAX; ++j)
                                    if (j < s)
                                        y = fmaf(
                                            -sD[v * WMAX + j],
                                            sGram[j * WMAX + s], y);
                                y *= sBeta[s];
                            }
                            sD[v * WMAX + s] = y;
                        }
                    }
                    __syncthreads();
                    // The checkpoint copy in sH is dead after H0Q.  Reuse its
                    // first W*V entries for transposed/scaled X, avoiding a
                    // second per-thread W-vector and an in-place transpose.
                    for (int x = t; x < W * V; x += NT) {
                        const int s = x / V, v = x - s * V;
                        sH[x] = sRep[s] * pd[x]
                                - total * sD[v * WMAX + s];
                    }
                } else {
                    for (int x = t; x < W * V; x += NT) {
                        const int s = x / V;
                        sH[x] = sRep[s] * pd[x];
                    }
                }
                __syncthreads();

                #pragma unroll
                for (int i = 0; i < VPT; ++i)
                    #pragma unroll
                    for (int j = 0; j < KPT; ++j)
                        acc[i][j] *= total;
                #pragma unroll 1
                for (int s = 0; s < ((dbg & 2) ? 0 : WMAX); ++s) {
                    float dv[VPT], kv[KPT];
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i)
                        dv[i] = sH[s * V + v0 + i];
                    #pragma unroll
                    for (int j = 0; j < KPT; ++j)
                        kv[j] = sK[s * K + k0 + j];
                    #pragma unroll
                    for (int i = 0; i < VPT; ++i)
                        #pragma unroll
                        for (int j = 0; j < KPT; ++j)
                            acc[i][j] = fmaf(dv[i], kv[j], acc[i][j]);
                }
                #pragma unroll
                for (int i = 0; i < VPT; ++i) {
                    if (dbg & 4) break;
                    float* row = ph + (long)(v0 + i) * K + k0;
                    #pragma unroll
                    for (int j = 0; j < KPT; j += 4)
                        *reinterpret_cast<float4*>(row + j) = make_float4(
                            acc[i][j], acc[i][j + 1],
                            acc[i][j + 2], acc[i][j + 3]);
                }
            }
            __syncthreads();

            if (latch && ls6_ubar != nullptr && ls6_zk != nullptr) {
                float* pu = ls6_ubar + cidx * s_u_slot
                            + (long)i_hv * G * V;
                const int mh = ls6_mh[i_hv];
                for (int x = t; x < mh * V; x += NT) {
                    const int g = x / V, v = x - g * V;
                    float u = sRep[WMAX] * pu[x];
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s)
                        u = fmaf(sZk[s * G + g], sH[s * V + v], u);
                    pu[x] = u;
                }
            }
            __syncthreads();
        }
    }
}

// Production candidate for the value-tiled block-WY ordering.  This keeps
// the checkpoint in 64 scalar accumulators/thread, streams those same values
// through two 128x32 shared panels for native tcgen05 P=H0 Q, and performs
// the triangular Y recurrence in-place in the 16 returned registers.  It is
// the working tcgen layout from gdn_flush_kernel with half its live state and
// without full-F materialization or a corrected-delta global round trip.
template <int K, int V, int TV=16>
__global__ void __launch_bounds__(TV * 16, 2)
gdn_exact_tile_tc_wy_kernel(
    float* __restrict__ h0, const float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    float* __restrict__ ls6_ubar, const float* __restrict__ ls6_zk,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_u_slot, long s_zk_slot, long s_beta_slot,
    int H, int HV, int W, int G, int row_offset, int row_limit)
{
    static_assert(K == 128 && V == 128, "tile tc-WY is Qwen-specialized");
    constexpr int TK2 = 16, NT = TV * TK2;
    constexpr int VPT = V / TV, KPT = K / TK2;
    static_assert(TV == 16 || TV == 32, "TV must be 16 or 32");
    __shared__ __align__(1024) float sK[WMAX * K];
    __shared__ __align__(16) float sRawK[WMAX * K];
    __shared__ __align__(16) float sD[WMAX * V];
    __shared__ __align__(16) float sZk[WMAX * GMAX];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sRep[WMAX + 1], sBeta[WMAX];
    __shared__ __align__(8) unsigned long long sMmaBar;
    __shared__ unsigned sTaddr[4];
    extern __shared__ __align__(1024) float sHtc[]; // two [V,32] panels

    const int t = threadIdx.x, tk = t & 15, tv = t >> 4;
    const int k0 = tk * KPT, v0 = tv * VPT;
    const int n_all = n_ptr[0];
    const int n_rows = n_all > row_offset
        ? min(row_limit, n_all - row_offset) : 0;
    const int hpg = HV / H;
    const int n_work = n_rows * H;
    if (t < 32) tc_alloc(sTaddr);
    __syncthreads();

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int r_local = work / H;
        const int i_h = work - r_local * H;
        const int r = r_local + row_offset;
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        for (int x = t; x < W * K; x += NT) {
            const int s = x / K, k = x - s * K;
            const float kv = pk[x];
            sRawK[x] = kv;
            sK[tc_swizzle<WMAX>(s, k)] = kv;
        }
        bool any_latch = false;
        #pragma unroll
        for (int sub = 0; sub < 3; ++sub)
            any_latch |= sub < hpg && ls6_mh[i_h * hpg + sub] > 0;
        __syncthreads();

        if (any_latch && t < 32) {
            using namespace nvcuda;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> gacc;
            wmma::fill_fragment(gacc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < K; kk += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::row_major> ga;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32,
                               wmma::col_major> gb;
                wmma::load_matrix_sync(ga, sRawK + kk, K);
                wmma::load_matrix_sync(gb, sRawK + kk, K);
                wmma::mma_sync(gacc, ga, gb, gacc);
            }
            wmma::store_matrix_sync(
                sGram, gacc, WMAX, wmma::mem_row_major);
        }
        if (any_latch && ls6_ubar != nullptr && ls6_zk != nullptr) {
            const float* pz = ls6_zk + cidx * s_zk_slot
                              + (long)i_h * W * G;
            for (int x = t; x < W * G; x += NT) sZk[x] = pz[x];
        }
        __syncthreads();

        #pragma unroll 1
        for (int sub = 0; sub < 3; ++sub) {
            if (sub >= hpg) break;
            const int i_hv = i_h * hpg + sub;
            const bool latch = ls6_mh[i_hv] > 0;
            float* ph = h0 + sidx * s_h0_slot + (long)i_hv * s_h0_h;
            const float* pd = d_cache + sidx * s_d_slot
                              + (long)i_hv * W * V;
            const float* pg = g_cache + sidx * s_g_slot
                              + (long)i_hv * W;
            const float* pb = ls6_beta + cidx * s_beta_slot
                              + (long)i_hv * W;
            float acc[VPT][KPT];
            #pragma unroll
            for (int i = 0; i < VPT; ++i) {
                const float* row = ph + (long)(v0 + i) * K + k0;
                #pragma unroll
                for (int j = 0; j < KPT; j += 4) {
                    const float4 x = *reinterpret_cast<const float4*>(row + j);
                    acc[i][j] = x.x; acc[i][j + 1] = x.y;
                    acc[i][j + 2] = x.z; acc[i][j + 3] = x.w;
                }
            }
            const float gv = t < W ? pg[t] : 0.f;
            if (t < 32) {
                float prefix = gv;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const float prev = __shfl_up_sync(
                        0xffffffffu, prefix, off);
                    if (t >= off) prefix += prev;
                }
                const float total_log = __shfl_sync(
                    0xffffffffu, prefix, WMAX - 1);
                if (t < W) {
                    sRep[t] = __expf(total_log - prefix);
                    sBeta[t] = pb[t];
                }
                if (t == 0) {
                    sRep[WMAX] = __expf(total_log);
                    if (latch) {
                        asm volatile(
                            "mbarrier.init.shared::cta.b64 [%0], %1;"
                            :: "r"(tc_smem_u32(&sMmaBar)), "r"(1));
                        asm volatile(
                            "fence.mbarrier_init.release.cluster;"
                            ::: "memory");
                    }
                }
            }
            __syncthreads();

            if (latch) {
                constexpr unsigned idesc =
                    (1u << 4) | (2u << 7) | (2u << 10)
                    | ((WMAX >> 3) << 17) | ((V >> 4) << 24);
                const unsigned long long bk = tc_sdesc(sK, V * 128);
                #pragma unroll
                for (int kp = 0; kp < K / 64; ++kp) {
                    const int own_kb = k0 / 32;
                    if (own_kb / 2 == kp) {
                        const int stage = own_kb & 1;
                        #pragma unroll
                        for (int i = 0; i < VPT; ++i) {
                            #pragma unroll
                            for (int j = 0; j < KPT; j += 4) {
                                *reinterpret_cast<float4*>(
                                    sHtc + stage * V * 32
                                    + tc_swizzle<V>(
                                        v0 + i,
                                        k0 - own_kb * 32 + j))
                                    = make_float4(
                                        acc[i][j], acc[i][j + 1],
                                        acc[i][j + 2], acc[i][j + 3]);
                            }
                        }
                    }
                    __syncthreads();
                    if (t == 0) {
                        #pragma unroll
                        for (int stage = 0; stage < 2; ++stage) {
                            const int kb = kp * 2 + stage;
                            const unsigned long long ah = tc_sdesc(
                                sHtc + stage * V * 32, V * 128);
                            #pragma unroll
                            for (int ki = 0; ki < 32 / 8; ++ki) {
                                const unsigned long long ad = ah
                                    + (unsigned long long)((ki * 32) >> 4);
                                const unsigned long long bd = bk
                                    + (unsigned long long)(
                                        (kb * WMAX * 128 + ki * 32) >> 4);
                                tc_mma_tf32(
                                    sTaddr[0], ad, bd, idesc,
                                    kb != 0 || ki != 0);
                            }
                        }
                        tc_commit(&sMmaBar);
                        tc_mbar_wait(&sMmaBar, kp & 1);
                    }
                    __syncthreads();
                }
                tc_fence_after_sync();
                if (t < V) {
                    float p[WMAX];
                    tc_load16(sTaddr[0], (t / 32) * 32, p);
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s) {
                        float y = p[s];
                        #pragma unroll
                        for (int j = 0; j < WMAX; ++j)
                            if (j < s)
                                y = fmaf(-p[j],
                                         sGram[j * WMAX + s], y);
                        y *= sBeta[s];
                        p[s] = y;
                        sD[s * V + t] = sRep[s] * pd[(long)s * V + t]
                                          - sRep[WMAX] * y;
                    }
                }
            } else {
                for (int x = t; x < W * V; x += NT) {
                    const int s = x / V;
                    sD[x] = sRep[s] * pd[x];
                }
            }
            __syncthreads();

            const float total = sRep[WMAX];
            #pragma unroll
            for (int i = 0; i < VPT; ++i)
                #pragma unroll
                for (int j = 0; j < KPT; ++j) acc[i][j] *= total;
            #pragma unroll 1
            for (int s = 0; s < WMAX; ++s) {
                float dv[VPT], kv[KPT];
                #pragma unroll
                for (int i = 0; i < VPT; ++i)
                    dv[i] = sD[s * V + v0 + i];
                #pragma unroll
                for (int j = 0; j < KPT; ++j)
                    kv[j] = sRawK[s * K + k0 + j];
                #pragma unroll
                for (int i = 0; i < VPT; ++i)
                    #pragma unroll
                    for (int j = 0; j < KPT; ++j)
                        acc[i][j] = fmaf(dv[i], kv[j], acc[i][j]);
            }
            #pragma unroll
            for (int i = 0; i < VPT; ++i) {
                float* row = ph + (long)(v0 + i) * K + k0;
                #pragma unroll
                for (int j = 0; j < KPT; j += 4)
                    *reinterpret_cast<float4*>(row + j) = make_float4(
                        acc[i][j], acc[i][j + 1],
                        acc[i][j + 2], acc[i][j + 3]);
            }
            __syncthreads();

            if (latch && ls6_ubar != nullptr && ls6_zk != nullptr) {
                float* pu = ls6_ubar + cidx * s_u_slot
                            + (long)i_hv * G * V;
                const int mh = ls6_mh[i_hv];
                for (int x = t; x < mh * V; x += NT) {
                    const int g = x / V, v = x - g * V;
                    float u = total * pu[x];
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s)
                        u = fmaf(sZk[s * G + g], sD[s * V + v], u);
                    pu[x] = u;
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (t < 32) tc_dealloc(sTaddr[0]);
}

// Stream the exact block-WY refresh in independent 64-value-row tiles.  The
// mature Replay16 kernel above keeps the complete 128x128 checkpoint in
// registers and uses scalar outer products.  That leaves only eight resident
// warps and serializes the extra S0*Q product with the final fold.  Here the
// checkpoint is still read and written exactly once, but both rank-16
// products are native tcgen05 MMAs:
//
//   P = H0 Q,
//   Y_s = beta_s(P_s - sum_{j<s}Y_j <k_j,k_s>),
//   X_s = r_s u_s - gamma Y_s,
//   H_W = gamma H0 + X Q^T.
//
// One CTA owns a (request,key-head), reuses Q/Q^T/Gram across its GQA value
// heads, and streams two M=64 row tiles per head.  The small tile keeps the
// state out of the register file; the only large traffic remains one H0 load
// and one H_W store.
template <int K, int V, int M=64>
__global__ void __launch_bounds__(128, 2)
gdn_exact_stream_tc_wy_kernel(
    float* __restrict__ h0, float* __restrict__ d_cache,
    const float* __restrict__ k_cache, const float* __restrict__ g_cache,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    float* __restrict__ ls6_ubar, const float* __restrict__ ls6_zk,
    const int* __restrict__ ls6_mh, const float* __restrict__ ls6_beta,
    const int* __restrict__ ls6_map,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot,
    long s_g_slot, long s_u_slot, long s_zk_slot, long s_beta_slot,
    int H, int HV, int W, int G, int row_offset, int row_limit, int dbg)
{
    static_assert(K == 128 && V == 128 && M == 64,
                  "stream tc-WY is Qwen-specialized");
    constexpr int NT = 128;
    // Shared tensor operands all use the native 128-B swizzle.  Q is kept in
    // both orientations so it is staged once per key head, not once per row
    // tile.  H and X are the only per-tile operands.
    extern __shared__ __align__(1024) float sH[];
    __shared__ __align__(1024) float sQ[WMAX * K];
    __shared__ __align__(1024) float sQt[K * 32];
    __shared__ __align__(1024) float sX[M * 32];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sPrefix[WMAX];
    __shared__ float sRep[WMAX];
    __shared__ float sBeta[WMAX];
    __shared__ float sGamma;
    __shared__ __align__(8) unsigned long long sMmaBar;
    __shared__ unsigned sTaddr[4];

    const int t = threadIdx.x;
    const int n_all = n_ptr[0];
    const int n_rows = n_all > row_offset
        ? min(row_limit, n_all - row_offset) : 0;
    const int hpg = HV / H;
    const int n_work = n_rows * H;
    if (t < 32) tc_alloc_n(sTaddr, 128);
    __syncthreads();

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int r_local = work / H;
        const int i_h = work - r_local * H;
        const int r = r_local + row_offset;
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;
        const float* pk = k_cache + sidx * s_k_slot
                          + (long)i_h * W * K;

        // Q for S0*Q and padded Q^T for X*Q^T.
        for (int x = t; x < W * K; x += NT) {
            const int s = x / K, k = x - s * K;
            sQ[tc_swizzle<WMAX>(s, k)] = pk[x];
        }
        for (int x = t; x < K * 32; x += NT) {
            const int k = x / 32, s = x - k * 32;
            const float q = s < W ? pk[(long)s * K + k] : 0.f;
            sQt[tc_swizzle<K>(k, s)] = q;
        }
        __syncthreads();

        bool any_latch = false;
        #pragma unroll
        for (int sub = 0; sub < 3; ++sub)
            any_latch |= sub < hpg && ls6_mh[i_h * hpg + sub] > 0;

        // Q^TQ is state independent and shared by the three GQA value heads.
        for (int ix = t; ix < W * W; ix += NT) {
            const int j = ix / W, s = ix - j * W;
            float dot = 0.f;
            if (any_latch && j < s) {
                #pragma unroll
                for (int k = 0; k < K; ++k)
                    dot = fmaf(sQ[tc_swizzle<WMAX>(j, k)],
                               sQ[tc_swizzle<WMAX>(s, k)], dot);
            }
            sGram[ix] = dot;
        }
        __syncthreads();

        #pragma unroll 1
        for (int sub = 0; sub < 3; ++sub) {
            if (sub >= hpg) break;
            const int i_hv = i_h * hpg + sub;
            const bool latch = ls6_mh[i_hv] > 0;
            float* ph = h0 + sidx * s_h0_slot
                        + (long)i_hv * s_h0_h;
            const float* pd = d_cache + sidx * s_d_slot
                              + (long)i_hv * W * V;
            const float* pg = g_cache + sidx * s_g_slot
                              + (long)i_hv * W;
            const float* pb = ls6_beta + cidx * s_beta_slot
                              + (long)i_hv * W;

            const float gv = t < W ? pg[t] : 0.f;
            if (t < 32) {
                float prefix = gv;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const float prev = __shfl_up_sync(
                        0xffffffffu, prefix, off);
                    if (t >= off) prefix += prev;
                }
                const float total_log = __shfl_sync(
                    0xffffffffu, prefix, WMAX - 1);
                if (t < W) {
                    sPrefix[t] = prefix;
                    sRep[t] = __expf(total_log - prefix);
                    sBeta[t] = latch ? pb[t] : 0.f;
                }
                if (t == 0) sGamma = __expf(total_log);
            }
            __syncthreads();

            #pragma unroll 1
            for (int vt = 0; vt < V / M; ++vt) {
                const int vb = vt * M;
                // Coalesced float4 HBM read; the swizzled shared copy feeds
                // both the first MMA and the final gamma*H0 addition.
                for (int q = t; q < M * K / 4; q += NT) {
                    const int e = q * 4;
                    const int v = e / K, k = e - v * K;
                    const float4 z = *reinterpret_cast<const float4*>(
                        ph + (long)(vb + v) * K + k);
                    float* dst = sH + tc_swizzle<M>(v, k);
                    dst[0] = z.x; dst[1] = z.y;
                    dst[2] = z.z; dst[3] = z.w;
                }
                // Raw u_s for latch heads, exact d_s for dense fallback.
                // Columns 16..31 are the reduction padding of the second MMA.
                for (int x = t; x < M * 32; x += NT) {
                    const int v = x / 32, s = x - v * 32;
                    const float d = s < W
                        ? pd[(long)s * V + vb + v] : 0.f;
                    sX[tc_swizzle<M>(v, s)] = d;
                }
                if (t == 0) {
                    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                                 :: "r"(tc_smem_u32(&sMmaBar)), "r"(1));
                    asm volatile("fence.mbarrier_init.release.cluster;"
                                 ::: "memory");
                }
                __syncthreads();

                if (latch) {
                    if (t == 0) {
                        constexpr unsigned idesc_p =
                            (1u << 4) | (2u << 7) | (2u << 10)
                            | ((WMAX >> 3) << 17) | ((M >> 4) << 24);
                        const unsigned long long ah = tc_sdesc(sH, M * 128);
                        const unsigned long long bq = tc_sdesc(sQ, V * 128);
                        #pragma unroll
                        for (int kb = 0; kb < K / 32; ++kb) {
                            #pragma unroll
                            for (int ki = 0; ki < 32 / 8; ++ki) {
                                const unsigned long long ad = ah
                                    + (unsigned long long)(
                                        (kb * M * 128 + ki * 32) >> 4);
                                const unsigned long long bd = bq
                                    + (unsigned long long)(
                                        (kb * WMAX * 128 + ki * 32) >> 4);
                                tc_mma_tf32(sTaddr[0], ad, bd, idesc_p,
                                             kb != 0 || ki != 0);
                            }
                        }
                        tc_commit(&sMmaBar);
                        tc_mbar_wait(&sMmaBar, 0);
                    }
                    __syncthreads();
                    tc_fence_after_sync();
                    if (t < 128) {
                        float p[WMAX];
                        const int warp = t >> 5, lane = t & 31;
                        tc_load16(sTaddr[0], warp * 32, p);
                        constexpr int ROWS_PER_WARP = M / 4;
                        if (lane < ROWS_PER_WARP) {
                            const int v = warp * ROWS_PER_WARP + lane;
                            if (dbg & 512) {
                                #pragma unroll
                                for (int s = 0; s < WMAX; ++s)
                                    d_cache[sidx * s_d_slot
                                            + (long)i_hv * W * V
                                            + (long)s * V + vb + v] = p[s];
                            }
                            #pragma unroll
                            for (int s = 0; s < WMAX; ++s) {
                                float y = p[s];
                                #pragma unroll
                                for (int j = 0; j < WMAX; ++j)
                                    if (j < s)
                                        y = fmaf(-p[j],
                                            sGram[j * WMAX + s], y);
                                y *= sBeta[s];
                                p[s] = y;
                                const int xo = tc_swizzle<M>(v, s);
                                sX[xo] = fmaf(
                                    -sGamma, y, sRep[s] * sX[xo]);
                            }
                        }
                    }
                } else {
                    for (int x = t; x < M * W; x += NT) {
                        const int v = x / W, s = x - v * W;
                        const int xo = tc_swizzle<M>(v, s);
                        sX[xo] *= sRep[s];
                    }
                }
                __syncthreads();

                if (t == 0) {
                    constexpr unsigned idesc_x =
                        (1u << 4) | (2u << 7) | (2u << 10)
                        | ((K >> 3) << 17) | ((M >> 4) << 24);
                    const unsigned long long ax = tc_sdesc(sX, M * 128);
                    const unsigned long long bqt = tc_sdesc(sQt, K * 128);
                    #pragma unroll
                    for (int ki = 0; ki < WMAX / 8; ++ki) {
                        const unsigned long long ad = ax
                            + (unsigned long long)((ki * 8 * 4) >> 4);
                        const unsigned long long bd = bqt
                            + (unsigned long long)((ki * 8 * 4) >> 4);
                        tc_mma_tf32(sTaddr[0], ad, bd, idesc_x,
                                     ki != 0);
                    }
                    tc_commit(&sMmaBar);
                    tc_mbar_wait(&sMmaBar, latch ? 1 : 0);
                }
                __syncthreads();
                tc_fence_after_sync();

                // A warp lane is one value row in TMEM.  Vector stores are
                // intentionally row-local; the many resident CTAs provide
                // the inter-row memory-level parallelism.
                if (t < 128) {
                    const int warp = t >> 5, lane = t & 31;
                    constexpr int ROWS_PER_WARP = M / 4;
                    const int v = warp * ROWS_PER_WARP + lane;
                    #pragma unroll
                    for (int cb = 0; cb < K / WMAX; ++cb) {
                        float out[WMAX];
                        tc_load16(sTaddr[0] + cb * WMAX,
                                  warp * 32, out);
                        if (lane < ROWS_PER_WARP) {
                            float* dst = ph + (long)(vb + v) * K
                                         + cb * WMAX;
                            #pragma unroll
                            for (int j = 0; j < WMAX; j += 4) {
                                const int k = cb * WMAX + j;
                                const float4 z = make_float4(
                                    fmaf(sGamma,
                                         sH[tc_swizzle<M>(v, k)], out[j]),
                                    fmaf(sGamma,
                                         sH[tc_swizzle<M>(v, k + 1)],
                                         out[j + 1]),
                                    fmaf(sGamma,
                                         sH[tc_swizzle<M>(v, k + 2)],
                                         out[j + 2]),
                                    fmaf(sGamma,
                                         sH[tc_swizzle<M>(v, k + 3)],
                                         out[j + 3]));
                                *reinterpret_cast<float4*>(dst + j) = z;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // Ubar is deliberately added only after the state-tile dataflow
            // is selected; microbench candidates pass null here.  Keep a
            // hard guard so an experimental dispatch cannot silently skip it.
            if (latch && ls6_ubar != nullptr)
                asm volatile("trap;");
        }
        __syncthreads();
    }
    __syncthreads();
    if (t < 32) tc_dealloc_n(sTaddr[0], 128);
}

template <int K, int V, int TV, bool TC=false, bool BLOCK_WY=false>
__global__ void __launch_bounds__(TV * TK, 2)
gdn_flush_kernel(
    float* __restrict__ h0, float* __restrict__ d_cache, const float* __restrict__ k_cache,
    const float* __restrict__ g_cache, const int* __restrict__ ssm_state_indices,
    const int* __restrict__ flush_list, const int* __restrict__ n_ptr,
    const int* __restrict__ fz_nf, float* __restrict__ fz_u, float* __restrict__ fz_z,
    const float* __restrict__ fz_qbar, const float* __restrict__ fz_kbar,
    float* __restrict__ ls6_ubar, const float* __restrict__ ls6_zk,
    const int* __restrict__ ls6_map, const int* __restrict__ ls6_mh,
    const float* __restrict__ ls6_beta,
    long s_h0_slot, long s_h0_h, long s_ind, long s_d_slot, long s_k_slot, long s_g_slot,
    long s_fz_slot, long s_fzb_slot, long s_u_slot, long s_zk_slot,
    long s_beta_slot,
    int H, int HV, int W, int G, int row_offset, int row_limit, int dbg)
{
    constexpr int NT = TV * TK;
    constexpr int KPT = K / TK;   // 스레드당 k 열
    constexpr int VPT = V / TV;   // 스레드당 v 행
    constexpr int ND4 = (WMAX * V / 4 + NT - 1) / NT;   // 스테이징 float4 / 스레드 (정적 → 로드가 한꺼번에 뜬다)
    constexpr int NK4 = (WMAX * K / 4 + NT - 1) / NT;
    static_assert(KPT % 4 == 0 || KPT == 2 || KPT == 1, "KPT");
    __shared__ __align__(16) float sD[WMAX * V];     // r'_s d_s[v]
    extern __shared__ __align__(1024) float sHtc[];
    __shared__ __align__(1024) float sK[WMAX * K];
    __shared__ __align__(16) float sRawK[WMAX * K];
    __shared__ __align__(16) float sZk[WMAX * GMAX];
    __shared__ float sGram[WMAX * WMAX];
    __shared__ float sRep[WMAX + 1];                 // [W] = tot'
    __shared__ float sPrefix[WMAX];
    __shared__ float sBeta[WMAX];
    __shared__ __align__(8) unsigned long long sMmaBar;
    __shared__ unsigned sTaddr[4];
    const int t = threadIdx.x;
    const int tk = t % TK, tv = t / TK;
    const int k0 = tk * KPT, v0 = tv * VPT;
    const int n_all = n_ptr[0];
    const int n_rows = n_all > row_offset ? min(row_limit, n_all - row_offset) : 0;
    const int hpg = HV / H;
    const bool group_gqa = (dbg & 65536) != 0;
    const int work_heads = group_gqa ? H : HV;
    const int n_work = n_rows * work_heads;
    const bool ls6 = ls6_ubar != nullptr;
    const int nd4 = W * V / 4, nk4 = W * K / 4;
    if constexpr (TC && K == 128 && V == 128) {
        if (t < 32) tc_alloc(sTaddr);
    }
    __syncthreads();
    for (int w = blockIdx.x; w < n_work; w += gridDim.x) {
        const int r_local = w / work_heads;
        const int work_head = w - r_local * work_heads;
        const int sub_count = group_gqa ? hpg : 1;
        for (int sub = 0; sub < sub_count; ++sub) {
        const int i_h = group_gqa ? work_head : work_head / hpg;
        const int i_hv = group_gqa ? i_h * hpg + sub : work_head;
        const bool stage_key = !group_gqa || sub == 0;
        const int r = r_local + row_offset;
        // compact 커널이 flush_list 에 **state_idx 를 직접** 넣는다(행 번호가 아니라) — 종속 로드 1단 절약.
        const long sidx = (long)flush_list[r];
        const long cidx = ls6_map ? (long)ls6_map[sidx] : sidx;   // fz_*/ls6_* 는 compact 슬롯 (스텝 커널과 같은 규약)
        const bool exact_head = ls6_beta != nullptr && ls6_mh[i_hv] > 0;
        if constexpr (TC && K == 128 && V == 128) {
            if (exact_head && t == 0) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :: "r"(tc_smem_u32(&sMmaBar)), "r"(1));
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
        }
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
        const float* pk = k_cache + sidx * s_k_slot + (long)i_h * W * K;
        const float4* pk4 = reinterpret_cast<const float4*>(pk);
        float4 dreg[ND4], kreg[NK4];
        #pragma unroll
        for (int q = 0; q < ND4; ++q) { const int i = t + q * NT; dreg[q] = (i < nd4 && !(dbg & 8)) ? pd4[i] : make_float4(0.f, 0.f, 0.f, 0.f); }
        #pragma unroll
        for (int q = 0; q < NK4; ++q) { const int i = t + q * NT; kreg[q] = (i < nk4 && stage_key) ? pk4[i] : make_float4(0.f, 0.f, 0.f, 0.f); }
        const float gv = (t < W) ? pg[t] : 0.f;
        if (ls6 && !(dbg & 32768)) {
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
            if (t < W) {
                sRep[t] = __expf(gt - pre);
                sPrefix[t] = pre;
            }
            if (t == 0) sRep[WMAX] = __expf(gt);
        }
        #pragma unroll
        for (int q = 0; q < NK4; ++q) {
            const int i = t + q * NT;
            if (i < nk4 && stage_key) {
                if constexpr (TC && K == 128 && V == 128) {
                    reinterpret_cast<float4*>(sRawK)[i] = kreg[q];
                } else {
                    reinterpret_cast<float4*>(sK)[i] = kreg[q];
                }
            }
        }
        __syncthreads();
        const float tot = sRep[WMAX];
        #pragma unroll
        for (int q = 0; q < ND4; ++q) {
            const int i = t + q * NT;
            if (i < nd4) reinterpret_cast<float4*>(sD)[i] = dreg[q];
        }
        if (exact_head && t < W) {
            sBeta[t] = ls6_beta[
                cidx * s_beta_slot + (long)i_hv * W + t
            ];
        }
        __syncthreads();
        if (exact_head) {
            if constexpr (TC && K == 128 && V == 128) {
                const bool do_gram = (group_gqa && sub == 0)
                    || (!group_gqa && exact_head);
                if (do_gram && t < 32) {
                    using namespace nvcuda;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> gacc;
                    wmma::fill_fragment(gacc, 0.f);
                    if (!(dbg & 1024)) {
                        #pragma unroll
                        for (int kk = 0; kk < K; kk += 8) {
                            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                                           wmma::precision::tf32,
                                           wmma::row_major> ga;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                           wmma::precision::tf32,
                                           wmma::col_major> gb;
                            wmma::load_matrix_sync(ga, sRawK + kk, K);
                            wmma::load_matrix_sync(gb, sRawK + kk, K);
                            wmma::mma_sync(gacc, ga, gb, gacc);
                        }
                    }
                    wmma::store_matrix_sync(
                        sGram, gacc, WMAX, wmma::mem_row_major);
                }
            } else {
                for (int ix = t; ix < W * W; ix += NT) {
                    const int j = ix / W, s = ix - j * W;
                    float kap = 0.f;
                    if (j < s && !(dbg & 1024)) {
                        #pragma unroll
                        for (int kk = 0; kk < K; ++kk)
                            kap = fmaf(
                                sK[j * K + kk], sK[s * K + kk], kap);
                        kap *= __expf(sPrefix[s] - sPrefix[j]);
                    }
                    sGram[ix] = kap;
                }
            }
            __syncthreads();

            if constexpr (TC && K == 128 && V == 128) {
                if (t < K) {
                    if constexpr (BLOCK_WY) {
                        // Value-tiled block WY: tensor cores consume the raw
                        // key panel Q.  The small triangular recurrence is
                        // applied after P=S0*Q, once per value row.  This
                        // avoids materializing the KxW full-WY factor F.
                        #pragma unroll
                        for (int s = 0; s < WMAX; ++s)
                            sK[tc_swizzle<WMAX>(s, t)] =
                                sRawK[s * K + t];
                    } else if (!(dbg & 16384)) {
                        // Direct realization: form the normalized full-WY
                        // factor in key space before P=S0*F.
                        float fh[WMAX];
                        #pragma unroll
                        for (int s = 0; s < WMAX; ++s) {
                            float f = sRawK[s * K + t];
                            #pragma unroll
                            for (int j = 0; j < WMAX; ++j)
                                if (j < s)
                                    f = fmaf(
                                        -fh[j], sGram[j * W + s], f);
                            f *= sBeta[s];
                            fh[s] = f;
                            sK[tc_swizzle<WMAX>(s, t)] = f;
                        }
                    }
                }
                __syncthreads();

                // Stream the already resident register state through two
                // 128x32 (2x16-KiB) shared tiles.  This avoids both a second
                // HBM state read and the old 64-KiB full-state allocation.
                if (!(dbg & 2048)) {
                    constexpr unsigned idesc =
                        (1u << 4) | (2u << 7) | (2u << 10)
                        | ((WMAX >> 3) << 17) | ((V >> 4) << 24);
                    const unsigned long long bk = tc_sdesc(sK, V * 128);
                    #pragma unroll
                    for (int kp = 0; kp < K / 64; ++kp) {
                        const int own_kb = k0 / 32;
                        if (own_kb / 2 == kp) {
                            const int stage = own_kb & 1;
                            #pragma unroll
                            for (int i = 0; i < VPT; ++i) {
                                #pragma unroll
                                for (int j = 0; j < KPT; j += 4) {
                                    *reinterpret_cast<float4*>(
                                        sHtc + stage * V * 32
                                        + tc_swizzle<V>(
                                            v0 + i,
                                            k0 - own_kb * 32 + j))
                                        = make_float4(
                                            acc[i][j], acc[i][j + 1],
                                            acc[i][j + 2], acc[i][j + 3]);
                                }
                            }
                        }
                        __syncthreads();
                        if (t == 0) {
                            #pragma unroll
                            for (int stage = 0; stage < 2; ++stage) {
                                const int kb = kp * 2 + stage;
                                const unsigned long long ah = tc_sdesc(
                                    sHtc + stage * V * 32, V * 128);
                                #pragma unroll
                                for (int ki = 0; ki < 32 / 8; ++ki) {
                                    const unsigned long long ad = ah
                                        + (unsigned long long)(
                                            (ki * 32) >> 4);
                                    const unsigned long long bd = bk
                                        + (unsigned long long)(
                                            (kb * WMAX * 128 + ki * 32) >> 4);
                                    tc_mma_tf32(
                                        sTaddr[0], ad, bd, idesc,
                                        kb != 0 || ki != 0);
                                }
                            }
                            tc_commit(&sMmaBar);
                            if (kp + 1 < K / 64)
                                tc_mbar_wait(&sMmaBar, kp & 1);
                        }
                        if (kp + 1 < K / 64) {
                            // Pair 0 must finish before its shared buffers are
                            // reused by the next K-pair.
                            __syncthreads();
                        } else {
                            // The last TMEM product only reads sHtc.  Hide the
                            // independent gamma*S0 register scaling under it.
                            #pragma unroll
                            for (int i = 0; i < VPT; ++i)
                                #pragma unroll
                                for (int j = 0; j < KPT; ++j)
                                    acc[i][j] *= tot;
                            if (t == 0)
                                tc_mbar_wait(&sMmaBar, kp & 1);
                            __syncthreads();
                        }
                    }
                    tc_fence_after_sync();
                }
                if (t < V && !(dbg & 4096)) {
                    float hp[WMAX];
                    tc_load16(sTaddr[0], (t / 32) * 32, hp);
                    if constexpr (BLOCK_WY) {
                        // P_s=S0*k_s -> Y_s=S0*F_s via the same normalized
                        // WY recurrence, then fuse correction and replay:
                        // X_s = r_s*u_s - gamma*Y_s,
                        // S_W = gamma*S0 + sum_s X_s*k_s^T.
                        #pragma unroll
                        for (int s = 0; s < WMAX; ++s) {
                            float y = hp[s];
                            #pragma unroll
                            for (int j = 0; j < WMAX; ++j)
                                if (j < s)
                                    y = fmaf(
                                        -hp[j], sGram[j * W + s], y);
                            y *= sBeta[s];
                            hp[s] = y;
                            const float u = sD[s * V + t];
                            const float d_corr = u
                                - __expf(sPrefix[s]) * y;
                            sD[s * V + t] = fmaf(-tot, y, sRep[s] * u);
                            // Profiling bit 26 drops this compatibility-only
                            // write.  The ring slot is dead after a successful
                            // flush; state and Ubar consume X from shared.
                            if (!(dbg & 67108864))
                                d_cache[sidx * s_d_slot
                                        + (long)i_hv * W * V
                                        + (long)s * V + t] = d_corr;
                        }
                    } else {
                        // P=S0*F already contains the complete exact
                        // correction in the direct realization.
                        #pragma unroll
                        for (int s = 0; s < WMAX; ++s) {
                            const float p = hp[s];
                            if (dbg & 512) {
                                d_cache[sidx * s_d_slot
                                        + (long)i_hv * W * V
                                        + (long)s * V + t] = p;
                                hp[s] = 0.f;
                                sD[s * V + t] *= sRep[s];
                                continue;
                            }
                            const float d_corr = sD[s * V + t]
                                - __expf(sPrefix[s]) * p;
                            sD[s * V + t] = d_corr * sRep[s];
                            d_cache[sidx * s_d_slot
                                    + (long)i_hv * W * V
                                    + (long)s * V + t] = d_corr;
                        }
                    }
                } else if (t < V) {
                    #pragma unroll
                    for (int s = 0; s < WMAX; ++s)
                        sD[s * V + t] *= sRep[s];
                }
            } else {
                // A 16-lane subgroup owns VPT rows.  Each lane retains the
                // W-step history while the subgroup reduces resident state
                // accumulator columns.
                float hp[WMAX];
                #pragma unroll
                for (int s = 0; s < WMAX; ++s) {
                    float mine = 0.f;
                    if (s < W) {
                        #pragma unroll
                        for (int i = 0; i < VPT; ++i) {
                            float dot = 0.f;
                            #pragma unroll
                            for (int j = 0; j < KPT; ++j)
                                dot = fmaf(acc[i][j], sK[s * K + k0 + j], dot);
                            #pragma unroll
                            for (int o = TK / 2; o > 0; o >>= 1)
                                dot += __shfl_xor_sync(0xffffffffu, dot, o);
                            if (tk == i) mine = dot;
                        }
                        if (tk < VPT) {
                            const int v = v0 + tk;
                            float prev = 0.f;
                            #pragma unroll
                            for (int j = 0; j < WMAX; ++j)
                                if (j < s) prev = fmaf(
                                    hp[j], sBeta[s] * sGram[j * W + s], prev);
                            hp[s] = sBeta[s] * __expf(sPrefix[s]) * mine - prev;
                            const float d_corr = sD[s * V + v] - hp[s];
                            sD[s * V + v] = d_corr * sRep[s];
                            d_cache[sidx * s_d_slot + (long)i_hv * W * V
                                    + (long)s * V + v] = d_corr;
                        }
                    } else if (tk < VPT) {
                        hp[s] = 0.f;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int q = 0; q < ND4; ++q) {
                const int i = t + q * NT;
                if (i < nd4) {
                    float4 x = reinterpret_cast<float4*>(sD)[i];
                    const float rr = sRep[(i * 4) / V];
                    x.x *= rr; x.y *= rr; x.z *= rr; x.w *= rr;
                    reinterpret_cast<float4*>(sD)[i] = x;
                }
            }
        }
        __syncthreads();
        if constexpr (TC && K == 128 && V == 128) {
            if (!exact_head || (dbg & 2048)) {
                #pragma unroll
                for (int i = 0; i < VPT; ++i)
                    #pragma unroll
                    for (int j = 0; j < KPT; ++j) acc[i][j] *= tot;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VPT; ++i)
                #pragma unroll
                for (int j = 0; j < KPT; ++j) acc[i][j] *= tot;
        }
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
                    float4 x;
                    if constexpr (TC && K == 128 && V == 128)
                        x = *reinterpret_cast<const float4*>(
                            sRawK + s * K + k0 + j);
                    else
                        x = *reinterpret_cast<const float4*>(ks + j);
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
        // LS6 never consumes the legacy FZ anchor readouts: latch heads use
        // Ubar/f_s and m_h==0 heads read the dense state directly.  Building
        // two extra S_new @ {qbar,kbar} vectors here was therefore dead work
        // (and about 2*K*V scalar FMAs per checkpoint head).
        if (fz_u != nullptr && ls6_mh == nullptr) {
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
        // ── LS6: Ubar[g,v] = tot*Ubar[g,v] + sum_s zk[s,g] D[s,v] ──
        if (ls6 && !(dbg & 32768)) {
            float* pu = ls6_ubar + cidx * s_u_slot + (long)i_hv * G * V;
            const int mh = ls6_mh ? ls6_mh[i_hv] : G;
            const int n_gv = mh * V;
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
                        for (int s = 0; s < W; ++s)
                            u = fmaf(
                                sZk[s * G + gi[q]],
                                sD[s * V + vi[q]], u);
                        pu[idx] = u;
                    }
                }
            }
        }
        __syncthreads();   // 다음 일감이 smem 을 덮기 전에
        }
    }
    if constexpr (TC && K == 128 && V == 128) {
        __syncthreads();
        if (t < 32) tc_dealloc(sTaddr[0]);
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
    int H, int HV, int W, int G, int row_offset, int row_limit, int dbg)
{
    const bool fz = fz_u.has_value();
    const bool ls6 = ls6_ubar.has_value();
    const bool exact = ls6_beta.has_value();
    if ((dbg & 8192) && exact) {
        if constexpr (K == 128 && V == 128) {
            constexpr int wy_smem = V * K * sizeof(float);
            cudaFuncSetAttribute(
                gdn_exact_wy_fused_kernel<K, V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, wy_smem);
            gdn_exact_wy_fused_kernel<K, V><<<grid, 256, wy_smem, st>>>(
                h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                flush_list.data_ptr<int>(), flush_list.data_ptr<int>() + n_off,
                ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                h0.stride(0), h0.stride(1), d_cache.stride(0),
                k_cache.stride(0), g_cache.stride(0), ls6_beta->stride(0),
                H, HV, W);
            return;
        }
    }
    // Development gate for the low-register causal formulation above.  It is
    // promoted to the mixed latch/dense production path only after numerical
    // and B300 latency gates pass.
    if ((dbg & 16) && exact) {
        gdn_exact_causal_kernel<K, V, 8, 4><<<grid, 8 * 32, 0, st>>>(
            h0.data_ptr<float>(), d_cache.data_ptr<float>(),
            k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
            flush_list.data_ptr<int>(), flush_list.data_ptr<int>() + n_off,
            ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
            ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
            h0.stride(0), h0.stride(1), d_cache.stride(0),
            k_cache.stride(0), g_cache.stride(0), ls6_beta->stride(0),
            H, HV, W);
        return;
    }
    if ((dbg & 32) && exact) {
        if constexpr (K == 128 && V == 128) {
            gdn_exact_wmma_kernel<K, V><<<grid, 256, 0, st>>>(
                h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                flush_list.data_ptr<int>(), flush_list.data_ptr<int>() + n_off,
                ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                h0.stride(0), h0.stride(1), d_cache.stride(0),
                k_cache.stride(0), g_cache.stride(0), ls6_beta->stride(0),
                H, HV, W, dbg);
        }
    }
    if ((dbg & 64) && exact) {
        if constexpr (K == 128 && V == 128) {
            gdn_exact_tiled_kernel<K, V><<<grid, 8 * TK, 0, st>>>(
                h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                flush_list.data_ptr<int>(), flush_list.data_ptr<int>() + n_off,
                ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                h0.stride(0), h0.stride(1), d_cache.stride(0),
                k_cache.stride(0), g_cache.stride(0), ls6_beta->stride(0),
                H, HV, W, row_offset, row_limit);
            return;
        }
    }
    if ((dbg & 128) && exact) {
        if constexpr (K == 128 && V == 128) {
            gdn_exact_tiled_kernel<K, V, 64, 16>
                <<<grid, 16 * TK, 0, st>>>(
                    h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                    k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                    flush_list.data_ptr<int>(),
                    flush_list.data_ptr<int>() + n_off,
                    ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                    ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                    h0.stride(0), h0.stride(1), d_cache.stride(0),
                    k_cache.stride(0), g_cache.stride(0),
                    ls6_beta->stride(0), H, HV, W,
                    row_offset, row_limit);
            return;
        }
    }
    if ((dbg & 524288) && exact) {
        if constexpr (K == 128 && V == 128) {
            constexpr int block_wy_smem = V * K * sizeof(float);
            cudaFuncSetAttribute(
                gdn_exact_block_wy_kernel<K, V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                block_wy_smem);
            gdn_exact_block_wy_kernel<K, V>
                <<<grid, 256, block_wy_smem, st>>>(
                    h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                    k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                    flush_list.data_ptr<int>(),
                    flush_list.data_ptr<int>() + n_off,
                    ls6 ? ls6_ubar->data_ptr<float>() : nullptr,
                    ls6 ? ls6_zk->data_ptr<float>() : nullptr,
                    ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                    ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                    h0.stride(0), h0.stride(1), d_cache.stride(0),
                    k_cache.stride(0), g_cache.stride(0),
                    ls6 ? ls6_ubar->stride(0) : 0,
                    ls6 ? ls6_zk->stride(0) : 0,
                    ls6_beta->stride(0), H, HV, W, G,
                    row_offset, row_limit, dbg);
            return;
        }
    }
    if ((dbg & 2097152) && exact) {
        if constexpr (K == 128 && V == 128) {
            constexpr int tile_wy_smem = V * K * sizeof(float);
            cudaFuncSetAttribute(
                gdn_exact_tile_wy_kernel<K, V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                tile_wy_smem);
            gdn_exact_tile_wy_kernel<K, V>
                <<<grid, 256, tile_wy_smem, st>>>(
                    h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                    k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                    flush_list.data_ptr<int>(),
                    flush_list.data_ptr<int>() + n_off,
                    ls6 ? ls6_ubar->data_ptr<float>() : nullptr,
                    ls6 ? ls6_zk->data_ptr<float>() : nullptr,
                    ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                    ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                    h0.stride(0), h0.stride(1), d_cache.stride(0),
                    k_cache.stride(0), g_cache.stride(0),
                    ls6 ? ls6_ubar->stride(0) : 0,
                    ls6 ? ls6_zk->stride(0) : 0,
                    ls6_beta->stride(0), H, HV, W, G,
                    row_offset, row_limit, dbg);
            return;
        }
    }
    if ((dbg & 268435456) && exact) {
        if constexpr (K == 128 && V == 128) {
            constexpr int stream_tc_smem = 64 * K * sizeof(float);
            cudaFuncSetAttribute(
                gdn_exact_stream_tc_wy_kernel<K, V, 64>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                stream_tc_smem);
            gdn_exact_stream_tc_wy_kernel<K, V, 64>
                <<<grid, 128, stream_tc_smem, st>>>(
                    h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                    k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                    flush_list.data_ptr<int>(),
                    flush_list.data_ptr<int>() + n_off,
                    ls6 ? ls6_ubar->data_ptr<float>() : nullptr,
                    ls6 ? ls6_zk->data_ptr<float>() : nullptr,
                    ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                    ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                    h0.stride(0), h0.stride(1), d_cache.stride(0),
                    k_cache.stride(0), g_cache.stride(0),
                    ls6 ? ls6_ubar->stride(0) : 0,
                    ls6 ? ls6_zk->stride(0) : 0,
                    ls6_beta->stride(0), H, HV, W, G,
                    row_offset, row_limit, dbg);
            return;
        }
    }
    if ((dbg & 8388608) && exact) {
        if constexpr (K == 128 && V == 128) {
            constexpr int tile_tc_smem = V * 64 * sizeof(float);
            const bool wide_tile = (dbg & 16777216) != 0;
            if (wide_tile)
                cudaFuncSetAttribute(
                    gdn_exact_tile_tc_wy_kernel<K, V, 32>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    tile_tc_smem);
            else
                cudaFuncSetAttribute(
                    gdn_exact_tile_tc_wy_kernel<K, V, 16>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    tile_tc_smem);
#define TILE_TC_ARGS \
                    h0.data_ptr<float>(), d_cache.data_ptr<float>(), \
                    k_cache.data_ptr<float>(), g_cache.data_ptr<float>(), \
                    flush_list.data_ptr<int>(), \
                    flush_list.data_ptr<int>() + n_off, \
                    ls6 ? ls6_ubar->data_ptr<float>() : nullptr, \
                    ls6 ? ls6_zk->data_ptr<float>() : nullptr, \
                    ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(), \
                    ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr, \
                    h0.stride(0), h0.stride(1), d_cache.stride(0), \
                    k_cache.stride(0), g_cache.stride(0), \
                    ls6 ? ls6_ubar->stride(0) : 0, \
                    ls6 ? ls6_zk->stride(0) : 0, \
                    ls6_beta->stride(0), H, HV, W, G, \
                    row_offset, row_limit
            if (wide_tile)
                gdn_exact_tile_tc_wy_kernel<K, V, 32>
                    <<<grid, 512, tile_tc_smem, st>>>(TILE_TC_ARGS);
            else
                gdn_exact_tile_tc_wy_kernel<K, V, 16>
                    <<<grid, 256, tile_tc_smem, st>>>(TILE_TC_ARGS);
#undef TILE_TC_ARGS
            return;
        }
    }
    if ((dbg & 262144) && exact) {
        if constexpr (K == 128 && V == 128) {
            gdn_exact_coord_kernel<K, V><<<grid, 8 * 32, 0, st>>>(
                h0.data_ptr<float>(), d_cache.data_ptr<float>(),
                k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
                flush_list.data_ptr<int>(), flush_list.data_ptr<int>() + n_off,
                ls6 ? ls6_ubar->data_ptr<float>() : nullptr,
                ls6 ? ls6_zk->data_ptr<float>() : nullptr,
                ls6_mh->data_ptr<int>(), ls6_beta->data_ptr<float>(),
                ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
                h0.stride(0), h0.stride(1), d_cache.stride(0),
                k_cache.stride(0), g_cache.stride(0),
                ls6 ? ls6_ubar->stride(0) : 0,
                ls6 ? ls6_zk->stride(0) : 0,
                ls6_beta->stride(0), H, HV, W, G,
                row_offset, row_limit);
            return;
        }
    }
    if ((dbg & 256) && exact) {
        if constexpr (K == 128 && V == 128 && TV == 8) {
            constexpr int tc_smem = V * 64 * sizeof(float);
            const bool block_wy = (dbg & 33554432) != 0;
#define GDN_TC_ARGS \
                h0.data_ptr<float>(), d_cache.data_ptr<float>(), \
                k_cache.data_ptr<float>(), g_cache.data_ptr<float>(), \
                ssm_state_indices.data_ptr<int>(), flush_list.data_ptr<int>(), \
                flush_list.data_ptr<int>() + n_off, \
                fz ? fz_nf->data_ptr<int>() : nullptr, \
                fz ? fz_u->data_ptr<float>() : nullptr, \
                fz ? fz_z->data_ptr<float>() : nullptr, \
                fz ? fz_qbar->data_ptr<float>() : nullptr, \
                fz ? fz_kbar->data_ptr<float>() : nullptr, \
                ls6 ? ls6_ubar->data_ptr<float>() : nullptr, \
                ls6 ? ls6_zk->data_ptr<float>() : nullptr, \
                ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr, \
                ls6_mh.has_value() ? ls6_mh->data_ptr<int>() : nullptr, \
                ls6_beta->data_ptr<float>(), \
                h0.stride(0), h0.stride(1), ssm_state_indices.stride(0), \
                d_cache.stride(0), k_cache.stride(0), g_cache.stride(0), \
                fz ? fz_u->stride(0) : 0, \
                fz ? fz_qbar->stride(0) : 0, \
                ls6 ? ls6_ubar->stride(0) : 0, \
                ls6 ? ls6_zk->stride(0) : 0, \
                ls6_beta->stride(0), H, HV, W, G, \
                row_offset, row_limit, dbg
            // Profiling bit 17 selects the 256-thread decomposition.  It
            // halves each thread's live state accumulator (64 versus 128
            // floats) while retaining the same one-read/one-write dataflow.
            if (dbg & 131072) {
                constexpr int TC_TV = 16;
                if (block_wy) {
                    cudaFuncSetAttribute(
                        gdn_flush_kernel<K, V, TC_TV, true, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, tc_smem);
                    gdn_flush_kernel<K, V, TC_TV, true, true>
                        <<<grid, TC_TV * TK, tc_smem, st>>>(GDN_TC_ARGS);
                } else {
                    cudaFuncSetAttribute(
                        gdn_flush_kernel<K, V, TC_TV, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, tc_smem);
                    gdn_flush_kernel<K, V, TC_TV, true>
                        <<<grid, TC_TV * TK, tc_smem, st>>>(GDN_TC_ARGS);
                }
                return;
            }
            constexpr int TC_TV = 8;
            if (block_wy) {
                cudaFuncSetAttribute(
                    gdn_flush_kernel<K, V, TC_TV, true, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, tc_smem);
                gdn_flush_kernel<K, V, TC_TV, true, true>
                    <<<grid, TC_TV * TK, tc_smem, st>>>(GDN_TC_ARGS);
            } else {
                cudaFuncSetAttribute(
                    gdn_flush_kernel<K, V, TC_TV, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, tc_smem);
                gdn_flush_kernel<K, V, TC_TV, true>
                    <<<grid, TC_TV * TK, tc_smem, st>>>(GDN_TC_ARGS);
            }
#undef GDN_TC_ARGS
            return;
        }
    }
    // Legacy two-kernel correction is retained above for bisecting, but the
    // production exact path now consumes beta inside gdn_flush_kernel so S0
    // is fetched only once.
    if (false && exact) {
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
    gdn_flush_kernel<K, V, TV, false><<<grid, TV * TK, 0, st>>>(
        h0.data_ptr<float>(), d_cache.data_ptr<float>(), k_cache.data_ptr<float>(), g_cache.data_ptr<float>(),
        ssm_state_indices.data_ptr<int>(), flush_list.data_ptr<int>(), flush_list.data_ptr<int>() + n_off,
        fz ? fz_nf->data_ptr<int>() : nullptr, fz ? fz_u->data_ptr<float>() : nullptr, fz ? fz_z->data_ptr<float>() : nullptr,
        fz ? fz_qbar->data_ptr<float>() : nullptr, fz ? fz_kbar->data_ptr<float>() : nullptr,
        ls6 ? ls6_ubar->data_ptr<float>() : nullptr, ls6 ? ls6_zk->data_ptr<float>() : nullptr,
        ls6_map.has_value() ? ls6_map->data_ptr<int>() : nullptr,
        ls6_mh.has_value() ? ls6_mh->data_ptr<int>() : nullptr,
        (exact && !(dbg & 32)) ? ls6_beta->data_ptr<float>() : nullptr,
        h0.stride(0), h0.stride(1), ssm_state_indices.stride(0), d_cache.stride(0), k_cache.stride(0), g_cache.stride(0),
        fz ? fz_u->stride(0) : 0, fz ? fz_qbar->stride(0) : 0, ls6 ? ls6_ubar->stride(0) : 0, ls6 ? ls6_zk->stride(0) : 0,
        exact ? ls6_beta->stride(0) : 0,
        H, HV, W, G, row_offset, row_limit, dbg);
}

void gdn_flush(int grid,
    torch::Tensor h0, torch::Tensor d_cache, torch::Tensor k_cache, torch::Tensor g_cache,
    torch::Tensor ssm_state_indices, torch::Tensor flush_list, int n_off,
    c10::optional<torch::Tensor> fz_nf, c10::optional<torch::Tensor> fz_u, c10::optional<torch::Tensor> fz_z,
    c10::optional<torch::Tensor> fz_qbar, c10::optional<torch::Tensor> fz_kbar,
    c10::optional<torch::Tensor> ls6_ubar, c10::optional<torch::Tensor> ls6_zk,
    c10::optional<torch::Tensor> ls6_xk, c10::optional<torch::Tensor> ls6_map,
    c10::optional<torch::Tensor> ls6_mh, c10::optional<torch::Tensor> ls6_beta,
    int H, int HV, int K, int V, int W, int G, int tv,
    int row_offset, int row_limit, int dbg)
{
    TORCH_CHECK(W <= WMAX && G <= GMAX, "W<=16, G<=128");
    auto st = at::cuda::getCurrentCUDAStream().stream();
#define CASE(KK, VV, TVV) if (K == KK && V == VV && tv == TVV) { launch<KK, VV, TVV>(grid, st, h0, d_cache, k_cache, g_cache, ssm_state_indices, flush_list, n_off, fz_nf, fz_u, fz_z, fz_qbar, fz_kbar, ls6_ubar, ls6_zk, ls6_xk, ls6_map, ls6_mh, ls6_beta, H, HV, W, G, row_offset, row_limit, dbg); return; }
    CASE(128, 128, 8)
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
    int H, int HV, int K, int V, int W, int G, int tv,
    int row_offset, int row_limit, int dbg);
"""

_EXT = None

# This translation unit intentionally instantiates only the Qwen Flash-Next
# geometry while the exact SM100 path is being tuned.  Advertising the older
# generic shapes made the wrapper select CUDA and fail at launch instead of
# taking its tested Triton fallback.
_SUPPORTED_KV = {(128, 128)}


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
            name="ns_gdn_flush_mh_v17", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["gdn_flush"], build_directory=bd,
            # CUDA 13 exposes tcgen05 through the Blackwell family target.
            # sm_100f binaries are forward-compatible within that family and
            # execute on B300 (sm_103); sm_100a is not forward-compatible.
            extra_cuda_cflags=[
                "-O3", "--use_fast_math", "-lineinfo",
                "-gencode=arch=compute_100f,code=sm_100f",
            ], verbose=False)
    return _EXT


def gdn_flush_cuda(grid, h0, d_cache, k_cache, g_cache, ssm_state_indices, flush_list, n_off,
                   fz_nf, fz_u, fz_z, fz_qbar, fz_kbar, ls6_ubar, ls6_zk, ls6_xk,
                   H, HV, K, V, W, G,
                   ls6_map=None, ls6_mh=None, ls6_beta=None,
                   row_offset=0, row_limit=None):
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
    # Ubar recurrence is a separate tensor-core kernel.  The remaining
    # state/FZ fold is fastest with TV=8 on B300.
    tv = min(int(os.environ.get("NS_GDN_FLUSH_TV", "8")), max(8, V // 4))
    if ls6_map is not None and (ls6_map.dtype != torch.int32 or not ls6_map.is_contiguous() or ls6_map.dim() != 1):
        raise TypeError(f"gdn_flush_cuda: ls6_map 은 연속 int32 (NX,) 여야 한다 (받음 {ls6_map.dtype} {tuple(ls6_map.shape)})")
    if ls6_beta is not None and ls6_mh is None:
        raise ValueError("gdn_flush_cuda: exact beta에는 ls6_mh가 필요하다")
    if ls6_mh is not None and ls6_mh.dtype != torch.int32:
        raise TypeError(
            f"gdn_flush_cuda: ls6_mh 는 int32여야 한다 ({ls6_mh.dtype})"
        )
    if row_limit is None:
        row_limit = n_off
    if row_offset < 0 or row_limit < 0 or row_offset + row_limit > n_off:
        raise ValueError(
            f"invalid flush row range [{row_offset}, {row_offset + row_limit}) "
            f"for list capacity {n_off}"
        )
    _ext().gdn_flush(int(grid), h0, d_cache, k_cache, g_cache, ssm_state_indices, flush_list, int(n_off),
                     fz_nf, fz_u, fz_z, fz_qbar, fz_kbar, ls6_ubar, ls6_zk, ls6_xk, ls6_map,
                     ls6_mh, ls6_beta, H, HV, K, V, W, G, tv,
                     int(row_offset), int(row_limit),
                     int(os.environ.get('NS_GDN_FLUSH_DBG', '0')))
