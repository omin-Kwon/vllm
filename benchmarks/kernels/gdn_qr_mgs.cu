// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research augmented QR: warp-distributed modified Gram-Schmidt in FP32.
#include <cuda_runtime.h>

__device__ float sum_warp(float value) {
  for (int d = 16; d; d >>= 1) value += __shfl_down_sync(0xffffffff, value, d);
  return __shfl_sync(0xffffffff, value, 0);
}

template <int C>
__global__ void mgs_metadata(const float* state, float* u, float* phi,
                             const int* rows, const int* mapping,
                             const int* widths, int max_rows, int hv, int g,
                             long state_stride, float ridge) {
  const int b = blockIdx.x, head = b % hv, ri = b / hv;
  const int m = widths[head];
  constexpr int LOW = C == 8 ? 0 : C / 2;
  if (ri >= rows[max_rows] || rows[ri] <= 0 || m <= LOW || m > C) return;
  const int slot = rows[ri], compact = mapping[slot];
  const float* h = state + slot * state_stride + (long)head * 16384;
  float* du = u + ((long)compact * hv + head) * g * 128;
  float* dp = phi + ((long)compact * hv + head) * g * 128;
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  __shared__ float sums[4], root_eta, q[160];
  __shared__ float sketch[128 * (C + 1)];
  float sq = 0.f;
  for (int o = t; o < 16384; o += 128) {
    const float value = h[o];
    sq = fmaf(value, value, sq);
    if (o % 128 < C) sketch[(o / 128) * (C + 1) + o % 128] = value;
  }
  sq = sum_warp(sq);
  if (lane == 0) sums[warp] = sq;
  __syncthreads();
  if (t == 0)
    root_eta =
        sqrtf(ridge * ((sums[0] + sums[1]) + (sums[2] + sums[3])) / 128.f);
  __syncthreads();
  float a[C / 4][5];
#pragma unroll
  for (int j = 0; j < C / 4; ++j) {
    const int col = warp + 4 * j;
#pragma unroll
    for (int r = 0; r < 5; ++r) {
      const int row = lane + 32 * r;
      a[j][r] = row < 128 ? (col < m ? sketch[row * (C + 1) + col] : 0.f)
                          : (row - 128 == col ? root_eta : 0.f);
    }
  }
// Each warp owns columns modulo four; all row reductions stay in a warp.
#pragma unroll
  for (int pivot = 0; pivot < C; ++pivot) {
    if (pivot < m) {
      if (warp == pivot % 4) {
        float norm = 0.f;
#pragma unroll
        for (int r = 0; r < 5; ++r)
          norm = fmaf(a[pivot / 4][r], a[pivot / 4][r], norm);
        norm = sum_warp(norm);
        const float scale = norm > 0.f ? rsqrtf(norm) : 0.f;
#pragma unroll
        for (int r = 0; r < 5; ++r) {
          a[pivot / 4][r] *= scale;
          q[lane + 32 * r] = a[pivot / 4][r];
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < C / 4; ++j) {
        const int col = warp + 4 * j;
        if (col > pivot && col < m) {
          float dot = 0.f;
#pragma unroll
          for (int r = 0; r < 5; ++r)
            dot = fmaf(q[lane + 32 * r], a[j][r], dot);
          dot = sum_warp(dot);
#pragma unroll
          for (int r = 0; r < 5; ++r)
            a[j][r] = fmaf(-dot, q[lane + 32 * r], a[j][r]);
        }
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (int j = 0; j < C / 4; ++j) {
    const int col = warp + 4 * j;
    if (col < g) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int k = lane + 32 * r;
        du[col * 128 + k] = col < m ? a[j][r] : 0.f;
        dp[col * 128 + k] =
            col < m && r == 0 && k < m ? root_eta * a[j][4] : 0.f;
      }
    }
  }
  for (int o = C * 128 + t; o < g * 128; o += 128) {
    du[o] = 0.f;
    dp[o] = 0.f;
  }
}

extern "C" int mgs_refresh(int g, int max_rows, int hv, long state_stride,
                           const float* state, float* u, float* phi,
                           const int* rows, const int* mapping,
                           const int* widths, float ridge, int split,
                           cudaStream_t stream) {
  if (g > 32 || g < 4 || g % 4) return -1;
  if (max_rows == 0) return 0;
#define LAUNCH(C)                                                          \
  mgs_metadata<C><<<max_rows * hv, 128, 0, stream>>>(                      \
      state, u, phi, rows, mapping, widths, max_rows, hv, g, state_stride, \
      ridge)
  LAUNCH(8);
  if (g > 8) {
    LAUNCH(16);
  }
  if (g > 16) {
    LAUNCH(32);
  }
#undef LAUNCH
  return int(cudaGetLastError());
}
