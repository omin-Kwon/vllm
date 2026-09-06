// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research candidate: register-resident Gauss-Jordan on [M | C], for SPD M.
#include <cuda_runtime.h>
#include <cmath>

// Eight warps own interleaved rows; lanes own four columns of the 128-column
// augmented system. Only the normalized pivot row travels through shared
// memory.
template <int CAP>
__global__ void gj_kernel(int m, const float* scratch, float* phi, int g, int n,
                          int hv, const int* rows, const int* mapping,
                          const int* widths, int* info) {
  const int b = blockIdx.x, row_slot = b / hv, head = b % hv;
  if (widths && widths[head] != m) return;
  if (rows && (row_slot >= rows[n / hv] || rows[row_slot] <= 0)) return;
  const int slot =
      rows ? (mapping ? mapping[rows[row_slot]] : rows[row_slot]) : row_slot;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const float* sc = scratch + (long)b * 128 * g;
  float* dest = phi + ((long)slot * hv + head) * g * 128;
  constexpr int Q = CAP / 8;
  float x[Q][4];
  __shared__ float pivot[128];
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    const int i = warp + 8 * q;
#pragma unroll
    for (int e = 0; e < 4; ++e)
      x[q][e] = i < m ? sc[(lane + 32 * e) * g + i] : 0.f;
  }
  if (t == 0) info[b] = 0;
  for (int k = 0; k < m; ++k) {
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      if (warp + 8 * q == k) {
        float v = x[q][0];
#pragma unroll
        for (int e = 1; e < 4; ++e)
          if (k / 32 == e) v = x[q][e];
        const float diagonal = __shfl_sync(0xffffffffu, v, k & 31);
        if (lane == 0 && !(diagonal > 0.f)) info[b] = k + 1;
        const float reciprocal = 1.f / fmaxf(diagonal, 1e-30f);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (lane + 32 * e >= k) x[q][e] *= reciprocal;
          pivot[lane + 32 * e] = x[q][e];
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int i = warp + 8 * q;
      if (i < m && i != k) {
        float v = x[q][0];
#pragma unroll
        for (int e = 1; e < 4; ++e)
          if (k / 32 == e) v = x[q][e];
        const float factor = __shfl_sync(0xffffffffu, v, k & 31);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int j = lane + 32 * e;
          if (j > k)
            x[q][e] = fmaf(-factor, pivot[j], x[q][e]);
          else if (j == k)
            x[q][e] = 0.f;
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int q = 0; q < Q; ++q) {
    const int i = warp + 8 * q;
    if (i < m) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = lane + 32 * e;
        dest[i * 128 + j] = j < m ? float(i == j) : x[q][e];
      }
    }
  }
  for (int o = m * 128 + t; o < g * 128; o += 256) dest[o] = 0.f;
}

extern "C" int gj_solve(int m, const float* scratch, float* phi, int g, int n,
                        int hv, const int* rows, const int* mapping,
                        const int* widths, int* info, cudaStream_t stream) {
#define LAUNCH(CAP)                                                      \
  gj_kernel<CAP><<<n, 256, 0, stream>>>(m, scratch, phi, g, n, hv, rows, \
                                        mapping, widths, info)
  if (m <= 8) {
    LAUNCH(8);
  } else if (m <= 16) {
    LAUNCH(16);
  } else if (m <= 32) {
    LAUNCH(32);
  } else if (m <= 48) {
    LAUNCH(48);
  } else if (m <= 64) {
    LAUNCH(64);
  } else if (m <= 96) {
    LAUNCH(96);
  } else {
    LAUNCH(128);
  }
#undef LAUNCH
  return int(cudaGetLastError());
}
