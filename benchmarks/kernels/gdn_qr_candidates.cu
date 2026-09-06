// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research only: exact ridge-preserving basis change, without Gram or solve.
#include <cuda_runtime.h>
#include <cusolverdx.hpp>
#include <algorithm>

template <int M, cusolverdx::function F>
using QR =
    decltype(cusolverdx::Size<128 + M, M, M>() +
             cusolverdx::Precision<float>() +
             cusolverdx::Type<cusolverdx::type::real>() +
             cusolverdx::Function<F>() +
             cusolverdx::Arrangement<cusolverdx::col_major>() +
             cusolverdx::SM<1000>() + cusolverdx::Block() +
             cusolverdx::BatchesPerBlock<1>() + cusolverdx::BlockDim<128>());

template <int M, bool DOT>
__global__ void qr_metadata(const float* state, float* u, float* phi,
                            const int* rows, const int* mapping,
                            const int* widths, int max_rows, int hv, int g,
                            long state_stride, float ridge, int library_smem) {
  const int b = blockIdx.x, head = b % hv, ri = b / hv;
  const int m = widths[head];
  constexpr int LOW = M == 8 ? 0 : M / 2;
  if (ri >= rows[max_rows] || rows[ri] <= 0 || m <= LOW || m > M) return;
  using Factor = QR<M, cusolverdx::function::geqrf>;
  using Form = QR<M, cusolverdx::function::ungqr>;
  static_assert(Factor::lda == Form::lda);
  constexpr int LD = Factor::lda;
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  const int slot = rows[ri], compact = mapping[slot];
  const float* h = state + slot * state_stride + (long)head * 16384;
  float* dest_u = u + ((long)compact * hv + head) * g * 128;
  float* dest_p = phi + ((long)compact * hv + head) * g * 128;
  // Library operators own the start of dynamic shared memory, including tau.
  extern __shared__ __align__(16) float sm[];
  float* a = sm;
  float* tau = sm + LD * M;
  float* hshared = sm + (library_smem + 15) / 16 * 4;
  constexpr int HS = (DOT ? 128 : M) + 1;
  __shared__ float partial[4], root_eta;
  float sq = 0.f;
  for (int o = t; o < 16384; o += 128) {
    const float value = h[o];
    if (o % 128 < HS - 1) hshared[(o / 128) * HS + o % 128] = value;
    sq = fmaf(value, value, sq);
  }
  for (int offset = 16; offset; offset >>= 1)
    sq += __shfl_down_sync(0xffffffff, sq, offset);
  if (lane == 0) partial[warp] = sq;
  __syncthreads();
  if (t == 0)
    root_eta =
        sqrtf(ridge * ((partial[0] + partial[1]) + (partial[2] + partial[3])) /
              128.f);
  __syncthreads();
  for (int o = t; o < (128 + M) * M; o += 128) {
    const int row = o % (128 + M), col = o / (128 + M);
    a[col * LD + row] = row < 128 ? (col < m ? hshared[row * HS + col] : 0.f)
                                  : (row - 128 == col ? root_eta : 0.f);
  }
  __syncthreads();
  Factor().execute(a, LD, tau);
  __syncthreads();
  Form().execute(a, LD, tau);
  __syncthreads();
  for (int o = t; o < g * 128; o += 128) {
    const int col = o / 128, k = o % 128;
    float value = 0.f;
    if (col < m) {
      // P' = Q_v^T H + sqrt(eta) Q_b^T [I_m, 0].
      if constexpr (DOT) {
        for (int v = 0; v < 128; ++v)
          value = fmaf(a[col * LD + v], hshared[v * HS + k], value);
      }
      if (k < m) value = fmaf(root_eta, a[col * LD + 128 + k], value);
    }
    dest_u[o] = col < m ? a[col * LD + k] : 0.f;
    dest_p[o] = value;
  }
}

extern "C" int qr_refresh(int g, int max_rows, int hv, long state_stride,
                          const float* state, float* u, float* phi,
                          const int* rows, const int* mapping,
                          const int* widths, float ridge, int split,
                          cudaStream_t stream) {
  if (g > 32 || g < 4 || g % 4) return -1;
  if (max_rows == 0) return 0;
#define LAUNCH_IMPL(M, DOT)                                                  \
  do {                                                                       \
    using A = QR<M, cusolverdx::function::geqrf>;                            \
    using B = QR<M, cusolverdx::function::ungqr>;                            \
    const int library_smem =                                                 \
        std::max(A::get_shared_memory_size(), B::get_shared_memory_size());  \
    const int smem = (library_smem + 15) / 16 * 16 +                         \
                     128 * ((DOT ? 128 : M) + 1) * sizeof(float);            \
    auto error = cudaFuncSetAttribute(                                       \
        qr_metadata<M, DOT>, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
        smem);                                                               \
    if (error != cudaSuccess) return int(error);                             \
    qr_metadata<M, DOT><<<max_rows * hv, 128, smem, stream>>>(               \
        state, u, phi, rows, mapping, widths, max_rows, hv, g, state_stride, \
        ridge, library_smem);                                                \
  } while (0)
#define LAUNCH(M)            \
  do {                       \
    if (split) {             \
      LAUNCH_IMPL(M, false); \
    } else {                 \
      LAUNCH_IMPL(M, true);  \
    }                        \
  } while (0)
  LAUNCH(8);
  if (g > 8) {
    LAUNCH(16);
  }
  if (g > 16) {
    LAUNCH(32);
  }
#undef LAUNCH
#undef LAUNCH_IMPL
  return int(cudaGetLastError());
}
