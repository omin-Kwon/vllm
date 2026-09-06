// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only adapters: same Gram scratch -> same U/P coefficient layout.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverdx.hpp>

struct Context {
  cublasHandle_t blas;
  cusolverDnHandle_t solver;
};

extern "C" void* create_context() {
  auto* c = new Context;
  if (cublasCreate(&c->blas) != CUBLAS_STATUS_SUCCESS) return nullptr;
  if (cusolverDnCreate(&c->solver) != CUSOLVER_STATUS_SUCCESS) return nullptr;
  cublasSetMathMode(c->blas, CUBLAS_PEDANTIC_MATH);
  return c;
}
extern "C" void destroy_context(void* p) {
  auto* c = static_cast<Context*>(p);
  cublasDestroy(c->blas);
  cusolverDnDestroy(c->solver);
  delete c;
}

__device__ bool active(int b, int m, int hv, int max_rows, const int* rows,
                       const int* widths) {
  if (widths && widths[b % hv] != m) return false;
  if (rows && (b / hv >= rows[max_rows] || rows[b / hv] <= 0)) return false;
  return true;
}
__device__ float* output(float* phi, int b, int hv, int g, const int* rows,
                         const int* mapping) {
  const int row = b / hv;
  const int slot = rows ? (mapping ? mapping[rows[row]] : rows[row]) : row;
  return phi + ((long)slot * hv + b % hv) * g * 128;
}

__global__ void pack(const float* scratch, float* a, float* rhs, float* inv,
                     float* x, float** pointers, int m, int g, int n, int hv,
                     const int* rows, const int* widths) {
  const int b = blockIdx.x, t = threadIdx.x, nrhs = 128 - m;
  const bool use = active(b, m, hv, n / hv, rows, widths);
  const float* sc = scratch + (long)b * 128 * g;
  float* ab = a + (long)b * m * m;
  float* bb = rhs + (long)b * m * nrhs;
  if (t == 0) {
    pointers[b] = ab;
    pointers[n + b] = bb;
    pointers[2 * n + b] = inv + (long)b * m * m;
    pointers[3 * n + b] = x + (long)b * m * nrhs;
  }
  for (int o = t; o < m * m; o += blockDim.x) {
    const int i = o % m, j = o / m;
    ab[o] = use ? sc[i * g + j] : float(i == j);
  }
  for (int o = t; o < m * nrhs; o += blockDim.x)
    bb[o] = use ? sc[(m + o / m) * g + o % m] : 0.f;
}

__global__ void scatter(const float* rhs, float* phi, int m, int g, int n,
                        int hv, const int* rows, const int* mapping,
                        const int* widths) {
  const int b = blockIdx.x, t = threadIdx.x;
  if (!active(b, m, hv, n / hv, rows, widths)) return;
  float* dest = output(phi, b, hv, g, rows, mapping);
  const float* x = rhs + (long)b * m * (128 - m);
  for (int o = t; o < g * 128; o += blockDim.x) {
    const int i = o / 128, j = o % 128;
    dest[o] = i >= m ? 0.f : j < m ? float(i == j) : x[(j - m) * m + i];
  }
}

extern "C" int library_solve(void* context, int variant, int m, int g, int n,
                             int hv, const float* scratch, float* phi, float* a,
                             float* rhs, float* inv, float* x, float** pointers,
                             int* info, int* pivots, const int* rows,
                             const int* mapping, const int* widths,
                             cudaStream_t stream) {
  auto* c = static_cast<Context*>(context);
  cublasSetStream(c->blas, stream);
  cusolverDnSetStream(c->solver, stream);
  const float one = 1.f, zero = 0.f;
  const int nrhs = 128 - m;
  pack<<<n, 256, 0, stream>>>(scratch, a, rhs, inv, x, pointers, m, g, n, hv,
                              rows, widths);
  auto ap = pointers;
  auto bp = pointers + n;
  auto ip = pointers + 2 * n;
  auto xp = pointers + 3 * n;
#define BLAS(call)                         \
  do {                                     \
    auto status = (call);                  \
    if (status) return 1000 + int(status); \
  } while (0)
  if (variant == 0) {
    auto status = cusolverDnSpotrfBatched(c->solver, CUBLAS_FILL_MODE_LOWER, m,
                                          ap, m, info, n);
    if (status) return 2000 + int(status);
    BLAS(cublasStrsmBatched(c->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one,
                            ap, m, bp, m, n));
    BLAS(cublasStrsmBatched(c->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one,
                            ap, m, bp, m, n));
  } else if (variant == 1) {
    BLAS(cublasSgetrfBatched(c->blas, m, ap, m, nullptr, info, n));
    // Two TRSM calls avoid getrsBatched's host-side info argument.
    BLAS(cublasStrsmBatched(c->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, nrhs, &one, ap, m,
                            bp, m, n));
    BLAS(cublasStrsmBatched(c->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one,
                            ap, m, bp, m, n));
  } else {
    if (variant == 2) {
      if (m > 32) return -2;
      BLAS(cublasSmatinvBatched(c->blas, m, ap, m, ip, m, info, n));
    } else {
      BLAS(cublasSgetrfBatched(c->blas, m, ap, m, pivots, info, n));
      BLAS(cublasSgetriBatched(c->blas, m, ap, m, pivots, ip, m, info, n));
    }
    BLAS(cublasSgemmBatched(c->blas, CUBLAS_OP_N, CUBLAS_OP_N, m, nrhs, m, &one,
                            ip, m, bp, m, &zero, xp, m, n));
  }
#undef BLAS
  scatter<<<n, 256, 0, stream>>>(variant < 2 ? rhs : x, phi, m, g, n, hv, rows,
                                 mapping, widths);
  return int(cudaGetLastError());
}

template <int M, bool PAD>
using PosvBase =
    decltype(cusolverdx::Size<PAD ? (M + 15) / 16 * 16 : M,
                              PAD ? (M + 15) / 16 * 16 : M, 128 - M>() +
             cusolverdx::Precision<float>() +
             cusolverdx::Type<cusolverdx::type::real>() +
             cusolverdx::Function<cusolverdx::function::posv>() +
             cusolverdx::FillMode<cusolverdx::lower>() +
             cusolverdx::Arrangement<cusolverdx::col_major>() +
             cusolverdx::SM<1000>() + cusolverdx::Block() +
             cusolverdx::BatchesPerBlock<1>());

template <int M, int THREADS, bool PAD>
using Posv = decltype(PosvBase<M, PAD>() +
                      cusolverdx::BlockDim<
                          THREADS ? THREADS
                                  : PosvBase<M, PAD>::suggested_block_dim.x>());

template <int M, int THREADS, bool PAD>
__global__ void dx_kernel(const float* scratch, float* phi, int g, int n,
                          int hv, const int* rows, const int* mapping,
                          const int* widths, int* info) {
  using Op = Posv<M, THREADS, PAD>;
  constexpr int LA = Op::lda, LB = Op::ldb, R = 128 - M;
  constexpr int P = Op::m_size, T = Op::max_threads_per_block;
  const int b = blockIdx.x, t = threadIdx.x;
  if (!active(b, M, hv, n / hv, rows, widths)) return;
  extern __shared__ __align__(16) float sm[];
  float* a = sm;
  float* rhs = a + LA * P;
  const float* sc = scratch + (long)b * 128 * g;
  for (int o = t; o < P * P; o += T) {
    const int i = o % P, j = o / P;
    a[j * LA + i] = i < M && j < M ? sc[i * g + j] : float(i == j);
  }
  for (int o = t; o < P * R; o += T)
    rhs[(o / P) * LB + o % P] = o % P < M ? sc[(M + o / P) * g + o % P] : 0.f;
  __syncthreads();
  Op().execute(a, LA, rhs, info + b);
  __syncthreads();
  float* dest = output(phi, b, hv, g, rows, mapping);
  for (int o = t; o < g * 128; o += T) {
    const int i = o / 128, j = o % 128;
    dest[o] = i >= M ? 0.f : j < M ? float(i == j) : rhs[(j - M) * LB + i];
  }
}

template <int M, int T, bool PAD>
int launch_dx(const float* scratch, float* phi, int g, int n, int hv,
              const int* rows, const int* mapping, const int* widths, int* info,
              cudaStream_t stream) {
  using Op = Posv<M, T, PAD>;
  static bool initialized = false;
  if (!initialized) {
    auto status = cudaFuncSetAttribute(
        dx_kernel<M, T, PAD>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        Op::shared_memory_size);
    if (status) return int(status);
    initialized = true;
  }
  dx_kernel<M, T, PAD><<<n, Op::block_dim, Op::shared_memory_size, stream>>>(
      scratch, phi, g, n, hv, rows, mapping, widths, info);
  return int(cudaGetLastError());
}

extern "C" int dx_solve(int m, int threads, const float* scratch, float* phi,
                        int g, int n, int hv, const int* rows,
                        const int* mapping, const int* widths, int* info,
                        cudaStream_t stream) {
#define CASE(M)                                                            \
  case M:                                                                  \
    if (threads == -1)                                                     \
      return launch_dx<M, 0, true>(scratch, phi, g, n, hv, rows, mapping,  \
                                   widths, info, stream);                  \
    if (threads == 0)                                                      \
      return launch_dx<M, 0, false>(scratch, phi, g, n, hv, rows, mapping, \
                                    widths, info, stream);                 \
    return launch_dx<M, 128, false>(scratch, phi, g, n, hv, rows, mapping, \
                                    widths, info, stream)
  switch (m) {
    CASE(4);
    CASE(8);
    CASE(16);
    CASE(17);
    CASE(24);
    CASE(32);
    CASE(33);
    CASE(48);
    CASE(64);
    CASE(80);
    CASE(96);
    CASE(120);
    CASE(127);
    default:
      return -1;
  }
#undef CASE
}
