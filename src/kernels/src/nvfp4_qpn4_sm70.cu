// QPN4: 1Cat's SM70 prefill GEMM route, ported verbatim from
// flash_qla .../csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu (their
// nvfp4_qpn4_dequantize_vector_sm70_kernel + nvfp4_prefill_fp32_mm_out).
//
// For M >= 256 (the measured crossover on our shapes, PROGRESS_PORT 192) the
// prefill route beats the QPN2 packed-mma route: dequantize the SAME QPN2
// tile-major codes to a dense FP16 [K, N] and run a cuBLAS FP32-accum GEMM at
// ~100 TF/s vs 42-48 TF/s for m32_twophase. Bit-identical to the QPN2 route at
// our output granularity (bench_gemm_prefill: max_abs=0 at every shape).
//
// Scale semantics are OURS (nvfp4_effective_scale: e4m3 * per-column global
// scale, one FP16 rounding; the 16384 bias folded the same way), so both routes
// read the same bytes with the same values.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace qpn4_sm70 {

constexpr int kPrepareThreads = 256;

// --- helpers duplicated from nvfp4_qpn2_sm70.cu, exactly as 1Cat duplicates
// them between their own files.
__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn2_logical_k(int physical_k) {
  const int local = physical_k & 7;
  return (physical_k & 8) + ((local & 3) << 1) + (local >> 2);
}

__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

__device__ __forceinline__ half2 nvfp4_effective_scale(uint8_t value,
                                                       float global_scale) {
  const half raw = __low2half(fp8e4m3_to_half2(value));
  const half scaled =
      __float2half_rn(__fmul_rn(__half2float(raw), global_scale));
  return __halves2half2(scaled, scaled);
}

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                                half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const half2 raw = *reinterpret_cast<half2 *>(&values[index]);
    output[index] = __hmul2(raw, __hmul2(scale, __float2half2_rn(16384.0f)));
  }
}

struct Nvfp4Qpn2CodeReader {
  const uint8_t *base;

  __device__ __forceinline__ Nvfp4Qpn2CodeReader(const uint8_t *codes, int tile,
                                                 int groups_k16, int lane) {
    base = codes + (static_cast<size_t>(tile) * groups_k16 * 32 + lane) * 8;
  }

  __device__ __forceinline__ uint2 load(int group) const {
    return __ldcs(reinterpret_cast<const uint2 *>(base) +
                  static_cast<size_t>(group) * 32);
  }
};

// 1Cat: nvfp4_qpn4_dequantize_vector_sm70_kernel<UseScaleCode=true,
// TurboMindLayout=true>. Their comment kept: "Materialize the same FP16 values
// using aligned vectors. Small prefill GEMMs are sensitive to the scalar
// stores in the original dequantization kernel."
__global__ __launch_bounds__(256, 4) void qpn4_dequantize_vector(
    half *__restrict__ output, const uint8_t *__restrict__ codes,
    const uint8_t *__restrict__ packed_scales, float global_scale,
    const float *__restrict__ column_gs, int n, int k) {
  __shared__ __align__(16) half tile[128 * 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int local_group = tid >> 5;
  const int group = blockIdx.y * 8 + local_group;
  const int col = blockIdx.x * 32 + qpn2_col_from_lane(lane);
  const size_t word =
      (static_cast<size_t>(blockIdx.x) * (k / 16) + group) * 32 + lane;
  const float gs =
      column_gs == nullptr
          ? global_scale
          : column_gs[static_cast<size_t>(blockIdx.x) * 32 +
                     qpn2_col_from_lane(lane)];
  const half2 scale2 =
      nvfp4_effective_scale(packed_scales[word], gs);
  Nvfp4Qpn2CodeReader reader(codes, blockIdx.x, k / 16, lane);
  half2 weights[8];
  dequant_e2m1x8(reader.load(group).x, scale2, weights);
  dequant_e2m1x8(reader.load(group).y, scale2, weights + 4);
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const half2 value = weights[pair];
    const int physical_k = (pair & 3) + ((pair & 4) ? 8 : 0);
    const int k0 = qpn2_logical_k(physical_k);
    const int k1 = qpn2_logical_k(physical_k + 4);
    tile[(local_group * 16 + k0) * 32 + (col & 31)] = __low2half(value);
    tile[(local_group * 16 + k1) * 32 + (col & 31)] = __high2half(value);
  }
  __syncthreads();
#pragma unroll
  for (int vector = tid; vector < 512; vector += 256) {
    const int row = blockIdx.y * 128 + vector / 4;
    const int column = blockIdx.x * 32 + (vector & 3) * 8;
    *reinterpret_cast<uint4 *>(output + static_cast<size_t>(row) * n + column) =
        reinterpret_cast<const uint4 *>(tile)[vector];
  }
}

}  // namespace qpn4_sm70

// ---------------------------------------------------------------------------
// extern "C" surface (FFI)
// ---------------------------------------------------------------------------

extern "C" void nvfp4_qpn4_dequantize_sm70(
    const uint8_t *codes, const uint8_t *scales, float global_scale,
    const float *column_global_scale, half *dense_out, int n, int k,
    int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const dim3 g(n / 32, k / 128);
  qpn4_sm70::qpn4_dequantize_vector<<<g, qpn4_sm70::kPrepareThreads, 0, s>>>(
      dense_out, codes, scales, global_scale, column_global_scale, n, k);
}

// 1Cat: nvfp4_prefill_fp32_mm_out. Row-major X[M,K] * W[K,N] is column-major
// W[N,K] * X[K,M]; FP32 accumulation, no reduced-precision reduction.
extern "C" void nvfp4_qpn4_prefill_mm_sm70(const half *input,
                                           const half *dense_weight,
                                           half *output, int m, int n, int k,
                                           int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static cublasHandle_t handle = [] {
    cublasHandle_t h;
    cublasCreate(&h);
    return h;
  }();
  cublasSetStream(handle, s);
  cublasMath_t saved;
  cublasGetMathMode(handle, &saved);
  cublasSetMathMode(handle, static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH |
                                                     CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  const float alpha = 1.f, beta = 0.f;
  const cublasStatus_t st = cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dense_weight,
      CUDA_R_16F, n, input, CUDA_R_16F, k, &beta, output, CUDA_R_16F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasSetMathMode(handle, saved);
  if (st != CUBLAS_STATUS_SUCCESS) {
    printf("nvfp4_qpn4_prefill_mm_sm70: cublas status %d\n", (int)st);
    abort();
  }
}
