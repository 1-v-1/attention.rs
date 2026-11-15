// NVFP4 QPN2 kernels for SM70 (Volta), ported from 1Cat-vLLM
// (csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu, nvfp4_qpn2_layout.cuh,
// activation_pack_sm70.cuh) with exactly two adaptations:
//
//   1. The global scale is per output column (`column_global_scale`, the
//      xinfer `weight_row_scales`), not 1Cat's uniform kernel argument.
//      Every QPN2 layout lane owns exactly one column, so the per-column
//      scale is loaded once per lane and the arithmetic is unchanged:
//      half_rn(e4m3 * gs) then half_rn(e2m1 * that), bit-identical to the
//      old nvfp4_repack_weight_sm70 fold.
//   2. The m32 two-phase kernel gained a grid.z row-block dimension so that
//      prefill chunks larger than 32 rows reuse the same verified kernel.
//
// Dispatch (mirrors 1Cat's kQpn2DispatchMaxRows ladder):
//   m <= 8   native projection, RowTiles = 1
//   9..16    same accumulate with RowTiles = 2
//   >16      pack_k16_input + m32 two-phase, one CTA per 32-row block
//
// The weights live in the native QPN2 layout [N/32][K/16][lane][uint2]; the
// E4M3 block scales in [N/32][K/16][lane]. Both are permutations of the
// checkpoint bytes, so the storage cost is unchanged.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qpn2_sm70 {

constexpr int kQpn2RowsPerCta = 8;

__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn2_logical_k(int physical_k) {
  const int local = physical_k & 7;
  return (physical_k & 8) + ((local & 3) << 1) + (local >> 2);
}

// 1Cat: nvfp4_qpn2_prepack_codes_kernel
__global__ void nvfp4_qpn2_prepack_codes_kernel(
    uint8_t *__restrict__ output, const uint8_t *__restrict__ weight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 2;
  if (index >= numel) {
    return;
  }
  const int physical_byte = static_cast<int>(index & 7);
  size_t outer = index >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  const int logical_k0 = group * 16 + qpn2_logical_k(physical_byte * 2);
  const int logical_k1 = group * 16 + qpn2_logical_k(physical_byte * 2 + 1);
  const int k_bytes = k >> 1;
  const uint8_t packed0 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k0 >> 1)];
  const uint8_t packed1 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k1 >> 1)];
  const uint8_t code0 =
      static_cast<uint8_t>((packed0 >> ((logical_k0 & 1) * 4)) & 0x0f);
  const uint8_t code1 =
      static_cast<uint8_t>((packed1 >> ((logical_k1 & 1) * 4)) & 0x0f);
  output[index] = static_cast<uint8_t>(code0 | (code1 << 4));
}

// 1Cat: nvfp4_qpn2_prepack_scales_kernel
__global__ void nvfp4_qpn2_prepack_scales_kernel(
    uint8_t *__restrict__ output, const uint8_t *__restrict__ scales, int n,
    int k, int logical_n) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 16;
  if (index >= numel) {
    return;
  }
  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  output[index] = column < logical_n
                      ? scales[static_cast<size_t>(column) * groups_k16 + group]
                      : 0;
}

// 1Cat: fp8e4m3_to_half2
__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

// 1Cat: nvfp4_effective_scale. Match TurboMind's W4A16 weights: multiply the
// exact E4M3 group scale by the FP32 global scale before rounding once to
// FP16. Rounding the global factor first changes the model's weights on
// every decode step.
__device__ __forceinline__ half2 nvfp4_effective_scale(uint8_t value,
                                                       float global_scale) {
  const half raw = __low2half(fp8e4m3_to_half2(value));
  const half scaled =
      __float2half_rn(__fmul_rn(__half2float(raw), global_scale));
  return __halves2half2(scaled, scaled);
}

// 1Cat: dequant_e2m1x8 (RebiasedScale = true: the admitted scale range makes
// the power-of-two rescaling exact, including FP16 subnormals).
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

// 1Cat: Nvfp4Qpn2CodeReader<false, false, false> (native layout).
struct Nvfp4Qpn2CodeReader {
  const uint8_t *base;

  __device__ __forceinline__ Nvfp4Qpn2CodeReader(const uint8_t *codes, int tile,
                                                 int groups_k16, int lane) {
    base = codes +
           (static_cast<size_t>(tile) * groups_k16 * 32 + lane) * 8;
  }

  __device__ __forceinline__ uint2 load(int group) const {
    return __ldcs(reinterpret_cast<const uint2 *>(base) +
                  static_cast<size_t>(group) * 32);
  }
};

#define VLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                        \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]),              \
        "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7])               \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// 1Cat: nvfp4_qpn2_accumulate_sm70<16, 2, RowTiles, false, false, false, 4,
// true>, with the per-column global scale (see file header).
// 1Cat: nvfp4_qpn2_accumulate_sm70 (fully parameterized). With TileColumns=16
// each physical warp owns two logical K partitions (one per 16-column half
// of the 32-row weight tile), preserving each partition's MMA chain and
// ascending reduction order; adjacent output tiles reuse the same weight tile.
template <int SplitK, int NAcc, int RowTiles = 1, int TileColumns = 32>
__device__ __forceinline__ void nvfp4_qpn2_accumulate_sm70(
    const uint8_t *__restrict__ codes, const uint8_t *__restrict__ scales,
    const half *__restrict__ input, half *__restrict__ output, int n, int k,
    int m, float global_scale, const float *__restrict__ column_gs,
    const half *__restrict__ bias) {
  constexpr int QuadsPerTile = TileColumns / 8;
  __shared__ float partials[SplitK][RowTiles * 8 * TileColumns];

  const int lane = threadIdx.x & 31;
  const int quadpair = (lane >> 2) & 3;
  const int warp =
      (threadIdx.x >> 5) * (32 / TileColumns) + quadpair / QuadsPerTile;
  const int output_tile = blockIdx.x;
  const int tile = output_tile * TileColumns / 32;
  const int mapped_lane =
      (lane & 19) |
      ((quadpair % QuadsPerTile +
        (output_tile % (32 / TileColumns)) * QuadsPerTile)
       << 2);
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const Nvfp4Qpn2CodeReader reader(codes, tile, groups_k16, mapped_lane);
  const uint8_t *scale_ptr =
      scales + static_cast<size_t>(tile) * groups_k16 * 32 + mapped_lane;
  // Each u8-scale lane owns exactly one column; fold its column's global
  // scale in (ours), or use the uniform one (1Cat).
  const float gs = column_gs == nullptr
                       ? global_scale
                       : column_gs[static_cast<size_t>(tile) * 32 +
                                  qpn2_col_from_lane(mapped_lane)];

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][chain][index] = 0.0f;
      }
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed = reader.load(group);
    const half2 scale = nvfp4_effective_scale(
        __ldg(scale_ptr + static_cast<size_t>(group) * 32), gs);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);

    const unsigned *b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int row = row_base + row_tile * kQpn2RowsPerCta + local_row;
      if (row < m) {
        const half *input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
      }
      const unsigned *a0 = reinterpret_cast<const unsigned *>(&input01);
      const unsigned *a1 = reinterpret_cast<const unsigned *>(&input23);
      VLLM_SM70_QPN2_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][chain][index];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[warp][output_row * TileColumns +
                     (quadpair % QuadsPerTile) * 8 + output_col] =
          accum[row_tile][0][index];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 8 * TileColumns;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    const int output_row = row_base + (element / TileColumns);
    const int output_col = element % TileColumns;
    if (output_row < m) {
      const int full_col = output_tile * TileColumns + output_col;
      if (bias != nullptr) value += __half2float(bias[full_col]);
      output[static_cast<size_t>(output_row) * n + full_col] =
          __float2half(value);
    }
  }
}

// 1Cat: nvfp4_qpn2_projection_sm70_kernel — the native m <= 8 path and its
// RowTiles=2 sibling (8 < m <= 16). Thread count and residency follow the
// tile width: 64/TileColumns CTAs per SM stay resident either way.
template <int K, int TileColumns, int SplitK, int RowTiles = 1>
__global__ __launch_bounds__(TileColumns * SplitK, 64 / TileColumns)
    void nvfp4_qpn2_projection_sm70_kernel(
    const uint8_t *__restrict__ codes, const uint8_t *__restrict__ scales,
    const half *__restrict__ input, half *__restrict__ output, int n, int k,
    int m, float global_scale, const float *__restrict__ column_gs,
    const half *__restrict__ bias) {
  nvfp4_qpn2_accumulate_sm70<SplitK, 2, RowTiles, TileColumns>(
      codes, scales, input, output, n, K > 0 ? K : k, m, global_scale,
      column_gs, bias);
}

// 1Cat: pack_k16_input ([M, K] -> [K/16, M, 16], half2 granularity).
__global__ void pack_k16_input_kernel(const half *__restrict__ input,
                                      half *__restrict__ packed, int m,
                                      int k) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * k / 2) return;
  const int pair = index & 7;
  const int row = (index >> 3) % m;
  const int group = (index >> 3) / m;
  reinterpret_cast<half2 *>(packed)[index] = reinterpret_cast<const half2 *>(
      input)[static_cast<size_t>(row) * k / 2 + group * 8 + pair];
}

// 1Cat: nvfp4_qpn2_m32_twophase_sm70_kernel<16, false, FullRows> plus the
// grid.z row-block dimension. `input` is the K16-packed activation.
template <bool FullRows>
__global__ __launch_bounds__(256, 2) void nvfp4_qpn2_m32_twophase_sm70_kernel(
    const uint8_t *__restrict__ codes, const uint8_t *__restrict__ scales,
    const half *__restrict__ input, half *__restrict__ output, int width,
    int k, int m, float global_scale,
    const float *__restrict__ column_gs, const half *__restrict__ bias) {
  constexpr int RowTiles = 4;
  constexpr int Phases = 2;
  constexpr int Split = 16;
  constexpr int Warps = Split / Phases;
  constexpr int Elements = RowTiles * 256;
  __shared__ float partial[Warps + 1][Elements];
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) % Warps;
  const int qp = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.z * RowTiles * 8;
  const int tile = blockIdx.y;
  const int groups = k / 16;
  const int groups_per_warp = groups / Split;
  const Nvfp4Qpn2CodeReader reader(codes, tile, groups, lane);
  const uint8_t *scale_ptr =
      scales + static_cast<size_t>(tile) * groups * 32 + lane;
  const float gs = column_gs == nullptr
                       ? global_scale
                       : column_gs[static_cast<size_t>(tile) * 32 +
                                  qpn2_col_from_lane(lane)];

#pragma unroll
  for (int phase = 0; phase < Phases; ++phase) {
    float accum[RowTiles][2][8] = {};
    const int begin = (warp + phase * Warps) * groups_per_warp;
    uint2 next_codes = make_uint2(0, 0);
    uint8_t next_scale = 0;
    if constexpr (FullRows) {
      next_codes = reader.load(begin);
      next_scale = __ldg(scale_ptr + static_cast<size_t>(begin) * 32);
    }
#pragma unroll 1
    for (int group = begin; group < begin + groups_per_warp; ++group) {
      const uint2 code = FullRows ? next_codes : reader.load(group);
      const half2 scale = nvfp4_effective_scale(
          FullRows ? next_scale
                   : __ldg(scale_ptr + static_cast<size_t>(group) * 32),
          gs);
      if constexpr (FullRows) {
        if (group + 1 < begin + groups_per_warp) {
          next_codes = reader.load(group + 1);
          next_scale =
              __ldg(scale_ptr + static_cast<size_t>(group + 1) * 32);
        }
      }
      half2 weights[8];
      dequant_e2m1x8(code.x, scale, weights);
      dequant_e2m1x8(code.y, scale, weights + 4);
      const unsigned *b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
      for (int tile_row = 0; tile_row < RowTiles; ++tile_row) {
        uint4 a01 = make_uint4(0, 0, 0, 0);
        uint4 a23 = make_uint4(0, 0, 0, 0);
        const int row = row_base + tile_row * 8 + local_row;
        if (FullRows || row < m) {
          const half *a = input + (static_cast<size_t>(group) * m + row) * 16;
          a01 = *reinterpret_cast<const uint4 *>(a);
          a23 = *reinterpret_cast<const uint4 *>(a + 8);
        }
        const unsigned *a0 = reinterpret_cast<const unsigned *>(&a01);
        const unsigned *a1 = reinterpret_cast<const unsigned *>(&a23);
        VLLM_SM70_QPN2_MMA(accum[tile_row][0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_QPN2_MMA(accum[tile_row][1], a0[2], a0[3], b[2], b[3]);
        VLLM_SM70_QPN2_MMA(accum[tile_row][0], a1[0], a1[1], b[4], b[5]);
        VLLM_SM70_QPN2_MMA(accum[tile_row][1], a1[2], a1[3], b[6], b[7]);
      }
    }
#pragma unroll
    for (int tile_row = 0; tile_row < RowTiles; ++tile_row) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r =
            tile_row * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partial[warp][r * 32 + qp * 8 + c] =
            accum[tile_row][0][i] + accum[tile_row][1][i];
      }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < Elements; e += blockDim.x) {
      float sum = 0.0f;
      if (phase > 0) sum = partial[Warps][e];
#pragma unroll
      for (int w = 0; w < Warps; ++w) sum += partial[w][e];
      if (phase + 1 < Phases) {
        partial[Warps][e] = sum;
      } else {
        const int row = row_base + e / 32;
        if (FullRows || row < m) {
          const int col = blockIdx.y * 32 + e % 32;
          if (bias != nullptr) sum += __half2float(bias[col]);
          output[static_cast<size_t>(row) * width + col] = __float2half(sum);
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace qpn2_sm70

// ---------------------------------------------------------------------------
// extern "C" surface (FFI)
// ---------------------------------------------------------------------------

extern "C" void nvfp4_qpn2_prepack_sm70(const uint8_t *weight,
                                        const uint8_t *scales,
                                        uint8_t *codes_out,
                                        uint8_t *scales_out, int n, int k,
                                        int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const size_t cnum = static_cast<size_t>(n) * k / 2;
  const size_t snum = static_cast<size_t>(n) * k / 16;
  qpn2_sm70::nvfp4_qpn2_prepack_codes_kernel
      <<<(cnum + 255) / 256, 256, 0, s>>>(codes_out, weight, n, k);
  qpn2_sm70::nvfp4_qpn2_prepack_scales_kernel
      <<<(snum + 255) / 256, 256, 0, s>>>(scales_out, scales, n, k, n);
}

extern "C" void nvfp4_qpn2_matmul_sm70(
    const half *input, const uint8_t *codes, const uint8_t *scales,
    float global_scale, const float *column_global_scale, const half *bias,
    half *output, int m, int n, int k, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (m <= 0 || n <= 0 || k <= 0) return;
  if (m <= 8) {
    if (k == 5120) {
      qpn2_sm70::nvfp4_qpn2_projection_sm70_kernel<5120, 32, 16>
          <<<dim3(n / 32, (m + 7) / 8), 512, 0, s>>>(
              codes, scales, input, output, n, k, m, global_scale,
              column_global_scale, bias);
    } else {
      qpn2_sm70::nvfp4_qpn2_projection_sm70_kernel<0, 32, 16>
          <<<dim3(n / 32, (m + 7) / 8), 512, 0, s>>>(
              codes, scales, input, output, n, k, m, global_scale,
              column_global_scale, bias);
    }
  } else if (m <= 16) {
    qpn2_sm70::nvfp4_qpn2_projection_sm70_kernel<0, 32, 16, 2>
        <<<dim3(n / 32, (m + 15) / 16), 512, 0, s>>>(
            codes, scales, input, output, n, k, m, global_scale,
            column_global_scale, bias);
  } else {
    // 1Cat route: the Rust wrapper first runs nvfp4_qpn2_pack_k16_sm70 into a
    // scratch of m * k halves, then passes the packed buffer here; one CTA
    // per 32-row block via grid.z.
    const dim3 g(1, n / 32, (m + 31) / 32);
    if (m % 32 == 0) {
      qpn2_sm70::nvfp4_qpn2_m32_twophase_sm70_kernel<true>
          <<<g, 256, 0, s>>>(codes, scales, input, output, n, k, m,
                            global_scale, column_global_scale, bias);
    } else {
      qpn2_sm70::nvfp4_qpn2_m32_twophase_sm70_kernel<false>
          <<<g, 256, 0, s>>>(codes, scales, input, output, n, k, m,
                            global_scale, column_global_scale, bias);
    }
  }
}

extern "C" void nvfp4_qpn2_pack_k16_sm70(const half *input, half *packed,
                                         int m, int k, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int n = m * k;
  qpn2_sm70::pack_k16_input_kernel<<<(n / 2 + 255) / 256, 256, 0, s>>>(input,
                                                                      packed, m, k);
}
