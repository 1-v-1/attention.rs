// SM70 (Volta) channel-wise FP8 GEMM fast path, ported from
// 1Cat-vLLM csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu (the QPN8 execution
// layout is derived from dnv2003/v100-skinny (MIT) and its block-scale
// adaptation in haohervchb/sglang-V100).
//
// Scope of this port: the dense channel-wise path only. The FusedBA, Gated,
// BlockScaled and qpn_pair variants of the upstream dispatch are dead for the
// channel-wise checkpoints this crate serves here, so their branches are
// dropped rather than stubbed. The math for the kept configurations is
// bit-identical to upstream.
//
// Layout contract (the part the hand-written 972355f attempt got wrong):
// `prepare` transposes the [N, K] row-major qweight into [K, N] tile-major
// QPN codes - each 32-column tile is stored lane-per-warp with the
// qpn8_lane_from_col permutation, and within each 16-element k-group the
// logical k lands at qpn8_physical_k(logical_k). The GEMM then reads one uint4
// (16 codes) per lane per k-group, exactly matching the m8n8k4 fragment order.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ void fp8x8_to_half2x4(uint2 q, half2 out[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned b0 = (q.x >> (8 * i)) & 0xffu;
    const unsigned b1 = (q.y >> (8 * i)) & 0xffu;
    const unsigned h0 = ((b0 & 0x80u) << 8) | ((b0 & 0x7fu) << 7);
    const unsigned h1 = ((b1 & 0x80u) << 8) | ((b1 & 0x7fu) << 7);
    const unsigned packed = h0 | (h1 << 16);
    out[i] = *reinterpret_cast<const half2*>(&packed);
  }
}

__device__ __forceinline__ void fp8x8_to_half2x4_fast(uint2 q, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3f803f80u;
  unsigned permuted[4];
  permuted[0] = __byte_perm(q.x, q.y, 0x0400);
  permuted[1] = __byte_perm(q.x, q.y, 0x0501);
  permuted[2] = __byte_perm(q.x, q.y, 0x0602);
  permuted[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned value =
        ((permuted[i] << 8) & kSign) | ((permuted[i] << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&value);
  }
}

constexpr int kQpn8PrepareThreads = 256;

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn8_lane_from_col(int col) {
  return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

__device__ __forceinline__ int qpn8_physical_k(int logical_k) {
  const int local = logical_k & 7;
  return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

__global__ void fp8_qpn8_prepack_sm70_kernel(
    uint8_t* __restrict__ codes, const uint8_t* __restrict__ qweight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k;
  if (index >= numel) {
    return;
  }

  const int row = static_cast<int>(index / k);
  const int logical_k = static_cast<int>(index % k);
  const int tile = row >> 5;
  const int lane = qpn8_lane_from_col(row & 31);
  const int group = logical_k >> 4;
  const int physical_k = qpn8_physical_k(logical_k & 15);
  const int groups_k16 = k >> 4;
  const size_t packed_index =
      (((static_cast<size_t>(tile) * groups_k16 + group) * 32 + lane) * 16 +
       physical_k);
  codes[packed_index] = qweight[index];
}

__global__ void fp8_qpn8_channel_scale_sm70_kernel(
    half* __restrict__ channel_scales, const float* __restrict__ scales,
    int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n) {
    channel_scales[col] = __float2half(scales[col] * 256.0f);
  }
}

__global__ void fp8_qpn8_silu_and_mul_sm70_kernel(
    half* __restrict__ output, const half* __restrict__ gate_up, int rows,
    int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const half* row_input = gate_up + static_cast<size_t>(row) * hidden * 2;
  half* row_output = output + static_cast<size_t>(row) * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float gate = __half2float(row_input[col]);
    const float silu = gate / (1.0f + __expf(-gate));
    row_output[col] = __hmul(__float2half(silu), row_input[hidden + col]);
  }
}

#define VLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                       \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false, bool SplitOutputs = false, int RowTiles = 1>
__global__ void fp8_qpn8_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ group_scales,
    const half* __restrict__ input, half* __restrict__ output,
    half* __restrict__ z_output, int qkv_n, int n, int k, int m,
    bool channel_scales) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "QPN8 supports one or two 8-row tiles");
  static_assert(!M1Only || RowTiles == 1,
                "QPN8 M=1 specialization uses one row tile");
  __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                         static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr = group_scales + tile;

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][chain][i] = 0.0f;
      }
    }
  }
  int loaded_scale_group = -1;
  half loaded_scale =
      channel_scales
          ? __ldg(group_scales + tile * 32 + qpn8_col_from_lane(lane))
          : __float2half(0.0f);
  uint4 prefetched = make_uint4(0, 0, 0, 0);
  if constexpr (PrefetchCodes) {
    prefetched = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
  }
  const half* single_input_row =
      (RowTiles == 1 && m > 0)
          ? (input + static_cast<size_t>(row < m ? row : (m - 1)) * k)
          : nullptr;

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const int scale_group = group >> 3;
    if (!channel_scales && scale_group != loaded_scale_group) {
      loaded_scale =
          __ldg(scale_ptr + static_cast<size_t>(scale_group) * (n >> 5));
      loaded_scale_group = scale_group;
    }

    const uint4 packed =
        PrefetchCodes ? prefetched
                      : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    uint4 next = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
      if (group + 1 < group_begin + groups_per_warp) {
        next = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
      }
    }
    half2 weights[8];
    if constexpr (FastDecoder) {
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
    } else {
      fp8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);
    }

    const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      if constexpr (RowTiles == 1) {
        if (single_input_row != nullptr) {
          input01 =
              *reinterpret_cast<const uint4*>(single_input_row + group * 16);
          input23 = *reinterpret_cast<const uint4*>(single_input_row +
                                                   group * 16 + 8);
        }
      } else {
        const int input_row_idx = row_tile * 8 + row;
        if (input_row_idx < m) {
          const half* input_row =
              input + static_cast<size_t>(input_row_idx) * k;
          input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
          input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
        }
      }

      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
    if constexpr (PrefetchCodes) {
      prefetched = next;
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][0][i] += accum[row_tile][chain][i];
      }
    }
  }

  if constexpr (M1Only) {
    if ((lane & 17) == 0) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
        for (int offset = 0; offset < 2; ++offset) {
          const int i = pair * 4 + offset;
          const int output_col =
              offset | (((lane >> 1) & 1) << 1) | (pair << 2);
          partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
        }
      }
    }
  } else {
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int output_row =
            row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partials[warp][output_row * 32 + quadpair * 8 + output_col] =
            accum[row_tile][0][i];
      }
    }
  }
  __syncthreads();

  constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
  for (int element = threadIdx.x; element < kOutputElements;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    if constexpr (M1Only) {
      const int output_col = tile * 32 + element;
      if constexpr (SplitOutputs) {
        if (output_col < qkv_n) {
          output[output_col] = __float2half(value);
        } else {
          z_output[output_col - qkv_n] = __float2half(value);
        }
      } else {
        output[output_col] = __float2half(value);
      }
    } else {
      const int output_row = element >> 5;
      const int output_col = element & 31;
      if (output_row < m) {
        const int col = tile * 32 + output_col;
        if constexpr (SplitOutputs) {
          if (col < qkv_n) {
            output[static_cast<size_t>(output_row) * qkv_n + col] =
                __float2half(value);
          } else {
            z_output[static_cast<size_t>(output_row) * (n - qkv_n) + col -
                     qkv_n] = __float2half(value);
          }
        } else {
          output[static_cast<size_t>(output_row) * n + col] =
              __float2half(value);
        }
      }
    }
  }
}

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false, int RowTiles = 1>
void launch_fp8_qpn8_sm70(const uint8_t* codes, const half* group_scales,
                          const half* input, half* output, int n, int k, int m,
                          bool channel_scales, cudaStream_t stream) {
  fp8_qpn8_sm70_kernel<SplitK, NAcc, FastDecoder, PrefetchCodes, M1Only, false,
                      RowTiles>
        <<<(n / 32), (32 * SplitK), 0, stream>>>(codes, group_scales, input,
                                                 output, nullptr, 0, n, k, m,
                                                 channel_scales);
}

// [M, K] -> [K/16, M, 16]. This packs only activations; quantized weights
// and their numerics remain unchanged.
__global__ void pack_k16_input(const half* __restrict__ input,
                               half* __restrict__ packed, int m, int k) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * k / 2) return;
  const int pair = index & 7;
  const int row = (index >> 3) % m;
  const int group = (index >> 3) / m;
  reinterpret_cast<half2*>(packed)[index] = reinterpret_cast<const half2*>(
      input)[static_cast<size_t>(row) * k / 2 + group * 8 + pair];
}

// M32-class two-phase GEMM with a templated row-tile count: RowTiles=4 covers
// M=17..32 (four independent 8-row accumulator tiles), RowTiles=5 covers
// M=33..40. Keeping all logical split-K warps resident would exceed the shared
// budget, so half as many physical warps execute the original logical warp
// ranges in two ordered phases. The compact first-half sum lets the final
// reduction retain the exact p0 + ... + p(SplitK-1) order.
template <int SplitK, bool PackedInput = false, bool FullRows = false,
          int RowTiles = 4, int KUnroll = 0>
__global__ __launch_bounds__(
    32 * (SplitK / 2),
    (32 * (SplitK / 2) == 128)
        ? 3
        : 2) void fp8_qpn8_m32_twophase_sm70_kernel(const uint8_t* __restrict__ codes,
                                                    const half* __restrict__ channel_scales,
                                                    const half* __restrict__ input,
                                                    half* __restrict__ output,
                                                    int n, int k, int m) {
  static_assert(SplitK == 8 || SplitK == 12 || SplitK == 16,
                "M32 two-phase QPN8 supports split-8, split-12 or split-16");
  static_assert(RowTiles == 2 || RowTiles == 4 || RowTiles == 5,
                "two-phase QPN8 supports two, four or five row tiles");
  static_assert(KUnroll == 0 || KUnroll == 1 || KUnroll == 2 || KUnroll == 4,
                "M32 two-phase QPN8 K-unroll must be 0 (auto), 1, 2 or 4");
  if constexpr (FullRows) m = RowTiles * 8;
  constexpr int kPhysicalWarps = SplitK / 2;
  constexpr int kRowTiles = RowTiles;
  constexpr int kOutputElements = kRowTiles * 256;
  // With four or five row tiles, unrolling four K groups exceeds 128 registers
  // and prevents a second CTA from residing on a SM70 SM.
  constexpr int kKUnroll =
      RowTiles >= 4 ? 1 : (KUnroll > 0 ? KUnroll : (PackedInput ? 1 : 4));
  constexpr int kThreads = 32 * kPhysicalWarps;
  constexpr int kPartialSlots = (kOutputElements + kThreads - 1) / kThreads;
  __shared__ float reduction_storage[kPhysicalWarps + 1][kOutputElements];
  __shared__ float first_sum_smem[RowTiles == 5 ? kOutputElements : 1];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                         static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  // Channel scales keep their per-column K-invariant semantics.
  half loaded_scale;
  int loaded_scale_group = -1;
  half next_scale = __float2half(0.0f);
  half2 scale2;
  loaded_scale = __ldg(channel_scales + tile * 32 + qpn8_col_from_lane(lane));
  scale2 = __halves2half2(loaded_scale, loaded_scale);

#pragma unroll
  for (int phase = 0; phase < 2; ++phase) {
    float accum[kRowTiles][2][8];
#pragma unroll
    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
      for (int chain = 0; chain < 2; ++chain) {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          accum[row_tile][chain][index] = 0.0f;
        }
      }
    }

    const int logical_warp = warp + phase * kPhysicalWarps;
    const int group_begin = logical_warp * groups_per_warp;
    constexpr bool kPrefetch = FullRows;
    uint4 next_codes = make_uint4(0, 0, 0, 0);
    if constexpr (kPrefetch) {
      next_codes = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
    }
#pragma unroll kKUnroll
    for (int group = group_begin; group < group_begin + groups_per_warp;
         ++group) {
      const uint4 packed =
          kPrefetch ? next_codes
                    : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
      if constexpr (kPrefetch) {
        if (group + 1 < group_begin + groups_per_warp) {
          next_codes = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
        }
      }
      half2 weights[8];
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        weights[index] = __hmul2(weights[index], scale2);
      }

      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        const int input_row_idx = row_tile * 8 + row;
        if (input_row_idx < m) {
          const half* input_row =
              PackedInput
                  ? input +
                        (static_cast<size_t>(group) * m + input_row_idx) * 16
                  : input + static_cast<size_t>(input_row_idx) * k + group * 16;
          input01 = *reinterpret_cast<const uint4*>(input_row);
          input23 = *reinterpret_cast<const uint4*>(input_row + 8);
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1], a0[2], a0[3], b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a1[0], a1[1], b[4], b[5]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1], a1[2], a1[3], b[6], b[7]);
      }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][1][index];
        const int output_row =
            row_tile * 8 + (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
        reduction_storage[warp]
                         [output_row * 32 + quadpair * 8 + output_col] =
                             accum[row_tile][0][index];
      }
    }
    __syncthreads();

    auto reduce_slot = [&](int element, int slot) {
      float value = 0.0f;
      if (phase != 0) {
        if constexpr (RowTiles == 5) {
          value = first_sum_smem[element];
        } else {
          value = reduction_storage[kPhysicalWarps][element];
        }
      }
#pragma unroll
      for (int k_warp = 0; k_warp < kPhysicalWarps; ++k_warp) {
        value += reduction_storage[k_warp][element];
      }
      if (phase == 0) {
        if constexpr (RowTiles == 5) {
          first_sum_smem[element] = value;
        } else {
          reduction_storage[kPhysicalWarps][element] = value;
        }
      } else {
        const int output_row = element >> 5;
        const int output_col = element & 31;
        if (output_row < m) {
          const half result = __float2half(value);
          output[static_cast<size_t>(output_row) * n + blockIdx.x * 32 +
                 output_col] = result;
        }
      }
    };
    if constexpr (RowTiles == 5) {
      // Constant indices keep M40 first-phase partials in registers.
#pragma unroll
      for (int slot = 0; slot < kPartialSlots; ++slot) {
        const int element = threadIdx.x + slot * kThreads;
        if (element < kOutputElements) reduce_slot(element, slot);
      }
    } else {
      // Preserve the measured M32 reduction schedule.
      for (int element = threadIdx.x; element < kOutputElements;
           element += blockDim.x) {
        reduce_slot(element, element / kThreads);
      }
    }
    __syncthreads();
  }
}
template <int SplitK, bool PackedInput = false, int RowTiles = 4,
          int KUnroll = 0, bool FullRows = false>
void launch_fp8_qpn8_m32_twophase_sm70(const uint8_t* codes,
                                      const half* channel_scales,
                                      const half* input, half* output, int n,
                                      int k, int m, cudaStream_t stream) {
  constexpr int kPhysicalWarps = SplitK / 2;
  if constexpr (!FullRows && RowTiles == 4) {
    if (m > RowTiles * 8 && m <= (RowTiles + 1) * 8) {
      launch_fp8_qpn8_m32_twophase_sm70<SplitK, PackedInput, RowTiles + 1,
                                       KUnroll>(codes, channel_scales, input,
                                               output, n, k, m, stream);
      return;
    }
  }
  auto kernel = fp8_qpn8_m32_twophase_sm70_kernel<SplitK, PackedInput, FullRows,
                                                 RowTiles, KUnroll>;
  if constexpr (!FullRows) {
    // A full row-tile set drops the per-group row guard; the math per row is
    // identical, so this only fires when the batch fills the tile set.
    if (m == RowTiles * 8) {
      kernel = fp8_qpn8_m32_twophase_sm70_kernel<SplitK, PackedInput, true,
                                                RowTiles, KUnroll>;
    }
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                      100);
  kernel<<<(n / 32), (32 * kPhysicalWarps), 0, stream>>>(codes, channel_scales,
                                                        input, output, n, k,
                                                        m);
}

}  // namespace

// Split the [N, K] row-major channel-wise FP8 weight into QPN-packed [K, N]
// codes plus the F16 channel scales (x256, matching the raw-bit decoder).
extern "C" void fp8_qpn8_prepare_f16(uint8_t* codes, half* group_scales,
                                    const uint8_t* qweight,
                                    const float* scales, int n, int k,
                                    cudaStream_t stream) {
  const int weight_blocks =
      static_cast<int>((static_cast<size_t>(n) * k + kQpn8PrepareThreads - 1) /
                      kQpn8PrepareThreads);
  fp8_qpn8_prepack_sm70_kernel<<<weight_blocks, kQpn8PrepareThreads, 0,
                                stream>>>(codes, qweight, n, k);
  const int scale_blocks = (n + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads;
  fp8_qpn8_channel_scale_sm70_kernel<<<scale_blocks, kQpn8PrepareThreads, 0,
                                      stream>>>(group_scales, scales, n);
}

extern "C" void fp8_qpn8_pack_input_f16(const half* input, half* packed, int m,
                                       int k, cudaStream_t stream) {
  constexpr int kThreads = 256;
  pack_k16_input<<<(static_cast<size_t>(m) * k / 2 + kThreads - 1) / kThreads,
                  kThreads, 0, stream>>>(input, packed, m, k);
}

// out[M, N] = input[M, K] @ dequant(codes). input_is_packed selects the
// [K/16, M, 16] activation twin produced by fp8_qpn8_pack_input_f16.
extern "C" void fp8_qpn8_gemm_f16(const half* input, const uint8_t* codes,
                                 const half* group_scales, half* output, int m,
                                 int n, int k, int split_k,
                                 int input_is_packed, cudaStream_t stream) {
  const bool packed = input_is_packed != 0;
  if (m <= 8) {
    switch (split_k) {
      case 4:
        launch_fp8_qpn8_sm70<4, 1, true, false>(codes, group_scales, input,
                                               output, n, k, m, true, stream);
        break;
      case 8:
        launch_fp8_qpn8_sm70<8, 1, true, false>(codes, group_scales, input,
                                               output, n, k, m, true, stream);
        break;
      case 16:
        launch_fp8_qpn8_sm70<16, 1, true, false>(codes, group_scales, input,
                                                output, n, k, m, true, stream);
        break;
      default:
        launch_fp8_qpn8_sm70<32, 1, true, false>(codes, group_scales, input,
                                                output, n, k, m, true, stream);
        break;
    }
  } else if (m <= 16) {
    switch (split_k) {
      case 4:
        launch_fp8_qpn8_sm70<4, 2, true, false, false, 2>(
            codes, group_scales, input, output, n, k, m, true, stream);
        break;
      case 8:
        launch_fp8_qpn8_sm70<8, 2, true, false, false, 2>(
            codes, group_scales, input, output, n, k, m, true, stream);
        break;
      default:
        launch_fp8_qpn8_sm70<16, 2, true, false, false, 2>(
            codes, group_scales, input, output, n, k, m, true, stream);
        break;
    }
  } else {
    if (packed) {
      if (split_k == 12) {
        launch_fp8_qpn8_m32_twophase_sm70<12, true>(codes, group_scales, input,
                                                  output, n, k, m, stream);
      } else {
        launch_fp8_qpn8_m32_twophase_sm70<16, true>(codes, group_scales, input,
                                                  output, n, k, m, stream);
      }
    } else if (split_k == 12) {
      launch_fp8_qpn8_m32_twophase_sm70<12>(codes, group_scales, input, output,
                                           n, k, m, stream);
    } else {
      launch_fp8_qpn8_m32_twophase_sm70<16>(codes, group_scales, input, output,
                                           n, k, m, stream);
    }
  }
}
