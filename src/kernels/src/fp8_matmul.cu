#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <cassert>
#include <cstring>
#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"
#include "attention/dtype_fp8.cuh"
#include "attention/dtype_e8m0.cuh"
using namespace nvcuda::wmma;

namespace vllm {
    __forceinline__ __device__ void from_float(half& out, float in) {
        out = __float2half(in);
    }
} // namespace vllm

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

template <typename ScaleT>
__device__ __forceinline__ float get_scale(const ScaleT *__restrict__ scale,
                                           int n, int k, int scale_stride,
                                           int block_size_y, int block_size_x) {
  int sr = n / block_size_y;
  int sc = k / block_size_x;
  return attention_rs::e8m0::read_scale(scale, sr * scale_stride + sc);
}

template <typename T, typename ScaleT, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_kernel(const T *__restrict__ input,
                                 const uint8_t *__restrict__ weight,
                                 const ScaleT *__restrict__ weight_scale,
                                 T *__restrict__ output, int M, int N, int K,
                                 int scale_row_stride, int block_size_y,
                                 int block_size_x) {
  __shared__ float s_input[BLOCK_M][BLOCK_K + 4];
  __shared__ float s_weight[BLOCK_N][BLOCK_K + 4];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = by * BLOCK_M + ty;
  const int col = bx * BLOCK_N + tx;

  float acc = 0.0f;

  const int num_threads = BLOCK_M * BLOCK_N;
  const int tid = ty * BLOCK_N + tx;

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    const bool tile_scale_uniform =
        block_size_x >= BLOCK_K &&
        ((k_tile % block_size_x) + BLOCK_K <= block_size_x);
    const int scale_k_idx_tile = tile_scale_uniform ? (k_tile / block_size_x) : 0;

    for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
      int lm = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gm = by * BLOCK_M + lm;
      int gk = k_tile + lk;

      float val = 0.0f;
      if (gm < M && gk < K) {
        if constexpr (std::is_same_v<T, half>) {
          val = __half2float(__ldg(&input[gm * K + gk]));
        } else {
#ifndef NO_BF16_KERNEL
          val = __bfloat162float(__ldg(&input[gm * K + gk]));
#endif
        }
      }
      s_input[lm][lk] = val;
    }

    // Vectorized FP8 weight loading: process 4 bytes per iteration
    constexpr int WEIGHT_ELEMS = BLOCK_N * BLOCK_K;
    constexpr int VEC4_ITERS = WEIGHT_ELEMS / 4;
    for (int i = tid; i < VEC4_ITERS; i += num_threads) {
      int flat = i * 4;
      int ln = flat / BLOCK_K;
      int lk = flat % BLOCK_K;
      int gn = bx * BLOCK_N + ln;
      int gk_base = k_tile + lk;

      if (gn < N && gk_base + 3 < K) {
        uint32_t w4 = __ldg(reinterpret_cast<const uint32_t*>(&weight[gn * K + gk_base]));
        float s;
        if (tile_scale_uniform) {
          int scale_row = gn / block_size_y;
          s = attention_rs::e8m0::read_scale(
              weight_scale, scale_row * scale_row_stride + scale_k_idx_tile);
        } else {
          s = get_scale(weight_scale, gn, gk_base, scale_row_stride, block_size_y,
                        block_size_x);
        }
        s_weight[ln][lk]     = vllm::fp8::dispatch_fp8_to_float((uint8_t)(w4 & 0xFF)) * s;
        s_weight[ln][lk + 1] = vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 8) & 0xFF)) * s;
        s_weight[ln][lk + 2] = vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 16) & 0xFF)) * s;
        s_weight[ln][lk + 3] = vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 24) & 0xFF)) * s;
      } else {
        for (int d = 0; d < 4; d++) {
          int cur_lk = lk + d;
          int cur_gk = gk_base + d;
          float val = 0.0f;
          if (gn < N && cur_gk < K) {
            uint8_t w_raw = __ldg(&weight[gn * K + cur_gk]);
            float s = 0.0f;
            if (tile_scale_uniform) {
              int scale_row = gn / block_size_y;
              s = attention_rs::e8m0::read_scale(
                  weight_scale, scale_row * scale_row_stride + scale_k_idx_tile);
            } else {
              s = get_scale(weight_scale, gn, cur_gk, scale_row_stride, block_size_y,
                            block_size_x);
            }
            val = vllm::fp8::dispatch_fp8_to_float(w_raw) * s;
          }
          s_weight[ln][cur_lk] = val;
        }
      }
    }
    // Handle remaining elements (when WEIGHT_ELEMS not divisible by 4)
    for (int i = VEC4_ITERS * 4 + tid; i < WEIGHT_ELEMS; i += num_threads) {
      int ln = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gn = bx * BLOCK_N + ln;
      int gk = k_tile + lk;
      float val = 0.0f;
      if (gn < N && gk < K) {
        uint8_t w_raw = __ldg(&weight[gn * K + gk]);
        float s = tile_scale_uniform
            ? attention_rs::e8m0::read_scale(
                  weight_scale, (gn / block_size_y) * scale_row_stride + scale_k_idx_tile)
            : get_scale(weight_scale, gn, gk, scale_row_stride, block_size_y, block_size_x);
        val = vllm::fp8::dispatch_fp8_to_float(w_raw) * s;
      }
      s_weight[ln][lk] = val;
    }

    __syncthreads();

    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < BLOCK_K; k += 4) {
        float4 in4 = *reinterpret_cast<float4*>(&s_input[ty][k]);
        float4 w4 = *reinterpret_cast<float4*>(&s_weight[tx][k]);
        acc = fmaf(in4.x, w4.x, acc);
        acc = fmaf(in4.y, w4.y, acc);
        acc = fmaf(in4.z, w4.z, acc);
        acc = fmaf(in4.w, w4.w, acc);
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    vllm::from_float(output[row * N + col], acc);
  }
}

#define BK 32 

template <typename T, typename ScaleT, int BM, int BN, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void fp8_wmma_matmul(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weight,
    const ScaleT *__restrict__ weight_scale,
    T *__restrict__ output,
    int M, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x) 
{
    // Warps layout:
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Specialized warp mapping
    int warp_row_px, warp_col_px;
    
    // Accumulators
    // Fragments count depends on (BM, BN) vs (WMMA_M, WMMA_N)
    constexpr int FRAGS_M = BM / WMMA_M;
    constexpr int FRAGS_N_PER_WARP = (BN / 4) / WMMA_N; // Assuming 1x4 Grid?
    
    // Let's make the grid explicit per config
    if constexpr (BM == 64 && BN == 64) {
        // Grid 2x2. Warp covers 32x32.
        // WMMA 16x16x16.
        // Frags: 32/16 x 32/16 = 2x2.
        warp_row_px = (warp_id / 2) * 32;
        warp_col_px = (warp_id % 2) * 32;
    } else if constexpr (BM == 16 && BN == 64) {
        // Grid 1x4. Warp covers 16x16. WMMA 16x16. Frags 1x1.
        warp_row_px = 0;             
        warp_col_px = warp_id * 16;  
    } else if constexpr (BM == 16 && BN == 128) {
        // Grid 1x4. Warp covers 16x32. WMMA 16x16. Frags 1x2.
        warp_row_px = 0;
        warp_col_px = warp_id * 32;
    } else if constexpr (BM == 8 && BN == 128) {
        // Grid 1x4. Warp covers 8x32. 
        // using WMMA 8x32. Frags 1x1.
        warp_row_px = 0;
        warp_col_px = warp_id * 32;
    } else {
        warp_row_px = 0; 
        warp_col_px = 0;
    }

    // Number of fragments per thread
    constexpr int NUM_FRAGS_M = (BM == 64) ? 2 : 1;
    constexpr int NUM_FRAGS_N = (BM == 64) ? 2 : (BN == 128 ? 2 : 1);
    // Special case for BM=8, BN=128 with WMMA 8x32 => 1 fragment N
    constexpr int ACTUAL_FRAGS_N = (BM == 8 && BN == 128) ? 1 : NUM_FRAGS_N;
    
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[NUM_FRAGS_M][ACTUAL_FRAGS_N];

    #pragma unroll
    for(int i=0; i<NUM_FRAGS_M; ++i)
        #pragma unroll
        for(int j=0; j<ACTUAL_FRAGS_N; ++j)
            fill_fragment(acc[i][j], 0.0f);

    // Shared Memory
    __shared__ T s_a[BM][BK]; 
    // B is now loaded as row-major chunks, essentially directly copying tiles.
    // Padding to avoid bank conflicts.
    // padding for bank conflicts (must maintain 128-bit/16-byte alignment of rows)
    // BK=32 (aligned). Padding should be 8.
    __shared__ T s_b[BN][BK + 8]; 
    // Output shared memory
    __shared__ float s_out[BM][BN + 8]; 

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tid = threadIdx.x;

    // Loop over K in chunks of BK
    for (int k_step = 0; k_step < K; k_step += BK) {
        // Cooperative Load A -> Smem ---
        #pragma unroll
        for(int i=0; i < (BM * BK) / 128; ++i) {
             int current_idx = i * 128 + tid;
             if (current_idx < BM * BK) {
                 int r = current_idx / BK;
                 int c = current_idx % BK;
                 int gr = by * BM + r;
                 int gc = k_step + c;
                 if(gr < M && gc < K) {
                    s_a[r][c] = input[gr * K + gc];
                 } else {
                    s_a[r][c] = T(0.0);
                 }
             }
        }

        // Weight is N x K. Loading tile [BN, BK].
        // Vectorized: load 4 FP8 bytes at once, dequant and store as half/bf16.
        
        constexpr int TILE_ELEMS = BN * BK;
        constexpr int VEC4_TILE = TILE_ELEMS / 4;
        
        #pragma unroll
        for (int i = tid; i < VEC4_TILE; i += blockDim.x) {
             int flat = i * 4;
             int n_rel = flat / BK;
             int k_rel = flat % BK;
             
             int gn = bx * BN + n_rel;
             int gk_base = k_step + k_rel;
             
             if (gn < N && gk_base + 3 < K) {
                 uint32_t w4 = __ldg(reinterpret_cast<const uint32_t*>(&weight[gn * K + gk_base]));
                 
                 int sr = gn / block_size_y;
                 int sc = gk_base / block_size_x;
                 float s = attention_rs::e8m0::read_scale(
                     weight_scale, sr * scale_row_stride + sc);
                 
                 vllm::from_float(s_b[n_rel][k_rel],     vllm::fp8::dispatch_fp8_to_float((uint8_t)(w4 & 0xFF)) * s);
                 vllm::from_float(s_b[n_rel][k_rel + 1], vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 8) & 0xFF)) * s);
                 vllm::from_float(s_b[n_rel][k_rel + 2], vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 16) & 0xFF)) * s);
                 vllm::from_float(s_b[n_rel][k_rel + 3], vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 24) & 0xFF)) * s);
             } else {
                 for (int d = 0; d < 4; d++) {
                     int cur_k = k_rel + d;
                     int gk = gk_base + d;
                     float val = 0.0f;
                     if (gn < N && gk < K) {
                         uint8_t w = weight[gn * K + gk];
                         int sr2 = gn / block_size_y;
                         int sc2 = gk / block_size_x;
                         float s2 = attention_rs::e8m0::read_scale(
                             weight_scale, sr2 * scale_row_stride + sc2);
                         val = vllm::fp8::dispatch_fp8_to_float(w) * s2;
                     }
                     vllm::from_float(s_b[n_rel][cur_k], val);
                 }
             }
        }
        // Handle remaining elements
        for (int i = VEC4_TILE * 4 + tid; i < TILE_ELEMS; i += blockDim.x) {
             int n_rel = i / BK;
             int k_rel = i % BK;
             int gn = bx * BN + n_rel;
             int gk = k_step + k_rel;
             float val = 0.0f;
             if (gn < N && gk < K) {
                 uint8_t w = weight[gn * K + gk];
                 int sr = gn / block_size_y;
                 int sc = gk / block_size_x;
                 float s = attention_rs::e8m0::read_scale(
                     weight_scale, sr * scale_row_stride + sc);
                 val = vllm::fp8::dispatch_fp8_to_float(w) * s;
             }
             vllm::from_float(s_b[n_rel][k_rel], val);
        }
        
        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag; 

        // Iterate over the loaded K tile (BK=32) in steps of WMMA_K
        #pragma unroll
        for (int k_sub = 0; k_sub < BK; k_sub += WMMA_K) {
             #pragma unroll
            for (int i = 0; i < NUM_FRAGS_M; ++i) {     // Rows of sub-tiles (M)
                #pragma unroll
                for (int j = 0; j < ACTUAL_FRAGS_N; ++j) { // Cols of sub-tiles (N)
                    
                    load_matrix_sync(a_frag, &s_a[warp_row_px + i*WMMA_M][k_sub], BK);

                    // B fragment: Load from s_b[warp_col + j*WMMA_N][k_sub]
                    // s_b is padded [BN][BK+8].
                    load_matrix_sync(b_frag, &s_b[warp_col_px + j*WMMA_N][k_sub], BK + 8);

                    mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NUM_FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < ACTUAL_FRAGS_N; ++j) {
            float *ptr = &s_out[warp_row_px + i * WMMA_M][warp_col_px + j * WMMA_N];
            store_matrix_sync(ptr, acc[i][j], BN + 8, mem_row_major);
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int r = i / BN;
        int c = i % BN;
            
        int global_r = by * BM + r;
        int global_c = bx * BN + c;
            
        if (global_r < M && global_c < N) {
             float val = s_out[r][c];
             vllm::from_float(output[global_r * N + global_c], val);
        }
    }
}

// Channel-wise compressed-tensors FP8 uses one F32 scale for every output
// row: scale[n] applies to all K values of weight[n, :].  The WMMA path above
// ---------------------------------------------------------------------------
// SM70 decode GEMV for channel-wise FP8 (M == 1).
//
// The generic `fp8_matmul_kernel` dequantizes every single weight into FP32
// shared memory using the branchy software FP8->FP32 conversion, which costs
// roughly 25 instructions per weight and leaves the memory system idle: on a
// V100 it sustains 86-96 GB/s while a plain read of the same bytes reaches
// 818 GB/s.
//
// This kernel uses the layout 1Cat's dense SM70 kernels use:
//   * one warp owns one output column and the 32 lanes split K in stripes
//     (lane l takes k-block l, l+32, ...), so the weight and activation loads
//     are fully coalesced and no shared-memory staging is needed;
//   * the weights are decoded straight from global memory into registers with
//     a branch-free bit trick;
//   * the products are accumulated in FP32, which is exact for F16 x F16.
//
// The result is bit-identical to `fp8_matmul_kernel` for channel-wise scales
// (verified against a double-precision CPU reference on the 27B shapes).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fp8_e4m3x8_to_half2x4(uint2 q, uint32_t out[4]) {
  // E4M3 (OCP) -> F16.  The plain trick `((code & 0x7f) << 7)` yields the
  // value scaled by 2^-8 (1Cat compensates by folding 256 into the block
  // scale); adding 0x2000 restores the exponent bias so that the F16
  // exponent field ends up as (E4M3 exponent + 8).  The addition cannot carry
  // past bit 14 because that bit is zero before it.
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kEM = 0x3f803f80u;    // F16 exponent field + top mantissa
  constexpr unsigned kBias = 0x20002000u;  // exponent bias fix-up
  unsigned p[4];
  p[0] = __byte_perm(q.x, q.y, 0x0400);
  p[1] = __byte_perm(q.x, q.y, 0x0501);
  p[2] = __byte_perm(q.x, q.y, 0x0602);
  p[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = ((p[i] << 8) & kSign) | ((((p[i] << 7) & kEM) + kBias) & ~kSign);
  }
}

__device__ __forceinline__ float2 fp8_half2_to_float2(uint32_t v) {
  return __half22float2(*reinterpret_cast<const __half2 *>(&v));
}

// ---------------------------------------------------------------------------
// SM70 batched-decode GEMV: 1Cat's fp8_qpn8_sm70_kernel (RowTiles=2) copied
// verbatim, with two adaptations: the channel-wise scale is F32 (this
// checkpoint's layout) instead of F16, and grid.y splits M into 16-row blocks.
// The decoder is the existing bit-identical `fp8_e4m3x8_to_half2x4` (its
// kBias fold replaces 1Cat's "256 folded into the scale" convention).
// ---------------------------------------------------------------------------
#define SM70_QPN8_MMA_8N8K4(C, A0, A1, B0, B1)                           \
  asm volatile(                                                          \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                 \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                   \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                      \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]),                    \
        "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                    \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

template <int SplitK, int NAcc, int RowTiles>
__global__ void fp8_qpn8_sm70_kernel(const uint8_t *__restrict__ codes,
                                     const float *__restrict__ scales,
                                     const half *__restrict__ input,
                                     half *__restrict__ output, int n, int k,
                                     int m) {
  __shared__ float partials[SplitK][RowTiles * 256];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int row0 = blockIdx.y * (RowTiles * 8);
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const uint4 *code_ptr = reinterpret_cast<const uint4 *>(codes) +
                         (size_t)tile * groups_k16 * 32 + lane;
  const float loaded_scale =
      __ldg(scales + tile * 32 + qpn8_col_from_lane(lane));

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int rt = 0; rt < RowTiles; ++rt)
    for (int c = 0; c < NAcc; ++c)
      for (int i = 0; i < 8; ++i) accum[rt][c][i] = 0.0f;

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint4 packed = __ldcs(code_ptr + (size_t)group * 32);
    half2 weights[8];
    fp8_e4m3x8_to_half2x4(make_uint2(packed.x, packed.y),
                          reinterpret_cast<uint32_t *>(weights));
    fp8_e4m3x8_to_half2x4(make_uint2(packed.z, packed.w),
                          reinterpret_cast<uint32_t *>(weights) + 4);
    const half hs = __float2half_rn(loaded_scale);
        const half2 scale2 = __halves2half2(hs, hs);
#pragma unroll
    for (int i = 0; i < 8; ++i) weights[i] = __hmul2(weights[i], scale2);
    const unsigned *b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
    for (int rt = 0; rt < RowTiles; ++rt) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int input_row_idx = row0 + rt * 8 + row;
      if (input_row_idx < m) {
        const half *ir = input + (size_t)input_row_idx * k;
        input01 = *reinterpret_cast<const uint4 *>(ir + group * 16);
        input23 = *reinterpret_cast<const uint4 *>(ir + group * 16 + 8);
      }
      const unsigned *a0 = reinterpret_cast<const unsigned *>(&input01);
      const unsigned *a1 = reinterpret_cast<const unsigned *>(&input23);
      SM70_QPN8_MMA_8N8K4(accum[rt][0], a0[0], a0[1], b[0], b[1]);
      SM70_QPN8_MMA_8N8K4(accum[rt][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      SM70_QPN8_MMA_8N8K4(accum[rt][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      SM70_QPN8_MMA_8N8K4(accum[rt][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int rt = 0; rt < RowTiles; ++rt)
    for (int c = 1; c < NAcc; ++c)
      for (int i = 0; i < 8; ++i) accum[rt][0][i] += accum[rt][c][i];

#pragma unroll
  for (int rt = 0; rt < RowTiles; ++rt)
    for (int i = 0; i < 8; ++i) {
      const int output_row = rt * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[rt][0][i];
    }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int kw = 0; kw < SplitK; ++kw) value += partials[kw][element];
    const int output_row = element >> 5;
    const int output_col = element & 31;
    if (row0 + output_row < m) {
      output[(size_t)(row0 + output_row) * n + tile * 32 + output_col] =
          __float2half(value);
    }
  }
}

// first converts each dequantized weight to T in shared memory.  That is a
// precision loss for this format (especially when T is BF16), and it is
// needlessly repeated for every K tile.  Keep this path on the FP32
// dequantizing kernel, whose accumulator already stays in FP32, and only
// round once when writing the requested output dtype.
template <typename T>
static void fp8_matmul_channelwise_impl(
    const T *input, const uint8_t *weight, const float *weight_scale,
    T *output, int M, int N, int K, cudaStream_t stream) {
  // SM70 (1Cat QPN8 semantics): the copied RowTiles=2 kernel decodes each
  // FP8 weight byte once with the branch-free bit trick and reuses it for
  // 16 activation rows, so batched decode (2 <= M < 256) streams weights at
  // the same rate as M == 1. M >= 256 keeps the dequant->cublas route
  // (fp8_linear.rs).
  if constexpr (std::is_same_v<T, __half>) {
    if (N % 32 == 0 && K % 128 == 0) {
      fp8_qpn8_sm70_kernel<8, 4, 2><<<dim3(N / 32, CEILDIV(M, 16)), 32 * 8, 0,
                                     stream>>>(weight, weight_scale, input,
                                               output, N, K, M);
      return;
    }
  }
  constexpr int BM = 8;
  constexpr int BN = 64;
  constexpr int BK_S = 32;
  dim3 block(BN, BM);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
  fp8_matmul_kernel<T, float, BM, BN, BK_S>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                   1, 1, K);
}

extern "C" void fp8_matmul_f16_channelwise(
    const __half *input, const uint8_t *weight, const float *weight_scale,
    __half *output, int M, int N, int K, cudaStream_t stream) {
  fp8_matmul_channelwise_impl(input, weight, weight_scale, output, M, N, K, stream);
}

// ---------------------------------------------------------------------------
// SM70 prefill dequant (1Cat QPN8 semantics): channel-wise FP8 -> dense F16.
// The generic channel-wise GEMM stages every weight into FP32 shared memory
// with a branchy conversion and re-reads tiles for every M block, so at
// prefill M it sustains only ~90 GB/s. Dequantizing once into an F16 buffer
// and running the plain tensor-core F16 GEMM measured ~103 TF/s.
// Requires K % 8 == 0; each element handles 8 packed bytes.
// ---------------------------------------------------------------------------
__global__ void fp8_dequant_channelwise_f16_kernel(
    const uint8_t *__restrict__ w, const float *__restrict__ scale,
    __half *__restrict__ out, long long numel8, int k_over_8) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < numel8;
       i += stride) {
    float s = scale[i / k_over_8];
    uint2 w8 = reinterpret_cast<const uint2 *>(w)[i];
    const uint8_t *b = reinterpret_cast<const uint8_t *>(&w8);
    __half2 h[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
      h[j] = __floats2half2_rn(vllm::fp8::dispatch_fp8_to_float(b[2 * j]) * s,
                               vllm::fp8::dispatch_fp8_to_float(b[2 * j + 1]) * s);
    reinterpret_cast<uint4 *>(out)[i] = *reinterpret_cast<uint4 *>(&h[0]);
  }
}

extern "C" void fp8_dequant_channelwise_f16(const uint8_t *w, const float *scale,
                                            __half *out, long long numel8,
                                            int k_over_8, cudaStream_t stream) {
  int block = 256;
  long long total = numel8;
  int grid = (int)min((long long)1024, (total + block - 1) / block);
  fp8_dequant_channelwise_f16_kernel<<<grid, block, 0, stream>>>(w, scale, out, numel8, k_over_8);
}

__global__ void fp8_gemv_sm70_kernel(const __half *__restrict__ input,
                                     const uint8_t *__restrict__ weight,
                                     const float *__restrict__ weight_scale,
                                     __half *__restrict__ output, int N, int K) {
  const int n = blockIdx.x;
  const int lane = threadIdx.x;
  const uint8_t *wrow = weight + (size_t)n * K;
  const float scale = weight_scale[n];
  const int blocks = K >> 4;  // 16 elements per k-block

  float acc = 0.0f;
  for (int kb = lane; kb < blocks; kb += 32) {
    const uint4 wv = *reinterpret_cast<const uint4 *>(wrow + kb * 16);
    uint32_t b[8];
    fp8_e4m3x8_to_half2x4(make_uint2(wv.x, wv.y), b);      // codes 0..7
    fp8_e4m3x8_to_half2x4(make_uint2(wv.z, wv.w), b + 4);  // codes 8..15
    // The decoder emits (code_i, code_{i+4}) pairs; re-pair them into k order.
    uint32_t p[8];
    p[0] = (b[0] & 0xffffu) | (b[1] << 16);
    p[1] = (b[2] & 0xffffu) | (b[3] << 16);
    p[2] = (b[0] >> 16) | (b[1] & 0xffff0000u);
    p[3] = (b[2] >> 16) | (b[3] & 0xffff0000u);
    p[4] = (b[4] & 0xffffu) | (b[5] << 16);
    p[5] = (b[6] & 0xffffu) | (b[7] << 16);
    p[6] = (b[4] >> 16) | (b[5] & 0xffff0000u);
    p[7] = (b[6] >> 16) | (b[7] & 0xffff0000u);

    const uint4 a0v = *reinterpret_cast<const uint4 *>(input + kb * 16);
    const uint4 a1v = *reinterpret_cast<const uint4 *>(input + kb * 16 + 8);

    const float2 a0 = fp8_half2_to_float2(a0v.x);
    const float2 a1 = fp8_half2_to_float2(a0v.y);
    const float2 a2 = fp8_half2_to_float2(a0v.z);
    const float2 a3 = fp8_half2_to_float2(a0v.w);
    const float2 a4 = fp8_half2_to_float2(a1v.x);
    const float2 a5 = fp8_half2_to_float2(a1v.y);
    const float2 a6 = fp8_half2_to_float2(a1v.z);
    const float2 a7 = fp8_half2_to_float2(a1v.w);

    const float2 w0 = fp8_half2_to_float2(p[0]);
    const float2 w1 = fp8_half2_to_float2(p[1]);
    const float2 w2 = fp8_half2_to_float2(p[2]);
    const float2 w3 = fp8_half2_to_float2(p[3]);
    const float2 w4 = fp8_half2_to_float2(p[4]);
    const float2 w5 = fp8_half2_to_float2(p[5]);
    const float2 w6 = fp8_half2_to_float2(p[6]);
    const float2 w7 = fp8_half2_to_float2(p[7]);

    acc = fmaf(a0.x, w0.x, acc);
    acc = fmaf(a0.y, w0.y, acc);
    acc = fmaf(a1.x, w1.x, acc);
    acc = fmaf(a1.y, w1.y, acc);
    acc = fmaf(a2.x, w2.x, acc);
    acc = fmaf(a2.y, w2.y, acc);
    acc = fmaf(a3.x, w3.x, acc);
    acc = fmaf(a3.y, w3.y, acc);
    acc = fmaf(a4.x, w4.x, acc);
    acc = fmaf(a4.y, w4.y, acc);
    acc = fmaf(a5.x, w5.x, acc);
    acc = fmaf(a5.y, w5.y, acc);
    acc = fmaf(a6.x, w6.x, acc);
    acc = fmaf(a6.y, w6.y, acc);
    acc = fmaf(a7.x, w7.x, acc);
    acc = fmaf(a7.y, w7.y, acc);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) output[n] = __float2half(acc * scale);
}

// Multi-row sibling of the M == 1 GEMV: one warp owns one output column and
// produces up to 8 activation rows. The FP8 weight row is decoded once per
// k-block and reused for every row, so batched decode (2 <= M < 256) streams
// the weights at the same rate as M == 1 instead of paying the tile-kernel
// latency bound. Row blocks beyond M are masked at the store.
__global__ void fp8_gemv_sm70_m_kernel(const __half *__restrict__ input,
                                     const uint8_t *__restrict__ weight,
                                     const float *__restrict__ weight_scale,
                                     __half *__restrict__ output, int N,
                                        int K, int m) {
  const int n = blockIdx.x;
  const int row0 = blockIdx.y * 8;
  const int lane = threadIdx.x;
  const uint8_t *wrow = weight + (size_t)n * K;
  const float scale = weight_scale[n];
  const int blocks = K >> 4;  // 16 elements per k-block

  float acc[8];
#pragma unroll
  for (int r = 0; r < 8; ++r) acc[r] = 0.0f;
  for (int kb = lane; kb < blocks; kb += 32) {
    const uint4 wv = *reinterpret_cast<const uint4 *>(wrow + kb * 16);
    uint32_t b[8];
    fp8_e4m3x8_to_half2x4(make_uint2(wv.x, wv.y), b);      // codes 0..7
    fp8_e4m3x8_to_half2x4(make_uint2(wv.z, wv.w), b + 4);  // codes 8..15
    // The decoder emits (code_i, code_{i+4}) pairs; re-pair them into k order.
    uint32_t p[8];
    p[0] = (b[0] & 0xffffu) | (b[1] << 16);
    p[1] = (b[2] & 0xffffu) | (b[3] << 16);
    p[2] = (b[0] >> 16) | (b[1] & 0xffff0000u);
    p[3] = (b[2] >> 16) | (b[3] & 0xffff0000u);
    p[4] = (b[4] & 0xffffu) | (b[5] << 16);
    p[5] = (b[6] & 0xffffu) | (b[7] << 16);
    p[6] = (b[4] >> 16) | (b[5] & 0xffff0000u);
    p[7] = (b[6] >> 16) | (b[7] & 0xffff0000u);

    const int rows = (m - row0 < 8) ? (m - row0) : 8;
    for (int r = 0; r < rows; ++r) {
    const half *ir = input + (size_t)(row0 + r) * K + kb * 16;
    const uint4 a0v = *reinterpret_cast<const uint4 *>(ir);
    const uint4 a1v = *reinterpret_cast<const uint4 *>(ir + 8);

    const float2 a0 = fp8_half2_to_float2(a0v.x);
    const float2 a1 = fp8_half2_to_float2(a0v.y);
    const float2 a2 = fp8_half2_to_float2(a0v.z);
    const float2 a3 = fp8_half2_to_float2(a0v.w);
    const float2 a4 = fp8_half2_to_float2(a1v.x);
    const float2 a5 = fp8_half2_to_float2(a1v.y);
    const float2 a6 = fp8_half2_to_float2(a1v.z);
    const float2 a7 = fp8_half2_to_float2(a1v.w);

    const float2 w0 = fp8_half2_to_float2(p[0]);
    const float2 w1 = fp8_half2_to_float2(p[1]);
    const float2 w2 = fp8_half2_to_float2(p[2]);
    const float2 w3 = fp8_half2_to_float2(p[3]);
    const float2 w4 = fp8_half2_to_float2(p[4]);
    const float2 w5 = fp8_half2_to_float2(p[5]);
    const float2 w6 = fp8_half2_to_float2(p[6]);
    const float2 w7 = fp8_half2_to_float2(p[7]);

    acc[r] = fmaf(a0.x, w0.x, acc[r]);
    acc[r] = fmaf(a0.y, w0.y, acc[r]);
    acc[r] = fmaf(a1.x, w1.x, acc[r]);
    acc[r] = fmaf(a1.y, w1.y, acc[r]);
    acc[r] = fmaf(a2.x, w2.x, acc[r]);
    acc[r] = fmaf(a2.y, w2.y, acc[r]);
    acc[r] = fmaf(a3.x, w3.x, acc[r]);
    acc[r] = fmaf(a3.y, w3.y, acc[r]);
    acc[r] = fmaf(a4.x, w4.x, acc[r]);
    acc[r] = fmaf(a4.y, w4.y, acc[r]);
    acc[r] = fmaf(a5.x, w5.x, acc[r]);
    acc[r] = fmaf(a5.y, w5.y, acc[r]);
    acc[r] = fmaf(a6.x, w6.x, acc[r]);
    acc[r] = fmaf(a6.y, w6.y, acc[r]);
    acc[r] = fmaf(a7.x, w7.x, acc[r]);
    acc[r] = fmaf(a7.y, w7.y, acc[r]);
  }

#pragma unroll
  for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
  if (lane == 0) {
    const int rows = (m - row0 < 8) ? (m - row0) : 8;
    for (int r = 0; r < rows; ++r)
      output[(size_t)(row0 + r) * N + n] = __float2half(acc[r] * scale);
  }
}

}extern "C" void fp8_gemv_sm70_m_f16(const __half *input, const uint8_t *weight,
                                    const float *weight_scale, __half *output,
                                    int N, int K, int m, cudaStream_t stream) {
  fp8_gemv_sm70_m_kernel<<<dim3(N, CEILDIV(m, 8)), 32, 0, stream>>>(
      input, weight, weight_scale, output, N, K, m);
}

extern "C" void fp8_gemv_sm70_f16(const __half *input, const uint8_t *weight,
                                  const float *weight_scale, __half *output,
                                  int N, int K, cudaStream_t stream) {
  if (N <= 0 || K <= 0 || (K & 15) != 0) return;
  fp8_gemv_sm70_kernel<<<N, 32, 0, stream>>>(input, weight, weight_scale,
                                              output, N, K);
}

extern "C" void fp8_matmul_f16(const __half *input, const uint8_t *weight,
                        const void *weight_scale, __half *output, int M,
                        int N, int K, int scale_row_stride, int block_size_y,
                        int block_size_x, int scale_dtype, cudaStream_t stream) {

  if (scale_dtype == 1) { // e8m0
    const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__half, uint8_t, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__half, uint8_t, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  } else { // f32
    const float* ws = reinterpret_cast<const float*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__half, float, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__half, float, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  }
}

extern "C" void fp8_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                        const void *weight_scale, __nv_bfloat16 *output, int M,
                        int N, int K, int scale_row_stride, int block_size_y,
                        int block_size_x, int scale_dtype, cudaStream_t stream) {

#ifndef NO_BF16_KERNEL
  if (scale_dtype == 1) { // e8m0
    const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__nv_bfloat16, uint8_t, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__nv_bfloat16, uint8_t, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  } else { // f32
    const float* ws = reinterpret_cast<const float*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__nv_bfloat16, float, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__nv_bfloat16, float, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  }
#endif
}

extern "C" void fp8_matmul_bf16_channelwise(
    const __nv_bfloat16 *input, const uint8_t *weight, const float *weight_scale,
    __nv_bfloat16 *output, int M, int N, int K, cudaStream_t stream) {
#ifndef NO_BF16_KERNEL
  fp8_matmul_channelwise_impl(input, weight, weight_scale, output, M, N, K, stream);
#endif
}
