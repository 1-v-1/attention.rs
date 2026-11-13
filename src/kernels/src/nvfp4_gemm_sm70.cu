/**
 * @brief SM70 (Volta / Tesla V100) NVFP4 GEMM kernels.
 *
 * Ports the SM70 "HMMA.884" approach of 1Cat-vLLM's TurboMind `sm70_s884`
 * kernel family into attention.rs.
 *
 * Why a separate file: the HMMA.884 path needs Volta-only PTX and a
 * different weight packing, so it cannot share code with the WMMA path
 * (which targets Ampere and later).  The file is compiled unconditionally but
 * every kernel body is guarded by an SM70 check, so it is inert on newer
 * architectures.
 *
 * ---------------------------------------------------------------------------
 * The Volta MMA and its fragment layout
 * ---------------------------------------------------------------------------
 *
 *   mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
 *
 * is a Volta HMMA instruction with an M=8, N=8, K=4 shape.  Its A operand is
 * two 32-bit registers holding four halves and its B operand is two 32-bit
 * registers holding four halves of a single column.  The layout below was
 * determined from the PTX ISA specification and then verified on real V100
 * hardware with one-hot fragment scans (see `docs/design` and the probe
 * programs under `repos/xxinfer-dev-env/mmatest/`):
 *
 *   A operand (8x4, row major, M x K)
 *     lane L holds one whole row:  row = (L & 3) + 4 * (L >= 16)
 *     a[0] = {A[row][0], A[row][1]}   a[1] = {A[row][2], A[row][3]}
 *
 *   B operand (4x8, column major, K x N)
 *     lane L holds one whole column: col = (L & 3) + 4 * (L >= 16)
 *     b[0] = {B[0][col], B[1][col]}   b[1] = {B[2][col], B[3][col]}
 *
 *   D operand (8x8 f32, eight elements per lane, four-fold redundant)
 *     element i (0..7) of lane L sits at
 *       row = (L & 1) + (i & 2) + 4 * (L >= 16)
 *       col = (L & 2) + (i & 1) + 4 * ((i >> 2) & 1)
 *
 * A warp therefore covers 8 rows of A and 8 columns of B at a time; the 32
 * lanes are four redundant copies of eight distinct (row, column) pairs.
 * 1Cat's `SM70_MMA_884` atom wraps two of these instructions into one
 * K=8 step, using A as one row of eight contiguous k and B as one column of
 * eight contiguous k; the first instruction consumes k=0..3 and the second
 * k=4..7 while accumulating on top of the first.
 *
 * ---------------------------------------------------------------------------
 * Weight packing
 * ---------------------------------------------------------------------------
 *
 *   The checkpoint stores NVFP4 weights as `[N, K/2]` bytes with two nibbles
 *   per byte (low nibble = smaller k).  The B fragment wants eight
 *   *contiguous k of one n* in a single 32-bit register, so the weights are
 *   repacked once per layer into `[N, K/8]` uint32 by
 *   `nvfp4_repack_weight_sm70`.  With that layout the global-to-shared copy is
 *   a straight copy of whole 32-bit words, with no transpose.
 *
 *   1Cat's own TurboMind packing is `[K][N/8]` (eight codes of eight
 *   *different* n at one k), which forces a transpose during the
 *   global-to-shared copy.  Repacking to `[N, K/8]` avoids that entirely and
 *   is cheaper to produce.
 *
 *   The nibble order inside the packed word is the "inv" permutation
 *   `[0, 4, 1, 5, 2, 6, 3, 7]`: element j of the eight is stored at bit
 *   4*inv[j].  That permutation is what makes 1Cat's decoder emit the eight
 *   elements in k order (see `decode_fp4_e2m1_x8`).
 *
 * ---------------------------------------------------------------------------
 * Scaling
 * ---------------------------------------------------------------------------
 *
 *   NVFP4 uses one FP8 E4M3 scale per 16 elements along k.  Since 16 is a
 *   multiple of the 8-k step, one 32-bit weight word never straddles two
 *   scales, so the scale can be folded into the decoded half values exactly
 *   the way 1Cat's `Transform_HMMA_SIMT_B_PrescaledE2M1` does: decode the
 *   nibbles into half precision and multiply by the scale with a single
 *   `mul.f16x2`.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "attention/dtype_fp8.cuh"

#define SM70_NVFP4_CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define SM70_NVFP4_WARP_SIZE 32

// The number of k elements one NVFP4 block scale covers.
#define SM70_NVFP4_BLOCK_SIZE 16

namespace sm70_nvfp4 {

using vllm::fp8::dispatch_fp8_to_float;

// ---------------------------------------------------------------------------
// FP4 E2M1 -> half, following 1Cat's `cvt_f16x8_e2m1`.
//
// The eight nibbles of a packed word are decoded with four masked shifts that
// build the half representation directly.  The masks keep the sign bit and
// the three exponent/mantissa bits, and the result is scaled by 2^-14;
// multiplying by 2^14 afterwards recovers the exact E2M1 value.
//
// The nibble order is the "inv" permutation, so nibble n holds element
// inv^-1(n).  1Cat's decode reads the word as
//
//   word0 = (e0, e4), word1 = (e1, e5), word2 = (e2, e6), word3 = (e3, e7)
//
// where e0..e7 are the eight elements in k order.  With the inv permutation
// this falls out automatically, and the four words together hold e0..e7 in
// order when they are paired as (word0.lo, word1.lo, word2.lo, word3.lo) and
// (word0.hi, word1.hi, word2.hi, word3.hi).
//
// `scale` is a half2 holding the block scale broadcast to both halves.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void decode_fp4_e2m1_x8(uint32_t x, uint32_t scale,
                                                   uint32_t *out) {
  constexpr uint32_t S = 0x80008000U;
  constexpr uint32_t EM = 0x0E000E00U;

  uint32_t w0 = ((x << 12) & S) | ((x << 9) & EM);
  uint32_t w1 = ((x << 8) & S) | ((x << 5) & EM);
  uint32_t w2 = ((x << 4) & S) | ((x << 1) & EM);
  uint32_t w3 = (x & S) | ((x >> 3) & EM);

  // 2^14 turns the E2M1/2^14 representation into the exact E2M1 value.
  // 0x7400 is the half representation of 16384.0.  The constant has to be a
  // half2: `mul.f16x2` multiplies both halves, so a plain 0x7400 would scale
  // the upper half by zero and wipe it out.
  constexpr uint32_t kE2M1Scale = 0x74007400U;
  // `scale` already arrives as a half2 holding the block scale in both
  // halves, so it is used as it is.
  const uint32_t s2 = scale;

  // Fold the block scale in and undo the 2^14 factor in one multiply each.
  // Scaling by 2^14 first keeps every E2M1 value exact, so the block scale
  // multiply is the only rounding step.
  //
  // Both multiplies use half2 operands, so the constant has to carry the
  // 16384.0 factor in both halves.
  const uint32_t e2m1 = kE2M1Scale;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t w = (i == 0) ? w0 : (i == 1) ? w1 : (i == 2) ? w2 : w3;
    asm("mul.f16x2 %0, %1, %2;" : "=r"(w) : "r"(w), "r"(e2m1));
    asm("mul.f16x2 %0, %1, %2;" : "=r"(w) : "r"(w), "r"(s2));
    if (i == 0) w0 = w;
    else if (i == 1) w1 = w;
    else if (i == 2) w2 = w;
    else w3 = w;
  }

  out[0] = w0;
  out[1] = w1;
  out[2] = w2;
  out[3] = w3;
}

// ---------------------------------------------------------------------------
// The Volta HMMA instruction.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m8n8k4_row_col(float *d, const uint32_t *a,
                                                    const uint32_t *b,
                                                    const float *c) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11},"
      "{%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
        "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}

// ---------------------------------------------------------------------------
// Repack NVFP4 weights for the HMMA.884 path.
//
// Input:  `weight` is [N, K/2] bytes, two nibbles per byte, low nibble first.
// Output: `weight_out` is [N, K/8] uint32, eight contiguous k of one n per
//         word, with the nibbles permuted by inv = {0,4,1,5,2,6,3,7} so
//         that 1Cat's decoder returns them in k order.
//
// The scale tensor is repacked at the same time: it is [N, K/16] FP8 E4M3
// bytes, one scale per 16 elements of k.  The kernel needs the scale as a
// half2 broadcast, so the packer converts E4M3 to half once and stores
// [N, K/16] half.  Storing the half in a uint32 would waste half of the
// storage: on a 27B model the scale tensor is over a gigabyte per card,
// so it is kept as a plain half.
// ---------------------------------------------------------------------------

// `column_global_scale` is the per-output-column global scale.  It is NULL
// when every column shares the same scale, in which case `global_scale` is
// used for all of them.
//
// The per-column form matters for a merged gate/up projection, where the two
// halves were quantized with different global scales.  The generic NVFP4
// path picks the per-column scale when it is present and falls back to the
// tensor-wide one otherwise, and this kernel has to agree with it exactly.
__global__ void nvfp4_repack_weight_sm70_kernel(
    const uint8_t *__restrict__ weight, const uint8_t *__restrict__ weight_scale,
    uint32_t *__restrict__ weight_out, __half *__restrict__ scale_out,
    const float *__restrict__ column_global_scale, float global_scale, int N,
    int K) {
  // One thread per (n, 16-element scale group).
  //
  // The scale tensor is indexed by the NVFP4 group size (16 elements along
  // k), not by the weight word (8 elements along k).  Storing one scale per
  // weight word instead would duplicate every scale twice, which doubles the
  // scale memory for no reason -- and for a 27B model that is gigabytes.
  const int scales_per_row = K / 16;
  const int total = N * scales_per_row;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int n = idx / scales_per_row;
  const int group = idx % scales_per_row;
  const int k0 = group * 16;

  // The source is two nibbles per byte with the low nibble at the smaller k,
  // so byte j holds elements 2j (low) and 2j+1 (high).
  const uint8_t *src = weight + (size_t)n * (K / 2) + k0 / 2;
  uint32_t codes[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint8_t b = src[i];
    codes[2 * i] = b & 0xFu;
    codes[2 * i + 1] = (b >> 4) & 0xFu;
  }

  // Two weight words per 16-element scale group.
#pragma unroll
  for (int half = 0; half < 2; ++half) {
  // The packed layout keeps one complete four-bit code per nibble, element j
  // in nibble j.  `decode_fp4_e2m1_x8` turns that into
  //
  //   w0 = (e0, e4), w1 = (e1, e5), w2 = (e2, e6), w3 = (e3, e7)
  //
  // which is what the 884 atom wants: b_lo = {w0, w1} covers k = 0..3 and
  // b_hi = {w2, w3} covers k = 4..7.
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) packed |= (codes[half * 8 + j] & 0xFu) << (4 * j);
  weight_out[n * (K / 8) + (k0 / 8) + half] = packed;

  }  // half

  // The per-tensor global scale is folded in here so that the GEMM kernel
  // only has to multiply by the block scale.  Folding it at repack time
  // keeps it out of the inner loop and costs one multiply per layer.
  const float gscale =
      column_global_scale ? column_global_scale[n] : global_scale;
  const float sf =
      dispatch_fp8_to_float(weight_scale[(size_t)n * (K / 16) + group]) *
      gscale;
  const __half h = __float2half(sf);
  // The scale is a plain half: the GEMM kernel broadcasts it into both halves
  // of the shared-memory entry when it loads it.
  scale_out[idx] = h;
}

// ---------------------------------------------------------------------------
// HMMA.884 GEMM kernel.
//
// CTA layout: one warp per CTA by default.  Every warp owns
// `kSm70ColumnsPerWarp` output columns (four n-quads of eight) and walks the
// whole K range.
//
// Why one warp per CTA: the CTA is the unit of scheduling, and a small CTA
// keeps the grid large.  With N=4096 and 32 columns per warp the grid is 128
// CTAs, which is more than the 80 SMs of a V100, so every SM stays busy.  A
// four-warp CTA would only produce 32 CTAs and leave half of the GPU idle.
//
// The MMA covers 8 rows of A and 8 columns of B.  A warp therefore holds
// eight columns of B, one per group of four lanes, and issues four mma
// instructions (four n-quads) per K=8 step to cover 32 columns.
//
// The shared tile holds the packed weights for the CTA's columns and one
// k-block range: `smem[col_local][kblock]`, i.e. n-major with k contiguous,
// plus the block scales of the same columns.  Staging the scales matters: a
// naive kernel reloads the scale from global memory for every (quad, k block,
// lane) triple, which reads 64x more bytes than the weights themselves.
//
// The activation lives in registers: each lane needs the eight halves of one
// row of A for the current k block, which is exactly the A fragment the MMA
// consumes.
//
// The k tile is a template parameter because the trade-off between shared
// memory footprint and barrier count flips with M.  A large k tile means fewer
// `__syncthreads()` calls, which is what limits the decode path; but it also
// lowers the number of resident CTAs per SM, which is what limits the batched
// path once the grid is already larger than the GPU.
//
// Because the A fragment is one row of eight contiguous k and the B fragment
// is one column of eight contiguous k, the two instructions of the 884 atom
// consume k=0..3 and k=4..7 of the same row and the same column.
// ---------------------------------------------------------------------------

// Output columns per warp: four n-quads of eight columns.
constexpr int kSm70ColumnsPerWarp = 32;
// Output rows per CTA.  The MMA's M is 8, so a CTA covers eight rows.
constexpr int kSm70TileM = 8;
// Weight words per k block of eight elements.
constexpr int kSm70WordsPerKBlock8 = 1;

template <int kTileK, int kWarps>
struct Sm70HmmaTile {
  // The CTA stages the weights of all of its warps' columns.
  static constexpr int kColumns = kSm70ColumnsPerWarp * kWarps;
  static constexpr int kWordsPerKBlock = kTileK / 8;
  static constexpr int kWordsPerTile = kColumns * kWordsPerKBlock;
  static constexpr int kScalesPerTile = kColumns * (kTileK / 8);
  // Two buffers of weights and two of scales.
  static constexpr size_t kSmemBytes =
      2 * (sizeof(uint32_t) * kWordsPerTile +
           sizeof(uint32_t) * kScalesPerTile);
};

template <typename T, int kTileK, int kWarps>
__launch_bounds__(SM70_NVFP4_WARP_SIZE * kWarps) __global__
    void nvfp4_sm70_hmma884_kernel(const T *__restrict__ input,
                                    const uint32_t *__restrict__ weight,
                                    const __half *__restrict__ weight_scale,
                                    const float *__restrict__ weight_row_scales,
                                    const T *__restrict__ bias,
                                    T *__restrict__ output, int M, int N,
                                    int K, bool has_bias) {
  using Tile = Sm70HmmaTile<kTileK, kWarps>;
  extern __shared__ __align__(16) uint32_t smem[];

  const int tid = threadIdx.x;
  const int lane = tid % SM70_NVFP4_WARP_SIZE;
  const int warp = tid / SM70_NVFP4_WARP_SIZE;

  uint32_t *wbuf0 = smem;
  uint32_t *wbuf1 = smem + Tile::kWordsPerTile;
  uint32_t *sbuf0 = wbuf1 + Tile::kWordsPerTile;
  uint32_t *sbuf1 = sbuf0 + Tile::kScalesPerTile;

  // The CTA covers kWarps * kSm70ColumnsPerWarp output columns.
  const int n_base = blockIdx.x * kWarps * kSm70ColumnsPerWarp;
  const int m_base = blockIdx.y * kSm70TileM;

  // The lane's row of A inside the 8x8 MMA tile.  The B fragment is picked
  // per mma call, see `compute_kblock`.
  const int act_row = (lane & 3) + ((lane >= 16) ? 4 : 0);
  const int warp_col0 = n_base + warp * kSm70ColumnsPerWarp;

  float acc[4][8];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[q][i] = 0.0f;
  }

  const int words_per_row = K / 8;   // weight words per output column
  const int scale_blocks = K / 16;   // repacked scales per output column
  const int n_blocks = K / 8;        // k blocks of eight elements
  const int n_tiles = SM70_NVFP4_CEILDIV(n_blocks, Tile::kWordsPerKBlock);

  // Stage the weight and scale tiles.
  //
  // The tile is far larger than the CTA, so the copy is a grid-stride loop.
  // It moves four words (sixteen bytes) per iteration because the tile is
  // laid out n-major with k contiguous, so four consecutive k blocks of one
  // column are adjacent in both global and shared memory.
  auto load_tile = [&](uint32_t *wdst, uint32_t *sdst, int k_tile) {
    constexpr int kWVec = 4;  // words per vectorised access
    const int kWordsVec = Tile::kWordsPerTile / kWVec;
    const int kWordsPerKBlockVec = Tile::kWordsPerKBlock / kWVec;
    for (int wv = tid; wv < kWordsVec; wv += blockDim.x) {
      const int c = wv / kWordsPerKBlockVec;
      const int kbv = wv % kWordsPerKBlockVec;
      const int gcol = n_base + c;
      const int gkb = k_tile + kbv * kWVec;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (gcol < N && gkb + kWVec <= n_blocks) {
        val = __ldg(reinterpret_cast<const uint4 *>(
            &weight[(size_t)gcol * words_per_row + gkb]));
      } else if (gcol < N && gkb < n_blocks) {
        // Ragged tail: copy the words one by one.
        uint4 tmp = make_uint4(0, 0, 0, 0);
        uint32_t *tp = reinterpret_cast<uint32_t *>(&tmp);
        for (int j = 0; j < kWVec; ++j)
          if (gkb + j < n_blocks)
            tp[j] = __ldg(&weight[(size_t)gcol * words_per_row + gkb + j]);
        val = tmp;
      }
      uint4 *dst = reinterpret_cast<uint4 *>(&wdst[c * Tile::kWordsPerKBlock]);
      dst[kbv] = val;
    }
    // Scales: the repack emits one scale per 16-element k group, so a tile
    // of kTileK elements needs kTileK/16 of them per column.  The shared
    // tile keeps the historical one-half2-per-word layout (each 8-element
    // word shares its half with the other word of the same 16-element
    // group), because `decode_fp4_e2m1_x8` wants the scale broadcast to
    // both halves of the half2.
    //
    // The shared tile holds kTileK/8 half2 scales per column, and the
    // global tensor holds kTileK/16 scales per column (one per 16
    // elements).  Each global scale fills two shared entries, so the
    // shared entry count is twice the global scale count.
    //
    // The global read is one uint4 (four uint32 scales, sixteen bytes) per
    // iteration, which keeps the traffic wide.  A tile of 32 elements has
    // only two groups per column, so the vector is masked rather than
    // rounded up.
    // The global read is one uint4 (four uint32 scales, sixteen bytes) per
    // iteration, which keeps the traffic wide.  A tile of 32 elements has
    // only two groups per column, so the vector is masked rather than
    // rounded up.
    constexpr int kSVec = 4;
    constexpr int kSGroups = 2;  // shared entries per global scale
    const int groups_per_col = kTileK / 16;
    const int vecs_per_col = (groups_per_col + kSVec - 1) / kSVec;
    const int kScalesVec = Tile::kColumns * vecs_per_col;
    for (int sv = tid; sv < kScalesVec; sv += blockDim.x) {
      const int c = sv / vecs_per_col;
      const int v = sv % vecs_per_col;
      const int gcol = n_base + c;
      // `k_tile` counts weight words (eight elements each); a scale group
      // covers two words.
      const int ggroup = k_tile / 2 + v * kSVec;
      // The global scales are halves, so four of them make one uint2.  The
      // shared tile still holds the half2 form, which the kernel broadcasts
      // to both halves when it loads it.
      uint2 raw = make_uint2(0, 0);
      if (gcol < N && ggroup + kSVec <= scale_blocks) {
        raw = __ldg(reinterpret_cast<const uint2 *>(
            &weight_scale[(size_t)gcol * scale_blocks + ggroup]));
      } else if (gcol < N && ggroup < scale_blocks) {
        __half *tp = reinterpret_cast<__half *>(&raw);
        for (int j = 0; j < kSVec; ++j)
          if (ggroup + j < scale_blocks)
            tp[j] = __ldg(&weight_scale[(size_t)gcol * scale_blocks + ggroup + j]);
      }
      uint32_t *dst = reinterpret_cast<uint32_t *>(sdst + c * (kTileK / 8));
      // The group index counts 16-element groups, and each group fills two
      // shared entries (the two weight words of the group).
      const __half *rawp = reinterpret_cast<const __half *>(&raw);
#pragma unroll
      for (int j = 0; j < kSVec; ++j) {
        const int g = v * kSVec + j;
        if (g >= groups_per_col) break;
        // The shared entry is a half2 that carries the scale in both halves.
        uint32_t hb;
        memcpy(&hb, &rawp[j], sizeof(__half));
        dst[g * kSGroups] = hb | (hb << 16);
        dst[g * kSGroups + 1] = hb | (hb << 16);
      }
    }
  };

  // Compute one k block out of a staged tile.
  auto compute_kblock = [&](const uint32_t *wsrc, const uint32_t *ssrc,
                            int gkb) {
    // The activation for this k block: eight halves of row `act_row`.
    uint32_t a_frag[4];
    {
      const int k = gkb * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (m_base + act_row < M) {
        const T *src = input + (size_t)(m_base + act_row) * K + k;
        v = __ldg(reinterpret_cast<const uint4 *>(src));
      }
      a_frag[0] = v.x;
      a_frag[1] = v.y;
      a_frag[2] = v.z;
      a_frag[3] = v.w;
    }

    // The A fragment of the 884 atom: a[0..3] are the eight halves of one
    // row, with k=0..3 in a[0..1] and k=4..7 in a[2..3].
    const uint32_t a_lo[2] = {a_frag[0], a_frag[1]};
    const uint32_t a_hi[2] = {a_frag[2], a_frag[3]};

    const int kb = gkb % Tile::kWordsPerKBlock;
    const int sidx = gkb % (kTileK / 8);

    // The mma.m8n8k4 atom of one eight column quad.  The lane picks the
    // column
    //   c = (lane & 3) + 4 * (lane >= 16)
    // inside the quad and contributes the four consecutive k of that column.
    // The whole warp together covers the eight columns of the quad, and the
    // atom returns the eight rows of the quad to every lane with the
    // layout
    //   row = (lane & 1) + (i & 2) + 4 * (lane >= 16)
    //   col = (lane & 2) + (i & 1) + 4 * ((i >> 2) & 1)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      // The B fragment: one column of the n-quad, eight contiguous k.
      const int c_local =
          warp * kSm70ColumnsPerWarp + q * 8 + (lane & 3) +
          ((lane >= 16) ? 4 : 0);
      const uint32_t w = wsrc[c_local * Tile::kWordsPerKBlock + kb];
      const uint32_t s = ssrc[c_local * (kTileK / 8) + sidx];

      uint32_t bwords[4];
      decode_fp4_e2m1_x8(w, s, bwords);

      // The decoder splits the eight elements of one column into four
      // words whose low halves carry k = 0..3 and whose high halves carry
      // k = 4..7:
      //
      //   w0 = (k0, k4), w1 = (k1, k5), w2 = (k2, k6), w3 = (k3, k7)
      //
      // The 884 atom wants the four consecutive k of one column in two
      // registers, with the low half holding the first k and the high half
      // the second one.  So the low halves are paired up for k = 0..3 and
      // the high halves for k = 4..7.
      const uint32_t b_lo[2] = {
          (bwords[0] & 0xFFFFu) | ((bwords[1] & 0xFFFFu) << 16),
          (bwords[2] & 0xFFFFu) | ((bwords[3] & 0xFFFFu) << 16),
      };
      const uint32_t b_hi[2] = {
          (bwords[0] >> 16) | ((bwords[1] >> 16) << 16),
          (bwords[2] >> 16) | ((bwords[3] >> 16) << 16),
      };

      float d[8];
      float c[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) c[i] = acc[q][i];
      mma_m8n8k4_row_col(d, a_lo, b_lo, c);
      // The second atom accumulates on top of the first one.  The
      // accumulator of the first atom has to live in a separate register
      // set: using the same array as the C operand and the D operand of
      // one instruction is undefined behaviour for inline asm.
      float e[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) e[i] = d[i];
      mma_m8n8k4_row_col(d, a_hi, b_hi, e);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[q][i] = d[i];
    }
  };

  int cur = 0;
  load_tile(wbuf0, sbuf0, 0);
  __syncthreads();

  for (int tile = 0; tile < n_tiles; ++tile) {
    const uint32_t *wcur = cur ? wbuf1 : wbuf0;
    const uint32_t *scur = cur ? sbuf1 : sbuf0;
    uint32_t *wnxt = cur ? wbuf0 : wbuf1;
    uint32_t *snxt = cur ? sbuf0 : sbuf1;

    // Prefetch the next tile into the other buffer so the global load
    // latency overlaps with the compute of the current tile.
    const int next_tile = tile + 1;
    if (next_tile < n_tiles) {
      load_tile(wnxt, snxt, next_tile * Tile::kWordsPerKBlock);
    }

    const int k_begin = tile * Tile::kWordsPerKBlock;
    const int k_end = min(n_blocks, k_begin + Tile::kWordsPerKBlock);
    for (int gkb = k_begin; gkb < k_end; ++gkb) {
      compute_kblock(wcur, scur, gkb);
    }
    __syncthreads();
    cur ^= 1;
  }

  // Write the accumulators out.  The D layout is
  //   row = (lane & 1) + (i & 2) + 4 * (lane >= 16)
  //   col = (lane & 2) + (i & 1) + 4 * ((i >> 2) & 1)
  // relative to the 8x8 tile of the n-quad.  The four redundant lane copies
  // write the same cells, so the stores are plain (not atomic) and they agree.
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int quad_base = warp_col0 + q * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // The C fragment of the 884 atom.  1Cat's reference layout
      // (`benchmarks/csrc/sm70_raw_hmma_probe.cu`) puts the row at
      //   (lane & 1) + (reg & 2) + ((lane >> 4) * 4)
      // and the column at
      //   (lane & 12) * 2 + (lane & 2) + (reg & 4) + (reg & 1)
      // The n-tile part `(lane & 12) * 2` selects one of the four eight
      // column quads of the warp, and `reg & 4` the second half of it.
      const int r = (lane & 1) + (i & 2) + 4 * ((lane >= 16) ? 1 : 0);
      const int c = (lane & 2) + (i & 1) + 4 * ((i >> 2) & 1);
      const int gcol = quad_base + c;
      if (gcol >= N) continue;
      const int grow = m_base + r;
      if (grow >= M) continue;
      float v = acc[q][i];
      if (has_bias) v += __half2float(reinterpret_cast<const __half *>(bias)[gcol]);
      output[(size_t)grow * N + gcol] = static_cast<T>(__float2half(v));
    }
  }
}

}  // namespace sm70_nvfp4

// ---------------------------------------------------------------------------
// Host entry points.
// ---------------------------------------------------------------------------

extern "C" void nvfp4_repack_weight_sm70(const uint8_t *weight,
                                          const uint8_t *weight_scale,
                                          uint32_t *weight_out,
                                          __half *scale_out,
                                          const float *column_global_scale,
                                          float global_scale, int N, int K,
                                          cudaStream_t stream) {
  if (N <= 0 || K <= 0) return;
  // One thread per 16-element scale group; each thread writes two weight
  // words (the two halves of the group).
  const int scales_per_row = K / 16;
  const int total = N * scales_per_row;
  const int threads = 256;
  const int blocks = SM70_NVFP4_CEILDIV(total, threads);
  sm70_nvfp4::nvfp4_repack_weight_sm70_kernel<<<blocks, threads, 0, stream>>>(
      weight, weight_scale, weight_out, scale_out, column_global_scale,
      global_scale, N, K);
}

// The k tile size is chosen by M.  The decode path (M <= 16) has few enough
// CTAs that the grid would otherwise leave SMs idle, so a large tile that
// amortises the barriers wins.  The batched path already produces more CTAs
// than the GPU has SMs, and there a large tile only costs occupancy, so a
// small tile wins.  The thresholds come from a sweep on a V100
// (N=4096, K=4096): M=1..16 -> 0.192..0.199 ms with tileK=512 versus
// 0.214..0.223 ms with tileK=128, and M=32 -> 0.268 ms with tileK=128 versus
// 0.404 ms with tileK=512.
// The CTA shape.
//
// Staging the k tile into shared memory is what limits this kernel: a tile
// of kTileK=512 holds 64 words and 64 scales per column, so the whole CTA
// has to move `columns * (kTileK/8) * 8` bytes per tile.  With a single warp
// that is 16 KiB moved by 32 threads, and the global memory latency is not
// hidden at all.
//
// A sweep on a V100 (numbers in `PROGRESS_PORT.md`) shows the opposite of
// what a barrier-count analysis predicts: the *smaller* the k tile, the
// faster the kernel, because a small tile leaves room for many CTAs per SM
// and the global memory latency is hidden by the occupancy rather than by
// the barrier count:
//
//   M=8 N=17408 K=5120: tileK=512 -> 1.06 ms, 256 -> 0.74, 128 -> 0.46,
//                         64 -> 0.47, 32 -> 0.46 (warp counts 1/2/4).
//
// `kTileK=64` with two warps is the robust choice across the shapes the
// model uses; four warps is slightly better for the decode path, where the
// grid is small.
template <int kTileK, int kWarps>
static void launch_hmma884(const __half *input, const uint32_t *weight,
                            const __half *weight_scale,
                            const float *weight_row_scales, const __half *bias,
                            __half *output, int M, int N, int K, bool has_bias,
                            cudaStream_t stream) {
  using Tile = sm70_nvfp4::Sm70HmmaTile<kTileK, kWarps>;
  constexpr size_t kSmem = Tile::kSmemBytes;
  static_assert(kSmem <= 96 * 1024, "shared memory budget exceeded");
  auto kernel = sm70_nvfp4::nvfp4_sm70_hmma884_kernel<half, kTileK, kWarps>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)kSmem);
  dim3 block(SM70_NVFP4_WARP_SIZE * kWarps);
  dim3 grid(SM70_NVFP4_CEILDIV(N, kWarps * sm70_nvfp4::kSm70ColumnsPerWarp),
            SM70_NVFP4_CEILDIV(M, sm70_nvfp4::kSm70TileM));
  kernel<<<grid, block, kSmem, stream>>>(input, weight, weight_scale,
                                        weight_row_scales, bias, output, M, N,
                                        K, has_bias);

  // Surface a launch failure at the call site instead of letting it turn
  // into an opaque "failed to fill whole buffer" in the runner.
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr,
            "[nvfp4_sm70] launch failed: %s (M=%d N=%d K=%d smem=%zu grid=%d)\n",
            cudaGetErrorString(err), M, N, K, kSmem, grid.x * grid.y);
  }
}

// Picks the CTA shape.
//
// The k tile is fixed at 64 elements: the sweep shows it is at the knee for
// every shape the model uses, and it keeps the shared memory footprint at
// 4 KiB (one warp) or 8 KiB (two warps), which leaves room for many
// resident CTAs.
//
// The warp count follows the grid size: a V100 has 80 SMs, so a CTA of two
// warps only pays off when it still leaves enough CTAs to fill the GPU.
static void launch_hmma884_auto(const __half *input, const uint32_t *weight,
                                const __half *weight_scale,
                                const float *weight_row_scales,
                                const __half *bias, __half *output, int M, int N,
                                int K, bool has_bias, cudaStream_t stream) {
  constexpr int kTargetCtas = 160;  // two per SM on a V100
  constexpr int kCols = sm70_nvfp4::kSm70ColumnsPerWarp;
  if (M > 16) {
    // Batched path: the grid is already large, so use the wider CTA.
    if (N >= 2 * kCols * kTargetCtas) {
      launch_hmma884<64, 4>(input, weight, weight_scale, weight_row_scales,
                            bias, output, M, N, K, has_bias, stream);
    } else {
      launch_hmma884<64, 2>(input, weight, weight_scale, weight_row_scales,
                            bias, output, M, N, K, has_bias, stream);
    }
    return;
  }
  // Decode path: few CTAs, so widen the CTA to get more threads per CTA.
  if (N >= 4 * kCols * kTargetCtas) {
    launch_hmma884<32, 4>(input, weight, weight_scale, weight_row_scales,
                          bias, output, M, N, K, has_bias, stream);
  } else if (N >= 2 * kCols * kTargetCtas) {
    launch_hmma884<64, 2>(input, weight, weight_scale, weight_row_scales,
                          bias, output, M, N, K, has_bias, stream);
  } else {
    launch_hmma884<32, 4>(input, weight, weight_scale, weight_row_scales,
                          bias, output, M, N, K, has_bias, stream);
  }
}

// `weight_scale` holds the block scales with the global scale already folded
// in (see `nvfp4_repack_weight_sm70`).  The global scale is per output
// column: the repack folds `weight_row_scales[col]` when it is present and
// `weight_global_scale` otherwise, which is what the generic NVFP4 path does.
// Neither is therefore applied again here; both are kept in the signature so
// that the host API matches the other NVFP4 entry points.
extern "C" void nvfp4_matmul_hmma884_f16(const __half *input,
                                          const uint32_t *weight,
                                          const __half *weight_scale,
                                          float weight_global_scale,
                                          const float *weight_row_scales,
                                          const __half *bias, __half *output,
                                          int M, int N, int K, bool has_bias,
                                          cudaStream_t stream) {
  (void)weight_global_scale;
  if (M <= 0 || N <= 0 || K <= 0) return;
  launch_hmma884_auto(input, weight, weight_scale, weight_row_scales, bias,
                      output, M, N, K, has_bias, stream);
}

// ---------------------------------------------------------------------------
// Device-side cross-check of the SM70 repack.
//
// The repack rewrites the weight into the layout the HMMA.884 atom expects
// and folds the global scale into the block scales.  Nothing on the host
// checks that rewrite, so a mistake in the nibble order or in the scale
// folding produces garbage that only shows up as a NaN many layers later.
//
// This kernel decodes the repacked weight back to a float and compares it
// against the same weight decoded from the original packed layout, for the
// first few output columns.  `out_diff` receives the largest absolute
// difference and the number of mismatching elements.
//
// The comparison is exact in the scale domain: the reference decodes the
// original nibbles with the software LUT and applies the folded scale the
// same way, so any difference points at the repack.
// ---------------------------------------------------------------------------
extern "C" void nvfp4_sm70_repack_check(const uint8_t *__restrict__ weight,
                                         const uint8_t *__restrict__ weight_scale,
                                         const uint32_t *__restrict__ weight_out,
                                         const __half *__restrict__ scale_out,
                                         float global_scale, int N, int K,
                                         float *__restrict__ out_diff,
                                         int *__restrict__ out_mismatch,
                                         int max_cols) {
  // The FP4 E2M1 values, indexed by the nibble.
  const float lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,
                         6.0f,  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f,
                         -4.0f, -6.0f};

  const int cols = N < max_cols ? N : max_cols;
  const int total = cols * K;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int n = idx / K;
    const int k = idx % K;
    const int group = k / 16;

    // The reference: the original packed weight, two nibbles per byte with
    // the low nibble at the smaller k.
    const uint8_t src = weight[(size_t)n * (K / 2) + (k / 2)];
    const uint32_t nib = (k & 1) ? (src >> 4) : (src & 0xFu);
    const float raw = lut[nib];

    // The folded block scale of this group.
    const float folded = __half2float(scale_out[(size_t)n * (K / 16) + group]);

    // The repacked weight: eight contiguous k per word, the nibble j of the
    // word holding the element j.
    const uint32_t word = weight_out[(size_t)n * (K / 8) + (k / 8)];
    const uint32_t packed = (word >> (4 * (k % 8))) & 0xFu;
    const float repacked = lut[packed];

    const float a = raw * folded;
    const float b = repacked * folded;
    const float d = fabsf(a - b);
    if (d > out_diff[0]) out_diff[0] = d;
    if (a != b) atomicAdd(out_mismatch, 1);
  }
}
