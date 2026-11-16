// SPDX-License-Identifier: Apache-2.0
// One-shot push allreduce for world_size == 2, ported in protocol from
// 1Cat-vLLM csrc/custom_all_reduce.cu sm70 push path: slots are preinitialised
// with a reserved FP16 NaN payload (the sentinel); the sender escapes its
// payload, pushes 16B packs straight into the peer's slot over NVLink, and the
// receiver spin-reads its own slot until no pack carries the sentinel any
// more. After consumption the slot is restored to the sentinel and the
// per-region epoch advances.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

#include "gpu_sampling.cuh"

namespace {
constexpr unsigned kSentinelBits = 0x7f7fu;  // reserved FP16 NaN payload
constexpr unsigned kEscapeBits = 0x7e00u;    // rewrite target for the escape
constexpr unsigned kEpochs = 2u;             // parity double-buffering

__device__ __forceinline__ bool push_is_sentinel(half v) {
  return *reinterpret_cast<unsigned short*>(&v) == kSentinelBits;
}

__device__ __forceinline__ void push_escape(half& v) {
  unsigned short* bits = reinterpret_cast<unsigned short*>(&v);
  if (*bits == kSentinelBits) *bits = kEscapeBits;
}

__global__ void oneshot_push_allreduce_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    char* own, char* peer, volatile unsigned* epoch_ptr, unsigned slot_stride,
    unsigned n, unsigned rank) {
  constexpr unsigned kPerPack = 8u;  // halves per 16B pack
  const unsigned n16 = n / kPerPack;
  const unsigned offset = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride_threads = gridDim.x * blockDim.x;
  // Per-region epoch counter lives in the 128B header at signal_off.
  const unsigned epoch = *epoch_ptr;
  const unsigned slot = epoch * slot_stride;
  for (unsigned i = offset; i < n16; i += stride_threads) {
    // 1) escape and push our half into the peer's slot for this epoch.
    float4 value = reinterpret_cast<const float4*>(input)[i];
    half* parts = reinterpret_cast<half*>(&value);
    #pragma unroll
    for (unsigned e = 0u; e < kPerPack; ++e) push_escape(parts[e]);
    auto* dest = reinterpret_cast<float4*>(peer + slot) + i;
    const uint4 bits = *reinterpret_cast<const uint4*>(&value);
    asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
        : : "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w),
            "l"(dest));
    // 2) spin on our own slot until the peer's half carries no sentinel.
    const float4* src = reinterpret_cast<const float4*>(own + slot) + i;
    float4 pv;
    uint4 got;
    while (true) {
      asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
          : "=r"(got.x), "=r"(got.y), "=r"(got.z), "=r"(got.w) : "l"(src));
      pv = *reinterpret_cast<const float4*>(&got);
      half* ph = reinterpret_cast<half*>(&pv);
      bool empty = false;
      #pragma unroll
      for (unsigned e = 0u; e < kPerPack; ++e) empty |= push_is_sentinel(ph[e]);
      if (!empty) break;
    }
    // 3) fold both halves and emit.
    half* mine = reinterpret_cast<half*>(&value);
    half* theirs = reinterpret_cast<half*>(&pv);
    float4 o;
    half* oh = reinterpret_cast<half*>(&o);
    #pragma unroll
    for (unsigned e = 0u; e < kPerPack; ++e) oh[e] = mine[e] + theirs[e];
    reinterpret_cast<float4*>(output)[i] = o;
    // 4) restore the slot to the sentinel for the next epoch.
    float4 empty_pack;
    half* eh = reinterpret_cast<half*>(&empty_pack);
    #pragma unroll
    for (unsigned e = 0u; e < kPerPack; ++e)
      *reinterpret_cast<unsigned short*>(&eh[e]) = kSentinelBits;
    const uint4 back = *reinterpret_cast<const uint4*>(&empty_pack);
    asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
        : : "r"(back.x), "r"(back.y), "r"(back.z),
            "r"(back.w), "l"(src));
  }
  __syncthreads();
  if (threadIdx.x == 0u && blockIdx.x == 0u)  *epoch_ptr = (epoch + 1u) % kEpochs;
}
}  // namespace

extern "C" void oneshot_clear(void* p, size_t bytes) {
  CUresult r = cuMemsetD8((CUdeviceptr)(uintptr_t)p, 0, bytes);
  if (r != CUDA_SUCCESS) { printf("oneshot clear failed\n"); }
}

extern "C" void oneshot_alloc(unsigned long long cap_bytes,
                              unsigned long long* out_own_ptr,
                              unsigned char* out_handle_bytes) {
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, cap_bytes));
  CUresult r = cuMemsetD8((CUdeviceptr)(uintptr_t)p, 0x7f, cap_bytes);
  if (r != CUDA_SUCCESS) { printf("oneshot memset failed\n"); }
  cudaIpcMemHandle_t h;
  CUDA_CHECK(cudaIpcGetMemHandle(&h, p));
  *out_own_ptr = (unsigned long long)(uintptr_t)p;
  cudaMemcpy(out_handle_bytes, &h, sizeof(h), cudaMemcpyHostToHost);
}

extern "C" void oneshot_open(const unsigned char* handle_bytes,
                             unsigned long long* out_peer_ptr) {
  cudaIpcMemHandle_t h;
  cudaMemcpy(&h, handle_bytes, sizeof(h), cudaMemcpyHostToHost);
  void* peer = nullptr;
  CUDA_CHECK(cudaIpcOpenMemHandle(&peer, h, cudaIpcMemLazyEnablePeerAccess));
  *out_peer_ptr = (unsigned long long)(uintptr_t)peer;
}

extern "C" void oneshot_fold(const half* input, half* output,
                             char* own, char* peer,
                             volatile unsigned* epoch_ptr,
                             unsigned slot_stride, unsigned n,
                             unsigned rank, cudaStream_t stream) {
  const unsigned n16 = n / 8u;
  const unsigned blocks = min((n16 + 127u) / 128u, 64u);
  oneshot_push_allreduce_kernel<<<blocks, 128, 0, stream>>>(
      input, output, own, peer, epoch_ptr, slot_stride, n, rank);
}
