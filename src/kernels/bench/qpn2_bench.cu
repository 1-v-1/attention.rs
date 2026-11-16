// Standalone timing bench: qpn2 projection kernel, m=1 vs m=2 vs m=8,
// real production shapes. Build next to the kernel source:
//   nvcc -O3 -arch=sm_70 qpn2_bench.cu ../src/nvfp4_qpn2_sm70.cu -o /tmp/qb
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void nvfp4_qpn2_matmul_sm70(const half *input, const uint8_t *codes,
                                       const uint8_t *scales,
                                       float global_scale,
                                       const float *column_global_scale,
                                       const half *bias, half *output, int m,
                                       int n, int k, int64_t stream);

static void bench(int n, int k, const char *name) {
  for (int m : {1, 2, 8}) {
    std::vector<half> h_in((size_t)m * k, __float2half_rn(0.01f));
    std::vector<uint8_t> h_codes((size_t)n * k / 2, 0x44);
    std::vector<uint8_t> h_scales((size_t)n * k / 16, 0x38);
    half *d_in; uint8_t *d_c; uint8_t *d_s; half *d_out;
    cudaMalloc(&d_in, h_in.size() * 2);
    cudaMalloc(&d_c, h_codes.size());
    cudaMalloc(&d_s, h_scales.size());
    cudaMalloc(&d_out, (size_t)m * n * 2);
    cudaMemcpy(d_in, h_in.data(), h_in.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 5; ++i)
      nvfp4_qpn2_matmul_sm70(d_in, d_c, d_s, 1.f, nullptr, nullptr, d_out, m,
                             n, k, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(e0, 0);
    for (int i = 0; i < 50; ++i)
      nvfp4_qpn2_matmul_sm70(d_in, d_c, d_s, 1.f, nullptr, nullptr, d_out, m,
                             n, k, 0);
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double gb = (double)n * k / 2.0 / 1e9;  // weight bytes streamed
    printf("%s n=%d k=%d m=%d: %.3f ms/call, %.1f GB/s eff\n", name, n, k, m,
           ms / 50, gb / (ms / 50 / 1e3));
    cudaFree(d_in); cudaFree(d_c); cudaFree(d_s); cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
}

int main() {
  int dev; cudaGetDevice(&dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  printf("dev=%s\n", p.name);
  bench(17408, 5120, "gate_up");   // mlp gate/up (TP2 per-rank)
  bench(5120, 8704, "down");       // mlp down (TP2 per-rank)
  bench(248320, 5120, "lm_head");  // vocab projection
  return 0;
}
