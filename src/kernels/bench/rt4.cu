// 193 verification: m<=8 native path — current <_,32,16,1> vs the RowTiles=4
// instance (m32_twophase, the 1Cat tile structure). Same inputs, same output
// layout; time both and diff bit-exactly.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/nvfp4_qpn2_sm70.cu"

int main() {
  const int k = 5120, n = 5120;
  const int groups = k / 16;
  // Weight in native QPN2 layout: [N/32][K/16][lane][uint2].
  std::vector<uint8_t> codes((size_t)n * k / 2);
  std::vector<uint8_t> scales((size_t)n * k / 16);
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t)((i * 7 + 3) & 0x77);
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = (uint8_t)((i * 11 + 5) & 0x7f);
  const int m_max = 32;
  std::vector<half> input((size_t)m_max * k), packed((size_t)m_max * k);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = __float2half((float)((i * 13 + 7) % 251) / 251.0f - 0.5f);

  half *d_in, *d_pk, *d_out, *d_out2;
  uint8_t *d_c, *d_s;
  cudaMalloc(&d_c, codes.size()); cudaMalloc(&d_s, scales.size());
  cudaMalloc(&d_in, input.size() * 2); cudaMalloc(&d_pk, input.size() * 2);
  cudaMalloc(&d_out, (size_t)m_max * n * 2); cudaMalloc(&d_out2, (size_t)m_max * n * 2);
  cudaMemcpy(d_c, codes.data(), codes.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s, scales.data(), scales.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, input.data(), input.size() * 2, cudaMemcpyHostToDevice);
  // K16-packed activation for the m32 path: [group][row][16].
  for (int g = 0; g < groups; ++g)
    for (int r = 0; r < m_max; ++r)
      for (int e = 0; e < 16; ++e)
        packed[(size_t)(g * m_max + r) * 16 + e] =
            input[(size_t)r * k + g * 16 + e];
  cudaMemcpy(d_pk, packed.data(), packed.size() * 2, cudaMemcpyHostToDevice);

  cudaEvent_t t0, t1; cudaEventCreate(&t0, cudaEventDefault); cudaEventCreate(&t1, cudaEventDefault);
  float ms_native = 0, ms_rt4 = 0;
  for (int m : {1, 2, 8}) {
    // native (current production path)
    auto run_native = [&] {
      qpn2_sm70::nvfp4_qpn2_projection_sm70_kernel<5120, 32, 16>
          <<<dim3(n / 32, (m + 7) / 8), 512>>>(d_c, d_s, d_in, d_out, n, k, m,
                                               1.0f, nullptr, nullptr);
    };
    // RowTiles=4 instance (m32 two-phase, packed input)
    auto run_rt4 = [&] {
      const dim3 g(1, n / 32, (m + 31) / 32);
      qpn2_sm70::nvfp4_qpn2_m32_twophase_sm70_kernel<false>
          <<<g, 256>>>(d_c, d_s, d_pk, d_out2, n, k, m, 1.0f, nullptr, nullptr);
    };
    for (int i = 0; i < 20; ++i) { run_native(); run_rt4(); }
    cudaDeviceSynchronize();
    cudaEventRecord(t0, 0);
    for (int i = 0; i < 100; ++i) run_native();
    cudaEventRecord(t1, 0); cudaEventSynchronize(t1);
    float a; cudaEventElapsedTime(&a, t0, t1);
    cudaEventRecord(t0, 0);
    for (int i = 0; i < 100; ++i) run_rt4();
    cudaEventRecord(t1, 0); cudaEventSynchronize(t1);
    float b; cudaEventElapsedTime(&b, t0, t1);
    // bit-exact check on m rows
    std::vector<half> ref((size_t)m * n), got((size_t)m * n);
    cudaMemcpy(ref.data(), d_out, (size_t)m * n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), d_out2, (size_t)m * n * 2, cudaMemcpyDeviceToHost);
    size_t diff = 0;
    for (size_t i = 0; i < ref.size(); ++i)
      if (memcmp(&ref[i], &got[i], 2) != 0) ++diff;
    printf("m=%d native=%.2fus rt4=%.2fus ratio=%.2fx diff=%zu\n", m,
           a * 1000.f / 100.f, b * 1000.f / 100.f, a / b, diff);
  }
  return 0;
}
