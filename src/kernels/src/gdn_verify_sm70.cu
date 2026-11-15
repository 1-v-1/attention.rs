// SPDX-License-Identifier: Apache-2.0
// GDN target-verification recurrent update (sigmoid gating + gated delta rule)
// for SM70. Copied verbatim from 1Cat-vLLM csrc/sm70_turbomind/ops/
// gdn_verify_sm70.cu; the torch wrapper became an extern "C" raw-pointer
// entry point (same convention as fp8_matmul.cu).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr int KD = 128, TMAX = 8;

__device__ __forceinline__ float tof(float v) { return v; }
__device__ __forceinline__ float tof(half v) { return __half2float(v); }

struct P {
  const half* qkv;
  const half* a;
  const half* b;
  float* state;
  half* o;
  const int* cu;
  const int* idx;
  const int* nacc;
  long long s_state;  // state slot stride (elements)
  int s_idx;          // ssm_state_indices row stride
  int qkv_stride;
  int H, HV;
  int lda, ldb;  // row strides of a / b (elements), HV when contiguous
  float scale;
  // replay mode: rbuf holds two sides x nmax sequences of the previous/current
  // round's normalized k [TMAX][H][KD], raw v [TMAX][HV][KD] and (decay, beta)
  // [TMAX][HV][2], fp32.
  float* rbuf;
  const int* rn;  // tokens of the previous round to replay (its accepted
                  // count), per sequence
  unsigned* ctr;  // [round, finished CTAs]
  int nmax;
  // Per-sequence replay slot and buffer side: rmap[n] = 2 * slot + side.
  // The round writes side `side` and replays side `side ^ 1`. Null: the
  // shared round counter `ctr` selects the side and slot = n.
  const int* rmap;
  // Current-round token index whose state is also stored to state slot
  // cks[n] (an aligned prefix checkpoint inside the round), -1 for none.
  const int* ckt;
  const int* cks;
};

__device__ __forceinline__ long long seq_floats(int H, int HV) {
  return (long long)TMAX * (H * KD + HV * KD + HV * 2);
}

// Replay mode: the last CTA to finish advances the round counter (selects the
// buffer side).
__device__ __forceinline__ void round_done(const P& p) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + 1, 1u) == gridDim.x * gridDim.y - 1) {
      p.ctr[1] = 0;
      __threadfence();
      atomicAdd(p.ctr, 1u);
    }
  }
}

template <int TT, int CTA_ROWS, bool REPLAY>
struct GdnVerifySmem {
  alignas(16) float q[TT][KD];
  alignas(16) float k[TT][KD];
  alignas(16) float rk[REPLAY ? TMAX : 1][KD];
  alignas(16) float v[TT][CTA_ROWS];
  alignas(16) float rv[REPLAY ? TMAX : 1][CTA_ROWS];
  float2 gb[TT];
  float2 rgb[REPLAY ? TMAX : 1];
};

template <typename TA, typename TD, int WARPS, int ROWS, bool REPLAY, int TT>
__global__ __launch_bounds__(WARPS * 32, (WARPS <= 4 ? 8 : 4)) void gdn_verify(
    P p, const TA* __restrict__ A_log, const TD* __restrict__ dt_bias) {
  constexpr int CTA_ROWS = WARPS * ROWS;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.y / p.HV, hv = blockIdx.y % p.HV,
            h = hv / (p.HV / p.H);
  const int bos = p.cu[n], T = p.cu[n + 1] - bos;
  // Replay mode commits the previous round even without current tokens.
  const int R0 = REPLAY ? min(p.rn[n], TMAX) : 0;
  if (T <= 0 && R0 <= 0) {
    if (REPLAY && !p.rmap) round_done(p);
    return;
  }
  const int slot0 =
      p.idx[n * p.s_idx + (REPLAY ? 0 : (p.nacc ? p.nacc[n] - 1 : 0))];
  if (slot0 < 0) {
    if (REPLAY && !p.rmap) round_done(p);
    return;
  }

  const int cta_row0 = blockIdx.x * CTA_ROWS;
  const int row0 = cta_row0 + warp * ROWS;
  const int qo = h * KD + lane * 4, ko = p.H * KD + h * KD + lane * 4,
            cta_vo = 2 * p.H * KD + hv * KD + cta_row0;

  // Issue state loads immediately so HBM latency overlaps with parallel
  // Q/K/V/gate precompute across the CTA's warps.
  float4 st[ROWS];
#pragma unroll
  for (int j = 0; j < ROWS; ++j) {
    st[j] = __ldcg(
        reinterpret_cast<const float4*>(p.state + slot0 * p.s_state +
                                        ((long long)hv * KD + row0 + j) * KD) +
        lane);
  }

  __shared__ GdnVerifySmem<TT, CTA_ROWS, REPLAY> smem;

  const unsigned side =
      !REPLAY ? 0u
              : (p.rmap ? (unsigned)(p.rmap[n] & 1) : (__ldcg(p.ctr) & 1u));
  const long long SEQ = seq_floats(p.H, p.HV);
  const int rs = REPLAY && p.rmap ? (p.rmap[n] >> 1) : n;
  float* cur =
      REPLAY ? p.rbuf + ((long long)side * p.nmax + rs) * SEQ : nullptr;
  const float* prv =
      REPLAY ? p.rbuf + ((long long)(side ^ 1u) * p.nmax + rs) * SEQ : nullptr;

  if constexpr (REPLAY) {
    for (int t = warp; t < R0; t += WARPS) {
      const float4 rk4 = __ldg(
          reinterpret_cast<const float4*>(prv + (t * p.H + h) * KD) + lane);
      *reinterpret_cast<float4*>(&smem.rk[t][lane * 4]) = rk4;
      if (lane < CTA_ROWS) {
        smem.rv[t][lane] = __ldg(prv + TMAX * p.H * KD + (t * p.HV + hv) * KD +
                                 cta_row0 + lane);
      }
      if (lane == 0) {
        smem.rgb[t] = __ldg(
            reinterpret_cast<const float2*>(prv + TMAX * (p.H + p.HV) * KD) +
            t * p.HV + hv);
      }
    }
  }

  const int Tc = min(T, TT);
  if (Tc > 0) {
    const float negA = -__expf(tof(A_log[hv]));
    const float dtb = tof(dt_bias[hv]);
    for (int t = warp; t < Tc; t += WARPS) {
      const half* row = p.qkv + (long long)(bos + t) * p.qkv_stride;
      const uint2 qr = *reinterpret_cast<const uint2*>(row + qo);
      const uint2 kr = *reinterpret_cast<const uint2*>(row + ko);
      if (lane < CTA_ROWS) {
        const float vf = __half2float(__ldg(row + cta_vo + lane));
        smem.v[t][lane] = vf;
        if constexpr (REPLAY) {
          __stcg(cur + TMAX * p.H * KD + (t * p.HV + hv) * KD + cta_row0 + lane,
                 vf);
        }
      }
      float q[4], k[4];
      {
        const half2 q01 = *reinterpret_cast<const half2*>(&qr.x),
                    q23 = *reinterpret_cast<const half2*>(&qr.y);
        const half2 k01 = *reinterpret_cast<const half2*>(&kr.x),
                    k23 = *reinterpret_cast<const half2*>(&kr.y);
        q[0] = __low2float(q01), q[1] = __high2float(q01),
        q[2] = __low2float(q23), q[3] = __high2float(q23);
        k[0] = __low2float(k01), k[1] = __high2float(k01),
        k[2] = __low2float(k23), k[3] = __high2float(k23);
      }
      float sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
      float sk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
        sq += __shfl_xor_sync(0xffffffff, sq, s);
        sk += __shfl_xor_sync(0xffffffff, sk, s);
      }
      const float rq = rsqrtf(sq + 1e-6f) * p.scale, rk = rsqrtf(sk + 1e-6f);
#pragma unroll
      for (int e = 0; e < 4; ++e) q[e] *= rq, k[e] *= rk;
      *reinterpret_cast<float4*>(&smem.q[t][lane * 4]) =
          make_float4(q[0], q[1], q[2], q[3]);
      *reinterpret_cast<float4*>(&smem.k[t][lane * 4]) =
          make_float4(k[0], k[1], k[2], k[3]);

      const float ar = __half2float(p.a[(bos + t) * p.lda + hv]);
      const float br = __half2float(p.b[(bos + t) * p.ldb + hv]);
      const float x = ar + dtb;
      const float sp = x <= 20.f ? logf(1.f + __expf(x)) : x;
      const float decay = __expf(negA * sp);
      const float beta = 1.f / (1.f + __expf(-br));
      if (lane == 0) {
        smem.gb[t] = make_float2(decay, beta);
      }
      if constexpr (REPLAY) {
        if (blockIdx.x == 0 && hv % (p.HV / p.H) == 0) {
          __stcg(reinterpret_cast<float4*>(cur + (t * p.H + h) * KD) + lane,
                 make_float4(k[0], k[1], k[2], k[3]));
        }
        if (blockIdx.x == 0 && lane == 0) {
          __stcg(reinterpret_cast<float2*>(cur + TMAX * (p.H + p.HV) * KD) +
                     t * p.HV + hv,
                 make_float2(decay, beta));
        }
      }
    }
  }
  __syncthreads();

  float hs[ROWS][4];
#pragma unroll
  for (int j = 0; j < ROWS; ++j) {
    hs[j][0] = st[j].x;
    hs[j][1] = st[j].y;
    hs[j][2] = st[j].z;
    hs[j][3] = st[j].w;
  }

  if constexpr (REPLAY) {
    for (int t = 0; t < R0; ++t) {
      const float4 rk4 =
          *reinterpret_cast<const float4*>(&smem.rk[t][lane * 4]);
      const float2 rgb = smem.rgb[t];
      const float k[4] = {rk4.x, rk4.y, rk4.z, rk4.w};
      float d[ROWS];
#pragma unroll
      for (int j = 0; j < ROWS; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) hs[j][e] *= rgb.x;
        d[j] = hs[j][0] * k[0] + hs[j][1] * k[1] + hs[j][2] * k[2] +
               hs[j][3] * k[3];
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1)
#pragma unroll
        for (int j = 0; j < ROWS; ++j)
          d[j] += __shfl_xor_sync(0xffffffff, d[j], s);
#pragma unroll
      for (int j = 0; j < ROWS; ++j) {
        const float u = (smem.rv[t][warp * ROWS + j] - d[j]) * rgb.y;
#pragma unroll
        for (int e = 0; e < 4; ++e) hs[j][e] += u * k[e];
      }
    }
    if (R0 > 0) {
#pragma unroll
      for (int j = 0; j < ROWS; ++j) {
        __stcg(reinterpret_cast<float4*>(p.state + slot0 * p.s_state +
                                         ((long long)hv * KD + row0 + j) * KD) +
                   lane,
               make_float4(hs[j][0], hs[j][1], hs[j][2], hs[j][3]));
      }
    }
  }

  for (int t = 0; t < Tc; ++t) {
    const float4 q4 = *reinterpret_cast<const float4*>(&smem.q[t][lane * 4]);
    const float4 k4 = *reinterpret_cast<const float4*>(&smem.k[t][lane * 4]);
    const float2 gb = smem.gb[t];
    const float q[4] = {q4.x, q4.y, q4.z, q4.w};
    const float k[4] = {k4.x, k4.y, k4.z, k4.w};
    float d[ROWS];
#pragma unroll
    for (int j = 0; j < ROWS; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) hs[j][e] *= gb.x;
      d[j] =
          hs[j][0] * k[0] + hs[j][1] * k[1] + hs[j][2] * k[2] + hs[j][3] * k[3];
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int j = 0; j < ROWS; ++j)
        d[j] += __shfl_xor_sync(0xffffffff, d[j], s);
    float ov[ROWS];
#pragma unroll
    for (int j = 0; j < ROWS; ++j) {
      const float u = (smem.v[t][warp * ROWS + j] - d[j]) * gb.y;
#pragma unroll
      for (int e = 0; e < 4; ++e) hs[j][e] += u * k[e];
      ov[j] =
          hs[j][0] * q[0] + hs[j][1] * q[1] + hs[j][2] * q[2] + hs[j][3] * q[3];
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int j = 0; j < ROWS; ++j)
        ov[j] += __shfl_xor_sync(0xffffffff, ov[j], s);
    if (lane < ROWS) {
      float mine = ov[0];
#pragma unroll
      for (int j = 1; j < ROWS; ++j)
        if (lane == j) mine = ov[j];
      p.o[((long long)(bos + t) * p.HV + hv) * KD + row0 + lane] =
          __float2half_rn(mine);
    }
    const int sl = REPLAY ? (p.ckt && p.ckt[n] == t ? p.cks[n] : -1)
                          : p.idx[n * p.s_idx + t];
    if (sl >= 0) {
#pragma unroll
      for (int j = 0; j < ROWS; ++j)
        __stcg(reinterpret_cast<float4*>(p.state + sl * p.s_state +
                                         ((long long)hv * KD + row0 + j) * KD) +
                   lane,
               make_float4(hs[j][0], hs[j][1], hs[j][2], hs[j][3]));
    }
  }
  if (REPLAY && !p.rmap) round_done(p);
}
}  // namespace
extern "C" void gdn_verify_sm70(
    const void* qkv, const void* a, const void* b, const void* a_log,
    const void* dt_bias, void* state, void* o, const int* cu, const int* idx,
    const int* nacc, int n, int h, int hv, float scale, int cfg,
    long long s_state, int s_idx, int qkv_stride, int lda, int ldb,
    const void* rbuf, const int* rn, unsigned* ctr, long long nmax,
    const int* rmap, const int* ckt, const int* cks, int tmax,
    int a_log_is_f32, int dt_bias_is_f32, void* stream) {
  P p{};
  p.qkv = reinterpret_cast<const half*>(qkv);
  p.a = reinterpret_cast<const half*>(a);
  p.b = reinterpret_cast<const half*>(b);
  p.state = reinterpret_cast<float*>(state);
  p.o = reinterpret_cast<half*>(o);
  p.cu = cu; p.idx = idx; p.nacc = nacc;
  p.s_state = s_state; p.s_idx = s_idx; p.qkv_stride = qkv_stride;
  p.H = h; p.HV = hv; p.scale = scale; p.lda = lda; p.ldb = ldb;
  const bool replay = rbuf != nullptr;
  if (replay) {
    p.rbuf = const_cast<float*>(reinterpret_cast<const float*>(rbuf));
    p.rn = rn; p.ctr = ctr; p.nmax = (int)nmax;
    if (rmap) p.rmap = rmap;
    if (ckt) { p.ckt = ckt; p.cks = cks; }
  }
  auto go3 = [&](auto wr, auto rp, auto tt) {
    constexpr int W = decltype(wr)::first_type::value,
                  R = decltype(wr)::second_type::value;
    constexpr bool RP = decltype(rp)::value;
    constexpr int TTV = decltype(tt)::value;
    const dim3 grid(128 / (W * R), (unsigned)(n * hv));
    if (a_log_is_f32 && dt_bias_is_f32)
      gdn_verify<float, float, W, R, RP, TTV>
          <<<grid, W * 32, 0, (cudaStream_t)stream>>>(
              p, reinterpret_cast<const float*>(a_log),
              reinterpret_cast<const float*>(dt_bias));
    else if (a_log_is_f32 && !dt_bias_is_f32)
      gdn_verify<float, half, W, R, RP, TTV>
          <<<grid, W * 32, 0, (cudaStream_t)stream>>>(
              p, reinterpret_cast<const float*>(a_log),
              reinterpret_cast<const half*>(dt_bias));
    else
      gdn_verify<half, half, W, R, RP, TTV>
          <<<grid, W * 32, 0, (cudaStream_t)stream>>>(
              p, reinterpret_cast<const half*>(a_log),
              reinterpret_cast<const half*>(dt_bias));
  };
  auto go2 = [&](auto wr, auto rp) {
    if (tmax <= 5) go3(wr, rp, std::integral_constant<int, 5>{});
    else go3(wr, rp, std::integral_constant<int, 8>{});
  };
  auto go = [&](auto wr) {
    if (replay) go2(wr, std::true_type{});
    else go2(wr, std::false_type{});
  };
  if (n == 1 && hv > 12)
    go(std::pair<std::integral_constant<int, 8>,
                 std::integral_constant<int, 2>>{});
  else
    go(std::pair<std::integral_constant<int, 8>,
                 std::integral_constant<int, 4>>{});
}
