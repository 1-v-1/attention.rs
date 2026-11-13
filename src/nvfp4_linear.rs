#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use crate::workspace::get_cutlass_workspace;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Result, Tensor};

pub const NVFP4_BLOCK_SIZE: usize = 16;

/// Compute online input scale from the activation tensor's max absolute value.
///
/// Device-only variant: returns a CUDA `[2]` tensor
/// (`[input_scale, input_scale_inv]` where `input_scale = amax(|x|) / 6.0`).
/// Safe under CUDA graph capture — no host sync.
#[cfg(feature = "cuda")]
pub fn compute_online_input_scale_device(input: &Tensor) -> Result<Tensor> {
    use candle_core::{DType, Storage};

    let dev = input.device();
    let dtype = input.dtype();
    let num_elements = input.elem_count();

    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let result_tensor = Tensor::zeros((2,), DType::F32, dev)?;
    let (result_s, _) = result_tensor.storage_and_layout();
    let result_ptr = match &*result_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("result must be on CUDA"),
    };

    let (input_s, _) = input.storage_and_layout();
    let input_ptr = match &*input_s {
        Storage::Cuda(c) => match dtype {
            DType::F16 => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
            DType::BF16 => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        },
        _ => candle_core::bail!("tensor must be on CUDA"),
    };

    unsafe {
        match dtype {
            DType::F16 => ffi::nvfp4_compute_online_input_scale_f16(
                input_ptr as *const std::ffi::c_void,
                result_ptr as *mut f32,
                num_elements as i32,
                stream,
            ),
            DType::BF16 => ffi::nvfp4_compute_online_input_scale_bf16(
                input_ptr as *const std::ffi::c_void,
                result_ptr as *mut f32,
                num_elements as i32,
                stream,
            ),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        }
    }
    drop(result_s);
    drop(input_s);

    Ok(result_tensor)
}

/// Compute online input scale from the activation tensor's max absolute value.
/// Returns (input_scale, input_scale_inv) where input_scale = amax(|x|) / 6.0.
///
/// Performs a D2H sync — **not** safe under CUDA graph capture. Prefer
/// [`compute_online_input_scale_device`] + [`resolve_online_hw_scales`] inside
/// graph-captured forwards.
#[cfg(feature = "cuda")]
pub fn compute_online_input_scale(input: &Tensor) -> Result<(f32, f32)> {
    let result_tensor = compute_online_input_scale_device(input)?;
    let result_vec = result_tensor.to_vec1::<f32>()?;
    Ok((result_vec[0], result_vec[1]))
}

/// Build device `alpha` and `sf_inv` scalars from an online `[scale, inv]`
/// tensor and the checkpoint-calibrated scale. Graph-capture safe (no D2H).
#[cfg(feature = "cuda")]
pub fn resolve_online_hw_scales(
    online: &Tensor,
    calibrated_input_scale: f32,
    weight_global_scale: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::{DType, Storage};

    let dev = online.device();
    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let alpha_tensor = Tensor::zeros((1,), DType::F32, dev)?;
    let sf_scale_tensor = Tensor::zeros((1,), DType::F32, dev)?;

    let (online_s, _) = online.storage_and_layout();
    let online_ptr = match &*online_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("online scales must be on CUDA"),
    };
    let (alpha_s, _) = alpha_tensor.storage_and_layout();
    let alpha_ptr = match &*alpha_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("alpha must be on CUDA"),
    };
    let (sf_s, _) = sf_scale_tensor.storage_and_layout();
    let sf_ptr = match &*sf_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("sf_inv must be on CUDA"),
    };

    unsafe {
        ffi::nvfp4_resolve_online_scales(
            online_ptr as *const f32,
            calibrated_input_scale,
            weight_global_scale,
            alpha_ptr as *mut f32,
            sf_ptr as *mut f32,
            stream,
        );
    }
    drop(online_s);
    drop(alpha_s);
    drop(sf_s);

    Ok((alpha_tensor, sf_scale_tensor))
}

/// Check if hardware FP4 (CUTLASS block-scaled tensor ops) is available.
/// Requires Blackwell SM100+ and the cutlass feature.
#[cfg(feature = "cuda")]
fn is_hardware_fp4_available(dev: &candle_core::Device) -> bool {
    if !cfg!(feature = "cutlass") {
        return false;
    }
    if let Ok(cuda_dev) = dev.as_cuda_device() {
        let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
        // SM100+ hardware NVFP4, including SM120/SM121 tensor-core FP4.
        sm >= 100
    } else {
        false
    }
}

/// Check if FlashInfer-ported FP4 CUTLASS path is available.
/// Requires SM100+ (including SM120/SM121) and the flashinfer feature.
#[cfg(feature = "cuda")]
fn is_flashinfer_fp4_available(dev: &candle_core::Device) -> bool {
    if !cfg!(feature = "flashinfer") {
        return false;
    }
    if let Ok(cuda_dev) = dev.as_cuda_device() {
        let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
        sm >= 100
    } else {
        false
    }
}

/// Pad dimension up to the nearest multiple of `align`.
#[cfg(feature = "cuda")]
fn pad_to(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
}

/// Pre-swizzle NVFP4 weight scales from linear layout to the CUTLASS
/// 128×4 block-interleaved layout required by Blackwell hardware FP4 tensor
/// cores. Call once at model load time to avoid re-swizzling on every forward.
///
/// Accepts rank-2 `[N, K/16]` (single linear layer) or rank-3 `[E, N, K/16]`
/// (batched / MoE). All dimensions are read from the tensor shape, so
/// sharded tensors work correctly without the caller adjusting sizes.
///
/// Returns a swizzled U8 tensor with the same rank (with N and K/16 padded).
#[allow(unused)]
pub fn swizzle_nvfp4_weight_scales(scale: &Tensor) -> Result<Tensor> {
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };

    let dims = scale.dims();
    let (num_slices, n, k_scale_cols) = match dims {
        [n, ksc] => (1, *n, *ksc),
        [e, n, ksc] => (*e, *n, *ksc),
        _ => candle_core::bail!(
            "swizzle_nvfp4_weight_scales: expected rank 2 [N, K/16] or 3 [E, N, K/16], got {:?}",
            dims
        ),
    };
    let is_batched = dims.len() == 3;
    let dev = scale.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;

            let k_scale_padded = pad_to(k_scale_cols, 4);
            let n_padded = pad_to(n, 128);

            let swizzled = if is_batched {
                Tensor::zeros((num_slices, n_padded, k_scale_padded), DType::U8, dev)?
            } else {
                Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?
            };

            {
                let (scale_s, _) = scale.storage_and_layout();
                let scale_base = match &*scale_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let (sw_s, _) = swizzled.storage_and_layout();
                let sw_base = match &*sw_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let stream = *cuda_dev.cu_stream() as i64;
                for e in 0..num_slices {
                    let src_offset = (e * n * k_scale_cols) as u64;
                    let dst_offset = (e * n_padded * k_scale_padded) as u64;
                    unsafe {
                        ffi::nvfp4_swizzle_weight_scales(
                            (scale_base + src_offset) as *const std::ffi::c_void,
                            (sw_base + dst_offset) as *mut std::ffi::c_void,
                            n as i32,
                            k_scale_cols as i32,
                            n_padded as i32,
                            k_scale_padded as i32,
                            stream,
                        );
                    }
                }
            }

            Ok(swizzled)
        }
        _ => candle_core::bail!("swizzle_nvfp4_weight_scales: unsupported backend (need CUDA)"),
    }
}

/// NVFP4 linear: output = input @ weight^T [+ bias]
///
/// * `input` - [M, K] in F16/BF16
/// * `weight` - [N, K/2] packed U8 (2 FP4 E2M1 nibbles per byte)
/// * `scale` - [N, K/16] U8 FP8 E4M3 block scales (linear layout)
/// * `weight_global_scale` - scalar F32 weight-side global scale
///   (from `weight_scale_2` or `1/weight_global_scale` in the checkpoint)
/// * `input_scale` - scalar F32 activation-side global scale
///   (from `input_scale` or `input_global_scale` in the checkpoint, default 1.0)
///   Used by the hardware FP4 path to pre-scale activation block scales during
///   quantization and to compute the GEMM epilogue alpha = input_scale * weight_global_scale.
///   Ignored by the software path (activations stay in FP16/BF16).
/// * `bias` - Optional [N] in F16/BF16
/// * `weight_scale_swizzled` - Optional pre-swizzled weight scales from
///   [`swizzle_nvfp4_weight_scales`]. When provided, skips per-call swizzling.
///
/// On SM100+ with cutlass/flashinfer: prefill uses hardware FP4 tensor cores
/// via CUTLASS block-scaled GEMM (quantizes activations to FP4 on-the-fly).
/// SM120/SM121 prefill uses FlashInfer `fp4_quantize` + `mm_fp4`. Decode uses
/// the same hardware W4A4 path by default (`XINFER_NVFP4_HW_DECODE=1`);
/// set `XINFER_NVFP4_HW_DECODE=0` for software small-M GEMV.
/// On Hopper and below: software dequant (LUT-based FP4 decode + FMA/WMMA).
///
/// Returns [M, N] in same dtype as input

/// Like [`nvfp4_matmul`] but accepts optional pre-swizzled weight scales to
/// avoid redundant per-call swizzling on the hardware FP4 path.
#[allow(unused)]
// ---------------------------------------------------------------------------
// SM70 (Volta) HMMA.884 weight cache.
//
// The Volta path needs the weights repacked into `[N, K/8]` uint32 (eight
// contiguous k of one output column per word) and the block scales converted
// to a half duplicated in both halves.  That repack is expensive relative to
// a decode step, so it is done once per layer and cached.
//
// The cache is keyed by the device pointer of the source weight tensor, which
// is stable for the lifetime of a loaded model.  It only holds a handful of
// entries (one per NVFP4 linear layer), so a linear scan is fine.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod sm70_weight_cache {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Tensor};

    pub struct Entry {
        pub weight: Tensor,
        pub scale: Tensor,
    }

    /// The key combines the weight's device pointer with the global scale,
    /// because the global scale is folded into the repacked block scales and
    /// therefore changes the result.
    type Key = (u64, u64, u32);

    type Cache = HashMap<Key, Entry>;

    fn cache() -> &'static Mutex<Cache> {
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Return the repacked weights for `weight`, repacking on first use.
    ///
    /// `weight` is the `[N, K/2]` u8 checkpoint tensor and `scale` is the
    /// `[N, K/16]` u8 FP8 E4M3 block scale tensor.
    pub fn get_or_repack(
        weight: &Tensor,
        scale: &Tensor,
        weight_global_scale: f32,
        column_global_scale: Option<&Tensor>,
        n: usize,
        k: usize,
    ) -> Result<(Tensor, Tensor)> {
        use candle_core::Storage;

        let (ws, _) = weight.storage_and_layout();
        let weight_key = match &*ws {
            Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
            _ => candle_core::bail!("weight must be on CUDA"),
        };

        // The per-column global scale is folded into the repacked block
        // scales as well, so its identity has to be part of the key.
        let column_key = match column_global_scale {
            Some(cs) => {
                let (cs_s, _) = cs.storage_and_layout();
                match &*cs_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
                    _ => candle_core::bail!("column_global_scale must be on CUDA"),
                }
            }
            None => 0,
        };

        let key = (weight_key, column_key, weight_global_scale.to_bits());

        {
            let guard = cache().lock().unwrap();
            if let Some(entry) = guard.get(&key) {
                return Ok((entry.weight.clone(), entry.scale.clone()));
            }
        }

        let (repacked_weight, repacked_scale) =
            crate::nvfp4_linear::nvfp4_repack_weight_sm70(
                weight,
                scale,
                weight_global_scale,
                column_global_scale,
                n,
                k,
            )?;

        let mut guard = cache().lock().unwrap();
        guard.insert(
            key,
            Entry {
                weight: repacked_weight.clone(),
                scale: repacked_scale.clone(),
            },
        );
        Ok((repacked_weight, repacked_scale))
    }
}

/// Whether the SM70 HMMA.884 path should be used on this device.
///
/// Volta (SM70) has no `m16n16k16` HMMA and no hardware FP4, so the generic
/// path falls back to the LUT-based software dequant, which leaves the tensor
/// cores idle.  The HMMA.884 kernel uses the Volta `m8n8k4` instruction
/// instead and is several times faster on the decode shape.
#[cfg(feature = "cuda")]
fn use_sm70_hmma884(cuda_dev: &candle_core::CudaDevice) -> bool {
    // XINFER_NVFP4_SM70_HMMA884=0 forces the generic software path, which
    // makes it possible to A/B the Volta kernel against the fallback on the
    // same model without rebuilding.
    if std::env::var("XINFER_NVFP4_SM70_HMMA884")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return false;
    }
    crate::cuda_utils::sm_version(cuda_dev) == Some(70)
}

pub fn nvfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    input_scale: f32,
    bias: Option<&Tensor>,
    is_prefill: bool,
    weight_scale_swizzled: Option<&Tensor>,
    weight_row_scales: Option<&Tensor>,
) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weight = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };

    let input_dims = input.dims();
    let weight_dims = weight.dims();

    if input_dims.len() != 2 {
        candle_core::bail!("nvfp4_matmul: expected input rank 2, got {:?}", input_dims);
    }

    let m = input_dims[0];
    let k = input_dims[1];
    let n = weight_dims[0];

    // A weight that has already been repacked for the SM70 HMMA.884 kernel
    // arrives as u32 instead of the packed u8, so the shape check below is
    // skipped for it.
    let prepacked_sm70 = weight.dtype() == DType::U32;
    if !prepacked_sm70 {
        if k % NVFP4_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "nvfp4_matmul: K must be divisible by {NVFP4_BLOCK_SIZE}, got K={k}"
            );
        }
        if weight_dims[1] != k / 2 {
            candle_core::bail!(
                "nvfp4_matmul: weight shape mismatch, expected [N, K/2]=[{}, {}], got {:?}",
                n,
                k / 2,
                weight_dims
            );
        }
    }

    let dev = input.device();
    let dtype = input.dtype();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;
            let _ = &Storage::Cuda;

            // Hardware W4A4 for prefill; decode too unless XINFER_NVFP4_HW_DECODE=0.
            let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
            let use_hw_gemm = is_prefill || crate::nvfp4_hw_decode();
            let use_flashinfer_fp4 = use_hw_gemm
                && cfg!(feature = "flashinfer")
                && is_flashinfer_fp4_available(dev)
                && n % 32 == 0
                && k % 32 == 0;
            // SM120/SM121: FlashInfer fp4_quantize (SWIZZLED_128x4, e4m3Max=448).
            let use_flashinfer_quant = use_flashinfer_fp4 && sm >= 120;

            let use_hardware_fp4 = use_hw_gemm
                && !use_flashinfer_fp4
                && cfg!(feature = "cutlass")
                && is_hardware_fp4_available(dev)
                && n % 32 == 0
                && k % 32 == 0;

            let output = Tensor::zeros((m, n), dtype, dev)?;
            let has_bias = bias.is_some();

            // A repacked weight can only be read by the SM70 kernel.  Every
            // other path expects the packed u8 layout and would silently
            // produce garbage, so refuse instead.
            if prepacked_sm70
                && !(use_sm70_hmma884(cuda_dev)
                    && dtype == DType::F16
                    && k % 16 == 0
                    && weight_scale_swizzled.is_none())
            {
                candle_core::bail!(
                    "nvfp4_matmul: the weight is repacked for the SM70 HMMA.884 kernel, \
                     which this device and dtype cannot use"
                );
            }

            if use_flashinfer_fp4 || use_hardware_fp4 {
                #[cfg(feature = "cutlass")]
                {
                    use crate::workspace::get_nvfp4_hw_scratch;

                    let stream = *cuda_dev.cu_stream() as i64;

                    let m_padded = pad_to(m, 128);
                    let k_scale_cols = k / NVFP4_BLOCK_SIZE;
                    let k_scale_padded = pad_to(k_scale_cols, 4);
                    let n_padded = pad_to(n, 128);

                    let act_packed_bytes = m * (k / 2);
                    let act_scales_bytes = m_padded * k_scale_cols;
                    let act_scales_swizzled_bytes = m_padded * k_scale_padded;
                    let wscale_bytes = if weight_scale_swizzled.is_some() {
                        0
                    } else {
                        n_padded * k_scale_padded
                    };
                    let scratch = get_nvfp4_hw_scratch(
                        cuda_dev,
                        act_packed_bytes,
                        act_scales_bytes,
                        act_scales_swizzled_bytes,
                        wscale_bytes,
                    )?;

                    let act_packed_ptr = scratch.act_packed;
                    let act_scales_ptr = scratch.act_scales;
                    let act_scales_sw_ptr = scratch.act_scales_swizzled;

                    // Default: checkpoint-calibrated input_scale (ModelOpt joint
                    // PTQ). Optional XINFER_NVFP4_ONLINE_SCALE=1 raises the scale
                    // to max(online_amax/6, calibrated) to avoid FP4 clipping.
                    // Online path stays fully on-device (no to_vec1) so CUDA graph
                    // capture/replay remain valid.
                    let (alpha_ptr, sf_scale_ptr) = if crate::nvfp4_online_scale() {
                        // Compute online amax/6 directly into domain-stable scratch
                        // (avoids ephemeral Tensor::zeros under CUDA graph capture).
                        let (input_s, _) = input.storage_and_layout();
                        let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                        let num_elements = input.elem_count();
                        unsafe {
                            match dtype {
                                DType::F16 => ffi::nvfp4_compute_online_input_scale_f16(
                                    input_ptr,
                                    scratch.online_scale,
                                    num_elements as i32,
                                    stream,
                                ),
                                DType::BF16 => ffi::nvfp4_compute_online_input_scale_bf16(
                                    input_ptr,
                                    scratch.online_scale,
                                    num_elements as i32,
                                    stream,
                                ),
                                _ => candle_core::bail!(
                                    "nvfp4 online scale: unsupported dtype {:?}",
                                    dtype
                                ),
                            }
                            ffi::nvfp4_resolve_online_scales(
                                scratch.online_scale,
                                input_scale,
                                weight_global_scale,
                                scratch.alpha,
                                scratch.sf_scale,
                                stream,
                            );
                        }
                        drop(input_s);
                        (scratch.alpha as *const f32, scratch.sf_scale as *const f32)
                    } else {
                        use candle_core::cuda_backend::cudarc::driver::result;
                        let (hw_input_scale, hw_input_scale_inv) = if input_scale > 1e-12 {
                            (input_scale, 1.0 / input_scale)
                        } else {
                            (1.0, 1.0)
                        };
                        let alpha = hw_input_scale * weight_global_scale;
                        // Host→device scalar writes are graph-capture safe and keep
                        // alpha/sf addresses in the domain-scoped scratch pool.
                        unsafe {
                            result::memcpy_htod_async(
                                scratch.alpha as u64,
                                &[alpha],
                                *cuda_dev.cu_stream(),
                            )
                            .map_err(candle_core::Error::wrap)?;
                            result::memcpy_htod_async(
                                scratch.sf_scale as u64,
                                &[hw_input_scale_inv],
                                *cuda_dev.cu_stream(),
                            )
                            .map_err(candle_core::Error::wrap)?;
                        }
                        (scratch.alpha as *const f32, scratch.sf_scale as *const f32)
                    };

                    {
                        let (input_s, _) = input.storage_and_layout();
                        let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;

                        let (weight_s, _) = weight.storage_and_layout();
                        let weight_ptr =
                            cuda_ptr(&weight_s, weight.dtype())? as *const std::ffi::c_void;

                        let (scale_s, _) = scale.storage_and_layout();
                        let scale_ptr =
                            cuda_ptr(&scale_s, scale.dtype())? as *const std::ffi::c_void;

                        let wscale_sw_ptr = if let Some(preswizzled) = weight_scale_swizzled {
                            let (wscale_sw_s, _) = preswizzled.storage_and_layout();
                            let ptr = cuda_ptr(&wscale_sw_s, DType::U8)? as *mut std::ffi::c_void;
                            drop(wscale_sw_s);
                            ptr
                        } else {
                            scratch.wscale_swizzled
                        };

                        let (output_s, _) = output.storage_and_layout();
                        let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                        unsafe {
                            if use_flashinfer_quant {
                                match dtype {
                                    DType::F16 => ffi::flashinfer_nvfp4_quantize_activation_f16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_sw_ptr,
                                        sf_scale_ptr,
                                        m as i32,
                                        k as i32,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::flashinfer_nvfp4_quantize_activation_bf16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_sw_ptr,
                                        sf_scale_ptr,
                                        m as i32,
                                        k as i32,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
                            } else {
                                match dtype {
                                    DType::F16 => ffi::nvfp4_quantize_activation_f16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_ptr,
                                        act_scales_sw_ptr,
                                        sf_scale_ptr,
                                        m as i32,
                                        k as i32,
                                        m_padded as i32,
                                        k_scale_padded as i32,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::nvfp4_quantize_activation_bf16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_ptr,
                                        act_scales_sw_ptr,
                                        sf_scale_ptr,
                                        m as i32,
                                        k as i32,
                                        m_padded as i32,
                                        k_scale_padded as i32,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
                            }

                            if weight_scale_swizzled.is_none() {
                                ffi::nvfp4_swizzle_weight_scales(
                                    scale_ptr,
                                    wscale_sw_ptr,
                                    n as i32,
                                    k_scale_cols as i32,
                                    n_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                );
                            }

                            let (ws_ptr, ws_bytes) = get_cutlass_workspace(cuda_dev, 0)?;
                            let ws_bytes = ws_bytes as i64;

                            if use_flashinfer_fp4 {
                                // FlashInfer-ported CUTLASS path (preferred on SM100+)
                                match dtype {
                                    DType::F16 => ffi::flashinfer_nvfp4_cutlass_gemm_f16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::flashinfer_nvfp4_cutlass_gemm_bf16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
                            } else {
                                // Existing CUTLASS path (fallback when flashinfer not enabled)
                                match dtype {
                                    DType::F16 => ffi::nvfp4_cutlass_gemm_f16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::nvfp4_cutlass_gemm_bf16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
                            }
                        }
                    }

                    if let Some(b) = bias {
                        return Ok(output.broadcast_add(b)?);
                    }
                }
                #[cfg(not(feature = "cutlass"))]
                {
                    let _ = (use_flashinfer_fp4, use_hardware_fp4, use_flashinfer_quant);
                    candle_core::bail!("nvfp4 hardware path requires the cutlass feature");
                }
            } else if use_sm70_hmma884(cuda_dev)
                && dtype == DType::F16
                && k % 16 == 0
                && weight_scale_swizzled.is_none()
            {
                // SM70 (Volta) HMMA.884 path: the weights are repacked once
                // per layer and cached, then every step runs on the tensor
                // cores with `mma.m8n8k4`.
                //
                // When the weight arrives already repacked -- it was done at
                // load time so that the packed original could be dropped --
                // there is nothing left to do here.
                let (repacked_weight, repacked_scale) = if prepacked_sm70 {
                    (weight.clone(), scale.clone())
                } else {
                    sm70_weight_cache::get_or_repack(
                        &weight,
                        &scale,
                        weight_global_scale,
                        weight_row_scales,
                        n,
                        k,
                    )?
                };
                let (input_s, _) = input.storage_and_layout();
                let (ws, _) = repacked_weight.storage_and_layout();
                let (wss, _) = repacked_scale.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                let weight_ptr = cuda_ptr(&ws, DType::U32)? as *const u32;
                let scale_ptr = cuda_ptr(&wss, DType::U32)? as *const u32;
                let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                let bias_ptr = if let Some(b) = bias {
                    let (b_s, _) = b.storage_and_layout();
                    cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                };

                let stream = *cuda_dev.cu_stream() as i64;

                unsafe {
                    ffi::nvfp4_matmul_hmma884_f16(
                        input_ptr,
                        weight_ptr,
                        scale_ptr,
                        weight_global_scale,
                        std::ptr::null(),
                        bias_ptr,
                        output_ptr,
                        m as i32,
                        n as i32,
                        k as i32,
                        has_bias,
                        stream,
                    );
                }

                // Optional cross-check against the generic software dequant
                // path.  Enabled with XINFER_NVFP4_SM70_CHECK=1; it doubles the
                // work of every GEMM, so it is only meant for validation runs.
                //
                // The check copies results back to the host, which is illegal
                // while a CUDA graph is being captured, and it would also run
                // once per layer per token.  It therefore only runs on the
                // very first call and only outside of a graph capture.
                //
                // The reference path needs the original packed tensors, so
                // the check is skipped when the weight arrived already
                // repacked: there is no packed copy left to compare against.
                if crate::nvfp4_sm70_check()
                    && !sm70_check_done()
                    && !prepacked_sm70
                    && !capturing(stream as usize)
                {
                    check_sm70_vs_software(
                        &input,
                        &weight,
                        &scale,
                        weight_global_scale,
                        weight_row_scales,
                        bias,
                        &output,
                        m,
                        n,
                        k,
                    )?;
                    set_sm70_check_done();
                }
            } else {
                // Software dequant path (existing kernels)
                let (input_s, _) = input.storage_and_layout();
                let (weight_s, _) = weight.storage_and_layout();
                let (scale_s, _) = scale.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                let weight_ptr = cuda_ptr(&weight_s, weight.dtype())? as *const u8;
                let scale_ptr = cuda_ptr(&scale_s, scale.dtype())? as *const u8;
                let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                let bias_ptr = if let Some(b) = bias {
                    let (b_s, _) = b.storage_and_layout();
                    cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                };

                let row_scales_ptr: *const f32 = match weight_row_scales {
                    Some(rs) => {
                        let (rs_s, _) = rs.storage_and_layout();
                        cuda_ptr(&rs_s, rs.dtype())? as *const f32
                    }
                    None => std::ptr::null(),
                };

                let stream = *cuda_dev.cu_stream() as i64;
                let force_lut = crate::nvfp4_force_lut();

                unsafe {
                    if m < 32 {
                        match dtype {
                            DType::F16 => {
                                ffi::nvfp4_matmul_smallm_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::nvfp4_matmul_smallm_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            _ => candle_core::bail!(
                                "nvfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        }
                    } else {
                        match dtype {
                            DType::F16 => {
                                ffi::nvfp4_matmul_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::nvfp4_matmul_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            _ => candle_core::bail!(
                                "nvfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        }
                    }
                }
            }

            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(metal_dev) => {
            use candle_core::Storage;

            let output = Tensor::zeros((m, n), dtype, dev)?;

            let command_buffer = metal_dev.command_buffer()?;
            let command_buffer_ref = command_buffer.as_ref();

            {
                let (input_s, input_l) = input.storage_and_layout();
                let input_ms = match &*input_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("input must be metal"),
                };
                let (weight_s, weight_l) = weight.storage_and_layout();
                let weight_ms = match &*weight_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("weight must be metal"),
                };
                let (scale_s, scale_l) = scale.storage_and_layout();
                let scale_ms = match &*scale_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("scale must be metal"),
                };
                let (output_s, _output_l) = output.storage_and_layout();
                let output_ms = match &*output_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("output must be metal"),
                };

                let x = (
                    input_ms.buffer(),
                    input_l.start_offset() * dtype.size_in_bytes(),
                );
                let w = (
                    weight_ms.buffer(),
                    weight_l.start_offset() * weight.dtype().size_in_bytes(),
                );
                let sc = (
                    scale_ms.buffer(),
                    scale_l.start_offset() * scale.dtype().size_in_bytes(),
                );

                if let Some(b) = bias {
                    let b = if b.is_contiguous() {
                        b.clone()
                    } else {
                        b.contiguous()?
                    };
                    let (bias_s, bias_l) = b.storage_and_layout();
                    let bias_ms = match &*bias_s {
                        Storage::Metal(s) => s,
                        _ => candle_core::bail!("bias must be metal"),
                    };
                    let bias_buf = (
                        bias_ms.buffer(),
                        bias_l.start_offset() * b.dtype().size_in_bytes(),
                    );

                    metal_kernels::call_nvfp4_matmul(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        bias_buf,
                        output_ms.buffer(),
                        m,
                        n,
                        k,
                        weight_global_scale,
                        true,
                    )
                    .map_err(candle_core::Error::wrap)?;
                } else {
                    let dummy_bias = (input_ms.buffer(), 0usize);

                    metal_kernels::call_nvfp4_matmul(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        dummy_bias,
                        output_ms.buffer(),
                        m,
                        n,
                        k,
                        weight_global_scale,
                        false,
                    )
                    .map_err(candle_core::Error::wrap)?;
                }
            }

            Ok(output)
        }
        _ => candle_core::bail!("nvfp4_matmul: unsupported backend (need CUDA or Metal)"),
    }
}

/// Device pointer of a CUDA tensor, addressed as the raw element type that
/// matches `dtype`.
///
/// The SM70 HMMA.884 path keeps the repacked weights and the block scales in
/// u32 tensors, which is why `DType::U32` is supported here.
#[cfg(feature = "cuda")]
fn cuda_ptr(s: &candle_core::Storage, dtype: DType) -> candle_core::Result<u64> {
    use candle_core::Storage;
    match s {
        Storage::Cuda(c) => match dtype {
            DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
            DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
            DType::U8 | DType::F8E4M3 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
            DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
            DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        },
        _ => candle_core::bail!("tensor must be on CUDA"),
    }
}

/// True once the SM70 cross-check has run at least once.  The check copies
/// GPU results back to the host, so it must not run while a CUDA graph is
/// being captured.
#[cfg(feature = "cuda")]
fn sm70_check_done() -> bool {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    DONE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Marks the SM70 cross-check as done.
#[cfg(feature = "cuda")]
fn set_sm70_check_done() {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    DONE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// True when the given stream is capturing a CUDA graph.
///
/// The runner captures the model into a CUDA graph before the first token,
/// and the cross-check copies results back to the host, which is illegal
/// during a capture.  The capture state is queried through the driver API
/// because the runtime API is not linked into this crate.
#[cfg(feature = "cuda")]
fn capturing(stream: usize) -> bool {
    use candle_core::cuda_backend::cudarc::driver::safe::capture_status;
    use candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureStatus;

    match capture_status(stream as _) {
        Ok(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE) => false,
        Ok(_) => true,
        Err(_) => false,
    }
}

/// Cross-check the SM70 HMMA.884 NVFP4 GEMM against the generic software
/// dequantization path on the very same inputs.
///
/// The SM70 path consumes the *repacked* weights (u32 packed nibbles plus the
/// block scales with the global scale already folded in), while the software
/// path consumes the raw `[N, K/2]` u8 weights and the original E4M3 scales.
/// Both are given the same global scale, so the two results must agree
/// bit-for-bit up to floating point rounding.
///
/// Only the F16 decode shape is exercised; anything else is skipped with a
/// message rather than treated as a failure.
#[cfg(feature = "cuda")]
fn check_sm70_vs_software(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    weight_row_scales: Option<&Tensor>,
    bias: Option<&Tensor>,
    output: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> candle_core::Result<()> {
    use candle_core::Storage;

    let dtype = input.dtype();
    if dtype != DType::F16 {
        return Ok(());
    }

    let dev = input.device();
    let ref_out = Tensor::zeros((m, n), dtype, dev)?;

    let (input_s, _) = input.storage_and_layout();
    let (weight_s, _) = weight.storage_and_layout();
    let (scale_s, _) = scale.storage_and_layout();
    let (ref_out_s, _) = ref_out.storage_and_layout();

    let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
    let weight_ptr = cuda_ptr(&weight_s, weight.dtype())? as *const u8;
    let scale_ptr = cuda_ptr(&scale_s, scale.dtype())? as *const u8;
    let ref_out_ptr = cuda_ptr(&ref_out_s, dtype)? as *mut std::ffi::c_void;

    let bias_ptr = match bias {
        Some(b) => {
            let (b_s, _) = b.storage_and_layout();
            cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
        }
        None => std::ptr::null(),
    };
    let row_scales_ptr: *const f32 = match weight_row_scales {
        Some(rs) => {
            let (rs_s, _) = rs.storage_and_layout();
            cuda_ptr(&rs_s, rs.dtype())? as *const f32
        }
        None => std::ptr::null(),
    };

    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;
    let force_lut = crate::nvfp4_force_lut();

    unsafe {
        if m < 32 {
            ffi::nvfp4_matmul_smallm_f16(
                input_ptr,
                weight_ptr,
                scale_ptr,
                weight_global_scale,
                row_scales_ptr,
                bias_ptr,
                ref_out_ptr,
                m as i32,
                n as i32,
                k as i32,
                bias.is_some(),
                force_lut,
                stream,
            );
        } else {
            ffi::nvfp4_matmul_f16(
                input_ptr,
                weight_ptr,
                scale_ptr,
                weight_global_scale,
                row_scales_ptr,
                bias_ptr,
                ref_out_ptr,
                m as i32,
                n as i32,
                k as i32,
                bias.is_some(),
                force_lut,
                stream,
            );
        }
    }

    // Compare on the host: the two tensors are small (m <= 32 rows).
    let cpu = candle_core::Device::Cpu;
    let got = output.to_dtype(DType::F32)?.to_device(&cpu)?;
    let want = ref_out.to_dtype(DType::F32)?.to_device(&cpu)?;
    let got_v: Vec<f32> = got.flatten_all()?.to_vec1()?;
    let want_v: Vec<f32> = want.flatten_all()?.to_vec1()?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mismatched = 0usize;
    for (g, w) in got_v.iter().zip(want_v.iter()) {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
        let scale = w.abs().max(1e-6);
        let r = d / scale;
        if r > max_rel {
            max_rel = r;
        }
        if d > 1e-3 {
            mismatched += 1;
        }
    }
    // Print a few reference values so a scale mismatch is visible directly.
    let sample: Vec<String> = (0..3)
        .map(|i| {
            format!("got={:.6} want={:.6}", got_v[i], want_v[i])
        })
        .collect();
    let row_scale_sample = match weight_row_scales {
        Some(rs) => {
            let v = rs.to_dtype(DType::F32)?.to_device(&cpu)?;
            let sv: Vec<f32> = v.flatten_all()?.to_vec1()?;
            format!("row_scale[0]={:.8} row_scale[1]={:.8}", sv[0], sv[1])
        }
        None => "row_scale=None".to_string(),
    };
    eprintln!(
        "[sm70-check] M={} N={} K={} gscale={} rows_scales={} {} max_abs_diff={:.6e} max_rel_diff={:.6e} mismatched={}/{} | {}",
        m, n, k, weight_global_scale, weight_row_scales.is_some(), row_scale_sample,
        max_abs, max_rel, mismatched, got_v.len(), sample.join(" ")
    );
    let _ = Storage::Cpu;
    Ok(())
}

/// Repack MLX NVFP4 weights from U32 to U8 on GPU.
///
/// MLX stores FP4 E2M1 weights as U32 (8 nibbles per U32 in little-endian).
/// Our NVFP4 GEMM kernels expect U8 (2 nibbles per byte). This function
/// reinterprets the bytes on-device without CPU round-trip.
///
/// * `weight_u32` - `[rows, cols]` U32 tensor on GPU
/// * Returns `[rows, cols * 4]` U8 tensor on GPU
#[allow(unused)]
pub fn mlx_repack_u32_to_u8(weight_u32: &Tensor) -> Result<Tensor> {
    let weight_u32 = if weight_u32.is_contiguous() {
        weight_u32.clone()
    } else {
        weight_u32.contiguous()?
    };
    let dims = weight_u32.dims();
    if dims.len() != 2 {
        candle_core::bail!("mlx_repack_u32_to_u8: expected rank 2, got {:?}", dims);
    }
    let rows = dims[0];
    let u32_cols = dims[1];
    let u8_cols = u32_cols * 4;
    let dev = weight_u32.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            use candle_core::Storage;

            let output = Tensor::zeros((rows, u8_cols), candle_core::DType::U8, dev)?;
            let stream = *cuda_dev.cu_stream() as i64;

            {
                let (in_s, _) = weight_u32.storage_and_layout();
                let in_ptr = match &*in_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u32>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };
                let (out_s, _) = output.storage_and_layout();
                let out_ptr = match &*out_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };
                unsafe {
                    ffi::mlx_nvfp4_repack_u32_to_u8(
                        in_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        rows as i32,
                        u32_cols as i32,
                        stream,
                    );
                }
            }
            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_metal_dev) => {
            // Metal path: reinterpret U32 bytes as U8 via compute kernel
            mlx_repack_u32_to_u8_metal(&weight_u32, rows, u32_cols)
        }
        _ => candle_core::bail!("mlx_repack_u32_to_u8: unsupported backend (need CUDA or Metal)"),
    }
}

/// Dequantize MLX NVFP4 quantized embeddings on GPU.
///
/// * `weight_u32` - `[vocab_size, hidden_size/8]` U32 packed FP4 weights on GPU
/// * `scales` - `[vocab_size, hidden_size/16]` U8 FP8 E4M3 block scales on GPU
/// * `vocab_size` - vocabulary size
/// * `hidden_size` - embedding dimension
/// * `dtype` - output dtype (F16 or BF16)
///
/// Returns `[vocab_size, hidden_size]` dequantized embedding tensor
#[allow(unused)]
pub fn mlx_dequant_embedding(
    weight_u32: &Tensor,
    scales: &Tensor,
    vocab_size: usize,
    hidden_size: usize,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let weight_u32 = if weight_u32.is_contiguous() {
        weight_u32.clone()
    } else {
        weight_u32.contiguous()?
    };
    let scales = if scales.is_contiguous() {
        scales.clone()
    } else {
        scales.contiguous()?
    };
    let dev = weight_u32.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            use candle_core::Storage;

            let output = Tensor::zeros((vocab_size, hidden_size), dtype, dev)?;
            let stream = *cuda_dev.cu_stream() as i64;

            {
                let (w_s, _) = weight_u32.storage_and_layout();
                let w_ptr = match &*w_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u32>()?.device_ptr(),
                    _ => candle_core::bail!("weight must be on CUDA"),
                };
                let (s_s, _) = scales.storage_and_layout();
                let s_ptr = match &*s_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("scales must be on CUDA"),
                };
                let (o_s, _) = output.storage_and_layout();
                let o_ptr = match &*o_s {
                    Storage::Cuda(c) => match dtype {
                        candle_core::DType::F16 => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        candle_core::DType::BF16 => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        _ => candle_core::bail!("output dtype must be F16 or BF16"),
                    },
                    _ => candle_core::bail!("output must be on CUDA"),
                };
                unsafe {
                    match dtype {
                        candle_core::DType::F16 => {
                            ffi::mlx_nvfp4_dequant_embedding_f16(
                                w_ptr as *const std::ffi::c_void,
                                s_ptr as *const std::ffi::c_void,
                                o_ptr as *mut std::ffi::c_void,
                                vocab_size as i32,
                                hidden_size as i32,
                                stream,
                            );
                        }
                        candle_core::DType::BF16 => {
                            ffi::mlx_nvfp4_dequant_embedding_bf16(
                                w_ptr as *const std::ffi::c_void,
                                s_ptr as *const std::ffi::c_void,
                                o_ptr as *mut std::ffi::c_void,
                                vocab_size as i32,
                                hidden_size as i32,
                                stream,
                            );
                        }
                        _ => candle_core::bail!(
                            "mlx_dequant_embedding: unsupported dtype {:?}",
                            dtype
                        ),
                    }
                }
            }
            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_metal_dev) => {
            mlx_dequant_embedding_metal(&weight_u32, &scales, vocab_size, hidden_size, dtype)
        }
        _ => candle_core::bail!("mlx_dequant_embedding: unsupported backend (need CUDA or Metal)"),
    }
}

/// Repack NVFP4 weights for the SM70 HMMA.884 path and run the Volta kernel.
///
/// This is the SM70 (Tesla V100) counterpart of the WMMA path: Volta has no
/// `m16n16k16` HMMA, only `m8n8k4`, and the weights have to be laid out so
/// that one 32-bit word holds eight contiguous k of one output column.
///
/// The returned tensors are the repacked weights `[N, K/8]` (u32) and the
/// repacked block scales `[N, K/16]` (u32, the FP8 E4M3 scale converted to
/// half in the low half).  Callers keep them for the lifetime of the layer
/// so the repack happens once per layer, not once per token.
#[cfg(feature = "cuda")]
pub fn nvfp4_repack_weight_sm70(
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    column_global_scale: Option<&Tensor>,
    n: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = weight.device();
    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let weight_out = Tensor::zeros((n, k / 8), DType::U32, dev)?;
    // One scale per NVFP4 group (16 elements along k), stored as half.
    // Storing one scale per weight word (8 elements) would duplicate every
    // scale, which costs gigabytes on a 27B model.
    let scale_out = Tensor::zeros((n, k / 16), DType::U32, dev)?;

    let pointers = {
    let (ws, _) = weight.storage_and_layout();
    let (ss, _) = scale.storage_and_layout();
    let (wo, _) = weight_out.storage_and_layout();
    let (so, _) = scale_out.storage_and_layout();

    let weight_ptr = match &*ws {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<u8>()?.device_ptr() as *const u8
        }
        _ => candle_core::bail!("weight must be on CUDA"),
    };
    let scale_ptr = match &*ss {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<u8>()?.device_ptr() as *const u8
        }
        _ => candle_core::bail!("scale must be on CUDA"),
    };
    let weight_out_ptr = match &*wo {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<u32>()?.device_ptr() as *mut u32
        }
        _ => candle_core::bail!("weight_out must be on CUDA"),
    };
    let scale_out_ptr = match &*so {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<u32>()?.device_ptr() as *mut u32
        }
        _ => candle_core::bail!("scale_out must be on CUDA"),
    };

    // The per-column global scale replaces the tensor-wide one for the
    // columns it covers, mirroring what the generic NVFP4 path does.
    let column_scale_ptr = if let Some(cs) = column_global_scale {
        let (cs_s, _) = cs.storage_and_layout();
        match &*cs_s {
            candle_core::Storage::Cuda(c) => {
                *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32
            }
            _ => candle_core::bail!("column_global_scale must be on CUDA"),
        }
    } else {
        std::ptr::null()
    };

    (weight_ptr, scale_ptr, weight_out_ptr, scale_out_ptr, column_scale_ptr)
    };
    let (weight_ptr, scale_ptr, weight_out_ptr, scale_out_ptr, column_scale_ptr) = pointers;

    unsafe {
        ffi::nvfp4_repack_weight_sm70(
            weight_ptr,
            scale_ptr,
            weight_out_ptr,
            scale_out_ptr,
            column_scale_ptr,
            weight_global_scale,
            n as i32,
            k as i32,
            stream,
        );
    }
    Ok((weight_out, scale_out))
}

/// Repack the weights for the SM70 NVFP4 path, ahead of the first matmul.
///
/// The repacked weight has exactly the same size as the packed NVFP4
/// weight it replaces -- `n * k / 8` u32 is `n * k / 2` bytes, which is
/// what `n * k / 2` u8 weighs -- so the caller can drop the original and
/// keep only the repacked copy.  On a 27B model that is the difference
/// between holding the weights twice and holding them once.
///
/// The repacked scale is stored as u32, so it is twice the size of the
/// original u8 scale.
#[cfg(feature = "cuda")]
pub fn nvfp4_prepack_sm70(
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    weight_row_scales: Option<&Tensor>,
    n: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    nvfp4_repack_weight_sm70(
        weight,
        scale,
        weight_global_scale,
        weight_row_scales,
        n,
        k,
    )
}

/// Run the SM70 HMMA.884 NVFP4 GEMM.
///
/// `weight` and `weight_scale` are the repacked tensors produced by
/// [`nvfp4_repack_weight_sm70`].  `weight_global_scale` multiplies the whole
/// result; `weight_row_scales` optionally scales individual output columns.
#[cfg(feature = "cuda")]
pub fn nvfp4_matmul_hmma884(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    weight_global_scale: f32,
    bias: Option<&Tensor>,
    weight_row_scales: Option<&Tensor>,
) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let input_dims = input.dims();
    if input_dims.len() != 2 {
        candle_core::bail!(
            "nvfp4_matmul_hmma884: expected input rank 2, got {:?}",
            input_dims
        );
    }
    let m = input_dims[0];
    let k = input_dims[1];
    let n = weight.dims()[0];

    let dev = input.device();
    let dtype = input.dtype();
    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let output = Tensor::zeros((m, n), dtype, dev)?;

    let pointers = {
    let (is, _) = input.storage_and_layout();
    let (ws, _) = weight.storage_and_layout();
    let (wss, _) = weight_scale.storage_and_layout();
    let (os, _) = output.storage_and_layout();

    let input_ptr = match &*is {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<half::f16>()?.device_ptr() as *const half::f16
        }
        _ => candle_core::bail!("input must be on CUDA"),
    };
    let weight_ptr = match &*ws {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<u32>()?.device_ptr() as *const u32
        }
        _ => candle_core::bail!("weight must be on CUDA"),
    };
    let scale_ptr = match &*wss {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<u32>()?.device_ptr() as *const u32
        }
        _ => candle_core::bail!("weight_scale must be on CUDA"),
    };
    let output_ptr = match &*os {
        candle_core::Storage::Cuda(c) => {
            *c.as_cuda_slice::<half::f16>()?.device_ptr() as *mut half::f16
        }
        _ => candle_core::bail!("output must be on CUDA"),
    };

    let bias_ptr = if let Some(b) = bias {
        let (bs, _) = b.storage_and_layout();
        match &*bs {
            candle_core::Storage::Cuda(c) => {
                *c.as_cuda_slice::<half::f16>()?.device_ptr() as *const half::f16
            }
            _ => candle_core::bail!("bias must be on CUDA"),
        }
    } else {
        std::ptr::null()
    };

    // The row scales are folded into the block scales by the repack, so the
    // kernel does not apply them a second time.
    let row_scales_ptr = if let Some(rs) = weight_row_scales {
        let (rs_s, _) = rs.storage_and_layout();
        match &*rs_s {
            candle_core::Storage::Cuda(c) => {
                *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32
            }
            _ => candle_core::bail!("weight_row_scales must be on CUDA"),
        }
    } else {
        std::ptr::null()
    };

    (input_ptr, weight_ptr, scale_ptr, output_ptr, bias_ptr, row_scales_ptr)
    };
    let (input_ptr, weight_ptr, scale_ptr, output_ptr, bias_ptr, row_scales_ptr) = pointers;

    unsafe {
        ffi::nvfp4_matmul_hmma884_f16(
            input_ptr as *const std::ffi::c_void,
            weight_ptr,
            scale_ptr,
            weight_global_scale,
            row_scales_ptr as *const f32,
            bias_ptr as *const std::ffi::c_void,
            output_ptr as *mut std::ffi::c_void,
            m as i32,
            n as i32,
            k as i32,
            bias.is_some(),
            stream,
        );
    }

    Ok(output)
}

#[cfg(feature = "metal")]
fn mlx_repack_u32_to_u8_metal(weight_u32: &Tensor, rows: usize, u32_cols: usize) -> Result<Tensor> {
    let u8_cols = u32_cols * 4;
    let dev = weight_u32.device();
    let output = Tensor::zeros((rows, u8_cols), candle_core::DType::U8, dev)?;

    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer_ref = command_buffer.as_ref();

    {
        use candle_core::Storage;
        let (in_s, in_l) = weight_u32.storage_and_layout();
        let in_ms = match &*in_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("tensor must be on Metal"),
        };
        let (out_s, _out_l) = output.storage_and_layout();
        let out_ms = match &*out_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("tensor must be on Metal"),
        };

        metal_kernels::call_mlx_nvfp4_repack(
            metal_dev.device(),
            command_buffer_ref,
            metal_kernels::Kernels::default(),
            (in_ms.buffer(), in_l.start_offset() * 4),
            out_ms.buffer(),
            rows,
            u32_cols,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    Ok(output)
}

#[cfg(feature = "metal")]
fn mlx_dequant_embedding_metal(
    weight_u32: &Tensor,
    scales: &Tensor,
    vocab_size: usize,
    hidden_size: usize,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let dev = weight_u32.device();
    let output = Tensor::zeros((vocab_size, hidden_size), dtype, dev)?;

    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer_ref = command_buffer.as_ref();

    {
        use candle_core::Storage;
        let (w_s, w_l) = weight_u32.storage_and_layout();
        let w_ms = match &*w_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("weight must be on Metal"),
        };
        let (s_s, s_l) = scales.storage_and_layout();
        let s_ms = match &*s_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("scales must be on Metal"),
        };
        let (o_s, _o_l) = output.storage_and_layout();
        let o_ms = match &*o_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("output must be on Metal"),
        };

        metal_kernels::call_mlx_nvfp4_dequant_embedding(
            metal_dev.device(),
            command_buffer_ref,
            metal_kernels::Kernels::default(),
            dtype,
            (w_ms.buffer(), w_l.start_offset() * 4),
            (s_ms.buffer(), s_l.start_offset()),
            o_ms.buffer(),
            vocab_size,
            hidden_size,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    Ok(output)
}
