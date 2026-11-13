#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[cfg(feature = "cuda")]
static SM_CACHE: OnceLock<Mutex<HashMap<usize, Option<i32>>>> = OnceLock::new();

/// The compute capability of a device, without creating a context.
///
/// `cuDeviceGetAttribute` is one of the few driver calls that answer before
/// a context exists, so this can be used from a process that must not
/// occupy device memory -- the engine process, which only orchestrates the
/// runners, would otherwise take a CUDA context on the first device and
/// its footprint would compete with the model for VRAM.
#[cfg(feature = "cuda")]
pub fn sm_version_without_context(ordinal: i32) -> Option<i32> {
    use candle_core::cuda_backend::cudarc::driver::result;

    // The driver has to be initialised before any other call.
    result::init().ok()?;
    let device = result::device::get(ordinal).ok()?;
    let major = unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .ok()?;
    let minor = unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .ok()?;
    Some(major * 10 + minor)
}

#[cfg(feature = "cuda")]
pub fn compute_capability(dev: &CudaDevice) -> Option<(i32, i32)> {
    let major = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    let minor = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .ok()?;
    Some((major, minor))
}

#[cfg(feature = "cuda")]
pub fn sm_version(dev: &CudaDevice) -> Option<i32> {
    // Key by the address of this device instance
    let key = (dev as *const CudaDevice) as usize;

    let cache = SM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Fast path: return cached value if present
    if let Some(v) = cache.lock().unwrap().get(&key).copied() {
        return v;
    }

    // Compute outside the lock to keep the critical section small
    let computed = compute_capability(dev).map(|(major, minor)| major * 10 + minor);

    // Store (a second thread might store first; that is fine)
    cache.lock().unwrap().insert(key, computed);

    computed
}

// Keep architecture checks available to backend-independent call sites. On a
// non-CUDA build there is no CUDA device whose compute capability can be read.
#[cfg(not(feature = "cuda"))]
pub fn sm_version<T>(_: &T) -> Option<i32> {
    None
}
