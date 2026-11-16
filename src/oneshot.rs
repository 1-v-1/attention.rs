// SPDX-License-Identifier: Apache-2.0
// One-shot push allreduce for world_size == 2 (1Cat custom_all_reduce push
// protocol): the peer partial arrives in our slot region over NVLink and the
// kernel spin-reads it until no pack carries the sentinel any more.
use candle_core::{CudaDevice, Device, Result};

use crate::kernels;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;

const HDR_BYTES: usize = 128;
static OWN_BASE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub struct OneShotArena {
    own_base: usize,
    peer_base: usize,
    cursor: std::sync::Mutex<usize>,
}

fn cuda_of(dev: &Device) -> Result<&CudaDevice> {
    dev.as_cuda_device()
}

/// Allocate the staging arena once and return its IPC handle so the caller
/// can hand it to the peer before opening the mirror.
pub fn alloc_handle(dev: &Device) -> Result<[u8; 64]> {
    let _ = cuda_of(dev)?;
    let mut h = [0u8; 64];
    unsafe {
        let mut p: u64 = 0;
        kernels::ffi::oneshot_alloc(192 << 20, &mut p, h.as_mut_ptr());
        let _ = OWN_BASE.set(p as usize);
    }
    Ok(h)
}

/// Map the peer's arena (already exchanged) and build the slot cursor.
pub fn open(dev: &Device, peer_handle: &[u8; 64]) -> Result<OneShotArena> {
    let _ = cuda_of(dev)?;
    let peer = unsafe {
        let mut p: u64 = 0;
        kernels::ffi::oneshot_open(peer_handle.as_ptr(), &mut p);
        p
    };
    Ok(OneShotArena {
        own_base: *OWN_BASE.get().unwrap(),
        peer_base: peer as usize,
        cursor: std::sync::Mutex::new(HDR_BYTES),
    })
}

impl OneShotArena {
    /// Carve a region for one layer: 128B header (epoch counter) plus four
    /// slots (two epochs x two ranks) of `slot_bytes` each.
    pub fn reserve(self: &Self, slot_bytes: usize) -> usize {
        let mut c = self.cursor.lock().unwrap();
        let at = *c;
        unsafe { kernels::ffi::oneshot_clear(
            (self.own_base + at) as *mut core::ffi::c_void, 4); }
        *c += HDR_BYTES + 2 * slot_bytes;
        at
    }

    /// world-2 one-shot fold: push `x` into the peer slot, spin-read our slot
    /// until the peer's half is complete, return the sum with `x`'s shape.
    pub fn fold(self: &Self, s: &candle_core::CudaStorage,
                l: &candle_core::Layout, off: usize, stride: usize,
                rank: usize) -> Result<candle_core::CudaStorage> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::WrapErr;
        let cuda = s.device().clone();
        let stream = *cuda.cu_stream() as i64;
        let n = l.shape().elem_count();
        let x_ptr = {
            let full = s.as_cuda_slice::<half::f16>()?;
            *full.device_ptr() as usize + l.start_offset() * 2
        };
        let mut out = unsafe { cuda.alloc::<half::f16>(n) }.w()?;
        let o_ptr = *out.device_ptr() as usize;
        if std::env::var("XINFER_ONESHOT_DEBUG").is_ok() { eprintln!("[oneshot] off={off} stride={stride} n={n} rank={rank}"); }
        unsafe {
            kernels::ffi::oneshot_fold(
                x_ptr as *const core::ffi::c_void,
                o_ptr as *mut core::ffi::c_void,
                (self.own_base + off + HDR_BYTES) as *mut core::ffi::c_void,
                (self.peer_base + off + HDR_BYTES) as *mut core::ffi::c_void,
                (self.own_base + off) as *mut u32,
                stride as u32, n as u32, rank as u32, stream);
        }
        Ok(candle_core::CudaStorage::wrap_cuda_slice(out, cuda))
    }
}
