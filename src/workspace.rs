//! Unified workspace buffer management for FlashInfer, CUTLASS, and other GPU operations.
//!
//! This module provides a centralized workspace allocation strategy that eliminates
//! redundant buffer allocations across different modules (moe.rs, fp8_linear.rs,
//! nvfp4_linear.rs, mxfp4_linear.rs, flashinfer.rs, mla.rs).
//!
//! # Workspace Layout (when `flashinfer` feature is enabled)
//!
//! ```text
//! FLASHINFER FLOAT BUFFER (512 MiB total):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  FlashInfer Plan Float (256 MiB)  │  FlashInfer Scratch (256 MiB)
//! │  [0..256 MiB)                     │  [256 MiB..512 MiB)        │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! INT BUFFER (128 MiB):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  FlashInfer Plan Int (128 MiB)                                  │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! PINNED HOST BUFFER (128 MiB):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Page-locked host memory for FlashInfer scheduler              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Workspace Layout (when `flashinfer` feature is disabled)
//!
//! CUTLASS keeps a separate per-thread device workspace. When `flashinfer` is
//! enabled, CUTLASS does not alias the FlashInfer float buffer because that
//! shared path is not stable for the FP8 CUTLASS GEMM kernels.
//!
//! # Native MLA decode workspace (FlashInfer disabled / skipped)
//!
//! Grow-only f32 buffer for split-K temps (`tmp_out`, `tmp_max`, `tmp_sum`),
//! lazily allocated on first `mla_paged_decode` use (graph warmup). Separate
//! eager / CUDA-graph slots keep captured addresses stable.
//!
//! # Thread Safety
//!
//! Workspaces are thread-local to avoid synchronization overhead. Each thread
//! maintains its own workspace buffers, which are lazily initialized on first use.

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{sys, CudaSlice, DevicePtr};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::WrapErr;
use candle_core::Result;

/// Size of the FlashInfer plan region within the float buffer.
pub const FLASHINFER_PLAN_FLOAT_SIZE: usize = 256 * 1024 * 1024;

/// Offset where GEMM scratch space begins within the float buffer.
pub const GEMM_SCRATCH_FLOAT_OFFSET: usize = FLASHINFER_PLAN_FLOAT_SIZE;

/// Size of the GEMM scratch region (used by CUTLASS operations).
///
/// The region only exists for the CUTLASS kernels, which are behind their
/// own feature flag.  Without it the float buffer would be half its size
/// for nothing: on a 27B model on a 16 GB card that is 512 MiB per card,
/// which is the difference between a KV cache and none at all.
#[cfg(feature = "cutlass")]
pub const GEMM_SCRATCH_FLOAT_SIZE: usize = 256 * 1024 * 1024;

/// The GEMM scratch region is empty without the CUTLASS kernels.
#[cfg(not(feature = "cutlass"))]
pub const GEMM_SCRATCH_FLOAT_SIZE: usize = 0;

/// Total size of the float workspace buffer.
///
/// The buffer holds the FlashInfer plan region, and the GEMM scratch region
/// when the CUTLASS kernels are compiled in.
pub const WORKSPACE_FLOAT_SIZE: usize = FLASHINFER_PLAN_FLOAT_SIZE + GEMM_SCRATCH_FLOAT_SIZE;

/// Size of the int workspace buffer (128 MiB).
///
/// This is the upper bound.  The buffer is allocated lazily at
/// [`WORKSPACE_INT_INITIAL_SIZE`] and grows to what the decode plan asks
/// for, so the 128 MiB is a cap rather than an allocation.
pub const WORKSPACE_INT_SIZE: usize = 128 * 1024 * 1024;

/// The size the int workspace starts at.
///
/// The decode plan carves five small index arrays out of it.  Their size is
/// driven by the padded batch size, so a few hundred kilobytes cover every
/// batch a 16 GB card can hold; the buffer grows on demand afterwards.
pub const WORKSPACE_INT_INITIAL_SIZE: usize = 4 * 1024 * 1024;

/// Size of the reduced float buffer used by the eager workspace on Volta.
///
/// The FlashInfer prefill kernels do not run on Volta, so the 256 MiB float
/// buffer would be wasted.  The decode plan still needs some float scratch
/// for the split-KV path, so the buffer is not dropped entirely.
pub const SM70_EAGER_FLOAT_SIZE: usize = 8 * 1024 * 1024;

/// Fallback workspace size when flashinfer is disabled.
#[cfg(feature = "cuda")]
pub const CUTLASS_WORKSPACE_FALLBACK_SIZE: usize = 512 * 1024 * 1024;

/// Describes a region within a workspace buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceRegion {
    pub offset: usize,
    pub size: usize,
}

/// Collection of all workspace regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceRegions {
    pub plan_float: WorkspaceRegion,
    pub gemm_scratch_float: WorkspaceRegion,
    pub plan_int: WorkspaceRegion,
}

/// Pre-computed workspace region definitions.
pub const WORKSPACE_REGIONS: WorkspaceRegions = WorkspaceRegions {
    plan_float: WorkspaceRegion {
        offset: 0,
        size: FLASHINFER_PLAN_FLOAT_SIZE,
    },
    gemm_scratch_float: WorkspaceRegion {
        offset: GEMM_SCRATCH_FLOAT_OFFSET,
        size: GEMM_SCRATCH_FLOAT_SIZE,
    },
    plan_int: WorkspaceRegion {
        offset: 0,
        size: WORKSPACE_INT_SIZE,
    },
};

/// The device memory one workspace slot occupies.
///
/// The caller needs this to reserve the right amount of VRAM: the buffer
/// is allocated lazily on first use, so a caller that plans its allocations
/// around the free memory has to know the size up front.
pub fn workspace_gpu_bytes() -> usize {
    WORKSPACE_FLOAT_SIZE + WORKSPACE_INT_INITIAL_SIZE
}

/// The device memory the reduced eager workspace occupies on Volta.
///
/// The eager slot keeps the int buffer and the page-locked host buffer, and
/// only the float buffer is reduced.
pub fn sm70_eager_workspace_gpu_bytes() -> usize {
    SM70_EAGER_FLOAT_SIZE + WORKSPACE_INT_INITIAL_SIZE
}

/// Returns the workspace regions after validating they don't overlap.
pub fn workspace_regions() -> WorkspaceRegions {
    debug_assert!(
        WORKSPACE_REGIONS.plan_float.offset + WORKSPACE_REGIONS.plan_float.size
            <= WORKSPACE_FLOAT_SIZE
    );
    debug_assert!(
        WORKSPACE_REGIONS.gemm_scratch_float.offset
            >= WORKSPACE_REGIONS.plan_float.offset + WORKSPACE_REGIONS.plan_float.size
    );
    debug_assert!(
        WORKSPACE_REGIONS.gemm_scratch_float.offset + WORKSPACE_REGIONS.gemm_scratch_float.size
            <= WORKSPACE_FLOAT_SIZE
    );
    debug_assert!(
        WORKSPACE_REGIONS.plan_int.offset + WORKSPACE_REGIONS.plan_int.size <= WORKSPACE_INT_SIZE
    );
    WORKSPACE_REGIONS
}

/// Adds an offset to a base pointer, returning a pointer to the specified region.
pub fn add_workspace_offset(
    base: *mut std::ffi::c_void,
    region: WorkspaceRegion,
) -> *mut std::ffi::c_void {
    debug_assert!(!base.is_null());
    unsafe { (base as *mut u8).add(region.offset) as *mut std::ffi::c_void }
}

// ============================================================================
// CUDA-specific workspace management
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// Page-locked host buffer for FlashInfer scheduler operations.
    pub struct PinnedHostBuffer {
        ptr: *mut std::ffi::c_void,
        size: usize,
    }

    impl PinnedHostBuffer {
        pub fn new(size: usize) -> Result<Self> {
            if size == 0 {
                candle_core::bail!("Pinned host buffer size must be > 0");
            }
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            unsafe {
                sys::lib()
                    .cuMemAllocHost_v2(&mut ptr, size)
                    .result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("cuMemAllocHost_v2 failed: {e:?}"))
                    })?
            }
            if ptr.is_null() {
                candle_core::bail!("cuMemAllocHost_v2 returned null pointer");
            }
            Ok(Self { ptr, size })
        }

        pub fn as_ptr(&self) -> *mut std::ffi::c_void {
            self.ptr
        }

        pub fn size(&self) -> usize {
            self.size
        }

        /// A placeholder that owns nothing.
        ///
        /// Used to move the real buffer out of a workspace before it is
        /// replaced, so that the old allocation can be kept alive instead of
        /// being freed.
        pub fn empty() -> Self {
            Self {
                ptr: std::ptr::null_mut(),
                size: 0,
            }
        }
    }

    impl Drop for PinnedHostBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    let _ = sys::lib().cuMemFreeHost(self.ptr).result();
                }
            }
        }
    }

    // SAFETY: The pinned buffer pointer is allocated per-thread and only accessed
    // within that thread's context.
    unsafe impl Send for PinnedHostBuffer {}

    /// Primary workspace for FlashInfer operations.
    pub struct FlashInferWorkspace {
        pub float_buffer: CudaSlice<u8>,
        pub int_buffer: CudaSlice<u8>,
        pub pinned_host: PinnedHostBuffer,
        pub device_ordinal: usize,
        /// Size of the float buffer, which is smaller than the region table
        /// on Volta.
        pub float_size: usize,
        /// Size of the int buffer.
        ///
        /// The decode plan carves a few small arrays out of it, so the
        /// buffer is sized to what the plan actually needs instead of a
        /// fixed 128 MiB.
        pub int_size: usize,
    }

    impl FlashInferWorkspace {
        /// Splits the workspace into the parts that a replacement needs and
        /// the page-locked buffer that has to survive the replacement.
        ///
        /// `CudaSlice` is reference counted, so cloning it keeps the device
        /// allocation alive.  The page-locked buffer is not cloneable, so it
        /// is handed over separately.
        fn split(&mut self) -> (CudaSlice<u8>, CudaSlice<u8>, usize, PinnedHostBuffer) {
            let pinned = std::mem::replace(&mut self.pinned_host, PinnedHostBuffer::empty());
            (
                self.float_buffer.clone(),
                self.int_buffer.clone(),
                self.float_size,
                pinned,
            )
        }
    }

    /// Dedicated CUTLASS workspace, kept separate from FlashInfer buffers.
    #[cfg(feature = "cutlass")]
    pub struct CutlassWorkspace {
        pub buffer: CudaSlice<u8>,
        pub size: usize,
        pub device_ordinal: usize,
    }

    /// Specialized FP8 workspace for FlashInfer blockscale operations.
    #[cfg(feature = "flashinfer")]
    pub struct FlashInferFp8Workspace {
        pub buffer: CudaSlice<u8>,
        pub size: usize,
        pub device_ordinal: usize,
    }

    thread_local! {
        /// Primary FlashInfer workspace (normal operations).
        ///
        /// The slot holds **one workspace per device ordinal**.  A single
        /// thread can serve several devices -- the multi-rank runner drives
        /// every rank from the same thread and switches the active device
        /// between steps -- and a CUDA graph captured for one rank bakes in
        /// the *host* address of the page-locked buffer that the FlashInfer
        /// plan was materialised into.  Re-creating that buffer whenever the
        /// active device changes invalidates every captured graph:
        /// `cuGraphLaunch` then fails with `CUDA_ERROR_INVALID_VALUE` and the
        /// KV cache is never updated.
        #[cfg(feature = "flashinfer")]
        pub static WORKSPACE: std::cell::RefCell<HashMap<usize, FlashInferWorkspace>> =
            std::cell::RefCell::new(HashMap::new());

        /// FlashInfer workspace for CUDA graph captures (separate to avoid interference).
        #[cfg(feature = "flashinfer")]
        pub static WORKSPACE_GRAPH: std::cell::RefCell<HashMap<usize, FlashInferWorkspace>> =
            std::cell::RefCell::new(HashMap::new());

        /// Dedicated CUTLASS workspaces, split by execution domain so decode-graph
        /// and MTP/DFlash-verify-graph captures never alias the same device pointer.
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        pub static CUTLASS_WORKSPACE_EAGER: std::cell::RefCell<HashMap<usize, CutlassWorkspace>> =
            std::cell::RefCell::new(HashMap::new());
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        pub static CUTLASS_WORKSPACE_DECODE_GRAPH: std::cell::RefCell<HashMap<usize, CutlassWorkspace>> =
            std::cell::RefCell::new(HashMap::new());
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        pub static CUTLASS_WORKSPACE_MTP_GRAPH: std::cell::RefCell<HashMap<usize, CutlassWorkspace>> =
            std::cell::RefCell::new(HashMap::new());

        /// Specialized FP8 blockscale workspace.
        #[cfg(feature = "flashinfer")]
        pub static FLASHINFER_FP8_WORKSPACE_EAGER: std::cell::RefCell<HashMap<usize, FlashInferFp8Workspace>> =
            std::cell::RefCell::new(HashMap::new());
        #[cfg(feature = "flashinfer")]
        pub static FLASHINFER_FP8_WORKSPACE_DECODE_GRAPH: std::cell::RefCell<HashMap<usize, FlashInferFp8Workspace>> =
            std::cell::RefCell::new(HashMap::new());
        #[cfg(feature = "flashinfer")]
        pub static FLASHINFER_FP8_WORKSPACE_MTP_GRAPH: std::cell::RefCell<HashMap<usize, FlashInferFp8Workspace>> =
            std::cell::RefCell::new(HashMap::new());
    }

    /// Initializes or retrieves the FlashInfer workspace for the given device.
    ///
    /// Returns (float_ptr, int_ptr, pinned_host_ptr, pinned_host_size).
    #[cfg(feature = "flashinfer")]
    pub fn get_or_init_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        for_cuda_graph: bool,
    ) -> Result<(
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        usize,
    )> {
        let ws_cell = if for_cuda_graph {
            &WORKSPACE_GRAPH
        } else {
            &WORKSPACE
        };
        ws_cell.with(|ws| {
            let mut ws = ws.borrow_mut();
            let ordinal = dev.ordinal();

            // On Volta the FlashInfer prefill kernels are not available --
            // they need the `cp.async` of SM80 -- so the prefill never runs
            // and the eager float buffer would only be touched by the
            // prefill plan.  The *decode* plan still needs the int buffer
            // and the page-locked host buffer: without them the adapter
            // gives up before it fills in the plan, the padded batch size
            // stays zero and the decode kernel is launched with an empty
            // grid, which fails with "invalid configuration argument".
            //
            // So the eager slot keeps the int buffer and the page-locked
            // buffer, and only the float buffer is left out on Volta.  That
            // saves 256 MiB per card, which on a 27B model on a 16 GB card is
            // the difference between having a KV cache and having none.
            // The CUDA-graph slot has the same problem: the FlashInfer
            // prefill kernels need SM80, so the prefill plan never runs on
            // Volta and the graph slot's float buffer is only touched by
            // that plan.  The decode plan uses the int buffer and the
            // page-locked host buffer, which are kept in both slots.
            //
            // Leaving the 256 MiB float buffer out of the graph slot as
            // well saves another 248 MiB per card.  On a 27B model on a
            // 16 GB card that is the difference between a usable KV cache
            // and a tiny one.
            let skip_float_buffer =
                crate::cuda_utils::sm_version(dev) == Some(70);
            // The decode plan still asks the float buffer for scratch when it
            // splits the KV cache over several chunks, so the reduced buffer
            // has to be big enough for that.  The split path needs
            // `num_qo_heads * padded_batch_size * HEAD_DIM` floats for the
            // partial results plus the same number of floats for the
            // softmax statistics; a few megabytes cover the batch sizes a
            // 16 GB card can hold.
            let float_size = if skip_float_buffer {
                SM70_EAGER_FLOAT_SIZE
            } else {
                WORKSPACE_FLOAT_SIZE
            };

            // The int buffer holds a handful of index arrays whose size is
            // driven by the padded batch size, not by the model.  It starts
            // small and grows on demand, so a 27B model on a 16 GB card does
            // not have to reserve 128 MiB per card for a few kilobytes of
            // indices.
            let int_size = WORKSPACE_INT_INITIAL_SIZE;

            if !ws.contains_key(&ordinal) {
                let float_buffer = unsafe { dev.alloc::<u8>(float_size.max(1)) }.w()?;
                let int_buffer = unsafe { dev.alloc::<u8>(int_size.max(1)) }.w()?;
                let pinned_host = PinnedHostBuffer::new(int_size)?;
                ws.insert(
                    ordinal,
                    FlashInferWorkspace {
                        float_buffer,
                        int_buffer,
                        pinned_host,
                        device_ordinal: ordinal,
                        float_size,
                        int_size,
                    },
                );
            }

            let workspace = ws.get(&ordinal).unwrap();
            Ok((
                *workspace.float_buffer.device_ptr() as *mut std::ffi::c_void,
                *workspace.int_buffer.device_ptr() as *mut std::ffi::c_void,
                workspace.pinned_host.as_ptr(),
                workspace.pinned_host.size(),
            ))
        })
    }

    /// Allocate the graph workspace up front, outside of a graph capture.
    ///
    /// The decode graph needs a plan workspace of its own, because the
    /// addresses it hands out have to stay stable across replays.  The
    /// obvious place to create it is the first call that needs it -- but that
    /// call happens *inside* the capture, and a buffer that is allocated
    /// during a capture comes out of the capture's memory pool.  A pool
    /// allocation cannot be handed back to the driver (the graphs keep
    /// referring to it), so the buffer would stay resident in the pool for
    /// the whole lifetime of the process, inflating the pool by the full
    /// size of the workspace.
    ///
    /// Creating it before the capture starts puts it in the device's default
    /// pool instead, where the allocator is free to reuse or release it.
    ///
    /// This is a no-op if the workspace already exists.
    pub fn preallocate_graph_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
    ) -> Result<()> {
        get_or_init_workspace(dev, true)?;
        Ok(())
    }

    /// Returns workspace pointers for FlashInfer plan operations.
    ///
    /// Grows the int buffer of a workspace slot to at least `needed` bytes.
    ///
    /// The decode plan reports how many bytes it carved out of the int
    /// workspace.  The buffer is only reallocated when the plan asks for
    /// more than the slot currently holds, so the common case costs
    /// nothing.
    ///
    /// The old buffers are **never released**.  A CUDA graph that was
    /// captured while the plan was materialised holds a `cudaMemcpyAsync`
    /// node whose source is the *host* address of the page-locked buffer
    /// (FlashInfer's `DecodePlanImpl` fills the page-locked buffer and then
    /// copies it to the device).  Freeing that buffer would leave the graph
    /// pointing at unmapped host memory, and every later `cuGraphLaunch`
    /// would fail with `CUDA_ERROR_INVALID_VALUE`.  Keeping the old buffers
    /// alive for the lifetime of the process costs a few megabytes and makes
    /// the captured addresses permanently valid.
    pub fn ensure_int_capacity(
        dev: &candle_core::cuda_backend::CudaDevice,
        for_cuda_graph: bool,
        needed: usize,
    ) -> Result<()> {
        let ws_cell = if for_cuda_graph {
            &WORKSPACE_GRAPH
        } else {
            &WORKSPACE
        };
        ws_cell.with(|ws| {
            let mut ws = ws.borrow_mut();
            let ordinal = dev.ordinal();
            let current = match ws.get(&ordinal) {
                Some(existing) => existing.int_size,
                None => 0,
            };
            if needed <= current {
                return Ok(());
            }
            // Round up to a power of two so that a plan that grows a little
            // does not reallocate on every call.
            let mut target = WORKSPACE_INT_INITIAL_SIZE.max(1);
            while target < needed {
                target *= 2;
            }
            let int_buffer = unsafe { dev.alloc::<u8>(target) }.w()?;
            let pinned_host = PinnedHostBuffer::new(target)?;
            let (float_buffer, _old_int_buffer, float_size, old_pinned_host) = {
                let existing = ws
                    .get_mut(&ordinal)
                    .ok_or_else(|| candle_core::Error::msg("workspace has no entry"))?;
                existing.split()
            };
            ws.insert(
                ordinal,
                FlashInferWorkspace {
                    float_buffer,
                    int_buffer,
                    pinned_host,
                    device_ordinal: ordinal,
                    float_size,
                    int_size: target,
                },
            );
            // Keep the previous page-locked buffer alive on purpose: a
            // captured CUDA graph refers to its host address in the memcpy
            // node it captured, and freeing it would leave the graph
            // pointing at unmapped memory.
            std::mem::forget(old_pinned_host);
            Ok(())
        })
    }

    /// Diagnostic: compare the device plan buffer with the page-locked host
    /// buffer that the plan is materialized from.
    ///
    /// The captured CUDA graph copies the plan into the device buffer with a
    /// memcpy node that was recorded at capture time.  If that node does not
    /// run -- or if it copies into a buffer that the driver has since
    /// reallocated -- the decode kernel reads the *capture-time* plan and
    /// therefore the wrong number of KV tiles.
    #[cfg(feature = "flashinfer")]
    pub fn diag_plan_device_matches(
        dev: &candle_core::cuda_backend::CudaDevice,
        for_cuda_graph: bool,
    ) -> Result<(bool, String)> {
        let ws_cell = if for_cuda_graph {
            &WORKSPACE_GRAPH
        } else {
            &WORKSPACE
        };
        let (int_buffer, pinned_ptr, n) = ws_cell.with(|ws| {
            let ws = ws.borrow();
            let existing = ws
                .get(&dev.ordinal())
                .expect("no flashinfer workspace");
            (
                existing.int_buffer.clone(),
                existing.pinned_host.as_ptr(),
                existing
                    .int_size
                    .min(existing.pinned_host.size())
                    .min(1 << 20),
            )
        };
        let dev_bytes = dev
            .dtoh_sync_copy(&int_buffer)
            .map_err(|e| candle_core::Error::Msg(format!("dtoh failed: {e:?}")))?;
        let host_bytes = unsafe { std::slice::from_raw_parts(pinned_ptr as *const u8, n) };
        let n = std::cmp::min(n, dev_bytes.len());
        let mut first_diff = None;
        for i in 0..n {
            if dev_bytes[i] != host_bytes[i] {
                first_diff = Some(i);
                break;
            }
        }
        let dump = |bytes: &[u8], off: usize| -> String {
            let mut s = String::new();
            for i in 0..8 {
                let b = off + 4 * i;
                if b + 4 > bytes.len() {
                    break;
                }
                let v = i32::from_le_bytes([
                    bytes[b],
                    bytes[b + 1],
                    bytes[b + 2],
                    bytes[b + 3],
                ]);
                s.push_str(&format!("{} ", v));
            }
            s
        };
        let report = format!(
            "match={} first_diff={:?} dev_o_indptr[0..8]={} host_o_indptr[0..8]={}",
            first_diff.is_none(),
            first_diff,
            dump(&dev_bytes, 1600),
            dump(host_bytes, 1600)
        );
        Ok((first_diff.is_none(), report))
    }

    /// Returns (float_ptr, float_size, int_ptr, int_size, page_locked_ptr, page_locked_size).
    #[cfg(feature = "flashinfer")]
    pub fn get_plan_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        for_cuda_graph: bool,
    ) -> Result<(
        *mut std::ffi::c_void,
        usize,
        *mut std::ffi::c_void,
        usize,
        *mut std::ffi::c_void,
        usize,
    )> {
        let regions = workspace_regions();
        let (float_ptr, int_ptr, page_locked_ptr, page_locked_size) =
            get_or_init_workspace(dev, for_cuda_graph)?;
        // The float buffer is smaller than the region on Volta, so the
        // reported size has to be the size that was actually allocated.
        let float_size = if float_ptr.is_null() {
            0
        } else {
            float_buffer_size(dev, for_cuda_graph)
        };
        // The int region is sized to what the buffer actually holds, not to
        // the 128 MiB upper bound: FlashInfer uses the size as the capacity
        // it may carve out of, and a smaller number means a smaller
        // allocation.
        let int_size = {
            let ws_cell = if for_cuda_graph {
                &WORKSPACE_GRAPH
            } else {
                &WORKSPACE
            };
            ws_cell.with(|ws| {
                let ws = ws.borrow();
                ws.get(&dev.ordinal()).map(|existing| existing.int_size).unwrap_or(0)
            })
        };
        Ok((
            add_workspace_offset(float_ptr, regions.plan_float),
            float_size,
            add_workspace_offset(int_ptr, regions.plan_int),
            int_size,
            page_locked_ptr,
            page_locked_size,
        ))
    }

    /// Returns the size of the float buffer that the eager or the graph
    /// workspace actually allocated.
    ///
    /// The Volta eager slot uses a reduced buffer, so the size cannot be
    /// taken from the region table.
    fn float_buffer_size(
        dev: &candle_core::cuda_backend::CudaDevice,
        for_cuda_graph: bool,
    ) -> usize {
        let ws_cell = if for_cuda_graph {
            &WORKSPACE_GRAPH
        } else {
            &WORKSPACE
        };
        ws_cell.with(|ws| {
            let ws = ws.borrow();
            ws.get(&dev.ordinal()).map(|existing| existing.float_size).unwrap_or(0)
        })
    }

    /// Returns the FlashInfer GEMM scratch region from the shared float buffer.
    ///
    /// This remains available for FlashInfer-owned scratch usage, but CUTLASS uses
    /// a separate dedicated workspace.
    #[cfg(feature = "flashinfer")]
    pub fn get_gemm_scratch_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        let regions = workspace_regions();
        let (float_ptr, _, _, _) = get_or_init_workspace(dev, false)?;
        Ok((
            add_workspace_offset(float_ptr, regions.gemm_scratch_float),
            regions.gemm_scratch_float.size,
        ))
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn get_or_init_cutlass_workspace_slot(
        cell: &std::cell::RefCell<HashMap<usize, CutlassWorkspace>>,
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        let mut slot = cell.borrow_mut();
        let ordinal = dev.ordinal();
        let alloc_size = required_size.max(CUTLASS_WORKSPACE_FALLBACK_SIZE).max(1);
        let needs_init = match slot.get(&ordinal) {
            None => true,
            Some(existing) => existing.size < alloc_size,
        };

        if needs_init {
            let buffer = unsafe { dev.alloc::<u8>(alloc_size) }.w()?;
            slot.insert(
                ordinal,
                CutlassWorkspace {
                    buffer,
                    size: alloc_size,
                    device_ordinal: ordinal,
                },
            );
        }

        let ws = slot.get(&ordinal).unwrap();
        Ok((*ws.buffer.device_ptr() as *mut std::ffi::c_void, ws.size))
    }

    /// Returns the dedicated CUTLASS workspace for the current FP8/NVFP4 execution domain.
    ///
    /// Even when `flashinfer` is enabled, CUTLASS does not alias the FlashInfer
    /// float buffer because that shared path is not stable for the FP8 CUTLASS GEMM kernels.
    /// Domain splitting keeps decode-graph and MTP-verify-graph captured addresses stable.
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    pub fn get_cutlass_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        match crate::graph_workspace::graph_workspace_domain() {
            crate::graph_workspace::GraphWorkspaceDomain::Eager => CUTLASS_WORKSPACE_EAGER
                .with(|cell| get_or_init_cutlass_workspace_slot(cell, dev, required_size)),
            crate::graph_workspace::GraphWorkspaceDomain::DecodeGraph => {
                CUTLASS_WORKSPACE_DECODE_GRAPH
                    .with(|cell| get_or_init_cutlass_workspace_slot(cell, dev, required_size))
            }
            crate::graph_workspace::GraphWorkspaceDomain::MtpGraph => CUTLASS_WORKSPACE_MTP_GRAPH
                .with(|cell| get_or_init_cutlass_workspace_slot(cell, dev, required_size)),
        }
    }

    /// Alias for CUTLASS workspace in MoE context.
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    pub fn get_moe_cutlass_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        get_cutlass_workspace(dev, required_size)
    }

    /// Grow-only scratch for dense NVFP4 hardware W4A4 (quantize + CUTLASS GEMM).
    ///
    /// Per-call `Tensor::zeros` for act_packed/scales/alpha is not CUDA-graph safe:
    /// those addresses are baked into the captured graph and must come from a
    /// stable, domain-scoped pool (same pattern as FP8 grouped-GEMM scratch).
    #[cfg(feature = "cutlass")]
    pub struct Nvfp4HwScratchPool {
        pub act_packed: CudaSlice<u8>,
        pub act_packed_bytes: usize,
        pub act_scales: CudaSlice<u8>,
        pub act_scales_bytes: usize,
        pub act_scales_swizzled: CudaSlice<u8>,
        pub act_scales_swizzled_bytes: usize,
        pub wscale_swizzled: CudaSlice<u8>,
        pub wscale_swizzled_bytes: usize,
        pub alpha: CudaSlice<u8>,
        pub sf_scale: CudaSlice<u8>,
        pub online_scale: CudaSlice<u8>,
        pub device_ordinal: usize,
    }

    #[cfg(feature = "cutlass")]
    thread_local! {
        static NVFP4_HW_SCRATCH_EAGER: std::cell::RefCell<HashMap<usize, Nvfp4HwScratchPool>> =
            std::cell::RefCell::new(HashMap::new());
        static NVFP4_HW_SCRATCH_DECODE_GRAPH: std::cell::RefCell<HashMap<usize, Nvfp4HwScratchPool>> =
            std::cell::RefCell::new(HashMap::new());
        static NVFP4_HW_SCRATCH_MTP_GRAPH: std::cell::RefCell<HashMap<usize, Nvfp4HwScratchPool>> =
            std::cell::RefCell::new(HashMap::new());
    }

    #[cfg(feature = "cutlass")]
    pub struct Nvfp4HwScratchPtrs {
        pub act_packed: *mut std::ffi::c_void,
        pub act_scales: *mut std::ffi::c_void,
        pub act_scales_swizzled: *mut std::ffi::c_void,
        pub wscale_swizzled: *mut std::ffi::c_void,
        pub alpha: *mut f32,
        pub sf_scale: *mut f32,
        pub online_scale: *mut f32,
    }

    #[cfg(feature = "cutlass")]
    fn get_nvfp4_hw_scratch_from_pool(
        cell: &std::cell::RefCell<HashMap<usize, Nvfp4HwScratchPool>>,
        dev: &candle_core::cuda_backend::CudaDevice,
        act_packed_bytes: usize,
        act_scales_bytes: usize,
        act_scales_swizzled_bytes: usize,
        wscale_swizzled_bytes: usize,
    ) -> Result<Nvfp4HwScratchPtrs> {
        let mut slot = cell.borrow_mut();
        let ordinal = dev.ordinal();
        let needs_realloc = match slot.get(&ordinal) {
            None => true,
            Some(p) => {
                p.act_packed_bytes < act_packed_bytes
                    || p.act_scales_bytes < act_scales_bytes
                    || p.act_scales_swizzled_bytes < act_scales_swizzled_bytes
                    || p.wscale_swizzled_bytes < wscale_swizzled_bytes
            }
        };

        if needs_realloc {
            let (ap, as_, asw, ws) = if let Some(prev) = slot.get(&ordinal) {
                (
                    act_packed_bytes.max(prev.act_packed_bytes),
                    act_scales_bytes.max(prev.act_scales_bytes),
                    act_scales_swizzled_bytes.max(prev.act_scales_swizzled_bytes),
                    wscale_swizzled_bytes.max(prev.wscale_swizzled_bytes),
                )
            } else {
                (
                    act_packed_bytes,
                    act_scales_bytes,
                    act_scales_swizzled_bytes,
                    wscale_swizzled_bytes,
                )
            };
            let pool = Nvfp4HwScratchPool {
                act_packed: unsafe { dev.alloc::<u8>(ap.max(1)) }.w()?,
                act_packed_bytes: ap,
                act_scales: unsafe { dev.alloc::<u8>(as_.max(1)) }.w()?,
                act_scales_bytes: as_,
                act_scales_swizzled: unsafe { dev.alloc::<u8>(asw.max(1)) }.w()?,
                act_scales_swizzled_bytes: asw,
                wscale_swizzled: unsafe { dev.alloc::<u8>(ws.max(1)) }.w()?,
                wscale_swizzled_bytes: ws,
                alpha: unsafe { dev.alloc::<u8>(4) }.w()?,
                sf_scale: unsafe { dev.alloc::<u8>(4) }.w()?,
                online_scale: unsafe { dev.alloc::<u8>(8) }.w()?,
                device_ordinal: ordinal,
            };
            slot.insert(ordinal, pool);
        }

        let pool = slot.get(&ordinal).unwrap();
        Ok(Nvfp4HwScratchPtrs {
            act_packed: *pool.act_packed.device_ptr() as *mut std::ffi::c_void,
            act_scales: *pool.act_scales.device_ptr() as *mut std::ffi::c_void,
            act_scales_swizzled: *pool.act_scales_swizzled.device_ptr() as *mut std::ffi::c_void,
            wscale_swizzled: *pool.wscale_swizzled.device_ptr() as *mut std::ffi::c_void,
            alpha: *pool.alpha.device_ptr() as *mut f32,
            sf_scale: *pool.sf_scale.device_ptr() as *mut f32,
            online_scale: *pool.online_scale.device_ptr() as *mut f32,
        })
    }

    #[cfg(feature = "cutlass")]
    pub fn get_nvfp4_hw_scratch(
        dev: &candle_core::cuda_backend::CudaDevice,
        act_packed_bytes: usize,
        act_scales_bytes: usize,
        act_scales_swizzled_bytes: usize,
        wscale_swizzled_bytes: usize,
    ) -> Result<Nvfp4HwScratchPtrs> {
        match crate::graph_workspace::graph_workspace_domain() {
            crate::graph_workspace::GraphWorkspaceDomain::Eager => {
                NVFP4_HW_SCRATCH_EAGER.with(|cell| {
                    get_nvfp4_hw_scratch_from_pool(
                        cell,
                        dev,
                        act_packed_bytes,
                        act_scales_bytes,
                        act_scales_swizzled_bytes,
                        wscale_swizzled_bytes,
                    )
                })
            }
            crate::graph_workspace::GraphWorkspaceDomain::DecodeGraph => {
                NVFP4_HW_SCRATCH_DECODE_GRAPH.with(|cell| {
                    get_nvfp4_hw_scratch_from_pool(
                        cell,
                        dev,
                        act_packed_bytes,
                        act_scales_bytes,
                        act_scales_swizzled_bytes,
                        wscale_swizzled_bytes,
                    )
                })
            }
            crate::graph_workspace::GraphWorkspaceDomain::MtpGraph => NVFP4_HW_SCRATCH_MTP_GRAPH
                .with(|cell| {
                    get_nvfp4_hw_scratch_from_pool(
                        cell,
                        dev,
                        act_packed_bytes,
                        act_scales_bytes,
                        act_scales_swizzled_bytes,
                        wscale_swizzled_bytes,
                    )
                }),
        }
    }

    /// Thread-local pool for MoE NVFP4 hardware activation buffers.
    ///
    /// The Blackwell CUTLASS path allocates several large temporary tensors
    /// (gathered, act_packed, act_scales, rep_out, output) that scale with
    /// `size_m = num_tokens × topk`. By pooling these as grow-only raw
    /// `CudaSlice` buffers, we avoid per-call `cudaMalloc`/`cudaFree` overhead
    /// and reduce peak memory from allocation-deallocation hysteresis.
    #[cfg(feature = "cutlass")]
    pub struct MoeActivationPool {
        pub gathered: CudaSlice<u8>,
        pub gathered_bytes: usize,
        pub act_packed: CudaSlice<u8>,
        pub act_packed_bytes: usize,
        pub act_scales: CudaSlice<u8>,
        pub act_scales_bytes: usize,
        pub rep_out: CudaSlice<u8>,
        pub rep_out_bytes: usize,
        pub metadata: CudaSlice<u8>,
        pub metadata_bytes: usize,
        pub device_ordinal: usize,
    }

    #[cfg(feature = "cutlass")]
    thread_local! {
        pub static MOE_ACTIVATION_POOL: std::cell::RefCell<HashMap<usize, MoeActivationPool>> =
            std::cell::RefCell::new(HashMap::new());
    }

    #[cfg(feature = "cutlass")]
    pub struct MoePoolBuffers {
        pub gathered_ptr: u64,
        pub act_packed_ptr: u64,
        pub act_scales_ptr: u64,
        pub rep_out_ptr: u64,
        pub metadata_ptr: u64,
    }

    #[cfg(feature = "cutlass")]
    pub fn get_moe_activation_pool(
        dev: &candle_core::cuda_backend::CudaDevice,
        gathered_bytes: usize,
        act_packed_bytes: usize,
        act_scales_bytes: usize,
        rep_out_bytes: usize,
        _output_bytes: usize,
        metadata_bytes: usize,
    ) -> Result<MoePoolBuffers> {
        MOE_ACTIVATION_POOL.with(|cell| {
            let mut slot = cell.borrow_mut();
            let ordinal = dev.ordinal();

            let needs_realloc = match slot.get(&ordinal) {
                None => true,
                Some(p) => {
                    p.gathered_bytes < gathered_bytes
                        || p.act_packed_bytes < act_packed_bytes
                        || p.act_scales_bytes < act_scales_bytes
                        || p.rep_out_bytes < rep_out_bytes
                        || p.metadata_bytes < metadata_bytes
                }
            };

            if needs_realloc {
                // Grow-only: CudaSlice Drop uses free_async on the device stream,
                // and the subsequent alloc is also stream-ordered — no host sync.
                let (g_sz, ap_sz, as_sz, r_sz, m_sz) = if let Some(prev) = slot.get(&ordinal) {
                    (
                        gathered_bytes.max(prev.gathered_bytes),
                        act_packed_bytes.max(prev.act_packed_bytes),
                        act_scales_bytes.max(prev.act_scales_bytes),
                        rep_out_bytes.max(prev.rep_out_bytes),
                        metadata_bytes.max(prev.metadata_bytes),
                    )
                } else {
                    (
                        gathered_bytes,
                        act_packed_bytes,
                        act_scales_bytes,
                        rep_out_bytes,
                        metadata_bytes,
                    )
                };
                let gathered = unsafe { dev.alloc::<u8>(g_sz.max(1)) }.w()?;
                let act_packed = unsafe { dev.alloc::<u8>(ap_sz.max(1)) }.w()?;
                let act_scales = unsafe { dev.alloc::<u8>(as_sz.max(1)) }.w()?;
                let rep_out = unsafe { dev.alloc::<u8>(r_sz.max(1)) }.w()?;
                let metadata = unsafe { dev.alloc::<u8>(m_sz.max(1)) }.w()?;
                let pool = MoeActivationPool {
                    gathered,
                    gathered_bytes: g_sz,
                    act_packed,
                    act_packed_bytes: ap_sz,
                    act_scales,
                    act_scales_bytes: as_sz,
                    rep_out,
                    rep_out_bytes: r_sz,
                    metadata,
                    metadata_bytes: m_sz,
                    device_ordinal: ordinal,
                };
                slot.insert(ordinal, pool);
            }

            let pool = slot.get(&ordinal).unwrap();
            Ok(MoePoolBuffers {
                gathered_ptr: *pool.gathered.device_ptr(),
                act_packed_ptr: *pool.act_packed.device_ptr(),
                act_scales_ptr: *pool.act_scales.device_ptr(),
                rep_out_ptr: *pool.rep_out.device_ptr(),
                metadata_ptr: *pool.metadata.device_ptr(),
            })
        })
    }

    /// Returns the FlashInfer FP8 blockscale workspace.
    ///
    /// This is a separate workspace used by the TRT-LLM blockscale FP8 GEMM runners,
    /// which have different allocation requirements than the standard GEMM scratch.
    #[cfg(feature = "flashinfer")]
    pub fn get_or_init_flashinfer_fp8_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        let init = |cell: &std::cell::RefCell<HashMap<usize, FlashInferFp8Workspace>>| {
            let mut slot = cell.borrow_mut();
            let ordinal = dev.ordinal();

            let needs_init = match slot.get(&ordinal) {
                None => true,
                Some(existing) => existing.size < required_size,
            };

            if needs_init {
                let alloc_size = required_size.max(1);
                let buffer = unsafe { dev.alloc::<u8>(alloc_size) }.w()?;
                slot.insert(
                    ordinal,
                    FlashInferFp8Workspace {
                        buffer,
                        size: alloc_size,
                        device_ordinal: ordinal,
                    },
                );
            }

            let ws = slot.get(&ordinal).unwrap();
            Ok((*ws.buffer.device_ptr() as *mut std::ffi::c_void, ws.size))
        };

        match crate::graph_workspace::graph_workspace_domain() {
            crate::graph_workspace::GraphWorkspaceDomain::Eager => {
                FLASHINFER_FP8_WORKSPACE_EAGER.with(init)
            }
            crate::graph_workspace::GraphWorkspaceDomain::DecodeGraph => {
                FLASHINFER_FP8_WORKSPACE_DECODE_GRAPH.with(init)
            }
            crate::graph_workspace::GraphWorkspaceDomain::MtpGraph => {
                FLASHINFER_FP8_WORKSPACE_MTP_GRAPH.with(init)
            }
        }
    }

    /// Grow-only workspace for native (non-FlashInfer) MLA split-K decode temps.
    ///
    /// Layout (all f32):
    /// ```text
    /// [tmp_out | tmp_max | tmp_sum]
    /// tmp_out: [num_seqs, num_heads, max_partitions, kv_lora_rank]
    /// tmp_max / tmp_sum: [num_seqs, num_heads, max_partitions]
    /// ```
    ///
    /// Lazily allocated on first `mla_paged_decode` use (typically graph warmup)
    /// and grown when a larger request arrives. Separate eager / graph slots keep
    /// CUDA-graph-captured addresses stable.
    pub struct MlaDecodeWorkspace {
        pub buffer: CudaSlice<u8>,
        pub size: usize,
        pub device_ordinal: usize,
    }

    thread_local! {
        static MLA_DECODE_WORKSPACE_EAGER: std::cell::RefCell<HashMap<usize, MlaDecodeWorkspace>> =
            std::cell::RefCell::new(HashMap::new());
        static MLA_DECODE_WORKSPACE_DECODE_GRAPH: std::cell::RefCell<HashMap<usize, MlaDecodeWorkspace>> =
            std::cell::RefCell::new(HashMap::new());
        static MLA_DECODE_WORKSPACE_MTP_GRAPH: std::cell::RefCell<HashMap<usize, MlaDecodeWorkspace>> =
            std::cell::RefCell::new(HashMap::new());
    }

    /// Pointers into the cached MLA decode workspace.
    pub struct MlaDecodeWorkspacePtrs {
        pub tmp_out: *mut std::ffi::c_void,
        pub tmp_max: *mut std::ffi::c_void,
        pub tmp_sum: *mut std::ffi::c_void,
    }

    /// Bytes required for native MLA split-K decode temps.
    pub fn mla_decode_workspace_bytes(
        num_seqs: usize,
        num_heads: usize,
        max_partitions: usize,
        kv_lora_rank: usize,
    ) -> usize {
        let out_elems = num_seqs
            .saturating_mul(num_heads)
            .saturating_mul(max_partitions)
            .saturating_mul(kv_lora_rank);
        let meta_elems = num_seqs
            .saturating_mul(num_heads)
            .saturating_mul(max_partitions);
        // tmp_out + tmp_max + tmp_sum, all f32
        (out_elems.saturating_add(meta_elems.saturating_mul(2))).saturating_mul(4)
    }

    /// Get or grow the native MLA decode workspace; returns (tmp_out, tmp_max, tmp_sum).
    pub fn get_or_init_mla_decode_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        num_seqs: usize,
        num_heads: usize,
        max_partitions: usize,
        kv_lora_rank: usize,
    ) -> Result<MlaDecodeWorkspacePtrs> {
        let out_elems = num_seqs
            .saturating_mul(num_heads)
            .saturating_mul(max_partitions)
            .saturating_mul(kv_lora_rank);
        let meta_elems = num_seqs
            .saturating_mul(num_heads)
            .saturating_mul(max_partitions);
        let out_bytes = out_elems.saturating_mul(4);
        let meta_bytes = meta_elems.saturating_mul(4);
        let required = out_bytes
            .saturating_add(meta_bytes)
            .saturating_add(meta_bytes)
            .max(1);

        let init = |cell: &std::cell::RefCell<HashMap<usize, MlaDecodeWorkspace>>| {
            let mut slot = cell.borrow_mut();
            let ordinal = dev.ordinal();
            let needs_init = match slot.get(&ordinal) {
                None => true,
                Some(existing) => existing.size < required,
            };

            if needs_init {
                let buffer = unsafe { dev.alloc::<u8>(required) }.w()?;
                slot.insert(
                    ordinal,
                    MlaDecodeWorkspace {
                        buffer,
                        size: required,
                        device_ordinal: ordinal,
                    },
                );
            }

            let ws = slot.get(&ordinal).unwrap();
            let base = *ws.buffer.device_ptr() as *mut u8;
            Ok(MlaDecodeWorkspacePtrs {
                tmp_out: base as *mut std::ffi::c_void,
                tmp_max: unsafe { base.add(out_bytes) } as *mut std::ffi::c_void,
                tmp_sum: unsafe { base.add(out_bytes.saturating_add(meta_bytes)) }
                    as *mut std::ffi::c_void,
            })
        };

        match crate::graph_workspace::graph_workspace_domain() {
            crate::graph_workspace::GraphWorkspaceDomain::Eager => {
                MLA_DECODE_WORKSPACE_EAGER.with(init)
            }
            crate::graph_workspace::GraphWorkspaceDomain::DecodeGraph => {
                MLA_DECODE_WORKSPACE_DECODE_GRAPH.with(init)
            }
            crate::graph_workspace::GraphWorkspaceDomain::MtpGraph => {
                MLA_DECODE_WORKSPACE_MTP_GRAPH.with(init)
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_regions_do_not_overlap_and_fit() {
        let regions = workspace_regions();
        assert_eq!(regions.plan_float.offset, 0);
        assert_eq!(regions.plan_int.offset, 0);
        assert_eq!(regions.gemm_scratch_float.size, GEMM_SCRATCH_FLOAT_SIZE);
        assert!(regions.plan_float.size <= WORKSPACE_FLOAT_SIZE);
        assert!(regions.plan_int.size <= WORKSPACE_INT_SIZE);
        assert_eq!(
            regions.plan_float.offset + regions.plan_float.size,
            regions.gemm_scratch_float.offset
        );
        assert!(
            regions.gemm_scratch_float.offset + regions.gemm_scratch_float.size
                <= WORKSPACE_FLOAT_SIZE
        );
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(WORKSPACE_INT_SIZE, 128 * 1024 * 1024);
        assert_eq!(FLASHINFER_PLAN_FLOAT_SIZE, 256 * 1024 * 1024);
        assert_eq!(GEMM_SCRATCH_FLOAT_OFFSET, FLASHINFER_PLAN_FLOAT_SIZE);
        assert_eq!(
            WORKSPACE_FLOAT_SIZE,
            FLASHINFER_PLAN_FLOAT_SIZE + GEMM_SCRATCH_FLOAT_SIZE
        );
        // The GEMM scratch region only exists for the CUTLASS kernels.
        #[cfg(feature = "cutlass")]
        assert_eq!(GEMM_SCRATCH_FLOAT_SIZE, 256 * 1024 * 1024);
        #[cfg(not(feature = "cutlass"))]
        assert_eq!(GEMM_SCRATCH_FLOAT_SIZE, 0);
    }
}
