//! Per-backend `ComputeClient<R>` construction with a typed device-absence skip
//! (D-05, RESEARCH Pattern 2).
//!
//! The CPU path constructs a client unconditionally (mirrors `lib.rs`'s
//! `CpuRuntime::client(&Default::default())`). The GPU paths
//! (`rocm`/`cuda`/`wgpu`, each behind its cargo feature) attempt the SAME
//! uniform `R::client(&<R::Device>::default())` construction but map a
//! construction failure on a missing device to a typed
//! [`CubeclError::DeviceUnavailable`] — the caller (harness/report) branches on
//! it as a SKIP, NEVER a silent CPU fallback.
//!
//! ## The A3 unknown (RESEARCH Open Question 1, Pitfall 1 — HIGH risk)
//!
//! cubecl 0.10's `Runtime::client(device) -> ComputeClient<R>` returns the
//! client by value; there is no `Result`. Whether a missing HIP/CUDA device
//! surfaces as a catchable Rust panic (trappable by [`std::panic::catch_unwind`])
//! or as a hard FFI `abort()` from the C driver (below `catch_unwind`'s reach)
//! is the single most important unknown gating every later backend
//! registration. As the FIRST attempt this module wraps construction in
//! `catch_unwind`; the `tests/device_absent.rs` spike RECORDS which behavior
//! actually occurs on this NVIDIA-less box. If the driver hard-aborts, Plan 03
//! MUST pre-probe device availability (a cubecl enumeration API confirmed via
//! `cargo doc`) and never call `client()` for an absent backend.

use crate::CubeclError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;

/// Generic per-backend client constructor: builds a [`ComputeClient<R>`] from
/// the runtime's default device, mapping a construction failure (e.g. no device
/// present) to a typed [`CubeclError::DeviceUnavailable`] rather than a panic
/// crossing the call boundary.
///
/// `backend` is the static feature tag carried into the error (`"rocm"` /
/// `"cuda"` / `"wgpu"` / `"cpu"`) so the caller knows which backend skipped.
///
/// As the FIRST attempt at device-absence handling this wraps construction in
/// [`std::panic::catch_unwind`] (D-05). A Rust panic on a missing device is
/// trapped and converted to the typed skip; a hard FFI `abort()` from the C
/// driver is NOT trappable here — that case is what the Wave-0 spike confirms
/// (see module docs / `tests/device_absent.rs`).
pub fn client<R: Runtime>(backend: &'static str) -> Result<ComputeClient<R>, CubeclError>
where
    R::Device: Default,
{
    // `ComputeClient<R>` is not `UnwindSafe` in general; the construction only
    // touches a freshly-defaulted device, so asserting unwind-safety here is
    // sound (no shared mutable state can be observed in a torn state).
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        R::client(&<R::Device as Default>::default())
    }))
    .map_err(|payload| classify_client_panic(backend, payload))
}

/// Substrings that identify a *device-absent* construction panic (WR-01).
///
/// The A3 spike established that a missing HIP/CUDA/wgpu device surfaces as a
/// catchable Rust panic whose payload message reports the dlopen / driver-init /
/// adapter-enumeration failure. We classify a trapped panic as the benign
/// [`CubeclError::DeviceUnavailable`] skip ONLY when its message matches one of
/// these known device-absence signatures; ANY other panic (a real driver fault,
/// OOM, or an internal cubecl assertion) is preserved as a genuine
/// [`CubeclError::ClientInit`] failure with its message carried in `detail`,
/// rather than silently swallowed as a skip.
///
/// Conservative-by-design: matching is case-insensitive and substring-based so a
/// minor driver wording change still classifies a true absence as a skip, while
/// an unrelated fault — which will not contain any of these phrases — is
/// surfaced. If a future runtime reports absence with wording absent from this
/// list, it degrades to a (loud) `ClientInit` failure, never a false skip.
const DEVICE_ABSENT_SIGNATURES: &[&str] = &[
    // cubecl/HIP/CUDA dlopen failure when the driver shared object is missing.
    "unable to dynamically load",
    // generic "no device / no adapter present" phrasings across backends.
    "no device available",
    "no available device",
    "no compatible adapter",
    "no suitable adapter",
    "failed to find an appropriate adapter",
    "no devices found",
];

/// Classify a trapped client-construction panic payload (WR-01).
///
/// Extracts the panic message (`String` then `&str` payloads), lower-cases it,
/// and returns [`CubeclError::DeviceUnavailable`] only when it matches a known
/// device-absence signature; otherwise returns [`CubeclError::ClientInit`] with
/// the original message preserved in `detail` so a genuine init fault is never
/// discarded. A non-string payload (neither `String` nor `&str`) cannot be a
/// recognized absence signature, so it is conservatively treated as a real
/// `ClientInit` fault with a marker detail rather than a silent skip.
fn classify_client_panic(
    backend: &'static str,
    payload: Box<dyn std::any::Any + Send>,
) -> CubeclError {
    let msg = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied());
    match msg {
        Some(m) => {
            let lower = m.to_ascii_lowercase();
            if DEVICE_ABSENT_SIGNATURES
                .iter()
                .any(|sig| lower.contains(sig))
            {
                CubeclError::DeviceUnavailable { backend }
            } else {
                CubeclError::ClientInit {
                    backend,
                    detail: m.to_string(),
                }
            }
        }
        None => CubeclError::ClientInit {
            backend,
            detail: "<non-string panic payload>".to_string(),
        },
    }
}

/// ROCm (AMD HIP) client. Returns [`CubeclError::DeviceUnavailable`] when no
/// HIP device is present (the typed skip, D-05). Behind the `rocm` cargo
/// feature, which forwards to `cubecl/rocm` → `cubecl/hip`.
#[cfg(feature = "rocm")]
pub fn rocm_client() -> Result<ComputeClient<cubecl::hip::HipRuntime>, CubeclError> {
    client::<cubecl::hip::HipRuntime>("rocm")
}

/// CUDA client. Returns [`CubeclError::DeviceUnavailable`] when no CUDA device
/// is present (the typed skip, D-05). Behind the `cuda` cargo feature.
///
/// On an NVIDIA-less box this is the device-ABSENT case the A3 spike exercises:
/// the spike asserts this returns `Err(DeviceUnavailable)` rather than aborting
/// the process.
#[cfg(feature = "cuda")]
pub fn cuda_client() -> Result<ComputeClient<cubecl::cuda::CudaRuntime>, CubeclError> {
    client::<cubecl::cuda::CudaRuntime>("cuda")
}

/// wgpu client. Returns [`CubeclError::DeviceUnavailable`] when no compatible
/// adapter is present (the typed skip, D-05). Behind the `wgpu` cargo feature.
///
/// `WgpuRuntime::Device` is `WgpuDevice`, whose `Default` is
/// `WgpuDevice::DefaultDevice` (the "best available" adapter) — so the generic
/// `<R::Device>::default()` path selects the default adapter uniformly with the
/// other backends.
#[cfg(feature = "wgpu")]
pub fn wgpu_client() -> Result<ComputeClient<cubecl::wgpu::WgpuRuntime>, CubeclError> {
    client::<cubecl::wgpu::WgpuRuntime>("wgpu")
}
