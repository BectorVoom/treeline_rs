//! Wave-0 device-absence spike (RESEARCH Open Question A3 / Pitfall 1 — HIGH
//! risk) + the typed-skip regression for [`CubeclError::DeviceUnavailable`].
//!
//! ## What this answers (A3)
//!
//! cubecl 0.10's `Runtime::client(device) -> ComputeClient<R>` returns the
//! client by value (no `Result`). The single most important unknown gating
//! every later Phase-7 backend registration is: when NO device is present, does
//! `client()` surface a **catchable Rust panic** (trappable by
//! [`std::panic::catch_unwind`], so `device::cuda_client()` can return
//! `Err(DeviceUnavailable)`) — or a **hard FFI `abort()`** from the C driver
//! that tears down the whole process below `catch_unwind`'s reach?
//!
//! This box has an AMD/ROCm GPU and NO NVIDIA device, so **CUDA is the
//! device-ABSENT backend** here. Built with `--features cuda`, this test calls
//! `cuda_client()` and asserts it returns `Err(DeviceUnavailable)`. The
//! outcome is recorded in `07-01-SUMMARY.md`:
//!   - If this test PASSES → `catch_unwind` traps the failure; the typed skip is
//!     reachable as-written and Plan 03 may construct clients directly.
//!   - If this test ABORTS the process (no Rust panic, the C driver calls
//!     `abort()`) → the SUMMARY MUST flag that Plan 03 needs a PRE-construction
//!     device probe (a cubecl enumeration API confirmed via `cargo doc`) so the
//!     harness never calls `client()` for an absent backend.
//!
//! ## Default (no-feature) build
//!
//! With no GPU feature enabled the cfg-gated `cuda_client` symbol does not
//! compile, so the test body is a no-op and `cargo test --workspace` stays
//! green without any GPU system libs present.

/// A3 spike: a compiled-in-but-device-absent CUDA backend must yield the typed
/// [`CubeclError::DeviceUnavailable`] skip — NOT a process abort.
///
/// Run with: `cargo test -p treelite-cubecl --features cuda --test device_absent`
#[cfg(feature = "cuda")]
#[test]
fn cuda_absent_device_is_typed_skip_not_abort() {
    use treelite_cubecl::CubeclError;
    use treelite_cubecl::device;

    // If this line aborts the process (hard FFI abort below catch_unwind), the
    // test never reports — that observation is itself the A3 finding the SUMMARY
    // records (Plan 03 then needs a pre-construction device probe).
    //
    // NOTE: `ComputeClient<R>` does not implement `Debug`, so we cannot
    // `{:?}`-format the `Ok` arm. We branch explicitly instead and only render
    // the typed-error arm.
    match device::cuda_client() {
        Err(CubeclError::DeviceUnavailable { backend: "cuda" }) => {
            // A3 = catchable: `catch_unwind` trapped the missing-device failure
            // and the typed skip is reachable as written.
        }
        Err(other) => panic!(
            "expected Err(DeviceUnavailable {{ backend: \"cuda\" }}) on an NVIDIA-less box, \
             got a different typed error: {other:?}"
        ),
        Ok(_client) => panic!(
            "expected Err(DeviceUnavailable {{ backend: \"cuda\" }}) on an NVIDIA-less box, \
             but cuda_client() returned Ok — a CUDA device appears to be present, \
             so this box is NOT the device-absent A3 case"
        ),
    }
}

/// Default build: no GPU feature, so there is no device to probe. Keeps
/// `cargo test --workspace` green without GPU libs while documenting that the
/// spike body above is feature-gated.
#[cfg(not(feature = "cuda"))]
#[test]
fn device_absent_spike_is_feature_gated() {
    // No-op: the A3 spike runs only under `--features cuda`. This placeholder
    // exists so the test target always has at least one test and the default
    // workspace run stays green without any GPU system libs.
}
