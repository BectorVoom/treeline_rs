//! Post-processing transforms (margin → probability).
//!
//! Ports `treelite-mainline/src/gtil/postprocessor.cc:19-37` VERBATIM. Phase 1
//! supports only the two postprocessors exercised by the `binary:logistic`
//! fixture: `identity` and `sigmoid`. The remaining upstream postprocessors
//! (`signed_square`, `hinge`, `exponential`, `softmax`, ...) are deferred.
//!
//! ## The 1e-5 cast-ordering contract
//!
//! Upstream `sigmoid` is instantiated with `InputT == float`, and
//! `model.sigmoid_alpha` is a `float`. Therefore both the multiplication and
//! `std::exp` run in **f32** — there is NO promotion to double precision.
//! Doing the math in double precision would shift the final ULPs past the
//! 1e-5 equivalence bound (RESEARCH §Pitfall 3/4). The Rust port keeps every
//! operation in `f32`.

/// `identity` postprocessor (`postprocessor.cc:19-20`): returns the margin
/// unchanged. The `_alpha` argument mirrors the upstream signature shape (the
/// `num_class` slot) and is unused.
pub fn identity(_alpha: f32, v: f32) -> f32 {
    v
}

/// `sigmoid` postprocessor (`postprocessor.cc:33-37`), ported verbatim:
///
/// ```cpp
/// InputT const val = *elem;
/// *elem = InputT(1) / (InputT(1) + std::exp(-model.sigmoid_alpha * val));
/// ```
///
/// `sigmoid_alpha` is `f32` and `exp` runs on the `f32` value — no
/// double-precision promotion anywhere (the cast-ordering contract).
pub fn sigmoid(sigmoid_alpha: f32, v: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-sigmoid_alpha * v).exp())
}
