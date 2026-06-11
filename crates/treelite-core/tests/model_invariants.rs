//! `Model` Wave-0 invariants: a `size_of::<Model>()` budget (Phase 9, MEM-02) and
//! the cross-thread shareability contract (Phase 10, PAR-03).
//!
//! These are the baseline guards downstream work must not break:
//!   * `model_size_not_bloated_by_smallvec` caps `size_of::<Model>()` so that
//!     choosing an over-large `SmallVec` inline `N` (Pitfall 2) is caught — an
//!     oversized inline buffer would push the struct past the budget.
//!   * `model_is_sync_for_readonly_predict` pins the Phase-10 `Sync` contract:
//!     `Model` is now soundly SHAREABLE across threads for read-only predict via
//!     the documented `unsafe impl Sync for Model` (model.rs), mirroring upstream
//!     OpenMP sharing `Model const&`. This SUPERSEDES the prior Phase-9 `!Send`
//!     invariant (the old not-Send compile check): the type intentionally became
//!     `Sync` so rayon can share `&Model` across workers. `requires_sync::<Model>()`
//!     compiles iff `Model: Sync`, and `&Model: Send` follows automatically (the
//!     property rayon relies on). Only `Sync` is asserted — NOT `Send` (A4); the
//!     model is never moved to another thread.

use treelite_core::Model;

/// Upper bound on `size_of::<Model>()`. Established in Plan 01 against the
/// CURRENT (pre-MEM-02) `Model`, whose measured size is 248 bytes; the budget is
/// `max(current_size, 512) == 512` (PATTERNS line 400). 512 leaves headroom for
/// Plan 02's `SmallVec`/`CompactString` field swap while staying a meaningful
/// guard against an over-large inline `N` (Pitfall 2). Plan 02 re-checks it.
const MODEL_SIZE_BUDGET: usize = 512;

/// PAR-03: `Model` is soundly SHAREABLE across threads for read-only predict
/// (documented `unsafe impl Sync for Model`, mirroring upstream OpenMP). This
/// SUPERSEDES the prior not-Send invariant — the type intentionally became
/// `Sync` so rayon can share `&Model` across workers.
#[test]
fn model_is_sync_for_readonly_predict() {
    fn requires_sync<T: Sync>() {}
    requires_sync::<Model>(); // compiles iff Model: Sync — the new contract.
    fn requires_send<T: Send>() {}
    requires_send::<&Model>(); // &Model: Send follows from Model: Sync (what rayon needs).
}

/// Pitfall 2: an over-large `SmallVec` inline `N` would bloat `Model` and hurt
/// cache locality (the opposite of the MEM-02 goal). Assert the struct stays
/// under a fixed byte budget so any future inline-`N` regression is caught.
#[test]
fn model_size_not_bloated_by_smallvec() {
    let size = std::mem::size_of::<Model>();
    assert!(
        size <= MODEL_SIZE_BUDGET,
        "size_of::<Model>() = {} exceeds the {}-byte budget (Pitfall 2: \
         an over-large SmallVec inline N bloats Model)",
        size,
        MODEL_SIZE_BUDGET,
    );
}
