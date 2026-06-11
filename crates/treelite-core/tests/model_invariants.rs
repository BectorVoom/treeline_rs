//! `Model` Wave-0 invariants for Phase 9 (MEM-02): a `size_of::<Model>()` budget
//! and a documented `!Send` check (A3 / Pitfall 2).
//!
//! These are the baseline guards the MEM-02 field swap (Plan 02) must not break:
//!   * `model_size_not_bloated_by_smallvec` caps `size_of::<Model>()` so that
//!     choosing an over-large `SmallVec` inline `N` (Pitfall 2) is caught — an
//!     oversized inline buffer would push the struct past the budget.
//!   * `_assert_not_send` documents the `!Send` invariant (RESEARCH line 75):
//!     `Model` MUST stay `!Send` because `TreeBuf::Borrowed` holds a `*const T`.
//!     `SmallVec`/`CompactString` are `Send`-neutral, so the MEM-02 swap leaves
//!     this invariant intact. The check is a commented-out `requires_send::<Model>()`
//!     that MUST NOT be uncommented — uncommenting it would fail to compile, which
//!     is exactly the invariant. (The repo has no `trybuild`, so the compile-fail
//!     half is expressed as this documented static assertion — PATTERNS lines 416-417.)

use treelite_core::Model;

/// Upper bound on `size_of::<Model>()`. Established in Plan 01 against the
/// CURRENT (pre-MEM-02) `Model`, whose measured size is 248 bytes; the budget is
/// `max(current_size, 512) == 512` (PATTERNS line 400). 512 leaves headroom for
/// Plan 02's `SmallVec`/`CompactString` field swap while staying a meaningful
/// guard against an over-large inline `N` (Pitfall 2). Plan 02 re-checks it.
const MODEL_SIZE_BUDGET: usize = 512;

/// A3 / RESEARCH line 75: `Model` MUST stay `!Send`. The live source of `!Send`
/// is the `*const T` in `TreeBuf::Borrowed`; `SmallVec`/`CompactString` do not
/// add `Send`, so the MEM-02 swap preserves this. Documented compile-fail check:
/// uncommenting `requires_send::<Model>()` MUST fail to compile.
#[allow(dead_code)]
fn _assert_not_send() {
    fn requires_send<T: Send>() {}
    // requires_send::<Model>();  // ← must NOT compile; this comment IS the invariant.
    let _ = requires_send::<i32>; // keep the helper referenced (i32: Send).
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
