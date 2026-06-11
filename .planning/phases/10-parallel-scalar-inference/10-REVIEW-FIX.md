---
phase: 10-parallel-scalar-inference
fixed_at: 2026-06-11T00:00:00Z
review_path: .planning/phases/10-parallel-scalar-inference/10-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-06-11T00:00:00Z
**Source review:** .planning/phases/10-parallel-scalar-inference/10-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (fix_scope = all: WR-01, WR-02, WR-03, IN-01, IN-02)
- Fixed: 5
- Skipped: 0

All five findings were addressed. Two received behavioral/soundness fixes
(WR-02 tightened a `Sync` bound; WR-03 added overflow guards on output
allocations), and three received the minimal-safe documentation fix that the
review and the per-finding guidance preferred (WR-01, IN-01, IN-02). No
behavior on valid inputs changed.

## Fixed Issues

### WR-02: `unsafe impl<T: Copy> Sync for TreeBuf<T>` over-broad

**Files modified:** `crates/treelite-core/src/tree_buf.rs`
**Commit:** 5b4331f
**Applied fix:** Narrowed the bound from `T: Copy` to `T: Copy + Sync`. `T: Copy`
alone permits interior-mutable `Copy` types such as `Cell<f32>` (`Copy` but
`!Sync`), which would make the concurrent `Owned`-path reads a data race.
Requiring `T: Sync` expresses the actual safety property. Verified every concrete
`T` in use (`f32`, `f64`, `i32`, `u32`, `u64`, `bool`, `Operator`,
`TreeNodeType`) satisfies `Sync` — `cargo build --workspace` compiles clean, so
this is the zero-impact tightening the reviewer described. The SAFETY comment was
updated to record the reasoning.

### WR-03: `predict_score_by_tree_preset` output allocation uses unchecked multiplication

**Files modified:** `crates/treelite-gtil/src/lib.rs`
**Commit:** 0c36c44
**Applied fix:** Replaced the unchecked `vec![O::zero(); num_row * num_tree * lvs]`
with a `checked_mul` chain that maps a wrapping product to a typed
`GtilError::InvalidInputShape` (`required = usize::MAX` flags a shape overflow,
not a feature mismatch) before the `Vec` is allocated. This prevents the 32-bit
`usize` buffer-overflow scenario the reviewer identified (a wrapped small
allocation paired with correctly-sized `par_chunks_mut` chunks → OOB writes).
The same cheap guard was applied to the two sibling allocations the finding
called out: `predict_leaf_preset` (`num_row * num_tree`) and `predict_preset`
(`num_row * cells_per_row`). Behavior on valid inputs is unchanged (the guard
only fires when the product would have wrapped). `cargo build -p treelite-gtil`
and the full gtil test suite pass.

### WR-01: `unsafe impl Send for SendModelRef` relies on an unenforced caller convention

**Files modified:** `crates/treelite-py/src/gtil.rs`
**Commit:** e862d35
**Applied fix:** Took the minimal-safe option from the finding (and per-finding
guidance): strengthened the SAFETY doc comment on the `unsafe impl Send` to state
the caller contract explicitly — the wrapped `Model` must contain only
`TreeBuf::Owned` columns, or every `TreeBuf::Borrowed` backing must outlive the
entire detached predict closure (not merely the `&Model`). A runtime
no-`Borrowed`-columns assertion was NOT added because it would require introducing
a new public column-ownership inspection API on `Model`/`TreeBuf` (a larger,
riskier public-API change than the phase warrants); the type-system seal is
likewise deferred. The contract holds in practice today (all models are
loader-produced with `Owned` columns) and is now documented at the construction
site so future `from_borrowed`-built models are audited before being routed here.

### IN-01: Multiple scoped thread pools created per `predict_preset` call

**Files modified:** `crates/treelite-gtil/src/config.rs`
**Commit:** cc4309f
**Applied fix:** Took the v1 documentation fix (the invasive single-pool refactor
through the call chain was deliberately avoided to protect the hot path). Added a
"Performance note (IN-01)" to `Config`'s doc comment explaining that `nthread > 0`
builds a separate scoped `rayon::ThreadPool` per parallel pass (traversal +
optional RF-average + base-scores → 2–3 pools per call), incurring 2–3× the
thread-creation/teardown cost of the `nthread <= 0` global-pool path, and
advising `nthread <= 0` for small batches.

### IN-02: `Config` default `nthread=0` and Python default `nthread=-1` diverge

**Files modified:** `crates/treelite-gtil/src/config.rs`, `crates/treelite-py/src/gtil.rs`
**Commit:** 1d40748
**Applied fix:** Aligned via documentation rather than changing the default value.
The reviewer's suggested change (`Config::default()` `0 → -1`) was NOT made because
`crates/treelite-gtil/tests/config_and_shape.rs:56` explicitly asserts
`cfg.nthread == 0` (mirroring upstream `gtil.h:51`), so the value change would
break an existing test. Instead documented on the `nthread` field doc (Rust side)
and on the `predict_f32`/`predict_f64` docstrings (Python side) that the two
sentinels (`0` and `-1`) are behavior-identical: both satisfy `nthread <= 0` and
route to the global "use all cores" pool; only the displayed default differs (`0`
matches upstream, `-1` matches the NumPy/scikit-learn convention). Behavior is
identical.

## Verification

All fixes verified per the 3-tier strategy (re-read + per-crate syntax/build
checks), then the full mandatory gate was run on the isolated worktree
(`/tmp/sv-10-reviewfix-*`):

- `cargo build --workspace` → **clean** (finished, no errors).
- `cargo test --workspace` → all `treelite-core`, `treelite-gtil`, `treelite-cubecl`,
  `treelite-builder` tests **pass**, including `config_and_shape` (the
  `nthread == 0` assertion preserved by the IN-02 doc-only approach),
  `parallel_nthread`, and the **1e-5 golden `gtil_matrix` (ok)** plus
  `gtil_matrix_cubecl (ok)`.
- `uv run pytest crates/treelite-py/tests/python/test_nthread.py -q` →
  **2 passed** (run on the main tree because `pyproject.toml`/`uv.lock`/venv are
  untracked and absent from the worktree; the Python-side fixes are doc-only and
  do not change the compiled extension's behavior, so the main-tree extension
  validates the same surface).

### Pre-existing failures (NOT caused by these fixes)

3 tests fail in the worktree, all from a single root cause unrelated to this
phase's fixes:

- `treelite-harness::lightgbm::lightgbm_numerical`
- `treelite-lightgbm::parse::tests::parses_vendored_header_and_counts`
- `treelite-lightgbm::tests::loads_vendored_numerical_model_to_f64_no_builder_errors`

All three fail with `No such file or directory` reading
`treelite-mainline/tests/examples/deep_lightgbm/model.txt`. That fixture exists
on the main working tree but is **untracked by git**, so it is absent from the
worktree checkout (the documented worktree-isolation artifact with dirty
untracked vendored trees). None of the five fixes touch LightGBM parsing or
fixtures — they modify only `tree_buf.rs`, `gtil/src/lib.rs`, `gtil/src/config.rs`,
and `py/src/gtil.rs`. The 1e-5 fidelity golden (`gtil_matrix`) passes.

---

_Fixed: 2026-06-11T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
