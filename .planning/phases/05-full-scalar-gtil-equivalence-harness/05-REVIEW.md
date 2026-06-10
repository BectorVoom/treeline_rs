---
phase: 05-full-scalar-gtil-equivalence-harness
reviewed: 2026-06-10T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - crates/treelite-gtil/Cargo.toml
  - crates/treelite-gtil/src/accessor.rs
  - crates/treelite-gtil/src/config.rs
  - crates/treelite-gtil/src/error.rs
  - crates/treelite-gtil/src/lib.rs
  - crates/treelite-gtil/src/postprocessor.rs
  - crates/treelite-gtil/src/shape.rs
  - crates/treelite-gtil/tests/config_and_shape.rs
  - crates/treelite-gtil/tests/generic_input.rs
  - crates/treelite-gtil/tests/predict.rs
  - crates/treelite-gtil/tests/predict_kinds.rs
  - crates/treelite-gtil/tests/sparse.rs
  - crates/treelite-harness/src/lib.rs
  - crates/treelite-harness/src/manifest.rs
  - crates/treelite-harness/tests/gtil_matrix.rs
  - fixtures/capture_gtil_matrix.py
  - fixtures/capture_gtil_models.py
findings:
  critical: 1
  warning: 6
  info: 4
  total: 11
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-06-10
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

This phase ports the full scalar GTIL inference engine and an equivalence
harness whose core contract is "predictions match upstream Treelite within
1e-5". The bounds-checking discipline is genuinely strong — every untrusted
loader/model field (`split_index`, child ids, `target_id`/`class_id`,
`row_ptr`, `col_ind`, negative `num_feature`, leaf-vector lengths) is routed to
a typed `GtilError` rather than an OOB panic, and the tests cover those paths
well. The cast-order reasoning for the dense numeric path, `leaf_as_out`,
`add_base_score`, averaging divisor, and `softmax` matches upstream.

The headline defect is a **numeric-fidelity deviation in the postprocessor
boundary for f64 input**: every postprocessor except `softmax` is run through
an f32 intermediate regardless of the output element `O`, while upstream runs
`sigmoid`/`exponential`/`multiclass_ova`/`logarithm_one_plus_exp`/`signed_square`
in `double` when `InputT == double`. This contradicts the code's own claim that
the O-generic postprocessor was "wired in Plan 05-03" and is only masked today
because the committed f64 goldens stay inside the 1e-5 band. A second
correctness concern is the matrix harness re-deriving the CSR from
NaN-presence rather than consuming the frozen sparse golden's own input, which
weakens the sparse-path guarantee. Several smaller shape/error inconsistencies
and dead code round out the list.

## Critical Issues

### CR-01: Postprocessor forced through f32 intermediate breaks the f64-input cast-order contract

**File:** `crates/treelite-gtil/src/lib.rs:1122-1141` (`apply_postprocessor`)
**Issue:**
`apply_postprocessor` narrows every output cell to `f32`, runs
`apply_postprocessor_f32`, and widens back to `O`:

```rust
let mut buf: Vec<f32> = output.iter().map(|v| v.out_to_f32()).collect();
apply_postprocessor_f32(model, shape, &mut buf, num_row)?;
for (dst, src) in output.iter_mut().zip(buf) {
    *dst = O::out_from_f32(src);
}
```

Upstream `ApplyPostProcessor<InputT>` (`predict.cc:307-323`) is instantiated
with `InputT == double` for f64 input, and the postprocessors are
`InputT`-templated (`postprocessor.cc:33-52, 77-82`): `sigmoid`, `exponential`,
`exponential_standard_ratio`, `logarithm_one_plus_exp`, `signed_square`, and
`multiclass_ova` run their `std::exp`/`std::exp2`/`std::log1p`/multiply in
**double** when `InputT == double`. The Rust port forces all of them through
`f32`, so for any f64-input model whose postprocessor is not `identity`/
`softmax`, the result is computed at f32 precision and then widened — the wrong
ULPs. (`softmax` is the one case that happens to match: upstream hardcodes
`float max_margin`/`float t` regardless of `InputT`, `postprocessor.cc:59-61`,
so the f32 intermediate is correct only there.)

This is the precise "postprocessor arithmetic precision" deviation the 1e-5
contract calls out. It is currently masked because the committed matrix
goldens (binary:logistic → sigmoid, multi:softprob → softmax) keep f64-input
sigmoid deviations near ~1e-7 to ~1e-8, inside the 1e-5 band — but the engine
is shipping wrong-precision output for the entire f64 × {sigmoid, exponential,
exp_standard_ratio, log1p, signed_square, ova} surface, and a large-margin or
edge input can push a sigmoid/exp deviation past 1e-5. The module comment at
`lib.rs:1128-1134` and `postprocessor.rs:13-20` claim the O-generic
postprocessor was "wired in Plan 05-03"; it was not — the f32 path is the only
path.

**Fix:** Make the postprocessor element arithmetic generic over `O` so f64
input runs the postprocessor in f64 (matching upstream `InputT`-templated
instantiation), while keeping `softmax`'s `max_margin`/`t` hardcoded to `f32`
(upstream does so for every `InputT`). Concretely, parameterize the
postprocessor functions over the element type (or add f64 variants) instead of
collapsing to `f32`:

```rust
// sigmoid<O>: O(1) / (O(1) + (-(sigmoid_alpha as O) * v).exp())
// softmax: keep `let mut max_margin: f32`, `let t: f32 = (cell_as_f32 - max_margin).exp()`
//          per upstream postprocessor.cc:59-73, even when O = f64.
```

At minimum, add a committed f64-input fixture whose postprocessor is `sigmoid`
or `exponential` with large-margin rows so the deviation is actually exercised
against the gate rather than silently absorbed.

## Warnings

### WR-01: Matrix harness re-derives the sparse CSR instead of consuming the frozen sparse golden's input

**File:** `crates/treelite-harness/tests/gtil_matrix.rs:240-294` (`run_cell`), `build_csr` at `188-212`
**Issue:**
For every cell (dense *and* sparse), `run_cell` decodes the golden's `input`
into a dense-with-NaN buffer, then synthesizes the CSR with `build_csr` by
treating every non-NaN cell as "present". For a fixture whose `manifest.layout
== "sparse"`, the golden's committed `input` is *already* the dense-with-NaN
materialization (see `capture_gtil_matrix.py:_freeze_cell` passing `dense_nan`
for the sparse layout), so the runner is reconstructing a CSR from a
reconstruction. This means the Rust sparse path is never fed a CSR that
originated from the upstream `scipy.sparse.csr_matrix`; it is fed a CSR whose
present-set == "non-NaN cells", which is only correct as long as the capture's
presence mask never selected a genuine NaN/inf value as "present". The
edge-seeded matrices inject `np.nan`/`±inf` at specific cells
(`capture_gtil_matrix.py:194-203`); if a presence-masked cell holds an injected
NaN, dense-with-NaN cannot distinguish "present NaN" from "absent", and the
reconstructed CSR diverges from the captured CSR — silently, because the test
never compares against the captured `indices`/`indptr`.
**Fix:** Freeze the actual CSR triple (`data`, `col_ind`/`indices`,
`row_ptr`/`indptr`) into the sparse golden payload at capture time and have the
runner load it verbatim for sparse cells, rather than re-deriving presence from
NaN. This makes the sparse path assert against the real captured CSR and
removes the "present NaN == absent" ambiguity.

### WR-02: `predict_score_by_tree` lvs clamp disagrees with `output_shape` (buffer length can mismatch the published shape)

**File:** `crates/treelite-gtil/src/lib.rs:1056-1061` vs `crates/treelite-gtil/src/shape.rs:51-60`
**Issue:**
`predict_score_by_tree` computes the third-dim size as
`lvs = (a * b).max(1)` with `a`/`b` defaulting to `1` on a missing/short
`leaf_vector_shape`, so the produced buffer length is `num_row * num_tree * 1`.
But the public `output_shape` for `ScorePerTree` computes `a * b` with
`unwrap_or(0)` (no `.max(1)`), so for the same malformed/short
`leaf_vector_shape` it reports a third dim of `0`. A Phase-8 caller that
allocates / reshapes against `output_shape` would then disagree with the actual
buffer `predict` returns (shape says `[r, t, 0]` → 0 elements; predict returns
`r * t * 1`). The two clamps must agree.
**Fix:** Use the same clamp in both. Either clamp `output_shape`'s product to
`>= 1`:

```rust
// shape.rs
let a = model.leaf_vector_shape.first().copied().unwrap_or(1).max(0) as u64;
let b = model.leaf_vector_shape.get(1).copied().unwrap_or(1).max(0) as u64;
Shape { dims: vec![num_row, num_tree, (a * b).max(1)] }
```

or have `predict_score_by_tree` honor a 0-width third dim. They cannot diverge.

### WR-03: `evaluate_tree` only bounds-checks the *next* node id, never node 0

**File:** `crates/treelite-gtil/src/lib.rs:407-456` (`evaluate_tree`)
**Issue:**
The loop validates `next` against `[0, num_nodes)` before re-entering, but the
initial `nid = 0` and the first-iteration accessors (`tree.is_leaf(0)`,
`tree.split_index(0)`, `tree.node_type(0)`, `tree.threshold(0)`, etc.) are
called with no check that `num_nodes >= 1`. For a malformed/empty tree
(`num_nodes == 0`, empty `cleft`/`split_index` columns), `tree.is_leaf(0)` or
`tree.split_index(0)` would index an empty `ContiguousArray` and panic — the
exact OOB-panic the ERR-01 contract says must become a typed error. The
declared invariant "a malformed `Model` must never index out of bounds" is not
fully held at node 0.
**Fix:** Guard the entry before the first access, e.g.:

```rust
if num_nodes == 0 {
    return Err(GtilError::NodeIndexOutOfBounds { node: 0 });
}
```

(or make the per-node accessors `get`-based and return the typed error). The
existing tests never construct a 0-node tree, so this gap is untested.

### WR-04: `category_list_safe` swallows malformed begin/end inversions as a silent non-match

**File:** `crates/treelite-gtil/src/lib.rs:375-391` (`category_list_safe`)
**Issue:**
On an out-of-range or inverted (`b > e`) category-list slice the function
returns `&[]`, which `next_node_categorical` then treats as "no category
matches" and routes accordingly. This is a *silent fallback that changes the
prediction*: a corrupt categorical node produces a definite (wrong) routing
instead of surfacing an error, directly the "silent fallback that could mask a
wrong prediction" failure mode the review brief flags. Upstream would slice
unchecked (and the loader guarantees well-formed offsets), so for a corrupt
model the two diverge silently rather than loudly.
**Fix:** Distinguish "legitimately empty list" (`b == e`, in bounds) from
"malformed offsets" (`b > e` or `e > values.len()` or missing offset) and
return a typed `GtilError` for the malformed case rather than `&[]`. Same
applies to the analogous `has_leaf_vector` fallthrough at `lib.rs:513-520`,
which silently treats a malformed leaf-vector offset as "scalar leaf".

### WR-05: `next_node` swallows `kEQ`/`kGT`/`kGE`/`kNone` differences silently relative to the upstream fatal check

**File:** `crates/treelite-gtil/src/lib.rs:316-329` (`next_node`)
**Issue:**
Upstream `NextNode` (`predict.cc:120-122`) hits `TREELITE_CHECK(false)` (throws)
on any operator outside the five comparison ops and returns `-1`. The Rust port
maps `Operator::kNone` to `cond = false` → routes to `right`. For a model that
somehow carries `kNone` on a numerical test node, upstream aborts loudly while
the Rust port silently produces a prediction (route right). The comment
acknowledges "Phase 1 fixtures never reach here", but this is a silent-fallback
divergence from the upstream fatal path on malformed input. (The non-`kNone`
ops `kLE/kEQ/kGT/kGE` are handled and correct.)
**Fix:** Return a typed `GtilError` (e.g. a new `UnrecognizedOperator` variant)
for `kNone` on a numerical node, matching upstream's fatal-on-unrecognized
behavior, rather than defaulting to `right`.

### WR-06: f32-input matrix fixture asserts against a golden but never cross-checks the f64 golden, hiding input-dtype-axis regressions

**File:** `crates/treelite-harness/tests/gtil_matrix.rs:392-400`
**Issue:**
The runner asserts each fixture's own-dtype result against its own-dtype golden
and asserts dense==sparse parity, but there is no assertion tying the f32 and
f64 cells of the *same model/kind/seed* together. The whole point of the D-05
input-dtype axis (per the file header) is that f32-input and f64-input are
*different* computations; nothing here verifies they actually differ where they
should (e.g. that an f32 categorical-gap routing diverges from f64), so an
accidental f32→f64 pre-cast inside a future backend would still pass every
assertion in this test as long as each cell matched its own golden. The
`f32_cells > 0` / `f64_cells > 0` counters only prove both axes ran, not that
the axis is meaningfully exercised.
**Fix:** Add at least one paired assertion that the f32 and f64 cells of the
categorical-gap model differ on the `2^24`-boundary row (or document that the
golden values themselves encode this), so a silent input-dtype collapse is
caught.

## Info

### IN-01: `Operator::kNone` arm comment references a non-existent "default arm returning right_child"

**File:** `crates/treelite-gtil/src/lib.rs:323-326`
**Issue:** The comment says the `kNone` arm "mirrors the upstream default arm
returning right_child after the fatal check", but upstream's default arm
returns `-1` after `TREELITE_CHECK(false)` (`predict.cc:120-123`), not
`right_child`. The comment misdescribes upstream behavior. (See WR-05 for the
behavioral concern.)
**Fix:** Correct the comment to state upstream aborts/returns `-1`, and that the
port chooses a non-fatal route (or a typed error per WR-05).

### IN-02: `LeafVectorTooShort` fields are populated with reversed semantics in score-per-tree

**File:** `crates/treelite-gtil/src/lib.rs:1097-1101`
**Issue:** When `leafvec.len() > lvs`, the error is built with
`needed: leafvec.len(), got: lvs` — but here `lvs` is the *buffer* width and
`leafvec.len()` is the data; the variant's doc comment defines `needed` as "the
minimum leaf-vector length the routing requires" and `got` as "actual
leaf-vector length", which is the opposite of how it is filled here (the leaf
vector is too *long* for the buffer, not too short). The message will read
backwards. Purely diagnostic; no behavioral impact.
**Fix:** Use a distinct variant (e.g. `LeafVectorTooLong { len, capacity }`) or
populate `needed`/`got` consistently with the doc.

### IN-03: `category_match` magnitude bound built per-call (recomputed each node)

**File:** `crates/treelite-gtil/src/lib.rs:219-220, 283-284`
**Issue:** `max_representable_int` is recomputed on every `category_match`
invocation (per node, per row). It is a compile-time constant per `O`
(`2^24` for f32, `u32::MAX` for f64). Out of scope as a perf issue, but it could
be a `const` for clarity and to make the per-dtype boundary self-documenting.
**Fix:** Promote to an associated `const MAX_REPRESENTABLE_INT: Self` on
`PredictOut`.

### IN-04: Fixture comment overstates the `2^24 + 1` f32 edge value

**File:** `fixtures/capture_gtil_matrix.py:197-200`, `lib.rs:1264`
**Issue:** The capture injects `float(2**24 + 1)` into an f64 matrix, then casts
the f32 cell via `astype(np.float32)`, which rounds `2^24 + 1` to exactly
`2^24` (the value *is* representable in f32 and sits *at* the boundary, not in
the `[2^24, u32::MAX]` gap). The comment claims it "is NOT representable in f32
and sits in the gap"; for the f32 cell it actually lands on the boundary
`2^24`, which `category_match` accepts (not rejects). The routing still matches
upstream (both round identically), so this is documentation-only, but the
claimed edge is not the edge being tested on the f32 path. The unit test at
`lib.rs:1258-1278` (which uses `2^24 + 64`, exactly representable) is the real
gap test.
**Fix:** Inject a value that is genuinely in the f32 gap and exactly
representable (e.g. `2**24 + 64`) if the matrix is meant to exercise the gap,
or correct the comment to say the f32 cell tests the boundary.

---

_Reviewed: 2026-06-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
