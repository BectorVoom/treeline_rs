# Phase 5: Code Review Fix Note

**Date:** 2026-06-10
**Source report:** `05-REVIEW.md` (reviewed 2026-06-10T09:28:01Z)
**Scope:** targeted code-review fix on the main working tree (no numbered plan).

## Findings fixed

| Finding | Severity | Status |
|---------|----------|--------|
| CR-01 — f64 `softmax` narrows the row to f32 before subtraction/exp | CRITICAL | Fixed |
| WR-03 — f64 softmax final `static_cast<float>(norm_const)` divide collapsed to f32 | WARNING | Fixed (by the CR-01 `softmax_f64`) |
| WR-04 — misleading `needed`/`got` payload on leaf-vector overflow | WARNING | Fixed |

### CR-01 / WR-03 — `softmax_f64`

- Added `postprocessor::softmax_f64`, porting upstream `softmax<double>`
  (`treelite-mainline/src/gtil/postprocessor.cc:57-75`) verbatim with respect to
  the mixed f32/f64 placement: the `double* row` cells stay **f64** for the
  `row[i] - max_margin` subtraction and `std::exp` (only the result narrows to
  the `f32` `t`), and the final divide is `*cell (f64) /= divisor (f64 from
  `norm_const as f32`)` — a `double /= float` divide. `max_margin` (`row[0]`
  narrowed to f32), `t`, and the divisor are the only `f32` quantities, matching
  the upstream template body literally. The max loop compares in f64 (the f32
  `max_margin` promotes), as upstream does.
- The f64 softmax arm in `apply_postprocessor_f64` (`lib.rs`) now dispatches to
  `softmax_f64` in place instead of the narrow-to-f32 / widen-back dance.
- Removed the now-dead `out_to_f32` / `out_from_f32` `PredictOut` trait methods
  and both impls (they existed only to feed the removed softmax f32-narrow arm),
  and corrected the contract docs in `postprocessor.rs` (module header +
  `softmax` doc) and `lib.rs` (trait `apply_named_postprocessor` doc and
  `apply_postprocessor_f64` doc) to state that softmax<double> keeps its cells in
  f64 — softmax is **not** uniformly f32-correct on every `InputT`.
- Added tests: `softmax_f64_diverges_from_collapsed_f32_on_precise_row` (proves
  the f64 path is not bit-identical to `softmax(row narrowed to f32)` on a
  double-precision multiclass near-tie row), `softmax_f64_matches_upstream_ordering_reference`
  (bit-exact against a hand-computed upstream-ordered reference), and
  `softmax_f64_empty_row_is_noop`.

### WR-04 — leaf-vector overflow payload

- In the ScorePerTree leaf-vector arm (`lib.rs`), the per-element bounds check
  (`if i >= lvs`) now reports `needed: i + 1, got: lvs` instead of
  `needed: leafvec.len(), got: lvs`, so the payload names the overflow point and
  is consistent with the `needed: li + 1` reporting in `output_leaf_vector`.
  Diagnostic-only; the check still prevents the OOB write.

## Commits

| Commit | Subject |
|--------|---------|
| `1e35209` | fix(05): softmax_f64 keeps cell in f64 per upstream (CR-01/WR-03) |
| `6c31263` | fix(05): correct leaf-vector overflow needed/got payload (WR-04) |

(The `05-REVIEW-FIX.md` note itself is committed in a following `docs(05)` commit.)

## Verification

- `cargo test --workspace`: **green** (0 failures), including the `gtil_matrix`
  equivalence harness (`gtil_matrix ... ok`) and the new `softmax_f64` unit tests.
- `cargo clippy -p treelite-gtil --all-targets`: **clean** (0 warnings). The
  only workspace clippy warnings are pre-existing `type_complexity` notes in
  `treelite-sklearn` test helpers — unrelated to this fix and out of scope.

### Worst observed |delta| of the f64 softmax cells vs golden (post-fix)

The `leaf_vec_mc.f32.f64.default.*` cells exercise the f64 softmax path
(softprob → softmax). Against the upstream f64 goldens, after the fix:

| Cell | max \|delta\| |
|------|------------|
| `leaf_vec_mc.f32.f64.default.dense.s1234` | 8.99e-8 |
| `leaf_vec_mc.f32.f64.default.dense.s5678` | 1.06e-7 |
| `leaf_vec_mc.f32.f64.default.sparse.s1234` | 1.08e-7 |
| `leaf_vec_mc.f32.f64.default.sparse.s5678` | 1.08e-7 |

**Worst f64 softmax cell |delta| = ~1.08e-7**, comfortably under the 1e-5 gate.
(The corrected f64 softmax is a different, more faithful computation than the
old narrow-to-f32 path, so the absolute delta vs the f64 golden shifted from the
review's reported ~9e-8 narrow-path value to ~1.08e-7 — still well inside 1e-5,
and now numerically faithful to upstream cast ordering rather than fitting under
tolerance by accident.)

### f32 path byte-identical

The f32 softmax arm (`apply_postprocessor_f32` → `postprocessor::softmax`) was
not touched. The `leaf_vec_mc.f32.f32.default.*` cells report an unchanged
max |delta| = 5.55e-17, confirming the f32 path is byte-identical to pre-fix.

## Intentionally NOT touched

Per the fix scope, the following review findings were left as design-judgement
items the user is deferring:

- **WR-01** — wording of the 1e-5 matrix-gate CR-01 comments (the gate confirms
  the f64 path matches upstream but is sub-1e-5 blind to the collapsed path).
- **WR-02** — the WR-06 `max_div > 0.0` strict bit-inequality guard (vs a
  relative-divergence floor).
- **WR-06** — assertion semantics.
- **IN-01 / IN-02 / IN-03** — informational items.
