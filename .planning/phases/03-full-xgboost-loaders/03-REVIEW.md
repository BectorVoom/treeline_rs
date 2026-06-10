---
phase: 03-full-xgboost-loaders
reviewed: 2026-06-10T00:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - crates/treelite-xgboost/src/lib.rs
  - crates/treelite-xgboost/src/json.rs
  - crates/treelite-xgboost/src/ubjson.rs
  - crates/treelite-xgboost/src/legacy.rs
  - crates/treelite-xgboost/src/detect.rs
  - crates/treelite-xgboost/src/objective.rs
  - crates/treelite-xgboost/src/error.rs
  - crates/treelite-xgboost/Cargo.toml
  - crates/treelite-xgboost/tests/json.rs
  - crates/treelite-xgboost/tests/ubjson.rs
  - crates/treelite-xgboost/tests/legacy.rs
  - crates/treelite-xgboost/tests/detect.rs
  - crates/treelite-xgboost/tests/nan_inf.rs
  - crates/treelite-harness/tests/three_format_equivalence.rs
  - crates/treelite-harness/tests/golden_v5.rs
  - fixtures/generate_xgb_3format.py
findings:
  critical: 4
  warning: 5
  info: 4
  total: 13
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-06-10
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Reviewed the three XGBoost loaders (JSON, UBJSON, legacy-binary), format detection,
the objective→postprocessor map, the version-gated base_score transform, and their
tests. The shared convergence path (`build_model_from_parsed`) and the byte-fidelity
strategy are sound, and most of the bounds-checking and typed-error discipline the
phase prompt calls for is genuinely in place.

However, the dominant risk surface — hand-rolled decoders over untrusted bytes —
has **four panic/abort-on-malformed-input defects** that directly contradict the
stated "typed errors only, never a panic/OOB/abort" invariant:

1. The JSON NaN/Inf pre-lexer (`replace_nonfinite`) panics on a non-ASCII byte in
   value position (non-char-boundary slice) — CR-01.
2. That same pre-lexer silently CORRUPTS non-ASCII UTF-8 string contents
   (`c as char` byte→codepoint re-encoding), breaking the documented
   "string contents BYTE-UNCHANGED" invariant and any model with non-ASCII feature
   names — CR-02.
3. The legacy cursor's `take` uses unchecked `self.pos + n` (the UBJSON cursor
   correctly uses `checked_add`); the leaf-vector skip path can drive `n` near
   `usize::MAX`, overflowing the add and panicking under the default
   debug/`cargo test` overflow-checks — CR-03.
4. The UBJSON recursive-descent decoder has NO recursion-depth limit; a deeply
   nested `[[[[…` stream overflows the stack and aborts the process — CR-04.

All four are reachable from attacker-controlled model files. The remaining warnings
concern misleading doc/size comments, an unused struct field, and a couple of
robustness gaps.

## Critical Issues

### CR-01: `replace_nonfinite` panics on a non-char-boundary slice for non-ASCII bytes in value position

**File:** `crates/treelite-xgboost/src/json.rs:75,79,83`
**Issue:** When a byte `>= 0x80` appears OUTSIDE a string (value position), the
fallthrough arm `_ => { out.push(c as char); i += 1; }` (line 87-90) advances `i`
by one byte into the middle of a multi-byte UTF-8 sequence. The next loop iteration
evaluates `input[i..].starts_with(...)` (lines 75/79/83), which slices the `&str`
at a non-char-boundary and **panics** ("byte index N is not a char boundary"). The
function is documented as operating safely on arbitrary input, and `load_xgboost_json`
feeds raw untrusted model text straight into it before any validation — so a
malformed/hostile `.json` with a stray non-ASCII byte outside a string panics the
process instead of returning `XgbError::Json`.
**Fix:** Operate purely on bytes and guard the literal checks with the byte slice,
never re-slicing the `&str`:
```rust
// match on b[i..] instead of input[i..]; compare against byte-string literals.
_ if b[i..].starts_with(b"-Infinity") => { out.push_str("\"@-Inf@\""); i += 9; }
_ if b[i..].starts_with(b"Infinity")  => { out.push_str("\"@Inf@\"");  i += 8; }
_ if b[i..].starts_with(b"NaN")       => { out.push_str("\"@NaN@\"");  i += 3; }
```
(Combined with the CR-02 fix below so non-ASCII bytes are copied verbatim.)

### CR-02: `replace_nonfinite` corrupts non-ASCII UTF-8 string contents (`c as char` re-encoding)

**File:** `crates/treelite-xgboost/src/json.rs:58,88`
**Issue:** Both the in-string copy (line 58) and the value-position fallthrough
(line 88) do `out.push(c as char)`, where `c: u8`. For any byte `0x80..=0xFF`,
`c as char` produces the Unicode scalar `U+0080..U+00FF`, and `String::push`
re-encodes it as **two** UTF-8 bytes. So every non-ASCII byte in the input is
rewritten (e.g. a UTF-8 feature name like `"température"` is mangled). This directly
violates the module's load-bearing promise that string contents are left
"BYTE-UNCHANGED" (json.rs:42, and the `nan_inf_string_contents_are_byte_unchanged`
test only covers ASCII). The corrupted output is then handed to `serde_json::from_str`,
so a model with non-ASCII feature names/attributes parses into mangled data or fails
— a silent correctness/data-integrity break, not just cosmetic.
**Fix:** Build the output as a `Vec<u8>` and push raw bytes, converting to `String`
via `String::from_utf8` (or `from_utf8_unchecked` given the input was already valid
UTF-8 and only ASCII sentinels are inserted):
```rust
let mut out: Vec<u8> = Vec::with_capacity(b.len());
// ... in_str branch: out.push(c);   fallthrough: out.push(c);
// sentinel arms: out.extend_from_slice(b"\"@Inf@\"");  etc.
String::from_utf8(out).expect("input was valid UTF-8; only ASCII inserted")
```

### CR-03: legacy `Cursor::take` uses unchecked `self.pos + n`; leaf-vector skip can overflow and panic

**File:** `crates/treelite-xgboost/src/legacy.rs:88` (and skip path at 379-383)
**Issue:** The legacy cursor computes `self.buf.get(self.pos..self.pos + n)` with a
plain `+` (line 88), unlike the UBJSON cursor which correctly uses `checked_add`
(ubjson.rs:76). The conditional leaf-vector tail reads an attacker-controlled `u64`
length, does `bytes = len_usize.checked_mul(4)` (line 379), then calls
`c.skip(bytes)` → `take(bytes)` WITHOUT first bounds-checking `bytes` against
`remaining()`. A crafted `len` of roughly `usize::MAX / 4` passes `checked_mul(4)`
(yielding `bytes` near `usize::MAX`), and `self.pos + bytes` then overflows `usize`.
Under the crate's default profile (`cargo test` is a debug build with
`overflow-checks = on`), this is an **arithmetic-overflow panic** on malformed input
— exactly the OOB/panic the T-03-L0x safety contract says cannot happen. (In release
the add wraps and `get` returns `None`, so the bug is masked there, which is why the
existing truncation tests miss it.)
**Fix:** Mirror the UBJSON cursor — use `checked_add` in `take`:
```rust
fn take(&mut self, n: usize) -> Result<&'a [u8], XgbError> {
    let end = self.pos.checked_add(n)
        .ok_or_else(|| XgbError::Legacy { pos: self.pos, detail: "length overflow".into() })?;
    match self.buf.get(self.pos..end) {
        Some(slice) => { self.pos = end; Ok(slice) }
        None => self.err(format!("truncated: need {n} bytes, only {} remain", self.remaining())),
    }
}
```
Also apply the same `checked_add` fix to `peek` (line 104). Optionally bounds-check
`bytes <= c.remaining()` before the leaf-vector `skip` for a clearer error.

### CR-04: UBJSON decoder has no recursion-depth limit → stack-overflow abort on deeply nested input

**File:** `crates/treelite-xgboost/src/ubjson.rs:110-113,168-169,249,300` (recursive `decode_value`/`decode_array`/`decode_object`)
**Issue:** `decode_value` → `decode_array`/`decode_object` → `decode_value` recurse
once per nesting level with no depth cap. Each array/object opener (`[` / `{`) is a
single byte, so a stream of N `[` bytes produces N-deep recursion. A small hostile
file (tens of KB of `[`) overflows the native stack and **aborts the process** (SIGSEGV
/ abort) — uncatchable, not an `XgbError::Ubjson`. This defeats the module's stated
guarantee that a malformed stream "returns a typed error, never a panic or an OOM"
(ubjson.rs:36-42, error.rs:84-99). The JSON path is protected (serde_json caps
recursion at 128) and the legacy path is iterative, so UBJSON is the sole exposure.
**Fix:** Thread a depth counter through the recursion and reject past a fixed bound:
```rust
const MAX_DEPTH: usize = 100; // upstream/serde_json use a similar small cap
fn decode_value(c: &mut Cursor, depth: usize) -> Result<Value, XgbError> {
    if depth > MAX_DEPTH { return Err(c.err("UBJSON nesting too deep")); }
    let tag = c.take_u8()?;
    decode_with_tag(c, tag, depth)
}
// pass depth + 1 into decode_array / decode_object recursive calls.
```

## Warnings

### WR-01: `SIZE_GBTREE_MODEL_PARAM` doc comments say "168 bytes" but the constant and check use 160

**File:** `crates/treelite-xgboost/src/legacy.rs:241` (and `read_gbtree_model_param` doc), `lib.rs`/comments referencing 168
**Issue:** The constant is correctly `160` (line 50, with a thorough justification at
lines 43-49), and the field-by-field reader consumes exactly 160 bytes, but the
doc comment on `read_gbtree_model_param` still says "Read the 168-byte
`GBTreeModelParam` … Asserts 168 bytes" (line 241), and lib.rs step-4 comment says
"GBTreeModelParam (168 bytes)" (line 446). A maintainer reconciling the reader
against the comment will be misled into thinking there is an 8-byte discrepancy.
This is a stale-comment defect, not a logic bug (the code is internally consistent).
**Fix:** Update both comments to "160-byte `GBTreeModelParam` … Asserts 160 bytes"
to match `SIZE_GBTREE_MODEL_PARAM`.

### WR-02: `nodes_bytes + stats_bytes` uses unchecked addition

**File:** `crates/treelite-xgboost/src/legacy.rs:317`
**Issue:** `if nodes_bytes + stats_bytes > c.remaining()` adds two `checked_mul`
results with a plain `+`. `num_nodes` is an `i32` (already rejected if `<= 0`), so on
64-bit the sum cannot overflow in practice, but on a 32-bit target a large `num_nodes`
could make `nodes_bytes` (×20) and `stats_bytes` (×16) sum past `u32::MAX` and wrap,
defeating the guard (and panicking under overflow-checks). Given the file's explicit
"validate before allocating" contract, the addition should be checked too.
**Fix:** `nodes_bytes.checked_add(stats_bytes).ok_or_else(|| XgbError::Legacy { … })?`
before comparing against `remaining()`.

### WR-03: `GBTreeModelParam.num_roots` field is read into a struct field that is never used

**File:** `crates/treelite-xgboost/src/legacy.rs:236-239,244-245,261-264`
**Issue:** `num_roots` is the only field besides `num_trees` kept on the
`GBTreeModelParam` struct, and the single-root invariant IS enforced from it in
`load_xgboost_legacy` (line 457), so it is used — but `num_roots` from the per-tree
`TreeParam` (read at legacy.rs:278 as `_num_roots_tp`) is decoded-and-discarded with
an explanatory comment (lines 394-398) that the per-tree check is intentionally
relaxed. That is defensible, but it means the per-tree `num_roots` is read purely for
stride and silently ignored even when it is malformed (e.g. 0 or negative), whereas
upstream gates on it for < 1.6 models. If a hostile file sets the GBTreeModelParam
`num_roots = 1` but a per-tree `num_roots = 0`, the loader accepts it. Confirm this
matches upstream's relaxed-path semantics for the formats in scope; otherwise it is a
validation gap.
**Fix:** Either document that the per-tree `num_roots` is deliberately unchecked for
≥1.6 parallel-tree compatibility (already partly done), or gate on it when
`major_version < 2` to match upstream's pre-1.6 enforcement.

### WR-04: `major_version` truncation silently clamps to `i32::MAX` instead of erroring

**File:** `crates/treelite-xgboost/src/legacy.rs:518`
**Issue:** `let major_version = i32::try_from(mparam.major_version).unwrap_or(i32::MAX);`
silently substitutes `i32::MAX` when the decoded `u32` major version exceeds
`i32::MAX`. Every other scalar in this function returns a typed `XgbError::Legacy` on
overflow (num_target line 509-512, num_feature line 514-517). Clamping the version to
`i32::MAX` forces the base_score→margin transform gate (`version[0] >= 1`) to ALWAYS
fire for such an input, which is a silent behavior change rather than a rejection.
For a corrupted header this is a fidelity hazard, not a crash, but it is inconsistent
with the surrounding fail-loud discipline.
**Fix:** Return a typed error like the sibling conversions:
```rust
let major_version = i32::try_from(mparam.major_version)
    .map_err(|_| XgbError::Legacy { pos: c.pos, detail: format!("major_version {} too large", mparam.major_version) })?;
```

### WR-05: `de_f32` accepts arbitrary numeric-string fallback, masking malformed thresholds

**File:** `crates/treelite-xgboost/src/json.rs:119-126`
**Issue:** The `visit_str` arm falls through to `other.parse()` for any string that
is not one of the three sentinels. The doc frames this as letting a "stray
numeric-as-string still round-trip," but it also means a `split_conditions` array
element supplied as the string `"1.5"` (rather than a JSON number) is silently
accepted where upstream's strongly-typed SAX handler would treat the node arrays as
numeric arrays. This loosens the schema beyond what XGBoost emits and can mask a
genuinely malformed model (e.g. mixed string/number arrays) instead of surfacing it.
Low severity because real XGBoost output never does this, but it is an
attacker-permissive deviation from the upstream contract.
**Fix:** Restrict `visit_str` to exactly the three sentinels and return
`de::Error::custom("unexpected string in f32 field")` for anything else, OR document
explicitly that the permissive fallback is intentional and bounded.

## Info

### IN-01: `info_raw` vector is built but never read

**File:** `crates/treelite-xgboost/src/legacy.rs:336,351`
**Issue:** `info_raw` is allocated with capacity `num_nodes_usize` and pushed to in
the node loop (line 351), but never consumed — `split_conditions` already carries the
same `info` union value (line 356) and is what `build_tree` reads. `info_raw` is dead
state (one wasted allocation + writes per tree).
**Fix:** Remove the `info_raw` declaration and its `push`.

### IN-02: `_ = tree_id` no-op in the DART weight-drop loop

**File:** `crates/treelite-xgboost/src/legacy.rs:491-492`
**Issue:** `for (tree_id, tree) in trees.iter_mut().enumerate() { let _ = tree_id; … }`
enumerates only to immediately discard the index with `let _ = tree_id;`. The
`.enumerate()` and the discard are both unnecessary.
**Fix:** Iterate `for tree in trees.iter_mut() { … }` and drop the `let _ = tree_id;`.

### IN-03: `read_tree` accepts a `weight_drop: Option<f32>` parameter that is always `None`

**File:** `crates/treelite-xgboost/src/legacy.rs:271-275,402-408,473`
**Issue:** `read_tree` is only ever called with `weight_drop = None` (line 473); the
actual DART fold is applied separately in `load_xgboost_legacy` (lines 491-500) after
all trees are read. The in-function fold block (lines 402-408) is therefore dead code,
and the parameter is vestigial.
**Fix:** Drop the `weight_drop` parameter and the dead fold block in `read_tree`,
keeping the single fold site in `load_xgboost_legacy`.

### IN-04: `float_to_value` uses `unwrap_or(Value::Null)` on a branch documented as unreachable

**File:** `crates/treelite-xgboost/src/ubjson.rs:184-187`
**Issue:** The finite branch calls `Number::from_f64(v).map(Value::Number).unwrap_or(Value::Null)`
with a comment asserting the `None` case is impossible (v is finite). The defensive
`unwrap_or(Value::Null)` is harmless but would, if the invariant ever broke, silently
turn a value into `Null` (data loss) rather than erroring. Minor.
**Fix:** Acceptable as-is; if stricter, return a typed error in the (unreachable)
`None` case rather than `Value::Null`.

---

_Reviewed: 2026-06-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
