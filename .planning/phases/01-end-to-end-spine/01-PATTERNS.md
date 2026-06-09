# Phase 1: End-to-End Spine - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 18 new files (4 crates + fixtures + workspace root)
**Analogs found:** 16 / 18 (2 have no analog — workspace config + golden capture)

> **Greenfield port note.** The Rust workspace is a stub today. There is *no* Rust code to copy from. The authoritative analogs are the vendored, read-only C++ upstream at `treelite-mainline/` (Treelite v4.7.0 — the porting source of truth) and `xgboost-master/` (XGBoost-JSON schema authority). Every Rust file below maps to the exact C++ source that defines the contract it must port. All line numbers were re-read this session and are exact. Where the C++ throws `TREELITE_LOG(FATAL)` / `TREELITE_CHECK`, the Rust port returns a `thiserror` `Err` (per ERR-01) instead of panicking.

## File Classification

| New File | Role | Data Flow | Closest Analog (C++ upstream) | Match Quality |
|----------|------|-----------|-------------------------------|---------------|
| `Cargo.toml` (workspace root) | config | — | — | no analog (Cargo-specific) |
| `crates/treelite-core/src/enums.rs` | model (vocabulary) | transform (enum↔string) | `src/enum/{task_type,tree_node_type,operator,typeinfo}.cc` | exact |
| `crates/treelite-core/src/tree_buf.rs` | model (storage primitive) | transform (owned/borrowed buffer) | `include/treelite/contiguous_array.h` | exact |
| `crates/treelite-core/src/tree.rs` | model | request-response (SoA getters) | `include/treelite/tree.h:78-335` | exact |
| `crates/treelite-core/src/model.rs` | model | — (header container) | `include/treelite/tree.h:437-573` | exact |
| `crates/treelite-core/src/error.rs` | utility (typed error) | — | `treelite::Error` via `TREELITE_LOG(FATAL)` | role-match |
| `crates/treelite-xgboost/src/lib.rs` | loader | transform (JSON→Model) | `src/model_loader/detail/xgboost_json/delegated_handler.cc` | exact |
| `crates/treelite-xgboost/src/objective.rs` | service | transform (objective→postproc, base_score→margin) | `src/model_loader/detail/xgboost.{h,cc}` | exact |
| `crates/treelite-xgboost/src/error.rs` | utility (typed error) | — | `TREELITE_CHECK`/`TREELITE_LOG(ERROR)` paths | role-match |
| `crates/treelite-gtil/src/lib.rs` | service (inference) | request-response (predict) | `src/gtil/predict.cc:99-323` | exact |
| `crates/treelite-gtil/src/postprocessor.rs` | service | transform (margin→prob) | `src/gtil/postprocessor.cc:19-37` | exact |
| `crates/treelite-gtil/src/error.rs` | utility (typed error) | — | GTIL `TREELITE_CHECK` paths | role-match |
| `crates/treelite-harness/src/lib.rs` | test (equivalence instrument) | request-response | upstream test harness (role only) | role-match |
| `crates/treelite-harness/tests/equivalence.rs` | test | request-response | `tests/cpp/` GTest structure (role only) | role-match |
| `fixtures/binary_logistic.model.json` | config (fixture) | file-I/O | XGBoost-JSON recognized-key list `delegated_handler.cc:484-490` | exact (schema) |
| `fixtures/capture_golden.py` | utility (one-time provenance) | file-I/O | — | no analog (Python wheel script) |
| `crates/*/Cargo.toml` (per-crate) | config | — | — | no analog (Cargo-specific) |

---

## Pattern Assignments

### `crates/treelite-core/src/enums.rs` (model/vocabulary, enum↔string transform)

**Analog:** `treelite-mainline/src/enum/{task_type,tree_node_type,operator,typeinfo}.cc`

**The four string tables — port EXACTLY (ENUM-01).** Strings are NON-uniform across the four enums; do not assume a consistent style. `FromString` on an unknown value is `TREELITE_LOG(FATAL)` upstream → Rust returns `Err(CoreError::UnknownEnumString { .. })`.

**`TaskType` — `kXxx`-style strings** (`task_type.cc:15-47`):
```cpp
case TaskType::kBinaryClf:       return "kBinaryClf";
case TaskType::kRegressor:       return "kRegressor";
case TaskType::kMultiClf:        return "kMultiClf";
case TaskType::kLearningToRank:  return "kLearningToRank";
case TaskType::kIsolationForest: return "kIsolationForest";
// FromString unknown → TREELITE_LOG(FATAL) << "Unknown task type: " << str;
```

**`TreeNodeType` — lowercase snake strings** (`tree_node_type.cc:15-39`):
```cpp
case TreeNodeType::kLeafNode:            return "leaf_node";
case TreeNodeType::kNumericalTestNode:   return "numerical_test_node";
case TreeNodeType::kCategoricalTestNode: return "categorical_test_node";
```

**`Operator` — symbolic strings** (`operator.cc:16-49`); note `kNone → ""`:
```cpp
case Operator::kEQ: return "==";
case Operator::kLT: return "<";
case Operator::kLE: return "<=";
case Operator::kGT: return ">";
case Operator::kGE: return ">=";
// default (incl. kNone): return "";
```

**`DType` (≡ upstream `TypeInfo`) — lowercase strings** (`typeinfo.cc:15-42`):
```cpp
case TypeInfo::kInvalid: return "invalid";
case TypeInfo::kUInt32:  return "uint32";
case TypeInfo::kFloat32: return "float32";
case TypeInfo::kFloat64: return "float64";
// NOTE: TypeInfoFromString does NOT accept "invalid" — only uint32/float32/float64;
//       anything else → kInvalid (Rust: Err for unknown, but map "invalid" deliberately).
```

**Integer reprs** (for `#[repr(..)]`, from the `.h` files cited in RESEARCH.md):
`TaskType: u8 {0..4}`; `TreeNodeType: i8 {0,1,2}`; `Operator: i8 {kNone=0,kEQ=1,kLT=2,kLE=3,kGT=4,kGE=5}`; `TypeInfo/DType: u8 {kInvalid=0,kUInt32=1,kFloat32=2,kFloat64=3}`.

---

### `crates/treelite-core/src/tree_buf.rs` (model/storage primitive, owned/borrowed)

**Analog:** `treelite-mainline/include/treelite/contiguous_array.h:16-63`

**Owned/borrowed contract** (CORE-03). Upstream is a manual `T* buffer_` + `bool owned_buffer_` with `UseForeignBuffer` for zero-copy aliasing, copy ctor deleted, POD-only:
```cpp
template <typename T> class ContiguousArray {
  ContiguousArray(ContiguousArray const&) = delete;          // no implicit copy → explicit Clone()
  ContiguousArray& operator=(ContiguousArray const&) = delete;
  ContiguousArray(ContiguousArray&& other) noexcept;          // move-only
  inline ContiguousArray Clone() const;                       // explicit deep copy
  inline void UseForeignBuffer(void* prealloc_buf, std::size_t size);  // zero-copy borrow
  inline T& operator[](std::size_t idx);                      // unsafe, no bounds check
  inline T& at(std::size_t idx);                              // safe, bounds-checked
  static_assert(std::is_pod<T>::value, "T must be POD");      // → Rust: T: Copy (bytemuck::Pod is Phase 9)
 private:
  T* buffer_; std::size_t size_; std::size_t capacity_; bool owned_buffer_;
};
```
**Rust port (discretion, RESEARCH §CORE-03):** `enum TreeBuf<T> { Owned(Vec<T>), Borrowed { ptr: *const T, len: usize } }` or `Cow<'a,[T]>`. Phase 1 only needs *both modes to exist* + a borrowed-slice round-trip unit test; the real borrowed consumer (Python buffer protocol) is Phase 8. Do NOT pull in `bytemuck` yet — that POD seam is Phase 9.

---

### `crates/treelite-core/src/tree.rs` (model, SoA columns + getters)

**Analog:** `treelite-mainline/include/treelite/tree.h:78-335`

**The ~20 parallel SoA columns — port the field set verbatim** (CORE-02), `tree.h:97-132`:
```cpp
ContiguousArray<TreeNodeType> node_type_;
ContiguousArray<std::int32_t> cleft_;          // -1 ⇒ leaf
ContiguousArray<std::int32_t> cright_;
ContiguousArray<std::int32_t> split_index_;    // feature index
ContiguousArray<bool> default_left_;           // missing-value direction
ContiguousArray<LeafOutputType> leaf_value_;
ContiguousArray<ThresholdType> threshold_;
ContiguousArray<Operator> cmp_;                // XGBoost always kLT
ContiguousArray<bool> category_list_right_child_;
// Leaf vector (empty for binary:logistic):
ContiguousArray<LeafOutputType> leaf_vector_;
ContiguousArray<std::uint64_t> leaf_vector_begin_;
ContiguousArray<std::uint64_t> leaf_vector_end_;
// Category list (empty for binary:logistic):
ContiguousArray<std::uint32_t> category_list_;
ContiguousArray<std::uint64_t> category_list_begin_;
ContiguousArray<std::uint64_t> category_list_end_;
// Node statistics (present-but-empty allowed in Phase 1):
ContiguousArray<std::uint64_t> data_count_;
ContiguousArray<double> sum_hess_;
ContiguousArray<double> gain_;
ContiguousArray<bool> data_count_present_; sum_hess_present_; gain_present_;
bool has_categorical_split_{false};            // tree.h:126
std::int32_t num_nodes{0};                      // tree.h:158
// num_opt_field_per_tree_/_per_node_ (tree.h:131-132) are serialization bookkeeping → Phase 2.
```

**Move-only invariant** (`tree.h:88-95`) — Rust must NOT derive `Clone` casually; mirror explicit `Clone()`:
```cpp
Tree(Tree const&) = delete;
Tree& operator=(Tree const&) = delete;
Tree(Tree&&) noexcept = default;                // move-only
inline Tree<ThresholdType, LeafOutputType> Clone() const;   // explicit deep copy
static_assert(std::is_same_v<ThresholdType, LeafOutputType>);  // tree.h:85 — NO mixed types
```

**Getters to port (the traversal contract)** (`tree.h:169-235`):
```cpp
inline int  LeftChild(int nid)    const { return cleft_[nid]; }                       // :169
inline int  RightChild(int nid)   const { return cright_[nid]; }                      // :176
inline int  DefaultChild(int nid) const { return default_left_[nid] ? cleft_[nid] : cright_[nid]; } // :183
inline std::int32_t SplitIndex(int nid) const { return split_index_[nid]; }           // :190
inline bool IsLeaf(int nid)       const { return cleft_[nid] == -1; }                  // :204
inline LeafOutputType LeafValue(int nid) const { return leaf_value_[nid]; }           // :211
inline bool HasLeafVector(int nid) const { return leaf_vector_begin_[nid] != leaf_vector_end_[nid]; } // :233
```
For the `binary:logistic` fixture `HasLeafVector` is always false (begin == end) → scalar leaf path only.

---

### `crates/treelite-core/src/model.rs` (model, two-variant enum + header metadata)

**Analog:** `treelite-mainline/include/treelite/tree.h:437-573`

**Two-variant model** (CORE-01), `tree.h:437`:
```cpp
using ModelPresetVariant = std::variant<ModelPreset<float,float>, ModelPreset<double,double>>;
```
**Rust port:** `enum ModelVariant { F32(ModelPreset<f32>), F64(ModelPreset<f64>) }`; replace `std::visit` (`tree.h:473-483`) with `match`. XGBoost-JSON only ever produces the **F32** variant. Header metadata lives on `Model`, OUTSIDE the variant (exactly as upstream).

**Move-only** (`tree.h:462-465`): `Model(Model const&) = delete;` → Rust move-only / explicit clone.

**Header field set — port verbatim** (CORE-04), `tree.h:535-553`. **Critical deviation from ROADMAP wording: `num_class`, `leaf_vector_shape`, `target_id`, `class_id` are ARRAYS (`ContiguousArray<i32>`), not scalars:**
```cpp
std::int32_t num_feature{0};                    // :535
TaskType task_type;                             // :537
bool average_tree_output{false};                // :539  (XGBoost hardcodes false)
std::int32_t num_target;                         // :542
ContiguousArray<std::int32_t> num_class;         // :543  → [1] for binary clf
ContiguousArray<std::int32_t> leaf_vector_shape; // :544  → [1,1]
ContiguousArray<std::int32_t> target_id;         // :546  per-tree → [0]
ContiguousArray<std::int32_t> class_id;          // :547  per-tree → [0]
std::string postprocessor;                       // :549  → "sigmoid"
float sigmoid_alpha{1.0f};                        // :550
float ratio_c{1.0f};                              // :551
ContiguousArray<double> base_scores;             // :552  f64, margin-transformed
std::string attributes;                          // :553  may be ""
// private serialization bookkeeping (Phase 2): num_tree_, num_opt_field_per_model_,
//   major/minor/patch_ver_ (default {4,7,0} OK for Phase 1), threshold_type_, leaf_output_type_ (:556-567)
```

---

### `crates/treelite-xgboost/src/objective.rs` (service, objective→postproc + base_score→margin)

**Analog:** `treelite-mainline/src/model_loader/detail/xgboost.{h,cc}` — **port the math verbatim; re-deriving it is the #1 way to silently break 1e-5.**

**objective → postprocessor map** (`xgboost.cc:28-50`):
```cpp
if (obj == "multi:softmax" || obj == "multi:softprob") return "softmax";
else if (obj == "reg:logistic" || obj == "binary:logistic") return "sigmoid";
else if (/* count:poisson, reg:gamma, reg:tweedie, survival:cox, survival:aft */) return "exponential";
else if (obj == "binary:hinge") return "hinge";
else if (/* reg:squarederror, reg:linear, ..., binary:logitraw, rank:* */) return "identity";
else TREELITE_LOG(FATAL) << "Unrecognized XGBoost objective";   // → Rust Err
```

**base_score → margin transform — MUST be f64** (`xgboost.h:16-23`, `xgboost.cc:52-60`):
```cpp
struct ProbToMargin {
  static double Sigmoid(double base_score)     { return -std::log(1.0 / base_score - 1.0); }
  static double Exponential(double base_score) { return std::log(base_score); }
};
double TransformBaseScoreToMargin(std::string const& pp, double base_score) {
  if (pp == "sigmoid")     return ProbToMargin::Sigmoid(base_score);
  if (pp == "exponential") return ProbToMargin::Exponential(base_score);
  return base_score;
}
```
**Pitfall (RESEARCH §Pitfall 2):** pick a fixture `base_score != 0.5` (e.g. `0.25`) so the transform is genuinely exercised — at 0.5 the margin is exactly 0 and masks the bug.

---

### `crates/treelite-xgboost/src/lib.rs` (loader, JSON→Model)

**Analog:** `treelite-mainline/src/model_loader/detail/xgboost_json/delegated_handler.cc`

**Recognized per-tree key list** the fixture must satisfy (`delegated_handler.cc:484-490`):
```cpp
loss_changes, sum_hessian, base_weights, categories_segments, categories_sizes,
categories_nodes, categories, leaf_child_counts, left_children, right_children,
parents, split_indices, split_type, split_conditions, default_left, tree_param, id, leaf_weights
```

**Array-length validation — mirror as typed `Err` not panic** (`delegated_handler.cc:423-432`, V5 input validation):
```cpp
if (num_nodes != split_conditions.size()) { TREELITE_LOG(ERROR) << "...incorrect dimension..."; return false; }
if (num_nodes != default_left.size())     { TREELITE_LOG(ERROR) << "...incorrect dimension..."; return false; }
```

**Per-node build loop — leaf vs numerical test** (`delegated_handler.cc:435-479`):
```cpp
if (left_children[node_id] == -1) {           // leaf
  if (size_leaf_vector > 1) LeafVector(...); else LeafScalar(split_conditions[node_id]);
} else {                                       // internal (Phase 1 fixture: numerical only)
  // split_type == kCategorical branch is Phase 5 — skip
  NumericalTest(split_indices[node_id], split_conditions[node_id],
                default_left[node_id], Operator::kLT,            // XGBoost ALWAYS kLT
                left_children[node_id], right_children[node_id]);
}
```

**Metadata finalize math — port verbatim** (`LearnerHandler::EndObject`, `delegated_handler.cc:811-903`). This computes the exact header values the golden depends on:
```cpp
bool const average_tree_output = false;                              // :814 (hardcoded)
// binary/regressor branch (num_class <= 1), :847-872:
if (StringStartsWith(objective_name, "binary:")) task_type = kBinaryClf;   // :849-850
num_class = std::vector<std::int32_t>(num_target, 1);                // :856  → [1]
class_id  = std::vector<std::int32_t>(num_tree, 0);                  // :857  → [0]
// size_leaf_vector <= 1 (scalar) path:
target_id[i] = tree_info[i];                                          // :867-868 → [0]
leaf_vector_shape = {1, 1};                                           // :870-871

// base_scores in f64 (:877-888): fill from scalar (<3.1) or copy vector (3.1+)
base_scores[*] = static_cast<double>(learner_params.base_score[*]);
// margin transform gate (:893-896): version empty OR version[0] >= 1
bool need_transform = output.version.empty() || output.version[0] >= 1;
if (need_transform) base_scores[e] = TransformBaseScoreToMargin(postprocessor.name, e);
```
**Set the fixture `"version": [4,7,0]`** so `version[0] >= 1` fires the margin transform (RESEARCH §Pitfall 2).

---

### `crates/treelite-gtil/src/lib.rs` (service/inference, predict)

**Analog:** `treelite-mainline/src/gtil/predict.cc:99-305`

**Scalar traversal `EvaluateTree` — port verbatim** (`predict.cc:152-172`). Phase 1 fixture has no categorical nodes, so the `NextNodeCategorical` branch (`:127-150`) is NOT ported (Phase 5):
```cpp
int EvaluateTree(Tree const& tree, row) {
  int node_id = 0;
  while (!tree.IsLeaf(node_id)) {
    auto split_index = tree.SplitIndex(node_id);
    InputT fvalue = row(split_index);
    if (std::isnan(fvalue)) { node_id = tree.DefaultChild(node_id); }      // missing → default
    else { node_id = NextNode(fvalue, tree.Threshold(node_id), tree.ComparisonOp(node_id),
                              tree.LeftChild(node_id), tree.RightChild(node_id)); }
  }
  return node_id;
}
```

**`NextNode` comparison** (`predict.cc:99-124`) — XGBoost always `kLT`:
```cpp
switch (op) {
  case Operator::kLT: cond = fvalue <  threshold; break;
  case Operator::kLE: cond = fvalue <= threshold; break;
  case Operator::kEQ: cond = fvalue == threshold; break;
  case Operator::kGT: cond = fvalue >  threshold; break;
  case Operator::kGE: cond = fvalue >= threshold; break;
}
return (cond ? left_child : right_child);
```

**Predict assembly order — THIS ORDERING IS THE 1e-5 CONTRACT** (`PredictRaw`, `predict.cc:231-305`):
```cpp
std::fill_n(output, size, InputT{});                                  // :238  zero-fill
// per row, SERIAL in tree_id (GTIL-08 — float add is non-associative):
output_view(row_id, target_id[tree], class_id[tree])
    += static_cast<InputT>(tree.LeafValue(leaf_id));                  // :228 OutputLeafValue
// skip averaging (average_tree_output == false for XGBoost), :259
// add base_scores — base_scores is f64, added into the InputT(f32) accumulator:
output_view(row_id, t, c) += base_score_view(t, c);                   // :294-304
```
**Pitfall (RESEARCH §Pitfall 3):** mirror cast ordering EXACTLY — f32 accumulator, f32 leaf/threshold, f64 `base_scores` added in, f32 `sigmoid_alpha`. Doing the whole chain in f64 (or all-f32) shifts the last ULPs past 1e-5. For `binary:logistic`: `num_target=1, num_class=[1]` → output shape `(num_row,1,1)` = one scalar per row.

---

### `crates/treelite-gtil/src/postprocessor.rs` (service, margin→prob transform)

**Analog:** `treelite-mainline/src/gtil/postprocessor.cc:19-37`

```cpp
template <typename InputT> void identity(Model const&, std::int32_t, InputT*) {}        // :20
template <typename InputT> void sigmoid(Model const& model, std::int32_t, InputT* elem) {// :34-37
  InputT const val = *elem;
  *elem = InputT(1) / (InputT(1) + std::exp(-model.sigmoid_alpha * val));
}
```
**Rust port** (RESEARCH §Code Examples) — `sigmoid_alpha` is f32, `exp` on the f32 value:
```rust
fn identity(_alpha: f32, v: f32) -> f32 { v }
fn sigmoid(sigmoid_alpha: f32, v: f32) -> f32 { 1.0f32 / (1.0f32 + (-sigmoid_alpha * v).exp()) }
```
**Pitfall (RESEARCH §Pitfall 4):** `exp` may differ by a ULP across libm versions — this is why D-07 mandates the libm/glibc manifest. 1e-5 comfortably absorbs single-ULP `exp` divergence for one scalar.

---

### `fixtures/binary_logistic.model.json` (config/fixture, hand-crafted XGBoost-JSON)

**Schema authority:** `delegated_handler.cc:484-490` (recognized keys) + `:423-432` (length checks). Author against the *loader's key list*, NOT the `saving_model.rst` config dump (RESEARCH §Pitfall 5). Required nesting:
```json
{"learner": {
   "learner_model_param": {"num_feature": ..., "num_class": "0", "num_target": "1", "base_score": "2.5E-1"},
   "gradient_booster": {"name":"gbtree","model":{
       "trees":[ {"tree_param":{"num_nodes":"3","size_leaf_vector":"0"},
                  "left_children":[1,-1,-1],"right_children":[2,-1,-1],
                  "split_indices":[0,0,0],"split_type":[0,0,0],
                  "split_conditions":[<thr>,<leaf>,<leaf>],"default_left":[1,0,0]} ],
       "tree_info":[0],"gbtree_model_param":{"num_trees":"1", ...}}},
   "objective":{"name":"binary:logistic"}},
 "version":[4,7,0]}
```
Use `base_score = 0.25` (not 0.5) so the sigmoid margin transform is exercised. Validate by actually running `capture_golden.py` against `treelite==4.7.0` — if it loads, the fixture is correct (Assumption A1).

---

## Shared Patterns

### Typed errors (ERR-01) — `thiserror`, per-crate
**Source pattern:** every upstream `TREELITE_LOG(FATAL)` / `TREELITE_CHECK` / `TREELITE_LOG(ERROR) ... return false` is a fatal path that the Rust port converts to a returned `Err`, NOT a panic.
**Apply to:** `treelite-core/src/error.rs`, `treelite-xgboost/src/error.rs`, `treelite-gtil/src/error.rs`.
**Concrete fatal-path sites to convert:** unknown enum string (`task_type.cc:44`, `operator.cc:46`, `tree_node_type.cc:36`, `typeinfo.cc:39`); unrecognized objective (`xgboost.cc:47`); array-length mismatch (`delegated_handler.cc:423-432`); out-of-bounds node index during traversal (use bounds-checked indexing — V5 input validation).

### Error context (ERR-02) — `anyhow` in harness/tests ONLY
**Apply to:** `treelite-harness/src/lib.rs`, `treelite-harness/tests/equivalence.rs`, and any integration test. NEVER in a library crate's public API.

### Move-only / no-implicit-copy invariant
**Source:** `tree.h:90-91` (Tree), `tree.h:462-463` (Model), `contiguous_array.h:22-23` (ContiguousArray) — all delete the copy ctor and provide explicit `Clone()`.
**Apply to:** `Tree<T>`, `Model`, `TreeBuf<T>` — do NOT `#[derive(Clone)]` casually; expose an explicit `.clone()`-style deep copy. (RESEARCH §Anti-Patterns.)

### Struct-of-Arrays, never a `Node` struct
**Source:** `tree.h:97-132` (parallel `ContiguousArray` columns).
**Apply to:** `tree.rs` — store every node field as a separate `TreeBuf<T>` column indexed by node id. A `Node` struct is an explicit anti-pattern (breaks zero-copy serialization in Phase 2).

### Numerical-order discipline (the 1e-5 contract)
**Source:** `predict.cc:231-305` + `postprocessor.cc:34-37` + `xgboost.cc:52-60`.
**Apply to:** `treelite-gtil/src/lib.rs` and `postprocessor.rs`. Three rules: (1) sum trees **serial in tree_id** (no reordering — float add non-associative); (2) base_score transform in **f64**; (3) predict accumulator types mirror upstream exactly (f32 acc, f64 base_scores added in, f32 sigmoid_alpha).

---

## No Analog Found

| File | Role | Reason |
|------|------|--------|
| `Cargo.toml` (workspace root + per-crate) | config | Cargo/Rust-specific; no C++ analog. Use RESEARCH §Recommended Project Structure + the pinned `[workspace.dependencies]` block (RESEARCH lines 117-132) directly. |
| `fixtures/capture_golden.py` | utility | One-time Python wheel script; no C++ analog. Use RESEARCH §Code Examples capture script (lines 476-493) verbatim; per Open Question #1, run `help(treelite.gtil.predict)` first to confirm the 4.7.0 keyword (`pred_margin=` vs `predict_type=`). |
| `treelite-harness/*` | test | The *role* maps to upstream `tests/cpp/` GTest harness, but the golden source (the wheel, D-06) and the 1e-5 instrument design are Phase-1-specific. Use RESEARCH §Equivalence-Harness Design (lines 536-541). |

## Metadata

**Analog search scope:** `treelite-mainline/src/enum/`, `treelite-mainline/include/treelite/{tree.h, contiguous_array.h}`, `treelite-mainline/src/model_loader/detail/xgboost{.h,.cc}` + `xgboost_json/delegated_handler.cc`, `treelite-mainline/src/gtil/{predict.cc, postprocessor.cc}`.
**Files read this session:** 4 enum `.cc` + `tree.h` (two ranges) + `contiguous_array.h` + `xgboost.{h,cc}` + `delegated_handler.cc` (three ranges) + `predict.cc` (two ranges) + `postprocessor.cc` = 10 distinct C++ sources.
**Pattern extraction date:** 2026-06-10
**Note:** RESEARCH.md already cites every line range used here; all excerpts above were re-read directly from the vendored source this session to confirm exactness.
