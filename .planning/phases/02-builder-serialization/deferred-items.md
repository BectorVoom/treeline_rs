# Deferred Items — Phase 02 (Builder & Serialization)

Out-of-scope discoveries logged during execution. These are NOT fixed in the
plan that discovered them; they are tracked here for a future plan.

## DEF-02-01: XGBoost loader is not byte-faithful to upstream's serialized model

- **Discovered during:** Plan 02-03 (v5 serializer), Task 1.
- **Subsystem:** `treelite-xgboost` (model loader layer) — OUTSIDE Plan 03's file
  scope (`serialize/*` + `golden_v5.rs`).
- **Symptom:** `serialize(load_xgboost_json(binary_logistic.model.json))` is 643
  bytes vs the frozen upstream `fixtures/golden_v5.bin` at 951 bytes; first
  divergence at offset 131 (`attributes`).
- **Root cause (loader, not serializer):** The Phase 1 Rust XGBoost loader
  (`crates/treelite-xgboost/src/lib.rs::build_tree`, documented "leaf-vector and
  category columns stay empty") produces a structurally leaner model than upstream
  `treelite.frontend.load_xgboost_model`. Specifically the loader does NOT populate:
  1. `attributes` — Rust sets `""`; upstream stamps `"{}"`.
  2. leaf-node `split_index` — Rust sets `0`; upstream sets `-1`.
  3. `category_list_right_child` — Rust leaves empty (count 0); upstream emits a
     present-but-empty `[false; num_nodes]` column.
  4. `leaf_vector_begin` / `leaf_vector_end` / `category_list_begin` /
     `category_list_end` — Rust empty; upstream emits `[0; num_nodes]`.
  5. `sum_hess` / `sum_hess_present` and `gain` / `gain_present` — Rust empty;
     upstream parses these from the XGBoost JSON (`sum_hessian`, `loss_changes`)
     and emits real per-node values with present-flags `[true; num_nodes]`
     (the `gain` present-flag is `[true,false,false]` — gain only on internal nodes).
- **Why deferred (not auto-fixed under Rule 1/2):**
  - It is a different crate/subsystem, not caused by the serializer changes
    (SCOPE BOUNDARY: only auto-fix issues directly caused by the current task).
  - Item 5 requires NEW XGBoost-JSON parsing (`sum_hessian`, `loss_changes`),
    which is substantive loader-domain work.
  - Touching the loader risks the currently-green `1e-5` equivalence test.
- **Impact on Plan 03:** NONE for serializer correctness. The serializer is proven
  byte-perfect by the golden round-trip `serialize(deserialize(golden)) == golden`
  (951 bytes, exact). The loader-path divergence is exercised as a NON-fatal
  diagnostic in `golden_v5.rs::loader_path_divergence_diagnostic`.
- **Suggested owner:** a follow-up loader-fidelity plan in this phase (or Phase 3)
  that upgrades `treelite-xgboost::build_tree` to populate the columns above and
  the model `attributes`, then flips `loader_path_divergence_diagnostic` into a
  hard byte-equality assertion (and removes this entry).
