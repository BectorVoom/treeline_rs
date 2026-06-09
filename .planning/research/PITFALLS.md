# Pitfalls Research

**Domain:** Numerically-precise tree-ensemble inference (Treelite/GTIL) ported C++ → Rust, with cubecl GPU compute, PyO3 binding, and a 1e-5 prediction-equivalence contract
**Researched:** 2026-06-09
**Confidence:** HIGH for floating-point/traversal pitfalls (verified directly against `treelite-mainline/src/gtil/*.cc` and `model_loader/*`), MEDIUM for cubecl-specific pitfalls (verified against cubecl docs/manuals; some behavior is version-dependent and evolving)

> **How to read this file.** The 1e-5 equivalence contract is the core value of the project. Pitfalls are ordered so that everything that can silently break that contract comes first. Each pitfall is mapped to a phase/subsystem and gives early warning signs. "Bit-exactness" is not the goal — *staying inside the 1e-5 budget* is — but several pitfalls below will blow a 1e-5 budget wide open if missed, so they are treated as critical.

---

## The single most important framing

The C++ reference is **deterministic per output element regardless of thread count**. In `PredictRaw` (`src/gtil/predict.cc:231-305`), OpenMP parallelizes over **rows** (`schedule(static)`), and within each row, trees are summed **serially in `tree_id` order** into a per-element accumulator. Base scores and tree-averaging are applied afterward in fixed order. **There is no cross-thread floating-point reduction in the reference.** This is the property that makes 1e-5 (even near-bit-exact on CPU) achievable. Any Rust/cubecl design that introduces a *different summation order over trees* — e.g. a parallel tree-reduction, atomic adds across threads, or GPU warp reductions over the tree dimension — changes accumulation order and is the #1 way this port silently drifts out of tolerance. Preserve "parallelize over rows, sum trees serially per element" everywhere, including in cubecl kernels.

---

## Critical Pitfalls

### Pitfall 1: Changing the tree-summation accumulation order (CPU or GPU)

**What goes wrong:**
Leaf contributions are summed in a different order than the reference's serial `for tree_id in 0..num_tree` loop. Floating-point addition is not associative, so a parallel reduction over trees, a tiled/blocked sum, or GPU warp/atomic accumulation produces a different rounded result. For deep ensembles (hundreds–thousands of trees) the divergence accumulates well past 1e-5, especially in f32.

**Why it happens:**
The "obvious" GPU parallelization is one thread per tree with an atomic add into the output — or a parallel reduction. That is faster but reorders the sum. Developers assume "addition is addition."

**How to avoid:**
- Mirror the reference exactly: parallelize over the **row** (and target/class) dimension; each unit accumulates *all* trees serially in `tree_id` order into a local register, then writes once. This is naturally how a per-row cubecl kernel maps (one unit = one row, loop over trees inside the kernel).
- Accumulate in the model's leaf-output type (`f32` for the float preset, `f64` for the double preset) — match `static_cast<InputT>(...)` semantics in `OutputLeafValue`/`OutputLeafVector`.
- Never use `atomicAdd` across units for the tree sum. Never use `cubecl-reduce` tree-based reductions over the tree axis.
- Apply base scores and tree-averaging as **separate passes after** the tree sum (matching `predict.cc:258-304`), not folded into the accumulator.

**Warning signs:**
Equivalence passes on 5-tree toy models but fails on real 500+ tree XGBoost/LightGBM models; error grows with `num_tree`; error larger in f32 preset than f64 preset; GPU and CPU backends disagree by more than rounding.

**Phase to address:** GTIL inference kernel design (cubecl traversal phase). Bake the "row-parallel, tree-serial" contract into the kernel signature before writing any reduction.

---

### Pitfall 2: Mixed-precision postprocessors not replicated bit-for-bit

**What goes wrong:**
`softmax` in `postprocessor.cc:57-75` is deliberately mixed precision and *must* be copied verbatim:
- `float max_margin` (always f32, even in the f64 preset path the arg type is `InputT` but the local is hard-coded `float`)
- `double norm_const` (always f64 accumulation)
- `float t`
- final `row[i] /= static_cast<float>(norm_const)` — divides by an f32-narrowed sum

A "clean" Rust rewrite that does the whole softmax in the generic float type, or accumulates `norm_const` in f32, or skips the `static_cast<float>`, produces different bits and can exceed 1e-5 on confident multiclass rows.

Similarly: `sigmoid` uses `model.sigmoid_alpha` (an f32-ish parameter) and `std::exp`; `exponential_standard_ratio` uses `std::exp2(-x / ratio_c)` (note: `exp2`, not `exp`); `signed_square` uses `std::copysign(margin*margin, margin)`; `logarithm_one_plus_exp` uses `std::log1p(std::exp(x))` (NOT `ln(1+exp(x))` naively — `log1p` matters for small x).

**Why it happens:**
The C++ looks like a bug ("why is `max_margin` a float when the row is double?") and a porter "fixes" it. That changes results. The upstream behavior is the spec, quirks included.

**How to avoid:**
- Port each postprocessor literally, preserving the exact local variable types and the order of operations. Add a code comment citing the source line so future "cleanups" don't undo it.
- Use `f64` for `norm_const`; narrow with an explicit `as f32` exactly where `static_cast<float>` appears.
- Use Rust equivalents that match libm semantics: `f64::exp2`, `f64::ln_1p` (= `log1p`), `f32::copysign`. Verify these match the platform libm the golden vectors were generated with (see Pitfall 17).
- For `softmax`, replicate the two-pass max-subtraction stabilization exactly; do not substitute a different stabilization.

**Warning signs:**
Multiclass softmax outputs drift in the 6th–7th decimal; binary sigmoid models off by ~1e-6 that grows after the transform; the f64 preset matches but f32 doesn't (or vice versa).

**Phase to address:** GTIL postprocessor phase. These are small, self-contained functions — port and unit-test them against captured C++ scalar outputs *before* wiring them into the predict path.

---

### Pitfall 3: Missing-value / default-direction handling diverges (the classic XGBoost/LightGBM mismatch)

**What goes wrong:**
The reference traversal (`predict.cc:152-172`) does: if `std::isnan(fvalue)` → go to `DefaultChild` (which is `default_left ? left : right`, per `tree.h:183-184`). **NaN is the only trigger for the default path** — not "missing from the sparse row," not "zero," not infinity. A Rust port that treats absent sparse entries, zeros, or sentinel values as "missing," or that uses `is_nan()` semantics differing from C++ `std::isnan` (e.g. on a non-float comparison), produces wrong routing and therefore wrong leaves and wildly wrong predictions (not just 1e-5 — whole branches differ).

For **sparse CSR input**, the reference materializes a dense row pre-filled with `quiet_NaN()` and scatters present values in (`predict.cc:80-86`). Absent columns are therefore NaN and take the default direction. A Rust sparse path that fills absent columns with 0.0 instead of NaN will route differently.

**Why it happens:**
"Missing value" is overloaded across frameworks. Sparse formats tempt you to treat absent = 0. The NaN-only rule is subtle and not the intuitive choice.

**How to avoid:**
- Traversal branch: `if fvalue.is_nan() { node = default_child } else { numeric/categorical test }`. Nothing else triggers default.
- Sparse CSR → dense materialization MUST pre-fill with `f32::NAN` / `f64::NAN`, then scatter present values. Reuse one scratch row per thread (matching `dense_row_` per-thread allocation, `predict.cc:67`).
- `DefaultChild` = `default_left ? left_child : right_child`. Store `default_left` per node exactly as upstream (`ContiguousArray<bool> default_left_`).
- Test explicitly with NaN-containing dense rows AND sparse rows with absent features, asserting identical routing to C++.

**Warning signs:**
Dense predictions match but sparse CSR predictions don't (parity failure between the two input paths); models with many missing features fail; predictions are correct except on rows that happen to contain NaN.

**Phase to address:** GTIL traversal phase + sparse-accessor phase. Build the dense/sparse parity test (Pitfall 18) as a gate.

---

### Pitfall 4: Categorical-split semantics ported incorrectly

**What goes wrong:**
`NextNodeCategorical` (`predict.cc:127-150`) has specific rules that differ from a naive "is value in set" check:
1. A feature value is a valid category only if `fvalue >= 0` AND `|fvalue| <= max_representable_int`, where `max_representable_int = min(uint32::MAX_as_InputT, 2^mantissa_digits_as_InputT)`. Values failing this are treated as **not matched** (not as an error).
2. The category is `static_cast<uint32_t>(fvalue)` — a truncation, not a round.
3. Membership is tested against `CategoryList(nid)`.
4. The match result is then flipped by `category_list_right_child`: if true, matched → right; if false, matched → left (`predict.cc:145-149`).
Getting the `category_list_right_child` polarity backwards, rounding instead of truncating, or treating out-of-range as an error, all produce wrong routing.

Cross-framework: XGBoost, LightGBM, and sklearn encode categorical splits differently upstream; the *loaders* normalize them into Treelite's `category_list` + `category_list_right_child` representation. The GTIL traversal only sees the normalized form, so the loader-side normalization is where the framework-specific semantics live and is a separate mismatch surface (LightGBM's bitset categories, XGBoost's one-hot vs partition, sklearn's lack of native categoricals).

**Why it happens:**
The float-representability guard and the truncation rule are non-obvious. The right-child polarity flag is easy to invert. Loader normalization is framework-specific and under-documented.

**How to avoid:**
- Port `NextNodeCategorical` literally including the `max_representable_int` computation (use `f32::MANTISSA_DIGITS`/`f64::MANTISSA_DIGITS` for the `numeric_limits<InputT>::digits` term; `1u64 << digits`).
- Truncate with `as u32` (Rust float→int cast saturates, which is acceptable since the guard already excludes out-of-range, but keep the guard to match exactly).
- Preserve `category_list_right_child` polarity exactly.
- Test against fixtures: `tests/examples/` contains `sparse_categorical` / `toy_categorical_model.txt` (LightGBM categorical) — use these as equivalence anchors.

**Warning signs:**
Categorical-model predictions are systematically wrong on a subset of rows; flipping a single boolean fixes one model but breaks another (polarity confusion); large categorical feature values route unexpectedly.

**Phase to address:** GTIL traversal phase (the `NextNodeCategorical` port) AND each loader phase (normalization correctness). Test categorical models per-loader.

---

### Pitfall 5: base_score margin transform (version-gated) omitted or applied to the wrong objectives

**What goes wrong:**
For XGBoost **legacy binary** models with `major_version >= 1`, the stored `base_score` is a *transformed* (probability-space) value that must be inverted back to margin space before being added (`xgboost_legacy.cc:453-465`, `detail/xgboost.cc:52-60`). `TransformBaseScoreToMargin` applies:
- `sigmoid` objective → `ProbToMargin::Sigmoid` (logit)
- `exponential` objective → `ProbToMargin::Exponential` (log)
- otherwise → identity
If you skip this transform, or apply it to the wrong objectives, or apply it for pre-1.0 models (where the bias is already a margin), every prediction is offset by a constant in margin space — which after sigmoid/softmax is a visible, non-1e-5 error on *every row*.

Additionally, XGBoost 3.1+ stores **vector** base_score (`ParseBaseScore`, `detail/xgboost.cc:60+`), parsed with rapidjson's NaN/Inf flag; pre-3.1 stores scalar via `std::stof`. The JSON/UBJSON path and the legacy binary path differ here.

base_scores are accumulated and stored in **`double`** (`CArray2DView<double>`, `predict.cc:296`) regardless of preset, then added to the (possibly f32) output. Preserve the f64 base-score storage + add-into-InputT pattern.

**Why it happens:**
The version gate is a one-line `if` easily lost in translation; the prob→margin inversion is counterintuitive; vector vs scalar base_score is a format-version detail.

**How to avoid:**
- Port the version check (`major_version >= 1`) and `TransformBaseScoreToMargin` exactly, per loader.
- Keep base_score storage in `f64`; add to output as `output += base_score_view(...)` matching `predict.cc:301`.
- Handle both scalar (`stof`) and array (rapidjson with NaN/Inf flag) base_score forms.
- Equivalence-test a logistic XGBoost model loaded from *each* of legacy-binary, JSON, and UBJSON — they share a model but exercise different base_score code paths.

**Warning signs:**
Constant offset on all predictions; binary classifier probabilities all shifted; legacy-binary models wrong but JSON models right (or vice versa); XGBoost 3.1+ models with per-class bias wrong.

**Phase to address:** XGBoost loader phases (legacy binary, JSON, UBJSON). Cross-format equivalence test.

---

### Pitfall 6: GPU vs CPU divergence from FMA contraction and transcendental functions

**What goes wrong:**
Even with identical accumulation order, a cubecl GPU kernel can produce different bits than CPU because: (a) the GPU may contract `a*b+c` into a fused multiply-add (FMA) with a single rounding, while the CPU path does two roundings (or vice versa); (b) `exp`, `exp2`, `log1p`, division, and `1/(1+exp(-x))` are implemented by *different transcendental approximations* on GPU than in CPU libm — GPU `exp` is typically ~1-2 ULP off the CPU result. Across a postprocessor + many trees this can approach or exceed 1e-5, particularly in f32. PROJECT.md explicitly accepts that "Bit-exact GPU reproducibility" is out of scope and the 1e-5 tolerance absorbs GPU reduction-order differences — but it does NOT automatically absorb f32 transcendental divergence on large models.

**Why it happens:**
FMA contraction is compiler/backend-controlled and silent. GPU math libraries trade accuracy for speed. Tree inference is mostly comparisons (exact) but the postprocessor and base-score/averaging arithmetic are where transcendentals/divisions live.

**How to avoid:**
- Treat the **CPU cubecl backend as the equivalence-validated default** (PROJECT.md decision). Validate 1e-5 against C++ on the CPU backend in CI.
- For the GPU backend, run the equivalence harness as a *separate, looser-budget* check and document the observed max deviation per model class. If a model class exceeds 1e-5 on GPU, fall back to f64 accumulation for the postprocessor, or compute the postprocessor on CPU after a GPU tree-sum.
- Keep the tree-traversal portion (pure comparisons + integer routing) on GPU — it's exact — and consider doing only the postprocessor in f64 to recover budget.
- Avoid relying on FMA: if a backend contracts and drifts, the comparison-only traversal is unaffected; only the arithmetic passes matter.

**Warning signs:**
CPU backend passes 1e-5, GPU backend fails by 1e-6..1e-5 on sigmoid/softmax/exponential models; deviation scales with `num_tree` and is worse in f32; deviation concentrated in postprocessed outputs, not raw margins.

**Phase to address:** cubecl GPU-backend bring-up phase (after CPU-backend equivalence is green). GPU is an opt-in acceleration, so this is a follow-on, not a v1 blocker.

---

### Pitfall 7: f16 half-precision blows the 1e-5 budget

**What goes wrong:**
f16 has ~10 mantissa bits (≈3 decimal digits) and the HALF_PRECISION_CUBECL manual itself uses a `1e-3` test tolerance. The 1e-5 contract is **two to three orders of magnitude tighter than f16 can represent.** Storing thresholds, leaf values, or accumulating in f16 cannot meet 1e-5. Even bf16 (8-bit mantissa) is worse on precision. f16 is a memory/throughput optimization that is fundamentally incompatible with the equivalence contract for the accumulation/output path.

**Why it happens:**
PROJECT.md lists "optional f16 half-precision via cubecl" as a memory-efficiency technique, and the manual makes it look easy. The trap is applying it to numerically-load-bearing data.

**How to avoid:**
- Do NOT use f16/bf16 for thresholds, leaf outputs, accumulation, or the postprocessor. The equivalence-validated presets are f32 and f64 only (matching the two upstream `ModelPreset` specializations).
- If f16 is used at all, restrict it to non-output-affecting auxiliary data, or expose it only as an explicit "fast, lower-accuracy" mode that is *excluded from the 1e-5 harness* and documented as such.
- Gate f16 behind `client.properties().features.supports_type(FloatKind::F16)` (per the manual) — it's not universally supported and will silently no-op or error otherwise.

**Warning signs:**
Any equivalence test run in f16 mode failing by ~1e-3; threshold comparisons flipping (a threshold rounded in f16 changes routing); "we enabled f16 for memory and now nothing matches."

**Phase to address:** Memory-efficiency phase. Decide early that f16 is out of the equivalence path; document it as an explicit non-goal for fidelity.

---

### Pitfall 8: Threshold comparison precision and operator semantics

**What goes wrong:**
`NextNode` (`predict.cc:99-125`) compares `InputT fvalue` against `ThresholdT threshold` using one of `<, <=, ==, >, >=`. Two traps:
1. **Type of the comparison.** In the f32 preset, threshold is f32 and input is f32 — direct. But if the Rust port up-casts thresholds to f64 for "safety," a value exactly on a threshold boundary can route differently (f32 0.1 ≠ f64 0.1). Compare in the same type the reference uses (input type drives it; threshold is the preset's `ThresholdType`).
2. **`==` operator on floats.** Some splits use `Operator::kEQ`. Exact float equality is brittle but is what the reference does — replicate it exactly, do not add an epsilon.

LightGBM thresholds are parsed as `double` (`strtod`, `lightgbm.cc:194,352`); XGBoost legacy thresholds are `bst_float`=f32. The loader must preserve the source precision into the chosen preset; rounding f64 LightGBM thresholds into an f32 preset (or vice versa) shifts boundary routing.

**Why it happens:**
"Promote everything to f64 for accuracy" is a tempting but incorrect instinct; it changes which side of a threshold a borderline value falls on.

**How to avoid:**
- Match comparison types to the preset exactly. Don't promote.
- Parse LightGBM with `strtof`/`strtod` matching `TextToNumber<float>`/`<double>` (`lightgbm.cc:66-90`) — same rounding as the source.
- Replicate `kEQ` as exact `==`.
- Test rows with feature values exactly equal to thresholds.

**Warning signs:**
Occasional single-row mismatches that flip an entire subtree; mismatches only on values near split thresholds; f32-preset model mismatching when thresholds were stored/parsed as f64.

**Phase to address:** Core model phase (preset/threshold typing) + loader phases (parse precision) + GTIL traversal (`NextNode`).

---

## Moderate Pitfalls

### Pitfall 9: XGBoost legacy binary endianness and struct layout

**What goes wrong:**
The legacy loader reads packed C structs (`LearnerModelParam` with `static_assert(sizeof == 136)`, `GBTreeModelParam`, `TreeParam`, `NodeStat` with `bst_float loss_chg; bst_float sum_hess; bst_float base_weight;` and `std::int32_t reserved[31]`) directly from the byte stream via `memcpy` (`xgboost_legacy.cc:154-193`). Porting traps:
- **Field order, padding, and reserved arrays** must be read byte-exact. The `reserved[31]` padding must be consumed even though unused.
- **Endianness:** XGBoost binary is little-endian; on the (rare) big-endian host the C++ would also be wrong, but a Rust port using `from_le_bytes` is correct and portable. Do NOT use native-endian transmute (`bytemuck` on the raw struct) for on-disk data unless you guarantee LE — use explicit `u32::from_le_bytes`/`f32::from_le_bytes`.
- The `sizeof(LearnerModelParam) == 136` invariant encodes exact padding; a Rust `#[repr(C)]` struct must reproduce the same 136-byte layout (including trailing reserved fields) or the read desynchronizes.

**Why it happens:**
`bytemuck`-casting a Rust struct over the buffer is tempting (it's in the optimiser playbook) but couples correctness to host endianness and to exact Rust layout matching C++.

**How to avoid:**
- Read scalar-by-scalar with explicit little-endian decoders (`from_le_bytes`), not whole-struct transmute, for all on-disk binary formats.
- Reproduce and assert the byte counts (e.g. consume exactly 136 bytes for the learner param, including reserved fields).
- Keep a `PeekableInputStream` equivalent for the format-detection peeking (`xgboost_legacy.cc:60-151`).

**Warning signs:**
Loader works on one machine, garbage on another (endianness); fields shifted by a few bytes producing nonsense feature counts / tree counts; off-by-padding desync after the first struct.

**Phase to address:** XGBoost legacy-binary loader phase.

---

### Pitfall 10: UBJSON and JSON numeric edge cases (NaN/Inf, int vs float, type widths)

**What goes wrong:**
- XGBoost JSON parses base_score with rapidjson's `kParseNanAndInfFlag` (`detail/xgboost.cc`), meaning `NaN`/`Infinity` literals are *valid* and expected in some fields. A Rust `serde_json` parser rejects these by default → load failure or silent coercion.
- UBJSON has typed numeric markers (int8/16/32/64, float32/float64, high-precision). Reading a value as the wrong width, or promoting an int field to float, changes values. Treelite's UBJSON path (`xgboost_ubjson.cc`) shares the JSON delegated handler; the binary number decoding must respect UBJSON type tags.
- `e.IsFloat()` checks in `ParseBaseScore` mean a JSON integer where a float is expected may be rejected or mishandled.

**Why it happens:**
JSON/UBJSON are assumed "just parse it" but Treelite relies on specific numeric-type handling and non-standard NaN/Inf extensions.

**How to avoid:**
- For JSON: use a parser configured to accept `NaN`/`Infinity` (serde_json has `arbitrary_precision`/feature considerations; may need a custom deserializer or `simd-json` with the right flags). Match rapidjson's NaN/Inf acceptance where the source uses it.
- For UBJSON: implement type-tag-aware decoding (don't assume f64); preserve int-vs-float distinctions.
- Test with a model whose base_score is `Infinity`/`NaN` and with integer-valued JSON fields.

**Warning signs:**
"invalid JSON" errors on real XGBoost JSON files; base_score parsed as 0 or wrong; UBJSON model loads but values are off by powers of 2 (width misread).

**Phase to address:** XGBoost JSON + UBJSON loader phases.

---

### Pitfall 11: LightGBM text-format quirks

**What goes wrong:**
LightGBM models are line-oriented `key=value` text (`lightgbm.cc:226-365`). Traps:
- Numbers parsed with `strtof`/`strtod` (`TextToNumber`) — Rust `str::parse::<f32>()` may round the last digit differently than C `strtof` for some decimal strings. Match the parse path used to generate goldens.
- `leaf_value` and `threshold` are `double`; `split_gain` is `float` (`lightgbm.cc:189-197`) — mixed precision per field; preserve it.
- Objective parsing drives the postprocessor + `sigmoid_alpha` (`lightgbm.cc:442-493`): `binary` → sigmoid with parsed alpha; `multiclassova` → `multiclass_ova` with alpha; `multiclass` → softmax; `average_output` flag (`lightgbm.cc:289`) toggles tree averaging. Missing the `sigmoid=` parameter parse changes the transform steepness.
- `class_id[i] = i % num_class_` (`lightgbm.cc:428`) — the round-robin class assignment for multiclass must be replicated.
- Empty `threshold`/`split_gain` for single-leaf trees (`lightgbm.cc:347,357`) — handle the degenerate tree.

**Why it happens:**
Text parsing feels trivial; the per-field precision and objective→postprocessor mapping are easy to under-specify.

**How to avoid:**
- Replicate per-field types (leaf_value/threshold = f64, split_gain = f32).
- Replicate the full objective→postprocessor+alpha mapping and `average_output` handling.
- Replicate `class_id = i % num_class` round-robin.
- Use the `tests/examples/` LightGBM fixtures for equivalence.

**Warning signs:**
LightGBM binary classifier off by a steepness factor (alpha missed); multiclass classes permuted (round-robin wrong); single-leaf-tree models crash or misparse.

**Phase to address:** LightGBM loader phase.

---

### Pitfall 12: scikit-learn float precision and the bulk-construction path

**What goes wrong:**
sklearn estimators expose tree arrays (thresholds, values) as numpy `float64`. Two traps:
- Precision: sklearn values are f64; deciding the Treelite preset (f32 vs f64) determines whether they're narrowed. For 1e-5 equivalence against sklearn-derived goldens, prefer the f64 preset for sklearn unless the reference narrows.
- The **bulk path** (`sklearn_bulk.cc`, flagged in ARCHITECTURE.md anti-patterns) calls `BulkConstructTree` directly, bypassing `ModelBuilder` validation (orphan/topology checks). A Rust port replicating this must either replicate the bypass faithfully *or* run validation — but adding validation that upstream skips could reject models upstream accepts (and vice versa). Match upstream behavior.
- sklearn has no native categorical splits; do not invent categorical handling for it.

**Why it happens:**
The dual construction path (validated builder vs bulk) is a hidden architectural fork; precision choice is silent.

**How to avoid:**
- Use f64 preset for sklearn (matches source dtype) unless goldens show otherwise.
- Decide deliberately whether the Rust bulk path validates; document it.
- Test with sklearn RandomForest/GradientBoosting fixtures via the equivalence harness (sklearn isn't bit-frozen — see Pitfall 17, you need a pinned sklearn version to regenerate goldens).

**Phase to address:** sklearn loader phase + model-builder phase (bulk vs validated decision).

---

### Pitfall 13: cubecl launch overhead makes GPU slower for small models

**What goes wrong:**
Tree inference per row is cheap (a handful of comparisons per tree). For small models / small batches, kernel launch + host↔device transfer dominates and the GPU is *slower* than the CPU backend, sometimes by orders of magnitude. Shipping "GPU = fast" as a blanket claim is wrong, and forcing GPU for tiny predict calls hurts.

**Why it happens:**
GPU acceleration is assumed universally beneficial. Tree inference is branch-heavy and low-arithmetic-intensity — not the GPU's strength.

**How to avoid:**
- Default to the CPU backend (already the PROJECT.md decision); make GPU opt-in.
- Add a heuristic threshold (rows × trees) below which CPU is used even when GPU is requested, or at least document the crossover.
- Benchmark the crossover point; don't transfer data to GPU for sub-threshold batches.
- Batch rows to amortize launch overhead when GPU is used.

**Warning signs:**
GPU-enabled predict slower than CPU on benchmarks for small inputs; profiling shows time dominated by transfer/launch, not compute.

**Phase to address:** cubecl GPU-backend phase + performance benchmarking phase.

---

### Pitfall 14: cubecl kernel non-determinism and branch-heavy traversal mapping

**What goes wrong:**
- Tree traversal is data-dependent control flow (`while !is_leaf`). On GPU/SIMT, divergent branches within a warp serialize and, more importantly, any design that uses shared-memory reductions or atomics introduces nondeterministic ordering. The reference is deterministic; reproduce determinism on the CPU backend (the validated one).
- cubecl's CPU backend is "still evolving and not fully optimized across every operator" (per cubecl release notes) and some operations are `not yet implemented` (e.g. issue #1019 cited unimplemented shared-memory free patterns). Relying on an unimplemented op silently fails or panics at runtime.
- `continue` is not supported in `#[cube]` (per the Loop Control manual) and normal Rust functions can't be called inside `#[cube]` (must be `#[cube]` themselves). A direct port of the C++ control flow may not compile.

**Why it happens:**
cubecl is young; kernel authoring constraints differ from normal Rust; determinism is assumed rather than designed.

**How to avoid:**
- Keep the per-row kernel free of cross-unit reductions/atomics for the tree sum (Pitfall 1) — this also gives determinism.
- Restructure traversal to cubecl's control-flow constraints: `while` loop with `break`, no `continue`, helper functions annotated `#[cube]`.
- Pin a known-good cubecl version and test every op used against the CPU backend early; have a plain-Rust CPU fallback path for the inference hot path so the project isn't blocked by a cubecl gap.
- Validate determinism: run the same input twice on the CPU backend, assert bit-identical output.

**Warning signs:**
Kernel compile errors about `continue`/function calls; runtime `not yet implemented` panics; non-reproducible outputs across runs on the same backend.

**Phase to address:** cubecl traversal-kernel phase. Spike a minimal kernel first to map control-flow + determinism before full port.

---

### Pitfall 15: Zero-copy/transmutation alignment, endianness, and lifetime hazards

**What goes wrong:**
- `bytemuck::cast_slice` panics (safely) on misaligned or non-divisible slices — but the panic is a runtime failure, and feeding it a `&[u8]` from a file at an arbitrary offset (e.g. inside the XGBoost binary stream) will panic if alignment isn't 4/8-byte. On-disk data is also LE-specific; `cast_slice` does a native-endian reinterpret, so transmuting on-disk f32/u32 is wrong on big-endian and couples to layout (see Pitfall 9).
- `ContiguousArray<T>`'s `UseForeignBuffer` mode (zero-copy over a Python buffer) means the Rust type may alias memory it doesn't own. A `Tree`/`Model` holding a borrowed buffer must encode the lifetime so the buffer outlives the model — otherwise dangling reads. The C++ side relies on the caller keeping the buffer alive; Rust must make this a lifetime or `Arc` ownership invariant, not a convention.
- CubeCL `Bytes::from_bytes_vec(...)` takes an owned `Vec` (`.to_vec()`) — the "zero-copy" host→device path in the manual actually *copies* into the owned container. Don't assume zero-copy all the way to the device.

**Why it happens:**
The optimiser playbook pushes zero-copy/transmute; the lifetime and endianness coupling is easy to overlook; "zero-copy" is overclaimed.

**How to avoid:**
- Use `bytemuck` only for in-memory, alignment-guaranteed, native-endian buffers (e.g. computed arrays staged for cubecl), NOT for parsing on-disk model bytes (use explicit `from_le_bytes` there).
- Model the foreign-buffer lifetime explicitly: either `ContiguousArray<'a, T>` borrowing with a lifetime, or own via `Arc<[u8]>`/an Arrow buffer with refcount. Prefer owned/refcounted for the PyO3 boundary.
- Don't claim end-to-end zero-copy to the GPU; measure the actual copies.

**Warning signs:**
`bytemuck` alignment panics on real model files; use-after-free / segfault when the Python array is dropped before predict returns; wrong values on big-endian (CI rarely catches this).

**Phase to address:** Core model phase (`ContiguousArray` design) + memory-efficiency phase + PyO3 phase.

---

### Pitfall 16: PyO3/maturin pitfalls — GIL, buffer lifetimes, error translation, abi3, numpy return

**What goes wrong:**
- **GIL & long predict calls:** Holding the GIL during a long multi-threaded predict blocks all other Python threads. Release it with `Python::allow_threads` around the rayon/cubecl compute, but ensure no `Py*` objects are touched inside.
- **Buffer-protocol borrow lifetimes:** Accepting a numpy array zero-copy gives a borrow valid only while the GIL is held / the array is alive. If you release the GIL (above) AND hold a raw pointer into the numpy buffer, you risk the array being mutated/freed. Either copy, or keep the GIL for the borrow and release only around pure-Rust compute on owned data. This directly interacts with Pitfall 15.
- **Error translation:** `thiserror` errors in library crates must map to `PyErr` at the boundary (`impl From<TreeliteError> for PyErr` or `#[derive]` via pyo3's error conversion). A panic across the FFI boundary is UB / abort — never let a library panic propagate into Python; convert `Result` to `PyResult`.
- **abi3:** Building an abi3 wheel (stable ABI, one wheel for many Python versions) restricts you to the limited API — some PyO3 features (e.g. certain buffer/`#[pyclass]` specializations) are unavailable or behave differently. Decide abi3 vs version-specific early; maturin config differs.
- **Returning numpy zero-copy:** Returning a Rust `Vec<f32>` as numpy without copy requires `rust-numpy` (`PyArray::from_vec` transfers ownership, or `IntoPyArray`); getting this wrong copies or leaks.

**Why it happens:**
The PyO3 GIL/lifetime/error model is subtle and the zero-copy temptation collides with the GIL-release optimization.

**How to avoid:**
- Use `rust-numpy` (`numpy` crate) for array in/out; `IntoPyArray`/`PyReadonlyArray` for borrows.
- Take input as `PyReadonlyArray2<f32>`, get `.as_slice()`, and either keep the GIL for the borrow or copy into an owned buffer before `allow_threads`.
- Implement `From<LibError> for PyErr`; wrap every boundary fn to return `PyResult`; add `catch_unwind` if any panic risk remains.
- Decide abi3 up front (PROJECT.md memory/portability goals); configure maturin accordingly.
- Verify against current PyO3 docs (API churns across major versions) — use the documentation-lookup workflow for the pinned PyO3 version.

**Warning signs:**
Deadlocks / no speedup from threading (GIL held); segfault when input array is GC'd mid-call; Python sees `pyo3_runtime.PanicException` (a panic leaked); abi3 wheel fails to import on a different Python minor version.

**Phase to address:** PyO3 binding phase. Establish the error-conversion and array-borrow patterns before exposing predict.

---

### Pitfall 17: Golden-vector generation not reproducible (RNG, libm, framework versions)

**What goes wrong:**
The equivalence harness captures C++ Treelite outputs as frozen goldens (PROJECT.md decision: no live C++ in CI). If the goldens aren't regenerable bit-for-bit, you can't tell whether a future mismatch is a Rust regression or golden drift. Sources of irreproducibility:
- **Input RNG:** "random seeded input matrices" — the seed AND the exact RNG algorithm must be pinned and documented. NumPy's `default_rng(seed)` vs `RandomState(seed)` differ; Rust's `rand` differs from NumPy. The harness must generate inputs identically on both sides, or store the exact input matrices alongside goldens.
- **libm divergence:** The C++ goldens embed the platform's `exp`/`log1p`/`exp2` results. If they were generated on glibc and Rust links a different libm (or musl), the transcendental results differ by ULPs — eating into the 1e-5 budget for no real reason.
- **Framework versions:** sklearn/XGBoost/LightGBM model outputs change across versions. The exact training framework versions must be pinned and recorded so models (and thus goldens) are regenerable.
- **f32 vs f64 goldens:** Capture goldens for *both* presets; a single-precision golden compared against a double-preset Rust run (or vice versa) is a false mismatch.

**Why it happens:**
Reproducibility is assumed; RNG/libm/version drift is invisible until a golden needs regenerating.

**How to avoid:**
- **Store the actual input matrices** (not just a seed) as fixtures, OR pin the exact generator (algorithm + seed) and verify both sides produce identical inputs.
- Record the toolchain: C++ compiler, libm/libc, OS, framework versions (XGBoost/LightGBM/sklearn) in a manifest committed with the goldens.
- Capture goldens per preset (f32 and f64) and per predict kind (default/raw/leaf-id/per-tree).
- Set the 1e-5 tolerance with explicit headroom for libm ULP differences; if CPU equivalence is tighter than 1e-5 you have margin — track the *actual* max deviation, not just pass/fail.

**Warning signs:**
Can't reproduce a golden you generated last month; tolerance "mysteriously" needs loosening; goldens regenerated on a different machine differ; one preset passes and the other has no golden.

**Phase to address:** Equivalence-harness phase (foundational — build this early, alongside the first ported postprocessor).

---

### Pitfall 18: Sparse-CSR vs dense parity, edge models, and predict-kind coverage gaps

**What goes wrong:**
The harness tests dense but not sparse (or only "default" predict kind), so a sparse-path bug (Pitfall 3's NaN-fill) or a `kPredictRaw`/`kPredictLeafID`/`kPredictPerTree` bug ships undetected. Edge models also slip through: single-leaf "stump" trees, single-tree models, multi-target models, models with `average_tree_output`, multiclass with `grove_per_class` vs `class_id` round-robin, models with leaf *vectors* (`OutputLeafVector`, `predict.cc:174-216`) vs scalar leaves.

**Why it happens:**
"Happy path" testing on a couple of canned models; the 4 predict kinds × 2 presets × dense/sparse × leaf-scalar/leaf-vector matrix is large and easy to under-cover.

**How to avoid:**
- Make dense↔sparse parity an explicit gate: for every test model, run both DenseMatrixAccessor and SparseMatrixAccessor inputs (same logical data, sparse with explicit-missing entries) and assert identical output.
- Cover all 4 predict kinds and both presets in the harness matrix.
- Include edge fixtures: stump tree, single tree, multi-target, `average_tree_output=true`, multiclass (both LightGBM round-robin and XGBoost softprob), leaf-vector models.
- Mirror upstream's parameterized test pattern (`INSTANTIATE_TEST_SUITE_P(GTIL, ..., testing::Values("dense","sparse"))`, per TESTING.md) so coverage is structurally enforced.

**Warning signs:**
Only `predict()` (default kind) tested; no sparse fixtures; multi-target / leaf-vector models absent from the harness; coverage matrix has empty cells.

**Phase to address:** Equivalence-harness phase + each GTIL predict-kind phase.

---

## Minor Pitfalls

### Pitfall 19: Pinning all crates to "latest" causes breaking-change churn

**What goes wrong:**
PROJECT.md mandates "all crates pinned to their latest published versions." cubecl is young and pre-1.0 (0.9.x on crates.io; the optimiser manual references 0.10.0 APIs that may not be published) — minor bumps break APIs (`supports_type` vs `feature_enabled`, launch signatures). PyO3, rust-numpy, bytemuck, and the half crate also churn. Blindly tracking latest mid-project causes repeated breakage and can desync the cubecl/half/bytemuck version triple (they must be ABI-compatible).

**How to avoid:**
Pin exact versions in `Cargo.lock` and bump deliberately, not continuously. Pin cubecl + its backend crates + half + bytemuck as a coherent set (match the versions the manuals were written against, or the latest mutually-compatible set). Verify the cubecl API surface used (`supports_type`/`FloatKind`) matches the pinned version before writing kernels.

**Warning signs:** Build breaks after `cargo update`; cubecl API names don't match the manual; version-mismatch errors between cubecl and half/bytemuck.

**Phase to address:** Workspace setup phase.

---

### Pitfall 20: Rust edition 2024 ecosystem friction

**What goes wrong:**
PROJECT.md selects edition 2024 (CONCERNS.md flags it as immature). Some crates / proc-macros (cubecl's `#[cube]`, pyo3 macros) may not be fully validated against 2024 semantics, and 2024 changed some defaults (e.g. unsafe-attribute, lifetime capture rules) that can surface as new errors.

**How to avoid:**
Confirm cubecl, pyo3, and rust-numpy compile cleanly under edition 2024 at workspace setup; have edition 2021 as a documented fallback (CONCERNS.md recommendation). Don't discover this mid-port.

**Warning signs:** Macro-expansion errors only under 2024; crates documenting "2021 only."

**Phase to address:** Workspace setup phase.

---

### Pitfall 21: Tree-averaging division and integer accumulator semantics

**What goes wrong:**
`average_tree_output` divides each output by an integer `average_factor` (`predict.cc:259-293`) computed by counting contributing trees per (target, class), then `output /= static_cast<InputT>(average_factor)`. Traps: computing the factor wrong (the per-target/class counting logic has 4 branches matching target_id/class_id being -1), dividing in the wrong type, or applying averaging before base-score (order matters: averaging is applied to the tree sum, base score added *after* — `predict.cc:294-304`). Random forests use this; gradient boosting does not.

**How to avoid:** Replicate the 4-branch average_factor counting and the apply-order (sum → average → base_score → postprocessor) exactly. Cast the integer factor to InputT for the division as upstream does.

**Warning signs:** RandomForest predictions off by a constant factor (≈ num_trees); GBM unaffected.

**Phase to address:** GTIL predict phase.

---

### Pitfall 22: Allocator (jemalloc/mimalloc) under Python / PyO3

**What goes wrong:**
Setting a global allocator (`#[global_allocator]` jemalloc/mimalloc) in a `cdylib` loaded by Python can conflict with Python's own allocator, cause issues on some platforms (jemalloc on macOS/musl is finicky), or interact badly with the GIL/fork model. A custom allocator in the extension module overrides allocation for the whole process once loaded.

**How to avoid:** Be cautious applying a global allocator in the PyO3 `cdylib`; consider scoping the allocator to the Rust-internal hot paths or benchmarking whether it actually helps tree inference (which is mostly arena-style contiguous allocation). Test wheel import on Linux/macOS. Don't enable jemalloc on musl without validation.

**Warning signs:** Wheel imports fine on dev machine, crashes/leaks on CI or another OS; memory not actually reduced.

**Phase to address:** Memory-efficiency phase + PyO3 phase.

---

### Pitfall 23: Denormals / flush-to-zero divergence

**What goes wrong:**
If a backend (GPU, or CPU with FTZ/DAZ flags set) flushes denormals to zero while the C++ reference does not, near-zero margins/probabilities can diverge. Tree leaf outputs are rarely denormal, but `exp(very negative)` in sigmoid/softmax can produce subnormals. GPU defaults often flush denormals.

**How to avoid:** Don't enable FTZ/DAZ on the CPU path. For GPU, note that denormal flushing is part of the GPU-vs-CPU divergence budget (Pitfall 6) — validate, don't assume. Generally a minor contributor relative to transcendental ULP differences.

**Warning signs:** Tiny probabilities (≈1e-30) differ between CPU and GPU; only extreme-margin rows affected.

**Phase to address:** GPU-backend phase (validation).

---

## Phase-to-Pitfall Map (for roadmap planning)

| Phase / Subsystem | Pitfalls to gate on |
|---|---|
| Workspace setup | 19 (version pinning), 20 (edition 2024) |
| Core model (`Tree`/`ContiguousArray`/presets) | 8 (threshold typing), 15 (foreign-buffer lifetimes) |
| XGBoost legacy-binary loader | 5 (base_score transform), 9 (endianness/struct layout) |
| XGBoost JSON/UBJSON loaders | 5, 10 (NaN/Inf, numeric widths) |
| LightGBM loader | 4 (categorical normalization), 8 (parse precision), 11 (text quirks) |
| sklearn loader | 12 (f64 precision, bulk path) |
| GTIL traversal kernel | 1 (accumulation order), 3 (missing/default), 4 (categorical), 8 (comparison), 14 (cubecl control flow/determinism) |
| GTIL postprocessors | 2 (mixed precision) — port + unit-test first |
| GTIL predict orchestration | 1, 21 (averaging order), 18 (predict-kind/sparse coverage) |
| Equivalence harness | 17 (reproducible goldens) — build early, 18 (coverage matrix) |
| cubecl GPU backend (opt-in) | 6 (FMA/transcendental divergence), 13 (launch overhead), 23 (denormals) |
| Memory efficiency | 7 (f16 incompatible with 1e-5), 15 (transmute/alignment), 22 (allocator) |
| PyO3 binding | 16 (GIL/lifetime/error/abi3/numpy), 15, 22 |

## Recommended sequencing implication

1. Build the **equivalence harness skeleton + reproducible golden generation (Pitfall 17)** and port the **postprocessors with scalar unit tests (Pitfall 2)** *before* the inference engine — they are the measurement instrument for everything else.
2. Get the **CPU cubecl backend to 1e-5 (Pitfalls 1, 3, 4, 8, 21)** as the validated baseline.
3. Treat the **GPU backend, f16, and aggressive zero-copy (Pitfalls 6, 7, 13, 15, 23)** as optimizations layered on top of a green CPU equivalence — never as prerequisites to it.

## Sources

- `treelite-mainline/src/gtil/predict.cc` (traversal, missing-value, accumulation order, sparse NaN-fill, averaging, base_score) — HIGH (primary source)
- `treelite-mainline/src/gtil/postprocessor.cc` (mixed-precision softmax/sigmoid/exp2/log1p) — HIGH (primary source)
- `treelite-mainline/include/treelite/tree.h` (default_left, DefaultChild, category_list_right_child) — HIGH (primary source)
- `treelite-mainline/include/treelite/detail/threading_utils.h` (OpenMP row-parallel, static schedule) — HIGH (primary source)
- `treelite-mainline/src/model_loader/xgboost_legacy.cc`, `detail/xgboost.cc` (struct layout, base_score margin transform, objective→postprocessor) — HIGH (primary source)
- `treelite-mainline/src/model_loader/lightgbm.cc` (text parse, per-field precision, objective/alpha, round-robin class) — HIGH (primary source)
- `/home/user/Documents/workspace/optimisor/manual/HALF_PRECISION_CUBECL.md` (f16 ~1e-3 tolerance, feature gating) — HIGH (manual)
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_TRANSMUTATION_CUBECL.md` (bytemuck alignment/panic, owned-Bytes copy) — HIGH (manual)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` (control-flow constraints, reduction, CPU backend) — MEDIUM (manual)
- cubecl release/version status (0.9.x published; CPU backend evolving; unimplemented-op issues) — MEDIUM (WebSearch, GitHub releases/issue #1019)
