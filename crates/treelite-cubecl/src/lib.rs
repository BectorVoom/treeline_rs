//! `treelite-cubecl` — the cubecl-accelerated GTIL kernel crate (GPU-01).
//!
//! This crate reimplements the GTIL inference hot path (numerical-dense
//! traversal + the 10 postprocessors) as `cubecl` `#[cube(launch)]` kernels,
//! defaulting to the cubecl **CPU** runtime (`features = ["cpu"]`) and
//! validated to 1e-5 against the frozen scalar golden matrix. Sparse CSR and
//! categorical splits ride the scalar [`treelite_gtil`] fallback this phase
//! (D-02).
//!
//! Wave layout (this crate is filled in across plans 06-01..06-05):
//! - Wave 0: crate scaffold, [`CubeclError`], the [`predict_cpu`] stub, and the
//!   placeholder [`upload`]/[`kernels`] modules.
//! - Wave 1: the descend spike (`tests/spike.rs`).
//! - Wave 2: per-column ragged-SoA upload ([`upload`]).
//! - Wave 3 (this plan, 06-04): the `#[cube(launch)]` kernels ([`kernels`]) for
//!   all four predict kinds + the real [`predict_cpu`] host launcher
//!   (validate → upload → select kernel by `Config.kind` → launch → read), with
//!   the categorical/sparse whole-model scalar fallback (D-02).
//! - Wave 4: determinism + the `gtil_matrix_cubecl` harness sibling.

pub mod error;

/// Per-backend `ComputeClient<R>` construction with a typed device-absence
/// skip (`CubeclError::DeviceUnavailable`, D-05). Holds the generic
/// `client::<R>()` plus the `#[cfg(feature = …)]`-gated `rocm_client`/
/// `cuda_client`/`wgpu_client` helpers (Phase-7 Wave 0).
pub mod device;

/// Per-column ragged-SoA host→device upload (Wave 2).
pub mod upload;

/// The `#[cube(launch)]` traversal + postprocessor kernels (Wave 3).
pub mod kernels;

pub use error::CubeclError;

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime};

use treelite_core::{Model, ModelVariant};
use treelite_gtil::{Config, PredictKind, postprocessor};

/// The output/input element of a cubecl prediction (`f32` / `f64`, D-05).
///
/// Bundles the `cubecl` element/`bytemuck` bounds the launcher needs plus the
/// per-element postprocessor dispatch that mirrors `treelite_gtil`'s private
/// `apply_postprocessor_{f32,f64}` (same public `postprocessor::*` functions, so
/// the result is byte-identical to the scalar reference). The output element
/// EQUALS the input element regardless of the model preset (Pitfall 6).
pub trait PredictCpuElem: Float + CubeElement + bytemuck::Pod + treelite_gtil::PredictOut {
    /// Apply the model's named postprocessor over the `(num_row, num_target,
    /// max_num_class)` `output` buffer in THIS element's precision, replicating
    /// `treelite_gtil::apply_postprocessor` arm-for-arm. `num_class` is the
    /// per-target class count; cells are laid out row-major
    /// `row * (num_target * max_num_class) + t * max_num_class + c`.
    fn apply_postprocessor(
        model: &Model,
        output: &mut [Self],
        num_row: usize,
        num_target: usize,
        max_num_class: usize,
        num_class: &[i32],
    ) -> Result<(), CubeclError>;
}

/// Per-`(row, target)` row span `[start, start + n)` for the softmax /
/// multiclass-ova row postprocessors, mirroring `apply_postprocessor`'s
/// `shape.idx(r, t, 0)` + `num_class_of(t)` spans.
#[inline]
fn row_span(
    r: usize,
    t: usize,
    num_target: usize,
    max_num_class: usize,
    num_class: &[i32],
) -> Option<(usize, usize)> {
    let n = if t >= num_class.len() {
        0
    } else {
        num_class[t].max(0) as usize
    };
    if n == 0 {
        return None;
    }
    let start = r * (num_target * max_num_class) + t * max_num_class;
    Some((start, start + n))
}

impl PredictCpuElem for f32 {
    fn apply_postprocessor(
        model: &Model,
        output: &mut [f32],
        num_row: usize,
        num_target: usize,
        max_num_class: usize,
        num_class: &[i32],
    ) -> Result<(), CubeclError> {
        // Mirror apply_postprocessor_f32 (lib.rs:1209-1292) arm-for-arm.
        match model.postprocessor.as_str() {
            "identity" => {
                for v in output.iter_mut() {
                    *v = postprocessor::identity(1.0, *v);
                }
            }
            "identity_multiclass" => {
                for v in output.iter_mut() {
                    *v = postprocessor::identity_multiclass(1.0, *v);
                }
            }
            "sigmoid" => {
                for v in output.iter_mut() {
                    *v = postprocessor::sigmoid(model.sigmoid_alpha, *v);
                }
            }
            "signed_square" => {
                for v in output.iter_mut() {
                    *v = postprocessor::signed_square(*v);
                }
            }
            "hinge" => {
                for v in output.iter_mut() {
                    *v = postprocessor::hinge(*v);
                }
            }
            "exponential" => {
                for v in output.iter_mut() {
                    *v = postprocessor::exponential(*v);
                }
            }
            "exponential_standard_ratio" => {
                for v in output.iter_mut() {
                    *v = postprocessor::exponential_standard_ratio(model.ratio_c, *v);
                }
            }
            "logarithm_one_plus_exp" => {
                for v in output.iter_mut() {
                    *v = postprocessor::logarithm_one_plus_exp(*v);
                }
            }
            "softmax" => {
                for r in 0..num_row {
                    for t in 0..num_target {
                        if let Some((s, e)) = row_span(r, t, num_target, max_num_class, num_class) {
                            postprocessor::softmax(&mut output[s..e]);
                        }
                    }
                }
            }
            "multiclass_ova" => {
                for r in 0..num_row {
                    for t in 0..num_target {
                        if let Some((s, e)) = row_span(r, t, num_target, max_num_class, num_class) {
                            postprocessor::multiclass_ova(model.sigmoid_alpha, &mut output[s..e]);
                        }
                    }
                }
            }
            other => {
                return Err(CubeclError::Unsupported(format!(
                    "unsupported postprocessor: {other}"
                )));
            }
        }
        Ok(())
    }
}

impl PredictCpuElem for f64 {
    fn apply_postprocessor(
        model: &Model,
        output: &mut [f64],
        num_row: usize,
        num_target: usize,
        max_num_class: usize,
        num_class: &[i32],
    ) -> Result<(), CubeclError> {
        // Mirror apply_postprocessor_f64 (lib.rs:1306-1385) arm-for-arm.
        match model.postprocessor.as_str() {
            "identity" => {}
            "identity_multiclass" => {}
            "sigmoid" => {
                for v in output.iter_mut() {
                    *v = postprocessor::sigmoid_f64(model.sigmoid_alpha, *v);
                }
            }
            "signed_square" => {
                for v in output.iter_mut() {
                    *v = postprocessor::signed_square_f64(*v);
                }
            }
            "hinge" => {
                for v in output.iter_mut() {
                    *v = if *v > 0.0 { 1.0 } else { 0.0 };
                }
            }
            "exponential" => {
                for v in output.iter_mut() {
                    *v = postprocessor::exponential_f64(*v);
                }
            }
            "exponential_standard_ratio" => {
                for v in output.iter_mut() {
                    *v = postprocessor::exponential_standard_ratio_f64(model.ratio_c, *v);
                }
            }
            "logarithm_one_plus_exp" => {
                for v in output.iter_mut() {
                    *v = postprocessor::logarithm_one_plus_exp_f64(*v);
                }
            }
            "softmax" => {
                for r in 0..num_row {
                    for t in 0..num_target {
                        if let Some((s, e)) = row_span(r, t, num_target, max_num_class, num_class) {
                            postprocessor::softmax_f64(&mut output[s..e]);
                        }
                    }
                }
            }
            "multiclass_ova" => {
                for r in 0..num_row {
                    for t in 0..num_target {
                        if let Some((s, e)) = row_span(r, t, num_target, max_num_class, num_class) {
                            postprocessor::multiclass_ova_f64(model.sigmoid_alpha, &mut output[s..e]);
                        }
                    }
                }
            }
            other => {
                return Err(CubeclError::Unsupported(format!(
                    "unsupported postprocessor: {other}"
                )));
            }
        }
        Ok(())
    }
}

/// Whole-model scalar-fallback predicate — the SINGLE source of truth for which
/// dense models defer entirely to [`treelite_gtil::predict`] (D-02) rather than
/// running the cubecl kLT kernels.
///
/// Two gates fire:
/// 1. **Categorical** (D-02 / Open Q1): any tree with a categorical split — the
///    numerical-only kernels cannot route a categorical test.
/// 2. **Non-`kLT` operator** (CR-01): any INTERNAL node (`cleft != -1`) whose
///    comparison operator is not `kLT`. The cubecl `descend` kernel reproduces
///    ONLY `fv < threshold` (predict.cc / traversal.rs), whereas the scalar
///    reference `next_node` dispatches per node on the stored `Operator`
///    (kLT/kLE/kEQ/kGT/kGE). E.g. EVERY LightGBM numerical model emits
///    `Operator::kLE`; reaching the kLT-hardcoded kernel would mis-route every
///    `fv == threshold` tie (kLT right, kLE left). Such a model defers WHOLE to
///    the proven scalar reference for exact 1e-5 fidelity. Only internal nodes
///    are inspected, so a leaf sentinel's unset/`kNone` operator never spuriously
///    trips the gate.
///
/// WR-04: `predict_cpu` and the `gtil_matrix_cubecl` provenance gate BOTH call
/// this function, so the two cannot drift — the test observes the executed
/// routing decision instead of re-deriving it from a parallel hand-rolled copy.
pub fn model_routes_to_scalar_fallback(model: &Model) -> bool {
    fn has_non_klt_split<V: Copy>(trees: &[treelite_core::Tree<V>]) -> bool {
        trees.iter().any(|t| {
            let cleft = t.cleft.as_slice();
            let cmp = t.cmp.as_slice();
            cleft.iter().zip(cmp.iter()).any(|(&cl, &op)| {
                // Internal node only (cleft != -1); the kernel reproduces kLT only.
                cl != -1 && op != treelite_core::Operator::kLT
            })
        })
    }
    match &model.variant {
        ModelVariant::F32(p) => {
            p.trees.iter().any(|t| t.has_categorical_split) || has_non_klt_split(&p.trees)
        }
        ModelVariant::F64(p) => {
            p.trees.iter().any(|t| t.has_categorical_split) || has_non_klt_split(&p.trees)
        }
    }
}

/// Runtime-generic host launcher for cubecl GTIL prediction — the single
/// central code change of Phase 7. Generic over `R: Runtime`, it selects its
/// client via [`device::client`] so the SAME kernels run on CPU/ROCm/CUDA/wgpu.
///
/// Mirrors [`treelite_gtil::predict`]:
/// `(&Model, &[F], num_row, &Config) -> Result<Vec<F>, _>`. The output element
/// EQUALS the input element `F` (Pitfall 6) — `f32` input always yields
/// `Vec<f32>`, `f64` input always yields `Vec<f64>`, independent of the preset.
///
/// Pipeline (RESEARCH Open Q1/Q2):
/// 1. If the model routes to the scalar fallback (categorical OR non-`kLT`
///    operator, [`model_routes_to_scalar_fallback`]), defer the WHOLE model to
///    [`treelite_gtil::predict`] (the checked scalar fallback, D-02) BEFORE any
///    device op — so a fallback-routed model succeeds even on a device-less
///    backend. Sparse input has its own entry point [`predict_cpu_sparse`].
/// 2. Construct the runtime `R`'s client via [`device::client`]; a missing
///    device propagates as the typed [`CubeclError::DeviceUnavailable`] skip
///    (D-05/D-09) — NO silent CPU fallback.
/// 3. Upload the forest ([`upload::upload_forest`]), the input matrix, and the
///    routing/averaging columns; launch the kernel selected by `Config.kind`;
///    read the output back into `Vec<F>`; and (for the `Default` kind only)
///    apply the postprocessor. `upload_forest` validates `num_feature` /
///    `num_row` / `data.len()` / per-node `split_index` bounds BEFORE any device
///    write (T-06-09; no OOB device write).
pub fn predict<R: Runtime, F: PredictCpuElem>(
    model: &Model,
    data: &[F],
    num_row: usize,
    cfg: &Config,
) -> Result<Vec<F>, CubeclError>
where
    R::Device: Default,
{
    // ---- Whole-model scalar fallback gate (D-02 / Open Q1 + CR-01) ----
    // Checked BEFORE any device op so a categorical / non-kLT model never reaches
    // the numerical-only kLT kernels — AND so a fallback-routed model on a
    // device-less backend still succeeds via the scalar reference (it never
    // touches the GPU / constructs a client). The predicate is the single source
    // of truth in [`model_routes_to_scalar_fallback`]; the provenance test
    // observes THAT same function rather than a parallel hand-rolled copy (WR-04).
    if model_routes_to_scalar_fallback(model) {
        return treelite_gtil::predict::<F>(model, data, num_row, cfg)
            .map_err(|e| CubeclError::Unsupported(format!("scalar fallback: {e}")));
    }

    // Construct the selected runtime's client. A missing device crosses back as
    // the typed `CubeclError::DeviceUnavailable` skip (D-05) and PROPAGATES via
    // `?` — there is NO silent CPU fallback (D-09). The backend tag carried into
    // the skip is the runtime's type name so the caller knows which backend
    // skipped on the generic path (CpuRuntime always succeeds via the shim).
    let client = crate::device::client::<R>(std::any::type_name::<R>())?;

    match cfg.kind {
        PredictKind::Default | PredictKind::Raw => {
            launch_default_raw::<R, F>(&client, model, data, num_row, cfg)
        }
        PredictKind::LeafId => launch_leaf_id::<R, F>(&client, model, data, num_row),
        PredictKind::ScorePerTree => launch_score_per_tree::<R, F>(&client, model, data, num_row),
    }
}

/// CPU-backend GTIL prediction — a thin shim over [`predict`] pinned to
/// `CpuRuntime` (registration-not-refactor). Keeps the Phase-6 cubecl-cpu surface
/// (`treelite_harness::cubecl_cpu_case`, `gtil_matrix_cubecl.rs`) byte-identical:
/// the CPU client always constructs, so this never returns `DeviceUnavailable`.
pub fn predict_cpu<F: PredictCpuElem>(
    model: &Model,
    data: &[F],
    num_row: usize,
    cfg: &Config,
) -> Result<Vec<F>, CubeclError> {
    predict::<CpuRuntime, F>(model, data, num_row, cfg)
}

/// SPARSE-CSR entry: routed WHOLE to the scalar fallback this phase (D-02).
///
/// Sparse input always rides the checked scalar [`treelite_gtil::predict_sparse`]
/// — the cubecl kernels are dense-numerical only this phase.
pub fn predict_cpu_sparse<F>(
    model: &Model,
    csr: treelite_gtil::SparseCsr<'_, F>,
    num_row: usize,
    cfg: &Config,
) -> Result<Vec<F>, CubeclError>
where
    F: treelite_gtil::PredictOut,
{
    treelite_gtil::predict_sparse::<F>(model, csr, num_row, cfg)
        .map_err(|e| CubeclError::Unsupported(format!("scalar fallback: {e}")))
}

/// `Default`/`Raw` launch path: upload forest + routing/averaging columns, launch
/// `predict_default_raw`, read back, and (for `Default`) apply the postprocessor.
fn launch_default_raw<R: Runtime, F: PredictCpuElem>(
    client: &cubecl::client::ComputeClient<R>,
    model: &Model,
    data: &[F],
    num_row: usize,
    cfg: &Config,
) -> Result<Vec<F>, CubeclError> {
    // Output layout (mirror predict_rows lib.rs:1007-1017).
    let num_target_i = model.num_target.max(0);
    let num_target = if num_target_i == 0 { 1 } else { num_target_i } as usize;
    let max_num_class = model.num_class.iter().copied().max().unwrap_or(1).max(1) as usize;
    let cells_per_row = num_target * max_num_class;

    // Per-cell num_class (length num_target), clamped (num_class_of semantics).
    let num_class: Vec<i32> = (0..num_target)
        .map(|t| {
            if t >= model.num_class.len() {
                0
            } else {
                model.num_class[t].max(0)
            }
        })
        .collect();

    // Base-score plane (num_target, max_num_class) as f64, zero-padded.
    let mut base_scores = vec![0.0f64; cells_per_row];
    for (i, slot) in base_scores.iter_mut().enumerate() {
        if let Some(&b) = model.base_scores.get(i) {
            *slot = b;
        }
    }

    // Routing columns (length num_tree).
    let (target_id, class_id) = routing_columns(model);

    // RF average factor per cell (mirror predict_preset lib.rs:680-708), 1.0 when
    // averaging is off (a divide by 1).
    let average_factor =
        average_factor(model, num_target, max_num_class, &num_class, &target_id, &class_id);

    // Dispatch over the preset width T; the kernel is generic over (F, T).
    let mut out = match &model.variant {
        ModelVariant::F32(p) => run_default_raw::<R, F, f32>(
            client, p, model.num_feature, data, num_row, num_target, max_num_class, &num_class,
            &base_scores, &target_id, &class_id, &average_factor,
        )?,
        ModelVariant::F64(p) => run_default_raw::<R, F, f64>(
            client, p, model.num_feature, data, num_row, num_target, max_num_class, &num_class,
            &base_scores, &target_id, &class_id, &average_factor,
        )?,
    };

    // Postprocessor for Default only (Raw returns the raw margin, gtil.h:33-36).
    if cfg.kind == PredictKind::Default {
        F::apply_postprocessor(model, &mut out, num_row, num_target, max_num_class, &num_class)?;
    }
    Ok(out)
}

/// `T`-monomorphic upload + launch of `predict_default_raw`.
#[allow(clippy::too_many_arguments)]
fn run_default_raw<R: Runtime, F: PredictCpuElem, T: Float + CubeElement + bytemuck::Pod + Copy>(
    client: &cubecl::client::ComputeClient<R>,
    preset: &treelite_core::ModelPreset<T>,
    num_feature: i32,
    data: &[F],
    num_row: usize,
    num_target: usize,
    max_num_class: usize,
    num_class: &[i32],
    base_scores: &[f64],
    target_id: &[i32],
    class_id: &[i32],
    average_factor: &[f64],
) -> Result<Vec<F>, CubeclError> {
    // Validate + upload the forest columns (validate_shape runs BEFORE any device
    // write, T-06-09).
    // The default_raw broadcast leaf-vector loop reads up to
    // `num_target * max_num_class` cells from `[leaf_vector_begin, ...)`; pass
    // that span so validate_leaf_vectors rejects a span the kernel would overrun.
    let forest = upload::upload_forest::<R, T>(
        client, preset, num_feature, num_row, data.len(), num_target, max_num_class, num_class,
        target_id, class_id,
    )?;
    let num_tree = preset.trees.len();
    let cells_per_row = num_target * max_num_class;

    // Offsets + routing/averaging/base columns as their own device handles.
    let h_node_off = forest.node_off(client);
    let h_leafvec_off = forest.leafvec_off(client);
    let h_target = client.create_from_slice(bytemuck::cast_slice(target_id));
    let h_class = client.create_from_slice(bytemuck::cast_slice(class_id));
    let h_numclass = client.create_from_slice(bytemuck::cast_slice(num_class));
    let h_base = client.create_from_slice(bytemuck::cast_slice(base_scores));
    let h_avg = client.create_from_slice(bytemuck::cast_slice(average_factor));
    let h_in = client.create_from_slice(bytemuck::cast_slice(data));
    let zero_out = vec![F::from_int(0); num_row * cells_per_row];
    let h_out = client.create_from_slice(bytemuck::cast_slice(&zero_out));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::Static((num_row.div_ceil(256)).max(1) as u32, 1, 1);

    kernels::default_raw::predict_default_raw::launch::<F, T, R>(
        client,
        cube_count,
        cube_dim,
        unsafe { ArrayArg::from_raw_parts(forest.cleft.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.cright.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.split_index.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.threshold.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_value.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_vector.clone(), forest.num_leafvec_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_vector_begin.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_vector_end.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.default_left.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(h_node_off.clone(), num_tree + 1) },
        unsafe { ArrayArg::from_raw_parts(h_leafvec_off.clone(), num_tree + 1) },
        unsafe { ArrayArg::from_raw_parts(h_target.clone(), target_id.len()) },
        unsafe { ArrayArg::from_raw_parts(h_class.clone(), class_id.len()) },
        unsafe { ArrayArg::from_raw_parts(h_numclass.clone(), num_class.len()) },
        unsafe { ArrayArg::from_raw_parts(h_base.clone(), base_scores.len()) },
        unsafe { ArrayArg::from_raw_parts(h_avg.clone(), average_factor.len()) },
        unsafe { ArrayArg::from_raw_parts(h_in.clone(), data.len()) },
        unsafe { ArrayArg::from_raw_parts(h_out.clone(), zero_out.len()) },
        num_row as u32,
        num_tree as u32,
        num_target as u32,
        max_num_class as u32,
        num_feature as u32,
    );

    let bytes = client.read_one_unchecked(h_out);
    Ok(bytemuck::cast_slice::<u8, F>(&bytes).to_vec())
}

/// `LeafId` launch path: per-`(row, tree)` leaf node id, no postprocess.
fn launch_leaf_id<R: Runtime, F: PredictCpuElem>(
    client: &cubecl::client::ComputeClient<R>,
    model: &Model,
    data: &[F],
    num_row: usize,
) -> Result<Vec<F>, CubeclError> {
    match &model.variant {
        ModelVariant::F32(p) => {
            run_leaf_id::<R, F, f32>(client, p, model.num_feature, data, num_row)
        }
        ModelVariant::F64(p) => {
            run_leaf_id::<R, F, f64>(client, p, model.num_feature, data, num_row)
        }
    }
}

fn run_leaf_id<R: Runtime, F: PredictCpuElem, T: Float + CubeElement + bytemuck::Pod + Copy>(
    client: &cubecl::client::ComputeClient<R>,
    preset: &treelite_core::ModelPreset<T>,
    num_feature: i32,
    data: &[F],
    num_row: usize,
) -> Result<Vec<F>, CubeclError> {
    // LeafId reads no leaf-vector elements (it returns the leaf node id), so the
    // broadcast span is 0; the begin<=end / end<=segment_len checks still apply.
    let forest = upload::upload_forest::<R, T>(
        client, preset, num_feature, num_row, data.len(), 0, 0, &[], &[], &[],
    )?;
    let num_tree = preset.trees.len();

    let h_node_off = forest.node_off(client);
    let h_in = client.create_from_slice(bytemuck::cast_slice(data));
    let zero_out = vec![F::from_int(0); num_row * num_tree];
    let h_out = client.create_from_slice(bytemuck::cast_slice(&zero_out));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::Static((num_row.div_ceil(256)).max(1) as u32, 1, 1);

    kernels::leaf_id::predict_leaf_id::launch::<F, T, R>(
        client,
        cube_count,
        cube_dim,
        unsafe { ArrayArg::from_raw_parts(forest.cleft.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.cright.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.split_index.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.threshold.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.default_left.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(h_node_off.clone(), num_tree + 1) },
        unsafe { ArrayArg::from_raw_parts(h_in.clone(), data.len()) },
        unsafe { ArrayArg::from_raw_parts(h_out.clone(), zero_out.len()) },
        num_row as u32,
        num_tree as u32,
        num_feature as u32,
    );

    let bytes = client.read_one_unchecked(h_out);
    Ok(bytemuck::cast_slice::<u8, F>(&bytes).to_vec())
}

/// `ScorePerTree` launch path: raw per-tree leaf data, no postprocess.
fn launch_score_per_tree<R: Runtime, F: PredictCpuElem>(
    client: &cubecl::client::ComputeClient<R>,
    model: &Model,
    data: &[F],
    num_row: usize,
) -> Result<Vec<F>, CubeclError> {
    // lvs = leaf_vector_shape[0] * leaf_vector_shape[1], clamped >= 1 (mirror
    // predict_score_by_tree lib.rs:1122-1124).
    let a = model.leaf_vector_shape.first().copied().unwrap_or(1).max(0) as usize;
    let b = model.leaf_vector_shape.get(1).copied().unwrap_or(1).max(0) as usize;
    let lvs = (a * b).max(1);
    match &model.variant {
        ModelVariant::F32(p) => {
            run_score_per_tree::<R, F, f32>(client, p, model.num_feature, data, num_row, lvs)
        }
        ModelVariant::F64(p) => {
            run_score_per_tree::<R, F, f64>(client, p, model.num_feature, data, num_row, lvs)
        }
    }
}

fn run_score_per_tree<R: Runtime, F: PredictCpuElem, T: Float + CubeElement + bytemuck::Pod + Copy>(
    client: &cubecl::client::ComputeClient<R>,
    preset: &treelite_core::ModelPreset<T>,
    num_feature: i32,
    data: &[F],
    num_row: usize,
    lvs: usize,
) -> Result<Vec<F>, CubeclError> {
    // ScorePerTree reads `[leaf_vector_begin, leaf_vector_end)` per leaf (bounded
    // by the per-tree segment length, which the end<=segment_len check covers);
    // there is no (num_target, max_num_class) broadcast, so the broadcast span is 0.
    let forest = upload::upload_forest::<R, T>(
        client, preset, num_feature, num_row, data.len(), 0, 0, &[], &[], &[],
    )?;
    let num_tree = preset.trees.len();

    let h_node_off = forest.node_off(client);
    let h_leafvec_off = forest.leafvec_off(client);
    let h_in = client.create_from_slice(bytemuck::cast_slice(data));
    let zero_out = vec![F::from_int(0); num_row * num_tree * lvs];
    let h_out = client.create_from_slice(bytemuck::cast_slice(&zero_out));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::Static((num_row.div_ceil(256)).max(1) as u32, 1, 1);

    kernels::score_per_tree::predict_score_per_tree::launch::<F, T, R>(
        client,
        cube_count,
        cube_dim,
        unsafe { ArrayArg::from_raw_parts(forest.cleft.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.cright.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.split_index.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.threshold.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_value.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_vector.clone(), forest.num_leafvec_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_vector_begin.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_vector_end.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.default_left.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(h_node_off.clone(), num_tree + 1) },
        unsafe { ArrayArg::from_raw_parts(h_leafvec_off.clone(), num_tree + 1) },
        unsafe { ArrayArg::from_raw_parts(h_in.clone(), data.len()) },
        unsafe { ArrayArg::from_raw_parts(h_out.clone(), zero_out.len()) },
        num_row as u32,
        num_tree as u32,
        lvs as u32,
        num_feature as u32,
    );

    let bytes = client.read_one_unchecked(h_out);
    Ok(bytemuck::cast_slice::<u8, F>(&bytes).to_vec())
}

/// Per-tree `target_id` / `class_id` routing columns (length num_tree), read
/// defensively with the `-1` default `predict_preset` uses (`unwrap_or(-1)`).
fn routing_columns(model: &Model) -> (Vec<i32>, Vec<i32>) {
    let num_tree = match &model.variant {
        ModelVariant::F32(p) => p.trees.len(),
        ModelVariant::F64(p) => p.trees.len(),
    };
    let target_id: Vec<i32> = (0..num_tree)
        .map(|i| model.target_id.get(i).copied().unwrap_or(-1))
        .collect();
    let class_id: Vec<i32> = (0..num_tree)
        .map(|i| model.class_id.get(i).copied().unwrap_or(-1))
        .collect();
    (target_id, class_id)
}

/// Per-cell RF average divisor (mirror predict_preset lib.rs:680-708). When
/// `average_tree_output` is false every cell is `1.0` (the kernel's divide by 1
/// is a no-op). When true, each cell's factor is the number of trees routed to it
/// via the same four-way `(target_id, class_id)` branch.
fn average_factor(
    model: &Model,
    num_target: usize,
    max_num_class: usize,
    num_class: &[i32],
    target_id: &[i32],
    class_id: &[i32],
) -> Vec<f64> {
    let cells = num_target * max_num_class;
    if !model.average_tree_output {
        return vec![1.0; cells];
    }
    let nt = num_target as i32;
    let mnc = max_num_class as i32;
    let num_class_of = |t: i32| -> i32 {
        if t < 0 || t as usize >= num_class.len() {
            0
        } else {
            num_class[t as usize]
        }
    };
    let mut factor = vec![0.0f64; cells];
    let num_tree = target_id.len();
    for tree_id in 0..num_tree {
        let tid = target_id[tree_id];
        let cid = class_id[tree_id];
        if tid < 0 && cid < 0 {
            for t in 0..nt {
                for c in 0..num_class_of(t) {
                    factor[t as usize * max_num_class + c as usize] += 1.0;
                }
            }
        } else if tid < 0 {
            if cid >= 0 && cid < mnc {
                for t in 0..nt {
                    factor[t as usize * max_num_class + cid as usize] += 1.0;
                }
            }
        } else if cid < 0 {
            if tid < nt {
                for c in 0..num_class_of(tid) {
                    factor[tid as usize * max_num_class + c as usize] += 1.0;
                }
            }
        } else if tid < nt && cid < mnc {
            factor[tid as usize * max_num_class + cid as usize] += 1.0;
        }
    }
    factor
}
