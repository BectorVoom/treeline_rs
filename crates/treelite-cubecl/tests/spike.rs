//! Wave 1 cubecl descent spike (plan 06-02) — D-04 CONFIRMATION.
//!
//! This turns the RED Wave-0 `spike.rs` scaffold green. It is a CONFIRMATION
//! step, NOT a go/no-go gate: the user has repeatedly validated that cubecl
//! reproduces scalar precision in-kernel including f64 and mixed precision
//! (memory `cubecl-precision-validated`), so these asserts are EXPECTED to pass
//! and carry NO scalar-engine hedge branch keyed on a 1e-5 miss
//! (D-04). Per the registration-not-refactor design (D-11) the
//! [`treelite_cubecl::kernels::traversal::descend`] helper exercised here is the
//! verbatim shape Wave 3 reuses.
//!
//! What the spike locks down (the four cubecl API-surface assumptions):
//! - A1 (`exp2`): RESOLVED via the `exp(x*ln2)` IDENTITY, NOT a direct `exp2`.
//!   The 06-01 scaffold pinned `Float::exp2` from `cubecl-core typemap.rs:680`,
//!   but that method lives on the DynamicScalar *runtime* path; the cube
//!   *frontend* expandable-intrinsic set (`frontend/operation/unary.rs`)
//!   exposes `Exp` but has NO `Exp2`, so `F::exp2(x)` does NOT resolve for a
//!   generic `F: Float` inside `#[cube]` (`no function exp2 found for type
//!   parameter F`). `exponential_standard_ratio` therefore uses the exact
//!   algebraic identity `exp2(x) == exp(x * ln(2))` computed in the element's
//!   own width `F` (`F::exp` IS a cube-frontend intrinsic). Verified within
//!   1e-5 against the `exponential_standard_ratio`/`_f64` scalar twins on f32
//!   AND f64. This is the spike's exact A1 lock-down (RESEARCH Pitfall 2 / D-04).
//! - A2 (mixed-width f32/f64 locals): RESOLVED — the f64 default kernel and the
//!   `softmax_f64` micro-kernel both run f64 cells with f32 `max_margin`/`t`/
//!   divisor locals in ONE kernel on `CpuRuntime` and match the scalar twins
//!   within 1e-5.
//! - A3 (upload entry): RESOLVED — `client.create_from_slice(&[u8])` is the
//!   upload entry (cubecl-runtime client.rs:287); `client.read_one_unchecked`
//!   reads back. (The owned `client.create(Bytes)` at :345 is the alternative.)
//! - A4 (import paths): RESOLVED — `cubecl::prelude::*` for the kernel surface,
//!   `cubecl::{CubeCount, CubeDim, Runtime}`, and `cubecl::cpu::CpuRuntime` for
//!   the CPU backend client.
//!
//! All four assumptions A1-A4 are retired by the passing asserts below.

use approx::assert_abs_diff_eq;
use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime};

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_cubecl::kernels::traversal::descend;
use treelite_gtil::{Config, predict};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// 2-tree numerical `default`-kind kernel: one unit per row (`ABSOLUTE_POS`),
/// serial `for tree_id` accumulation (GTIL-08 — NO tree-axis reduction), the
/// f64 base-score add, and the `identity` postprocessor (a no-op, so the output
/// is the raw margin sum + base score — exactly `predict` with `postprocessor =
/// "identity"`).
///
/// Mirrors `predict_preset` (`lib.rs:643-741`) for the scalar binary
/// `(num_row, 1, 1)` shape: every tree routes into the single output cell `row`.
/// The forest columns are the ragged-SoA concatenation; `node_off[t]` is the
/// prefix-sum base of tree `t`'s nodes, so [`descend`] addresses
/// `concat[node_off[t] + nid]`.
#[cube(launch)]
fn predict_default_2tree<F: Float>(
    cleft: &Array<i32>,
    cright: &Array<i32>,
    split_index: &Array<i32>,
    threshold: &Array<F>,
    leaf_value: &Array<F>,
    default_left: &Array<u32>,
    node_off: &Array<u32>,
    input: &Array<F>,
    output: &mut Array<F>,
    base_score: &Array<F>,
    num_row: u32,
    num_tree: u32,
    num_feature: u32,
) {
    // ABSOLUTE_POS is `usize` in cubecl 0.10.0; cast to u32 so all index math
    // (and the `descend` u32 offset params) stays in one width.
    let row = ABSOLUTE_POS as u32;
    if row < num_row {
        let row_off = row * num_feature;
        // Serial tree accumulation in tree_id order — do NOT parallelize.
        let mut acc = F::new(0.0);
        for tree_id in 0..num_tree {
            let base = node_off[tree_id as usize];
            let leaf = descend::<F>(
                cleft,
                cright,
                split_index,
                threshold,
                default_left,
                base,
                row_off,
                input,
            );
            acc += leaf_value[(base + leaf) as usize];
        }
        // Base-score add (a single (target,class)=(0,0) cell, uploaded as a
        // 1-element Array to sidestep the Float-scalar launch ambiguity) then
        // the identity postprocessor (a no-op).
        output[row as usize] = acc + base_score[0];
    }
}

/// Standalone `exponential_standard_ratio` micro-kernel: `exp2(-v / ratio_c)`
/// per element (`postprocessor.cc:44-47`, base-2).
///
/// A1 RESOLUTION (the spike's whole point, RESEARCH Pitfall 2): on cubecl
/// 0.10.0 there is NO cube-frontend `exp2` for a generic `F: Float` — `exp2`
/// lives only on the `typemap.rs` DynamicScalar runtime path, not the
/// `frontend/operation/unary.rs` expandable-intrinsic set (which has `Exp` but
/// no `Exp2`). So `F::exp2(x)` fails to resolve inside `#[cube]`
/// (`no function exp2 found for type parameter F`). The chosen form is the
/// exact algebraic identity `exp2(x) == exp(x * ln(2))` computed in the
/// element's own width `F` — `F::exp(x * ln2)` where `ln2` is a 1-element
/// uploaded `Array<F>` carrying `ln(2)` in `F`. `F::exp(...)` IS a
/// cube-frontend intrinsic (the `Exp` trait). `ratio_c` rides as a 1-element
/// `Array<F>` (an f32 model field cast to `F` at the divide site).
#[cube(launch)]
fn exp_standard_ratio_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    ratio_c: &Array<F>,
    ln2: &Array<F>,
    n: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        // exp2(-v / ratio_c) == exp((-v / ratio_c) * ln2). `F::exp` is the
        // expandable intrinsic (associated fn, NOT the method `.exp()`).
        let x = -input[i as usize] / ratio_c[0];
        output[i as usize] = F::exp(x * ln2[0]);
    }
}

/// Standalone `softmax_f64` micro-kernel over ONE `n`-class row, launched with a
/// single unit. Reproduces `postprocessor::softmax_f64`'s EXACT mixed f32/f64
/// cast order (the 1e-5 contract, CR-01 / Pitfall 3): cells stay f64, while
/// `max_margin`, `t`, and the final divisor are f32. Confirms A2 — f64 cells +
/// f32 locals coexist in one cubecl kernel on `CpuRuntime`.
#[cube(launch)]
fn softmax_f64_kernel(row: &mut Array<f64>, n: u32) {
    if ABSOLUTE_POS == 0 {
        // float max_margin = row[0];  (f64 -> f32 narrow)
        let mut max_margin: f32 = f32::cast_from(row[0]);
        // if (row[i] > max_margin) max_margin = row[i];
        // Comparison promotes max_margin (f32) to f64; assignment narrows to f32.
        let mut i: u32 = 1;
        while i < n {
            if row[i as usize] > f64::cast_from(max_margin) {
                max_margin = f32::cast_from(row[i as usize]);
            }
            i += 1;
        }
        let mut norm_const: f64 = 0.0;
        let mut j: u32 = 0;
        while j < n {
            // t = exp(row[j] - max_margin): f64 - f32 -> f64, exp in f64, narrow
            // the result to the f32 t.
            let t: f32 = f32::cast_from(f64::exp(row[j as usize] - f64::cast_from(max_margin)));
            norm_const += f64::cast_from(t); // norm_const (f64) += t (f32 promotes)
            row[j as usize] = f64::cast_from(t); // row[j] (f64) = t (f32 promotes)
            j += 1;
        }
        // static_cast<float>(norm_const), then f64 /= f32 (promoted back to f64).
        let divisor: f64 = f64::cast_from(f32::cast_from(norm_const));
        let mut k: u32 = 0;
        while k < n {
            row[k as usize] /= divisor;
            k += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Hand-built 2-tree forests (mirror tests/predict.rs split_tree shape)
// ---------------------------------------------------------------------------

/// Single-split `Tree<T>`: node 0 numerical `kLT` test on `feature`,
/// default-left; node 1 = leaf `left_leaf`; node 2 = leaf `right_leaf`.
fn split_tree<T: Copy + Default>(
    feature: i32,
    threshold: T,
    left_leaf: T,
    right_leaf: T,
) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![feature, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![threshold, T::default(), T::default()]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![T::default(), left_leaf, right_leaf]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    t
}

/// Wrap trees into a binary scalar `(num_row, 1, 1)` `Model` with the given
/// postprocessor + base score (mirrors `tests/predict.rs::model_of`).
fn model_of<T, F>(trees: Vec<Tree<T>>, wrap: F, postprocessor: &str, base_score: f64) -> Model
where
    T: Copy,
    F: Fn(ModelPreset<T>) -> ModelVariant,
{
    let num_tree = trees.len();
    let mut m = Model::new(wrap(ModelPreset::new(trees)));
    m.num_feature = 2;
    m.num_target = 1;
    m.num_class = vec![1];
    m.leaf_vector_shape = vec![1, 1];
    m.target_id = vec![0; num_tree];
    m.class_id = vec![0; num_tree];
    m.postprocessor = postprocessor.to_string();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![base_score];
    m
}

/// Flatten a 2-tree forest into the ragged-SoA columns the kernel uploads.
fn soa_columns<T: Copy>(
    trees: &[Tree<T>],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<T>, Vec<T>, Vec<u32>, Vec<u32>) {
    let mut cleft = Vec::new();
    let mut cright = Vec::new();
    let mut split_index = Vec::new();
    let mut threshold = Vec::new();
    let mut leaf_value = Vec::new();
    let mut default_left = Vec::new();
    let mut node_off = Vec::with_capacity(trees.len());
    let mut off: u32 = 0;
    for t in trees {
        node_off.push(off);
        cleft.extend_from_slice(t.cleft.as_slice());
        cright.extend_from_slice(t.cright.as_slice());
        split_index.extend_from_slice(t.split_index.as_slice());
        threshold.extend_from_slice(t.threshold.as_slice());
        leaf_value.extend_from_slice(t.leaf_value.as_slice());
        default_left.extend(t.default_left.as_slice().iter().map(|&b| b as u32));
        off += t.cleft.as_slice().len() as u32;
    }
    (
        cleft,
        cright,
        split_index,
        threshold,
        leaf_value,
        default_left,
        node_off,
    )
}

// ---------------------------------------------------------------------------
// Spike: 2-tree default kernel (f32 AND f64) vs treelite_gtil::predict
// ---------------------------------------------------------------------------

#[test]
fn spike_default_2tree_f32_and_f64_descend() {
    // Two single-split trees on features {0, 1}. Threshold 0.5; base score 0.5.
    //   tree0: feature0 < 0.5 ? +1.0 : -1.0
    //   tree1: feature1 < 0.5 ? +2.0 : -3.0
    // Rows exercise both branches of both trees.
    //   row0 = [0.0, 0.0] -> +1.0 + 2.0 + 0.5 = 3.5
    //   row1 = [1.0, 1.0] -> -1.0 - 3.0 + 0.5 = -3.5
    //   row2 = [0.0, 1.0] -> +1.0 - 3.0 + 0.5 = -1.5
    let client = CpuRuntime::client(&Default::default());

    // ---- f32 path (retires the f32 leg) ----
    {
        let trees = vec![
            split_tree::<f32>(0, 0.5, 1.0, -1.0),
            split_tree::<f32>(1, 0.5, 2.0, -3.0),
        ];
        let (cl, cr, si, th, lv, dl, no) = soa_columns(&trees);
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let num_row = 3usize;
        let model = model_of(
            trees,
            ModelVariant::F32,
            "identity",
            0.5,
        );
        let expected = predict::<f32>(&model, &data, num_row, &Config::default()).unwrap();

        let got = launch_default::<f32>(&client, &cl, &cr, &si, &th, &lv, &dl, &no, &data, 0.5, num_row, 2);
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
        }
    }

    // ---- f64 path (retires A2 — f64 in-kernel) ----
    {
        let trees = vec![
            split_tree::<f64>(0, 0.5, 1.0, -1.0),
            split_tree::<f64>(1, 0.5, 2.0, -3.0),
        ];
        let (cl, cr, si, th, lv, dl, no) = soa_columns(&trees);
        let data: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let num_row = 3usize;
        let model = model_of(
            trees,
            ModelVariant::F64,
            "identity",
            0.5,
        );
        let expected = predict::<f64>(&model, &data, num_row, &Config::default()).unwrap();

        let got = launch_default::<f64>(&client, &cl, &cr, &si, &th, &lv, &dl, &no, &data, 0.5, num_row, 2);
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
        }
    }
}

/// Upload the SoA columns, launch `predict_default_2tree::<F>`, read back.
///
/// `u32` scalars are passed to the generated `launch` as plain Rust values (the
/// `#[cube(launch)]` macro wraps them); arrays as `ArrayArg::from_raw_parts(
/// handle, len)`. The `F`-width `base_score` rides as a 1-element `Array<F>`
/// (sidestepping the Float-scalar launch ambiguity).
#[allow(clippy::too_many_arguments)]
fn launch_default<F: Float + CubeElement + bytemuck::Pod>(
    client: &cubecl::client::ComputeClient<CpuRuntime>,
    cleft: &[i32],
    cright: &[i32],
    split_index: &[i32],
    threshold: &[F],
    leaf_value: &[F],
    default_left: &[u32],
    node_off: &[u32],
    input: &[F],
    base_score: F,
    num_row: usize,
    num_tree: u32,
) -> Vec<F> {
    let num_feature = input.len() / num_row;
    let zero_out = vec![F::from_int(0); num_row];
    let base_vec = vec![base_score];
    let h_base = client.create_from_slice(bytemuck::cast_slice(&base_vec));
    let h_cleft = client.create_from_slice(bytemuck::cast_slice(cleft));
    let h_cright = client.create_from_slice(bytemuck::cast_slice(cright));
    let h_si = client.create_from_slice(bytemuck::cast_slice(split_index));
    let h_th = client.create_from_slice(bytemuck::cast_slice(threshold));
    let h_lv = client.create_from_slice(bytemuck::cast_slice(leaf_value));
    let h_dl = client.create_from_slice(bytemuck::cast_slice(default_left));
    let h_no = client.create_from_slice(bytemuck::cast_slice(node_off));
    let h_in = client.create_from_slice(bytemuck::cast_slice(input));
    let h_out = client.create_from_slice(bytemuck::cast_slice(&zero_out));

    let cube_dim = CubeDim::new_1d(256);
    let blocks = num_row.div_ceil(256) as u32;
    let cube_count = CubeCount::Static(blocks.max(1), 1, 1);

    predict_default_2tree::launch::<F, CpuRuntime>(
        client,
        cube_count,
        cube_dim,
        unsafe { ArrayArg::from_raw_parts(h_cleft.clone(), cleft.len()) },
        unsafe { ArrayArg::from_raw_parts(h_cright.clone(), cright.len()) },
        unsafe { ArrayArg::from_raw_parts(h_si.clone(), split_index.len()) },
        unsafe { ArrayArg::from_raw_parts(h_th.clone(), threshold.len()) },
        unsafe { ArrayArg::from_raw_parts(h_lv.clone(), leaf_value.len()) },
        unsafe { ArrayArg::from_raw_parts(h_dl.clone(), default_left.len()) },
        unsafe { ArrayArg::from_raw_parts(h_no.clone(), node_off.len()) },
        unsafe { ArrayArg::from_raw_parts(h_in.clone(), input.len()) },
        unsafe { ArrayArg::from_raw_parts(h_out.clone(), num_row) },
        unsafe { ArrayArg::from_raw_parts(h_base.clone(), base_vec.len()) },
        num_row as u32,
        num_tree,
        num_feature as u32,
    );

    let bytes = client.read_one_unchecked(h_out);
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

// ---------------------------------------------------------------------------
// Spike: exponential_standard_ratio micro-kernel vs scalar twin (A1)
// ---------------------------------------------------------------------------

#[test]
fn spike_exp_standard_ratio_matches_scalar_twin() {
    use treelite_gtil::postprocessor::{exponential_standard_ratio, exponential_standard_ratio_f64};
    let client = CpuRuntime::client(&Default::default());
    let ratio_c: f32 = 2.5;

    // f32 leg
    {
        let input: Vec<f32> = vec![-1.5, 0.0, 0.75, 3.2];
        let ratio_vec = vec![ratio_c];
        let ln2_vec = vec![core::f32::consts::LN_2];
        let h_in = client.create_from_slice(bytemuck::cast_slice(&input));
        let h_out = client.create_from_slice(bytemuck::cast_slice(&vec![0.0f32; input.len()]));
        let h_rc = client.create_from_slice(bytemuck::cast_slice(&ratio_vec));
        let h_ln2 = client.create_from_slice(bytemuck::cast_slice(&ln2_vec));
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static(input.len().div_ceil(256).max(1) as u32, 1, 1);
        exp_standard_ratio_kernel::launch::<f32, CpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            unsafe { ArrayArg::from_raw_parts(h_in.clone(), input.len()) },
            unsafe { ArrayArg::from_raw_parts(h_out.clone(), input.len()) },
            unsafe { ArrayArg::from_raw_parts(h_rc.clone(), ratio_vec.len()) },
            unsafe { ArrayArg::from_raw_parts(h_ln2.clone(), ln2_vec.len()) },
            input.len() as u32,
        );
        let bytes = client.read_one_unchecked(h_out);
        let got = bytemuck::cast_slice::<u8, f32>(&bytes);
        for (g, v) in got.iter().zip(input.iter()) {
            let e = exponential_standard_ratio(ratio_c, *v);
            assert_abs_diff_eq!(*g, e, epsilon = 1e-5);
        }
    }

    // f64 leg (the ratio_c stays an f32 field cast at the divide site)
    {
        let input: Vec<f64> = vec![-1.5, 0.0, 0.75, 3.2];
        let ratio_vec = vec![ratio_c as f64];
        let ln2_vec = vec![core::f64::consts::LN_2];
        let h_in = client.create_from_slice(bytemuck::cast_slice(&input));
        let h_out = client.create_from_slice(bytemuck::cast_slice(&vec![0.0f64; input.len()]));
        let h_rc = client.create_from_slice(bytemuck::cast_slice(&ratio_vec));
        let h_ln2 = client.create_from_slice(bytemuck::cast_slice(&ln2_vec));
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static(input.len().div_ceil(256).max(1) as u32, 1, 1);
        exp_standard_ratio_kernel::launch::<f64, CpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            unsafe { ArrayArg::from_raw_parts(h_in.clone(), input.len()) },
            unsafe { ArrayArg::from_raw_parts(h_out.clone(), input.len()) },
            unsafe { ArrayArg::from_raw_parts(h_rc.clone(), ratio_vec.len()) },
            unsafe { ArrayArg::from_raw_parts(h_ln2.clone(), ln2_vec.len()) },
            input.len() as u32,
        );
        let bytes = client.read_one_unchecked(h_out);
        let got = bytemuck::cast_slice::<u8, f64>(&bytes);
        for (g, v) in got.iter().zip(input.iter()) {
            let e = exponential_standard_ratio_f64(ratio_c, *v);
            assert_abs_diff_eq!(*g, e, epsilon = 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------
// Spike: softmax_f64 micro-kernel vs scalar twin (A2 mixed f32/f64)
// ---------------------------------------------------------------------------

#[test]
fn spike_softmax_f64_matches_scalar_twin() {
    use treelite_gtil::postprocessor::softmax_f64;
    let client = CpuRuntime::client(&Default::default());

    // 3-class margins with a large spread to exercise the mixed-width cast order.
    let row: Vec<f64> = vec![12.5, -3.0, 7.25];
    let n = row.len() as u32;

    let h_row = client.create_from_slice(bytemuck::cast_slice(&row));
    let cube_dim = CubeDim::new_1d(1);
    let cube_count = CubeCount::Static(1, 1, 1);
    softmax_f64_kernel::launch::<CpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        unsafe { ArrayArg::from_raw_parts(h_row.clone(), row.len()) },
        n,
    );
    let bytes = client.read_one_unchecked(h_row);
    let got = bytemuck::cast_slice::<u8, f64>(&bytes);

    let mut expected = row.clone();
    softmax_f64(&mut expected);

    for (g, e) in got.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
    }
}
