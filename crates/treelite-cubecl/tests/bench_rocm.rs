//! ROCm-backend benchmark for the cubecl GTIL kernels (gfx-class AMD GPU).
//!
//! `#[ignore]` + `rocm`-feature-gated (D-06/D-05): the developer runs it
//! explicitly on the AMD/ROCm box. It times `predict::<HipRuntime, _>`
//! (full per-call cost: forest+input upload, kernel launch, readback) against
//! the `CpuRuntime` baseline over synthetic forests sized to expose the two
//! optimizations under test:
//!   - descend node-index CSE  → deep trees (long descent),
//!   - comptime single-cell    → many rows, shallow forest (per-row pre/post).
//!
//! A missing HIP device surfaces as the typed `DeviceUnavailable` skip
//! (catchable, not an FFI abort — Plan-01 A3); the bench prints "skip" and the
//! test PASSES. Uses ONLY the public `predict` API, so this file compiles
//! unchanged against both the optimized HEAD and the pre-optimization source.
//!
//! Run:
//!   ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
//!   LD_LIBRARY_PATH=/opt/rocm/lib \
//!   cargo test -p treelite-cubecl --release --features rocm --test bench_rocm \
//!     -- --nocapture --ignored
#![cfg(feature = "rocm")]

use std::time::Instant;

use cubecl::Runtime;
use cubecl::cpu::CpuRuntime;
use cubecl::hip::HipRuntime;

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_cubecl::{CubeclError, predict};
use treelite_gtil::{Config, PredictKind};

/// Perfect binary tree of the given depth; internal nodes split kLT on a
/// rotating feature at 0.5, leaves carry distinct bounded values.
fn perfect_tree(depth: u32, num_feature: i32, seed: f32) -> Tree<f32> {
    let total: usize = (1usize << (depth + 1)) - 1;
    let internal: usize = (1usize << depth) - 1;

    let mut cleft = vec![-1i32; total];
    let mut cright = vec![-1i32; total];
    let mut split_index = vec![-1i32; total];
    let mut default_left = vec![false; total];
    let mut threshold = vec![0.0f32; total];
    let mut cmp = vec![Operator::kNone; total];
    let mut leaf_value = vec![0.0f32; total];
    let mut node_type = vec![TreeNodeType::kLeafNode; total];

    for i in 0..total {
        if i < internal {
            cleft[i] = (2 * i + 1) as i32;
            cright[i] = (2 * i + 2) as i32;
            split_index[i] = (i as i32) % num_feature;
            default_left[i] = true;
            threshold[i] = 0.5;
            cmp[i] = Operator::kLT;
            node_type[i] = TreeNodeType::kNumericalTestNode;
        } else {
            leaf_value[i] = seed + ((i - internal) as f32) * 1e-3;
        }
    }

    let mut t = Tree::<f32>::new();
    t.num_nodes = total as i32;
    t.cleft = TreeBuf::from_owned(cleft);
    t.cright = TreeBuf::from_owned(cright);
    t.split_index = TreeBuf::from_owned(split_index);
    t.default_left = TreeBuf::from_owned(default_left);
    t.threshold = TreeBuf::from_owned(threshold);
    t.cmp = TreeBuf::from_owned(cmp);
    t.leaf_value = TreeBuf::from_owned(leaf_value);
    t.node_type = TreeBuf::from_owned(node_type);
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0u64; total]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![0u64; total]);
    t
}

fn single_cell_model(trees: Vec<Tree<f32>>, num_feature: i32) -> Model {
    let num_tree = trees.len();
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = num_feature;
    m.num_target = 1;
    m.num_class = vec![1].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0; num_tree].into();
    m.class_id = vec![0; num_tree].into();
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.0].into();
    m
}

fn make_input(num_row: usize, num_feature: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut out = vec![0.0f32; num_row * num_feature];
    for v in out.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((state >> 40) as f32) / (1u64 << 24) as f32;
    }
    out
}

/// Time `predict::<R, f32>` over a warmed workload; returns (best_ms, output).
fn time_predict<R: Runtime>(
    model: &Model,
    data: &[f32],
    num_row: usize,
    cfg: &Config,
    iters: usize,
) -> Result<(f64, Vec<f32>), CubeclError>
where
    R::Device: Default,
{
    // Warm-up: JIT-compile the kernel + init driver/alloc pools (excluded).
    let warm = predict::<R, f32>(model, data, num_row, cfg)?;
    let mut best = f64::INFINITY;
    let mut last = warm;
    for _ in 0..iters {
        let t0 = Instant::now();
        last = predict::<R, f32>(model, data, num_row, cfg)?;
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&last);
        if dt < best {
            best = dt;
        }
    }
    Ok((best * 1e3, last))
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).fold(0.0, f64::max)
}

fn run_case(name: &str, depth: u32, num_tree: usize, num_feature: usize, num_row: usize) {
    let iters = 10usize;
    let trees: Vec<Tree<f32>> =
        (0..num_tree).map(|i| perfect_tree(depth, num_feature as i32, i as f32 * 1e-2)).collect();
    let model = single_cell_model(trees, num_feature as i32);
    let data = make_input(num_row, num_feature);
    let cfg = Config { kind: PredictKind::Raw, nthread: 0 };

    let (cpu_ms, cpu_out) =
        time_predict::<CpuRuntime>(&model, &data, num_row, &cfg, iters).expect("cpu predict");

    match time_predict::<HipRuntime>(&model, &data, num_row, &cfg, iters) {
        Ok((gpu_ms, gpu_out)) => {
            let delta = max_abs_delta(&cpu_out, &gpu_out);
            let node_visits = num_row as f64 * num_tree as f64 * (depth as f64 + 1.0);
            eprintln!("---- {name} ----");
            eprintln!(
                "  depth={depth} trees={num_tree} feat={num_feature} rows={num_row} iters={iters}"
            );
            eprintln!("  CPU best  = {cpu_ms:8.3} ms  ({:7.1} Mrows/s)", num_row as f64 / cpu_ms / 1e3);
            eprintln!("  ROCm best = {gpu_ms:8.3} ms  ({:7.1} Mrows/s)", num_row as f64 / gpu_ms / 1e3);
            eprintln!("  speedup (CPU/ROCm) = {:.2}x", cpu_ms / gpu_ms);
            eprintln!("  ROCm node-visit throughput = {:.1} M/s", node_visits / gpu_ms / 1e3);
            eprintln!("  max |delta| CPU vs ROCm = {delta:.3e}  (record-only; GPU rounding is impl-defined)");
        }
        Err(CubeclError::DeviceUnavailable { backend }) => {
            eprintln!("---- {name}: SKIP — no {backend} device available ----");
        }
        Err(e) => panic!("ROCm predict failed (not a device-absence skip): {e}"),
    }
}

#[test]
#[ignore]
fn bench_rocm_kernel() {
    eprintln!("================= ROCm BENCH =================");
    // Descent-heavy (descend CSE target): deep trees, long traversal.
    run_case("descent-heavy", 14, 64, 32, 100_000);
    // Per-row-overhead (comptime single-cell target): shallow, many rows.
    run_case("per-row-overhead", 2, 8, 16, 1_000_000);
    // Balanced realistic GBDT shape (XGBoost-ish).
    run_case("balanced-gbdt", 8, 200, 32, 200_000);
    eprintln!("=============================================");
}
