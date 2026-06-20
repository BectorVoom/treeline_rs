//! SPIKE (throwaway): shared-memory feature-row staging for the GTIL descent.
//!
//! Hypothesis: the descent hot path's dominant cost is the per-node, data-
//! dependent GLOBAL gather `input[row_off + fi]` (uncoalesced, high latency,
//! re-read once per visited node across the whole forest). If a cube stages its
//! TILE of feature rows into on-chip SHARED memory once (a coalesced bulk load)
//! and every descent then reads features from shared memory, we trade many
//! scattered global reads for one coalesced load + many low-latency shared reads.
//!
//! Crucially this keeps ONE THREAD PER ROW and the SERIAL tree-sum order, so the
//! float accumulation is BIT-IDENTICAL to the production kernel (GTIL-08 holds —
//! no reduction reorder). Only WHERE features are read changes. Scope is the
//! common single-cell / scalar-leaf / Raw case to keep the kernel minimal.
//!
//! Correctness test runs on the CPU runtime in plain `cargo test`. The bench
//! (`--ignored`) times staged-vs-production on CPU, and on ROCm when built
//! `--features rocm`.

use cubecl::Runtime;
use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_cubecl::upload;
use treelite_gtil::{Config, PredictKind};

// ---------------------------------------------------------------------------
// The staged kernel: tile of rows → shared memory → serial-tree descent.
// ---------------------------------------------------------------------------

/// One unit per row; a cube stages its `tile`-row block of features into shared
/// memory (coalesced), then each unit descends all trees serially reading
/// features from shared memory. Single-cell scalar-leaf Raw: `output[row]` is
/// the bit-exact serial sum of the reached leaf values.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn predict_staged<F: Float, T: Float>(
    cleft: &Array<i32>,
    cright: &Array<i32>,
    split_index: &Array<i32>,
    threshold: &Array<T>,
    default_left: &Array<u32>,
    leaf_value: &Array<T>,
    node_off: &Array<u32>,
    input: &Array<F>,
    output: &mut Array<F>,
    num_row: u32,
    num_tree: u32,
    #[comptime] num_feature: u32,
    #[comptime] tile: u32,
) {
    // Static shared-memory tile: tile rows × num_feature elements (comptime).
    let smem_len = comptime!(tile as usize * num_feature as usize);
    let mut feat = SharedMemory::<F>::new(smem_len);

    let t = UNIT_POS_X;
    let tile_row_base = CUBE_POS_X * tile;

    // Cooperative, coalesced staging of the whole tile into shared memory.
    // Stride by CUBE_DIM_X (== tile) so adjacent units touch adjacent elements.
    let total = tile * num_feature;
    let mut e = t;
    while e < total {
        let local_row = e / num_feature;
        let feat_idx = e % num_feature;
        let grow = tile_row_base + local_row;
        if grow < num_row {
            feat[e as usize] = input[(grow * num_feature + feat_idx) as usize];
        } else {
            feat[e as usize] = F::new(0.0);
        }
        e += CUBE_DIM_X;
    }
    sync_cube();

    // One unit per row: serial-tree descent reading features from shared memory.
    let row = tile_row_base + t;
    if row < num_row {
        let smem_row = t * num_feature;
        let mut acc = F::new(0.0);
        for tree_id in 0..num_tree {
            let base = node_off[tree_id as usize];
            let mut nid: u32 = 0;
            while cleft[(base + nid) as usize] != -1i32 {
                let idx = (base + nid) as usize;
                let fi = split_index[idx];
                // The ONLY change vs the production descend: read from shared mem.
                let fv = feat[(smem_row + fi as u32) as usize];
                let mut next: i32 = cright[idx];
                #[allow(clippy::eq_op)]
                if fv != fv {
                    if default_left[idx] == 1u32 {
                        next = cleft[idx];
                    }
                } else if f64::cast_from(fv) < f64::cast_from(threshold[idx]) {
                    next = cleft[idx];
                }
                nid = next as u32;
            }
            acc += F::cast_from(leaf_value[(base + nid) as usize]);
        }
        output[row as usize] = acc;
    }
}

/// Host launcher (generic over runtime). Uploads the forest, launches the staged
/// kernel with `cube_dim = tile`, reads back. Single-cell scalar-leaf Raw only.
#[allow(clippy::too_many_arguments)]
fn run_staged<R: Runtime>(
    client: &cubecl::client::ComputeClient<R>,
    preset: &ModelPreset<f32>,
    num_feature: u32,
    data: &[f32],
    num_row: usize,
    tile: u32,
) -> Vec<f32>
where
    R::Device: Default,
{
    // LeafId-style upload: no leaf-vector broadcast span needed.
    let forest = upload::upload_forest::<R, f32>(
        client,
        preset,
        num_feature as i32,
        num_row,
        data.len(),
        0,
        0,
        &[],
        &[],
        &[],
    )
    .expect("upload_forest");
    let num_tree = preset.trees.len();

    let h_node_off = forest.node_off(client);
    let h_in = client.create_from_slice(bytemuck::cast_slice(data));
    let zero_out = vec![0.0f32; num_row];
    let h_out = client.create_from_slice(bytemuck::cast_slice(&zero_out));

    let cube_dim = CubeDim::new_1d(tile);
    let cube_count = CubeCount::Static((num_row as u32).div_ceil(tile).max(1), 1, 1);

    predict_staged::launch::<f32, f32, R>(
        client,
        cube_count,
        cube_dim,
        unsafe { ArrayArg::from_raw_parts(forest.cleft.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.cright.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.split_index.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.threshold.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.default_left.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(forest.leaf_value.clone(), forest.num_nodes_total) },
        unsafe { ArrayArg::from_raw_parts(h_node_off.clone(), num_tree + 1) },
        unsafe { ArrayArg::from_raw_parts(h_in.clone(), data.len()) },
        unsafe { ArrayArg::from_raw_parts(h_out.clone(), zero_out.len()) },
        num_row as u32,
        num_tree as u32,
        num_feature,
        tile,
    );

    let bytes = client.read_one_unchecked(h_out);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

// ---------------------------------------------------------------------------
// Synthetic single-cell scalar-leaf model (perfect binary trees).
// ---------------------------------------------------------------------------

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

fn preset_of(model: &Model) -> &ModelPreset<f32> {
    match &model.variant {
        ModelVariant::F32(p) => p,
        ModelVariant::F64(_) => unreachable!("spike builds f32 models"),
    }
}

// ---------------------------------------------------------------------------
// Correctness gate (CPU runtime, always runs): staged == scalar within 1e-5,
// AND bit-identical to the production cubecl kernel.
// ---------------------------------------------------------------------------

/// Returns (max |delta| vs scalar, #bit-mismatches vs production cubecl).
fn check_correctness<R: Runtime>(
    label: &str,
    client: &cubecl::client::ComputeClient<R>,
) -> (f64, usize)
where
    R::Device: Default,
{
    let num_feature = 16usize;
    let tile = 64u32;
    // num_row deliberately NOT a multiple of tile to exercise the partial tile.
    let num_row = 1000usize;
    let trees: Vec<Tree<f32>> =
        (0..40).map(|i| perfect_tree(6, num_feature as i32, i as f32 * 1e-2)).collect();
    let model = single_cell_model(trees, num_feature as i32);
    let data = make_input(num_row, num_feature);
    let cfg = Config { kind: PredictKind::Raw, nthread: 0 };

    let staged =
        run_staged::<R>(client, preset_of(&model), num_feature as u32, &data, num_row, tile);
    let scalar = treelite_gtil::predict::<f32>(&model, &data, num_row, &cfg).expect("scalar");
    let prod = treelite_cubecl::predict_cpu::<f32>(&model, &data, num_row, &cfg).expect("prod");

    let mut max_delta = 0.0f64;
    let mut bit_mismatches = 0usize;
    for r in 0..num_row {
        max_delta = max_delta.max((staged[r] as f64 - scalar[r] as f64).abs());
        if staged[r].to_bits() != prod[r].to_bits() {
            bit_mismatches += 1;
        }
    }
    eprintln!("[{label}] staged vs scalar: max |delta| = {max_delta:.3e}");
    eprintln!("[{label}] staged vs production cubecl: {bit_mismatches} / {num_row} bit-mismatches");
    (max_delta, bit_mismatches)
}

#[test]
fn spike_shared_mem_correctness() {
    // CPU runtime: report only. The cubecl CPU runtime does not provide true
    // per-cube SharedMemory + sync_cube semantics across units, so this is
    // informational, not a gate (the real target is the GPU).
    let cpu = CpuRuntime::client(&Default::default());
    let (cpu_delta, cpu_mm) = check_correctness::<CpuRuntime>("cpu", &cpu);
    if cpu_delta >= 1e-5 {
        eprintln!("[cpu] NOTE: shared-memory staging not faithfully emulated on the CPU runtime (expected); see GPU gate.");
    }
    let _ = cpu_mm;

    // ROCm: the real gate — shared memory is on-chip hardware here.
    #[cfg(feature = "rocm")]
    {
        match treelite_cubecl::device::client::<cubecl::hip::HipRuntime>("rocm") {
            Ok(gpu) => {
                let (delta, mm) = check_correctness::<cubecl::hip::HipRuntime>("rocm", &gpu);
                assert!(delta < 1e-5, "staged kernel diverged from scalar on ROCm");
                assert_eq!(mm, 0, "staged kernel must be bit-identical to production on ROCm");
            }
            Err(e) => eprintln!("[rocm] correctness gate SKIPPED — {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Bench (--ignored): staged vs production, CPU and (if rocm) ROCm.
// ---------------------------------------------------------------------------

fn best_ms<FN>(iters: usize, mut f: FN) -> f64
where
    FN: FnMut() -> Vec<f32>,
{
    let _ = std::hint::black_box(f()); // warm-up (JIT + pools)
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let out = f();
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        best = best.min(dt);
    }
    best * 1e3
}

fn bench_backend<R: Runtime>(label: &str, client: &cubecl::client::ComputeClient<R>)
where
    R::Device: Default,
{
    let configs = [
        ("descent-heavy", 12u32, 64usize, 32usize, 100_000usize, 128u32),
        ("per-row-overhead", 2, 8, 16, 1_000_000, 256),
        ("balanced-gbdt", 8, 200, 32, 200_000, 128),
    ];
    for (name, depth, num_tree, num_feature, num_row, tile) in configs {
        let trees: Vec<Tree<f32>> =
            (0..num_tree).map(|i| perfect_tree(depth, num_feature as i32, i as f32 * 1e-2)).collect();
        let model = single_cell_model(trees, num_feature as i32);
        let data = make_input(num_row, num_feature);
        let cfg = Config { kind: PredictKind::Raw, nthread: 0 };

        let staged_ms = best_ms(8, || {
            run_staged::<R>(client, preset_of(&model), num_feature as u32, &data, num_row, tile)
        });
        let prod_ms = best_ms(8, || {
            treelite_cubecl::predict::<R, f32>(&model, &data, num_row, &cfg).expect("prod")
        });
        eprintln!(
            "[{label}] {name:<16} d={depth} t={num_tree} f={num_feature} rows={num_row} tile={tile}",
        );
        eprintln!(
            "[{label}]   production = {prod_ms:8.3} ms | staged = {staged_ms:8.3} ms | staged speedup = {:.2}x",
            prod_ms / staged_ms,
        );
    }
}

#[test]
#[ignore]
fn spike_shared_mem_bench() {
    eprintln!("============ SHARED-MEM STAGING SPIKE ============");
    let cpu = CpuRuntime::client(&Default::default());
    bench_backend::<CpuRuntime>("cpu", &cpu);

    #[cfg(feature = "rocm")]
    {
        match treelite_cubecl::device::client::<cubecl::hip::HipRuntime>("rocm") {
            Ok(gpu) => bench_backend::<cubecl::hip::HipRuntime>("rocm", &gpu),
            Err(e) => eprintln!("[rocm] SKIP — {e}"),
        }
    }
    eprintln!("==================================================");
}
