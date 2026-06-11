//! Wave 0 (plan 10-00) — RED scaffold for parallel-scalar determinism (GTIL-08).
//!
//! Once Wave 1 (plan 10-01) row-parallelizes the scalar GTIL loop, N repeated
//! runs of the SAME `treelite_gtil::predict` over the SAME multi-row input must
//! stay element-wise bit-identical. The inner per-tree accumulation stays SERIAL
//! (float add is non-associative), so parallelizing ONLY the independent per-row
//! axis must not introduce any run-to-run reordering — determinism holds
//! structurally, and this test proves it observationally.
//!
//! `.to_bits()` (not `==`) is the assertion: it distinguishes `+0.0`/`-0.0` and
//! makes a `NaN` payload difference a failure, so a non-deterministic
//! accumulation that happened to land on an equal-but-different bit pattern
//! cannot masquerade as a pass (T-06-13).
//!
//! RED status: the test is `#[ignore]`d with a MISSING reason string — it carries
//! the intended assertion shape that Wave 1 un-ignores and makes green after the
//! loop conversion lands.

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_gtil::{Config, PredictKind};

/// Number of repeated predict runs compared against the first (RESEARCH "N runs").
const N_RUNS: usize = 4;

/// Single-split numerical `Tree<T>` (node 0 `kLT` default-left; node 1/2 leaves).
/// Duplicated from `treelite-cubecl/tests/determinism.rs` (cargo integration
/// tests do not share a module), retargeted to the gtil scalar engine.
fn split_tree<T: Copy + Default>(feature: i32, threshold: T, left: T, right: T) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![feature, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![threshold, T::default(), T::default()]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![T::default(), left, right]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0, 0, 0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![0, 0, 0]);
    t
}

/// Binary scalar `(num_row, 1, 1)` model over `trees` (sigmoid postprocessor).
fn scalar_model<T, W>(trees: Vec<Tree<T>>, wrap: W, num_feature: i32) -> Model
where
    T: Copy,
    W: Fn(ModelPreset<T>) -> ModelVariant,
{
    let num_tree = trees.len();
    let mut m = Model::new(wrap(ModelPreset::new(trees)));
    m.num_feature = num_feature;
    m.num_target = 1;
    m.num_class = vec![1].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0; num_tree].into();
    m.class_id = vec![0; num_tree].into();
    m.postprocessor = "sigmoid".to_string().into();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![0.5].into();
    m
}

/// Build a MULTI-row 2-feature dense input that is big enough for rayon to split
/// across workers (a determinism test on a single row proves nothing about
/// parallel row reordering). Rows cycle through all four branch combinations.
fn multi_row_input(num_row: usize) -> Vec<f64> {
    let patterns = [[0.0_f64, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
    let mut data = Vec::with_capacity(num_row * 2);
    for r in 0..num_row {
        let p = patterns[r % patterns.len()];
        data.push(p[0]);
        data.push(p[1]);
    }
    data
}

/// GTIL-08: `N_RUNS` repeated `predict::<f64>` calls over an identical multi-row
/// input are element-wise bit-identical, for every `PredictKind`.
#[test]
#[ignore = "MISSING — Wave 1 (10-01) parallelizes predict; byte-identical determinism asserted after"]
fn determinism_byte_identical_n_runs() {
    let trees = vec![
        split_tree::<f64>(0, 0.5, 1.0, -1.0),
        split_tree::<f64>(1, 0.5, 2.0, -3.0),
    ];
    let model = scalar_model(trees, ModelVariant::F64, 2);

    // Many rows so the Wave-1 par_chunks_mut split is real (not a 1-row no-op).
    let num_row = 64;
    let data = multi_row_input(num_row);

    for kind in [
        PredictKind::Default,
        PredictKind::Raw,
        PredictKind::LeafId,
        PredictKind::ScorePerTree,
    ] {
        let cfg = Config { kind, nthread: 0 };
        let first = treelite_gtil::predict::<f64>(&model, &data, num_row, &cfg).unwrap();
        for run in 1..N_RUNS {
            let next = treelite_gtil::predict::<f64>(&model, &data, num_row, &cfg).unwrap();
            assert_eq!(
                first.len(),
                next.len(),
                "{kind:?}: length differs on run {run}"
            );
            for (i, (x, y)) in first.iter().zip(next.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{kind:?}: cell {i} not bit-identical on run {run} (GTIL-08): \
                     {x} (bits {:#x}) vs {y} (bits {:#x})",
                    x.to_bits(),
                    y.to_bits(),
                );
            }
        }
    }

    // TODO Wave 1: sparse determinism — repeat the above via `predict_sparse`
    // over a `SparseCsr` fixture once the sparse path is parallelized.
}
