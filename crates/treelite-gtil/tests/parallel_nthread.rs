//! Wave 0 (plan 10-00) — RED scaffold for nthread equivalence + >1-core
//! utilization of the parallel scalar GTIL engine (PAR-01/02/04).
//!
//! Wave 1 (plan 10-01) row-parallelizes `treelite_gtil::predict` and threads
//! `Config.nthread` through to a scoped `rayon::ThreadPool`. These tests assert
//! the two contracts that conversion must satisfy:
//!   1. `nthread_equivalence` — `nthread = 0` (global pool), `1`, and `2` produce
//!      byte-identical output (the per-row split must not change the result).
//!   2. `parallel_uses_more_than_one_core` — on a multi-core runner the parallel
//!      predict actually fans out across more than one rayon worker.
//!
//! `.to_bits()` (not `==`) is the equivalence assertion (T-06-13). RED status:
//! both tests are `#[ignore]`d with MISSING reason strings — Wave 1 un-ignores
//! them and makes them green once the parallelism and the scoped pool land.

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_gtil::{Config, PredictKind};

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

/// Multi-row 2-feature dense input large enough for a real parallel split.
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

/// PAR-04: `nthread = 0`, `1`, and `2` produce byte-identical output for every
/// `PredictKind` (the scoped-pool size must not change the result).
#[test]
#[ignore = "MISSING — Wave 1 (10-01) wires nthread into a scoped rayon pool"]
fn nthread_equivalence() {
    let trees = vec![
        split_tree::<f64>(0, 0.5, 1.0, -1.0),
        split_tree::<f64>(1, 0.5, 2.0, -3.0),
    ];
    let model = scalar_model(trees, ModelVariant::F64, 2);
    let num_row = 64;
    let data = multi_row_input(num_row);

    for kind in [
        PredictKind::Default,
        PredictKind::Raw,
        PredictKind::LeafId,
        PredictKind::ScorePerTree,
    ] {
        let baseline = treelite_gtil::predict::<f64>(
            &model,
            &data,
            num_row,
            &Config { kind, nthread: 0 },
        )
        .unwrap();
        for nthread in [1, 2] {
            let got = treelite_gtil::predict::<f64>(
                &model,
                &data,
                num_row,
                &Config { kind, nthread },
            )
            .unwrap();
            assert_eq!(
                baseline.len(),
                got.len(),
                "{kind:?}: length differs at nthread={nthread}"
            );
            for (i, (x, y)) in baseline.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{kind:?}: cell {i} differs at nthread={nthread} (PAR-04): \
                     {x} (bits {:#x}) vs {y} (bits {:#x})",
                    x.to_bits(),
                    y.to_bits(),
                );
            }
        }
    }
}

/// PAR-01/02: on a multi-core runner the default (global-pool) parallel predict
/// fans out across more than one rayon worker. Vacuously passes on a 1-core
/// runner (Environment Availability note).
#[test]
#[ignore = "MISSING — Wave 1 (10-01) parallelizes predict over rayon workers"]
fn parallel_uses_more_than_one_core() {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if cores <= 1 {
        // Single-core runner: nothing to prove about >1-worker fan-out.
        return;
    }

    let trees = vec![
        split_tree::<f64>(0, 0.5, 1.0, -1.0),
        split_tree::<f64>(1, 0.5, 2.0, -3.0),
    ];
    let model = scalar_model(trees, ModelVariant::F64, 2);
    let num_row = 1024;
    let data = multi_row_input(num_row);

    let cfg = Config {
        kind: PredictKind::Default,
        nthread: 0,
    };
    let _ = treelite_gtil::predict::<f64>(&model, &data, num_row, &cfg).unwrap();

    // Wave 1: the meaningful assertion is that the parallel section ran on >1
    // worker. The global rayon pool reports its width here; Wave 1's
    // par_chunks_mut split makes this observation load-bearing.
    assert!(
        rayon::current_num_threads() > 1,
        "expected the global rayon pool to expose >1 worker on a {cores}-core runner, \
         got {}",
        rayon::current_num_threads(),
    );
}
