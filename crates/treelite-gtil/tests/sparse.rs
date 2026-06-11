//! Sparse-CSR predict tests (Plan 05-04 Task 1, GTIL-02 / D-04).
//!
//! Asserts:
//! - the per-row NaN materialization (absent columns become NaN, NOT 0);
//! - dense==sparse structural parity (`predict_sparse(csr)` ==
//!   `predict(dense_with_nan)` on identical logical data, D-04);
//! - malformed `col_ind` / `row_ptr` surface as typed `GtilError`, never a
//!   panic / OOB (T-05-09 / T-05-10).

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_gtil::{Config, GtilError, SparseCsr, predict, predict_sparse};

/// Single-split `Tree<f32>`: node 0 numerical test on `feature` (`kLT`,
/// default-left), nodes 1/2 leaves with `left_leaf`/`right_leaf`.
fn split_tree(feature: i32, threshold: f32, left_leaf: f32, right_leaf: f32) -> Tree<f32> {
    let mut t = Tree::<f32>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![feature, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![threshold, 0.0, 0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![0.0, left_leaf, right_leaf]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    t
}

/// Binary scalar `(num_row, 1, 1)` F32 model over `trees` with `num_feature`
/// features and an identity postprocessor (so the margin is the raw output).
fn model_of(trees: Vec<Tree<f32>>, num_feature: i32) -> Model {
    let num_tree = trees.len();
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = num_feature;
    m.num_target = 1;
    m.num_class = vec![1].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0; num_tree].into();
    m.class_id = vec![0; num_tree].into();
    m.postprocessor = "identity".to_string().into();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![0.0].into();
    m
}

/// A CSR row with entries at columns {0, 3} materializes a scratch row where
/// columns {1, 2, 4} are NaN and {0, 3} hold the data values (absent = NaN, NOT
/// 0). We probe this via a tree that splits on an ABSENT column with
/// `default_left = true`: the absent (NaN) value must route to the default
/// child — which it only does if the absent cell is NaN, not 0.
#[test]
fn absent_csr_entries_materialize_as_nan_not_zero() {
    // num_feature = 5; tree splits on feature 1 (ABSENT in the CSR row),
    // threshold 0.5, default_left=true. left leaf = 100.0, right leaf = -100.0.
    let m = model_of(vec![split_tree(1, 0.5, 100.0, -100.0)], 5);

    // One row, present at columns {0, 3} with values 7.0 and 9.0; column 1 is
    // ABSENT → NaN → routes to default_left → LEFT leaf (100.0).
    let data = [7.0_f32, 9.0_f32];
    let col_ind = [0_u64, 3_u64];
    let row_ptr = [0_u64, 2_u64];
    let csr = SparseCsr {
        data: &data,
        col_ind: &col_ind,
        row_ptr: &row_ptr,
    };
    let out = predict_sparse(&m, csr, 1, &Config::default()).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0], 100.0_f32,
        "absent feature must be NaN (route default_left), not 0 (which would be < 0.5 too \
         but a 0 fill would also pick left here — so cross-check the != 0 case below)"
    );

    // Cross-check that an absent cell is genuinely NaN, not 0: split on the
    // absent feature with default_left=FALSE. A 0-fill would compare 0 < 0.5 →
    // LEFT; a NaN-fill routes to default (RIGHT). The right answer is RIGHT.
    let mut t = split_tree(1, 0.5, 100.0, -100.0);
    t.default_left = TreeBuf::from_owned(vec![false, false, false]);
    let m2 = model_of(vec![t], 5);
    let out2 = predict_sparse(&m2, csr, 1, &Config::default()).unwrap();
    assert_eq!(
        out2[0], -100.0_f32,
        "absent feature is NaN → default_child (right); a wrong 0-fill would route left"
    );
}

/// `predict_sparse(csr)` equals `predict(dense_with_nan)` when the dense matrix
/// has NaN exactly where the CSR is absent (D-04 structural parity).
#[test]
fn dense_with_nan_equals_sparse() {
    // num_feature = 4, two trees, three rows. Build a presence mask, then a
    // dense matrix (NaN in absent positions) and the equivalent CSR.
    let num_feature = 4usize;
    let m = model_of(
        vec![split_tree(0, 0.5, 1.0, -1.0), split_tree(2, 1.5, 0.5, -0.5)],
        num_feature as i32,
    );

    // Logical rows (None = absent → NaN):
    let rows: [[Option<f32>; 4]; 3] = [
        [Some(0.2), None, Some(1.0), None], // f0=0.2(<0.5→+1), f2=1.0(<1.5→+0.5)
        [Some(0.9), Some(3.0), None, Some(2.0)], // f0=0.9(≥0.5→-1), f2 absent→NaN→default(right tree2, default_left=true→+0.5)
        [None, None, None, None],                // all absent → both trees default_left
    ];

    // Dense-with-NaN buffer.
    let mut dense: Vec<f32> = Vec::with_capacity(3 * num_feature);
    for row in &rows {
        for cell in row {
            dense.push(cell.unwrap_or(f32::NAN));
        }
    }

    // Equivalent CSR (present cells only, row-major).
    let mut data: Vec<f32> = Vec::new();
    let mut col_ind: Vec<u64> = Vec::new();
    let mut row_ptr: Vec<u64> = vec![0];
    for row in &rows {
        for (c, cell) in row.iter().enumerate() {
            if let Some(v) = cell {
                data.push(*v);
                col_ind.push(c as u64);
            }
        }
        row_ptr.push(data.len() as u64);
    }

    let dense_out = predict(&m, &dense, 3, &Config::default()).unwrap();
    let csr = SparseCsr {
        data: &data,
        col_ind: &col_ind,
        row_ptr: &row_ptr,
    };
    let sparse_out = predict_sparse(&m, csr, 3, &Config::default()).unwrap();

    assert_eq!(dense_out.len(), sparse_out.len());
    for (i, (d, s)) in dense_out.iter().zip(sparse_out.iter()).enumerate() {
        // Bitwise-equal: identical logical data through the identical traversal.
        assert_eq!(d, s, "dense vs sparse mismatch at cell {i}: {d} != {s}");
    }
}

/// The same parity for f64 input (D-05 × D-04).
#[test]
fn dense_with_nan_equals_sparse_f64() {
    let num_feature = 3usize;
    let m = model_of(vec![split_tree(0, 0.5, 2.0, -2.0)], num_feature as i32);

    let dense: Vec<f64> = vec![
        0.1,
        f64::NAN,
        f64::NAN, // row0: f0=0.1 (<0.5 → +2.0)
        0.9,
        5.0,
        f64::NAN, // row1: f0=0.9 (≥0.5 → -2.0)
    ];
    let data: Vec<f64> = vec![0.1, 0.9, 5.0];
    let col_ind: Vec<u64> = vec![0, 0, 1];
    let row_ptr: Vec<u64> = vec![0, 1, 3];

    let dense_out = predict(&m, &dense, 2, &Config::default()).unwrap();
    let csr = SparseCsr {
        data: &data,
        col_ind: &col_ind,
        row_ptr: &row_ptr,
    };
    let sparse_out = predict_sparse(&m, csr, 2, &Config::default()).unwrap();
    assert_eq!(dense_out, sparse_out);
}

/// `col_ind[k] >= num_feature` returns `SparseColumnOutOfBounds`, never panics.
#[test]
fn col_ind_out_of_bounds_is_typed_error() {
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], 3); // num_feature = 3
    let data = [1.0_f32, 2.0_f32];
    let col_ind = [0_u64, 5_u64]; // 5 >= num_feature(3) → OOB
    let row_ptr = [0_u64, 2_u64];
    let csr = SparseCsr {
        data: &data,
        col_ind: &col_ind,
        row_ptr: &row_ptr,
    };
    let err = predict_sparse(&m, csr, 1, &Config::default()).unwrap_err();
    match err {
        GtilError::SparseColumnOutOfBounds { col, num_feature } => {
            assert_eq!(col, 5);
            assert_eq!(num_feature, 3);
        }
        other => panic!("expected SparseColumnOutOfBounds, got {other:?}"),
    }
}

/// Non-monotonic `row_ptr` returns `SparseRowPtrInvalid`, never panics.
#[test]
fn non_monotonic_row_ptr_is_typed_error() {
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], 3);
    let data = [1.0_f32, 2.0_f32, 3.0_f32];
    let col_ind = [0_u64, 1_u64, 2_u64];
    // row_ptr [0, 2, 1]: the third fence (1) is < the second (2) → non-monotone.
    let row_ptr = [0_u64, 2_u64, 1_u64];
    let csr = SparseCsr {
        data: &data,
        col_ind: &col_ind,
        row_ptr: &row_ptr,
    };
    let err = predict_sparse(&m, csr, 2, &Config::default()).unwrap_err();
    assert!(
        matches!(err, GtilError::SparseRowPtrInvalid { .. }),
        "expected SparseRowPtrInvalid, got {err:?}"
    );
}

/// `row_ptr[num_row] > data.len()` returns `SparseRowPtrInvalid`, never panics.
#[test]
fn row_ptr_past_data_len_is_typed_error() {
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], 3);
    let data = [1.0_f32, 2.0_f32]; // len 2
    let col_ind = [0_u64, 1_u64];
    // trailing fence 5 > data.len() 2 → out of range.
    let row_ptr = [0_u64, 5_u64];
    let csr = SparseCsr {
        data: &data,
        col_ind: &col_ind,
        row_ptr: &row_ptr,
    };
    let err = predict_sparse(&m, csr, 1, &Config::default()).unwrap_err();
    assert!(
        matches!(err, GtilError::SparseRowPtrInvalid { .. }),
        "expected SparseRowPtrInvalid, got {err:?}"
    );
}

/// `row_ptr.len() != num_row + 1` returns `SparseRowPtrInvalid`, never panics.
#[test]
fn wrong_row_ptr_length_is_typed_error() {
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], 3);
    let data = [1.0_f32];
    let col_ind = [0_u64];
    // num_row = 2 needs row_ptr.len() == 3, but only 2 entries supplied.
    let row_ptr = [0_u64, 1_u64];
    let csr = SparseCsr {
        data: &data,
        col_ind: &col_ind,
        row_ptr: &row_ptr,
    };
    let err = predict_sparse(&m, csr, 2, &Config::default()).unwrap_err();
    assert!(
        matches!(err, GtilError::SparseRowPtrInvalid { .. }),
        "expected SparseRowPtrInvalid, got {err:?}"
    );
}
