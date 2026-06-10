//! Row accessors that materialize a per-row dense feature view for traversal.
//!
//! Ports upstream's `DenseMatrixAccessor` / `SparseMatrixAccessor`
//! (`treelite-mainline/src/gtil/predict.cc:38-97`). Both produce a contiguous
//! `&[O]` row that `evaluate_tree` walks VERBATIM — so the dense and sparse
//! predict paths share the exact same traversal and dense==sparse parity on
//! identical logical data is *structural*, not coincidental (D-04).
//!
//! The sparse accessor is the load-bearing one: absent CSR entries are
//! materialized as `O::nan()` (the "missing value" sentinel that
//! `evaluate_tree` routes to the default child), NOT `0` — exactly
//! `SparseMatrixAccessor::GetRow` (`predict.cc:80-85`). Every `col_ind` /
//! `row_ptr` read is bounds-checked into a typed [`GtilError`] so a corrupt CSR
//! can never OOB-write the scratch row or OOB-slice `data` (T-05-09 / T-05-10).

use crate::PredictOut;
use crate::error::GtilError;

/// A borrowed compressed-sparse-row (CSR) feature matrix (`PredictSparse`
/// signature, `gtil.h:85-88`): `data`/`col_ind`/`row_ptr` are the three CSR
/// arrays, untrusted and validated on use.
///
/// - `data[k]` is the `k`-th stored (present) value, in row-major CSR order;
/// - `col_ind[k]` is the column (feature) index of `data[k]`;
/// - `row_ptr[r]..row_ptr[r+1]` is the half-open range of `data`/`col_ind`
///   indices belonging to row `r` (so `row_ptr.len() == num_row + 1`).
///
/// All three are borrowed (`&'a [_]`) — zero-copy over the caller's buffers,
/// mirroring the upstream pointer-triple accessor.
#[derive(Debug, Clone, Copy)]
pub struct SparseCsr<'a, O> {
    /// The present feature values, row-major CSR order.
    pub data: &'a [O],
    /// Column (feature) index of each `data` entry.
    pub col_ind: &'a [u64],
    /// Per-row offsets into `data`/`col_ind`; length `num_row + 1`.
    pub row_ptr: &'a [u64],
}

impl<'a, O: PredictOut> SparseCsr<'a, O> {
    /// Validate the CSR structure against `num_row` / `num_feature` ONCE, before
    /// any row is materialized (`SparseMatrixAccessor` ctor invariants,
    /// `predict.cc:61-69`).
    ///
    /// Checks, in order (all → a typed error, never a panic):
    /// - `row_ptr.len() == num_row + 1` (`SparseRowPtrInvalid`);
    /// - `row_ptr` is monotone non-decreasing and `row_ptr[num_row] <=
    ///   data.len()` and `<= col_ind.len()` — so every per-row slice is in
    ///   bounds (T-05-10, `SparseRowPtrInvalid`);
    /// - every `col_ind[k] < num_feature` — so the scratch write at
    ///   `scratch[col_ind[k]]` can never go out of bounds (T-05-09,
    ///   `SparseColumnOutOfBounds`).
    pub fn validate(&self, num_row: usize, num_feature: usize) -> Result<(), GtilError> {
        // row_ptr must have exactly num_row + 1 entries (one fence per row plus
        // the trailing total). A short/long row_ptr is a malformed CSR.
        if self.row_ptr.len() != num_row + 1 {
            return Err(GtilError::SparseRowPtrInvalid {
                index: num_row,
                value: self.row_ptr.len() as u64,
                limit: (num_row + 1) as u64,
            });
        }
        // Monotone non-decreasing; the final fence bounds both backing arrays.
        let mut prev: u64 = 0;
        for (i, &p) in self.row_ptr.iter().enumerate() {
            if p < prev {
                // Non-monotonic: row i would slice [p..) below the previous
                // fence, an inverted/overlapping range.
                return Err(GtilError::SparseRowPtrInvalid {
                    index: i,
                    value: p,
                    limit: prev,
                });
            }
            prev = p;
        }
        // The last fence must not exceed either backing array length, else a
        // per-row slice would read past the end of data/col_ind.
        let total = *self.row_ptr.last().unwrap(); // len checked == num_row+1 ≥ 1
        let backing = self.data.len().min(self.col_ind.len()) as u64;
        if total > backing {
            return Err(GtilError::SparseRowPtrInvalid {
                index: num_row,
                value: total,
                limit: backing,
            });
        }
        // Every column index must be a valid feature, so scratch[col] is in
        // bounds. Only the entries actually referenced by row_ptr matter, but
        // checking the whole prefix [0, total) is equivalent and simpler.
        let nf = num_feature as u64;
        for k in 0..(total as usize) {
            let col = self.col_ind[k];
            if col >= nf {
                return Err(GtilError::SparseColumnOutOfBounds {
                    col,
                    num_feature: nf,
                });
            }
        }
        Ok(())
    }

    /// Materialize row `r` into `scratch` (`SparseMatrixAccessor::GetRow`,
    /// `predict.cc:72-86`): fill the whole scratch with `O::nan()` (absent =
    /// NaN, SC1), then overwrite the present columns with their `data` values.
    ///
    /// MUST be called only after [`validate`](Self::validate) has succeeded for
    /// the same `num_row`/`num_feature`; under that precondition every index
    /// here is in bounds. `scratch.len()` must equal `num_feature`.
    pub fn get_row(&self, r: usize, scratch: &mut [O]) {
        // 1) absent positions are NaN (predict.cc:81) — NOT 0. This is what
        //    makes dense(NaN)==sparse parity structural: a missing feature hits
        //    the same default-child route in evaluate_tree on both paths.
        for v in scratch.iter_mut() {
            *v = O::nan();
        }
        // 2) write the present (column, value) pairs for this row
        //    (predict.cc:83-85).
        let begin = self.row_ptr[r] as usize;
        let end = self.row_ptr[r + 1] as usize;
        for k in begin..end {
            scratch[self.col_ind[k] as usize] = self.data[k];
        }
    }
}
