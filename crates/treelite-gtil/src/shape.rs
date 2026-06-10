//! Public per-kind output-shape descriptor (D-07).
//!
//! Ports `GetOutputShape` (`treelite-mainline/src/gtil/output_shape.cc:17-39`)
//! verbatim. Callers (Phase-8 numpy reshape) allocate / reshape their output
//! buffer against [`Shape::dims`]. This is the *public* shape returned to
//! callers; the predict-internal `(num_row, num_target, max_num_class)` indexer
//! is a separate private `OutputLayout` in `lib.rs` (the two were disambiguated
//! per RESEARCH Open Q3).

use crate::config::{Config, PredictKind};
use treelite_core::Model;

/// Per-kind output shape, a flat dimension vector (`std::vector<std::uint64_t>`
/// upstream). For `Default`/`Raw` this is `[num_row, num_target_or_1,
/// max_num_class]`; for `LeafId` it is `[num_row, num_tree]`; for
/// `ScorePerTree` it is `[num_row, num_tree, leaf_vector_shape[0] *
/// leaf_vector_shape[1]]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Output dimensions, outermost (rows) first.
    pub dims: Vec<u64>,
}

/// Compute the output shape for `num_row` rows under `config` (`GetOutputShape`,
/// `output_shape.cc:17-39`).
///
/// `max_num_class` is `max(num_class[0..num_target])`, clamped to `>= 1` so a
/// degenerate / malformed model never produces a zero or negative dimension
/// (threat T-05-04; mirrors the `lib.rs` clamp at the predict entry). The
/// `default`/`raw` branch collapses `num_target == 1` to dim `1` (it is NOT
/// omitted) exactly as upstream.
pub fn output_shape(model: &Model, num_row: u64, config: &Config) -> Shape {
    let num_tree = model.num_tree();

    match config.kind {
        PredictKind::Default | PredictKind::Raw => {
            let max_num_class = max_num_class(model);
            if model.num_target > 1 {
                Shape {
                    dims: vec![num_row, model.num_target as u64, max_num_class],
                }
            } else {
                Shape {
                    dims: vec![num_row, 1, max_num_class],
                }
            }
        }
        PredictKind::LeafId => Shape {
            dims: vec![num_row, num_tree],
        },
        PredictKind::ScorePerTree => {
            // leaf_vector_shape[0] * leaf_vector_shape[1]; read defensively so a
            // short/malformed shape vector yields 0 rather than panicking
            // (threat T-05-04). Upstream sizes it to exactly 2.
            let a = model.leaf_vector_shape.first().copied().unwrap_or(0) as u64;
            let b = model.leaf_vector_shape.get(1).copied().unwrap_or(0) as u64;
            Shape {
                dims: vec![num_row, num_tree, a * b],
            }
        }
    }
}

/// `max(num_class[0..num_target])`, clamped to `>= 1` (T-05-04).
///
/// Upstream reads `num_class.Data()[0..num_target]` unchecked
/// (`output_shape.cc:20-21`); here the iterator is bounded by the actual
/// `num_class` length and the result is clamped to `1` for an empty / degenerate
/// model so the dimension is always valid.
fn max_num_class(model: &Model) -> u64 {
    model.num_class.iter().copied().max().unwrap_or(1).max(1) as u64
}
