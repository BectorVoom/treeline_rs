//! Line-based `key=value` tokenizer → typed `LGBModel` / `LGBTree` structs.
//!
//! Ports the text parser in `treelite-mainline/src/model_loader/lightgbm.cc`:
//! - the global/tree `key=value` split (`lightgbm.cc:248-291`)
//! - the per-tree field parse with EXACT per-field precision and array lengths
//!   (`lightgbm.cc:293-414`)
//!
//! **Per-field precision is load-bearing (LGB-02 precursor, 1e-5 fidelity):**
//! `leaf_value` and `threshold` are `f64`, `split_gain` is `f32`,
//! `decision_type` is `i8`, `num_leaves`/`num_cat`/`num_class`/`max_feature_idx`
//! are `i32`, `cat_boundaries` are `u64`, `cat_threshold` are `u32`. Downcasting
//! `leaf_value`/`threshold` to `f32` would silently shift the last ULPs past the
//! 1e-5 target.
//!
//! Every malformed count returns [`LgbError::Parse`] / [`LgbError::DimensionMismatch`]
//! — never an out-of-bounds slice (T-04-07, ASVS V5).

use std::collections::HashMap;
use std::str::FromStr;

use crate::error::LgbError;

/// One parsed LightGBM tree, with the raw LightGBM negative-index leaf encoding
/// preserved (re-numbering happens later in `lib.rs`).
#[derive(Debug, Clone, Default)]
pub struct LGBTree {
    /// Number of leaves. Determines every array length below.
    pub num_leaves: i32,
    /// Number of categorical splits (0 for a purely-numerical tree).
    pub num_cat: i32,
    /// Per-leaf output value (length `num_leaves`). f64 — NO downcast.
    pub leaf_value: Vec<f64>,
    /// Per-internal-node decision-type bitfield (length `num_leaves - 1`). i8.
    pub decision_type: Vec<i8>,
    /// Categorical-split boundaries (length `num_cat + 1`). u64. LGB-02.
    pub cat_boundaries: Vec<u64>,
    /// Categorical-split bitsets (length `cat_boundaries.back()`). u32. LGB-02.
    pub cat_threshold: Vec<u32>,
    /// Per-internal-node split feature index (length `num_leaves - 1`). i32.
    pub split_feature: Vec<i32>,
    /// Per-internal-node threshold (length `num_leaves - 1`). f64 — NO downcast.
    pub threshold: Vec<f64>,
    /// Per-internal-node left child (length `num_leaves - 1`). i32. Negative ⇒ leaf.
    pub left_child: Vec<i32>,
    /// Per-internal-node right child (length `num_leaves - 1`). i32. Negative ⇒ leaf.
    pub right_child: Vec<i32>,
    /// Per-internal-node split gain (length `num_leaves - 1`, or empty). f32.
    pub split_gain: Vec<f32>,
    /// Per-internal-node sample count (length `num_leaves - 1`, or empty). i32.
    pub internal_count: Vec<i32>,
    /// Per-leaf sample count (length `num_leaves`, or empty). i32.
    pub leaf_count: Vec<i32>,
}

/// A parsed LightGBM model: global header fields + the per-tree structs.
#[derive(Debug, Clone, Default)]
pub struct LGBModel {
    /// `max_feature_idx` (`num_feature = max_feature_idx + 1`). i32.
    pub max_feature_idx: i32,
    /// `num_class`. i32.
    pub num_class: i32,
    /// Whether the global `average_output` key was present.
    pub average_output: bool,
    /// The raw objective name (first token of the `objective=` line), or
    /// `"custom"` when the key is absent (`lightgbm.cc:272-273`).
    pub objective_name: String,
    /// The whitespace-split tail of the `objective=` line (params).
    pub objective_params: Vec<String>,
    /// The parsed trees, in file order.
    pub trees: Vec<LGBTree>,
}

/// Split `text` on `delim`, keeping every token (LightGBM's `Split` helper,
/// `lightgbm.cc:157-165`).
fn split_on(text: &str, delim: char) -> Vec<String> {
    text.split(delim).map(|s| s.to_string()).collect()
}

/// Parse a single number token, attributing a failure to `where_` (`lightgbm.cc`
/// `TextToNumber`). Trims surrounding whitespace first (LightGBM tokens are
/// space-delimited and may carry trailing CR on Windows-authored files).
fn parse_num<T: FromStr>(where_: &str, field: &'static str, token: &str) -> Result<T, LgbError> {
    token.trim().parse::<T>().map_err(|_| LgbError::Parse {
        line: where_.to_string(),
        detail: format!("field {field:?}: cannot parse {token:?} as a number"),
    })
}

/// Parse a space-delimited array of exactly `num_entry` numbers
/// (`lightgbm.cc:167-180` `TextToArray`). An empty text with `num_entry > 0`, or
/// a token count below `num_entry`, is a typed error — never an OOB slice
/// (T-04-07).
fn parse_array<T: FromStr>(
    where_: &str,
    field: &'static str,
    text: &str,
    num_entry: usize,
) -> Result<Vec<T>, LgbError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        if num_entry > 0 {
            return Err(LgbError::Parse {
                line: where_.to_string(),
                detail: format!("field {field:?}: cannot convert empty text into {num_entry} entries"),
            });
        }
        return Ok(Vec::new());
    }
    let tokens: Vec<&str> = trimmed.split(' ').filter(|t| !t.is_empty()).collect();
    if tokens.len() < num_entry {
        return Err(LgbError::Parse {
            line: where_.to_string(),
            detail: format!(
                "field {field:?}: expected {num_entry} entries, found {}",
                tokens.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(num_entry);
    for token in tokens.into_iter().take(num_entry) {
        out.push(parse_num::<T>(where_, field, token)?);
    }
    Ok(out)
}

/// Look up a required key, returning a typed error when it is missing.
fn require<'a>(
    dict: &'a HashMap<String, String>,
    where_: &str,
    key: &'static str,
) -> Result<&'a String, LgbError> {
    dict.get(key).ok_or_else(|| LgbError::Parse {
        line: where_.to_string(),
        detail: format!("missing required key {key:?}"),
    })
}

/// Parse a full LightGBM text model into an [`LGBModel`].
///
/// Ports `ParseStream` (`lightgbm.cc:234-414`): split each line on the FIRST
/// `=` (re-appending any further `=` to the value, e.g. `feature_infos=[0:100]`),
/// route lines into the global dict or the current tree dict (a `Tree=` line
/// opens a new tree), then parse each section with exact per-field precision.
pub fn parse_lightgbm(model_str: &str) -> Result<LGBModel, LgbError> {
    let mut global_dict: HashMap<String, String> = HashMap::new();
    let mut tree_dicts: Vec<HashMap<String, String>> = Vec::new();
    let mut in_tree = false;

    for raw_line in model_str.lines() {
        // StringTrimFromEnd (lightgbm.cc:227): drop trailing whitespace/CR.
        let line = raw_line.trim_end();
        if line.is_empty() {
            continue;
        }
        // Split on the FIRST '=' only; the remainder (which may contain '=') is
        // the value (lightgbm.cc:249-257).
        let (key, value) = match line.split_once('=') {
            Some((k, v)) => (k, v),
            // A bare line with no '=' that isn't a tree/section marker is ignored
            // (upstream getline leaves value empty). Section markers like
            // "end of trees" fall here and are harmless.
            None => (line, ""),
        };
        if key == "Tree" {
            in_tree = true;
            tree_dicts.push(HashMap::new());
        } else if in_tree {
            tree_dicts
                .last_mut()
                .expect("in_tree implies a tree dict exists")
                .insert(key.to_string(), value.to_string());
        } else {
            global_dict.insert(key.to_string(), value.to_string());
        }
    }

    // --- global header (lightgbm.cc:270-291) ---
    let (objective_name, objective_params) = match global_dict.get("objective") {
        None => ("custom".to_string(), Vec::new()),
        Some(obj) => {
            let toks = split_on(obj.trim(), ' ');
            let name = toks.first().cloned().unwrap_or_default();
            let params = toks.into_iter().skip(1).filter(|t| !t.is_empty()).collect();
            (name, params)
        }
    };

    let max_feature_idx =
        parse_num::<i32>("global", "max_feature_idx", require(&global_dict, "global", "max_feature_idx")?)?;
    let num_class =
        parse_num::<i32>("global", "num_class", require(&global_dict, "global", "num_class")?)?;
    let average_output = global_dict.contains_key("average_output");

    // --- per-tree (lightgbm.cc:293-414) ---
    let mut trees = Vec::with_capacity(tree_dicts.len());
    for (idx, dict) in tree_dicts.iter().enumerate() {
        trees.push(parse_tree(idx, dict)?);
    }

    Ok(LGBModel {
        max_feature_idx,
        num_class,
        average_output,
        objective_name,
        objective_params,
        trees,
    })
}

/// Parse one tree dict into an [`LGBTree`] (`lightgbm.cc:293-414`).
fn parse_tree(idx: usize, dict: &HashMap<String, String>) -> Result<LGBTree, LgbError> {
    let where_ = format!("Tree {idx}");
    let num_leaves = parse_num::<i32>(&where_, "num_leaves", require(dict, &where_, "num_leaves")?)?;
    let num_cat = parse_num::<i32>(&where_, "num_cat", require(dict, &where_, "num_cat")?)?;

    if num_leaves < 0 {
        return Err(LgbError::Parse {
            line: where_,
            detail: format!("num_leaves must be non-negative, got {num_leaves}"),
        });
    }
    let n_leaf = num_leaves as usize;
    // Internal-node arrays have length num_leaves - 1 (clamped to 0 for a
    // single-leaf / empty tree, mirroring the `num_leaves <= 1` upstream branch).
    let n_internal = n_leaf.saturating_sub(1);

    // leaf_value: length num_leaves, required, f64 (lightgbm.cc:305-308).
    let leaf_value: Vec<f64> =
        parse_array(&where_, "leaf_value", require(dict, &where_, "leaf_value")?, n_leaf)?;

    // decision_type: length num_leaves-1; default-zero when absent and
    // num_leaves > 1 (lightgbm.cc:310-322). i8.
    let decision_type: Vec<i8> = if n_internal == 0 {
        Vec::new()
    } else {
        match dict.get("decision_type") {
            None => vec![0i8; n_internal],
            Some(v) => parse_array(&where_, "decision_type", v, n_internal)?,
        }
    };

    // Categorical arrays (lightgbm.cc:324-333). LGB-02; parsed for completeness.
    let (cat_boundaries, cat_threshold) = if num_cat > 0 {
        let cb: Vec<u64> = parse_array(
            &where_,
            "cat_boundaries",
            require(dict, &where_, "cat_boundaries")?,
            (num_cat as usize) + 1,
        )?;
        let total = *cb.last().unwrap_or(&0) as usize;
        let ct: Vec<u32> = parse_array(
            &where_,
            "cat_threshold",
            require(dict, &where_, "cat_threshold")?,
            total,
        )?;
        (cb, ct)
    } else {
        (Vec::new(), Vec::new())
    };

    // split_feature / threshold / left_child / right_child: length num_leaves-1,
    // required when num_leaves > 1 (lightgbm.cc:335-413).
    let split_feature: Vec<i32> = if n_internal == 0 {
        Vec::new()
    } else {
        parse_array(&where_, "split_feature", require(dict, &where_, "split_feature")?, n_internal)?
    };
    let threshold: Vec<f64> = if n_internal == 0 {
        Vec::new()
    } else {
        parse_array(&where_, "threshold", require(dict, &where_, "threshold")?, n_internal)?
    };
    let left_child: Vec<i32> = if n_internal == 0 {
        Vec::new()
    } else {
        parse_array(&where_, "left_child", require(dict, &where_, "left_child")?, n_internal)?
    };
    let right_child: Vec<i32> = if n_internal == 0 {
        Vec::new()
    } else {
        parse_array(&where_, "right_child", require(dict, &where_, "right_child")?, n_internal)?
    };

    // split_gain: length num_leaves-1, optional (lightgbm.cc:355-367). f32.
    let split_gain: Vec<f32> = if n_internal == 0 {
        Vec::new()
    } else {
        match dict.get("split_gain") {
            Some(v) if !v.trim().is_empty() => parse_array(&where_, "split_gain", v, n_internal)?,
            _ => Vec::new(),
        }
    };

    // internal_count: length num_leaves-1, optional (lightgbm.cc:369-381). i32.
    let internal_count: Vec<i32> = if n_internal == 0 {
        Vec::new()
    } else {
        match dict.get("internal_count") {
            Some(v) if !v.trim().is_empty() => parse_array(&where_, "internal_count", v, n_internal)?,
            _ => Vec::new(),
        }
    };

    // leaf_count: length num_leaves, optional (lightgbm.cc:383-393). i32.
    let leaf_count: Vec<i32> = if n_leaf == 0 {
        Vec::new()
    } else {
        match dict.get("leaf_count") {
            Some(v) if !v.trim().is_empty() => parse_array(&where_, "leaf_count", v, n_leaf)?,
            _ => Vec::new(),
        }
    };

    Ok(LGBTree {
        num_leaves,
        num_cat,
        leaf_value,
        decision_type,
        cat_boundaries,
        cat_threshold,
        split_feature,
        threshold,
        left_child,
        right_child,
        split_gain,
        internal_count,
        leaf_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VENDORED: &str = "../../treelite-mainline/tests/examples/deep_lightgbm/model.txt";

    fn load_vendored() -> LGBModel {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(VENDORED);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        parse_lightgbm(&text).expect("parse deep_lightgbm/model.txt")
    }

    #[test]
    fn parses_vendored_header_and_counts() {
        let m = load_vendored();
        assert_eq!(m.num_class, 1);
        assert_eq!(m.max_feature_idx, 0);
        assert_eq!(m.objective_name, "regression");
        assert_eq!(m.trees.len(), 1);
        assert_eq!(m.trees[0].num_leaves, 32);
        assert_eq!(m.trees[0].num_cat, 0);
        // Per-field precision: leaf_value/threshold f64, split_gain f32.
        assert_eq!(m.trees[0].leaf_value.len(), 32);
        assert_eq!(m.trees[0].threshold.len(), 31);
        assert_eq!(m.trees[0].split_gain.len(), 31);
        // leaf_value first element is 31 (regression toy labels).
        assert_eq!(m.trees[0].leaf_value[0], 31.0_f64);
    }

    #[test]
    fn malformed_count_returns_typed_error_not_oob() {
        // num_leaves says 4 leaves but leaf_value supplies only 2 → typed error.
        let bad = "num_class=1\nmax_feature_idx=0\nobjective=regression\nTree=0\nnum_leaves=4\nnum_cat=0\nleaf_value=1 2\n";
        let err = parse_lightgbm(bad).unwrap_err();
        assert!(matches!(err, LgbError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn missing_required_global_key_errors() {
        // No max_feature_idx → typed Parse error.
        let bad = "num_class=1\nobjective=regression\n";
        assert!(matches!(parse_lightgbm(bad), Err(LgbError::Parse { .. })));
    }
}
