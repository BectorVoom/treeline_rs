//! Unit test of `run_equivalence` against a HAND-COMPUTED scalar model — no
//! dependency on `fixtures/golden.json`.
//!
//! This is the safety net for the core-value assertion: it proves
//! `run_equivalence` actually CATCHES a `> 1e-5` deviation (the 1e-5 assertion
//! fires) rather than being merely build-checked. A wrong pipeline impl is
//! caught HERE, not deferred to the golden spine test.
//!
//! ## The hand-authored model
//!
//! A single-tree `binary:logistic` model with `base_score = 0.5`. The
//! version-gated margin transform maps `base_score 0.5` to
//! `-ln(1/0.5 - 1) = -ln(1) = 0.0`, so the added margin is EXACTLY `0.0` and the
//! expected output is a pure sigmoid of the leaf value:
//!
//! ```text
//! root: feature[0] < 0.5 ? left(leaf L_left) : right(leaf L_right)
//! expected[row] = 1.0_f32 / (1.0_f32 + (-L).exp())   // sigmoid(1.0, L + 0.0)
//! ```

use std::io::Write;

/// Hand-authored single-tree `binary:logistic` XGBoost-JSON model.
///
/// Node 0: numerical test on `feature[0]` at `0.5` (default_left=1).
/// Node 1 (left): leaf `L_LEFT`. Node 2 (right): leaf `L_RIGHT`.
const L_LEFT: f32 = -0.75;
const L_RIGHT: f32 = 1.25;

fn model_json() -> String {
    format!(
        r#"{{
  "learner": {{
    "learner_model_param": {{
      "num_feature": "1",
      "num_class": "0",
      "num_target": "1",
      "base_score": "5E-1"
    }},
    "gradient_booster": {{
      "name": "gbtree",
      "model": {{
        "trees": [
          {{
            "id": 0,
            "tree_param": {{ "num_nodes": "3", "size_leaf_vector": "0" }},
            "left_children": [1, -1, -1],
            "right_children": [2, -1, -1],
            "parents": [2147483647, 0, 0],
            "split_indices": [0, 0, 0],
            "split_type": [0, 0, 0],
            "split_conditions": [0.5, {l_left}, {l_right}],
            "default_left": [1, 0, 0],
            "loss_changes": [10.0, 0.0, 0.0],
            "sum_hessian": [6.0, 3.0, 3.0],
            "base_weights": [0.0, {l_left}, {l_right}],
            "leaf_child_counts": [0, 0, 0],
            "categories": [],
            "categories_nodes": [],
            "categories_segments": [],
            "categories_sizes": []
          }}
        ],
        "tree_info": [0],
        "gbtree_model_param": {{ "num_trees": "1", "num_parallel_tree": "1" }}
      }}
    }},
    "objective": {{ "name": "binary:logistic" }},
    "attributes": {{}}
  }},
  "version": [4, 7, 0]
}}"#,
        l_left = L_LEFT,
        l_right = L_RIGHT,
    )
}

/// f32 sigmoid of a leaf value with zero added margin (the hand computation).
fn expected_sigmoid(leaf: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-leaf).exp())
}

/// Write the model JSON to a uniquely-named temp file and return its path.
fn write_temp_model(tag: &str) -> String {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "treelite_harness_run_equivalence_{tag}_{}.json",
        std::process::id()
    ));
    let mut f = std::fs::File::create(&path).expect("create temp model file");
    f.write_all(model_json().as_bytes())
        .expect("write temp model file");
    path.to_string_lossy().into_owned()
}

/// Build an in-memory `Golden` whose output is hand-computed for the model.
///
/// Input rows: one routing left (`feature[0] = 0.0 < 0.5`) → `L_LEFT`, one
/// routing right (`feature[0] = 1.0 >= 0.5`) → `L_RIGHT`. Manifest is filled
/// with the running environment so `check_manifest` would not warn (it is not
/// exercised here).
fn build_golden(output: Vec<f32>) -> treelite_harness::Golden {
    let raw = format!(
        r#"{{
  "input": [[0.0], [1.0]],
  "output": [{}, {}],
  "manifest": {{
    "treelite": "4.7.0",
    "xgboost": "3.2.0",
    "os": "test",
    "arch": "test",
    "libc": ["glibc", "0.0"],
    "python": "3.13.0"
  }}
}}"#,
        output[0], output[1]
    );
    serde_json::from_str(&raw).expect("build in-memory golden")
}

#[test]
fn run_equivalence_matches_hand_computed_model() {
    let model_path = write_temp_model("match");
    let golden = build_golden(vec![expected_sigmoid(L_LEFT), expected_sigmoid(L_RIGHT)]);

    let max_dev = treelite_harness::run_equivalence(&model_path, &golden)
        .expect("run_equivalence must succeed against the hand-computed golden");

    assert!(
        max_dev < 1e-5,
        "max observed |delta| ({max_dev:e}) must be < 1e-5 for a matching golden"
    );

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn run_equivalence_catches_perturbation_beyond_1e5() {
    let model_path = write_temp_model("perturb");
    // Perturb the LEFT expected output by 1e-3 (>> 1e-5) — the 1e-5 assertion
    // inside run_equivalence MUST fire, proving a numerically wrong pipeline is
    // caught here, not silently passed.
    let perturbed = expected_sigmoid(L_LEFT) + 1e-3;
    let golden = build_golden(vec![perturbed, expected_sigmoid(L_RIGHT)]);

    let result =
        std::panic::catch_unwind(|| treelite_harness::run_equivalence(&model_path, &golden));

    // `approx::assert_abs_diff_eq!` panics on a >1e-5 mismatch; catch_unwind
    // turns that into an `Err`. Either a panic (Err) or a returned `Err` proves
    // the deviation was caught.
    let caught = match result {
        Err(_) => true,                 // assertion panicked → caught
        Ok(Err(_)) => true,             // returned an error → caught
        Ok(Ok(max_dev)) => {
            panic!(
                "run_equivalence must NOT succeed on a >1e-5 perturbation \
                 (got Ok(max_dev = {max_dev:e})) — the 1e-5 gate failed to fire"
            )
        }
    };
    assert!(caught, "the >1e-5 deviation must be caught");

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn load_golden_on_missing_path_returns_err_with_context() {
    let result = treelite_harness::load_golden("/nonexistent/path/to/golden.json");
    assert!(
        result.is_err(),
        "load_golden on a missing path must return Err (no raw panic)"
    );
    // The anyhow context chain should mention the read step.
    let err = result.unwrap_err();
    let chain = format!("{err:#}");
    assert!(
        chain.contains("golden.json"),
        "error context chain should mention golden.json, got: {chain}"
    );
}
