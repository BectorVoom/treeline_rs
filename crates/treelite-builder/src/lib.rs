//! `treelite-builder` — the validated model-construction layer (D-10).
//!
//! Ports `treelite-mainline/src/model_builder/model_builder.cc` (the fluent
//! [`ModelBuilder`] state machine), `treelite-mainline/src/model_concat.cc`
//! (the [`concat::concatenate`] free function), and the `BulkConstructTree`
//! fast path in `treelite-mainline/src/model_loader/sklearn_bulk.cc`
//! ([`bulk::bulk_construct_tree`]).
//!
//! All three semantics are ported verbatim from the vendored upstream C++ —
//! fidelity over invention. Every upstream `TREELITE_CHECK*` / `TREELITE_LOG(FATAL)`
//! becomes a returned [`BuilderError`] carrying the offending key/index (D-07),
//! never an abort or out-of-bounds index.

pub mod bulk;
pub mod concat;
pub mod error;

pub use error::BuilderError;

use std::collections::BTreeMap;

use treelite_core::enums::{Operator, TaskType, TreeNodeType};
use treelite_core::{Model, ModelPreset, ModelVariant, Tree, TreeBuf};

/// The 5-state machine gating which methods are legal at each point
/// (`model_builder.cc:50-56`, RESEARCH Pattern 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuilderState {
    /// Expect `start_tree()` or `commit_model()`.
    ExpectTree,
    /// Expect `start_node()` or `end_tree()`.
    ExpectNode,
    /// Expect a node detail: `numerical_test`/`categorical_test`/`leaf_*`/`gain`/`data_count`/`sum_hess`.
    ExpectDetail,
    /// Node detail supplied; expect `end_node()` (or further `gain`/`data_count`/`sum_hess`).
    NodeComplete,
    /// `commit_model()` has produced the final model; no further calls legal.
    ModelComplete,
}

/// Metadata supplied at construction (subset of upstream `Metadata` +
/// `TreeAnnotation`, `model_builder.cc:336-388`).
///
/// `expected_num_tree` is `tree_annotation.num_tree`; `expected_leaf_size` is the
/// product of `leaf_vector_shape` (the leaf-output length expected of every leaf).
#[derive(Debug, Clone)]
pub struct BuilderMetadata {
    /// Number of input features; gates `split_index` range checks.
    pub num_feature: i32,
    /// Prediction task kind.
    pub task_type: TaskType,
    /// Whether tree outputs are averaged.
    pub average_tree_output: bool,
    /// Number of targets.
    pub num_target: i32,
    /// Per-target class counts (`tree.h:543`).
    pub num_class: Vec<i32>,
    /// Leaf-vector shape (`tree.h:544`); product is the expected leaf size.
    pub leaf_vector_shape: Vec<i32>,
    /// Per-tree target id.
    pub target_id: Vec<i32>,
    /// Per-tree class id.
    pub class_id: Vec<i32>,
    /// Postprocessor name.
    pub postprocessor: String,
    /// Margin-transformed base scores.
    pub base_scores: Vec<f64>,
    /// Free-form attributes blob (`"{}"` when omitted).
    pub attributes: Option<String>,
}

// Raw per-node staging columns for the tree currently under construction.
// Children are stored as RAW user keys (no resolution yet); resolved to internal
// indices at `end_tree` (`model_builder.cc:104-153`, RESEARCH Pitfall 6).
struct NodeStaging {
    node_type: TreeNodeType,
    // raw child keys (user-defined); -1 marks a leaf
    raw_left: i32,
    raw_right: i32,
    split_index: i32,
    default_left: bool,
    leaf_value: f32,
    threshold: f32,
    cmp: Operator,
    data_count: u64,
    data_count_present: bool,
    sum_hess: f64,
    sum_hess_present: bool,
    gain: f64,
    gain_present: bool,
    // true once the node was given a leaf OR test detail (state guard backstop)
    detail_set: bool,
    is_leaf: bool,
}

impl NodeStaging {
    fn new() -> Self {
        NodeStaging {
            node_type: TreeNodeType::kLeafNode,
            raw_left: -1,
            raw_right: -1,
            split_index: -1,
            default_left: false,
            leaf_value: 0.0,
            threshold: 0.0,
            cmp: Operator::kNone,
            data_count: 0,
            data_count_present: false,
            sum_hess: 0.0,
            sum_hess_present: false,
            gain: 0.0,
            gain_present: false,
            detail_set: false,
            is_leaf: false,
        }
    }
}

/// A fluent, always-strict model builder (BLD-01, D-07/D-08).
///
/// Ports `ModelBuilderImpl` (`model_builder.cc:58-389`). Validates per-node
/// well-formedness eagerly (at the detail call / `end_node`) and tree topology
/// (orphans, dangling child keys via forward-reference resolution) at
/// [`Self::end_tree`]. The orphan check is ALWAYS on — the upstream validation
/// toggle for orphan checking is intentionally NOT ported (D-08).
///
/// Only the `<f32, f32>` preset is produced in Phase 2 (the variant XGBoost
/// yields); the threshold/leaf type is `f32`.
pub struct ModelBuilder {
    state: BuilderState,
    metadata: Option<BuilderMetadata>,
    expected_num_tree: usize,
    expected_leaf_size: usize,

    // accumulated finished trees
    trees: Vec<Tree<f32>>,

    // current tree under construction
    // node_id_map: user key -> internal index (declaration order). BTreeMap to
    // mirror upstream `std::map` deterministic iteration for orphan-error key
    // selection (RESEARCH Pattern 1).
    node_id_map: BTreeMap<i32, i32>,
    nodes: Vec<NodeStaging>,
    current_node_key: i32,
    current_node_id: i32,
}

impl ModelBuilder {
    /// Construct a builder with metadata already initialized
    /// (`model_builder.cc:73-87`).
    pub fn new(metadata: BuilderMetadata) -> Result<Self, BuilderError> {
        let mut b = ModelBuilder::empty();
        b.initialize_metadata(metadata)?;
        Ok(b)
    }

    /// Construct a builder with NO metadata yet (`model_builder.cc:61-71`).
    /// `initialize_metadata` must be called before `commit_model`.
    pub fn empty() -> Self {
        ModelBuilder {
            state: BuilderState::ExpectTree,
            metadata: None,
            expected_num_tree: 0,
            expected_leaf_size: 0,
            trees: Vec::new(),
            node_id_map: BTreeMap::new(),
            nodes: Vec::new(),
            current_node_key: 0,
            current_node_id: 0,
        }
    }

    /// Initialize metadata (`model_builder.cc:336-388`). May be called once.
    pub fn initialize_metadata(&mut self, metadata: BuilderMetadata) -> Result<(), BuilderError> {
        // expected_leaf_size = product of leaf_vector_shape, starting at 1
        // (`model_builder.cc:385-386`).
        let expected_leaf_size: i32 = metadata
            .leaf_vector_shape
            .iter()
            .copied()
            .fold(1_i32, |acc, x| acc.saturating_mul(x));
        self.expected_num_tree = metadata.target_id.len();
        self.expected_leaf_size = expected_leaf_size.max(0) as usize;
        self.metadata = Some(metadata);
        Ok(())
    }

    /// Begin a tree (`model_builder.cc:95-102`). Legal only in `ExpectTree`.
    pub fn start_tree(&mut self) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectTree],
            "start_tree() or commit_model()",
        )?;
        self.node_id_map.clear();
        self.nodes.clear();
        self.state = BuilderState::ExpectNode;
        Ok(())
    }

    /// Begin a node with user key `node_key` (`model_builder.cc:155-166`).
    /// Rejects negative and duplicate keys.
    pub fn start_node(&mut self, node_key: i32) -> Result<(), BuilderError> {
        self.check_state(&[BuilderState::ExpectNode], "start_node() or end_tree()")?;
        if node_key < 0 {
            return Err(BuilderError::NegativeNodeKey { key: node_key });
        }
        if self.node_id_map.contains_key(&node_key) {
            return Err(BuilderError::DuplicateNodeKey { key: node_key });
        }
        let node_id = self.nodes.len() as i32; // AllocNode (`model_builder.cc:159`)
        self.nodes.push(NodeStaging::new());
        self.node_id_map.insert(node_key, node_id);
        self.current_node_key = node_key;
        self.current_node_id = node_id;
        self.state = BuilderState::ExpectDetail;
        Ok(())
    }

    /// Configure the current node as a numerical test
    /// (`model_builder.cc:173-190`). Children are stored RAW (resolved at
    /// `end_tree`).
    pub fn numerical_test(
        &mut self,
        split_index: i32,
        threshold: f32,
        default_left: bool,
        cmp: Operator,
        left_child_key: i32,
        right_child_key: i32,
    ) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectDetail],
            "numerical_test(), categorical_test(), leaf_scalar(), leaf_vector(), gain(), data_count(), or sum_hess()",
        )?;
        self.validate_test_children(split_index, left_child_key, right_child_key)?;

        let node = &mut self.nodes[self.current_node_id as usize];
        node.node_type = TreeNodeType::kNumericalTestNode;
        node.split_index = split_index;
        node.threshold = threshold;
        node.default_left = default_left;
        node.cmp = cmp;
        node.raw_left = left_child_key;
        node.raw_right = right_child_key;
        node.is_leaf = false;
        node.detail_set = true;

        self.state = BuilderState::NodeComplete;
        Ok(())
    }

    /// Configure the current node as a categorical test
    /// (`model_builder.cc:192-212`). Phase 2 stores the split as a
    /// categorical-test node but does not yet exercise category lists in GTIL;
    /// children are stored RAW like the numerical path.
    pub fn categorical_test(
        &mut self,
        split_index: i32,
        default_left: bool,
        left_child_key: i32,
        right_child_key: i32,
    ) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectDetail],
            "numerical_test(), categorical_test(), leaf_scalar(), leaf_vector(), gain(), data_count(), or sum_hess()",
        )?;
        self.validate_test_children(split_index, left_child_key, right_child_key)?;

        let node = &mut self.nodes[self.current_node_id as usize];
        node.node_type = TreeNodeType::kCategoricalTestNode;
        node.split_index = split_index;
        node.default_left = default_left;
        node.cmp = Operator::kNone;
        node.raw_left = left_child_key;
        node.raw_right = right_child_key;
        node.is_leaf = false;
        node.detail_set = true;

        self.state = BuilderState::NodeComplete;
        Ok(())
    }

    /// Set a scalar leaf output (`model_builder.cc:214-224`). Mutually exclusive
    /// with the test methods via the state machine.
    pub fn leaf_scalar(&mut self, value: f32) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectDetail],
            "numerical_test(), categorical_test(), leaf_scalar(), leaf_vector(), gain(), data_count(), or sum_hess()",
        )?;
        if self.metadata.is_some() && self.expected_leaf_size != 1 {
            return Err(BuilderError::LeafVectorSizeMismatch {
                expected: self.expected_leaf_size,
                got: 1,
            });
        }
        let node = &mut self.nodes[self.current_node_id as usize];
        node.node_type = TreeNodeType::kLeafNode;
        node.raw_left = -1;
        node.raw_right = -1;
        node.leaf_value = value;
        node.cmp = Operator::kNone;
        node.is_leaf = true;
        node.detail_set = true;

        self.state = BuilderState::NodeComplete;
        Ok(())
    }

    /// Set a leaf vector (`model_builder.cc:226-256`). In Phase 2 the leaf-vector
    /// columns are not yet wired into the built `Tree`; this validates the
    /// expected length and marks the node a leaf. Provided for interface
    /// completeness; the XGBoost rewiring (Plan 05) uses `leaf_scalar`.
    pub fn leaf_vector(&mut self, leaf_vector: &[f32]) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectDetail],
            "numerical_test(), categorical_test(), leaf_scalar(), leaf_vector(), gain(), data_count(), or sum_hess()",
        )?;
        if self.metadata.is_some() && self.expected_leaf_size != leaf_vector.len() {
            return Err(BuilderError::LeafVectorSizeMismatch {
                expected: self.expected_leaf_size,
                got: leaf_vector.len(),
            });
        }
        let node = &mut self.nodes[self.current_node_id as usize];
        node.node_type = TreeNodeType::kLeafNode;
        node.raw_left = -1;
        node.raw_right = -1;
        node.cmp = Operator::kNone;
        node.is_leaf = true;
        node.detail_set = true;

        self.state = BuilderState::NodeComplete;
        Ok(())
    }

    /// Set the split gain of the current node (`model_builder.cc:258-263`).
    /// Legal in both `ExpectDetail` and `NodeComplete`.
    pub fn gain(&mut self, gain: f64) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectDetail, BuilderState::NodeComplete],
            "a node detail, gain(), data_count(), sum_hess(), or end_node()",
        )?;
        let node = &mut self.nodes[self.current_node_id as usize];
        node.gain = gain;
        node.gain_present = true;
        Ok(())
    }

    /// Set the data count of the current node (`model_builder.cc:265-270`).
    pub fn data_count(&mut self, data_count: u64) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectDetail, BuilderState::NodeComplete],
            "a node detail, gain(), data_count(), sum_hess(), or end_node()",
        )?;
        let node = &mut self.nodes[self.current_node_id as usize];
        node.data_count = data_count;
        node.data_count_present = true;
        Ok(())
    }

    /// Set the sum of hessians of the current node (`model_builder.cc:272-277`).
    pub fn sum_hess(&mut self, sum_hess: f64) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::ExpectDetail, BuilderState::NodeComplete],
            "a node detail, gain(), data_count(), sum_hess(), or end_node()",
        )?;
        let node = &mut self.nodes[self.current_node_id as usize];
        node.sum_hess = sum_hess;
        node.sum_hess_present = true;
        Ok(())
    }

    /// End the current node (`model_builder.cc:168-171`).
    pub fn end_node(&mut self) -> Result<(), BuilderError> {
        self.check_state(
            &[BuilderState::NodeComplete],
            "end_node(), gain(), data_count(), or sum_hess()",
        )?;
        self.state = BuilderState::ExpectNode;
        Ok(())
    }

    /// End the current tree (`model_builder.cc:104-153`).
    ///
    /// Resolves every non-leaf node's RAW child keys to internal indices via the
    /// node-id map (`DanglingChildKey` on miss), marks reachable children
    /// non-orphaned, then any still-orphaned node → `OrphanedNode`. The orphan
    /// check is ALWAYS on (D-08). Finally finalizes the `Tree<f32>` via the
    /// column-fill → `TreeBuf::from_owned` pattern.
    pub fn end_tree(&mut self) -> Result<(), BuilderError> {
        self.check_state(&[BuilderState::ExpectNode], "start_node() or end_tree()")?;

        let num_nodes = self.nodes.len();
        if num_nodes == 0 {
            return Err(BuilderError::EmptyTree);
        }

        // Resolve raw child keys to internal indices and detect orphans.
        // orphaned[0] = false (root); all others start orphaned
        // (`model_builder.cc:110-111`).
        let mut orphaned = vec![true; num_nodes];
        orphaned[0] = false;

        // Resolved child indices, written back after the resolution pass.
        let mut resolved: Vec<(i32, i32)> = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            if self.nodes[i].is_leaf {
                resolved.push((-1, -1));
                continue;
            }
            let left_key = self.nodes[i].raw_left;
            let right_key = self.nodes[i].raw_right;
            let cleft = *self
                .node_id_map
                .get(&left_key)
                .ok_or(BuilderError::DanglingChildKey { key: left_key })?;
            let cright = *self
                .node_id_map
                .get(&right_key)
                .ok_or(BuilderError::DanglingChildKey { key: right_key })?;
            orphaned[cleft as usize] = false;
            orphaned[cright as usize] = false;
            resolved.push((cleft, cright));
        }

        // Any still-orphaned node → OrphanedNode, keyed by the user key that maps
        // to it (BTreeMap iteration order mirrors upstream `std::map`,
        // `model_builder.cc:133-145`).
        if let Some(orphan_idx) = orphaned.iter().position(|&o| o) {
            for (k, v) in &self.node_id_map {
                if *v == orphan_idx as i32 {
                    return Err(BuilderError::OrphanedNode { key: *k });
                }
            }
            // Fallback: an index with no user key (cannot happen, but stay typed).
            return Err(BuilderError::OrphanedNode {
                key: orphan_idx as i32,
            });
        }

        // Build the Tree<f32> columns (`xgboost::build_tree` structural template).
        let mut node_type = Vec::with_capacity(num_nodes);
        let mut cleft = Vec::with_capacity(num_nodes);
        let mut cright = Vec::with_capacity(num_nodes);
        let mut split_index = Vec::with_capacity(num_nodes);
        let mut default_left = Vec::with_capacity(num_nodes);
        let mut leaf_value = Vec::with_capacity(num_nodes);
        let mut threshold = Vec::with_capacity(num_nodes);
        let mut cmp = Vec::with_capacity(num_nodes);
        let mut data_count = Vec::with_capacity(num_nodes);
        let mut data_count_present = Vec::with_capacity(num_nodes);
        let mut sum_hess = Vec::with_capacity(num_nodes);
        let mut sum_hess_present = Vec::with_capacity(num_nodes);
        let mut gain = Vec::with_capacity(num_nodes);
        let mut gain_present = Vec::with_capacity(num_nodes);

        for (n, &(rl, rr)) in self.nodes.iter().zip(resolved.iter()) {
            node_type.push(n.node_type);
            cleft.push(rl);
            cright.push(rr);
            split_index.push(n.split_index);
            default_left.push(n.default_left);
            leaf_value.push(n.leaf_value);
            threshold.push(n.threshold);
            cmp.push(n.cmp);
            data_count.push(n.data_count);
            data_count_present.push(n.data_count_present);
            sum_hess.push(n.sum_hess);
            sum_hess_present.push(n.sum_hess_present);
            gain.push(n.gain);
            gain_present.push(n.gain_present);
        }

        let mut tree = Tree::<f32>::new();
        tree.node_type = TreeBuf::from_owned(node_type);
        tree.cleft = TreeBuf::from_owned(cleft);
        tree.cright = TreeBuf::from_owned(cright);
        tree.split_index = TreeBuf::from_owned(split_index);
        tree.default_left = TreeBuf::from_owned(default_left);
        tree.leaf_value = TreeBuf::from_owned(leaf_value);
        tree.threshold = TreeBuf::from_owned(threshold);
        tree.cmp = TreeBuf::from_owned(cmp);
        tree.data_count = TreeBuf::from_owned(data_count);
        tree.data_count_present = TreeBuf::from_owned(data_count_present);
        tree.sum_hess = TreeBuf::from_owned(sum_hess);
        tree.sum_hess_present = TreeBuf::from_owned(sum_hess_present);
        tree.gain = TreeBuf::from_owned(gain);
        tree.gain_present = TreeBuf::from_owned(gain_present);
        tree.has_categorical_split = self
            .nodes
            .iter()
            .any(|n| n.node_type == TreeNodeType::kCategoricalTestNode);
        tree.num_nodes = num_nodes as i32;

        self.trees.push(tree);
        self.node_id_map.clear();
        self.nodes.clear();
        self.state = BuilderState::ExpectTree;
        Ok(())
    }

    /// Number of trees committed so far (`GetNumTree`).
    pub fn num_tree(&self) -> usize {
        self.trees.len()
    }

    /// Finalize and produce a `Model` (`model_builder.cc:279-288`).
    ///
    /// Requires metadata initialized and exactly `expected_num_tree` trees built.
    pub fn commit_model(mut self) -> Result<Model, BuilderError> {
        self.check_state(
            &[BuilderState::ExpectTree],
            "start_tree() or commit_model()",
        )?;
        let metadata = self
            .metadata
            .take()
            .ok_or(BuilderError::MetadataNotInitialized)?;
        if self.trees.len() != self.expected_num_tree {
            return Err(BuilderError::CommitTreeCountMismatch {
                expected: self.expected_num_tree,
                got: self.trees.len(),
            });
        }
        self.state = BuilderState::ModelComplete;

        let trees = std::mem::take(&mut self.trees);
        let mut model = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
        model.num_feature = metadata.num_feature;
        model.task_type = metadata.task_type;
        model.average_tree_output = metadata.average_tree_output;
        model.num_target = metadata.num_target;
        model.num_class = metadata.num_class;
        model.leaf_vector_shape = metadata.leaf_vector_shape;
        model.target_id = metadata.target_id;
        model.class_id = metadata.class_id;
        model.postprocessor = metadata.postprocessor;
        model.base_scores = metadata.base_scores;
        model.attributes = metadata.attributes.unwrap_or_else(|| "{}".to_string());
        Ok(model)
    }

    // --- internal helpers ---

    fn check_state(
        &self,
        valid: &[BuilderState],
        expected: &'static str,
    ) -> Result<(), BuilderError> {
        if valid.contains(&self.state) {
            Ok(())
        } else {
            Err(BuilderError::WrongState { expected })
        }
    }

    /// Shared child-key + split-index validation for the two test methods
    /// (`model_builder.cc:176-183,197-203`).
    fn validate_test_children(
        &self,
        split_index: i32,
        left: i32,
        right: i32,
    ) -> Result<(), BuilderError> {
        if left < 0 {
            return Err(BuilderError::NegativeNodeKey { key: left });
        }
        if right < 0 {
            return Err(BuilderError::NegativeNodeKey { key: right });
        }
        if self.current_node_key == left || self.current_node_key == right || left == right {
            return Err(BuilderError::SelfOrEqualChildKey {
                node: self.current_node_key,
            });
        }
        if let Some(meta) = &self.metadata
            && split_index >= meta.num_feature
        {
            return Err(BuilderError::SplitIndexOutOfRange {
                split_index,
                num_feature: meta.num_feature,
            });
        }
        Ok(())
    }
}
