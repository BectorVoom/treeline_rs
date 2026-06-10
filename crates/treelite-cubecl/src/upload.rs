//! Per-column ragged-SoA host→device upload (Wave 2 / plan 06-03).
//!
//! Placeholder module. Wave 2 fills in the per-column concatenation across the
//! forest (one device handle per column, no per-tree handle explosion), the
//! parallel `tree_node_offset`/`tree_leafvec_offset` prefix-sum index, and the
//! `as_bytes()` → `bytemuck::cast_slice` → upload path (GPU-05 zero-copy).
//
// cubecl 0.10.0 API: upload=ComputeClient::create_from_slice(&[u8]) -> Handle (zero-copy SoA, cubecl-runtime client.rs:287; create(Bytes) at :345 is the owned variant), exp2=Float::exp2(self) -> Self (base-2 exp intrinsic, cubecl-core typemap.rs:680 — direct, no exp(x*ln2)/powf identity needed). Retires RESEARCH assumptions A1/A3.
