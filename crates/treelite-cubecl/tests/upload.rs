//! RED scaffold — Wave 2 per-column ragged-SoA upload round-trip (plan 06-03).
//!
//! Nyquist MISSING marker: asserts the host→device upload (per-column
//! concatenation across the forest, one handle per column, the
//! `tree_node_offset`/`tree_leafvec_offset` prefix-sum index) round-trips
//! byte-exact via `TreeBuf::as_bytes()` → `bytemuck::cast_slice` →
//! `client.create_from_slice` → `client.read` (GPU-05 zero-copy).

#[test]
#[ignore = "MISSING — Wave 2: per-column ragged SoA upload round-trip"]
fn upload_ragged_soa_roundtrip() {
    todo!("Wave 2 (plan 06-03): per-column concat + offset index + as_bytes upload round-trip");
}
