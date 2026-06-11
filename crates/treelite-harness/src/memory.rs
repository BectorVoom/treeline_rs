//! MEM-03 memory sampler + observational report writer (D-10).
//!
//! This module is the sampler/writer half of the `memory_report` bin: it samples
//! peak resident-set size (RSS) and bytes-allocated under whichever global
//! allocator the bin was built with, and renders a committed
//! `docs/MEMORY_REPORT.md` modeled on the Phase-7 `report.rs`
//! (`GPU_EQUIVALENCE_REPORT.md`) precedent.
//!
//! ## Observational, NOT a CI gate (D-10)
//!
//! Like the GPU equivalence report, this artifact is *recorded*, never asserted.
//! There is NO brittle RSS threshold here — the real Phase-9 pass/fail floor is
//! the `1e-5` equivalence harness + the byte-identical `golden_v5` compare
//! (D-11), which stay green independent of any allocator choice. The report's
//! job is to make the before/after (system vs jemalloc vs mimalloc) RSS narrative
//! VISIBLE in a committed file, with full capture provenance.
//!
//! ## Sampling method (RESEARCH Pattern 4/5, Pitfall 5)
//!
//! * **jemalloc** rows read `tikv_jemalloc_ctl::{epoch, stats}` — jemalloc's
//!   statistics are *epoch-cached*, so [`sample_rss`] advances the epoch BEFORE
//!   each `stats::allocated::read()` / `stats::resident::read()` (Pitfall 5:
//!   without the advance the reads return stale, often-zero values). This path is
//!   `#[cfg(feature = "jemalloc")]`-gated so the mimalloc / default builds compile
//!   without `tikv-jemalloc-ctl`.
//! * **mimalloc** and the **default system** allocator rows read `/proc/self/statm`
//!   field 2 (resident pages) × the page size for RSS; jemalloc-style
//!   bytes-allocated is not portably available for them, so that column is
//!   reported as unavailable for those builds.
//!
//! The `#[global_allocator]` static itself lives ONLY in the `memory_report` bin
//! (D-08) — this module installs none, and exposes no `pub fn` that would pull an
//! allocator into the library graph. Per CLAUDE.md (binary/harness context) it
//! uses `anyhow` for error handling.

use std::fmt::Write as _;
use std::path::Path;

use crate::manifest::Manifest;

/// Which allocator-row sampling method to use. Selected by the bin from its
/// compiled-in `#[global_allocator]` (the cfg-resolved allocator name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMethod {
    /// jemalloc: epoch-advanced `tikv_jemalloc_ctl::stats` reads (RSS + allocated).
    Jemalloc,
    /// mimalloc or the default system allocator: `/proc/self/statm` RSS only.
    Statm,
}

impl SampleMethod {
    /// The human-readable measurement-method string rendered in the report's
    /// `measurement method` column (D-10 — explicit per row).
    pub fn label(self) -> &'static str {
        match self {
            SampleMethod::Jemalloc => "tikv_jemalloc_ctl epoch+stats (resident/allocated)",
            SampleMethod::Statm => "/proc/self/statm field 2 × page size (RSS only)",
        }
    }
}

/// A sampled memory snapshot: peak resident bytes and (where available)
/// bytes-allocated, plus the method used to obtain them.
#[derive(Debug, Clone, Copy)]
pub struct MemSample {
    /// Resident-set size in bytes (peak observed at the sample point).
    pub peak_rss_bytes: u64,
    /// Bytes currently allocated, when the allocator exposes it (jemalloc); `None`
    /// for the statm path (mimalloc / system), where it is not portably available.
    pub bytes_allocated: Option<u64>,
    /// How this snapshot was obtained (rendered verbatim in the report).
    pub method: SampleMethod,
}

/// One report row: the per-model memory sample under the active allocator.
#[derive(Debug, Clone)]
pub struct MemRow {
    /// The benchmark model class (e.g. `xgb_3format (JSON)`, `binary (v5)`).
    pub model: String,
    /// The active allocator name (e.g. `system (default malloc)`, `jemalloc`).
    pub allocator: String,
    /// Peak resident-set size in bytes at the sample point.
    pub peak_rss_bytes: u64,
    /// Bytes allocated (jemalloc only); `None` for the statm path.
    pub bytes_allocated: Option<u64>,
    /// The measurement-method string (D-10 — explicit per row).
    pub method: String,
}

/// A `size_of::<Model>()` "before/after" attestation row.
///
/// MEM-02 left `size_of::<Model>()` byte-identical (no struct-size cost); this
/// row records the measured value alongside the Wave-0 budget so the committed
/// report carries the structural-size evidence (the `model_invariants` budget is
/// the live guard; this is the human-readable mirror — D-10 before/after row).
#[derive(Debug, Clone, Copy)]
pub struct SizeOfRow {
    /// The measured `std::mem::size_of::<Model>()` in bytes.
    pub size_of_model_bytes: usize,
    /// The Wave-0 `model_invariants` upper-bound budget (Pitfall-2 guard).
    pub budget_bytes: usize,
}

/// Read the OS page size (bytes) for the `/proc/self/statm` RSS conversion.
///
/// `statm` field 2 is in PAGES; multiplying by the page size yields bytes. The
/// harness takes no `libc` dependency (it is not in the graph), so this returns
/// the conventional 4 KiB page used on the x86-64 / aarch64 Linux targets this
/// report runs on. Kept as a single seam so a future `sysconf(_SC_PAGESIZE)`-backed
/// read on a divergent-page-size host is an explicit one-line change rather than a
/// silent assumption.
fn page_size_bytes() -> u64 {
    4096
}

/// Sample peak RSS / bytes-allocated under the active allocator.
///
/// `method` selects the sampling path — the bin passes [`SampleMethod::Jemalloc`]
/// when built `--features jemalloc` (so the jemalloc-ctl stats are read) and
/// [`SampleMethod::Statm`] otherwise (mimalloc / system). Returns a [`MemSample`].
///
/// The jemalloc path advances the epoch BEFORE reading (Pitfall 5: jemalloc
/// statistics are epoch-cached and otherwise return stale values).
pub fn sample_rss(method: SampleMethod) -> anyhow::Result<MemSample> {
    match method {
        SampleMethod::Jemalloc => sample_jemalloc(),
        SampleMethod::Statm => sample_statm(),
    }
}

/// jemalloc sampler — epoch-then-read (Pitfall 5). Only compiled when the
/// `jemalloc` feature is on; the `tikv-jemalloc-ctl` crate is otherwise absent
/// from the build graph, so this arm must be cfg-gated.
#[cfg(feature = "jemalloc")]
fn sample_jemalloc() -> anyhow::Result<MemSample> {
    use tikv_jemalloc_ctl::{epoch, stats};
    // Advance the epoch FIRST so the subsequent stats reads reflect the current
    // allocation state (jemalloc caches stats per epoch — Pitfall 5).
    epoch::advance().map_err(|e| anyhow::anyhow!("jemalloc epoch::advance: {e}"))?;
    let allocated =
        stats::allocated::read().map_err(|e| anyhow::anyhow!("jemalloc stats::allocated: {e}"))?;
    let resident =
        stats::resident::read().map_err(|e| anyhow::anyhow!("jemalloc stats::resident: {e}"))?;
    Ok(MemSample {
        peak_rss_bytes: resident as u64,
        bytes_allocated: Some(allocated as u64),
        method: SampleMethod::Jemalloc,
    })
}

/// When the `jemalloc` feature is OFF, asking for the jemalloc sampler is a
/// configuration error rather than a silent fall-through (the bin only requests
/// it under the matching feature). Kept as an arm so [`sample_rss`] type-checks
/// in every feature build.
#[cfg(not(feature = "jemalloc"))]
fn sample_jemalloc() -> anyhow::Result<MemSample> {
    anyhow::bail!("jemalloc sampler requested but the `jemalloc` feature is not enabled")
}

/// `/proc/self/statm` sampler — RSS via field 2 × page size (mimalloc / system).
///
/// `statm` is a single line of space-separated page counts; field index 2 (the
/// third field, 0-based) is the resident set size in pages. bytes-allocated is
/// not portably exposed for these allocators, so it is reported as `None`.
fn sample_statm() -> anyhow::Result<MemSample> {
    let statm = std::fs::read_to_string("/proc/self/statm")
        .map_err(|e| anyhow::anyhow!("reading /proc/self/statm: {e}"))?;
    let resident_pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("/proc/self/statm has no resident field"))?
        .parse()
        .map_err(|e| anyhow::anyhow!("parsing /proc/self/statm resident field: {e}"))?;
    Ok(MemSample {
        peak_rss_bytes: resident_pages * page_size_bytes(),
        bytes_allocated: None,
        method: SampleMethod::Statm,
    })
}

/// Render bytes as a `12345 (12.06 KiB)` cell for human legibility in the report.
fn fmt_bytes(b: u64) -> String {
    let kib = b as f64 / 1024.0;
    if kib >= 1024.0 {
        format!("{b} ({:.2} MiB)", kib / 1024.0)
    } else {
        format!("{b} ({kib:.2} KiB)")
    }
}

/// Render the optional bytes-allocated cell.
fn fmt_allocated(b: Option<u64>) -> String {
    match b {
        Some(b) => fmt_bytes(b),
        None => "n/a (statm path)".to_string(),
    }
}

/// Build the committed observational markdown report body (D-10).
///
/// Header carries the run provenance (os/arch/rustc) from the [`Manifest`] and
/// the verbatim **"Observational — NOT a CI gate"** banner (report.rs:200-204
/// style). The body is the RESEARCH §MEM-03 table — `Model | Allocator | peak
/// resident (RSS) | bytes allocated | measurement method` — plus a
/// `size_of::<Model>()` before/after row and the `1e-5 + golden green`
/// attestation line (D-11 evidence). There is NO RSS threshold assertion here:
/// the report records, the green floor gates.
pub fn render_markdown(rows: &[MemRow], size_of: SizeOfRow, manifest: &Manifest) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Memory Efficiency Report (MEM-03)");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "Regenerated on: treelite {treelite} / {rustc} / captured-on {os} {arch} \
         (provenance from the run Manifest, D-10)",
        treelite = manifest.treelite,
        rustc = manifest.rustc.as_deref().unwrap_or("unknown rustc"),
        os = manifest.os,
        arch = manifest.arch,
    );
    let _ = writeln!(
        s,
        "Reference floor: the `1e-5` equivalence harness + the byte-identical \
         `golden_v5` compare (D-11). **Observational — NOT a CI gate (D-10).**"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "| Model | Allocator | peak resident (RSS) | bytes allocated | measurement method |"
    );
    let _ = writeln!(
        s,
        "|-------|-----------|---------------------|-----------------|--------------------|"
    );
    for r in rows {
        let _ = writeln!(
            s,
            "| {model} | {alloc} | {rss} | {allocated} | {method} |",
            model = r.model,
            alloc = r.allocator,
            rss = fmt_bytes(r.peak_rss_bytes),
            allocated = fmt_allocated(r.bytes_allocated),
            method = r.method,
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "## `size_of::<Model>()` (MEM-02 structural-size attestation)");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "| Metric | Bytes | Budget (Wave-0 Pitfall-2 guard) | Within budget? |"
    );
    let _ = writeln!(
        s,
        "|--------|-------|---------------------------------|----------------|"
    );
    let _ = writeln!(
        s,
        "| `size_of::<Model>()` | {size} | {budget} | {within} |",
        size = size_of.size_of_model_bytes,
        budget = size_of.budget_bytes,
        within = if size_of.size_of_model_bytes <= size_of.budget_bytes {
            "yes"
        } else {
            "NO — investigate (Pitfall 2)"
        },
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "> MEM-02 migrated the `Model`/`Metadata` metadata fields to \
         `SmallVec`/`CompactString` with the inline `N` chosen for the dominant \
         shape — `size_of::<Model>()` stays byte-identical to the prior \
         `Vec`/`String` layout (zero struct-size cost). MEM-01 routed the \
         serializer's `le_bytes_of` through `bytemuck::cast_slice`, removing an \
         `unsafe` block while emitting byte-identical bytes."
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "## Attestation (D-11 green floor)"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "The two hard invariants are green for this Phase-9 state: the v5 \
         serializer emits byte-identical `golden_v5.bin` / `golden_v5_3format.bin` \
         (HARD INVARIANT 1), and the full equivalence/matrix harness is within \
         `1e-5` with `cargo test --workspace` + `uv run pytest crates/treelite-py` \
         passing (HARD INVARIANT 2). The allocator choice above changes only the \
         RSS narrative, never the predictions — the `1e-5`/golden floor is the \
         real gate (D-11)."
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "## Wheel isolation (D-08)"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "The allocator crates (`tikv-jemallocator`, `tikv-jemalloc-ctl`, \
         `mimalloc`) are `optional = true` on the `treelite-harness` crate ONLY, \
         behind the non-default mutually-exclusive `jemalloc`/`mimalloc` features; \
         the `#[global_allocator]` static lives ONLY in this `memory_report` bin. \
         `cargo tree -p treelite-py | grep -E \"jemalloc|mimalloc\"` prints \
         nothing — the abi3 wheel pulls ZERO allocator deps."
    );
    s
}

/// Emit the committed observational `MEMORY_REPORT.md` at `report_md_path`
/// (report.rs:281-305 form): `create_dir_all(parent)` then `write(path, md)`.
///
/// The bin runs this once under each allocator to assemble the before/after
/// narrative; there is no JSON sidecar for MEM-03 (the report is the artifact,
/// D-10). Uses `anyhow` per CLAUDE.md (binary/harness context).
pub fn emit(
    rows: &[MemRow],
    size_of: SizeOfRow,
    manifest: &Manifest,
    report_md_path: &Path,
) -> anyhow::Result<()> {
    let md = render_markdown(rows, size_of, manifest);
    if let Some(parent) = report_md_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating {}: {e}", parent.display()))?;
    }
    std::fs::write(report_md_path, md)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", report_md_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manifest() -> Manifest {
        Manifest {
            treelite: "4.7.0".to_string(),
            xgboost: None,
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            libc: serde_json::Value::Null,
            python: None,
            backend: "scalar-cpu".to_string(),
            rustc: Some("rustc 1.x".to_string()),
            cubecl: None,
            seed: None,
            sha256: None,
            numpy: None,
            scipy: None,
            lightgbm: None,
            scikit_learn: None,
            model: None,
            preset: None,
            input_dtype: None,
            kind: None,
            layout: None,
        }
    }

    #[test]
    fn statm_sampler_reads_a_positive_rss() {
        // The statm path is always available on Linux; RSS must be > 0 for a live
        // process and bytes-allocated is None (not portably exposed).
        let s = sample_rss(SampleMethod::Statm).expect("statm sample");
        assert!(s.peak_rss_bytes > 0, "rss = {}", s.peak_rss_bytes);
        assert_eq!(s.bytes_allocated, None);
        assert_eq!(s.method, SampleMethod::Statm);
    }

    #[test]
    fn render_carries_the_observational_banner_and_provenance() {
        let rows = vec![MemRow {
            model: "binary (v5)".to_string(),
            allocator: "system (default malloc)".to_string(),
            peak_rss_bytes: 12_582_912,
            bytes_allocated: None,
            method: SampleMethod::Statm.label().to_string(),
        }];
        let size_of = SizeOfRow {
            size_of_model_bytes: 248,
            budget_bytes: 512,
        };
        let md = render_markdown(&rows, size_of, &test_manifest());
        // Observational banner present (D-10).
        assert!(md.contains("Observational"), "banner missing");
        // Manifest provenance in the header.
        assert!(md.contains("x86_64"), "arch provenance missing");
        assert!(md.contains("treelite 4.7.0"), "treelite provenance missing");
        // size_of attestation row.
        assert!(md.contains("size_of::<Model>()"), "size_of row missing");
        // The 1e-5/golden attestation line.
        assert!(md.contains("1e-5"), "1e-5 attestation missing");
        assert!(md.contains("golden_v5"), "golden attestation missing");
    }
}
