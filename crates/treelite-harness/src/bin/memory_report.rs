//! `memory_report` — MEM-03 allocator-comparison report generator (D-09/D-10).
//!
//! This bin is the ONLY site in the workspace that installs a custom
//! `#[global_allocator]` (D-08). It is built three ways to compare allocators:
//!   * default (no allocator feature)  → system malloc
//!   * `--features jemalloc`           → tikv-jemallocator
//!   * `--features mimalloc`           → mimalloc
//! The `jemalloc`/`mimalloc` features are non-default and mutually exclusive
//! (D-07); enabling both is a compile error (see the guard below).
//!
//! It loads a small representative benchmark model set (one XGBoost-JSON model +
//! the frozen v5 `binary` / `large_margin` / `lgbm_numerical` serialized models),
//! runs a prediction pass over each to drive allocation, samples peak RSS /
//! bytes-allocated via [`treelite_harness::memory::sample_rss`], and writes the
//! committed observational `docs/MEMORY_REPORT.md` (D-10). The report MERGES the
//! rows for the active allocator into the committed file so, after running once
//! under each allocator, the file shows the before/after (system vs jemalloc vs
//! mimalloc) narrative.
//!
//! There is NO brittle RSS threshold (D-10): the report is observational; the
//! real Phase-9 gate is the `1e-5` equivalence harness + the byte-identical
//! `golden_v5` compare (D-11).
//!
//! Per CLAUDE.md this binary uses `anyhow` for error handling.

use std::path::{Path, PathBuf};

use anyhow::Context;
use treelite_core::Model;
use treelite_gtil::Config;
use treelite_harness::Manifest;
use treelite_harness::memory::{MemRow, SampleMethod, SizeOfRow, emit, sample_rss};

// --- `#[global_allocator]` selection (the ONLY such static in the workspace) ---
// jemalloc when ONLY the jemalloc feature is on.
#[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// mimalloc when ONLY the mimalloc feature is on.
#[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Both-on is a configuration error: the features are mutually exclusive (D-07).
#[cfg(all(feature = "jemalloc", feature = "mimalloc"))]
compile_error!("features `jemalloc` and `mimalloc` are mutually exclusive (D-07)");

/// Name of the allocator active in this build (cfg-resolved at compile time).
const fn active_allocator() -> &'static str {
    #[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
    {
        "jemalloc (tikv-jemallocator)"
    }
    #[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
    {
        "mimalloc"
    }
    #[cfg(not(any(feature = "jemalloc", feature = "mimalloc")))]
    {
        "system (default malloc)"
    }
}

/// The sampling method for the active allocator: jemalloc reads `tikv_jemalloc_ctl`
/// stats (epoch-then-read); mimalloc + the system allocator read `/proc/self/statm`.
const fn active_sample_method() -> SampleMethod {
    #[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
    {
        SampleMethod::Jemalloc
    }
    #[cfg(not(all(feature = "jemalloc", not(feature = "mimalloc"))))]
    {
        SampleMethod::Statm
    }
}

/// The `model_invariants` Wave-0 `size_of::<Model>()` budget (Pitfall-2 guard).
/// Mirrored here so the report's structural-size row carries the same budget.
const SIZE_OF_MODEL_BUDGET: usize = 512;

/// Resolve a path under the workspace-root `fixtures/` dir (gtil_matrix_gpu.rs form).
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
}

/// Resolve a path under the workspace-root `docs/` dir (gtil_matrix_gpu.rs:111-115).
fn docs_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs")
        .join(name)
}

/// A loaded benchmark model + a representative dense f32 input that exercises a
/// prediction pass (to drive allocation before the sample point).
struct Bench {
    name: &'static str,
    model: Model,
}

/// Load the small representative benchmark set (Claude's discretion per D-06/D-10):
/// one XGBoost-JSON model (the loader path) + three frozen v5 serialized models
/// (`binary`, `large_margin`, `lgbm_numerical` — a LightGBM-derived model),
/// covering the loader and the deserialize paths without the dev-only
/// LightGBM/sklearn crates (which the bin cannot depend on).
fn load_benchmarks() -> anyhow::Result<Vec<Bench>> {
    let mut benches = Vec::new();

    // 1) XGBoost-JSON via the loader (the import path).
    let xgb_json = std::fs::read_to_string(fixture_path("xgb_3format.json"))
        .context("reading xgb_3format.json")?;
    let xgb_model = treelite_xgboost::load_xgboost_json(&xgb_json)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("loading xgb_3format.json")?;
    benches.push(Bench {
        name: "xgb_3format (XGBoost JSON)",
        model: xgb_model,
    });

    // 2-4) Frozen v5 serialized models via the deserialize path.
    for (label, file) in [
        ("binary (v5)", "gtil/binary.model.bin"),
        ("large_margin (v5)", "gtil/large_margin.model.bin"),
        ("lgbm_numerical (v5, LightGBM-derived)", "gtil/lgbm_numerical.model.bin"),
    ] {
        let bytes = std::fs::read(fixture_path(file))
            .with_context(|| format!("reading {file}"))?;
        let model = treelite_core::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("deserializing {file}"))?;
        benches.push(Bench { name: label, model });
    }

    Ok(benches)
}

/// Run a prediction pass over `model` with a synthetic dense f32 input sized to
/// the model's `num_feature`, to DRIVE allocation before the sample point. The
/// numeric result is intentionally discarded — this bin measures memory, not
/// fidelity (the `1e-5` gate lives in the equivalence harness, D-11).
fn drive_prediction(model: &Model) -> anyhow::Result<()> {
    let num_feature = model.num_feature.max(0) as usize;
    // A modest synthetic batch: enough rows to exercise the predict allocation
    // path without dominating the process RSS.
    let num_row = 256usize;
    let data: Vec<f32> = (0..num_row * num_feature.max(1))
        .map(|i| (i % 17) as f32 * 0.5 - 4.0)
        .collect();
    // Discard the output; a typed error is surfaced (never silently swallowed).
    let _ = treelite_gtil::predict::<f32>(model, &data, num_row, &Config::default())
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("driving a prediction pass for allocation")?;
    Ok(())
}

/// Build the run-provenance [`Manifest`] for the report header (os/arch/rustc).
fn build_manifest() -> Manifest {
    Manifest {
        treelite: "4.7.0".to_string(),
        xgboost: None,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        libc: serde_json::Value::Null,
        python: None,
        backend: "scalar-cpu".to_string(),
        rustc: option_env!("RUSTC_VERSION_AT_BUILD").map(str::to_string),
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

fn main() -> anyhow::Result<()> {
    let allocator = active_allocator();
    let method = active_sample_method();
    println!("memory_report: active global allocator = {allocator}");

    let benches = load_benchmarks()?;

    let mut rows: Vec<MemRow> = Vec::with_capacity(benches.len());
    for b in &benches {
        // Drive allocation, then sample under the active allocator.
        drive_prediction(&b.model)?;
        let sample = sample_rss(method)
            .with_context(|| format!("sampling RSS for {}", b.name))?;
        println!(
            "  {name:<40}  RSS={rss} bytes  allocated={alloc:?}",
            name = b.name,
            rss = sample.peak_rss_bytes,
            alloc = sample.bytes_allocated,
        );
        rows.push(MemRow {
            model: b.name.to_string(),
            allocator: allocator.to_string(),
            peak_rss_bytes: sample.peak_rss_bytes,
            bytes_allocated: sample.bytes_allocated,
            method: method.label().to_string(),
        });
    }

    let size_of = SizeOfRow {
        size_of_model_bytes: std::mem::size_of::<Model>(),
        budget_bytes: SIZE_OF_MODEL_BUDGET,
    };

    let manifest = build_manifest();
    let report_path = docs_path("MEMORY_REPORT.md");

    // Merge this allocator's rows into the committed report so running once under
    // each allocator assembles the before/after (system vs jemalloc vs mimalloc)
    // narrative in one committed file (D-10).
    let merged = merge_existing_rows(&report_path, allocator, rows)?;
    emit(&merged, size_of, &manifest, &report_path)?;

    println!(
        "memory_report: wrote {} ({} rows for `{allocator}`)",
        report_path.display(),
        merged
            .iter()
            .filter(|r| r.allocator == allocator)
            .count()
    );
    Ok(())
}

/// Merge the freshly-sampled `new_rows` (all for `allocator`) with the rows of
/// OTHER allocators already present in a previously-committed report, so the
/// committed `MEMORY_REPORT.md` accumulates one block per allocator across the
/// three runs (system → jemalloc → mimalloc).
///
/// Rows for the CURRENT `allocator` are replaced by `new_rows` (a re-run refreshes
/// its own block); rows for other allocators are preserved verbatim. When no
/// prior report exists this is just `new_rows`. The parse is best-effort: an
/// unreadable/garbled prior report degrades to "just the new rows" rather than
/// failing the run (the report is observational, D-10).
fn merge_existing_rows(
    report_path: &Path,
    allocator: &str,
    new_rows: Vec<MemRow>,
) -> anyhow::Result<Vec<MemRow>> {
    let mut preserved = parse_existing_rows(report_path)
        .into_iter()
        .filter(|r| r.allocator != allocator)
        .collect::<Vec<_>>();
    preserved.extend(new_rows);
    // Stable order: group by allocator name so the committed table reads as
    // contiguous per-allocator blocks.
    preserved.sort_by(|a, b| a.allocator.cmp(&b.allocator).then(a.model.cmp(&b.model)));
    Ok(preserved)
}

/// Best-effort parse of the per-model rows out of a previously-committed
/// `MEMORY_REPORT.md`. Reads only the main `Model | Allocator | …` table; any
/// non-data line (header, banner, the `size_of` sub-table) is skipped. Returns an
/// empty vec when the file is absent or unparseable — never an error (D-10).
fn parse_existing_rows(report_path: &Path) -> Vec<MemRow> {
    let Ok(text) = std::fs::read_to_string(report_path) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        // Data rows are pipe-delimited with exactly the 5 MEM-03 columns.
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        if cells.len() != 5 {
            continue;
        }
        // Skip the header row and the markdown separator row.
        if cells[0] == "Model" || cells[0].starts_with("---") {
            continue;
        }
        // Skip the size_of sub-table (its first column is the metric name, not a
        // model, and its 2nd column is a plain integer — distinguish by parsing
        // the RSS cell, which is "<bytes> (<n> KiB/MiB)").
        let Some(peak_rss_bytes) = parse_leading_u64(cells[2]) else {
            continue;
        };
        let bytes_allocated = parse_leading_u64(cells[3]);
        rows.push(MemRow {
            model: cells[0].to_string(),
            allocator: cells[1].to_string(),
            peak_rss_bytes,
            bytes_allocated,
            method: cells[4].to_string(),
        });
    }
    rows
}

/// Parse the leading integer out of a `"12345 (12.06 KiB)"` cell. Returns `None`
/// for the `"n/a (statm path)"` / non-numeric cells.
fn parse_leading_u64(cell: &str) -> Option<u64> {
    cell.split_whitespace().next()?.parse().ok()
}
