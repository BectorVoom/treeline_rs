//! `memory_report` — MEM-03 allocator-comparison report generator (Wave-0 skeleton).
//!
//! This bin is the ONLY site in the workspace that installs a custom
//! `#[global_allocator]` (D-08). It is built three ways to compare allocators:
//!   * default (no allocator feature)  → system malloc
//!   * `--features jemalloc`           → tikv-jemallocator
//!   * `--features mimalloc`           → mimalloc
//! The `jemalloc`/`mimalloc` features are non-default and mutually exclusive
//! (D-07); enabling both is a compile error (see the guard below).
//!
//! Wave 0 (this plan): the bin only reports which allocator is active and exits
//! `Ok(())`. The real RSS / `bytes-allocated` sampling and the
//! `docs/MEMORY_REPORT.md` write land in Plan 04 (D-10) — they reuse the
//! `report.rs` markdown-writer precedent and `tikv_jemalloc_ctl::{epoch, stats}`
//! for the jemalloc row plus `/proc/self/statm` for the mimalloc/system rows.
//!
//! Per CLAUDE.md this binary uses `anyhow` for error handling.

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

fn main() -> anyhow::Result<()> {
    // Wave-0 skeleton: just report the active allocator. RSS sampling + the
    // docs/MEMORY_REPORT.md write are Plan 04 (D-10).
    println!("memory_report: active global allocator = {}", active_allocator());
    Ok(())
}
