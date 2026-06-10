//! Capture-environment provenance manifest (D-07, D-09).
//!
//! The [`Manifest`] is the diagnosability record frozen alongside every golden:
//! the upstream Treelite version, OS/arch/libc, framework versions, and — added
//! this phase (D-09) — the `backend` that produced/asserts the vector
//! (`scalar-cpu` for Phase 5) plus full forward-looking provenance (`rustc`,
//! `cubecl` placeholder, `seed`, `sha256`, per-framework versions).
//!
//! ## Backward compatibility (T-05-12)
//!
//! Every field NOT present in the older per-loader manifests
//! (`golden_v5.manifest.json`, `xgb_3format.manifest.json`, the original
//! `golden.json`) is `#[serde(default)]` — an `Option` or a defaulted scalar —
//! so an old manifest still deserializes unchanged. The `backend` field
//! defaults to `"scalar-cpu"` (the only backend that has ever existed) so a
//! pre-D-09 manifest reads as the scalar reference it implicitly was.
//!
//! [`check_manifest`] only ever `eprintln!`s a warning on drift; it NEVER fails
//! the equivalence gate (D-07/D-09 are diagnosability, not gates).

use serde::Deserialize;

/// Default for [`Manifest::backend`]: the only backend that existed before the
/// D-09 field was added is the plain-Rust scalar reference.
fn default_backend() -> String {
    "scalar-cpu".to_string()
}

/// Capture-environment provenance (D-07, D-09).
///
/// Keys match what the `capture_*.py` scripts write. The pre-D-09 fields
/// (`treelite`, `xgboost`, `os`, `arch`, `libc`, `python`) keep their original
/// shape so every previously-frozen manifest still parses; the new D-09 fields
/// are all defaulted/optional for the same reason.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    // --- pre-D-09 fields (unchanged shape) --------------------------------- //
    /// Upstream Treelite version the golden was captured against (e.g. `4.7.0`).
    pub treelite: String,
    /// XGBoost version used to author the fixture (optional).
    #[serde(default)]
    pub xgboost: Option<String>,
    /// `platform.platform()` string (e.g. `Linux-...-x86_64-with-glibc2.39`).
    pub os: String,
    /// `platform.machine()` (e.g. `x86_64`).
    pub arch: String,
    /// `platform.libc_ver()` tuple (e.g. `["glibc", "2.39"]`). Optional because
    /// the GTIL-matrix manifests record `os`/`arch` but not a separate `libc`.
    #[serde(default)]
    pub libc: serde_json::Value,
    /// Python version of the capture environment (optional).
    #[serde(default)]
    pub python: Option<String>,

    // --- D-09 backend + provenance (all defaulted for back-compat) ---------- //
    /// Which `R: Runtime` produced/asserts the vector (`scalar-cpu` this phase;
    /// `cubecl-cpu`/`cuda`/`wgpu`/`rocm` in Phase 6/7). Defaults to `scalar-cpu`
    /// so pre-D-09 manifests read as the scalar reference they implicitly were.
    #[serde(default = "default_backend")]
    pub backend: String,
    /// `rustc --version` string of the capture host (provenance only).
    #[serde(default)]
    pub rustc: Option<String>,
    /// cubecl version placeholder (`"n/a"` this phase; recorded forward, D-09).
    #[serde(default)]
    pub cubecl: Option<String>,
    /// RNG seed that drew the input matrix (documentation only; D-08 — CI never
    /// re-draws from it).
    #[serde(default)]
    pub seed: Option<u64>,
    /// SHA-256 of the `{input, output}` payload (per-fixture integrity).
    #[serde(default)]
    pub sha256: Option<String>,
    /// numpy version of the capture environment.
    #[serde(default)]
    pub numpy: Option<String>,
    /// scipy version of the capture environment.
    #[serde(default)]
    pub scipy: Option<String>,
    /// LightGBM version (when the fixture is a LightGBM golden).
    #[serde(default)]
    pub lightgbm: Option<String>,
    /// scikit-learn version (when the fixture is an sklearn golden).
    #[serde(default, alias = "sklearn")]
    pub scikit_learn: Option<String>,
    /// Model axis tag (e.g. `binary`/`leaf_vec_mc`) for the GTIL matrix.
    #[serde(default)]
    pub model: Option<String>,
    /// Preset axis tag (`f32`/`f64`).
    #[serde(default)]
    pub preset: Option<String>,
    /// Input-dtype axis tag (`f32`/`f64`).
    #[serde(default)]
    pub input_dtype: Option<String>,
    /// Predict-kind axis tag (`default`/`raw`/`leaf_id`/`score_per_tree`).
    #[serde(default)]
    pub kind: Option<String>,
    /// Layout axis tag (`dense`/`sparse`).
    #[serde(default)]
    pub layout: Option<String>,
}

/// Warn (never fail) when the running environment differs from the capture
/// environment recorded in the golden's manifest (D-07, D-09, T-04-02 / T-05-13).
///
/// A `1e-5` failure on a different distro is most often a libm/glibc divergence
/// (RESEARCH Pitfall 4). Surfacing OS/arch/`rustc`/`backend` drift here makes
/// such a failure immediately diagnosable per backend, without the manifest
/// itself ever passing/failing the equivalence gate.
pub fn check_manifest(manifest: &Manifest) {
    let running_os = std::env::consts::OS;
    let running_arch = std::env::consts::ARCH;

    // `manifest.os` is `platform.platform()` (a verbose descriptor), so a
    // substring check against the coarse `std::env::consts::OS` (e.g. "linux")
    // is the right granularity here.
    if !manifest.os.to_lowercase().contains(running_os) {
        eprintln!(
            "WARNING: golden captured on OS '{}' but running on '{}' — \
             a 1e-5 deviation here may be a libm/environment divergence (D-07/D-09).",
            manifest.os, running_os
        );
    }
    if manifest.arch.to_lowercase() != running_arch.to_lowercase() {
        eprintln!(
            "WARNING: golden captured on arch '{}' but running on '{}' — \
             a 1e-5 deviation here may be an environment divergence (D-07/D-09).",
            manifest.arch, running_arch
        );
    }
    // Backend drift: this phase runs ONLY the scalar reference, so a manifest
    // recording any other backend while we assert with `scalar-cpu` is worth
    // surfacing (D-09). Never fails — Phase 6/7 will assert cross-backend.
    if manifest.backend != "scalar-cpu" {
        eprintln!(
            "WARNING: golden manifest backend is '{}' but the scalar reference \
             (scalar-cpu) is asserting it — backend drift (D-09).",
            manifest.backend
        );
    }
    // rustc drift is provenance-only (compiled-in vs captured); surface it so a
    // toolchain-sensitive 1e-5 miss is diagnosable.
    if let (Some(captured), Some(running)) = (
        manifest.rustc.as_deref(),
        option_env!("RUSTC_VERSION_AT_BUILD"),
    ) && captured != running
    {
        eprintln!(
            "WARNING: golden captured with rustc '{captured}' but built with \
             '{running}' — toolchain drift (D-09).",
        );
    }
}
