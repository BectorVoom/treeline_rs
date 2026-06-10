//! The ten GTIL postprocessors as `#[cube]` helpers (D-03), porting
//! `treelite_gtil::postprocessor` VERBATIM with respect to cast order.
//!
//! The cast order IS the 1e-5 contract (CR-01): these ports reproduce
//! `postprocessor.rs` line-by-line and NEVER "simplify" to an all-f32 or
//! all-f64 rewrite. Each `#[cube]` helper is unit-tested in `tests/postproc.rs`
//! against its `treelite_gtil::postprocessor` scalar twin to 1e-5.
//!
//! cubecl 0.10.0 constraints that shape these ports (RESEARCH Pitfalls 1-3, and
//! the Wave-1 spike's retired assumptions A1-A4):
//! - Math intrinsics are ASSOCIATED functions, never methods (Pitfall 1):
//!   `F::exp(x)`, `F::log1p(x)`, `F::abs(x)`, `F::powf(b, e)` — the method forms
//!   (e.g. exp/ln_1p called on the value) fail E0599 `__expand_*_method` inside
//!   `#[cube]`.
//! - `exp2(x)` has NO cube-frontend intrinsic for a generic `F`; it is computed
//!   via the exact algebraic identity `exp2(x) == exp(x * ln 2)` in the element
//!   width `F` (the spike's A1 resolution, `exp_standard_ratio_kernel`).
//! - `copysign` has no cube-frontend intrinsic. `signed_square`'s
//!   `copysign(m*m, m)` is re-expressed as an if-STATEMENT flipping the sign of
//!   the (always non-negative) square by `m`'s sign — the verbatim equivalent.
//! - if-STATEMENTS that assign a `let mut`, never an if-EXPRESSION value
//!   (E0308); `while` loops with a counter, no loop-skip keyword.
//! - `softmax` / `softmax_f64` keep `max_margin` / `t` / divisor as `f32` and
//!   `norm_const` as `f64` EXACTLY where the scalar twin does (Pitfall 3); the
//!   f64 variant keeps the row cells in f64. This mixed-width split is NOT
//!   collapsed.
//!
//! All postprocessors take their `F`-width scalar model fields (`sigmoid_alpha`,
//! `ratio_c`) as already-cast `F` arguments — the host casts the `f32` model
//! field into `F` at the call site (mirroring `sigmoid_f64`'s `alpha as f64`).
//! `ln2` (the `exp2` identity constant) likewise rides in as an `F` argument.

use cubecl::prelude::*;

// ---------------------------------------------------------------------------
// Scalar-element postprocessors (one value in, one value out)
// ---------------------------------------------------------------------------

/// `identity` (`postprocessor.cc:19-20`): returns the margin unchanged. The
/// `_alpha` slot mirrors the upstream signature shape and is unused.
#[cube]
pub fn identity<F: Float>(_alpha: F, v: F) -> F {
    v
}

/// `identity_multiclass` (`postprocessor.cc:55`): a no-op (the upstream body is
/// empty). Mirrors the shape of [`identity`].
#[cube]
pub fn identity_multiclass<F: Float>(_alpha: F, v: F) -> F {
    v
}

/// `sigmoid` (`postprocessor.cc:33-37`): `1 / (1 + exp(-alpha * v))`, run in the
/// element width `F`. `sigmoid_alpha` is an `f32` model field cast into `F` at
/// the call site (the f64 twin's `alpha as f64`). `F::exp` is the associated-fn
/// intrinsic (Pitfall 1).
#[cube]
pub fn sigmoid<F: Float>(sigmoid_alpha: F, v: F) -> F {
    F::new(1.0) / (F::new(1.0) + F::exp(-sigmoid_alpha * v))
}

/// `exponential` (`postprocessor.cc:39-42`): `exp(v)` in width `F`.
#[cube]
pub fn exponential<F: Float>(v: F) -> F {
    F::exp(v)
}

/// `exponential_standard_ratio` (`postprocessor.cc:44-47`): `exp2(-v / ratio_c)`
/// — base-2. cubecl 0.10.0 has no generic `exp2` frontend intrinsic, so this
/// uses the spike-resolved identity `exp2(x) == exp(x * ln 2)` in width `F`
/// (`ln2` passed in as `F`). `ratio_c` is the `f32` model field cast into `F`.
#[cube]
pub fn exponential_standard_ratio<F: Float>(ratio_c: F, ln2: F, v: F) -> F {
    let x = -v / ratio_c;
    F::exp(x * ln2)
}

/// `logarithm_one_plus_exp` (`postprocessor.cc:49-52`): `log1p(exp(v))` in width
/// `F`. `F::log1p` is the cube-frontend `ln(1 + x)` intrinsic (the analog of the
/// scalar twin's exp-then-ln_1p method chain).
#[cube]
pub fn logarithm_one_plus_exp<F: Float>(v: F) -> F {
    F::log1p(F::exp(v))
}

/// `signed_square` (`postprocessor.cc:22-26`): `copysign(margin * margin,
/// margin)`. There is no cube-frontend `copysign`; since `margin * margin` is
/// always non-negative, the sign is purely `margin`'s sign, re-expressed here as
/// an if-STATEMENT (never an if-expr value). Verbatim equivalent of the scalar
/// twin within 1e-5.
#[cube]
pub fn signed_square<F: Float>(v: F) -> F {
    let sq = v * v;
    let mut out = sq;
    if v < F::new(0.0) {
        out = -sq;
    }
    out
}

/// `hinge` (`postprocessor.cc:28-31`): `v > 0 ? 1 : 0` (strict). Re-expressed as
/// an if-STATEMENT assigning a `let mut` (never an if-expr value).
#[cube]
pub fn hinge<F: Float>(v: F) -> F {
    let mut out = F::new(0.0);
    if v > F::new(0.0) {
        out = F::new(1.0);
    }
    out
}

// ---------------------------------------------------------------------------
// Row postprocessors (operate in place over one row of `n` class cells)
// ---------------------------------------------------------------------------

/// `multiclass_ova` (`postprocessor.cc:77-82`): an INDEPENDENT per-class sigmoid
/// over the row's `n` cells (NOT a softmax — the cells do not sum to 1). Each
/// cell `c` becomes `1 / (1 + exp(-alpha * c))` in width `F`. Operates in place
/// on `row` over `n` cells starting at `row_off` (the ragged-SoA offset). A
/// `while` loop with a counter (no loop-skip keyword).
#[cube]
pub fn multiclass_ova<F: Float>(sigmoid_alpha: F, row: &mut Array<F>, row_off: u32, n: u32) {
    let mut c: u32 = 0;
    while c < n {
        let idx = (row_off + c) as usize;
        row[idx] = sigmoid::<F>(sigmoid_alpha, row[idx]);
        c += 1;
    }
}

/// `softmax` (`postprocessor.cc:57-75`) for the `f32`-cell instantiation
/// (`ApplyPostProcessor<float>`): `max_margin` / `t` / divisor are `f32`,
/// `norm_const` is `f64`. Operates in place over `n` cells at `row_off`. The
/// mixed-width split IS the 1e-5 contract (Pitfall 3) — NOT collapsed. An empty
/// row (`n == 0`) is a no-op (the loops never execute).
#[cube]
pub fn softmax_f32(row: &mut Array<f32>, row_off: u32, n: u32) {
    if n > 0 {
        let mut max_margin: f32 = row[row_off as usize];
        let mut i: u32 = 1;
        while i < n {
            let x = row[(row_off + i) as usize];
            if x > max_margin {
                max_margin = x;
            }
            i += 1;
        }
        let mut norm_const: f64 = 0.0;
        let mut j: u32 = 0;
        while j < n {
            // t = exp(row[j] - max_margin) in f32; accumulate in f64.
            let t: f32 = f32::exp(row[(row_off + j) as usize] - max_margin);
            norm_const += f64::cast_from(t);
            row[(row_off + j) as usize] = t;
            j += 1;
        }
        // divisor = static_cast<float>(norm_const); f32 /= f32.
        let divisor: f32 = f32::cast_from(norm_const);
        let mut k: u32 = 0;
        while k < n {
            row[(row_off + k) as usize] /= divisor;
            k += 1;
        }
    }
}

/// `softmax_f64` (`postprocessor.cc:57-75`) for the `f64`-cell instantiation
/// (`ApplyPostProcessor<double>`): the row cells stay `f64`, but `max_margin` /
/// `t` / divisor are `f32` and `norm_const` is `f64` — the EXACT mixed-width
/// cast order of the scalar `softmax_f64` twin (Pitfall 3 / WR-03, CR-01's exact
/// band). The subtraction is `f64 - f32 -> f64`, `exp` runs in f64, the result
/// narrows to the f32 `t`, and the final divide is `f64 /= f32` (promoted back to
/// f64). NOT collapsed to all-f32 or all-f64. Empty row (`n == 0`) is a no-op.
#[cube]
pub fn softmax_f64(row: &mut Array<f64>, row_off: u32, n: u32) {
    if n > 0 {
        // float max_margin = row[0];  (f64 -> f32 narrow)
        let mut max_margin: f32 = f32::cast_from(row[row_off as usize]);
        let mut i: u32 = 1;
        while i < n {
            // Comparison promotes max_margin (f32) to f64; assignment narrows.
            if row[(row_off + i) as usize] > f64::cast_from(max_margin) {
                max_margin = f32::cast_from(row[(row_off + i) as usize]);
            }
            i += 1;
        }
        let mut norm_const: f64 = 0.0;
        let mut j: u32 = 0;
        while j < n {
            // t = exp(row[j] - max_margin): f64 - f32 -> f64, exp in f64, narrow
            // the result to the f32 t.
            let t: f32 =
                f32::cast_from(f64::exp(row[(row_off + j) as usize] - f64::cast_from(max_margin)));
            norm_const += f64::cast_from(t); // norm_const (f64) += t (f32 promotes)
            row[(row_off + j) as usize] = f64::cast_from(t); // row[j] (f64) = t (f32 promotes)
            j += 1;
        }
        // static_cast<float>(norm_const), then f64 /= f32 (promoted back to f64).
        let divisor: f64 = f64::cast_from(f32::cast_from(norm_const));
        let mut k: u32 = 0;
        while k < n {
            row[(row_off + k) as usize] /= divisor;
            k += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrappers (the thin `#[cube(launch)]` drivers the unit tests exercise;
// Wave 3's full launch kernels call the `#[cube]` helpers above directly).
// ---------------------------------------------------------------------------

/// Apply a scalar postprocessor element-wise over `input` into `output`, keyed
/// by `which`:
///   0 = identity, 1 = identity_multiclass, 2 = sigmoid, 3 = exponential,
///   4 = exponential_standard_ratio (uses `ratio_c`/`ln2`),
///   5 = logarithm_one_plus_exp, 6 = signed_square, 7 = hinge.
/// `alpha`/`ratio_c`/`ln2` ride as 1-element `Array<F>` (the spike's
/// Float-scalar launch convention). One unit per element.
#[cube(launch)]
pub fn scalar_postproc<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    alpha: &Array<F>,
    ratio_c: &Array<F>,
    ln2: &Array<F>,
    which: u32,
    n: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let v = input[i as usize];
        let mut out = v;
        if which == 0 {
            out = identity::<F>(alpha[0], v);
        }
        if which == 1 {
            out = identity_multiclass::<F>(alpha[0], v);
        }
        if which == 2 {
            out = sigmoid::<F>(alpha[0], v);
        }
        if which == 3 {
            out = exponential::<F>(v);
        }
        if which == 4 {
            out = exponential_standard_ratio::<F>(ratio_c[0], ln2[0], v);
        }
        if which == 5 {
            out = logarithm_one_plus_exp::<F>(v);
        }
        if which == 6 {
            out = signed_square::<F>(v);
        }
        if which == 7 {
            out = hinge::<F>(v);
        }
        output[i as usize] = out;
    }
}

/// Launch `multiclass_ova` over ONE row of `n` cells (single unit).
#[cube(launch)]
pub fn multiclass_ova_kernel<F: Float>(row: &mut Array<F>, alpha: &Array<F>, n: u32) {
    if ABSOLUTE_POS == 0 {
        multiclass_ova::<F>(alpha[0], row, 0, n);
    }
}

/// Launch the `f32`-cell [`softmax_f32`] over ONE row of `n` cells (single unit).
#[cube(launch)]
pub fn softmax_f32_kernel(row: &mut Array<f32>, n: u32) {
    if ABSOLUTE_POS == 0 {
        softmax_f32(row, 0, n);
    }
}

/// Launch the `f64`-cell [`softmax_f64`] over ONE row of `n` cells (single unit).
#[cube(launch)]
pub fn softmax_f64_kernel(row: &mut Array<f64>, n: u32) {
    if ABSOLUTE_POS == 0 {
        softmax_f64(row, 0, n);
    }
}
