//! Post-processing transforms (margin → probability).
//!
//! Ports `treelite-mainline/src/gtil/postprocessor.cc:19-82` VERBATIM. Phase 1
//! shipped the two postprocessors exercised by the `binary:logistic` fixture
//! (`identity` and `sigmoid`). Plan 04-02 pulled forward the four postprocessors
//! the Phase-4 loaders need to verify 1e-5: `exponential`,
//! `exponential_standard_ratio`, `logarithm_one_plus_exp`, and the row-wise
//! `softmax`. Plan 05-03 completes the surface with the final three —
//! `signed_square`, `hinge`, and `multiclass_ova` — so all 10 upstream
//! postprocessors are now ported (GTIL-04). (`identity_multiclass` was already
//! shipped by the sklearn loaders in Phase 4.)
//!
//! ## The 1e-5 cast-ordering contract
//!
//! Upstream `ApplyPostProcessor<InputT>` (`predict.cc:307-323`) is instantiated
//! with `InputT == float` for f32 input and `InputT == double` for f64 input
//! (`predict.cc:236`, `c_api/gtil.cc:50-55`). Each postprocessor body in
//! `postprocessor.cc:19-82` is therefore templated on `InputT`, so for an
//! f64-input model the multiply / `std::exp` / `std::exp2` / `std::log1p` /
//! `copysign` run in **f64**, and for an f32-input model they run in **f32**.
//! Running an f64-input model's postprocessor through an f32 intermediate would
//! shift the final ULPs past the 1e-5 equivalence bound on a large-margin value
//! (CR-01, RESEARCH §Pitfall 3/4).
//!
//! This module therefore ships each non-softmax/non-identity postprocessor in
//! BOTH widths: the `f32` fns below run the element arithmetic in `f32`
//! (matching `ApplyPostProcessor<float>`), and the `*_f64` twins run it in
//! `f64` (matching `ApplyPostProcessor<double>`). `model.sigmoid_alpha` /
//! `model.ratio_c` are `f32` model fields on BOTH paths — they are cast into
//! the element type at the operation site, never stored as the element type.
//!
//! `softmax` is a PARTIAL exception: upstream `softmax<InputT>`
//! (`postprocessor.cc:57-75`) hardcodes `float max_margin` and `float t` for
//! EVERY `InputT`, but it still operates on the `InputT* row` in place — so for
//! the `double` instantiation (`ApplyPostProcessor<double>`) the cell reads
//! `row[i]` stay **f64**: `t = std::exp(row[i] - max_margin)` is
//! `double - float -> double`, `std::exp` runs in **double** (narrowing only the
//! result to `float t`), and the final `row[i] /= static_cast<float>(norm_const)`
//! is a `double /= float` divide. Collapsing the whole row to f32 before the
//! subtraction/exp/divide (the pre-05-08 path) shifts ULPs and was CR-01's exact
//! defect, just for softmax. Therefore [`softmax`] (f32 cells, for
//! `ApplyPostProcessor<float>`) and [`softmax_f64`] (f64 cells with the
//! `max_margin`/`t`/divisor float split, for `ApplyPostProcessor<double>`) are
//! BOTH shipped; only `max_margin`, `t`, and the final divisor are `f32` on
//! either path.

/// `identity` postprocessor (`postprocessor.cc:19-20`): returns the margin
/// unchanged. The `_alpha` argument mirrors the upstream signature shape (the
/// `num_class` slot) and is unused.
pub fn identity(_alpha: f32, v: f32) -> f32 {
    v
}

/// `identity_multiclass` postprocessor (`postprocessor.cc:55`), a no-op ported
/// verbatim (the upstream body is empty: `void identity_multiclass(...) {}`).
///
/// Used by the sklearn RandomForest/ExtraTrees classifier loaders (SKL-01),
/// whose averaged leaf-vector outputs are already normalized class
/// probabilities at load time — no further transform is applied. Mirrors the
/// shape of [`identity`] (the `_alpha` slot is unused).
pub fn identity_multiclass(_alpha: f32, v: f32) -> f32 {
    v
}

/// `sigmoid` postprocessor (`postprocessor.cc:33-37`), ported verbatim:
///
/// ```cpp
/// InputT const val = *elem;
/// *elem = InputT(1) / (InputT(1) + std::exp(-model.sigmoid_alpha * val));
/// ```
///
/// `sigmoid_alpha` is `f32` and `exp` runs on the `f32` value — no
/// double-precision promotion anywhere (the cast-ordering contract).
pub fn sigmoid(sigmoid_alpha: f32, v: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-sigmoid_alpha * v).exp())
}

/// f64 twin of [`sigmoid`] for an f64-input model (`ApplyPostProcessor<double>`
/// instantiating `sigmoid<double>`, `postprocessor.cc:33-37`). The element
/// `val` and the `std::exp` run in `f64`; only `sigmoid_alpha` is an `f32` model
/// field, cast into `f64` at the multiply site:
///
/// ```cpp
/// double const val = *elem;
/// *elem = double(1) / (double(1) + std::exp(-model.sigmoid_alpha * val));
/// ```
///
/// On a large-margin `v` (e.g. near ±40) this is NOT bit-equal to
/// `sigmoid(alpha, v as f32) as f64` — the f64 `exp` retains precision the
/// collapsed-f32 path loses (CR-01).
pub fn sigmoid_f64(sigmoid_alpha: f32, v: f64) -> f64 {
    1.0_f64 / (1.0_f64 + (-(sigmoid_alpha as f64) * v).exp())
}

/// `exponential` postprocessor (`postprocessor.cc:39-42`), ported verbatim:
///
/// ```cpp
/// *elem = std::exp(*elem);
/// ```
///
/// Instantiated with `InputT == float` in GTIL, so `std::exp` runs in `f32` —
/// the Rust port keeps the operation in `f32` (the cast-ordering contract).
pub fn exponential(v: f32) -> f32 {
    v.exp()
}

/// f64 twin of [`exponential`] (`exponential<double>`,
/// `postprocessor.cc:39-42`): `std::exp(*elem)` in `f64`.
pub fn exponential_f64(v: f64) -> f64 {
    v.exp()
}

/// `exponential_standard_ratio` postprocessor (`postprocessor.cc:44-47`),
/// ported verbatim:
///
/// ```cpp
/// *elem = std::exp2(-*elem / model.ratio_c);
/// ```
///
/// Note `std::exp2` — base-**2** exponential, NOT `std::exp`. `ratio_c` is a
/// `float` model field and the entire expression runs in `f32` (no
/// double-precision promotion). Used by XGBoost `survival:aft` /
/// `count:poisson`-adjacent objectives whose Phase-4 fixtures need it.
pub fn exponential_standard_ratio(ratio_c: f32, v: f32) -> f32 {
    (-v / ratio_c).exp2()
}

/// f64 twin of [`exponential_standard_ratio`]
/// (`exponential_standard_ratio<double>`, `postprocessor.cc:44-47`):
/// `std::exp2(-*elem / model.ratio_c)` in `f64`. `ratio_c` stays an `f32` model
/// field, cast into `f64` at the divide site.
pub fn exponential_standard_ratio_f64(ratio_c: f32, v: f64) -> f64 {
    (-v / (ratio_c as f64)).exp2()
}

/// `logarithm_one_plus_exp` postprocessor (`postprocessor.cc:49-52`), ported
/// verbatim:
///
/// ```cpp
/// *elem = std::log1p(std::exp(*elem));
/// ```
///
/// `std::log1p(x)` computes `ln(1 + x)` with improved precision near zero;
/// Rust's [`f32::ln_1p`] is the exact analog. The intermediate `exp` and the
/// `ln_1p` both run in `f32` (the cast-ordering contract).
pub fn logarithm_one_plus_exp(v: f32) -> f32 {
    v.exp().ln_1p()
}

/// f64 twin of [`logarithm_one_plus_exp`] (`logarithm_one_plus_exp<double>`,
/// `postprocessor.cc:49-52`): `std::log1p(std::exp(*elem))` in `f64`.
pub fn logarithm_one_plus_exp_f64(v: f64) -> f64 {
    v.exp().ln_1p()
}

/// `softmax` postprocessor (`postprocessor.cc:57-75`), ported VERBATIM with
/// respect to the mixed f32/f64 reduction order — this ordering IS the 1e-5
/// contract, do not "simplify":
///
/// ```cpp
/// float max_margin = row[0];
/// double norm_const = 0.0;
/// float t;
/// for (i = 1; i < num_class; ++i) if (row[i] > max_margin) max_margin = row[i];
/// for (i = 0; i < num_class; ++i) { t = std::exp(row[i] - max_margin);
///                                   norm_const += t; row[i] = t; }
/// for (i = 0; i < num_class; ++i) row[i] /= static_cast<float>(norm_const);
/// ```
///
/// Operates in place on one row's `num_class` cells. `max_margin` is `f32`,
/// each `t = exp(row[i] - max_margin)` is computed in `f32`, the accumulator
/// `norm_const` is `f64` (the `f32` `t` is promoted on add-into), and the final
/// divisor is `norm_const` cast back to `f32`. An empty row is a no-op (mirrors
/// the upstream loops never executing for `num_class == 0`).
pub fn softmax(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mut max_margin: f32 = row[0];
    let mut norm_const: f64 = 0.0;
    for &x in row.iter().skip(1) {
        if x > max_margin {
            max_margin = x;
        }
    }
    for cell in row.iter_mut() {
        let t: f32 = (*cell - max_margin).exp();
        norm_const += t as f64;
        *cell = t;
    }
    let divisor = norm_const as f32;
    for cell in row.iter_mut() {
        *cell /= divisor;
    }
}

/// f64 twin of [`softmax`] (`softmax<double>`, `postprocessor.cc:57-75`),
/// ported VERBATIM with respect to the mixed f32/f64 placement — this ordering
/// IS the 1e-5 cast-ordering contract, do not "simplify" by narrowing the row
/// to f32 (that was CR-01).
///
/// For the `double` instantiation the upstream body keeps the `double* row`
/// cells in **f64**; only `max_margin`, `t`, and the final divisor are `float`:
///
/// ```cpp
/// float max_margin = row[0];        // double row[0] narrowed to float
/// double norm_const = 0.0;
/// float t;
/// for (i = 1; i < n; ++i) if (row[i] > max_margin) max_margin = row[i];
/// for (i = 0; i < n; ++i) { t = std::exp(row[i] - max_margin);   // double - float -> double; std::exp in double; narrow to float t
///                           norm_const += t; row[i] = t; }        // double += float; double cell = float t
/// for (i = 0; i < n; ++i) row[i] /= static_cast<float>(norm_const);  // double /= float
/// ```
///
/// Cast-placement notes that must be preserved exactly:
/// - The max loop compares `row[i] > max_margin` in **f64** (the `f32`
///   `max_margin` promotes to `f64` for the comparison; the assignment narrows
///   the `f64` cell back to `f32`).
/// - `t = std::exp(row[i] - max_margin)`: the subtraction is `f64 - f32 -> f64`,
///   `exp` runs in **f64**, and only the result narrows to the `f32` `t`.
/// - The final divide is `*cell (f64) /= divisor (f64 derived from
///   `norm_const as f32`)` — a `double /= float` divide, NOT an f32 divide
///   (WR-03).
///
/// An empty row is a no-op (the upstream loops never execute for
/// `num_class == 0`).
pub fn softmax_f64(row: &mut [f64]) {
    if row.is_empty() {
        return;
    }
    // float max_margin = row[0];  (double -> float narrow)
    let mut max_margin: f32 = row[0] as f32;
    // if (row[i] > max_margin) max_margin = row[i];
    // Comparison promotes max_margin (f32) to f64; assignment narrows row[i] to f32.
    for &x in row.iter().skip(1) {
        if x > max_margin as f64 {
            max_margin = x as f32;
        }
    }
    let mut norm_const: f64 = 0.0;
    for cell in row.iter_mut() {
        // t = std::exp(row[i] - max_margin): double - float -> double, exp in
        // double, narrow the result to f32 t.
        let t: f32 = (*cell - max_margin as f64).exp() as f32;
        norm_const += t as f64; // norm_const (double) += t (float promotes)
        *cell = t as f64; // row[i] (double) = t (float promotes)
    }
    // static_cast<float>(norm_const), then double /= float.
    let divisor: f64 = norm_const as f32 as f64;
    for cell in row.iter_mut() {
        *cell /= divisor;
    }
}

/// `signed_square` postprocessor (`postprocessor.cc:22-26`), ported verbatim:
///
/// ```cpp
/// InputT const margin = *elem;
/// *elem = std::copysign(margin * margin, margin);
/// ```
///
/// Squares the margin then re-applies the margin's sign via `copysign`, so a
/// negative margin yields a negative result (`signed_square(-3.0) == -9.0`).
/// Instantiated with `InputT == float` in GTIL, so the multiply and `copysign`
/// run in `f32` — the Rust port keeps the operation in `f32` (the cast-ordering
/// contract).
pub fn signed_square(v: f32) -> f32 {
    (v * v).copysign(v)
}

/// f64 twin of [`signed_square`] (`signed_square<double>`,
/// `postprocessor.cc:22-26`): `std::copysign(margin * margin, margin)` in `f64`.
pub fn signed_square_f64(v: f64) -> f64 {
    (v * v).copysign(v)
}

/// `hinge` postprocessor (`postprocessor.cc:28-31`), ported verbatim:
///
/// ```cpp
/// *elem = (*elem > 0 ? InputT(1) : InputT(0));
/// ```
///
/// A strict step: `1.0` iff the margin is strictly greater than zero, else
/// `0.0` (so `hinge(0.0) == 0.0`). Runs in `f32` (the cast-ordering contract).
pub fn hinge(v: f32) -> f32 {
    if v > 0.0 { 1.0 } else { 0.0 }
}

/// `multiclass_ova` postprocessor (`postprocessor.cc:77-82`), ported verbatim:
///
/// ```cpp
/// for (i = 0; i < num_class; ++i)
///   row[i] = InputT(1) / (InputT(1) + std::exp(-model.sigmoid_alpha * row[i]));
/// ```
///
/// One-vs-all: applies an *independent* per-class sigmoid in place over the
/// row's `num_class` cells (this is the per-class form of [`sigmoid`], NOT a
/// `softmax` — the cells do NOT sum to 1). `sigmoid_alpha` is a `float` model
/// field and every cell's multiply and `std::exp` run in `f32` — there is NO
/// double-precision promotion (the cast-ordering contract, RESEARCH Pitfall 2).
pub fn multiclass_ova(sigmoid_alpha: f32, row: &mut [f32]) {
    for c in row.iter_mut() {
        *c = 1.0_f32 / (1.0_f32 + (-sigmoid_alpha * *c).exp());
    }
}

/// f64 twin of [`multiclass_ova`] (`multiclass_ova<double>`,
/// `postprocessor.cc:77-82`): an independent per-class sigmoid run in `f64` over
/// the row's `num_class` cells. `sigmoid_alpha` stays an `f32` model field, cast
/// into `f64` at each multiply site. Like [`multiclass_ova`] the cells do NOT
/// sum to 1 (this is the per-class form of [`sigmoid_f64`], NOT a softmax).
pub fn multiclass_ova_f64(sigmoid_alpha: f32, row: &mut [f64]) {
    for c in row.iter_mut() {
        *c = 1.0_f64 / (1.0_f64 + (-(sigmoid_alpha as f64) * *c).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_standard_ratio_uses_base2_exp2() {
        // v = -2.0, ratio_c = 4.0  =>  exp2(-(-2.0)/4.0) = exp2(0.5) = sqrt(2)
        let got = exponential_standard_ratio(4.0, -2.0);
        let expected = 2.0_f32.sqrt();
        assert!(
            (got - expected).abs() < 1e-7,
            "exp2 base-2 expected {expected}, got {got}"
        );
        // A second pair: v = 3.0, ratio_c = 1.0 => exp2(-3.0) = 0.125
        let got2 = exponential_standard_ratio(1.0, 3.0);
        assert!((got2 - 0.125_f32).abs() < 1e-7, "got {got2}");
    }

    #[test]
    fn exponential_returns_exp() {
        let got = exponential(1.0);
        assert!((got - std::f32::consts::E).abs() < 1e-6, "got {got}");
        assert!((exponential(0.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn logarithm_one_plus_exp_returns_log1p_exp() {
        // ln(1 + exp(0)) = ln(2)
        let got = logarithm_one_plus_exp(0.0);
        assert!((got - 2.0_f32.ln()).abs() < 1e-6, "got {got}");
        // ln(1 + exp(1))
        let got2 = logarithm_one_plus_exp(1.0);
        let expected2 = (1.0_f32 + 1.0_f32.exp()).ln();
        assert!((got2 - expected2).abs() < 1e-6, "got {got2}");
    }

    #[test]
    fn softmax_sums_to_one_and_matches_reference() {
        // Hand reference for [1.0, 2.0, 3.0] with max-subtraction.
        let mut row = [1.0_f32, 2.0, 3.0];
        softmax(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax row sums to {sum}");

        // Reference: same f32-exp / f64-accumulate / f32-divide ordering.
        let m = 3.0_f32;
        let t: [f32; 3] = [
            (1.0_f32 - m).exp(),
            (2.0_f32 - m).exp(),
            (3.0_f32 - m).exp(),
        ];
        let norm = t[0] as f64 + t[1] as f64 + t[2] as f64;
        let d = norm as f32;
        for i in 0..3 {
            let expected = t[i] / d;
            assert!(
                (row[i] - expected).abs() < 1e-6,
                "class {i}: expected {expected}, got {}",
                row[i]
            );
        }
    }

    #[test]
    fn softmax_empty_row_is_noop() {
        let mut row: [f32; 0] = [];
        softmax(&mut row);
        assert_eq!(row.len(), 0);
    }

    // ----------------------------------------------------------------------- //
    // GREEN (Plan 05-03): the THREE remaining postprocessors (GTIL-04).
    //
    // These were RED Wave-0 scaffolds in Plan 01 (`#[ignore]`d, hand references
    // only). Plan 05-03 ports `signed_square`/`hinge`/`multiclass_ova` verbatim
    // (`postprocessor.cc:22-31, 77-82`) and these tests now call the real fns
    // and pass — completing the 10/10 postprocessor surface.
    // ----------------------------------------------------------------------- //

    /// `signed_square(v: f32) -> f32` = `(v*v).copysign(v)`
    /// (`postprocessor.cc:22-26`). `signed_square(-3.0) == -9.0`,
    /// `signed_square(2.0) == 4.0` (copysign preserves the margin sign).
    #[test]
    fn signed_square_matches_copysign_reference() {
        assert!((signed_square(-3.0) - (-9.0)).abs() < 1e-7);
        assert!((signed_square(2.0) - 4.0).abs() < 1e-7);
        // copysign carries the sign of zero / negatives through faithfully.
        assert!((signed_square(-0.5) - (-0.25)).abs() < 1e-7);
        assert_eq!(signed_square(0.0), 0.0);
    }

    /// `hinge(v: f32) -> f32` = `1.0` if `v > 0` else `0.0`
    /// (`postprocessor.cc:28-31`). `hinge(0.5) == 1.0`, `hinge(-1.0) == 0.0`,
    /// `hinge(0.0) == 0.0` (strict `> 0`).
    #[test]
    fn hinge_matches_step_reference() {
        assert_eq!(hinge(0.5), 1.0);
        assert_eq!(hinge(-1.0), 0.0);
        assert_eq!(hinge(0.0), 0.0);
    }

    /// `multiclass_ova(sigmoid_alpha: f32, row: &mut [f32])` = per-class
    /// independent sigmoid (NOT softmax), `sigmoid_alpha` stays `f32`
    /// (`postprocessor.cc:77-82`). With `alpha = 1.0`, each cell `c` becomes
    /// `1 / (1 + exp(-c))`; the cells do NOT sum to 1.
    #[test]
    fn multiclass_ova_matches_per_class_sigmoid_reference() {
        let alpha = 1.0_f32;
        let mut row = [-1.0_f32, 0.0, 2.0];
        multiclass_ova(alpha, &mut row);
        let expect = [
            1.0_f32 / (1.0 + 1.0_f32.exp()),
            0.5_f32,
            1.0_f32 / (1.0 + (-2.0_f32).exp()),
        ];
        for i in 0..3 {
            assert!((row[i] - expect[i]).abs() < 1e-7, "ova class {i}");
        }
        // A true OVA row does NOT sum to 1 (distinguishes it from softmax).
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() > 1e-3,
            "OVA must not normalize like softmax"
        );
    }

    // ----------------------------------------------------------------------- //
    // CR-01 (Plan 05-06): the f64-input postprocessor surface must run in f64
    // (matching `ApplyPostProcessor<double>`), NOT through an f32 intermediate.
    // These tests PROVE the f64 twins are a genuinely higher-precision
    // computation: on a large-margin value the f64 path diverges from the
    // collapsed-f32 path (`*_fn(alpha, v as f32) as f64`) by more than 1e-7.
    // softmax is excluded — it is f32 on every InputT upstream.
    // ----------------------------------------------------------------------- //

    /// `sigmoid_f64(alpha, v)` must diverge from the collapsed-f32 path
    /// `sigmoid(alpha, v as f32) as f64` on a large margin — proving the f64
    /// sigmoid runs `std::exp` in double precision (CR-01). The divergence sits
    /// in the ~1e-8 RELATIVE band (NOT 1e-7 absolute on a saturated value): this
    /// is EXACTLY the regime that masked CR-01 — the collapsed-f32 path stays
    /// ~1e-7 inside the 1e-5 gate, so a coarser threshold would never catch it.
    /// We require a relative divergence well above the f64 rounding floor
    /// (~2.2e-16) on at least one probe, and that the two paths are not
    /// bit-identical.
    #[test]
    fn sigmoid_f64_diverges_from_collapsed_f32_on_large_margin() {
        let alpha = 1.0_f32;
        // Margins in the pre-/post-saturation slope where f32 vs f64 `exp`
        // genuinely differ (empirically ~1e-8 to ~9e-8 relative).
        let probes = [-18.0_f64, -16.0, -12.0, -10.0, 12.0, 15.0];
        let mut max_rel = 0.0_f64;
        let mut any_bit_diff = false;
        for &v in &probes {
            let f64_path = sigmoid_f64(alpha, v);
            let collapsed = sigmoid(alpha, v as f32) as f64;
            if f64_path.to_bits() != collapsed.to_bits() {
                any_bit_diff = true;
            }
            if f64_path != 0.0 {
                max_rel = max_rel.max((f64_path - collapsed).abs() / f64_path.abs());
            }
        }
        assert!(
            any_bit_diff,
            "sigmoid_f64 was bit-identical to the collapsed-f32 path on every probe — the f64 path is not running in f64"
        );
        assert!(
            max_rel > 1e-8,
            "sigmoid_f64 max relative divergence from collapsed-f32 was {max_rel:.3e}, \
             not above the 1e-8 band that distinguishes a genuine f64 computation"
        );
    }

    /// `exponential_f64(v)` must diverge from `exponential(v as f32) as f64` on a
    /// large argument — the f32 `exp` carries far fewer significant digits than
    /// the native f64 `exp`, so the widened f32 value is a different number. The
    /// divergence is in the ~1e-8 relative band (CR-01's masking regime).
    #[test]
    fn exponential_f64_diverges_from_collapsed_f32_on_large_arg() {
        let v = 80.0_f64;
        let f64_path = exponential_f64(v);
        let collapsed = exponential(v as f32) as f64;
        let rel = (f64_path - collapsed).abs() / f64_path.abs();
        assert_ne!(
            f64_path.to_bits(),
            collapsed.to_bits(),
            "exponential_f64 was bit-identical to the collapsed-f32 path — not running in f64"
        );
        assert!(
            rel > 1e-8,
            "exponential_f64 ({f64_path}) did not diverge from collapsed-f32 ({collapsed}); rel = {rel:.3e}"
        );
    }

    /// The f64 twins reduce to their f32 cousins' VALUE on small, exactly
    /// representable inputs (sanity: same math, just wider) — guards against a
    /// transcription error in a twin.
    /// `softmax_f64` must keep its row cells in f64 for the subtraction/exp/
    /// divide — so on a double-precision row whose values do NOT survive an f32
    /// round-trip, it diverges from the collapsed path
    /// `softmax(row narrowed to f32) widened back to f64` (CR-01 / WR-03). The
    /// divergence sits in the ~1e-8 relative band (the regime that masked CR-01),
    /// and the two paths must not be bit-identical on at least one cell.
    #[test]
    fn softmax_f64_diverges_from_collapsed_f32_on_precise_row() {
        // A 4-class multiclass row with f64 fractional bits below the f32 ULP,
        // plus a near-tie between the top two margins (the worst case for the
        // max-subtraction). These mimic the leaf_vec_mc softprob margins.
        let base = [12.300_000_017_3_f64, 12.300_000_009_1, -3.7, 5.55];

        let mut f64_row = base;
        softmax_f64(&mut f64_row);

        let mut collapsed: Vec<f32> = base.iter().map(|&v| v as f32).collect();
        softmax(&mut collapsed);
        let collapsed_f64: Vec<f64> = collapsed.iter().map(|&v| v as f64).collect();

        let mut any_bit_diff = false;
        let mut max_rel = 0.0_f64;
        for (a, b) in f64_row.iter().zip(collapsed_f64.iter()) {
            if a.to_bits() != b.to_bits() {
                any_bit_diff = true;
            }
            if *a != 0.0 {
                max_rel = max_rel.max((a - b).abs() / a.abs());
            }
        }
        assert!(
            any_bit_diff,
            "softmax_f64 was bit-identical to the collapsed-f32 path on every cell — \
             the f64 softmax is not running its row cells in f64"
        );
        assert!(
            max_rel > 1e-9,
            "softmax_f64 max relative divergence from collapsed-f32 was {max_rel:.3e}, \
             below the band that distinguishes a genuine f64 softmax"
        );

        // It is still a valid probability distribution (sums to ~1). The divisor
        // is `norm_const as f32` (the upstream `static_cast<float>`), so the sum
        // carries the f32 divisor's rounding — it is ~1, not bit-exactly 1.
        let sum: f64 = f64_row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax_f64 row sums to {sum}");
    }

    /// `softmax_f64` matches the upstream `softmax<double>` body exactly on a
    /// hand-computed reference (same f32 max_margin / f64 cell / f32 t / f64
    /// accumulate / `double /= float` divide ordering). Guards against a
    /// transcription error in the twin.
    #[test]
    fn softmax_f64_matches_upstream_ordering_reference() {
        let mut row = [1.0_f64, 2.0, 3.0];
        softmax_f64(&mut row);

        // Reference: max_margin is f32(3.0); t = exp(cell - max_margin) in f64
        // narrowed to f32; norm_const accumulates in f64; divisor = norm as f32.
        let m: f32 = 3.0;
        let t: [f32; 3] = [
            (1.0_f64 - m as f64).exp() as f32,
            (2.0_f64 - m as f64).exp() as f32,
            (3.0_f64 - m as f64).exp() as f32,
        ];
        let norm: f64 = t[0] as f64 + t[1] as f64 + t[2] as f64;
        let divisor: f64 = norm as f32 as f64;
        for i in 0..3 {
            let expected: f64 = (t[i] as f64) / divisor;
            assert_eq!(
                row[i].to_bits(),
                expected.to_bits(),
                "softmax_f64 class {i} differs from the upstream-ordered reference"
            );
        }
    }

    #[test]
    fn softmax_f64_empty_row_is_noop() {
        let mut row: [f64; 0] = [];
        softmax_f64(&mut row);
        assert_eq!(row.len(), 0);
    }

    #[test]
    fn f64_twins_agree_with_f32_on_small_exact_inputs() {
        assert!((signed_square_f64(-3.0) - (-9.0)).abs() < 1e-12);
        assert!((exponential_standard_ratio_f64(4.0, -2.0) - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((logarithm_one_plus_exp_f64(0.0) - 2.0_f64.ln()).abs() < 1e-12);
        let mut row = [0.0_f64];
        multiclass_ova_f64(1.0, &mut row);
        assert!((row[0] - 0.5).abs() < 1e-12);
    }
}
