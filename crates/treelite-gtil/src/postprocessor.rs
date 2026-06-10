//! Post-processing transforms (margin → probability).
//!
//! Ports `treelite-mainline/src/gtil/postprocessor.cc:19-82` VERBATIM. Phase 1
//! shipped the two postprocessors exercised by the `binary:logistic` fixture
//! (`identity` and `sigmoid`). Plan 04-02 pulls forward the four postprocessors
//! the Phase-4 loaders need to verify 1e-5: `exponential`,
//! `exponential_standard_ratio`, `logarithm_one_plus_exp`, and the row-wise
//! `softmax`. The remaining upstream postprocessors (`signed_square`, `hinge`,
//! `identity_multiclass`, `multiclass_ova`) are deferred to Phase 5 (complete
//! GTIL surface).
//!
//! ## The 1e-5 cast-ordering contract
//!
//! Upstream `sigmoid` is instantiated with `InputT == float`, and
//! `model.sigmoid_alpha` is a `float`. Therefore both the multiplication and
//! `std::exp` run in **f32** — there is NO promotion to double precision.
//! Doing the math in double precision would shift the final ULPs past the
//! 1e-5 equivalence bound (RESEARCH §Pitfall 3/4). The Rust port keeps every
//! operation in `f32`.

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
    // RED Wave-0 scaffolds for the THREE remaining postprocessors (GTIL-04).
    //
    // These are the Nyquist Wave-0 targets: hand-computed references for the
    // not-yet-implemented `signed_square`, `hinge`, and `multiclass_ova`
    // postprocessors. They are `#[ignore]`d with a "RED until Plan 03" reason
    // (the Wave-0 MISSING marker the Nyquist gate reads). Each test names the
    // EXACT fn signature Plan 03 must add to this module (porting
    // `postprocessor.cc:22-31, 77-82` verbatim) and asserts the hand reference;
    // until those fns exist the call sites stay commented so the crate still
    // compiles (acceptance: `cargo test -p treelite-gtil --no-run` exits 0).
    // ----------------------------------------------------------------------- //

    /// RED (Plan 03): `signed_square(v: f32) -> f32` = `(v*v).copysign(v)`
    /// (`postprocessor.cc:22-26`). Hand reference: `signed_square(-3.0) == -9.0`,
    /// `signed_square(2.0) == 4.0`.
    #[test]
    #[ignore = "RED until Plan 03 (signed_square postprocessor not yet implemented)"]
    fn signed_square_matches_copysign_reference() {
        // TODO Plan 03: implement `pub fn signed_square(v: f32) -> f32`, then
        // replace the hand references below with calls to it.
        // let got = signed_square(-3.0); assert!((got - (-9.0)).abs() < 1e-7);
        let reference_neg3 = (-3.0_f32 * -3.0_f32).copysign(-3.0); // == -9.0
        let reference_pos2 = (2.0_f32 * 2.0_f32).copysign(2.0); // == 4.0
        assert!((reference_neg3 - (-9.0)).abs() < 1e-7);
        assert!((reference_pos2 - 4.0).abs() < 1e-7);
    }

    /// RED (Plan 03): `hinge(v: f32) -> f32` = `1.0` if `v > 0` else `0.0`
    /// (`postprocessor.cc:28-31`). Hand reference: `hinge(0.5) == 1.0`,
    /// `hinge(-0.5) == 0.0`, `hinge(0.0) == 0.0`.
    #[test]
    #[ignore = "RED until Plan 03 (hinge postprocessor not yet implemented)"]
    fn hinge_matches_step_reference() {
        // TODO Plan 03: implement `pub fn hinge(v: f32) -> f32`, then replace
        // the hand references below with calls to it.
        // assert_eq!(hinge(0.5), 1.0); assert_eq!(hinge(-0.5), 0.0);
        let r_pos = if 0.5_f32 > 0.0 { 1.0_f32 } else { 0.0 };
        let r_neg = if -0.5_f32 > 0.0 { 1.0_f32 } else { 0.0 };
        let r_zero = if 0.0_f32 > 0.0 { 1.0_f32 } else { 0.0 };
        assert_eq!(r_pos, 1.0);
        assert_eq!(r_neg, 0.0);
        assert_eq!(r_zero, 0.0);
    }

    /// RED (Plan 03): `multiclass_ova(sigmoid_alpha: f32, row: &mut [f32])` =
    /// per-class independent sigmoid (NOT softmax), `sigmoid_alpha` stays `f32`
    /// (`postprocessor.cc:77-82`). Hand reference: with `alpha = 1.0`, each cell
    /// `c` becomes `1 / (1 + exp(-c))`.
    #[test]
    #[ignore = "RED until Plan 03 (multiclass_ova postprocessor not yet implemented)"]
    fn multiclass_ova_matches_per_class_sigmoid_reference() {
        // TODO Plan 03: implement `pub fn multiclass_ova(sigmoid_alpha: f32,
        // row: &mut [f32])`, then drive `row` through it and compare.
        let alpha = 1.0_f32;
        let mut row = [-1.0_f32, 0.0, 2.0];
        // Hand reference: independent per-class sigmoid (NOT normalized).
        for c in row.iter_mut() {
            *c = 1.0_f32 / (1.0_f32 + (-alpha * *c).exp());
        }
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
}
