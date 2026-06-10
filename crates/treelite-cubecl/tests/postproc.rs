//! Wave 3 postprocessor parity (plan 06-03) — each `#[cube]` port vs its scalar
//! twin to 1e-5.
//!
//! The cast order IS the 1e-5 contract (CR-01). These tests launch each
//! `treelite_cubecl::kernels::postproc` `#[cube]` helper on `CpuRuntime` and
//! assert its output equals the `treelite_gtil::postprocessor` scalar twin
//! within `epsilon = 1e-5` over a representative input row — including a
//! large-margin softmax row and a `multiclass_ova` row that exercise the
//! mixed-precision cast order (a collapsed single-precision rewrite fails the
//! gate, T-06-08).

use approx::assert_abs_diff_eq;
use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime};

use treelite_cubecl::kernels::postproc::{
    multiclass_ova_kernel, scalar_postproc, softmax_f32_kernel, softmax_f64_kernel,
};
use treelite_gtil::postprocessor as pp;

// `which` codes shared with `scalar_postproc`.
const IDENTITY: u32 = 0;
const IDENTITY_MULTICLASS: u32 = 1;
const SIGMOID: u32 = 2;
const EXPONENTIAL: u32 = 3;
const EXP_STD_RATIO: u32 = 4;
const LOG1P_EXP: u32 = 5;
const SIGNED_SQUARE: u32 = 6;
const HINGE: u32 = 7;

/// Launch `scalar_postproc::<F>` over `input` and read the result back.
#[allow(clippy::too_many_arguments)]
fn run_scalar<F: Float + CubeElement + bytemuck::Pod>(
    client: &cubecl::client::ComputeClient<CpuRuntime>,
    which: u32,
    input: &[F],
    alpha: F,
    ratio_c: F,
    ln2: F,
) -> Vec<F> {
    let zero = vec![F::from_int(0); input.len()];
    let h_in = client.create_from_slice(bytemuck::cast_slice(input));
    let h_out = client.create_from_slice(bytemuck::cast_slice(&zero));
    let h_alpha = client.create_from_slice(bytemuck::cast_slice(&[alpha]));
    let h_rc = client.create_from_slice(bytemuck::cast_slice(&[ratio_c]));
    let h_ln2 = client.create_from_slice(bytemuck::cast_slice(&[ln2]));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::Static(input.len().div_ceil(256).max(1) as u32, 1, 1);
    scalar_postproc::launch::<F, CpuRuntime>(
        client,
        cube_count,
        cube_dim,
        unsafe { ArrayArg::from_raw_parts(h_in.clone(), input.len()) },
        unsafe { ArrayArg::from_raw_parts(h_out.clone(), input.len()) },
        unsafe { ArrayArg::from_raw_parts(h_alpha.clone(), 1) },
        unsafe { ArrayArg::from_raw_parts(h_rc.clone(), 1) },
        unsafe { ArrayArg::from_raw_parts(h_ln2.clone(), 1) },
        which,
        input.len() as u32,
    );
    let bytes = client.read_one_unchecked(h_out);
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

#[test]
fn identity_and_identity_multiclass_match_scalar_twins() {
    let client = CpuRuntime::client(&Default::default());
    let row_f32: Vec<f32> = vec![-3.5, 0.0, 1.25, 42.0];
    // identity (f32)
    let got = run_scalar::<f32>(&client, IDENTITY, &row_f32, 1.0, 1.0, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(row_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::identity(1.0, *v), epsilon = 1e-5);
    }
    // identity_multiclass (f32)
    let got = run_scalar::<f32>(&client, IDENTITY_MULTICLASS, &row_f32, 1.0, 1.0, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(row_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::identity_multiclass(1.0, *v), epsilon = 1e-5);
    }
    // identity (f64)
    let row_f64: Vec<f64> = vec![-3.5, 0.0, 1.25, 42.0];
    let got = run_scalar::<f64>(&client, IDENTITY, &row_f64, 1.0, 1.0, core::f64::consts::LN_2);
    for (g, v) in got.iter().zip(row_f64.iter()) {
        // identity has no f64 twin (it is a pure no-op); the value must round-trip.
        assert_abs_diff_eq!(*g, *v, epsilon = 1e-5);
    }
}

#[test]
fn sigmoid_matches_scalar_twin_f32_and_f64() {
    let client = CpuRuntime::client(&Default::default());
    let alpha: f32 = 1.5;
    let probes_f32: Vec<f32> = vec![-12.0, -2.0, 0.0, 2.0, 12.0];
    let got = run_scalar::<f32>(&client, SIGMOID, &probes_f32, alpha, 1.0, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::sigmoid(alpha, *v), epsilon = 1e-5);
    }
    // f64 twin: alpha is an f32 model field cast into f64 at the call site.
    let probes_f64: Vec<f64> = vec![-18.0, -2.0, 0.0, 2.0, 18.0];
    let got = run_scalar::<f64>(&client, SIGMOID, &probes_f64, alpha as f64, 1.0, core::f64::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f64.iter()) {
        assert_abs_diff_eq!(*g, pp::sigmoid_f64(alpha, *v), epsilon = 1e-5);
    }
}

#[test]
fn exponential_matches_scalar_twin_f32_and_f64() {
    let client = CpuRuntime::client(&Default::default());
    let probes_f32: Vec<f32> = vec![-2.0, 0.0, 1.0, 3.5];
    let got = run_scalar::<f32>(&client, EXPONENTIAL, &probes_f32, 1.0, 1.0, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::exponential(*v), epsilon = 1e-5);
    }
    let probes_f64: Vec<f64> = vec![-2.0, 0.0, 1.0, 3.5];
    let got = run_scalar::<f64>(&client, EXPONENTIAL, &probes_f64, 1.0, 1.0, core::f64::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f64.iter()) {
        assert_abs_diff_eq!(*g, pp::exponential_f64(*v), epsilon = 1e-5);
    }
}

#[test]
fn exponential_standard_ratio_matches_scalar_twin_f32_and_f64() {
    let client = CpuRuntime::client(&Default::default());
    let ratio_c: f32 = 2.5;
    let probes_f32: Vec<f32> = vec![-1.5, 0.0, 0.75, 3.2];
    let got = run_scalar::<f32>(&client, EXP_STD_RATIO, &probes_f32, 1.0, ratio_c, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::exponential_standard_ratio(ratio_c, *v), epsilon = 1e-5);
    }
    let probes_f64: Vec<f64> = vec![-1.5, 0.0, 0.75, 3.2];
    let got = run_scalar::<f64>(&client, EXP_STD_RATIO, &probes_f64, 1.0, ratio_c as f64, core::f64::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f64.iter()) {
        assert_abs_diff_eq!(*g, pp::exponential_standard_ratio_f64(ratio_c, *v), epsilon = 1e-5);
    }
}

#[test]
fn logarithm_one_plus_exp_matches_scalar_twin_f32_and_f64() {
    let client = CpuRuntime::client(&Default::default());
    let probes_f32: Vec<f32> = vec![-3.0, 0.0, 1.0, 5.0];
    let got = run_scalar::<f32>(&client, LOG1P_EXP, &probes_f32, 1.0, 1.0, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::logarithm_one_plus_exp(*v), epsilon = 1e-5);
    }
    let probes_f64: Vec<f64> = vec![-3.0, 0.0, 1.0, 5.0];
    let got = run_scalar::<f64>(&client, LOG1P_EXP, &probes_f64, 1.0, 1.0, core::f64::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f64.iter()) {
        assert_abs_diff_eq!(*g, pp::logarithm_one_plus_exp_f64(*v), epsilon = 1e-5);
    }
}

#[test]
fn signed_square_matches_scalar_twin_f32_and_f64() {
    let client = CpuRuntime::client(&Default::default());
    let probes_f32: Vec<f32> = vec![-3.0, -0.5, 0.0, 2.0, 4.0];
    let got = run_scalar::<f32>(&client, SIGNED_SQUARE, &probes_f32, 1.0, 1.0, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::signed_square(*v), epsilon = 1e-5);
    }
    let probes_f64: Vec<f64> = vec![-3.0, -0.5, 0.0, 2.0, 4.0];
    let got = run_scalar::<f64>(&client, SIGNED_SQUARE, &probes_f64, 1.0, 1.0, core::f64::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f64.iter()) {
        assert_abs_diff_eq!(*g, pp::signed_square_f64(*v), epsilon = 1e-5);
    }
}

#[test]
fn hinge_matches_scalar_twin() {
    let client = CpuRuntime::client(&Default::default());
    let probes_f32: Vec<f32> = vec![-1.0, 0.0, 0.5, 100.0];
    let got = run_scalar::<f32>(&client, HINGE, &probes_f32, 1.0, 1.0, core::f32::consts::LN_2);
    for (g, v) in got.iter().zip(probes_f32.iter()) {
        assert_abs_diff_eq!(*g, pp::hinge(*v), epsilon = 1e-5);
    }
}

#[test]
fn multiclass_ova_matches_scalar_twin_f32_and_f64() {
    let client = CpuRuntime::client(&Default::default());
    let alpha: f32 = 1.0;

    // f32 row
    let mut expected_f32: Vec<f32> = vec![-1.0, 0.0, 2.0, -5.0];
    let row = expected_f32.clone();
    pp::multiclass_ova(alpha, &mut expected_f32);
    let h_row = client.create_from_slice(bytemuck::cast_slice(&row));
    let h_alpha = client.create_from_slice(bytemuck::cast_slice(&[alpha]));
    multiclass_ova_kernel::launch::<f32, CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(h_row.clone(), row.len()) },
        unsafe { ArrayArg::from_raw_parts(h_alpha.clone(), 1) },
        row.len() as u32,
    );
    let bytes = client.read_one_unchecked(h_row);
    let got = bytemuck::cast_slice::<u8, f32>(&bytes);
    for (g, e) in got.iter().zip(expected_f32.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
    }

    // f64 row (alpha is an f32 model field cast into f64 at the call site)
    let mut expected_f64: Vec<f64> = vec![-1.0, 0.0, 2.0, -5.0];
    let row64 = expected_f64.clone();
    pp::multiclass_ova_f64(alpha, &mut expected_f64);
    let h_row = client.create_from_slice(bytemuck::cast_slice(&row64));
    let h_alpha = client.create_from_slice(bytemuck::cast_slice(&[alpha as f64]));
    multiclass_ova_kernel::launch::<f64, CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(h_row.clone(), row64.len()) },
        unsafe { ArrayArg::from_raw_parts(h_alpha.clone(), 1) },
        row64.len() as u32,
    );
    let bytes = client.read_one_unchecked(h_row);
    let got = bytemuck::cast_slice::<u8, f64>(&bytes);
    for (g, e) in got.iter().zip(expected_f64.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
    }
}

#[test]
fn softmax_matches_scalar_twin_large_margin_row() {
    let client = CpuRuntime::client(&Default::default());

    // f32 softmax over a large-margin row (exercises the f32-max / f64-norm /
    // f32-divisor cast order, Pitfall 3 / CR-01).
    let mut expected_f32: Vec<f32> = vec![12.5, -3.0, 7.25, 11.9];
    let row = expected_f32.clone();
    pp::softmax(&mut expected_f32);
    let h_row = client.create_from_slice(bytemuck::cast_slice(&row));
    softmax_f32_kernel::launch::<CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(h_row.clone(), row.len()) },
        row.len() as u32,
    );
    let bytes = client.read_one_unchecked(h_row);
    let got = bytemuck::cast_slice::<u8, f32>(&bytes);
    for (g, e) in got.iter().zip(expected_f32.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
    }

    // f64 softmax: f64 cells with f32 max_margin/t/divisor — the worst-case
    // near-tie large-margin row from the scalar twin's CR-01 test.
    let mut expected_f64: Vec<f64> = vec![12.300_000_017_3, 12.300_000_009_1, -3.7, 5.55];
    let row64 = expected_f64.clone();
    pp::softmax_f64(&mut expected_f64);
    let h_row = client.create_from_slice(bytemuck::cast_slice(&row64));
    softmax_f64_kernel::launch::<CpuRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(h_row.clone(), row64.len()) },
        row64.len() as u32,
    );
    let bytes = client.read_one_unchecked(h_row);
    let got = bytemuck::cast_slice::<u8, f64>(&bytes);
    for (g, e) in got.iter().zip(expected_f64.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
    }
}
