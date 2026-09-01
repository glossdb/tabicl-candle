//! Stage 2 of the fidelity gate: the Rust wrapper matches the sklearn
//! wrapper pinned to one member (n_estimators=1, norm method "none").
//! Preprocessing is graded in f64 against the wrapper's own model
//! inputs; the end-to-end predictions need converted weights and skip
//! with a message when they are absent.

use std::path::Path;

use candle_core::Device;
use ndarray_npy::NpzReader;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Workspace root: weights/ and fixtures/DIGESTS live at the
/// workspace level, shared by every crate (see the weights policy).
fn ws() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

struct Fixture {
    x_train: ndarray::Array2<f64>,
    y_train: ndarray::Array1<f64>,
    x_test: ndarray::Array2<f64>,
    model_x: ndarray::Array3<f64>,
    model_y: ndarray::Array2<f64>,
    y_mean: f64,
    y_scale: f64,
    mean: ndarray::Array1<f32>,
    quantiles: ndarray::Array2<f32>,
    raw_quantiles: ndarray::Array2<f32>,
    alphas: Vec<f64>,
}

fn fixture() -> Fixture {
    let file = std::fs::File::open(root().join("fixtures/wrapper_regressor.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let y_mean: ndarray::Array1<f64> = npz.by_name("y_mean").unwrap();
    let y_scale: ndarray::Array1<f64> = npz.by_name("y_scale").unwrap();
    let alphas: ndarray::Array1<f64> = npz.by_name("alphas").unwrap();
    Fixture {
        x_train: npz.by_name("x_train").unwrap(),
        y_train: npz.by_name("y_train").unwrap(),
        x_test: npz.by_name("x_test").unwrap(),
        model_x: npz.by_name("model_x").unwrap(),
        model_y: npz.by_name("model_y").unwrap(),
        y_mean: y_mean[0],
        y_scale: y_scale[0],
        mean: npz.by_name("mean").unwrap(),
        quantiles: npz.by_name("quantiles").unwrap(),
        raw_quantiles: npz.by_name("raw_quantiles").unwrap(),
        alphas: alphas.to_vec(),
    }
}

fn max_diff(got: &[f64], want: &[f64]) -> f64 {
    got.iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max)
}

/// The preprocessing plane alone, graded in f64 against the wrapper's
/// own model inputs. Needs no weights.
#[test]
fn preprocessing_matches_sklearn() {
    let fx = fixture();
    let (rows, cols) = fx.x_train.dim();
    let (test_rows, _) = fx.x_test.dim();

    let prep =
        tabicl_inference::regressor::Preprocessor::fit(fx.x_train.as_slice().unwrap(), rows, cols);
    assert_eq!(prep.n_kept(), fx.model_x.dim().2, "kept-column count");

    let mut got = prep.x_train.clone();
    got.extend(prep.transform(fx.x_test.as_slice().unwrap(), test_rows));
    // the npz stores model_x in Fortran order; normalize before slicing
    let want = fx.model_x.as_standard_layout();
    let diff = max_diff(&got, want.as_slice().unwrap());
    println!("preprocessing: max |diff| = {diff:e}");
    assert!(diff < 1e-10, "preprocessing {diff:e}");
}

/// The y standardization: f32 cast then stats, as the wrapper does.
#[test]
fn y_scaler_matches_sklearn() {
    let fx = fixture();
    let yf: Vec<f64> = fx.y_train.iter().map(|v| *v as f32 as f64).collect();
    let mean = yf.iter().sum::<f64>() / yf.len() as f64;
    let std = (yf.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / yf.len() as f64).sqrt();
    println!(
        "y scaler: mean diff {:e}, scale diff {:e}",
        (mean - fx.y_mean).abs(),
        (std - fx.y_scale).abs()
    );
    // sklearn computes these in f32; the cast quantization is the gap
    assert!((mean - fx.y_mean).abs() < 1e-5);
    assert!((std - fx.y_scale).abs() < 1e-5);

    let scaled: Vec<f64> = yf.iter().map(|v| (v - mean) / std).collect();
    let diff = max_diff(&scaled, fx.model_y.as_slice().unwrap());
    println!("y scaled: max |diff| = {diff:e}");
    assert!(diff < 1e-5, "y scaled {diff:e}");
}

fn run_end_to_end(device: &Device, tolerance: f32) {
    if !ws().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run verify/python/convert_weights.py first");
        return;
    }
    let fx = fixture();
    let ckpt = tabicl_model::weights::load(ws(), "regressor", device).unwrap();
    let model = tabicl_model::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let (rows, cols) = fx.x_train.dim();
    let (test_rows, _) = fx.x_test.dim();
    let reg = tabicl_inference::regressor::TabIclRegressor::fit(
        &model,
        fx.x_train.as_slice().unwrap(),
        rows,
        cols,
        fx.y_train.as_slice().unwrap(),
    );
    let pred = reg
        .predict(fx.x_test.as_slice().unwrap(), test_rows, device)
        .unwrap();

    let check = |name: &str, got: candle_core::Tensor, want: &[f32]| {
        let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let diff = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("wrapper {name}: max |diff| = {diff:e}");
        assert!(diff < tolerance, "{name}: {diff:e} exceeds {tolerance:e}");
    };
    check("mean", pred.mean().unwrap(), fx.mean.as_slice().unwrap());
    check(
        "quantiles",
        pred.quantiles(&fx.alphas).unwrap(),
        fx.quantiles.as_slice().unwrap(),
    );
    check(
        "raw_quantiles",
        pred.raw_quantiles().unwrap(),
        fx.raw_quantiles.as_slice().unwrap(),
    );
}

#[test]
fn wrapper_matches_sklearn_cpu() {
    run_end_to_end(&Device::Cpu, 5e-4);
}

#[cfg(feature = "metal")]
#[test]
fn wrapper_matches_sklearn_metal() {
    run_end_to_end(&Device::new_metal(0).unwrap(), 5e-4);
}
