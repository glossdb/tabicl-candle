//! Stage 1 of the fidelity gate: the candle forward matches the
//! committed torch train-mode forwards on every fixture. Skips with a
//! message when converted weights are absent (same pattern as load.rs).

use std::path::Path;

use candle_core::{Device, Tensor};
use ndarray_npy::NpzReader;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

const SHAPES: [(usize, usize, usize); 3] = [(10, 4, 7), (16, 5, 12), (24, 3, 16)];

fn run_fixtures(which: &str, device: &Device, tolerance: f32) {
    if !root().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run scripts/convert_weights.py first");
        return;
    }
    let ckpt = tabicl_candle::weights::load(root(), which, device).unwrap();
    let model = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    for (t, h, tr) in SHAPES {
        let file =
            std::fs::File::open(root().join(format!("fixtures/{which}_{t}x{h}x{tr}.npz"))).unwrap();
        let mut npz = NpzReader::new(file).unwrap();
        let x: ndarray::Array3<f32> = npz.by_name("x").unwrap();
        let y: ndarray::Array2<f32> = npz.by_name("y").unwrap();
        let expected: ndarray::Array3<f32> = npz.by_name("out").unwrap();

        let x = Tensor::from_slice(x.as_slice().unwrap(), (1, t, h), device).unwrap();
        let y = Tensor::from_slice(y.as_slice().unwrap(), (1, tr), device).unwrap();
        let out = model.forward(&x, &y).unwrap();
        assert_eq!(out.dims3().unwrap(), expected.dim(), "{which} {t}x{h}x{tr}");

        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("{which} {t}x{h}x{tr}: max |diff| = {max_diff:e}");
        assert!(
            max_diff < tolerance,
            "{which} {t}x{h}x{tr}: max |diff| {max_diff:e} exceeds {tolerance:e}"
        );
    }
}

#[test]
fn regressor_matches_torch_cpu() {
    run_fixtures("regressor", &Device::Cpu, 1e-4);
}

#[test]
fn classifier_matches_torch_cpu() {
    run_fixtures("classifier", &Device::Cpu, 1e-4);
}

#[cfg(feature = "metal")]
#[test]
fn regressor_matches_torch_metal() {
    run_fixtures("regressor", &Device::new_metal(0).unwrap(), 1e-4);
}

#[cfg(feature = "metal")]
#[test]
fn classifier_matches_torch_metal() {
    run_fixtures("classifier", &Device::new_metal(0).unwrap(), 1e-4);
}
