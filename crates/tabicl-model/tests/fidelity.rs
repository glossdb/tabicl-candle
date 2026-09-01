//! Stage 1 of the fidelity gate: the candle forward matches the
//! committed torch train-mode forwards on every fixture. Skips with a
//! message when converted weights are absent (same pattern as load.rs).

use std::path::Path;

use candle_core::{Device, Tensor};
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

const SHAPES: [(usize, usize, usize); 3] = [(10, 4, 7), (16, 5, 12), (24, 3, 16)];

fn run_fixtures(which: &str, device: &Device, tolerance: f32) {
    if !ws().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run verify/python/convert_weights.py first");
        return;
    }
    let ckpt = tabicl_model::weights::load(ws(), which, device).unwrap();
    let model = tabicl_model::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

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

/// The quantile head against the model's own QuantileToDistribution,
/// isolated from the forward: input is the fixture's raw torch output.
/// Needs no weights — the head is weight-free post-processing.
#[test]
fn quantile_head_matches_torch() {
    let alphas = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];
    for (t, h, tr) in SHAPES {
        let file = std::fs::File::open(root().join(format!("fixtures/quantile_{t}x{h}x{tr}.npz")))
            .unwrap();
        let mut npz = NpzReader::new(file).unwrap();
        let raw: ndarray::Array3<f32> = npz.by_name("raw").unwrap();
        let expected_q: ndarray::Array3<f32> = npz.by_name("quantiles").unwrap();
        let expected_mean: ndarray::Array2<f32> = npz.by_name("mean").unwrap();
        let expected_bands: ndarray::Array3<f32> = npz.by_name("bands").unwrap();

        let (b, test, n) = raw.dim();
        let raw = Tensor::from_slice(raw.as_slice().unwrap(), (b, test, n), &Device::Cpu).unwrap();
        let dist = tabicl_model::quantile::QuantileDist::new(&raw).unwrap();

        let check = |name: &str, got: &Tensor, want: &[f32]| {
            let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            let max_diff = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("quantile {t}x{h}x{tr} {name}: max |diff| = {max_diff:e}");
            assert!(max_diff < 1e-4, "{name} {t}x{h}x{tr}: {max_diff:e}");
        };
        check("quantiles", &dist.quantiles, expected_q.as_slice().unwrap());
        check(
            "mean",
            &dist.mean().unwrap(),
            expected_mean.as_slice().unwrap(),
        );
        check(
            "bands",
            &dist.icdf(&alphas).unwrap(),
            expected_bands.as_slice().unwrap(),
        );

        // log_prob at probes covering both tails, the interior, and
        // exact knot landings
        let z: ndarray::Array3<f32> = npz.by_name("z").unwrap();
        let expected_lp: ndarray::Array3<f32> = npz.by_name("log_prob").unwrap();
        let (zb, zt, zk) = z.dim();
        let z = Tensor::from_slice(z.as_slice().unwrap(), (zb, zt, zk), &Device::Cpu).unwrap();
        check(
            "log_prob",
            &dist.log_prob(&z).unwrap(),
            expected_lp.as_slice().unwrap(),
        );
    }
}

/// candle 0.9's CPU argsort ignores a view's start offset (the argsort
/// permutation comes from the wrong storage window; gather then reads
/// the right one). QuantileDist materializes a zero-offset copy before
/// sorting — this pins that guarantee for offset views regardless of
/// how candle evolves.
#[test]
fn quantile_dist_is_offset_view_safe() {
    let dev = Device::Cpu;
    let data: Vec<f32> = (0..32)
        .map(|i| if i < 16 { i as f32 } else { (64 - i) as f32 })
        .collect();
    let t = Tensor::from_vec(data, (1, 4, 8), &dev).unwrap();
    let view = t.narrow(1, 2, 2).unwrap(); // contiguous strides, offset 16
    let from_view = tabicl_model::quantile::QuantileDist::new(&view).unwrap();
    let materialized = view.affine(1.0, 0.0).unwrap();
    let from_copy = tabicl_model::quantile::QuantileDist::new(&materialized).unwrap();
    let a: Vec<f32> = from_view
        .quantiles
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b: Vec<f32> = from_copy
        .quantiles
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(a, b, "QuantileDist must sort offset views correctly");
    let sorted = a.chunks(8).all(|row| row.windows(2).all(|w| w[0] <= w[1]));
    assert!(sorted, "quantiles must be monotone per row");
}
