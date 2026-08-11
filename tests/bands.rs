//! Stage 3, bands half: the E2.1 walk-forward band calibration from
//! the Rust side. Every fit is graded against the pinned-member sklearn
//! oracle (bands_pinned.npz), then the harness summary — coverage at
//! nominal 80/90 and median relative width, per grain — is recomputed
//! and compared. Coverage is a step function of band edges vs actuals,
//! so a handful of edge flips from fp noise is tolerated at row level;
//! the summary must agree to a few rows.

use std::path::Path;

use candle_core::Device;
use ndarray_npy::NpzReader;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

const ALPHAS: [f64; 5] = [0.05, 0.10, 0.50, 0.90, 0.95];

struct Summary {
    n: usize,
    cov80: f64,
    cov90: f64,
    medw80: f64,
}

fn summarize(
    index: &ndarray::Array2<i64>,
    bands: &[Vec<f64>],
    actual: &[f64],
    grain: i64,
) -> Summary {
    let rows: Vec<usize> = (0..index.dim().0)
        .filter(|&i| index[[i, 0]] == grain)
        .collect();
    let mut in80 = 0usize;
    let mut in90 = 0usize;
    let mut widths: Vec<f64> = Vec::with_capacity(rows.len());
    for &i in &rows {
        let b = &bands[i];
        let a = actual[i];
        let (lo80, hi80) = (b[1].min(b[3]), b[1].max(b[3]));
        let (lo90, hi90) = (b[0].min(b[4]), b[0].max(b[4]));
        in80 += usize::from(lo80 <= a && a <= hi80);
        in90 += usize::from(lo90 <= a && a <= hi90);
        widths.push((hi80 - lo80).abs() / f64::max(1e-9, a.abs()));
    }
    widths.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = rows.len();
    let medw80 = if n % 2 == 1 {
        widths[n / 2]
    } else {
        (widths[n / 2 - 1] + widths[n / 2]) / 2.0
    };
    Summary {
        n,
        cov80: in80 as f64 / n as f64,
        cov90: in90 as f64 / n as f64,
        medw80,
    }
}

fn run(device: &Device) {
    if !root().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run scripts/convert_weights.py first");
        return;
    }
    let file = std::fs::File::open(root().join("fixtures/bands_walk.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let index: ndarray::Array2<i64> = npz.by_name("index").unwrap();
    let train_x_all: ndarray::Array2<f64> = npz.by_name("train_x_all").unwrap();
    let train_y_all: ndarray::Array1<f64> = npz.by_name("train_y_all").unwrap();
    let test_x: ndarray::Array2<f64> = npz.by_name("test_x").unwrap();
    let actual: ndarray::Array1<f64> = npz.by_name("actual").unwrap();

    let file = std::fs::File::open(root().join("fixtures/bands_pinned.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let oracle: ndarray::Array2<f64> = npz.by_name("bands").unwrap();

    let ckpt = tabicl_candle::weights::load(root(), "regressor", device).unwrap();
    let model = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let n_fits = index.dim().0;
    let cols = train_x_all.dim().1;
    let mut bands: Vec<Vec<f64>> = Vec::with_capacity(n_fits);
    let mut max_rel = 0f64;
    for i in 0..n_fits {
        let (off, n) = (index[[i, 4]] as usize, index[[i, 5]] as usize);
        let x: Vec<f64> = train_x_all
            .slice(ndarray::s![off..off + n, ..])
            .iter()
            .copied()
            .collect();
        let y: Vec<f64> = train_y_all.slice(ndarray::s![off..off + n]).to_vec();
        let xt: Vec<f64> = test_x.row(i).to_vec();

        let est = tabicl_candle::regressor::TabIclRegressor::fit(&model, &x, n, cols, &y);
        let pred = est.predict(&xt, 1, device).unwrap();
        let q: Vec<f32> = pred
            .quantiles(&ALPHAS)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let q: Vec<f64> = q.into_iter().map(|v| v as f64).collect();

        for (a, got) in q.iter().enumerate() {
            let want = oracle[[i, a]];
            let rel = (got - want).abs() / f64::max(1.0, want.abs());
            if rel > max_rel {
                max_rel = rel;
                if rel > 1e-3 {
                    println!(
                        "fit {i} (grain={} seed={} months={} t={} n={}): alpha {} got {got} want {want}",
                        index[[i, 0]],
                        index[[i, 1]],
                        index[[i, 2]],
                        index[[i, 3]],
                        n,
                        ALPHAS[a],
                    );
                }
            }
        }
        bands.push(q);
    }
    println!("bands vs pinned oracle: max relative |diff| = {max_rel:e}");
    assert!(
        max_rel < 1e-3,
        "band values drift from the pinned oracle: {max_rel:e}"
    );

    let oracle_bands: Vec<Vec<f64>> = (0..n_fits).map(|i| oracle.row(i).to_vec()).collect();
    let actual: Vec<f64> = actual.to_vec();
    for (grain, name) in [(0i64, "month"), (1, "segment")] {
        let got = summarize(&index, &bands, &actual, grain);
        let want = summarize(&index, &oracle_bands, &actual, grain);
        println!(
            "{name}: n={} rust cov80={:.4} cov90={:.4} medw80={:.4} | \
             pinned-oracle cov80={:.4} cov90={:.4} medw80={:.4}",
            got.n, got.cov80, got.cov90, got.medw80, want.cov80, want.cov90, want.medw80
        );
        // allow at most 2 edge-flip rows per coverage figure
        let slack = 2.0 / got.n as f64;
        assert!((got.cov80 - want.cov80).abs() <= slack, "{name} cov80");
        assert!((got.cov90 - want.cov90).abs() <= slack, "{name} cov90");
        assert!((got.medw80 - want.medw80).abs() < 1e-3, "{name} medw80");
    }
}

#[test]
fn bands_walk_matches_pinned_oracle_cpu() {
    run(&Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn bands_walk_matches_pinned_oracle_metal() {
    run(&Device::new_metal(0).unwrap());
}
