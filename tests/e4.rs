//! The E4 what-if walk from the Rust side: all 21 fits (fine grid,
//! coarse grid, two-lever interaction) graded band-for-band against
//! the pinned sklearn oracle, then the harness grades (median APE,
//! effect recovery, coverage, width) recomputed from both sides and
//! compared. The pinned-vs-recorded ensemble comparison is Python-side
//! evidence (gen_e4_oracle.py output, recorded in the README): one
//! member carries dense support; the ensemble buys 2-3x point error
//! on sparse support.

use std::path::Path;

use candle_core::Device;
use ndarray_npy::NpzReader;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

const ALPHAS: [f64; 5] = [0.05, 0.10, 0.50, 0.90, 0.95];

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

struct Grade {
    mape_p50: f64,
    recovery: f64,
    cov80: f64,
    cov90: f64,
    width_rel: f64,
}

/// The harness grade over one fit's bands (rows = alphas, 6 columns).
fn grade(q: &[Vec<f64>], truth: &[f64], base: &[f64]) -> Grade {
    let n = truth.len();
    let p50 = &q[2];
    let mut ape: Vec<f64> = (0..n)
        .map(|i| (p50[i] - truth[i]).abs() / truth[i].abs())
        .collect();
    let mut rec: Vec<f64> = (0..n)
        .filter(|&i| truth[i] - base[i] != 0.0)
        .map(|i| (p50[i] - base[i]) / (truth[i] - base[i]))
        .collect();
    let mut width: Vec<f64> = (0..n)
        .map(|i| (q[3][i] - q[1][i]).abs() / truth[i].abs())
        .collect();
    let inside = |lo: &[f64], hi: &[f64]| -> f64 {
        (0..n)
            .filter(|&i| {
                let (l, h) = (lo[i].min(hi[i]), lo[i].max(hi[i]));
                l <= truth[i] && truth[i] <= h
            })
            .count() as f64
            / n as f64
    };
    Grade {
        mape_p50: median(&mut ape),
        recovery: median(&mut rec),
        cov80: inside(&q[1], &q[3]),
        cov90: inside(&q[0], &q[4]),
        width_rel: median(&mut width),
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_bands(
    model: &tabicl_candle::tabicl::TabIcl,
    train_x: &[f64],
    rows: usize,
    cols: usize,
    train_y: &[f64],
    test_x: &[f64],
    test_rows: usize,
    device: &Device,
) -> Vec<Vec<f64>> {
    let est = tabicl_candle::regressor::TabIclRegressor::fit(model, train_x, rows, cols, train_y);
    let pred = est.predict(test_x, test_rows, device).unwrap();
    let q: Vec<f32> = pred
        .quantiles(&ALPHAS)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    // predict returns (test, alphas); the grade wants rows = alphas
    (0..ALPHAS.len())
        .map(|a| {
            (0..test_rows)
                .map(|r| q[r * ALPHAS.len() + a] as f64)
                .collect()
        })
        .collect()
}

fn run(device: &Device) {
    if !root().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run scripts/convert_weights.py first");
        return;
    }
    let file = std::fs::File::open(root().join("fixtures/e4_walk.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let grid_index: ndarray::Array2<i64> = npz.by_name("grid_index").unwrap();
    let grid_offsets: ndarray::Array2<i64> = npz.by_name("grid_offsets").unwrap();
    let grid_train_x: ndarray::Array2<f64> = npz.by_name("grid_train_x").unwrap();
    let grid_train_y: ndarray::Array1<f64> = npz.by_name("grid_train_y").unwrap();
    let grid_test_x: ndarray::Array3<f64> = npz.by_name("grid_test_x").unwrap();
    let grid_truth: ndarray::Array2<f64> = npz.by_name("grid_truth").unwrap();
    let grid_base: ndarray::Array2<f64> = npz.by_name("grid_base").unwrap();
    let inter_train_x: ndarray::Array3<f64> = npz.by_name("inter_train_x").unwrap();
    let inter_train_y: ndarray::Array2<f64> = npz.by_name("inter_train_y").unwrap();
    let inter_test_x: ndarray::Array3<f64> = npz.by_name("inter_test_x").unwrap();
    let inter_truth: ndarray::Array2<f64> = npz.by_name("inter_truth").unwrap();

    let file = std::fs::File::open(root().join("fixtures/e4_pinned.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let want_grid: ndarray::Array3<f64> = npz.by_name("grid_bands").unwrap();
    let want_inter: ndarray::Array3<f64> = npz.by_name("inter_bands").unwrap();

    let ckpt = tabicl_candle::weights::load(root(), "regressor", device).unwrap();
    let model = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let mut max_rel = 0f64;
    let mut compare = |got: &[Vec<f64>], want: ndarray::ArrayView2<f64>, label: &str| {
        for (a, row) in got.iter().enumerate() {
            for (i, v) in row.iter().enumerate() {
                let w = want[[a, i]];
                let rel = (v - w).abs() / f64::max(1.0, w.abs());
                if rel > max_rel {
                    max_rel = rel;
                }
                assert!(rel < 1e-3, "{label} alpha {a} col {i}: rel {rel:e}");
            }
        }
    };

    let n_grid = grid_index.dim().0;
    let mut grid_bands: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_grid);
    for i in 0..n_grid {
        let (off, n) = (grid_offsets[[i, 0]] as usize, grid_offsets[[i, 1]] as usize);
        let train_x: Vec<f64> = grid_train_x
            .slice(ndarray::s![off..off + n, ..])
            .iter()
            .copied()
            .collect();
        let train_y: Vec<f64> = grid_train_y.slice(ndarray::s![off..off + n]).to_vec();
        let test_x: Vec<f64> = grid_test_x
            .index_axis(ndarray::Axis(0), i)
            .iter()
            .copied()
            .collect();
        let q = predict_bands(&model, &train_x, n, 2, &train_y, &test_x, 6, device);
        compare(
            &q,
            want_grid.index_axis(ndarray::Axis(0), i),
            &format!("grid fit {i}"),
        );
        grid_bands.push(q);
    }
    for i in 0..inter_train_x.dim().0 {
        let train_x: Vec<f64> = inter_train_x
            .index_axis(ndarray::Axis(0), i)
            .iter()
            .copied()
            .collect();
        let train_y: Vec<f64> = inter_train_y.index_axis(ndarray::Axis(0), i).to_vec();
        let test_x: Vec<f64> = inter_test_x
            .index_axis(ndarray::Axis(0), i)
            .iter()
            .copied()
            .collect();
        let rows = inter_train_y.dim().1;
        let q = predict_bands(&model, &train_x, rows, 3, &train_y, &test_x, 6, device);
        compare(
            &q,
            want_inter.index_axis(ndarray::Axis(0), i),
            &format!("inter fit {i}"),
        );
        // interaction grade: median absolute error of p50 vs truth
        let truth: Vec<f64> = inter_truth.index_axis(ndarray::Axis(0), i).to_vec();
        let mut err: Vec<f64> = (0..6).map(|r| (q[2][r] - truth[r]).abs()).collect();
        let mut werr: Vec<f64> = (0..6)
            .map(|r| (want_inter[[i, 2, r]] - truth[r]).abs())
            .collect();
        let (e, we) = (median(&mut err), median(&mut werr));
        assert!(
            (e - we).abs() / f64::max(1.0, we.abs()) < 1e-3,
            "inter fit {i} err: {e} vs {we}"
        );
    }
    println!("bands vs pinned oracle: max relative |diff| = {max_rel:e} over 21 fits");

    // Grid grades, Rust vs pinned-oracle bands through the same grader.
    for i in 0..n_grid {
        let truth: Vec<f64> = grid_truth.index_axis(ndarray::Axis(0), i).to_vec();
        let base: Vec<f64> = grid_base.index_axis(ndarray::Axis(0), i).to_vec();
        let got = grade(&grid_bands[i], &truth, &base);
        let want_q: Vec<Vec<f64>> = (0..5)
            .map(|a| (0..6).map(|r| want_grid[[i, a, r]]).collect())
            .collect();
        let want = grade(&want_q, &truth, &base);
        assert!((got.mape_p50 - want.mape_p50).abs() < 2e-3, "fit {i} mape");
        assert!(
            (got.recovery - want.recovery).abs() < 5e-3,
            "fit {i} recovery"
        );
        assert!(
            (got.width_rel - want.width_rel).abs() < 2e-3,
            "fit {i} width"
        );
        // coverage is a step function over 6 points; allow one edge flip
        assert!((got.cov80 - want.cov80).abs() < 0.17, "fit {i} cov80");
        assert!((got.cov90 - want.cov90).abs() < 0.17, "fit {i} cov90");
    }
    println!("grid grades match the pinned oracle on all {n_grid} fits");
}

#[test]
fn e4_walk_matches_pinned_oracle_cpu() {
    run(&Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn e4_walk_matches_pinned_oracle_metal() {
    run(&Device::new_metal(0).unwrap());
}
