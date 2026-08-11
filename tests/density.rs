//! The density read against the oracle's score_samples, permutations
//! and noise replayed from the fixture. Log densities compound one
//! forward per conditional, so the gate is looser than the wrapper's —
//! and the preprocessing here runs on f32-cast data (the sklearn class
//! stores X_ in f32) while the Rust statistics are f64, so ~1e-7
//! input-level differences are expected before the forward's own ~2e-5.

use std::collections::VecDeque;
use std::path::Path;

use candle_core::Device;
use ndarray_npy::NpzReader;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn run(device: &Device, tolerance: f64) {
    if !root().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run scripts/convert_weights.py first");
        return;
    }
    let file = std::fs::File::open(root().join("fixtures/density_scores.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let x_train: ndarray::Array2<f32> = npz.by_name("x_train").unwrap();
    let x_test: ndarray::Array2<f32> = npz.by_name("x_test").unwrap();
    let perms: ndarray::Array2<i64> = npz.by_name("perms").unwrap();
    let want_log: ndarray::Array2<f64> = npz.by_name("log_densities").unwrap();
    let want_scores: ndarray::Array1<f64> = npz.by_name("scores").unwrap();

    // The oracle's noise stream, in consumption order: train then test
    // per permutation.
    let mut noise_queue: VecDeque<Vec<f32>> = VecDeque::new();
    for k in 0..perms.dim().0 {
        for role in ["train", "test"] {
            let arr: ndarray::Array2<f32> = npz.by_name(&format!("noise_{role}_{k}")).unwrap();
            noise_queue.push_back(arr.iter().copied().collect());
        }
    }

    let ckpt = tabicl_candle::weights::load(root(), "regressor", device).unwrap();
    let model = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let (rows, cols) = x_train.dim();
    let (test_rows, _) = x_test.dim();
    let unsup = tabicl_candle::unsupervised::Unsupervised::fit(
        &model,
        x_train.iter().copied().collect(),
        rows,
        cols,
    );

    let permutations: Vec<Vec<usize>> = perms
        .outer_iter()
        .map(|p| p.iter().map(|&v| v as usize).collect())
        .collect();
    let mut noise = |n: usize| -> Vec<f32> {
        let next = noise_queue.pop_front().expect("noise stream exhausted");
        assert_eq!(next.len(), n, "noise draw size mismatch");
        next
    };

    // Grade each permutation's log density, then the composed scores.
    let x_test_flat: Vec<f32> = x_test.iter().copied().collect();
    let mut all_log: Vec<Vec<f64>> = Vec::new();
    for (k, perm) in permutations.iter().enumerate() {
        let got = unsup
            .log_density(&x_test_flat, test_rows, perm, &mut noise, device)
            .unwrap();
        let want: Vec<f64> = want_log.row(k).to_vec();
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let mean = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / got.len() as f64;
        println!("density perm {k} {perm:?}: max |diff| = {diff:e}, mean = {mean:e}");
        assert!(diff < tolerance, "perm {k}: {diff:e} exceeds {tolerance:e}");
        all_log.push(got);
    }
    assert!(noise_queue.is_empty(), "noise stream not fully consumed");

    let n_perms = all_log.len() as f64;
    let scores: Vec<f64> = (0..test_rows)
        .map(|r| (all_log.iter().map(|lp| lp[r]).sum::<f64>() / n_perms).exp())
        .collect();
    let diff = scores
        .iter()
        .zip(want_scores.iter())
        .map(|(a, b)| ((a.ln()) - (b.ln())).abs())
        .fold(0f64, f64::max);
    println!("density scores (log space): max |diff| = {diff:e}");
    assert!(diff < tolerance, "scores: {diff:e} exceeds {tolerance:e}");

    // The product read is the ranking — it must match exactly.
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        idx
    };
    assert_eq!(
        rank(&scores),
        rank(want_scores.as_slice().unwrap()),
        "score ranking diverges from the oracle"
    );
}

// The gate is set by the read's conditioning, not by porting slack:
// log_prob is the log of a finite difference of adjacent quantiles, so
// the forward's ~2e-5 parity is amplified by the local grid spacing
// into smooth ~1e-3-level log-density noise (a genuine mismatch — a
// branch or tail error — shows up at ~1e-1). Ranking is unaffected:
// the scores span many log-density units.
#[test]
fn density_matches_oracle_cpu() {
    run(&Device::Cpu, 5e-3);
}

#[cfg(feature = "metal")]
#[test]
fn density_matches_oracle_metal() {
    run(&Device::new_metal(0).unwrap(), 5e-3);
}
