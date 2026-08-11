//! Stage 3, density half, Rust side: the E1.2s3 shuffled-variant
//! surface on a reduced instance (400-row context sampled from the real
//! reference, 200-row frames — the full protocol's 15k-row fits belong
//! to the Python-side verdict run, not a standing test). Grades the
//! chain-rule read per permutation against the pinned oracle, then the
//! read-out contract end to end: NLL, q95 threshold, batch score,
//! AUROC.

use std::collections::VecDeque;
use std::path::Path;

use candle_core::Device;
use ndarray_npy::NpzReader;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn run(device: &Device) {
    if !root().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run scripts/convert_weights.py first");
        return;
    }
    let file = std::fs::File::open(root().join("fixtures/e12s3_reduced.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let context: ndarray::Array2<f32> = npz.by_name("context").unwrap();
    let frames: [(&str, ndarray::Array2<f32>); 2] = [
        ("ref", npz.by_name("ref_score").unwrap()),
        ("bat", npz.by_name("bat_score").unwrap()),
    ];
    let labels: ndarray::Array1<i64> = npz.by_name("labels").unwrap();
    let categorical: ndarray::Array1<i64> = npz.by_name("categorical").unwrap();
    let perms: ndarray::Array2<i64> = npz.by_name("perms").unwrap();
    let want_log: [(&str, ndarray::Array2<f64>); 2] = [
        ("ref", npz.by_name("log_densities_ref").unwrap()),
        ("bat", npz.by_name("log_densities_bat").unwrap()),
    ];
    let want_nll_ref: ndarray::Array1<f64> = npz.by_name("nll_ref").unwrap();
    let want_nll_bat: ndarray::Array1<f64> = npz.by_name("nll_bat").unwrap();
    let want_threshold: ndarray::Array1<f64> = npz.by_name("threshold").unwrap();
    let want_batch_score: ndarray::Array1<f64> = npz.by_name("batch_score").unwrap();
    let want_auroc: ndarray::Array1<f64> = npz.by_name("auroc").unwrap();

    let mut noise: std::collections::HashMap<String, Vec<f32>> = Default::default();
    for name in ["ref", "bat"] {
        for k in 0..perms.dim().0 {
            for role in ["train", "test"] {
                let arr: ndarray::Array2<f32> =
                    npz.by_name(&format!("noise_{name}_{role}_{k}")).unwrap();
                noise.insert(format!("{name}_{role}_{k}"), arr.iter().copied().collect());
            }
        }
    }

    let ckpt = tabicl_candle::weights::load(root(), "regressor", device).unwrap();
    let reg = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();
    let ckpt = tabicl_candle::weights::load(root(), "classifier", device).unwrap();
    let clf = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let (rows, cols) = context.dim();
    let unsup = tabicl_candle::unsupervised::Unsupervised::fit(
        &reg,
        Some(&clf),
        context.iter().copied().collect(),
        rows,
        cols,
        categorical.iter().map(|&v| v as usize).collect(),
    );

    let permutations: Vec<Vec<usize>> = perms
        .outer_iter()
        .map(|p| p.iter().map(|&v| v as usize).collect())
        .collect();

    // Per-row log densities are NOT tightly gateable on this surface,
    // and that is a finding, not slack: amount and amount_gap are
    // near-deterministic conditionals (amount ~= amount_inv on clean
    // rows), so their quantile grids are spikes whose local gaps sit at
    // ~1e-4 — the same order as legitimate fp32 forward jitter. Torch's
    // own fp32 forward sits 1.6e-4 from its fp64 reference there, the
    // Rust forward 1.3e-4 from the same reference, log_prob on shared
    // quantiles agrees to 6e-5, and the per-feature medians split
    // cleanly (well-conditioned numericals 2-7e-4, categoricals ~1e-6,
    // the two spiked columns ~3e-1) — the row-level read is chaotic in
    // ANY fp32 implementation on such columns. The rank-based decision
    // reads are the stable quantities, and they get the tight gates;
    // the bulk bounds below only catch gross semantic breakage (a
    // missed mask or floor shifts rows by ~23 log units). The tight
    // per-row grading of the orchestration lives in the synthetic
    // density fixtures, where the conditionals are well-conditioned.
    let median_gate = 1.5;
    let flip_fraction_gate = 0.6; // rows allowed beyond 0.5 log units
    let stats = |got: &[f64], want: &[f64]| -> (f64, f64, f64) {
        let mut diffs: Vec<f64> = got.iter().zip(want).map(|(a, b)| (a - b).abs()).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = diffs.len();
        let median = (diffs[(n - 1) / 2] + diffs[n / 2]) / 2.0;
        let flips = diffs.iter().filter(|d| **d > 0.5).count() as f64 / n as f64;
        (median, flips, diffs[n - 1])
    };

    let mut nlls: Vec<Vec<f64>> = Vec::new();
    for ((name, frame), (_, want)) in frames.iter().zip(&want_log) {
        let (test_rows, _) = frame.dim();
        let flat: Vec<f32> = frame.iter().copied().collect();
        let mut queue: VecDeque<Vec<f32>> = (0..permutations.len())
            .flat_map(|k| {
                ["train", "test"].map(|role| noise[&format!("{name}_{role}_{k}")].clone())
            })
            .collect();
        let mut draw = |n: usize| -> Vec<f32> {
            let next = queue.pop_front().expect("noise stream exhausted");
            assert_eq!(next.len(), n, "noise draw size mismatch");
            next
        };

        let mut all_log: Vec<Vec<f64>> = Vec::new();
        for (k, perm) in permutations.iter().enumerate() {
            let got = unsup
                .log_density(&flat, test_rows, perm, &mut draw, device)
                .unwrap();
            let (median, flips, max) = stats(&got, &want.row(k).to_vec());
            println!(
                "{name} perm {k}: median |diff| = {median:e}, flip fraction = {flips:.3}, max = {max:e}"
            );
            assert!(median < median_gate, "{name} perm {k} median: {median:e}");
            assert!(
                flips <= flip_fraction_gate,
                "{name} perm {k} flips: {flips}"
            );
            all_log.push(got);
        }
        assert!(queue.is_empty(), "noise stream not fully consumed");

        let n_perms = all_log.len() as f64;
        let scores: Vec<f64> = (0..test_rows)
            .map(|r| (all_log.iter().map(|lp| lp[r]).sum::<f64>() / n_perms).exp())
            .collect();
        nlls.push(tabicl_candle::readout::nll(&scores));
    }

    let (nll_ref, nll_bat) = (&nlls[0], &nlls[1]);
    for (name, got, want) in [
        ("ref", nll_ref, &want_nll_ref),
        ("bat", nll_bat, &want_nll_bat),
    ] {
        let (median, flips, max) = stats(got, want.as_slice().unwrap());
        println!(
            "nll {name}: median |diff| = {median:e}, flip fraction = {flips:.3}, max = {max:e}"
        );
        assert!(median < median_gate, "nll {name} median: {median:e}");
        assert!(flips <= flip_fraction_gate, "nll {name} flips: {flips}");
    }

    // The read-out contract — the decision quantities the harness
    // consumed. The threshold is a quantile of the noisy-tail NLLs, so
    // it gets an absolute bound; the step-function batch score gets
    // row-flip slack; AUROC is the headline and must agree closely.
    let threshold = tabicl_candle::readout::quantile(nll_ref, 0.95);
    println!("threshold: got {threshold} want {}", want_threshold[0]);
    assert!((threshold - want_threshold[0]).abs() < 0.5);

    let batch_score = tabicl_candle::readout::batch_score(nll_bat, threshold);
    println!(
        "batch_score: got {batch_score} want {}",
        want_batch_score[0]
    );
    assert!((batch_score - want_batch_score[0]).abs() <= 2.0 / nll_bat.len() as f64);

    let auroc = tabicl_candle::readout::auroc(labels.as_slice().unwrap(), nll_bat);
    println!("auroc: got {auroc:.4} want {:.4}", want_auroc[0]);
    assert!((auroc - want_auroc[0]).abs() < 5e-3);
}

#[test]
fn e12s3_reduced_matches_pinned_oracle_cpu() {
    run(&Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn e12s3_reduced_matches_pinned_oracle_metal() {
    run(&Device::new_metal(0).unwrap());
}
