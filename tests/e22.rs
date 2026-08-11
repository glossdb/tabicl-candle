//! E2.2 diagnosis reproduction: the cause classifier over the recorded
//! evidence episodes (24 cause-labeled episodes, leave-one-seed-out,
//! FULL vs BLIND feature sets). The fixture carries the pinned sklearn
//! oracle (scripts/gen_e22_fixture.py), which itself reproduces the
//! recorded ensemble accuracies exactly (full 18/24, blind 13/24, with
//! 23/24 individual call agreement per set). Here the ported classifier
//! wrapper is graded against the pinned oracle per fold, and the
//! accuracy figures are asserted at the recorded values. Skips without
//! converted weights.

use std::path::Path;

use candle_core::Device;
use ndarray_npy::NpzReader;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn e22_diagnosis_reproduces_recorded_accuracy() {
    if !root()
        .join("weights/tabicl-classifier.safetensors")
        .exists()
    {
        eprintln!("skipping: run scripts/convert_weights.py first");
        return;
    }
    let device = Device::Cpu;
    let ckpt = tabicl_candle::weights::load(root(), "classifier", &device).unwrap();
    let model = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let file = std::fs::File::open(root().join("fixtures/e22_diagnosis.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();

    for (label, want_hits) in [("full", 18usize), ("blind", 13usize)] {
        let mut hits = 0usize;
        let mut n = 0usize;
        let mut max_diff = 0f64;
        for fold in 0..3 {
            let tag = format!("{label}_f{fold}");
            let train_x: ndarray::Array2<f64> = npz.by_name(&format!("{tag}_train_x")).unwrap();
            let train_y: ndarray::Array1<i64> = npz.by_name(&format!("{tag}_train_y")).unwrap();
            let test_x: ndarray::Array2<f64> = npz.by_name(&format!("{tag}_test_x")).unwrap();
            let test_y: ndarray::Array1<i64> = npz.by_name(&format!("{tag}_test_y")).unwrap();
            let want_proba: ndarray::Array2<f64> = npz.by_name(&format!("{tag}_proba")).unwrap();

            let (rows, cols) = train_x.dim();
            let (test_rows, _) = test_x.dim();
            let y: Vec<f64> = train_y.iter().map(|&v| v as f64).collect();
            let clf = tabicl_candle::classifier::TabIclClassifier::fit(
                &model,
                train_x.as_standard_layout().as_slice().unwrap(),
                rows,
                cols,
                &y,
            );

            let proba = clf
                .predict_proba(
                    test_x.as_standard_layout().as_slice().unwrap(),
                    test_rows,
                    &device,
                )
                .unwrap();
            let want = want_proba.as_standard_layout();
            let want = want.as_slice().unwrap();
            assert_eq!(proba.len(), want.len(), "{tag} proba shape");
            for (a, b) in proba.iter().zip(want) {
                max_diff = max_diff.max((*a as f64 - b).abs());
            }

            let k = clf.classes.len();
            for (i, &truth) in test_y.iter().enumerate() {
                let row = &proba[i * k..(i + 1) * k];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                hits += usize::from(pred as i64 == truth);
                n += 1;
            }
        }
        println!("{label}: {hits}/{n}, max proba |diff| = {max_diff:e}");
        assert!(max_diff < 5e-4, "{label}: proba diff {max_diff:e}");
        assert_eq!(n, 24, "{label}: episode count");
        assert_eq!(hits, want_hits, "{label}: accuracy vs the recorded figure");
    }
}
