//! The Yeo-Johnson power stage against sklearn's
//! PowerTransformer(standardize=True) golden fixtures: lambdas from
//! the same bounded-Brent MLE, transformed train and test matrices
//! through transform. Pure f64 host math — the gates are tight.

use std::path::Path;

use ndarray_npy::NpzReader;
use tabicl_candle::power::PowerStage;

const CASES: [&str; 6] = ["mixed", "pos", "neg", "const", "tiny", "outlier"];

#[test]
fn power_stage_matches_sklearn() {
    let file =
        std::fs::File::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/power.npz"))
            .unwrap();
    let mut npz = NpzReader::new(file).unwrap();

    for case in CASES {
        let train: ndarray::Array2<f64> = npz.by_name(&format!("{case}_train")).unwrap();
        let test: ndarray::Array2<f64> = npz.by_name(&format!("{case}_test")).unwrap();
        let want_lambdas: ndarray::Array1<f64> = npz.by_name(&format!("{case}_lambdas")).unwrap();
        let want_train: ndarray::Array2<f64> = npz.by_name(&format!("{case}_out_train")).unwrap();
        let want_test: ndarray::Array2<f64> = npz.by_name(&format!("{case}_out_test")).unwrap();

        let (rows, cols) = train.dim();
        let x: Vec<f64> = train.iter().copied().collect();
        let stage = PowerStage::fit(&x, rows, cols);

        for (c, (got, want)) in stage.lambdas.iter().zip(want_lambdas.iter()).enumerate() {
            let diff = (got - want).abs() / f64::max(1.0, want.abs());
            assert!(diff < 1e-6, "{case} lambda {c}: {got} vs {want}");
        }

        let out_train = stage.transform(&x, rows, cols);
        for (i, (got, want)) in out_train.iter().zip(want_train.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "{case} train value {i}: {got} vs {want}"
            );
        }

        let (test_rows, _) = test.dim();
        let xt: Vec<f64> = test.iter().copied().collect();
        let out_test = stage.transform(&xt, test_rows, cols);
        for (i, (got, want)) in out_test.iter().zip(want_test.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "{case} test value {i}: {got} vs {want}"
            );
        }
    }
}
