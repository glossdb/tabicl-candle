//! The classifier wrapper against the sklearn wrapper pinned to one
//! member (n_estimators=1, norm method "none"). The label encoding and
//! preprocessing plane are graded in f64 against the wrapper's own
//! model inputs; the end-to-end probabilities need converted weights
//! and skip with a message when they are absent.

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
    classes: ndarray::Array1<f64>,
    proba: ndarray::Array2<f32>,
    labels: ndarray::Array1<f64>,
}

fn fixture() -> Fixture {
    let file = std::fs::File::open(root().join("fixtures/wrapper_classifier.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    Fixture {
        x_train: npz.by_name("x_train").unwrap(),
        y_train: npz.by_name("y_train").unwrap(),
        x_test: npz.by_name("x_test").unwrap(),
        model_x: npz.by_name("model_x").unwrap(),
        model_y: npz.by_name("model_y").unwrap(),
        classes: npz.by_name("classes").unwrap(),
        proba: npz.by_name("proba").unwrap(),
        labels: npz.by_name("labels").unwrap(),
    }
}

/// Label encoding plus the preprocessing plane, graded in f64 against
/// the wrapper's own model inputs. Needs no weights.
#[test]
fn encoding_and_preprocessing_match_sklearn() {
    let fx = fixture();
    let (rows, cols) = fx.x_train.dim();
    let (test_rows, _) = fx.x_test.dim();

    // LabelEncoder: sorted unique classes, y mapped to indices
    let mut classes: Vec<f64> = fx.y_train.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();
    assert_eq!(classes, fx.classes.to_vec(), "classes_");
    let encoded: Vec<f64> = fx
        .y_train
        .iter()
        .map(|v| classes.partition_point(|c| c < v) as f64)
        .collect();
    let model_y = fx.model_y.as_standard_layout();
    assert_eq!(encoded, model_y.as_slice().unwrap().to_vec(), "encoded y");

    let prep =
        tabicl_inference::regressor::Preprocessor::fit(fx.x_train.as_slice().unwrap(), rows, cols);
    assert_eq!(prep.n_kept(), fx.model_x.dim().2, "kept-column count");

    let mut got = prep.x_train.clone();
    got.extend(prep.transform(fx.x_test.as_slice().unwrap(), test_rows));
    // the npz stores model_x in Fortran order; normalize before slicing
    let want = fx.model_x.as_standard_layout();
    let diff = got
        .iter()
        .zip(want.as_slice().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    println!("preprocessing: max |diff| = {diff:e}");
    assert!(diff < 1e-10, "preprocessing {diff:e}");
}

fn run_end_to_end(device: &Device, tolerance: f32) {
    if !ws()
        .join("weights/tabicl-classifier.safetensors")
        .exists()
    {
        eprintln!("skipping: run verify/python/convert_weights.py first");
        return;
    }
    let fx = fixture();
    let ckpt = tabicl_model::weights::load(ws(), "classifier", device).unwrap();
    let model = tabicl_model::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let (rows, cols) = fx.x_train.dim();
    let (test_rows, _) = fx.x_test.dim();
    let clf = tabicl_inference::classifier::TabIclClassifier::fit(
        &model,
        fx.x_train.as_slice().unwrap(),
        rows,
        cols,
        fx.y_train.as_slice().unwrap(),
    );
    assert_eq!(clf.classes, fx.classes.to_vec());

    let proba = clf
        .predict_proba(fx.x_test.as_slice().unwrap(), test_rows, device)
        .unwrap();
    let want = fx.proba.as_slice().unwrap();
    assert_eq!(proba.len(), want.len());
    let diff = proba
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("predict_proba: max |diff| = {diff:e}");
    assert!(diff < tolerance, "proba: {diff:e} exceeds {tolerance:e}");

    let labels = clf
        .predict(fx.x_test.as_slice().unwrap(), test_rows, device)
        .unwrap();
    assert_eq!(labels, fx.labels.to_vec(), "predicted labels");
}

#[test]
fn classifier_matches_sklearn_cpu() {
    run_end_to_end(&Device::Cpu, 5e-4);
}

#[cfg(feature = "metal")]
#[test]
fn classifier_matches_sklearn_metal() {
    run_end_to_end(&Device::new_metal(0).unwrap(), 5e-4);
}
