//! The unit holds together: fixtures read without Python, weights load
//! digest-verified when present (skip with a message when absent — the
//! oracle-test pattern the glossql suites use).

use std::path::Path;

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

#[test]
fn fixtures_are_readable_and_shaped() {
    for name in ["regressor_10x4x7", "classifier_10x4x7"] {
        let file = std::fs::File::open(root().join(format!("fixtures/{name}.npz"))).unwrap();
        let mut npz = NpzReader::new(file).unwrap();
        let x: ndarray::Array3<f32> = npz.by_name("x").unwrap();
        let y: ndarray::Array2<f32> = npz.by_name("y").unwrap();
        let out: ndarray::Array3<f32> = npz.by_name("out").unwrap();
        assert_eq!(x.dim(), (1, 10, 4), "{name}");
        assert_eq!(y.dim(), (1, 7), "{name}");
        // test rows = T - train; the last axis is the head's support
        // (999 raw quantiles / 10 logits).
        assert_eq!(out.dim().1, 3, "{name}");
    }
}

#[test]
fn weights_load_and_verify() {
    if !ws().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run verify/python/convert_weights.py first");
        return;
    }
    for which in ["regressor", "classifier"] {
        let ckpt = tabicl_model::weights::load(ws(), which, &candle_core::Device::Cpu).unwrap();
        assert!(
            ckpt.tensors.len() > 100,
            "{which}: expected a full state dict, got {}",
            ckpt.tensors.len()
        );
    }
}

#[test]
fn weights_load_from_bytes() {
    let st = ws().join("weights/tabicl-regressor.safetensors");
    if !st.exists() {
        eprintln!("skipping: run verify/python/convert_weights.py first");
        return;
    }
    let bytes = std::fs::read(&st).unwrap();
    let config =
        std::fs::read_to_string(ws().join("weights/tabicl-regressor.config.json")).unwrap();
    let ckpt =
        tabicl_model::weights::load_bytes(&bytes, &config, &candle_core::Device::Cpu).unwrap();
    assert!(
        ckpt.tensors.len() > 100,
        "expected a full state dict, got {}",
        ckpt.tensors.len()
    );
    // The bytes-loaded checkpoint builds the same model the file path does.
    tabicl_model::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();
}
