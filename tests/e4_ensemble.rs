//! The multi-member ensemble against the recorded-configuration
//! sklearn oracle: all 21 E4 fits rerun with the exact member sets the
//! seeded generator produced (norm + feature permutation per member,
//! injected from the fixture), final averaged bands compared
//! band-for-band. This is the configuration of the recorded tfmeval
//! run — the oracle rerun sits on the recorded figures to four
//! decimals — so passing here closes the ensemble against the archive.
//!
//! Member generation itself (this crate's own RNG) is exercised for
//! shape and validity only; which permutation a member draws is
//! deliberately not sklearn's (see src/ensemble.rs).

use std::path::Path;

use candle_core::Device;
use ndarray_npy::NpzReader;
use tabicl_candle::ensemble::{EnsembleMember, TabIclEnsemble};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

const ALPHAS: [f64; 5] = [0.05, 0.10, 0.50, 0.90, 0.95];

fn members_from(rows: ndarray::ArrayView2<i64>) -> Vec<EnsembleMember> {
    rows.outer_iter()
        .map(|row| EnsembleMember {
            power: row[0] == 1,
            shuffle: row.iter().skip(1).map(|v| *v as usize).collect(),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn ensemble_bands(
    model: &tabicl_candle::tabicl::TabIcl,
    train_x: &[f64],
    rows: usize,
    cols: usize,
    train_y: &[f64],
    test_x: &[f64],
    test_rows: usize,
    members: Vec<EnsembleMember>,
    device: &Device,
) -> Vec<f64> {
    let est = TabIclEnsemble::fit(model, train_x, rows, cols, train_y, members);
    est.predict_quantiles(test_x, test_rows, &ALPHAS, device)
        .unwrap()
}

#[test]
fn e4_ensemble_matches_recorded_configuration_oracle() {
    let device = Device::Cpu;
    if !root().join("weights/tabicl-regressor.safetensors").exists() {
        eprintln!("skipping: run scripts/convert_weights.py first");
        return;
    }
    let file = std::fs::File::open(root().join("fixtures/e4_walk.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let grid_offsets: ndarray::Array2<i64> = npz.by_name("grid_offsets").unwrap();
    let grid_train_x: ndarray::Array2<f64> = npz.by_name("grid_train_x").unwrap();
    let grid_train_y: ndarray::Array1<f64> = npz.by_name("grid_train_y").unwrap();
    let grid_test_x: ndarray::Array3<f64> = npz.by_name("grid_test_x").unwrap();
    let inter_train_x: ndarray::Array3<f64> = npz.by_name("inter_train_x").unwrap();
    let inter_train_y: ndarray::Array2<f64> = npz.by_name("inter_train_y").unwrap();
    let inter_test_x: ndarray::Array3<f64> = npz.by_name("inter_test_x").unwrap();

    let file = std::fs::File::open(root().join("fixtures/e4_ensemble.npz")).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let want_grid: ndarray::Array3<f64> = npz.by_name("grid_bands").unwrap();
    let grid_members: ndarray::Array3<i64> = npz.by_name("grid_members").unwrap();
    let want_inter: ndarray::Array3<f64> = npz.by_name("inter_bands").unwrap();
    let inter_members: ndarray::Array3<i64> = npz.by_name("inter_members").unwrap();

    let ckpt = tabicl_candle::weights::load(root(), "regressor", &device).unwrap();
    let model = tabicl_candle::tabicl::TabIcl::from_checkpoint(ckpt).unwrap();

    let mut max_rel = 0f64;
    let mut compare = |got: &[f64], want: ndarray::ArrayView2<f64>, label: &str| {
        // got is (test_rows, alphas); want is (alphas, test_rows)
        for (a, row) in want.outer_iter().enumerate() {
            for (i, w) in row.iter().enumerate() {
                let v = got[i * ALPHAS.len() + a];
                let rel = (v - w).abs() / f64::max(1.0, w.abs());
                if rel > max_rel {
                    max_rel = rel;
                }
                assert!(rel < 1e-3, "{label} alpha {a} col {i}: rel {rel:e}");
            }
        }
    };

    for i in 0..grid_offsets.dim().0 {
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
        let members = members_from(grid_members.index_axis(ndarray::Axis(0), i));
        assert_eq!(members.len(), 4, "grid fits carry 4 members");
        let q = ensemble_bands(
            &model, &train_x, n, 2, &train_y, &test_x, 6, members, &device,
        );
        compare(
            &q,
            want_grid.index_axis(ndarray::Axis(0), i),
            &format!("grid fit {i}"),
        );
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
        let members = members_from(inter_members.index_axis(ndarray::Axis(0), i));
        assert_eq!(members.len(), 6, "interaction fits carry 6 members");
        let q = ensemble_bands(
            &model, &train_x, rows, 3, &train_y, &test_x, 6, members, &device,
        );
        compare(
            &q,
            want_inter.index_axis(ndarray::Axis(0), i),
            &format!("inter fit {i}"),
        );
    }
    println!("ensemble bands vs recorded-config oracle: max rel |diff| = {max_rel:e} over 21 fits");
}

#[test]
fn generated_members_are_valid_permutations() {
    for (n_features, n_members) in [(2, 4), (3, 6), (5, 8), (12, 8)] {
        let members = EnsembleMember::generate(n_features, n_members, 7);
        assert_eq!(members.len(), n_members.min(2 * n_features));
        for m in &members {
            let mut seen = vec![false; n_features];
            for &c in &m.shuffle {
                assert!(!seen[c], "duplicate column in shuffle");
                seen[c] = true;
            }
            assert!(seen.iter().all(|s| *s), "incomplete permutation");
        }
        // both norms present once members outnumber shuffles
        assert!(members.iter().any(|m| m.power) && members.iter().any(|m| !m.power));
    }
    // the pinned short-circuit
    let pinned = EnsembleMember::generate(4, 1, 7);
    assert_eq!(pinned.len(), 1);
    assert!(!pinned[0].power);
    assert_eq!(pinned[0].shuffle, vec![0, 1, 2, 3]);
}
