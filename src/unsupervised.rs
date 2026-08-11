//! The chain-rule density read (`TabICLUnsupervised.score_samples`),
//! numerical columns only: for each permutation, each feature is scored
//! by a conditional regressor fit on the preceding features, and the
//! per-feature log densities sum; the score is exp of the mean over
//! permutations. The categorical conditional (the classifier wrapper's
//! `predict_proba`) is the open half and is not ported yet.
//!
//! Two oracle mechanics are inputs here instead of ports: the
//! permutations (the sklearn source draws them from its latin Shuffler
//! seeded with Python's Mersenne Twister — nothing semantic rides on
//! that stream) and the standard-normal dummy column used when a
//! feature has no conditioning (numpy-Generator noise in the source).
//! Grading replays the oracle's recorded streams; production supplies
//! its own.

use candle_core::{Device, Tensor};

use crate::quantile::QuantileDist;
use crate::regressor::TabIclRegressor;
use crate::tabicl::TabIcl;

/// Features with fewer non-NaN training rows contribute nothing (the
/// sklearn `_MIN_SAMPLES_PER_CONDITIONAL`); they still appear in later
/// conditioning sets.
pub const MIN_SAMPLES_PER_CONDITIONAL: usize = 5;

pub struct Unsupervised<'a> {
    model: &'a TabIcl,
    x: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl<'a> Unsupervised<'a> {
    /// x: row-major (rows, cols) training data, NaN allowed. Stored in
    /// f32 as the sklearn class stores `X_`.
    pub fn fit(model: &'a TabIcl, x: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(x.len(), rows * cols);
        Self {
            model,
            x,
            rows,
            cols,
        }
    }

    /// Outlier scores: higher = more normal. `noise` supplies the dummy
    /// standard-normal column for empty conditionings, called once for
    /// the train rows and once for the test rows of each permutation's
    /// first feature (skipped features consume nothing).
    pub fn score_samples(
        &self,
        x_test: &[f32],
        test_rows: usize,
        permutations: &[Vec<usize>],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        device: &Device,
    ) -> anyhow::Result<Vec<f64>> {
        let mut acc = vec![0f64; test_rows];
        for perm in permutations {
            let lp = self.log_density(x_test, test_rows, perm, noise, device)?;
            for (a, v) in acc.iter_mut().zip(lp) {
                *a += v;
            }
        }
        let k = permutations.len() as f64;
        Ok(acc.into_iter().map(|s| (s / k).exp()).collect())
    }

    /// One permutation's summed per-feature log densities, (test_rows,).
    pub fn log_density(
        &self,
        x_test: &[f32],
        test_rows: usize,
        perm: &[usize],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        device: &Device,
    ) -> anyhow::Result<Vec<f64>> {
        assert_eq!(x_test.len(), test_rows * self.cols);
        let mut log_p = vec![0f64; test_rows];

        for (i, &col) in perm.iter().enumerate() {
            let train_rows: Vec<usize> = (0..self.rows)
                .filter(|&r| !self.x[r * self.cols + col].is_nan())
                .collect();
            if train_rows.len() < MIN_SAMPLES_PER_CONDITIONAL {
                continue;
            }

            let cond = &perm[..i];
            let (x_tr, x_te, n_cond): (Vec<f64>, Vec<f64>, usize) = if cond.is_empty() {
                let tr = noise(train_rows.len()).iter().map(|v| *v as f64).collect();
                let te = noise(test_rows).iter().map(|v| *v as f64).collect();
                (tr, te, 1)
            } else {
                let tr = train_rows
                    .iter()
                    .flat_map(|&r| cond.iter().map(move |&c| self.x[r * self.cols + c] as f64))
                    .collect();
                let te = (0..test_rows)
                    .flat_map(|r| cond.iter().map(move |&c| x_test[r * self.cols + c] as f64))
                    .collect();
                (tr, te, cond.len())
            };
            let y: Vec<f64> = train_rows
                .iter()
                .map(|&r| self.x[r * self.cols + col] as f64)
                .collect();

            let est = TabIclRegressor::fit(self.model, &x_tr, train_rows.len(), n_cond, &y);
            let pred = est.predict(&x_te, test_rows, device)?;

            // The oracle rebuilds the quantile distribution over the
            // y-space ensemble average; with one member that is the
            // member's sorted quantiles, affine-mapped.
            let dist = QuantileDist::new(&pred.raw_quantiles()?)?;
            let observed: Vec<f32> = (0..test_rows)
                .map(|r| x_test[r * self.cols + col])
                .collect();
            let z = Tensor::from_vec(observed, (test_rows, 1), &Device::Cpu)?;
            let lp: Vec<f32> = dist.log_prob(&z)?.flatten_all()?.to_vec1()?;
            for (a, v) in log_p.iter_mut().zip(lp) {
                *a += v as f64;
            }
        }
        Ok(log_p)
    }
}
