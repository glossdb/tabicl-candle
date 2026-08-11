//! The chain-rule density read (`TabICLUnsupervised.score_samples`):
//! for each permutation, each feature is scored by a conditional
//! estimator fit on the preceding features — a regressor read through
//! the quantile distribution for numerical features, a classifier
//! `predict_proba` lookup for categorical ones — and the per-feature
//! log densities sum; the score is exp of the mean over permutations.
//!
//! Two oracle mechanics are inputs here instead of ports: the
//! permutations (the sklearn source draws them from its latin Shuffler
//! seeded with Python's Mersenne Twister — nothing semantic rides on
//! that stream) and the standard-normal dummy column used when a
//! feature has no conditioning (numpy-Generator noise in the source).
//! Grading replays the oracle's recorded streams; production supplies
//! its own.

use candle_core::{Device, Tensor};

use crate::classifier::TabIclClassifier;
use crate::quantile::QuantileDist;
use crate::regressor::TabIclRegressor;
use crate::tabicl::TabIcl;

/// Features with fewer non-NaN training rows contribute nothing (the
/// sklearn `_MIN_SAMPLES_PER_CONDITIONAL`); they still appear in later
/// conditioning sets.
pub const MIN_SAMPLES_PER_CONDITIONAL: usize = 5;

/// The probability floor for categorical conditionals: unseen classes
/// score this, and every looked-up probability is clipped to it.
pub const PROBA_FLOOR: f64 = 1e-10;

pub struct Unsupervised<'a> {
    reg: &'a TabIcl,
    /// The classifier checkpoint; required iff `categorical` is
    /// non-empty (the sklearn class loads it on the same condition).
    clf: Option<&'a TabIcl>,
    x: Vec<f32>,
    rows: usize,
    cols: usize,
    categorical: Vec<usize>,
}

impl<'a> Unsupervised<'a> {
    /// x: row-major (rows, cols) training data, NaN allowed. Stored in
    /// f32 as the sklearn class stores `X_`. `categorical` lists the
    /// column indices read through the classifier.
    pub fn fit(
        reg: &'a TabIcl,
        clf: Option<&'a TabIcl>,
        x: Vec<f32>,
        rows: usize,
        cols: usize,
        categorical: Vec<usize>,
    ) -> Self {
        assert_eq!(x.len(), rows * cols);
        assert!(
            categorical.is_empty() || clf.is_some(),
            "categorical features need the classifier checkpoint"
        );
        Self {
            reg,
            clf,
            x,
            rows,
            cols,
            categorical,
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
            let observed: Vec<f32> = (0..test_rows)
                .map(|r| x_test[r * self.cols + col])
                .collect();

            if self.categorical.contains(&col) {
                self.categorical_conditional(
                    &x_tr,
                    &x_te,
                    n_cond,
                    &train_rows,
                    &y,
                    &observed,
                    &mut log_p,
                    device,
                )?;
            } else {
                self.numerical_conditional(
                    &x_tr,
                    &x_te,
                    n_cond,
                    &train_rows,
                    &y,
                    &observed,
                    &mut log_p,
                    device,
                )?;
            }
        }
        Ok(log_p)
    }

    /// Regressor fit + quantile-distribution log_prob of the observed
    /// values. NaN observations propagate, as the oracle's do.
    #[allow(clippy::too_many_arguments)]
    fn numerical_conditional(
        &self,
        x_tr: &[f64],
        x_te: &[f64],
        n_cond: usize,
        train_rows: &[usize],
        y: &[f64],
        observed: &[f32],
        log_p: &mut [f64],
        device: &Device,
    ) -> anyhow::Result<()> {
        let est = TabIclRegressor::fit(self.reg, x_tr, train_rows.len(), n_cond, y);
        let pred = est.predict(x_te, log_p.len(), device)?;

        // The oracle rebuilds the quantile distribution over the
        // y-space ensemble average; with one member that is the
        // member's sorted quantiles, affine-mapped.
        let dist = QuantileDist::new(&pred.raw_quantiles()?)?;
        let z = Tensor::from_vec(observed.to_vec(), (log_p.len(), 1), &Device::Cpu)?;
        let lp: Vec<f32> = dist.log_prob(&z)?.flatten_all()?.to_vec1()?;
        for (a, v) in log_p.iter_mut().zip(lp) {
            *a += v as f64;
        }
        Ok(())
    }

    /// Classifier fit + probability lookup for the observed class:
    /// searchsorted over the sorted classes, unseen classes floored at
    /// 1e-10, missing observations contributing 0.0 — the oracle's
    /// `_log_prob_categorical`. Class values pass through the oracle's
    /// int cast (truncation) on both the training and observed side.
    #[allow(clippy::too_many_arguments)]
    fn categorical_conditional(
        &self,
        x_tr: &[f64],
        x_te: &[f64],
        n_cond: usize,
        train_rows: &[usize],
        y: &[f64],
        observed: &[f32],
        log_p: &mut [f64],
        device: &Device,
    ) -> anyhow::Result<()> {
        let clf = self.clf.expect("checked at fit");
        let y_int: Vec<f64> = y.iter().map(|v| v.trunc()).collect();
        let est = TabIclClassifier::fit(clf, x_tr, train_rows.len(), n_cond, &y_int);
        let proba = est.predict_proba(x_te, log_p.len(), device)?;

        let n_classes = est.classes.len();
        for (r, (a, &obs)) in log_p.iter_mut().zip(observed).enumerate() {
            if obs.is_nan() {
                continue; // missing observation: 0.0 contribution
            }
            let obs = (obs as f64).trunc();
            let idx = est.classes.partition_point(|c| *c < obs).min(n_classes - 1);
            let p = if est.classes[idx] == obs {
                proba[r * n_classes + idx] as f64
            } else {
                PROBA_FLOOR
            };
            *a += p.max(PROBA_FLOOR).ln();
        }
        Ok(())
    }
}
