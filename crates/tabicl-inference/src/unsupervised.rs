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
//!
//! Scheduling is the caller's: `tasks` decomposes the read into
//! independent conditionals (noise pre-drawn in sequential order) and
//! `run` executes one — this library never spawns threads or picks a
//! pool. The convenience reads (`score_samples`, `score_log_mean`)
//! run the decomposition sequentially.

use candle_core::{Device, Result, Tensor};

use crate::classifier::TabIclClassifier;
use crate::regressor::TabIclRegressor;
use tabicl_model::quantile::QuantileDist;
use tabicl_model::tabicl::TabIcl;

/// Features with fewer non-NaN training rows contribute nothing (the
/// sklearn `_MIN_SAMPLES_PER_CONDITIONAL`); they still appear in later
/// conditioning sets.
pub const MIN_SAMPLES_PER_CONDITIONAL: usize = 5;

/// The probability floor for categorical conditionals: unseen classes
/// score this, and every looked-up probability is clipped to it.
pub const PROBA_FLOOR: f64 = 1e-10;

/// One independent unit of the chain-rule read: one feature's
/// conditional within one permutation. `cond` preserves the
/// permutation's order; the pre-drawn dummy-noise columns (empty
/// conditioning only) ride along, so tasks can run under any schedule
/// and still reproduce the sequential read.
pub struct Task {
    pub cond: Vec<usize>,
    pub col: usize,
    dummy: Option<(Vec<f32>, Vec<f32>)>,
}

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
    ) -> Result<Vec<f64>> {
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

    /// Decompose the read into its independent units — the caller
    /// schedules: sequentially, on its own thread pool (the
    /// conditionals are independent), or in order on an accelerator's
    /// one queue. Tasks come back in the sequential (permutation,
    /// position) order, and the dummy-noise streams are drawn here in
    /// that order, so any schedule reproduces the sequential read.
    pub fn tasks(
        &self,
        test_rows: usize,
        permutations: &[Vec<usize>],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Vec<Task> {
        let mut tasks = Vec::new();
        for perm in permutations {
            for (i, &col) in perm.iter().enumerate() {
                let train_len = self.usable_train_rows(col);
                if train_len < MIN_SAMPLES_PER_CONDITIONAL {
                    continue;
                }
                let dummy = (i == 0).then(|| (noise(train_len), noise(test_rows)));
                tasks.push(Task {
                    cond: perm[..i].to_vec(),
                    col,
                    dummy,
                });
            }
        }
        tasks
    }

    /// Run one task: (test_rows,) per-row conditional log densities.
    pub fn run(
        &self,
        x_test: &[f32],
        test_rows: usize,
        task: Task,
        device: &Device,
    ) -> Result<Vec<f64>> {
        assert_eq!(x_test.len(), test_rows * self.cols);
        self.conditional(x_test, test_rows, &task.cond, task.col, task.dummy, device)
    }

    /// One permutation's summed per-feature log densities, (test_rows,).
    pub fn log_density(
        &self,
        x_test: &[f32],
        test_rows: usize,
        perm: &[usize],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        device: &Device,
    ) -> Result<Vec<f64>> {
        let perm = perm.to_vec();
        let tasks = self.tasks(test_rows, std::slice::from_ref(&perm), noise);
        self.sum_tasks(x_test, test_rows, tasks, device)
    }

    /// Mean per-row log density across permutations — the log-space read
    /// the misfit door consumes (`score_samples` keeps the oracle's exp).
    /// Sequential over the decomposition; a caller that wants
    /// parallelism schedules `tasks` on its own pool and averages the
    /// summed results itself.
    pub fn score_log_mean(
        &self,
        x_test: &[f32],
        test_rows: usize,
        permutations: &[Vec<usize>],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        device: &Device,
    ) -> Result<Vec<f64>> {
        let tasks = self.tasks(test_rows, permutations, noise);
        let acc = self.sum_tasks(x_test, test_rows, tasks, device)?;
        let k = permutations.len() as f64;
        Ok(acc.into_iter().map(|s| s / k).collect())
    }

    fn sum_tasks(
        &self,
        x_test: &[f32],
        test_rows: usize,
        tasks: Vec<Task>,
        device: &Device,
    ) -> Result<Vec<f64>> {
        let mut acc = vec![0f64; test_rows];
        for task in tasks {
            let lp = self.run(x_test, test_rows, task, device)?;
            for (a, v) in acc.iter_mut().zip(lp) {
                *a += v;
            }
        }
        Ok(acc)
    }

    fn usable_train_rows(&self, col: usize) -> usize {
        (0..self.rows)
            .filter(|&r| !self.x[r * self.cols + col].is_nan())
            .count()
    }

    /// One feature's conditional log densities, (test_rows,): the body
    /// the permutation walks share. `dummy` carries the pre-drawn
    /// standard-normal (train, test) columns for an empty conditioning.
    fn conditional(
        &self,
        x_test: &[f32],
        test_rows: usize,
        cond: &[usize],
        col: usize,
        dummy: Option<(Vec<f32>, Vec<f32>)>,
        device: &Device,
    ) -> Result<Vec<f64>> {
        let train_rows: Vec<usize> = (0..self.rows)
            .filter(|&r| !self.x[r * self.cols + col].is_nan())
            .collect();
        let (x_tr, x_te, n_cond): (Vec<f64>, Vec<f64>, usize) = match dummy {
            Some((tr, te)) => {
                debug_assert!(cond.is_empty());
                (
                    tr.iter().map(|v| *v as f64).collect(),
                    te.iter().map(|v| *v as f64).collect(),
                    1,
                )
            }
            None => {
                let tr = train_rows
                    .iter()
                    .flat_map(|&r| cond.iter().map(move |&c| self.x[r * self.cols + c] as f64))
                    .collect();
                let te = (0..test_rows)
                    .flat_map(|r| cond.iter().map(move |&c| x_test[r * self.cols + c] as f64))
                    .collect();
                (tr, te, cond.len())
            }
        };
        let y: Vec<f64> = train_rows
            .iter()
            .map(|&r| self.x[r * self.cols + col] as f64)
            .collect();
        let observed: Vec<f32> = (0..test_rows)
            .map(|r| x_test[r * self.cols + col])
            .collect();

        let mut log_p = vec![0f64; test_rows];
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
    ) -> Result<()> {
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
    ) -> Result<()> {
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
