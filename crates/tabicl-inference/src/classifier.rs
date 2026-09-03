//! The classifier wrapper pinned to one member (n_estimators=1, norm
//! method "none") — the same preprocessing plane as the regressor
//! (identical `PreprocessingPipeline` construction; the classification
//! flag never reaches it), plus label encoding and the logit read-out:
//! slice to the classes present, temperature softmax, renormalize.
//!
//! With one member both the feature and the class shuffle short-circuit
//! to identity, so the ensemble average is the member itself and the
//! class-shuffle correction is a no-op — the same shape the density
//! read's inner classifiers have.

use candle_core::{D, Device, Result, Tensor};

use crate::regressor::Preprocessor;
use tabicl_model::tabicl::TabIcl;

pub const SOFTMAX_TEMPERATURE: f32 = 0.9;

/// The fitted classifier: preprocessing statistics, the sorted class
/// values (LabelEncoder's `classes_`), and the encoded training labels.
pub struct TabIclClassifier<'a> {
    model: &'a TabIcl,
    pub prep: Preprocessor,
    /// Sorted unique class values; `predict_proba` columns follow this
    /// order, as the sklearn wrapper's do.
    pub classes: Vec<f64>,
    y_encoded: Vec<f32>,
}

impl<'a> TabIclClassifier<'a> {
    /// x: row-major (rows, cols) training features, NaN allowed.
    /// y: class values (NaN not allowed — encode/drop missing upstream,
    /// as the density read does).
    pub fn fit(model: &'a TabIcl, x: &[f64], rows: usize, cols: usize, y: &[f64]) -> Self {
        assert_eq!(y.len(), rows);
        assert!(y.iter().all(|v| !v.is_nan()), "NaN class label");

        // LabelEncoder: sorted unique values -> 0..n_classes
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        let y_encoded = y
            .iter()
            .map(|v| classes.partition_point(|c| c < v) as f32)
            .collect();

        Self {
            model,
            prep: Preprocessor::fit(x, rows, cols),
            classes,
            y_encoded,
        }
    }

    /// One forward over [train; test], logits sliced to the classes
    /// present, softmax at the wrapper's temperature, renormalized.
    /// Returns row-major (rows, n_classes) probabilities, columns in
    /// `classes` order. The softmax runs host-side in f32, as the
    /// wrapper's numpy helper does after the f32 logits leave torch.
    pub fn predict_proba(&self, x: &[f64], rows: usize, device: &Device) -> Result<Vec<f32>> {
        let k = self.prep.n_kept();
        let t = self.prep.n_train + rows;
        let mut all = self.prep.x_train.clone();
        all.extend(self.prep.transform(x, rows));
        let x32: Vec<f32> = all.iter().map(|v| *v as f32).collect();

        let x = Tensor::from_vec(x32, (1, t, k), device)?;
        let y = Tensor::from_slice(&self.y_encoded, (1, self.prep.n_train), device)?;
        let out = self.model.forward(&x, &y)?; // (1, test, max_classes)

        let n_classes = self.classes.len();
        if n_classes > out.dim(D::Minus1)? {
            candle_core::bail!(
                "{} classes exceed the model's max_classes {}",
                n_classes,
                out.dim(D::Minus1)?
            );
        }
        let logits: Vec<f32> = out
            .narrow(D::Minus1, 0, n_classes)?
            .contiguous()?
            .flatten_all()?
            .to_vec1()?;

        let mut proba = Vec::with_capacity(rows * n_classes);
        for row in logits.chunks_exact(n_classes) {
            let scaled: Vec<f32> = row.iter().map(|v| v / SOFTMAX_TEMPERATURE).collect();
            let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scaled.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let softmaxed: Vec<f32> = exp.iter().map(|v| v / sum).collect();
            // the wrapper renormalizes once more after averaging
            let sum: f32 = softmaxed.iter().sum();
            proba.extend(softmaxed.iter().map(|v| v / sum));
        }
        Ok(proba)
    }

    /// Argmax over `predict_proba`, mapped back to class values.
    pub fn predict(&self, x: &[f64], rows: usize, device: &Device) -> Result<Vec<f64>> {
        let proba = self.predict_proba(x, rows, device)?;
        Ok(proba
            .chunks_exact(self.classes.len())
            .map(|row| {
                // np.argmax: first index of the maximum
                let best = row
                    .iter()
                    .enumerate()
                    .fold(
                        (0, f32::NEG_INFINITY),
                        |acc, (i, &v)| {
                            if v > acc.1 { (i, v) } else { acc }
                        },
                    )
                    .0;
                self.classes[best]
            })
            .collect())
    }
}
