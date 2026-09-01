//! Stage 2: the sklearn wrapper pinned to one member (n_estimators=1,
//! norm method "none") — mean imputation, the unique-value filter,
//! standard scaling with clipping, the two-stage outlier soft clip, and
//! y standardization. The fit statistics are computed host-side in f64
//! exactly as numpy computes them; the model sees the f32 cast, as the
//! wrapper's `torch.from_numpy(...).float()` does.
//!
//! Multi-member ensembling (norm methods beyond "none", feature
//! shuffles) is deliberately out of scope until this member is graded.

use candle_core::{Device, Result, Tensor};

use crate::power::PowerStage;
use tabicl_model::quantile::QuantileDist;
use tabicl_model::tabicl::TabIcl;

/// The wrapper's preprocessing plane for one member: impute -> unique
/// filter -> standard scale (z clipped to +-100) -> optional power
/// normalization (the "power" norm method) -> outlier soft clip.
pub struct Preprocessor {
    cols: usize,
    impute_means: Vec<f64>,
    kept: Vec<usize>,
    // kept-column space from here on
    scale_mean: Vec<f64>,
    scale_scale: Vec<f64>,
    power: Option<PowerStage>,
    clip_lower: Vec<f64>,
    clip_upper: Vec<f64>,
    /// Preprocessed training rows (row-major, kept columns), cached at
    /// fit like the sklearn pipeline's `X_transformed_`.
    pub x_train: Vec<f64>,
    pub n_train: usize,
}

const SCALE_EPS: f64 = 1e-6;
const Z_CLIP: f64 = 100.0;
const OUTLIER_THRESHOLD: f64 = 4.0;

fn column(x: &[f64], cols: usize, c: usize) -> impl Iterator<Item = f64> + '_ {
    x.iter().skip(c).step_by(cols).copied()
}

impl Preprocessor {
    /// The norm-"none" pipeline (the pinned member's).
    pub fn fit(x: &[f64], rows: usize, cols: usize) -> Self {
        Self::fit_norm(x, rows, cols, false)
    }

    /// The norm-"power" pipeline: Yeo-Johnson between scaling and the
    /// outlier stage, as the sklearn PreprocessingPipeline orders it.
    pub fn fit_power(x: &[f64], rows: usize, cols: usize) -> Self {
        Self::fit_norm(x, rows, cols, true)
    }

    fn fit_norm(x: &[f64], rows: usize, cols: usize, use_power: bool) -> Self {
        assert_eq!(x.len(), rows * cols);

        // SimpleImputer: column means over non-NaN training values
        let impute_means: Vec<f64> = (0..cols)
            .map(|c| {
                let (s, n) = column(x, cols, c)
                    .filter(|v| !v.is_nan())
                    .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
                s / n as f64
            })
            .collect();
        let imputed: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if v.is_nan() {
                    impute_means[i % cols]
                } else {
                    *v
                }
            })
            .collect();

        // UniqueFeatureFilter: drop columns with a single unique value
        // (all columns kept when there are too few rows to judge)
        let kept: Vec<usize> = (0..cols)
            .filter(|&c| rows <= 1 || column(&imputed, cols, c).any(|v| v != imputed[c]))
            .collect();
        let k = kept.len();
        let filtered: Vec<f64> = (0..rows)
            .flat_map(|r| kept.iter().map(move |&c| (r, c)))
            .map(|(r, c)| imputed[r * cols + c])
            .collect();

        // CustomStandardScaler: z = (x - mean) / (std_pop + 1e-6), clip
        let scale_mean: Vec<f64> = (0..k)
            .map(|c| column(&filtered, k, c).sum::<f64>() / rows as f64)
            .collect();
        let scale_scale: Vec<f64> = (0..k)
            .map(|c| {
                let var = column(&filtered, k, c)
                    .map(|v| (v - scale_mean[c]).powi(2))
                    .sum::<f64>()
                    / rows as f64;
                var.sqrt() + SCALE_EPS
            })
            .collect();
        let scaled: Vec<f64> = filtered
            .iter()
            .enumerate()
            .map(|(i, v)| ((v - scale_mean[i % k]) / scale_scale[i % k]).clamp(-Z_CLIP, Z_CLIP))
            .collect();

        // Optional power normalization, fitted on the scaled matrix.
        let (power, normalized) = if use_power {
            let stage = PowerStage::fit(&scaled, rows, k);
            let normalized = stage.transform(&scaled, rows, k);
            (Some(stage), normalized)
        } else {
            (None, scaled)
        };

        // OutlierRemover, stage 1: sample-std bounds over all values
        let nanstats = |data: &[f64], c: usize| -> (f64, f64) {
            let vals: Vec<f64> = column(data, k, c).filter(|v| !v.is_nan()).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let ddof = if vals.len() > 1 { vals.len() - 1 } else { 1 };
            let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ddof as f64).sqrt();
            (mean, std.max(SCALE_EPS))
        };
        let stage1: Vec<(f64, f64)> = (0..k).map(|c| nanstats(&normalized, c)).collect();

        // Stage 2: recompute without the values outside stage-1 bounds
        let cleaned: Vec<f64> = normalized
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let (mean, std) = stage1[i % k];
                if (v - mean).abs() > OUTLIER_THRESHOLD * std {
                    f64::NAN
                } else {
                    *v
                }
            })
            .collect();
        let mut clip_lower = Vec::with_capacity(k);
        let mut clip_upper = Vec::with_capacity(k);
        for c in 0..k {
            let (mean, std) = nanstats(&cleaned, c);
            clip_lower.push(mean - OUTLIER_THRESHOLD * std);
            clip_upper.push(mean + OUTLIER_THRESHOLD * std);
        }

        let mut prep = Self {
            cols,
            impute_means,
            kept,
            scale_mean,
            scale_scale,
            power,
            clip_lower,
            clip_upper,
            x_train: Vec::new(),
            n_train: rows,
        };
        prep.x_train = prep.soft_clip(normalized);
        prep
    }

    /// The sequential log1p soft clip, second bound applied to the
    /// already-lower-clipped values (as the sklearn transform does).
    fn soft_clip(&self, mut x: Vec<f64>) -> Vec<f64> {
        let k = self.kept.len();
        for (i, v) in x.iter_mut().enumerate() {
            *v = f64::max(-(v.abs().ln_1p()) + self.clip_lower[i % k], *v);
            *v = f64::min(v.abs().ln_1p() + self.clip_upper[i % k], *v);
        }
        x
    }

    /// The test-time path: impute with training means, filter, scale,
    /// optional power normalization, soft clip. Returns row-major
    /// (rows, n_kept()).
    pub fn transform(&self, x: &[f64], rows: usize) -> Vec<f64> {
        assert_eq!(x.len(), rows * self.cols);
        let scaled: Vec<f64> = (0..rows)
            .flat_map(|r| self.kept.iter().enumerate().map(move |(i, &c)| (r, i, c)))
            .map(|(r, i, c)| {
                let v = x[r * self.cols + c];
                let v = if v.is_nan() { self.impute_means[c] } else { v };
                ((v - self.scale_mean[i]) / self.scale_scale[i]).clamp(-Z_CLIP, Z_CLIP)
            })
            .collect();
        let normalized = match &self.power {
            Some(stage) => stage.transform(&scaled, rows, self.kept.len()),
            None => scaled,
        };
        self.soft_clip(normalized)
    }

    pub fn n_kept(&self) -> usize {
        self.kept.len()
    }
}

/// The fitted regressor: preprocessing statistics, the standardized
/// training targets, and the y scale for mapping read-outs back.
pub struct TabIclRegressor<'a> {
    model: &'a TabIcl,
    pub prep: Preprocessor,
    pub y_mean: f64,
    pub y_scale: f64,
    y_scaled: Vec<f32>,
}

impl<'a> TabIclRegressor<'a> {
    /// x: row-major (rows, cols) training features, NaN allowed.
    pub fn fit(model: &'a TabIcl, x: &[f64], rows: usize, cols: usize, y: &[f64]) -> Self {
        // The wrapper casts y to float32 before standardizing; the f32
        // quantization is the visible part, so mirror the cast and
        // compute the statistics in f64 over the cast values.
        let yf: Vec<f64> = y.iter().map(|v| *v as f32 as f64).collect();
        let y_mean = yf.iter().sum::<f64>() / rows as f64;
        let std = (yf.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / rows as f64).sqrt();
        let y_scale = if std < 10.0 * f32::EPSILON as f64 {
            1.0
        } else {
            std
        };
        let y_scaled = yf.iter().map(|v| ((v - y_mean) / y_scale) as f32).collect();

        Self {
            model,
            prep: Preprocessor::fit(x, rows, cols),
            y_mean,
            y_scale,
            y_scaled,
        }
    }

    /// One forward over [train; test], read out as a quantile
    /// distribution still carrying the y scale.
    pub fn predict(&self, x: &[f64], rows: usize, device: &Device) -> Result<Prediction> {
        let k = self.prep.n_kept();
        let t = self.prep.n_train + rows;
        let mut all = self.prep.x_train.clone();
        all.extend(self.prep.transform(x, rows));
        let x32: Vec<f32> = all.iter().map(|v| *v as f32).collect();

        let x = Tensor::from_vec(x32, (1, t, k), device)?;
        let y = Tensor::from_slice(&self.y_scaled, (1, self.prep.n_train), device)?;
        let raw = self.model.forward(&x, &y)?; // (1, test, n_quantiles)

        Ok(Prediction {
            dist: QuantileDist::new(&raw)?,
            y_mean: self.y_mean,
            y_scale: self.y_scale,
        })
    }
}

/// Read-outs in the original y space. With one member the ensemble
/// average is the member itself, so each read is the distribution's,
/// affine-mapped by the y scaler — exactly the sklearn wrapper's order
/// (stats in scaled space, then inverse transform).
pub struct Prediction {
    pub dist: QuantileDist,
    y_mean: f64,
    y_scale: f64,
}

impl Prediction {
    fn unscale(&self, t: Tensor) -> Result<Tensor> {
        Ok(t.affine(self.y_scale, self.y_mean)?)
    }

    /// (test,) — the wrapper's `output_type="mean"`.
    pub fn mean(&self) -> Result<Tensor> {
        self.unscale(self.dist.mean()?.squeeze(0)?)
    }

    /// (test, alphas.len()) — the wrapper's `output_type="quantiles"`.
    pub fn quantiles(&self, alphas: &[f64]) -> Result<Tensor> {
        self.unscale(self.dist.icdf(alphas)?.squeeze(0)?)
    }

    /// (test, n_quantiles) — the wrapper's `output_type="raw_quantiles"`
    /// (monotone: the wrapper returns the distribution's sorted grid).
    pub fn raw_quantiles(&self) -> Result<Tensor> {
        self.unscale(self.dist.quantiles.squeeze(0)?)
    }
}
