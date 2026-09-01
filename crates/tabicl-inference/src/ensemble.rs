//! The multi-member ensemble over the regressor: per-norm
//! preprocessing pipelines ("none" and "power"), per-member feature
//! permutations, one forward per member, and the sklearn wrapper's
//! aggregation — quantiles from each member's own distribution,
//! inverse-scaled, arithmetic-meaned across members.
//!
//! Member *machinery* is graded against the recorded sklearn run with
//! injected configs (tests/e4_ensemble.rs). Member *generation* for
//! production use follows the same construction (latin-square feature
//! shuffles crossed with the two norm methods) but draws from this
//! crate's own deterministic RNG, so which permutation a member gets
//! differs from sklearn's Python-`random` selection. The member set's
//! diversity, not its identity, is what the ensemble buys.

use candle_core::{Device, Result, Tensor};

use tabicl_model::quantile::QuantileDist;
use crate::regressor::Preprocessor;
use tabicl_model::tabicl::TabIcl;

/// One ensemble member: which pipeline, and the feature permutation
/// (over kept columns) its forward sees.
#[derive(Clone, Debug)]
pub struct EnsembleMember {
    pub power: bool,
    pub shuffle: Vec<usize>,
}

/// SplitMix64 — a small deterministic generator for member creation.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// A random reduced latin square, the same construction the sklearn
/// side uses (recursive insertion, then row shuffle / transpose / row
/// shuffle) with this crate's RNG.
fn latin_squares(n: usize, rng: &mut SplitMix64) -> Vec<Vec<usize>> {
    fn rls(symbols: &mut Vec<usize>, rng: &mut SplitMix64) -> Vec<Vec<usize>> {
        let n = symbols.len();
        if n == 1 {
            return vec![symbols.clone()];
        }
        let sym = symbols.remove(rng.below(n));
        let mut square = rls(symbols, rng);
        square.push(square[0].clone());
        for (i, row) in square.iter_mut().enumerate() {
            row.insert(i, sym);
        }
        square
    }
    let mut symbols: Vec<usize> = (0..n).collect();
    let mut square = rls(&mut symbols, rng);
    rng.shuffle(&mut square);
    let mut trans: Vec<Vec<usize>> = (0..n)
        .map(|c| square.iter().map(|row| row[c]).collect())
        .collect();
    rng.shuffle(&mut trans);
    trans
}

impl EnsembleMember {
    /// Generate member configs for production use: latin feature
    /// shuffles crossed with the norm methods, truncated to
    /// `n_members`. `n_members == 1` short-circuits to the pinned
    /// member (identity shuffle, norm "none"), as the wrapper does.
    pub fn generate(n_features: usize, n_members: usize, seed: u64) -> Vec<EnsembleMember> {
        if n_members <= 1 || n_features == 0 {
            return vec![EnsembleMember {
                power: false,
                shuffle: (0..n_features).collect(),
            }];
        }
        let mut rng = SplitMix64(seed);
        let mut shuffles = latin_squares(n_features, &mut rng);
        rng.shuffle(&mut shuffles);
        shuffles
            .iter()
            .flat_map(|s| {
                [false, true].map(|power| EnsembleMember {
                    power,
                    shuffle: s.clone(),
                })
            })
            .take(n_members)
            .collect()
    }
}

/// The fitted ensemble: the pipelines its members need, the shared y
/// scaling, and the member list.
pub struct TabIclEnsemble<'a> {
    model: &'a TabIcl,
    prep_none: Option<Preprocessor>,
    prep_power: Option<Preprocessor>,
    pub members: Vec<EnsembleMember>,
    pub y_mean: f64,
    pub y_scale: f64,
    y_scaled: Vec<f32>,
    n_train: usize,
}

impl<'a> TabIclEnsemble<'a> {
    /// x: row-major (rows, cols) training features, NaN allowed.
    /// Member shuffles index the kept-column space after the unique
    /// filter.
    pub fn fit(
        model: &'a TabIcl,
        x: &[f64],
        rows: usize,
        cols: usize,
        y: &[f64],
        members: Vec<EnsembleMember>,
    ) -> Self {
        assert!(!members.is_empty());
        // The wrapper's y standardization, shared by every member.
        let yf: Vec<f64> = y.iter().map(|v| *v as f32 as f64).collect();
        let y_mean = yf.iter().sum::<f64>() / rows as f64;
        let std = (yf.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / rows as f64).sqrt();
        let y_scale = if std < 10.0 * f32::EPSILON as f64 {
            1.0
        } else {
            std
        };
        let y_scaled = yf.iter().map(|v| ((v - y_mean) / y_scale) as f32).collect();

        let prep_none = members
            .iter()
            .any(|m| !m.power)
            .then(|| Preprocessor::fit(x, rows, cols));
        let prep_power = members
            .iter()
            .any(|m| m.power)
            .then(|| Preprocessor::fit_power(x, rows, cols));

        Self {
            model,
            prep_none,
            prep_power,
            members,
            y_mean,
            y_scale,
            y_scaled,
            n_train: rows,
        }
    }

    /// Bands in the original y space: one forward per member, each
    /// member's quantiles at `alphas` from its own distribution,
    /// averaged. Returns row-major (rows, alphas.len()).
    pub fn predict_quantiles(
        &self,
        x: &[f64],
        rows: usize,
        alphas: &[f64],
        device: &Device,
    ) -> Result<Vec<f64>> {
        let t = self.n_train + rows;
        let mut acc = vec![0.0f64; rows * alphas.len()];
        for member in &self.members {
            let prep = if member.power {
                self.prep_power.as_ref().expect("power pipeline fitted")
            } else {
                self.prep_none.as_ref().expect("none pipeline fitted")
            };
            let k = prep.n_kept();
            assert_eq!(member.shuffle.len(), k, "shuffle over kept columns");
            let mut all = prep.x_train.clone();
            all.extend(prep.transform(x, rows));
            let all = &all;
            let permuted: Vec<f32> = (0..t)
                .flat_map(|r| member.shuffle.iter().map(move |&c| all[r * k + c] as f32))
                .collect();
            let xt = Tensor::from_vec(permuted, (1, t, k), device)?;
            let yt = Tensor::from_slice(&self.y_scaled, (1, self.n_train), device)?;
            let raw = self.model.forward(&xt, &yt)?;
            let dist = QuantileDist::new(&raw)?;
            let q: Vec<f32> = dist.icdf(alphas)?.squeeze(0)?.flatten_all()?.to_vec1()?;
            for (slot, v) in acc.iter_mut().zip(q) {
                *slot += v as f64 * self.y_scale + self.y_mean;
            }
        }
        let n = self.members.len() as f64;
        acc.iter_mut().for_each(|v| *v /= n);
        Ok(acc)
    }
}
