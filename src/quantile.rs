//! The regressor read-out: raw quantiles -> QuantileDistribution, as
//! the model's own `QuantileToDistribution` defaults do it — sort-based
//! crossing fix, piecewise-linear interior on the i/(n+1) grid, and
//! exponential tails fit by log-space regression over the outer
//! min(20, n/4) quantiles. `icdf` gives the bands; `mean` is the mean
//! of the monotone quantiles (the model's `predict_stats("mean")`).

use candle_core::{D, Tensor};

const TOL: f64 = 1e-6;
const MIN_BETA: f64 = 0.01;
const MAX_BETA: f64 = 100.0;
const TAIL_QUANTILES: usize = 20;

pub struct QuantileDist {
    /// Monotone quantiles, shape (*batch, n).
    pub quantiles: Tensor,
    tail_a_l: Tensor, // (*batch, 1) each
    tail_b_l: Tensor,
    tail_a_r: Tensor,
    tail_b_r: Tensor,
    grid: Vec<f64>, // alpha levels, length n
}

impl QuantileDist {
    /// raw: (*batch, n) predicted quantiles, crossings allowed.
    pub fn new(raw: &Tensor) -> anyhow::Result<Self> {
        let n = raw.dim(D::Minus1)?;
        // candle 0.9's CPU argsort reads its input from storage index 0,
        // ignoring the view's start offset (candle-core sort.rs, asort) —
        // a narrow'd forward output sorts the wrong storage window while
        // gather then reads the right one, silently mis-sorting.
        // `.contiguous()` is a no-op on such views (strides pass the
        // check), so materialize a zero-offset copy via an elementwise
        // op before sorting. Metal applies the offset correctly; the
        // copy is cheap and uniform across devices.
        let (quantiles, _) = raw.affine(1.0, 0.0)?.sort_last_dim(true)?;
        let grid: Vec<f64> = (1..=n).map(|i| i as f64 / (n + 1) as f64).collect();
        let k = TAIL_QUANTILES.min(n / 4);

        // Exponential tails: regress Q against ln(alpha) (left) and
        // ln(1 - alpha) (right) over the outer k quantiles;
        // beta = |cov / var| clamped, then Q(a) = a_tail * ln(.) + b.
        let dev = raw.device();
        let regress = |q_k: &Tensor, ln_x: &[f64]| -> anyhow::Result<Tensor> {
            let mean = ln_x.iter().sum::<f64>() / k as f64;
            let centered: Vec<f32> = ln_x.iter().map(|v| (v - mean) as f32).collect();
            let var = centered.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / k as f64;
            let centered = Tensor::from_vec(centered, k, dev)?;
            let q_mean = q_k.mean_keepdim(D::Minus1)?;
            let cov = q_k
                .broadcast_sub(&q_mean)?
                .broadcast_mul(&centered)?
                .mean_keepdim(D::Minus1)?;
            Ok((cov / var.max(TOL))?)
        };

        let ln_alpha_left: Vec<f64> = grid[..k].iter().map(|a| a.max(TOL).ln()).collect();
        let beta_l = regress(&quantiles.narrow(D::Minus1, 0, k)?, &ln_alpha_left)?
            .abs()?
            .clamp(MIN_BETA, MAX_BETA)?;
        let ln_1ma_right: Vec<f64> = grid[n - k..]
            .iter()
            .map(|a| (1.0 - a).max(TOL).ln())
            .collect();
        let beta_r = regress(&quantiles.narrow(D::Minus1, n - k, k)?, &ln_1ma_right)?
            .neg()?
            .abs()?
            .clamp(MIN_BETA, MAX_BETA)?;

        let q_l = quantiles.narrow(D::Minus1, 0, 1)?;
        let q_r = quantiles.narrow(D::Minus1, n - 1, 1)?;
        let alpha_l = grid[0].max(TOL);
        let alpha_r = grid[n - 1].min(1.0 - TOL);
        let tail_a_l = beta_l;
        let tail_b_l = (&q_l - (&tail_a_l * alpha_l.ln())?)?;
        let tail_a_r = beta_r.neg()?;
        let tail_b_r = (&q_r - (&tail_a_r * (1.0 - alpha_r).ln())?)?;

        Ok(Self {
            quantiles,
            tail_a_l,
            tail_b_l,
            tail_a_r,
            tail_b_r,
            grid,
        })
    }

    /// Bands at the given levels: (*batch, alphas.len()).
    pub fn icdf(&self, alphas: &[f64]) -> anyhow::Result<Tensor> {
        let n = self.grid.len();
        let (alpha_l, alpha_r) = (self.grid[0], self.grid[n - 1]);
        let mut cols = Vec::with_capacity(alphas.len());
        for &alpha in alphas {
            let col = if alpha < alpha_l {
                (&self.tail_a_l * alpha.max(TOL).ln())?.add(&self.tail_b_l)?
            } else if alpha > alpha_r {
                (&self.tail_a_r * (1.0 - alpha).max(TOL).ln())?.add(&self.tail_b_r)?
            } else if alpha >= alpha_r {
                self.quantiles.narrow(D::Minus1, n - 1, 1)?
            } else {
                // searchsorted(grid[..n-1], alpha, right) - 1, clamped
                let seg =
                    match self.grid[..n - 1].binary_search_by(|g| g.partial_cmp(&alpha).unwrap()) {
                        Ok(i) => i,
                        Err(i) => i.saturating_sub(1),
                    }
                    .min(n - 2);
                let lo = self.quantiles.narrow(D::Minus1, seg, 1)?;
                let hi = self.quantiles.narrow(D::Minus1, seg + 1, 1)?;
                let t = ((alpha - self.grid[seg]) / (self.grid[seg + 1] - self.grid[seg]).max(TOL))
                    .clamp(0.0, 1.0);
                (&lo + ((hi - &lo)? * t)?)?
            };
            cols.push(col);
        }
        Ok(Tensor::cat(&cols.iter().collect::<Vec<_>>(), D::Minus1)?)
    }

    /// Mean of the monotone quantiles: (*batch).
    pub fn mean(&self) -> anyhow::Result<Tensor> {
        Ok(self.quantiles.mean(D::Minus1)?)
    }

    /// Log density at z: (*batch, k) -> (*batch, k), via
    /// `-log(clamp(Q'(F(z)), MIN_SLOPE, MAX_SLOPE))` — the read the
    /// chain-rule density orchestration uses per numerical conditional.
    /// Scalar per-element work, computed host-side in f32 with torch's
    /// exact branch conditions (f32 grid comparisons matter at knots).
    pub fn log_prob(&self, z: &Tensor) -> anyhow::Result<Tensor> {
        const MIN_SLOPE: f32 = 1e-6;
        const MAX_SLOPE: f32 = 1e6;
        let tol = TOL as f32;

        let n = self.grid.len();
        let z_dims = z.dims().to_vec();
        let k = *z_dims.last().unwrap();
        let m: usize = z_dims[..z_dims.len() - 1].iter().product();

        let grid: Vec<f32> = self.grid.iter().map(|&a| a as f32).collect();
        let (alpha_l, alpha_r) = (grid[0], grid[n - 1]);
        let q: Vec<f32> = self.quantiles.flatten_all()?.to_vec1()?;
        let a_l: Vec<f32> = self.tail_a_l.flatten_all()?.to_vec1()?;
        let b_l: Vec<f32> = self.tail_b_l.flatten_all()?.to_vec1()?;
        let a_r: Vec<f32> = self.tail_a_r.flatten_all()?.to_vec1()?;
        let b_r: Vec<f32> = self.tail_b_r.flatten_all()?.to_vec1()?;
        let zs: Vec<f32> = z.contiguous()?.flatten_all()?.to_vec1()?;

        let mut out = Vec::with_capacity(m * k);
        for row in 0..m {
            let qr = &q[row * n..(row + 1) * n];
            let (q_l, q_r) = (qr[0], qr[n - 1]);
            for &zv in &zs[row * k..(row + 1) * k] {
                // cdf: F(z), region by z against the boundary quantiles
                let alpha = if zv < q_l {
                    let a_safe = a_l[row].abs().max(tol);
                    ((zv - b_l[row]) / a_safe)
                        .min(0.0)
                        .exp()
                        .clamp(0.0, alpha_l)
                } else if zv > q_r {
                    let a_safe = a_r[row].abs().max(tol);
                    let one_m = ((zv - b_r[row]) / -a_safe).min(0.0).exp();
                    (1.0 - one_m).clamp(alpha_r, 1.0)
                } else {
                    // searchsorted(q_lo, z, right) - 1 over this row's
                    // segment starts, clamped to a valid segment
                    let seg = qr[..n - 1].partition_point(|v| *v <= zv).saturating_sub(1);
                    let t = ((zv - qr[seg]) / (qr[seg + 1] - qr[seg]).max(tol)).clamp(0.0, 1.0);
                    let a = grid[seg] + t * (grid[seg + 1] - grid[seg]);
                    if zv >= q_r { alpha_r } else { a }
                };
                // dQ/dalpha at F(z), region by alpha against the grid ends
                let deriv = if alpha < alpha_l {
                    a_l[row] / alpha.max(tol)
                } else if alpha > alpha_r {
                    -a_r[row] / (1.0 - alpha).max(tol)
                } else {
                    let seg = grid[..n - 1]
                        .partition_point(|v| *v <= alpha)
                        .saturating_sub(1);
                    (qr[seg + 1] - qr[seg]) / (grid[seg + 1] - grid[seg]).max(tol)
                };
                out.push(-deriv.clamp(MIN_SLOPE, MAX_SLOPE).ln());
            }
        }
        Ok(Tensor::from_vec(out, z_dims, z.device())?)
    }
}
