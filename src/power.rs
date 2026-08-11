//! The Yeo-Johnson power stage: sklearn's
//! `PowerTransformer(method="yeo-johnson", standardize=True)` — the
//! "power" half of the wrapper's default norm methods. Lambda per
//! feature by MLE exactly as scipy computes it (`yeojohnson_normmax`
//! with data-dependent bounds, minimized by the bounded Brent
//! algorithm behind `fminbound`), the transform in the same
//! log1p/expm1 stable forms, then standard scaling of the transformed
//! features. All statistics in f64, matching the sklearn side.

const EPS: f64 = f64::EPSILON;

/// The Yeo-Johnson transform of one value at a given lambda, in
/// scipy's numerically stable form.
pub fn yeojohnson(x: f64, lmbda: f64) -> f64 {
    if x >= 0.0 {
        if lmbda.abs() < EPS {
            x.ln_1p()
        } else {
            (lmbda * x.ln_1p()).exp_m1() / lmbda
        }
    } else if (lmbda - 2.0).abs() > EPS {
        -((2.0 - lmbda) * (-x).ln_1p()).exp_m1() / (2.0 - lmbda)
    } else {
        -(-x).ln_1p()
    }
}

fn population_var(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

fn logsumexp(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = v.clone().fold(f64::NEG_INFINITY, f64::max);
    if m.is_infinite() {
        return m;
    }
    m + v.map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// log|e^a - e^b|, stable; NEG_INFINITY when a == b.
fn log_abs_diff_exp(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    if d == 0.0 {
        f64::NEG_INFINITY
    } else {
        a.max(b) + (-(-d).exp()).ln_1p()
    }
}

/// scipy's `_log_var`: log of the population variance of e^logx,
/// computed from logx without leaving log space.
fn log_var(logx: &[f64]) -> f64 {
    let n = logx.len() as f64;
    let logmean = logsumexp(logx.iter().copied()) - n.ln();
    let logxmu: Vec<f64> = logx.iter().map(|&a| log_abs_diff_exp(a, logmean)).collect();
    logsumexp(logxmu.iter().map(|v| 2.0 * v)) - n.ln()
}

/// scipy's `yeojohnson_llf` over finite data: the log-likelihood of
/// the transformed data under a normal model, with the all-positive /
/// all-negative branches computed in log space for stability.
pub fn yeojohnson_llf(lmb: f64, data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let all_pos = data.iter().all(|v| *v >= 0.0);
    let all_neg = data.iter().all(|v| *v < 0.0);

    let logvar = if all_pos {
        if lmb.abs() < EPS {
            let logs: Vec<f64> = data.iter().map(|x| x.ln_1p()).collect();
            population_var(&logs).ln()
        } else {
            let logs: Vec<f64> = data.iter().map(|x| lmb * x.ln_1p()).collect();
            log_var(&logs) - 2.0 * lmb.abs().ln()
        }
    } else if all_neg {
        if (lmb - 2.0).abs() < EPS {
            let logs: Vec<f64> = data.iter().map(|x| (-x).ln_1p()).collect();
            population_var(&logs).ln()
        } else {
            let logs: Vec<f64> = data.iter().map(|x| (2.0 - lmb) * (-x).ln_1p()).collect();
            log_var(&logs) - 2.0 * (2.0 - lmb).abs().ln()
        }
    } else {
        let y: Vec<f64> = data.iter().map(|&x| yeojohnson(x, lmb)).collect();
        let sigma = population_var(&y);
        if sigma >= f64::MIN_POSITIVE {
            sigma.ln()
        } else {
            f64::NEG_INFINITY
        }
    };

    let pull: f64 = data
        .iter()
        .map(|x| {
            if *x == 0.0 {
                0.0
            } else {
                x.signum() * x.abs().ln_1p()
            }
        })
        .sum();
    -n / 2.0 * logvar + (lmb - 1.0) * pull
}

/// scipy's `fminbound` core (`_minimize_scalar_bounded`): bounded
/// minimization by golden section with parabolic interpolation.
fn fminbound(f: impl Fn(f64) -> f64, x1: f64, x2: f64, xatol: f64, maxfun: usize) -> f64 {
    let sqrt_eps = 2.2e-16f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let fulc = a + golden_mean * (b - a);
    let (mut fulc, mut nfc, mut xf) = (fulc, fulc, fulc);
    let (mut rat, mut e) = (0.0f64, 0.0f64);
    let mut fx = f(xf);
    let mut num = 1usize;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;

    let sign_or_one = |v: f64| -> f64 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            1.0
        }
    };

    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    rat = tol1 * sign_or_one(xm - xf);
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }

        let x = xf + sign_or_one(rat) * rat.abs().max(tol1);
        let fu = f(x);
        num += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            (fulc, ffulc) = (nfc, fnfc);
            (nfc, fnfc) = (xf, fx);
            (xf, fx) = (x, fu);
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                (fulc, ffulc) = (nfc, fnfc);
                (nfc, fnfc) = (x, fu);
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                (fulc, ffulc) = (x, fu);
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxfun {
            break;
        }
    }
    xf
}

/// scipy's `yeojohnson_normmax` without a bracket: MLE lambda over
/// data-dependent bounds that keep the transform inside f64 range,
/// minimized by `fminbound` at brent tolerance.
pub fn yeojohnson_normmax(data: &[f64]) -> f64 {
    if data.iter().all(|v| *v == 0.0) {
        return 1.0;
    }
    let max_abs = data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let log1p_max_x = (20.0 * max_abs).ln_1p();
    let log_eps = EPS.ln();
    let log_tiny_float = (f64::MIN_POSITIVE.ln() - log_eps) / 2.0;
    let log_max_float = (f64::MAX.ln() + log_eps) / 2.0;
    let mut lb = log_tiny_float / log1p_max_x;
    let mut ub = log_max_float / log1p_max_x;
    if data.iter().all(|v| *v < 0.0) {
        (lb, ub) = (2.0 - ub, 2.0 - lb);
    } else if data.iter().any(|v| *v < 0.0) {
        (lb, ub) = ((2.0 - ub).max(lb), (2.0 - lb).min(ub));
    }
    let neg_llf = |l: f64| -> f64 {
        let v = yeojohnson_llf(l, data);
        if v.is_infinite() { f64::INFINITY } else { -v }
    };
    fminbound(neg_llf, lb, ub, 1.48e-8, 500)
}

/// sklearn's `_is_constant_feature`: variance indistinguishable from
/// the two-pass algorithm's error bound.
fn is_constant_feature(var: f64, mean: f64, n: usize) -> bool {
    let n = n as f64;
    var <= n * EPS * var + (n * mean * EPS).powi(2)
}

/// The fitted `PowerTransformer(standardize=True)`: per-feature
/// lambdas plus the standard scaler over the transformed features.
pub struct PowerStage {
    pub lambdas: Vec<f64>,
    means: Vec<f64>,
    scales: Vec<f64>,
}

impl PowerStage {
    /// x: row-major (rows, cols), finite (the pipeline feeds scaled,
    /// imputed data). Constant features keep lambda = 1 (identity).
    pub fn fit(x: &[f64], rows: usize, cols: usize) -> Self {
        assert_eq!(x.len(), rows * cols);
        let mut lambdas = Vec::with_capacity(cols);
        let mut transformed = vec![0.0f64; rows * cols];
        for c in 0..cols {
            let col: Vec<f64> = (0..rows).map(|r| x[r * cols + c]).collect();
            let n = col.len() as f64;
            let mean = col.iter().sum::<f64>() / n;
            let var = population_var(&col);
            let lambda = if is_constant_feature(var, mean, rows) {
                1.0
            } else {
                yeojohnson_normmax(&col)
            };
            lambdas.push(lambda);
            for r in 0..rows {
                transformed[r * cols + c] = yeojohnson(col[r], lambda);
            }
        }
        // StandardScaler over the transformed features, near-constant
        // scales pinned to 1 as sklearn's zero handling does.
        let mut means = Vec::with_capacity(cols);
        let mut scales = Vec::with_capacity(cols);
        for c in 0..cols {
            let col: Vec<f64> = (0..rows).map(|r| transformed[r * cols + c]).collect();
            let mean = col.iter().sum::<f64>() / rows as f64;
            let var = population_var(&col);
            let scale = if is_constant_feature(var, mean, rows) || var.sqrt() == 0.0 {
                1.0
            } else {
                var.sqrt()
            };
            means.push(mean);
            scales.push(scale);
        }
        Self {
            lambdas,
            means,
            scales,
        }
    }

    /// Row-major (rows, cols) in, same shape out.
    pub fn transform(&self, x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        assert_eq!(cols, self.lambdas.len());
        assert_eq!(x.len(), rows * cols);
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let c = i % cols;
                (yeojohnson(v, self.lambdas[c]) - self.means[c]) / self.scales[c]
            })
            .collect()
    }
}
