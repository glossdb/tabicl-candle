//! Small primitives shared by every stage. Everything is fp32 and
//! mirrors torch semantics exactly: LayerNorm uses biased variance and
//! eps 1e-5, GELU is the erf form (torch's default, not tanh-approx).

use candle_core::{D, Tensor};

pub const SKIP_VALUE: f64 = -100.0;

/// The checkpoint's tensors, with lookups that name what's missing.
pub struct TensorMap(pub std::collections::HashMap<String, Tensor>);

impl TensorMap {
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no tensor named {name}"))
    }

    pub fn linear(&self, prefix: &str) -> anyhow::Result<Linear> {
        Ok(Linear {
            w: self.get(&format!("{prefix}.weight"))?,
            b: Some(self.get(&format!("{prefix}.bias"))?),
        })
    }

    pub fn layer_norm(&self, prefix: &str) -> anyhow::Result<LayerNorm> {
        // bias_free_ln=true in both shipped checkpoints; a bias tensor,
        // if a future checkpoint carries one, is picked up.
        Ok(LayerNorm {
            w: self.get(&format!("{prefix}.weight"))?,
            b: self.0.get(&format!("{prefix}.bias")).cloned(),
        })
    }
}

/// Torch nn.Linear: y = x W^T + b, weight stored (out, in).
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    /// x: (..., in) -> (..., out); flattens batch dims for the matmul.
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let dims = x.dims().to_vec();
        let (n_in, n_out) = (self.w.dim(1)?, self.w.dim(0)?);
        let n: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.contiguous()?.reshape((n, n_in))?;
        let mut y = x2.matmul(&self.w.t()?)?;
        if let Some(b) = &self.b {
            y = y.broadcast_add(b)?;
        }
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = n_out;
        Ok(y.reshape(out_dims)?)
    }
}

pub struct LayerNorm {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl LayerNorm {
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mu = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mu)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let mut y = xc
            .broadcast_div(&(var + 1e-5)?.sqrt()?)?
            .broadcast_mul(&self.w)?;
        if let Some(b) = &self.b {
            y = y.broadcast_add(b)?;
        }
        Ok(y)
    }
}

/// SkippableLinear: rows whose inputs are all SKIP_VALUE come out as
/// SKIP_VALUE, everything else is a plain linear.
pub fn skippable_linear(lin: &Linear, x: &Tensor) -> anyhow::Result<Tensor> {
    let y = lin.forward(x)?;
    // mask: 1.0 where the whole input row equals the skip value
    let dev = (x - SKIP_VALUE)?.abs()?.max_keepdim(D::Minus1)?;
    let mask = dev.eq(0f64)?.to_dtype(candle_core::DType::F32)?;
    let keep = (1.0 - &mask)?;
    Ok(y.broadcast_mul(&keep)?
        .broadcast_add(&(mask * SKIP_VALUE)?)?)
}

/// One-hot against `num_classes` then linear — torch's OneHotAndLinear.
pub fn one_hot_linear(lin: &Linear, y: &Tensor, num_classes: usize) -> anyhow::Result<Tensor> {
    let classes = Tensor::arange(0f32, num_classes as f32, y.device())?;
    let one_hot = y
        .unsqueeze(D::Minus1)?
        .broadcast_eq(&classes)?
        .to_dtype(candle_core::DType::F32)?;
    lin.forward(&one_hot)
}
