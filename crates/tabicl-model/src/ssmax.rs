//! QASSMax, elementwise variant — the reason this port exists at all:
//! the query scale is `base_mlp(log n) * (1 + tanh(query_mlp(q)))` with
//! n the *runtime* context length, which a traced export bakes in as a
//! constant (the 2026-08-11 export evaluation).

use candle_core::Tensor;

use crate::nn::Linear;

pub struct QassMax {
    pub base0: Linear,  // 1 -> 64
    pub base2: Linear,  // 64 -> num_heads * head_dim
    pub query0: Linear, // head_dim -> 64
    pub query2: Linear, // 64 -> head_dim
    pub num_heads: usize,
}

impl QassMax {
    pub fn load(tm: &crate::nn::TensorMap, prefix: &str, num_heads: usize) -> anyhow::Result<Self> {
        Ok(Self {
            base0: tm.linear(&format!("{prefix}.base_mlp.0"))?,
            base2: tm.linear(&format!("{prefix}.base_mlp.2"))?,
            query0: tm.linear(&format!("{prefix}.query_mlp.0"))?,
            query2: tm.linear(&format!("{prefix}.query_mlp.2"))?,
            num_heads,
        })
    }

    /// q: (batch, heads, len, head_dim); n is the source length the
    /// queries attend over. Returns the scaled queries.
    pub fn forward(&self, q: &Tensor, n: usize) -> anyhow::Result<Tensor> {
        let head_dim = q.dim(3)?;
        let logn = Tensor::full((n.max(1) as f64).ln() as f32, (1, 1), q.device())?;
        let base = self
            .base2
            .forward(&self.base0.forward(&logn)?.gelu_erf()?)?
            .reshape((1, self.num_heads, 1, head_dim))?;
        let modulation = (self
            .query2
            .forward(&self.query0.forward(q)?.gelu_erf()?)?
            .tanh()?
            + 1.0)?;
        Ok(q.broadcast_mul(&base)?.mul(&modulation)?)
    }
}
