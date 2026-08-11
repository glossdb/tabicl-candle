//! Multi-head attention and the pre-norm block, mirroring the model's
//! `MultiheadAttention` / `MultiheadAttentionBlock`. Context restriction
//! (train-prefix attention) is done the way the source does it — by
//! slicing K/V, never with masks — and softmax scale is 1/sqrt(head_dim)
//! applied after SSMax has scaled the queries.

use candle_core::Tensor;

use crate::nn::{LayerNorm, Linear, TensorMap};
use crate::rope::Rope;
use crate::ssmax::QassMax;

pub struct Mha {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    out: Linear,
    pub num_heads: usize,
    pub ssmax: Option<QassMax>,
}

impl Mha {
    /// `prefix` names the torch module holding `in_proj_weight`,
    /// `in_proj_bias`, `out_proj.*` and optionally `ssmax_layer.*`.
    pub fn load(
        tm: &TensorMap,
        prefix: &str,
        num_heads: usize,
        ssmax: bool,
    ) -> anyhow::Result<Self> {
        let w = tm.get(&format!("{prefix}.in_proj_weight"))?;
        let b = tm.get(&format!("{prefix}.in_proj_bias"))?;
        let e = w.dim(1)?;
        let slice = |i: usize| -> anyhow::Result<Linear> {
            Ok(Linear {
                w: w.narrow(0, i * e, e)?.contiguous()?,
                b: Some(b.narrow(0, i * e, e)?.contiguous()?),
            })
        };
        Ok(Self {
            wq: slice(0)?,
            wk: slice(1)?,
            wv: slice(2)?,
            out: tm.linear(&format!("{prefix}.out_proj"))?,
            num_heads,
            ssmax: if ssmax {
                Some(QassMax::load(
                    tm,
                    &format!("{prefix}.ssmax_layer"),
                    num_heads,
                )?)
            } else {
                None
            },
        })
    }

    /// q: (batch, len_q, E), kv: (batch, len_k, E) -> (batch, len_q, E).
    pub fn forward(&self, q: &Tensor, kv: &Tensor, rope: Option<&Rope>) -> anyhow::Result<Tensor> {
        let (b, len_q, e) = q.dims3()?;
        let len_k = kv.dim(1)?;
        let (nh, hd) = (self.num_heads, e / self.num_heads);

        let split = |t: Tensor, len: usize| -> anyhow::Result<Tensor> {
            Ok(t.reshape((b, len, nh, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let mut qh = split(self.wq.forward(q)?, len_q)?;
        let mut kh = split(self.wk.forward(kv)?, len_k)?;
        let vh = split(self.wv.forward(kv)?, len_k)?;

        if let Some(rope) = rope {
            qh = rope.apply(&qh)?;
            kh = rope.apply(&kh)?;
        }
        if let Some(ssmax) = &self.ssmax {
            qh = ssmax.forward(&qh, len_k)?;
        }

        let qh = qh.reshape((b * nh, len_q, hd))?;
        let kh = kh
            .reshape((b * nh, len_k, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let vh = vh.reshape((b * nh, len_k, hd))?;
        let scores = (qh.matmul(&kh)? * (1.0 / (hd as f64).sqrt()))?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&vh)?;
        let attn = attn
            .reshape((b, nh, len_q, hd))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, len_q, e))?;
        self.out.forward(&attn)
    }
}

/// How a block sources its K/V.
pub enum Kv<'a> {
    /// Self-attention over the full sequence.
    SelfFull,
    /// Self-attention where K/V are the first `n` positions of the
    /// normalized queries (train-prefix attention).
    SelfTrain(usize),
    /// Cross-attention: K/V are this tensor, normalized by the block's
    /// own norm1 (the source's `k_normed = self.norm1(k)`).
    Cross(&'a Tensor),
}

/// Pre-norm transformer block: x + attn(norm1(x)), then
/// x + linear2(gelu(linear1(norm2(x)))).
pub struct Block {
    pub attn: Mha,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Block {
    /// `prefix` names the torch block holding `attn.*`, `norm1.*`,
    /// `norm2.*`, `linear1.*`, `linear2.*`.
    pub fn load(
        tm: &TensorMap,
        prefix: &str,
        num_heads: usize,
        ssmax: bool,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attn: Mha::load(tm, &format!("{prefix}.attn"), num_heads, ssmax)?,
            norm1: tm.layer_norm(&format!("{prefix}.norm1"))?,
            norm2: tm.layer_norm(&format!("{prefix}.norm2"))?,
            linear1: tm.linear(&format!("{prefix}.linear1"))?,
            linear2: tm.linear(&format!("{prefix}.linear2"))?,
        })
    }

    pub fn forward(&self, q: &Tensor, kv: Kv, rope: Option<&Rope>) -> anyhow::Result<Tensor> {
        let q_normed = self.norm1.forward(q)?;
        let kv_normed = match kv {
            Kv::SelfFull => q_normed.clone(),
            Kv::SelfTrain(n) => q_normed.narrow(1, 0, n)?.contiguous()?,
            Kv::Cross(t) => self.norm1.forward(t)?,
        };
        let attn = self.attn.forward(&q_normed, &kv_normed, rope)?;
        let x = (q + attn)?;
        let ff = self
            .linear2
            .forward(&self.linear1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        Ok((x + ff)?)
    }
}
