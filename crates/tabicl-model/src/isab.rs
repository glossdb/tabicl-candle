//! Induced self-attention block: inducing points attend to the train
//! prefix (stage 1, the only SSMax'd attention in the column stage),
//! then the full sequence attends back to the induced summary (stage 2).
//! Batch elements that are entirely the skip value pass through as the
//! skip value — the source computes only the live elements; computing
//! everything and overwriting the skipped rows is elementwise identical
//! because batch elements never mix.

use candle_core::{D, Tensor};

use crate::attention::{Block, Kv};
use crate::nn::{SKIP_VALUE, TensorMap};

pub struct Isab {
    attn1: Block,
    attn2: Block,
    ind_vectors: Tensor, // (num_inds, d_model)
}

impl Isab {
    pub fn load(
        tm: &TensorMap,
        prefix: &str,
        num_heads: usize,
        ssmax: bool,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attn1: Block::load(tm, &format!("{prefix}.multihead_attn1"), num_heads, ssmax)?,
            attn2: Block::load(tm, &format!("{prefix}.multihead_attn2"), num_heads, false)?,
            ind_vectors: tm.get(&format!("{prefix}.ind_vectors"))?,
        })
    }

    /// src: (batch, T, E); inducing points see only the first
    /// `train_size` rows.
    pub fn forward(&self, src: &Tensor, train_size: usize) -> anyhow::Result<Tensor> {
        let (b, _, e) = src.dims3()?;
        let (num_inds, _) = self.ind_vectors.dims2()?;
        let ind = self
            .ind_vectors
            .unsqueeze(0)?
            .expand((b, num_inds, e))?
            .contiguous()?;
        let train = src.narrow(1, 0, train_size)?.contiguous()?;
        let hidden = self.attn1.forward(&ind, Kv::Cross(&train), None)?;
        let out = self.attn2.forward(src, Kv::Cross(&hidden), None)?;

        // Skip protocol: a batch element whose every value is the skip
        // value stays the skip value.
        let dev = (src - SKIP_VALUE)?.abs()?.max(D::Minus1)?.max(D::Minus1)?;
        let flags: Vec<f32> = dev.to_vec1()?;
        if flags.iter().all(|&f| f != 0.0) {
            return Ok(out);
        }
        let mask = dev
            .eq(0f64)?
            .to_dtype(candle_core::DType::F32)?
            .reshape((b, 1, 1))?;
        let keep = (1.0 - &mask)?;
        Ok(out
            .broadcast_mul(&keep)?
            .broadcast_add(&(mask * SKIP_VALUE)?)?)
    }
}
