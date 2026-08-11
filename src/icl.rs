//! In-context learning: targets embed into the train prefix of the row
//! representations, twelve pre-norm blocks run train-prefix attention
//! (every one SSMax'd — the log(train_size) runtime dependency), then
//! the decoder head maps to quantiles (regressor) or logits
//! (classifier). Train mode returns the test rows.

use candle_core::{D, Tensor};

use crate::attention::{Block, Kv};
use crate::nn::{LayerNorm, Linear, TensorMap, one_hot_linear};

pub struct IclPredictor {
    blocks: Vec<Block>,
    ln: LayerNorm,
    y_encoder: Linear,
    decoder0: Linear, // D -> 2D
    decoder2: Linear, // 2D -> out_dim
    pub max_classes: usize,
}

impl IclPredictor {
    pub fn load(
        tm: &TensorMap,
        num_blocks: usize,
        num_heads: usize,
        max_classes: usize,
    ) -> anyhow::Result<Self> {
        let blocks = (0..num_blocks)
            .map(|i| {
                Block::load(
                    tm,
                    &format!("icl_predictor.tf_icl.blocks.{i}"),
                    num_heads,
                    true,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            ln: tm.layer_norm("icl_predictor.ln")?,
            y_encoder: tm.linear("icl_predictor.y_encoder")?,
            decoder0: tm.linear("icl_predictor.decoder.0")?,
            decoder2: tm.linear("icl_predictor.decoder.2")?,
            max_classes,
        })
    }

    /// r: (B, T, D), y: (B, train_size) -> (B, T - train_size, out_dim).
    pub fn forward(&self, r: &Tensor, y: &Tensor) -> anyhow::Result<Tensor> {
        let (_, t, _) = r.dims3()?;
        let train_size = y.dim(1)?;

        let y_emb = if self.max_classes > 0 {
            one_hot_linear(&self.y_encoder, y, self.max_classes)?
        } else {
            self.y_encoder.forward(&y.unsqueeze(D::Minus1)?)?
        };
        let r_train = (r.narrow(1, 0, train_size)? + y_emb)?;
        let r_rest = r.narrow(1, train_size, t - train_size)?;
        let mut x = Tensor::cat(&[&r_train, &r_rest], 1)?;

        for block in &self.blocks {
            x = block.forward(&x, Kv::SelfTrain(train_size), None)?;
        }
        let x = self.ln.forward(&x)?;
        let out = self
            .decoder2
            .forward(&self.decoder0.forward(&x)?.gelu_erf()?)?;
        Ok(out.narrow(1, train_size, t - train_size)?.contiguous()?)
    }
}
